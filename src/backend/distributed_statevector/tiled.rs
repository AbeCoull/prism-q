//! Out-of-core tiled statevector prototype: file-backed tiles streamed
//! through a bounded DRAM window. Not part of production builds, and no
//! production selector exists yet.
//!
//! The `2^n` amplitude vector lives in one raw `Complex64` file inside a
//! run-scoped temporary directory, split into `2^(n - w)` tiles of `2^w`
//! amplitudes. The low `w` qubits index inside a tile; the high `n - w`
//! qubits select the tile, mirroring how rank bits sit above local bits in
//! the sharded backend. Qubit 0 is the least significant bit. The file
//! layout is a private implementation detail with no version guarantee, and
//! the storage is deleted on drop.
//!
//! Consecutive instructions whose non-diagonal targets are all window-local
//! share one streaming pass over the tiles, applied through the inner dense
//! kernels. Diagonal action and control bits on tile-bit qubits are free
//! inside a pass, mirroring the rank-bit rules. A non-diagonal gate on a
//! tile-bit qubit pairs the tiles differing in that bit and costs its own
//! pass. A schedule needing more than [`DEFAULT_MAX_PASS_RATE`] passes per
//! gate is rejected before any storage I/O once it needs at least
//! [`MIN_PASSES_FOR_REJECTION`] passes; [`TileConfig::max_pass_rate`] at or
//! above 1.0 disables the rejection.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use num_complex::Complex64;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::backend::statevector::StatevectorBackend;
use crate::backend::{Backend, dense_statevector_len, measurement_inv_norm, simd};
use crate::circuit::{Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

const BACKEND_NAME: &str = "tiled_statevector";

/// Default ceiling on full-shard streaming passes per scheduled gate.
///
/// Above this rate, window batching is failing for most of the circuit and
/// runtime is predictably storage-bound.
pub const DEFAULT_MAX_PASS_RATE: f64 = 0.5;

/// Schedules needing fewer passes than this are never rejected: a handful of
/// streams is the unavoidable minimum for any schedule, including one gate.
pub const MIN_PASSES_FOR_REJECTION: u64 = 4;

static STORE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Geometry and policy for a [`TiledStatevector`].
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Qubits resident per tile; the DRAM window holds `2^window_qubits`
    /// amplitudes, twice that during a pair pass.
    pub window_qubits: usize,
    /// Parent of the run-scoped storage directory. Defaults to the system
    /// temporary directory.
    pub dir: Option<PathBuf>,
    /// Rejection threshold on streaming passes per gate. Values at or above
    /// 1.0 disable the rejection; tests forcing tiny tiles rely on that.
    pub max_pass_rate: f64,
}

impl TileConfig {
    /// Config with the default pass-rate threshold and storage location.
    pub fn new(window_qubits: usize) -> Self {
        Self {
            window_qubits,
            dir: None,
            max_pass_rate: DEFAULT_MAX_PASS_RATE,
        }
    }
}

enum WindowOp {
    Inner(Instruction),
    Masked {
        tile_mask: usize,
        instr: Instruction,
    },
    Masked1q {
        tile_mask: usize,
        qubit: usize,
        mat: [[Complex64; 2]; 2],
    },
    ScaleAll {
        tile_mask: usize,
        factor: Complex64,
    },
    Diag1q {
        tile_bit: usize,
        d0: Complex64,
        d1: Complex64,
    },
    ParityScale {
        bit0: usize,
        bit1: usize,
        same: Complex64,
        diff: Complex64,
    },
    ParityDiag {
        tile_bit: usize,
        qubit: usize,
        same: Complex64,
        diff: Complex64,
    },
}

enum PairOp {
    Combine {
        tile_bit: usize,
        tile_mask: usize,
        ctrl_mask: usize,
        mat: [[Complex64; 2]; 2],
    },
    SwapMixed {
        tile_bit: usize,
        window_qubit: usize,
    },
    SwapTiles {
        bit0: usize,
        bit1: usize,
    },
}

enum Translated {
    Window(WindowOp),
    Pair(PairOp),
}

/// Single-process out-of-core statevector over file-backed tiles.
///
/// The window and partner staging buffers are allocated at construction, so
/// gate passes never allocate. Measurement draws from the same seeded stream
/// the distributed backend uses, so outcomes agree with it for a given seed.
pub struct TiledStatevector {
    num_qubits: usize,
    window_qubits: usize,
    tile_bits: usize,
    /// `Some` until drop; taken there so the handle closes before the file
    /// is removed, which Windows requires.
    file: Option<File>,
    path: PathBuf,
    dir: PathBuf,
    /// `inner.state` is the primary DRAM window; tiles are read into it and
    /// window-local instructions dispatch to the inner kernels.
    inner: StatevectorBackend,
    partner: Vec<Complex64>,
    pending_norm: f64,
    classical_bits: Vec<bool>,
    meas_rng: ChaCha8Rng,
    max_pass_rate: f64,
    passes: u64,
    tiles_streamed: u64,
}

fn complex_bytes(buf: &[Complex64]) -> &[u8] {
    // SAFETY: Complex64 is repr(C) over two f64 values with no padding, so
    // the slice is size_of_val(buf) initialized bytes.
    unsafe { std::slice::from_raw_parts(buf.as_ptr().cast(), std::mem::size_of_val(buf)) }
}

fn complex_bytes_mut(buf: &mut [Complex64]) -> &mut [u8] {
    // SAFETY: same layout as `complex_bytes`; every byte pattern is a valid
    // Complex64 and the mutable borrow is exclusive.
    unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), std::mem::size_of_val(buf)) }
}

fn storage_error(path: &Path, operation: &str, error: &std::io::Error) -> PrismError {
    PrismError::BackendUnsupported {
        backend: BACKEND_NAME.to_string(),
        operation: format!("{operation} at {}: {error}", path.display()),
    }
}

fn combine_pair(
    lo: &mut [Complex64],
    hi: &mut [Complex64],
    ctrl_mask: usize,
    mat: &[[Complex64; 2]; 2],
) {
    for (k, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
        if k & ctrl_mask == ctrl_mask {
            let (x, y) = (*a, *b);
            *a = mat[0][0] * x + mat[0][1] * y;
            *b = mat[1][0] * x + mat[1][1] * y;
        }
    }
}

fn apply_window_op(inner: &mut StatevectorBackend, op: &WindowOp, tile: usize) {
    match op {
        WindowOp::Inner(instr) => inner.apply(instr).expect("window-local instruction"),
        WindowOp::Masked { tile_mask, instr } => {
            if tile & tile_mask == *tile_mask {
                inner
                    .apply(instr)
                    .expect("window-local controlled residual");
            }
        }
        WindowOp::Masked1q {
            tile_mask,
            qubit,
            mat,
        } => {
            if tile & tile_mask == *tile_mask {
                inner
                    .apply_1q_matrix(*qubit, mat)
                    .expect("window-local 1q residual");
            }
        }
        WindowOp::ScaleAll { tile_mask, factor } => {
            if tile & tile_mask == *tile_mask {
                simd::scale_complex_slice(&mut inner.state, *factor);
            }
        }
        WindowOp::Diag1q { tile_bit, d0, d1 } => {
            let factor = if (tile >> tile_bit) & 1 == 1 {
                *d1
            } else {
                *d0
            };
            simd::scale_complex_slice(&mut inner.state, factor);
        }
        WindowOp::ParityScale {
            bit0,
            bit1,
            same,
            diff,
        } => {
            let parity = ((tile >> bit0) ^ (tile >> bit1)) & 1;
            simd::scale_complex_slice(&mut inner.state, [*same, *diff][parity]);
        }
        WindowOp::ParityDiag {
            tile_bit,
            qubit,
            same,
            diff,
        } => {
            let gbit = (tile >> tile_bit) & 1;
            let d0 = [*same, *diff][gbit];
            let d1 = [*same, *diff][gbit ^ 1];
            let z = Complex64::new(0.0, 0.0);
            inner
                .apply_1q_matrix(*qubit, &[[d0, z], [z, d1]])
                .expect("window-local parity residual");
        }
    }
}

impl TiledStatevector {
    /// Create a `|0...0>` state with its tile storage on disk.
    ///
    /// Rejects a window of zero qubits, a window wider than the register, and
    /// a window pair exceeding the statevector memory cap. Storage failures
    /// surface as `PrismError` naming the path.
    pub fn new(
        num_qubits: usize,
        num_classical_bits: usize,
        seed: u64,
        config: TileConfig,
    ) -> Result<Self> {
        if num_qubits == 0 || num_qubits >= usize::BITS as usize {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "tiled statevector supports 1 to {} qubits, got {num_qubits}",
                    usize::BITS - 1
                ),
            });
        }
        let w = config.window_qubits;
        if w == 0 || w > num_qubits {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "window of {w} qubits is invalid for a {num_qubits} qubit register"
                ),
            });
        }
        let cap = crate::backend::memory::max_statevector_qubits();
        if w + 1 > cap {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "window pair of {} qubits exceeds the statevector cap of {cap} on this \
                     machine (set PRISM_MAX_SV_QUBITS to override)",
                    w + 1
                ),
            });
        }
        let total_bytes: u64 = ((1u128 << num_qubits) * std::mem::size_of::<Complex64>() as u128)
            .try_into()
            .map_err(|_| PrismError::InvalidParameter {
                message: format!("tile storage for {num_qubits} qubits exceeds addressable size"),
            })?;

        let base = config.dir.clone().unwrap_or_else(std::env::temp_dir);
        let dir = base.join(format!(
            "prismq-tiles-{}-{}",
            std::process::id(),
            STORE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir)
            .map_err(|e| storage_error(&dir, "create storage directory", &e))?;
        let path = dir.join("tiles.bin");
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| storage_error(&path, "create tile file", &e))?;

        let mut inner = StatevectorBackend::new(seed);
        inner.init(w, 0)?;
        let mut state = Self {
            num_qubits,
            window_qubits: w,
            tile_bits: num_qubits - w,
            file: Some(file),
            path,
            dir,
            inner,
            partner: vec![Complex64::new(0.0, 0.0); 1usize << w],
            pending_norm: 1.0,
            classical_bits: vec![false; num_classical_bits],
            meas_rng: ChaCha8Rng::seed_from_u64(seed),
            max_pass_rate: config.max_pass_rate,
            passes: 0,
            tiles_streamed: 0,
        };
        state.init_storage(total_bytes)?;
        Ok(state)
    }

    fn init_storage(&mut self, total_bytes: u64) -> Result<()> {
        let file = self.file.as_mut().expect("storage open until drop");
        file.set_len(total_bytes)
            .and_then(|_| file.seek(SeekFrom::Start(0)))
            .and_then(|_| file.write_all(complex_bytes(&[Complex64::new(1.0, 0.0)])))
            .map_err(|e| storage_error(&self.path, "initialize tile file", &e))
    }

    #[inline]
    fn tile_len(&self) -> usize {
        1usize << self.window_qubits
    }

    #[inline]
    fn tile_count(&self) -> usize {
        1usize << self.tile_bits
    }

    #[inline]
    fn tile_byte_len(&self) -> u64 {
        (self.tile_len() * std::mem::size_of::<Complex64>()) as u64
    }

    fn read_at(
        file: Option<&mut File>,
        path: &Path,
        offset: u64,
        buf: &mut [Complex64],
    ) -> Result<()> {
        let file = file.expect("storage open until drop");
        file.seek(SeekFrom::Start(offset))
            .and_then(|_| file.read_exact(complex_bytes_mut(buf)))
            .map_err(|e| storage_error(path, "read tile", &e))
    }

    fn write_at(
        file: Option<&mut File>,
        path: &Path,
        offset: u64,
        buf: &[Complex64],
    ) -> Result<()> {
        let file = file.expect("storage open until drop");
        file.seek(SeekFrom::Start(offset))
            .and_then(|_| file.write_all(complex_bytes(buf)))
            .map_err(|e| storage_error(path, "write tile", &e))
    }

    fn read_window_tile(&mut self, tile: usize) -> Result<()> {
        let offset = tile as u64 * self.tile_byte_len();
        Self::read_at(
            self.file.as_mut(),
            &self.path,
            offset,
            &mut self.inner.state,
        )?;
        self.tiles_streamed += 1;
        Ok(())
    }

    fn read_partner_tile(&mut self, tile: usize) -> Result<()> {
        let offset = tile as u64 * self.tile_byte_len();
        Self::read_at(self.file.as_mut(), &self.path, offset, &mut self.partner)?;
        self.tiles_streamed += 1;
        Ok(())
    }

    fn write_window_tile(&mut self, tile: usize) -> Result<()> {
        let offset = tile as u64 * self.tile_byte_len();
        Self::write_at(self.file.as_mut(), &self.path, offset, &self.inner.state)
    }

    fn write_partner_tile(&mut self, tile: usize) -> Result<()> {
        let offset = tile as u64 * self.tile_byte_len();
        Self::write_at(self.file.as_mut(), &self.path, offset, &self.partner)
    }

    /// Apply `instructions` after a dry scheduling pass that rejects
    /// tile-thrashing schedules and unsupported gate shapes before any
    /// storage I/O.
    pub fn run(&mut self, instructions: &[Instruction]) -> Result<()> {
        self.check_pass_rate(instructions)?;
        let mut batch: Vec<WindowOp> = Vec::new();
        for instruction in instructions {
            self.step(instruction, &mut batch)?;
        }
        self.flush(&mut batch)
    }

    fn step(&mut self, instruction: &Instruction, batch: &mut Vec<WindowOp>) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.enqueue(gate, targets, batch),
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                // Classical bits only change at measure and reset, which flush
                // the batch, so the bits seen here are the bits the flush sees.
                if condition.evaluate(&self.classical_bits) {
                    self.enqueue(gate, targets, batch)
                } else {
                    Ok(())
                }
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.flush(batch)?;
                self.measure(*qubit, *classical_bit)
            }
            Instruction::Reset { qubit } => {
                self.flush(batch)?;
                self.reset(*qubit)
            }
            Instruction::Barrier { .. } => Ok(()),
        }
    }

    fn enqueue(&mut self, gate: &Gate, targets: &[usize], batch: &mut Vec<WindowOp>) -> Result<()> {
        match self.translate(gate, targets)? {
            Translated::Window(op) => {
                batch.push(op);
                Ok(())
            }
            Translated::Pair(op) => {
                self.flush(batch)?;
                self.pair_pass(&op)
            }
        }
    }

    fn translate(&self, gate: &Gate, targets: &[usize]) -> Result<Translated> {
        let w = self.window_qubits;
        let mut all_window = true;
        super::for_each_gate_qubit(gate, targets, |q| all_window &= q < w);
        if all_window {
            return Ok(Translated::Window(WindowOp::Inner(Instruction::Gate {
                gate: gate.clone(),
                targets: targets.into(),
            })));
        }
        match gate {
            Gate::Swap => {
                let (a, b) = (targets[0], targets[1]);
                Ok(Translated::Pair(match (a < w, b < w) {
                    (true, false) => PairOp::SwapMixed {
                        tile_bit: b - w,
                        window_qubit: a,
                    },
                    (false, true) => PairOp::SwapMixed {
                        tile_bit: a - w,
                        window_qubit: b,
                    },
                    _ => PairOp::SwapTiles {
                        bit0: a - w,
                        bit1: b - w,
                    },
                }))
            }
            Gate::Cx => {
                Ok(self.translate_controlled(&targets[..1], targets[1], Gate::X.matrix_2x2()))
            }
            Gate::Cz => Ok(Translated::Window(
                self.corner_phase(&[targets[0], targets[1]], -Complex64::new(1.0, 0.0)),
            )),
            Gate::Rzz(theta) => {
                let same = Complex64::from_polar(1.0, -theta / 2.0);
                let diff = Complex64::from_polar(1.0, theta / 2.0);
                let (q0, q1) = (targets[0], targets[1]);
                Ok(Translated::Window(match (q0 < w, q1 < w) {
                    (false, false) => WindowOp::ParityScale {
                        bit0: q0 - w,
                        bit1: q1 - w,
                        same,
                        diff,
                    },
                    (true, false) => WindowOp::ParityDiag {
                        tile_bit: q1 - w,
                        qubit: q0,
                        same,
                        diff,
                    },
                    (false, true) => WindowOp::ParityDiag {
                        tile_bit: q0 - w,
                        qubit: q1,
                        same,
                        diff,
                    },
                    (true, true) => unreachable!("handled by the all-window fast path"),
                }))
            }
            Gate::Cu(mat) => {
                if let Some(phase) = gate.controlled_phase() {
                    Ok(Translated::Window(
                        self.corner_phase(&[targets[0], targets[1]], phase),
                    ))
                } else {
                    Ok(self.translate_controlled(&targets[..1], targets[1], **mat))
                }
            }
            Gate::Mcu(data) => {
                let num_controls = data.num_controls as usize;
                if let Some(phase) = gate.controlled_phase() {
                    Ok(Translated::Window(self.corner_phase(targets, phase)))
                } else {
                    Ok(self.translate_controlled(
                        &targets[..num_controls],
                        targets[num_controls],
                        data.mat,
                    ))
                }
            }
            g if g.num_qubits() == 1 => {
                let qubit = targets[0];
                let mat = g.matrix_2x2();
                if g.is_diagonal_1q() {
                    Ok(Translated::Window(WindowOp::Diag1q {
                        tile_bit: qubit - w,
                        d0: mat[0][0],
                        d1: mat[1][1],
                    }))
                } else {
                    Ok(Translated::Pair(PairOp::Combine {
                        tile_bit: qubit - w,
                        tile_mask: 0,
                        ctrl_mask: 0,
                        mat,
                    }))
                }
            }
            _ => Err(PrismError::BackendUnsupported {
                backend: BACKEND_NAME.to_string(),
                operation: "fused or batched gate spanning a tile-bit qubit".to_string(),
            }),
        }
    }

    /// Split a controlled 2x2 application into tile-bit controls (a tile
    /// mask), window controls, and the target side. A zero tile-mask bit
    /// deactivates the gate for that tile with no amplitude traffic.
    fn translate_controlled(
        &self,
        controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) -> Translated {
        let w = self.window_qubits;
        let mut tile_mask = 0usize;
        let mut window_controls: SmallVec<[usize; 4]> = SmallVec::new();
        for &c in controls {
            if c < w {
                window_controls.push(c);
            } else {
                tile_mask |= 1 << (c - w);
            }
        }
        if target >= w {
            let ctrl_mask = window_controls.iter().map(|&c| 1usize << c).sum();
            return Translated::Pair(PairOp::Combine {
                tile_bit: target - w,
                tile_mask,
                ctrl_mask,
                mat,
            });
        }
        Translated::Window(match window_controls.len() {
            0 => WindowOp::Masked1q {
                tile_mask,
                qubit: target,
                mat,
            },
            n => {
                let gate = if n == 1 {
                    Gate::cu(mat)
                } else {
                    Gate::mcu(mat, n as u8)
                };
                let mut instr_targets: SmallVec<[usize; 4]> = window_controls;
                instr_targets.push(target);
                WindowOp::Masked {
                    tile_mask,
                    instr: Instruction::Gate {
                        gate,
                        targets: instr_targets,
                    },
                }
            }
        })
    }

    /// Phase on the all-ones corner of `qubits`: tile-bit qubits become a
    /// tile mask and window qubits carry the residual controlled phase.
    fn corner_phase(&self, qubits: &[usize], phase: Complex64) -> WindowOp {
        let w = self.window_qubits;
        let mut window_qubits: SmallVec<[usize; 8]> = SmallVec::new();
        let mut tile_mask = 0usize;
        for &q in qubits {
            if q >= w {
                tile_mask |= 1 << (q - w);
            } else if !window_qubits.contains(&q) {
                window_qubits.push(q);
            }
        }
        let z = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let mat = [[one, z], [z, phase]];
        match window_qubits.len() {
            0 => WindowOp::ScaleAll {
                tile_mask,
                factor: phase,
            },
            1 => WindowOp::Masked1q {
                tile_mask,
                qubit: window_qubits[0],
                mat,
            },
            n => {
                let gate = if n == 2 {
                    Gate::cu(mat)
                } else {
                    Gate::mcu(mat, (n - 1) as u8)
                };
                WindowOp::Masked {
                    tile_mask,
                    instr: Instruction::Gate {
                        gate,
                        targets: window_qubits.iter().copied().collect(),
                    },
                }
            }
        }
    }

    fn flush(&mut self, batch: &mut Vec<WindowOp>) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        self.passes += 1;
        for tile in 0..self.tile_count() {
            self.read_window_tile(tile)?;
            for op in batch.iter() {
                apply_window_op(&mut self.inner, op, tile);
            }
            self.write_window_tile(tile)?;
        }
        batch.clear();
        Ok(())
    }

    fn pair_pass(&mut self, op: &PairOp) -> Result<()> {
        self.passes += 1;
        match *op {
            PairOp::Combine {
                tile_bit,
                tile_mask,
                ctrl_mask,
                mat,
            } => {
                for t0 in 0..self.tile_count() {
                    if (t0 >> tile_bit) & 1 == 1 || t0 & tile_mask != tile_mask {
                        continue;
                    }
                    let t1 = t0 | (1 << tile_bit);
                    self.read_window_tile(t0)?;
                    self.read_partner_tile(t1)?;
                    combine_pair(&mut self.inner.state, &mut self.partner, ctrl_mask, &mat);
                    self.write_window_tile(t0)?;
                    self.write_partner_tile(t1)?;
                }
            }
            PairOp::SwapMixed {
                tile_bit,
                window_qubit,
            } => {
                let half = 1usize << window_qubit;
                for t0 in 0..self.tile_count() {
                    if (t0 >> tile_bit) & 1 == 1 {
                        continue;
                    }
                    let t1 = t0 | (1 << tile_bit);
                    self.read_window_tile(t0)?;
                    self.read_partner_tile(t1)?;
                    let (lo, hi) = (&mut self.inner.state, &mut self.partner);
                    for k in 0..lo.len() {
                        if k & half != 0 {
                            std::mem::swap(&mut lo[k], &mut hi[k ^ half]);
                        }
                    }
                    self.write_window_tile(t0)?;
                    self.write_partner_tile(t1)?;
                }
            }
            PairOp::SwapTiles { bit0, bit1 } => {
                for t0 in 0..self.tile_count() {
                    if (t0 >> bit0) & 1 == 0 || (t0 >> bit1) & 1 == 1 {
                        continue;
                    }
                    let t1 = t0 ^ (1 << bit0) ^ (1 << bit1);
                    self.read_window_tile(t0)?;
                    self.read_partner_tile(t1)?;
                    self.write_window_tile(t1)?;
                    self.write_partner_tile(t0)?;
                }
            }
        }
        Ok(())
    }

    /// Dry scheduling pass: count full-shard streaming passes without
    /// touching storage and reject a schedule whose passes-per-gate ratio
    /// exceeds the configured threshold.
    fn check_pass_rate(&self, instructions: &[Instruction]) -> Result<()> {
        let mut gate_passes = 0u64;
        let mut gate_count = 0u64;
        let mut open_batch = false;
        for instruction in instructions {
            let (gate, targets) = match instruction {
                Instruction::Gate { gate, targets } => (gate, targets),
                Instruction::Conditional { gate, targets, .. } => (gate, targets),
                Instruction::Barrier { .. } => continue,
                Instruction::Measure { .. } | Instruction::Reset { .. } => {
                    open_batch = false;
                    continue;
                }
            };
            gate_count += 1;
            match self.translate(gate, targets)? {
                Translated::Window(_) => {
                    if !open_batch {
                        gate_passes += 1;
                        open_batch = true;
                    }
                }
                Translated::Pair(_) => {
                    gate_passes += 1;
                    open_batch = false;
                }
            }
        }
        if gate_count == 0 || gate_passes < MIN_PASSES_FOR_REJECTION {
            return Ok(());
        }
        let rate = gate_passes as f64 / gate_count as f64;
        if rate > self.max_pass_rate {
            return Err(PrismError::BackendUnsupported {
                backend: BACKEND_NAME.to_string(),
                operation: format!(
                    "schedule needs {gate_passes} full-shard streaming passes for {gate_count} \
                     gates (pass rate {rate:.2} exceeds the threshold {:.2}); batch window-local \
                     work or raise TileConfig::max_pass_rate to opt in",
                    self.max_pass_rate
                ),
            });
        }
        Ok(())
    }

    /// Scaled weight of the `qubit == 1` subspace, streamed over the tiles.
    fn prob_one(&mut self, qubit: usize) -> Result<f64> {
        self.passes += 1;
        let mut acc = 0.0f64;
        if qubit < self.window_qubits {
            let half = 1usize << qubit;
            let block = half << 1;
            for tile in 0..self.tile_count() {
                self.read_window_tile(tile)?;
                for chunk in self.inner.state.chunks(block) {
                    let (_, hi) = chunk.split_at(half);
                    acc += simd::norm_sqr_sum(hi);
                }
            }
        } else {
            let bit = qubit - self.window_qubits;
            for tile in 0..self.tile_count() {
                if (tile >> bit) & 1 == 0 {
                    continue;
                }
                self.read_window_tile(tile)?;
                acc += simd::norm_sqr_sum(&self.inner.state);
            }
        }
        Ok(acc * self.pending_norm * self.pending_norm)
    }

    /// Zero the amplitudes inconsistent with `qubit == outcome`. Tiles that
    /// are eliminated whole are overwritten without being read.
    fn collapse(&mut self, qubit: usize, outcome: bool) -> Result<()> {
        self.passes += 1;
        if qubit < self.window_qubits {
            let half = 1usize << qubit;
            let block = half << 1;
            for tile in 0..self.tile_count() {
                self.read_window_tile(tile)?;
                for chunk in self.inner.state.chunks_mut(block) {
                    let (lo, hi) = chunk.split_at_mut(half);
                    simd::zero_slice(if outcome { lo } else { hi });
                }
                self.write_window_tile(tile)?;
            }
        } else {
            let bit = qubit - self.window_qubits;
            simd::zero_slice(&mut self.partner);
            for tile in 0..self.tile_count() {
                if ((tile >> bit) & 1 == 1) == outcome {
                    continue;
                }
                self.write_partner_tile(tile)?;
            }
        }
        Ok(())
    }

    fn measure(&mut self, qubit: usize, classical_bit: usize) -> Result<()> {
        let prob_one = self.prob_one(qubit)?;
        let outcome = self.meas_rng.random::<f64>() < prob_one;
        self.classical_bits[classical_bit] = outcome;
        self.collapse(qubit, outcome)?;
        self.pending_norm *= measurement_inv_norm(outcome, prob_one);
        Ok(())
    }

    /// Reset as one trajectory of the reset channel: sample, collapse, then
    /// apply X when the outcome is 1, matching the distributed backend.
    fn reset(&mut self, qubit: usize) -> Result<()> {
        let prob_one = self.prob_one(qubit)?;
        let outcome = self.meas_rng.random::<f64>() < prob_one;
        self.collapse(qubit, outcome)?;
        self.pending_norm *= measurement_inv_norm(outcome, prob_one);
        if outcome {
            let mat = Gate::X.matrix_2x2();
            if qubit < self.window_qubits {
                let mut batch = vec![WindowOp::Masked1q {
                    tile_mask: 0,
                    qubit,
                    mat,
                }];
                self.flush(&mut batch)?;
            } else {
                self.pair_pass(&PairOp::Combine {
                    tile_bit: qubit - self.window_qubits,
                    tile_mask: 0,
                    ctrl_mask: 0,
                    mat,
                })?;
            }
        }
        Ok(())
    }

    /// Assemble the full statevector in tile order, folding the deferred
    /// norm. Carries the same dense output cap as every gathering query.
    pub fn export_statevector(&mut self) -> Result<Vec<Complex64>> {
        let len = dense_statevector_len(BACKEND_NAME, "statevector export", self.num_qubits)?;
        let mut out = Vec::with_capacity(len);
        for tile in 0..self.tile_count() {
            self.read_window_tile(tile)?;
            out.extend_from_slice(&self.inner.state);
        }
        if self.pending_norm != 1.0 {
            let s = Complex64::new(self.pending_norm, 0.0);
            for amp in &mut out {
                *amp *= s;
            }
        }
        Ok(out)
    }

    /// Classical bits recorded by measurements, in bit order.
    pub fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    /// Full-shard streaming passes issued so far, gate and measurement alike.
    pub fn passes(&self) -> u64 {
        self.passes
    }

    /// Tiles read through the staging buffers so far.
    pub fn tiles_streamed(&self) -> u64 {
        self.tiles_streamed
    }

    /// Run-scoped storage directory; removed when the state drops.
    pub fn storage_dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TiledStatevector {
    fn drop(&mut self) {
        self.file = None;
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_dir(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::distributed_statevector::DistributedStatevectorBackend;
    use crate::circuit::builder::CircuitBuilder;
    use crate::circuit::{Circuit, ClassicalCondition, smallvec};
    use crate::distributed::DistributedContext;

    const SEED: u64 = 42;
    const TOL: f64 = 1e-10;

    fn tiled_run(circuit: &Circuit, window_qubits: usize) -> TiledStatevector {
        let mut config = TileConfig::new(window_qubits);
        config.max_pass_rate = 1.0;
        let mut state =
            TiledStatevector::new(circuit.num_qubits, circuit.num_classical_bits, SEED, config)
                .expect("tiled state");
        state.run(&circuit.instructions).expect("tiled run");
        state
    }

    fn dense_reference(circuit: &Circuit) -> Vec<Complex64> {
        let mut backend = StatevectorBackend::new(SEED);
        backend
            .init(circuit.num_qubits, circuit.num_classical_bits)
            .expect("dense init");
        for instruction in &circuit.instructions {
            backend.apply(instruction).expect("dense apply");
        }
        backend.export_statevector().expect("dense export")
    }

    fn serial_distributed_reference(circuit: &Circuit) -> (Vec<Complex64>, Vec<bool>) {
        let mut backend = DistributedStatevectorBackend::new(DistributedContext::serial(), SEED);
        backend
            .init(circuit.num_qubits, circuit.num_classical_bits)
            .expect("serial init");
        for instruction in &circuit.instructions {
            backend.apply(instruction).expect("serial apply");
        }
        let state = backend.export_statevector().expect("serial export");
        (state, backend.classical_results().to_vec())
    }

    fn assert_states_match(expected: &[Complex64], actual: &[Complex64], label: &str) {
        assert_eq!(expected.len(), actual.len(), "{label}: length mismatch");
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (e - a).norm() < TOL,
                "{label}: amplitude {i} expected {e}, got {a}"
            );
        }
    }

    #[test]
    fn fresh_state_is_the_zero_ket() {
        let mut state = TiledStatevector::new(5, 0, SEED, TileConfig::new(2)).expect("state");
        let amps = state.export_statevector().expect("export");
        assert_eq!(amps.len(), 32);
        assert!((amps[0] - Complex64::new(1.0, 0.0)).norm() < TOL);
        assert!(amps[1..].iter().all(|a| a.norm() < TOL));
    }

    #[test]
    fn x_on_qubit_zero_sets_index_one() {
        let mut b = CircuitBuilder::new(4);
        b.x(0);
        let mut state = tiled_run(&b.build(), 2);
        let amps = state.export_statevector().expect("export");
        assert!((amps[1] - Complex64::new(1.0, 0.0)).norm() < TOL);
    }

    fn mixed_unitary_circuit(n: usize) -> Circuit {
        let x = Gate::X.matrix_2x2();
        let mut b = CircuitBuilder::new(n);
        for q in 0..n {
            b.h(q);
        }
        for q in 0..n - 1 {
            b.cx(q, q + 1);
        }
        b.t(0)
            .s(n - 1)
            .rz(0.37, n / 2)
            .rx(0.9, 1)
            .ry(-0.4, n - 2)
            .cz(0, n - 1)
            .rzz(0.7, 1, n - 1)
            .swap(0, n - 1)
            .swap(n - 2, n - 1)
            .cx(n - 1, 0)
            .cphase(0.5, n - 2, 1)
            .mcu(x, &[0, n - 1], n - 2);
        for q in 0..n {
            b.h(q);
        }
        b.build()
    }

    #[test]
    fn unitary_circuit_matches_dense_at_every_window_split() {
        let n = 6;
        let circuit = mixed_unitary_circuit(n);
        let expected = dense_reference(&circuit);
        for w in 1..=n {
            let mut state = tiled_run(&circuit, w);
            let actual = state.export_statevector().expect("export");
            assert_states_match(&expected, &actual, &format!("window {w}"));
        }
    }

    #[test]
    fn diagonal_wall_joins_the_open_pass() {
        let n = 6;
        let mut b = CircuitBuilder::new(n);
        b.h(0)
            .h(1)
            .t(0)
            .t(4)
            .t(5)
            .cz(0, 5)
            .cz(4, 5)
            .rzz(0.3, 2, 5)
            .rz(1.1, 4)
            .cphase(0.25, 3, 4);
        let circuit = b.build();
        let mut state = tiled_run(&circuit, 2);
        assert_eq!(state.passes(), 1, "one streaming pass for the whole wall");
        assert_eq!(state.tiles_streamed(), 16, "each tile read once");
        let actual = state.export_statevector().expect("export");
        assert_states_match(&dense_reference(&circuit), &actual, "diagonal wall");
    }

    #[test]
    fn tile_bit_controls_are_free() {
        let mut b = CircuitBuilder::new(6);
        b.h(5).cx(5, 0).cz(5, 1).rzz(0.4, 5, 0);
        let circuit = b.build();
        let mut state = tiled_run(&circuit, 2);
        assert_eq!(state.passes(), 2, "one pair pass for H, one window pass");
        let actual = state.export_statevector().expect("export");
        assert_states_match(&dense_reference(&circuit), &actual, "tile-bit controls");
    }

    #[test]
    fn measurement_reset_and_conditional_match_the_serial_distributed_backend() {
        let n = 5;
        let mut b = CircuitBuilder::new_with_classical(n, 2);
        b.h(0)
            .cx(0, 4)
            .rx(0.3, 2)
            .measure(4, 0)
            .conditional(ClassicalCondition::BitIsOne(0), Gate::X, &[1])
            .reset(0)
            .ry(0.8, 3)
            .measure(1, 1);
        let circuit = b.build();
        let (expected_state, expected_bits) = serial_distributed_reference(&circuit);
        for w in 1..=n {
            let mut state = tiled_run(&circuit, w);
            assert_eq!(
                state.classical_results(),
                &expected_bits[..],
                "window {w}: classical bits"
            );
            let actual = state.export_statevector().expect("export");
            assert_states_match(&expected_state, &actual, &format!("window {w}"));
        }
    }

    #[test]
    fn tile_thrashing_schedule_is_rejected_before_any_streaming() {
        let mut b = CircuitBuilder::new(5);
        b.h(3).h(4).h(3).h(4);
        let circuit = b.build();
        let mut state = TiledStatevector::new(5, 0, SEED, TileConfig::new(2)).expect("state");
        let err = state.run(&circuit.instructions).unwrap_err();
        match err {
            PrismError::BackendUnsupported { operation, .. } => {
                assert!(
                    operation.contains("pass rate"),
                    "unexpected message: {operation}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(state.tiles_streamed(), 0, "rejection precedes storage I/O");

        let mut config = TileConfig::new(2);
        config.max_pass_rate = 1.0;
        let mut state = TiledStatevector::new(5, 0, SEED, config).expect("state");
        state.run(&circuit.instructions).expect("override run");
        let actual = state.export_statevector().expect("export");
        assert_states_match(&dense_reference(&circuit), &actual, "override");
    }

    #[test]
    fn short_schedules_run_under_the_default_threshold() {
        let mut b = CircuitBuilder::new(4);
        b.h(3);
        let mut state = TiledStatevector::new(4, 0, SEED, TileConfig::new(2)).expect("state");
        state.run(&b.build().instructions).expect("run");
    }

    #[test]
    fn invalid_window_geometry_is_rejected() {
        assert!(TiledStatevector::new(4, 0, SEED, TileConfig::new(0)).is_err());
        assert!(TiledStatevector::new(4, 0, SEED, TileConfig::new(5)).is_err());
        assert!(TiledStatevector::new(0, 0, SEED, TileConfig::new(1)).is_err());
    }

    #[test]
    fn fused_gate_spanning_a_tile_bit_is_rejected() {
        let z = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let identity = [
            [one, z, z, z],
            [z, one, z, z],
            [z, z, one, z],
            [z, z, z, one],
        ];
        let spanning = Instruction::Gate {
            gate: Gate::Fused2q(Box::new(identity)),
            targets: smallvec![0, 3],
        };
        let mut state = TiledStatevector::new(4, 0, SEED, TileConfig::new(2)).expect("state");
        let err = state.run(std::slice::from_ref(&spanning)).unwrap_err();
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));

        let local = Instruction::Gate {
            gate: Gate::Fused2q(Box::new(identity)),
            targets: smallvec![0, 1],
        };
        state
            .run(std::slice::from_ref(&local))
            .expect("window-local fused gate");
    }

    #[test]
    fn storage_is_removed_on_drop() {
        let mut b = CircuitBuilder::new(4);
        b.h(0);
        let circuit = b.build();
        let mut state = TiledStatevector::new(4, 0, SEED, TileConfig::new(2)).expect("state");
        state.run(&circuit.instructions).expect("run");
        let dir = state.storage_dir().to_path_buf();
        assert!(dir.exists());
        drop(state);
        assert!(!dir.exists());
    }
}

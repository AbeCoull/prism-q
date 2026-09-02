//! Exact density-matrix backend.
//!
//! Stores the full density operator `rho` for `n` qubits as a `2^(2n)` amplitude
//! buffer laid out row-major: buffer index `(r << n) | c` holds `<r|rho|c>`. That
//! layout is isomorphic to a `2n`-qubit statevector whose high `n` qubits index
//! the ket (row `r`) and whose low `n` qubits index the bra (column `c`). Gate
//! application therefore reuses the validated statevector kernels directly: a
//! unitary `U` on the ket register yields the left product `U rho`, and the same
//! `U` applied to the bra register on a conjugated buffer yields the right product
//! `rho U^dagger`, giving `U rho U^dagger` with no gate math of its own.
//!
//! # Memory layout
//!
//! Memory is `16 * 4^n` bytes (`4^n` `Complex64` entries), so the practical
//! ceiling is about 14 qubits on a 16 GiB host and 15 on a 32 GiB host. With a
//! device attached through [`DensityMatrixBackend::with_gpu`] the buffer lives in
//! VRAM instead, where an 11 GiB card holds 13 qubits (1 GiB at 13, 4 GiB at 14
//! plus scratch), and every sweep runs as a kernel over the embedded buffer. This
//! backend is explicit-dispatch only and is never chosen by `Auto`.
//!
//! # Gate support
//!
//! Supported: exact unitary evolution, basis-state probabilities, the one-qubit
//! reduced density matrix, projective measurement with stochastic collapse,
//! reset, classically-conditioned gates, exact one-qubit Kraus channels
//! (`apply_1q_kraus`), exact two-qubit Kraus channels (`apply_2q_kraus`, with
//! `apply_2q_depolarizing` taking the twirled closed form instead), and
//! exact `Tr(rho P)` expectation (`expectation_pauli`, reachable through
//! `pauli_expectations`). Fused gates are accepted, so `sim` fuses for this
//! backend, and at the `2n` width its buffer actually costs rather than at the
//! circuit width. `QftBlock` carries qubit indices outside the instruction
//! targets and is remapped onto the ket register before the left product; the
//! tiled shapes (`MultiFused`, `Multi2q`) apply their constituent gates one at
//! a time instead. See [`Backend::supports_fused_gates`] for the ordering
//! contract that requires it. Diagonal gates do not take the two-product route
//! at all: their two factors combine into one table, so `Rzz` and the diagonal
//! batches (`BatchPhase`, `BatchRzz`, `DiagonalBatch`) each sweep the buffer
//! once.
//!
//! # When to prefer this backend
//!
//! - Exact noise-channel evolution: one run yields the exact mixed state,
//!   where trajectory averaging converges as `1/sqrt(shots)`. Selecting it
//!   with a noise model attached routes every `Simulate` terminal except the two
//!   gradient terminals to that one evolution.
//! - Mixed-state diagnostics: purity and exact `Tr(rho P)` observables.
//!
//! # When NOT to use this backend
//!
//! - Pure-state circuits; a statevector covers twice the qubits in the same
//!   memory.
//! - Qubit counts past the `4^n` ceiling; trajectory sampling on a pure-state
//!   backend scales further.
//! - Noisy circuits with mid-circuit measurement or classical conditioning.
//!   The mixture holds every branch at once, so per-shot feedback cannot be
//!   replayed from it and the noisy terminals reject those shapes.
//! - Gradients under noise. Both gradient terminals decline a noise model, for
//!   different reasons: the adjoint backpropagates against a pure state and a
//!   channel has no reverse evolution to walk, while parameter shift is exact
//!   on a mixture and simply not wired.

use std::borrow::Cow;
#[cfg(feature = "gpu")]
use std::sync::Arc;

use num_complex::Complex64;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use crate::backend::simd;
use crate::backend::statevector::{StatevectorBackend, insert_zero_bit, kernels};
use crate::backend::{Backend, NORM_CLAMP_MIN};
use crate::circuit::{ClassicalCondition, Instruction};
use crate::error::Result;
use crate::gates::{DiagEntry, Gate, McuData, diag_entries_phase, mat_mul_4x4};
#[cfg(feature = "gpu")]
use crate::gpu::{GpuContext, GpuState, kernels::density as dk};
use crate::sim::i_pow;
use crate::sim::unified_pauli::PauliTerm;

/// Compile a one-qubit Kraus set into the 4x4 superoperator acting on the
/// `(row-bit, col-bit)` block of `rho`, where block index `i = 2*a + b` orders
/// `(row-bit a, col-bit b)`:
/// `S[2a+b][2a'+b'] = sum_k K_k[a][a'] * conj(K_k[b][b'])`.
///
/// A single-element set `[U]` gives the unitary sandwich `U rho U^dagger`.
fn block_superoperator(kraus: &[[[Complex64; 2]; 2]]) -> [[Complex64; 4]; 4] {
    let mut s = [[Complex64::new(0.0, 0.0); 4]; 4];
    for k in kraus {
        for a in 0..2 {
            for b in 0..2 {
                for ap in 0..2 {
                    for bp in 0..2 {
                        s[2 * a + b][2 * ap + bp] += k[a][ap] * k[b][bp].conj();
                    }
                }
            }
        }
    }
    s
}

/// Collapse one tile of `rho` onto the `outcome` subspace of the qubit whose
/// row and column bits in the buffer index are `rmask` and `cmask`, scaling the
/// survivors by `scale`. `base` is the tile's buffer index, and the tile must
/// not straddle `rmask`, so one test fixes its row class: a tile on the
/// eliminated row is a contiguous zero fill, and only the surviving row pays a
/// per-entry column test.
fn project_tile(
    tile: &mut [Complex64],
    base: usize,
    rmask: usize,
    cmask: usize,
    outcome: bool,
    scale: Complex64,
) {
    if ((base & rmask) != 0) != outcome {
        simd::zero_slice(tile);
        return;
    }
    let zero = Complex64::new(0.0, 0.0);
    for (j, amp) in tile.iter_mut().enumerate() {
        *amp = if (((base + j) & cmask) != 0) == outcome {
            *amp * scale
        } else {
            zero
        };
    }
}

/// Fold the row-set half of one `2 * rmask` block of `rho` onto the row-clear
/// half, then clear it. `r0` and `r1` are the two halves, or matching sub-tiles
/// of them, and `cmask` is the qubit's column bit.
fn reset_fold_pair(r0: &mut [Complex64], r1: &mut [Complex64], cmask: usize) {
    let zero = Complex64::new(0.0, 0.0);
    for (j, amp) in r0.iter_mut().enumerate() {
        *amp = if j & cmask == 0 {
            *amp + r1[j | cmask]
        } else {
            zero
        };
    }
    simd::zero_slice(r1);
}

/// Tile size for a parallel sweep over `rho` whose body reads the qubit's row
/// class from the tile's own base index. Both `rmask` and `2 * cmask` are powers
/// of two and `rmask >= 2 * cmask`, so the result is a multiple of the column run
/// that divides `rmask`: a tile can never straddle the row bit.
#[cfg(feature = "parallel")]
fn row_aligned_tile(cmask: usize, rmask: usize) -> usize {
    (cmask << 1).max(crate::backend::MIN_PAR_ELEMS).min(rmask)
}

/// Base buffer index of block `m`, expanding the four block bit positions out
/// of the compacted index. `positions` must be ascending.
#[inline(always)]
fn block_base(m: usize, positions: &[usize; 4]) -> usize {
    let mut base = m;
    for &pos in positions {
        base = insert_zero_bit(base, pos);
    }
    base
}

/// Index into the 16-entry superoperator diagonal for the amplitude at buffer
/// index `idx`: `4 * tr + tc` with `tr` read from the ket bits and `tc` from
/// the bra bits of `(q0, q1)`, matching the flat order in
/// [`DensityMatrixBackend::block_layout`].
#[inline(always)]
fn diag_slot(idx: usize, q0: usize, q1: usize, n: usize) -> usize {
    ((idx >> (q0 + n)) & 1) << 3
        | ((idx >> (q1 + n)) & 1) << 2
        | ((idx >> q0) & 1) << 1
        | ((idx >> q1) & 1)
}

fn conjugate_2x2(m: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [m[0][0].conj(), m[0][1].conj()],
        [m[1][0].conj(), m[1][1].conj()],
    ]
}

fn conjugate_4x4(m: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let mut out = *m;
    for row in &mut out {
        for entry in row {
            *entry = entry.conj();
        }
    }
    out
}

/// The gate's 2x2 matrix, or `None` for anything else. `Gate::num_qubits` is
/// not a usable test here: it reports 1 for a `BatchPhase` with no phases, a
/// single-qubit `DiagonalBatch`, and `QftBlock { num: 1 }`, none of which
/// `matrix_2x2` accepts.
fn matrix_1q(gate: &Gate) -> Option<[[Complex64; 2]; 2]> {
    match gate {
        Gate::Id
        | Gate::X
        | Gate::Y
        | Gate::Z
        | Gate::H
        | Gate::S
        | Gate::Sdg
        | Gate::T
        | Gate::Tdg
        | Gate::SX
        | Gate::SXdg
        | Gate::Rx(_)
        | Gate::Ry(_)
        | Gate::Rz(_)
        | Gate::P(_)
        | Gate::Fused(_) => Some(gate.matrix_2x2()),
        _ => None,
    }
}

/// `conj(gate)` as a native gate variant, which is what the bra register of
/// `rho -> U rho U^dagger` needs. `Cx`, `Cz`, and `Swap` are real, so they are
/// their own conjugate. `None` means the variant has no native conjugate form
/// and the caller falls back to conjugating the buffer around the gate.
fn conjugate_gate(gate: &Gate) -> Option<Gate> {
    match gate {
        Gate::Cx | Gate::Cz | Gate::Swap => Some(gate.clone()),
        Gate::Rzz(theta) => Some(Gate::Rzz(-*theta)),
        Gate::Fused2q(mat) => Some(Gate::Fused2q(Box::new(conjugate_4x4(mat)))),
        Gate::Cu(mat) => Some(Gate::Cu(Box::new(conjugate_2x2(mat)))),
        Gate::Mcu(data) => Some(Gate::Mcu(Box::new(McuData {
            mat: conjugate_2x2(&data.mat),
            num_controls: data.num_controls,
        }))),
        _ => None,
    }
}

/// The gate with every qubit index stored inside its payload shifted onto the
/// ket register. Offsetting the instruction targets is not enough for
/// `QftBlock`, whose whole range lives in the variant. Borrows the gate when it
/// carries no such index.
///
/// `MultiFused` and `Multi2q` are deliberately absent: shifting a `Multi2q`
/// payload reorders it, so both apply their constituents directly. The diagonal
/// batches are absent for a different reason, that
/// [`DensityMatrixBackend::apply_diagonal_sandwich`] takes them before the
/// two-product route is reached.
/// The batch payload as a flat [`DiagEntry`] list, or `None` for a gate that is
/// not a diagonal batch. `BatchPhase` keeps its shared control in the
/// instruction targets rather than in the payload, so it arrives separately.
fn diagonal_batch_entries(gate: &Gate, targets: &[usize]) -> Option<Vec<DiagEntry>> {
    match gate {
        Gate::BatchPhase(data) => Some(
            data.phases
                .iter()
                .map(|&(target, phase)| DiagEntry::Phase2q {
                    q0: targets[0],
                    q1: target,
                    phase,
                })
                .collect(),
        ),
        Gate::BatchRzz(data) => Some(
            data.edges
                .iter()
                .map(|&(q0, q1, theta)| DiagEntry::Parity2q {
                    q0,
                    q1,
                    same: Complex64::from_polar(1.0, -theta / 2.0),
                    diff: Complex64::from_polar(1.0, theta / 2.0),
                })
                .collect(),
        ),
        Gate::DiagonalBatch(data) => Some(data.entries.clone()),
        _ => None,
    }
}

fn ket_register_gate(gate: &Gate, n: usize) -> Cow<'_, Gate> {
    match gate {
        Gate::QftBlock { start, num } => Cow::Owned(Gate::QftBlock {
            start: start + n as u8,
            num: *num,
        }),
        _ => Cow::Borrowed(gate),
    }
}

/// Reject a mixture the device cannot hold before anything is allocated: the
/// `4^n` buffer at 16 bytes per entry plus the `2^n` diagonal scratch. A
/// context that cannot report free memory falls through to the allocation,
/// whose own error stays authoritative.
#[cfg(feature = "gpu")]
fn check_device_budget(context: &GpuContext, num_qubits: usize) -> Result<()> {
    if 2 * num_qubits >= usize::BITS as usize - 4 {
        return Ok(());
    }
    let Ok(free) = context.vram_available() else {
        return Ok(());
    };
    let needed = (1usize << (2 * num_qubits)) * 16 + (1usize << num_qubits) * 8;
    if needed > free {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: "density_matrix-gpu".to_string(),
            reason: format!(
                "circuit has {num_qubits} qubits needing {} MiB of device memory for the \
                 4^n mixture, exceeding the {} MiB free on the GPU",
                needed >> 20,
                free >> 20
            ),
        });
    }
    Ok(())
}

/// The channel entry points are infallible by signature, so a device launch
/// that fails under one has nowhere to report and stops the run instead.
#[cfg(feature = "gpu")]
fn launched<T>(result: Result<T>) -> T {
    result.unwrap_or_else(|e| panic!("density-matrix device kernel failed: {e}"))
}

/// Exact density-matrix simulator. See the module docs for the state layout.
pub struct DensityMatrixBackend {
    num_qubits: usize,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
    sv: StatevectorBackend,
    #[cfg(feature = "gpu")]
    gpu_context: Option<Arc<GpuContext>>,
}

impl DensityMatrixBackend {
    /// Create a new density-matrix backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            sv: StatevectorBackend::new(seed),
            #[cfg(feature = "gpu")]
            gpu_context: None,
        }
    }

    /// Hold the mixture on the device bound to `context`.
    ///
    /// [`Backend::init`] then allocates the `4^n` buffer in device memory after a
    /// budget check against the free VRAM, and every sweep runs as a kernel. A
    /// device that cannot hold the state is an error at `init`, never a host
    /// fallback. The channel entry points keep their infallible signatures, so
    /// a kernel launch that fails under one of them panics.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, context: Arc<GpuContext>) -> Self {
        self.gpu_context = Some(context.clone());
        self.sv = self.sv.with_gpu(context);
        self
    }

    /// The device state with its context, when the mixture is resident there.
    #[cfg(feature = "gpu")]
    fn device(&mut self) -> Option<(Arc<GpuContext>, &mut GpuState)> {
        let gpu = self.sv.gpu_state_mut()?;
        let ctx = gpu.context().clone();
        Some((ctx, gpu))
    }

    /// The full `4^n` buffer, row-major as the module docs lay it out. Reads
    /// back from the device when the mixture is resident there.
    pub fn density_matrix(&self) -> Result<Vec<Complex64>> {
        self.sv.export_statevector()
    }

    /// Purity `Tr(rho^2)`, equal to `1` for a pure state and less otherwise.
    pub fn purity(&self) -> f64 {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.sv.gpu_state() {
            return launched(dk::norm_sqr(gpu.context(), gpu, self.num_qubits));
        }
        crate::backend::state_norm_sqr(&self.sv.state)
    }

    #[inline]
    fn dim(&self) -> usize {
        1usize << self.num_qubits
    }

    fn conjugate_buffer(&mut self) -> Result<()> {
        #[cfg(feature = "gpu")]
        {
            let n = self.num_qubits;
            if let Some((ctx, gpu)) = self.device() {
                return dk::conjugate(&ctx, gpu, n);
            }
        }
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            if self.sv.state.len() >= (1 << crate::backend::PARALLEL_THRESHOLD_QUBITS) {
                self.sv
                    .state
                    .par_iter_mut()
                    .for_each(|amp| *amp = amp.conj());
                return Ok(());
            }
        }
        for amp in self.sv.state.iter_mut() {
            *amp = amp.conj();
        }
        Ok(())
    }

    /// A two-qubit matrix on the embedded buffer, on whichever side holds it.
    fn fused_2q(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) -> Result<()> {
        #[cfg(feature = "gpu")]
        if let Some((ctx, gpu)) = self.device() {
            return crate::gpu::kernels::dense::launch_apply_fused_2q(&ctx, gpu, q0, q1, mat);
        }
        self.sv.apply_fused_2q(q0, q1, mat);
        Ok(())
    }

    /// Apply a compiled one-qubit block superoperator in a single buffer pass.
    ///
    /// The `(row-bit, col-bit)` block of `qubit` is the two-qubit subspace
    /// `(qubit + n, qubit)` of the embedded `2n`-qubit statevector, so the
    /// statevector two-qubit kernel applies `S` directly. That kernel indexes
    /// its matrix as `2 * bit(q0) + bit(q1)`, matching the block index `2a + b`
    /// once the row bit is passed as `q0`.
    fn apply_block_superoperator(&mut self, qubit: usize, s: &[[Complex64; 4]; 4]) -> Result<()> {
        let n = self.num_qubits;
        self.fused_2q(qubit + n, qubit, s)
    }

    /// Evolve `rho -> U rho U^dagger` for the unitary `gate` on `targets`.
    ///
    /// `U` applies to the ket register (targets and payload indices offset by
    /// `n`, see [`ket_register_gate`]) for the left product `U rho`, and
    /// `conj(U)` to the bra register (indices unchanged) for the right product
    /// `rho U^dagger`. Variants with no native conjugate form fall back to
    /// conjugating the whole buffer around the gate, which costs two extra
    /// passes.
    ///
    /// One-qubit gates take [`DensityMatrixBackend::apply_1q_sandwich`], `Rzz`
    /// takes [`DensityMatrixBackend::apply_rzz_sandwich`], and the diagonal
    /// batches take [`DensityMatrixBackend::apply_diagonal_sandwich`]. All
    /// three fold both products into one pass rather than taking a conjugate
    /// form here.
    ///
    /// `Multi2q` carries a gate list that the statevector kernel partitions by
    /// cache tier and runs one tier at a time, which preserves application
    /// order only while the whole list sits in one tier. Fusion guarantees that
    /// against the circuit's own qubit indices, and the `+n` shift onto the ket
    /// register moves gates across the tier bounds, so the ket half applies its
    /// constituents one at a time. The bra half keeps the circuit's indices, so
    /// it batches through the tiled pass whenever
    /// [`kernels::multi_2q_single_tier`] holds, and falls back to
    /// per-constituent application otherwise. `MultiFused` entries
    /// are one per qubit and commute, so its list is order independent; it takes
    /// the same treatment because the one-qubit sandwich is cheaper than the
    /// tiled pass here.
    fn apply_unitary(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        let n = self.num_qubits;

        if let Some(mat) = matrix_1q(gate) {
            return self.apply_1q_sandwich(targets[0], &mat);
        }

        match gate {
            Gate::MultiFused(data) => {
                for (qubit, mat) in data.gates.iter() {
                    self.apply_1q_sandwich(*qubit, mat)?;
                }
                return Ok(());
            }
            Gate::Rzz(theta) => {
                return self.apply_rzz_sandwich(targets[0], targets[1], *theta);
            }
            Gate::Multi2q(data) => {
                for &(q0, q1, ref mat) in data.gates.iter() {
                    self.fused_2q(q0 + n, q1 + n, mat)?;
                }
                if !self.sv.is_gpu_resident() && kernels::multi_2q_single_tier(&data.gates) {
                    let conjugated: Vec<(usize, usize, [[Complex64; 4]; 4])> = data
                        .gates
                        .iter()
                        .map(|&(q0, q1, ref mat)| (q0, q1, conjugate_4x4(mat)))
                        .collect();
                    self.sv.apply_multi_2q(&conjugated);
                } else {
                    for &(q0, q1, ref mat) in data.gates.iter() {
                        self.fused_2q(q0, q1, &conjugate_4x4(mat))?;
                    }
                }
                return Ok(());
            }
            _ => {}
        }

        if let Some(entries) = diagonal_batch_entries(gate, targets) {
            return self.apply_diagonal_sandwich(&entries);
        }

        let ket_targets: SmallVec<[usize; 4]> = targets.iter().map(|&t| t + n).collect();
        self.sv.apply(&Instruction::Gate {
            gate: ket_register_gate(gate, n).into_owned(),
            targets: ket_targets,
        })?;

        let bra_targets: SmallVec<[usize; 4]> = targets.iter().copied().collect();
        if let Some(conjugate) = conjugate_gate(gate) {
            return self.sv.apply(&Instruction::Gate {
                gate: conjugate,
                targets: bra_targets,
            });
        }

        self.conjugate_buffer()?;
        self.sv.apply(&Instruction::Gate {
            gate: gate.clone(),
            targets: bra_targets,
        })?;
        self.conjugate_buffer()
    }

    /// Evolve `rho -> U rho U^dagger` for a one-qubit `U`.
    ///
    /// Above the parallel threshold, and always on the device, the two products
    /// compile to one block superoperator and sweep the buffer once; below it
    /// the superoperator's dense 4x4 block loses to two cheap register passes.
    fn apply_1q_sandwich(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        let n = self.num_qubits;
        if self.sv.is_gpu_resident() || 2 * n >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            let s = block_superoperator(&[*matrix]);
            return self.apply_block_superoperator(qubit, &s);
        }
        self.sv.apply_1q_matrix(qubit + n, matrix)?;
        self.sv.apply_1q_matrix(qubit, &conjugate_2x2(matrix))
    }

    /// Evolve `rho -> R rho R^dagger` for `R = Rzz(theta)` in one buffer pass.
    ///
    /// Both factors are diagonal, so the entry at `(r, c)` picks up
    /// `phase(p_r) * conj(phase(p_c))` where `p` is the parity of the target
    /// pair in that register and `phase` is the statevector kernel's
    /// `[exp(-i theta/2), exp(+i theta/2)]`. Conjugate phases cancel wherever
    /// the two registers agree on parity, which is half the sixteen entry
    /// classes, and the rest carry `exp(-i theta)` or its conjugate. The
    /// generic route's two register passes therefore collapse onto the single
    /// contiguous pass [`DensityMatrixBackend::kraus_2q_diagonal_sweep`]
    /// already runs for a diagonal channel.
    fn apply_rzz_sandwich(&mut self, q0: usize, q1: usize, theta: f64) -> Result<()> {
        let phase = [
            Complex64::from_polar(1.0, -theta / 2.0),
            Complex64::from_polar(1.0, theta / 2.0),
        ];
        let mut diag = [Complex64::new(1.0, 0.0); 16];
        for tr in 0..4usize {
            for tc in 0..4usize {
                let pr = ((tr >> 1) ^ tr) & 1;
                let pc = ((tc >> 1) ^ tc) & 1;
                diag[4 * tr + tc] = phase[pr] * phase[pc].conj();
            }
        }
        self.kraus_2q_diagonal_sweep(&diag, q0, q1)
    }

    /// Evolve `rho -> D rho D^dagger` for a diagonal batch `D` in one buffer
    /// pass.
    ///
    /// A diagonal `D` scales `<r|rho|c>` by `f(r) * conj(f(c))`, where `f` is
    /// the combined phase the payload puts on a basis index. The embedded
    /// layout splits the buffer index into exactly those two operands, `r` in
    /// the high `n` bits and `c` in the low `n`, so one table of `2^n` phases
    /// serves both registers and the sweep reads it twice per amplitude. The
    /// table costs `2^n` against the `4^n` buffer it saves three passes on: the
    /// generic route has no conjugate form for these variants and pays two
    /// register passes plus two buffer conjugations.
    ///
    /// Entry order does not matter, all of them being diagonal, so the tier
    /// reordering the tiled payloads have to avoid cannot arise here.
    fn apply_diagonal_sandwich(&mut self, entries: &[DiagEntry]) -> Result<()> {
        let n = self.num_qubits;
        let d = self.dim();
        let ket: Vec<Complex64> = (0..d).map(|r| diag_entries_phase(r, entries)).collect();

        #[cfg(feature = "gpu")]
        if let Some((ctx, gpu)) = self.device() {
            return dk::diagonal_sandwich(&ctx, gpu, n, &ket);
        }

        let bra: Vec<Complex64> = ket.iter().map(|f| f.conj()).collect();

        #[cfg(feature = "parallel")]
        if 2 * n >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            use crate::backend::MIN_PAR_ELEMS;
            use rayon::prelude::*;

            self.sv
                .state
                .par_iter_mut()
                .enumerate()
                .with_min_len(MIN_PAR_ELEMS)
                .for_each(|(idx, amp)| {
                    *amp *= ket[idx >> n] * bra[idx & (d - 1)];
                });
            return Ok(());
        }

        for (idx, amp) in self.sv.state.iter_mut().enumerate() {
            *amp *= ket[idx >> n] * bra[idx & (d - 1)];
        }
        Ok(())
    }

    /// `P(qubit = |1>) = Tr(P_1 rho)`, the sum of the diagonal `rho` entries
    /// whose row index has `qubit` set.
    fn prob_one(&self, qubit: usize) -> Result<f64> {
        let d = self.dim();
        let bit = 1usize << qubit;
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.sv.gpu_state() {
            let diag = dk::diagonal(gpu.context(), gpu, self.num_qubits)?;
            let p1: f64 = diag
                .iter()
                .enumerate()
                .filter(|(r, _)| r & bit != 0)
                .map(|(_, p)| p)
                .sum();
            return Ok(p1.clamp(0.0, 1.0));
        }
        let mut p1 = 0.0;
        for r in 0..d {
            if r & bit != 0 {
                p1 += self.sv.state[r * d + r].re;
            }
        }
        Ok(p1.clamp(0.0, 1.0))
    }

    /// Sample and project qubit `qubit`, recording the outcome in
    /// `classical_bit`. Collapses `rho -> P_m rho P_m / p_m`, which in the
    /// embedded layout zeroes every entry whose row or column disagrees with
    /// the outcome on `qubit` and rescales the survivors to unit trace.
    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) -> Result<()> {
        let p1 = self.prob_one(qubit)?;
        let u: f64 = self.rng.random();
        let outcome = u < p1;
        self.classical_bits[classical_bit] = outcome;
        let p = if outcome { p1 } else { 1.0 - p1 };
        self.project(qubit, outcome, p)
    }

    /// Deterministic reset `rho -> |0><0| (x) tr_q rho`: fold the block with
    /// `qubit` set on both row and column into the block with it clear on both,
    /// then zero the three sibling entries that still touch `qubit`. The four
    /// entry classes are contiguous runs of the buffer, so the pass walks
    /// `2 * rmask` blocks rather than scattering per block base.
    fn apply_reset(&mut self, qubit: usize) -> Result<()> {
        let n = self.num_qubits;
        #[cfg(feature = "gpu")]
        if let Some((ctx, gpu)) = self.device() {
            return dk::reset(&ctx, gpu, n, qubit);
        }
        let rmask = 1usize << (qubit + n);
        let cmask = 1usize << qubit;
        let block_size = rmask << 1;

        #[cfg(feature = "parallel")]
        if 2 * n >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            use crate::backend::chunk_min_len;
            use rayon::prelude::*;

            if self.sv.state.len() / block_size >= 4 {
                self.sv
                    .state
                    .par_chunks_mut(block_size)
                    .with_min_len(chunk_min_len(block_size))
                    .for_each(|block| {
                        let (r0, r1) = block.split_at_mut(rmask);
                        reset_fold_pair(r0, r1, cmask);
                    });
                return Ok(());
            }

            let tile = row_aligned_tile(cmask, rmask);
            for block in self.sv.state.chunks_mut(block_size) {
                let (r0, r1) = block.split_at_mut(rmask);
                r0.par_chunks_mut(tile)
                    .zip(r1.par_chunks_mut(tile))
                    .for_each(|(t0, t1)| reset_fold_pair(t0, t1, cmask));
            }
            return Ok(());
        }

        for block in self.sv.state.chunks_mut(block_size) {
            let (r0, r1) = block.split_at_mut(rmask);
            reset_fold_pair(r0, r1, cmask);
        }
        Ok(())
    }

    /// Project `rho` onto the `outcome` subspace of `qubit` with outcome
    /// probability `p`, renormalizing the survivors to unit trace.
    fn project(&mut self, qubit: usize, outcome: bool, p: f64) -> Result<()> {
        let n = self.num_qubits;
        let rmask = 1usize << (qubit + n);
        let cmask = 1usize << qubit;
        let scale = Complex64::new(1.0 / p.clamp(NORM_CLAMP_MIN, 1.0), 0.0);

        #[cfg(feature = "gpu")]
        if let Some((ctx, gpu)) = self.device() {
            return dk::project(&ctx, gpu, n, qubit, outcome, scale.re);
        }

        #[cfg(feature = "parallel")]
        if 2 * n >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            use crate::backend::chunk_min_len;
            use rayon::prelude::*;

            let tile = row_aligned_tile(cmask, rmask);
            self.sv
                .state
                .par_chunks_mut(tile)
                .with_min_len(chunk_min_len(tile))
                .enumerate()
                .for_each(|(t, chunk)| {
                    project_tile(chunk, t * tile, rmask, cmask, outcome, scale);
                });
            return Ok(());
        }

        for (t, chunk) in self.sv.state.chunks_mut(rmask).enumerate() {
            project_tile(chunk, t * rmask, rmask, cmask, outcome, scale);
        }
        Ok(())
    }

    fn apply_conditional(
        &mut self,
        condition: &ClassicalCondition,
        gate: &Gate,
        targets: &[usize],
    ) -> Result<()> {
        if condition.evaluate(&self.classical_bits) {
            self.apply_unitary(gate, targets)?;
        }
        Ok(())
    }

    /// Apply a general single-qubit channel `rho -> sum_k K_k rho K_k^dagger`
    /// on `qubit`. The Kraus set is compiled once into a 4x4 block
    /// superoperator acting on each `(row-bit, col-bit)` block of `rho`, so the
    /// buffer is swept once with no per-element allocation.
    pub fn apply_1q_kraus(&mut self, qubit: usize, kraus: &[[[Complex64; 2]; 2]]) {
        let s = block_superoperator(kraus);
        self.apply_block_superoperator_infallible(qubit, &s);
    }

    /// [`DensityMatrixBackend::apply_block_superoperator`] for the entry points
    /// that carry no error channel: the host path cannot fail and a device
    /// launch failure stops the run.
    fn apply_block_superoperator_infallible(&mut self, qubit: usize, s: &[[Complex64; 4]; 4]) {
        let result = self.apply_block_superoperator(qubit, s);
        #[cfg(feature = "gpu")]
        launched(result);
        #[cfg(not(feature = "gpu"))]
        result.expect("host block superoperator is infallible");
    }

    /// Apply `gate` and then every Kraus set in `channels`, all one-qubit maps
    /// on `qubit`, in one buffer sweep instead of one per map.
    ///
    /// Returns false with the buffer untouched when `gate` has no one-qubit
    /// matrix. Each block superoperator acts on the same `(row-bit, col-bit)`
    /// 4-vector of `qubit`, so the composition is their matrix product in
    /// application order and costs 64 complex multiplies per factor, once per
    /// instruction rather than once per amplitude.
    pub(crate) fn try_apply_fused_1q_channels(
        &mut self,
        gate: &Gate,
        qubit: usize,
        channels: &[Vec<[[Complex64; 2]; 2]>],
    ) -> bool {
        let Some(mat) = matrix_1q(gate) else {
            return false;
        };
        let mut s = block_superoperator(&[mat]);
        for kraus in channels {
            s = mat_mul_4x4(&block_superoperator(kraus), &s);
        }
        self.apply_block_superoperator_infallible(qubit, &s);
        true
    }

    /// Apply symmetric two-qubit depolarizing on `(q0, q1)`:
    /// `rho -> (1-p) rho + (p/15) sum_{P != I(x)I} P rho P`, summed over the 15
    /// non-identity two-qubit Paulis.
    ///
    /// The Pauli twirl `sum_P P B P = 4 Tr(B) I4` over all 16 two-qubit Paulis
    /// holds for any 4x4 `B`, so on each `(q0, q1)` block the map collapses to
    /// `B -> alpha B + beta Tr(B) I4` with `alpha = 1 - 16p/15` and
    /// `beta = 4p/15`. That is one real scale per amplitude against the 16
    /// complex multiply-accumulates the equivalent 16x16 superoperator pays in
    /// [`DensityMatrixBackend::apply_2q_kraus`].
    ///
    /// # Panics
    ///
    /// If `q0` and `q1` are equal or outside the register, as
    /// [`DensityMatrixBackend::apply_2q_kraus`] does.
    pub fn apply_2q_depolarizing(&mut self, q0: usize, q1: usize, p: f64) {
        let alpha = 1.0 - 16.0 * p / 15.0;
        let beta = 4.0 * p / 15.0;
        let (positions, flats, num_groups) = self.block_layout(q0, q1);

        #[cfg(feature = "gpu")]
        {
            let n = self.num_qubits;
            if let Some((ctx, gpu)) = self.device() {
                launched(dk::depolarizing_2q(&ctx, gpu, n, q0, q1, alpha, beta));
                return;
            }
        }

        #[cfg(feature = "parallel")]
        if 2 * self.num_qubits >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            use crate::backend::MIN_PAR_ITERS;
            use crate::backend::statevector::SendPtr;
            use rayon::prelude::*;

            let ptr = SendPtr(self.sv.state.as_mut_ptr());
            (0..num_groups)
                .into_par_iter()
                .with_min_len(MIN_PAR_ITERS)
                .for_each(move |m| {
                    let base = block_base(m, &positions);
                    // SAFETY: inserting a zero bit at each of the four block
                    // positions is a bijection from `m` onto the block bases, so
                    // the 16 offsets one iteration touches are disjoint from
                    // every other iteration's. The safe alternative needs the
                    // 16 entries as disjoint sub-slices, which they are not:
                    // they are strided across four independent bit positions.
                    unsafe {
                        let mut trace = Complex64::new(0.0, 0.0);
                        for t in 0..4 {
                            trace += ptr.load(base | flats[5 * t]);
                        }
                        let shift = trace * beta;
                        for (i, &off) in flats.iter().enumerate() {
                            let mut out = ptr.load(base | off) * alpha;
                            if i % 5 == 0 {
                                out += shift;
                            }
                            ptr.store(base | off, out);
                        }
                    }
                });
            return;
        }

        for m in 0..num_groups {
            let base = block_base(m, &positions);
            let mut trace = Complex64::new(0.0, 0.0);
            for t in 0..4 {
                trace += self.sv.state[base | flats[5 * t]];
            }
            let shift = trace * beta;
            for (i, &off) in flats.iter().enumerate() {
                let mut out = self.sv.state[base | off] * alpha;
                if i % 5 == 0 {
                    out += shift;
                }
                self.sv.state[base | off] = out;
            }
        }
    }

    /// Block traversal for a two-qubit map on `(q0, q1)`: the four buffer bit
    /// positions the block spans in ascending order, the 16 offsets from a
    /// block base indexed `4 * tr + tc`, and the number of blocks. Block index
    /// `5 * t` is the diagonal entry `tr == tc == t`.
    ///
    /// The four positions must be distinct and inside the register, or the
    /// zero-bit insertion in [`block_base`] stops being a bijection onto the
    /// block bases and the sweep reads and writes the wrong entries. Both channel
    /// entry points are public and route through here, so this is their choke
    /// point. It is not the sweep's:
    /// [`DensityMatrixBackend::apply_rzz_sandwich`] builds its table and calls
    /// the sweep directly, where equal targets yield the identity table that the
    /// two register passes it replaced also produced.
    fn block_layout(&self, q0: usize, q1: usize) -> ([usize; 4], [usize; 16], usize) {
        let n = self.num_qubits;
        assert!(
            q0 != q1 && q0 < n && q1 < n,
            "two-qubit channel on ({q0}, {q1}) needs distinct targets inside the {n}-qubit register"
        );
        let mut positions = [q0, q1, q0 + n, q1 + n];
        positions.sort_unstable();

        let mut flats = [0usize; 16];
        for tr in 0..4 {
            for tc in 0..4 {
                flats[4 * tr + tc] = (if tr & 2 != 0 { 1usize << (q0 + n) } else { 0 })
                    | (if tr & 1 != 0 { 1usize << (q1 + n) } else { 0 })
                    | (if tc & 2 != 0 { 1usize << q0 } else { 0 })
                    | (if tc & 1 != 0 { 1usize << q1 } else { 0 });
            }
        }

        let d = self.dim();
        (positions, flats, (d * d) >> 4)
    }

    /// Apply a general two-qubit channel `rho -> sum_k K_k rho K_k^dagger` on
    /// `(q0, q1)`. Each operator is indexed `K[t][t']` with
    /// `t = 2 * bit(q0) + bit(q1)`, matching [`Gate::matrix_4x4`].
    ///
    /// The Kraus set is compiled once into a 16x16 block superoperator over the
    /// four-bit `(q0-row, q1-row, q0-col, q1-col)` block, so the buffer is swept
    /// once. Block index `4*tr + tc` orders the two-qubit row value `tr` against
    /// the column value `tc`. An all-diagonal set compiles to a diagonal
    /// superoperator and is applied as one complex multiply per amplitude in a
    /// contiguous pass instead of the dense sweep. On hosts with detected AVX2
    /// and FMA the dense sweep issues each row inner product two complex terms
    /// per FMA through `simd::PreparedKraus2q`.
    ///
    /// # Panics
    ///
    /// If `q0` and `q1` are equal or outside the register. A model built through
    /// [`NoiseModel`](crate::NoiseModel) is rejected before it reaches here.
    pub fn apply_2q_kraus(&mut self, q0: usize, q1: usize, kraus: &[[[Complex64; 4]; 4]]) {
        // S[4*tr+tc][4*trp+tcp] = sum_k K_k[tr][trp] * conj(K_k[tc][tcp]).
        let mut s = [[Complex64::new(0.0, 0.0); 16]; 16];
        for k in kraus {
            for tr in 0..4 {
                for trp in 0..4 {
                    let kr = k[tr][trp];
                    if kr == Complex64::new(0.0, 0.0) {
                        continue;
                    }
                    for tc in 0..4 {
                        for tcp in 0..4 {
                            s[4 * tr + tc][4 * trp + tcp] += kr * k[tc][tcp].conj();
                        }
                    }
                }
            }
        }

        let (positions, flats, num_groups) = self.block_layout(q0, q1);

        // Off-diagonal entries of an all-diagonal set only ever accumulate
        // `kr * conj(0.0)`, so exact zero is the test.
        let zero = Complex64::new(0.0, 0.0);
        let diagonal = s
            .iter()
            .enumerate()
            .all(|(r, row)| row.iter().enumerate().all(|(c, &e)| r == c || e == zero));
        if diagonal {
            let diag = std::array::from_fn(|i| s[i][i]);
            let result = self.kraus_2q_diagonal_sweep(&diag, q0, q1);
            #[cfg(feature = "gpu")]
            launched(result);
            #[cfg(not(feature = "gpu"))]
            result.expect("host diagonal sweep is infallible");
            return;
        }

        #[cfg(feature = "gpu")]
        {
            let n = self.num_qubits;
            if let Some((ctx, gpu)) = self.device() {
                launched(dk::kraus_2q_dense(&ctx, gpu, n, q0, q1, &s));
                return;
            }
        }

        #[cfg(target_arch = "x86_64")]
        if simd::has_avx2_fma() && simd::kraus_2q_wide_enabled() {
            // SAFETY: AVX2 and FMA checked above; AVX2 implies the AVX the
            // constructor requires.
            let prepared = unsafe { simd::PreparedKraus2q::new(&s) };
            self.kraus_2q_sweep_wide(&prepared, &positions, &flats, num_groups);
            return;
        }

        match positions[0] {
            0 | 1 => self.kraus_2q_sweep::<1>(&s, &positions, &flats, num_groups),
            _ => self.kraus_2q_sweep::<4>(&s, &positions, &flats, num_groups),
        }
    }

    /// [`DensityMatrixBackend::kraus_2q_sweep`] with the row inner products
    /// issued through [`simd::PreparedKraus2q`], one block per iteration
    /// whatever the qubit pair: the vector axis is the superoperator's 16-wide
    /// `j` axis, which does not depend on `min(q0, q1)` the way the block-run
    /// width does.
    #[cfg(target_arch = "x86_64")]
    fn kraus_2q_sweep_wide(
        &mut self,
        prepared: &simd::PreparedKraus2q,
        positions: &[usize; 4],
        flats: &[usize; 16],
        num_groups: usize,
    ) {
        #[cfg(feature = "parallel")]
        if 2 * self.num_qubits >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            use crate::backend::MIN_PAR_ITERS;
            use crate::backend::statevector::SendPtr;
            use rayon::prelude::*;

            let ptr = SendPtr(self.sv.state.as_mut_ptr());
            let positions = *positions;
            let flats = *flats;
            (0..num_groups)
                .into_par_iter()
                .with_min_len(MIN_PAR_ITERS)
                .for_each(move |m| {
                    let base = block_base(m, &positions);
                    // SAFETY: AVX2 and FMA detected. Inserting a zero bit at
                    // each of the four block positions is a bijection from
                    // `m` onto the block bases, so every `base | flats[j]`
                    // stays under `4^n` and the 16 offsets one iteration
                    // touches are disjoint from every other iteration's.
                    unsafe { prepared.apply_block_ptr(ptr.as_f64_ptr(), base, &flats) };
                });
            return;
        }

        let ptr = self.sv.state.as_mut_ptr() as *mut f64;
        for m in 0..num_groups {
            let base = block_base(m, positions);
            // SAFETY: AVX2 and FMA detected. The block bases are disjoint and
            // in bounds as in the parallel arm, on one thread.
            unsafe { prepared.apply_block_ptr(ptr, base, flats) };
        }
    }

    /// Apply a diagonal block superoperator in one contiguous pass: the
    /// amplitude at `idx` is scaled by `diag[diag_slot(idx, q0, q1, n)]`, one
    /// complex multiply against the dense sweep's 16 multiply-accumulates.
    fn kraus_2q_diagonal_sweep(
        &mut self,
        diag: &[Complex64; 16],
        q0: usize,
        q1: usize,
    ) -> Result<()> {
        let n = self.num_qubits;

        #[cfg(feature = "gpu")]
        if let Some((ctx, gpu)) = self.device() {
            return dk::kraus_2q_diagonal(&ctx, gpu, n, q0, q1, diag);
        }

        #[cfg(feature = "parallel")]
        if 2 * n >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            use crate::backend::MIN_PAR_ELEMS;
            use rayon::prelude::*;

            let diag = *diag;
            self.sv
                .state
                .par_iter_mut()
                .enumerate()
                .with_min_len(MIN_PAR_ELEMS)
                .for_each(move |(idx, amp)| {
                    *amp *= diag[diag_slot(idx, q0, q1, n)];
                });
            return Ok(());
        }

        for (idx, amp) in self.sv.state.iter_mut().enumerate() {
            *amp *= diag[diag_slot(idx, q0, q1, n)];
        }
        Ok(())
    }

    /// Sweep the buffer applying the compiled 16x16 block superoperator `s`,
    /// `W` blocks per iteration.
    ///
    /// `W` must divide `2^positions[0]`. Blocks whose compacted indices form an
    /// aligned run of that length have consecutive bases, because
    /// [`insert_zero_bit`] preserves every bit below its position and no `flats`
    /// entry has a bit below `positions[0]`. Each of the 16 slots is then a
    /// contiguous run of `W` amplitudes, which amortizes the base and offset
    /// arithmetic over `W` blocks and gives the accumulator chains room to
    /// overlap. The arithmetic itself stays scalar: the crate builds at the
    /// x86-64 baseline and every wide kernel here takes its width from
    /// `#[target_feature]` dispatch in [`crate::backend::simd`], which this does
    /// not use.
    ///
    /// Only `W = 1` and `W = 4` are instantiated. `W = 1` is forced when
    /// `min(q0, q1) == 0`, where one block bit is bit 0 and no run exists, and
    /// it also serves `min(q0, q1) == 1`: a two-block step measured +0.2% and
    /// +1.9% against one, so it earned no arm of its own. Hosts with detected
    /// AVX2 and FMA route past this sweep to
    /// [`DensityMatrixBackend::kraus_2q_sweep_wide`] unless
    /// `PRISM_NO_AVX2_KRAUS` is set.
    fn kraus_2q_sweep<const W: usize>(
        &mut self,
        s: &[[Complex64; 16]; 16],
        positions: &[usize; 4],
        flats: &[usize; 16],
        num_groups: usize,
    ) {
        let zero = Complex64::new(0.0, 0.0);
        let steps = num_groups / W;

        #[cfg(feature = "parallel")]
        if 2 * self.num_qubits >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            use crate::backend::MIN_PAR_ITERS;
            use crate::backend::statevector::SendPtr;
            use rayon::prelude::*;

            let ptr = SendPtr(self.sv.state.as_mut_ptr());
            let s = *s;
            let positions = *positions;
            let flats = *flats;
            (0..steps)
                .into_par_iter()
                .with_min_len(MIN_PAR_ITERS.div_ceil(W))
                .for_each(move |step| {
                    let base = block_base(step * W, &positions);
                    let mut v = [[zero; W]; 16];
                    // SAFETY: inserting a zero bit at each of the four block
                    // positions is a bijection from the compacted index onto the
                    // block bases, so the `16 * W` offsets one iteration touches
                    // are disjoint from every other iteration's. The safe
                    // alternative needs them as disjoint sub-slices, which they
                    // are not: the 16 slots are strided across four independent
                    // bit positions.
                    unsafe {
                        for (slot, &off) in v.iter_mut().zip(flats.iter()) {
                            for (j, lane) in slot.iter_mut().enumerate() {
                                *lane = ptr.load((base | off) + j);
                            }
                        }
                        for (row, &off) in s.iter().zip(flats.iter()) {
                            let mut acc = [zero; W];
                            for (&coeff, slot) in row.iter().zip(v.iter()) {
                                for (a, &lane) in acc.iter_mut().zip(slot.iter()) {
                                    *a += coeff * lane;
                                }
                            }
                            for (j, &a) in acc.iter().enumerate() {
                                ptr.store((base | off) + j, a);
                            }
                        }
                    }
                });
            return;
        }

        for step in 0..steps {
            let base = block_base(step * W, positions);
            let mut v = [[zero; W]; 16];
            for (k, &off) in flats.iter().enumerate() {
                v[k].copy_from_slice(&self.sv.state[(base | off)..(base | off) + W]);
            }
            for (row, &off) in s.iter().zip(flats.iter()) {
                let mut acc = [zero; W];
                for (&coeff, slot) in row.iter().zip(v.iter()) {
                    for (a, &lane) in acc.iter_mut().zip(slot.iter()) {
                        *a += coeff * lane;
                    }
                }
                self.sv.state[(base | off)..(base | off) + W].copy_from_slice(&acc);
            }
        }
    }

    /// Exact `Tr(rho P)` for the joint Pauli reduced to `(xmask, zmask, num_y)`,
    /// where `P|j> = i^{num_y} * (-1)^{popcount(j & zmask)} * |j ^ xmask>`. The
    /// trace collapses to a single diagonal-offset sweep:
    /// `Tr(rho P) = i^{num_y} sum_j (-1)^{popcount(j & zmask)} rho[j][j ^ xmask]`.
    pub fn expectation_pauli(&self, xmask: usize, zmask: usize, num_y: u32) -> f64 {
        self.expectations_pauli(&[(xmask, zmask, num_y)])[0]
    }

    /// [`DensityMatrixBackend::expectation_pauli`] for every mask triple in one
    /// sweep of the `4^n` buffer.
    ///
    /// Each row is visited once and contributes one entry per observable, so
    /// the strided sweep is paid once instead of once per observable. Values
    /// match the single-observable form exactly: the accumulation order within
    /// an observable is unchanged.
    pub fn expectations_pauli(&self, masks: &[(usize, usize, u32)]) -> Vec<f64> {
        let d = self.dim();
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.sv.gpu_state() {
            let pairs: Vec<(u64, u64)> = masks
                .iter()
                .map(|&(x, z, _)| (x as u64, z as u64))
                .collect();
            let sums = launched(dk::pauli_sums(gpu.context(), gpu, self.num_qubits, &pairs));
            return sums
                .iter()
                .zip(masks)
                .map(|(value, &(_, _, num_y))| (value * i_pow(num_y)).re)
                .collect();
        }
        let mut acc = vec![Complex64::new(0.0, 0.0); masks.len()];
        for j in 0..d {
            let row = &self.sv.state[j * d..(j + 1) * d];
            for (slot, &(xmask, zmask, _)) in acc.iter_mut().zip(masks) {
                let sign = if (j & zmask).count_ones() & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                *slot += row[j ^ xmask] * sign;
            }
        }
        acc.iter()
            .zip(masks)
            .map(|(value, &(_, _, num_y))| (value * i_pow(num_y)).re)
            .collect()
    }
}

impl Backend for DensityMatrixBackend {
    fn name(&self) -> &'static str {
        "density_matrix"
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::DensityMatrix
    }

    fn placement(&self) -> crate::sim::Placement {
        if self.sv.is_gpu_resident() {
            crate::sim::Placement::Device
        } else {
            crate::sim::Placement::Host
        }
    }

    /// With a device attached the host cap does not apply: the mixture is
    /// budgeted against free VRAM instead, before anything is allocated.
    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        #[cfg(feature = "gpu")]
        if let Some(ctx) = &self.gpu_context {
            check_device_budget(ctx, num_qubits)?;
            self.num_qubits = num_qubits;
            self.classical_bits = vec![false; num_classical_bits];
            return self.sv.init(2 * num_qubits, 0);
        }

        crate::backend::check_state_allocation(
            "density_matrix",
            num_qubits,
            crate::backend::max_density_matrix_qubits(),
            crate::backend::DM_QUBIT_CAP_ENV,
        )?;

        self.num_qubits = num_qubits;
        self.classical_bits = vec![false; num_classical_bits];
        self.sv.init(2 * num_qubits, 0)
    }

    fn supports_initial_state(&self) -> bool {
        true
    }

    /// Starts from the pure mixture `|psi><psi|`, so the buffer is the `4^n`
    /// outer product of `amplitudes` with its own conjugate. The cap check in
    /// [`init`](Self::init) runs first, before that buffer is sized.
    fn init_from_amplitudes(
        &mut self,
        amplitudes: Vec<Complex64>,
        num_classical_bits: usize,
    ) -> Result<()> {
        crate::backend::validate_initial_amplitudes(&amplitudes)?;
        let num_qubits = amplitudes.len().trailing_zeros() as usize;
        self.init(num_qubits, num_classical_bits)?;

        #[cfg(feature = "gpu")]
        if let Some((ctx, gpu)) = self.device() {
            return dk::outer_product(&ctx, gpu, num_qubits, &amplitudes);
        }

        let d = amplitudes.len();
        for (row, &amp) in amplitudes.iter().enumerate() {
            let dst = &mut self.sv.state[row * d..(row + 1) * d];
            for (entry, &col) in dst.iter_mut().zip(amplitudes.iter()) {
                *entry = amp * col.conj();
            }
        }
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.apply_unitary(gate, targets),
            Instruction::Barrier { .. } => Ok(()),
            Instruction::Measure {
                qubit,
                classical_bit,
            } => self.apply_measure(*qubit, *classical_bit),
            Instruction::Reset { qubit } => self.apply_reset(*qubit),
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => self.apply_conditional(condition, gate, targets),
            Instruction::Region(region) => self.apply_region(region),
        }
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let d = self.dim();
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.sv.gpu_state() {
            let mut diag = dk::diagonal(gpu.context(), gpu, self.num_qubits)?;
            for p in &mut diag {
                *p = p.max(0.0);
            }
            return Ok(diag);
        }
        let mut probs = vec![0.0f64; d];
        for (k, p) in probs.iter_mut().enumerate() {
            *p = self.sv.state[k * d + k].re.max(0.0);
        }
        Ok(probs)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// `MultiFused` and `Multi2q` apply their constituents one at a time here
    /// rather than through the tiled kernels, which the ket register's shifted
    /// indices would reorder.
    fn supports_fused_gates(&self) -> bool {
        true
    }

    /// The mixture is a `2n`-qubit statevector, so every fusion floor is
    /// reached at half the circuit width the statevector needs.
    fn fusion_state_qubits(&self, num_qubits: usize) -> usize {
        2 * num_qubits
    }

    fn qubit_probability(&self, qubit: usize) -> Result<f64> {
        self.prob_one(qubit)
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset(qubit)
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// `Tr(rho P_k)` per observable, the mixed-state reading of the trait's
    /// `<psi|P_k|psi>`. `rho` is trace-one by construction, so no
    /// normalization divide is needed.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        let masks = observables
            .iter()
            .map(|observable| crate::sim::pauli_masks(observable, self.num_qubits))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.expectations_pauli(&masks))
    }

    /// On the device the four entries come from the Pauli sums `T`, `Z`, `X`,
    /// and `Y` on `qubit`: the diagonal is `(T +- Z) / 2` and the off-diagonal
    /// pair is `(X +- Y) / 2`, where `Y` carries the row sign and no `i`.
    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        let n = self.num_qubits;
        let d = self.dim();
        let bit = 1usize << qubit;
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.sv.gpu_state() {
            let b = bit as u64;
            let sums = dk::pauli_sums(gpu.context(), gpu, n, &[(0, 0), (0, b), (b, 0), (b, b)])?;
            let (t, z, x, y) = (sums[0], sums[1], sums[2], sums[3]);
            return Ok([
                [(t + z) * 0.5, (x + y) * 0.5],
                [(x - y) * 0.5, (t - z) * 0.5],
            ]);
        }
        let others = 1usize << (n - 1);
        let mut r00 = Complex64::new(0.0, 0.0);
        let mut r01 = Complex64::new(0.0, 0.0);
        let mut r10 = Complex64::new(0.0, 0.0);
        let mut r11 = Complex64::new(0.0, 0.0);
        for m in 0..others {
            let base = (m & (bit - 1)) | ((m >> qubit) << (qubit + 1));
            let i1 = base | bit;
            r00 += self.sv.state[base * d + base];
            r01 += self.sv.state[base * d + i1];
            r10 += self.sv.state[i1 * d + base];
            r11 += self.sv.state[i1 * d + i1];
        }
        Ok([[r00, r01], [r10, r11]])
    }

    /// Evolve `rho -> K rho K^dagger` for an arbitrary `K`, on the same kernel
    /// selection as the one-qubit branch of `apply_unitary`. Trajectories never
    /// route here, since `supports_noisy_per_shot` excludes the density matrix in
    /// favour of `apply_1q_kraus` on the mixture; this keeps the trait method
    /// allocation-free on every backend that holds a state.
    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        self.apply_1q_sandwich(qubit, matrix)
    }
}

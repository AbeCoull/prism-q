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
//! ceiling is about 14 qubits on a 16 GiB host and 15 on a 32 GiB host. CPU only.
//! This backend is explicit-dispatch only and is never chosen by `Auto`.
//!
//! # Gate support
//!
//! Supported: exact unitary evolution, basis-state probabilities, the one-qubit
//! reduced density matrix, projective measurement with stochastic collapse,
//! reset, classically-conditioned gates, exact one-qubit Kraus channels
//! (`apply_1q_kraus`), two-qubit depolarizing (`apply_2q_depolarizing`), and
//! exact `Tr(rho P)` expectation (`expectation_pauli`, reachable through
//! `pauli_expectations`). Batched payload shapes (`QftBlock`, `BatchPhase`,
//! `BatchRzz`, `DiagonalBatch`, `MultiFused`, `Multi2q`) carry qubit indices
//! outside the instruction targets and are remapped onto the ket register
//! before the left product; `sim` does not produce them here, since
//! `supports_fused_gates` returns `false`, but a direct `Backend::apply` can.
//!
//! # When to prefer this backend
//!
//! - Exact noise-channel evolution: one run yields the exact mixed state,
//!   where trajectory averaging converges as `1/sqrt(shots)`. Selecting it
//!   with a noise model attached routes every `Simulate` terminal to that one
//!   evolution.
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

use std::borrow::Cow;

use num_complex::Complex64;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use crate::backend::simd;
use crate::backend::statevector::{StatevectorBackend, insert_zero_bit};
use crate::backend::{Backend, NORM_CLAMP_MIN};
use crate::circuit::{ClassicalCondition, Instruction};
use crate::error::Result;
use crate::gates::{DiagEntry, Gate, McuData};
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

fn conjugate_2x2(m: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [m[0][0].conj(), m[0][1].conj()],
        [m[1][0].conj(), m[1][1].conj()],
    ]
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
        Gate::Cu(mat) => Some(Gate::Cu(Box::new(conjugate_2x2(mat)))),
        Gate::Mcu(data) => Some(Gate::Mcu(Box::new(McuData {
            mat: conjugate_2x2(&data.mat),
            num_controls: data.num_controls,
        }))),
        _ => None,
    }
}

/// The gate with every qubit index stored inside its payload shifted onto the
/// ket register. Offsetting the instruction targets is not enough for the
/// batched shapes, whose application indices live in the payload, nor for
/// `QftBlock`, whose whole range lives in the variant. Borrows the gate when it
/// carries no such index.
fn ket_register_gate(gate: &Gate, n: usize) -> Cow<'_, Gate> {
    match gate {
        Gate::BatchPhase(data) => {
            let mut data = data.clone();
            for entry in &mut data.phases {
                entry.0 += n;
            }
            Cow::Owned(Gate::BatchPhase(data))
        }
        Gate::BatchRzz(data) => {
            let mut data = data.clone();
            for entry in &mut data.edges {
                entry.0 += n;
                entry.1 += n;
            }
            Cow::Owned(Gate::BatchRzz(data))
        }
        Gate::DiagonalBatch(data) => {
            let mut data = data.clone();
            for entry in &mut data.entries {
                match entry {
                    DiagEntry::Phase1q { qubit, .. } => *qubit += n,
                    DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } => {
                        *q0 += n;
                        *q1 += n;
                    }
                }
            }
            Cow::Owned(Gate::DiagonalBatch(data))
        }
        Gate::MultiFused(data) => {
            let mut data = data.clone();
            for entry in &mut data.gates {
                entry.0 += n;
            }
            Cow::Owned(Gate::MultiFused(data))
        }
        Gate::Multi2q(data) => {
            let mut data = data.clone();
            for entry in &mut data.gates {
                entry.0 += n;
                entry.1 += n;
            }
            Cow::Owned(Gate::Multi2q(data))
        }
        Gate::QftBlock { start, num } => Cow::Owned(Gate::QftBlock {
            start: start + n as u8,
            num: *num,
        }),
        _ => Cow::Borrowed(gate),
    }
}

/// Exact density-matrix simulator. See the module docs for the state layout.
pub struct DensityMatrixBackend {
    num_qubits: usize,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
    sv: StatevectorBackend,
}

impl DensityMatrixBackend {
    /// Create a new density-matrix backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            sv: StatevectorBackend::new(seed),
        }
    }

    /// Purity `Tr(rho^2)`, equal to `1` for a pure state and less otherwise.
    pub fn purity(&self) -> f64 {
        crate::backend::state_norm_sqr(&self.sv.state)
    }

    #[inline]
    fn dim(&self) -> usize {
        1usize << self.num_qubits
    }

    fn conjugate_buffer(&mut self) {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            if self.sv.state.len() >= (1 << crate::backend::PARALLEL_THRESHOLD_QUBITS) {
                self.sv
                    .state
                    .par_iter_mut()
                    .for_each(|amp| *amp = amp.conj());
                return;
            }
        }
        for amp in self.sv.state.iter_mut() {
            *amp = amp.conj();
        }
    }

    /// Apply a compiled one-qubit block superoperator in a single buffer pass.
    ///
    /// The `(row-bit, col-bit)` block of `qubit` is the two-qubit subspace
    /// `(qubit + n, qubit)` of the embedded `2n`-qubit statevector, so the
    /// statevector two-qubit kernel applies `S` directly. That kernel indexes
    /// its matrix as `2 * bit(q0) + bit(q1)`, matching the block index `2a + b`
    /// once the row bit is passed as `q0`.
    fn apply_block_superoperator(&mut self, qubit: usize, s: &[[Complex64; 4]; 4]) {
        let n = self.num_qubits;
        self.sv.apply_fused_2q(qubit + n, qubit, s);
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
    /// A one-qubit gate can instead compile to a block superoperator and sweep
    /// the buffer once, which wins once the buffer stops fitting in cache. Below
    /// that the superoperator's dense 4x4 block loses to two cheap passes, so
    /// the crossover is the point where the embedded `2n`-qubit statevector
    /// reaches the kernels' own parallel threshold.
    fn apply_unitary(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        let n = self.num_qubits;

        if let Some(mat) = matrix_1q(gate) {
            if 2 * n >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
                let s = block_superoperator(&[mat]);
                self.apply_block_superoperator(targets[0], &s);
                return Ok(());
            }
            self.sv.apply_1q_matrix(targets[0] + n, &mat)?;
            return self.sv.apply_1q_matrix(targets[0], &conjugate_2x2(&mat));
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

        self.conjugate_buffer();
        self.sv.apply(&Instruction::Gate {
            gate: gate.clone(),
            targets: bra_targets,
        })?;
        self.conjugate_buffer();
        Ok(())
    }

    /// `P(qubit = |1>) = Tr(P_1 rho)`, the sum of the diagonal `rho` entries
    /// whose row index has `qubit` set.
    fn prob_one(&self, qubit: usize) -> f64 {
        let d = self.dim();
        let bit = 1usize << qubit;
        let mut p1 = 0.0;
        for r in 0..d {
            if r & bit != 0 {
                p1 += self.sv.state[r * d + r].re;
            }
        }
        p1.clamp(0.0, 1.0)
    }

    /// Sample and project qubit `qubit`, recording the outcome in
    /// `classical_bit`. Collapses `rho -> P_m rho P_m / p_m`, which in the
    /// embedded layout zeroes every entry whose row or column disagrees with
    /// the outcome on `qubit` and rescales the survivors to unit trace.
    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        let p1 = self.prob_one(qubit);
        let u: f64 = self.rng.random();
        let outcome = u < p1;
        self.classical_bits[classical_bit] = outcome;
        let p = if outcome { p1 } else { 1.0 - p1 };
        self.project(qubit, outcome, p);
    }

    /// Deterministic reset `rho -> |0><0| (x) tr_q rho`: fold the block with
    /// `qubit` set on both row and column into the block with it clear on both,
    /// then zero the three sibling entries that still touch `qubit`. The four
    /// entry classes are contiguous runs of the buffer, so the pass walks
    /// `2 * rmask` blocks rather than scattering per block base.
    fn apply_reset(&mut self, qubit: usize) {
        let n = self.num_qubits;
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
                return;
            }

            let tile = row_aligned_tile(cmask, rmask);
            for block in self.sv.state.chunks_mut(block_size) {
                let (r0, r1) = block.split_at_mut(rmask);
                r0.par_chunks_mut(tile)
                    .zip(r1.par_chunks_mut(tile))
                    .for_each(|(t0, t1)| reset_fold_pair(t0, t1, cmask));
            }
            return;
        }

        for block in self.sv.state.chunks_mut(block_size) {
            let (r0, r1) = block.split_at_mut(rmask);
            reset_fold_pair(r0, r1, cmask);
        }
    }

    /// Project `rho` onto the `outcome` subspace of `qubit` with outcome
    /// probability `p`, renormalizing the survivors to unit trace.
    fn project(&mut self, qubit: usize, outcome: bool, p: f64) {
        let n = self.num_qubits;
        let rmask = 1usize << (qubit + n);
        let cmask = 1usize << qubit;
        let scale = Complex64::new(1.0 / p.clamp(NORM_CLAMP_MIN, 1.0), 0.0);

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
            return;
        }

        for (t, chunk) in self.sv.state.chunks_mut(rmask).enumerate() {
            project_tile(chunk, t * rmask, rmask, cmask, outcome, scale);
        }
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
        self.apply_block_superoperator(qubit, &s);
    }

    /// Apply symmetric two-qubit depolarizing on `(q0, q1)`:
    /// `rho -> (1-p) rho + (p/15) sum_{P != I(x)I} P rho P`, summed over the 15
    /// non-identity two-qubit Paulis. The 16 two-qubit Pauli Kraus operators are
    /// compiled once into a 16x16 block superoperator over the four-bit
    /// `(q0-row, q1-row, q0-col, q1-col)` block, so the buffer is swept once.
    ///
    /// Block index `4*tr + tc` orders the two-qubit row value `tr` (bit 0 = q0,
    /// bit 1 = q1) against the column value `tc`.
    pub fn apply_2q_depolarizing(&mut self, q0: usize, q1: usize, p: f64) {
        let c1 = Complex64::new(1.0, 0.0);
        let c0 = Complex64::new(0.0, 0.0);
        let ci = Complex64::new(0.0, 1.0);
        let paulis: [[[Complex64; 2]; 2]; 4] = [
            [[c1, c0], [c0, c1]],
            [[c0, c1], [c1, c0]],
            [[c0, -ci], [ci, c0]],
            [[c1, c0], [c0, -c1]],
        ];
        // K_(a,b) = weight * (paulis[b] (x) paulis[a]) with q0 as the low bit;
        // S[4*tr+tc][4*trp+tcp] = sum_(a,b) K[tr][trp] * conj(K[tc][tcp]).
        let mut s = [[Complex64::new(0.0, 0.0); 16]; 16];
        for a in 0..4 {
            for b in 0..4 {
                let w = if a == 0 && b == 0 {
                    (1.0 - p).sqrt()
                } else {
                    (p / 15.0).sqrt()
                };
                let kentry = |t: usize, tp: usize| {
                    Complex64::new(w, 0.0) * paulis[b][t >> 1][tp >> 1] * paulis[a][t & 1][tp & 1]
                };
                for tr in 0..4 {
                    for trp in 0..4 {
                        let kr = kentry(tr, trp);
                        if kr == Complex64::new(0.0, 0.0) {
                            continue;
                        }
                        for tc in 0..4 {
                            for tcp in 0..4 {
                                s[4 * tr + tc][4 * trp + tcp] += kr * kentry(tc, tcp).conj();
                            }
                        }
                    }
                }
            }
        }

        let n = self.num_qubits;
        let d = self.dim();
        let mut positions = [q0, q1, q0 + n, q1 + n];
        positions.sort_unstable();
        let flat_offset = |tr: usize, tc: usize| {
            (if tr & 1 != 0 { 1usize << (q0 + n) } else { 0 })
                | (if tr & 2 != 0 { 1usize << (q1 + n) } else { 0 })
                | (if tc & 1 != 0 { 1usize << q0 } else { 0 })
                | (if tc & 2 != 0 { 1usize << q1 } else { 0 })
        };
        let mut flats = [0usize; 16];
        for tr in 0..4 {
            for tc in 0..4 {
                flats[4 * tr + tc] = flat_offset(tr, tc);
            }
        }

        let num_groups = (d * d) >> 4;

        #[cfg(feature = "parallel")]
        if 2 * n >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            use crate::backend::MIN_PAR_ITERS;
            use crate::backend::statevector::SendPtr;
            use rayon::prelude::*;

            let ptr = SendPtr(self.sv.state.as_mut_ptr());
            (0..num_groups)
                .into_par_iter()
                .with_min_len(MIN_PAR_ITERS)
                .for_each(move |m| {
                    let mut base = m;
                    for &pos in &positions {
                        base = insert_zero_bit(base, pos);
                    }
                    let mut v = [Complex64::new(0.0, 0.0); 16];
                    // SAFETY: inserting a zero bit at each of the four block
                    // positions is a bijection from `m` onto the block bases, so
                    // the 16 offsets one iteration touches are disjoint from
                    // every other iteration's. The safe alternative needs the
                    // 16 entries as disjoint sub-slices, which they are not:
                    // they are strided across four independent bit positions.
                    unsafe {
                        for (k, &off) in flats.iter().enumerate() {
                            v[k] = ptr.load(base | off);
                        }
                        for (i, &off) in flats.iter().enumerate() {
                            let mut acc = Complex64::new(0.0, 0.0);
                            for j in 0..16 {
                                acc += s[i][j] * v[j];
                            }
                            ptr.store(base | off, acc);
                        }
                    }
                });
            return;
        }

        for m in 0..num_groups {
            let mut base = m;
            for &pos in &positions {
                base = insert_zero_bit(base, pos);
            }
            let mut v = [Complex64::new(0.0, 0.0); 16];
            for (k, &off) in flats.iter().enumerate() {
                v[k] = self.sv.state[base | off];
            }
            for (i, &off) in flats.iter().enumerate() {
                let mut acc = Complex64::new(0.0, 0.0);
                for j in 0..16 {
                    acc += s[i][j] * v[j];
                }
                self.sv.state[base | off] = acc;
            }
        }
    }

    /// Exact `Tr(rho P)` for the joint Pauli reduced to `(xmask, zmask, num_y)`,
    /// where `P|j> = i^{num_y} * (-1)^{popcount(j & zmask)} * |j ^ xmask>`. The
    /// trace collapses to a single diagonal-offset sweep:
    /// `Tr(rho P) = i^{num_y} sum_j (-1)^{popcount(j & zmask)} rho[j][j ^ xmask]`.
    pub fn expectation_pauli(&self, xmask: usize, zmask: usize, num_y: u32) -> f64 {
        let d = self.dim();
        let mut acc = Complex64::new(0.0, 0.0);
        for j in 0..d {
            let sign = if (j & zmask).count_ones() & 1 == 1 {
                -1.0
            } else {
                1.0
            };
            acc += self.sv.state[j * d + (j ^ xmask)] * sign;
        }
        (acc * i_pow(num_y)).re
    }
}

impl Backend for DensityMatrixBackend {
    fn name(&self) -> &'static str {
        "density_matrix"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
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

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.apply_unitary(gate, targets),
            Instruction::Barrier { .. } => Ok(()),
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.apply_measure(*qubit, *classical_bit);
                Ok(())
            }
            Instruction::Reset { qubit } => {
                self.apply_reset(*qubit);
                Ok(())
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => self.apply_conditional(condition, gate, targets),
        }
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let d = self.dim();
        let mut probs = vec![0.0f64; d];
        for (k, p) in probs.iter_mut().enumerate() {
            *p = self.sv.state[k * d + k].re.max(0.0);
        }
        Ok(probs)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_fused_gates(&self) -> bool {
        false
    }

    fn qubit_probability(&self, qubit: usize) -> Result<f64> {
        Ok(self.prob_one(qubit))
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset(qubit);
        Ok(())
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// `Tr(rho P_k)` per observable, the mixed-state reading of the trait's
    /// `<psi|P_k|psi>`. `rho` is trace-one by construction, so no
    /// normalization divide is needed.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        observables
            .iter()
            .map(|observable| {
                let (xmask, zmask, num_y) = crate::sim::pauli_masks(observable, self.num_qubits)?;
                Ok(self.expectation_pauli(xmask, zmask, num_y))
            })
            .collect()
    }

    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        let n = self.num_qubits;
        let d = self.dim();
        let bit = 1usize << qubit;
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
        let n = self.num_qubits;
        if 2 * n >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            let s = block_superoperator(&[*matrix]);
            self.apply_block_superoperator(qubit, &s);
            return Ok(());
        }
        self.sv.apply_1q_matrix(qubit + n, matrix)?;
        self.sv.apply_1q_matrix(qubit, &conjugate_2x2(matrix))
    }
}

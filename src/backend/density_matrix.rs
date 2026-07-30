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
//! Memory is `16 * 4^n` bytes (`4^n` `Complex64` entries), so the practical
//! ceiling is about 14 qubits on a 16 GiB host and 15 on a 32 GiB host. CPU only.
//! This backend is explicit-dispatch only and is never chosen by `Auto`.
//!
//! Supported: exact unitary evolution, basis-state probabilities, the one-qubit
//! reduced density matrix, projective measurement with stochastic collapse,
//! reset, classically-conditioned gates, exact one-qubit Kraus channels
//! (`apply_1q_kraus`), two-qubit depolarizing (`apply_2q_depolarizing`), and
//! exact `Tr(rho P)` expectation (`expectation_pauli`). Fusion is disabled
//! (`supports_fused_gates` returns `false`) so every instruction reaching the
//! backend is a primitive whose qubits live in the instruction targets.

use num_complex::Complex64;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use crate::backend::statevector::{StatevectorBackend, insert_zero_bit};
use crate::backend::{Backend, NORM_CLAMP_MIN};
use crate::circuit::{ClassicalCondition, Instruction};
use crate::error::Result;
use crate::gates::{Gate, McuData};
use crate::sim::i_pow;

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
        self.sv.state.iter().map(Complex64::norm_sqr).sum()
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
    /// `U` applies to the ket register (targets offset by `n`) for the left
    /// product `U rho`, and `conj(U)` to the bra register (original targets) for
    /// the right product `rho U^dagger`. Variants with no native conjugate form
    /// fall back to conjugating the whole buffer around the gate, which costs
    /// two extra passes.
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
            gate: gate.clone(),
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
    /// then zero the three sibling entries that still touch `qubit`. One pass
    /// over the `d*d/4` block bases.
    fn apply_reset(&mut self, qubit: usize) {
        let n = self.num_qubits;
        let d = self.dim();
        let rmask = 1usize << (qubit + n);
        let cmask = 1usize << qubit;
        let both = rmask | cmask;
        let zero = Complex64::new(0.0, 0.0);
        for m in 0..((d * d) >> 2) {
            let base = insert_zero_bit(insert_zero_bit(m, qubit), qubit + n);
            let folded = self.sv.state[base | both];
            self.sv.state[base] += folded;
            self.sv.state[base | rmask] = zero;
            self.sv.state[base | cmask] = zero;
            self.sv.state[base | both] = zero;
        }
    }

    /// Project `rho` onto the `outcome` subspace of `qubit` with outcome
    /// probability `p`, renormalizing the survivors to unit trace.
    fn project(&mut self, qubit: usize, outcome: bool, p: f64) {
        let n = self.num_qubits;
        let d = self.dim();
        let rmask = 1usize << (qubit + n);
        let cmask = 1usize << qubit;
        let keep = if outcome { rmask | cmask } else { 0 };
        let inv = 1.0 / p.clamp(NORM_CLAMP_MIN, 1.0);
        for idx in 0..d * d {
            if idx & (rmask | cmask) == keep {
                self.sv.state[idx] *= inv;
            } else {
                self.sv.state[idx] = Complex64::new(0.0, 0.0);
            }
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

        for m in 0..((d * d) >> 4) {
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
        // The state is a 2n-qubit statevector, so the statevector cap binds at
        // half its value. Taking the tighter of the two here keeps this backend
        // the one that reports the rejection, whatever the two caps are set to.
        crate::backend::check_state_allocation(
            "density_matrix",
            num_qubits,
            crate::backend::max_density_matrix_qubits()
                .min(crate::backend::max_statevector_qubits() / 2),
            "PRISM_MAX_DM_QUBITS",
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

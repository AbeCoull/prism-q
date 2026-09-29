//! Terminal shot sampling for Clifford+T circuits as a Clifford tableau over a dense
//! register that holds only the qubits the T rotations reach.

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_8};

use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::qec::camps_prefix::{SignedCliffordPrefix, SignedPauli};

/// Widest register [`TerminalSampler::compile`] builds. Diagonalizing the measured Paulis on it
/// costs about `k^2 2^k` amplitude updates.
pub(super) const MAX_REGISTER_QUBITS: usize = 20;

/// The state `C (|0> ⊗ φ)`: a Clifford `C` over `n` qubits, `|0>` on every qubit that no
/// T rotation has reached, and dense amplitudes `φ` over the rest.
struct CliffordRegister {
    prefix: SignedCliffordPrefix,
    /// Qubit `q`'s bit in `amps`, or `None` while it holds `|0>`.
    slot: Vec<Option<usize>>,
    amps: Vec<Complex64>,
}

/// Pauli string on the register in the letter convention of [`SignedPauli`]: bit `b` of
/// `x` and `z` gives the letter on register bit `b`, times `i^phase4`.
#[derive(Clone, Copy)]
struct RegisterPauli {
    x: u64,
    z: u64,
    phase4: u8,
}

impl RegisterPauli {
    /// `self · other`, both Hermitian and commuting.
    fn mul(self, other: RegisterPauli) -> RegisterPauli {
        let phase = u32::from(self.phase4)
            + u32::from(other.phase4)
            + letter_phase(self.x, self.z, other.x, other.z);
        RegisterPauli {
            x: self.x ^ other.x,
            z: self.z ^ other.z,
            phase4: (phase & 3) as u8,
        }
    }
}

/// Sum of the `i^k` factors from multiplying letters position by position, packed.
#[inline]
fn letter_phase(ax: u64, az: u64, bx: u64, bz: u64) -> u32 {
    let a_x = ax & !az;
    let a_y = ax & az;
    let a_z = !ax & az;
    let b_x = bx & !bz;
    let b_y = bx & bz;
    let b_z = !bx & bz;
    let plus = (a_x & b_y) | (a_y & b_z) | (a_z & b_x);
    let minus = (a_x & b_z) | (a_y & b_x) | (a_z & b_y);
    plus.count_ones() + 3 * minus.count_ones()
}

/// `dst ← dst · src` over packed rows.
fn mul_rows(dst: &mut SignedPauli, src: &SignedPauli) {
    let mut phase = u32::from(dst.phase4) + u32::from(src.phase4);
    for w in 0..dst.x.len() {
        phase += letter_phase(dst.x[w], dst.z[w], src.x[w], src.z[w]);
        dst.x[w] ^= src.x[w];
        dst.z[w] ^= src.z[w];
    }
    dst.phase4 = (phase & 3) as u8;
}

#[inline]
fn bit(words: &[u64], q: usize) -> bool {
    (words[q >> 6] >> (q & 63)) & 1 == 1
}

impl CliffordRegister {
    fn new(num_qubits: usize) -> Self {
        Self {
            prefix: SignedCliffordPrefix::identity(num_qubits),
            slot: vec![None; num_qubits],
            amps: vec![Complex64::new(1.0, 0.0)],
        }
    }

    fn width(&self) -> usize {
        self.amps.len().trailing_zeros() as usize
    }

    /// Apply `exp(-i θ Z_q)` after the Clifford: `exp(-i θ P)` on `|0> ⊗ φ` for
    /// `P = C† Z_q C`. When `P` flips a qubit still in `|0>`, CX gates controlled on
    /// that qubit, which leave the state alone, clear the flips from every other such
    /// qubit and fold into `C`, and the qubit joins the register. Returns `false` past
    /// `max_width`.
    fn rotate_z(&mut self, q: usize, theta: f64, max_width: usize) -> bool {
        let p = self.prefix.conjugate_z(q);
        let anchor = (0..self.slot.len()).find(|&j| self.slot[j].is_none() && bit(&p.x, j));
        let p = match anchor {
            None => p,
            Some(anchor) => {
                if self.width() == max_width {
                    return false;
                }
                for i in 0..self.slot.len() {
                    if i != anchor && self.slot[i].is_none() && bit(&p.x, i) {
                        self.prefix
                            .fold_right_state_gate(&Gate::Cx, &[anchor, i])
                            .expect("CX folds into the prefix");
                    }
                }
                self.slot[anchor] = Some(self.width());
                let len = self.amps.len();
                self.amps.resize(2 * len, Complex64::new(0.0, 0.0));
                self.prefix.conjugate_z(q)
            }
        };
        let reg = self.restrict(&p);
        self.apply_rotation(reg, theta);
        true
    }

    /// Restrict a Pauli with no `X` on a `|0>` qubit to the register; its `Z` letters
    /// there act as `+1`.
    fn restrict(&self, p: &SignedPauli) -> RegisterPauli {
        let mut reg = RegisterPauli {
            x: 0,
            z: 0,
            phase4: p.phase4,
        };
        for (q, slot) in self.slot.iter().enumerate() {
            match slot {
                Some(b) => {
                    reg.x |= u64::from(bit(&p.x, q)) << b;
                    reg.z |= u64::from(bit(&p.z, q)) << b;
                }
                None => debug_assert!(!bit(&p.x, q), "X left on a qubit held in |0>"),
            }
        }
        reg
    }

    /// `φ ← cos θ φ - i sin θ P φ`, with `P|b> = i^(phase4 + #Y) (-1)^|b & z| |b ^ x>`.
    fn apply_rotation(&mut self, p: RegisterPauli, theta: f64) {
        let quarter = (u32::from(p.phase4) + (p.x & p.z).count_ones()) & 3;
        let (sin, cos) = theta.sin_cos();
        // -i sin θ · i^quarter
        let coeff = Complex64::new(0.0, -sin) * Complex64::i().powu(quarter);
        let sign = |b: usize| {
            if (b as u64 & p.z).count_ones() & 1 == 1 {
                -coeff
            } else {
                coeff
            }
        };
        if p.x == 0 {
            for (b, amp) in self.amps.iter_mut().enumerate() {
                *amp *= cos + sign(b);
            }
            return;
        }
        let x = p.x as usize;
        let low = x & x.wrapping_neg();
        for b in 0..self.amps.len() {
            if b & low != 0 {
                continue;
            }
            let b2 = b ^ x;
            let (a, a2) = (self.amps[b], self.amps[b2]);
            self.amps[b] = a * cos + sign(b2) * a2;
            self.amps[b2] = a2 * cos + sign(b) * a;
        }
    }
}

enum RegisterGate {
    H(usize),
    Sdg(usize),
    Cx(usize, usize),
}

fn conjugate(p: &mut RegisterPauli, gate: &RegisterGate) {
    match *gate {
        RegisterGate::H(m) => {
            let (xb, zb) = ((p.x >> m) & 1, (p.z >> m) & 1);
            if xb & zb == 1 {
                p.phase4 = (p.phase4 + 2) & 3;
            }
            p.x = (p.x & !(1 << m)) | (zb << m);
            p.z = (p.z & !(1 << m)) | (xb << m);
        }
        RegisterGate::Sdg(m) => {
            // Sdg P S: X -> -Y, Y -> X.
            if (p.x >> m) & 1 == 1 {
                if (p.z >> m) & 1 == 0 {
                    p.phase4 = (p.phase4 + 2) & 3;
                }
                p.z ^= 1 << m;
            }
        }
        RegisterGate::Cx(c, t) => {
            let (xc, zc) = ((p.x >> c) & 1 == 1, (p.z >> c) & 1 == 1);
            let (xt, zt) = ((p.x >> t) & 1 == 1, (p.z >> t) & 1 == 1);
            if xc && zt && !(xt ^ zc) {
                p.phase4 = (p.phase4 + 2) & 3;
            }
            if xc {
                p.x ^= 1 << t;
            }
            if zt {
                p.z ^= 1 << c;
            }
        }
    }
}

fn apply_to_amps(amps: &mut [Complex64], gate: &RegisterGate) {
    match *gate {
        RegisterGate::H(m) => {
            let mask = 1usize << m;
            for b in (0..amps.len()).filter(|b| b & mask == 0) {
                let (a0, a1) = (amps[b], amps[b | mask]);
                amps[b] = (a0 + a1) * FRAC_1_SQRT_2;
                amps[b | mask] = (a0 - a1) * FRAC_1_SQRT_2;
            }
        }
        RegisterGate::Sdg(m) => {
            let mask = 1usize << m;
            for (b, amp) in amps.iter_mut().enumerate() {
                if b & mask != 0 {
                    *amp *= Complex64::new(0.0, -1.0);
                }
            }
        }
        RegisterGate::Cx(c, t) => {
            let (cm, tm) = (1usize << c, 1usize << t);
            for b in (0..amps.len()).filter(|b| b & cm != 0 && b & tm == 0) {
                amps.swap(b, b | tm);
            }
        }
    }
}

/// Map independent commuting register Paulis to single `Z` letters with a Clifford `W`,
/// apply `W` to `amps`, and return the cumulative table of `|W φ|^2` over those letters
/// with each row's sign.
fn register_marginals(
    amps: &mut [Complex64],
    mut rows: Vec<RegisterPauli>,
    width: usize,
) -> (Vec<f64>, Vec<bool>) {
    let mut gates = Vec::new();
    let mut pivots: Vec<usize> = Vec::with_capacity(rows.len());
    let mut flips = Vec::with_capacity(rows.len());
    for i in 0..rows.len() {
        let p = rows[i];
        let c = (0..width)
            .find(|b| !pivots.contains(b) && ((p.x | p.z) >> b) & 1 == 1)
            .expect("independent commuting rows keep a free letter");
        let mut step = |gate: RegisterGate, rows: &mut [RegisterPauli]| {
            for q in rows.iter_mut() {
                conjugate(q, &gate);
            }
            gates.push(gate);
        };
        for b in (0..width).filter(|b| !pivots.contains(b) && (p.x >> b) & 1 == 1) {
            if (p.z >> b) & 1 == 1 {
                step(RegisterGate::Sdg(b), &mut rows[i..]);
            }
            step(RegisterGate::H(b), &mut rows[i..]);
        }
        let zs: Vec<usize> = (0..width)
            .filter(|&b| b != c && (rows[i].z >> b) & 1 == 1)
            .collect();
        for b in zs {
            step(RegisterGate::Cx(b, c), &mut rows[i..]);
        }
        debug_assert!(rows[i].x == 0 && rows[i].z == 1 << c);
        flips.push(rows[i].phase4 == 2);
        pivots.push(c);
    }
    for gate in &gates {
        apply_to_amps(amps, gate);
    }
    let mut table = vec![0.0f64; 1 << pivots.len()];
    for (b, amp) in amps.iter().enumerate() {
        let index = pivots
            .iter()
            .enumerate()
            .fold(0, |acc, (i, &c)| acc | (((b >> c) & 1) << i));
        table[index] += amp.norm_sqr();
    }
    let mut total = 0.0;
    let cumulative = table
        .into_iter()
        .map(|p| {
            total += p;
            total
        })
        .collect();
    (cumulative, flips)
}

/// Where a measured row's outcome comes from.
enum RowOutcome {
    /// Uniform: the row flips a qubit held in `|0>`.
    Random,
    /// Fixed by the row's sign.
    Fixed(bool),
    /// Bit `index` of the drawn register outcome, flipped by the sign.
    Register { index: usize, flip: bool },
}

/// Terminal measurements of a unitary Clifford+T circuit, reduced once and drawn per shot.
///
/// The measured `Z` images `C† Z_q C` are row reduced: a row that flips a `|0>` qubit
/// takes a uniform bit independent of the rest, and the remaining rows act on the
/// register alone, where a Clifford maps the independent ones to single `Z` letters so
/// one table of `|φ|^2` marginals draws them all.
pub(super) struct TerminalSampler {
    outcome: Vec<RowOutcome>,
    /// Column `r` of the inverse of the row-combination matrix: the measured qubits'
    /// bits are the XOR of `inverse[r]` over rows `r` whose outcome is 1.
    inverse: Vec<Vec<u64>>,
    cumulative: Vec<f64>,
    measures: Vec<(usize, usize)>,
    row_of_qubit: Vec<usize>,
    num_classical_bits: usize,
}

impl TerminalSampler {
    /// Reduce `circuit`, or return `None` when its register would pass `max_width`
    /// qubits.
    pub(super) fn compile(circuit: &Circuit, max_width: usize) -> Result<Option<Self>> {
        let n = circuit.num_qubits;
        let mut state = CliffordRegister::new(n);
        let mut measures = Vec::new();
        for inst in &circuit.instructions {
            match inst {
                Instruction::Gate { gate, targets } => match gate {
                    Gate::T | Gate::Tdg => {
                        let theta = if matches!(gate, Gate::T) {
                            FRAC_PI_8
                        } else {
                            -FRAC_PI_8
                        };
                        if !state.rotate_z(targets[0], theta, max_width) {
                            return Ok(None);
                        }
                    }
                    _ => state.prefix.apply_state_gate(gate, targets).map_err(|_| {
                        PrismError::BackendUnsupported {
                            backend: "stabilizer_rank".into(),
                            operation: format!("gate `{}` in the Clifford register", gate.name()),
                        }
                    })?,
                },
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => measures.push((*qubit, *classical_bit)),
                Instruction::Barrier { .. } => {}
                _ => {
                    return Err(PrismError::IncompatibleBackend {
                        backend: "stabilizer_rank".into(),
                        reason: "the Clifford register samples terminal measurements of a unitary \
                             circuit"
                            .into(),
                    });
                }
            }
        }

        let mut measured: Vec<usize> = Vec::new();
        let mut row_of_qubit = vec![usize::MAX; n];
        for &(q, _) in &measures {
            if row_of_qubit[q] == usize::MAX {
                row_of_qubit[q] = measured.len();
                measured.push(q);
            }
        }
        let m = measured.len();
        let words = m.div_ceil(64).max(1);
        let mut rows: Vec<SignedPauli> = measured
            .iter()
            .map(|&q| state.prefix.conjugate_z(q))
            .collect();
        let mut inverse: Vec<Vec<u64>> = (0..m)
            .map(|r| {
                let mut col = vec![0u64; words];
                col[r >> 6] |= 1 << (r & 63);
                col
            })
            .collect();
        let mut outcome: Vec<Option<RowOutcome>> = (0..m).map(|_| None).collect();

        for f in (0..n).filter(|&f| state.slot[f].is_none()) {
            let Some(pivot) = (0..m).find(|&r| outcome[r].is_none() && bit(&rows[r].x, f)) else {
                continue;
            };
            outcome[pivot] = Some(RowOutcome::Random);
            let pivot_row = rows[pivot].clone();
            for r in 0..m {
                if r == pivot || !bit(&rows[r].x, f) {
                    continue;
                }
                mul_rows(&mut rows[r], &pivot_row);
                let (lo, hi) = inverse.split_at_mut(pivot.max(r));
                let (dst, src) = if pivot < r {
                    (&mut lo[pivot], &hi[0])
                } else {
                    (&mut hi[0], &lo[r])
                };
                for (d, s) in dst.iter_mut().zip(src) {
                    *d ^= s;
                }
            }
        }

        let mut reg_rows: Vec<(usize, RegisterPauli)> = (0..m)
            .filter(|&r| outcome[r].is_none())
            .map(|r| (r, state.restrict(&rows[r])))
            .collect();
        let width = state.width();
        let mut independent: Vec<usize> = Vec::new();
        for column in 0..2 * width {
            let has = |p: &RegisterPauli| {
                let word = if column < width { p.x } else { p.z };
                (word >> (column % width.max(1))) & 1 == 1
            };
            let Some(pivot) =
                (0..reg_rows.len()).find(|&i| !independent.contains(&i) && has(&reg_rows[i].1))
            else {
                continue;
            };
            independent.push(pivot);
            let pivot_pauli = reg_rows[pivot].1;
            let pivot_row = reg_rows[pivot].0;
            for (i, (r, pauli)) in reg_rows.iter_mut().enumerate() {
                if i != pivot && has(pauli) {
                    *pauli = pauli.mul(pivot_pauli);
                    let src = inverse[*r].clone();
                    for (d, s) in inverse[pivot_row].iter_mut().zip(&src) {
                        *d ^= s;
                    }
                }
            }
        }
        for (i, &(r, p)) in reg_rows.iter().enumerate() {
            if !independent.contains(&i) {
                debug_assert!(p.x == 0 && p.z == 0 && p.phase4 & 1 == 0);
                outcome[r] = Some(RowOutcome::Fixed(p.phase4 == 2));
            }
        }

        let rows = independent.iter().map(|&i| reg_rows[i].1).collect();
        let (cumulative, flips) = register_marginals(&mut state.amps, rows, width);
        for (index, (&i, flip)) in independent.iter().zip(flips).enumerate() {
            outcome[reg_rows[i].0] = Some(RowOutcome::Register { index, flip });
        }

        let outcome = outcome
            .into_iter()
            .map(|o| o.expect("every measured row is classified"))
            .collect();
        Ok(Some(Self {
            outcome,
            inverse,
            cumulative,
            measures,
            row_of_qubit,
            num_classical_bits: circuit.num_classical_bits,
        }))
    }

    pub(super) fn sample(&self, num_shots: usize, seed: u64) -> Vec<Vec<bool>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let total = *self
            .cumulative
            .last()
            .expect("the table has one entry at least");
        (0..num_shots)
            .map(|_| {
                let draw = rng.random::<f64>() * total;
                let index = self
                    .cumulative
                    .partition_point(|&c| c <= draw)
                    .min(self.cumulative.len() - 1);
                self.classical_bits(index, || rng.random::<bool>())
            })
            .collect()
    }

    /// Classical bits for register outcome `index`, drawing each uniform row from
    /// `coin`.
    fn classical_bits(&self, index: usize, mut coin: impl FnMut() -> bool) -> Vec<bool> {
        let mut bits = vec![0u64; self.inverse.first().map_or(1, Vec::len)];
        for (r, how) in self.outcome.iter().enumerate() {
            let one = match *how {
                RowOutcome::Random => coin(),
                RowOutcome::Fixed(one) => one,
                RowOutcome::Register { index: i, flip } => ((index >> i) & 1 == 1) ^ flip,
            };
            if one {
                for (b, s) in bits.iter_mut().zip(&self.inverse[r]) {
                    *b ^= s;
                }
            }
        }
        let mut classical = vec![false; self.num_classical_bits];
        for &(q, cb) in &self.measures {
            classical[cb] = bit(&bits, self.row_of_qubit[q]);
        }
        classical
    }
}

#[cfg(test)]
#[path = "clifford_register_tests.rs"]
mod tests;

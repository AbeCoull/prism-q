//! Terminal shots and exact output distributions for Clifford+T circuits as a Clifford
//! tableau over a dense register that holds only the qubits the T rotations reach.

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
#[cfg(not(prism_lowered_thresholds))]
pub(super) const MAX_REGISTER_QUBITS: usize = 20;
#[cfg(prism_lowered_thresholds)]
pub(super) const MAX_REGISTER_QUBITS: usize = 2;

/// The state `C (|0> ⊗ φ)`: a Clifford `C` over `n` qubits, `|0>` on every qubit that no
/// T rotation has reached, and dense amplitudes `φ` over the rest.
struct CliffordRegister {
    prefix: SignedCliffordPrefix,
    /// Qubit `q`'s bit in `amps`, or `None` while it holds `|0>`.
    slot: Vec<Option<usize>>,
    width: usize,
    /// `2^width` amplitudes, or the starting amplitude alone when only the width is
    /// tracked.
    amps: Vec<Complex64>,
    /// Amplitudes the rotations sweep: `2^width` per T gate, at the width after it.
    rotation_work: u64,
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
            width: 0,
            amps: vec![Complex64::new(1.0, 0.0)],
            rotation_work: 0,
        }
    }

    /// Return `P = C† Z_q C` with its support inside the register. When `P` flips a
    /// qubit still in `|0>`, CX gates controlled on that qubit, which leave the state
    /// alone, clear the flips from every other such qubit and fold into `C`, and the
    /// qubit joins the register. Returns `None` past `max_width`.
    fn admit_z(&mut self, q: usize, max_width: usize) -> Option<SignedPauli> {
        let p = self.prefix.conjugate_z(q);
        let Some(anchor) = (0..self.slot.len()).find(|&j| self.slot[j].is_none() && bit(&p.x, j))
        else {
            return Some(p);
        };
        if self.width == max_width {
            return None;
        }
        for i in 0..self.slot.len() {
            if i != anchor && self.slot[i].is_none() && bit(&p.x, i) {
                self.prefix
                    .fold_right_state_gate(&Gate::Cx, &[anchor, i])
                    .expect("CX folds into the prefix");
            }
        }
        self.slot[anchor] = Some(self.width);
        self.width += 1;
        Some(self.prefix.conjugate_z(q))
    }

    /// Apply `exp(-i θ Z_q)` after the Clifford: `exp(-i θ P)` on `|0> ⊗ φ` for
    /// `P = C† Z_q C`. Returns `false` past `max_width`.
    fn rotate_z(&mut self, q: usize, theta: f64, max_width: usize) -> bool {
        let Some(p) = self.admit_z(q, max_width) else {
            return false;
        };
        let len = self.amps.len();
        if len < 1 << self.width {
            self.amps.resize(2 * len, Complex64::new(0.0, 0.0));
        }
        let reg = self.restrict(&p);
        self.apply_rotation(reg, theta);
        true
    }

    /// Zero all but the `budget` largest amplitudes, keeping every amplitude strictly
    /// above the cut before any tie and ties in index order.
    fn cut(&mut self, prune: &mut Pruning) {
        let budget = prune.budget;
        if self.amps.len() <= budget {
            return;
        }
        prune.scratch.clear();
        prune.scratch.extend(self.amps.iter().map(|a| a.norm_sqr()));
        let (larger, &mut threshold, _) = prune
            .scratch
            .select_nth_unstable_by(budget - 1, |a, b| b.total_cmp(a));
        if threshold == 0.0 {
            return;
        }
        let mut ties_left = budget - larger.iter().filter(|&&m| m > threshold).count();
        let mut lost = 0.0;
        for amp in &mut self.amps {
            let m = amp.norm_sqr();
            if m > threshold {
                continue;
            }
            if m == threshold && ties_left > 0 {
                ties_left -= 1;
                continue;
            }
            if m != 0.0 {
                lost += m;
                prune.pruned += 1;
                *amp = Complex64::new(0.0, 0.0);
            }
        }
        prune.discarded += lost.sqrt();
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
    /// CX from every bit of `controls` onto `target`. The gates commute, so together
    /// they flip `target` where the controls have odd parity, in one pass.
    CxFan {
        controls: usize,
        target: usize,
    },
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
        RegisterGate::CxFan {
            controls,
            target: t,
        } => {
            let mut rest = controls;
            while rest != 0 {
                let c = rest.trailing_zeros() as usize;
                rest &= rest - 1;
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
}

fn apply_to_amps(amps: &mut [Complex64], gate: &RegisterGate) {
    match *gate {
        RegisterGate::H(m) => for_each_pair(amps, m, |_, a0, a1| {
            let (x, y) = (*a0, *a1);
            *a0 = (x + y) * FRAC_1_SQRT_2;
            *a1 = (x - y) * FRAC_1_SQRT_2;
        }),
        RegisterGate::Sdg(m) => for_each_pair(amps, m, |_, _, a1| {
            *a1 = Complex64::new(a1.im, -a1.re);
        }),
        RegisterGate::CxFan { controls, target } => for_each_pair(amps, target, |b, a0, a1| {
            if (b & controls).count_ones() & 1 == 1 {
                std::mem::swap(a0, a1);
            }
        }),
    }
}

/// Call `f(b, amps[b], amps[b | 1 << bit])` for every index `b` with `bit` clear.
#[inline(always)]
fn for_each_pair<F>(amps: &mut [Complex64], bit: usize, f: F)
where
    F: Fn(usize, &mut Complex64, &mut Complex64),
{
    let half = 1usize << bit;
    for (j, block) in amps.chunks_exact_mut(2 * half).enumerate() {
        let (lo, hi) = block.split_at_mut(half);
        for (k, (a0, a1)) in lo.iter_mut().zip(hi).enumerate() {
            f(2 * j * half + k, a0, a1);
        }
    }
}

/// Map independent commuting register Paulis to single `Z` letters with a Clifford `W`,
/// apply `W` to `amps`, and return the table of `|W φ|^2` over those letters with each
/// row's sign and the number of passes over `amps` that `W` took.
fn register_marginals(
    amps: &mut [Complex64],
    mut rows: Vec<RegisterPauli>,
    width: usize,
) -> (Vec<f64>, Vec<bool>, usize) {
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
        let controls = rows[i].z as usize & !(1 << c);
        if controls != 0 {
            step(
                RegisterGate::CxFan {
                    controls,
                    target: c,
                },
                &mut rows[i..],
            );
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
    (table, flips, gates.len())
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
    /// `|W φ|^2` indexed by register outcome.
    table: Vec<f64>,
    measures: Vec<(usize, usize)>,
    row_of_qubit: Vec<usize>,
    num_classical_bits: usize,
    register_width: usize,
    /// Passes over the register in the reduction: one per diagonalizing gate, plus the
    /// table fold.
    #[cfg(feature = "bench-internal")]
    reduction_passes: usize,
}

/// Amplitude budget of the register on the approximate probability route, with what its
/// cuts dropped.
pub(super) struct Pruning {
    budget: usize,
    /// Nonzero amplitudes zeroed, summed over the cuts.
    pub(super) pruned: usize,
    /// Sum over the cuts of the 2-norm of the amplitudes each one zeroed.
    pub(super) discarded: f64,
    /// `sum |φ|^2` once the circuit has run, before any renormalization; exactly 1.0 when
    /// no cut dropped anything.
    pub(super) kept_norm_sqr: f64,
    /// Nonzero amplitudes held once the circuit has run.
    pub(super) held: usize,
    scratch: Vec<f64>,
}

impl Pruning {
    /// `budget` must be at least 1.
    pub(super) fn new(budget: usize) -> Self {
        Self {
            budget,
            pruned: 0,
            discarded: 0.0,
            kept_norm_sqr: 1.0,
            held: 1,
            scratch: Vec::new(),
        }
    }
}

/// Run `circuit` through a register, with its amplitudes only when `dense`, pushing its
/// measurements onto `measures`; `None` when the register would pass `max_width` qubits.
/// With `prune`, a dense register is cut back to the budget after every rotation.
fn build_register(
    circuit: &Circuit,
    max_width: usize,
    dense: bool,
    mut prune: Option<&mut Pruning>,
    measures: &mut Vec<(usize, usize)>,
) -> Result<Option<CliffordRegister>> {
    let mut state = CliffordRegister::new(circuit.num_qubits);
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => match gate {
                Gate::T | Gate::Tdg => {
                    let admitted = if !dense {
                        state.admit_z(targets[0], max_width).is_some()
                    } else if matches!(gate, Gate::T) {
                        state.rotate_z(targets[0], FRAC_PI_8, max_width)
                    } else {
                        state.rotate_z(targets[0], -FRAC_PI_8, max_width)
                    };
                    if !admitted {
                        return Ok(None);
                    }
                    if let Some(prune) = prune.as_deref_mut() {
                        state.cut(prune);
                    }
                    state.rotation_work += 1 << state.width;
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
    Ok(Some(state))
}

/// Width and rotation work of the register of `circuit`, read off the tableau without
/// amplitudes; `None` exactly when [`TerminalSampler::compile`] returns `None`.
pub(super) fn register_price(circuit: &Circuit, max_width: usize) -> Result<Option<(usize, u64)>> {
    Ok(
        build_register(circuit, max_width, false, None, &mut Vec::new())?
            .map(|state| (state.width, state.rotation_work)),
    )
}

/// Register counts of one terminal Clifford+T circuit, read off the tableau without
/// amplitudes.
#[cfg(feature = "bench-internal")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliffordRegisterCensus {
    /// T and Tdg gates after lowering.
    pub t_count: usize,
    /// Register qubits once every T gate is admitted, uncapped.
    pub width: usize,
    /// Amplitudes the rotations sweep: `2^w` per T gate, `w` the width after it.
    pub rotation_work: u64,
    /// Full-width passes of the readout reduction, the table fold included; `None` past
    /// the width cap.
    pub reduction_passes: Option<usize>,
    /// Independent measured rows left on the register, the index bits of the outcome
    /// table; `None` past the width cap.
    pub register_rows: Option<usize>,
    /// Linked sets of register qubits in the product gauge, widest first.
    pub parts: Vec<RegisterPart>,
    /// Pairs of register rows compared while merging parts whose rows anticommute.
    pub row_pair_checks: u64,
}

/// One linked set of register qubits in the product gauge, where each anchoring T gate
/// adds a qubit in a known one-qubit state. Parts join where an excess rotation acts on
/// both, or where two measured register rows anticommute inside each of them.
#[cfg(feature = "bench-internal")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterPart {
    pub width: usize,
    /// T gates applied as a rotation over qubits already in the part.
    pub excess: usize,
    /// Rank of the register rows cut down to the part, the index bits of its table.
    pub rows: usize,
}

/// Parts of the register of `circuit` in the product gauge, from the tableau alone, with
/// the row pairs the merge compared. `circuit` must already have passed
/// [`build_register`].
#[cfg(feature = "bench-internal")]
fn register_parts(circuit: &Circuit) -> (Vec<RegisterPart>, u64) {
    let n = circuit.num_qubits;
    let mut prefix = SignedCliffordPrefix::identity(n);
    let mut slot: Vec<Option<usize>> = vec![None; n];
    let mut parent: Vec<usize> = Vec::new();
    let mut excess: Vec<u64> = Vec::new();
    let mut measured: Vec<usize> = Vec::new();
    let fold = |prefix: &mut SignedCliffordPrefix, gate: Gate, targets: &[usize]| {
        prefix
            .fold_right_state_gate(&gate, targets)
            .expect("Clifford folds into the prefix");
    };
    fn find(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    fn join(parent: &mut [usize], mask: u64) {
        let first = mask.trailing_zeros() as usize;
        let mut rest = mask & mask.wrapping_sub(1);
        while rest != 0 {
            let b = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            let (ra, rb) = (find(parent, first), find(parent, b));
            parent[rb] = ra;
        }
    }
    let slot_mask = |slot: &[Option<usize>], words: &[u64]| {
        slot.iter()
            .enumerate()
            .filter_map(|(q, s)| s.filter(|_| bit(words, q)))
            .fold(0u64, |m, b| m | 1 << b)
    };

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate {
                gate: Gate::T | Gate::Tdg,
                targets,
            } => {
                let q = targets[0];
                let p = prefix.conjugate_z(q);
                let Some(anchor) = (0..n).find(|&j| slot[j].is_none() && bit(&p.x, j)) else {
                    let support: Vec<u64> = p.x.iter().zip(&p.z).map(|(x, z)| x | z).collect();
                    let mask = slot_mask(&slot, &support);
                    if mask != 0 {
                        join(&mut parent, mask);
                        excess.push(mask);
                    }
                    continue;
                };
                for i in (0..n).filter(|&i| i != anchor && slot[i].is_none() && bit(&p.x, i)) {
                    fold(&mut prefix, Gate::Cx, &[anchor, i]);
                }
                // Fold the controlled rest of `P = A_anchor ⊗ R`: it fixes `|0>` on the
                // anchor and leaves the rotation acting on the anchor alone.
                let p = prefix.conjugate_z(q);
                for j in (0..n).filter(|&j| j != anchor) {
                    match (bit(&p.x, j), bit(&p.z, j)) {
                        (false, false) => {}
                        (true, false) => fold(&mut prefix, Gate::Cx, &[anchor, j]),
                        (false, true) => fold(&mut prefix, Gate::Cz, &[anchor, j]),
                        (true, true) => {
                            fold(&mut prefix, Gate::S, &[j]);
                            fold(&mut prefix, Gate::Cx, &[anchor, j]);
                            fold(&mut prefix, Gate::Sdg, &[j]);
                        }
                    }
                }
                debug_assert!({
                    let g = prefix.conjugate_z(q);
                    (0..n).all(|j| j == anchor || !(bit(&g.x, j) || bit(&g.z, j)))
                });
                slot[anchor] = Some(parent.len());
                parent.push(parent.len());
            }
            Instruction::Gate { gate, targets } => prefix
                .apply_state_gate(gate, targets)
                .expect("the register build accepted this gate"),
            Instruction::Measure { qubit, .. } if !measured.contains(qubit) => {
                measured.push(*qubit);
            }
            _ => {}
        }
    }

    let mut rows: Vec<SignedPauli> = measured.iter().map(|&q| prefix.conjugate_z(q)).collect();
    let mut random = vec![false; rows.len()];
    for f in (0..n).filter(|&f| slot[f].is_none()) {
        let Some(pivot) = (0..rows.len()).find(|&r| !random[r] && bit(&rows[r].x, f)) else {
            continue;
        };
        random[pivot] = true;
        let pivot_row = rows[pivot].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if r != pivot && bit(&row.x, f) {
                mul_rows(row, &pivot_row);
            }
        }
    }
    let mut basis: Vec<u128> = Vec::new();
    for (row, _) in rows.iter().zip(&random).filter(|(_, random)| !**random) {
        let mut v =
            u128::from(slot_mask(&slot, &row.x)) | u128::from(slot_mask(&slot, &row.z)) << 64;
        for b in &basis {
            v = v.min(v ^ b);
        }
        if v != 0 {
            basis.push(v);
        }
    }

    let groups = |parent: &mut [usize]| {
        let mut masks = vec![0u64; parent.len()];
        for b in 0..parent.len() {
            masks[find(parent, b)] |= 1 << b;
        }
        masks.retain(|&m| m != 0);
        masks
    };
    let mut checks = 0u64;
    let mut parts = groups(&mut parent);
    loop {
        let mut merged = false;
        for (i, &a) in basis.iter().enumerate() {
            for &b in &basis[i + 1..] {
                checks += 1;
                let (ax, az, bx, bz) = (a as u64, (a >> 64) as u64, b as u64, (b >> 64) as u64);
                let anti = (ax & bz) ^ (az & bx);
                let clash = parts
                    .iter()
                    .filter(|&&c| (anti & c).count_ones() & 1 == 1)
                    .fold(0u64, |m, &c| m | c);
                if clash != 0 {
                    join(&mut parent, clash);
                    parts = groups(&mut parent);
                    merged = true;
                }
            }
        }
        if !merged {
            break;
        }
    }

    let mut out: Vec<RegisterPart> = parts
        .iter()
        .map(|&c| {
            let cut = u128::from(c) | u128::from(c) << 64;
            let mut piece_basis: Vec<u128> = Vec::new();
            for &row in &basis {
                let mut v = row & cut;
                for b in &piece_basis {
                    v = v.min(v ^ b);
                }
                if v != 0 {
                    piece_basis.push(v);
                }
            }
            RegisterPart {
                width: c.count_ones() as usize,
                excess: excess.iter().filter(|&&m| m & c != 0).count(),
                rows: piece_basis.len(),
            }
        })
        .collect();
    out.sort_by(|a, b| b.width.cmp(&a.width).then(b.excess.cmp(&a.excess)));
    (out, checks)
}

#[cfg(feature = "bench-internal")]
pub(super) fn census(circuit: &Circuit) -> Result<CliffordRegisterCensus> {
    let mut measures = Vec::new();
    let state = build_register(circuit, circuit.num_qubits, false, None, &mut measures)?
        .expect("an uncapped register always fits");
    let (parts, row_pair_checks) = register_parts(circuit);
    debug_assert_eq!(
        parts.iter().map(|p| p.width).sum::<usize>(),
        state.width,
        "the product gauge anchors as many qubits as the register holds"
    );
    let t_count = circuit
        .instructions
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                Instruction::Gate {
                    gate: Gate::T | Gate::Tdg,
                    ..
                }
            )
        })
        .count();
    let (width, rotation_work) = (state.width, state.rotation_work);
    let reduced = (width <= MAX_REGISTER_QUBITS)
        .then(|| TerminalSampler::reduce(state, measures, circuit.num_classical_bits));
    Ok(CliffordRegisterCensus {
        t_count,
        width,
        rotation_work,
        reduction_passes: reduced.as_ref().map(|s| s.reduction_passes),
        register_rows: reduced
            .as_ref()
            .map(|s| s.table.len().trailing_zeros() as usize),
        parts,
        row_pair_checks,
    })
}

impl TerminalSampler {
    /// Reduce `circuit`, or return `None` when its register would pass `max_width`
    /// qubits.
    pub(super) fn compile(circuit: &Circuit, max_width: usize) -> Result<Option<Self>> {
        let mut measures = Vec::new();
        let Some(state) = build_register(circuit, max_width, true, None, &mut measures)? else {
            return Ok(None);
        };
        Ok(Some(Self::reduce(
            state,
            measures,
            circuit.num_classical_bits,
        )))
    }

    /// Like [`Self::compile`], with qubit `q` measured into classical bit `q` in place of
    /// the circuit's own measurements.
    pub(super) fn compile_every_qubit(circuit: &Circuit, max_width: usize) -> Result<Option<Self>> {
        let n = circuit.num_qubits;
        let Some(state) = build_register(circuit, max_width, true, None, &mut Vec::new())? else {
            return Ok(None);
        };
        Ok(Some(Self::reduce(
            state,
            (0..n).map(|q| (q, q)).collect(),
            n,
        )))
    }

    /// Like [`Self::compile_every_qubit`], with the register cut back to `prune`'s budget
    /// after every rotation. [`Self::distribution`] then sums to `prune.kept_norm_sqr`.
    pub(super) fn compile_every_qubit_pruned(
        circuit: &Circuit,
        max_width: usize,
        prune: &mut Pruning,
    ) -> Result<Option<Self>> {
        let n = circuit.num_qubits;
        let Some(state) = build_register(circuit, max_width, true, Some(prune), &mut Vec::new())?
        else {
            return Ok(None);
        };
        if prune.pruned > 0 {
            prune.kept_norm_sqr = state.amps.iter().map(|a| a.norm_sqr()).sum();
        }
        prune.held = state.amps.iter().filter(|a| a.norm_sqr() != 0.0).count();
        Ok(Some(Self::reduce(
            state,
            (0..n).map(|q| (q, q)).collect(),
            n,
        )))
    }

    pub(super) fn register_width(&self) -> usize {
        self.register_width
    }

    fn reduce(
        mut state: CliffordRegister,
        measures: Vec<(usize, usize)>,
        num_classical_bits: usize,
    ) -> Self {
        let n = state.slot.len();
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
        let width = state.width;
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
        let (table, flips, _diagonalizing_passes) =
            register_marginals(&mut state.amps, rows, width);
        for (index, (&i, flip)) in independent.iter().zip(flips).enumerate() {
            outcome[reg_rows[i].0] = Some(RowOutcome::Register { index, flip });
        }

        let outcome = outcome
            .into_iter()
            .map(|o| o.expect("every measured row is classified"))
            .collect();
        Self {
            outcome,
            inverse,
            table,
            measures,
            row_of_qubit,
            num_classical_bits,
            register_width: width,
            #[cfg(feature = "bench-internal")]
            reduction_passes: _diagonalizing_passes + 1,
        }
    }

    pub(super) fn sample(&self, num_shots: usize, seed: u64) -> Vec<Vec<bool>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut total = 0.0;
        let cumulative: Vec<f64> = self
            .table
            .iter()
            .map(|p| {
                total += p;
                total
            })
            .collect();
        (0..num_shots)
            .map(|_| {
                let draw = rng.random::<f64>() * total;
                let index = cumulative
                    .partition_point(|&c| c <= draw)
                    .min(cumulative.len() - 1);
                self.classical_bits(index, || rng.random::<bool>())
            })
            .collect()
    }

    /// Exact probability of every classical outcome, indexed with classical bit `i` at
    /// bit `i`. Dense over `2^num_classical_bits` entries.
    pub(super) fn distribution(&self) -> Vec<f64> {
        let mut writer = vec![usize::MAX; self.num_classical_bits];
        for &(q, cb) in &self.measures {
            writer[cb] = self.row_of_qubit[q];
        }
        let key_of = |r: usize| {
            writer
                .iter()
                .enumerate()
                .filter(|&(_, &row)| row != usize::MAX && bit(&self.inverse[r], row))
                .fold(0usize, |acc, (cb, _)| acc | 1 << cb)
        };
        let mut base = 0usize;
        let mut register_masks = vec![0usize; self.table.len().trailing_zeros() as usize];
        let mut coin_masks = Vec::new();
        for (r, how) in self.outcome.iter().enumerate() {
            let key = key_of(r);
            match *how {
                RowOutcome::Random if key != 0 => coin_masks.push(key),
                RowOutcome::Random | RowOutcome::Fixed(false) => {}
                RowOutcome::Fixed(true) => base ^= key,
                RowOutcome::Register { index, flip } => {
                    register_masks[index] = key;
                    if flip {
                        base ^= key;
                    }
                }
            }
        }
        // Counting from `i - 1` to `i` flips bits `0..=trailing_zeros(i)`.
        let gray_steps = |masks: &mut [usize]| {
            for j in 1..masks.len() {
                masks[j] ^= masks[j - 1];
            }
        };
        gray_steps(&mut register_masks);
        gray_steps(&mut coin_masks);

        let coin_weight = 0.5f64.powi(coin_masks.len() as i32);
        let mut dist = vec![0.0f64; 1 << self.num_classical_bits];
        let mut key = base;
        for (index, &p) in self.table.iter().enumerate() {
            if index > 0 {
                key ^= register_masks[index.trailing_zeros() as usize];
            }
            let p = p * coin_weight;
            let mut coin_key = key;
            dist[coin_key] += p;
            for a in 1..1usize << coin_masks.len() {
                coin_key ^= coin_masks[a.trailing_zeros() as usize];
                dist[coin_key] += p;
            }
        }
        dist
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

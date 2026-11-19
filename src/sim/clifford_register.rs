//! Terminal shots and exact output distributions for Clifford+T circuits as a Clifford
//! tableau over a dense register that holds only the qubits the T rotations reach.

use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_8};

use crate::backend::simd::norm_sqr_to_slice;
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::qec::camps_prefix::{SignedCliffordPrefix, SignedPauli};

/// Widest register [`TerminalSampler::compile_every_qubit`] builds. Up to this width the
/// split register holds every part whole.
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
    /// `self · other`; anticommuting Hermitian factors leave an odd `phase4`.
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
pub(super) fn letter_phase(ax: u64, az: u64, bx: u64, bz: u64) -> u32 {
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
        apply_rotation(&mut self.amps, reg, theta);
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
}

/// `φ ← cos θ φ - i sin θ P φ`, with `P|b> = i^(phase4 + #Y) (-1)^|b & z| |b ^ x>`.
fn apply_rotation(amps: &mut [Complex64], p: RegisterPauli, theta: f64) {
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
        for (b, amp) in amps.iter_mut().enumerate() {
            *amp *= cos + sign(b);
        }
        return;
    }
    let x = p.x as usize;
    let low = x & x.wrapping_neg();
    for b in 0..amps.len() {
        if b & low != 0 {
            continue;
        }
        let b2 = b ^ x;
        let (a, a2) = (amps[b], amps[b2]);
        amps[b] = a * cos + sign(b2) * a2;
        amps[b2] = a2 * cos + sign(b) * a;
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
    let independent = rows.len();
    let (pivots, flips, passes) = diagonalize(amps, &mut rows, independent, width);
    (fold_table(amps, &pivots, width), flips, passes)
}

/// Map the first `independent` of `rows`, independent and commuting, to single `Z` letters
/// with a Clifford `W`, apply `W` to `amps`, and conjugate the rest of `rows` through it.
/// Returns the letter each mapped row lands on, whether it lands on `-Z`, and the number of
/// passes over `amps` that `W` took.
fn diagonalize(
    amps: &mut [Complex64],
    rows: &mut [RegisterPauli],
    independent: usize,
    width: usize,
) -> (Vec<usize>, Vec<bool>, usize) {
    let mut gates = Vec::new();
    let mut pivots: Vec<usize> = Vec::with_capacity(independent);
    let mut flips = Vec::with_capacity(independent);
    for i in 0..independent {
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
    (pivots, flips, gates.len())
}

/// Sum `|amps[b]|^2` into `table[i]` over every `b` whose bit `pivots[j]` is bit `j` of
/// `i`, adding in increasing `b`.
///
/// When the low `lead` pivots are the identity, a block of `2^lead` amplitudes lands on
/// one contiguous run of the table; otherwise a lookup over the low byte of `b` gives
/// each amplitude's offset from its block's base.
fn fold_table(amps: &[Complex64], pivots: &[usize], width: usize) -> Vec<f64> {
    let gather = |b: usize| {
        pivots
            .iter()
            .enumerate()
            .fold(0, |acc, (i, &c)| acc | (((b >> c) & 1) << i))
    };
    let lead = pivots
        .iter()
        .enumerate()
        .take_while(|&(i, &c)| i == c)
        .count();
    let mut table = vec![0.0f64; 1 << pivots.len()];
    if lead >= width.min(8) {
        for (h, block) in amps.chunks(1 << lead).enumerate() {
            let start = gather(h << lead);
            let run = &mut table[start..start + block.len()];
            if pivots.len() == width {
                norm_sqr_to_slice(block, run);
            } else {
                for (t, amp) in run.iter_mut().zip(block) {
                    *t += amp.norm_sqr();
                }
            }
        }
    } else {
        let mut low = [0usize; 256];
        for (j, offset) in low.iter_mut().enumerate() {
            *offset = gather(j);
        }
        for (h, block) in amps.chunks(256).enumerate() {
            let base = gather(h << 8);
            for (amp, &offset) in block.iter().zip(&low) {
                table[base | offset] += amp.norm_sqr();
            }
        }
    }
    table
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

/// Terminal measurements of a unitary Clifford+T circuit over one dense register, reduced
/// once and read out as an exact distribution.
///
/// The measured `Z` images `C† Z_q C` are row reduced: a row that flips a `|0>` qubit
/// takes a uniform bit independent of the rest, and the remaining rows act on the
/// register alone, where a Clifford maps the independent ones to single `Z` letters so
/// one table of `|φ|^2` marginals holds them all.
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
                    let sweep = 1u64.checked_shl(state.width as u32).unwrap_or(u64::MAX);
                    state.rotation_work = state.rotation_work.saturating_add(sweep);
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
/// amplitudes; `None` exactly when [`TerminalSampler::compile_every_qubit`] returns `None`.
pub(super) fn register_price(circuit: &Circuit, max_width: usize) -> Result<Option<(usize, u64)>> {
    Ok(
        build_register(circuit, max_width, false, None, &mut Vec::new())?
            .map(|state| (state.width, state.rotation_work)),
    )
}

/// Fold the controlled register part of `p = A_anchor ⊗ R`, `A` an `X` or `Y` on `anchor`,
/// into `prefix`. Controlled on a qubit in `|0>` it leaves the state alone, and it leaves
/// the rotation about `p` acting on the anchor and on `Z` letters of qubits held in `|0>`,
/// which act on the state as `+1`. Those letters stay unfolded: the choice sets the letters
/// of later anchors, so seeded shots depend on it.
fn fold_anchor_rest(
    prefix: &mut SignedCliffordPrefix,
    slot: &[Option<usize>],
    anchor: usize,
    p: &SignedPauli,
) {
    let mut fold = |gate: Gate, targets: &[usize]| {
        prefix
            .fold_right_state_gate(&gate, targets)
            .expect("Clifford folds into the prefix");
    };
    for j in (0..slot.len()).filter(|&j| j != anchor && slot[j].is_some()) {
        match (bit(&p.x, j), bit(&p.z, j)) {
            (false, false) => {}
            (true, false) => fold(Gate::Cx, &[anchor, j]),
            (false, true) => fold(Gate::Cz, &[anchor, j]),
            (true, true) => {
                fold(Gate::S, &[j]);
                fold(Gate::Cx, &[anchor, j]);
                fold(Gate::Sdg, &[j]);
            }
        }
    }
}

fn find(parent: &mut [usize], mut a: usize) -> usize {
    while parent[a] != a {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    a
}

/// Union every register bit of `mask` into one set.
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

/// The sets of `parent` as register bit masks.
fn groups(parent: &mut [usize]) -> Vec<u64> {
    let mut masks = vec![0u64; parent.len()];
    for b in 0..parent.len() {
        masks[find(parent, b)] |= 1 << b;
    }
    masks.retain(|&m| m != 0);
    masks
}

/// Register counts of one terminal Clifford+T circuit, read off the tableau without
/// amplitudes.
#[cfg(feature = "bench-internal")]
#[derive(Debug, Clone, PartialEq)]
pub struct CliffordRegisterCensus {
    /// T and Tdg gates after lowering.
    pub t_count: usize,
    /// Register qubits once every T gate is admitted, uncapped.
    pub width: usize,
    /// Amplitudes the rotations sweep: `2^w` per T gate, `w` the width after it, saturating.
    pub rotation_work: u64,
    /// Full-width passes of the readout reduction, the table fold included; `None` past
    /// the width cap.
    pub reduction_passes: Option<usize>,
    /// Independent measured rows left on the register, the index bits of the outcome
    /// table; `None` past the width cap.
    pub register_rows: Option<usize>,
    /// The register split into parts; `None` past 64 anchors.
    pub split: Option<RegisterSplit>,
}

/// The register of one circuit split into parts in the product gauge.
#[cfg(feature = "bench-internal")]
#[derive(Debug, Clone, PartialEq)]
pub struct RegisterSplit {
    /// Linked sets of register qubits, widest first, from the pairwise merge run to a
    /// fixed point.
    pub parts: Vec<RegisterPart>,
    /// Pairs of register rows compared while merging parts whose rows anticommute.
    pub row_pair_checks: u64,
    /// Part widths, widest first, of the canonical merge: the connected components of
    /// the binary matroid of the row commutation forms of the units, the sets that excess
    /// rotations alone link. They depend on the row space alone. A component can still
    /// split into parts whose forms each sum to zero, so these can be wider than `parts`.
    pub canonical_widths: Vec<usize>,
    /// Whether the canonical merge yields the same parts as the fixed point; `false`
    /// when a canonical part holds several fixed point parts, which are still valid.
    pub canonical_agrees: bool,
    /// The cut the sampler would take of the widest canonical part, whose sides are unions
    /// of units so no rotation crosses it; `None` when that part is one unit.
    pub cut: Option<RegisterCut>,
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

#[cfg(feature = "bench-internal")]
impl RegisterPart {
    /// Amplitude updates of the part's dense readout: about `3 (k + 1)` passes over its
    /// `2^k` amplitudes.
    pub fn readout_work(&self) -> f64 {
        3.0 * (self.width + 1) as f64 * 2f64.powi(self.width as i32)
    }
}

/// A part cut in two, each side held densely and the row pairs that anticommute across
/// the cut paid per shot.
#[cfg(feature = "bench-internal")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegisterCut {
    /// Side widths, wider first.
    pub sides: (usize, usize),
    /// Row pairs whose pieces anticommute across the cut, `E`.
    pub pairs: usize,
    /// Amplitude updates at 1024 shots: `40 (2^kL + 2^kR)` to build the sides, then
    /// `(E + 3) 2^E` a shot.
    pub work: f64,
}

/// Parts of the register of `circuit` in the product gauge, from the tableau alone, or
/// `None` past the 64 anchors [`ProductGauge::build`] holds. `circuit` must already have
/// passed [`build_register`].
#[cfg(feature = "bench-internal")]
fn register_parts(circuit: &Circuit) -> Option<RegisterSplit> {
    let gauge = ProductGauge::build(circuit)?;
    let register = &gauge.register;
    let excess: Vec<u64> = gauge.excess.iter().map(|(p, _)| p.x | p.z).collect();
    let mut measures = Vec::new();
    for inst in &circuit.instructions {
        if let Instruction::Measure {
            qubit,
            classical_bit,
        } = inst
        {
            measures.push((*qubit, *classical_bit));
        }
    }
    let rows = measured_rows(&register.prefix, &register.slot, &measures);
    let basis = split::row_basis(
        (0..rows.rows.len())
            .filter(|&r| !rows.uniform[r])
            .map(|r| register.restrict(&rows.rows[r])),
    );

    let anti = |a: u128, b: u128| {
        let (ax, az, bx, bz) = (a as u64, (a >> 64) as u64, b as u64, (b >> 64) as u64);
        (ax & bz) ^ (az & bx)
    };
    let units = split::units(register.width, &gauge.excess);
    let mut parent: Vec<usize> = (0..register.width).collect();
    for &mask in &excess {
        join(&mut parent, mask);
    }
    let mut checks = 0u64;
    let mut parts = units.clone();
    loop {
        let mut merged = false;
        for (i, &a) in basis.iter().enumerate() {
            for &b in &basis[i + 1..] {
                checks += 1;
                let anti = anti(a, b);
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

    let forms = split::unit_forms(&units, &basis);
    let mut canonical = split::canonical_parts(&units, &forms);
    debug_assert!(
        canonical.iter().all(|&c| {
            let mut sum = vec![0u64; basis.len()];
            for (u, form) in units.iter().zip(&forms).filter(|(u, _)| *u & c != 0) {
                debug_assert_eq!(u & !c, 0, "a canonical part holds whole units");
                for (s, f) in sum.iter_mut().zip(form) {
                    *s ^= f;
                }
            }
            sum.iter().all(|&w| w == 0)
        }),
        "the rows commute inside every canonical part"
    );
    let mut canonical_widths: Vec<usize> =
        canonical.iter().map(|c| c.count_ones() as usize).collect();
    canonical_widths.sort_unstable_by(|a, b| b.cmp(a));
    let widest = canonical
        .iter()
        .copied()
        .max_by_key(|c| (c.count_ones(), std::cmp::Reverse(c.trailing_zeros())));
    let mut fixed = parts.clone();
    fixed.sort_unstable();
    canonical.sort_unstable();
    let canonical_agrees = canonical == fixed;

    let mut out: Vec<(u64, RegisterPart)> = parts
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
            let part = RegisterPart {
                width: c.count_ones() as usize,
                excess: excess.iter().filter(|&&m| m & c != 0).count(),
                rows: piece_basis.len(),
            };
            (c, part)
        })
        .collect();
    out.sort_by(|(_, a), (_, b)| b.width.cmp(&a.width).then(b.excess.cmp(&a.excess)));
    let cut = widest.and_then(|part| {
        let inside: Vec<usize> = (0..units.len()).filter(|&i| units[i] & part != 0).collect();
        let inside_units: Vec<u64> = inside.iter().map(|&i| units[i]).collect();
        let inside_forms: Vec<&[u64]> = inside.iter().map(|&i| &forms[i][..]).collect();
        split::frozen_cut(&inside_units, &inside_forms).map(|cut| RegisterCut {
            sides: (
                cut.left_width.max(cut.right_width),
                cut.left_width.min(cut.right_width),
            ),
            pairs: cut.pairs,
            work: cut.work as f64,
        })
    });
    Some(RegisterSplit {
        parts: out.into_iter().map(|(_, part)| part).collect(),
        row_pair_checks: checks,
        canonical_widths,
        canonical_agrees,
        cut,
    })
}

#[cfg(feature = "bench-internal")]
pub(super) fn census(circuit: &Circuit) -> Result<CliffordRegisterCensus> {
    let mut measures = Vec::new();
    let state = build_register(circuit, circuit.num_qubits, false, None, &mut measures)?
        .expect("an uncapped register always fits");
    let split = register_parts(circuit);
    if let Some(split) = &split {
        debug_assert_eq!(
            split.parts.iter().map(|p| p.width).sum::<usize>(),
            state.width,
            "the product gauge anchors as many qubits as the register holds"
        );
    }
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
        split,
    })
}

/// The `Z` images `C† Z_q C` of the measured qubits, in order of first measurement, with
/// the rows that flip a qubit held in `|0>` split off as uniform bits.
struct MeasuredRows {
    row_of_qubit: Vec<usize>,
    /// The rows after the reduction: a uniform row is the only one left flipping its
    /// qubit, and no other row flips a qubit held in `|0>`.
    rows: Vec<SignedPauli>,
    uniform: Vec<bool>,
    /// Column `r` of the inverse of the row-combination matrix: the measured qubits' bits
    /// are the XOR of `inverse[r]` over rows `r` whose outcome is 1.
    inverse: Vec<Vec<u64>>,
}

fn measured_rows(
    prefix: &SignedCliffordPrefix,
    slot: &[Option<usize>],
    measures: &[(usize, usize)],
) -> MeasuredRows {
    let n = slot.len();
    let mut measured: Vec<usize> = Vec::new();
    let mut row_of_qubit = vec![usize::MAX; n];
    for &(q, _) in measures {
        if row_of_qubit[q] == usize::MAX {
            row_of_qubit[q] = measured.len();
            measured.push(q);
        }
    }
    let m = measured.len();
    let words = m.div_ceil(64).max(1);
    let mut rows: Vec<SignedPauli> = measured.iter().map(|&q| prefix.conjugate_z(q)).collect();
    let mut inverse: Vec<Vec<u64>> = (0..m)
        .map(|r| {
            let mut col = vec![0u64; words];
            col[r >> 6] |= 1 << (r & 63);
            col
        })
        .collect();
    let mut uniform = vec![false; m];
    for f in (0..n).filter(|&f| slot[f].is_none()) {
        let Some(pivot) = (0..m).find(|&r| !uniform[r] && bit(&rows[r].x, f)) else {
            continue;
        };
        uniform[pivot] = true;
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
    MeasuredRows {
        row_of_qubit,
        rows,
        uniform,
        inverse,
    }
}

impl TerminalSampler {
    /// Reduce `circuit` with qubit `q` measured into classical bit `q` in place of the
    /// circuit's own measurements, or return `None` when its register would pass
    /// `max_width` qubits.
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
        let MeasuredRows {
            row_of_qubit,
            rows,
            uniform,
            mut inverse,
        } = measured_rows(&state.prefix, &state.slot, &measures);
        let m = rows.len();
        let mut outcome: Vec<Option<RowOutcome>> = uniform
            .iter()
            .map(|&u| u.then_some(RowOutcome::Random))
            .collect();

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
}

#[path = "clifford_parts.rs"]
mod parts;
#[path = "clifford_split.rs"]
mod split;
#[cfg(feature = "bench-internal")]
use parts::ProductGauge;
pub(super) use parts::RegisterExpectations;
pub(super) use split::{SplitPlan, SplitSampler};

#[cfg(test)]
#[path = "clifford_register_tests.rs"]
mod tests;

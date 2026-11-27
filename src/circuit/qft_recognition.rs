//! Recognition of expanded QFT gate sequences as [`Gate::QftBlock`], and the textbook
//! expansion that reverses it.

use std::borrow::Cow;
use std::f64::consts::PI;

use num_complex::Complex64;

use super::region::any_gate;
use super::{Circuit, GuardedRegion, Instruction, smallvec};
use crate::gates::Gate;

/// Narrowest QFT on part of the register that becomes a block. Below it the gates
/// stay, since fusion's tiled passes run a QFT that small in a sweep or two while the
/// block costs an FFT pass plus a bit-reversal pass and splits the fusion window.
const MIN_SUBRANGE_QFT_QUBITS: usize = 10;

/// Largest `|e^{i theta} - e^{i pi / 2^d}|` a controlled phase may carry and match.
const PHASE_EPS: f64 = 1e-12;

/// Replace each exact expanded QFT on qubits `0..num` with one [`Gate::QftBlock`],
/// the range the CPU statevector FFT runs natively.
///
/// Matches every sequence [`qft_textbook_steps`](super::qft_textbook_steps) emits:
/// forward or inverse, with or without the bit-reversal swaps, in either label order,
/// with the controlled phases of each column in any order and either qubit as
/// control, and the swaps in any order. The gates must be consecutive top-level
/// instructions. A block narrower than the register needs at least
/// `MIN_SUBRANGE_QFT_QUBITS` (10) qubits. Returns `Cow::Borrowed` when nothing
/// matches.
pub fn recognize_qft_blocks(circuit: &Circuit) -> Cow<'_, Circuit> {
    let insts = &circuit.instructions;
    let mut out: Option<Vec<Instruction>> = None;
    let mut copied = 0;
    let mut i = 0;
    while i < insts.len() {
        let found = match &insts[i] {
            Instruction::Gate { gate: Gate::H, .. } => {
                match_forward(insts, i).or_else(|| match_inverse(insts, i))
            }
            Instruction::Gate {
                gate: Gate::Swap, ..
            } => match_inverse(insts, i),
            _ => None,
        };
        let Some(m) = found.filter(|m| {
            m.start == 0
                && m.num <= u8::MAX as usize
                && (m.num == circuit.num_qubits || m.num >= MIN_SUBRANGE_QFT_QUBITS)
        }) else {
            i += 1;
            continue;
        };
        let out = out.get_or_insert_with(|| Vec::with_capacity(insts.len()));
        out.extend_from_slice(&insts[copied..i]);
        out.push(Instruction::Gate {
            gate: Gate::QftBlock {
                start: m.start as u8,
                num: m.num as u8,
                inverse: m.inverse,
                swaps: m.swaps,
                big_endian: m.big_endian,
            },
            targets: (m.start..m.start + m.num).collect(),
        });
        i = m.end;
        copied = i;
    }
    match out {
        None => Cow::Borrowed(circuit),
        Some(mut out) => {
            out.extend_from_slice(&insts[copied..]);
            Cow::Owned(circuit.with_instructions(out))
        }
    }
}

/// A recognized QFT ending before instruction `end`.
struct QftMatch {
    end: usize,
    start: usize,
    num: usize,
    inverse: bool,
    swaps: bool,
    big_endian: bool,
}

#[derive(Clone, Copy)]
enum Op {
    H(usize),
    Phase(usize, usize, Complex64),
    Swap(usize, usize),
    Other,
}

fn op(insts: &[Instruction], i: usize) -> Op {
    let Some(Instruction::Gate { gate, targets }) = insts.get(i) else {
        return Op::Other;
    };
    match gate {
        Gate::H => Op::H(targets[0]),
        Gate::Swap => Op::Swap(targets[0], targets[1]),
        Gate::Cu(_) => gate
            .controlled_phase()
            .map_or(Op::Other, |phase| Op::Phase(targets[0], targets[1], phase)),
        _ => Op::Other,
    }
}

/// The other qubit of a two-qubit `(a, b)` that touches `q`.
fn partner(a: usize, b: usize, q: usize) -> Option<usize> {
    if a == q {
        Some(b)
    } else if b == q {
        Some(a)
    } else {
        None
    }
}

/// Whether `phase` is `e^{sign * i pi / 2^d}`.
fn phase_matches(phase: Complex64, d: usize, sign: f64) -> bool {
    let theta = sign * PI * 0.5f64.powi(d as i32);
    (phase - Complex64::from_polar(1.0, theta)).norm() <= PHASE_EPS
}

/// Set of distances below 256.
#[derive(Default)]
struct Distances([u64; 4]);

impl Distances {
    /// Insert `d`, false when it was present or out of range.
    fn insert(&mut self, d: usize) -> bool {
        if d >= 256 {
            return false;
        }
        let (word, bit) = (d / 64, 1u64 << (d % 64));
        let fresh = self.0[word] & bit == 0;
        self.0[word] |= bit;
        fresh
    }
}

/// Match `len` controlled phases joining `q` to `q + dir * d` for each `d` in
/// `1..=len`, in any order, with phase `e^{sign * i pi / 2^d}`. Returns the index
/// after them.
fn match_column(
    insts: &[Instruction],
    mut j: usize,
    q: usize,
    dir: isize,
    len: usize,
    sign: f64,
) -> Option<usize> {
    let mut seen = Distances::default();
    for _ in 0..len {
        let Op::Phase(a, b, phase) = op(insts, j) else {
            return None;
        };
        let delta = (partner(a, b, q)? as isize - q as isize) * dir;
        if delta < 1 || delta as usize > len {
            return None;
        }
        let d = delta as usize;
        if !seen.insert(d) || !phase_matches(phase, d, sign) {
            return None;
        }
        j += 1;
    }
    Some(j)
}

/// Match the `num / 2` swaps of `lo..lo + num` in any order from `j`, returning the
/// index after them.
fn match_swap_set(insts: &[Instruction], mut j: usize, lo: usize, num: usize) -> Option<usize> {
    let mirror = 2 * lo + num - 1;
    let mut seen = Distances::default();
    for _ in 0..num / 2 {
        let Op::Swap(a, b) = op(insts, j) else {
            return None;
        };
        let low = a.min(b);
        if low < lo || a + b != mirror || !seen.insert(low - lo) {
            return None;
        }
        j += 1;
    }
    Some(j)
}

/// Match a forward QFT whose first H is instruction `i`. The first column fixes the
/// width and the label direction; the swaps are taken when all of them follow.
fn match_forward(insts: &[Instruction], i: usize) -> Option<QftMatch> {
    let Op::H(top) = op(insts, i) else {
        return None;
    };
    let mut j = i + 1;
    let mut dir = 0isize;
    let mut seen = Distances::default();
    let mut widest = 0;
    while let Op::Phase(a, b, phase) = op(insts, j) {
        let Some(other) = partner(a, b, top) else {
            break;
        };
        let delta = other as isize - top as isize;
        if dir == 0 {
            dir = delta.signum();
        } else if delta.signum() != dir {
            break;
        }
        let d = delta.unsigned_abs();
        if !seen.insert(d) || !phase_matches(phase, d, 1.0) {
            return None;
        }
        widest = widest.max(d);
        j += 1;
    }
    let num = j - i;
    if num < 2 || widest != num - 1 {
        return None;
    }
    let qubit = |c: usize| (top as isize + dir * c as isize) as usize;
    for c in 1..num {
        if !matches!(op(insts, j), Op::H(q) if q == qubit(c)) {
            return None;
        }
        j = match_column(insts, j + 1, qubit(c), dir, num - 1 - c, 1.0)?;
    }
    let start = top.min(qubit(num - 1));
    let after_swaps = match_swap_set(insts, j, start, num);
    Some(QftMatch {
        end: after_swaps.unwrap_or(j),
        start,
        num,
        inverse: false,
        swaps: after_swaps.is_some(),
        big_endian: dir > 0,
    })
}

/// Match an inverse QFT starting at instruction `i`, its swaps or its first H.
///
/// After the swaps the width is fixed and every column must be present. Without
/// them the match takes columns while they continue the pattern, since every
/// prefix of the no-swap inverse is itself a narrower no-swap inverse.
fn match_inverse(insts: &[Instruction], i: usize) -> Option<QftMatch> {
    let mut j = i;
    let mut range = None;
    if let Op::Swap(a, b) = op(insts, i) {
        let mirror = a + b;
        let mut lo = a.min(b);
        while let Op::Swap(a, b) = op(insts, j + 1) {
            if a + b != mirror {
                break;
            }
            lo = lo.min(a.min(b));
            j += 1;
        }
        let num = mirror - 2 * lo + 1;
        j = match_swap_set(insts, i, lo, num)?;
        range = Some((lo, num));
    }

    let Op::H(first) = op(insts, j) else {
        return None;
    };
    j += 1;
    let dir = match range {
        Some((lo, _)) if first == lo => 1,
        Some((lo, num)) if first == lo + num - 1 => -1,
        Some(_) => return None,
        None => {
            let Op::Phase(a, b, _) = op(insts, j) else {
                return None;
            };
            match partner(a, b, first)? as isize - first as isize {
                1 => 1,
                -1 => -1,
                _ => return None,
            }
        }
    };

    let mut num = 1;
    loop {
        if range.is_some_and(|(_, width)| num == width) {
            break;
        }
        let q = first as isize + dir * num as isize;
        let column = if q < 0 {
            None
        } else {
            match_column(insts, j, q as usize, -dir, num, -1.0)
                .filter(|&k| matches!(op(insts, k), Op::H(h) if h == q as usize))
        };
        match column {
            Some(k) => {
                j = k + 1;
                num += 1;
            }
            None if range.is_some() => return None,
            None => break,
        }
    }
    if num < 2 {
        return None;
    }
    let last = (first as isize + dir * (num as isize - 1)) as usize;
    Some(QftMatch {
        end: j,
        start: first.min(last),
        num,
        inverse: true,
        swaps: range.is_some(),
        big_endian: dir < 0,
    })
}

/// One step of the textbook QFT decomposition yielded by [`qft_textbook_steps`].
#[derive(Debug, Clone, Copy)]
pub enum QftTextbookStep {
    Hadamard(usize),
    CPhase {
        control: usize,
        target: usize,
        theta: f64,
    },
    Swap(usize, usize),
}

fn neg_pow2(exp: usize) -> f64 {
    if exp <= 1022 {
        f64::from_bits((1023 - exp as u64) << 52)
    } else if exp <= 1074 {
        f64::from_bits(1u64 << (1074 - exp))
    } else {
        0.0
    }
}

/// Yield the textbook decomposition of the [`Gate::QftBlock`] with these fields.
///
/// Forward order: for each qubit `q` from `num - 1` down to `0`, `H q[start+q]`
/// followed by `cphase(pi / 2^(q-k))` controlled by `q[start+k]` for `k in 0..q`,
/// then, with `swaps`, the bit-reversal swaps. `big_endian` maps `q[start+i]` to
/// `q[start+num-1-i]` throughout, and `inverse` yields the forward sequence
/// reversed with negated angles.
pub fn qft_textbook_steps(
    start: usize,
    num: usize,
    inverse: bool,
    swaps: bool,
    big_endian: bool,
) -> impl Iterator<Item = QftTextbookStep> {
    let label = move |i: usize| {
        if big_endian {
            start + num - 1 - i
        } else {
            start + i
        }
    };
    let columns = (0..num).rev().flat_map(move |q| {
        let head = std::iter::once(QftTextbookStep::Hadamard(label(q)));
        let phases = (0..q).map(move |k| QftTextbookStep::CPhase {
            control: label(k),
            target: label(q),
            theta: std::f64::consts::TAU * neg_pow2(q - k + 1),
        });
        head.chain(phases)
    });
    let swap_count = if swaps { num / 2 } else { 0 };
    let forward = columns
        .chain((0..swap_count).map(move |i| QftTextbookStep::Swap(start + i, start + num - 1 - i)));
    let (forward, reversed) = if inverse {
        (None, Some(forward.rev().map(QftTextbookStep::adjoint)))
    } else {
        (Some(forward), None)
    };
    forward
        .into_iter()
        .flatten()
        .chain(reversed.into_iter().flatten())
}

impl QftTextbookStep {
    fn adjoint(self) -> Self {
        match self {
            QftTextbookStep::CPhase {
                control,
                target,
                theta,
            } => QftTextbookStep::CPhase {
                control,
                target,
                theta: -theta,
            },
            step => step,
        }
    }
}

/// Expand `Gate::QftBlock` instructions to textbook QFT gates.
///
/// Backends without native support call this before dispatch. A block under an
/// [`Instruction::Conditional`] becomes an [`Instruction::Region`] holding its
/// gates. Returns `Cow::Borrowed` when there is nothing to expand.
pub fn expand_qft_blocks(circuit: &Circuit) -> std::borrow::Cow<'_, Circuit> {
    if !any_gate(&circuit.instructions, &mut |gate| {
        matches!(gate, Gate::QftBlock { .. })
    }) {
        return std::borrow::Cow::Borrowed(circuit);
    }
    std::borrow::Cow::Owned(circuit.with_instructions(expanded_qft_blocks(&circuit.instructions)))
}

fn expanded_qft_blocks(instructions: &[Instruction]) -> Vec<Instruction> {
    let mut out: Vec<Instruction> = Vec::with_capacity(instructions.len() * 2);
    for inst in instructions {
        if let Instruction::Gate {
            gate:
                Gate::QftBlock {
                    start,
                    num,
                    inverse,
                    swaps,
                    big_endian,
                },
            ..
        } = inst
        {
            for step in qft_textbook_steps(
                *start as usize,
                *num as usize,
                *inverse,
                *swaps,
                *big_endian,
            ) {
                match step {
                    QftTextbookStep::Hadamard(q) => out.push(Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![q],
                    }),
                    QftTextbookStep::CPhase {
                        control,
                        target,
                        theta,
                    } => out.push(Instruction::Gate {
                        gate: Gate::cphase(theta),
                        targets: smallvec![control, target],
                    }),
                    QftTextbookStep::Swap(a, b) => out.push(Instruction::Gate {
                        gate: Gate::Swap,
                        targets: smallvec![a, b],
                    }),
                }
            }
        } else if let Instruction::Conditional {
            condition,
            gate: gate @ Gate::QftBlock { .. },
            targets,
        } = inst
        {
            let bare = Instruction::Gate {
                gate: gate.clone(),
                targets: targets.clone(),
            };
            out.push(Instruction::Region(Box::new(GuardedRegion::new(
                condition.clone(),
                expanded_qft_blocks(&[bare]),
            ))));
        } else if let Instruction::Region(region) = inst {
            out.push(Instruction::Region(Box::new(GuardedRegion::new(
                region.condition().clone(),
                expanded_qft_blocks(region.body()),
            ))));
        } else {
            out.push(inst.clone());
        }
    }
    out
}

// `Gate::QftBlock` holds `start` and `num` as `u8`. The second step of a forward block
// is its finest controlled phase, `tau / 2^num` from qubit 0 onto qubit `num - 1`, so it
// takes the largest shift of any step. Kani's overflow check covers that shift, and the
// angle must equal `tau` scaled by `2^-num` built from its exponent bits, which no shift
// can wrap.
#[cfg(kani)]
mod proofs {
    use super::{QftTextbookStep, qft_textbook_steps};

    #[kani::proof]
    #[kani::unwind(5)]
    fn qft_textbook_finest_phase_has_no_shift_overflow() {
        let (start, num): (u8, u8) = (kani::any(), kani::any());
        let mut steps =
            qft_textbook_steps(start.into(), num.into(), false, kani::any(), kani::any());
        steps.next();
        if let Some(QftTextbookStep::CPhase { theta, .. }) = steps.next() {
            let scale = f64::from_bits((1023 - u64::from(num)) << 52);
            assert_eq!(theta.to_bits(), (std::f64::consts::TAU * scale).to_bits());
        }
    }
}

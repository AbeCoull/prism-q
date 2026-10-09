//! Recognition of expanded QFT gate sequences as [`Gate::QftBlock`].

use std::borrow::Cow;
use std::f64::consts::PI;

use num_complex::Complex64;

use super::{Circuit, Instruction};
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

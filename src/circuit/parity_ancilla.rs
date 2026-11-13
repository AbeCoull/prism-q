//! Rewrite of parity ancillas into parity measurements on the data qubits they read,
//! so the outcome-branching walk runs at the data width.

use smallvec::SmallVec;

use super::{Circuit, ClassicalCondition, Instruction, for_each_measure, remap_instruction};
use crate::gates::Gate;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Zero,
    Open,
    Measured { bit: usize, clobbered: bool },
    Failed,
}

/// One closed span: the parity of `parity` (inverted when `invert`) replaces the
/// measurement at instruction `measure_at`.
struct Span {
    measure_at: usize,
    parity: SmallVec<[usize; 4]>,
    invert: bool,
}

struct Ancilla {
    phase: Phase,
    /// Sources picked up since the span opened, each frozen until the measurement.
    touched: SmallVec<[usize; 4]>,
    parity: SmallVec<[usize; 4]>,
    invert: bool,
    spans: Vec<Span>,
    dropped: Vec<usize>,
}

struct Scan {
    ancillas: Vec<Ancilla>,
    /// Ancillas with an open span that picked up each qubit.
    holders: Vec<SmallVec<[usize; 2]>>,
    /// Ancilla whose last measurement wrote each classical bit.
    measured_by: Vec<Option<usize>>,
}

impl Scan {
    fn fail(&mut self, a: usize) {
        let ancilla = &mut self.ancillas[a];
        if ancilla.phase == Phase::Failed {
            return;
        }
        ancilla.phase = Phase::Failed;
        for d in std::mem::take(&mut ancilla.touched) {
            self.holders[d].retain(|&mut h| h != a);
        }
    }

    /// `q` changed in a way that does not commute with `Z_q`: no span holding it as a
    /// source survives.
    fn disturb(&mut self, q: usize) {
        for a in std::mem::take(&mut self.holders[q]) {
            self.fail(a);
        }
    }

    fn touch(&mut self, q: usize) {
        self.fail(q);
        self.disturb(q);
    }

    fn pick_up(&mut self, a: usize, d: usize, at: usize) {
        self.fail(d);
        self.disturb(a);
        let ancilla = &mut self.ancillas[a];
        match ancilla.phase {
            Phase::Zero | Phase::Open => {
                ancilla.phase = Phase::Open;
                match ancilla.parity.iter().position(|&s| s == d) {
                    Some(i) => {
                        ancilla.parity.swap_remove(i);
                    }
                    None => ancilla.parity.push(d),
                }
                if !ancilla.touched.contains(&d) {
                    ancilla.touched.push(d);
                    self.holders[d].push(a);
                }
                ancilla.dropped.push(at);
            }
            Phase::Measured { .. } => self.fail(a),
            Phase::Failed => {}
        }
    }

    fn flip(&mut self, a: usize, at: usize) {
        self.disturb(a);
        let ancilla = &mut self.ancillas[a];
        match ancilla.phase {
            Phase::Zero | Phase::Open => {
                ancilla.phase = Phase::Open;
                ancilla.invert = !ancilla.invert;
                ancilla.dropped.push(at);
            }
            Phase::Measured { .. } => self.fail(a),
            Phase::Failed => {}
        }
    }

    fn write_bit(&mut self, bit: usize) {
        if let Some(a) = self.measured_by[bit].take()
            && let Phase::Measured { bit: b, clobbered } = &mut self.ancillas[a].phase
            && *b == bit
        {
            *clobbered = true;
        }
    }

    fn measure(&mut self, a: usize, bit: usize, at: usize) {
        self.write_bit(bit);
        self.disturb(a);
        let ancilla = &mut self.ancillas[a];
        match ancilla.phase {
            Phase::Open if !ancilla.parity.is_empty() => {
                ancilla.spans.push(Span {
                    measure_at: at,
                    parity: std::mem::take(&mut ancilla.parity),
                    invert: std::mem::take(&mut ancilla.invert),
                });
                ancilla.phase = Phase::Measured {
                    bit,
                    clobbered: false,
                };
                for d in std::mem::take(&mut ancilla.touched) {
                    self.holders[d].retain(|&mut h| h != a);
                }
                self.measured_by[bit] = Some(a);
            }
            Phase::Failed => {}
            _ => self.fail(a),
        }
    }

    fn reset(&mut self, a: usize, at: usize) {
        self.disturb(a);
        let ancilla = &mut self.ancillas[a];
        match ancilla.phase {
            Phase::Zero | Phase::Measured { .. } => {
                ancilla.phase = Phase::Zero;
                ancilla.dropped.push(at);
            }
            Phase::Open => self.fail(a),
            Phase::Failed => {}
        }
    }

    /// Take `if (bit) x a` right after `a` was measured into `bit` as a reset.
    fn active_reset(
        &mut self,
        condition: &ClassicalCondition,
        gate: &Gate,
        targets: &[usize],
        at: usize,
    ) -> bool {
        let (Gate::X, &[a], &ClassicalCondition::BitIsOne(bit)) = (gate, targets, condition) else {
            return false;
        };
        let ancilla = &mut self.ancillas[a];
        if ancilla.phase
            != (Phase::Measured {
                bit,
                clobbered: false,
            })
        {
            return false;
        }
        ancilla.phase = Phase::Zero;
        ancilla.dropped.push(at);
        true
    }
}

/// Rewrite every parity ancilla of `circuit` into a parity measurement on its data
/// qubits and drop it from the register, or `None` when no qubit qualifies.
///
/// A qubit qualifies when every instruction touching it, over the whole circuit, reads
/// `(CX(d -> a) | X a)* Measure a (Reset a | if (bit) x a)?`, the closing group optional
/// only on the last span, with a reset allowed wherever `a` already holds `|0>`. Each
/// span must pick up an odd count of at least one source, and from its first pickup to
/// the measurement a source takes nothing but CX gates it controls. Any save declines
/// the pass. Each span becomes `cx d h` over the other sources onto `h`, the lowest,
/// then `measure h` (between two `x h` when the span flipped `a` an odd number of
/// times), then the fan undone; the other qubits keep their order and every
/// classical bit keeps its index. The run must start from `|0...0>` and carry no
/// noise, since both read the ancilla.
pub(crate) fn rewrite_parity_ancillas(circuit: &Circuit) -> Option<Circuit> {
    let n = circuit.num_qubits;
    let mut scan = Scan {
        ancillas: (0..n)
            .map(|_| Ancilla {
                phase: Phase::Zero,
                touched: SmallVec::new(),
                parity: SmallVec::new(),
                invert: false,
                spans: Vec::new(),
                dropped: Vec::new(),
            })
            .collect(),
        holders: vec![SmallVec::new(); n],
        measured_by: vec![None; circuit.num_classical_bits],
    };
    for (at, inst) in circuit.instructions.iter().enumerate() {
        match inst {
            Instruction::Save { .. } => return None,
            Instruction::Barrier { .. } => {}
            Instruction::Gate {
                gate: Gate::Cx,
                targets,
            } => scan.pick_up(targets[1], targets[0], at),
            Instruction::Gate {
                gate: Gate::X,
                targets,
            } => scan.flip(targets[0], at),
            Instruction::Gate { targets, .. } => {
                for &q in targets {
                    scan.touch(q);
                }
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => scan.measure(*qubit, *classical_bit, at),
            Instruction::Reset { qubit } => scan.reset(*qubit, at),
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if !scan.active_reset(condition, gate, targets, at) {
                    for &q in targets {
                        scan.touch(q);
                    }
                }
            }
            Instruction::Region(region) => {
                for &q in region.qubits() {
                    scan.touch(q);
                }
                for_each_measure(region.body(), &mut |_, bit| scan.write_bit(bit));
            }
        }
    }

    let kept: Vec<bool> = scan
        .ancillas
        .iter()
        .map(|a| matches!(a.phase, Phase::Zero | Phase::Measured { .. }) && !a.spans.is_empty())
        .collect();
    if !kept.contains(&true) {
        return None;
    }

    let mut index = vec![usize::MAX; n];
    let mut width = 0;
    for q in (0..n).filter(|&q| !kept[q]) {
        index[q] = width;
        width += 1;
    }
    let mut dropped = vec![false; circuit.instructions.len()];
    let mut spans: Vec<Option<&Span>> = vec![None; circuit.instructions.len()];
    for ancilla in scan
        .ancillas
        .iter()
        .zip(&kept)
        .filter_map(|(a, &k)| k.then_some(a))
    {
        for &at in &ancilla.dropped {
            dropped[at] = true;
        }
        for span in &ancilla.spans {
            spans[span.measure_at] = Some(span);
        }
    }

    let mut out = Circuit::new(width, circuit.num_classical_bits);
    out.instructions.reserve(circuit.instructions.len());
    for (at, inst) in circuit.instructions.iter().enumerate() {
        if dropped[at] {
            continue;
        }
        if let (Some(span), Instruction::Measure { classical_bit, .. }) = (spans[at], inst) {
            let mut parity: SmallVec<[usize; 4]> = span.parity.iter().map(|&d| index[d]).collect();
            parity.sort_unstable();
            let (&h, rest) = parity.split_first().expect("a span has a source");
            let fan = |d: usize| Instruction::Gate {
                gate: Gate::Cx,
                targets: SmallVec::from_slice(&[d, h]),
            };
            let flip = || Instruction::Gate {
                gate: Gate::X,
                targets: SmallVec::from_slice(&[h]),
            };
            out.instructions.extend(rest.iter().map(|&d| fan(d)));
            if span.invert {
                out.instructions.push(flip());
            }
            out.instructions.push(Instruction::Measure {
                qubit: h,
                classical_bit: *classical_bit,
            });
            if span.invert {
                out.instructions.push(flip());
            }
            out.instructions.extend(rest.iter().rev().map(|&d| fan(d)));
            continue;
        }
        if let Instruction::Barrier { qubits } = inst {
            let qubits: SmallVec<[usize; 4]> = qubits
                .iter()
                .filter(|&&q| !kept[q])
                .map(|&q| index[q])
                .collect();
            if !qubits.is_empty() {
                out.instructions.push(Instruction::Barrier { qubits });
            }
            continue;
        }
        out.instructions
            .push(remap_instruction(inst, &|q| index[q], &|bit| bit));
    }
    Some(out)
}

#[cfg(test)]
#[path = "parity_ancilla_tests.rs"]
mod tests;

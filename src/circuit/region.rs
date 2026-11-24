//! Classical conditions, guarded regions, and the walks that descend into region bodies.

use super::{Instruction, SmallVec};
use crate::gates::Gate;

/// Condition for classically-controlled gate execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ClassicalCondition {
    /// True when the classical bit at `bit` is 1.
    BitIsOne(usize),
    /// True when the classical bit at `bit` is 0.
    BitIsZero(usize),
    /// True when bits `offset..offset+size`, bit `offset + i` weighted `1 << i`,
    /// equal `value`.
    RegisterEquals {
        offset: usize,
        size: usize,
        value: u64,
    },
    /// Negation of `RegisterEquals`.
    RegisterNotEquals {
        offset: usize,
        size: usize,
        value: u64,
    },
    /// True when the XOR of the listed bits equals `expected`.
    ///
    /// The bits need not be contiguous, as a detector predicate over measurement
    /// records requires. Boxed to keep the enum at the register variants' size.
    /// An empty list has parity zero and no OpenQASM form, so
    /// [`qasm_export`](super::qasm_export) rejects it.
    Parity { bits: Box<[usize]>, expected: bool },
}

impl ClassicalCondition {
    /// The condition that holds exactly when this one does not.
    pub fn negate(&self) -> ClassicalCondition {
        match self {
            ClassicalCondition::BitIsOne(bit) => ClassicalCondition::BitIsZero(*bit),
            ClassicalCondition::BitIsZero(bit) => ClassicalCondition::BitIsOne(*bit),
            ClassicalCondition::RegisterEquals {
                offset,
                size,
                value,
            } => ClassicalCondition::RegisterNotEquals {
                offset: *offset,
                size: *size,
                value: *value,
            },
            ClassicalCondition::RegisterNotEquals {
                offset,
                size,
                value,
            } => ClassicalCondition::RegisterEquals {
                offset: *offset,
                size: *size,
                value: *value,
            },
            ClassicalCondition::Parity { bits, expected } => ClassicalCondition::Parity {
                bits: bits.clone(),
                expected: !expected,
            },
        }
    }

    pub(super) fn for_each_bit(&self, mut f: impl FnMut(usize)) {
        match self {
            ClassicalCondition::BitIsOne(bit) | ClassicalCondition::BitIsZero(bit) => f(*bit),
            ClassicalCondition::RegisterEquals { offset, size, .. }
            | ClassicalCondition::RegisterNotEquals { offset, size, .. } => {
                (*offset..offset + size).for_each(f)
            }
            ClassicalCondition::Parity { bits, .. } => bits.iter().copied().for_each(f),
        }
    }

    pub fn evaluate(&self, classical_bits: &[bool]) -> bool {
        match self {
            ClassicalCondition::BitIsOne(bit) => classical_bits[*bit],
            ClassicalCondition::BitIsZero(bit) => !classical_bits[*bit],
            ClassicalCondition::RegisterEquals {
                offset,
                size,
                value,
            }
            | ClassicalCondition::RegisterNotEquals {
                offset,
                size,
                value,
            } => {
                // Bits from 64 up weigh more than any `value` can hold, so a
                // set one there decides inequality.
                let (low, high) = classical_bits[*offset..offset + size].split_at((*size).min(64));
                let reg_val = low
                    .iter()
                    .enumerate()
                    .fold(0u64, |acc, (i, &bit)| acc | (u64::from(bit) << i));
                let eq = reg_val == *value && !high.contains(&true);
                if matches!(self, ClassicalCondition::RegisterEquals { .. }) {
                    eq
                } else {
                    !eq
                }
            }
            ClassicalCondition::Parity { bits, expected } => {
                bits.iter()
                    .fold(false, |acc, &bit| acc ^ classical_bits[bit])
                    == *expected
            }
        }
    }
}

/// Deepest guarded-region nesting the parser accepts.
///
/// Every pass walks a region body recursively, so the depth is bounded at
/// construction to keep those walks finite.
pub const MAX_REGION_DEPTH: usize = 16;

/// A guarded region: `body` executes in order iff `condition` holds.
///
/// The condition is evaluated once, against the classical bits as they stand
/// when control reaches the region. Bits written by measurements inside a taken
/// body are visible to later instructions.
///
/// Fields are private because [`GuardedRegion::qubits`] caches the union over
/// the body; build a new region rather than editing one in place.
#[derive(Debug, Clone)]
pub struct GuardedRegion {
    condition: ClassicalCondition,
    body: Vec<Instruction>,
    qubits: SmallVec<[usize; 4]>,
    noise_slots: usize,
}

impl GuardedRegion {
    /// Compute the body's qubit union once, so each pass reads it instead of
    /// re-walking the body.
    pub fn new(condition: ClassicalCondition, body: Vec<Instruction>) -> Self {
        let mut qubits: SmallVec<[usize; 4]> = SmallVec::new();
        collect_region_qubits(&body, &mut qubits);
        qubits.sort_unstable();
        qubits.dedup();
        let noise_slots = body.len() + guarded_noise_slots(&body);
        Self {
            condition,
            body,
            qubits,
            noise_slots,
        }
    }

    pub fn condition(&self) -> &ClassicalCondition {
        &self.condition
    }

    pub fn body(&self) -> &[Instruction] {
        &self.body
    }

    /// Sorted union of every qubit the body touches, nested regions included.
    pub fn qubits(&self) -> &[usize] {
        &self.qubits
    }

    /// Noise slots the body takes after the top-level ones: one per body
    /// instruction plus those of the guards inside it. See
    /// [`NoiseModel::after_gate`](crate::sim::noise::NoiseModel::after_gate).
    pub(crate) fn noise_slots(&self) -> usize {
        self.noise_slots
    }

    /// Nesting depth, counting this region as 1.
    pub fn depth(&self) -> usize {
        1 + self
            .body
            .iter()
            .filter_map(|inst| match inst {
                Instruction::Region(inner) => Some(inner.depth()),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }
}

/// Visit every gate, including a guarded one, descending into region bodies.
pub(super) fn for_each_gate(instructions: &[Instruction], visit: &mut impl FnMut(&Gate)) {
    for inst in instructions {
        match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => visit(gate),
            Instruction::Region(region) => for_each_gate(region.body(), visit),
            _ => {}
        }
    }
}

/// Short-circuiting scan over unguarded [`Instruction::Gate`] payloads only,
/// which is the set the lowering passes rewrite.
pub(super) fn any_bare_gate(
    instructions: &[Instruction],
    pred: &mut impl FnMut(&Gate) -> bool,
) -> bool {
    instructions.iter().any(|inst| match inst {
        Instruction::Gate { gate, .. } => pred(gate),
        Instruction::Region(region) => any_bare_gate(region.body(), pred),
        _ => false,
    })
}

/// Short-circuiting [`for_each_gate`].
pub(crate) fn any_gate(instructions: &[Instruction], pred: &mut impl FnMut(&Gate) -> bool) -> bool {
    instructions.iter().any(|inst| match inst {
        Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => pred(gate),
        Instruction::Region(region) => any_gate(region.body(), pred),
        _ => false,
    })
}

/// True when `pred` holds for `inst` or for anything inside a region body.
pub(super) fn any_instruction(
    instructions: &[Instruction],
    pred: &mut impl FnMut(&Instruction) -> bool,
) -> bool {
    instructions.iter().any(|inst| {
        pred(inst)
            || match inst {
                Instruction::Region(region) => any_instruction(region.body(), pred),
                _ => false,
            }
    })
}

pub(super) fn mark_coupled(words: &mut [u64], targets: &[usize]) {
    if targets.len() >= 2 {
        for &q in targets {
            words[q / 64] |= 1 << (q % 64);
        }
    }
}

pub(super) fn strip_measurements(instructions: &[Instruction]) -> Vec<Instruction> {
    instructions
        .iter()
        .filter(|inst| !matches!(inst, Instruction::Measure { .. }))
        .filter_map(|inst| match inst {
            // A region whose body was all measurements drops out entirely,
            // rather than surviving as the empty region `guarded` never builds.
            Instruction::Region(region) => guarded(
                region.condition().clone(),
                strip_measurements(region.body()),
            ),
            other => Some(other.clone()),
        })
        .collect()
}

/// Visit every measurement in `instructions`, descending into region bodies.
pub(super) fn for_each_measure(instructions: &[Instruction], visit: &mut impl FnMut(usize, usize)) {
    for inst in instructions {
        match inst {
            Instruction::Measure {
                qubit,
                classical_bit,
            } => visit(*qubit, *classical_bit),
            Instruction::Region(region) => for_each_measure(region.body(), visit),
            _ => {}
        }
    }
}

/// Which classical bits a preceding measurement may have written, plus the
/// all-zero vector a statically decidable condition evaluates against.
pub(super) struct FoldState {
    pub(super) written: Vec<bool>,
    pub(super) zeros: Vec<bool>,
    pub(super) folded: bool,
}

impl FoldState {
    /// `Some(value)` when no preceding measurement can have written a bit the
    /// condition reads, so its value is fixed before the run starts.
    fn static_value(&self, condition: &ClassicalCondition) -> Option<bool> {
        let mut read = Vec::new();
        collect_condition_bits(condition, &mut read);
        if read
            .iter()
            .any(|&bit| bit >= self.written.len() || self.written[bit])
        {
            return None;
        }
        Some(condition.evaluate(&self.zeros))
    }

    fn mark_written(&mut self, body: &[Instruction]) {
        let written = &mut self.written;
        for_each_measure(body, &mut |_, classical_bit| {
            if let Some(slot) = written.get_mut(classical_bit) {
                *slot = true;
            }
        });
    }
}

pub(super) fn fold_static_guards_into(
    instructions: &[Instruction],
    state: &mut FoldState,
    out: &mut Vec<Instruction>,
) {
    for inst in instructions {
        match inst {
            Instruction::Measure { classical_bit, .. } => {
                if let Some(slot) = state.written.get_mut(*classical_bit) {
                    *slot = true;
                }
                out.push(inst.clone());
            }
            Instruction::Save { .. } => out.push(inst.clone()),
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => match state.static_value(condition) {
                Some(true) => {
                    state.folded = true;
                    out.push(Instruction::Gate {
                        gate: gate.clone(),
                        targets: targets.clone(),
                    });
                }
                Some(false) => state.folded = true,
                None => out.push(inst.clone()),
            },
            Instruction::Region(region) => match state.static_value(region.condition()) {
                Some(true) => {
                    state.folded = true;
                    fold_static_guards_into(region.body(), state, out);
                }
                Some(false) => state.folded = true,
                None => {
                    state.mark_written(region.body());
                    out.push(inst.clone());
                }
            },
            Instruction::Gate { .. } | Instruction::Reset { .. } | Instruction::Barrier { .. } => {
                out.push(inst.clone())
            }
        }
    }
}

/// True when a measurement anywhere in `body` writes a bit `condition` reads.
///
/// Lowering `else` and `switch` to a chain of guarded regions is only sound
/// while this is false. Each region in the chain re-evaluates its condition
/// after the earlier bodies have run, so a body that rewrites its own guard bits
/// could take two arms of what the source wrote as one choice.
pub(crate) fn body_writes_condition_bits(
    body: &[Instruction],
    condition: &ClassicalCondition,
) -> bool {
    let mut read = Vec::new();
    collect_condition_bits(condition, &mut read);
    let mut written = false;
    for_each_measure(body, &mut |_, classical_bit| {
        written |= read.contains(&classical_bit)
    });
    written
}

pub(super) fn collect_condition_bits(condition: &ClassicalCondition, out: &mut Vec<usize>) {
    match condition {
        ClassicalCondition::BitIsOne(bit) | ClassicalCondition::BitIsZero(bit) => out.push(*bit),
        ClassicalCondition::RegisterEquals { offset, size, .. }
        | ClassicalCondition::RegisterNotEquals { offset, size, .. } => {
            out.extend(*offset..offset.saturating_add(*size))
        }
        ClassicalCondition::Parity { bits, .. } => out.extend_from_slice(bits),
    }
}

/// Every classical bit read by a condition anywhere in `body`, nested regions
/// included. A subsystem partition that ignored these would split a circuit
/// whose halves are coupled through the classical bits alone.
pub(super) fn collect_body_condition_bits(body: &[Instruction], out: &mut Vec<usize>) {
    for inst in body {
        match inst {
            Instruction::Conditional { condition, .. } => collect_condition_bits(condition, out),
            Instruction::Region(region) => {
                collect_condition_bits(region.condition(), out);
                collect_body_condition_bits(region.body(), out);
            }
            _ => {}
        }
    }
}

fn remap_condition(
    condition: &ClassicalCondition,
    cbit: &impl Fn(usize) -> usize,
) -> ClassicalCondition {
    // An empty register reads no bit, so its offset has no image to look up.
    let run_start = |offset: usize, size: usize| if size == 0 { 0 } else { cbit(offset) };
    match condition {
        ClassicalCondition::BitIsOne(bit) => ClassicalCondition::BitIsOne(cbit(*bit)),
        ClassicalCondition::BitIsZero(bit) => ClassicalCondition::BitIsZero(cbit(*bit)),
        ClassicalCondition::RegisterEquals {
            offset,
            size,
            value,
        } => ClassicalCondition::RegisterEquals {
            offset: run_start(*offset, *size),
            size: *size,
            value: *value,
        },
        ClassicalCondition::RegisterNotEquals {
            offset,
            size,
            value,
        } => ClassicalCondition::RegisterNotEquals {
            offset: run_start(*offset, *size),
            size: *size,
            value: *value,
        },
        ClassicalCondition::Parity { bits, expected } => ClassicalCondition::Parity {
            bits: bits.iter().map(|&bit| cbit(bit)).collect(),
            expected: *expected,
        },
    }
}

/// `gate` with the qubit range a [`Gate::QftBlock`] carries in its payload moved
/// under `qubit`, which must keep the block's qubits consecutive and ascending.
pub(super) fn remap_gate(gate: &Gate, qubit: &impl Fn(usize) -> usize) -> Gate {
    match gate {
        Gate::QftBlock {
            start,
            num,
            inverse,
            swaps,
            big_endian,
        } => Gate::QftBlock {
            start: qubit(*start as usize) as u8,
            num: *num,
            inverse: *inverse,
            swaps: *swaps,
            big_endian: *big_endian,
        },
        other => other.clone(),
    }
}

/// Rebuild `inst` under a qubit and classical-bit relabeling.
///
/// Callers must have established that every qubit `inst` touches has an image
/// under `qubit`, which for a region means its whole body.
pub(crate) fn remap_instruction(
    inst: &Instruction,
    qubit: &impl Fn(usize) -> usize,
    cbit: &impl Fn(usize) -> usize,
) -> Instruction {
    match inst {
        Instruction::Gate { gate, targets } => Instruction::Gate {
            gate: remap_gate(gate, qubit),
            targets: targets.iter().map(|&t| qubit(t)).collect(),
        },
        Instruction::Measure {
            qubit: q,
            classical_bit,
        } => Instruction::Measure {
            qubit: qubit(*q),
            classical_bit: cbit(*classical_bit),
        },
        Instruction::Reset { qubit: q } => Instruction::Reset { qubit: qubit(*q) },
        Instruction::Save {
            spec,
            qubits,
            label,
        } => Instruction::Save {
            spec: *spec,
            qubits: qubits.iter().map(|&q| qubit(q)).collect(),
            label: label.clone(),
        },
        Instruction::Barrier { qubits } => Instruction::Barrier {
            qubits: qubits.iter().map(|&q| qubit(q)).collect(),
        },
        Instruction::Conditional {
            condition,
            gate,
            targets,
        } => Instruction::Conditional {
            condition: remap_condition(condition, cbit),
            gate: remap_gate(gate, qubit),
            targets: targets.iter().map(|&t| qubit(t)).collect(),
        },
        Instruction::Region(region) => Instruction::Region(Box::new(GuardedRegion::new(
            remap_condition(region.condition(), cbit),
            region
                .body()
                .iter()
                .map(|inner| remap_instruction(inner, qubit, cbit))
                .collect(),
        ))),
    }
}

/// Noise slots the guards in `instructions` take beyond the list's own: a
/// region's subtree, and one taken-only slot per conditional gate.
pub(super) fn guarded_noise_slots(instructions: &[Instruction]) -> usize {
    instructions
        .iter()
        .map(|inst| match inst {
            Instruction::Region(region) => region.noise_slots(),
            Instruction::Conditional { .. } => 1,
            _ => 0,
        })
        .sum()
}

/// Visit each instruction of `list` with its noise slot, `list` itself at
/// `base` onward and the bodies of its guards from `nested` onward. A
/// conditional's taken-only slot is visited with its gate as a bare
/// [`Instruction::Gate`].
pub(super) fn visit_noise_slots(
    list: &[Instruction],
    base: usize,
    mut nested: usize,
    visit: &mut impl FnMut(usize, &Instruction),
) {
    for (offset, inst) in list.iter().enumerate() {
        visit(base + offset, inst);
        match inst {
            Instruction::Region(region) => {
                visit_noise_slots(region.body(), nested, nested + region.body().len(), visit);
                nested += region.noise_slots();
            }
            Instruction::Conditional { gate, targets, .. } => {
                let bare = Instruction::Gate {
                    gate: gate.clone(),
                    targets: targets.clone(),
                };
                visit(nested, &bare);
                nested += 1;
            }
            _ => {}
        }
    }
}

fn collect_region_qubits(body: &[Instruction], out: &mut SmallVec<[usize; 4]>) {
    for inst in body {
        out.extend_from_slice(inst.qubits());
    }
}

/// Build the guarded form of `body`, or `None` when the body is empty.
///
/// A single-gate body lowers to [`Instruction::Conditional`] so the common
/// `if (c) x q[0];` keeps its allocation-free representation; anything else
/// becomes an [`Instruction::Region`].
pub fn guarded(condition: ClassicalCondition, mut body: Vec<Instruction>) -> Option<Instruction> {
    if body.is_empty() {
        return None;
    }
    if body.len() == 1 && matches!(body[0], Instruction::Gate { .. }) {
        let Some(Instruction::Gate { gate, targets }) = body.pop() else {
            unreachable!("length and variant both checked above")
        };
        return Some(Instruction::Conditional {
            condition,
            gate,
            targets,
        });
    }
    Some(Instruction::Region(Box::new(GuardedRegion::new(
        condition, body,
    ))))
}

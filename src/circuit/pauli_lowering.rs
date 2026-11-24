//! Pauli rotation lowering to CNOT ladders, and the basis rotations behind Pauli-product
//! measurement.

use super::region::any_gate;
use super::{Circuit, GuardedRegion, Instruction, SmallVec, smallvec};
use crate::gates::{Gate, PauliRotData};
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

/// Emit the CNOT-ladder lowering of `exp(-i θ P / 2)` gate by gate.
///
/// Basis layer first (`H` for an X letter, `Sdg` then `H` for Y), a CX chain
/// accumulating the parity on the last target, one `Rz(θ)`, then the chain and
/// basis layer unwound. Shared by [`expand_pauli_rotations`] and the GPU
/// inline expansion so both routes lower identically.
pub(crate) fn pauli_rotation_lowering(
    theta: f64,
    targets: &[usize],
    axes: &[PauliAxis],
    mut emit: impl FnMut(Gate, &[usize]),
) {
    for (&q, axis) in targets.iter().zip(axes) {
        match axis {
            PauliAxis::X => emit(Gate::H, &[q]),
            PauliAxis::Y => {
                emit(Gate::Sdg, &[q]);
                emit(Gate::H, &[q]);
            }
            PauliAxis::Z => {}
        }
    }
    for pair in targets.windows(2) {
        emit(Gate::Cx, &[pair[0], pair[1]]);
    }
    emit(Gate::Rz(theta), &[targets[targets.len() - 1]]);
    for pair in targets.windows(2).rev() {
        emit(Gate::Cx, &[pair[0], pair[1]]);
    }
    for (&q, axis) in targets.iter().zip(axes) {
        match axis {
            PauliAxis::X => emit(Gate::H, &[q]),
            PauliAxis::Y => {
                emit(Gate::H, &[q]);
                emit(Gate::S, &[q]);
            }
            PauliAxis::Z => {}
        }
    }
}

/// Lower a Pauli rotation to the gate and target order a circuit stores it as,
/// the recognizing step behind [`Circuit::add_pauli_rotation`].
///
/// Shared with the OpenQASM parser, which builds the instruction itself. Targets
/// come back sorted by qubit with the letters aligned to them.
///
/// # Panics
/// Panics if `factors` is empty or names a qubit twice.
pub(crate) fn pauli_rotation_gate(
    theta: f64,
    factors: &[PauliTerm],
) -> (Gate, SmallVec<[usize; 4]>) {
    assert!(
        !factors.is_empty(),
        "Pauli rotation needs at least one factor"
    );
    let mut sorted: SmallVec<[PauliTerm; 4]> = SmallVec::from_slice(factors);
    sorted.sort_unstable_by_key(|term| term.qubit);
    for pair in sorted.windows(2) {
        assert_ne!(
            pair[0].qubit, pair[1].qubit,
            "Pauli rotation has duplicate factor on qubit {}",
            pair[0].qubit
        );
    }
    match sorted.as_slice() {
        [term] => {
            let gate = match term.axis {
                PauliAxis::X => Gate::Rx(theta),
                PauliAxis::Y => Gate::Ry(theta),
                PauliAxis::Z => Gate::Rz(theta),
            };
            (gate, smallvec![term.qubit])
        }
        [a, b] if a.axis == PauliAxis::Z && b.axis == PauliAxis::Z => {
            (Gate::Rzz(theta), smallvec![a.qubit, b.qubit])
        }
        _ => {
            let targets: SmallVec<[usize; 4]> = sorted.iter().map(|term| term.qubit).collect();
            let axes: Vec<PauliAxis> = sorted.iter().map(|term| term.axis).collect();
            (
                Gate::PauliRot(Box::new(PauliRotData { theta, axes })),
                targets,
            )
        }
    }
}

/// Destination of a gate lowering: a [`Circuit`], or a flat gate list that skips the
/// [`Instruction`] encoding.
pub(crate) trait GateSink {
    fn gate(&mut self, gate: Gate, targets: &[usize]);
}

impl GateSink for Circuit {
    fn gate(&mut self, gate: Gate, targets: &[usize]) {
        self.add_gate(gate, targets);
    }
}

/// Rotate `qubit` so that a Z-basis measurement reads the `axis` eigenvalue:
/// X appends H, Y appends Sdg then H, Z appends nothing.
pub(crate) fn append_axis_to_z_rotation<S: GateSink>(sink: &mut S, axis: PauliAxis, qubit: usize) {
    match axis {
        PauliAxis::X => sink.gate(Gate::H, &[qubit]),
        PauliAxis::Y => {
            sink.gate(Gate::Sdg, &[qubit]);
            sink.gate(Gate::H, &[qubit]);
        }
        PauliAxis::Z => {}
    }
}

/// Inverse of [`append_axis_to_z_rotation`].
pub(crate) fn append_z_to_axis_rotation<S: GateSink>(sink: &mut S, axis: PauliAxis, qubit: usize) {
    match axis {
        PauliAxis::X => sink.gate(Gate::H, &[qubit]),
        PauliAxis::Y => {
            sink.gate(Gate::H, &[qubit]);
            sink.gate(Gate::S, &[qubit]);
        }
        PauliAxis::Z => {}
    }
}

/// Lower a Pauli-product measurement onto a scratch qubit holding |0>: rotate
/// each term into the Z basis, accumulate parity on the scratch via CX, then
/// undo the rotations in reverse order. The caller measures the scratch.
pub(crate) fn append_parity_rotations<S: GateSink>(
    sink: &mut S,
    terms: &[PauliTerm],
    scratch: usize,
) {
    for term in terms {
        append_axis_to_z_rotation(sink, term.axis, term.qubit);
    }
    for term in terms {
        sink.gate(Gate::Cx, &[term.qubit, scratch]);
    }
    for term in terms.iter().rev() {
        append_z_to_axis_rotation(sink, term.axis, term.qubit);
    }
}

/// Expand `Gate::PauliRot` instructions to the CNOT-ladder lowering, for backends
/// without the native kernel. A rotation under an [`Instruction::Conditional`]
/// becomes an [`Instruction::Region`] holding its ladder. Returns `Cow::Borrowed`
/// when there is nothing to expand.
pub fn expand_pauli_rotations(circuit: &Circuit) -> std::borrow::Cow<'_, Circuit> {
    if !any_gate(&circuit.instructions, &mut |gate| {
        matches!(gate, Gate::PauliRot(_))
    }) {
        return std::borrow::Cow::Borrowed(circuit);
    }
    std::borrow::Cow::Owned(
        circuit.with_instructions(expanded_pauli_rotations(&circuit.instructions)),
    )
}

fn expanded_pauli_rotations(instructions: &[Instruction]) -> Vec<Instruction> {
    let mut out: Vec<Instruction> = Vec::with_capacity(instructions.len() * 2);
    for inst in instructions {
        if let Instruction::Gate {
            gate: Gate::PauliRot(data),
            targets,
        } = inst
        {
            pauli_rotation_lowering(data.theta, targets, &data.axes, |gate, tgts| {
                out.push(Instruction::Gate {
                    gate,
                    targets: SmallVec::from_slice(tgts),
                });
            });
        } else if let Instruction::Conditional {
            condition,
            gate: gate @ Gate::PauliRot(_),
            targets,
        } = inst
        {
            let bare = Instruction::Gate {
                gate: gate.clone(),
                targets: targets.clone(),
            };
            out.push(Instruction::Region(Box::new(GuardedRegion::new(
                condition.clone(),
                expanded_pauli_rotations(&[bare]),
            ))));
        } else if let Instruction::Region(region) = inst {
            out.push(Instruction::Region(Box::new(GuardedRegion::new(
                region.condition().clone(),
                expanded_pauli_rotations(region.body()),
            ))));
        } else {
            out.push(inst.clone());
        }
    }
    out
}

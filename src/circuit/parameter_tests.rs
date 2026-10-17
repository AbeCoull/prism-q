use super::*;

fn two_rotations() -> Circuit {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Rx(0.1), &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rz(0.2), &[1]);
    c
}

#[test]
fn all_rotations_declares_one_slot_per_gate() {
    let p = Parameters::all_rotations(&two_rotations());
    assert_eq!(p.num_slots(), 2);
    assert_eq!(p.links().len(), 2);
    assert_eq!(p.links()[1].instruction, 2);
}

#[test]
fn bind_writes_angles_and_leaves_structure() {
    let template = two_rotations();
    let p = Parameters::all_rotations(&template);
    let bound = p.bind(&template, &[1.5, 2.5]).unwrap();
    assert_eq!(bound.instructions.len(), 3);
    assert!(matches!(
        bound.instructions[0],
        Instruction::Gate {
            gate: Gate::Rx(t),
            ..
        } if t == 1.5
    ));
    assert!(matches!(
        bound.instructions[2],
        Instruction::Gate {
            gate: Gate::Rz(t),
            ..
        } if t == 2.5
    ));
}

#[test]
fn shared_slot_writes_every_linked_gate() {
    let template = two_rotations();
    let mut p = Parameters::new(1);
    p.link(0, 0);
    p.link(2, 0);
    let bound = p.bind(&template, &[0.75]).unwrap();
    assert_eq!(super::angle_of(&bound.instructions[0]), 0.75);
    assert_eq!(super::angle_of(&bound.instructions[2]), 0.75);
}

#[test]
fn wrong_arity_is_an_error() {
    let template = two_rotations();
    let p = Parameters::all_rotations(&template);
    assert!(p.bind(&template, &[1.0]).is_err());
    assert!(p.bind(&template, &[1.0, 2.0, 3.0]).is_err());
}

#[test]
fn non_finite_value_is_an_error() {
    let template = two_rotations();
    let p = Parameters::all_rotations(&template);
    assert!(p.bind(&template, &[f64::NAN, 0.0]).is_err());
    assert!(p.bind(&template, &[0.0, f64::INFINITY]).is_err());
}

#[test]
fn link_past_end_is_an_error() {
    let template = two_rotations();
    let mut p = Parameters::new(1);
    p.link(99, 0);
    assert!(p.bind(&template, &[0.5]).is_err());
}

#[test]
fn link_to_non_bindable_gate_is_an_error() {
    let template = two_rotations();
    let mut p = Parameters::new(1);
    p.link(1, 0);
    assert!(p.bind(&template, &[0.5]).is_err());
}

#[test]
fn slot_no_gate_reads_is_accepted_and_reported() {
    let template = two_rotations();
    let mut p = Parameters::new(2);
    p.link(0, 0);
    assert!(p.bind(&template, &[0.5, 0.5]).is_ok());
    assert_eq!(p.unread_slots(), vec![1]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn slot_past_declared_count_panics() {
    let mut p = Parameters::new(1);
    p.link(0, 4);
}

#[test]
fn values_round_trip_through_bind() {
    let template = two_rotations();
    let p = Parameters::all_rotations(&template);
    let bound = p.bind(&template, &[0.3, 0.4]).unwrap();
    assert_eq!(p.values(&bound).unwrap(), vec![0.3, 0.4]);
}

// `pauli_generator` decides which gates bind, and `angle_of`, `angle_mut`,
// and the replay recipe's `write_angle` each match the same variants by
// hand. A new angle-carrying variant fails here until all four agree.
// `BatchRzz` is the one deliberate exception: replay writes its edges
// although the fused gate itself is not bindable.
#[test]
fn angle_sites_agree_on_every_gate() {
    use crate::circuit::plan::write_angle;
    use crate::gates::PauliRotData;
    use crate::sim::unified_pauli::PauliAxis;

    let h = Gate::H.matrix_2x2();
    let gates = [
        Gate::Rx(0.1),
        Gate::Ry(0.2),
        Gate::Rz(0.3),
        Gate::P(0.4),
        Gate::Rzz(0.5),
        Gate::PauliRot(Box::new(PauliRotData {
            theta: 0.6,
            axes: vec![PauliAxis::X, PauliAxis::Z],
        })),
        Gate::Id,
        Gate::H,
        Gate::T,
        Gate::SX,
        Gate::Cx,
        Gate::Cz,
        Gate::Swap,
        Gate::Cu(Box::new(h)),
        Gate::Fused(Box::new(h)),
        Gate::Fused2q(Box::new(Gate::Cx.matrix_4x4())),
        Gate::QftBlock {
            start: 0,
            num: 2,
            inverse: false,
            swaps: true,
            big_endian: false,
        },
    ];
    for gate in gates {
        let targets: SmallVec<[usize; 4]> = (0..gate.num_qubits()).collect();
        let mut inst = Instruction::Gate {
            gate: gate.clone(),
            targets,
        };
        let bindable = gate.pauli_generator().is_some();
        assert_eq!(write_angle(&mut inst, 0, 1.25), bindable, "{gate}");
        if bindable {
            assert_eq!(angle_of(&inst), 1.25, "{gate}");
            *angle_mut(&mut inst) = 2.5;
            assert_eq!(angle_of(&inst), 2.5, "{gate}");
        }
    }
}

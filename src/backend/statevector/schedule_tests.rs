use super::*;
use crate::backend::{
    Backend, density_matrix::DensityMatrixBackend, statevector::StatevectorBackend,
};
use crate::circuit::{Circuit, smallvec};
use crate::gates::{BatchPhaseData, DiagonalBatchData, PauliRotData};
use crate::sim::unified_pauli::PauliAxis;

fn fixture(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 1);
    for q in 0..n {
        c.add_gate(Gate::Ry(0.17 + q as f64 * 0.13), &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q + 1, q]);
        c.add_gate(Gate::Rzz(0.31), &[q, n - 1]);
        c.add_gate(Gate::Rz(-0.43), &[n - 1]);
        c.add_gate(Gate::Rx(0.12), &[q]);
    }
    c.add_gate(
        Gate::PauliRot(Box::new(PauliRotData {
            theta: 0.27,
            axes: vec![PauliAxis::X, PauliAxis::Y, PauliAxis::Z],
        })),
        &[0, 2, n - 1],
    );
    c.add_gate(
        Gate::BatchPhase(Box::new(BatchPhaseData::new([(
            0,
            Complex64::from_polar(1.0, 0.37),
        )]))),
        &[n - 1, 0],
    );
    c.add_gate(
        Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(vec![DiagEntry::Phase1q {
            qubit: n - 1,
            d0: Complex64::from_polar(1.0, 0.11),
            d1: Complex64::from_polar(1.0, -0.19),
        }]))),
        &[n - 1],
    );
    c
}

fn compare(c: &Circuit) -> Vec<Complex64> {
    let mut a = StatevectorBackend::new(42);
    let mut b = StatevectorBackend::new(42);
    a.init(c.num_qubits, 1).unwrap();
    b.init(c.num_qubits, 1).unwrap();
    for inst in &c.instructions {
        a.apply(inst).unwrap();
    }
    b.apply_schedule(&c.instructions).unwrap();
    for (x, y) in a.state.iter().zip(&b.state) {
        assert!((*x - *y).norm() < 1e-11, "{x} != {y}");
    }
    b.state
}

#[test]
fn mixed_schedule_matches_native_gates() {
    for n in [6, 16, 18, 20] {
        compare(&fixture(n));
    }
    assert_eq!(size_of::<Gate>(), 16);
}

#[test]
fn mixed_schedule_rebuilds_on_structure_and_refreshes_payloads() {
    let mut c = fixture(8);
    let mut backend = StatevectorBackend::new(42);
    for gate in [Gate::Ry(0.2), Gate::Rz(0.3), Gate::Ry(-0.4)] {
        c.instructions[0] = Instruction::Gate {
            gate,
            targets: smallvec![7],
        };
        backend.init(8, 1).unwrap();
        backend.apply_schedule(&c.instructions).unwrap();
        let expected = compare(&c);
        assert_eq!(backend.state, expected);
    }
}

#[test]
fn mixed_schedule_refreshes_target_order_and_pauli_letters() {
    let mut c = fixture(8);
    let mut backend = StatevectorBackend::new(42);
    let mut replacements = vec![
        Instruction::Gate {
            gate: Gate::Cx,
            targets: smallvec![7, 0],
        },
        Instruction::Gate {
            gate: Gate::Cx,
            targets: smallvec![0, 7],
        },
    ];
    for axis in [PauliAxis::X, PauliAxis::Y, PauliAxis::Z] {
        replacements.push(Instruction::Gate {
            gate: Gate::PauliRot(Box::new(PauliRotData {
                theta: 0.47,
                axes: vec![axis, PauliAxis::Y, PauliAxis::Z],
            })),
            targets: smallvec![0, 2, 7],
        });
    }
    for inst in replacements {
        c.instructions[0] = inst;
        backend.init(8, 1).unwrap();
        backend.apply_schedule(&c.instructions).unwrap();
        assert_eq!(backend.state, compare(&c));
    }
}

#[test]
fn doubled_schedule_matches_density_matrix_native_gates() {
    let c = fixture(7);
    let mut a = DensityMatrixBackend::new(42);
    a.init(7, 1).unwrap();
    for inst in &c.instructions {
        a.apply(inst).unwrap();
    }
    let mut b = StatevectorBackend::new(42);
    b.init(14, 0).unwrap();
    b.apply_doubled_schedule(&c.instructions, 7).unwrap();
    for (x, y) in a.density_matrix().unwrap().iter().zip(&b.state) {
        assert!((*x - *y).norm() < 1e-11, "{x} != {y}");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn mixed_schedule_is_equal_across_thread_counts() {
    let c = fixture(18);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| compare(&c));
    let four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| compare(&c));
    assert_eq!(one, four);
}

#[test]
fn mixed_schedule_keeps_opaque_boundaries() {
    let mut c = fixture(8);
    c.instructions.insert(
        9,
        Instruction::Barrier {
            qubits: smallvec![0, 7],
        },
    );
    c.instructions.insert(
        20,
        Instruction::Measure {
            qubit: 7,
            classical_bit: 0,
        },
    );
    c.instructions.insert(24, Instruction::Reset { qubit: 2 });
    c.instructions.insert(
        21,
        Instruction::Conditional {
            condition: crate::circuit::ClassicalCondition::BitIsOne(0),
            gate: Gate::Ry(0.53),
            targets: smallvec![0],
        },
    );
    compare(&c);
}

#[test]
fn mixed_schedule_folds_repeated_homogeneous_targets_in_order() {
    let mut circuit = Circuit::new(20, 1);
    for (gate, q) in [
        (Gate::Rx(0.31), 19),
        (Gate::Ry(0.27), 0),
        (Gate::Ry(0.59), 19),
        (Gate::Rx(-0.43), 0),
    ] {
        circuit.add_gate(gate, &[q]);
    }
    compare(&circuit);
}

#[test]
fn single_batch_kind_ignores_opaque_boundaries() {
    let two = Action::Two(0, 1, Gate::Cz.matrix_4x4());
    let one = Action::One(2, Gate::H.matrix_2x2());
    let diagonal = Action::Diagonal(0, 1, [Complex64::new(1.0, 0.0); 4]);
    assert!(single_batch_kind(&[]));
    assert!(single_batch_kind(&[
        two.clone(),
        Action::Opaque(1),
        two.clone()
    ]));
    assert!(single_batch_kind(&[one.clone(), one.clone()]));
    assert!(!single_batch_kind(&[two.clone(), one]));
    assert!(!single_batch_kind(&[two, diagonal.clone()]));
    assert!(!single_batch_kind(&[diagonal.clone(), diagonal]));
    assert!(!single_batch_kind(&[Action::Cx(0, 1)]));
}

#[test]
fn mixed_schedule_long_stream_keeps_every_operation() {
    let actions: Vec<_> = (0..4096)
        .map(|i| Action::Cx((i + 1) % 24, i % 24))
        .collect();
    let mut schedule = Schedule::default();
    schedule.update(&actions, 24);
    let mut emitted: Vec<_> = schedule.stages.iter().flatten().copied().collect();
    emitted.sort_unstable();
    assert_eq!(emitted, (0..actions.len()).collect::<Vec<_>>());
}

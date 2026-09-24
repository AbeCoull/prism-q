//! Parameter binding: agreement with independently constructed circuits, plan
//! reuse fidelity, and the rejection paths.

use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::fusion::fuse_circuit;
use prism_q::{
    Circuit, CircuitBuilder, Gate, Instruction, ObservableExpectation, ParamLink, Parameters,
    PauliObservable, PauliTerm, PreparedCircuit, circuits,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 42;

fn angles(count: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| rng.random::<f64>() * std::f64::consts::TAU)
        .collect()
}

// Apply `circuit` verbatim, without a further fusion pass, so an already fused
// stream is measured as the backend would execute it.
fn statevector(circuit: &Circuit) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(SEED);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .expect("init failed");
    backend
        .apply_instructions(&circuit.instructions)
        .expect("apply failed");
    backend.state_vector().to_vec()
}

fn assert_states_match(a: &[Complex64], b: &[Complex64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: dimension");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).norm() < 1e-12,
            "{what}: amplitude {i} differs, {x} vs {y}"
        );
    }
}

// The instruction streams must agree gate for gate, not merely up to the state
// they produce: a replayed plan with a different block structure would still
// simulate correctly while losing the performance the plan exists for.
fn assert_streams_match(a: &Circuit, b: &Circuit, what: &str) {
    assert_eq!(
        a.instructions.len(),
        b.instructions.len(),
        "{what}: instruction count, {:?} vs {:?}",
        a.instructions.len(),
        b.instructions.len()
    );
    for (i, (x, y)) in a.instructions.iter().zip(&b.instructions).enumerate() {
        match (x, y) {
            (
                Instruction::Gate {
                    gate: gx,
                    targets: tx,
                },
                Instruction::Gate {
                    gate: gy,
                    targets: ty,
                },
            ) => {
                assert_eq!(
                    std::mem::discriminant(gx),
                    std::mem::discriminant(gy),
                    "{what}: instruction {i} gate kind, {gx:?} vs {gy:?}"
                );
                assert_eq!(tx, ty, "{what}: instruction {i} targets");
                assert_payloads_match(gx, gy, &format!("{what}: instruction {i}"));
            }
            (px, py) => assert_eq!(
                format!("{px:?}"),
                format!("{py:?}"),
                "{what}: instruction {i}"
            ),
        }
    }
}

fn assert_close(a: Complex64, b: Complex64, what: &str) {
    assert!((a - b).norm() < 1e-12, "{what}: {a} vs {b}");
}

fn assert_mat2(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2], what: &str) {
    for r in 0..2 {
        for c in 0..2 {
            assert_close(a[r][c], b[r][c], &format!("{what} [{r}][{c}]"));
        }
    }
}

fn assert_mat4(a: &[[Complex64; 4]; 4], b: &[[Complex64; 4]; 4], what: &str) {
    for r in 0..4 {
        for c in 0..4 {
            assert_close(a[r][c], b[r][c], &format!("{what} [{r}][{c}]"));
        }
    }
}

// Gate kind and targets alone would let a replayed plan pass while carrying the
// wrong matrices, which is exactly what a mis-recorded recipe produces.
fn assert_payloads_match(a: &Gate, b: &Gate, what: &str) {
    match (a, b) {
        (Gate::Fused(x), Gate::Fused(y)) => assert_mat2(x, y, what),
        (Gate::Fused2q(x), Gate::Fused2q(y)) => assert_mat4(x, y, what),
        (Gate::MultiFused(x), Gate::MultiFused(y)) => {
            assert_eq!(
                x.gates().len(),
                y.gates().len(),
                "{what}: multi_fused arity"
            );
            assert_eq!(x.all_diagonal(), y.all_diagonal(), "{what}: all_diagonal");
            for (k, (gx, gy)) in x.gates().iter().zip(y.gates()).enumerate() {
                assert_eq!(gx.0, gy.0, "{what}: multi_fused entry {k} qubit");
                assert_mat2(&gx.1, &gy.1, &format!("{what} entry {k}"));
            }
        }
        (Gate::Multi2q(x), Gate::Multi2q(y)) => {
            assert_eq!(x.gates.len(), y.gates.len(), "{what}: multi_2q arity");
            for (k, (gx, gy)) in x.gates.iter().zip(&y.gates).enumerate() {
                assert_eq!(
                    (gx.0, gx.1),
                    (gy.0, gy.1),
                    "{what}: multi_2q entry {k} pair"
                );
                assert_mat4(&gx.2, &gy.2, &format!("{what} entry {k}"));
            }
        }
        (Gate::BatchRzz(x), Gate::BatchRzz(y)) => {
            assert_eq!(x.edges.len(), y.edges.len(), "{what}: batch_rzz arity");
            for (k, (ex, ey)) in x.edges.iter().zip(&y.edges).enumerate() {
                assert_eq!(
                    (ex.0, ex.1),
                    (ey.0, ey.1),
                    "{what}: batch_rzz edge {k} pair"
                );
                assert!(
                    (ex.2 - ey.2).abs() < 1e-12,
                    "{what}: batch_rzz edge {k} angle, {} vs {}",
                    ex.2,
                    ey.2
                );
            }
        }
        (
            Gate::Rx(x) | Gate::Ry(x) | Gate::Rz(x) | Gate::Rzz(x) | Gate::P(x),
            Gate::Rx(y) | Gate::Ry(y) | Gate::Rz(y) | Gate::Rzz(y) | Gate::P(y),
        ) => assert!((x - y).abs() < 1e-12, "{what}: angle, {x} vs {y}"),
        _ => {}
    }
}

// A Trotter layer over the native Pauli rotation: the strings the constructor
// does not recognize stay as `PauliRot`, whose angle a binding has to reach
// through the plan like any other rotation.
fn trotter_layer(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Ry(0.17 + 0.03 * q as f64), &[q]);
    }
    for q in 0..n - 2 {
        c.add_pauli_rotation(
            0.21 + 0.01 * q as f64,
            &[PauliTerm::x(q), PauliTerm::y(q + 1), PauliTerm::z(q + 2)],
        );
        c.add_pauli_rotation(0.13, &[PauliTerm::y(q), PauliTerm::x(q + 2)]);
        c.add_pauli_rotation(0.09, &[PauliTerm::z(q), PauliTerm::z(q + 1)]);
    }
    c
}

fn ansatz_cases() -> Vec<(&'static str, Circuit)> {
    vec![
        ("trotter/12", trotter_layer(12)),
        ("hea/6", circuits::hardware_efficient_ansatz(6, 2, SEED)),
        ("hea/12", circuits::hardware_efficient_ansatz(12, 3, SEED)),
        ("hea/16", circuits::hardware_efficient_ansatz(16, 2, SEED)),
        ("hea/20", circuits::hardware_efficient_ansatz(20, 2, SEED)),
        ("qaoa/12", circuits::qaoa_circuit(12, 2, SEED)),
        ("qaoa/16", circuits::qaoa_circuit(16, 2, SEED)),
        ("qaoa/20", circuits::qaoa_circuit(20, 2, SEED)),
    ]
}

#[test]
fn bound_fusion_matches_independent_fusion() {
    for (name, template) in ansatz_cases() {
        let params = Parameters::all_rotations(&template);
        let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
        for point in 0..8 {
            let values = angles(params.num_slots(), 1000 + point);
            let independent = params.bind(&template, &values).unwrap();
            let expected = fuse_circuit(&independent, true).into_owned();
            let got = prepared.bind_fused(&values).unwrap();
            assert_streams_match(got, &expected, &format!("{name} point {point}"));
        }
    }
}

#[test]
fn n_bindings_match_n_independent_circuits_on_statevector() {
    for (name, template) in [
        ("hea/6", circuits::hardware_efficient_ansatz(6, 3, SEED)),
        ("hea/10", circuits::hardware_efficient_ansatz(10, 2, SEED)),
        ("hea/12", circuits::hardware_efficient_ansatz(12, 2, SEED)),
        ("qaoa/12", circuits::qaoa_circuit(12, 2, SEED)),
    ] {
        let params = Parameters::all_rotations(&template);
        let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
        for point in 0..5 {
            let values = angles(params.num_slots(), 2000 + point);
            let independent = params.bind(&template, &values).unwrap();
            let expected = statevector(&independent);
            let got = statevector(prepared.bind_fused(&values).unwrap());
            assert_states_match(&got, &expected, &format!("{name} point {point}"));
        }
    }
}

// A weight-2 `PauliRot` anchors a `Fused2q`, so its angle now reaches the state
// through the `Mat4` recipe rather than as a gate the plan carries unfused. The
// recipe reads the bound gate through `matrix_4x4`, and a missing arm there
// would replay the template's angle instead of the new one, silently.
#[test]
fn rxx_angles_rebind_through_a_fused_template() {
    let n = 16;
    let mut template = Circuit::new(n, 0);
    for q in 0..n {
        template.add_gate(Gate::H, &[q]);
    }
    for q in 0..n - 1 {
        template.add_gate(Gate::Rz(0.31 + 0.02 * q as f64), &[q]);
        template.add_pauli_rotation(
            0.19 + 0.01 * q as f64,
            &[PauliTerm::x(q), PauliTerm::x(q + 1)],
        );
        template.add_gate(Gate::Ry(0.23), &[q + 1]);
    }

    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
    for point in 0..5 {
        let values = angles(params.num_slots(), 4000 + point);
        let independent = params.bind(&template, &values).unwrap();
        let expected = statevector(&independent);
        let bound = prepared.bind_fused(&values).unwrap();
        assert!(
            bound.instructions.iter().any(|i| matches!(
                i,
                Instruction::Gate {
                    gate: Gate::Fused2q(_) | Gate::Multi2q(_),
                    ..
                }
            )),
            "point {point}: no anchored block in the bound stream, so the Mat4              recipe is not the path under test"
        );
        assert_states_match(
            &statevector(bound),
            &expected,
            &format!("rxx point {point}"),
        );
    }
}

// Fusion folds a weight-2 `PauliRot` with pending 1q neighbours into a dense
// 4x4, so the plan has to rebuild that block from the bound angle. The
// neighbours are generic rotations rather than named gates, which would bail
// capture and hide the replay path behind the fallback.
#[test]
fn an_absorbed_pauli_rotation_rebinds_through_the_plan() {
    let n = 12;
    let mut template = Circuit::new(n, 0);
    for q in 0..n {
        template.add_gate(Gate::Ry(0.11 + 0.02 * q as f64), &[q]);
        template.add_gate(Gate::Rz(0.29 + 0.03 * q as f64), &[q]);
    }
    for q in (0..n - 1).step_by(2) {
        template.add_pauli_rotation(
            0.37 + 0.01 * q as f64,
            &[PauliTerm::x(q), PauliTerm::y(q + 1)],
        );
    }
    for q in 0..n {
        template.add_gate(Gate::Ry(0.19 + 0.02 * q as f64), &[q]);
    }

    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
    assert!(prepared.reuses_fusion_plan(), "no fusion plan captured");
    assert!(
        prepared
            .bind_fused(&params.values(&template).unwrap())
            .unwrap()
            .instructions
            .iter()
            .all(|i| !matches!(
                i,
                Instruction::Gate {
                    gate: Gate::PauliRot(_),
                    ..
                }
            )),
        "a rotation survived fusion unabsorbed, so the replay path is not under test"
    );

    for point in 0..4 {
        let values = angles(params.num_slots(), 5000 + point);
        let independent = params.bind(&template, &values).unwrap();
        let expected = statevector(&independent);
        let bound = prepared.bind_fused(&values).unwrap();
        assert_states_match(
            &statevector(bound),
            &expected,
            &format!("absorbed pauli_rot point {point}"),
        );
    }
}

#[test]
fn plan_is_captured_for_the_ansatz_bench_shapes() {
    for (name, template) in ansatz_cases() {
        let params = Parameters::all_rotations(&template);
        let prepared = PreparedCircuit::new(template, params).unwrap();
        assert!(
            prepared.reuses_fusion_plan(),
            "{name}: no fusion plan captured"
        );
    }
}

// Zero angles drive several fused blocks to the identity, which fusion elides.
// The plan cannot express that, so the guard has to send the binding back
// through the pass pipeline rather than emit a stale block.
#[test]
fn degenerate_angles_fall_back_and_stay_correct() {
    let template = circuits::hardware_efficient_ansatz(12, 3, SEED);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();

    for values in [
        vec![0.0; params.num_slots()],
        vec![std::f64::consts::PI; params.num_slots()],
        vec![std::f64::consts::FRAC_PI_2; params.num_slots()],
    ] {
        let independent = params.bind(&template, &values).unwrap();
        let expected = fuse_circuit(&independent, true).into_owned();
        let got = prepared.bind_fused(&values).unwrap();
        assert_streams_match(got, &expected, "degenerate binding");
        assert_states_match(
            &statevector(got),
            &statevector(&expected),
            "degenerate state",
        );
    }
}

#[test]
fn rebinding_the_same_values_is_stable() {
    let template = circuits::hardware_efficient_ansatz(12, 2, SEED);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template, params.clone()).unwrap();
    let values = angles(params.num_slots(), 7);

    let first = statevector(prepared.bind_fused(&values).unwrap());
    let other = angles(params.num_slots(), 8);
    let _ = prepared.bind_fused(&other).unwrap();
    let again = statevector(prepared.bind_fused(&values).unwrap());
    assert_states_match(&again, &first, "rebinding the same values");
}

#[test]
fn shared_slots_bind_every_linked_gate() {
    let mut template = Circuit::new(3, 0);
    template.add_gate(Gate::Ry(0.0), &[0]);
    template.add_gate(Gate::Cx, &[0, 1]);
    template.add_gate(Gate::Ry(0.0), &[1]);
    template.add_gate(Gate::Rz(0.0), &[2]);

    let params = Parameters::from_links(
        vec![
            ParamLink {
                instruction: 0,
                slot: 0,
            },
            ParamLink {
                instruction: 2,
                slot: 0,
            },
            ParamLink {
                instruction: 3,
                slot: 1,
            },
        ],
        2,
    );

    let bound = params.bind(&template, &[0.6, 1.1]).unwrap();
    let mut expected = Circuit::new(3, 0);
    expected.add_gate(Gate::Ry(0.6), &[0]);
    expected.add_gate(Gate::Cx, &[0, 1]);
    expected.add_gate(Gate::Ry(0.6), &[1]);
    expected.add_gate(Gate::Rz(1.1), &[2]);
    assert_states_match(&statevector(&bound), &statevector(&expected), "shared slot");
}

#[test]
fn wrong_arity_is_rejected_without_panicking() {
    let template = circuits::hardware_efficient_ansatz(4, 1, SEED);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template, params.clone()).unwrap();

    assert!(prepared.bind_fused(&[]).is_err());
    assert!(
        prepared
            .bind_fused(&vec![0.1; params.num_slots() - 1])
            .is_err()
    );
    assert!(
        prepared
            .bind_fused(&vec![0.1; params.num_slots() + 1])
            .is_err()
    );
    assert!(prepared.bind_fused(&vec![0.1; params.num_slots()]).is_ok());
}

#[test]
fn non_finite_angle_is_rejected_without_panicking() {
    let template = circuits::hardware_efficient_ansatz(4, 1, SEED);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template, params.clone()).unwrap();

    let mut values = vec![0.1; params.num_slots()];
    values[0] = f64::NAN;
    assert!(prepared.bind_fused(&values).is_err());
    values[0] = f64::NEG_INFINITY;
    assert!(prepared.bind_fused(&values).is_err());
}

#[test]
fn out_of_range_link_is_rejected_without_panicking() {
    let template = circuits::hardware_efficient_ansatz(4, 1, SEED);
    let params = Parameters::from_links(
        vec![ParamLink {
            instruction: 10_000,
            slot: 0,
        }],
        1,
    );
    assert!(PreparedCircuit::new(template, params).is_err());
}

#[test]
fn link_to_a_gate_without_an_angle_is_rejected() {
    let mut template = Circuit::new(2, 0);
    template.add_gate(Gate::H, &[0]);
    template.add_gate(Gate::Cx, &[0, 1]);
    let params = Parameters::from_links(
        vec![ParamLink {
            instruction: 0,
            slot: 0,
        }],
        1,
    );
    assert!(PreparedCircuit::new(template, params).is_err());
}

#[test]
fn slot_that_no_gate_reads_binds_and_is_reported() {
    let template = circuits::hardware_efficient_ansatz(4, 1, SEED);
    let base = Parameters::all_rotations(&template);
    let widened = Parameters::from_links(base.links().to_vec(), base.num_slots() + 3);
    assert_eq!(
        widened.unread_slots(),
        (base.num_slots()..base.num_slots() + 3).collect::<Vec<_>>()
    );

    let mut prepared = PreparedCircuit::new(template.clone(), widened.clone()).unwrap();
    let values = angles(widened.num_slots(), 11);
    let narrow = base.bind(&template, &values[..base.num_slots()]).unwrap();
    let got = prepared.bind_fused(&values).unwrap();
    assert_states_match(
        &statevector(got),
        &statevector(&fuse_circuit(&narrow, true)),
        "widened slot set",
    );
}

// A replayed plan patches payloads in place, so a rotation it records no site
// for keeps the template's angle while the states still look plausible. Read
// the angle back off the fused stream rather than trusting agreement.
#[test]
fn a_bound_pauli_rotation_reaches_the_fused_stream() {
    let template = trotter_layer(12);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
    assert!(prepared.reuses_fusion_plan());

    let values = angles(params.num_slots(), 77);
    let fused = prepared.bind_fused(&values).unwrap();
    let bound: Vec<f64> = fused
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Gate {
                gate: Gate::PauliRot(data),
                ..
            } => Some(data.theta()),
            _ => None,
        })
        .collect();
    let expected: Vec<f64> = params
        .links()
        .iter()
        .filter(|link| {
            matches!(
                template.instructions[link.instruction],
                Instruction::Gate {
                    gate: Gate::PauliRot(_),
                    ..
                }
            )
        })
        .map(|link| values[link.slot])
        .collect();
    assert_eq!(bound.len(), 20, "two native rotations per site, 10 sites");

    assert_eq!(bound, expected);
}

// A circuit with no parameters must still bind and fuse exactly as the
// ordinary path does.
#[test]
fn unparameterized_template_is_unchanged() {
    let template = circuits::qft_circuit(12);
    let params = Parameters::new(0);
    let mut prepared = PreparedCircuit::new(template.clone(), params).unwrap();
    let expected = fuse_circuit(&template, true).into_owned();
    let got = prepared.bind_fused(&[]).unwrap();
    assert_streams_match(got, &expected, "unparameterized");
    assert_states_match(
        &statevector(got),
        &statevector(&expected),
        "unparameterized",
    );
}

// A 1q run that is diagonal at capture time commutes backwards past a CX
// control, and can then be absorbed into a Fused2q that carries no 2x2 payload
// of its own. Binding an angle that makes the run non-diagonal stops the
// reorder, so the guard on the run has to catch it even though no site does.
#[test]
fn binding_that_flips_1q_diagonality_falls_back() {
    let mut template = Circuit::new(12, 0);
    template.add_gate(Gate::Cx, &[0, 1]);
    template.add_gate(Gate::Ry(0.0), &[0]);
    template.add_gate(Gate::Rz(0.7), &[0]);

    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
    assert!(prepared.reuses_fusion_plan());

    for values in [[1.1, 0.7], [std::f64::consts::FRAC_PI_3, 0.4], [0.0, 0.7]] {
        let independent = params.bind(&template, &values).unwrap();
        let expected = fuse_circuit(&independent, true).into_owned();
        let got = prepared.bind_fused(&values).unwrap();
        assert_streams_match(got, &expected, &format!("{values:?}"));
        assert_states_match(
            &statevector(got),
            &statevector(&expected),
            &format!("{values:?}"),
        );
    }
}

#[test]
fn run_matches_simulate_on_every_binding() {
    let template = circuits::hardware_efficient_ansatz(10, 3, SEED);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();

    for point in 0..4 {
        let values = angles(params.num_slots(), 5000 + point);
        let independent = params.bind(&template, &values).unwrap();
        let expected = prism_q::simulate(&independent)
            .seed(SEED)
            .run()
            .unwrap()
            .probabilities
            .expect("no probabilities")
            .to_vec();
        let got = prepared
            .run(&values, SEED)
            .unwrap()
            .probabilities
            .expect("no probabilities")
            .to_vec();
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "point {point} outcome {i}: {a} vs {b}"
            );
        }
    }
}

// A held backend would carry its RNG from one call into the next, so repeated
// calls with one seed must each measure what a fresh `simulate` run measures.
#[test]
fn run_with_mid_circuit_measurement_is_independent_of_call_history() {
    let mut template = Circuit::new(3, 3);
    template.add_gate(Gate::H, &[0]);
    template.add_gate(Gate::Rx(0.0), &[1]);
    template.add_gate(Gate::Cx, &[0, 2]);
    template.add_measure(0, 0);
    template.add_gate(Gate::Rx(0.0), &[1]);
    template.add_gate(Gate::Cx, &[1, 2]);
    template.add_measure(1, 1);
    template.add_measure(2, 2);
    let params = Parameters::from_links(
        vec![
            ParamLink {
                instruction: 1,
                slot: 0,
            },
            ParamLink {
                instruction: 4,
                slot: 1,
            },
        ],
        2,
    );
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();

    let a = [1.3, 0.4];
    let b = [0.7, 2.2];
    let c = [2.9, 1.1];
    for values in [a, a, a, b, a, c, b, a] {
        let independent = params.bind(&template, &values).unwrap();
        let expected = prism_q::simulate(&independent).seed(SEED).run().unwrap();
        let got = prepared.run(&values, SEED).unwrap();
        assert_eq!(got.classical_bits, expected.classical_bits, "{values:?}");
        if let (Some(got), Some(expected)) = (got.probabilities, expected.probabilities) {
            let (got, expected) = (got.to_vec(), expected.to_vec());
            assert_eq!(got.len(), expected.len());
            for (i, (x, y)) in got.iter().zip(&expected).enumerate() {
                assert!((x - y).abs() < 1e-12, "{values:?} outcome {i}: {x} vs {y}");
            }
        }
    }
}

// The density matrix backend does not accept fused gates, so `run` has to hand
// it the bound template rather than the replayed skeleton.
#[test]
fn run_on_a_backend_without_fused_gates_binds_unfused() {
    let template = circuits::hardware_efficient_ansatz(6, 2, SEED);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::with_backend(
        template.clone(),
        params.clone(),
        prism_q::BackendKind::DensityMatrix,
    )
    .unwrap();

    for point in 0..3 {
        let values = angles(params.num_slots(), 6000 + point);
        let independent = params.bind(&template, &values).unwrap();
        let expected = prism_q::simulate(&independent)
            .backend(prism_q::BackendKind::DensityMatrix)
            .seed(SEED)
            .run()
            .unwrap()
            .probabilities
            .expect("no probabilities")
            .to_vec();
        let got = prepared
            .run(&values, SEED)
            .unwrap()
            .probabilities
            .expect("no probabilities")
            .to_vec();
        assert_eq!(got.len(), expected.len());
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "point {point} outcome {i}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn a_clone_carries_the_plan_and_agrees() {
    let template = circuits::hardware_efficient_ansatz(12, 2, SEED);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template, params.clone()).unwrap();
    let mut copy = prepared.clone();
    assert!(copy.reuses_fusion_plan());

    let values = angles(params.num_slots(), 77);
    let a = statevector(prepared.bind_fused(&values).unwrap());
    let b = statevector(copy.bind_fused(&values).unwrap());
    assert_states_match(&a, &b, "clone");
}

// Links are instruction indices, so inserting a gate shifts every later link.
// A set built by `all_rotations` pins the gate kinds it saw, so the edited
// circuit is rejected instead of binding the wrong gates.
#[test]
fn editing_the_circuit_after_recording_links_is_rejected() {
    let mut template = Circuit::new(3, 0);
    for q in 0..3 {
        template.add_gate(Gate::Ry(0.1), &[q]);
        template.add_gate(Gate::Rz(0.2), &[q]);
    }
    let params = Parameters::all_rotations(&template);
    assert!(
        params
            .bind(&template, &vec![0.5; params.num_slots()])
            .is_ok()
    );

    let mut edited = template.clone();
    edited.instructions.insert(
        0,
        Instruction::Gate {
            gate: Gate::Rz(0.9),
            targets: prism_q::circuit::smallvec![0],
        },
    );
    assert!(
        params
            .bind(&edited, &vec![0.5; params.num_slots()])
            .is_err()
    );
}

#[test]
fn named_slots_resolve_both_ways() {
    let template = circuits::hardware_efficient_ansatz(4, 1, SEED);
    let base = Parameters::all_rotations(&template);
    let names: Vec<String> = (0..base.num_slots())
        .map(|k| format!("theta_{k}"))
        .collect();
    let named = base.clone().with_names(names);

    assert_eq!(named.name_of(0), Some("theta_0"));
    assert_eq!(named.slot_of("theta_3"), Some(3));
    assert_eq!(named.slot_of("nope"), None);
    assert_eq!(base.name_of(0), None);
}

#[test]
fn the_builder_lowers_a_pauli_rotation_the_way_the_circuit_does() {
    let factors = [PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)];
    let built = CircuitBuilder::new(3)
        .h(0)
        .pauli_rotation(0.7, &factors)
        .pauli_rotation(0.3, &[PauliTerm::z(0), PauliTerm::z(1)])
        .build();

    let mut direct = Circuit::new(3, 0);
    direct.add_gate(Gate::H, &[0]);
    direct.add_pauli_rotation(0.7, &factors);
    direct.add_pauli_rotation(0.3, &[PauliTerm::z(0), PauliTerm::z(1)]);

    assert_eq!(
        format!("{:?}", built.instructions),
        format!("{:?}", direct.instructions)
    );
}

#[test]
fn a_builder_pauli_rotation_takes_a_parameter_slot() {
    let mut builder = CircuitBuilder::new(3);
    builder
        .h(0)
        .pauli_rotation(0.1, &[PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)])
        .param(0);
    let (template, params) = builder.build_parametric();

    let bound = params.bind(&template, &[1.25]).expect("bind");
    assert_eq!(params.values(&bound).expect("values"), vec![1.25]);
}

// A prepared circuit crosses a thread boundary in the Python bindings, which
// release the GIL around `run`. The backend the route holds is what makes this
// non-obvious, so pin it here rather than discovering it downstream.
#[test]
fn a_prepared_circuit_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<PreparedCircuit>();
}

fn pauli_strings(n: usize) -> Vec<Vec<PauliTerm>> {
    let mut strings: Vec<Vec<PauliTerm>> = (0..n).map(|q| vec![PauliTerm::z(q)]).collect();
    strings.push(vec![PauliTerm::x(0), PauliTerm::x(1)]);
    strings.push(vec![PauliTerm::y(2), PauliTerm::z(4), PauliTerm::x(n - 1)]);
    strings
}

// A `ZZ` chain and an `X` field each form one group past the pair budget, so
// both the Z-only and the basis-rotated moments passes run, and the `Y` term
// lands in a small group that takes the pair expansion.
fn energy(n: usize) -> PauliObservable {
    let chain = (0..n - 1).map(|q| (1.0, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]));
    let field = (0..n).map(|q| (0.5, vec![PauliTerm::x(q)]));
    let extra = [
        (0.25, vec![PauliTerm::y(0), PauliTerm::y(1)]),
        (-0.75, vec![]),
    ];
    PauliObservable::from_terms(chain.chain(field).chain(extra).collect::<Vec<_>>()).unwrap()
}

// Random bindings, then the three degenerate ones that send `bind_fused` back
// through the pass pipeline.
fn bindings_with_fallback(slots: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut points: Vec<Vec<f64>> = (0..3).map(|k| angles(slots, seed + k)).collect();
    points.push(vec![0.0; slots]);
    points.push(vec![std::f64::consts::PI; slots]);
    points.push(angles(slots, seed + 3));
    points.push(vec![std::f64::consts::FRAC_PI_2; slots]);
    points
}

fn assert_values_close(got: &[f64], expected: &[f64], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}");
    for (i, (a, b)) in got.iter().zip(expected).enumerate() {
        assert!((a - b).abs() < 1e-12, "{what} value {i}: {a} vs {b}");
    }
}

fn assert_observables_close(
    got: &ObservableExpectation,
    expected: &ObservableExpectation,
    what: &str,
) {
    assert_values_close(&[got.mean], &[expected.mean], &format!("{what} mean"));
    assert_eq!(
        got.variance.is_some(),
        expected.variance.is_some(),
        "{what}"
    );
    if let (Some(a), Some(b)) = (got.variance, expected.variance) {
        assert_values_close(&[a], &[b], &format!("{what} variance"));
    }
    match (&got.group_variances, &expected.group_variances) {
        (Some(a), Some(b)) => assert_values_close(a, b, &format!("{what} group variances")),
        (None, None) => {}
        _ => panic!("{what}: group variances present on one side only"),
    }
    assert_eq!(got.metadata.backend, expected.metadata.backend, "{what}");
}

// Eighteen qubits is where `Auto` offers the circuit to the tensor route first.
#[test]
fn expectation_values_match_simulate_on_every_binding() {
    for n in [12, 18] {
        let template = circuits::hardware_efficient_ansatz(n, 3, SEED);
        let params = Parameters::all_rotations(&template);
        let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
        assert!(prepared.reuses_fusion_plan());
        let observables = pauli_strings(n);

        for (point, values) in bindings_with_fallback(params.num_slots(), 7000)
            .iter()
            .enumerate()
        {
            let independent = params.bind(&template, values).unwrap();
            let expected = prism_q::simulate(&independent)
                .seed(SEED)
                .expectation_values(&observables)
                .unwrap();
            let got = prepared
                .expectation_values(values, &observables, SEED)
                .unwrap();
            assert_values_close(&got, &expected, &format!("{n} qubits, point {point}"));
        }
    }
}

#[test]
fn observable_expectation_matches_simulate_on_every_binding() {
    let template = circuits::hardware_efficient_ansatz(12, 3, SEED);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
    let observable = energy(12);

    for (point, values) in bindings_with_fallback(params.num_slots(), 7100)
        .iter()
        .enumerate()
    {
        let independent = params.bind(&template, values).unwrap();
        let expected = prism_q::simulate(&independent)
            .seed(SEED)
            .observable_expectation(&observable)
            .unwrap();
        assert!(expected.variance.is_some());
        let got = prepared
            .observable_expectation(values, &observable, SEED)
            .unwrap();
        assert_observables_close(&got, &expected, &format!("point {point}"));
    }
}

// Two halves that never interact, which `Auto` decomposes, so no held route
// exists and every terminal asks `simulate`.
fn independent_halves() -> Circuit {
    let mut c = Circuit::new(6, 0);
    for half in [0, 3] {
        for q in half..half + 3 {
            c.add_gate(Gate::Ry(0.3), &[q]);
            c.add_gate(Gate::Rz(0.4), &[q]);
        }
        c.add_gate(Gate::Cx, &[half, half + 1]);
        c.add_gate(Gate::Cx, &[half + 1, half + 2]);
        for q in half..half + 3 {
            c.add_gate(Gate::Rx(0.5), &[q]);
        }
    }
    c
}

// Explicit kinds that take the grouped and the per-term routes, one of which
// holds no fused gates, and an `Auto` template with no held route at all.
#[test]
fn expectation_terminals_match_simulate_across_routes() {
    let ansatz = circuits::hardware_efficient_ansatz(6, 2, SEED);
    let observables = pauli_strings(6);
    let observable = energy(6);

    for (kind, template) in [
        (prism_q::BackendKind::Statevector, ansatz.clone()),
        (
            prism_q::BackendKind::Mps { max_bond_dim: 64 },
            ansatz.clone(),
        ),
        (prism_q::BackendKind::Factored, ansatz.clone()),
        (prism_q::BackendKind::DensityMatrix, ansatz),
        (prism_q::BackendKind::Auto, independent_halves()),
    ] {
        let params = Parameters::all_rotations(&template);
        let mut prepared =
            PreparedCircuit::with_backend(template.clone(), params.clone(), kind.clone()).unwrap();
        for (point, values) in bindings_with_fallback(params.num_slots(), 7200)
            .iter()
            .enumerate()
        {
            let what = format!("{kind:?} point {point}");
            let independent = params.bind(&template, values).unwrap();
            let sim = || {
                prism_q::simulate(&independent)
                    .backend(kind.clone())
                    .seed(SEED)
            };
            let expected = sim().expectation_values(&observables).unwrap();
            let got = prepared
                .expectation_values(values, &observables, SEED)
                .unwrap();
            assert_values_close(&got, &expected, &what);

            let expected = sim().observable_expectation(&observable).unwrap();
            let got = prepared
                .observable_expectation(values, &observable, SEED)
                .unwrap();
            assert_observables_close(&got, &expected, &what);
        }
    }
}

#[test]
fn expectation_terminals_reject_a_measured_template_as_simulate_does() {
    let mut template = Circuit::new(4, 1);
    template.instructions = circuits::hardware_efficient_ansatz(4, 1, SEED).instructions;
    template.add_measure(0, 0);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template, params.clone()).unwrap();
    let values = angles(params.num_slots(), 1);

    assert!(
        prepared
            .expectation_values(&values, &pauli_strings(4), SEED)
            .is_err()
    );
    assert!(
        prepared
            .observable_expectation(&values, &energy(4), SEED)
            .is_err()
    );
    assert!(prepared.run(&values, SEED).is_ok());
}

// Eight qubits splits across workers under `parallel`; fifteen runs in order.
#[test]
fn many_terminals_are_bit_identical_to_a_loop() {
    for n in [8, 15] {
        let template = circuits::hardware_efficient_ansatz(n, 2, SEED);
        let params = Parameters::all_rotations(&template);
        let points = bindings_with_fallback(params.num_slots(), 7300 + n as u64);
        let observables = pauli_strings(n);
        let observable = energy(n);

        let mut looped = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
        let mut many = PreparedCircuit::new(template, params).unwrap();

        let runs = many.run_many(&points, SEED).unwrap();
        let values = many
            .expectation_values_many(&points, &observables, SEED)
            .unwrap();
        let energies = many
            .observable_expectation_many(&points, &observable, SEED)
            .unwrap();
        assert_eq!(runs.len(), points.len());
        assert_eq!(values.len(), points.len());
        assert_eq!(energies.len(), points.len());

        for (point, binding) in points.iter().enumerate() {
            let what = format!("{n} qubits, point {point}");
            let run = looped.run(binding, SEED).unwrap();
            assert_eq!(
                runs[point].probabilities.as_ref().unwrap().to_vec(),
                run.probabilities.unwrap().to_vec(),
                "{what}"
            );
            assert_eq!(
                values[point],
                looped
                    .expectation_values(binding, &observables, SEED)
                    .unwrap(),
                "{what}"
            );
            let energy = looped
                .observable_expectation(binding, &observable, SEED)
                .unwrap();
            assert_eq!(energies[point].mean, energy.mean, "{what}");
            assert_eq!(energies[point].variance, energy.variance, "{what}");
            assert_eq!(
                energies[point].group_variances, energy.group_variances,
                "{what}"
            );
        }
    }
}

#[test]
fn many_terminals_report_the_first_failing_binding() {
    let template = circuits::hardware_efficient_ansatz(4, 1, SEED);
    let params = Parameters::all_rotations(&template);
    let mut prepared = PreparedCircuit::new(template, params.clone()).unwrap();
    let good = angles(params.num_slots(), 3);
    let short = vec![0.1];
    let err = prepared
        .run_many(&[good.clone(), short, good], SEED)
        .unwrap_err();
    assert!(matches!(err, prism_q::PrismError::InvalidParameter { .. }));
    assert!(prepared.run_many::<Vec<f64>>(&[], SEED).unwrap().is_empty());
}

// Each binding in a sweep has to measure what a solo run with the same seed
// measures, whichever worker it lands on.
#[test]
fn run_many_with_mid_circuit_measurement_matches_simulate() {
    let mut template = Circuit::new(3, 3);
    template.add_gate(Gate::H, &[0]);
    template.add_gate(Gate::Rx(0.0), &[1]);
    template.add_gate(Gate::Cx, &[0, 2]);
    template.add_measure(0, 0);
    template.add_gate(Gate::Rx(0.0), &[1]);
    template.add_gate(Gate::Cx, &[1, 2]);
    template.add_measure(1, 1);
    template.add_measure(2, 2);
    let params = Parameters::from_links(
        vec![
            ParamLink {
                instruction: 1,
                slot: 0,
            },
            ParamLink {
                instruction: 4,
                slot: 1,
            },
        ],
        2,
    );
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();

    let points: Vec<[f64; 2]> = (0..24)
        .map(|k| {
            let v = angles(2, 8000 + k % 5);
            [v[0], v[1]]
        })
        .collect();
    for seed in [SEED, 9] {
        let outcomes = prepared.run_many(&points, seed).unwrap();
        for (values, got) in points.iter().zip(&outcomes) {
            let independent = params.bind(&template, values).unwrap();
            let expected = prism_q::simulate(&independent).seed(seed).run().unwrap();
            assert_eq!(got.classical_bits, expected.classical_bits, "{values:?}");
        }
    }
}

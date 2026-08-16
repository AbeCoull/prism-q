//! Parameter binding: agreement with independently constructed circuits, plan
//! reuse fidelity, and the rejection paths.

use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::fusion::fuse_circuit;
use prism_q::{
    Circuit, Gate, Instruction, ParamLink, Parameters, PauliTerm, PreparedCircuit, circuits,
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
            assert_eq!(x.gates.len(), y.gates.len(), "{what}: multi_fused arity");
            assert_eq!(x.all_diagonal, y.all_diagonal, "{what}: all_diagonal");
            for (k, (gx, gy)) in x.gates.iter().zip(&y.gates).enumerate() {
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

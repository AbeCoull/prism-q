//! Pipeline coverage for the native Pauli rotation: the statevector keeps the
//! native gate through fusion, every declining backend receives the ladder
//! expansion, and both routes agree at parallel-kernel sizes.

use crate::common;

use common::{MPS_EPS, SEED, SV_EPS, assert_probs_close, mix_seed};
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::{Circuit, Instruction, expand_pauli_rotations};
use prism_q::gates::Gate;
use prism_q::{BackendKind, PauliTerm, sim};

fn trotter_like_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Ry(0.23 + 0.11 * q as f64), &[q]);
    }
    c.add_pauli_rotation(0.41, &[PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)]);
    c.add_pauli_rotation(0.29, &[PauliTerm::z(0), PauliTerm::z(2), PauliTerm::z(3)]);
    c.add_pauli_rotation(0.37, &[PauliTerm::x(1), PauliTerm::x(3)]);
    c.add_pauli_rotation(0.19, &[PauliTerm::z(0), PauliTerm::y(n - 1)]);
    // Top-qubit pivot with a nonzero low X-mask, so the tiled pair kernel
    // pairs across permuted tile indices rather than identity ones.
    c.add_pauli_rotation(0.23, &[PauliTerm::x(n - 3), PauliTerm::y(n - 1)]);
    c
}

fn state_of(backend: &mut StatevectorBackend, circuit: &Circuit) -> Vec<Complex64> {
    sim::run_on(backend, circuit).unwrap();
    backend.state_vector().to_vec()
}

// Native route pin, per the QftBlock lesson: the fused stream the statevector
// executes must still contain the native gate, not just produce matching
// amplitudes.
//
// Weight 2 is the one width fusion may take, as the anchor of a `Fused2q`, and
// only when the rotation has a neighbouring 1q gate to absorb. Both weight-3
// terms survive on width alone. So does `x(1) x(3)`, whose qubits the two
// weight-3 terms ahead of it already flushed, leaving it nothing to absorb; the
// two rotations reaching an untouched qubit are the ones that anchor. At 16
// qubits the three survivors run back to back, so they reach the kernel as one
// `MultiPauliRot` batch.
#[test]
fn fusion_keeps_the_native_gate() {
    let circuit = trotter_like_circuit(16);
    let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
    let mut native_widths: Vec<usize> = Vec::new();
    let mut batches = 0;
    for inst in &fused.instructions {
        match inst {
            Instruction::Gate {
                gate: Gate::PauliRot(data),
                ..
            } => native_widths.push(data.axes().len()),
            Instruction::Gate {
                gate: Gate::MultiPauliRot(data),
                ..
            } => {
                batches += 1;
                native_widths.extend(
                    data.rotations()
                        .iter()
                        .map(|&(x, z, _)| (x | z).count_ones() as usize),
                );
            }
            _ => {}
        }
    }
    assert_eq!(
        native_widths,
        vec![3, 3, 2],
        "fusion took a rotation it has no anchor for, or left one unabsorbed"
    );
    assert_eq!(batches, 1, "the surviving rotations did not batch");
    assert!(StatevectorBackend::new(SEED).supports_pauli_rotation());
}

fn jordan_wigner_step(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for (coefficient, factors) in prism_q::circuits::jordan_wigner_hamiltonian(n, 200, SEED) {
        if !factors.is_empty() {
            c.add_pauli_rotation(0.1 * coefficient, &factors);
        }
    }
    c
}

// The batched stream against the same circuit applied gate by gate without
// fusion, at a width that gathers subcubes and one that runs mostly in place.
#[test]
fn batched_trotter_step_matches_per_gate_application() {
    for n in [16, 20] {
        let circuit = jordan_wigner_step(n);
        let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
        assert!(
            fused.instructions.iter().any(|inst| matches!(
                inst,
                Instruction::Gate {
                    gate: Gate::MultiPauliRot(_),
                    ..
                }
            )),
            "{n}q: no MultiPauliRot in the fused stream"
        );
        let batched = state_of(&mut StatevectorBackend::new(SEED), &circuit);
        let mut per_gate = StatevectorBackend::new(SEED);
        per_gate.init(n, 0).unwrap();
        per_gate.apply_instructions(&circuit.instructions).unwrap();
        for (i, (a, e)) in batched.iter().zip(per_gate.state_vector()).enumerate() {
            let diff = (a - e).norm();
            assert!(diff < 1e-12, "{n}q amplitude {i}: |diff| = {diff:e}");
        }
    }
}

// Native kernel against the ladder lowering across the parallel threshold,
// covering the low-pivot, high-pivot, and Z-diagonal kernel branches.
#[test]
fn native_matches_ladder_at_parallel_sizes() {
    for n in [6, 16] {
        let circuit = trotter_like_circuit(n);
        let expanded = expand_pauli_rotations(&circuit);
        assert!(matches!(expanded, std::borrow::Cow::Owned(_)));
        let native = state_of(&mut StatevectorBackend::new(SEED), &circuit);
        let lowered = state_of(&mut StatevectorBackend::new(SEED), &expanded);
        // 1e-12 rather than SV_EPS: one native pass against a ladder of tens
        // of gates accumulates only rounding, not backend error.
        for (i, (a, e)) in native.iter().zip(&lowered).enumerate() {
            let diff = (a - e).norm();
            assert!(diff < 1e-12, "{n}q amplitude {i}: |diff| = {diff:e}");
        }
    }
}

// A weight-k string lowers to 2(k-1) CX around one Rz, plus the basis layer.
#[test]
fn ladder_lowering_shape() {
    let mut c = Circuit::new(4, 0);
    c.add_pauli_rotation(
        0.5,
        &[
            PauliTerm::x(0),
            PauliTerm::y(1),
            PauliTerm::z(2),
            PauliTerm::z(3),
        ],
    );
    let expanded = expand_pauli_rotations(&c);
    let count = |wanted: fn(&Gate) -> bool| {
        expanded
            .instructions
            .iter()
            .filter(|inst| matches!(inst, Instruction::Gate { gate, .. } if wanted(gate)))
            .count()
    };
    assert_eq!(count(|g| matches!(g, Gate::PauliRot(_))), 0);
    assert_eq!(count(|g| matches!(g, Gate::Cx)), 6);
    assert_eq!(count(|g| matches!(g, Gate::Rz(_))), 1);
    // X basis change: H twice; Y: Sdg + H in, H + S out.
    assert_eq!(count(|g| matches!(g, Gate::H)), 4);
    assert_eq!(count(|g| matches!(g, Gate::Sdg)), 1);
    assert_eq!(count(|g| matches!(g, Gate::S)), 1);
}

#[test]
fn expansion_borrows_without_native_gates() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_pauli_rotation(0.3, &[PauliTerm::z(0), PauliTerm::z(1)]);
    assert!(matches!(
        expand_pauli_rotations(&c),
        std::borrow::Cow::Borrowed(_)
    ));
}

// Declining backends receive the ladder through the sim-layer expansion: a
// raw PauliRot reaching MPS or sparse dispatch panics on arity, and the
// density matrix would pay two extra conjugation passes per gate.
#[test]
fn declining_backends_match_statevector_through_expansion() {
    let circuit = trotter_like_circuit(6);
    let reference = common::sv_reference_probs(&circuit);

    let cases: [(&str, Box<dyn Backend>, f64); 3] = [
        ("mps", Box::new(MpsBackend::new(SEED, 64)), MPS_EPS),
        ("sparse", Box::new(SparseBackend::new(SEED)), SV_EPS),
        (
            "density_matrix",
            Box::new(DensityMatrixBackend::new(SEED)),
            SV_EPS,
        ),
    ];
    for (label, mut backend, eps) in cases {
        assert!(!backend.supports_pauli_rotation(), "{label}");
        sim::run_on(&mut *backend, &circuit).unwrap();
        let probs = backend.probabilities().unwrap();
        assert_probs_close(&probs, &reference, eps, label);
    }
}

// Mid-circuit measurement forces the per-shot route, which expands per block
// plan before fusing; MPS erroring here would mean that expansion was skipped.
#[test]
fn per_shot_route_expands_for_mps() {
    let mut c = trotter_like_circuit(4);
    c.num_classical_bits = 2;
    c.add_measure(0, 0);
    c.add_pauli_rotation(0.31, &[PauliTerm::x(1), PauliTerm::y(2), PauliTerm::z(3)]);
    c.add_measure(3, 1);

    let sv: Vec<Vec<bool>> = (0..32)
        .map(|i| {
            sim::simulate(&c)
                .backend(BackendKind::Statevector)
                .seed(mix_seed(SEED, i))
                .run()
                .unwrap()
                .classical_bits
        })
        .collect();
    let mps = sim::simulate(&c)
        .backend(BackendKind::Mps { max_bond_dim: 64 })
        .seed(SEED)
        .shots(32)
        .unwrap();
    assert_eq!(sv, mps.shots);
}

#[test]
fn z_only_rotation_preserves_sparsity() {
    let mut z_only = Circuit::new(3, 0);
    z_only.add_pauli_rotation(0.4, &[PauliTerm::z(0), PauliTerm::z(1), PauliTerm::z(2)]);
    assert!(z_only.is_sparse_friendly());
    assert!(!z_only.is_clifford_only());
    assert!(z_only.has_entangling_gates());

    let mut mixed = Circuit::new(3, 0);
    mixed.add_pauli_rotation(0.4, &[PauliTerm::x(0), PauliTerm::z(1), PauliTerm::z(2)]);
    assert!(!mixed.is_sparse_friendly());
}

// Agreement with the ladder is not the property to assert: the ladder passes it
// whether or not the native gate survived. Assert the reparsed instruction is a
// `PauliRot`, the way `the_native_gate_carries_the_gradient` does on the
// gradient side.
#[test]
fn qasm_export_keeps_the_native_pauli_rotation() {
    let circuit = trotter_like_circuit(4);
    let qasm = prism_q::circuit::qasm_export::to_qasm3(&circuit).unwrap();
    let round = prism_q::circuit::openqasm::parse(&qasm).expect("reparse");

    assert_eq!(circuit.instructions.len(), round.instructions.len());
    let native = |c: &Circuit| {
        c.instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    Instruction::Gate {
                        gate: Gate::PauliRot(_),
                        ..
                    }
                )
            })
            .count()
    };
    assert!(native(&circuit) > 0, "fixture carries no PauliRot");
    assert_eq!(native(&circuit), native(&round));
    assert!(
        !qasm.contains("cx"),
        "export fell back to the ladder: {qasm}"
    );
}

// The ladder is still reachable for a consumer that cannot read the extension.
#[test]
fn expanding_first_exports_the_portable_ladder() {
    let circuit = trotter_like_circuit(4);
    let lowered = prism_q::circuit::expand_pauli_rotations(&circuit);
    let qasm = prism_q::circuit::qasm_export::to_qasm3(&lowered).unwrap();
    assert!(qasm.contains("cx"));
    assert!(!qasm.contains("rxyz"));
}

// Noise events are indexed per instruction, so the trajectory engine applies
// the stream raw: the statevector serves the gate natively, and a backend
// that needs the lowering is rejected with a typed error, never a panic.
#[test]
fn noisy_trajectories_run_native_and_reject_declining_backends() {
    let circuit = trotter_like_circuit(4);
    let noise = prism_q::NoiseModel::uniform_depolarizing(&circuit, 0.01);

    let sv = sim::simulate(&circuit)
        .backend(BackendKind::Statevector)
        .noise(&noise)
        .seed(SEED)
        .shots(8)
        .unwrap();
    assert_eq!(sv.shots.len(), 8);

    let err = sim::simulate(&circuit)
        .backend(BackendKind::Mps { max_bond_dim: 64 })
        .noise(&noise)
        .seed(SEED)
        .shots(8)
        .unwrap_err();
    assert!(matches!(
        err,
        prism_q::PrismError::IncompatibleBackend { .. }
    ));
}

#[test]
#[should_panic(expected = "duplicate factor")]
fn constructor_rejects_duplicate_qubits() {
    let mut c = Circuit::new(2, 0);
    c.add_pauli_rotation(0.1, &[PauliTerm::x(0), PauliTerm::z(0)]);
}

#[test]
#[should_panic(expected = "at least one factor")]
fn constructor_rejects_empty_strings() {
    let mut c = Circuit::new(1, 0);
    c.add_pauli_rotation(0.1, &[]);
}

// `rxx` and `ryy` parsed to a seven-gate basis-change ladder before the native
// rotation gained a spelling. Both forms are the same unitary, so pin that
// rather than trusting the change: the ladder is written out here so the
// comparison does not run through the code that replaced it.
#[test]
fn parsed_rxx_and_ryy_agree_with_the_ladder_they_replaced() {
    use prism_q::circuit::openqasm;

    let cases = [
        (
            "rxx",
            "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nrxx(0.7) q[0], q[1];",
            "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\n\
             h q[0];\nh q[1];\ncx q[0], q[1];\nrz(0.7) q[1];\ncx q[0], q[1];\nh q[0];\nh q[1];",
        ),
        (
            "ryy",
            "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nryy(0.7) q[0], q[1];",
            "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\n\
             rx(pi/2) q[0];\nrx(pi/2) q[1];\ncx q[0], q[1];\nrz(0.7) q[1];\ncx q[0], q[1];\n\
             rx(-pi/2) q[0];\nrx(-pi/2) q[1];",
        ),
    ];

    for (label, native_src, ladder_src) in cases {
        let native = openqasm::parse(native_src).expect("parse native");
        assert_eq!(
            native.instructions.len(),
            2,
            "{label}: expected one rotation"
        );

        let ladder = openqasm::parse(ladder_src).expect("parse ladder");
        let mut a = StatevectorBackend::new(SEED);
        let mut b = StatevectorBackend::new(SEED);
        let native_probs = sim::run_on(&mut a, &native)
            .expect("run native")
            .probabilities
            .expect("probs")
            .to_vec();
        let ladder_probs = sim::run_on(&mut b, &ladder)
            .expect("run ladder")
            .probabilities
            .expect("probs")
            .to_vec();
        assert_probs_close(&native_probs, &ladder_probs, SV_EPS, label);
    }
}

fn entangled_pauli_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Ry(0.23 + 0.11 * q as f64), &[q]);
        c.add_gate(Gate::Rz(0.67 - 0.07 * q as f64), &[q]);
    }
    for q in 0..n.saturating_sub(1) {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c
}

#[test]
fn pauli_rot_run_lt4_and_pivot0_match_ladder() {
    for n in [6usize, 16] {
        let mut c = entangled_pauli_circuit(n);
        let terms_list: Vec<(f64, Vec<PauliTerm>)> = vec![
            // pivot == 0 (xmask == 1), X0=true, Z0=false/true
            (0.41, vec![PauliTerm::x(0)]),
            (0.37, vec![PauliTerm::y(0)]),
            (0.29, vec![PauliTerm::x(0), PauliTerm::z(1)]),
            (
                0.53,
                vec![PauliTerm::y(0), PauliTerm::z(1), PauliTerm::z(2)],
            ),
            (0.19, vec![PauliTerm::x(0), PauliTerm::z(n - 1)]),
            (
                0.31,
                vec![PauliTerm::y(0), PauliTerm::z(1), PauliTerm::z(n - 1)],
            ),
            // run == 1 with pivot > 0 across all (X0, Z0, num_y % 4)
            (0.23, vec![PauliTerm::x(0), PauliTerm::x(2)]),
            (0.43, vec![PauliTerm::x(0), PauliTerm::y(2)]),
            (0.47, vec![PauliTerm::y(0), PauliTerm::x(2)]),
            (0.59, vec![PauliTerm::y(0), PauliTerm::y(2)]),
            (0.61, vec![PauliTerm::z(0), PauliTerm::x(2)]),
            (0.17, vec![PauliTerm::z(0), PauliTerm::y(2)]),
            (
                0.27,
                vec![
                    PauliTerm::y(0),
                    PauliTerm::y(1),
                    PauliTerm::y(2),
                    PauliTerm::x(3),
                ],
            ),
            (
                0.33,
                vec![
                    PauliTerm::x(0),
                    PauliTerm::y(1),
                    PauliTerm::y(2),
                    PauliTerm::y(3),
                    PauliTerm::x(4),
                ],
            ),
            // run == 2 (!X0 && !Z0) across num_y % 4 in 0..4
            (0.21, vec![PauliTerm::x(1), PauliTerm::x(3)]),
            (0.35, vec![PauliTerm::x(1), PauliTerm::y(3)]),
            (0.49, vec![PauliTerm::y(1), PauliTerm::y(3)]),
            (
                0.55,
                vec![PauliTerm::y(1), PauliTerm::y(3), PauliTerm::y(4)],
            ),
            (
                0.63,
                vec![
                    PauliTerm::x(1),
                    PauliTerm::y(2),
                    PauliTerm::y(3),
                    PauliTerm::y(4),
                    PauliTerm::y(5),
                ],
            ),
            // num_blocks < 4 at n == 16 (pivot = n - 1)
            (0.13, vec![PauliTerm::x(0), PauliTerm::x(n - 1)]),
            (0.25, vec![PauliTerm::y(0), PauliTerm::y(n - 1)]),
            (0.39, vec![PauliTerm::z(0), PauliTerm::x(n - 1)]),
            (0.45, vec![PauliTerm::z(0), PauliTerm::y(n - 1)]),
        ];
        for (theta, factors) in terms_list {
            c.add_pauli_rotation(theta, &factors);
        }

        let expanded = expand_pauli_rotations(&c);
        let lowered = state_of(&mut StatevectorBackend::new(SEED), &expanded);

        let mut per_gate = StatevectorBackend::new(SEED);
        per_gate.init(n, 0).unwrap();
        per_gate.apply_instructions(&c.instructions).unwrap();
        for (i, (a, e)) in per_gate.state_vector().iter().zip(&lowered).enumerate() {
            let diff = (a - e).norm();
            assert!(
                diff < 1e-12,
                "{n}q per-gate amplitude {i}: |diff| = {diff:e}"
            );
        }

        let fused_native = state_of(&mut StatevectorBackend::new(SEED), &c);
        for (i, (a, e)) in fused_native.iter().zip(&lowered).enumerate() {
            let diff = (a - e).norm();
            assert!(
                diff < 1e-12,
                "{n}q fused-native amplitude {i}: |diff| = {diff:e}"
            );
        }
    }
}

#[test]
fn z_only_pauli_rot_and_multi_parity_match_ladder() {
    for n in [6usize, 16] {
        let mut c = entangled_pauli_circuit(n);
        c.add_pauli_rotation(0.41, &[PauliTerm::z(0), PauliTerm::z(2)]);
        c.add_pauli_rotation(0.33, &[PauliTerm::z(1), PauliTerm::z(3)]);
        c.add_pauli_rotation(0.27, &[PauliTerm::z(2), PauliTerm::z(4)]);
        c.add_pauli_rotation(0.51, &[PauliTerm::z(3), PauliTerm::z(5)]);
        if n >= 16 {
            c.add_pauli_rotation(0.19, &[PauliTerm::z(12), PauliTerm::z(14)]);
            c.add_pauli_rotation(0.29, &[PauliTerm::z(13), PauliTerm::z(15)]);
        }

        let expanded = expand_pauli_rotations(&c);
        let lowered = state_of(&mut StatevectorBackend::new(SEED), &expanded);

        let mut per_gate = StatevectorBackend::new(SEED);
        per_gate.init(n, 0).unwrap();
        per_gate.apply_instructions(&c.instructions).unwrap();
        for (i, (a, e)) in per_gate.state_vector().iter().zip(&lowered).enumerate() {
            let diff = (a - e).norm();
            assert!(diff < 1e-12, "{n}q z-only amplitude {i}: |diff| = {diff:e}");
        }
    }

    // Back-to-back MultiPauliRot at 16q mixing low X/Y rotations with Z-only strings
    // whose Z letters sit solely outside the tile (`tile_z == 0 && zrest != 0`).
    let n = 16;
    let mut c = entangled_pauli_circuit(n);
    c.add_pauli_rotation(0.31, &[PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)]);
    c.add_pauli_rotation(0.47, &[PauliTerm::z(13), PauliTerm::z(15)]);
    c.add_pauli_rotation(0.23, &[PauliTerm::z(0), PauliTerm::z(1), PauliTerm::z(14)]);
    c.add_pauli_rotation(0.19, &[PauliTerm::x(1), PauliTerm::y(2), PauliTerm::z(15)]);

    let fused = prism_q::circuit::fusion::fuse_circuit(&c, true);
    assert!(
        fused.instructions.iter().any(|inst| matches!(
            inst,
            Instruction::Gate {
                gate: Gate::MultiPauliRot(_),
                ..
            }
        )),
        "expected MultiPauliRot batch in 16q tiled Z-parity test"
    );

    let expanded = expand_pauli_rotations(&c);
    let lowered = state_of(&mut StatevectorBackend::new(SEED), &expanded);
    let native = state_of(&mut StatevectorBackend::new(SEED), &c);
    for (i, (a, e)) in native.iter().zip(&lowered).enumerate() {
        let diff = (a - e).norm();
        assert!(
            diff < 1e-12,
            "16q multi-parity amplitude {i}: |diff| = {diff:e}"
        );
    }
}

#[test]
fn unfused_rzz_and_parity2q_match_reference() {
    use prism_q::backend::factored::FactoredBackend;
    use prism_q::gates::{DiagEntry, DiagonalBatchData};

    let cases: &[(usize, &[(usize, usize)])] = &[
        (
            6,
            &[
                (0, 1),
                (1, 0),
                (0, 3),
                (3, 0),
                (1, 2),
                (2, 1),
                (2, 5),
                (5, 2),
            ],
        ),
        (
            16,
            &[
                (0, 1),
                (1, 0),
                (0, 3),
                (0, 14),
                (14, 0),
                (1, 2),
                (1, 13),
                (13, 1),
                (2, 5),
                (3, 14),
                (13, 15),
                (15, 13),
            ],
        ),
    ];

    for &(n, pairs) in cases {
        let base_circuit = entangled_pauli_circuit(n);
        let mut base_sv = StatevectorBackend::new(SEED);
        base_sv.init(n, 0).unwrap();
        base_sv
            .apply_instructions(&base_circuit.instructions)
            .unwrap();
        let start = base_sv.state_vector().to_vec();

        let theta = 0.63;
        let same_rzz = Complex64::from_polar(1.0, -theta / 2.0);
        let diff_rzz = Complex64::from_polar(1.0, theta / 2.0);

        let mut c_rzz = Circuit::new(n, 0);
        let mut expected_rzz = start.clone();
        for &(q0, q1) in pairs {
            c_rzz.add_gate(Gate::Rzz(theta), &[q0, q1]);
            for (i, amp) in expected_rzz.iter_mut().enumerate() {
                let parity = ((i >> q0) ^ (i >> q1)) & 1;
                *amp *= if parity == 0 { same_rzz } else { diff_rzz };
            }
        }

        let mut sv = StatevectorBackend::new(SEED);
        sv.init_from_amplitudes(start.clone(), 0).unwrap();
        sv.apply_instructions(&c_rzz.instructions).unwrap();
        for (i, (a, e)) in sv.state_vector().iter().zip(&expected_rzz).enumerate() {
            let diff = (a - e).norm();
            assert!(
                diff < 1e-12,
                "{n}q unfused Rzz sv amplitude {i}: |diff| = {diff:e}"
            );
        }

        let mut fb = FactoredBackend::new(SEED);
        fb.init(n, 0).unwrap();
        fb.apply_instructions(&base_circuit.instructions).unwrap();
        fb.apply_instructions(&c_rzz.instructions).unwrap();
        let fb_state = fb.export_statevector().unwrap();
        for (i, (a, e)) in fb_state.iter().zip(&expected_rzz).enumerate() {
            let diff = (a - e).norm();
            assert!(
                diff < 1e-12,
                "{n}q unfused Rzz factored amplitude {i}: |diff| = {diff:e}"
            );
        }

        // Non-conjugate arbitrary `(same, diff)` via DiagEntry::Parity2q on FactoredBackend.
        let same_arb = Complex64::new(0.8, 0.35);
        let diff_arb = Complex64::new(-0.4, 0.7);
        let mut expected_arb = start.clone();
        let mut c_diag = Circuit::new(n, 0);
        for &(q0, q1) in pairs {
            for (i, amp) in expected_arb.iter_mut().enumerate() {
                let parity = ((i >> q0) ^ (i >> q1)) & 1;
                *amp *= if parity == 0 { same_arb } else { diff_arb };
            }

            let mut targets = [q0, q1];
            targets.sort_unstable();
            c_diag.add_gate(
                Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(vec![
                    DiagEntry::Parity2q {
                        q0,
                        q1,
                        same: same_arb,
                        diff: diff_arb,
                    },
                ]))),
                &targets,
            );
        }

        let mut fb_diag = FactoredBackend::new(SEED);
        fb_diag.init(n, 0).unwrap();
        fb_diag
            .apply_instructions(&base_circuit.instructions)
            .unwrap();
        fb_diag.apply_instructions(&c_diag.instructions).unwrap();
        let got_diag = fb_diag.export_statevector().unwrap();
        for (i, (a, e)) in got_diag.iter().zip(&expected_arb).enumerate() {
            let diff = (a - e).norm();
            assert!(
                diff < 1e-12,
                "{n}q Parity2q factored amplitude {i}: |diff| = {diff:e}"
            );
        }
    }
}

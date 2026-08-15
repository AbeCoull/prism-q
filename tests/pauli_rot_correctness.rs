//! Pipeline coverage for the native Pauli rotation: the statevector keeps the
//! native gate through fusion, every declining backend receives the ladder
//! expansion, and both routes agree at parallel-kernel sizes.

mod common;

use common::{MPS_EPS, SEED, SV_EPS, assert_probs_close};
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
#[test]
fn fusion_keeps_the_native_gate() {
    let circuit = trotter_like_circuit(16);
    let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
    let native_count = fused
        .instructions
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                Instruction::Gate {
                    gate: Gate::PauliRot(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        native_count, 5,
        "fusion must pass PauliRot through untouched"
    );
    assert!(StatevectorBackend::new(SEED).supports_pauli_rotation());
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

    let sv = sim::simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .shots(32)
        .unwrap();
    let mps = sim::simulate(&c)
        .backend(BackendKind::Mps { max_bond_dim: 64 })
        .seed(SEED)
        .shots(32)
        .unwrap();
    assert_eq!(sv.shots, mps.shots);
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

#[test]
fn qasm_export_emits_the_lowering() {
    let circuit = trotter_like_circuit(4);
    let qasm = prism_q::circuit::qasm_export::to_qasm3(&circuit).unwrap();
    assert!(!qasm.contains("pauli_rot"));
    assert!(qasm.contains("cx"));
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

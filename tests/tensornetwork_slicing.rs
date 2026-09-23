//! Index slicing at the tensor-network memory ceiling. Isolated in its own
//! test binary: it pins `PRISM_MAX_TN_PEAK_QUBITS` and `PRISM_MAX_TN_SLICES`,
//! which are cached per process, low enough that ordinary circuits cross the
//! cap and have to be sliced.

mod common;

use common::{SEED, caps};
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::{TensorNetworkBackend, last_slice_count};
use prism_q::gates::Gate;
use prism_q::{Circuit, PauliTerm, PrismError, circuits};

/// Peak cap of `2^10` elements against a slice budget of 64, so a 14-qubit
/// ansatz slices under the cap and a 12-qubit register behind one 10-qubit
/// multi-controlled gate cannot.
const PEAK_CAP: usize = 10;
const SLICE_BUDGET: usize = 64;

fn small_caps() {
    caps::set_once(&[
        ("PRISM_MAX_TN_PEAK_QUBITS", "10"),
        ("PRISM_MAX_TN_SLICES", "64"),
    ]);
}

fn loaded(circuit: &Circuit) -> TensorNetworkBackend {
    let mut tn = TensorNetworkBackend::new(SEED);
    tn.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    tn.apply_instructions(&circuit.instructions).unwrap();
    tn
}

fn statevector_expectation(circuit: &Circuit, terms: &[PauliTerm]) -> f64 {
    let mut sv = StatevectorBackend::new(SEED);
    sv.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    sv.apply_instructions(&circuit.instructions).unwrap();
    sv.pauli_expectations(&[terms.to_vec()]).unwrap()[0]
}

#[test]
fn a_sliced_expectation_matches_the_unsliced_one() {
    small_caps();
    let circuit = circuits::hardware_efficient_ansatz(14, 4, SEED);
    let terms = [PauliTerm::z(0), PauliTerm::x(7)];

    let tn = loaded(&circuit);
    let sliced = tn.pauli_expectations(&[terms.to_vec()]).unwrap()[0];
    let slices = last_slice_count();
    assert!(slices > 1, "the fixture contracted whole");
    assert!(
        slices <= SLICE_BUDGET,
        "{slices} slices past the pinned budget"
    );

    let exact = statevector_expectation(&circuit, &terms);
    assert!(
        (sliced - exact).abs() < 1e-10,
        "{sliced} vs {exact} over {slices} slices"
    );
}

#[test]
fn a_sliced_marginal_matches_the_unsliced_one() {
    small_caps();
    let circuit = circuits::hardware_efficient_ansatz(14, 4, SEED);

    let tn = loaded(&circuit);
    let rho = tn.reduced_density_matrix_1q(7).unwrap();
    assert!(last_slice_count() > 1, "the fixture contracted whole");

    let mut sv = StatevectorBackend::new(SEED);
    sv.init(14, 0).unwrap();
    sv.apply_instructions(&circuit.instructions).unwrap();
    let expected = sv.reduced_density_matrix_1q(7).unwrap();
    for (row, expected_row) in rho.iter().zip(&expected) {
        for (entry, want) in row.iter().zip(expected_row) {
            assert!((entry - want).norm() < 1e-10, "{entry} vs {want}");
        }
    }
}

// A 12-qubit register behind one 10-qubit multi-controlled gate: the doubled
// network cannot be brought under the pinned cap inside the slice budget, so
// the cap rejects by name exactly as it did before slicing existed.
#[test]
fn a_contraction_the_budget_cannot_reach_names_the_cap() {
    small_caps();
    let n = 12;
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    let x_mat = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    circuit.add_gate(Gate::mcu(x_mat, 9), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let tn = loaded(&circuit);
    let err = tn.pauli_expectations(&[vec![PauliTerm::z(0)]]).unwrap_err();
    let reason = caps::incompatible_reason(err, "tensornetwork");
    assert!(
        reason.contains("pauli expectation") && reason.contains(&format!("2^{PEAK_CAP}")),
        "{reason}"
    );
}

// The guard rejects before the replay allocates, so an oversize contraction
// that cannot be sliced must not have moved any data first.
#[test]
fn the_rejection_is_an_incompatible_backend_error() {
    small_caps();
    let mut circuit = Circuit::new(12, 0);
    for q in 0..12 {
        circuit.add_gate(Gate::H, &[q]);
    }
    let x_mat = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    circuit.add_gate(Gate::mcu(x_mat, 9), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let tn = loaded(&circuit);
    let err = tn.reduced_density_matrix_1q(0).unwrap_err();
    assert!(
        matches!(err, PrismError::IncompatibleBackend { .. }),
        "{err:?}"
    );
}

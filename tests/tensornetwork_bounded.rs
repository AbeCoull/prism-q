//! Bounded tensor-network contraction: intermediate bonds truncated at a
//! caller-supplied tolerance. Isolated in its own test binary: it pins
//! `PRISM_MAX_TN_PEAK_QUBITS` and `PRISM_MAX_TN_SLICES`, which are cached per
//! process, low enough that the contractions here cross the cap.

mod common;

use common::{SEED, caps};
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::gates::Gate;
use prism_q::sim::Exactness;
use prism_q::{BackendKind, Circuit, PauliTerm, PrismError, simulate};

fn small_caps() {
    caps::set_once(&[
        ("PRISM_MAX_TN_PEAK_QUBITS", "10"),
        ("PRISM_MAX_TN_SLICES", "64"),
    ]);
}

/// A weakly entangling chain: small rotations ahead of each CZ ladder, so
/// the intermediates the contraction builds are close to low rank and the
/// singular spectrum has something for a tolerance to drop.
fn weak_chain(n: usize, layers: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    for layer in 0..layers {
        for q in 0..n {
            circuit.add_gate(Gate::Ry(0.18 + 0.01 * layer as f64), &[q]);
        }
        for q in 0..n - 1 {
            circuit.add_gate(Gate::Cz, &[q, q + 1]);
        }
    }
    circuit
}

fn loaded(circuit: &Circuit, tolerance: f64) -> TensorNetworkBackend {
    let mut tn = TensorNetworkBackend::with_tolerance(SEED, tolerance);
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
fn a_bounded_contraction_stays_inside_the_bound_it_reports() {
    small_caps();
    let circuit = weak_chain(14, 4);
    let terms = [PauliTerm::z(0), PauliTerm::z(7)];

    let tn = loaded(&circuit, 1e-3);
    let approximate = tn.pauli_expectations(&[terms.to_vec()]).unwrap()[0];
    let Exactness::Approximate {
        fidelity_lower_bound,
    } = tn.exactness()
    else {
        panic!("a tolerance must report Approximate");
    };
    let bound = fidelity_lower_bound.expect("the bounded route reports a bound");
    assert!(
        tn.truncation_discarded() > 0.0,
        "the fixture contracted without cutting anything"
    );

    // A 2-norm state error of `delta` moves a Pauli expectation by at most
    // `2 * delta`, and `1 - bound` is the accumulated squared weight.
    let exact = statevector_expectation(&circuit, &terms);
    let allowed = 2.0 * (1.0 - bound).max(0.0).sqrt() + 1e-9;
    assert!(
        (approximate - exact).abs() <= allowed,
        "{approximate} vs {exact}, outside {allowed} at bound {bound}"
    );
}

#[test]
fn a_zero_tolerance_stays_exact() {
    small_caps();
    let circuit = weak_chain(14, 4);
    let terms = [PauliTerm::z(0), PauliTerm::z(7)];

    let tn = loaded(&circuit, 0.0);
    let value = tn.pauli_expectations(&[terms.to_vec()]).unwrap()[0];
    assert_eq!(tn.exactness(), Exactness::Exact);
    assert_eq!(tn.truncation_discarded(), 0.0);

    let exact = statevector_expectation(&circuit, &terms);
    assert!((value - exact).abs() < 1e-12, "{value} vs {exact}");
}

#[test]
fn require_exact_rejects_a_bounded_tensor_network() {
    small_caps();
    let circuit = weak_chain(6, 2);
    let err = simulate(&circuit)
        .backend(BackendKind::TensorNetworkBounded { tolerance: 1e-3 })
        .require_exact()
        .seed(SEED)
        .run()
        .unwrap_err();
    assert!(format!("{err}").contains("TensorNetworkBounded"), "{err}");

    simulate(&circuit)
        .backend(BackendKind::TensorNetwork)
        .require_exact()
        .seed(SEED)
        .run()
        .expect("the exact route passes the same gate");
}

// A tolerance the truncation rule cannot read as a fraction would keep rank 1
// and discard nearly everything, so it is refused rather than served. Zero is
// refused too: the bounded kind is always approximate, and the exact
// contraction has its own kind.
#[test]
fn an_unusable_tolerance_is_refused() {
    small_caps();
    let circuit = weak_chain(6, 2);
    for bad in [-1e-3, 0.0, f64::NAN, f64::INFINITY] {
        let err = simulate(&circuit)
            .backend(BackendKind::TensorNetworkBounded { tolerance: bad })
            .seed(SEED)
            .run()
            .unwrap_err();
        assert!(
            matches!(err, PrismError::InvalidParameter { .. }),
            "{err:?}"
        );
    }
}

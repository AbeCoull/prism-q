//! Bounded tensor-network contraction: intermediate bonds truncated at a
//! caller-supplied tolerance.

use crate::common::SEED;
use crate::small_caps;
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::gates::Gate;
use prism_q::sim::{Exactness, RunMetadata};
use prism_q::{BackendKind, Circuit, Limits, PauliTerm, PrismError, ShotsResult, simulate};

const BOUNDED: BackendKind = BackendKind::TensorNetworkBounded { tolerance: 1e-3 };

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

// A truncating pair read is Hermitian and trace one, sits within the
// reported bound of the exact pair, and the terminal reports the bound the
// query's own contraction left rather than the one before it.
#[test]
fn a_bounded_pair_density_is_hermitian_and_inside_its_bound() {
    small_caps();
    let circuit = weak_chain(14, 4);
    let subsystem = [3usize, 10];

    let mut tn = loaded(&circuit, 1e-3);
    let rho = tn.reduced_density_matrix(&subsystem).unwrap();
    assert!(
        tn.truncation_discarded() > 0.0,
        "the fixture contracted without cutting anything"
    );
    let trace: Complex64 = (0..4).map(|t| rho[t * 4 + t]).sum();
    assert!((trace - 1.0).norm() < 1e-12, "trace {trace}");
    for t in 0..4 {
        for tp in 0..4 {
            assert_eq!(rho[t * 4 + tp], rho[tp * 4 + t].conj(), "({t}, {tp})");
        }
    }

    let Exactness::Approximate {
        fidelity_lower_bound: Some(bound),
    } = tn.exactness()
    else {
        panic!("a tolerance must report Approximate with a bound");
    };
    let mut sv = StatevectorBackend::new(SEED);
    sv.init(circuit.num_qubits, 0).unwrap();
    sv.apply_instructions(&circuit.instructions).unwrap();
    let exact = sv.reduced_density_matrix(&subsystem).unwrap();
    let allowed = 2.0 * (1.0 - bound).max(0.0).sqrt() + 1e-9;
    for (entry, want) in rho.iter().zip(&exact) {
        assert!(
            (entry - want).norm() <= allowed,
            "{entry} vs {want}, outside {allowed} at bound {bound}"
        );
    }

    let result = simulate(&circuit)
        .backend(BackendKind::TensorNetworkBounded { tolerance: 1e-3 })
        .seed(SEED)
        .reduced_density_matrix(&subsystem)
        .unwrap();
    let reported = result
        .metadata
        .fidelity_lower_bound()
        .expect("the bounded route reports a bound");
    assert!(reported < 1.0, "the terminal read the bound before the cut");
}

// Below 14 so the sampler takes the per-shot sweep rather than the dense arm.
fn sweep_limits() -> Limits {
    Limits::new().max_probability_qubits(12)
}

fn measured_chain() -> Circuit {
    let mut circuit = weak_chain(14, 4);
    circuit.measure_all();
    circuit
}

fn bounded_shots(num_shots: usize) -> ShotsResult {
    simulate(&measured_chain())
        .backend(BOUNDED)
        .limits(sweep_limits())
        .seed(SEED)
        .shots(num_shots)
        .unwrap()
}

fn bound_of(metadata: &RunMetadata) -> f64 {
    metadata
        .fidelity_lower_bound()
        .expect("the bounded route reports a bound")
}

// Every contraction of the sweep runs inside the sampling call, so a bound
// read before it reports the untouched 1.0. Each shot is its own answer, so
// the loss does not accumulate across shots, though a later shot whose
// outcomes cut more can still lower the bound.
#[test]
fn bounded_shots_report_the_worst_shot_the_sweep_cut() {
    small_caps();
    let eight = bounded_shots(8);
    let sixty_four = bounded_shots(64);
    assert_eq!(
        eight.shots[..],
        sixty_four.shots[..8],
        "the first 8 shots of a longer run are no longer the 8-shot run"
    );

    let (eight, sixty_four) = (bound_of(&eight.metadata), bound_of(&sixty_four.metadata));
    assert!(eight < 1.0, "the terminal read the bound before the sweep");
    assert!(
        sixty_four <= eight,
        "8 shots {eight} vs 64 shots {sixty_four}"
    );
    // A sum over shots would carry the loss to about 8 times the 8-shot loss.
    assert!(
        1.0 - sixty_four < 2.0 * (1.0 - eight),
        "8 shots {eight} vs 64 shots {sixty_four}"
    );
}

#[test]
fn bounded_counts_report_the_cut_the_sweep_made() {
    small_caps();
    let counts = simulate(&measured_chain())
        .backend(BOUNDED)
        .limits(sweep_limits())
        .seed(SEED)
        .sample_counts(8)
        .unwrap();
    let bound = bound_of(&counts.metadata);
    assert!(bound < 1.0, "the terminal read the bound before the sweep");
    let shots = bound_of(&bounded_shots(8).metadata);
    assert!(
        (bound - shots).abs() < 1e-12,
        "counts {bound} vs shots {shots}"
    );
}

// Each observable truncates its own copy of the network, so listing one
// observable eight times answers the same query eight times. Twelve qubits,
// since the fused 14-qubit chain contracts under the cap without a cut.
#[test]
fn a_repeated_observable_keeps_the_bound_of_one() {
    small_caps();
    let circuit = weak_chain(12, 4);
    let observable = vec![PauliTerm::z(0), PauliTerm::z(6)];
    let bound = |observables: &[Vec<PauliTerm>]| {
        simulate(&circuit)
            .backend(BOUNDED)
            .seed(SEED)
            .expectation_values_reported(observables)
            .unwrap()
            .metadata
            .fidelity_lower_bound()
            .expect("the bounded route reports a bound")
    };
    let once = bound(std::slice::from_ref(&observable));
    assert!(
        once < 1.0,
        "the fixture contracted without cutting anything"
    );
    let repeated = bound(&vec![observable; 8]);
    assert!(
        (once - repeated).abs() < 1e-12,
        "once {once} vs eight {repeated}"
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

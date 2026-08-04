//! Start states other than |0...0>: the builder entry point, the validation it
//! applies to a caller-supplied amplitude vector, and the backends and
//! terminals that decline one.

mod common;

use common::SEED;
use num_complex::Complex64;
use prism_q::gates::Gate;
use prism_q::{
    BackendKind, Circuit, NoiseModel, ParamLink, ParameterMap, PauliTerm, PrismError, simulate,
};

const EPS: f64 = 1e-12;

/// cos(pi/8)|0> + sin(pi/8)|1>. Normalized, not a basis state, and not a
/// stabilizer state, so any route that assumed |0...0> answers differently.
fn tilted() -> Vec<Complex64> {
    let theta = std::f64::consts::FRAC_PI_8;
    vec![
        Complex64::new(theta.cos(), 0.0),
        Complex64::new(theta.sin(), 0.0),
    ]
}

/// cos(pi/8)^2 = (1 + cos(pi/4))/2 and sin(pi/8)^2 = (1 - cos(pi/4))/2.
fn tilted_probs() -> (f64, f64) {
    let a = std::f64::consts::FRAC_1_SQRT_2;
    ((1.0 + a) / 2.0, (1.0 - a) / 2.0)
}

fn assert_close(got: f64, want: f64, label: &str) {
    assert!((got - want).abs() < EPS, "{label}: got {got}, want {want}");
}

#[test]
fn run_evolves_the_given_start_state() {
    // H (c|0> + s|1>) = ((c+s)|0> + (c-s)|1>)/sqrt(2), and (c+s)^2/2 =
    // (1 + 2cs)/2 = (1 + sin(pi/4))/2.
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);

    let probs = simulate(&circuit)
        .initial_state(&tilted())
        .seed(SEED)
        .run()
        .expect("statevector accepts a start state")
        .probabilities
        .expect("dense probabilities")
        .to_vec();

    let (hi, lo) = tilted_probs();
    assert_close(probs[0], hi, "p(0)");
    assert_close(probs[1], lo, "p(1)");
}

// Nothing in this circuit is outside the Clifford group, so structure-based
// routing sends it to the tableau, which can only start from |0...0>. The
// start state constrains the route to the statevector instead; a tableau
// answer would be [1, 0, 0, 0].
#[test]
fn a_clifford_circuit_with_a_start_state_does_not_route_to_the_tableau() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::Cx, &[0, 1]);

    let mut state = vec![Complex64::new(0.0, 0.0); 4];
    state[0] = tilted()[0];
    state[1] = tilted()[1];

    let probs = simulate(&circuit)
        .initial_state(&state)
        .seed(SEED)
        .run()
        .expect("auto routes a start state to the statevector")
        .probabilities
        .expect("dense probabilities")
        .to_vec();

    let (hi, lo) = tilted_probs();
    assert_close(probs[0], hi, "p(00)");
    assert_close(probs[1], 0.0, "p(01)");
    assert_close(probs[2], 0.0, "p(10)");
    assert_close(probs[3], lo, "p(11)");
}

#[test]
fn a_start_state_of_the_wrong_length_is_rejected() {
    let circuit = Circuit::new(2, 0);
    let err = simulate(&circuit)
        .initial_state(&tilted())
        .seed(SEED)
        .run()
        .expect_err("a 2-amplitude state cannot start a 2-qubit circuit");

    match err {
        PrismError::InvalidParameter { message } => {
            assert!(message.contains('4') && message.contains('2'), "{message}");
        }
        other => panic!("expected InvalidParameter, got {other:?}"),
    }
}

#[test]
fn an_unnormalized_start_state_is_rejected() {
    let circuit = Circuit::new(1, 0);
    let state = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
    let err = simulate(&circuit)
        .initial_state(&state)
        .seed(SEED)
        .run()
        .expect_err("norm 2 is not renormalized silently");

    match err {
        PrismError::InvalidParameter { message } => {
            assert!(message.contains("norm"), "{message}");
        }
        other => panic!("expected InvalidParameter, got {other:?}"),
    }
}

#[test]
fn a_non_finite_start_state_is_rejected() {
    let circuit = Circuit::new(1, 0);
    let state = vec![Complex64::new(f64::NAN, 0.0), Complex64::new(0.0, 0.0)];
    let err = simulate(&circuit)
        .initial_state(&state)
        .seed(SEED)
        .run()
        .expect_err("NaN amplitudes are rejected before the norm check");

    assert!(
        matches!(err, PrismError::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

#[test]
fn backends_without_a_start_state_name_themselves() {
    let circuit = Circuit::new(1, 0);
    for (kind, name) in [
        (BackendKind::Stabilizer, "Stabilizer"),
        (BackendKind::Sparse, "Sparse"),
        (BackendKind::ProductState, "ProductState"),
        (BackendKind::Mps { max_bond_dim: 64 }, "Mps"),
        (BackendKind::Factored, "Factored"),
    ] {
        let err = simulate(&circuit)
            .backend(kind)
            .initial_state(&tilted())
            .seed(SEED)
            .run()
            .expect_err("only the statevector and density matrix take a start state");

        match err {
            PrismError::IncompatibleBackend { backend, reason } => {
                assert!(backend.contains(name), "{backend}");
                assert!(reason.contains("statevector"), "{reason}");
            }
            other => panic!("expected IncompatibleBackend for {name}, got {other:?}"),
        }
    }
}

#[test]
fn the_density_matrix_accepts_a_start_state() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);

    let probs = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .initial_state(&tilted())
        .seed(SEED)
        .run()
        .expect("the outer product of a pure start state fits the cap")
        .probabilities
        .expect("dense probabilities")
        .to_vec();

    let (hi, lo) = tilted_probs();
    assert_close(probs[0], hi, "p(0)");
    assert_close(probs[1], lo, "p(1)");
}

// X takes the populations to (s^2, c^2), then depolarizing with total
// probability p flips the bit with probability 2p/3: p(0) = (1 - 2p/3) s^2 +
// (2p/3) c^2 = 1/2 - (1 - 4p/3) cos(pi/4) / 2, which is 1/2 - 0.3/sqrt(2) at
// p = 0.3.
#[test]
fn a_start_state_evolves_under_the_exact_mixture() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::X, &[0]);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.3);

    let probs = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .initial_state(&tilted())
        .seed(SEED)
        .run()
        .expect("the noisy density-matrix route takes a start state")
        .probabilities
        .expect("dense probabilities")
        .to_vec();

    let a = std::f64::consts::FRAC_1_SQRT_2;
    assert_close(probs[0], 0.5 - 0.3 * a, "p(0)");
    assert_close(probs[1], 0.5 + 0.3 * a, "p(1)");
}

#[test]
fn marginals_read_the_start_state() {
    let circuit = Circuit::new(1, 0);
    let marginals = simulate(&circuit)
        .initial_state(&tilted())
        .seed(SEED)
        .marginals()
        .expect("marginals accept a start state")
        .into_vec();

    let (hi, lo) = tilted_probs();
    assert_close(marginals[0].0, hi, "p(0)");
    assert_close(marginals[0].1, lo, "p(1)");
}

// <Z> = c^2 - s^2 = cos(pi/4) and <X> = 2cs = sin(pi/4) on the same state.
#[test]
fn expectation_values_read_the_start_state() {
    let circuit = Circuit::new(1, 0);
    let observables = vec![vec![PauliTerm::z(0)], vec![PauliTerm::x(0)]];

    let values = simulate(&circuit)
        .initial_state(&tilted())
        .seed(SEED)
        .expectation_values(&observables)
        .expect("expectation values accept a start state");

    let a = std::f64::consts::FRAC_1_SQRT_2;
    assert_close(values[0], a, "<Z>");
    assert_close(values[1], a, "<X>");

    let dm = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .initial_state(&tilted())
        .seed(SEED)
        .expectation_values(&observables)
        .expect("the density matrix answers Tr(rho P) from a start state");
    assert_close(dm[0], a, "dm <Z>");
    assert_close(dm[1], a, "dm <X>");
}

// |1> as an amplitude vector: every shot reads back 1, so the check needs no
// tolerance and no reference distribution.
#[test]
fn terminal_shots_sample_from_the_start_state() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_measure(0, 0);
    let state = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

    let shots = simulate(&circuit)
        .initial_state(&state)
        .seed(SEED)
        .shots(16)
        .expect("shots accept a start state");
    assert_eq!(shots.shots.len(), 16);
    assert!(shots.shots.iter().all(|shot| shot[0]));

    let counts = simulate(&circuit)
        .initial_state(&state)
        .seed(SEED)
        .sample_counts(16)
        .expect("counts accept a start state")
        .into_counts();
    assert_eq!(counts.len(), 1);
    assert_eq!(counts.get(&vec![1u64]).copied(), Some(16));
}

// A gate after the measurement takes the circuit off the terminal path, so
// every shot replays the start state instead of sampling one distribution.
#[test]
fn mid_circuit_shots_replay_the_start_state() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::H, &[0]);
    let state = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

    let shots = simulate(&circuit)
        .initial_state(&state)
        .seed(SEED)
        .shots(16)
        .expect("per-shot replay accepts a start state");
    assert_eq!(shots.shots.len(), 16);
    assert!(shots.shots.iter().all(|shot| shot[0]));
}

#[test]
fn noisy_shots_decline_a_start_state() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.1);

    let err = simulate(&circuit)
        .noise(&noise)
        .initial_state(&tilted())
        .seed(SEED)
        .shots(4)
        .expect_err("trajectory replay has no start-state path");
    assert!(
        matches!(err, PrismError::IncompatibleBackend { .. }),
        "expected IncompatibleBackend, got {err:?}"
    );
}

#[test]
fn the_adjoint_gradient_declines_a_start_state() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::Rz(0.3), &[0]);
    let params = ParameterMap::from_links(vec![ParamLink {
        instruction: 0,
        param: 0,
    }]);

    let err = simulate(&circuit)
        .initial_state(&tilted())
        .seed(SEED)
        .expectation_gradient(&[(1.0, vec![PauliTerm::z(0)])], &params)
        .expect_err("the adjoint pass initializes its own state");
    assert!(
        matches!(err, PrismError::IncompatibleBackend { .. }),
        "expected IncompatibleBackend, got {err:?}"
    );
}

//! The `observable_variance` terminal: the spread of the operator itself,
//! which is not the grouped-measurement variance beside it.

mod common;

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

use prism_q::sim::noise::{NoiseEvent, NoiseModel};
use prism_q::{BackendKind, Circuit, CircuitBuilder, PauliObservable, PauliTerm, simulate};

use common::{SEED, SV_EPS};

type Terms = Vec<(f64, Vec<PauliTerm>)>;

fn observable(terms: &[(f64, Vec<PauliTerm>)]) -> PauliObservable {
    PauliObservable::from_terms(terms.iter().cloned()).unwrap()
}

fn variance_of(circuit: &Circuit, terms: &[(f64, Vec<PauliTerm>)]) -> f64 {
    simulate(circuit)
        .seed(SEED)
        .observable_variance(&observable(terms))
        .unwrap()
        .variance
}

fn rotated(theta: f64) -> Circuit {
    let mut builder = CircuitBuilder::new(1);
    builder.ry(theta, 0);
    builder.build()
}

// Hand algebra on one qubit, each case reachable without a simulator:
// on |+> the Z expectation is 0 and Z squares to the identity, so Var is 1;
// on |0> the Hadamard observable (X + Z)/sqrt(2) has mean 1/sqrt(2) and
// second moment 1, so Var is 1/2.
#[test]
fn single_qubit_variances_match_hand_algebra() {
    let hadamard = [
        (FRAC_1_SQRT_2, vec![PauliTerm::x(0)]),
        (FRAC_1_SQRT_2, vec![PauliTerm::z(0)]),
    ];
    let cases: [(&str, Circuit, Terms, f64); 5] = [
        (
            "<Z> on |0>",
            rotated(0.0),
            vec![(1.0, vec![PauliTerm::z(0)])],
            0.0,
        ),
        (
            "<Z> on |+>",
            {
                let mut builder = CircuitBuilder::new(1);
                builder.h(0);
                builder.build()
            },
            vec![(1.0, vec![PauliTerm::z(0)])],
            1.0,
        ),
        ("<H> on |0>", rotated(0.0), hadamard.to_vec(), 0.5),
        (
            "<H> on its own eigenstate",
            rotated(FRAC_PI_4),
            hadamard.to_vec(),
            0.0,
        ),
        (
            "a constant has no spread",
            rotated(0.9),
            vec![(2.5, Vec::new())],
            0.0,
        ),
    ];
    for (name, circuit, terms, expected) in cases {
        let actual = variance_of(&circuit, &terms);
        assert!(
            (actual - expected).abs() < SV_EPS,
            "{name}: got {actual}, expected {expected}"
        );
    }
}

// A tensor product of Paulis squares to the identity, so its variance is
// `1 - <P>^2` whatever the state. This holds across the whole corpus and is the
// cheapest statement the terminal owes.
#[test]
fn a_pauli_string_reads_one_minus_the_squared_mean() {
    let mut builder = CircuitBuilder::new(3);
    builder.h(0).cx(0, 1).t(1).ry(0.7, 2).cx(1, 2);
    let circuit = builder.build();
    for string in [
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(1)],
        vec![PauliTerm::y(2)],
        vec![PauliTerm::z(0), PauliTerm::z(1)],
        vec![PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)],
    ] {
        let result = simulate(&circuit)
            .seed(SEED)
            .observable_variance(&observable(&[(1.0, string.clone())]))
            .unwrap();
        let expected = 1.0 - result.mean * result.mean;
        assert!(
            (result.variance - expected).abs() < SV_EPS,
            "{string:?}: got {}, expected {expected}",
            result.variance
        );
    }
}

// The operator variance is not the grouped-measurement variance beside it. On
// the +1 eigenstate of (X + Z)/sqrt(2) the operator has no spread at all, while
// measuring X and Z in separate groups reads 1/2 from each: the grouped number
// drops the covariance between groups, which is the whole difference.
#[test]
fn the_operator_variance_is_not_the_grouped_variance() {
    let circuit = rotated(FRAC_PI_4);
    let terms = [(1.0, vec![PauliTerm::x(0)]), (1.0, vec![PauliTerm::z(0)])];
    let sum = observable(&terms);
    assert_eq!(sum.num_groups(), 2, "X and Z do not qubit-wise commute");

    let operator = simulate(&circuit)
        .seed(SEED)
        .observable_variance(&sum)
        .unwrap();
    assert!(operator.variance.abs() < SV_EPS, "{}", operator.variance);

    let grouped = simulate(&circuit)
        .seed(SEED)
        .observable_expectation(&sum)
        .unwrap();
    assert!((grouped.variance.expect("a grouped variance") - 1.0).abs() < SV_EPS);
    assert!((grouped.mean - operator.mean).abs() < SV_EPS, "same mean");
}

#[test]
fn every_backend_that_answers_agrees() {
    let mut builder = CircuitBuilder::new(4);
    builder.h(0).cx(0, 1).t(1).ry(0.4, 2).cx(2, 3);
    let circuit = builder.build();
    let terms = [
        (0.8, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (-0.3, vec![PauliTerm::x(2)]),
        (0.5, vec![PauliTerm::y(1), PauliTerm::y(3)]),
        (1.2, Vec::new()),
    ];
    let reference = simulate(&circuit)
        .seed(SEED)
        .backend(BackendKind::Statevector)
        .observable_variance(&observable(&terms))
        .unwrap()
        .variance;
    for backend in [
        BackendKind::Auto,
        BackendKind::Sparse,
        BackendKind::TensorNetwork,
        BackendKind::Mps { max_bond_dim: 32 },
    ] {
        let actual = simulate(&circuit)
            .seed(SEED)
            .backend(backend.clone())
            .observable_variance(&observable(&terms))
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"))
            .variance;
        assert!(
            (actual - reference).abs() < SV_EPS,
            "{backend:?}: got {actual}, expected {reference}"
        );
    }
}

// Under noise the variance is read from the exact mixture. Dephasing one half
// of a Bell pair leaves <ZZ> alone and so leaves Var(ZZ) at zero, while it
// drives <XX> to zero and Var(XX) to one.
#[test]
fn a_noise_model_gives_the_exact_mixture() {
    let mut builder = CircuitBuilder::new(2);
    builder.h(0).cx(0, 1);
    let circuit = builder.build();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[1].push(NoiseEvent::pauli(1, 0.0, 0.0, 0.5));

    for (string, expected) in [
        (vec![PauliTerm::z(0), PauliTerm::z(1)], 0.0),
        (vec![PauliTerm::x(0), PauliTerm::x(1)], 1.0),
    ] {
        let actual = simulate(&circuit)
            .seed(SEED)
            .backend(BackendKind::DensityMatrix)
            .noise(&noise)
            .observable_variance(&observable(&[(1.0, string.clone())]))
            .unwrap()
            .variance;
        assert!(
            (actual - expected).abs() < SV_EPS,
            "{string:?}: got {actual}, expected {expected}"
        );
    }
}

// The stabilizer tableau shares no arithmetic with the amplitude backends, so a
// Clifford case anchors the terminal independently.
#[test]
fn the_stabilizer_answers_a_clifford_circuit() {
    let mut builder = CircuitBuilder::new(2);
    builder.h(0).cx(0, 1);
    let circuit = builder.build();
    let stabilized = observable(&[(1.0, vec![PauliTerm::z(0), PauliTerm::z(1)])]);
    let anticommuting = observable(&[(1.0, vec![PauliTerm::z(0)])]);
    for (name, sum, expected) in [
        ("ZZ stabilizes the pair", stabilized, 0.0),
        ("Z alone is undetermined", anticommuting, 1.0),
    ] {
        let actual = simulate(&circuit)
            .seed(SEED)
            .backend(BackendKind::Stabilizer)
            .observable_variance(&sum)
            .unwrap()
            .variance;
        assert!(
            (actual - expected).abs() < SV_EPS,
            "{name}: got {actual}, expected {expected}"
        );
    }
}

// A constant offset moves the mean and leaves the spread alone, which catches
// an identity term leaking into the second moment.
#[test]
fn a_constant_offset_moves_the_mean_and_not_the_variance() {
    let mut builder = CircuitBuilder::new(2);
    builder.h(0).cx(0, 1).t(1);
    let circuit = builder.build();
    let base = [
        (0.8, vec![PauliTerm::z(0)]),
        (-0.4, vec![PauliTerm::x(0), PauliTerm::x(1)]),
    ];
    let plain = simulate(&circuit)
        .seed(SEED)
        .observable_variance(&observable(&base))
        .unwrap();

    // A mild offset only checks the algebra. A large one checks that the
    // constant is held out of the square rather than cancelled inside it: at
    // 1e6 the second moment and the squared mean agree to every digit the
    // variance was meant to carry.
    let mut shifted = base.to_vec();
    shifted.push((1.0e6, Vec::new()));
    let offset = simulate(&circuit)
        .seed(SEED)
        .observable_variance(&observable(&shifted))
        .unwrap();

    assert!((offset.mean - (plain.mean + 1.0e6)).abs() < 1e-6);
    assert!(
        (offset.variance - plain.variance).abs() < SV_EPS,
        "{} against {}",
        offset.variance,
        plain.variance
    );
}

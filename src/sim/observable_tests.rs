use num_complex::Complex64;

use super::{PauliObservable, weighted_group_moments};
use crate::sim::unified_pauli::PauliTerm;

fn qubit_wise_commutes(a: &[PauliTerm], b: &[PauliTerm]) -> bool {
    a.iter().all(|ta| {
        b.iter()
            .all(|tb| ta.qubit != tb.qubit || ta.axis == tb.axis)
    })
}

fn assert_grouping_valid(observable: &PauliObservable) {
    let terms = observable.terms();
    let grouping = observable.grouping();
    let mut seen = vec![false; terms.len()];
    for group in &grouping.groups {
        for (i, &a) in group.term_indices.iter().enumerate() {
            assert!(!seen[a], "term {a} appears in two groups");
            seen[a] = true;
            for &b in &group.term_indices[i + 1..] {
                assert!(
                    qubit_wise_commutes(&terms[a].1, &terms[b].1),
                    "terms {a} and {b} share a group but do not qubit-wise commute"
                );
            }
        }
    }
    for (i, covered) in seen.iter().enumerate() {
        assert_eq!(
            *covered,
            !terms[i].1.is_empty(),
            "non-identity terms are covered exactly; identity terms never"
        );
    }
}

#[test]
fn add_term_merges_identical_strings_regardless_of_factor_order() {
    let mut obs = PauliObservable::new();
    obs.add_term(0.5, vec![PauliTerm::z(1), PauliTerm::x(0)])
        .unwrap();
    obs.add_term(0.25, vec![PauliTerm::x(0), PauliTerm::z(1)])
        .unwrap();
    assert_eq!(obs.num_terms(), 1);
    let (coefficient, factors) = &obs.terms()[0];
    assert_eq!(*coefficient, 0.75);
    assert_eq!(factors, &vec![PauliTerm::x(0), PauliTerm::z(1)]);
}

#[test]
fn add_term_rejects_duplicate_qubit_and_non_finite_coefficient() {
    let mut obs = PauliObservable::new();
    assert!(
        obs.add_term(1.0, vec![PauliTerm::x(2), PauliTerm::z(2)])
            .is_err()
    );
    assert!(obs.add_term(f64::NAN, vec![PauliTerm::z(0)]).is_err());
    assert_eq!(obs.num_terms(), 0);
}

#[test]
fn identity_term_is_allowed_and_ungrouped() {
    let obs = PauliObservable::from_terms([(2.5, vec![]), (1.0, vec![PauliTerm::z(0)])]).unwrap();
    assert_eq!(obs.num_terms(), 2);
    assert_eq!(obs.num_groups(), 1);
    assert_grouping_valid(&obs);
}

#[test]
fn arithmetic_composes() {
    let a = PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();
    let b =
        PauliObservable::from_terms([(2.0, vec![PauliTerm::z(0)]), (3.0, vec![PauliTerm::x(1)])])
            .unwrap();
    let sum = (a + b) * 2.0;
    assert_eq!(sum.num_terms(), 2);
    let coefficients: Vec<f64> = sum.terms().iter().map(|(c, _)| *c).collect();
    assert_eq!(coefficients, vec![6.0, 6.0]);

    let x1 = PauliObservable::from_terms([(3.0, vec![PauliTerm::x(1)])]).unwrap();
    let diff = sum - x1 * 2.0;
    let coefficients: Vec<f64> = diff.terms().iter().map(|(c, _)| *c).collect();
    assert_eq!(coefficients, vec![6.0, 0.0]);
}

#[test]
fn qwc_terms_share_a_group_and_anticommuting_terms_split() {
    let obs = PauliObservable::from_terms([
        (1.0, vec![PauliTerm::z(0)]),
        (1.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
    ])
    .unwrap();
    assert_eq!(obs.num_groups(), 1);

    let obs =
        PauliObservable::from_terms([(1.0, vec![PauliTerm::x(0)]), (1.0, vec![PauliTerm::z(0)])])
            .unwrap();
    assert_eq!(obs.num_groups(), 2);

    let obs = PauliObservable::from_terms([
        (1.0, vec![PauliTerm::x(0), PauliTerm::x(1)]),
        (1.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (1.0, vec![PauliTerm::y(0), PauliTerm::y(1)]),
    ])
    .unwrap();
    assert_eq!(obs.num_groups(), 3);
    assert_grouping_valid(&obs);
}

#[test]
fn grouping_is_valid_and_deterministic_on_a_jordan_wigner_hamiltonian() {
    let terms = crate::circuits::jordan_wigner_hamiltonian(8, 400, 42);
    assert!(terms.len() > 100);
    let obs = PauliObservable::from_terms(terms.clone()).unwrap();
    assert_grouping_valid(&obs);
    assert!(obs.num_groups() < obs.num_terms() / 2);

    let again = PauliObservable::from_terms(terms).unwrap();
    let shape: Vec<Vec<usize>> = obs
        .grouping()
        .groups
        .iter()
        .map(|g| g.term_indices.clone())
        .collect();
    let shape_again: Vec<Vec<usize>> = again
        .grouping()
        .groups
        .iter()
        .map(|g| g.term_indices.clone())
        .collect();
    assert_eq!(shape, shape_again);
}

#[test]
fn mutation_invalidates_the_cached_grouping() {
    let mut obs = PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();
    assert_eq!(obs.num_groups(), 1);
    obs.add_term(1.0, vec![PauliTerm::x(0)]).unwrap();
    assert_eq!(obs.num_groups(), 2);
}

#[test]
fn weighted_group_moments_matches_hand_values() {
    // |psi> = (|00> + |11>)/sqrt(2); H_g = Z0 + Z0 Z1 has h(00) = 2, h(11) = 0.
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let state = [
        Complex64::new(s, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(s, 0.0),
    ];
    let (m1, m2) = weighted_group_moments(&state, &[0b01, 0b11], &[1.0, 1.0], 1.0);
    assert!((m1 - 1.0).abs() < 1e-12);
    assert!((m2 - 2.0).abs() < 1e-12);
}

#[test]
fn jordan_wigner_fixture_pool_exceeds_the_bench_budget() {
    for n in [16usize, 20] {
        let pool = crate::circuits::jordan_wigner_hamiltonian(n, usize::MAX, 42);
        assert!(pool.len() > 2500, "pool at n={n} is {}", pool.len());
        let truncated = crate::circuits::jordan_wigner_hamiltonian(n, 2000, 42);
        assert_eq!(truncated.len(), 2000);
    }
}

use num_complex::Complex64;

use super::{PauliObservable, weighted_group_moments};
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

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

/// Dense matrix of a Pauli sum over `n` qubits, qubit 0 the lowest bit. Each
/// Pauli string holds one non-zero per row, so the term walks rows and picks
/// the column that string pairs each with.
fn dense(observable: &PauliObservable, n: usize) -> Vec<Vec<Complex64>> {
    let dim = 1usize << n;
    let mut matrix = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    for (coefficient, string) in observable.terms() {
        for (row, entries) in matrix.iter_mut().enumerate() {
            let mut column = row;
            let mut value = Complex64::new(*coefficient, 0.0);
            for factor in string {
                let bit = row >> factor.qubit & 1;
                match factor.axis {
                    PauliAxis::X => column ^= 1 << factor.qubit,
                    PauliAxis::Y => {
                        column ^= 1 << factor.qubit;
                        value *= Complex64::new(0.0, if bit == 0 { -1.0 } else { 1.0 });
                    }
                    PauliAxis::Z if bit == 1 => value = -value,
                    PauliAxis::Z => {}
                }
            }
            entries[column] += value;
        }
    }
    matrix
}

type Terms = Vec<(f64, Vec<PauliTerm>)>;

fn multiply(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    a.iter()
        .map(|row| {
            (0..b.len())
                .map(|column| {
                    row.iter()
                        .enumerate()
                        .map(|(k, entry)| entry * b[k][column])
                        .sum()
                })
                .collect()
        })
        .collect()
}

// The square is anchored against matrix multiplication rather than against a
// second Pauli-sum routine, so a sign or phase convention has somewhere to fail.
#[test]
fn square_matches_the_matrix_product() {
    let cases: [(&str, usize, Terms); 6] = [
        ("z", 1, vec![(1.0, vec![PauliTerm::z(0)])]),
        (
            "hadamard",
            1,
            vec![
                (std::f64::consts::FRAC_1_SQRT_2, vec![PauliTerm::x(0)]),
                (std::f64::consts::FRAC_1_SQRT_2, vec![PauliTerm::z(0)]),
            ],
        ),
        (
            "anticommuting pair",
            1,
            vec![
                (0.75, vec![PauliTerm::x(0)]),
                (-0.25, vec![PauliTerm::y(0)]),
            ],
        ),
        (
            "with an identity offset",
            1,
            vec![(2.0, Vec::new()), (0.5, vec![PauliTerm::z(0)])],
        ),
        (
            "two qubit mix",
            2,
            vec![
                (0.3, vec![PauliTerm::x(0), PauliTerm::y(1)]),
                (-1.1, vec![PauliTerm::z(0)]),
                (0.7, vec![PauliTerm::y(0), PauliTerm::z(1)]),
            ],
        ),
        (
            "three qubit chain",
            3,
            vec![
                (1.0, vec![PauliTerm::x(0), PauliTerm::x(1)]),
                (1.0, vec![PauliTerm::y(1), PauliTerm::y(2)]),
                (-0.5, vec![PauliTerm::z(0), PauliTerm::z(2)]),
            ],
        ),
    ];
    for (name, n, terms) in cases {
        let observable = PauliObservable::from_terms(terms).unwrap();
        let expected = multiply(&dense(&observable, n), &dense(&observable, n));
        let actual = dense(&observable.square(), n);
        for (row, (want, got)) in expected.iter().zip(&actual).enumerate() {
            for (column, (want, got)) in want.iter().zip(got).enumerate() {
                assert!(
                    (want - got).norm() < 1e-12,
                    "{name}: ({row}, {column}) is {got}, expected {want}"
                );
            }
        }
    }
}

// A Pauli string squares to the identity, so the square of a single-term
// observable is its coefficient squared on the empty string.
#[test]
fn square_of_a_pauli_string_is_the_identity() {
    let observable =
        PauliObservable::from_terms([(1.0, vec![PauliTerm::x(0), PauliTerm::y(1)])]).unwrap();
    let squared = observable.square();
    assert_eq!(squared.terms().len(), 1);
    assert!(
        squared.terms()[0].1.is_empty(),
        "expected the identity string"
    );
    assert!((squared.terms()[0].0 - 1.0).abs() < 1e-12);
}

fn random_state(n: usize, seed: u64) -> Vec<Complex64> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..1usize << n)
        .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect()
}

// Sizes run below one Walsh-Hadamard block, at one block, across several, and
// past the parallel reduction floor; masks include the empty string and the
// top qubit.
#[test]
fn weighted_group_moments_matches_the_per_index_sum() {
    for n in [1usize, 2, 5, 6, 7, 10, 17] {
        let state = random_state(n, 42 + n as u64);
        let top = 1usize << (n - 1);
        let mut zmasks = vec![0, top, (1 << n) - 1, 1];
        zmasks.extend((0..n.saturating_sub(1)).map(|q| 0b11 << q));
        zmasks.extend((0..n).map(|q| (0x5a5a_5a5a >> q) & ((1 << n) - 1)));
        let coefficients: Vec<f64> = (0..zmasks.len()).map(|i| 0.7 - 0.13 * i as f64).collect();
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();

        let (mut want1, mut want2) = (0.0, 0.0);
        for (j, amp) in state.iter().enumerate() {
            let h: f64 = zmasks
                .iter()
                .zip(&coefficients)
                .map(|(&z, &c)| if (j & z).count_ones() % 2 == 1 { -c } else { c })
                .sum();
            want1 += amp.norm_sqr() * h;
            want2 += amp.norm_sqr() * h * h;
        }
        let (m1, m2) = weighted_group_moments(&state, &zmasks, &coefficients, norm);
        let (want1, want2) = (want1 / norm, want2 / norm);
        assert!(
            (m1 - want1).abs() < 1e-12 * want1.abs().max(1.0),
            "n={n}: {m1} vs {want1}"
        );
        assert!(
            (m2 - want2).abs() < 1e-12 * want2.max(1.0),
            "n={n}: {m2} vs {want2}"
        );
    }
}

/// `sum_i c_i P_i |state>` over the listed terms, each string applied factor by
/// factor.
fn apply_terms(
    terms: &[(f64, Vec<PauliTerm>)],
    members: &[usize],
    state: &[Complex64],
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); state.len()];
    for &i in members {
        let (coefficient, string) = &terms[i];
        for (j, &amp) in state.iter().enumerate() {
            let mut target = j;
            let mut value = amp * coefficient;
            for factor in string {
                let bit = j >> factor.qubit & 1;
                match factor.axis {
                    PauliAxis::X => target ^= 1 << factor.qubit,
                    PauliAxis::Y => {
                        target ^= 1 << factor.qubit;
                        value *= Complex64::new(0.0, if bit == 0 { 1.0 } else { -1.0 });
                    }
                    PauliAxis::Z if bit == 1 => value = -value,
                    PauliAxis::Z => {}
                }
            }
            out[target] += value;
        }
    }
    out
}

/// A `ZZ` chain with Z fields and an alternating X/Y chain with fields, `2n - 1`
/// terms each, plus seeded 1- to 4-local strings on random axes and an
/// identity offset.
fn mixed_observable(n: usize, seed: u64) -> PauliObservable {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let alternating = |q: usize| {
        if q.is_multiple_of(2) {
            PauliTerm::x(q)
        } else {
            PauliTerm::y(q)
        }
    };
    let mut terms: Terms = vec![(0.4, Vec::new())];
    for q in 0..n {
        terms.push((0.3 + 0.05 * q as f64, vec![PauliTerm::z(q)]));
        terms.push((-0.2 + 0.07 * q as f64, vec![alternating(q)]));
        if q + 1 < n {
            terms.push((
                0.9 - 0.1 * q as f64,
                vec![PauliTerm::z(q), PauliTerm::z(q + 1)],
            ));
            terms.push((-0.6, vec![alternating(q), alternating(q + 1)]));
        }
    }
    for _ in 0..10 {
        let mut qubits: Vec<usize> = (0..n).collect();
        let weight = rng.random_range(1..=4.min(n));
        let string = (0..weight)
            .map(|_| {
                let q = qubits.swap_remove(rng.random_range(0..qubits.len()));
                match rng.random_range(0..3u32) {
                    0 => PauliTerm::x(q),
                    1 => PauliTerm::y(q),
                    _ => PauliTerm::z(q),
                }
            })
            .collect();
        terms.push((rng.random::<f64>() - 0.5, string));
    }
    PauliObservable::from_terms(terms).unwrap()
}

// Each group's operator applied to the state gives `<H_g>` and
// `||H_g psi||^2` with no mask reduction. Across the sizes the grouping holds
// single-term groups, small groups that expand in pairs, and Z-only and
// rotated groups past the pair budget that take the moments pass.
#[test]
fn grouped_mean_and_variance_match_the_applied_operator() {
    use crate::backend::Backend;
    use crate::backend::statevector::StatevectorBackend;

    let mut shapes = [false; 4];
    for n in 3..=12usize {
        let circuit = crate::circuits::random_circuit(n, 6, 42 + n as u64);
        let mut backend = StatevectorBackend::new(42);
        crate::sim::run_on(&mut backend, &circuit).unwrap();
        let mut state = backend.export_statevector().unwrap();
        let norm = state.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        state.iter_mut().for_each(|a| *a /= norm);

        let ising = (0..n - 1)
            .map(|q| (1.0, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]))
            .chain((0..n).map(|q| (0.5, vec![PauliTerm::x(q)])))
            .chain([(-0.3, Vec::new())]);
        let observables = [
            mixed_observable(n, 42 + n as u64),
            PauliObservable::from_terms(ising.collect::<Vec<_>>()).unwrap(),
        ];
        for observable in &observables {
            let result = crate::sim::simulate(&circuit)
                .backend(crate::sim::BackendKind::Statevector)
                .seed(42)
                .observable_expectation(observable)
                .unwrap();
            let terms = observable.terms();
            let groups = &observable.grouping().groups;
            let variances = result.group_variances.as_ref().unwrap();
            let mut mean: f64 = terms
                .iter()
                .filter(|(_, string)| string.is_empty())
                .map(|(c, _)| c)
                .sum();
            for (g, group) in groups.iter().enumerate() {
                let size = group.term_indices.len();
                shapes[0] |= size == 1;
                shapes[1] |= (2..=6).contains(&size);
                shapes[2] |= size > 6 && group.is_z_only();
                shapes[3] |= size > 6 && !group.is_z_only();

                let applied = apply_terms(terms, &group.term_indices, &state);
                let first: f64 = state
                    .iter()
                    .zip(&applied)
                    .map(|(a, b)| (a.conj() * b).re)
                    .sum();
                let second: f64 = applied.iter().map(|a| a.norm_sqr()).sum();
                mean += first;
                let want = second - first * first;
                assert!(
                    (variances[g] - want).abs() < 1e-12,
                    "n={n} group {g}: {} vs {want}",
                    variances[g]
                );
            }
            assert!(
                (result.mean - mean).abs() < 1e-12,
                "n={n}: {} vs {mean}",
                result.mean
            );
            let total: f64 = variances.iter().sum();
            assert!((result.variance.unwrap() - total).abs() < 1e-12);
        }
    }
    assert_eq!(
        shapes, [true; 4],
        "single, small, large Z-only, large rotated"
    );
}

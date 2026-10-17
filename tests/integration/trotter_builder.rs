//! The product-formula builder against exact evolution: the rotation angle
//! convention, exactness on commuting terms, the convergence order of each
//! formula, and the native rotation route through fusion.

use crate::common;

use common::SEED;
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::Instruction;
use prism_q::gates::Gate;
use prism_q::{Circuit, PauliAxis, PauliObservable, PauliTerm, TrotterOrder, sim};
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

type Matrix = Vec<Vec<Complex64>>;

/// Heisenberg chain with seeded couplings on all three axes plus seeded
/// transverse and longitudinal fields, so no pair of terms commutes by
/// accident and the two-qubit `XX` and `YY` strings reach `Gate::PauliRot`.
fn heisenberg(n: usize) -> PauliObservable {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut coupling = |lo: f64, hi: f64| lo + (hi - lo) * rng.random::<f64>();
    let mut terms = Vec::new();
    for q in 0..n - 1 {
        for axis in [PauliAxis::X, PauliAxis::Y, PauliAxis::Z] {
            terms.push((
                coupling(0.5, 1.0),
                vec![PauliTerm::new(q, axis), PauliTerm::new(q + 1, axis)],
            ));
        }
    }
    for q in 0..n {
        terms.push((coupling(0.2, 0.5), vec![PauliTerm::x(q)]));
        terms.push((coupling(0.2, 0.5), vec![PauliTerm::z(q)]));
    }
    PauliObservable::from_terms(terms).unwrap()
}

fn seeded_state(n: usize) -> Vec<Complex64> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut state: Vec<Complex64> = (0..1usize << n)
        .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect();
    let norm = state.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    for amp in &mut state {
        *amp /= norm;
    }
    state
}

/// `H` as a dense matrix, `q[0]` the least significant bit of the row index.
/// `Y|b> = i (-1)^b |1 - b>`, so a string flips its X and Y qubits and picks up
/// `i^(#Y) (-1)^(parity of b on the Y and Z qubits)`.
fn dense_hamiltonian(hamiltonian: &PauliObservable, n: usize) -> Matrix {
    let dim = 1usize << n;
    let mut h = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    for (coefficient, string) in hamiltonian.terms() {
        let mut flip = 0usize;
        let mut sign_mask = 0usize;
        let mut num_y = 0u32;
        for term in string {
            match term.axis {
                PauliAxis::X => flip |= 1 << term.qubit,
                PauliAxis::Y => {
                    flip |= 1 << term.qubit;
                    sign_mask |= 1 << term.qubit;
                    num_y += 1;
                }
                PauliAxis::Z => sign_mask |= 1 << term.qubit,
            }
        }
        let i_pow = Complex64::new(0.0, 1.0).powu(num_y);
        for b in 0..dim {
            let sign = if (b & sign_mask).count_ones() & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            h[b ^ flip][b] += coefficient * sign * i_pow;
        }
    }
    h
}

fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let dim = a.len();
    let mut out = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    for i in 0..dim {
        for k in 0..dim {
            let aik = a[i][k];
            if aik.norm_sqr() == 0.0 {
                continue;
            }
            for j in 0..dim {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

fn one_norm(a: &Matrix) -> f64 {
    (0..a.len())
        .map(|j| a.iter().map(|row| row[j].norm()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// `exp(A)` by scaling and squaring over a Taylor series: scale `A` by `2^-s`
/// until its 1-norm is at most one half, sum the series until a term falls
/// below `1e-18`, then square the result `s` times.
fn expm(a: &Matrix) -> Matrix {
    let dim = a.len();
    let norm = one_norm(a);
    let squarings = if norm > 0.5 {
        (norm / 0.5).log2().ceil() as u32
    } else {
        0
    };
    let scale = 0.5f64.powi(squarings as i32);
    let scaled: Matrix = a
        .iter()
        .map(|row| row.iter().map(|x| x * scale).collect())
        .collect();
    let mut identity = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    for (i, row) in identity.iter_mut().enumerate() {
        row[i] = Complex64::new(1.0, 0.0);
    }
    let mut result = identity.clone();
    let mut term = identity;
    for k in 1..=40 {
        term = matmul(&term, &scaled);
        for row in &mut term {
            for x in row.iter_mut() {
                *x /= k as f64;
            }
        }
        for (acc, add) in result.iter_mut().zip(&term) {
            for (x, y) in acc.iter_mut().zip(add) {
                *x += y;
            }
        }
        if one_norm(&term) < 1e-18 {
            break;
        }
    }
    for _ in 0..squarings {
        result = matmul(&result, &result);
    }
    result
}

/// `exp(-i H t) |state>` from the dense matrix exponential.
fn exact_evolution(
    hamiltonian: &PauliObservable,
    n: usize,
    t: f64,
    state: &[Complex64],
) -> Vec<Complex64> {
    let h = dense_hamiltonian(hamiltonian, n);
    let a: Matrix = h
        .iter()
        .map(|row| row.iter().map(|x| x * Complex64::new(0.0, -t)).collect())
        .collect();
    let u = expm(&a);
    u.iter()
        .map(|row| row.iter().zip(state).map(|(u, s)| u * s).sum())
        .collect()
}

/// The circuit run through the default pipeline from `state`.
fn trotter_evolution(
    hamiltonian: &PauliObservable,
    n: usize,
    t: f64,
    steps: usize,
    order: TrotterOrder,
    state: &[Complex64],
) -> Vec<Complex64> {
    let mut circuit = Circuit::new(n, 0);
    circuit.add_trotter(hamiltonian, t, steps, order);
    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on_state(&mut backend, &circuit, state).unwrap();
    backend.export_statevector().unwrap()
}

fn distance(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn infidelity(a: &[Complex64], b: &[Complex64]) -> f64 {
    let overlap: Complex64 = a.iter().zip(b).map(|(x, y)| x.conj() * y).sum();
    1.0 - overlap.norm_sqr()
}

/// Least-squares slope of `log(error)` against `log(steps)`, negated so a
/// formula of order `p` reads as `p`.
fn convergence_order(steps: &[usize], errors: &[f64]) -> f64 {
    let xs: Vec<f64> = steps.iter().map(|&s| (s as f64).ln()).collect();
    let ys: Vec<f64> = errors.iter().map(|e| e.ln()).collect();
    let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let cov: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum();
    let var: f64 = xs.iter().map(|x| (x - x_mean).powi(2)).sum();
    -cov / var
}

fn count_native(circuit: &Circuit) -> usize {
    circuit
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
        .count()
}

// exp(-i c t X)|0> = cos(ct)|0> - i sin(ct)|1>, so the amplitude on |1> pins
// both the sign and the factor of two between the coefficient and the
// `exp(-i theta P / 2)` angle the rotation gate takes. The weight-2 string goes
// through `Gate::PauliRot`: X Y |00> = i |11>, so the |11> amplitude is
// sin(ct), real and positive.
#[test]
fn rotation_angle_is_twice_the_coefficient_times_dt() {
    let c = 0.3;
    let t = 0.7;

    let single = PauliObservable::from_terms([(c, vec![PauliTerm::x(0)])]).unwrap();
    let mut circuit = Circuit::new(1, 0);
    circuit.add_trotter(&single, t, 1, TrotterOrder::First);
    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on(&mut backend, &circuit).unwrap();
    let state = backend.export_statevector().unwrap();
    assert!((state[0] - Complex64::new((c * t).cos(), 0.0)).norm() < 1e-12);
    assert!((state[1] - Complex64::new(0.0, -(c * t).sin())).norm() < 1e-12);

    let pair = PauliObservable::from_terms([(c, vec![PauliTerm::x(0), PauliTerm::y(1)])]).unwrap();
    let mut circuit = Circuit::new(2, 0);
    circuit.add_trotter(&pair, t, 1, TrotterOrder::First);
    assert_eq!(count_native(&circuit), 1);
    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on(&mut backend, &circuit).unwrap();
    let state = backend.export_statevector().unwrap();
    assert!((state[0] - Complex64::new((c * t).cos(), 0.0)).norm() < 1e-12);
    assert!((state[3] - Complex64::new((c * t).sin(), 0.0)).norm() < 1e-12);
}

#[test]
fn first_order_is_exact_on_a_single_term() {
    let n = 4;
    let state = seeded_state(n);
    let single =
        PauliObservable::from_terms([(0.9, vec![PauliTerm::x(0), PauliTerm::y(2)])]).unwrap();
    let exact = exact_evolution(&single, n, 1.3, &state);
    let got = trotter_evolution(&single, n, 1.3, 1, TrotterOrder::First, &state);
    assert!(distance(&got, &exact) < 1e-12);
}

#[test]
fn first_order_is_exact_on_commuting_terms() {
    let n = 4;
    let state = seeded_state(n);
    let diagonal = PauliObservable::from_terms([
        (0.4, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (0.7, vec![PauliTerm::z(1), PauliTerm::z(2)]),
        (0.2, vec![PauliTerm::z(0)]),
        (0.5, vec![PauliTerm::z(1), PauliTerm::z(2), PauliTerm::z(3)]),
    ])
    .unwrap();
    let exact = exact_evolution(&diagonal, n, 2.1, &state);
    let got = trotter_evolution(&diagonal, n, 2.1, 1, TrotterOrder::First, &state);
    assert!(distance(&got, &exact) < 1e-12);
}

#[test]
fn identity_term_is_a_global_phase() {
    let n = 3;
    let state = seeded_state(n);
    let with_offset = PauliObservable::from_terms([
        (1.5, vec![]),
        (0.6, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (0.3, vec![PauliTerm::x(2)]),
    ])
    .unwrap();
    let t = 0.8;
    let exact = exact_evolution(&with_offset, n, t, &state);
    let got = trotter_evolution(&with_offset, n, t, 1, TrotterOrder::First, &state);
    let phase = Complex64::new(0.0, -1.5 * t).exp();
    let rephased: Vec<Complex64> = got.iter().map(|amp| amp * phase).collect();
    assert!(distance(&rephased, &exact) < 1e-12);
}

#[test]
fn convergence_orders_match_the_formulas() {
    let n = 7;
    let t = 0.6;
    let hamiltonian = heisenberg(n);
    let state = seeded_state(n);
    let exact = exact_evolution(&hamiltonian, n, t, &state);

    for (order, expected, steps) in [
        (TrotterOrder::First, 1.0, [8usize, 16, 32, 64]),
        (TrotterOrder::Second, 2.0, [4, 8, 16, 32]),
        (TrotterOrder::Suzuki(2), 4.0, [2, 4, 8, 16]),
    ] {
        let mut errors = Vec::new();
        let mut infidelities = Vec::new();
        for &s in &steps {
            let got = trotter_evolution(&hamiltonian, n, t, s, order, &state);
            errors.push(distance(&got, &exact));
            infidelities.push(infidelity(&exact, &got));
        }
        assert!(
            infidelities.windows(2).all(|pair| pair[1] < pair[0]),
            "{order:?}: infidelity did not fall with steps: {infidelities:?}"
        );
        assert!(
            *errors.last().unwrap() > 1e-11,
            "{order:?}: finest error sits on the rounding floor, the fit is meaningless"
        );
        let slope = convergence_order(&steps, &errors);
        assert!(
            (slope - expected).abs() < 0.3,
            "{order:?}: fitted order {slope:.3}, expected {expected}: errors {errors:?}"
        );
    }
}

#[test]
fn native_rotations_survive_fusion_with_the_state_unchanged() {
    let n = 16;
    let hamiltonian = heisenberg(n);
    let mut circuit = Circuit::new(n, 0);
    circuit.add_trotter(&hamiltonian, 0.3, 2, TrotterOrder::Second);
    assert!(count_native(&circuit) > 0, "no term reached Gate::PauliRot");

    let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
    let native_in_fused = fused.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Gate {
                gate: Gate::PauliRot(_) | Gate::MultiPauliRot(_),
                ..
            }
        )
    });
    assert!(native_in_fused, "fusion lowered every native rotation");

    let mut unfused = StatevectorBackend::new(SEED);
    unfused.init(n, 0).unwrap();
    for inst in &circuit.instructions {
        unfused.apply(inst).unwrap();
    }
    let expected = unfused.export_statevector().unwrap();

    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on(&mut backend, &circuit).unwrap();
    let got = backend.export_statevector().unwrap();
    assert!(distance(&got, &expected) < 1e-10);
}

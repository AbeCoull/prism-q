use super::*;

const EPS: f64 = 1e-12;

fn one_qubit_gates() -> Vec<Gate> {
    let fused = [
        [Complex64::new(0.6, 0.0), Complex64::new(0.0, 0.8)],
        [Complex64::new(0.0, 0.8), Complex64::new(0.6, 0.0)],
    ];
    vec![
        Gate::Id,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::SX,
        Gate::SXdg,
        Gate::Rx(0.37),
        Gate::Ry(-1.21),
        Gate::Rz(2.05),
        Gate::P(0.83),
        Gate::Fused(Box::new(fused)),
    ]
}

fn random_ket(rng: &mut ChaCha8Rng, dim: usize) -> Vec<Complex64> {
    let mut psi: Vec<Complex64> = (0..dim)
        .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect();
    let norm = psi.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    for a in &mut psi {
        *a /= norm;
    }
    psi
}

/// `sum_k w_k |psi_k><psi_k|` over `weights.len()` random kets, row-major.
fn random_rho(rng: &mut ChaCha8Rng, n: usize, weights: &[f64]) -> Vec<Complex64> {
    let dim = 1usize << n;
    let mut rho = vec![Complex64::new(0.0, 0.0); dim * dim];
    for &w in weights {
        let psi = random_ket(rng, dim);
        for r in 0..dim {
            for c in 0..dim {
                rho[r * dim + c] += w * psi[r] * psi[c].conj();
            }
        }
    }
    rho
}

#[test]
fn rzz_run_sweep_equals_full_sweep() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let one = Complex64::new(1.0, 0.0);
    // n = 6 runs the scalar arms and n = 8 the rayon ones.
    for n in [6, 8] {
        let rho = random_rho(&mut rng, n, &[0.5, 0.3, 0.2]);
        let mut full = DensityMatrixBackend::new(42);
        full.init_from_density_matrix(rho.clone(), 0).unwrap();
        let mut runs = DensityMatrixBackend::new(42);
        runs.init_from_density_matrix(rho.clone(), 0).unwrap();
        let mut routed = DensityMatrixBackend::new(42);
        routed.init_from_density_matrix(rho.clone(), 0).unwrap();
        for theta in [0.7, 0.3, -2.4] {
            let diag = DensityMatrixBackend::rzz_sandwich_diagonal(theta);
            assert_eq!(
                diag.iter().filter(|&&d| d == one).count(),
                8,
                "theta={theta}"
            );
            for q0 in 0..n {
                for q1 in 0..n {
                    full.sv.state.copy_from_slice(&rho);
                    full.kraus_2q_diagonal_sweep(&diag, q0, q1).unwrap();

                    runs.sv.state.copy_from_slice(&rho);
                    runs.kraus_2q_diagonal_runs(&diag, q0, q1);
                    assert!(
                        runs.sv.state == full.sv.state,
                        "run sweep n={n} ({q0}, {q1}) theta={theta}"
                    );

                    routed.sv.state.copy_from_slice(&rho);
                    routed.apply_rzz_sandwich(q0, q1, theta).unwrap();
                    assert!(
                        routed.sv.state == full.sv.state,
                        "routed sandwich n={n} ({q0}, {q1}) theta={theta}"
                    );
                }
            }
        }
    }
}

#[test]
fn blocked_1q_sandwich_matches_two_register_passes() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for n in 1..=7 {
        let states = [
            random_rho(&mut rng, n, &[1.0]),
            random_rho(&mut rng, n, &[0.5, 0.3, 0.2]),
        ];
        for rho in &states {
            for gate in one_qubit_gates() {
                let mat = matrix_1q(&gate).expect("one-qubit gate");
                for qubit in 0..n {
                    let mut reference = DensityMatrixBackend::new(42);
                    reference.init_from_density_matrix(rho.clone(), 0).unwrap();
                    reference.sv.apply_1q_matrix(qubit + n, &mat).unwrap();
                    reference
                        .sv
                        .apply_1q_matrix(qubit, &conjugate_2x2(&mat))
                        .unwrap();

                    let mut blocked = rho.clone();
                    simd::apply_1q_sandwich_blocked(&mut blocked, n, qubit, &mat);

                    for (i, (got, want)) in blocked.iter().zip(&reference.sv.state).enumerate() {
                        assert!(
                            (got - want).norm() < EPS,
                            "{gate:?} n={n} qubit={qubit} entry {i}: {got} vs {want}",
                        );
                    }
                }
            }
        }
    }
}

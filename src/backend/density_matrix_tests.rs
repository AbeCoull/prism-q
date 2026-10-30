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

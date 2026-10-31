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

#[test]
fn expectations_pauli_match_a_serial_trace_across_the_parallel_floor() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for n in 1..=7 {
        let d = 1usize << n;
        let rho = random_rho(&mut rng, n, &[0.5, 0.3, 0.2]);
        let masks: Vec<(usize, usize, u32)> = (0..d)
            .flat_map(|x| (0..d).map(move |z| (x, z, (x & z).count_ones())))
            .collect();
        let mut dm = DensityMatrixBackend::new(42);
        dm.init_from_density_matrix(rho.clone(), 0).unwrap();
        let got = dm.expectations_pauli(&masks);

        for (&(xmask, zmask, num_y), got) in masks.iter().zip(&got) {
            let trace: Complex64 = (0..d)
                .map(|j| {
                    let sign = if (j & zmask).count_ones() % 2 == 1 {
                        -1.0
                    } else {
                        1.0
                    };
                    rho[j * d + (j ^ xmask)] * sign
                })
                .sum();
            let want = (trace * i_pow(num_y)).re;
            assert!(
                (got - want).abs() < EPS,
                "n={n} x={xmask:#b} z={zmask:#b}: {got} vs {want}",
            );
        }
    }
}

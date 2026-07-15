//! Regression: the native whole-state `QftBlock` FFT and the textbook
//! expansion (`expand_qft_blocks` / `qft_textbook_steps`) must be identical
//! transforms. The `PRISM_NO_QFT_BLOCK` kill switch documents them as A/B
//! equivalent, and every non-statevector backend (MPS, sparse, factored,
//! product, tensornetwork) plus the GPU path executes the textbook expansion.
//!
//! Existing cross-backend tests compare probabilities from the |0...0> input,
//! where every QFT convention yields a uniform distribution and phase errors
//! are invisible. These checks feed a non-|0> input and compare amplitudes.

use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::{Circuit, expand_qft_blocks};
use prism_q::circuits::{phase_estimation_circuit, qft_circuit};
use prism_q::gates::Gate;
use prism_q::sim::{BackendKind, run_on, simulate};

fn prep_basis(n: usize, j: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        if j & (1 << q) != 0 {
            c.add_gate(Gate::X, &[q]);
        }
    }
    c
}

fn amplitudes(circuit: &Circuit) -> Vec<(f64, f64)> {
    let mut backend = StatevectorBackend::new(42);
    run_on(&mut backend, circuit).unwrap();
    backend
        .state_vector()
        .iter()
        .map(|z| (z.re, z.im))
        .collect()
}

fn max_abs_diff(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|((ar, ai), (br, bi))| ((ar - br).powi(2) + (ai - bi).powi(2)).sqrt())
        .fold(0.0f64, f64::max)
}

#[test]
fn qft_native_matches_analytic_dft_nontrivial_input() {
    let n = 4;
    let j = 5;
    let mut circuit = prep_basis(n, j);
    circuit.instructions.extend(qft_circuit(n).instructions);
    let native = amplitudes(&circuit);

    let size = 1usize << n;
    let norm = 1.0 / (size as f64).sqrt();
    let analytic: Vec<(f64, f64)> = (0..size)
        .map(|k| {
            let angle = std::f64::consts::TAU * (j * k) as f64 / size as f64;
            (norm * angle.cos(), norm * angle.sin())
        })
        .collect();

    let diff = max_abs_diff(&native, &analytic);
    assert!(
        diff < 1e-9,
        "native QFT deviates from analytic DFT: {diff:e}"
    );
}

#[test]
fn qft_textbook_expansion_matches_native_nontrivial_input() {
    let n = 4;
    for j in [1usize, 3, 5, 6, 9, 10] {
        let mut circuit = prep_basis(n, j);
        circuit.instructions.extend(qft_circuit(n).instructions);

        let native = amplitudes(&circuit);
        let expanded = amplitudes(&expand_qft_blocks(&circuit));

        let diff = max_abs_diff(&native, &expanded);
        assert!(
            diff < 1e-9,
            "QftBlock textbook expansion disagrees with native FFT for input |{j}>: {diff:e}"
        );
    }
}

#[test]
fn qpe_recovers_known_eigenphase() {
    // phase_estimation_circuit applies (pi/4)*2^k = 2*pi*(1/8)*2^k to counting
    // qubit k, so it estimates phase = 1/8. With `n_counting` counting qubits
    // this is exactly representable, so the register collapses deterministically
    // to m = 2^n_counting / 8, with the target qubit left in |1>.
    for n in [4usize, 5, 7] {
        let n_counting = n - 1;
        let expected_m = (1usize << n_counting) / 8;
        let expected_index = expected_m + (1 << n_counting);

        let outcome = simulate(&phase_estimation_circuit(n))
            .seed(42)
            .backend(BackendKind::Statevector)
            .run()
            .unwrap();
        let probs = outcome.probabilities.unwrap().to_vec();

        let (peak, peak_p) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, p)| (i, *p))
            .unwrap();

        assert_eq!(
            peak, expected_index,
            "QPE (n={n}) peak at index {peak}, expected {expected_index}"
        );
        assert!(
            peak_p > 0.999,
            "QPE (n={n}) peak probability {peak_p} not deterministic"
        );
    }
}

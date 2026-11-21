//! Assertions and fixtures shared by the unit tests.

use num_complex::Complex64;
use rand::RngExt;
use rand_chacha::ChaCha8Rng;

use crate::backend::Backend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::{Circuit, Instruction};
use crate::sim::unified_pauli::PauliTerm;

/// Assert equal lengths and `|actual[i] - expected[i]| < eps` at every index.
#[track_caller]
pub(crate) fn assert_probs_close(actual: &[f64], expected: &[f64], eps: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "probability vector length mismatch"
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < eps,
            "prob[{i}]: expected {e}, got {a} (diff {})",
            (a - e).abs()
        );
    }
}

/// [`assert_probs_close`] for amplitudes, on the modulus of each difference.
#[track_caller]
pub(crate) fn assert_amps_close(actual: &[Complex64], expected: &[Complex64], eps: f64) {
    assert_eq!(actual.len(), expected.len(), "state vector length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() < eps,
            "amp[{i}]: expected {e}, got {a} (diff {})",
            (a - e).norm()
        );
    }
}

/// [`assert_probs_close`] with `label` leading each failure message.
#[track_caller]
pub(crate) fn assert_close(label: &str, got: &[f64], want: &[f64], eps: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < eps, "{label}[{i}]: got {g}, want {w}");
    }
}

/// Assert `|actual[i][j] - expected[i][j]| < eps` at every entry.
#[track_caller]
pub(crate) fn assert_mat_close<const N: usize>(
    actual: &[[Complex64; N]; N],
    expected: &[[Complex64; N]; N],
    eps: f64,
) {
    for i in 0..N {
        for j in 0..N {
            assert!(
                (actual[i][j] - expected[i][j]).norm() < eps,
                "mat[{i}][{j}]: expected {:?}, got {:?}",
                expected[i][j],
                actual[i][j]
            );
        }
    }
}

/// Probabilities after the circuit's gates on a statevector, skipping every other
/// instruction.
pub(crate) fn gate_only_statevector_probabilities(circuit: &Circuit) -> Vec<f64> {
    let mut sv = StatevectorBackend::new(42);
    sv.init(circuit.num_qubits, 0).unwrap();
    for inst in &circuit.instructions {
        if matches!(inst, Instruction::Gate { .. }) {
            sv.apply(inst).unwrap();
        }
    }
    sv.probabilities().unwrap()
}

/// A normalized state of `num_qubits` qubits, each component uniform in `[-0.5, 0.5)`
/// before scaling.
pub(crate) fn random_state(num_qubits: usize, rng: &mut ChaCha8Rng) -> Vec<Complex64> {
    let raw: Vec<Complex64> = (0..1usize << num_qubits)
        .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect();
    let norm = raw.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    raw.into_iter().map(|a| a / norm).collect()
}

/// Pauli strings read off the stabilizer half of an eager tableau: each
/// generator, then the product of each adjacent pair. Every one is in the
/// group, so its expectation is +1 or -1.
pub(crate) fn group_strings(backend: &StabilizerBackend) -> Vec<Vec<PauliTerm>> {
    let n = backend.num_qubits();
    let nw = n.div_ceil(64);
    let (xz, _) = backend.raw_tableau();
    let string_of = |rows: &[usize]| -> Vec<PauliTerm> {
        (0..n)
            .filter_map(|q| {
                let bit =
                    |r: usize, half: usize| (xz[r * 2 * nw + half * nw + q / 64] >> (q % 64)) & 1;
                let x = rows.iter().fold(0, |acc, &r| acc ^ bit(r, 0)) == 1;
                let z = rows.iter().fold(0, |acc, &r| acc ^ bit(r, 1)) == 1;
                match (x, z) {
                    (true, false) => Some(PauliTerm::x(q)),
                    (false, true) => Some(PauliTerm::z(q)),
                    (true, true) => Some(PauliTerm::y(q)),
                    (false, false) => None,
                }
            })
            .collect()
    };
    let mut strings: Vec<Vec<PauliTerm>> = (n..2 * n).map(|r| string_of(&[r])).collect();
    strings.extend((n..2 * n - 1).map(|r| string_of(&[r, r + 1])));
    strings
}

/// The device context for a test, or `None` after printing a skip line when no GPU is
/// usable; panics instead under `PRISM_REQUIRE_GPU`.
#[cfg(feature = "gpu")]
pub(crate) fn gpu_context_or_skip() -> Option<std::sync::Arc<crate::gpu::GpuContext>> {
    match crate::gpu::GpuContext::new(0) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "PRISM_REQUIRE_GPU is set but no usable GPU was found ({e})"
            );
            eprintln!("SKIP: no usable GPU ({e})");
            None
        }
    }
}

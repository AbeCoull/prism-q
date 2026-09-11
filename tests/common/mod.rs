//! Shared helpers for the cross-backend correctness tests. Tolerance
//! constants are split per backend so the same helper runs at each backend's
//! precision.

#![allow(dead_code)]

pub mod caps;
pub mod circuits;
pub mod framework;
pub mod gate_fixtures;
pub mod matrix;

use num_complex::Complex64;
use prism_q::Parameters;
use prism_q::backend::Backend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::{Circuit, Instruction};
use prism_q::gates::Gate;
use prism_q::sim;

pub const SV_EPS: f64 = 1e-10;
pub const STAB_EPS: f64 = 1e-12;
pub const SPARSE_EPS: f64 = 1e-10;
pub const MPS_EPS: f64 = 1e-9;
pub const TN_EPS: f64 = 1e-10;
pub const PRODUCT_EPS: f64 = 1e-12;
pub const FACTORED_EPS: f64 = 1e-10;
pub const DM_EPS: f64 = 1e-12;

pub const SEED: u64 = 42;

/// The 16 two-qubit Pauli Kraus operators of symmetric depolarizing with
/// parameter `p`, weighted `sqrt(1-p)` on `I(x)I` and `sqrt(p/15)` elsewhere,
/// indexed `2*bit(q0) + bit(q1)`. The closed-form
/// `DensityMatrixBackend::apply_2q_depolarizing` replaces this set, so the
/// two are independent implementations of the same channel.
pub fn depolarizing_2q_kraus(p: f64) -> Vec<[[Complex64; 4]; 4]> {
    let c = Complex64::new;
    let paulis: [[[Complex64; 2]; 2]; 4] = [
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]],
        [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]],
        [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]],
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]],
    ];
    let mut kraus = vec![[[c(0.0, 0.0); 4]; 4]; 16];
    for a in 0..4 {
        for b in 0..4 {
            let w = if a == 0 && b == 0 {
                (1.0 - p).sqrt()
            } else {
                (p / 15.0).sqrt()
            };
            let k = &mut kraus[4 * a + b];
            for (t, row) in k.iter_mut().enumerate() {
                for (tp, entry) in row.iter_mut().enumerate() {
                    *entry = c(w, 0.0) * paulis[a][t >> 1][tp >> 1] * paulis[b][t & 1][tp & 1];
                }
            }
        }
    }
    kraus
}

/// Every `n`-qubit Pauli as `(xmask, zmask, num_y)`; the `4^n` expectations
/// determine `rho` uniquely.
pub fn all_pauli_masks(n: usize) -> Vec<(usize, usize, u32)> {
    let d = 1usize << n;
    let mut masks = Vec::with_capacity(d * d);
    for xmask in 0..d {
        for zmask in 0..d {
            masks.push((xmask, zmask, (xmask & zmask).count_ones()));
        }
    }
    masks
}

/// Statevector probabilities used as the reference for the backend matrices.
///
/// The statevector is a participant, not an independent authority: a helper
/// built on it can only report that two implementations disagree, and it will
/// name the other backend as the failure. Anything this reference is expected
/// to get right therefore needs a closed-form anchor in
/// `tests/golden_small_circuits.rs` that names the statevector directly. The
/// `reset` contract went unanchored there for a long time, and a
/// projection-onto-|0> implementation survived a green matrix as a result:
/// four correct backends were the ones reported as failing.
pub fn sv_reference_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

pub fn assert_probs_close(actual: &[f64], expected: &[f64], eps: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: probability vector length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < eps,
            "{label} prob[{i}]: expected {e:.12}, got {a:.12} (diff {diff:.2e}, eps {eps:.0e})"
        );
    }
}

pub fn assert_backend_matches_sv<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
    eps: f64,
    label: &str,
) {
    sim::run_on(backend, circuit).unwrap();
    let actual = backend.probabilities().unwrap();
    let expected = sv_reference_probs(circuit);
    assert_probs_close(&actual, &expected, eps, label);
}

pub fn assert_backend_outcome_matches_sv<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
    eps: f64,
    label: &str,
) {
    let expected = sim::run_on(&mut StatevectorBackend::new(SEED), circuit).unwrap();
    let actual = sim::run_on(backend, circuit).unwrap();
    assert_eq!(
        actual.classical_bits, expected.classical_bits,
        "{label}: classical bits mismatch"
    );
    assert_probs_close(
        &actual.probabilities.expect("actual probabilities").to_vec(),
        &expected
            .probabilities
            .expect("expected probabilities")
            .to_vec(),
        eps,
        label,
    );
}

pub fn assert_backend_repeatable<B: Backend, F: Fn() -> B>(
    new_backend: F,
    circuit: &Circuit,
    eps: f64,
    label: &str,
) {
    let first = sim::run_on(&mut new_backend(), circuit).unwrap();
    let second = sim::run_on(&mut new_backend(), circuit).unwrap();
    assert_eq!(
        first.classical_bits, second.classical_bits,
        "{label}: fixed seed produced different classical bits"
    );
    assert_probs_close(
        &first.probabilities.expect("first probabilities").to_vec(),
        &second.probabilities.expect("second probabilities").to_vec(),
        eps,
        label,
    );
}

pub fn run_unfused_probs<B: Backend>(backend: &mut B, circuit: &Circuit) -> Vec<f64> {
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    let expanded = if backend.supports_qft_block() {
        std::borrow::Cow::Borrowed(circuit)
    } else {
        prism_q::circuit::expand_qft_blocks(circuit)
    };
    let expanded = if backend.supports_pauli_rotation() {
        expanded
    } else {
        std::borrow::Cow::Owned(prism_q::circuit::expand_pauli_rotations(&expanded).into_owned())
    };
    for instr in &expanded.instructions {
        backend.apply(instr).unwrap();
    }
    backend.probabilities().unwrap()
}

pub fn run_fused_probs<B: Backend>(backend: &mut B, circuit: &Circuit) -> Vec<f64> {
    sim::run_on(backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

pub fn run_and_probs(circuit: &Circuit) -> Vec<f64> {
    run_fused_probs(&mut StatevectorBackend::new(SEED), circuit)
}

pub fn run_and_state(circuit: &Circuit) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.state_vector().to_vec()
}

pub fn run_stabilizer_probs(circuit: &Circuit) -> Vec<f64> {
    run_fused_probs(&mut StabilizerBackend::new(SEED), circuit)
}

pub fn assert_fused_matches_unfused<B: Backend, F: Fn() -> B>(
    new_backend: F,
    circuit: &Circuit,
    eps: f64,
    label: &str,
) {
    let mut unfused_backend = new_backend();
    let unfused = run_unfused_probs(&mut unfused_backend, circuit);
    let mut fused_backend = new_backend();
    let fused = run_fused_probs(&mut fused_backend, circuit);
    assert_probs_close(&fused, &unfused, eps, label);
}

pub fn is_clifford(circuit: &Circuit) -> bool {
    for instr in &circuit.instructions {
        match instr {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => {
                if !gate.is_clifford() {
                    return false;
                }
            }
            Instruction::Region(region) => {
                if !is_clifford_body(region.body()) {
                    return false;
                }
            }
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Barrier { .. } => {}
        }
    }
    true
}

fn is_clifford_body(instructions: &[Instruction]) -> bool {
    let mut probe = Circuit::new(1, 0);
    probe.instructions = instructions.to_vec();
    is_clifford(&probe)
}

pub fn count_gates(circuit: &Circuit, want: impl Fn(&Gate) -> bool) -> usize {
    circuit
        .instructions
        .iter()
        .filter(|inst| matches!(inst, Instruction::Gate { gate, .. } if want(gate)))
        .count()
}

/// The two Kraus operators of amplitude damping at rate `gamma`.
pub fn amplitude_damping(gamma: f64) -> Vec<[[Complex64; 2]; 2]> {
    let c = Complex64::new;
    let zero = c(0.0, 0.0);
    vec![
        [[c(1.0, 0.0), zero], [zero, c((1.0 - gamma).sqrt(), 0.0)]],
        [[zero, c(gamma.sqrt(), 0.0)], [zero, zero]],
    ]
}

/// A copy of `circuit` with `delta` added to the angle of every gate bound to
/// parameter `slot`.
pub fn shift_slot(circuit: &Circuit, params: &Parameters, slot: usize, delta: f64) -> Circuit {
    let mut out = circuit.clone();
    for link in params.links().iter().filter(|l| l.slot == slot) {
        if let Instruction::Gate { gate, .. } = &mut out.instructions[link.instruction] {
            *gate = shifted_gate(gate, delta);
        }
    }
    out
}

fn shifted_gate(gate: &Gate, delta: f64) -> Gate {
    match gate {
        Gate::Rx(t) => Gate::Rx(t + delta),
        Gate::Ry(t) => Gate::Ry(t + delta),
        Gate::Rz(t) => Gate::Rz(t + delta),
        Gate::Rzz(t) => Gate::Rzz(t + delta),
        Gate::P(t) => Gate::P(t + delta),
        Gate::PauliRot(data) => {
            let mut shifted = data.clone();
            shifted.set_theta(data.theta() + delta);
            Gate::PauliRot(shifted)
        }
        other => panic!("gate {} is not differentiable", other.name()),
    }
}

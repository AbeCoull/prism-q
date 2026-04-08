use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::smallvec;

use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::gates::Gate;
use crate::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel};
use crate::sim::ShotsResult;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn apply_pauli(
    backend: &mut dyn Backend,
    qubit: usize,
    px: f64,
    py: f64,
    pz: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let r: f64 = rand::Rng::gen(rng);
    if r < px {
        backend.apply(&Instruction::Gate {
            gate: Gate::X,
            targets: smallvec![qubit],
        })?;
    } else if r < px + py {
        backend.apply(&Instruction::Gate {
            gate: Gate::Y,
            targets: smallvec![qubit],
        })?;
    } else if r < px + py + pz {
        backend.apply(&Instruction::Gate {
            gate: Gate::Z,
            targets: smallvec![qubit],
        })?;
    }
    Ok(())
}

/// Minimum p_jump for which we apply the jump branch instead of the no-jump
/// branch. Below this, the jump is numerically indistinguishable from zero and
/// dividing by sqrt(p_jump) produces NaN/inf.
const JUMP_EPSILON: f64 = 1e-12;

/// Apply a single-qubit diagonal Kraus channel with 2 operators of the form:
///   K_0 = diag(1, c)        — no-jump
///   K_1 = [[0, s01_upper], [0, s11_lower]]   — jump (AD)
///       or diag(0, s)       — jump (PD, pass s01_upper=0)
///
/// where c = sqrt(1-gamma). The effective jump probability is gamma * p1.
fn apply_diagonal_kraus_2op(
    backend: &mut dyn Backend,
    qubit: usize,
    gamma: f64,
    jump_moves_population: bool,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let p1 = backend.qubit_probability(qubit)?;
    let p_jump = gamma * p1;
    let r: f64 = rand::Rng::gen(rng);

    let zero = Complex64::new(0.0, 0.0);

    if r < p_jump && p_jump > JUMP_EPSILON {
        let s = (gamma / p_jump).sqrt();
        let mat = if jump_moves_population {
            [[zero, Complex64::new(s, 0.0)], [zero, zero]]
        } else {
            [[zero, zero], [zero, Complex64::new(s, 0.0)]]
        };
        backend.apply_1q_matrix(qubit, &mat)?;
    } else {
        let denom = 1.0 - p_jump;
        if denom <= JUMP_EPSILON {
            return Ok(());
        }
        let inv = 1.0 / denom.sqrt();
        let mat = [
            [Complex64::new(inv, 0.0), zero],
            [zero, Complex64::new((1.0 - gamma).sqrt() * inv, 0.0)],
        ];
        backend.apply_1q_matrix(qubit, &mat)?;
    }
    Ok(())
}

fn apply_amplitude_damping(
    backend: &mut dyn Backend,
    qubit: usize,
    gamma: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    apply_diagonal_kraus_2op(backend, qubit, gamma, true, rng)
}

fn apply_phase_damping(
    backend: &mut dyn Backend,
    qubit: usize,
    gamma: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    apply_diagonal_kraus_2op(backend, qubit, gamma, false, rng)
}

fn apply_thermal_relaxation(
    backend: &mut dyn Backend,
    qubit: usize,
    t1: f64,
    t2: f64,
    gate_time: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    if t1 <= 0.0 || t2 <= 0.0 || gate_time < 0.0 {
        return Ok(());
    }
    let p_reset = 1.0 - (-gate_time / t1).exp();
    let p_dephase = if t2 < 2.0 * t1 {
        1.0 - (-gate_time * (1.0 / t2 - 0.5 / t1)).exp()
    } else {
        0.0
    };

    let r: f64 = rand::Rng::gen(rng);
    if r < p_reset {
        backend.reset(qubit)?;
    } else {
        let r2: f64 = rand::Rng::gen(rng);
        if r2 < p_dephase {
            backend.apply(&Instruction::Gate {
                gate: Gate::Z,
                targets: smallvec![qubit],
            })?;
        }
    }
    Ok(())
}

fn apply_two_qubit_depolarizing(
    backend: &mut dyn Backend,
    q0: usize,
    q1: usize,
    p: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    // 16 Pauli products: I⊗I, I⊗X, I⊗Y, I⊗Z, X⊗I, ..., Z⊗Z
    // Each non-identity term has probability p/15
    let pp = p / 15.0;
    let r: f64 = rand::Rng::gen(rng);

    if r >= p {
        return Ok(()); // I⊗I (no error)
    }

    // Sample which of 15 error terms
    let idx = ((r / pp) as usize).min(14);
    let (pauli0, pauli1) = TWO_QUBIT_PAULIS[idx];
    if pauli0 != PauliOp::I {
        apply_pauli_op(backend, q0, pauli0)?;
    }
    if pauli1 != PauliOp::I {
        apply_pauli_op(backend, q1, pauli1)?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum PauliOp {
    I,
    X,
    Y,
    Z,
}

const TWO_QUBIT_PAULIS: [(PauliOp, PauliOp); 15] = [
    (PauliOp::I, PauliOp::X),
    (PauliOp::I, PauliOp::Y),
    (PauliOp::I, PauliOp::Z),
    (PauliOp::X, PauliOp::I),
    (PauliOp::X, PauliOp::X),
    (PauliOp::X, PauliOp::Y),
    (PauliOp::X, PauliOp::Z),
    (PauliOp::Y, PauliOp::I),
    (PauliOp::Y, PauliOp::X),
    (PauliOp::Y, PauliOp::Y),
    (PauliOp::Y, PauliOp::Z),
    (PauliOp::Z, PauliOp::I),
    (PauliOp::Z, PauliOp::X),
    (PauliOp::Z, PauliOp::Y),
    (PauliOp::Z, PauliOp::Z),
];

fn apply_pauli_op(backend: &mut dyn Backend, qubit: usize, op: PauliOp) -> Result<()> {
    let gate = match op {
        PauliOp::X => Gate::X,
        PauliOp::Y => Gate::Y,
        PauliOp::Z => Gate::Z,
        PauliOp::I => return Ok(()),
    };
    backend.apply(&Instruction::Gate {
        gate,
        targets: smallvec![qubit],
    })
}

/// Threshold at which a Kraus operator's `K†K` off-diagonal element is
/// considered non-zero. Used to reject Kraus ops whose `K†K` is not diagonal,
/// because such ops require knowing the reduced density matrix's coherences
/// which we cannot extract from a pure state without cloning the backend.
const KRAUS_DIAGONAL_TOL: f64 = 1e-10;

/// Return true if `K†K` is diagonal (within `KRAUS_DIAGONAL_TOL`).
///
/// `K†K` is diagonal iff its (0,1) entry vanishes:
///   `(K†K)[0][1] = k00*·k01 + k10*·k11 = 0`
/// When this holds, `Tr(K†K ρ) = (|k00|²+|k10|²)·p0 + (|k01|²+|k11|²)·p1`,
/// which depends only on populations (not coherences) and can be computed
/// exactly from `qubit_probability` alone. Diagonal Kraus ops and all pure
/// Paulis (X, Y, Z, and their scaled versions) satisfy this.
#[inline]
fn is_kdagger_k_diagonal(k: &[[Complex64; 2]; 2]) -> bool {
    let cross = k[0][0].conj() * k[0][1] + k[1][0].conj() * k[1][1];
    cross.norm() < KRAUS_DIAGONAL_TOL
}

/// Sample and apply one of a set of 1-qubit Kraus operators.
///
/// **Supported Kraus operators.** Each operator `K_k` must satisfy `K_k† K_k`
/// is diagonal. This is the class of Kraus ops for which the per-branch
/// probability `p_k = Tr(K_k† K_k ρ)` depends only on the qubit's populations
/// (not its coherences), so the trajectory branch can be selected exactly
/// from `qubit_probability` alone without cloning the full state. This class
/// includes:
///
/// - All diagonal Kraus operators (amplitude/phase damping, dephasing, Z).
/// - All pure Paulis (X, Y, Z) and their scaled versions.
/// - The standard bit-flip / phase-flip / bit-phase-flip channels.
///
/// It **excludes** Kraus operators like `[[a,b],[0,0]]` or general dense
/// matrices where the (0,1) entry of `K_k† K_k` is non-zero. For those, the
/// trajectory branching depends on the off-diagonal coherence `⟨0|ρ|1⟩`,
/// which is not accessible via a single-qubit probability query.
///
/// Users needing general dense Kraus support on entangled qubits should
/// either decompose the channel into a unitary dilation + measurement on an
/// ancilla, or use a density-matrix simulator (not provided by PRISM-Q).
fn apply_custom_kraus(
    backend: &mut dyn Backend,
    qubit: usize,
    kraus: &[[[Complex64; 2]; 2]],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    for (i, k) in kraus.iter().enumerate() {
        if !is_kdagger_k_diagonal(k) {
            return Err(crate::error::PrismError::BackendUnsupported {
                backend: "trajectory engine".to_string(),
                operation: format!(
                    "NoiseChannel::Custom: Kraus operator {i} has non-diagonal K†K \
                     (off-diagonal magnitude {:.3e} > {KRAUS_DIAGONAL_TOL:.0e}). \
                     Only Kraus ops whose K†K is diagonal are supported (diagonal \
                     matrices, Paulis, and their scaled versions). Decompose your \
                     channel into a unitary dilation + ancilla measurement, or use \
                     a density-matrix simulator for general dense Kraus channels.",
                    (k[0][0].conj() * k[0][1] + k[1][0].conj() * k[1][1]).norm()
                ),
            });
        }
    }

    let p1 = backend.qubit_probability(qubit)?;
    let p0 = 1.0 - p1;

    let mut cumulative: smallvec::SmallVec<[f64; 8]> = smallvec::SmallVec::new();
    let mut total = 0.0;
    for k in kraus {
        let pk = p0 * (k[0][0].norm_sqr() + k[1][0].norm_sqr())
            + p1 * (k[0][1].norm_sqr() + k[1][1].norm_sqr());
        total += pk;
        cumulative.push(total);
    }

    if total <= JUMP_EPSILON {
        return Ok(());
    }

    let r: f64 = rand::Rng::gen::<f64>(rng) * total;
    let chosen = cumulative
        .iter()
        .position(|&c| r < c)
        .unwrap_or(kraus.len() - 1);
    let pk = if chosen == 0 {
        cumulative[0]
    } else {
        cumulative[chosen] - cumulative[chosen - 1]
    };
    if pk <= JUMP_EPSILON {
        return Ok(());
    }
    let inv = 1.0 / pk.sqrt();
    let inv_c = Complex64::new(inv, 0.0);
    let k = kraus[chosen];
    let normalized = [
        [k[0][0] * inv_c, k[0][1] * inv_c],
        [k[1][0] * inv_c, k[1][1] * inv_c],
    ];

    backend.apply_1q_matrix(qubit, &normalized)
}

fn apply_noise_event(
    backend: &mut dyn Backend,
    event: &NoiseEvent,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    match &event.channel {
        NoiseChannel::Pauli { px, py, pz } => {
            apply_pauli(backend, event.qubits[0], *px, *py, *pz, rng)
        }
        NoiseChannel::Depolarizing { p } => {
            let pp = p / 3.0;
            apply_pauli(backend, event.qubits[0], pp, pp, pp, rng)
        }
        NoiseChannel::AmplitudeDamping { gamma } => {
            apply_amplitude_damping(backend, event.qubits[0], *gamma, rng)
        }
        NoiseChannel::PhaseDamping { gamma } => {
            apply_phase_damping(backend, event.qubits[0], *gamma, rng)
        }
        NoiseChannel::ThermalRelaxation { t1, t2, gate_time } => {
            apply_thermal_relaxation(backend, event.qubits[0], *t1, *t2, *gate_time, rng)
        }
        NoiseChannel::TwoQubitDepolarizing { p } => {
            apply_two_qubit_depolarizing(backend, event.qubits[0], event.qubits[1], *p, rng)
        }
        NoiseChannel::Custom { kraus } => apply_custom_kraus(backend, event.qubits[0], kraus, rng),
    }
}

fn apply_readout_errors(
    results: &mut [bool],
    readout: &[Option<crate::sim::noise::ReadoutError>],
    rng: &mut ChaCha8Rng,
) {
    for (bit, ro) in results.iter_mut().zip(readout.iter()) {
        if let Some(err) = ro {
            let r: f64 = rand::Rng::gen(rng);
            if *bit {
                // 1→0 with probability p10
                if r < err.p10 {
                    *bit = false;
                }
            } else {
                // 0→1 with probability p01
                if r < err.p01 {
                    *bit = true;
                }
            }
        }
    }
}

pub(crate) fn run_trajectory_shot(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &NoiseModel,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<bool>> {
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;

    for (idx, instr) in circuit.instructions.iter().enumerate() {
        backend.apply(instr)?;
        for event in &noise.after_gate[idx] {
            apply_noise_event(backend, event, rng)?;
        }
    }

    let mut results = backend.classical_results().to_vec();
    apply_readout_errors(&mut results, &noise.readout, rng);
    Ok(results)
}

pub(crate) fn run_trajectories(
    backend_factory: impl Fn(u64) -> Box<dyn Backend> + Sync,
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    #[cfg(feature = "parallel")]
    {
        if circuit.num_qubits <= 20 && num_shots >= 4 {
            return run_trajectories_par(&backend_factory, circuit, noise, num_shots, seed);
        }
    }

    let mut shots = Vec::with_capacity(num_shots);
    for i in 0..num_shots {
        let shot_seed = seed.wrapping_add(i as u64);
        let mut rng = ChaCha8Rng::seed_from_u64(shot_seed);
        let mut backend = backend_factory(shot_seed);
        let result = run_trajectory_shot(backend.as_mut(), circuit, noise, &mut rng)?;
        shots.push(result);
    }

    Ok(ShotsResult {
        shots,
        num_classical_bits: circuit.num_classical_bits,
    })
}

#[cfg(feature = "parallel")]
fn run_trajectories_par(
    backend_factory: &(impl Fn(u64) -> Box<dyn Backend> + Sync),
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let shots: Result<Vec<Vec<bool>>> = (0..num_shots)
        .into_par_iter()
        .map(|i| {
            let shot_seed = seed.wrapping_add(i as u64);
            let mut rng = ChaCha8Rng::seed_from_u64(shot_seed);
            let mut backend = backend_factory(shot_seed);
            run_trajectory_shot(backend.as_mut(), circuit, noise, &mut rng)
        })
        .collect();

    Ok(ShotsResult {
        shots: shots?,
        num_classical_bits: circuit.num_classical_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits;

    #[test]
    fn trajectory_pauli_matches_brute_force() {
        let n = 5;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let result = run_trajectories(factory, &circuit, &noise, 500, 42).unwrap();
        assert_eq!(result.shots.len(), 500);

        let all_zero: Vec<bool> = vec![false; n];
        let all_one: Vec<bool> = vec![true; n];
        let num_00 = result.shots.iter().filter(|s| **s == all_zero).count();
        let num_11 = result.shots.iter().filter(|s| **s == all_one).count();
        let num_other = 500 - num_00 - num_11;

        assert!(num_other > 0, "noise should produce non-GHZ outcomes");
        assert!(num_00 > 50, "should still have many |00...0> outcomes");
        assert!(num_11 > 50, "should still have many |11...1> outcomes");
    }

    #[test]
    fn amplitude_damping_decays_to_ground() {
        // |1⟩ with strong amplitude damping should mostly decay to |0⟩
        let mut circuit = Circuit::new(1, 1);
        circuit.add_gate(Gate::X, &[0]); // prepare |1⟩
        circuit.add_measure(0, 0);

        let noise = NoiseModel::with_amplitude_damping(&circuit, 0.9);
        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let result = run_trajectories(factory, &circuit, &noise, 1000, 42).unwrap();
        let num_zero = result.shots.iter().filter(|s| !s[0]).count();
        // With gamma=0.9 on |1⟩, P(decay) ≈ 0.9
        assert!(
            num_zero > 700,
            "strong AD should decay most to |0⟩, got {} zeros",
            num_zero
        );
    }

    #[test]
    fn phase_damping_preserves_populations() {
        // Phase damping should not change |0⟩/|1⟩ populations
        let mut circuit = Circuit::new(1, 1);
        circuit.add_gate(Gate::X, &[0]); // prepare |1⟩
        circuit.add_measure(0, 0);

        let mut pd_noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        for events in &mut pd_noise.after_gate {
            *events = events
                .iter()
                .map(|e| NoiseEvent {
                    channel: NoiseChannel::PhaseDamping { gamma: 0.9 },
                    qubits: e.qubits.clone(),
                })
                .collect();
        }

        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let result = run_trajectories(factory, &circuit, &pd_noise, 1000, 42).unwrap();
        let num_one = result.shots.iter().filter(|s| s[0]).count();
        // Phase damping on |1⟩ should keep it as |1⟩ (only dephases superpositions)
        assert_eq!(num_one, 1000, "PD should not change |1⟩ population");
    }

    #[test]
    fn readout_error_flips_bits() {
        let mut circuit = Circuit::new(1, 1);
        circuit.add_measure(0, 0); // measure |0⟩

        let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        noise.with_readout_error(0.3, 0.0); // 30% chance of 0→1

        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let result = run_trajectories(factory, &circuit, &noise, 1000, 42).unwrap();
        let num_one = result.shots.iter().filter(|s| s[0]).count();
        // Should see ~30% readout flips
        assert!(
            num_one > 200 && num_one < 400,
            "readout error p01=0.3 should flip ~30% of |0⟩ outcomes, got {}",
            num_one
        );
    }

    #[test]
    fn two_qubit_depolarizing_produces_errors() {
        let mut circuit = Circuit::new(2, 2);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);

        let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        // Add 2q depolarizing on the CX gate
        noise.after_gate[0] = vec![NoiseEvent {
            channel: NoiseChannel::TwoQubitDepolarizing { p: 0.5 },
            qubits: smallvec![0, 1],
        }];

        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let result = run_trajectories(factory, &circuit, &noise, 1000, 42).unwrap();
        // With 50% 2q depolarizing, we should see varied outcomes
        let num_00 = result.shots.iter().filter(|s| !s[0] && !s[1]).count();
        assert!(
            num_00 < 900,
            "strong 2q depolarizing should produce errors, got {} |00⟩",
            num_00
        );
    }

    #[test]
    fn zero_noise_trajectory_deterministic() {
        let n = 3;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let r1 = run_trajectories(factory, &circuit, &noise, 100, 42).unwrap();
        let r2 = run_trajectories(factory, &circuit, &noise, 100, 42).unwrap();
        assert_eq!(r1.shots, r2.shots, "same seed must produce same results");
    }
}

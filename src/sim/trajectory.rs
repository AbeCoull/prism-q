//! Monte Carlo trajectory execution for noisy circuits: one pure-state
//! simulation per shot, sampling a noise branch after each instruction.

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::smallvec;

use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::gates::Gate;
use crate::sim::ShotsResult;
use crate::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Noise sampler for one shot, on a ChaCha stream of its own.
///
/// The backend is seeded from the same shot seed and is also `ChaCha8Rng`, so on
/// the default stream the two emit the same sequence and a branch draw lands on
/// the same value as the measurement draw after it. Amplitude damping at
/// `gamma = 0.15` on `H|0>` read `P(1) = 0.384` over 200k shots against an exact
/// 0.425. A second stream keeps one seed reproducible and the draws independent.
pub(crate) fn noise_rng(shot_seed: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(shot_seed);
    rng.set_stream(1);
    rng
}

fn apply_pauli(
    backend: &mut dyn Backend,
    qubit: usize,
    px: f64,
    py: f64,
    pz: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let r: f64 = rand::RngExt::random(rng);
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

/// Minimum p_jump for applying the jump branch instead of the no-jump
/// branch. Below this, the jump is numerically indistinguishable from zero and
/// dividing by sqrt(p_jump) produces NaN/inf.
const JUMP_EPSILON: f64 = 1e-12;

/// Apply a single-qubit diagonal Kraus channel with 2 operators of the form:
///   K_0 = diag(1, c), no-jump
///   K_1 = [[0, s01_upper], [0, s11_lower]], jump (AD)
///       or diag(0, s), jump (PD, pass s01_upper=0)
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
    let r: f64 = rand::RngExt::random(rng);

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

/// Unravel thermal relaxation as the amplitude-damping and phase-damping
/// composition [`crate::sim::noise::kraus_1q`] lowers it to, so the trajectory
/// average reproduces `exp(-t/t1)` populations and `exp(-t/t2)` coherences. A
/// mixture of reset and `Z` cannot: it needs a negative dephasing probability
/// whenever `t1 < t2 <= 2*t1`.
///
/// The composition has four Kraus operators, but every one of them has a
/// diagonal `Kdagger K`, so the branch probabilities read off `P(1)` alone and
/// need no reduced density matrix. The two that move population are both
/// proportional to `|0><1|` and leave the same normalized state, so they merge.
/// That leaves three branches behind one probability read and one matrix pass.
fn apply_thermal_relaxation(
    backend: &mut dyn Backend,
    qubit: usize,
    t1: f64,
    t2: f64,
    gate_time: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    if t1 <= 0.0 || t2 <= 0.0 || gate_time <= 0.0 {
        return Ok(());
    }
    let (gad, gpd) = crate::sim::noise::thermal_relaxation_rates(t1, t2, gate_time);
    let p1 = backend.qubit_probability(qubit)?;
    let p_relax = gad * p1;
    let p_dephase = (1.0 - gad) * gpd * p1;

    let zero = Complex64::new(0.0, 0.0);
    let r: f64 = rand::RngExt::random(rng);
    let mat = if r < p_relax {
        let s = (1.0 / p1).sqrt();
        [[zero, Complex64::new(s, 0.0)], [zero, zero]]
    } else if r < p_relax + p_dephase {
        let s = (1.0 / p1).sqrt();
        [[zero, zero], [zero, Complex64::new(s, 0.0)]]
    } else {
        let denom = 1.0 - p_relax - p_dephase;
        if denom <= JUMP_EPSILON {
            return Ok(());
        }
        let inv = 1.0 / denom.sqrt();
        let keep = ((1.0 - gad) * (1.0 - gpd)).sqrt() * inv;
        [
            [Complex64::new(inv, 0.0), zero],
            [zero, Complex64::new(keep, 0.0)],
        ]
    };
    backend.apply_1q_matrix(qubit, &mat)
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
    let r: f64 = rand::RngExt::random(rng);

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

/// Compute the branch effect `Kdagger K` for a single-qubit Kraus operator.
#[inline]
pub(crate) fn kdagger_k(k: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [
            k[0][0].conj() * k[0][0] + k[1][0].conj() * k[1][0],
            k[0][0].conj() * k[0][1] + k[1][0].conj() * k[1][1],
        ],
        [
            k[0][1].conj() * k[0][0] + k[1][1].conj() * k[1][0],
            k[0][1].conj() * k[0][1] + k[1][1].conj() * k[1][1],
        ],
    ]
}

#[inline]
fn kraus_probability(k: &[[Complex64; 2]; 2], rho: &[[Complex64; 2]; 2]) -> f64 {
    let effect = kdagger_k(k);
    let p = effect[0][0] * rho[0][0]
        + effect[0][1] * rho[1][0]
        + effect[1][0] * rho[0][1]
        + effect[1][1] * rho[1][1];
    p.re.max(0.0)
}

/// Sample and apply one of a set of 1-qubit Kraus operators.
///
/// Branch probabilities use `p_k = Tr(Kdagger K rho_q)`, where `rho_q` is
/// the qubit's reduced density matrix. This handles dense Kraus operators
/// whose branch probabilities depend on coherence.
fn apply_custom_kraus(
    backend: &mut dyn Backend,
    qubit: usize,
    kraus: &[[[Complex64; 2]; 2]],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let rho = backend.reduced_density_matrix_1q(qubit)?;

    let mut cumulative: smallvec::SmallVec<[f64; 8]> = smallvec::SmallVec::new();
    let mut total = 0.0;
    for k in kraus {
        let pk = kraus_probability(k, &rho);
        total += pk;
        cumulative.push(total);
    }

    if total <= JUMP_EPSILON {
        return Ok(());
    }

    let r: f64 = rand::RngExt::random::<f64>(rng) * total;
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

pub(crate) fn apply_readout_errors(
    results: &mut [bool],
    readout: &[Option<crate::sim::noise::ReadoutError>],
    rng: &mut ChaCha8Rng,
) {
    for (bit, ro) in results.iter_mut().zip(readout.iter()) {
        if let Some(err) = ro {
            let r: f64 = rand::RngExt::random(rng);
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

/// Shot-level parallelism stops where kernel-level parallelism starts. At and
/// above the backend parallel threshold the statevector kernels already
/// saturate the pool, and nesting shot tasks inside kernel joins piles stolen
/// shot frames onto one worker stack until it overflows. Same guard as
/// `MAX_BLOCK_QUBITS_FOR_PAR` in the decomposed path.
/// Replica cap for parallel trajectories. Each Rayon thread holds its own
/// backend, so peak memory is `threads * state(num_qubits)` rather than one
/// state. Bounding the qubit count bounds the replica set: at 14 qubits a
/// statevector replica is 256 KiB, so even a large thread pool stays in the
/// tens of megabytes. Above this the trajectories run serially, one live
/// backend at a time, and the backend's own `init` cap is the only limit.
#[cfg(feature = "parallel")]
const MAX_QUBITS_FOR_PAR_SHOTS: usize = 14;

/// `force_serial` keeps every trajectory on one thread. Device-resident
/// backends set it: parallel trajectories would allocate one device state per
/// Rayon thread against a single-state VRAM verdict, and concurrent launches
/// on one device serialize anyway.
pub(crate) fn run_trajectories(
    backend_factory: impl Fn(u64) -> Box<dyn Backend> + Sync,
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
    force_serial: bool,
    route: crate::sim::ResolvedBackend,
) -> Result<ShotsResult> {
    #[cfg(not(feature = "parallel"))]
    let _ = force_serial;
    #[cfg(feature = "parallel")]
    {
        if !force_serial && circuit.num_qubits < MAX_QUBITS_FOR_PAR_SHOTS && num_shots >= 4 {
            return run_trajectories_par(&backend_factory, circuit, noise, num_shots, seed, route);
        }
    }

    let mut shots = Vec::with_capacity(num_shots);
    let mut metadata = crate::sim::RunMetadata::exact(route);
    for i in 0..num_shots {
        let shot_seed = seed.wrapping_add(i as u64);
        let mut rng = noise_rng(shot_seed);
        let mut backend = backend_factory(shot_seed);
        let result = run_trajectory_shot(backend.as_mut(), circuit, noise, &mut rng)?;
        let shot_metadata = crate::sim::backend_metadata(backend.as_ref());
        if i == 0 {
            metadata = shot_metadata;
        } else {
            metadata.weaken_with(&shot_metadata);
        }
        shots.push(result);
    }

    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata))
}

#[cfg(feature = "parallel")]
fn run_trajectories_par(
    backend_factory: &(impl Fn(u64) -> Box<dyn Backend> + Sync),
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
    route: crate::sim::ResolvedBackend,
) -> Result<ShotsResult> {
    let results: Result<Vec<(Vec<bool>, crate::sim::RunMetadata)>> = (0..num_shots)
        .into_par_iter()
        .map(|i| {
            let shot_seed = seed.wrapping_add(i as u64);
            let mut rng = noise_rng(shot_seed);
            let mut backend = backend_factory(shot_seed);
            let bits = run_trajectory_shot(backend.as_mut(), circuit, noise, &mut rng)?;
            Ok((bits, crate::sim::backend_metadata(backend.as_ref())))
        })
        .collect();

    let mut metadata = crate::sim::RunMetadata::exact(route);
    let mut shots = Vec::with_capacity(num_shots);
    for (index, (bits, shot_metadata)) in results?.into_iter().enumerate() {
        if index == 0 {
            metadata = shot_metadata;
        } else {
            metadata.weaken_with(&shot_metadata);
        }
        shots.push(bits);
    }
    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits;

    // The replica cap is only a memory bound if it stays at or below the cap
    // governing a single state. Raising `MAX_QUBITS_FOR_PAR_SHOTS` past the
    // statevector cap would let the parallel path allocate one oversize state
    // per thread, which is the failure `run_trajectories` avoids by falling
    // back to serial execution.
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_trajectory_replicas_stay_within_a_single_state_cap() {
        let cap = crate::backend::max_statevector_qubits();
        assert!(
            MAX_QUBITS_FOR_PAR_SHOTS <= cap,
            "parallel replica cap {MAX_QUBITS_FOR_PAR_SHOTS} exceeds the statevector cap {cap}"
        );
        let replica_bytes = (1usize << MAX_QUBITS_FOR_PAR_SHOTS) * size_of::<Complex64>();
        assert_eq!(
            replica_bytes,
            256 * 1024,
            "replica size documented as 256 KiB"
        );
    }

    #[test]
    fn trajectory_pauli_matches_brute_force() {
        let n = 5;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.measure_all();

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.02);
        let num_shots = 5000;

        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };
        let trajectory = run_trajectories(
            factory,
            &circuit,
            &noise,
            num_shots,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        let brute = crate::sim::noise::run_shots_noisy_brute_with(
            |s| Box::new(crate::backend::stabilizer::StabilizerBackend::new(s)),
            &circuit,
            &noise,
            num_shots,
            42,
        )
        .unwrap();

        let traj_coh = trajectory.coherent_fraction();
        let brute_coh = brute.coherent_fraction();
        // 5 sigma for a pairwise comparison at 5000 shots.
        assert!(
            (traj_coh - brute_coh).abs() < 0.05,
            "coherent fraction: trajectory={traj_coh:.3}, brute={brute_coh:.3}"
        );
        assert!(traj_coh < 1.0, "noise should produce non-GHZ outcomes");

        for bit in 0..n {
            let t = trajectory.marginal(bit);
            let b = brute.marginal(bit);
            assert!(
                (t - b).abs() < 0.05,
                "bit {bit}: trajectory marginal {t:.3} vs brute {b:.3}"
            );
        }
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

        let result = run_trajectories(
            factory,
            &circuit,
            &noise,
            1000,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
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

        let result = run_trajectories(
            factory,
            &circuit,
            &pd_noise,
            1000,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
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

        let result = run_trajectories(
            factory,
            &circuit,
            &noise,
            1000,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
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

        let result = run_trajectories(
            factory,
            &circuit,
            &noise,
            1000,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        // With 50% 2q depolarizing, varied outcomes should appear.
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
        circuit.measure_all();

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let r1 = run_trajectories(
            factory,
            &circuit,
            &noise,
            100,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        let r2 = run_trajectories(
            factory,
            &circuit,
            &noise,
            100,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        assert_eq!(r1.shots, r2.shots, "same seed must produce same results");
    }
}

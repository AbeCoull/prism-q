//! Kraus forms of the noise channels and exact density-matrix evolution under a noise model.

use num_complex::Complex64;

use crate::backend::Backend;
use crate::backend::density_matrix::{
    DeferredSuperoperators, DensityMatrixBackend, MIN_DEFERRED_QUBITS, PairChannel,
};
use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::gates::Gate;
use crate::limits::ResolvedLimits;
use crate::sim::BackendKind;

use super::{NoiseChannel, NoiseEvent, NoiseModel};

fn amplitude_damping_kraus(gamma: f64) -> [[[Complex64; 2]; 2]; 2] {
    let s = (1.0 - gamma).max(0.0).sqrt();
    let g = gamma.max(0.0).sqrt();
    [
        [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(s, 0.0)],
        ],
        [
            [Complex64::new(0.0, 0.0), Complex64::new(g, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        ],
    ]
}

/// Kraus set of generalized amplitude damping: decay toward `|0>` at rate
/// `gamma` weighted by `1 - excited`, excitation toward `|1>` at the same rate
/// weighted by `excited`. At `excited == 0` the last two operators vanish and
/// the set is the amplitude-damping pair.
fn generalized_amplitude_damping_kraus(gamma: f64, excited: f64) -> [[[Complex64; 2]; 2]; 4] {
    let s = (1.0 - gamma).max(0.0).sqrt();
    let g = gamma.max(0.0).sqrt();
    let cold = (1.0 - excited).max(0.0).sqrt();
    let hot = excited.max(0.0).sqrt();
    let c = |re: f64| Complex64::new(re, 0.0);
    [
        [[c(cold), c(0.0)], [c(0.0), c(cold * s)]],
        [[c(0.0), c(cold * g)], [c(0.0), c(0.0)]],
        [[c(hot * s), c(0.0)], [c(0.0), c(hot)]],
        [[c(0.0), c(0.0)], [c(hot * g), c(0.0)]],
    ]
}

fn phase_damping_kraus(gamma: f64) -> [[[Complex64; 2]; 2]; 2] {
    let s = (1.0 - gamma).max(0.0).sqrt();
    let g = gamma.max(0.0).sqrt();
    [
        [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(s, 0.0)],
        ],
        [
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(g, 0.0)],
        ],
    ]
}

/// Amplitude-damping and phase-damping rates whose composition decays
/// populations as `exp(-t/t1)` and coherences as `exp(-t/t2)`. Shared by the
/// exact Kraus lowering and the trajectory unraveling so the two routes cannot
/// disagree on what a declared channel means. Requires `t2 <= 2*t1`, which
/// [`NoiseChannel::validate`] enforces; the clamp only absorbs rounding at the
/// boundary.
pub(crate) fn thermal_relaxation_rates(t1: f64, t2: f64, gate_time: f64) -> (f64, f64) {
    let gad = 1.0 - (-gate_time / t1).exp();
    let gpd = (1.0 - (gate_time / t1 - 2.0 * gate_time / t2).exp()).clamp(0.0, 1.0);
    (gad, gpd)
}

/// Lower a single-qubit noise channel to its Kraus operators for exact
/// density-matrix evolution. Thermal relaxation (`t2 <= 2*t1`, zero
/// temperature) is the amplitude-damping channel composed with a pure-dephasing
/// channel chosen so populations decay as `exp(-t/t1)` and coherences as
/// `exp(-t/t2)`.
pub(crate) fn kraus_1q(channel: &NoiseChannel) -> Vec<[[Complex64; 2]; 2]> {
    let id = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let x = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let y = [
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
    ];
    let z = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ];
    let scale = |m: [[Complex64; 2]; 2], s: f64| {
        let s = Complex64::new(s, 0.0);
        [[m[0][0] * s, m[0][1] * s], [m[1][0] * s, m[1][1] * s]]
    };
    let pauli = |px: f64, py: f64, pz: f64| {
        let pi = (1.0 - px - py - pz).max(0.0);
        vec![
            scale(id, pi.sqrt()),
            scale(x, px.sqrt()),
            scale(y, py.sqrt()),
            scale(z, pz.sqrt()),
        ]
    };
    match channel {
        NoiseChannel::Pauli { px, py, pz } => pauli(*px, *py, *pz),
        NoiseChannel::Depolarizing { p } => pauli(p / 3.0, p / 3.0, p / 3.0),
        NoiseChannel::AmplitudeDamping { gamma } => amplitude_damping_kraus(*gamma).to_vec(),
        NoiseChannel::PhaseDamping { gamma } => phase_damping_kraus(*gamma).to_vec(),
        NoiseChannel::ThermalRelaxation {
            t1,
            t2,
            gate_time,
            excited_population,
        } => {
            let (gad, gpd) = thermal_relaxation_rates(*t1, *t2, *gate_time);
            let ad = generalized_amplitude_damping_kraus(gad, *excited_population);
            let pd = phase_damping_kraus(gpd);
            let mut out = Vec::with_capacity(8);
            for a in &ad {
                for p in &pd {
                    let k = crate::gates::mat_mul_2x2(a, p);
                    // A zero operator is a branch that never fires, and at zero
                    // temperature half the set is zero.
                    if k.iter().flatten().any(|e| e.norm_sqr() > 0.0) {
                        out.push(k);
                    }
                }
            }
            out
        }
        NoiseChannel::Custom { kraus } => kraus.clone(),
        NoiseChannel::TwoQubitDepolarizing { .. }
        | NoiseChannel::Kraus2q { .. }
        | NoiseChannel::LeakageTransport { .. } => {
            unreachable!("two-qubit channels have no single-qubit Kraus lowering")
        }
        NoiseChannel::Leakage { .. }
        | NoiseChannel::Seepage { .. }
        | NoiseChannel::QuasiStatic { .. } => {
            unreachable!("evolve_density_matrix rejects channels with per-shot memory")
        }
    }
}

/// Apply one noise event to a density-matrix backend exactly.
pub(crate) fn apply_noise_event_dm(dm: &mut DensityMatrixBackend, event: &NoiseEvent) {
    match &event.channel {
        NoiseChannel::TwoQubitDepolarizing { p } => {
            dm.apply_2q_depolarizing(event.qubits[0], event.qubits[1], *p);
        }
        NoiseChannel::Kraus2q { kraus } => {
            dm.apply_2q_kraus(event.qubits[0], event.qubits[1], kraus);
        }
        other => {
            let kraus = kraus_1q(other);
            dm.apply_1q_kraus(event.qubits[0], &kraus);
        }
    }
}

pub(in crate::sim) fn validate_mixture_inputs(
    circuit: &Circuit,
    noise: Option<&NoiseModel>,
    initial_state: Option<crate::sim::StartState<'_>>,
) -> Result<()> {
    if !circuit.has_terminal_measurements_under_reset_channel() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: "density_matrix".into(),
            reason: "the mixture holds every measurement branch at once, so mid-circuit \
                     measurement and classical conditioning cannot be replayed from it; move the \
                     measurements to the end of the circuit, or run trajectories on a backend \
                     with a per-shot pure state"
                .into(),
        });
    }
    if let Some(noise) = noise {
        noise.validate_for(circuit)?;
        noise.ensure_memoryless("density_matrix")?;
    }

    if let Some(state) = initial_state {
        crate::sim::check_initial_state_len(state, circuit.num_qubits)?;
        match state {
            crate::sim::StartState::Amplitudes(a) => {
                crate::backend::validate_initial_amplitudes(a)?
            }
            crate::sim::StartState::DensityMatrix(rho) => {
                crate::backend::validate_initial_density_matrix(rho)?;
            }
        }
    }
    Ok(())
}

/// Evolve a density-matrix backend through `circuit` and optional `noise`,
/// starting from `initial_state` when one is given and |0...0> otherwise.
///
/// Measurements are not collapsed; observables and marginals are read off the
/// final mixed state, so the circuit must carry terminal measurements only and
/// no classical conditionals, which
/// [`Circuit::has_terminal_measurements_under_reset_channel`] decides and this
/// function rejects. The circuit is checked against the
/// density-matrix qubit cap and the noise model is validated before any
/// allocation.
pub(crate) fn evolve_density_matrix(
    kind: &BackendKind,
    circuit: &Circuit,
    noise: Option<&NoiseModel>,
    initial_state: Option<crate::sim::StartState<'_>>,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<DensityMatrixBackend> {
    use crate::sim::dispatch::{Accel, Family, accel_for, build_density_matrix};

    let accel = accel_for(kind, Family::DensityMatrix, circuit.num_qubits);
    if matches!(accel, Accel::Cpu) {
        crate::backend::check_state_allocation(
            "density_matrix",
            circuit.num_qubits,
            limits.density_matrix_qubits,
            crate::limits::LimitKnob::DensityMatrixQubits,
        )?;
    }
    validate_mixture_inputs(circuit, noise, None)?;

    let mut dm = build_density_matrix(&accel, seed, limits);
    match initial_state {
        Some(state) => {
            crate::sim::check_initial_state_len(state, circuit.num_qubits)?;
            state.load(&mut dm, circuit.num_classical_bits)?;
        }
        None => dm.init(circuit.num_qubits, circuit.num_classical_bits)?,
    }
    let Some(noise) = noise else {
        let expanded = crate::sim::expand_for_backend(&dm, circuit);
        let fused = crate::sim::fuse_for_backend(&dm, &expanded);
        for inst in &fused.instructions {
            if !matches!(inst, Instruction::Measure { .. }) {
                dm.apply(inst)?;
            }
        }
        return Ok(dm);
    };

    let defer = circuit.num_qubits >= MIN_DEFERRED_QUBITS;
    let mut deferred = DeferredSuperoperators::new(circuit.num_qubits);
    for (inst, events) in circuit.instructions.iter().zip(&noise.after_gate) {
        let mut trailing: &[NoiseEvent] = events;
        let mut folded_pair = None;
        match inst {
            Instruction::Gate { gate, targets } => match *targets.as_slice() {
                [q] if (defer || events.iter().any(|e| e.qubits[..] == [q]))
                    && deferred.defer_gate(gate, q) => {}
                [q0, q1] if defer && fold_2q(&mut dm, &mut deferred, gate, q0, q1, events) => {
                    folded_pair = Some((q0, q1));
                }
                _ => {
                    deferred.flush_qubits(&mut dm, targets);
                    dm.apply(inst)?;
                }
            },
            Instruction::Measure { .. } | Instruction::Barrier { .. } => {}
            _ => {
                deferred.flush(&mut dm);
                dm.apply(inst)?;
            }
        }
        if let Some((q0, q1)) = folded_pair {
            let split = events
                .iter()
                .position(|e| straddles(e, q0, q1))
                .unwrap_or(events.len());
            for event in events[..split]
                .iter()
                .filter(|e| e.qubits.iter().all(|&q| q != q0 && q != q1))
            {
                defer_event(&mut dm, &mut deferred, event);
            }
            trailing = &events[split..];
        }
        for event in trailing {
            defer_event(&mut dm, &mut deferred, event);
        }
        if !defer {
            deferred.flush(&mut dm);
        }
    }
    deferred.flush(&mut dm);
    Ok(dm)
}

/// Whether `event` touches exactly one qubit of `(q0, q1)` along with one
/// outside it, which a sweep over the pair can neither absorb nor commute past.
fn straddles(event: &NoiseEvent, q0: usize, q1: usize) -> bool {
    let inside = event.qubits.iter().filter(|&&q| q == q0 || q == q1).count();
    inside > 0 && inside < event.qubits.len()
}

/// Fold the two-qubit `gate` on `(q0, q1)` and the channels in `events` that
/// act on the pair, up to the first that straddles it, into one sweep. False,
/// with nothing applied, when [`DeferredSuperoperators::apply_2q`] declines.
fn fold_2q(
    dm: &mut DensityMatrixBackend,
    deferred: &mut DeferredSuperoperators,
    gate: &Gate,
    q0: usize,
    q1: usize,
    events: &[NoiseEvent],
) -> bool {
    let channels: Vec<PairChannel> = events
        .iter()
        .take_while(|e| !straddles(e, q0, q1))
        .filter(|e| e.qubits.iter().all(|&q| q == q0 || q == q1))
        .map(|e| match &e.channel {
            NoiseChannel::TwoQubitDepolarizing { p } => PairChannel::depolarizing_2q(*p),
            NoiseChannel::Kraus2q { kraus } => PairChannel::kraus_2q(kraus, e.qubits[0] != q0),
            other => PairChannel::block(e.qubits[0], &kraus_1q(other)),
        })
        .collect();
    deferred.apply_2q(dm, gate, q0, q1, &channels)
}

/// Hold a one-qubit channel on its qubit's pending map, or sweep a two-qubit
/// one after flushing both its qubits.
fn defer_event(
    dm: &mut DensityMatrixBackend,
    deferred: &mut DeferredSuperoperators,
    event: &NoiseEvent,
) {
    if event.channel.num_qubits() == 1 {
        deferred.defer_channel(event.qubits[0], &kraus_1q(&event.channel));
    } else {
        deferred.flush_qubits(dm, &event.qubits);
        apply_noise_event_dm(dm, event);
    }
}

/// Exact output distribution of `circuit` under `noise`, read off the diagonal
/// of the evolved mixture. Measurements are not collapsed during evolution, so
/// the distribution answers for a whole shot only when every measurement is
/// terminal; callers gate on that shape. Events past the last measurement of
/// every qubit they touch are dropped, see [`settle_measured_qubits`].
pub(crate) fn density_matrix_probabilities(
    kind: &BackendKind,
    circuit: &Circuit,
    noise: &NoiseModel,
    initial_state: Option<crate::sim::StartState<'_>>,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<Vec<f64>> {
    let noise = if circuit.has_terminal_measurements_under_reset_channel() {
        settle_measured_qubits(circuit, noise)?
    } else {
        std::borrow::Cow::Borrowed(noise)
    };
    evolve_density_matrix(kind, circuit, Some(&noise), initial_state, seed, limits)?.probabilities()
}

/// `noise` without the events that fire after the last measurement of every
/// qubit they act on. A shot records the bit before such an event, but the
/// mixture is read once at the end, so applying it would rewrite the outcome.
///
/// # Errors
/// `IncompatibleBackend` for an event on a qubit between two of its
/// measurements, or one coupling a measured qubit to an unmeasured one: the
/// record and the state part ways there, and the mixture holds only the state.
fn settle_measured_qubits<'a>(
    circuit: &Circuit,
    noise: &'a NoiseModel,
) -> Result<std::borrow::Cow<'a, NoiseModel>> {
    let mut first = vec![usize::MAX; circuit.num_qubits];
    let mut last = vec![usize::MAX; circuit.num_qubits];
    for (idx, inst) in circuit.instructions.iter().enumerate() {
        if let Instruction::Measure { qubit, .. } = *inst {
            first[qubit] = first[qubit].min(idx);
            last[qubit] = idx;
        }
    }
    let measured = |q: usize, idx: usize| first.get(q).is_some_and(|&m| m <= idx);
    let settled = |event: &NoiseEvent, idx: usize| {
        event
            .qubits
            .iter()
            .all(|&q| last.get(q).is_some_and(|&m| m <= idx))
    };

    let mut any_settled = false;
    for (idx, events) in noise.after_gate.iter().enumerate() {
        for event in events {
            if settled(event, idx) {
                any_settled = true;
            } else if event.qubits.iter().any(|&q| measured(q, idx)) {
                return Err(crate::error::PrismError::IncompatibleBackend {
                    backend: "density_matrix".into(),
                    reason: format!(
                        "the noise event after instruction {idx} acts on a measured qubit that \
                         is measured again or coupled to an unmeasured one, so the recorded bit \
                         and the state diverge and one mixture cannot hold both; run \
                         trajectories on a backend with a per-shot pure state"
                    ),
                });
            }
        }
    }
    if !any_settled {
        return Ok(std::borrow::Cow::Borrowed(noise));
    }
    let after_gate = noise
        .after_gate
        .iter()
        .enumerate()
        .map(|(idx, events)| {
            events
                .iter()
                .filter(|event| !settled(event, idx))
                .cloned()
                .collect()
        })
        .collect();
    Ok(std::borrow::Cow::Owned(NoiseModel {
        after_gate,
        readout: noise.readout.clone(),
    }))
}

/// Exact `Tr(rho P_k)` for each joint Pauli `P_k`, evolving the density-matrix
/// backend through `circuit` and (optionally) `noise`: the exact (zero-variance)
/// analogue of trajectory-averaged expectation values. The circuit must fit the
/// density-matrix qubit cap and contain no measurement, which returns
/// `IncompatibleBackend`; resets are applied as channels. Observables reuse the same
/// Pauli-mask reduction as [`run_expectation_values`](crate::sim::run_expectation_values).
pub fn density_matrix_expectation_values(
    circuit: &Circuit,
    observables: &[Vec<crate::PauliTerm>],
    noise: Option<&NoiseModel>,
    seed: u64,
) -> Result<Vec<f64>> {
    dm_expectation_values(
        &BackendKind::DensityMatrix,
        circuit,
        observables,
        noise,
        None,
        seed,
        &ResolvedLimits::process_default(),
    )
}

/// [`density_matrix_expectation_values`] with a caller-supplied start state,
/// which `Simulate::expectation_values` carries and the public entry point
/// leaves at |0...0>.
pub(crate) fn dm_expectation_values(
    kind: &BackendKind,
    circuit: &Circuit,
    observables: &[Vec<crate::PauliTerm>],
    noise: Option<&NoiseModel>,
    initial_state: Option<crate::sim::StartState<'_>>,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<Vec<f64>> {
    if circuit
        .instructions
        .iter()
        .any(|inst| matches!(inst, Instruction::Measure { .. }))
    {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: "density_matrix".into(),
            reason: "density-matrix expectation values require a circuit without \
                     measurements, since the evolution skips them and a measured qubit would \
                     report its pre-measurement value; remove the measurements"
                .into(),
        });
    }
    let dm = evolve_density_matrix(kind, circuit, noise, initial_state, seed, limits)?;
    let masks = observables
        .iter()
        .map(|obs| crate::sim::pauli_masks(obs, circuit.num_qubits))
        .collect::<Result<Vec<_>>>()?;
    Ok(dm.expectations_pauli(&masks))
}

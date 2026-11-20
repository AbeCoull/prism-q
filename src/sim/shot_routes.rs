//! Shot and count sampling: the shared shot source, the per-shot walk, and noisy
//! trajectories.

use super::*;

pub(crate) fn supports_compiled_measurement_sampling(circuit: &Circuit) -> bool {
    circuit.is_clifford_only()
        && !circuit.has_resets()
        && circuit.has_terminal_measurements_only()
        && circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Measure { .. }))
}

pub(super) fn supports_deferred_measurement_sampling(circuit: &Circuit) -> bool {
    circuit.is_clifford_only()
        && (circuit.has_resets() || !circuit.has_terminal_measurements_only())
        && circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Measure { .. }))
}

pub(super) fn is_clifford_sampler_kind(kind: &BackendKind) -> bool {
    if kind.is_auto() {
        return true;
    }
    match kind {
        BackendKind::Stabilizer | BackendKind::FactoredStabilizer => true,
        #[cfg(feature = "gpu")]
        BackendKind::StabilizerGpu { .. } => true,
        _ => false,
    }
}

pub(super) fn should_use_compiled_clifford_sampling(
    kind: &BackendKind,
    circuit: &Circuit,
    num_shots: usize,
) -> bool {
    num_shots >= 1
        && supports_compiled_measurement_sampling(circuit)
        && is_clifford_sampler_kind(kind)
}

pub(super) fn should_use_deferred_clifford_sampling(
    kind: &BackendKind,
    circuit: &Circuit,
    num_shots: usize,
) -> bool {
    num_shots >= 2
        && supports_deferred_measurement_sampling(circuit)
        && is_clifford_sampler_kind(kind)
}

pub(super) fn compile_measurements_for_kind(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> Result<compiled::CompiledSampler> {
    #[cfg(not(feature = "gpu"))]
    let _ = kind;

    let sampler = compiled::compile_measurements(circuit, seed)?;

    #[cfg(feature = "gpu")]
    if let BackendKind::StabilizerGpu { context } = kind {
        return Ok(sampler.with_gpu(context.clone()));
    }

    Ok(sampler)
}

/// The circuit an `Auto` per-shot run walks in place of `circuit`, once every route
/// ahead of the walk has declined: its parity ancillas folded into parity measurements
/// on the data qubits ([`rewrite_parity_ancillas`]).
pub(super) fn parity_rewrite(kind: &BackendKind, circuit: &Circuit) -> Option<Circuit> {
    if !kind.is_auto() {
        return None;
    }
    rewrite_parity_ancillas(circuit)
}

pub(super) type TerminalStatevector = (StatevectorBackend, Vec<(usize, usize)>, bool);

pub(super) fn prepare_terminal_statevector_backend(
    kind: &BackendKind,
    terminal: TerminalCircuit,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<TerminalStatevector> {
    let TerminalCircuit {
        circuit: stripped,
        meas_map,
    } = terminal;
    let (backend, expected_device, ()) =
        evolved_statevector(kind, &stripped, seed, limits, || Ok(()))?;
    Ok((backend, meas_map, expected_device))
}

/// A source that answers every measurement in a circuit from a single state,
/// in the precedence [`prepare_shot_source`] applies. Shots and counts both
/// consume it, so the two cannot disagree about which shortcut applies.
pub(super) enum ShotSource {
    /// `deferred` marks a sampler built from the measure/reset-deferred
    /// rewrite, whose renumbered measurements only read through `meas_map`.
    Compiled {
        sampler: Box<compiled::CompiledSampler>,
        meas_map: Vec<(usize, usize)>,
        deferred: bool,
    },
    /// The deferred rewrite whose guarded Paulis land after the compiled sample.
    Guarded(Box<compiled::DeferredCircuit>),
    TerminalStatevector {
        backend: Box<StatevectorBackend>,
        expected_device: bool,
        meas_map: Vec<(usize, usize)>,
    },
    /// A backend that draws basis states from its own representation.
    Native {
        backend: Box<dyn Backend>,
        meas_map: Vec<(usize, usize)>,
    },
    /// Dense output distribution of the measurement-stripped circuit, with the
    /// provenance of the run that produced it.
    TerminalProbabilities {
        probs: Probabilities,
        meas_map: Vec<(usize, usize)>,
        metadata: RunMetadata,
    },
    StabilizerRank,
    PerShot,
}

impl ShotSource {
    /// Which engine will answer, decided by `prepare_shot_source` and read here
    /// so the shot and count entry points do not re-derive it. `None` for the
    /// per-shot route, which builds one backend per shot and stamps its own.
    fn metadata(&self) -> Option<RunMetadata> {
        match self {
            ShotSource::Compiled { .. } => Some(
                RunMetadata::exact(ResolvedBackend::CompiledStabilizer)
                    .with_engine(Engine::CompiledSampler)
                    .with_reason(RouteRule::CompiledCliffordSampler),
            ),
            ShotSource::Guarded(_) => Some(
                RunMetadata::exact(ResolvedBackend::CompiledStabilizer)
                    .with_engine(Engine::NoisyCompiledSampler)
                    .with_reason(RouteRule::CompiledCliffordSampler),
            ),
            ShotSource::TerminalStatevector {
                backend,
                expected_device,
                ..
            } => Some(route_metadata(
                &**backend,
                *expected_device,
                RouteRule::TerminalStatevector,
            )),
            ShotSource::Native { backend, .. } => {
                Some(backend_metadata(&**backend).with_reason(RouteRule::NativeSampling))
            }
            ShotSource::TerminalProbabilities { metadata, .. } => Some(metadata.clone()),
            ShotSource::StabilizerRank => Some(
                RunMetadata::exact(ResolvedBackend::StabilizerRank)
                    .with_reason(RouteRule::CliffordRegister),
            ),
            ShotSource::PerShot => None,
        }
    }
}

/// Select and prepare the sampling source for `circuit`.
///
/// Preparation is real work: compiling a sampler, building and running a
/// backend, or executing the stripped circuit once. Call once per entry point
/// and match on the result rather than re-deriving the choice.
pub(super) fn prepare_shot_source(
    kind: &BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ShotSource> {
    match decide_shots(kind, circuit, num_shots, limits)? {
        ShotRoute::Compiled(deferred) => {
            let source = deferred.as_ref().unwrap_or(circuit);
            Ok(ShotSource::Compiled {
                sampler: Box::new(compile_measurements_for_kind(kind, source, seed)?),
                meas_map: source.measurement_map(),
                deferred: deferred.is_some(),
            })
        }
        ShotRoute::Guarded(deferred) => Ok(ShotSource::Guarded(deferred)),
        ShotRoute::StabilizerRank => Ok(ShotSource::StabilizerRank),
        ShotRoute::TerminalStatevector(terminal) => {
            let (backend, meas_map, expected_device) =
                prepare_terminal_statevector_backend(kind, terminal, seed, limits)?;
            Ok(ShotSource::TerminalStatevector {
                backend: Box::new(backend),
                expected_device,
                meas_map,
            })
        }
        ShotRoute::Native(plan, terminal) => {
            let mut backend = plan.build(seed, limits);
            execute(
                &mut *backend,
                &terminal.circuit,
                &SimOptions::classical_only(),
            )?;
            Ok(ShotSource::Native {
                backend,
                meas_map: terminal.meas_map,
            })
        }
        ShotRoute::TerminalProbabilities(terminal) => {
            let result = run_with_internal(
                kind.clone(),
                &terminal.circuit,
                seed,
                SimOptions::default(),
                limits,
            )?;
            Ok(match result.probabilities {
                Some(probs) => ShotSource::TerminalProbabilities {
                    probs,
                    meas_map: terminal.meas_map,
                    metadata: result.metadata,
                },
                None => ShotSource::PerShot,
            })
        }
        ShotRoute::PerShot => Ok(ShotSource::PerShot),
    }
}

/// Execute a circuit multiple times with explicit backend selection and return counts.
///
/// For Clifford circuits with terminal measurements and no resets, Auto,
/// Stabilizer, FactoredStabilizer, and explicit `StabilizerGpu` route through
/// the compiled sampler's optimized counting path. Explicit `StabilizerGpu`
/// carries its GPU context into the compiled sampler so large shot runs avoid
/// the raw tableau measurement round-trips. Other circuits fall back to
/// per-shot simulation with counting.
///
/// Optimized terminal statevector paths sample counts directly from the output
/// distribution. The distribution is equivalent to materializing shots first,
/// but finite seeded counts may differ from `run_shots_with(..., limits).counts()`.
pub(crate) fn run_counts_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<(HashMap<Vec<u64>, u64>, RunMetadata)> {
    #[cfg(feature = "distributed")]
    if matches!(kind, BackendKind::StatevectorDistributed { .. }) {
        let shots = run_shots_with(kind, circuit, num_shots, seed, limits)?;
        return Ok((shots.counts(), shots.metadata));
    }

    let folded = circuit.fold_static_guards();
    let circuit = folded.as_ref();

    let bits = circuit.num_classical_bits;
    let source = prepare_shot_source(&kind, circuit, num_shots, seed, limits)?;
    let Some(metadata) = source.metadata() else {
        let shots = run_shots_per_shot(
            kind,
            circuit,
            num_shots,
            seed,
            ShotRows::CountedOnly,
            limits,
        )?;
        return Ok((
            shots.counts(),
            shots.metadata.with_reason(RouteRule::PerShot),
        ));
    };
    let counts = match source {
        ShotSource::Compiled {
            mut sampler,
            meas_map,
            deferred,
        } => {
            if deferred {
                let packed = sampler.try_sample_bulk_packed(num_shots)?;
                counts_of(
                    packed_shots_to_classical_bits(&packed, &meas_map, bits),
                    bits,
                )
            } else {
                record_counts_to_classical_bits(
                    sampler.try_sample_counts(num_shots)?,
                    &meas_map,
                    bits,
                )
            }
        }
        ShotSource::TerminalStatevector {
            backend, meas_map, ..
        } => {
            if backend.is_gpu_resident() {
                let probs = backend.probabilities()?;
                sample_counts_from_probs(&probs, &meas_map, bits, num_shots, seed, limits)
            } else {
                sample_counts_from_state(
                    backend.state_vector(),
                    backend.probability_scale(),
                    &meas_map,
                    bits,
                    num_shots,
                    seed,
                    limits,
                )
            }
        }
        ShotSource::Native {
            mut backend,
            meas_map,
        } => {
            let samples = backend.sample_basis_states(num_shots, seed)?;
            counts_from_basis_samples(&samples, &meas_map, bits)
        }
        ShotSource::TerminalProbabilities {
            probs, meas_map, ..
        } => counts_of(sample_shots(&probs, &meas_map, bits, num_shots, seed), bits),
        ShotSource::Guarded(deferred) => {
            noise::run_shots_noisy_guarded(&deferred, num_shots, seed)?.counts()
        }
        ShotSource::StabilizerRank => {
            stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed)?.counts()
        }
        ShotSource::PerShot => unreachable!("handled above"),
    };
    Ok((counts, metadata.with_shots(num_shots)))
}

pub(super) fn counts_of(
    shots: Vec<Vec<bool>>,
    num_classical_bits: usize,
) -> HashMap<Vec<u64>, u64> {
    ShotsResult::from_shots(shots, num_classical_bits).counts()
}

/// Multi-shot execution for the distributed statevector backend.
///
/// Every rank runs this function in lockstep. Circuits with only terminal
/// measurements run once and sample basis indices without gathering the dense
/// state on any rank. Circuits with mid-circuit measurements run once per shot,
/// prefused, with per-shot seeds matching the generic slow path.
#[cfg(feature = "distributed")]
pub(super) fn run_shots_distributed(
    context: std::sync::Arc<crate::distributed::DistributedContext>,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ShotsResult> {
    use crate::backend::distributed_statevector::DistributedStatevectorBackend;

    let meas_map = circuit.measurement_map();
    if meas_map.is_empty() {
        // No measurements means every shot is all false, but init must still
        // run so invalid rank counts and local qubit floor violations surface as
        // errors instead of fabricated output.
        let mut backend = DistributedStatevectorBackend::with_limits(context, seed, limits);
        backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
        return Ok(ShotsResult::from_shots(
            vec![vec![false; circuit.num_classical_bits]; num_shots],
            circuit.num_classical_bits,
        )
        .with_metadata(backend_metadata(&backend)));
    }

    if circuit.has_terminal_measurements_only() {
        let stripped = circuit.without_measurements();
        let mut backend = DistributedStatevectorBackend::with_limits(context, seed, limits);
        execute(&mut backend, &stripped, &SimOptions::classical_only())?;
        let samples = backend.sample_basis_states(num_shots, seed)?;
        return Ok(ShotsResult::from_shots(
            shots_from_basis_samples(&samples, &meas_map, circuit.num_classical_bits),
            circuit.num_classical_bits,
        )
        .with_metadata(backend_metadata(&backend)));
    }

    let probe = DistributedStatevectorBackend::with_limits(context.clone(), seed, limits);
    let expanded = expand_for_backend(&probe, circuit);
    let fused = fuse_for_backend(&probe, &expanded);
    let opts = SimOptions::classical_only();
    let mut shots = Vec::with_capacity(num_shots);
    let mut metadata = RunMetadata::exact(ResolvedBackend::Distributed);
    for i in 0..num_shots {
        let shot_seed = mix_seed(seed, i);
        let mut backend =
            DistributedStatevectorBackend::with_limits(context.clone(), shot_seed, limits);
        let result = execute_circuit(&mut backend, &fused, &opts)?;
        metadata.weaken_with(&result.metadata);
        shots.push(result.classical_bits);
    }
    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata))
}

/// Noisy shots on the distributed statevector: trajectories run serially in shot
/// order on every rank, on one sharded backend that each shot re-initializes.
///
/// Shot `i` seeds its noise stream and the backend's measurement stream from the
/// run seed and `i`, as the host trajectory loop does, so every rank draws the same
/// branches without exchanging them. The weights a draw is compared against come
/// from reductions every rank receives whole, so every rank takes the same branch
/// and issues the same collectives. Validation and the first `init` read limits each
/// rank resolves for itself, so every rank agrees on their outcome before the first
/// shot. Past that point a shot's errors follow only from the shared inputs and the
/// replicated qubit map, so each one fires on every rank at the same instruction.
#[cfg(feature = "distributed")]
pub(super) fn run_noisy_shots_distributed(
    kind: &BackendKind,
    context: std::sync::Arc<crate::distributed::DistributedContext>,
    circuit: &Circuit,
    noise_model: &NoiseModel,
    num_shots: usize,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ShotsResult> {
    use crate::backend::distributed_statevector::DistributedStatevectorBackend;

    let mut backend = DistributedStatevectorBackend::with_limits(context, seed, limits);
    let fingerprint =
        DistributedStatevectorBackend::debug_fingerprint(&(circuit, noise_model, num_shots, seed));
    let validated = decide_noisy(kind, circuit, noise_model, limits).map(|_| ());
    backend.fail_together(fingerprint, validated)?;
    let ready = backend.init(circuit.num_qubits, circuit.num_classical_bits);
    backend.fail_together(fingerprint, ready)?;
    trajectory::run_trajectories_held(
        &mut backend,
        DistributedStatevectorBackend::set_seed,
        circuit,
        noise_model,
        num_shots,
        seed,
        ResolvedBackend::Distributed,
    )
    .map(|mut result| {
        result.metadata.reason = RouteRule::PerShot;
        result
    })
}

/// Execute a circuit multiple times with explicit backend selection.
pub(crate) fn run_shots_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ShotsResult> {
    // The distributed backend runs every rank in lockstep, so shot execution
    // must not route through shortcuts that reshape the collective call
    // sequence. Dispatch directly.
    #[cfg(feature = "distributed")]
    if let BackendKind::StatevectorDistributed { context } = &kind {
        return run_shots_distributed(context.clone(), circuit, num_shots, seed, limits).map(
            |mut result| {
                result.metadata.reason = RouteRule::ExplicitBackend;
                result
            },
        );
    }

    // Once per shots call, not per shot: a guard that cannot depend on a
    // measurement is resolved here so the sampling predicates below see the
    // circuit that actually runs. A circuit with no guard borrows through.
    let folded = circuit.fold_static_guards();
    let circuit = folded.as_ref();

    let bits = circuit.num_classical_bits;
    let source = prepare_shot_source(&kind, circuit, num_shots, seed, limits)?;
    let Some(metadata) = source.metadata() else {
        return run_shots_per_shot(
            kind,
            circuit,
            num_shots,
            seed,
            ShotRows::Exchangeable,
            limits,
        )
        .map(|mut result| {
            result.metadata.reason = RouteRule::PerShot;
            result
        });
    };
    let result = match source {
        ShotSource::Compiled {
            mut sampler,
            meas_map,
            ..
        } => {
            let packed = sampler.try_sample_bulk_packed(num_shots)?;
            ShotsResult::from_shots(
                packed_shots_to_classical_bits(&packed, &meas_map, bits),
                bits,
            )
        }
        ShotSource::TerminalStatevector {
            backend, meas_map, ..
        } => {
            let shots = if backend.is_gpu_resident() {
                let probs = backend.probabilities()?;
                sample_shots_from_probs(&probs, &meas_map, bits, num_shots, seed)
            } else {
                sample_shots_from_state(
                    backend.state_vector(),
                    backend.probability_scale(),
                    &meas_map,
                    bits,
                    num_shots,
                    seed,
                )
            };
            ShotsResult::from_shots(shots, bits)
        }
        ShotSource::Native {
            mut backend,
            meas_map,
        } => {
            let samples = backend.sample_basis_states(num_shots, seed)?;
            ShotsResult::from_shots(shots_from_basis_samples(&samples, &meas_map, bits), bits)
        }
        ShotSource::TerminalProbabilities {
            probs, meas_map, ..
        } => ShotsResult::from_shots(sample_shots(&probs, &meas_map, bits, num_shots, seed), bits),
        ShotSource::Guarded(deferred) => {
            noise::run_shots_noisy_guarded(&deferred, num_shots, seed)?
        }
        ShotSource::StabilizerRank => {
            stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed)?
        }
        ShotSource::PerShot => unreachable!("handled above"),
    };
    Ok(result.with_metadata(metadata))
}

/// Whether a caller of [`run_shots_per_shot`] reads the rows themselves or only
/// counts them, which spares the shuffle that keeps branched rows exchangeable.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ShotRows {
    Exchangeable,
    CountedOnly,
}

/// Run `circuit` once per shot, which mid-circuit measurements force, or on
/// the host statevector once per outcome history
/// ([`branching::branch_shots`]).
pub(super) fn run_shots_per_shot(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
    rows: ShotRows,
    limits: &ResolvedLimits,
) -> Result<ShotsResult> {
    // Pre-compute seed-independent analysis to avoid redundant work.
    validate_explicit_backend(&kind, circuit, limits)?;

    let (decompose, has_partial_independence) = analyze_independence(circuit);

    if matches!(kind, BackendKind::StabilizerRank) {
        return stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed);
    }
    if matches!(
        kind,
        BackendKind::StochasticPauli { .. } | BackendKind::DeterministicPauli { .. }
    ) {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "Pauli propagation backends do not support mid-circuit measurements".into(),
        });
    }

    if has_temporal_clifford_opportunity(&kind, circuit, limits) {
        if decompose.is_none()
            && let Some(tc) = plan_temporal_clifford(&kind, circuit, limits)
        {
            let route = ResolvedBackend::Statevector;
            let states = [(route, circuit.num_qubits)];
            return collect_shots(
                circuit,
                num_shots,
                seed,
                route,
                &kind,
                &states,
                |shot_seed| {
                    let outcome = run_temporal_clifford(&tc, shot_seed, false)?;
                    Ok((outcome.classical_bits, outcome.metadata))
                },
            );
        }
        // Decomposable circuits with a temporal prefix keep the per-shot
        // full-pipeline route; the prefix spans blocks that decomposition
        // would otherwise split. Its blocks are partitioned and planned once.
        let opts = SimOptions::classical_only();
        let route = resolve_backend(&kind, circuit, has_partial_independence, limits).resolved();
        let plan = plan_probability_route(&kind, circuit, limits);
        let blocks = match &plan {
            ProbabilityRoute::Decomposed(components) => Some(PreparedBlocks::new(
                &kind,
                components.clone(),
                circuit,
                limits,
            )),
            _ => None,
        };
        let states = [(route, circuit.num_qubits)];
        return collect_shots(
            circuit,
            num_shots,
            seed,
            route,
            &kind,
            &states,
            |shot_seed| {
                let outcome = match &blocks {
                    Some(blocks) => blocks.run(&kind, circuit, shot_seed, &opts, limits)?,
                    None => run_route(&kind, circuit, shot_seed, opts, &plan, limits)?,
                };
                Ok((outcome.classical_bits, outcome.metadata))
            },
        );
    }

    let rewritten = parity_rewrite(&kind, circuit);
    let (circuit, decompose, has_partial_independence) = match &rewritten {
        Some(rewritten) => {
            let (decompose, partial) = analyze_independence(rewritten);
            (rewritten, decompose, partial)
        }
        None => (circuit, decompose, has_partial_independence),
    };
    let opts = SimOptions::classical_only();

    if let Some((ref comps, _)) = decompose {
        let partitions = circuit.partition_subcircuits(comps);
        let block_plans: Vec<BackendPlan> = partitions
            .iter()
            .map(|(sub, _, _)| {
                validate_explicit_backend(&kind, sub, limits)?;
                Ok(resolve_backend(&kind, sub, false, limits))
            })
            .collect::<Result<_>>()?;
        let fused_blocks: Vec<std::borrow::Cow<'_, Circuit>> = partitions
            .iter()
            .zip(&block_plans)
            .map(|((sub, _, _), plan)| {
                let probe = plan.build(seed, limits);
                let expanded = expand_for_backend(&*probe, sub);
                std::borrow::Cow::Owned(fuse_for_backend(&*probe, &expanded).into_owned())
            })
            .collect();

        if num_shots > 1
            && circuit.num_qubits > 64
            && circuit.has_terminal_measurements_only()
            && !circuit.has_resets()
            && circuit.save_count() == 0
            && block_plans.iter().zip(comps).all(|(plan, comp)| {
                plan.is_host_statevector() && comp.len() < limits.statevector_qubits.value
            })
        {
            return decomposed::sample_terminal_blocks(
                comps,
                &partitions,
                &fused_blocks,
                circuit,
                num_shots,
                seed,
                limits,
            );
        }

        let block_states: Vec<(ResolvedBackend, usize)> = block_plans
            .iter()
            .zip(comps)
            .map(|(plan, comp)| (plan.resolved(), comp.len()))
            .collect();
        collect_shots(
            circuit,
            num_shots,
            seed,
            ResolvedBackend::Decomposed,
            &kind,
            &block_states,
            |shot_seed| {
                let result = run_decomposed_prefused(
                    &block_plans,
                    comps,
                    &partitions,
                    &fused_blocks,
                    shot_seed,
                    &opts,
                    circuit,
                    limits,
                )?;
                Ok((result.classical_bits, result.metadata))
            },
        )
    } else {
        let plan = resolve_backend(&kind, circuit, has_partial_independence, limits);
        let probe = plan.build(seed, limits);
        let expanded = expand_for_backend(&*probe, circuit);
        let fused = fuse_for_backend(&*probe, &expanded);

        let route = plan.resolved();
        let states = [(route, circuit.num_qubits)];
        if plan.is_host_statevector() && circuit.num_qubits < limits.statevector_qubits.value {
            #[cfg(feature = "parallel")]
            let parallel = shots_split_across_workers(&kind, &states)
                && circuit.num_qubits >= branching::MIN_QUBITS_FOR_PARALLEL_SPLITS;
            #[cfg(not(feature = "parallel"))]
            let parallel = false;
            let branch_limits =
                branching::BranchLimits::for_width(circuit.num_qubits, parallel, limits);
            return branching::branch_shots(
                &fused,
                num_shots,
                seed,
                &branch_limits,
                rows == ShotRows::Exchangeable,
            );
        }
        collect_shots(
            circuit,
            num_shots,
            seed,
            route,
            &kind,
            &states,
            |shot_seed| {
                let mut backend = plan.build(shot_seed, limits);
                let outcome = execute_circuit(&mut *backend, &fused, &opts)?;
                Ok((outcome.classical_bits, outcome.metadata))
            },
        )
    }
}

/// Each shot evolves its own state, so `shot` returns the provenance of its own
/// run and the ensemble keeps the weakest claim across them. `route` names the
/// engine for a zero-shot request, which runs nothing to read provenance off.
///
/// Under the `parallel` feature the shots split across Rayon workers when
/// [`shots_split_across_workers`] accepts `kind` for `states`, the family and width
/// of each state one shot holds. Shot `i` always runs on `mix_seed(seed, i)` and
/// the fold runs in shot order, so the result does not depend on the thread
/// count.
pub(super) fn collect_shots(
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
    route: ResolvedBackend,
    kind: &BackendKind,
    states: &[(ResolvedBackend, usize)],
    shot: impl Fn(u64) -> Result<(Vec<bool>, RunMetadata)> + Sync,
) -> Result<ShotsResult> {
    #[cfg(feature = "parallel")]
    if num_shots > 1 && shots_split_across_workers(kind, states) {
        use rayon::prelude::*;
        let runs: Vec<Result<(Vec<bool>, RunMetadata)>> = (0..num_shots)
            .into_par_iter()
            .map(|i| shot(mix_seed(seed, i)))
            .collect();
        return fold_shots(circuit, route, runs);
    }
    #[cfg(not(feature = "parallel"))]
    let _ = (kind, states);
    fold_shots(
        circuit,
        route,
        (0..num_shots).map(|i| shot(mix_seed(seed, i))),
    )
}

pub(super) fn fold_shots(
    circuit: &Circuit,
    route: ResolvedBackend,
    runs: impl IntoIterator<Item = Result<(Vec<bool>, RunMetadata)>>,
) -> Result<ShotsResult> {
    let runs = runs.into_iter();
    let mut shots = Vec::with_capacity(runs.size_hint().0);
    let mut metadata = RunMetadata::exact(route);
    for (i, run) in runs.enumerate() {
        let (bits, shot_metadata) = run?;
        metadata.fold(i, shot_metadata);
        shots.push(bits);
    }
    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata))
}

/// Family choice for auto-routed non-Pauli noise trajectories. Restricted to
/// families whose trajectory operations (1q Kraus, qubit probability, reduced
/// density matrix, reset) are supported; the statevector leaf carries the
/// kind's acceleration unless the device backend cannot run a two-qubit Kraus
/// event the model holds. Such an event entangles its pair, so it keeps the
/// circuit off the product state as an entangling gate would.
pub(super) fn general_noise_plan(
    kind: &BackendKind,
    circuit: &Circuit,
    noise_model: &NoiseModel,
    limits: &ResolvedLimits,
) -> BackendPlan {
    let family = if !circuit.has_entangling_gates() && !noise_model.has_two_qubit_kraus() {
        Family::ProductState
    } else if circuit.num_qubits > limits.statevector_qubits.value {
        if circuit.is_sparse_friendly() && circuit.num_qubits <= MAX_SPARSE_INDEX_QUBITS {
            Family::Sparse
        } else {
            Family::Mps
        }
    } else {
        Family::Statevector
    };
    let plan = plan_for_family(kind, family, circuit.num_qubits, limits);
    if plan.is_gpu() && noise_model.has_two_qubit_kraus() && !plan.supports_two_qubit_kraus(limits)
    {
        return plan_for_family(&BackendKind::Auto, family, circuit.num_qubits, limits);
    }
    plan
}

/// Execute a noisy circuit for multiple shots with explicit backend selection.
///
/// For Clifford circuits with Auto/Stabilizer/FactoredStabilizer backends,
/// uses the compiled noisy sampler (fast O(n²·m) compile + O(events·m/64) per shot).
/// For all other cases, falls back to per-shot simulation with noise injection.
/// A circuit with resets, mid-circuit measurements or guarded Pauli gates takes
/// the compiled path through its rewrite to terminal measurements, as
/// [`noise::run_shots_noisy`] describes. Pauli noise on the host statevector, or on a
/// tensor network whose probabilities fit the dense cap, under the same limits
/// draws every shot's errors first and simulates each distinct pattern once;
/// see [`trajectory::PauliGroups`]. On the host statevector, a circuit with a
/// mid-circuit measurement, reset or condition shares the gates before the first
/// of them the same way, and each shot finishes its own trajectory from there.
/// Damping on the host statevector walks the no-jump state its shots share once,
/// and each shot finishes its own trajectory from its first jump draw; see
/// [`trajectory::run_shared_prefix`].
pub(crate) fn run_shots_with_noise(
    kind: BackendKind,
    circuit: &Circuit,
    noise_model: &noise::NoiseModel,
    num_shots: usize,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ShotsResult> {
    #[cfg(feature = "distributed")]
    if let BackendKind::StatevectorDistributed { context } = &kind {
        return run_noisy_shots_distributed(
            &kind,
            context.clone(),
            circuit,
            noise_model,
            num_shots,
            seed,
            limits,
        );
    }
    let plan = match decide_noisy(&kind, circuit, noise_model, limits)? {
        NoisyRoute::Mixture => {
            let probs = exact_noisy_probabilities(&kind, circuit, noise_model, None, seed, limits)?;
            return Ok(ShotsResult::from_shots(
                sample_exact_noisy_shots(&probs, circuit, noise_model, num_shots, seed),
                circuit.num_classical_bits,
            )
            .with_metadata(exact_mixture_metadata(&kind)));
        }
        NoisyRoute::Compiled(deferred) => {
            let (circuit, noise_model) = match deferred.as_deref() {
                Some(deferred) if !deferred.guarded.is_empty() => {
                    return noise::run_shots_noisy_guarded(deferred, num_shots, seed);
                }
                Some(deferred) => (&deferred.circuit, &deferred.noise),
                None => (circuit, noise_model),
            };
            #[cfg(feature = "gpu")]
            if let BackendKind::StabilizerGpu { context } = &kind {
                return noise::run_shots_noisy_with_gpu(
                    circuit,
                    noise_model,
                    num_shots,
                    seed,
                    context.clone(),
                );
            }
            return noise::run_validated_shots_noisy(circuit, noise_model, num_shots, seed);
        }
        NoisyRoute::Trajectory(plan) => plan,
    };
    let host_statevector = matches!(plan, BackendPlan::Statevector { .. }) && !plan.is_gpu();
    let dense_tensor_network = matches!(plan, BackendPlan::TensorNetwork { .. })
        && crate::backend::tensor_probability_len("tensor network", circuit.num_qubits, limits)
            .is_ok();
    if (host_statevector || dense_tensor_network)
        && let Some(groups) = trajectory::PauliGroups::sample(circuit, noise_model, num_shots, seed)
    {
        let build = |s| plan.build(s, limits);
        return trajectory::run_pauli_groups(
            &groups,
            circuit,
            noise_model,
            seed,
            dense_tensor_network.then_some(&build as trajectory::GroupBackendFactory<'_>),
            plan.resolved(),
            limits,
        )
        .map(|mut result| {
            result.metadata.reason = RouteRule::PerShot;
            result
        });
    }
    // A checkpoint, a group state and a shot state at once.
    if host_statevector
        && circuit.num_qubits + 1 < limits.statevector_qubits.value
        && let Some(groups) =
            trajectory::PauliGroups::sample_replayed(circuit, noise_model, num_shots, seed)
    {
        return trajectory::run_replayed_groups(
            &groups,
            circuit,
            noise_model,
            seed,
            plan.resolved(),
            limits,
        )
        .map(|mut result| {
            result.metadata.reason = RouteRule::PerShot;
            result
        });
    }
    // A walked state and a shot state at once.
    if host_statevector
        && circuit.num_qubits < limits.statevector_qubits.value
        && let Some(mut result) = trajectory::run_shared_prefix(
            circuit,
            noise_model,
            num_shots,
            seed,
            plan.resolved(),
            limits,
        )?
    {
        result.metadata.reason = RouteRule::PerShot;
        return Ok(result);
    }
    let route = plan.resolved();
    trajectory::run_trajectories(
        |s| plan.build(s, limits),
        circuit,
        noise_model,
        num_shots,
        seed,
        plan.is_gpu(),
        route,
    )
    .map(|mut result| {
        result.metadata.reason = if plan.is_gpu() && result.metadata.placement == Placement::Host {
            RouteRule::DeviceInitializationFallback
        } else {
            RouteRule::PerShot
        };
        result
    })
}

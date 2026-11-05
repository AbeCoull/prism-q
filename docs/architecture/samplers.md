# Compiled Samplers

Multi-shot sampling that does not evolve a full statevector per shot.

## Noiseless compiled sampler (`src/sim/compiled/`)

**Backward path** (`compile_measurements`): propagates Pauli Z observables backward through the circuit. Each measured qubit becomes a row of a GF(2) parity matrix M recording which input qubits it depends on; a Clifford gate conjugates a Pauli string in O(1).

**Forward path** (`compile_forward`): tracks stabilizer generator dependencies forward through the circuit and produces the same parity matrix.

**Sampling**: random bits for the independent generators, then an XOR cascade through the parity matrix.

| Strategy | Condition | Method |
|----------|-----------|--------|
| `FlipLut` | Small rank | 256-entry XOR lookup table |
| `SparseParity` | Sparse rows | Only flip non-zero columns |
| `ParityBlocks` | Blocked structure | Per-block independent sampling |

**ShotAccumulator trait**: pluggable result collection.

| Accumulator | Output | Use case |
|-------------|--------|----------|
| `HistogramAccumulator` | Bitstring → count map | Standard shot output |
| `MarginalsAccumulator` | Per-qubit P(1) | Marginal probabilities |
| `PauliExpectationAccumulator` | ⟨P⟩ for Pauli observables | VQE/QAOA |
| `CorrelatorAccumulator` | ⟨Z_i Z_j⟩ correlations | Entanglement analysis |
| `NullAccumulator` | Nothing | Benchmarking raw sampling speed |

**PackedShots raw format**: `PackedShots::RAW_FORMAT_VERSION` is the replay contract
for `raw_data()` and `into_data()`. Version 1 stores little-endian bit order within each
`u64`. `ShotMajor` stores one row per shot with `m_words() = ceil(num_measurements /
64)`; `MeasMajor` one row per measurement with `s_words() = ceil(num_shots / 64)`.
`try_from_shot_major` and `try_from_meas_major` reject shape mismatches and non-zero
semantic padding. Histograms, marginals, parity rows and accumulators mask only
semantic padding: measurement-tail bits in shot-major data, shot-tail bits in
measurement-major data.

**Detector sampler** (`compile_detector_sampler`): compiles Clifford circuits with
measurement and reset reuse into the same packed measurement sampler, then derives
detector and observable records as packed parity rows over measurement record indices.
A reset becomes a fresh qubit alias, so repeated syndrome extraction needs no per-shot
tableau replay. The sampler returns packed measurements, detectors or observables,
detector counts, or feeds packed detector chunks into any `ShotAccumulator`.

## Native backend sampling (`Backend::sample_basis_states`)

A non-Clifford circuit sampled through `Backend::probabilities()` pays a dense `2^n`
allocation, which caps shots at the dense width even on a backend whose state is
polynomial.

Two `Backend` hooks lift that cap. `supports_native_sampling` declares that a backend
draws outcomes from its own representation, and `sample_basis_states(num_shots, seed)`
returns packed per-qubit outcomes as `BasisSamples` (`ceil(n / 64)` words per shot).
Seeding comes from the argument, not the backend's RNG, so a shot request replays
exactly, and the call does not collapse the state. It takes `&mut self` because a
backend may first reorganize its storage: the distributed backend restores its qubit
map, a collective, so each rank owns a contiguous slice in circuit order.
`supports_pauli_expectation` and `pauli_expectations(observables)` are the observable
pair; they are normalization independent, so a truncated MPS is divided by `⟨ψ|ψ⟩`
rather than assumed unit.

| Backend | Sampling cost | Method |
|---------|---------------|--------|
| Sparse | `O(k log k)` once, `O(log k)` per shot | CDF over the `k` stored amplitudes, ordered by basis index |
| Factored | `O(Σ 2^kᵢ)` once, `O(B log)` per shot | One draw per sub-state, concatenated; `B` blocks |
| MPS | `O(n·χ³)` once, `O(n·χ²)` per shot | Sequential conditional sampling against precomputed right environments |
| Product State | `O(n)` once, `O(n)` per shot | One Bernoulli draw per qubit against that qubit's own weight on `1` |
| Distributed | `O(2^(n-p))` once, `O(log)` per shot | CDF over the rank-local slice; one scalar gathered per rank picks the owner |
| Everything else | dense | Unchanged: `probabilities()` then CDF |

The sparse, factored and product-state samplers draw in blocks of 256 shots, each from
its own ChaCha8 stream keyed on the seed and block index, so blocks fill in parallel
and the bits do not depend on the thread count. The dense sampler over factored block
probabilities walks the same blocks in the same order, so it matches the factored
backend's native draws shot for shot.

`run_shots_with` picks the native path through `try_native_terminal_backend`, which
requires the route to land on a single backend and probes the capability before
`init`, so a backend without one costs an allocation and nothing else.
`run_counts_with` shares that setup but has its own native arm that counts
`sample_basis_states` directly instead of materializing shots, so seeded counts from
the two can differ at a finite shot count. Only the distributed branch delegates to
`run_shots_with(..).counts()`.

Drawing every shot from one evolved state needs
`Circuit::has_terminal_measurements_only`, which counts a reset as a measurement unless
no multi-qubit gate has touched its qubit. Such a qubit is still a product factor, so
the reset is deterministic, and the leading resets of an OpenQASM program keep the
terminal route. A reset on an entangled qubit draws an outcome that one evolution would
share across every shot, so the circuit replays per shot. The density matrix is the
exception: it applies reset as the channel, and its terminal draw stays exact.

The product state is the one backend taken past subsystem decomposition. It already
stores one factor per qubit, so splitting a non-entangling circuit into blocks pays a
backend, a partition and a merge per block to rebuild what one native draw reads off
the state, and past 64 qubits the merged block distribution has no representation.
Every other backend keeps the block split;
`only_the_product_state_takes_the_native_sampler_past_decomposition`
(`src/sim/tests.rs`) pins both sides.

MPS records each bit against the logical qubit hosted at a site, not the site index, so
a layout permuted by SWAP routing needs no canonicalization pass.
`tests/integration/native_sampling.rs` pins that case.
`mps_conditional_path_probabilities_match_the_dense_vector`
(`src/backend/mps/mps_tests.rs`) checks the conditional decomposition against the dense
vector to 1e-12, and the query matrix in `tests/integration/conformance_matrix.rs` is
the corpus-wide comparison.

The tensor network answers below its dense ceiling from one contraction of the full
distribution. Past it, each bit draws from the conditioned single-qubit marginal,
contracted on the doubled network, and the outcome projector is absorbed before the
next qubit. A shot costs one doubled contraction per qubit, with peaks set by
treewidth rather than `2^n`. The measured crossover that put the dense arm below the
ceiling is in the backend's module docstring.

## Weighted observables and commuting-set grouping (`src/sim/observable.rs`)

`PauliObservable` carries a weighted Pauli sum `H = Σ c_k P_k` in the
`(f64, Vec<PauliTerm>)` term shape the gradient surface takes. Terms are canonical
(factors sorted by qubit, identical strings merged). The qubit-wise-commuting (QWC)
grouping is greedy first-fit-decreasing coloring, cached in the observable and
invalidated on mutation. Two strings qubit-wise commute when every shared qubit
carries the same axis, so a group has one axis per qubit it touches.

`observable_expectation` computes `Var(H_g) = ⟨H_g²⟩ - ⟨H_g⟩²` per group on the
statevector family. Small groups share one batched traversal
(`pauli_expectations_from_masks`). Two strings in a QWC group multiply phase-free
(shared qubits carry equal axes and cancel to identity), so each `P_i P_j` is another
Pauli string, and a small group's `⟨H_g²⟩` expands into pairwise product masks
appended to the traversal that serves the term means.

A group past the pair budget (`MAX_PAIR_MASKS_PER_GROUP`, set where the quadratic
expansion would cost more than a state sweep) takes a single-pass moment accumulation:
on the state as run when the group is Z-only, otherwise on a copy rotated by H on
X-assigned qubits and Sdg then H on Y-assigned qubits, after which members are
plus-sign Z strings. Both moments, and so the group's share of the mean, come from that
pass. `h(j)` is built 64 indices at a time: with the high index bits fixed, each member
reduces to a signed coefficient on its low-bit pattern, and a Walsh-Hadamard transform
over those patterns gives `h` across the block. The pass costs one scalar step per
member per block and six butterflies per index. Summing members per index (one parity
and one add per member per index) ran 20x to 50x slower at 12 to 14 qubits.

The per-group pass is the fallback because it loses at molecular shapes. On the
2000-string Jordan-Wigner bench fixture the grouping yields about 850 groups of mean
size 2.3, and paying copy, rotate and sweep per group measured 8.3-8.5x slower than the
ungrouped batched traversal. That traversal already computes the mean in one pass
regardless of grouping; grouping buys the variance, priced at the pair-mask expansion.

The reported variance is `Σ_g Var(H_g)`: the variance of a grouped estimate drawing one
shot per group, and the input to shot allocation via the per-group vector. It excludes
cross-group covariances, so it equals `Var(H)` only when one group covers every term;
including them would cost `O(M²)` Pauli products at molecular term counts. Routes
without the grouped evaluator (Clifford/SPD, the per-backend native paths, a run with a
noise model or start state) report the weighted mean with no variance.

Grouping costs `O(M · G)` word operations for `M` terms and `G` groups,
sub-millisecond at thousands of terms. Stronger colorings were declined: fewer groups
shrink neither the mean's single traversal nor the pair expansion, which scales with
group size rather than group count.

## Noisy compiled sampler (`src/sim/noise.rs`)

Backward Pauli propagation through the circuit gives each noise location an X-flip and
a Z-flip sensitivity row. At sample time Bernoulli draws decide which channels fire,
and their sensitivity rows XOR into the sample.

`NoiseModel` holds per-instruction noise events. Every noisy engine supports Pauli and
depolarizing channels, and every one but `noisy_marginals_analytical` supports
two-qubit depolarizing. Amplitude damping, phase damping, thermal relaxation, and one-
and two-qubit custom Kraus operators need the trajectory engine. Readout error acts on
the measurement record rather than the state, so the route is chosen on the channels
alone and each engine applies readout itself.

`NoiseBuilder` (`src/sim/noise_builder.rs`) compiles declarative rules into the same
per-instruction vector: per-gate-type and per-qubit rates, idle decoherence against
circuit layers, crosstalk through a coupling map, coherent over-rotation proportional
to a rotation gate's angle, reset error, pre-measurement error, per-bit readout, idle
relaxation and detuning timed by an as-soon-as-possible schedule of gate durations,
and per-shot amplitude drift. Every rule is evaluated once at build time, so nothing
reaches a per-shot or per-instruction loop; the drift rules emit fixed weights over
draws the trajectory makes per shot. A timed rule places each qubit's events for a
layer just before that qubit's next instruction rather than after the layer's last
one, because instruction order can list a later layer's gate or measurement on the
qubit ahead of an earlier layer's last instruction.

Two-qubit Kraus operators are indexed `K[t][t']` with `t = 2*bit(q0) + bit(q1)`, the
packing `Gate::matrix_4x4` uses. The density matrix compiles the set into a 16x16
block superoperator (`apply_2q_kraus`, which `apply_2q_depolarizing` lowers onto). The
trajectory engine draws a branch from `Tr(Kdagger K rho)` over
`Backend::reduced_density_matrix_2q` and applies the normalized operator as a
`Fused2q`. The host statevector, sparse, factored, MPS and distributed statevector
backends answer that reduction, the distributed one through
`Backend::reduced_density_matrix`; the tableau backends, the product state and the
tensor network decline it. `Auto` counts a two-qubit Kraus event as entangling when it
picks the trajectory backend, so one-qubit gates carrying a crosstalk channel run on
the statevector (or the sparse map or MPS past the dense cap), not the product state.
Routing asks the trajectory plan's backend for `Backend::supports_two_qubit_kraus`
before the first shot, so an explicit kind that declines the channel, a device
statevector included, is named at dispatch rather than mid-trajectory, and `Auto` with
a device keeps such a model on the host statevector.

## Noisy engine routing and the observable-result contract

Noisy sampling is reachable through `simulate(...).noise(...).shots(n)` /
`.sample_counts(n)`; the other builder terminals reject an inline noise model.
`run_shots_with_noise` (`src/sim/mod.rs`) routes Pauli-only noise on Clifford circuits
to the compiled family when the backend is Auto or stabilizer-family and the
measurements are terminal or rewrite to terminal (below). Every other accepted
combination runs the trajectory engine over the resolved backend. Within the compiled
family, `run_shots_noisy` (`src/sim/noise.rs`) picks one engine per call:

| Engine | Selected when | Limitations |
|---|---|---|
| Brute-force replay (`run_shots_noisy_brute_with`) | Dynamic circuits the rewrite declines | Per-shot tableau replay, O(shots) simulations; non-Clifford circuits error here (the public entry point routes them to the trajectory engine instead) |
| Pauli frame | Shallow circuits: gate count / qubits < 3, or < 5 at >= 200 qubits | Clifford, terminal measurements, no event in a guarded body |
| Compiled Pauli (`NoisyCompiledSampler`) | Remaining Clifford circuits | Clifford, terminal measurements after the rewrite |

A circuit with resets, mid-circuit measurements or guarded Pauli gates is first
rewritten to terminal measurements (`defer_noisy_circuit` in
`src/sim/compiled/mod.rs`). A reset moves its qubit to a fresh alias, and a measured
alias is never touched again, so its measurement moves to the end unchanged. A guarded
Pauli `P` on qubit `t` becomes `P` controlled by the alias of each record its condition
reads, plus an unconditional `P` when the condition is inverted. This is deferred
measurement: a Pauli fired on a record's value is a controlled Pauli from an alias
holding that value. The condition must therefore be a parity of measured bits: the
rewrite takes a single bit, a `Parity`, or a register comparison reading at most one
bit a measurement has written (an unwritten bit reads 0), and a guarded body holds
Pauli gates only. An event on a measured alias drops, since the record already holds
the value the shot keeps; for the same reason a qubit measured twice without a reset
between declines the noisy rewrite. The noiseless shot route takes the same rewrite at
two shots or more.

An event in a guarded body fires only on shots whose guard holds, which is not linear
in the records, so no fixed map holds it. The compiled sampler draws every shot's
records under the every-shot events; then each guarded event, in execution order,
draws at its rate on every shot and lands its precomputed record flips on the shots
whose guard parity holds; readout comes last. A guard reads only records measured
before its event, which no later event reaches, so execution order decides every guard
against its final value. Such a circuit always takes the compiled sampler, not the
frame sampler.

The frame and compiled engines, and the noiseless compiled sampler, all resolve to
`ResolvedBackend::CompiledStabilizer`. `RunMetadata::engine` names which one ran, so a
test pins a route off the result rather than off the predicates that picked it.

The trajectory engine (`src/sim/trajectory.rs`) covers everything the compiled family
rejects (non-Pauli channels, non-Clifford gates, dynamic circuits the rewrite declines)
at per-shot state evolution cost. On the distributed statevector the trajectories run
serially in shot order on every rank, so the ranks issue their collectives in
lockstep.

Pauli noise on the host statevector skips the per-shot cost when all measurements are
terminal, there is no reset, and no live event follows the first measurement. A Pauli
branch weight does not depend on the state and no draw depends on an outcome, so every
shot's error pattern is drawn first, from that shot's own seed, and shots with the same
pattern share one evolution. Each shot then draws its record from its group's final
state on its own stream, followed by its readout flips. The result is distributionally
exact but not seed-identical to the per-shot path. At uniform depolarizing `p = 1e-3`
on a 12-qubit, 232-event circuit, 1000 shots fall into about 190 patterns, four in five
of them the error-free one. When distinct patterns exceed nine tenths of the shots, the
run falls back to per-shot trajectories.

Amplitude and phase damping on the host statevector share the gates before the first
instruction that is not a gate or barrier, or whose events include a channel other than
Pauli, depolarizing or damping. A damping draw at or above `gamma` takes the no-jump
pass for any `P(1)` up to one, and a Pauli draw past its last threshold is the
identity, so until a shot draws below one of those bounds its state is the one every
such shot holds, and the event where it leaves depends only on its own noise stream.
The run draws each shot's leave event first and walks the shared state once; each shot
copies that state where it leaves, or at the end of the prefix, reseeds, redraws its
stream that far and finishes its own trajectory. The records are seed-identical to the
per-shot path. At `gamma = 1e-3` on a 12-qubit, 138-event ansatz, 875 of 1000 shots
draw nothing below `gamma` and reach the end of the prefix, and the run makes 15% of
the per-shot full-state passes, half of them in the terminal measurements. A walk that
reads `P(1)` past one by more than its headroom falls back to per-shot trajectories, as
do leakage and quasi-static channels.

A two-qubit channel is one joint draw over its 15 non-identity Pauli products, never
two single-qubit draws: the probability that both letters move is `0.6p`, where
independent draws give `(0.8p)^2`. The compiled sampler stores the propagated X and Z
components of both targets, four rows an event, in a table parallel to the
single-qubit one, and runs them as a second pass so the single-qubit rows keep the
positions the flip LUT and device buffers were built against. Component rows rather
than precomputed branches is a storage choice: 4 rows an event against 15. A pair whose
targets fall in different subsystem blocks takes the monolithic compile, since the
block-filtered one holds only one block's propagated masks at a time.

A non-empty pair table keeps noise application on the host: the device noise kernel
derives each thread's bit from `(seed, event, batch)` for the one measurement row it
owns and would draw the two qubits independently. The noiseless parity sample and the
bit transpose still run on the device, so the pair table costs the fused device noise
kernel and the counts and marginals reductions behind it, not GPU sampling as a whole.

The frame and compiled samplers apply readout error to the packed measurement record
after the reference outcomes are folded in, so a set bit is a measured one. Records
are walked one at a time and thinned at `max(p01, p10)`, then a candidate is accepted
at the rate its live bit selects; that is what asymmetric rates cost over a single flip
mask. Brute-force replay flips each shot's unpacked record on that shot's own stream,
as the trajectory engine does.

Every entry point that draws shots calls `NoiseModel::validate_for` against the
circuit before allocating state: one event slot per instruction, channel parameters in
range, distinct targets on a two-qubit channel, and every target inside the register.
Bounds cannot be checked from the model alone, and an out-of-register target reaches
kernels that index amplitudes without one. `noisy_marginals_analytical` is the
exception, having no per-shot state to allocate.

Slots `0..n` follow the `n` top-level instructions and fire every shot, a guarded
instruction's included. Guarded-body slots come after: each region in order lays out
its body's slots and then its nested guards', and a conditional takes one slot for its
gate. Top-level indices keep the meaning they had before guards had slots, and a
circuit without guards has exactly `n`. A body slot fires only on the shots its
instruction runs, so the untaken branch adds no noise, as in a hand-lowered circuit.
The built-in constructors and every gate rule of `NoiseBuilder` put a guarded gate's
noise in its body slot; rules that read a schedule decline a circuit holding a region,
whose duration depends on the branch.

Trajectories and the stabilizer replay walk the guards per shot. `compile_noisy` and
the error chain complex fix their maps before a shot, so they reject a model whose body
slots carry an event; the noisy shot route lands such an event after the map, as
above. The density matrix rejects any guard before reading a slot. A model of exactly
`n` slots is still accepted on a circuit whose guards are all conditionals (which then
carry no gate noise) and, readout only, on a circuit with a region.

Custom Kraus sets must be trace preserving, `sum_k Kdagger_k K_k = I` to 1e-9. The
exact route applies a declared set literally while the trajectory route normalizes its
branch probabilities, so a non-trace-preserving set would mean different things on the
two engines.

`ThermalRelaxation` is generalized amplitude damping composed with pure dephasing on
both routes, at rates chosen so populations decay as `exp(-gate_time/t1)` and
coherences as `exp(-gate_time/t2)` toward a steady state of `excited_population`. At
zero excited population the two operators that excite or hold a hot steady state are
zero, leaving the amplitude-damping pair and a three-branch trajectory unraveling. A
mixture of reset and `Z` reproduces the population decay but reaches the coherence
decay only for `t2 <= t1`, and needs a negative dephasing probability above it.

All engines sample the same measurement-record distribution for the models and
circuits they accept. The equivalence is statistical: engines consume independent RNG
streams, so one seed gives different shots with matching marginals, correlators and
histograms. Cross-engine tests pin every engine to the analytic marginals from
`noisy_marginals_analytical` and to each other's correlator statistics:
`pauli_engines_share_observable_statistics` (`src/sim/noise_tests.rs`),
`trajectory_pauli_matches_brute_force` (`src/sim/trajectory.rs`), and the channel-level
analytic checks in `tests/integration/trajectory_correctness.rs`.

GPU reductions (`gpu` feature): with a context attached via `with_gpu`, the noisy
compiled sampler can sample, apply noise, and reduce counts or marginals on the device.
Device noise masks come from a device-seeded RNG stream, so GPU output matches CPU
output statistically, not bit for bit. On-device counts are limited to 512 measurements
(8 packed words); larger circuits fall back to the CPU reduction. Golden test:
`noisy_compiled_gpu_reductions_match_cpu_statistics` (`tests/golden_gpu.rs`).

## Leakage and quasi-static drift (`src/sim/trajectory.rs`, `src/qec/leakage.rs`)

Two channel families carry state from one event to the next within a shot: leakage
(`Leakage`, `Seepage`, `LeakageTransport`) keeps a leak flag per qubit, and
`QuasiStatic` reads standard normal draws made once at the start of the shot. Neither
is a CPTP map applied event by event, so the density matrix and the Pauli-path
estimator decline them and they run on the per-shot trajectory engines only. A model
holding neither takes the trajectory loop unchanged, with the same draws;
`MemorySpec::of` decides that once per run.

**Leaked qubit in a trajectory.** When a qubit leaks, the engine resets it and flips
it to `|1>`, then leaves it alone until it returns: gates naming it are skipped, other
channels skip events naming it, and a measurement of it reports 1. The reset is one
trajectory of the channel `rho -> |0><0| (x) tr_q rho` (see `Backend::reset`), so
averaged over shots the rest of the register holds exactly `tr_q rho`, as if the qubit
had left the register. After the reset the qubit is an unentangled basis state, which
keeps the rest cheap:

- Measuring it is a deterministic outcome that collapses nothing else, so the ordinary
  measurement path reports 1, and a classical condition reading the bit sees the same
  1.
- A skipped gate needs no kernel. The stand-in for a multi-qubit gate it blocks is one
  uniformly random Pauli (`I`, `X`, `Y`, `Z` at a quarter each) on each unleaked
  target: complete depolarization.
- Seepage returns the qubit to a uniformly random basis state with one coin and at most
  one `X`, uncorrelated with the rest because nothing entangled it.

Holding the leaked qubit at `|0>` would cost the same, but the measurement, and any
conditional reading that bit, would need a special case to report 1. The reset costs
one probability pass and one collapse per leak event, rare at realistic rates, and
nothing per gate.

`ShotsResult::leaked` records, per shot, which qubits leaked at any point: the erasure
herald.

**Quasi-static draws.** `QuasiStatic { axis, weights }` rotates its qubit by
`sum_k w_k z_k`, where `z` holds one independent standard normal per source index,
drawn by Box-Muller from the shot's noise stream before its first event. `NoiseBuilder`
lowers a `DriftDistribution` with covariance `Sigma` through its Cholesky factor `L`:
qubit `q`'s offset is row `q` of `L` against `z`, so the offsets have covariance
`L L^T = Sigma` exactly, and detuning scales the row by each scheduled layer's
duration. A singular `Sigma` (perfect correlation) keeps zero columns where the factor
has no variance left.

**Leakage on the compiled QEC path.** A Pauli frame cannot skip a gate, and does not
need to. `QecLeakageSampler` walks the deferred circuit forward per unit of 8192 shots,
holding a packed leak row per alias next to the frame rows. A qubit that leaks gets a
uniformly random frame Pauli, and every two-qubit gate meeting a leaked qubit scrambles
the frames of both its qubits on the shots where either is leaked. Scrambling both
fully depolarizes the pair, and `tr_ab(U rho U^dag) = tr_ab(rho)` for any `U` on the
pair, so the rest of the register sees what it sees in a trajectory where the gate is
skipped and the partner depolarized. Pauli annotations replay in program order with
the leakage ones, and a firing that names a leaked qubit is dropped, as a trajectory
skips it. The leaked qubit's own record is forced to 1 after the frame flips land. The
result matches trajectory leakage in distribution on a Clifford program
(`erasure_heralds_agree_with_trajectory_leakage_in_distribution`,
`tests/integration/qec_leakage.rs`). A measured alias's flag cannot change before its
record is read, since the lowering admits no gate on it and freezes its leakage
targets, so the record reads the flag at measurement. `LEAK` targets add herald
columns, `QecSampleResult::heralds`. Forcing a bit is not a linear flip, so these
programs bypass the record-noise and parity-projection samplers.

## Error chain complex (`src/sim/homological.rs`)

`ErrorChainComplex` is a GF(2) chain complex over the circuit's noise locations. The kernel (null space) of its boundary map gives the error cycles syndrome measurements cannot detect.

`noisy_marginals_analytical` gives closed-form marginals from the parity matrix and noise rates, with no Monte Carlo sampling.

See the [Noise and QEC guide](../guides/qec.md) for how these fit together in practice.

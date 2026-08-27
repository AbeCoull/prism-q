# Compiled Samplers

For multi-shot sampling without materializing the full statevector on every shot.

## Noiseless compiled sampler (`src/sim/compiled/`)

**Backward path** (`compile_measurements`): Propagates Pauli Z observables backward through the circuit. Each measurement qubit becomes a row in a GF(2) parity matrix M. Clifford gates conjugate Pauli strings in O(1). The resulting M encodes which input qubits each measurement depends on.

**Forward path** (`compile_forward`): Tracks stabilizer generator dependencies forward through the circuit. Produces the same parity matrix via dependency tracking.

**Sampling**: Random bits for independent generators, then XOR-cascade through the parity matrix. Multiple dispatch tiers:

| Strategy | Condition | Method |
|----------|-----------|--------|
| `FlipLut` | Small rank | 256-entry XOR lookup table |
| `SparseParity` | Sparse rows | Only flip non-zero columns |
| `XorDag` | General | Optimal XOR-reduction DAG |
| `ParityBlocks` | Blocked structure | Per-block independent sampling |

**ShotAccumulator trait**: Pluggable result collection.

| Accumulator | Output | Use case |
|-------------|--------|----------|
| `HistogramAccumulator` | Bitstring → count map | Standard shot output |
| `MarginalsAccumulator` | Per-qubit P(1) | Marginal probabilities |
| `PauliExpectationAccumulator` | ⟨P⟩ for Pauli observables | VQE/QAOA |
| `CorrelatorAccumulator` | ⟨Z_i Z_j⟩ correlations | Entanglement analysis |
| `NullAccumulator` | Nothing | Benchmarking raw sampling speed |

**PackedShots raw format**: `PackedShots::RAW_FORMAT_VERSION` is the replay
contract for `raw_data()` and `into_data()`. Version 1 stores little-endian bit
order within each `u64`. `ShotMajor` stores one row per shot with
`m_words() = ceil(num_measurements / 64)`. `MeasMajor` stores one row per
measurement with `s_words() = ceil(num_shots / 64)`. The checked
`try_from_shot_major` and `try_from_meas_major` constructors reject shape
mismatches and non-zero semantic padding. Histograms, marginals, parity rows,
and accumulators mask only semantic padding: measurement-tail bits in
shot-major data and shot-tail bits in measurement-major data.

**Detector sampler** (`compile_detector_sampler`): Compiles Clifford circuits
with measurement and reset reuse into the same packed measurement sampler, then
derives detector and observable records as packed parity rows over measurement
record indices. Reset reuse is represented by fresh qubit aliases, so repeated
syndrome extraction avoids per-shot tableau replay. The sampler can return
packed measurements, packed detectors, packed observables, detector counts, or
feed packed detector chunks into any `ShotAccumulator`.

## Native backend sampling (`Backend::sample_basis_states`)

The compiled samplers above cover Clifford circuits. Everything else used to
funnel through `Backend::probabilities()`, a dense `2^n` allocation, and sample
from that, which put a hard qubit ceiling on shots for backends whose state is
polynomial.

Two `Backend` hooks lift it. `supports_native_sampling` declares that a backend
draws outcomes from its own representation, and `sample_basis_states(num_shots,
seed)` returns packed per-qubit outcomes as `BasisSamples`
(`ceil(n / 64)` words per shot). Seeding is from the argument, not the backend's
RNG, so a shot request replays exactly, and the call does not collapse the
state. It takes `&mut self` because a backend may first have to reorganize its
own storage: the distributed backend restores its qubit map, a collective, so
that each rank owns a contiguous slice in circuit order.
`supports_pauli_expectation` and `pauli_expectations(observables)` are
the observable-side pair, normalization independent so a truncated MPS is
divided by `⟨ψ|ψ⟩` rather than assumed unit.

| Backend | Sampling cost | Method |
|---------|---------------|--------|
| Sparse | `O(k log k)` once, `O(log k)` per shot | CDF over the `k` stored amplitudes, ordered by basis index |
| Factored | `O(Σ 2^kᵢ)` once, `O(B log)` per shot | One draw per sub-state, concatenated; `B` blocks |
| MPS | `O(n·χ³)` once, `O(n·χ²)` per shot | Sequential conditional sampling against precomputed right environments |
| Product State | `O(n)` once, `O(n)` per shot | One Bernoulli draw per qubit against that qubit's own weight on `1` |
| Distributed | `O(2^(n-p))` once, `O(log)` per shot | CDF over the rank-local slice; one scalar gathered per rank picks the owner |
| Everything else | dense | Unchanged: `probabilities()` then CDF |

`run_shots_with` picks the native path through `try_native_terminal_backend`,
which requires the route to land on a single backend and probes the capability
before `init`, so a backend without one costs an allocation and nothing else.
`run_counts_with` needs no separate path: its tail is `run_shots_with(..).counts()`.

The product state is the one backend taken past subsystem decomposition. It
already stores one factor per qubit, so splitting a non-entangling circuit into
independent blocks pays a backend, a partition, and a merge per block to rebuild
what one native draw reads off the state, and past 64 qubits the merged block
distribution has no representation at all. Every other backend keeps the block
split it had before, which `only_the_product_state_takes_the_native_sampler_past_decomposition`
(`src/sim/mod.rs`) pins from both sides.

MPS records each bit against the logical qubit currently hosted at a site rather
than the site index, so a layout permuted by SWAP routing needs no
canonicalization pass. `tests/native_sampling.rs` pins that case; the exact
check that the conditional decomposition reproduces the dense vector to 1e-12
lives in `mps_conditional_path_probabilities_match_the_dense_vector`
(`src/backend/mps.rs`), and the corpus-wide comparison is the query matrix in
`tests/conformance_matrix.rs`.

The tensor network answers below its dense ceiling from one contraction of
the full distribution, and past it samples qubit by qubit: each bit draws
from the conditioned single-qubit marginal, contracted on the doubled
network, and the outcome projector is absorbed before the next qubit's
marginal. A sweep shot costs one doubled contraction per qubit with peaks set
by treewidth rather than `2^n`, which is what carries sampling past the
ceiling; the measured crossover that put the dense arm below it is recorded
in the backend's module docstring.

## Weighted observables and commuting-set grouping (`src/sim/observable.rs`)

`PauliObservable` carries a weighted Pauli sum `H = Σ c_k P_k` in the same
`(f64, Vec<PauliTerm>)` term shape the gradient surface takes. Terms are kept
canonical (factors sorted by qubit, identical strings merged) and the
qubit-wise-commuting grouping is computed by greedy first-fit-decreasing
coloring, cached inside the observable, and invalidated on mutation. Two
strings qubit-wise commute when every shared qubit carries the same axis, so a
group has one well-defined axis per qubit it touches.

`observable_expectation` computes `Var(H_g) = ⟨H_g²⟩ - ⟨H_g⟩²` per group on
the statevector family, with the mean and most group variances served by one
shared batched traversal (`pauli_expectations_from_masks`). Two strings in a
QWC group multiply phase-free: shared qubits carry equal axes and cancel to
identity, so each `P_i P_j` is just another Pauli string, and a small group's
`⟨H_g²⟩` expands into pairwise product masks appended to the same traversal
that serves the term means. A group past the pair budget
(`MAX_PAIR_MASKS_PER_GROUP`, set where the quadratic expansion would cost
more than a state sweep) takes a dedicated single-pass moment accumulation
instead: on the state as run when the group is Z-only, otherwise on a copy
rotated by H on X-assigned qubits and Sdg then H on Y-assigned qubits, after
which members are plus-sign Z strings, and `h(j)` accumulates per element
before squaring so both moments come from the one pass.

A per-group state pass is the fallback rather than the default because it
loses at molecular shapes: on the 2000-string Jordan-Wigner bench fixture
the grouping yields about 850 groups of mean size 2.3, and an engine paying
copy, rotate, and sweep per group measured 8.3-8.5x slower than the
ungrouped batched traversal. That traversal already made the mean one pass
regardless of grouping, so grouping buys no mean throughput; what it buys is
the variance, priced at the pair-mask expansion.

The reported variance is `Σ_g Var(H_g)`: the variance of a grouped measurement
estimate drawing one shot per group, and the input to shot allocation via the
per-group vector. It excludes cross-group covariances, so it equals `Var(H)`
of the full operator only when a single group covers every term; including
them would cost `O(M²)` Pauli products at molecular term counts. Routes
without the grouped evaluator (Clifford/SPD, the per-backend native paths, a
run with a noise model or start state) report the weighted mean with no
variance.

Grouping cost is `O(M · G)` word operations for `M` terms and `G` groups,
sub-millisecond at thousands of terms; stronger colorings were declined
because fewer groups would shrink neither the mean's single traversal nor the
pair expansion, which scales with group size rather than group count.

## Noisy compiled sampler (`src/sim/noise.rs`)

Backward Pauli propagation through circuit + noise sensitivity analysis. Each noise location gets an X-flip and Z-flip sensitivity row. During sampling, Bernoulli coin flips determine which noise channels fire, then XOR the sensitivity rows into the sample.

`NoiseModel`: per-instruction noise events. Pauli and depolarizing channels are
supported by every noisy engine. Amplitude damping, phase damping, thermal
relaxation, two-qubit depolarizing, one- and two-qubit custom Kraus operators,
and readout error require the trajectory engine.

`NoiseBuilder` (`src/sim/noise_builder.rs`) compiles declarative rules into that
same per-instruction vector: per-gate-type and per-qubit rates, idle
decoherence against circuit layers, crosstalk through a coupling map, coherent
over-rotation proportional to a rotation gate's own angle, reset error,
pre-measurement error, and per-bit readout. Every rule is evaluated once at
build time, so nothing it expresses reaches a per-shot or per-instruction loop.

Two-qubit Kraus operators are indexed `K[t][t']` with `t = 2*bit(q0) +
bit(q1)`, the packing `Gate::matrix_4x4` uses. The density matrix compiles the
set into a 16x16 block superoperator (`apply_2q_kraus`, which
`apply_2q_depolarizing` lowers onto). The trajectory engine draws a branch from
`Tr(Kdagger K rho)` over `Backend::reduced_density_matrix_2q` and applies the
normalized operator as a `Fused2q`. Only the host statevector implements that
reduction, and `run_shots_with_noise` checks `Backend::supports_two_qubit_kraus`
before the first shot, so an `Auto` route that picked another backend is named
at dispatch rather than part way through a trajectory.

## Noisy engine routing and the observable-result contract

Noisy sampling is reachable through `simulate(...).noise(...).shots(n)` /
`.sample_counts(n)`; the other builder terminals reject an inline noise model.
`run_shots_with_noise` (`src/sim/mod.rs`) routes Pauli-only noise on Clifford
circuits with terminal measurements (no resets, no classical conditionals) to
the compiled family when the backend is Auto or stabilizer-family; every other
accepted combination runs the trajectory engine over the resolved backend.
Within the compiled family, `run_shots_noisy` (`src/sim/noise.rs`) picks one
engine per call:

| Engine | Selected when | Limitations |
|---|---|---|
| Brute-force replay (`run_shots_noisy_brute_with`) | Resets, classical conditionals, or mid-circuit measurements | Per-shot tableau replay, O(shots) simulations; non-Clifford circuits error here (the public entry point routes them to the trajectory engine instead) |
| Homological (`src/sim/homological.rs`) | >= 1000 shots and the error complex compiles (syndrome rank <= 20) | Falls through to frame/compiled above rank 20 |
| Pauli frame | Shallow circuits: gate count / qubits < 3, or < 5 at >= 200 qubits | Clifford, terminal measurements only |
| Compiled Pauli (`NoisyCompiledSampler`) | Remaining Clifford + terminal-measurement circuits | Clifford, terminal measurements only |

The trajectory engine (`src/sim/trajectory.rs`) covers everything the compiled
family rejects: non-Pauli channels, readout error, mid-circuit measurement,
reset, classical conditionals, and non-Clifford gates, at per-shot state
evolution cost. Distributed backends reject noisy sampling entirely; per-shot
trajectories cannot keep rank collectives in lockstep.

Every noisy entry point calls `NoiseModel::validate_for` against the circuit
before allocating state: one event slot per instruction, channel parameters in
range, distinct targets on a two-qubit channel, and every target inside the
register. Bounds cannot be checked from the model alone, and a target outside
the register reaches kernels that index amplitudes without one.

A guarded region (`Instruction::Region`) is rejected whenever the model carries
at least one quantum event: slots are indexed per top-level instruction, so a
region body has none and would run noiselessly. A readout-only model has
nothing to lose there and is accepted. Reaching noise inside a region body
needs the event stream keyed by something other than a top-level index, which
the compiled sampler, the homological builder, and the density-matrix evolution
all walk today.

Custom Kraus sets must be trace preserving, `sum_k Kdagger_k K_k = I` to 1e-9.
The exact route applies a declared set literally while the trajectory route
normalizes its branch probabilities, so a set that is not trace preserving would
mean two different things depending on which engine ran it.

`ThermalRelaxation { t1, t2, gate_time }` is amplitude damping composed with
pure dephasing on both routes, at rates chosen so populations decay as
`exp(-gate_time/t1)` and coherences as `exp(-gate_time/t2)`. A mixture of reset
and `Z` reproduces the population decay but reaches the coherence decay only for
`t2 <= t1`, and needs a negative dephasing probability above it.

All engines sample from the same measurement-record distribution for the noise
models and circuits they accept. The equivalence is statistical, not
shot-for-shot: engines consume independent RNG streams, so the same seed
produces different shots with matching observable statistics (marginals,
correlators, histograms). Cross-engine tests pin every engine to the analytic
marginals from `noisy_marginals_analytical` and to each other's correlator
statistics: `pauli_engines_share_observable_statistics` (`src/sim/noise.rs`),
`trajectory_pauli_matches_brute_force` (`src/sim/trajectory.rs`), and the
channel-level analytic checks in `tests/trajectory_correctness.rs`.

GPU reductions (`gpu` feature): with a context attached via `with_gpu`, the
noisy compiled sampler can sample, apply noise, and reduce counts or marginals
on the device. Device noise masks come from a device-seeded RNG stream, so GPU
output matches CPU output statistically, not bit for bit. On-device counts are
limited to 512 measurements (8 packed words); larger circuits fall back to the
CPU reduction. Golden test: `noisy_compiled_gpu_reductions_match_cpu_statistics`
(`tests/golden_gpu.rs`).

## Homological sampler (`src/sim/homological.rs`)

`ErrorChainComplex`: GF(2) chain complex over the circuit's noise locations. Computes the kernel (null space) of the boundary map to identify error cycles that are undetectable by syndrome measurements. `HomologicalSampler` uses this for sampling with topological error correction awareness.

`noisy_marginals_analytical`: Closed-form marginal computation using the parity matrix and noise rates. Avoids Monte Carlo sampling entirely.

See the [Noise and QEC guide](../guides/qec.md) for how these fit together in practice.

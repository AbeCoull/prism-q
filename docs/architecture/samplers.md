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
state. `supports_pauli_expectation` and `pauli_expectations(observables)` are
the observable-side pair, normalization independent so a truncated MPS is
divided by `⟨ψ|ψ⟩` rather than assumed unit.

| Backend | Sampling cost | Method |
|---------|---------------|--------|
| Sparse | `O(k log k)` once, `O(log k)` per shot | CDF over the `k` stored amplitudes, ordered by basis index |
| Factored | `O(Σ 2^kᵢ)` once, `O(B log)` per shot | One draw per sub-state, concatenated; `B` blocks |
| MPS | `O(n·χ³)` once, `O(n·χ²)` per shot | Sequential conditional sampling against precomputed right environments |
| Everything else | dense | Unchanged: `probabilities()` then CDF |

`run_shots_with` picks the native path through `try_native_terminal_backend`,
which requires the route to land on a single backend and probes the capability
before `init`, so a backend without one costs an allocation and nothing else.
`run_counts_with` needs no separate path: its tail is `run_shots_with(..).counts()`.

MPS records each bit against the logical qubit currently hosted at a site rather
than the site index, so a layout permuted by SWAP routing needs no
canonicalization pass. `tests/native_sampling.rs` pins that case; the exact
check that the conditional decomposition reproduces the dense vector to 1e-12
lives in `mps_conditional_path_probabilities_match_the_dense_vector`
(`src/backend/mps.rs`), and the corpus-wide comparison is the query matrix in
`tests/conformance_matrix.rs`.

The tensor network stays on the dense route by decision, not by omission; its
module docstring records why.

## Noisy compiled sampler (`src/sim/noise.rs`)

Backward Pauli propagation through circuit + noise sensitivity analysis. Each noise location gets an X-flip and Z-flip sensitivity row. During sampling, Bernoulli coin flips determine which noise channels fire, then XOR the sensitivity rows into the sample.

`NoiseModel`: per-instruction noise events. Pauli and depolarizing channels are
supported by every noisy engine. Amplitude damping, phase damping, thermal
relaxation, two-qubit depolarizing, custom Kraus operators, and readout error
require the trajectory engine.

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

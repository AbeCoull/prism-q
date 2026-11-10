# Compiled Samplers

Shot sampling that does not evolve a full statevector per shot. Noisy sampling, noise
models and leakage are on [Noise Compilation](./noise.md).

## Noiseless compiled sampler (`src/sim/compiled/`)

`compile_measurements` propagates Pauli Z observables backward through the circuit. Each
measured qubit becomes a row of a GF(2) parity matrix M recording which input qubits it
depends on, and a Clifford gate conjugates a Pauli string in O(1). `compile_forward` tracks
stabilizer generator dependencies forward through the circuit and produces the same
matrix.

A shot draws random bits for the independent generators, then XORs them through the
parity matrix:

| Strategy | Condition | Method |
|----------|-----------|--------|
| `FlipLut` | Small rank | 256-entry XOR lookup table |
| `SparseParity` | Sparse rows | Flip only the nonzero columns |
| `ParityBlocks` | Blocked structure | Each block sampled independently |

Results collect through the `ShotAccumulator` trait:

| Accumulator | Output | Use case |
|-------------|--------|----------|
| `HistogramAccumulator` | Bitstring → count map | Standard shot output |
| `MarginalsAccumulator` | P(1) per qubit | Marginal probabilities |
| `PauliExpectationAccumulator` | ⟨P⟩ for Pauli observables | VQE/QAOA |
| `CorrelatorAccumulator` | ⟨Z_i Z_j⟩ correlations | Entanglement analysis |
| `NullAccumulator` | Nothing | Timing raw sampling |

`PackedShots::RAW_FORMAT_VERSION` is the replay contract for `raw_data()` and
`into_data()`. Version 1 stores little-endian bit order within each `u64`. `ShotMajor`
stores one row per shot with `m_words() = ceil(num_measurements / 64)`, and `MeasMajor`
one row per measurement with `s_words() = ceil(num_shots / 64)`. `try_from_shot_major` and
`try_from_meas_major` reject shape mismatches and nonzero semantic padding. Histograms,
marginals, parity rows and accumulators mask only semantic padding: the bits past the
last measurement in shot major data, and past the last shot in measurement major data.

`compile_detector_sampler` compiles Clifford circuits that reuse qubits after
measurement and reset into the same packed sampler, then derives detector and
observable records as packed parity rows over measurement record indices. A reset
becomes a fresh qubit alias, so repeated syndrome extraction replays no tableau per
shot. The sampler returns packed measurements, detectors or observables,
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
pair. They do not assume a unit norm, so a truncated MPS is divided by `⟨ψ|ψ⟩`.

| Backend | Sampling cost | Method |
|---------|---------------|--------|
| Sparse | `O(k log k)` once, `O(log k)` per shot | CDF over the `k` stored amplitudes, ordered by basis index |
| Factored | `O(Σ 2^kᵢ)` once, `O(B log)` per shot | One draw per block state, concatenated; `B` blocks |
| MPS | `O(n·χ³)` once, `O(n·χ²)` per shot | Sequential conditional sampling against precomputed right environments |
| Product State | `O(n)` once, `O(n)` per shot | One Bernoulli draw per qubit against that qubit's own weight on `1` |
| Distributed | `O(2^(n-p))` once, `O(log)` per shot | CDF over the local slice; one scalar gathered per rank picks the owner |
| Everything else | dense | Unchanged: `probabilities()` then CDF |

The sparse, factored and product-state samplers draw in blocks of 256 shots, each from
its own ChaCha8 stream keyed on the seed and block index, so blocks fill in parallel
and the bits do not depend on the thread count. The dense sampler over factored block
probabilities walks the same blocks in the same order, so it matches the factored
backend's native draws shot for shot.

`run_shots_with` and `run_counts_with` share `prepare_shot_source`. Terminal route
selection builds the circuit without measurements and its measurement map once,
then passes both to preparation. A native sampler requires a single backend whose
capability is checked before initialization. The counts path aggregates its draws
without materializing a `ShotsResult`.

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
Every other backend keeps the block split.
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

Weighted observables and their commuting groups are on
[the engine page](./engine.md#weighted-observables).

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

## Noisy compiled sampler (`src/sim/noise.rs`)

Backward Pauli propagation through circuit + noise sensitivity analysis. Each noise location gets an X-flip and Z-flip sensitivity row. During sampling, Bernoulli coin flips determine which noise channels fire, then XOR the sensitivity rows into the sample.

`NoiseModel`: Per-instruction depolarizing noise. `NoiseOp { qubit, px, py, pz }`.

## Homological sampler (`src/sim/homological.rs`)

`ErrorChainComplex`: GF(2) chain complex over the circuit's noise locations. Computes the kernel (null space) of the boundary map to identify error cycles that are undetectable by syndrome measurements. `HomologicalSampler` uses this for sampling with topological error correction awareness.

`noisy_marginals_analytical`: Closed-form marginal computation using the parity matrix and noise rates. Avoids Monte Carlo sampling entirely.

See the [Noise and QEC guide](../guides/qec.md) for how these fit together in practice.

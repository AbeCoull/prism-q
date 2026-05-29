# Clifford+T Simulation

Circuits that mix Clifford gates with a modest number of `T` gates sit between the
efficient stabilizer regime and the exponential statevector regime. PRISM-Q offers three
strategies. The right one depends on your T-count, qubit count, and whether you need
exact answers or can tolerate Monte Carlo error.

```admonish tip title="Which strategy?"
- **Few T gates, exact result needed**: stabilizer rank (`run_stabilizer_rank`).
- **Many T gates, marginals only**: stochastic Pauli propagation (`run_spp`).
- **Moderate T-count, exact or bounded-error expectation values**: deterministic sparse
  Pauli dynamics (`run_spd`).
```

These route through the Clifford+T strategies before the standard
[dispatch tree](../architecture/engine.md) when the T-count permits.

## Stabilizer rank (`src/sim/stabilizer_rank.rs`)

Exact probability output remains capped because it returns a dense vector with
2^n entries. Shot sampling uses coherent weighted MPS branches instead of a
dense statevector fallback. Clifford gates mutate each branch state, `T` and
`Tdg` split branches, and measurement computes outcome probabilities from the
weighted branch ensemble before projecting every branch to the sampled outcome.
This removes the hard qubit-count cap from `run_stabilizer_rank_shots`;
practical scaling is governed by branch count, MPS bond growth, and measurement
count.

The dense probability path maintains a weighted sum of stabilizer states. Each T
gate doubles the term count via the `T = alpha*I + beta*Z` decomposition.
Clifford gates are O(n²) per term and weighted amplitudes are accumulated for
exact probabilities.

| Function | Use |
|----------|-----|
| `run_stabilizer_rank` | Exact probabilities (t ≤ 20, n ≤ 25) |
| `run_stabilizer_rank_approx` | Approximate with Monte Carlo (higher t counts) |
| `run_stabilizer_rank_shots` | Shot-based sampling with no fixed qubit cap |
| `stabilizer_overlap_sq` | Inner product between stabilizer states |

## Stochastic Pauli Propagation (`src/sim/unified_pauli.rs`)

Backward-propagates measurement observables as Pauli strings. Clifford gates conjugate in O(1). T gates branch stochastically into two Pauli paths with appropriate weights. Per-path cost O(d×n/64), independent of T-gate count. Returns marginal probabilities via Monte Carlo estimation.

```rust
run_spp(circuit, num_samples, seed) // -> SppResult
```

## Deterministic Sparse Pauli Dynamics (`src/sim/unified_pauli.rs`)

Backward-propagates as a weighted sum of Pauli strings stored in a HashMap. T gates deterministically branch X/Y terms. Identical strings auto-merge. Optional ε-truncation for approximate mode. Exact for small T-counts, approximate with bounded error for larger ones.

```rust
run_spd(circuit, epsilon, max_terms) // -> SpdResult
```

You can also build Clifford+T test circuits directly with
[`clifford_t_circuit`](../reference/builders.md).

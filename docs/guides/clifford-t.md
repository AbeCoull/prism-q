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

## Pauli path propagation under noise (`src/sim/unified_pauli.rs`)

The same weighted Pauli sum, carried through a noise model. Select
`BackendKind::PauliPath { epsilon, max_terms }` and attach a noise model; the engine
answers `expectation_values` and `observable_expectation`, and nothing else.

```rust
let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
let values = simulate(&circuit)
    .backend(BackendKind::PauliPath { epsilon: 1e-8, max_terms: 1 << 16 })
    .noise(&noise)
    .expectation_values(&observables)?;
```

The error model is worth stating plainly, because it has two independent parts and only
one of them is approximate.

The channels are exact. Each one is applied as its adjoint on the Pauli basis rather than
as a twirl, so depolarizing, dephasing, thermal relaxation, and amplitude damping all
reproduce the density matrix to machine precision at `max_terms = 0`. Amplitude damping
is the case worth naming: it is not unital, and the identity term its adjoint produces
from `Z` is carried rather than dropped. A channel with no Pauli-basis form (custom
Kraus, two-qubit Kraus, readout error) is rejected rather than approximated.

The truncation is the approximate part, and it reports its own bound. With
`max_terms = 0` nothing is dropped and the value is exact. With a budget set, terms whose
coefficient magnitude falls below `epsilon` are dropped once the sum exceeds the budget,
and the total dropped magnitude bounds the error in the returned value. That bound is a
worst case rather than an estimate: it holds because every remaining operation is a
contraction in the Pauli 1-norm.

What decides whether the engine is usable is the term count, not the qubit count. Every
non-Clifford rotation in the observable's backward light cone can double the sum; every
channel shrinks it. Circuits where noise wins stay cheap at widths no dense
representation reaches, and circuits where it does not will hit the budget and report a
large discarded mass, which is the signal to use the density matrix or trajectory
averaging instead.

In practice the observable's weight is what moves that count, ahead of width and depth,
because the backward light cone opens from every letter it starts with. On a two-layer
hardware-efficient ansatz under 1% depolarizing, `Z` on one qubit stays at 11 terms
whether the register is 20 qubits or 100; `Z` on two adjacent qubits fills a
16384-term budget by 30 qubits; a `Z` on every qubit fills it at 20 and returns a
discarded mass larger than the observable's own norm, which is the engine saying the
answer is not usable rather than returning a wrong one quietly. Check the reported
discarded mass against the precision the caller needs before trusting a truncated run.

You can also build Clifford+T test circuits directly with
[`clifford_t_circuit`](../reference/builders.md).

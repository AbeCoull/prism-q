# Choosing a Backend

By default PRISM-Q inspects the circuit and picks a backend. Name one when you know
something the dispatcher cannot infer, or to benchmark a specific representation.

## Let it choose

```rust
use prism_q::simulate;

let result = simulate(&circuit).seed(42).run().unwrap();   // BackendKind::Auto
```

Auto reads the circuit's structure and the statevector memory cap. A circuit with no
entangling gates runs as a product state and a Clifford circuit on a stabilizer
tableau. Past the memory cap, a circuit that keeps few basis states goes to the sparse
backend and the rest to MPS at bond 256. Below it, a circuit with partly independent
registers runs factored and everything else on the dense statevector. The
[decision tree](../architecture/engine.md#auto-dispatch-decision-tree) draws these steps
with their thresholds.

Some routes sit outside the tree:

- A Clifford circuit of 16 or more qubits that splits into independent blocks goes to
  `FactoredStabilizer`.
- Shots on a Clifford circuit go to the `CompiledStabilizer` sampler, which compiles the
  measurement record once and draws every shot from it.
- On a Clifford+T circuit with few T gates, shots can take the stabilizer-rank sampler,
  and marginals at 12 qubits and above take deterministic Pauli propagation.
- Expectation and marginal queries go to the tensor network when its planned
  contraction stays inside the peak bound.

Probabilities always come from the tree. The memory cap is half of physical memory,
about 29 qubits on a 16 GiB host, unless `PRISM_MAX_SV_QUBITS` sets it.

## Choose explicitly

```rust
use prism_q::{simulate, BackendKind};

let result = simulate(&circuit)
    .backend(BackendKind::Stabilizer)
    .seed(42)
    .run()
    .unwrap();
```

From Python, pass a `BackendKind` constructor such as `BackendKind.stabilizer()` to
`.backend(...)`.

## Symptom to backend

| If your circuit... | Use | Why |
|--------------------|-----|-----|
| Is Clifford only (H, S, CX, CZ, ...) | `Stabilizer` | O(n²), scales to thousands of qubits |
| Is Clifford only and splits into independent blocks | `FactoredStabilizer` | One tableau per block |
| Has no entangling gates | `ProductState` | O(n), one state per qubit |
| Is dense and fits the [memory cap](../architecture/backends.md#memory-budget) | `Statevector` | Exact, fastest for the general case |
| Stays concentrated in few basis states | `Sparse` | O(k) in nonzero amplitudes |
| Has low entanglement but many qubits | `Mps { max_bond_dim }` | Polynomial memory in bond dimension |
| Splits into independent registers | `Factored` | Simulates blocks separately, merges lazily |
| Has low treewidth and you need an expectation or marginal | `TensorNetwork` | Contracts only what the query reads |
| Needs exact noisy probabilities or expectations | `DensityMatrix` | Evolves the mixed state once, 4ⁿ entries |
| Is Clifford + a few T gates | See [Clifford+T](../guides/clifford-t.md) | Beats a dense statevector |

```admonish warning title="ProductState rejects entanglement"
`ProductState` errors on any entangling gate. Auto selects it only for circuits that
have none; choose it explicitly only for a circuit that stays a product state
throughout.
```

## Scaling

| Backend | Memory | Ceiling |
|---------|--------|---------|
| Statevector | $O(2^n)$ | The memory cap: half of physical memory, about 29 qubits on 16 GiB |
| Stabilizer | $O(n^2)$ | Thousands of qubits |
| Factored Stabilizer | $O(n^2)$ per cluster | Thousands of qubits |
| Sparse | $O(k)$ nonzero | Large $n$, small $k$ |
| MPS | $O(n\chi^2)$ | Large $n$, bounded $\chi$ |
| Product | $O(n)$ | Unbounded |
| Tensor Network | Set by contraction order | Peak intermediate under `PRISM_MAX_TN_PEAK_QUBITS`; dense readout under `PRISM_MAX_PROB_QUBITS` |
| Factored | $O(2^n)$ worst case | Largest merged block |
| Density Matrix | $O(4^n)$ | Half the statevector cap, about 14 qubits on 16 GiB |

The [memory budget](../architecture/backends.md#memory-budget) derives every cap from the
same detected budget. The `distributed` feature adds a statevector sharded across MPI
ranks with exact results; [Distributed Statevector and MPI](../guides/distributed.md) has
its layout and limits. CPU and GPU support per backend is in the
[Capability and Support Matrix](../guides/capabilities.md), and each backend's internals
in the [architecture reference](../architecture/backends.md).

Next: work through a task end to end in the [Tutorials](../tutorials/index.md).

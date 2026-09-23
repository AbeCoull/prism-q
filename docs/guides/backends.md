# Backends Deep Dive

PRISM-Q has nine backends, each holding a different kind of state, plus four Clifford+T
and Pauli-propagation engines that hold no state and answer observables directly (see
[Clifford+T Simulation](./clifford-t.md)). This page covers scaling and when to reach for
each backend; the [architecture reference](../architecture/backends.md) covers the
kernels. To select a backend in code, see
[Choosing a Backend](../getting-started/choosing-a-backend.md). For the CPU and GPU
architectures each backend supports, see the
[Capability and Support Matrix](./capabilities.md).

## Scaling at a glance

| Backend | Memory | Best for | Ceiling |
|---------|--------|----------|---------|
| Statevector | $O(2^n)$ | Dense general circuits | ~28 qubits (RAM-bound) |
| Stabilizer | $O(n^2)$ | Clifford-only circuits | Thousands of qubits |
| Factored Stabilizer | $O(n^2)$ per cluster | Clifford with independent blocks | Thousands of qubits |
| Sparse | $O(k)$ nonzero | Concentrated support | Large $n$, small $k$ |
| MPS | $O(n\chi^2)$ | Low entanglement | Large $n$, bounded $\chi$ |
| Product | $O(n)$ | No entanglement | Unbounded |
| Tensor Network | order-dependent | Shallow / structured | $\le 25$ prob qubits |
| Factored | $O(2^n)$ worst case | Partially independent | Block-bound |
| Density Matrix | $O(4^n)$ | Exact noisy evolution | ~14 qubits (RAM-bound) |

The distributed statevector backend (behind the `distributed` feature) shards the dense
state across MPI ranks; see the
[Capability and Support Matrix](./capabilities.md) for its status.

## Statevector

The default for dense circuits: exact, fully general, and the fastest option whenever the
state fits in RAM. The memory cap is derived from system RAM (override with
`PRISM_MAX_SV_QUBITS`). Above it, auto-dispatch falls back to Sparse or MPS.

## Stabilizer

A circuit of only Clifford gates (H, S, Sdg, SX, SXdg, X, Y, Z, Id, CX, CZ, SWAP,
measurement) runs on the stabilizer tableau in $O(n^2)$ and scales to thousands of qubits.
Auto-dispatch selects it whenever the circuit is Clifford-only.

```admonish tip
Add even a single non-Clifford gate (`T`, `Rz(θ)` with arbitrary θ) and the stabilizer
backend no longer applies. For a small number of such gates, see
[Clifford+T Simulation](./clifford-t.md).
```

## Sparse, MPS, Product, Tensor Network, Factored, Density Matrix

- **Sparse** wins when the state stays concentrated in a handful of computational-basis
  states (amplitude pruning keeps the map small).
- **MPS** trades exactness for memory polynomial in the bond dimension, for
  low-entanglement circuits over many qubits.
- **Product** is the entanglement-free case: $O(n)$ memory, $O(1)$ per 1q gate.
- **Tensor Network** defers contraction until measurement, for shallow or structured
  circuits.
- **Factored** detects partial independence and simulates sub-registers separately,
  merging lazily via a Kronecker product computed on demand.
- **Density Matrix** evolves the full mixed state exactly, for noise studies below the
  $4^n$ memory ceiling. Explicit dispatch only; `Auto` never selects it.

The kernels behind each are in the
[architecture reference](../architecture/backends.md), and the speed mechanics in
[Performance and SIMD](./performance.md).

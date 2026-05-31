# Backends Deep Dive

PRISM-Q does not have one simulation algorithm. It has eight, each optimal for a
different class of circuit. This guide is the task-oriented companion to the
[architecture reference](../architecture/backends.md): it focuses on scaling and when to
reach for each one. To select a backend in code, see
[Choosing a Backend](../getting-started/choosing-a-backend.md). For which CPU and
GPU architectures each backend supports, see the
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

## Statevector

The default for dense circuits. Exact, fully general, and the fastest option whenever the
state fits in RAM. The memory cap is derived from system RAM (overridable with
`PRISM_MAX_SV_QUBITS`). Above it, auto-dispatch falls back to Sparse or MPS.

## Stabilizer

If your circuit uses only Clifford gates (H, S, Sdg, SX, SXdg, CX, CZ, SWAP, measurement),
the stabilizer tableau simulates it in $O(n^2)$ and scales to thousands of qubits.
Auto-dispatch selects it whenever the circuit is Clifford-only.

```admonish tip
Add even a single non-Clifford gate (`T`, `Rz(θ)` with arbitrary θ) and the stabilizer
backend no longer applies. For a small number of such gates, see
[Clifford+T Simulation](./clifford-t.md).
```

## Sparse, MPS, Product, Tensor Network, Factored

- **Sparse** wins when the state stays concentrated in a handful of computational-basis
  states (amplitude pruning keeps the map small).
- **MPS** trades exactness for polynomial memory in the bond dimension. Ideal for
  low-entanglement circuits over many qubits.
- **Product** is the degenerate, entanglement-free case: $O(n)$ memory, $O(1)$ per 1q gate.
- **Tensor Network** defers contraction until measurement, useful for shallow or
  structured circuits.
- **Factored** detects partial independence and simulates sub-registers separately,
  merging lazily via a Kronecker product computed on demand.

For the internal kernels behind each of these, read the
[architecture reference](../architecture/backends.md). For raw speed mechanics, see
[Performance and SIMD](./performance.md).

# Backends

PRISM-Q ships eight CPU backends plus an optional CUDA path. The
[simulation engine](./engine.md) picks one automatically, or you can select explicitly.
For a task-oriented version of this material, see the
[Backends Deep Dive guide](../guides/backends.md).

The diagrams below are rendered directly from PRISM-Q's own SVG circuit renderer.

![GHZ state preparation circuit](../diagrams/ghz_5.svg)

## Statevector

Full-state simulation in a flat `Vec<Complex64>` of 2^n amplitudes. The primary backend for circuits up to ~28 qubits.

Gate kernels use enum dispatch with specialized routines for CX, CZ, SWAP, Cu, MCU, Rzz, BatchRzz, BatchPhase, DiagonalBatch, and MultiFused. Single-qubit gates go through `PreparedGate1q` with FMA-vectorized SIMD. MultiFused gates use a three-tier tiled kernel (L2 16K / L3 131K / individual passes) for cache locality. MultiFused batches where all gates are diagonal dispatch to a dedicated fast path (1 complex multiply/element vs 4+2 for full 2×2).

Rayon parallelism at ≥14 qubits with `par_chunks_mut` and `MIN_PAR_ELEMS = 4096` per task. BMI2 `_pext_u64` accelerates BatchPhase, BatchRzz, and DiagonalBatch LUT indexing.

Deferred measurement normalization: `pending_norm` accumulates normalization factors without full-state scaling passes. Zero-cost for circuits without measurements.

The Quantum Fourier Transform is a representative statevector workload, dense with
controlled-phase gates that the fusion pipeline batches:

![Quantum Fourier Transform circuit](../diagrams/qft_4.svg)

## Stabilizer

Aaronson-Gottesman bit-packed tableau for Clifford circuits. O(n²) time and space. Scales to thousands of qubits. Gate kernels use wordwise bitwise ops and `popcount` for phase computation. Supports H, S, Sdg, SX, SXdg, X, Y, Z, Id, CX, CZ, SWAP, plus measurement, reset, and classical conditionals.

Word-group batching fuses multiple 1q gate flushes into single tableau passes. Type-grouped masks apply all gates of the same Pauli type with one wordwise op instead of per-gate dispatch. Sparse Generator Indexing (SGI) tracks per-qubit active generator lists, enabling targeted row operations instead of full-tableau scans. Lazy destabilizer materialization defers destabilizer rows until probabilities are requested.

Probability extraction uses coset-based enumeration with GF(2) Gaussian elimination. O(2^k) where k is the number of non-diagonal generators, rather than O(2^n).

**Factored Stabilizer** (`FactoredStabilizerBackend`): Per-cluster tableaux with dynamic merging. Starts with one qubit per cluster. Cross-cluster 2q gates merge tableaux. Measurement and reset can split independent sub-tableaux again. Independent subsystems avoid full-tableau work when product structure is preserved.

## Sparse

`HashMap<usize, Complex64>` for states with few non-zero amplitudes. O(k) memory. Amplitude pruning (|a|² < 1e-16) after each gate. Best for circuits whose support stays concentrated in computational-basis states at large qubit counts.

## MPS (Matrix Product State)

Chain of rank-3 tensors with adaptive bond dimension (default max 256). O(n·χ²) memory. Single-qubit gates absorb via FMA-vectorized SIMD over bond-dimension slices. Two-qubit gates contract adjacent sites, apply the gate, then SVD-truncate back. Non-adjacent gates route through SWAP chains.

Hybrid SVD dispatch: faer (bidiag+D&C) for matrices with m×n ≥ 256, hand-rolled Jacobi for small matrices.

## Product State

Per-qubit `[Complex64; 2]` storage. O(n) memory, O(1) per single-qubit gate. Rejects entangling gates. Selected automatically for circuits with no 2q gates.

## Tensor Network

Deferred contraction with a greedy min-size heuristic. Gates append tensors; contraction happens lazily at measurement or probability extraction. `MAX_PROB_QUBITS = 25` guards against exponential blowup.

## Factored

Dynamic split-state simulation. Starts with n independent 1-qubit states, merges via tensor product only when 2q gates bridge groups. Parallel kernels match statevector patterns for sub-states ≥14 qubits. Selected when subsystem decomposition detects partial independence.

The [GPU backend](../guides/gpu.md) is documented as a user guide.

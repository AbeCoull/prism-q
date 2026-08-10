# Backends

PRISM-Q ships nine CPU backends, an optional CUDA path attached to the statevector and
stabilizer backends, and a feature-gated distributed statevector backend that shards the
dense state across MPI ranks. The [simulation engine](./engine.md) picks a backend
automatically (the density matrix, tensor network, and distributed backends are
explicit-dispatch only), or you can select explicitly. For a task-oriented version of
this material, see the [Backends Deep Dive guide](../guides/backends.md).

The diagrams below are rendered directly from PRISM-Q's own SVG circuit renderer.

![GHZ state preparation circuit](../diagrams/ghz_5.svg)

## Reset semantics

`reset` is the channel `rho -> |0⟩⟨0| ⊗ tr_q rho` on every backend: the qubit is traced
out and replaced by `|0⟩`, leaving the rest of the register in the mixture the trace
produces. Projecting onto `|0⟩` and renormalizing is not equivalent. The two agree only
when the reset qubit is unentangled; when it is entangled, projection also collapses its
partners into the branch correlated with the `|0⟩` outcome. Resetting qubit 1 of a Bell
pair leaves `⟨Z0⟩ = 0` under the channel and `⟨Z0⟩ = 1` under projection.

A backend holding a single pure state cannot represent the resulting mixture, so it runs
one trajectory of the channel: sample the measurement outcome, collapse onto it, and
apply X when the outcome is 1. Averaged over shots that reproduces the channel, and a
reset consumes one draw from the backend's RNG stream. The density-matrix backend holds
the mixture and applies the channel directly, with no draw. `tests/reset_channel.rs`
pins the contract across backends against the density-matrix oracle.

## Memory budget

A circuit that does not fit in memory is an error, not a fallback. No backend silently
hands the work to a different one when its state would not fit: it returns
`PrismError::IncompatibleBackend` naming itself, the qubit count, the cap, and the
environment variable that overrides it. Choosing a different backend is the caller's
decision, and `BackendKind::Auto` makes it from circuit structure before any backend is
constructed.

The check lives in `Backend::init`, which is the one point every execution path passes
through before reserving its state. Putting it there means a caller that drives a backend
directly, through `run_on` rather than `simulate`, gets the same guard as one that goes
through dispatch.

| Cap | Variable | Default |
|-----|----------|---------|
| Statevector state | `PRISM_MAX_SV_QUBITS` | Largest `2^n` `Complex64` state fitting half of detected physical memory |
| Density-matrix state | `PRISM_MAX_DM_QUBITS`, bounded by `PRISM_MAX_SV_QUBITS` | `floor(cap_sv / 2)`, since a density matrix of `n` qubits is a `2n`-qubit statevector |
| Dense probability output | `PRISM_MAX_PROB_QUBITS` | Same budget over `f64` |
| Dense statevector export | `PRISM_MAX_EXPORT_QUBITS` | Same budget over `Complex64` |
| Dense outcome sampling | `PRISM_MAX_DENSE_OUTCOME_BITS` | Same budget over two `f64` per outcome |

The density-matrix cap is the tighter of its own override and half the statevector cap,
computed in one place so dispatch-time validation and the backend's `init` guard cannot
disagree about where the ceiling is. Raising `PRISM_MAX_DM_QUBITS` past that bound needs
`PRISM_MAX_SV_QUBITS` raised with it, which is what the rejection says: the backend
reports it itself rather than surfacing an error naming the statevector it allocates
internally. When physical memory cannot be detected the caps are
disabled and a warning is printed, because guessing a budget is worse than saying the
budget is unknown.

Parallel noisy trajectories are the one path holding more than one state at a time: each
Rayon thread runs its own backend, so peak memory is `threads * state(n)`. That path is
restricted to circuits below 14 qubits, where a statevector replica is 256 KiB and a full
thread pool stays in the tens of megabytes. Above it trajectories run serially with one
live backend, bounded by the ordinary state cap.

Three growth paths are not yet bounded and can still exhaust memory on an adversarial
input: MPS bond dimension when `max_bond_dim` is set very high, sparse-state entry count
when a circuit densifies, and the transient peak inside a factored sub-state merge.

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

Shots and Pauli expectations answer from the per-qubit states rather than the `2^n` probability vector, so both stay O(n) and the backend runs queries at widths no dense route reaches. See [Sampling Architecture](./samplers.md).

## Tensor Network

Deferred contraction with a greedy min-size heuristic. Gates append tensors; contraction happens lazily at measurement or probability extraction. `MAX_PROB_QUBITS = 25` guards against exponential blowup.

Two queries stay off the dense route by contracting the network against its conjugate.
The bra copy's legs are shifted clear of the ket index space, and each qubit's boundary
is either closed against its twin, which is a trace, or joined through an operator. A
one-qubit reduced density matrix leaves that qubit's ket and bra indices open and
returns a `2x2`; a Pauli expectation joins every non-identity factor through its
operator and contracts to a scalar. Both follow the doubled network's cost rather than
the qubit count. An identity factor is a closed leg rather than an appended tensor, so a
weight-`k` observable adds `k` tensors and not `n`.

Nothing about the planner changed: an index is open when exactly one tensor holds it,
which the greedy ordering already carries through to its result.

The reduced density matrix is the half of general-noise support the backend was missing,
so the trajectory engine now runs amplitude damping, phase damping, thermal relaxation,
and custom Kraus channels here under explicit dispatch.

## Factored

Dynamic split-state simulation. Starts with n independent 1-qubit states, merges via tensor product only when 2q gates bridge groups. Parallel kernels match statevector patterns for sub-states ≥14 qubits. Selected when subsystem decomposition detects partial independence.

## Density Matrix

Exact mixed-state evolution. Stores the full density operator `rho` for `n` qubits as a
`4^n` `Complex64` buffer laid out row-major: index `(r << n) | c` holds `⟨r|rho|c⟩`. That
layout is isomorphic to a `2n`-qubit statevector whose high `n` qubits index the ket (row)
and low `n` qubits index the bra (column), so gate application reuses the statevector
kernels with no new gate math. A unitary `U` on the ket register gives the left product
`U rho`; the same `U` on the bra register of a conjugated buffer gives the right product
`rho U^dagger`, so `U rho U^dagger` costs two statevector passes and two conjugations.

Memory is `16 * 4^n` bytes, so the ceiling is about 14 qubits on a 16 GiB host and 15 on
32 GiB (`PRISM_MAX_DM_QUBITS` moves it within the statevector budget). This backend is
CPU-only and explicit-dispatch only; `Auto` never selects it.

Selecting it with a noise model attached is the exact route for every `Simulate`
terminal: the mixture is evolved once and observables, marginals, probabilities, and
shots all read that one evolution. See [Noise across the terminals](./engine.md) for
what that route accepts and what stays on trajectory averaging.

The [GPU backend](../guides/gpu.md) is documented as a user guide. The distributed
statevector backend is covered in the
[Capability and Support Matrix](../guides/capabilities.md).

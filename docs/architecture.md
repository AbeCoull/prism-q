# PRISM-Q Architecture

## Goals

- **Primary**: Fastest practical quantum circuit simulation in Rust.
- Correct simulation of supported gate sets across multiple backend strategies.
- Clean backend plugin model — add new simulation strategies without touching the core.

## Non-goals

- Full OpenQASM 3.0 compliance (supports a practical subset).
- GUI or notebook integration (library-first).
- Hardware backend / QPU connectivity.

## Layered design

```text
┌─────────────────────────────────────┐
│          User / Application         │
├─────────────────────────────────────┤
│       run_qasm(&str, seed)          │  ← Public API (src/lib.rs)
├─────────────────────────────────────┤
│         OpenQASM 3.0 Parser         │  ← src/circuit/openqasm.rs
│         &str → Circuit IR           │
├─────────────────────────────────────┤
│          Circuit IR                 │  ← src/circuit/mod.rs
│   (backend-agnostic instruction     │
│    sequence: gates, measures,       │
│    barriers)                        │
├─────────────────────────────────────┤
│        Fusion Pipeline              │  ← src/circuit/fusion.rs
│   (cancel, fuse, reorder, batch)    │
├─────────────────────────────────────┤
│        Simulation Engine            │  ← src/sim/mod.rs
│   (dispatch, decompose, execute)    │
├─────────────────────────────────────┤
│        Backend Trait                │  ← src/backend/mod.rs
│   init | apply | measure | probs   │
├──────┬──────┬──────┬──────┬──────┬──────┬──────┤
│  SV  │  TN  │ MPS  │Sparse│ Prod │ Stab │ Fact │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

## Parser

Hand-rolled line-by-line parser targeting a practical OpenQASM 3.0 subset. Converts `&str` directly to `Circuit` IR with no intermediate AST.

**Supported**: `qubit`/`bit` declarations, standard gates (x, y, z, h, s, sdg, t, tdg, rx, ry, rz, cx, cz, swap, sx, sxdg, p, cy, ch, crx, cry, crz, csx, ccx/toffoli, ccz, cswap/fredkin, rzz, rxx, ryy, ecr, iswap, dcx, u1, u2, u3/u), gate modifiers (`ctrl @`, `inv @`, `pow(k) @`), user-defined `gate` blocks, classical `if` conditionals, multi-register broadcast, measure, barrier, expression evaluator with math functions. OpenQASM 2.0 backward compatibility (`qreg`/`creg`, `measure q -> c` syntax).

**Unsupported**: `for`/`while` loops, subroutines, classical expressions beyond `if`.

## Circuit IR

`Circuit` holds `num_qubits`, `num_classical_bits`, and `Vec<Instruction>`. Instructions are an enum: `Gate { gate, targets }`, `Measure`, `Barrier`, `Conditional`. Targets use `SmallVec<[usize; 4]>` — inline storage for up to 4 qubits, no heap allocation for typical gates.

`Gate` is a `Clone` enum kept at 16 bytes. Simple variants (e.g., `Rx(f64)`) carry parameters inline. Composite variants (`Fused`, `MultiFused`, `BatchPhase`, `BatchRzz`, `Cu`, `Mcu`) use `Box` to stay within the 16-byte budget for cache-friendly dispatch.

## Fusion pipeline

Six-pass gate optimization before execution, gated by qubit count thresholds:

1. **Self-inverse cancellation** — removes CX·CX, CZ·CZ, SWAP·SWAP pairs. Non-adjacent cancellation via per-qubit tracking.
2. **Rzz recognition** — CX(a,b)·Rz(θ,b)·CX(a,b) → native Rzz(θ,a,b). BatchRzz groups consecutive Rzz gates into a single LUT-based pass.
3. **Single-qubit fusion** — fuses consecutive 1q gates on the same qubit into one precomputed matrix.
4. **Commutation-aware reorder** — moves 1q gates earlier past non-conflicting 2q gates (diagonal gates commute through CX control, CZ, and Rzz).
5. **Multi-gate tiling** — batches 1q and 2q gates into MultiFused/Multi2q for L2/L3-tiled execution.
6. **Controlled-phase batching** — consecutive cphase gates sharing a control qubit collapse into BatchPhase with LUT+BMI2 kernels.

Returns `Cow::Borrowed` at each stage when no optimization applies — zero overhead for circuits that don't benefit.

## Backend trait

```rust
pub trait Backend {
    fn name(&self) -> &'static str;
    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()>;
    fn apply(&mut self, instruction: &Instruction) -> Result<()>;
    fn classical_results(&self) -> &[bool];
    fn probabilities(&self) -> Result<Vec<f64>>;
    fn num_qubits(&self) -> usize;
}
```

Contract: `init` before `apply`. Instructions arrive in circuit order. Measurement is destructive. Deterministic given same RNG seed.

## Simulation engine

Orchestration layer in `src/sim/mod.rs`. Entry points: `run()` (auto-dispatch), `run_with()` (explicit backend), `run_on()` (pre-constructed backend).

- **Auto-dispatch**: `BackendKind::Auto` selects the optimal backend — ProductState for single-qubit-only circuits, Stabilizer for all-Clifford, MPS for >28 qubits, Factored when independent subsystems exist, Statevector otherwise.
- **Subsystem decomposition**: Union-find detects independent qubit groups. Each block runs separately, results merge via Kronecker product.
- **Temporal Clifford decomposition**: Clifford prefix runs on Stabilizer, non-Clifford tail on Statevector with the exported state.

## Backends

### Statevector

Full-state simulation in a flat `Vec<Complex64>` of 2^n amplitudes. The primary backend for circuits up to ~28 qubits.

Gate kernels use enum dispatch with specialized routines for CX, CZ, SWAP, Cu, MCU, Rzz, BatchRzz, and BatchPhase. Single-qubit gates go through `PreparedGate1q` with FMA-vectorized SIMD. MultiFused gates use a three-tier tiled kernel (L2/L3/individual passes) for cache locality.

Rayon parallelism kicks in at ≥14 qubits with `par_chunks_mut` and `MIN_PAR_ELEMS = 4096` per task. BMI2 `_pext_u64` accelerates BatchPhase and BatchRzz LUT indexing where available.

### Stabilizer

Aaronson-Gottesman bit-packed tableau for Clifford circuits. O(n²) time and space — scales to thousands of qubits where statevector is impossible. Gate kernels use wordwise bitwise ops and `popcount` for phase computation. Supports H, S, Sdg, SX, SXdg, CX, CZ, SWAP, and measurement.

Word-group batching fuses multiple 1q gate flushes into single tableau passes. Type-grouped masks apply all gates of the same Pauli type with one wordwise op instead of per-gate dispatch.

Probability extraction uses coset-based enumeration with GF(2) Gaussian elimination — O(2^k) where k is the number of non-diagonal generators, rather than O(2^n).

### Sparse

`HashMap<usize, Complex64>` for states with few non-zero amplitudes. O(k) memory where k = nonzero entries. Amplitude pruning (|a|² < 1e-16) after each gate keeps the map compact. Best for computational-basis-heavy circuits at large qubit counts.

### MPS (Matrix Product State)

Chain of rank-3 tensors with adaptive bond dimension (default max 64). O(n·χ²) memory. Single-qubit gates absorb in-place using FMA-vectorized SIMD over bond-dimension slices. Two-qubit gates contract adjacent sites, apply the gate, then SVD-truncate back. Non-adjacent gates route through SWAP chains.

SVD uses a hybrid dispatch: faer (bidiag+D&C) for matrices with m×n ≥ 256, hand-rolled Jacobi for small matrices. Right-tensor transposition before contraction ensures sequential memory access in the inner reduction loop.

### Product State

Per-qubit `[Complex64; 2]` storage. O(n) memory, O(1) per single-qubit gate. Rejects entangling gates with `BackendUnsupported`. Selected automatically for circuits with no 2q gates.

### Tensor Network

Deferred contraction with a greedy min-size heuristic. Gates append tensors to the network; contraction happens lazily at measurement or probability extraction. Tensors use `SmallVec<[usize; 6]>` for shape and legs. Performance depends on circuit treewidth. `MAX_PROB_QUBITS = 25` guards against exponential blowup.

### Factored

Dynamic split-state simulation. Starts with n independent 1-qubit states, merges via tensor product only when 2q gates bridge groups. Gate dispatch translates global→local qubit indices per sub-state. Parallel kernels match the statevector patterns for sub-states ≥14 qubits. MultiFused gates group by sub-state for independent application.

Selected automatically when subsystem decomposition detects partial independence but the blocks aren't large enough to justify separate simulations.

## Memory layout

| Backend | State representation | Memory | Access pattern |
|---------|---------------------|--------|----------------|
| Statevector | `Vec<Complex64>` (2^n) | O(2^n) | Strided pair iteration |
| Stabilizer | Bit-packed `Vec<u64>` tableau | O(n²/8) bytes | Sequential row iteration |
| Sparse | `HashMap<usize, Complex64>` | O(k), k = nonzero | Hash-based random access |
| MPS | Chain of rank-3 tensors | O(n·χ²) | Sequential site access |
| Product | `Vec<[Complex64; 2]>` | O(n) | Per-qubit independent |
| Tensor Network | Network of dense tensors | O(gates × local dim) | Contraction-order dependent |
| Factored | `Vec<Option<SubState>>` | O(2^n) worst case | Per-sub-state dispatch |

## Threading and SIMD

### Threading

Gate kernels have `_par` variants using `par_chunks_mut` for safe Rayon parallelism (behind the `parallel` feature flag):

- **<14 qubits**: Single-threaded. Thread-pool overhead exceeds computation.
- **≥14 qubits**: Rayon parallel iterators with `MIN_PAR_ELEMS = 4096` (64KB per task).

### SIMD

`Complex64` maps to 128-bit SIMD naturally. Single-qubit gate kernels use `PreparedGate1q` with runtime CPU detection and tiered dispatch:

1. **AVX2+FMA** (256-bit): 2 complex pairs per iteration. Gated by `MAX_AVX2_STATE` for full-state passes (Skylake frequency throttling), but used freely within MultiFused L2 tiles where data is cache-resident.
2. **FMA** (128-bit): Default for larger states. 3-op complex multiply (permute + mul + fmaddsub).
3. **BMI2**: `_pext_u64` for BatchPhase and BatchRzz LUT indexing. Single-instruction bit extraction replaces multi-iteration shift-and-or loops.
4. **Scalar fallback**: No intrinsics.

### Determinism

Same circuit + same seed = same result, regardless of thread count. Parallel backends use deterministic work partitioning.

## Error model

All public APIs return `Result<T, PrismError>`. Error variants cover parsing (`Parse`, `UnsupportedConstruct`, `UndefinedRegister`), validation (`InvalidQubit`, `InvalidClassicalBit`, `GateArity`, `InvalidParameter`), and runtime (`BackendUnsupported`, `IncompatibleBackend`, `Simulation`).

No panics on user input. `debug_assert!` for internal invariants only.

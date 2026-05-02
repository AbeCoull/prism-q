# PRISM-Q Architecture

## Goals

- **Primary**: Fastest practical quantum circuit simulation in Rust.
- Correct simulation of supported gate sets across multiple backend strategies.
- Clean backend plugin model. New simulation strategies can be added without touching the core.

See the [Architecture Glossary](./glossary.md) for definitions of terms used throughout this document.

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
│    barriers, conditionals)          │
├─────────────────────────────────────┤
│        Fusion Pipeline              │  ← src/circuit/fusion.rs
│   (cancel, fuse, reorder, batch)    │
├─────────────────────────────────────┤
│        Simulation Engine            │  ← src/sim/mod.rs
│   (dispatch, decompose, execute)    │
├──────────┬──────────────────────────┤
│ Backends │  Compiled Samplers       │  ← src/sim/compiled/, noise.rs
│          │  (shot-based simulation) │     homological.rs, unified_pauli.rs
├──────┬───┴──┬──────┬──────┬──────┬──────┬──────┤
│  SV  │  TN  │ MPS  │Sparse│ Prod │ Stab │ Fact │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

## Parser

Handwritten parser targeting a practical OpenQASM 3.0 subset. It processes input line by line and converts `&str` directly to `Circuit` IR with no intermediate AST.

**Supported**: `qubit`/`bit` declarations, OpenQASM standard gates and aliases (x, y, z, h, s, sdg, t, tdg, sx, rx, ry, rz, p/phase, cx/CX/cnot, cy, cz, cp/cphase, crx, cry, crz, ch, swap, ccx/toffoli, cswap/fredkin, cu, u1, u2, u3/u/U), Qiskit and exporter gates (sxdg, cs, csdg, csx, ccz, r, rzz, rxx, ryy, xx_plus_yy, xx_minus_yy, ecr, iswap, dcx, c3x, c4x, mcx, rccx, rc3x/rcccx), hardware-native gates (gpi, gpi2, ms, syc, sqrt_iswap, sqrt_iswap_inv), gate modifiers (`ctrl @`, `inv @`, `pow(k) @`), user-defined `gate` blocks, classical `if` conditionals, multi-register broadcast, measure, barrier, expression evaluator with math functions. OpenQASM 2.0 backward compatibility (`qreg`/`creg`, `measure q -> c` syntax).

**Unsupported**: `for`/`while` loops, subroutines, classical expressions beyond `if`.

## Circuit IR

`Circuit` holds `num_qubits`, `num_classical_bits`, and `Vec<Instruction>`. Instructions are an enum:

| Variant | Fields | Description |
|---------|--------|-------------|
| `Gate` | `gate`, `targets` | Gate application |
| `Measure` | `qubit`, `classical_bit` | Destructive measurement |
| `Barrier` | `qubits` | Synchronization barrier |
| `Conditional` | `condition`, `gate`, `targets` | Classical-controlled gate |

Targets use `SmallVec<[usize; 4]>`, inline storage for up to 4 qubits, no heap allocation for typical gates.

### Gate enum

`Gate` is a `Clone` enum kept at **16 bytes**. Simple variants carry parameters inline. Composite variants use `Box` to stay within the 16-byte budget for cache-friendly dispatch.

| Variant | Data | Size |
|---------|------|------|
| `Id`, `X`, `Y`, `Z`, `H`, `S`, `Sdg`, `T`, `Tdg`, `SX`, `SXdg` | None | 16B |
| `Rx(f64)`, `Ry(f64)`, `Rz(f64)`, `P(f64)`, `Rzz(f64)` | Inline f64 | 16B |
| `Cx`, `Cz`, `Swap` | None | 16B |
| `Cu(Box<[[Complex64; 2]; 2]>)` | Boxed 2×2 | 16B |
| `Mcu(Box<McuData>)` | Boxed matrix + control count | 16B |
| `Fused(Box<[[Complex64; 2]; 2]>)` | Boxed pre-fused 1q matrix | 16B |
| `Fused2q(Box<[[Complex64; 4]; 4]>)` | Boxed pre-fused 2q matrix | 16B |
| `MultiFused(Box<MultiFusedData>)` | Batched 1q gates for tiled pass | 16B |
| `Multi2q(Box<Multi2qData>)` | Batched 2q gates for tiled pass | 16B |
| `BatchPhase(Box<BatchPhaseData>)` | Batched cphase with shared control | 16B |
| `BatchRzz(Box<BatchRzzData>)` | Batched ZZ rotations | 16B |
| `DiagonalBatch(Box<DiagonalBatchData>)` | Mixed diagonal 1q/2q batch | 16B |

## Fusion pipeline

Gate optimizations before execution, gated by qubit count thresholds. Every pass returns `Cow<Circuit>`. `Borrowed` when no optimization applies, so circuits that do not benefit pay zero overhead.

```text
  Input Circuit
    │
    ├─ pass0:  cancel_self_inverse_pairs     (always)
    ├─ pass0r: fuse_rzz                      (always)  CX·Rz·CX → Rzz
    ├─ pass0b: fuse_batch_rzz                (≥16q)    N×Rzz → BatchRzz
    │
    │  ─── MIN_QUBITS_FOR_FUSION (10) gate ───
    │
    ├─ pass1:  fuse_single_qubit_gates       (≥10q)    consecutive 1q → Fused
    ├─ pass1r: reorder_1q_gates              (≥10q)    commutation-aware reorder
    ├─ pass1c: cancel_self_inverse_pairs     (≥10q)    re-cancel after reorder
    ├─ pass1f: fuse_single_qubit_gates       (≥10q)    re-fuse after reorder
    │
    ├─ pass_2q:  fuse_2q_gates              (≥12q)    CX/CZ + adjacent 1q → Fused2q
    ├─ pass_2qb: fuse_same_pair_2q_blocks   (≥12q)    same-pair Fused2q blocks → Fused2q
    ├─ pass2:    fuse_multi_1q_gates        (≥14q)    1q batch → MultiFused
    ├─ pass_2qr: reorder_disjoint_fused2q   (≥12q)    disjoint Fused2q tier grouping
    ├─ pass_m2q: fuse_multi_2q_gates        (≥12q)    2q batch → Multi2q
    ├─ pass_cp:  fuse_controlled_phases     (≥16q)    cphase batch → BatchPhase
    ├─ pass_db:  fuse_diagonal_batch        (≥16q)    mixed diagonal → DiagonalBatch
    └─ pass_pp:  batch_post_phase_1q        (≥18q)    re-batch 1q after cphase
    │
    Output Circuit
```

**Threshold constants:**

| Constant | Value | Rationale |
|----------|-------|-----------|
| `MIN_QUBITS_FOR_FUSION` | 10 | Below this, clone cost exceeds simulation savings |
| `MIN_QUBITS_FOR_MULTI_FUSION` | 14 | MultiFused tiling overhead vs benefit |
| `MIN_QUBITS_FOR_DIAG_BATCH` | 16 | Diagonal batch, cphase, and Rzz batching |
| `MIN_QUBITS_FOR_POST_PHASE_BATCH` | 18 | Post-phase 1q re-batching |
| `MIN_QUBITS_FOR_2Q_FUSION` | 12 | Benchmarked QV and random sweeps show memory-pass reduction wins from 12q |
| `MIN_QUBITS_FOR_MULTI_2Q_FUSION` | 12 | Same as 2q fusion |

## Backend trait

```rust
pub trait Backend {
    fn name(&self) -> &'static str;
    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()>;
    fn apply(&mut self, instruction: &Instruction) -> Result<()>;
    fn classical_results(&self) -> &[bool];
    fn probabilities(&self) -> Result<Vec<f64>>;
    fn num_qubits(&self) -> usize;

    // Optional overrides:
    fn apply_instructions(&mut self, instructions: &[Instruction]) -> Result<()>;  // batch apply
    fn supports_fused_gates(&self) -> bool;   // false for symbolic backends (stabilizer)
    fn export_statevector(&self) -> Result<Vec<Complex64>>;  // for backend transitions
}
```

Contract: `init` before `apply`. Instructions arrive in circuit order. Measurement is destructive. Deterministic given same RNG seed.

## Simulation engine

Orchestration layer in `src/sim/mod.rs`. Entry points:

| Function | Description |
|----------|-------------|
| `run(circuit, seed)` | Auto-dispatch, full output |
| `run_with(kind, circuit, seed)` | Explicit backend selection |
| `run_on(backend, circuit)` | Pre-constructed backend |
| `run_qasm(qasm, seed)` | Parse + simulate |
| `run_shots(circuit, shots, seed)` | Multi-shot sampling |
| `run_shots_with(kind, circuit, shots, seed)` | Multi-shot with backend selection |
| `run_shots_with_noise(kind, circuit, noise, shots, seed)` | Noisy multi-shot |
| `run_counts(kind, circuit, shots, seed)` | Frequency histogram |
| `run_marginals(kind, circuit, seed)` | Per-qubit marginal probabilities |

### Auto-dispatch decision tree

```text
BackendKind::Auto
  ├─ No entangling gates        → ProductState (O(n))
  ├─ All Clifford               → Stabilizer (O(n²))
  ├─ Above memory limit:
  │   ├─ Sparse-friendly        → Sparse (O(k))
  │   └─ Otherwise              → MPS (bond dim 256)
  ├─ Partial independence       → Factored (dynamic split-state)
  └─ Otherwise                  → Statevector (exact)
```

Memory limit is dynamically computed from available system RAM (50% budget, capped at 33 qubits). Overridable via `PRISM_MAX_SV_QUBITS` environment variable. Falls back to 28 qubits (4 GB) when detection unavailable.

### Subsystem decomposition

Union-find detects independent qubit groups in O(n·α(n)). Each block runs separately with per-block Auto dispatch. Results merge lazily via `Probabilities::Factored`, a Kronecker product computed on demand per element in O(K), avoiding the O(2^N) dense materialization unless explicitly requested.

Block-level Rayon parallelism when all blocks are <14 qubits (avoids oversubscription with block-internal parallelism).

### Temporal Clifford decomposition

For Clifford+T circuits: Clifford prefix runs on the Stabilizer backend, state is exported to Statevector for the non-Clifford tail. Saves exponential memory for circuits with a long Clifford preamble.

## Backends

### Statevector

Full-state simulation in a flat `Vec<Complex64>` of 2^n amplitudes. The primary backend for circuits up to ~28 qubits.

Gate kernels use enum dispatch with specialized routines for CX, CZ, SWAP, Cu, MCU, Rzz, BatchRzz, BatchPhase, DiagonalBatch, and MultiFused. Single-qubit gates go through `PreparedGate1q` with FMA-vectorized SIMD. MultiFused gates use a three-tier tiled kernel (L2 16K / L3 131K / individual passes) for cache locality. MultiFused batches where all gates are diagonal dispatch to a dedicated fast path (1 complex multiply/element vs 4+2 for full 2×2).

Rayon parallelism at ≥14 qubits with `par_chunks_mut` and `MIN_PAR_ELEMS = 4096` per task. BMI2 `_pext_u64` accelerates BatchPhase, BatchRzz, and DiagonalBatch LUT indexing.

Deferred measurement normalization: `pending_norm` accumulates normalization factors without full-state scaling passes. Zero-cost for circuits without measurements.

### Stabilizer

Aaronson-Gottesman bit-packed tableau for Clifford circuits. O(n²) time and space. Scales to thousands of qubits. Gate kernels use wordwise bitwise ops and `popcount` for phase computation. Supports H, S, Sdg, SX, SXdg, CX, CZ, SWAP, and measurement.

Word-group batching fuses multiple 1q gate flushes into single tableau passes. Type-grouped masks apply all gates of the same Pauli type with one wordwise op instead of per-gate dispatch. Sparse Generator Indexing (SGI) tracks per-qubit active generator lists, enabling targeted row operations instead of full-tableau scans. Lazy destabilizer materialization defers destabilizer rows until probabilities are requested.

Probability extraction uses coset-based enumeration with GF(2) Gaussian elimination. O(2^k) where k is the number of non-diagonal generators, rather than O(2^n).

**Filtered Stabilizer** (`FilteredStabilizerBackend`): Per-cluster tableaux with dynamic merging. Starts with one qubit per cluster. Cross-cluster 2q gates merge tableaux. Independent subsystems never merge, giving O(block_size²) per gate vs O(n²).

### Sparse

`HashMap<usize, Complex64>` for states with few non-zero amplitudes. O(k) memory. Amplitude pruning (|a|² < 1e-16) after each gate. Best for circuits whose support stays concentrated in computational-basis states at large qubit counts.

### MPS (Matrix Product State)

Chain of rank-3 tensors with adaptive bond dimension (default max 256). O(n·χ²) memory. Single-qubit gates absorb via FMA-vectorized SIMD over bond-dimension slices. Two-qubit gates contract adjacent sites, apply the gate, then SVD-truncate back. Non-adjacent gates route through SWAP chains.

Hybrid SVD dispatch: faer (bidiag+D&C) for matrices with m×n ≥ 256, hand-rolled Jacobi for small matrices.

### Product State

Per-qubit `[Complex64; 2]` storage. O(n) memory, O(1) per single-qubit gate. Rejects entangling gates. Selected automatically for circuits with no 2q gates.

### Tensor Network

Deferred contraction with a greedy min-size heuristic. Gates append tensors; contraction happens lazily at measurement or probability extraction. `MAX_PROB_QUBITS = 25` guards against exponential blowup.

### Factored

Dynamic split-state simulation. Starts with n independent 1-qubit states, merges via tensor product only when 2q gates bridge groups. Parallel kernels match statevector patterns for sub-states ≥14 qubits. Selected when subsystem decomposition detects partial independence.

## GPU backend (optional, `gpu` feature)

CUDA acceleration covers statevector execution, stabilizer execution, and compiled
BTS sampling. Five entry points are available:

- **`BackendKind::StatevectorGpu { context }`**. Public dispatch path for statevector
  GPU execution. It routes through `sim::run_with`, keeps fusion and subsystem
  decomposition, and uses `crate::gpu::min_qubits()` (default 14,
  `PRISM_GPU_MIN_QUBITS` override) to keep small sub-circuits on CPU.
- **`BackendKind::StabilizerGpu { context }`**. Public dispatch path for stabilizer
  GPU execution. Gate application uses a device tableau and one batched Clifford
  kernel (`stab_apply_batch`). Measurement and reset keep pivot search, row cascade,
  phase fixup, and deterministic outcomes on the device. The default crossover stays
  conservative (`STABILIZER_MIN_QUBITS_DEFAULT = 100_000`,
  `PRISM_STABILIZER_GPU_MIN_QUBITS` override) until benchmarks justify lowering it.
  Direct backend benchmarks should use `StabilizerBackend::with_gpu(ctx)` to exclude
  diagnostic readbacks from `probabilities()`, `export_tableau()`, and
  `export_statevector()`. Golden tests cover every kernel path, including 500q GHZ
  measure-all.
- **`StatevectorBackend::new(seed).with_gpu(ctx)`**. Direct statevector GPU opt-in.
  Every instruction routes to CUDA after the context is attached. No crossover or
  subsystem decomposition applies.
- **`StabilizerBackend::new(seed).with_gpu(ctx)`**. Direct stabilizer GPU opt-in for
  kernel benchmarks and targeted correctness tests.
- **`run_shots_compiled_with_gpu`** (or `CompiledSampler::with_gpu(ctx)`). GPU BTS
  sampling for flat sparse parity. The path launches one kernel per `65_536`-shot
  chunk, uses random bits generated on the host, and preserves the CPU
  `sample_bts_meas_major` layout. The sampler caches sparse parity CSR arrays, packed
  reference bits, and reusable scratch on the device. It is active only when
  `num_shots >= BTS_MIN_SHOTS_DEFAULT` (`131_072` by default,
  `PRISM_GPU_BTS_MIN_SHOTS` override). `sample_bulk_packed_device` returns a
  `DevicePackedShots` handle. Marginals reduce to one counter per measurement row on
  the device. Exact counts use a bounded device hash reduction for up to 8 packed
  measurement words when the compact result is cheaper to transfer than the full shot
  matrix. Otherwise the API uses a host copy for correctness.

When a GPU context is attached, `Backend::init` allocates state on the device instead
of a host `Vec<Complex64>` and every instruction routes to a CUDA kernel.

**Module layout** (`src/gpu/`):

| File | Role |
| ---- | ---- |
| `mod.rs` | `GpuContext`, `GpuState` public entry points |
| `device.rs` | `GpuDevice`: cudarc wrapper, compiles PTX at device construction |
| `memory.rs` | `GpuBuffer`: device `Complex64` storage |
| `kernels/mod.rs` | `KERNEL_NAMES`, composed `kernel_source()` concatenating dense + stabilizer |
| `kernels/dense.rs` | PTX source and Rust launcher for every `Gate` variant |
| `kernels/stabilizer.rs` | PTX source and launchers for tableau init, 11 Clifford gates, `rowmul_words` |

**Kernel coverage:** every variant in the `Gate` enum has a dedicated kernel. Batched
variants (`BatchPhase`, `BatchRzz`, `DiagonalBatch`, `MultiFused { all_diagonal: true }`)
use LUT kernels that consume the same host table builders as the CPU path.
Non-diagonal `MultiFused` uses a shared memory tiled kernel (`apply_multi_fused_tiled`,
`TILE_Q = 10`, `TILE_SIZE = 1024`). Sub-gates whose target bit is inside the tile apply in
shared memory. Sub-gates whose target bit is outside the tile fall back to per gate
launches. `Multi2q` still launches once per sub-gate; rare in practice.

**PTX template substitution:** the CUDA C source is held as a template string
(`KERNEL_SOURCE_TEMPLATE`) with placeholders such as `{{BP_TABLE_SIZE}}` and
`{{TILE_Q}}`. The `kernel_source()` function substitutes them at device construction from
the Rust constants in `src/backend/statevector/kernels.rs`, keeping CPU and GPU in sync.

**Correctness:** `tests/golden_gpu.rs` drives 20 cross-checks comparing GPU amplitudes
against the CPU statevector within 1e-10. Covers every gate variant, fusion paths, and
the `BackendKind::StatevectorGpu` public dispatch path at the crossover boundary.

**Current limits:**

- `BackendKind::Auto` does not yet dispatch to GPU. The `StatevectorGpu` variant is the
  explicit opt-in until `Auto` can receive a default GPU context.
- Several fused launchers (`launch_apply_fused_2q`, `launch_apply_mcu`,
  `launch_apply_mcu_phase`, `launch_apply_batch_phase`, `launch_apply_batch_rzz`,
  `launch_apply_diagonal_batch`, `launch_apply_multi_fused_nondiag`) upload small
  metadata tables per dispatch via `clone_htod`. This is the next launch overhead
  target.
- `measure_prob_one` allocates a device partials buffer and reduces on the host every
  call. Matters for measurement-heavy circuits.
- Kernel design and crossover analysis live in the module docstrings on
  `src/gpu/kernels/dense.rs`. Cross-simulator comparison scripts are provided outside
  the public surface of the crate.

## Compiled samplers

For multi-shot sampling without materializing the full statevector on every shot.

### Noiseless compiled sampler (`src/sim/compiled/`)

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

**Detector sampler** (`compile_detector_sampler`): Compiles Clifford circuits
with measurement and reset reuse into the same packed measurement sampler, then
derives detector and observable records as packed parity rows over measurement
record indices. Reset reuse is represented by fresh qubit aliases, so repeated
syndrome extraction avoids per-shot tableau replay. The sampler can return
packed measurements, packed detectors, packed observables, detector counts, or
feed packed detector chunks into any `ShotAccumulator`.

### Noisy compiled sampler (`src/sim/noise.rs`)

Backward Pauli propagation through circuit + noise sensitivity analysis. Each noise location gets an X-flip and Z-flip sensitivity row. During sampling, Bernoulli coin flips determine which noise channels fire, then XOR the sensitivity rows into the sample.

`NoiseModel`: Per-instruction depolarizing noise. `NoiseOp { qubit, px, py, pz }`.

### Homological sampler (`src/sim/homological.rs`)

`ErrorChainComplex`: GF(2) chain complex over the circuit's noise locations. Computes the kernel (null space) of the boundary map to identify error cycles that are undetectable by syndrome measurements. `HomologicalSampler` uses this for sampling with topological error correction awareness.

`noisy_marginals_analytical`: Closed-form marginal computation using the parity matrix and noise rates. Avoids Monte Carlo sampling entirely.

## Clifford+T simulation

Three strategies for circuits mixing Clifford and T gates:

### Stabilizer rank (`src/sim/stabilizer_rank.rs`)

Maintains a weighted sum of stabilizer states. Each T gate doubles the term count via T = α·I + β·Z decomposition. Clifford gates are O(n²) per term. Accumulates weighted amplitudes for exact probabilities.

- `run_stabilizer_rank`: Exact probabilities (t ≤ 20, n ≤ 25)
- `run_stabilizer_rank_approx`: Approximate with Monte Carlo (higher t counts)
- `run_stabilizer_rank_shots`: Shot-based sampling
- `stabilizer_overlap_sq`: Inner product between stabilizer states

### Stochastic Pauli Propagation (`src/sim/unified_pauli.rs`)

Backward-propagates measurement observables as Pauli strings. Clifford gates conjugate in O(1). T gates branch stochastically into two Pauli paths with appropriate weights. Per-path cost O(d×n/64), independent of T-gate count. Returns marginal probabilities via Monte Carlo estimation.

`run_spp(circuit, num_samples, seed) → SppResult`

### Deterministic Sparse Pauli Dynamics (`src/sim/unified_pauli.rs`)

Backward-propagates as a weighted sum of Pauli strings stored in a HashMap. T gates deterministically branch X/Y terms. Identical strings auto-merge. Optional ε-truncation for approximate mode. Exact for small T-counts, approximate with bounded error for larger ones.

`run_spd(circuit, epsilon, max_terms) → SpdResult`

## Memory layout

| Backend | State representation | Memory | Access pattern |
|---------|---------------------|--------|----------------|
| Statevector | `Vec<Complex64>` (2^n) | O(2^n) | Strided pair iteration |
| Stabilizer | Bit-packed `Vec<u64>` tableau | O(n²/8) bytes | Sequential row iteration |
| Sparse | `HashMap<usize, Complex64>` | O(k), k = nonzero | Hash-based random access |
| MPS | Chain of rank-3 tensors | O(n·χ²) | Sequential site access |
| Product | `Vec<[Complex64; 2]>` | O(n) | Per-qubit independent |
| Tensor Network | Network of dense tensors | O(gates × local dim) | Contraction-order dependent |
| Factored | `Vec<Option<SubState>>` | O(2^n) worst case | Dispatch per substate |

## Threading and SIMD

### Threading

Gate kernels have `_par` variants using `par_chunks_mut` for safe Rayon parallelism (behind the `parallel` feature flag):

- **<14 qubits**: Single-threaded. Thread-pool overhead exceeds computation.
- **≥14 qubits**: Rayon parallel iterators with `MIN_PAR_ELEMS = 4096` (64KB per task).

Thread pool defaults to all logical cores (HT helps at 24q+ by hiding memory latency). Overridable via `RAYON_NUM_THREADS`.

### SIMD

`Complex64` maps to 128-bit SIMD naturally. Single-qubit gate kernels use `PreparedGate1q` with runtime CPU detection and tiered dispatch:

1. **AVX2+FMA** (256-bit): 2 complex pairs per iteration. Gated by `MAX_AVX2_STATE` for full-state passes (Skylake frequency throttling), but used freely within MultiFused L2 tiles where data is cache-resident.
2. **FMA** (128-bit): Default for larger states. 3-op complex multiply (permute + mul + fmaddsub).
3. **BMI2**: `_pext_u64` for BatchPhase, BatchRzz, and DiagonalBatch LUT indexing. One BMI2 bit extraction replaces loops with repeated shifts and ORs.
4. **Scalar fallback**: No intrinsics. All SIMD functions have a `#[cfg(not(target_arch = "x86_64"))]` fallback.

Two key SIMD structs hoist matrix broadcast at construction time, avoiding per-element dispatch:

- **`PreparedGate1q`**: Broadcasts 2×2 matrix into SIMD registers. Methods: `apply_full_sequential` (full state), `apply_tiled` (cache-resident tile, no AVX2 throttle guard), `apply_slice_pairs` (MPS bond-dimension slices), `apply_pair_ptr` (Cu/Mcu parallel).
- **`PreparedGate2q`**: Broadcasts 4×4 matrix. Methods: `apply_full` (mask-based iteration), `apply_tiled` (cache-resident Multi2q tiles, AVX2 paired-group kernel when available), `apply_group_ptr` (4 scattered indices).

The 2q tiled AVX2 path processes paired `k` and `k + 1` groups when the lower target qubit is above 0, which makes each row load contiguous. It falls back to the 128-bit FMA kernel for `lo == 0` and when AVX2+FMA is unavailable. Set `PRISM_NO_AVX2_2Q` to compare against the 128-bit FMA path, or `PRISM_NO_REORDER` to disable disjoint Fused2q tier grouping for A/B timing.

### Determinism

Same circuit + same seed = same result, regardless of thread count. Parallel backends use deterministic work partitioning.


## Backend dispatch

All `BackendKind` variants:

| Variant | Backend | Selection |
|---------|---------|-----------|
| `Auto` | Decision tree (see above) | Default |
| `Statevector` | Full state-vector | Explicit |
| `Stabilizer` | Aaronson-Gottesman tableau | Explicit or auto (all Clifford) |
| `FilteredStabilizer` | Per-cluster tableaux | Explicit |
| `Sparse` | HashMap state | Explicit or auto (above memory limit, sparse-friendly) |
| `Mps { max_bond_dim }` | Matrix Product State | Explicit or auto (above memory limit) |
| `ProductState` | Per-qubit product | Explicit or auto (no entangling) |
| `TensorNetwork` | Deferred contraction | Explicit |
| `Factored` | Dynamic split-state | Explicit or auto (partial independence) |
| `StabilizerRank` | Weighted stabilizer sum | Explicit |
| `StochasticPauli { num_samples }` | SPP | Explicit |
| `DeterministicPauli { epsilon, max_terms }` | SPD | Explicit |

## Circuit builders

Pre-built circuits for benchmarking and testing (`src/circuits.rs`):

| Function | Description |
|----------|-------------|
| `qft_circuit(n)` | Quantum Fourier Transform |
| `random_circuit(n, depth, seed)` | Random gates at given depth |
| `hardware_efficient_ansatz(n, layers, seed)` | HEA with Ry/Rz + CX |
| `clifford_heavy_circuit(n, depth, seed)` | Random Clifford (adjacent CX) |
| `clifford_random_pairs(n, depth, seed)` | Random Clifford (random pair CX) |
| `ghz_circuit(n)` | GHZ state (H + CX chain) |
| `qaoa_circuit(n, layers, seed)` | QAOA MaxCut |
| `single_qubit_rotation_circuit(n, depth, seed)` | 1q rotations only |
| `clifford_t_circuit(n, depth, t_fraction, seed)` | Clifford+T with tunable T ratio |
| `w_state_circuit(n)` | W state preparation |
| `quantum_volume_circuit(n, depth, seed)` | Quantum volume (random SU(4)) |
| `cz_chain_circuit(n, depth, seed)` | CZ chains |
| `phase_estimation_circuit(n)` | Quantum phase estimation |
| `independent_bell_pairs(n_pairs)` | Independent Bell pairs |
| `independent_random_blocks(blocks, size, depth, seed)` | Independent random blocks |

## Error model

All public APIs return `Result<T, PrismError>`. Error variants:

| Variant | Category | Description |
|---------|----------|-------------|
| `Parse` | Parsing | OpenQASM parse error with line number |
| `UnsupportedConstruct` | Parsing | Valid OpenQASM not supported by PRISM-Q |
| `UndefinedRegister` | Parsing | Reference to undeclared register |
| `InvalidQubit` | Validation | Qubit index exceeds register size |
| `InvalidClassicalBit` | Validation | Classical bit index exceeds register |
| `GateArity` | Validation | Wrong number of qubits for gate |
| `InvalidParameter` | Validation | Invalid gate parameter (NaN, etc.) |
| `BackendUnsupported` | Runtime | Backend can't perform requested operation |
| `IncompatibleBackend` | Runtime | Backend incompatible with circuit |

No panics on user input. `debug_assert!` for internal invariants only.

## Public API surface

Top-level re-exports from `src/lib.rs`:

**Simulation:**
`run`, `run_with`, `run_on`, `run_qasm`, `run_shots`, `run_shots_with`, `run_shots_with_noise`, `run_counts`, `run_marginals`, `bitstring`

**Compiled sampling:**
`compile_measurements`, `compile_forward`, `compile_detector_sampler`, `compile_noisy`, `run_shots_compiled`, `run_shots_noisy`, `run_shots_homological`, `noisy_marginals_analytical`

**Clifford+T:**
`run_stabilizer_rank`, `run_stabilizer_rank_approx`, `stabilizer_overlap_sq`, `run_spp`, `run_spd`, `spp_to_probabilities`, `spd_to_probabilities`

**Types:**
`Circuit`, `CircuitBuilder`, `Instruction`, `ClassicalCondition`, `Gate`, `BackendKind`, `SimulationResult`, `Probabilities`, `FactoredBlock`, `ShotsResult`, `PrismError`, `Result`

**Backends:**
`StatevectorBackend`, `StabilizerBackend`, `SparseBackend`, `MpsBackend`, `ProductStateBackend`, `TensorNetworkBackend`, `FactoredBackend`

**Accumulators:**
`ShotAccumulator`, `HistogramAccumulator`, `MarginalsAccumulator`, `PauliExpectationAccumulator`, `CorrelatorAccumulator`, `NullAccumulator`, `PackedShots`, `ShotLayout`

**Data types:**
`CompiledSampler`, `CompiledDetectorSampler`, `DetectorSampleBatch`, `NoisyCompiledSampler`, `HomologicalSampler`, `ErrorChainComplex`, `NoiseModel`, `NoiseOp`, `StabRankResult`, `SppResult`, `SpdResult`, `SparseParity`, `ParityStats`, `PauliVec`, `MultiFusedData`, `BatchPhaseData`, `McuData`, `Multi2qData`

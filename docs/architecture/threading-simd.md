# Threading, SIMD, and Memory Layout

For which SIMD tiers and architectures each backend supports, see the
[Capability and Support Matrix](../guides/capabilities.md).

## Memory layout

| Backend | State representation | Memory | Access pattern |
|---------|---------------------|--------|----------------|
| Statevector | `Vec<Complex64>` (2^n) | $O(2^n)$ | Strided pair iteration |
| Stabilizer | Bit-packed `Vec<u64>` tableau | $O(n^2/2)$ bytes | Sequential row iteration |
| Sparse | `HashMap<usize, Complex64>` | $O(k)$, $k$ = nonzero | Hash-based random access |
| MPS | Chain of rank-3 tensors | $O(n\chi^2)$ | Sequential site access |
| Product | `Vec<[Complex64; 2]>` | $O(n)$ | Per-qubit independent |
| Tensor Network | Network of dense tensors | $O(\text{gates} \times \text{local dim})$ | Contraction-order dependent |
| Factored | `Vec<Option<SubState>>` | $O(2^n)$ worst case | Dispatch per substate |

## Threading

Under the `parallel` feature, gate kernels have `_par` variants built on
`par_chunks_mut`. They engage at `PARALLEL_THRESHOLD_QUBITS = 14` (`src/backend/mod.rs`);
below that, thread-pool overhead exceeds the work. Each task covers at least
`MIN_PAR_ELEMS = 4096` amplitudes, 64 KiB.

The pool defaults to every logical core, and `RAYON_NUM_THREADS` overrides it.
Hyperthreads help at 24 qubits and above by hiding memory latency.

That pool is the process-wide Rayon pool, sized on the first simulation call. An
application that owns it can hand PRISM-Q a bounded one instead:
`ThreadPool::with_threads(n)` builds it and `install` runs a closure on it, leaving the
global pool unbuilt and unresized. Calls made outside `install` take the global path.
Pool width can change results; see [Determinism](#determinism).

## SIMD

One `Complex64` fills a 128-bit register. Single-qubit gate kernels use `PreparedGate1q` with runtime CPU detection and tiered dispatch:

1. **AVX2+FMA** (256-bit): 2 complex pairs per iteration. Gated by `MAX_AVX2_STATE` for full-state passes (Skylake frequency throttling), but used freely within MultiFused L2 tiles where data is cache-resident.
2. **FMA** (128-bit): Default for larger states. 3-op complex multiply (permute + mul + fmaddsub).
3. **SSE2** (128-bit): x86_64 baseline, 2 mul plus shuffle, xor and add.
4. **NEON** (128-bit): the aarch64 baseline, 1 mul plus 1 fma against a pre-negated matrix.
5. **Scalar**: no intrinsics, for everything that is neither x86_64 nor aarch64.

`_pext_u64` is not a tier. BatchPhase, BatchRzz and DiagonalBatch use it for LUT
indexing where BMI2 is present, one bit extraction in place of a shift-and-or loop, and
fall back to that loop where it is not.

Two structs broadcast the matrix into registers once, at construction, so the per-element loop does no dispatch:

- `PreparedGate1q`: the 2×2 matrix. Methods: `apply_full_sequential` (full state), `apply_tiled` (cache-resident tile, no AVX2 throttle guard), `apply_slice_pairs` (MPS bond-dimension slices), `apply_pair_ptr` (Cu/Mcu parallel).
- `PreparedGate2q`: the 4×4 matrix. Methods: `apply_full` (mask-based iteration), `apply_tiled` (cache-resident Multi2q tiles, AVX2 paired-group kernel when available), `apply_group_ptr` (4 scattered indices).

The 2q tiled AVX2 path processes paired `k` and `k + 1` groups when the lower target qubit is above 0, which makes each row load contiguous. It falls back to the 128-bit FMA kernel for `lo == 0` and when AVX2+FMA is unavailable. Set `PRISM_NO_AVX2_2Q` to compare against the 128-bit FMA path, or `PRISM_NO_REORDER` to disable disjoint Fused2q tier grouping for A/B timing.

## Determinism

Reproducibility is stated per path. Gate application never reduces across tasks, so it
is exactly reproducible; everything that sums floating-point values in parallel is
reproducible to the last ulp only; the batched compiled sampler is reproducible only at
a fixed thread count. `tests/determinism.rs` pins the dense unitary,
terminal sampling, reduction, and compiled-sampler claims below by running the same
seeded circuits in scoped 1-thread and 4-thread pools; the trajectory, SPD, and
stabilizer bullets stand on the mechanisms they state.

Every clause turns on thread count, not on which pool supplied the threads, so a
caller-supplied `ThreadPool` moves exactly the results a different `RAYON_NUM_THREADS`
would: the bitwise claims survive it, the reduction bound holds across it, and compiled
shot payloads change with it whenever the pool is narrower or wider than the one the
comparison run used.

- **Unitary evolution on the dense kernels: bitwise, at any thread count.** Gate kernels
  partition the state into index-derived disjoint ranges (fixed chunk boundaries, index
  bijections for the `SendPtr` kernels) and write elementwise, so the schedule cannot
  reach any value. Pinned bitwise for representative fused circuits and a QFT.
- **Terminal sampling on the dense route: bitwise for a given seed, at any thread
  count**, when the measurement map covers the register directly. Shot thresholds are
  drawn and sorted up front from the seeded generator, the probability vector is an
  elementwise transform, and the CDF walk is sequential. A measurement map that compacts
  into fewer outcomes builds its histogram through a parallel reduction and moves to the
  ulp-stable class below.
- **Noisy trajectory shots: bitwise for a given seed, at any thread count.** Each shot's
  generator is seeded from the shot index, not the worker, and results are collected in
  shot order.
- **Parallel reductions: stable to about 1e-12, not bitwise.** Norms, measurement
  collapse probabilities, reduced density matrices, and expectation values sum
  deterministic per-chunk partials in Rayon's combine order, which varies with pool width
  and work stealing. A mid-circuit measurement compares a seeded draw against such a sum,
  so an outcome flip is possible in principle when the draw lands inside the ulp gap.
- **Compiled (BTS) sampling: reproducible at a fixed thread count only.** The batched
  sampler derives one RNG stream per worker and splits shots by
  `rayon::current_num_threads()`, so a different pool width yields a different, equally
  distributed shot set. Pin the width when byte-identical shot payloads matter across
  machines, through `RAYON_NUM_THREADS` on the global path or through the width passed to
  `ThreadPool::with_threads` on the scoped one. The GPU analogue is documented in the [GPU guide](../guides/gpu.md).
- **SPD analytic estimates: stable to about 1e-12 between runs.** Hash-order term
  accumulation moves the last ulp even at a fixed thread count; tests carry that
  tolerance.
- **Stabilizer tableau: bitwise, at any thread count.** Row operations are integer and
  exactly associative. MPS carries no bitwise claim: truncation runs through faer's
  threaded SVD.

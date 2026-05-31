# Threading, SIMD, and Memory Layout

For which SIMD tiers and architectures each backend supports, see the
[Capability and Support Matrix](../guides/capabilities.md).

## Memory layout

| Backend | State representation | Memory | Access pattern |
|---------|---------------------|--------|----------------|
| Statevector | `Vec<Complex64>` (2^n) | $O(2^n)$ | Strided pair iteration |
| Stabilizer | Bit-packed `Vec<u64>` tableau | $O(n^2/8)$ bytes | Sequential row iteration |
| Sparse | `HashMap<usize, Complex64>` | $O(k)$, $k$ = nonzero | Hash-based random access |
| MPS | Chain of rank-3 tensors | $O(n\chi^2)$ | Sequential site access |
| Product | `Vec<[Complex64; 2]>` | $O(n)$ | Per-qubit independent |
| Tensor Network | Network of dense tensors | $O(\text{gates} \times \text{local dim})$ | Contraction-order dependent |
| Factored | `Vec<Option<SubState>>` | $O(2^n)$ worst case | Dispatch per substate |

## Threading

Gate kernels have `_par` variants using `par_chunks_mut` for safe Rayon parallelism (behind the `parallel` feature flag):

- **<14 qubits**: Single-threaded. Thread-pool overhead exceeds computation.
- **≥14 qubits**: Rayon parallel iterators with `MIN_PAR_ELEMS = 4096` (64KB per task).

Thread pool defaults to all logical cores (HT helps at 24q+ by hiding memory latency). Overridable via `RAYON_NUM_THREADS`.

## SIMD

`Complex64` maps to 128-bit SIMD naturally. Single-qubit gate kernels use `PreparedGate1q` with runtime CPU detection and tiered dispatch:

1. **AVX2+FMA** (256-bit): 2 complex pairs per iteration. Gated by `MAX_AVX2_STATE` for full-state passes (Skylake frequency throttling), but used freely within MultiFused L2 tiles where data is cache-resident.
2. **FMA** (128-bit): Default for larger states. 3-op complex multiply (permute + mul + fmaddsub).
3. **BMI2**: `_pext_u64` for BatchPhase, BatchRzz, and DiagonalBatch LUT indexing. One BMI2 bit extraction replaces loops with repeated shifts and ORs.
4. **Scalar fallback**: No intrinsics. All SIMD functions have a `#[cfg(not(target_arch = "x86_64"))]` fallback.

Two key SIMD structs hoist matrix broadcast at construction time, avoiding per-element dispatch:

- **`PreparedGate1q`**: Broadcasts 2×2 matrix into SIMD registers. Methods: `apply_full_sequential` (full state), `apply_tiled` (cache-resident tile, no AVX2 throttle guard), `apply_slice_pairs` (MPS bond-dimension slices), `apply_pair_ptr` (Cu/Mcu parallel).
- **`PreparedGate2q`**: Broadcasts 4×4 matrix. Methods: `apply_full` (mask-based iteration), `apply_tiled` (cache-resident Multi2q tiles, AVX2 paired-group kernel when available), `apply_group_ptr` (4 scattered indices).

The 2q tiled AVX2 path processes paired `k` and `k + 1` groups when the lower target qubit is above 0, which makes each row load contiguous. It falls back to the 128-bit FMA kernel for `lo == 0` and when AVX2+FMA is unavailable. Set `PRISM_NO_AVX2_2Q` to compare against the 128-bit FMA path, or `PRISM_NO_REORDER` to disable disjoint Fused2q tier grouping for A/B timing.

## Determinism

Same circuit + same seed = same result, regardless of thread count. Parallel backends use deterministic work partitioning.

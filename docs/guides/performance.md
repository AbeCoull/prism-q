# Performance and SIMD

The mechanisms that make a run fast and the variables that tune them. Internals are in
[Fusion Pipeline](../architecture/fusion.md) and
[Threading, SIMD, and Memory Layout](../architecture/threading-simd.md).

## Fusion, tiling and SIMD

1. Fusion merges many small gate passes into fewer, larger ones before execution,
   cutting memory traffic over the statevector. It is gated on qubit count and costs
   nothing when it does not apply.
2. Cache-resident tiling keeps batched gates on data that stays hot across a pass,
   in two shapes. Both size tiles from one budget read off the cache topology at
   startup: the smaller of the L2 per core and a quarter of the last-level cache per
   logical CPU sharing it, floored at 256 KB and capped at 1 MB, so every concurrent
   tile and the lines it gathers fit half that cache. On the i7-6700K used for
   measurement that is 256 KB; a 1 MB tile ran 2x slower there at 22 and 24 qubits.

   `MultiFused` splits its gates by target: targets whose period fits the budget run on
   tiles of the budget size (bit 13 and below at 256 KB), bits 14 through 16 on 2 MB tiles
   when those fit the last-level cache, and higher targets run untiled. A `Multi2q`
   tile is a contiguous run of the low index bits plus gathered high qubits, 64
   amplitudes per run and up to eight high qubits in a 2^14 tile, so a batch of gates
   costs one pass over the state as long as it spans at most eight qubits above bit 6.
   A `MultiPauliRot` batch uses the same tile, counting only qubits that carry an X or
   Y letter, since a Z letter outside the tile fixes one sign per tile.

   `PRISM_TILE_KB` overrides the budget for every tile, and `PRISM_MULTI_2Q_TILE_BITS`
   and `PRISM_MULTI_2Q_LOW_BITS` set the `Multi2q` shape by hand. Budgets past 256 KB
   are derived, not measured, so a core with a larger cache is worth a sweep of
   `PRISM_TILE_KB`.
3. SIMD vectorizes the inner complex-arithmetic loop. The tier is picked at
   runtime: AVX-512 (opt-in, see `PRISM_AVX512` below), AVX2+FMA, FMA or SSE2 on x86_64,
   NEON on aarch64, scalar elsewhere.

Removing arithmetic beats issuing it faster. A kernel whose operations an algebraic
identity or the operator's structure deletes is bounded by its memory floor, while a
vectorized kernel is still bounded by complex-arithmetic issue.

## Threading

Rayon parallel kernels engage at 15 qubits, below which pool overhead dominates, with
`MIN_PAR_ELEMS = 4096` per task. The pool defaults to all logical cores.

```admonish tip title="Control the thread pool"
Set `RAYON_NUM_THREADS` to cap parallelism. Hyperthreading helps at 24+ qubits by hiding
memory latency, but on a contended host it adds noise to benchmarks.

An application that already owns the process-wide Rayon pool can keep it: build a
`ThreadPool::with_threads(n)` and run simulations inside `install`. The global pool is
left unbuilt on that path.
```

## Determinism

Unitary evolution and seeded terminal sampling on the dense backends are bitwise
reproducible at any thread count; parallel reductions and the batched compiled sampler
are not. [Determinism](../architecture/threading-simd.md#determinism) states the
contract per path.

## Tuning environment variables

Every knob is read once per process, on first use, and cached. A value that does not
parse, or falls below the knob's minimum, prints a stderr warning naming the variable
and is treated as unset, so a typo neither fails a run nor passes silently. A flag needs
only to be present: any value switches the path off.

### Limits (also settable in code)

Each cap below is both a `Limits` field and a variable. `Simulate::limits` (Python
`Simulation.limits`) sets it for one request and `set_default_limits` for every later
call in the process; a request value beats the process default, which beats the
variable. `memory_budget_bytes` sets the budget every cap derives from (half of physical
memory by default), and a layer that sets it discards the per-cap values below it.
`auto_mps_bond_dim` (default 256) has no variable.

| `Limits` field | Variable | Default | Effect |
|----------------|----------|---------|--------|
| `max_statevector_qubits` | `PRISM_MAX_SV_QUBITS` | detected | Statevector qubit cap; see [Backends](../architecture/backends.md#memory-budget) for every cap |
| `max_density_matrix_qubits` | `PRISM_MAX_DM_QUBITS` | `floor(cap_sv / 2)` | Density-matrix qubit cap, bounded by the statevector cap |
| `max_probability_qubits` | `PRISM_MAX_PROB_QUBITS` | detected | Dense probability output cap |
| `max_export_qubits` | `PRISM_MAX_EXPORT_QUBITS` | detected | Dense statevector export cap |
| `max_sparse_qubits` | `PRISM_MAX_SPARSE_QUBITS` | detected | Sparse amplitude map holds at most `2^q` entries |
| `max_factored_merge_qubits` | `PRISM_MAX_FACTORED_MERGE_QUBITS` | detected | Factored merged-block width |
| `max_mps_workspace_qubits` | `PRISM_MAX_MPS_WORKSPACE_QUBITS` | detected | MPS contraction workspace, at most `2^q` amplitudes |
| `max_tn_peak_qubits` | `PRISM_MAX_TN_PEAK_QUBITS` | detected | Largest planned tensor-network intermediate, `2^q` elements |
| `max_tn_slices` | `PRISM_MAX_TN_SLICES` | `1024` | Independent slices a tensor-network contraction may sum over to fit the peak cap (minimum 1; `1` leaves slicing off) |
| `max_stabilizer_cluster_qubits` | `PRISM_MAX_STABILIZER_CLUSTER_QUBITS` | detected | Factored stabilizer merged-cluster width |

### Tuning (environment only)

| Variable | Default | Effect |
|----------|---------|--------|
| `PRISM_MAX_DENSE_OUTCOME_BITS` | detected | Measured-bit cap for dense terminal sampling |
| `PRISM_QFT_TWIDDLE_CACHE_LIMIT_MB` | `256` | Soft cap on cached QFT twiddle tables; `0` disables the cache |
| `PRISM_AVX512` | unset | `1` runs on AVX-512 the statevector gate kernels (single-qubit, `Multi2q` tiles, parallel `Fused2q` runs), QFT butterflies and bulk amplitude ops, the density-matrix two-qubit Kraus block, the stabilizer row kernel (with its streaming-store form), bulk word XOR, and the compiled sampler's propagation, BTS, random-word and transpose kernels where the CPU has `avx512f` and `avx512vpopcntdq`; verified under emulation, not yet timed on AVX-512 hardware |
| `PRISM_GPU_MIN_QUBITS` | `14` | Auto GPU crossover qubit count (`gpu` feature) |
| `PRISM_STABILIZER_GPU_MIN_QUBITS` | `100000` | Stabilizer GPU crossover qubit count |
| `PRISM_GPU_BTS_MIN_SHOTS` | `131072` | Shot count from which compiled BTS sampling runs on the device |
| `PRISM_GPU_BTS_MIN_RANK` | `4` | Compiled-sampler rank from which BTS sampling runs on the device (minimum 1) |
| `PRISM_GPU_BTS_MIN_WEIGHT_FACTOR` | `2` | Parity-weight factor for the BTS device route (minimum 1) |
| `PRISM_DIST_MIN_LOCAL_QUBITS` | `10` | Minimum local qubits per rank before distribution is worthwhile (minimum 1) |
| `PRISM_DIST_EXCHANGE_CHUNK` | `536870912` (2^29) | Amplitudes per message on the rank exchange paths (minimum 1) |
| `PRISM_DIST_RELABEL` | `1` | `0`/`false` disables qubit relabeling in the distributed backend |
| `RAYON_NUM_THREADS` | all cores | Rayon thread count, read by Rayon itself |
| `PRISM_NO_AVX2_2Q` | unset | Flag: force the 128-bit FMA two-qubit kernel |
| `PRISM_MIN_TILES_PER_CPU` | `1` | Fewest tiles a parallel tiled pass gives each hardware thread; the tile shrinks to reach the count. `0` keeps every pass at its budget tile |
| `PRISM_TILE_KB` | detected | Cache-resident tile budget in KB, rounded down to a power of two, 128 to 2048; derived from the cache topology when unset (256 on a 256 KB L2) |
| `PRISM_MULTI_2Q_TILE_BITS` | detected | Log2 of the `Multi2q` tile in amplitudes, 13 to 18; the tile budget when unset (14 at 256 KB) |
| `PRISM_MULTI_2Q_LOW_BITS` | 6 | Log2 of the contiguous run a `Multi2q` tile gathers, 6 to one below the tile bits; fewer means more high qubits per pass |
| `PRISM_NO_AVX2_KRAUS` | unset | Flag: disable the AVX2 and AVX-512 dense two-qubit Kraus kernels |
| `PRISM_NO_REORDER` | unset | Flag: disable the `Fused2q` reordering that fills each `Multi2q` tile |
| `PRISM_NO_QFT_BLOCK` | unset | Flag: expand `QftBlock` to the textbook sequence and leave expanded QFTs as gates |

## Benchmarking

Run benchmarks with `--features parallel`, one process at a time.
[Benchmark Methodology](./benchmarking.md) has the targets, the A/B gate and the
comparison controls, and [Performance](../performance/index.md) the current numbers.

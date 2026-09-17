# Performance and SIMD

Performance is the primary product requirement. This guide explains the mechanisms that
make PRISM-Q fast and the knobs you can turn. The internals live in the architecture
reference under [Fusion Pipeline](../architecture/fusion.md) and
[Threading, SIMD, and Memory Layout](../architecture/threading-simd.md).

## The three levers

1. **Fusion** collapses many small gate passes into fewer, larger ones before execution,
   reducing memory traffic over the statevector. It is qubit-count gated and zero-cost
   when it does not apply.
2. **Cache-resident tiling** keeps batched gates (`MultiFused`, `Multi2q`) operating on
   L2/L3-sized tiles so repeated passes reuse hot data.
3. **SIMD** vectorizes the inner complex-arithmetic loop with AVX2+FMA, FMA, and BMI2,
   with a scalar fallback on non-x86_64.

The levers are ordered, and lever 3 comes with a prior question: can the arithmetic be
removed rather than issued faster? A kernel whose operations an algebraic identity or an
operator structure deletes is bounded by its memory floor; vectorizing what remains is
bounded by the complex-arithmetic issue ceiling, near 23% of FMA peak in the interleaved
layout.

## Threading

Rayon parallel kernels engage at **≥14 qubits** (below that, thread-pool overhead
dominates), with `MIN_PAR_ELEMS = 4096` per task. The pool defaults to all logical cores.

```admonish tip title="Control the thread pool"
Set `RAYON_NUM_THREADS` to cap parallelism. Hyperthreading helps at 24+ qubits by hiding
memory latency, but on a contended host it adds noise to benchmarks.

An application that already owns the process-wide Rayon pool can keep it: build a
`ThreadPool::with_threads(n)` and run simulations inside `install`. The global pool is
left unbuilt on that path.
```

## Determinism

Deterministic partitioning makes unitary evolution and seeded terminal sampling on the
dense backends bitwise reproducible at any thread count. Parallel reductions (norms,
collapse probabilities, expectation values) are stable to about 1e-12 but not bitwise,
and the batched compiled sampler seeds one RNG stream per worker, so its shots reproduce
only at a fixed thread count. The per-path contract is in
[Threading, SIMD, and Memory Layout](../architecture/threading-simd.md).

## Tuning environment variables

Every knob is read once per process, on first use, and cached. A value that
does not parse, or falls below the knob's minimum, prints a warning on stderr
naming the variable and behaves as if the variable were unset: a typo never
fails a run and never passes silently. Flags are presence-only: setting the
variable to anything switches the path off.

| Variable | Default | Effect |
|----------|---------|--------|
| `PRISM_MAX_SV_QUBITS` | detected | Statevector qubit cap; see [Backends](../architecture/backends.md#memory-budget) for every cap |
| `PRISM_MAX_DM_QUBITS` | `floor(cap_sv / 2)` | Density-matrix qubit cap, bounded by the statevector cap |
| `PRISM_MAX_PROB_QUBITS` | detected | Dense probability output cap |
| `PRISM_MAX_EXPORT_QUBITS` | detected | Dense statevector export cap |
| `PRISM_MAX_DENSE_OUTCOME_BITS` | detected | Measured-bit cap for dense terminal sampling |
| `PRISM_MAX_SPARSE_QUBITS` | detected | Sparse amplitude map holds at most `2^q` entries |
| `PRISM_MAX_FACTORED_MERGE_QUBITS` | detected | Factored merged-block width |
| `PRISM_MAX_MPS_WORKSPACE_QUBITS` | detected | MPS contraction workspace, at most `2^q` amplitudes |
| `PRISM_MAX_TN_PEAK_QUBITS` | detected | Largest planned tensor-network intermediate, `2^q` elements |
| `PRISM_MAX_STABILIZER_CLUSTER_QUBITS` | detected | Factored stabilizer merged-cluster width |
| `PRISM_QFT_TWIDDLE_CACHE_LIMIT_MB` | `256` | Soft cap on cached QFT twiddle tables; `0` disables the cache |
| `PRISM_GPU_MIN_QUBITS` | `14` | Auto GPU crossover qubit count (`gpu` feature) |
| `PRISM_STABILIZER_GPU_MIN_QUBITS` | `100000` | Stabilizer GPU crossover qubit count |
| `PRISM_GPU_BTS_MIN_SHOTS` | `131072` | Shot count from which compiled BTS sampling runs on the device |
| `PRISM_GPU_BTS_MIN_RANK` | `4` | Compiled-sampler rank from which BTS sampling runs on the device (minimum 1) |
| `PRISM_GPU_BTS_MIN_WEIGHT_FACTOR` | `2` | Parity-weight factor for the BTS device route (minimum 1) |
| `PRISM_DIST_MIN_LOCAL_QUBITS` | `10` | Minimum local qubits per rank before distribution is worthwhile (minimum 1) |
| `PRISM_DIST_EXCHANGE_CHUNK` | unbounded | Amplitudes per message on the rank exchange paths (minimum 1) |
| `PRISM_DIST_RELABEL` | `1` | `0`/`false` disables qubit relabeling in the distributed backend |
| `RAYON_NUM_THREADS` | all cores | Rayon thread count, read by Rayon itself |
| `PRISM_NO_AVX2_2Q` | unset | Flag: force the 128-bit FMA two-qubit kernel |
| `PRISM_NO_AVX2_KRAUS` | unset | Flag: disable the AVX2 dense two-qubit Kraus kernel |
| `PRISM_NO_REORDER` | unset | Flag: disable disjoint `Fused2q` tier grouping |
| `PRISM_NO_QFT_BLOCK` | unset | Flag: expand `QftBlock` to the textbook sequence |

## Benchmarking

```admonish warning title="Benchmark with the parallel feature"
Always run benchmarks with `--features parallel`. The baselines were taken with Rayon
enabled; without it, large circuits run single-threaded and are not comparable. Never run
two `cargo bench` processes at once: competing Rayon pools cause large swings.
```

```bash
cargo bench --bench circuits --features parallel       # circuit macrobenchmarks
cargo bench --bench bench_driver --features parallel   # gate microbenchmarks
```

For current wall-clock numbers across the circuit suite, see the
[Benchmarks](../benchmarks.md) page.

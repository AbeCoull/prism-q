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

## Threading

Rayon parallel kernels engage at **≥14 qubits** (below that, thread-pool overhead
dominates), with `MIN_PAR_ELEMS = 4096` per task. The pool defaults to all logical cores.

```admonish tip title="Control the thread pool"
Set `RAYON_NUM_THREADS` to cap parallelism. Hyperthreading helps at 24+ qubits by hiding
memory latency, but on a contended host it adds noise to benchmarks.
```

## Determinism

Same circuit plus same seed yields the same result regardless of thread count. Parallel
backends use deterministic work partitioning, so reproducibility never costs correctness.

## Tuning environment variables

| Variable | Effect |
|----------|--------|
| `PRISM_MAX_SV_QUBITS` | Override the statevector memory cap |
| `RAYON_NUM_THREADS` | Cap Rayon thread count |
| `PRISM_NO_AVX2_2Q` | Force the 128-bit FMA 2q kernel (A/B comparison) |
| `PRISM_NO_REORDER` | Disable disjoint `Fused2q` tier grouping |
| `PRISM_GPU_MIN_QUBITS` | GPU crossover qubit count (with the `gpu` feature) |

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

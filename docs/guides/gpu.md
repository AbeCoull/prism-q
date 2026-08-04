# GPU Backend

```admonish info
The GPU backend is optional and gated behind the `gpu` feature. It requires the CUDA
toolkit (12.x or newer) and a CUDA-capable device.
```

```bash
cargo build --release --features "parallel gpu"
cargo test --features "parallel gpu" --test golden_gpu
```

CUDA acceleration covers statevector execution, stabilizer execution, and compiled
BTS sampling. Six entry points are available:

- **`BackendKind::AutoGpu { context }`** (`simulate(circuit).gpu_auto(ctx)`).
  Automatic backend selection with the device opted in. The shape-based decision
  tree runs unchanged; a selected statevector or stabilizer workload that clears the
  family's qubit crossover and fits in VRAM runs on the device. Everything else,
  including a device allocation that fails at `init`, takes the identical CPU path
  (the soft VRAM fallback).
- **`BackendKind::StatevectorGpu { context }`**. Public dispatch path for statevector
  GPU execution. It routes through `simulate(circuit).backend(kind).seed(seed).run()`,
  keeps fusion and subsystem
  decomposition, and uses `crate::gpu::min_qubits()` (default 14,
  `PRISM_GPU_MIN_QUBITS` override) to keep small sub-circuits on CPU.
- **`BackendKind::StabilizerGpu { context }`**. Public dispatch path for stabilizer
  GPU execution. Gate application uses a device tableau and one word-grouped batched
  Clifford kernel (`stab_apply_word_grouped`). Measurement and reset keep pivot
  search, row cascade,
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

The three `BackendKind` entry points are also reachable from Python, from a build
carrying the `gpu` feature. See [Python Bindings](python.md#gpu-backends).

## Module layout (`src/gpu/`)

| File | Role |
| ---- | ---- |
| `mod.rs` | `GpuContext`, `GpuState` public entry points |
| `device.rs` | `GpuDevice`: cudarc wrapper, compiles PTX at device construction |
| `memory.rs` | `GpuBuffer`: device `Complex64` storage |
| `kernels/mod.rs` | `KERNEL_NAMES`, `LauncherScratch`, composed `kernel_source()` concatenating dense + stabilizer + BTS |
| `kernels/dense.rs` | PTX source and Rust launcher for every `Gate` variant |
| `kernels/stabilizer.rs` | PTX source and launchers for tableau init, 11 Clifford gates, `rowmul_words` |
| `kernels/bts.rs` | PTX source and launchers for compiled BTS shot sampling |

## Kernel coverage

Every variant in the `Gate` enum has a dedicated kernel. Batched
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

## Correctness

`tests/golden_gpu.rs` drives 20 cross-checks comparing GPU amplitudes
against the CPU statevector within 1e-10. Covers every gate variant, fusion paths, and
the `BackendKind::StatevectorGpu` public dispatch path at the crossover boundary.

### Shot reproducibility

Two limits bound what a seed guarantees, and neither is visible from the golden
equality tests.

- **CPU against GPU:** agreement in distribution, not bit for bit. Both paths draw the
  same RNG stream for a given shot seed, but the device reduction that produces a
  measurement probability sums in tree order with FMA contraction, so it can differ from
  the host sum in the last ulp, and a uniform draw landing between the two flips that
  outcome and every outcome after it. Where the probability is a dyadic rational (0.5,
  1.0, and the amplitudes reachable from Clifford gates) both sums are exact and the
  shots do match, which is what `statevector_gpu_mid_measure_shots_match_cpu` pins.
  `statevector_gpu_shot_frequencies_match_cpu_off_dyadic` pins the general case: equal
  frequencies within 5 sigma after `Rx(0.3)`.
- **GPU BTS sampling:** reproducible at a fixed Rayon thread count. Above
  `MIN_PAR_DRAWS` random bits per chunk, `fill_random_bits` seeds one stream per worker
  and partitions the draws by `rayon::current_num_threads()`, so the same seed on a host
  with a different worker count produces different shots. Below that threshold the
  serial single-stream path runs and the seed reproduces outright. Pin
  `RAYON_NUM_THREADS` when byte-identical shot payloads matter across machines.

## Current limits

- Device placement is silent. Circuits below the crossover run on the host, and the
  `AutoGpu` soft VRAM fallback degrades to host execution without a report; nothing
  user-visible says whether a run executed on the device.
- Stabilizer `probabilities()`, `export_tableau()`, and `export_statevector()` read
  back to the CPU.
- A `Custom` Kraus noise event reads the full state back to the host, and every
  trajectory shot rebuilds the backend, reallocating the device buffer.
- Kernel design and crossover analysis live in the module docstrings on
  `src/gpu/kernels/dense.rs`.

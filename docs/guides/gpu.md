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
BTS sampling. Five entry points are available:

- **`BackendKind::StatevectorGpu { context }`**. Public dispatch path for statevector
  GPU execution. It routes through `simulate(circuit).backend(kind).seed(seed).run()`,
  keeps fusion and subsystem
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

## Module layout (`src/gpu/`)

| File | Role |
| ---- | ---- |
| `mod.rs` | `GpuContext`, `GpuState` public entry points |
| `device.rs` | `GpuDevice`: cudarc wrapper, compiles PTX at device construction |
| `memory.rs` | `GpuBuffer`: device `Complex64` storage |
| `kernels/mod.rs` | `KERNEL_NAMES`, composed `kernel_source()` concatenating dense + stabilizer |
| `kernels/dense.rs` | PTX source and Rust launcher for every `Gate` variant |
| `kernels/stabilizer.rs` | PTX source and launchers for tableau init, 11 Clifford gates, `rowmul_words` |

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

## Current limits

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
  `src/gpu/kernels/dense.rs`.

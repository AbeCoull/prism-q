# GPU Backend

```admonish info
The GPU backend is optional and gated behind the `gpu` feature. Building it needs no
CUDA toolkit: the NVIDIA driver loads when the first `GpuContext` opens, and NVRTC
when kernels need compiling.
Running it needs a CUDA-capable device, a driver for CUDA 12.0 or newer, and NVRTC
from a CUDA 12 toolkit on the loader path (`PATH` on Windows, the `ld.so` search path
on Linux). A missing driver or NVRTC surfaces as a `PrismError` from `GpuContext::new`.
```

```bash
cargo build --release --features "parallel gpu"
cargo nextest run --features "parallel gpu" --test golden_gpu --test golden_gpu_density_matrix
```

Constructing a `GpuContext` compiles the CUDA kernels through NVRTC, one to three
seconds on a GTX 1080 Ti. NVRTC emits a cubin for the device's exact architecture, which
loads on any CUDA 12 driver. An NVRTC older than the device compiles PTX for the newest
architecture it knows, and the driver JITs it. That path needs a driver at least as new
as the NVRTC, and `GpuContext::new` names both versions when it is not.

The compiled image is cached in `prism-q-ptx` under the user cache directory, keyed by
device architecture, crate version and kernel source, so later processes skip the
compile. A cached cubin loads without NVRTC. Delete the directory to force a recompile.

The device runs statevector, stabilizer and density matrix execution and compiled BTS
sampling. `BackendKind` has four device kinds, and the backends and samplers below opt in
directly:

- `BackendKind::AutoGpu { context }` (`simulate(circuit).gpu_auto(ctx)`). The decision
  tree runs unchanged, and a selected statevector or stabilizer workload that clears its
  family's qubit crossover and fits in VRAM runs on the device. Everything else,
  including a device allocation that fails at `init`, takes the identical CPU path.
  `RunMetadata::placement` reports `Host` or `Device` for the run.
- `BackendKind::StatevectorGpu { context }` (`simulate(circuit).gpu(ctx)`). Keeps fusion
  and subsystem decomposition, and leaves sub-circuits below `crate::gpu::min_qubits()`
  (default 14, `PRISM_GPU_MIN_QUBITS` override) on the CPU.
- `BackendKind::StabilizerGpu { context }`. Gates apply to a device tableau through
  one word-grouped batched Clifford kernel (`stab_apply_word_grouped`). Measurement and
  reset keep pivot search, row cascade, phase fixup and deterministic outcomes on the
  device. The default crossover is conservative (`STABILIZER_MIN_QUBITS_DEFAULT =
  100_000`, `PRISM_STABILIZER_GPU_MIN_QUBITS` override). Direct backend benchmarks should
  use `StabilizerBackend::with_gpu(ctx)` to exclude diagnostic readbacks from
  `probabilities()`, `export_tableau()` and `export_statevector()`. Golden tests cover
  every kernel path, including 500q GHZ measure-all.
- `BackendKind::DensityMatrixGpu { context }`. The exact mixture in device memory.
  Explicit only: neither `Auto` nor `AutoGpu` selects it, and it has no crossover and
  no host fallback. `init` budgets the `4^n` buffer against free VRAM and errors before
  allocating when it does not fit, so an 11 GiB card holds 14 qubits (4 GiB at 14
  plus a `2^n` diagonal scratch; 15 needs 16 GiB). The unitary half reuses the dense
  statevector kernels on the embedded `2n`-qubit buffer, and every channel, measurement
  and readout sweep has a kernel of its own. The noisy `Simulate` terminals answer from
  the device mixture exactly as `DensityMatrix` does.
  `DensityMatrixBackend::new(seed).with_gpu(ctx)` is the direct form.
- `StatevectorBackend::new(seed).with_gpu(ctx)`. Every instruction routes to CUDA once
  the context is attached, with no crossover or subsystem decomposition.
- `StabilizerBackend::new(seed).with_gpu(ctx)`, for kernel benchmarks and targeted
  correctness tests.
- `run_shots_compiled_with_gpu` (or `CompiledSampler::with_gpu(ctx)`). BTS sampling for
  flat sparse parity: one kernel per chunk of `65_536` shots, random bits generated on
  the host, and the CPU `sample_bts_meas_major` layout. Sparse parity CSR arrays, packed
  reference bits and scratch stay cached on the device. A sampler stays on the CPU,
  where launch and transfer setup would cost more than the kernel saves, below `131_072`
  shots (`PRISM_GPU_BTS_MIN_SHOTS`), below rank 4 (`PRISM_GPU_BTS_MIN_RANK`; GHZ and
  other low-rank samplers expand each shot from a few bits), or with fewer than 2 parity
  entries per measurement (`PRISM_GPU_BTS_MIN_WEIGHT_FACTOR`).
  `sample_bulk_packed_device` returns a `DevicePackedShots` handle. Marginals reduce to
  one counter per measurement row on the device. Exact counts use a bounded device hash
  reduction for up to 8 packed measurement words when the compact result is cheaper to
  transfer than the full shot matrix, and a host copy otherwise.
- `NoisyCompiledSampler::with_gpu(ctx)`. The same BTS path with Pauli noise. Noise masks
  come from an RNG stream seeded on the device when the event probabilities allow it, or
  from the host otherwise, and counts and marginals reduce on the device. The device
  stream differs from the CPU one, so output matches the CPU statistically, not shot for
  shot. Readout error keeps materialized shots on the CPU.
- `CompiledDetectorSampler::with_gpu(ctx)`. Detector and observable sampling over the
  device measurement sampler, with parity rows derived as on the CPU.

On the hard statevector path (`StatevectorGpu`, `with_gpu`), a state that does not fit
the free VRAM is rejected at `init` with an error naming the requested and free device
memory.

`GpuContext` reports on its device before a run: `device_name`, `vram_bytes` and
`vram_available`, `max_qubits_for_statevector` (the advisory cap from free memory), and
`fits_statevector(n)` for the amplitudes alone. `fits_statevector_with_scratch(n)` and
`fits_tableau(n)` are the checks `AutoGpu` runs before placing a block. `synchronize`
waits for queued kernels, which timing device work needs. `gpu::is_available()` opens no
context. `gpu::nvrtc_available()` reports whether NVRTC loads, which matters only when no
cached kernel image fits the device. Under `bench-internal`, `pass_census` counts
launches, passes over the state and host copies since the context opened.

Python reaches the four device kinds too. The Linux and Windows wheels carry the `gpu`
feature, and `pip install "prism-q[cuda12]"` adds NVRTC from PyPI. See
[Python Bindings](python-running.md#gpu-backends).

## Module layout (`src/gpu/`)

| File | Role |
| ---- | ---- |
| `mod.rs` | `GpuContext`, `GpuState` and `GpuTableau` public entry points, crossover knobs, `GpuPassCensus` (`bench-internal`) |
| `device.rs` | `GpuDevice`: cudarc wrapper, compiles the kernel image at device construction |
| `memory.rs` | `GpuBuffer`: device `Complex64` storage |
| `tiles.rs` | Lowering of a fused stream to mixed tile ops and the tile planner |
| `kernels/mod.rs` | `KERNEL_NAMES`, `LauncherScratch`, composed `kernel_source()` concatenating dense, stabilizer, BTS and density |
| `kernels/dense.rs` | Rust launchers for every `Gate` variant; CUDA C source in `kernels/dense.cu` |
| `kernels/stabilizer.rs` | Launchers for tableau init, the batched Clifford kernel `stab_apply_word_grouped`, `stab_rowmul_words`, and measurement as pivot search, cascade, fixup and a deterministic branch; source in `kernels/stabilizer.cu` |
| `kernels/bts.rs` | Launchers for compiled BTS shot sampling; source in `kernels/bts.cu` |
| `kernels/density.rs` | Launchers for the density-matrix path: sandwiches, Kraus channels, projection and Pauli expectation; source in `kernels/density.cu` |

## Kernel coverage

Every `Gate` variant runs on the device, though not each through a kernel of its own.
The one- and two-qubit gates with a fixed matrix share one dense matrix launcher.
`QftBlock` runs as shared memory DIF passes (`qft_dif_pass`, up to 11 stages each) plus
a bit reversal pass where the flags need one, and `PauliRot` lowers to a CNOT ladder
around a parity-phase kernel. Batched variants (`BatchPhase`, `BatchRzz`,
`DiagonalBatch`, `MultiFused { all_diagonal: true }`) use LUT kernels fed by the same
host table builders as the CPU path.

Non-diagonal `MultiFused` uses a shared memory tiled kernel (`apply_multi_fused_tiled`,
`TILE_Q = 10`, `TILE_SIZE = 1024`) over ten qubits per pass: the five lowest, which keep
a warp's loads contiguous, plus up to five of the sub-gates' higher targets. A
`MultiFused` over `n` qubits takes about `(n - 5) / 5` passes, and a pass with fewer than
three sub-gates falls back to per gate launches. `Multi2q` launches once per sub-gate. A
sub-gate or `Fused2q` whose matrix is exactly a CX, swap, CZ or controlled phase runs on
that gate's kernel, and one with a single nonzero per row (a permutation with phases, or
a diagonal) on `apply_monomial_2q`. Only the rest pay the dense 4x4 multiply, which
takes three real multiply-adds per complex product (the Gauss form). On a card with
reduced FP64 rate such as the GTX 1080 Ti it costs about one sweep of the state instead
of one and a half.

From 18 qubits the statevector fuses for the device (`FusionProfile::Device`, see
[Fusion Pipeline](../architecture/fusion.md#device-profile)) and `apply_instructions`
lowers the stream to tiles. One `apply_mixed_tile` block gathers the `2^11` amplitudes of
11 qubits (the two lowest plus up to nine the planner picks) into 32 KB of shared memory,
applies every op the tile holds, and writes them back: one pass for a run of 1q
matrices, diagonals, CX, swap, CZ, controlled phases and `Rzz`. A real 1q matrix, or one
shaped like `Rx`, takes half the multiplies of a general one. Dense 4x4 blocks, `Mcu`,
`QftBlock` and the rest keep their kernels. On a GTX 1080 Ti a five-layer hardware
efficient ansatz at 24 qubits drops from 115 dense passes to 7 tile passes, and a random
depth-10 circuit from 73 to 7. The tiles are FP64 bound on that card, so a dense 1q gate
still costs about half a sweep.

The CUDA C source sits in `.cu` files beside the launchers and reaches
`KERNEL_SOURCE_TEMPLATE` through `include_str!`. Placeholders such as `{{BP_TABLE_SIZE}}`
and `{{TILE_Q}}` in `kernels/dense.cu` are filled at device construction from the
constants in `src/backend/statevector/kernels.rs`, so CPU and GPU share one value.

## Correctness

`tests/golden_gpu.rs` compares GPU amplitudes against the CPU statevector within
1e-12 for every gate variant, the fusion paths, and the `BackendKind::StatevectorGpu`
public dispatch path at the crossover boundary. `tests/golden_gpu_density_matrix.rs`
does the same for the full density matrix buffer. Both suites skip when no device
opens, and no CI job opens one. `scripts/test-gpu.ps1` runs them with
`PRISM_REQUIRE_GPU=1` so a missing or unusable device fails the run.

### Shot reproducibility

The golden equality tests do not cover two limits on what a seed guarantees.

- CPU against GPU: agreement in distribution, not bit for bit. Both paths draw the
  same RNG stream for a given shot seed, but the device reduction behind a measurement
  probability sums in tree order with FMA contraction and can differ from the host sum
  in the last ulp. A uniform draw landing between the two flips that outcome and every
  one after it. Where the probability is a dyadic rational (0.5, 1.0, and the
  amplitudes Clifford gates reach) both sums are exact and the shots match, as
  `statevector_gpu_mid_measure_shots_match_cpu` pins.
  `statevector_gpu_shot_frequencies_match_cpu_off_dyadic` pins the general case: equal
  frequencies within 5 sigma after `Rx(0.3)`.
- GPU BTS sampling: reproducible at a fixed Rayon thread count. Above
  `MIN_PAR_DRAWS` random bits per chunk, `fill_random_bits` seeds one stream per worker
  and partitions the draws by `rayon::current_num_threads()`, so another worker count
  gives other shots. Below that threshold the serial single-stream path runs and the
  seed reproduces outright. Pin `RAYON_NUM_THREADS` when byte-identical shot payloads
  matter across machines.

## Current limits

- Stabilizer `probabilities()`, `export_tableau()`, and `export_statevector()` read
  back to the CPU.
- Every trajectory shot rebuilds the backend and reallocates the device buffer,
  measured at 0.1 ms per shot at 20 qubits. `Custom` Kraus branch probabilities come
  from the qubit's reduced density matrix, computed on the device (`rdm_qubit`).
- Kernel design notes live in the module docstrings under `src/gpu/kernels/`. The
  statevector crossover measurement is on `MIN_QUBITS_DEFAULT` in `src/gpu/mod.rs`.

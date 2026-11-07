# Python Bindings

The PyO3 extension `prism_q._prism_q` is re-exported by the pure Python `prism_q`
package. Simulation runs in Rust with the GIL released, so the wrapper adds no cost per
gate.

Wheels are `abi3` for Python 3.11 and newer, one per platform: Linux x86_64 and aarch64
(manylinux), macOS arm64 and Windows x64. Other platforms build from the source
distribution, which needs a Rust toolchain.

## Install

```bash
pip install prism-q
```

NumPy is the only runtime dependency. The Linux and Windows wheels also carry the
CUDA paths (see [GPU backends](python-running.md#gpu-backends)), which need an NVIDIA
driver and NVRTC, the CUDA runtime compiler. The `cuda12` extra installs NVRTC from PyPI:

```bash
pip install "prism-q[cuda12]"
```

Neither is loaded at import: without them the package runs on the CPU and only
`GpuContext(...)` raises. The macOS wheel has no CUDA paths.

Building from a source checkout needs [maturin](https://www.maturin.rs/):

```bash
pip install maturin
maturin develop --manifest-path bindings/python/Cargo.toml
```

`parallel` is on by default; `gpu` and `distributed-mpi` are optional. `--features gpu`
adds the CUDA paths to a source build and compiles without a CUDA toolkit. No wheel
carries the sharded statevector
([Distributed backend](python-running.md#distributed-backend)), so `distributed-mpi`
always means a source build.

## Quick start

```python
from prism_q import CircuitBuilder, simulate

circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
counts = simulate(circuit).seed(42).shots(1000).counts()
print(counts)          # {'00': 507, '11': 493}
```

```admonish warning title="q[0] is the least significant qubit"
`x q[0]` produces state index 1, not 2. In a counts key, character `i` is
classical bit `i` with bit 0 leftmost.
A Bell pair gives `'00'` and `'11'`, which look the same either way, but
`CircuitBuilder(2, 2).x(0).measure_all()` gives `'10'`, not `'01'`.
```

## The rest of the guide

| Page | Covers |
|------|--------|
| [Circuits and Parameters](python-circuits.md) | `CircuitBuilder`, classical control, OpenQASM export and drawing, parameter sweeps, dynamic programs |
| [Running Simulations](python-running.md) | the terminal methods, result metadata, route inspection, memory caps, backend selection, GPU and MPI backends, batches |
| [Observables and Gradients](python-observables.md) | expectation values, exact noisy expectations, adjoint gradients |
| [Noise and Shots](python-noise.md) | noise models and rules, leakage, timed and drifting noise, compiled samplers |
| [Quantum Error Correction](python-qec.md) | `QecProgram`, memory experiments, detector error models, decoders, feedforward, packed records |
| [Advanced State Handling](python-state.md) | start states and mixtures, entropy and overlap, Clifford tableau hand-off, mid-circuit saves |
| [Errors, Typing and Names](python-names.md) | `PrismError`, type stubs, how Python names map to the Rust API |

The [tutorials](../tutorials/index.md) walk through whole tasks:
[variational circuits](../tutorials/variational.md) for sweeps and gradients,
[noisy simulation](../tutorials/noisy-simulation.md),
[a QEC memory experiment](../tutorials/qec-memory.md) and
[dynamic circuits](../tutorials/dynamic-circuits.md).
[Python API](../reference/python-api.md) lists every Rust function with a Python
counterpart.

## Pickling and multiprocessing

`Circuit`, `Gate`, `Parameters`, `NoiseChannel`, `NoiseModel`, `QecProgram`,
`PauliObservable`, `ClassicalCondition`, `QecRecordRef`, `QecNoise`, `SaveSpec`, and
`QecBasis` pickle, so they cross `multiprocessing`, `concurrent.futures`, Ray, and Dask
boundaries. The payload is a versioned binary encoding with every float as its IEEE
bits, so an unpickled value is identical to the original, and a `Parameters` set keeps
the edit guard it was pinned with.

```python
import pickle

from prism_q import CircuitBuilder, simulate

circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
restored = pickle.loads(pickle.dumps(circuit))
assert pickle.dumps(restored) == pickle.dumps(circuit)
assert simulate(restored).seed(1).shots(100).counts() == simulate(circuit).seed(1).shots(
    100
).counts()
```

A payload from a newer format version raises `PrismError` naming the version. Of the
results, only `ExpectationResult` and `ExpectationGradient` pickle. Builders, other
results, samplers, backends, and contexts do not; return arrays or counts from a worker
instead.

## Threads

Parallel kernels run on one pool for the whole process, sized to every logical core. Set
`RAYON_NUM_THREADS` before the first simulation to cap it; the pool is built once and
ignores later changes. To run some calls at another width, call through a `ThreadPool`:

```python
from prism_q import ThreadPool, simulate

pool = ThreadPool(2)
outcome = pool.install(lambda: simulate(circuit).seed(42).run())
```

`install(fn, *args, **kwargs)` calls `fn` on one of the pool's workers, blocks, and
passes its result or exception through. Every parallel kernel reached from `fn` runs on
those workers. `fn` runs on another OS thread and does not see the caller's
thread-local state. Pool width changes results only where thread count does, per the
[threading contract](../architecture/threading-simd.md).

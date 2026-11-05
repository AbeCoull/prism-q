# Python Bindings

`prism-q` ships Python bindings built with PyO3. They are a thin wrapper over the
Rust crate: the compiled extension is `prism_q._prism_q` and the pure-Python
`prism_q` package re-exports it. Simulation runs in Rust with the GIL released,
so the wrapper adds no per-gate overhead.

Wheels are `abi3` for Python 3.11 and newer, so one wheel per platform covers
every supported interpreter. PyPI carries wheels for Linux x86_64 and aarch64
(manylinux), macOS arm64 and Windows x64. Other platforms build from the source
distribution, which needs a Rust toolchain.

## Install

```bash
pip install prism-q
```

NumPy is the only runtime dependency. The Linux and Windows wheels also carry the
CUDA paths (see [GPU backends](python-running.md#gpu-backends)), which need an NVIDIA driver and
NVRTC, the CUDA runtime compiler. The `cuda12` extra installs NVRTC from PyPI:

```bash
pip install "prism-q[cuda12]"
```

Neither is loaded at import, so on a host without them the package runs on the CPU
as usual and only `GpuContext(...)` raises. The macOS wheel has no CUDA paths.

Building from a source checkout needs [maturin](https://www.maturin.rs/):

```bash
pip install maturin
maturin develop --manifest-path bindings/python/Cargo.toml
```

The bindings enable the `parallel` feature by default. `gpu` and `distributed-mpi`
are optional. `--features gpu` adds the CUDA paths to a source build and needs no
CUDA toolkit to compile. The sharded statevector (see
[Distributed backend](python-running.md#distributed-backend)) is in no wheel, so `distributed-mpi` always
means a source build.

## Quick start

```python
from prism_q import CircuitBuilder, simulate

circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
counts = simulate(circuit).seed(42).shots(1000).counts()
print(counts)          # {'00': 507, '11': 493}
```

```admonish warning title="q[0] is the least significant qubit"
`x q[0]` produces state index 1, not 2. In a counts key, character `i` is
classical bit `i` with bit 0 leftmost, so keys read reversed relative to Qiskit.
A Bell pair gives `'00'` and `'11'`, which look the same either way, but
`CircuitBuilder(2, 2).x(0).measure_all()` gives `'10'`, not `'01'`.
```

## The rest of the guide

| Page | Covers |
|------|--------|
| [Circuits and Parameters](python-circuits.md) | `CircuitBuilder`, classical control, OpenQASM export and drawing, parameter sweeps, pickling, dynamic programs |
| [Running Simulations](python-running.md) | the terminal methods, result metadata, memory caps, backend selection, GPU and MPI backends, batches and threads |
| [Observables and Gradients](python-observables.md) | expectation values, exact noisy expectations, adjoint gradients |
| [Noise and Shots](python-noise.md) | noise models and rules, leakage, timed and drifting noise, compiled samplers |
| [Quantum Error Correction](python-qec.md) | `QecProgram`, detector error models, decoders, feedforward, packed records |
| [Advanced State Handling](python-state.md) | start states and mixtures, Clifford tableau hand-off, mid-circuit saves, route inspection |
| [Rust and Python Names](python-names.md) | how Python names map to the Rust API |

## Errors and typing

Every failure surfaces as `prism_q.PrismError`, carrying the message from the
Rust error. Backend limits, unsupported operations, and invalid arguments all
raise it:

```python
import prism_q

try:
    simulate(huge).density_matrix_expectation_values([[(0, "Z")]])
except prism_q.PrismError as exc:
    print(exc)   # backend `density_matrix` is incompatible: circuit has ... qubits
```

The package ships type stubs (`prism_q/_prism_q.pyi`) and a `py.typed` marker,
so mypy and Pyright see the full surface.

`PrismError` carries a `kind` string naming the variant it came from, so a
caller can branch on the failure without matching its message. A run over a memory
cap raises kind `"resource_limit"` with the numbers attached: `resource` names the
unit (`"qubits"`, `"amplitudes"`, `"entries"`, `"elements"` or
`"bytes of device memory"`), `required` and `limit` are integers in that unit, and
`env_var` names the variable that overrides the cap, or is `None`. `knob` names the
[`Limits`](python-running.md#memory-caps) field that overrides it, and `limit_source`
says which layer set it: `"detected"`, `"env"`, `"process_default"` or `"request"`. On
every other kind those six attributes are `None`.

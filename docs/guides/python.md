# Python Bindings

`prism-q` ships Python bindings built with PyO3. They are a thin wrapper over the
Rust crate: the compiled extension is `prism_q._prism_q` and the pure-Python
`prism_q` package re-exports it. Simulation runs in Rust with the GIL released,
so the wrapper adds no per-gate overhead.

Wheels are `abi3` for Python 3.11 and newer, so one wheel per platform covers
every supported interpreter.

## Install

```bash
pip install prism-q
```

NumPy is the only runtime dependency. Building from a source checkout needs
[maturin](https://www.maturin.rs/):

```bash
pip install maturin
maturin develop --manifest-path bindings/python/Cargo.toml
```

The bindings enable the `parallel` feature by default. The `gpu` feature is
optional and off in the published wheels (see [GPU backends](#gpu-backends));
the distributed backend is not reachable from Python.

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

## Building circuits

`CircuitBuilder` is a fluent API. Every gate method returns the builder, and
`build()` produces the `Circuit` that simulation consumes.

```python
from prism_q import CircuitBuilder

circuit = (
    CircuitBuilder(3, 3)
    .h(0)
    .cx(0, 1)
    .rz(0.5, 2)
    .cphase(0.25, 1, 2)
    .measure_all()
    .build()
)
```

| Group | Methods |
|-------|---------|
| Single qubit | `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg` |
| Rotations | `rx(theta, q)`, `ry(theta, q)`, `rz(theta, q)`, `p(theta, q)` |
| Two qubit | `cx(control, target)`, `cz(q0, q1)`, `swap(q0, q1)`, `rzz(theta, q0, q1)`, `cphase(theta, control, target)` |
| Arbitrary unitary | `cu(matrix, control, target)`, `mcu(matrix, controls, target)`, `gate(gate, targets)` |
| Non-unitary | `measure(qubit, bit)`, `measure_all()`, `barrier(qubits)` |
| Gradients | `param(slot)`, `parameter_links()` |

`cu` and `mcu` take a 2x2 matrix as nested Python sequences of complex numbers.
Out-of-range qubits raise `PrismError` at build time rather than at simulation
time.

Three other routes produce a `Circuit`:

```python
from prism_q import Circuit, circuits, parse_qasm

manual = Circuit(2, 2)                    # imperative, add_gate / add_measure / add_reset
ghz = circuits.ghz(10)                    # pre-built corpus
parsed = parse_qasm(qasm_source)          # OpenQASM 3.0, with 2.0 accepted
```

The `circuits` submodule mirrors the Rust builders documented in
[Circuit Builders](../reference/builders.md): `qft`, `ghz`, `w_state`, `random`,
`hardware_efficient_ansatz`, `clifford_heavy`, `clifford_random_pairs`, `qaoa`,
`single_qubit_rotation`, `clifford_t`, `quantum_volume`, `cz_chain`,
`phase_estimation`, `independent_bell_pairs`, `independent_random_blocks`, and
`local_clifford_blocks`. Seeded builders default to seed 42.

## Running a simulation

`simulate(circuit)` returns a `Simulation` you configure with `.seed()`,
`.backend()`, and `.noise()`, then finish with a terminal method. The default
seed is 42.

```python
from prism_q import BackendKind, simulate

sim = simulate(circuit).seed(7).backend(BackendKind.statevector())
outcome = sim.run()
```

| Terminal | Returns | Honors `.noise()` | Honors `.initial_state()` |
|----------|---------|-------------------|---------------------------|
| `run()` | `RunOutcome`: classical bits and the full probability array | density matrix only | yes |
| `shots(n)` | `ShotsResult`: per-shot measurement records | yes | without `.noise()` |
| `sample_counts(n)` | `CountsResult`: frequency histogram | yes | without `.noise()` |
| `marginals()` | `list[tuple[float, float]]`, per-qubit `(p0, p1)` | density matrix only | yes |
| `state_vector()` | `complex128` amplitudes | no | yes |
| `expectation_values(obs)` | `list[float]`, `⟨ψ\|P\|ψ⟩` per observable | density matrix only | yes |
| `density_matrix_expectation_values(obs)` | `list[float]`, exact `Tr(rho P)` | yes | no |
| `expectation_gradient(h, params)` | `(value, gradient)` via the adjoint method | no | no |

`shots()` and `sample_counts()` average trajectories on any backend holding a
per-shot pure state. The three rows marked "density matrix only" read the exact
mixed state instead, so they need
`.backend(BackendKind.density_matrix())`; auto dispatch never selects it. There
the mixture is evolved once and every terminal reads that one evolution, so the
probabilities are seed independent and the observables carry no sampling error.
Circuits with mid-circuit measurement or classical conditioning are rejected on
that route, since the mixture holds every measurement branch at once.

Terminals that cannot honor a model raise `PrismError` naming the reason, rather
than silently ignoring it. `state_vector()` always uses the statevector backend
and `density_matrix_expectation_values()` always uses the density-matrix
backend, both regardless of `.backend(...)`.

`ShotsResult` and `CountsResult` both expose `counts()`, returning a dict keyed
by bitstring.

## Result metadata

`RunOutcome`, `ShotsResult`, and `CountsResult` each carry a `metadata` object
describing how the result was produced.

```python
result = simulate(circuit).seed(42).run()
print(result.metadata.backend)               # 'Statevector'
print(result.metadata.is_exact)              # True
print(result.metadata.fidelity_lower_bound)  # None when exact
print(result.metadata.placement)             # 'host' or 'device'
```

`is_exact` is False when the engine that ran can discard state weight or
estimate by sampling. It marks the route, not the run: an MPS whose bond
dimension the circuit never fills reports `is_exact == False` with
`fidelity_lower_bound == 1.0`, so the flag answers whether the answer could have
been approximated and the bound answers whether it was.

Automatic dispatch sends a circuit past the statevector cap to a bounded-bond
MPS, which is the only route those circuits have. That is taken by default and
the result says so. `.require_exact()` rejects it instead, raising `PrismError`
naming the engine it would have used.

```python
simulate(big_circuit).seed(42).require_exact().marginals()  # raises
```

## Starting from a state other than |0...0>

`.initial_state(amplitudes)` replaces the default all-zero start. It takes any
sequence of complex numbers, a `complex128` NumPy array included, indexed with
qubit 0 in the least significant bit.

```python
import math
import numpy as np
from prism_q import CircuitBuilder, simulate

theta = math.pi / 8
start = np.array([math.cos(theta), math.sin(theta)], dtype=np.complex128)
circuit = CircuitBuilder(1).h(0).build()
probs = simulate(circuit).initial_state(start).seed(42).run().probabilities
```

The vector must have `2 ** num_qubits` entries and unit norm. A wrong length, a
non-finite entry, or a norm off unity raises `PrismError`; an unnormalized
vector is rejected rather than rescaled, so a mistake surfaces instead of
becoming a silent factor on every amplitude.

A start state also narrows the route. Auto dispatch reads circuit structure, and
its shortcuts (tableau, product state, subsystem decomposition, Pauli
propagation) are only valid from |0...0>: a Clifford circuit produces a
stabilizer state only when its input is one. So `auto()` resolves to the
statevector, `density_matrix()` is the only other backend that accepts one, and
every other choice raises `PrismError` naming itself. `run()`, `shots()`,
`sample_counts()`, `marginals()`, `expectation_values()`, and `state_vector()`
carry it; `expectation_gradient()` and `density_matrix_expectation_values()`
reject it, as do `shots()` and `sample_counts()` with a noise model attached,
since trajectory replay reinitializes a pure state per shot. To evolve a start
state under noise, read the exact mixture with `run()`, `marginals()`, or
`expectation_values()` on `density_matrix()`.

## Selecting a backend

`BackendKind.auto()` is the default and picks a backend from circuit structure.
Pass an explicit one to override it.

| Constructor | Notes |
|-------------|-------|
| `auto()` | Structure-driven dispatch. See [Choosing a Backend](../getting-started/choosing-a-backend.md). |
| `statevector()` | Dense amplitudes, the general-purpose path |
| `stabilizer()`, `factored_stabilizer()` | Clifford-only circuits |
| `stabilizer_rank()` | Clifford+T, requires at least one T gate |
| `sparse()` | Sparse states, for circuits that stay concentrated |
| `product_state()` | No entangling gates |
| `factored()` | Partially independent subsystems |
| `tensor_network()` | Contraction over a network |
| `mps(max_bond_dim=256)` | Approximate, truncates at the bond dimension |
| `density_matrix()` | Exact mixed states, never chosen by `auto()` |
| `stochastic_pauli(num_samples=1000)` | Sampled Pauli propagation |
| `deterministic_pauli(epsilon=0.0, max_terms=65536)` | Truncated Pauli propagation |
| `auto_gpu(context)`, `statevector_gpu(context)`, `stabilizer_gpu(context)` | CUDA device paths, see [GPU backends](#gpu-backends) |

The density-matrix backend stores `4^n` amplitudes, so its qubit ceiling is
about half the statevector cap; exceeding it raises `PrismError` naming the cap.
The distributed statevector backend has no Python constructor: `MPI_Init`
ownership between the interpreter, mpi4py, and the extension is unsettled.

## GPU backends

The GPU constructors take a `GpuContext`, an opaque handle to one CUDA device
and its compiled kernels. Build it once and reuse it: construction compiles the
kernel module, and passing the same handle to several simulations shares that
work.

```python
from prism_q import BackendKind, GpuContext, circuits, simulate

context = GpuContext(0)
outcome = simulate(circuits.qft(16)).backend(BackendKind.auto_gpu(context)).seed(42).run()
print(outcome.probabilities)
```

`GpuContext(device_id)` is where a missing or unusable device is reported, and
it raises `PrismError` rather than falling back. Past construction, routing is
soft by design and matches the Rust API: `statevector_gpu` runs circuits below
the crossover (`PRISM_GPU_MIN_QUBITS`, default 14) on the host, `auto_gpu`
routes each block independently, and a block whose device allocation fails
degrades to the host rather than erroring. A run that produces host results is
therefore normal, not a failure signal.

`stabilizer_gpu` sets its crossover at 100000 qubits
(`PRISM_STABILIZER_GPU_MIN_QUBITS`), so it runs on the host tableau unless that
override is lowered. The device tableau is correct; the default stays high until
benchmarks justify lowering it.

The published wheels are built without CUDA, because two of the three wheel
targets have no CUDA toolkit and macOS has no CUDA at all. In those wheels the
constructors still exist and `GpuContext(...)` raises `PrismError` naming the
missing build feature, so code written against the GPU API fails with a message
rather than an `AttributeError`. Two predicates separate the cases:

```python
GpuContext.is_supported()   # was this build compiled with CUDA support
GpuContext.is_available()   # ... and is a usable device present
```

To get a build with CUDA support, install the CUDA toolkit (12.x or newer) and
build from a checkout:

```bash
maturin develop --manifest-path bindings/python/Cargo.toml --features gpu
```

On Windows that build links the toolkit's NVRTC library (`nvrtc64_120_0.dll`
for CUDA 12.x) from the toolkit `bin` directory, which Python does not search.
The package adds it on import when `CUDA_PATH` is set, which the toolkit
installer does; without it the import fails with `DLL load failed while
importing _prism_q`.

## Noise

Build a `NoiseModel` from a circuit, then attach it. A model is sized to the
circuit it was built from, and using it with a different circuit raises
`PrismError`.

```python
from prism_q import NoiseChannel, NoiseModel, simulate

model = NoiseModel.uniform_depolarizing(circuit, 0.01)
counts = simulate(circuit).seed(42).noise(model).sample_counts(4000).counts()
```

`NoiseModel.uniform_depolarizing(circuit, p)` and
`NoiseModel.with_amplitude_damping(circuit, gamma)` cover the common cases.
For per-instruction control, start from `NoiseModel.empty(circuit)` and attach
events:

```python
model = NoiseModel.empty(circuit)
model.add_event(0, NoiseChannel.amplitude_damping(0.05), [0])
model.add_event(1, NoiseChannel.two_qubit_depolarizing(0.02), [0, 1])
model.with_readout_error(0.01, 0.01)
model.validate()
```

Channels are `pauli(px, py, pz)`, `depolarizing(p)`, `amplitude_damping(gamma)`,
`phase_damping(gamma)`, `thermal_relaxation(t1, t2, gate_time)`,
`two_qubit_depolarizing(p)`, and `custom(kraus)` for an explicit list of 2x2
Kraus operators. `validate()` checks probabilities and Kraus completeness;
`is_pauli_only()` reports whether the model can run on a stabilizer backend.

## Expectation values

An observable is a list of `(qubit, axis)` factors with `axis` one of `"X"`,
`"Y"`, `"Z"`. Identity factors are omitted, so `[(0, "Z"), (2, "X")]` means
`Z0 ⊗ I1 ⊗ X2`. Both expectation terminals take a list of observables and return
one float each.

```python
observables = [[(0, "Z")], [(0, "Z"), (1, "Z")]]
values = simulate(circuit).seed(42).expectation_values(observables)
```

`expectation_values` requires a unitary circuit and gives `⟨ψ|P|ψ⟩`.
`density_matrix_expectation_values` evolves the density matrix through the
circuit and any attached noise model and gives exact `Tr(rho P)`, with
measurements read off the final mixed state without collapse. It is the
zero-variance analogue of averaging over trajectories:

```python
model = NoiseModel.empty(circuit)
model.add_event(0, NoiseChannel.amplitude_damping(0.3), [0])
exact = simulate(circuit).seed(42).noise(model).density_matrix_expectation_values(
    [[(0, "Z")]]
)
```

```admonish note title="Compare with a tolerance"
Analytic means are not bit-stable across separate invocations. Hash-ordered term
accumulation can move the last ulp, so compare against `1e-12` rather than
asserting exact equality.
```

## Gradients

`expectation_gradient` computes `⟨H⟩` and its exact gradient with respect to the
bound parameters by the adjoint method, at a cost independent of the
parameter count. Mark parameters with `param(slot)` while building, then
pass `parameter_links()` through.

```python
import numpy as np
from prism_q import CircuitBuilder, simulate

builder = CircuitBuilder(2)
builder.ry(0.3, 0).param(0)
builder.cx(0, 1)
builder.rz(0.7, 1).param(1)
circuit, links = builder.build(), builder.parameter_links()

hamiltonian = [(1.0, [(0, "Z")]), (0.5, [(0, "Z"), (1, "Z")])]
value, gradient = simulate(circuit).seed(42).expectation_gradient(hamiltonian, links)
```

A Hamiltonian term is `(coefficient, observable)`. Several gates may share a
slot, in which case their gradients accumulate. `param()` rejects anything
but a differentiable gate (`rx`, `ry`, `rz`, `rzz`, `p`), and the circuit must be
unitary.

## Quantum error correction

`QecProgram` exposes the native QEC IR: `reset`, `measure`, `detector`,
`observable_include`, `postselect`, and `noise`, with `QecBasis`, `QecNoise`,
and `RecordRef` as the supporting types. `run()` returns a `QecResult` carrying
detector, observable, and measurement arrays as NumPy `bool_` matrices, plus
`logical_error_rates()` and `survivor_rate()`. Programs can also be parsed from
text with `QecProgram.from_text`. See the [Noise and QEC guide](./qec.md) for the
model itself.

## Errors and typing

Every failure surfaces as `prism_q.PrismError`, carrying the message from the
Rust error. Backend limits, unsupported operations, and invalid arguments all
raise it rather than panicking:

```python
import prism_q

try:
    simulate(huge).density_matrix_expectation_values([[(0, "Z")]])
except prism_q.PrismError as exc:
    print(exc)   # backend `density_matrix` is incompatible: circuit has ... qubits
```

The package ships type stubs (`prism_q/_prism_q.pyi`) and a `py.typed` marker,
so mypy and Pyright see the full surface.

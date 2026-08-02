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

The bindings enable the `parallel` feature and nothing else. The `gpu` and
`distributed` backends are not reachable from Python.

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
| Gradients | `trainable(slot)`, `parameter_links()` |

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

| Terminal | Returns | Honors `.noise()` |
|----------|---------|-------------------|
| `run()` | `RunOutcome`: classical bits and the full probability array | density matrix only |
| `shots(n)` | `ShotsResult`: per-shot measurement records | yes |
| `sample_counts(n)` | `CountsResult`: frequency histogram | yes |
| `marginals()` | `list[tuple[float, float]]`, per-qubit `(p0, p1)` | density matrix only |
| `state_vector()` | `complex128` amplitudes | no |
| `expectation_values(obs)` | `list[float]`, `⟨ψ\|P\|ψ⟩` per observable | density matrix only |
| `density_matrix_expectation_values(obs)` | `list[float]`, exact `Tr(rho P)` | yes |
| `expectation_gradient(h, params)` | `(value, gradient)` via the adjoint method | no |

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

The GPU and distributed backends have no Python constructor. The
density-matrix backend stores `4^n` amplitudes, so its qubit ceiling is about
half the statevector cap; exceeding it raises `PrismError` naming the cap.

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
trainable parameters by the adjoint method, at a cost independent of the
parameter count. Mark parameters with `trainable(slot)` while building, then
pass `parameter_links()` through.

```python
import numpy as np
from prism_q import CircuitBuilder, simulate

builder = CircuitBuilder(2)
builder.ry(0.3, 0).trainable(0)
builder.cx(0, 1)
builder.rz(0.7, 1).trainable(1)
circuit, links = builder.build(), builder.parameter_links()

hamiltonian = [(1.0, [(0, "Z")]), (0.5, [(0, "Z"), (1, "Z")])]
value, gradient = simulate(circuit).seed(42).expectation_gradient(hamiltonian, links)
```

A Hamiltonian term is `(coefficient, observable)`. Several gates may share a
slot, in which case their gradients accumulate. `trainable()` rejects anything
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

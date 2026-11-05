# Advanced State Handling

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
statevector, the GPU and distributed statevectors and `density_matrix()` are the
only other backends that accept one, and every other choice raises `PrismError`
naming itself. Every terminal whose row in the
[terminal table](python-running.md#running-a-simulation) says so carries it;
`expectation_gradient()` and `density_matrix_expectation_values()` reject it, as
do `shots()` and `sample_counts()` with a noise model attached, since trajectory
replay reinitializes a pure state per shot. To evolve a start
state under noise, read the exact mixture with `run()`, `marginals()`, or
`expectation_values()` on `density_matrix()`.

## Starting from a mixture

`.initial_density_matrix(rho)` starts the density-matrix backend from a mixed
state. It takes a square `complex128` NumPy array, or a sequence of rows, in
the layout `reduced_density_matrix()` over the whole register returns:
row-major `2 ** n` by `2 ** n` with qubit 0 the least significant bit of both
indices. A noisy run can be paused, its mixture stored, and resumed later:

```python
from prism_q import BackendKind, NoiseModel, simulate

dm = BackendKind.density_matrix()
rho = (
    simulate(first)
    .backend(dm)
    .noise(NoiseModel.uniform_depolarizing(first, 0.02))
    .reduced_density_matrix(range(first.num_qubits))
    .matrix
)
probs = (
    simulate(second)
    .backend(dm)
    .noise(NoiseModel.uniform_depolarizing(second, 0.02))
    .initial_density_matrix(rho)
    .run()
    .probabilities
)
```

The matrix must have `4 ** num_qubits` entries, be Hermitian to 1e-12 per
entry, and have unit trace to 1e-9; one failing a check raises `PrismError`
naming it. Positive semidefiniteness is not checked. Only `density_matrix()`
and its GPU sibling accept a mixture, so every other backend, `auto()`
included, raises `PrismError` naming itself. The
[terminal table](python-running.md#running-a-simulation) applies unchanged, and setting
a mixture clears an earlier `.initial_state()`, as that call clears a mixture.

## Handing a Clifford state to a second run

`StabilizerBackend` is a tableau held across calls. `run(circuit)` resets it to
the circuit's width and runs the circuit; `export_tableau()` returns the rows as
a `uint64` array of bit-packed words and a `bool` array of row signs;
`import_tableau(num_qubits, words, phases, num_classical_bits)` starts a backend
from that pair; and `apply(circuit)` runs a circuit on the held state without
resetting it. The pair is the raw tableau, so it is the checkpoint format for a
Clifford prefix that is expensive to replay.

```python
from prism_q import StabilizerBackend

prep = StabilizerBackend(seed=42)
prep.run(clifford_prefix)
words, phases = prep.export_tableau()

resumed = StabilizerBackend(seed=42)
resumed.import_tableau(clifford_prefix.num_qubits, words, phases, num_classical_bits=8)
bits = resumed.apply(measurement_suffix)
```

The import checks the lengths and that each destabilizer anticommutes with its
stabilizer partner, nothing more; the intended input is an export. The random
stream restarts from the importing backend's seed, so a resumed run and an
uninterrupted one draw the same outcomes only when neither drew before the
split.

## Mid-circuit saves

A save point records the state where it sits, so a circuit can be inspected part
way through without being cut in two and run twice.

```python
from prism_q import CircuitBuilder, Gate, SaveSpec, simulate

builder = CircuitBuilder(4)
for q in range(4):
    builder.h(q)
circuit = builder.build()
circuit.add_save(SaveSpec.StateVector, "after_hadamards")
circuit.add_gate(Gate.cx(), [0, 1])

outcome = simulate(circuit).seed(42).run()
for record in outcome.saves:
    print(record["label"], record["kind"], record["value"].shape)
```

Each record is a dictionary with `label`, `kind`, and `value`. `SaveSpec.StateVector`
and `SaveSpec.DensityMatrix` come back as `complex128` arrays, the density matrix flat
and row major over `2**n` rows; `SaveSpec.Probabilities` comes back as `float64`.
Records arrive in the order their points were reached, and labels need not be unique.

A save is a barrier across the whole register, so no gate is fused or reordered across
it. Only `run` returns the records: `shots`, `marginals` and the rest decline a circuit
carrying a save point rather than running it and dropping what it recorded. The same
goes for routes that hold no state to read, and for OpenQASM export, which has no save
syntax to write.

## Inspect a terminal before execution

`Simulation.plan(Query)` keeps the simulation request available for execution and
reports the route selected for that terminal. A query can change the route: an
expectation may use an exact scalar tensor contraction where a full-state run would
choose a bounded MPS.

```python
from prism_q import CircuitBuilder, Query, simulate

circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1).build()
request = simulate(circuit).seed(42)
plan = request.plan(Query.counts(1000))
assert plan.would_fail is None
result = request.sample_counts(1000)
assert result.metadata.reason == plan.rule
```

`Plan.backend`, `engine`, `placement`, and `rule` describe the selected route.
`RunMetadata.reason` gives its snake_case decision name after execution. A bounded
SPD attempt or soft GPU initialization can take a runtime fallback, declared in
`Plan.fallbacks`; the result names that fallback when it happens. Each fallback
can carry its own `would_fail`, which does not imply failure of the primary route.

`Plan.exactness` is `"exact"`, `"approximate"`, or `"estimated"`.
`bound_reported` says whether an approximate route reports a bound, and `samples`
gives an estimator's sample count. A possible fallback is not a guaranteed failure:
`require_exact()` rejects an approximate fallback only if execution needs it.

`would_fail` is a `PrismError` instance for a predictable resource or capability
failure. Malformed query inputs raise `PrismError` directly. `alternatives` lists
explicit backend selections rejected by the same validation used during execution.
It does not rank accepted backends by performance.

`peak_memory` contains `bytes` and `kind`. Known bounds cover the state and terminal
buffer payload, excluding circuit/fusion storage, allocator overhead and process
memory. Other routes return `{"bytes": None, "kind": "unknown"}`; state size alone
is not reported as a peak bound. Route inspection may perform bounded sparse-support
or tensor-contraction planning probes, but does not execute the requested terminal.

Query constructors are `run()`, `shots(n)`, `counts(n)`, `marginals()`,
`expectation(observables)`, `observable(hamiltonian)`, `state_vector()`,
`reduced_density_matrix(qubits)`, and `entropy(qubits)`. The gradient constructors,
`gradient(hamiltonian, parameters)` and `gradient_shift(hamiltonian, parameters)`,
take the same Hamiltonian and `(instruction, slot)` links as the execution terminals.
A shift plan's `evaluations` contains its baseline, followed by plus/minus evaluations
in link order. Its top-level backend names the baseline; child reports preserve any
different route chosen for another evaluation, including prepared backend reuse.

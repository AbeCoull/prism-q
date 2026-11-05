# Advanced State Handling

## Starting from a state other than |0...0>

`.initial_state(amplitudes)` replaces the all-zero start. It takes any sequence of
complex numbers, a `complex128` NumPy array included, with qubit 0 the least
significant bit of the index.

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
non-finite entry, or a norm off unity raises `PrismError`; an unnormalized vector is
rejected, not rescaled.

A start state also narrows the route. The dispatch shortcuts (tableau, product state,
subsystem decomposition, Pauli propagation) hold only from |0...0>: a Clifford circuit
yields a stabilizer state only from one. So `auto()` resolves to the statevector, the
GPU and distributed statevectors and `density_matrix()` are the only other backends
that accept a start state, and any other choice raises `PrismError` naming itself.
Terminals carry it where the [terminal table](python-running.md#running-a-simulation)
says so; `expectation_gradient()` rejects it, as do `shots()` and `sample_counts()` with
a noise model, since trajectory replay reinitializes a pure state per shot. Under noise,
read the exact mixture with `run()`, `marginals()`, or `expectation()` on
`density_matrix()`.

## Starting from a mixture

`.initial_density_matrix(rho)` starts the density-matrix backend from a mixed state: a
square `complex128` NumPy array or a sequence of rows, in the layout
`reduced_density_matrix()` returns over the whole register (row major, `2 ** n` by
`2 ** n`, qubit 0 the least significant bit of both indices). A noisy run can be
paused, its mixture stored, and resumed later:

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

The matrix must have `4 ** num_qubits` entries, be Hermitian to 1e-12 per entry, and
have unit trace to 1e-9; a failed check raises `PrismError` naming it. Positive
semidefiniteness is not checked. Only `density_matrix()` and its GPU sibling accept a
mixture; every other backend, `auto()` included, raises `PrismError` naming itself. The
[terminal table](python-running.md#running-a-simulation) applies unchanged. A mixture
clears an earlier `.initial_state()`, and the reverse.

## Handing a Clifford state to a second run

`StabilizerBackend` holds a tableau across calls. `run(circuit)` resets it to the
circuit's width and runs the circuit; `export_tableau()` returns the rows as a `uint64`
array of bit-packed words and a `bool` array of row signs;
`import_tableau(num_qubits, words, phases, num_classical_bits)` starts a backend from
that pair; `apply(circuit)` runs a circuit on the held state without resetting it. The
raw tableau serves as a checkpoint for a Clifford prefix that is expensive to replay.

```python
from prism_q import StabilizerBackend

prep = StabilizerBackend(seed=42)
prep.run(clifford_prefix)
words, phases = prep.export_tableau()

resumed = StabilizerBackend(seed=42)
resumed.import_tableau(clifford_prefix.num_qubits, words, phases, num_classical_bits=8)
bits = resumed.apply(measurement_suffix)
```

The import checks only the lengths and that each destabilizer anticommutes with its
stabilizer partner; the intended input is an export. The random stream restarts from
the importing backend's seed, so a resumed run matches an uninterrupted one only when
neither drew before the split.

## Mid-circuit saves

A save point records the state where it sits, so a circuit can be inspected part way
through without splitting it and running twice.

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
Records arrive in the order their points were reached; labels need not be unique.

A save is a barrier across the whole register, so no gate is fused or reordered across
it. Only `run` returns the records: `shots`, `marginals` and the other terminals decline
a circuit with a save point instead of dropping what it recorded. So do routes that
hold no state to read, and OpenQASM export, which has no save syntax.

## Inspect a terminal before execution

`Simulation.plan(Query)` reports the route a terminal would take and leaves the request
ready to run. The query can change the route: an expectation may use an exact scalar
tensor contraction where a full-state run would pick a bounded MPS.

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
`RunMetadata.reason` gives its snake_case decision name after execution. A bounded SPD
attempt or soft GPU initialization can take a runtime fallback, listed in
`Plan.fallbacks`; the result names the fallback when it happens. A fallback's own
`would_fail` says nothing about the primary route.

`Plan.exactness` is `"exact"`, `"approximate"`, or `"estimated"`. `bound_reported` says
whether an approximate route reports a bound, and `samples` gives an estimator's sample
count. `require_exact()` rejects an approximate fallback only if execution needs it.

`would_fail` is a `PrismError` instance for a predictable resource or capability
failure; malformed query inputs raise `PrismError` directly. `alternatives` lists
explicit backend selections rejected by the validation execution uses, without ranking
accepted backends by speed.

`peak_memory` holds `bytes` and `kind`. Known bounds cover the state and terminal buffer
payload, not circuit or fusion storage, allocator overhead, or process memory. Other
routes return `{"bytes": None, "kind": "unknown"}`; state size alone is not reported as
a peak bound. Inspection may run bounded sparse-support or tensor-contraction planning
probes but does not execute the terminal.

Query constructors are `run()`, `shots(n)`, `counts(n)`, `marginals()`,
`expectation(observables)`, `observable(hamiltonian)`, `state_vector()`,
`reduced_density_matrix(qubits)`, and `entropy(qubits)`. The gradient constructors,
`gradient(hamiltonian, parameters)` and `gradient_shift(hamiltonian, parameters)`,
take the same Hamiltonian and `(instruction, slot)` links as the execution terminals.
A shift plan's `evaluations` lists its baseline, then the plus and minus evaluations in
link order. The top-level backend names the baseline; child reports keep any different
route chosen for another evaluation, including prepared backend reuse.

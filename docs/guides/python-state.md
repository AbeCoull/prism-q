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
non-finite entry, or a norm off unity raises `PrismError`. An unnormalized vector is not
rescaled.

A start state also narrows the route. The dispatch shortcuts (tableau, product state,
subsystem decomposition, Pauli propagation) hold only from |0...0>, since a Clifford
circuit yields a stabilizer state only from one. So `auto()` resolves to the
statevector. The GPU and distributed statevectors and `density_matrix()` are the only
other backends that accept a start state, and any other choice raises `PrismError`
naming itself.

Terminals carry it where the [terminal table](python-running.md#running-a-simulation)
says so. `expectation_gradient()` rejects it, as do `shots()` and `sample_counts()` with
a noise model, since trajectory replay reinitializes a pure state per shot. Under noise,
read the exact mixture with `run()`, `marginals()`, or `expectation()` on
`density_matrix()`.

## Starting from a mixture

`.initial_density_matrix(rho)` starts the density matrix backend from a mixed state: a
square `complex128` NumPy array or a sequence of rows, in the layout
`reduced_density_matrix()` returns over the whole register (row major, `2 ** n` by
`2 ** n`, qubit 0 the least significant bit of both indices). A noisy run can be
paused, its mixture stored, and resumed later:

```python
from prism_q import BackendKind, CircuitBuilder, NoiseModel, simulate

first = CircuitBuilder(2).h(0).cx(0, 1).build()
second = CircuitBuilder(2).rx(0.3, 0).cz(0, 1).build()
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
have unit trace to 1e-9, and a failed check raises `PrismError` naming it. Positive
semidefiniteness is not checked. Only `density_matrix()` and its GPU sibling accept a
mixture. Every other backend, `auto()` included, raises `PrismError` naming itself. The
[terminal table](python-running.md#running-a-simulation) applies unchanged. A mixture
clears an earlier `.initial_state()`, and the reverse.

## Entropy, overlap and reduced states

```python
bell = CircuitBuilder(2).h(0).cx(0, 1).build()
cut = simulate(bell).entanglement_entropy([0])
print(cut.entropy, cut.schmidt_values)          # 0.6931... [0.7071 0.7071]
print(simulate(bell).reduced_density_matrix([0]).purity)   # 0.5
plus = CircuitBuilder(2).h(0).h(1).build()
print(simulate(bell).overlap(simulate(plus)).fidelity)     # 0.5
```

`entanglement_entropy(subsystem)` returns the von Neumann entropy of the cut in nats and
`schmidt_values`, the Schmidt coefficients in descending order with squares summing to 1.
`schmidt_values` is `None` where the backend holds the entropy without the spectrum, as
a stabilizer cut past the export cap does. The cut must leave qubits on both sides.
`reduced_density_matrix(qubits)` returns `.matrix` with `qubits[0]` the lowest bit, its
`.purity`, and `.qubits`. `overlap(other)` compares two pure states and reports only
`fidelity`, `|<a|b>|^2`, since a tableau keeps no global phase and truncation moves one.
`.left` and `.right` carry each side's route.

## Handing a Clifford state to a second run

`StabilizerBackend` holds a tableau across calls. `run(circuit)` resets it to the
circuit's width and runs the circuit, and `apply(circuit)` runs a circuit on the held
state without resetting it. `export_tableau()` returns the rows as a `uint64` array of
bit-packed words and a `bool` array of row signs, and
`import_tableau(num_qubits, words, phases, num_classical_bits)` starts a backend from
that pair. The pair serves as a checkpoint for a Clifford prefix that is expensive to
replay.

```python
from prism_q import StabilizerBackend

clifford_prefix = CircuitBuilder(4).h(0).cx(0, 1).cx(1, 2).s(3).build()
measurement_suffix = CircuitBuilder(4, 8).h(3).measure_all().build()
prep = StabilizerBackend(seed=42)
prep.run(clifford_prefix)
words, phases = prep.export_tableau()

resumed = StabilizerBackend(seed=42)
resumed.import_tableau(clifford_prefix.num_qubits, words, phases, num_classical_bits=8)
bits = resumed.apply(measurement_suffix)
```

The import checks only the lengths and that each destabilizer anticommutes with its
stabilizer partner, so feed it an export. The random stream restarts from
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
and row major over `2**n` rows. `SaveSpec.Probabilities` comes back as `float64`.
Records arrive in the order their points were reached, and labels need not be unique.

A save is a barrier across the whole register, so no gate is fused or reordered across
it. Only `run` returns the records: `shots`, `marginals` and the other terminals decline
a circuit with a save point instead of dropping what it recorded. So do routes that
hold no state to read, and OpenQASM export, which has no save syntax.

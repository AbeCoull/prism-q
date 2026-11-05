# Observables and Gradients

## Expectation values

An observable is a Pauli string or a list of `(qubit, axis)` factors. In a string,
character `i` acts on qubit `i`, so `"ZIX"` is `Z0 ⊗ I1 ⊗ X2`: the leftmost character is
qubit 0, the order count keys use. A short string leaves the remaining qubits as
identity, and a string longer than the register raises `PrismError`. The factor form
names only non-identity qubits, so `[(0, "Z"), (2, "X")]` is the same observable and the
shorter spelling on a wide register. Both forms mix in one list, and Hamiltonian terms
take either as `(coefficient, observable)`.

`expectation()` is the one expectation terminal. A list of observables returns an
`ExpectationResult`: `.values` holds one `⟨ψ|P|ψ⟩` per observable as a `float64` array,
`.std_errors` the standard errors of a sampling route (else `None`), and `.metadata` the
route that served them. The result also indexes, slices to a `float64` array, has a
length, converts with `np.asarray` as its `.values`, and pickles with its metadata. A `PauliObservable` returns an `ObservableExpectation` for
the weighted sum. Weighted terms must be wrapped in `PauliObservable([...])`; a bare
`(coefficient, observable)` list raises `TypeError`, as does a single factor list such as
`[(0, "Z")]`, which is written `[[(0, "Z")]]`. The circuit must be unitary, except on the
density matrix below.

```python
observables = ["Z", "ZZ", [(0, "X"), (1, "X")]]
values = simulate(circuit).seed(42).expectation(observables).values
energy = simulate(circuit).seed(42).expectation(PauliObservable([(1.0, "ZZ"), (0.5, "XI")])).mean
```

On `BackendKind.density_matrix()` the circuit and any attached noise model evolve the
density matrix, and the values are exact `Tr(rho P)`, with none of the variance of
averaging trajectories. A reset evolves as a channel there, while a circuit with a
measurement raises `PrismError`, so strip measurements first:

```python
model = NoiseModel.empty(circuit)
model.add_event(0, NoiseChannel.amplitude_damping(0.3), [0])
dm = BackendKind.density_matrix()
exact = simulate(circuit).seed(42).noise(model).backend(dm).expectation([[(0, "Z")]]).values
```

```admonish note title="Compare with a tolerance"
Analytic means are not bit-stable across separate invocations. Hash-ordered term
accumulation can move the last ulp, so compare against `1e-12` rather than
asserting exact equality.
```

## Gradients

`expectation_gradient` computes `⟨H⟩` and its exact gradient over the bound parameters
by the adjoint method, at a cost independent of the parameter count. Mark parameters
with `param(slot)` while building; `build_parametric()` returns the circuit with its
`Parameters`, whose `links()` the gradient takes.

```python
import numpy as np
from prism_q import CircuitBuilder, simulate

builder = CircuitBuilder(2)
builder.ry(0.3, 0).param(0)
builder.cx(0, 1)
builder.rz(0.7, 1).param(1)
circuit, params = builder.build_parametric()
links = params.links()

hamiltonian = [(1.0, [(0, "Z")]), (0.5, [(0, "Z"), (1, "Z")])]
result = simulate(circuit).seed(42).expectation_gradient(hamiltonian, links)
energy, gradient = result.value, result.gradient
```

The result is an `ExpectationGradient` holding `value` and a `float64` `gradient` with
one entry per slot. It unpacks, indexes and slices as the pair `(value, gradient)`, and
pickles. A Hamiltonian term is `(coefficient, observable)`. Gates sharing a slot accumulate
their gradients. `param()` accepts only differentiable gates (`rx`, `ry`, `rz`, `rzz`, `p`,
`pauli_rotation`), and the circuit must be unitary.

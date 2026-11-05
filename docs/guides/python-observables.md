# Observables and Gradients

## Expectation values

An observable is a list of `(qubit, axis)` factors, `axis` one of `"X"`, `"Y"`, `"Z"`,
with identity factors omitted: `[(0, "Z"), (2, "X")]` is `Z0 ⊗ I1 ⊗ X2`. Both
expectation terminals take a list of observables and return one float each.

```python
observables = [[(0, "Z")], [(0, "Z"), (1, "Z")]]
values = simulate(circuit).seed(42).expectation_values(observables)
```

`expectation_values` requires a unitary circuit and gives `⟨ψ|P|ψ⟩`.
`density_matrix_expectation_values` evolves the density matrix through the circuit and
any attached noise model and gives exact `Tr(rho P)`, with none of the variance of
averaging trajectories. A circuit with a measurement raises `PrismError`, so strip
measurements first:

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

`expectation_gradient` computes `⟨H⟩` and its exact gradient over the bound parameters
by the adjoint method, at a cost independent of the parameter count. Mark parameters
with `param(slot)` while building and pass `parameter_links()`.

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

A Hamiltonian term is `(coefficient, observable)`. Gates sharing a slot accumulate their
gradients. `param()` accepts only differentiable gates (`rx`, `ry`, `rz`, `rzz`, `p`,
`pauli_rotation`), and the circuit must be unitary.

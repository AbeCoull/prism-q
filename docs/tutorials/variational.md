# Variational Circuits and Gradients

Goal: minimize the energy of a four-site transverse-field Ising chain with a two-layer
ansatz, using exact gradients, then evaluate a batch of parameter points in one call.

## Mark the parameters

`param(slot)` tags the gate just appended as trainable and assigns it a slot. Several
gates may share a slot.

```python
import numpy as np
from prism_q import CircuitBuilder

n = 4
builder = CircuitBuilder(n)
for q in range(n):
    builder.ry(0.1, q).param(q)
for q in range(n - 1):
    builder.cx(q, q + 1)
for q in range(n):
    builder.ry(0.1, q).param(n + q)

circuit = builder.build()
params = builder.parameters()        # Parameters: 8 slots
links = builder.parameter_links()    # [(instruction, slot), ...]
print(params.num_slots, links[:2])   # 8 [(0, 0), (1, 1)]
```

`params` binds new angles into the circuit; `links` tells the gradient which instruction
feeds which slot.

## The observable

A Hamiltonian is a list of `(coefficient, [(qubit, axis), ...])` terms. Here
`H = -sum Z_i Z_{i+1} - 0.5 sum X_i`. Wrapping it in `PauliObservable` parses it once and
caches the grouping of commuting terms, which pays off inside a loop.

```python
from prism_q import PauliObservable

hamiltonian = PauliObservable(
    [(-1.0, [(q, "Z"), (q + 1, "Z")]) for q in range(n - 1)]
    + [(-0.5, [(q, "X")]) for q in range(n)]
)
print(hamiltonian.num_terms, hamiltonian.num_groups)   # 7 2
```

## Adjoint and parameter-shift gradients

`expectation_gradient` uses the adjoint method: one forward and one backward pass, whatever
the parameter count. `expectation_gradient_shift` uses the parameter-shift rule, two extra
circuit runs per parameter, and works on backends that have no adjoint pass. Both return
`(value, gradient)`, with one gradient entry per slot.

```python
from prism_q import simulate

value, adjoint = simulate(circuit).seed(42).expectation_gradient(hamiltonian, links)
_, shifted = simulate(circuit).seed(42).expectation_gradient_shift(hamiltonian, links)
print(round(value, 6))                            # -3.206449
print(np.allclose(adjoint, shifted, atol=1e-10))  # True
```

## Optimize

Plain gradient descent: bind the current angles, take the gradient, step.

```python
theta = np.full(params.num_slots, 0.1)
for step in range(100):
    bound = params.bind(circuit, theta)
    energy, gradient = simulate(bound).seed(42).expectation_gradient(hamiltonian, links)
    theta -= 0.2 * gradient
print(round(energy, 6))                         # -3.377401
```

The exact ground energy of this chain is -3.4270. Two layers of `ry` rotations get within
1.5% of it; a third layer closes more of the gap.

## Sweep many points at once

`PreparedCircuit` settles fusion and backend selection once and replays them for every
binding. The `*_many` calls take a `(points, num_slots)` array and cross into Rust once;
up to 16 qubits the rows split across cores.

```python
from prism_q import BackendKind, PreparedCircuit

prepared = PreparedCircuit(circuit, params, BackendKind.statevector())
rng = np.random.default_rng(42)
points = rng.uniform(-np.pi, np.pi, size=(256, params.num_slots))
results = prepared.observable_expectation_many(points, hamiltonian, seed=42)
energies = np.array([r.mean for r in results])
print(energies.shape, round(energies.min(), 4))  # (256,) -2.5401
```

`run_many` and `expectation_values_many` do the same for full runs and for lists of
single observables. Pass the backend explicitly when the template angles are not
representative: a template with every rotation at zero reads as Clifford, and automatic
dispatch would settle on a stabilizer backend that then rejects the bound circuit.

## In Rust

`build_parametric` returns the circuit and its `Parameters` together, and the gradient
terminals take the `Parameters` directly.

```rust
use prism_q::{CircuitBuilder, PauliTerm, simulate};

let n = 4;
let mut builder = CircuitBuilder::new(n);
for q in 0..n {
    builder.ry(0.1, q).param(q);
}
for q in 0..n - 1 {
    builder.cx(q, q + 1);
}
for q in 0..n {
    builder.ry(0.1, q).param(n + q);
}
let (circuit, params) = builder.build_parametric();

let mut hamiltonian: Vec<(f64, Vec<PauliTerm>)> = (0..n - 1)
    .map(|q| (-1.0, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]))
    .collect();
hamiltonian.extend((0..n).map(|q| (-0.5, vec![PauliTerm::x(q)])));

let mut theta = vec![0.1; params.num_slots()];
for _ in 0..100 {
    let bound = params.bind(&circuit, &theta)?;
    let step = simulate(&bound).seed(42).expectation_gradient(&hamiltonian, &params)?;
    for (t, g) in theta.iter_mut().zip(&step.gradient) {
        *t -= 0.2 * g;
    }
}
# Ok::<(), prism_q::PrismError>(())
```

[`examples/gradients.rs`](https://github.com/AbeCoull/prism-q/blob/main/examples/gradients.rs)
adds the parameter-shift check and the batched sweep.

Next: [Noisy Simulation](./noisy-simulation.md).

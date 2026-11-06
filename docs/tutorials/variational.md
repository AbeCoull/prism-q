# Variational Circuits and Gradients

Goal: minimize the energy of a four-site transverse-field Ising chain with a two-layer
ansatz, using exact gradients, then evaluate a batch of parameter points in one call.

## Mark the parameters

`param(slot)` marks the gate just appended as trainable in that slot; several gates may
share a slot.

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
params = builder.parameters()                # Parameters: 8 slots
print(params.num_slots, params.links()[:2])  # 8 [(0, 0), (1, 1)]
```

`params` binds new angles into the circuit and tells the gradient which instruction feeds
which slot.

## The observable

A Hamiltonian is a list of `(coefficient, [(qubit, axis), ...])` terms, here
`H = -sum Z_i Z_{i+1} - 0.5 sum X_i`. `PauliObservable` parses it once and caches the
grouping of commuting terms for reuse inside a loop.

```python
from prism_q import PauliObservable

hamiltonian = PauliObservable(
    [(-1.0, [(q, "Z"), (q + 1, "Z")]) for q in range(n - 1)]
    + [(-0.5, [(q, "X")]) for q in range(n)]
)
print(hamiltonian.num_terms, hamiltonian.num_groups)   # 7 2
```

## Adjoint and parameter-shift gradients

`expectation_gradient` uses the adjoint method: one forward and one backward pass at any
parameter count. `expectation_gradient_shift` uses the parameter-shift rule, two extra
circuit runs per parameter, and works on backends with no adjoint pass. Both return an
`ExpectationGradient`, which unpacks as `(value, gradient)`, one gradient entry per slot.

```python
from prism_q import simulate

value, adjoint = simulate(circuit).seed(42).expectation_gradient(hamiltonian, params)
_, shifted = simulate(circuit).seed(42).expectation_gradient_shift(hamiltonian, params)
print(round(value, 6))                            # -3.206449
print(np.allclose(adjoint, shifted, atol=1e-10))  # True
```

## Optimize

Plain gradient descent. `PreparedCircuit` settles fusion and backend selection once, so
each step only binds the new angles and replays them.

```python
from prism_q import BackendKind, PreparedCircuit

prepared = PreparedCircuit(circuit, params, BackendKind.statevector())
theta = np.full(params.num_slots, 0.1)
for step in range(100):
    energy, gradient = prepared.expectation_gradient(theta, hamiltonian, seed=42)
    theta -= 0.2 * gradient
print(round(energy, 6))                         # -3.377401
```

The exact ground energy is -3.4270. Two layers of `ry` rotations get within 1.5% of it; a
third layer closes more of the gap.

## Sweep many points at once

The `*_many` calls take a `(points, num_slots)` array and cross into Rust once; up to 16
qubits, rows split across cores (14 for gradients).

```python
rng = np.random.default_rng(42)
points = rng.uniform(-np.pi, np.pi, size=(256, params.num_slots))
results = prepared.expectation_many(points, hamiltonian, seed=42)
energies = np.array([r.mean for r in results])
print(energies.shape, round(energies.min(), 4))  # (256,) -2.5401
```

`expectation_gradient_many` returns a `(points,)` array of values and a
`(points, num_slots)` array of gradients:

```python
values, gradients = prepared.expectation_gradient_many(points[:8], hamiltonian, seed=42)
print(values.shape, gradients.shape)  # (8,) (8, 8)
```

`run_many` does the same for full runs, and `expectation_many` with a list of Pauli
strings returns a `(points, observables)` array. Pass the backend explicitly when the
template angles are not typical: with every rotation at zero the template reads as
Clifford, and automatic dispatch would pick a stabilizer backend that then rejects the
bound circuit.

## In Rust

`build_parametric` returns the circuit and its `Parameters` together, and the gradient
terminals take a `PauliObservable` and the `Parameters` directly.

```rust
use prism_q::{CircuitBuilder, PauliObservable, PauliTerm, PreparedCircuit};

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

let hamiltonian = PauliObservable::from_terms(
    (0..n - 1)
        .map(|q| (-1.0, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]))
        .chain((0..n).map(|q| (-0.5, vec![PauliTerm::x(q)]))),
)?;

let mut theta = vec![0.1; params.num_slots()];
let mut prepared = PreparedCircuit::new(circuit, params)?;
for _ in 0..100 {
    let step = prepared.expectation_gradient(&theta, &hamiltonian, 42)?;
    for (t, g) in theta.iter_mut().zip(&step.gradient) {
        *t -= 0.2 * g;
    }
}

// One row per binding; each row has one gradient entry per slot.
let batch = [theta.clone(), vec![0.0; theta.len()]];
let rows = prepared.expectation_gradient_many(&batch, &hamiltonian, 42)?;
assert_eq!(rows[0].gradient.len(), theta.len());
# Ok::<(), prism_q::PrismError>(())
```

[`examples/gradients.rs`](https://github.com/AbeCoull/prism-q/blob/main/examples/gradients.rs)
adds the parameter-shift check and the batched sweep.

Next: [Noisy Simulation](./noisy-simulation.md).

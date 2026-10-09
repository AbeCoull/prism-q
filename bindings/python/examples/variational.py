"""Minimize a transverse-field Ising energy with adjoint gradients, then sweep it."""

import numpy as np

from prism_q import BackendKind, CircuitBuilder, PauliObservable, PreparedCircuit, simulate

n = 4
builder = CircuitBuilder(n)
for q in range(n):
    builder.ry(0.1, q).param(q)
for q in range(n - 1):
    builder.cx(q, q + 1)
for q in range(n):
    builder.ry(0.1, q).param(n + q)
circuit = builder.build()
params = builder.parameters()
links = builder.parameter_links()

hamiltonian = PauliObservable(
    [(-1.0, [(q, "Z"), (q + 1, "Z")]) for q in range(n - 1)]
    + [(-0.5, [(q, "X")]) for q in range(n)]
)

value, adjoint = simulate(circuit).seed(42).expectation_gradient(hamiltonian, links)
_, shifted = simulate(circuit).seed(42).expectation_gradient_shift(hamiltonian, links)
print(f"energy at the template angles: {value:.6f}")
print("adjoint and parameter shift agree:", np.allclose(adjoint, shifted, atol=1e-10))

theta = np.full(params.num_slots, 0.1)
for step in range(100):
    bound = params.bind(circuit, theta)
    energy, gradient = simulate(bound).seed(42).expectation_gradient(hamiltonian, links)
    theta -= 0.2 * gradient
print(f"energy after 100 steps: {energy:.6f}")

prepared = PreparedCircuit(circuit, params, BackendKind.statevector())
rng = np.random.default_rng(42)
points = rng.uniform(-np.pi, np.pi, size=(256, params.num_slots))
energies = [r.mean for r in prepared.observable_expectation_many(points, hamiltonian, seed=42)]
print(f"lowest of 256 random points: {min(energies):.6f}")

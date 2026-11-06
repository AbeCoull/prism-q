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

hamiltonian = PauliObservable(
    [(-1.0, [(q, "Z"), (q + 1, "Z")]) for q in range(n - 1)]
    + [(-0.5, [(q, "X")]) for q in range(n)]
)

value, adjoint = simulate(circuit).seed(42).expectation_gradient(hamiltonian, params)
_, shifted = simulate(circuit).seed(42).expectation_gradient_shift(hamiltonian, params)
print(f"energy at the template angles: {value:.6f}")
print("adjoint and parameter shift agree:", np.allclose(adjoint, shifted, atol=1e-10))

prepared = PreparedCircuit(circuit, params, BackendKind.statevector())
theta = np.full(params.num_slots, 0.1)
for step in range(100):
    energy, gradient = prepared.expectation_gradient(theta, hamiltonian, seed=42)
    theta -= 0.2 * gradient
print(f"energy after 100 steps: {energy:.6f}")

rng = np.random.default_rng(42)
points = rng.uniform(-np.pi, np.pi, size=(256, params.num_slots))
energies = [r.mean for r in prepared.expectation_many(points, hamiltonian, seed=42)]
print(f"lowest of 256 random points: {min(energies):.6f}")

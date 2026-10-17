"""The product-formula builder: angle convention, convergence, and argument checks."""

import math

import numpy as np
import pytest

from prism_q import Circuit, CircuitBuilder, Gate, PauliObservable, PrismError, simulate

_PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _dense(n, terms):
    h = np.zeros((2**n, 2**n), dtype=complex)
    for coefficient, factors in terms:
        letters = dict(factors)
        string = np.array([[1.0 + 0j]])
        for q in reversed(range(n)):
            string = np.kron(string, _PAULI[letters.get(q, "I")])
        h += coefficient * string
    return h


def _exact(n, terms, t, state):
    energies, vectors = np.linalg.eigh(_dense(n, terms))
    return vectors @ (np.exp(-1j * energies * t) * (vectors.conj().T @ state))


# exp(-i c t X Y)|00> = cos(ct)|00> + sin(ct)|11>, which pins the sign and the
# factor of two between the coefficient and the rotation angle.
def test_single_term_matches_its_closed_form():
    c, t = 0.3, 0.7
    circuit = (
        CircuitBuilder(2)
        .trotter([(c, [(0, "X"), (1, "Y")])], t, steps=1, order=1)
        .build()
    )
    state = simulate(circuit).seed(42).state_vector()
    assert circuit.gate_count() == 1
    np.testing.assert_allclose(state[0], math.cos(c * t), atol=1e-12)
    np.testing.assert_allclose(state[3], math.sin(c * t), atol=1e-12)


def test_second_order_converges_from_an_observable():
    n = 3
    terms = [
        (1.0, [(0, "X"), (1, "X")]),
        (0.8, [(1, "Y"), (2, "Y")]),
        (0.6, [(0, "Z"), (1, "Z")]),
        (0.4, [(0, "Z")]),
        (0.3, [(2, "X")]),
    ]
    hamiltonian = PauliObservable(terms)
    plus = simulate(CircuitBuilder(n).h(0).h(1).h(2).build()).seed(42).state_vector()
    exact = _exact(n, terms, 0.9, plus)
    errors = []
    for steps in (2, 4, 8):
        circuit = Circuit(n)
        for q in range(n):
            circuit.add_gate(Gate.h(), [q])
        circuit.add_trotter(hamiltonian, 0.9, steps=steps, order=2)
        assert circuit.gate_count() == n + steps * (2 * len(terms) - 1) - (steps - 1)
        errors.append(np.linalg.norm(simulate(circuit).seed(42).state_vector() - exact))
    assert errors[0] / errors[1] > 3.0
    assert errors[1] / errors[2] > 3.0


def test_fourth_order_beats_second_at_equal_steps():
    terms = [(1.0, [(0, "X"), (1, "X")]), (0.7, [(0, "Z")]), (0.5, [(1, "Z")])]
    plus = simulate(CircuitBuilder(2).h(0).h(1).build()).seed(42).state_vector()
    exact = _exact(2, terms, 1.0, plus)
    by_order = {}
    for order in (2, 4):
        circuit = CircuitBuilder(2).h(0).h(1).trotter(terms, 1.0, steps=4, order=order).build()
        by_order[order] = np.linalg.norm(simulate(circuit).seed(42).state_vector() - exact)
    assert by_order[4] < by_order[2] / 10


def test_rejects_zero_steps_odd_orders_and_out_of_range_qubits():
    term = [(1.0, [(0, "X"), (1, "X")])]
    with pytest.raises(PrismError):
        Circuit(2).add_trotter(term, 1.0, steps=0)
    with pytest.raises(PrismError):
        Circuit(2).add_trotter(term, 1.0, order=3)
    with pytest.raises(PrismError):
        CircuitBuilder(2).trotter([(1.0, [(0, "X"), (5, "X")])], 1.0)

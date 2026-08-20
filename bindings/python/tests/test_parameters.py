import math

import numpy as np
import pytest

from prism_q import (
    BackendKind,
    Circuit,
    CircuitBuilder,
    Gate,
    Parameters,
    PreparedCircuit,
    PrismError,
    simulate,
)

# Template angles are generic rather than zero: dispatch reads the template, and
# an all-zero rotation layer looks Clifford.
TEMPLATE = [0.1] * 8

POINTS = [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    [1.7, -0.9, 2.4, 0.05, -1.2, 3.0, 0.33, -2.7],
    [0.0, math.pi, math.pi / 2, -math.pi / 2, 0.0, 0.0, math.pi, 1.0],
]


def _ansatz(angles):
    builder = CircuitBuilder(4)
    for q, theta in enumerate(angles[:4]):
        builder.ry(theta, q).param(q)
    for q in range(3):
        builder.cx(q, q + 1)
    for q, theta in enumerate(angles[4:]):
        builder.rz(theta, q).param(4 + q)
    return builder


def _prepared_ansatz():
    builder = _ansatz(TEMPLATE)
    return PreparedCircuit(builder.build(), builder.parameters())


def test_bind_agrees_with_a_rebuilt_circuit():
    prepared = _prepared_ansatz()
    for values in POINTS:
        bound = prepared.bind(values)
        rebuilt = _ansatz(values).build()
        assert bound.gate_count() == rebuilt.gate_count()
        np.testing.assert_allclose(
            simulate(bound).seed(42).state_vector(),
            simulate(rebuilt).seed(42).state_vector(),
            atol=1e-12,
        )


def test_run_agrees_with_a_rebuilt_circuit():
    prepared = _prepared_ansatz()
    for values in POINTS:
        held = prepared.run(values, seed=42)
        fresh = simulate(_ansatz(values).build()).seed(42).run()
        np.testing.assert_allclose(held.probabilities, fresh.probabilities, atol=1e-12)
        assert held.metadata.backend == fresh.metadata.backend


def test_run_reuses_one_object_across_a_sweep():
    prepared = _prepared_ansatz()
    assert prepared.reuses_fusion_plan
    first = prepared.run(POINTS[0], seed=42).probabilities
    prepared.run(POINTS[1], seed=42)
    again = prepared.run(POINTS[0], seed=42).probabilities
    np.testing.assert_allclose(first, again, atol=1e-15)


def test_explicit_backend_is_honored():
    builder = _ansatz(TEMPLATE)
    prepared = PreparedCircuit(
        builder.build(), builder.parameters(), BackendKind.statevector()
    )
    assert prepared.run(POINTS[0], seed=42).metadata.backend == "Statevector"


def test_template_and_parameters_survive_binding():
    prepared = _prepared_ansatz()
    prepared.run(POINTS[0], seed=42)
    assert prepared.template.num_qubits == 4
    assert prepared.parameters.num_slots == 8
    np.testing.assert_allclose(prepared.parameters.values(prepared.template), TEMPLATE)


def test_shared_slot_writes_every_linked_gate():
    builder = CircuitBuilder(2)
    builder.rx(0.1, 0).param(0).cx(0, 1).rx(0.1, 1).param(0)
    params = builder.parameters()
    assert params.num_slots == 1
    assert params.links() == [(0, 0), (2, 0)]

    bound = params.bind(builder.build(), [0.75])
    assert params.values(bound) == [0.75]


def test_all_rotations_gives_one_slot_per_gate():
    circuit = _ansatz(TEMPLATE).build()
    params = Parameters.all_rotations(circuit)
    assert params.num_slots == 8
    assert params.unread_slots() == []
    np.testing.assert_allclose(params.values(circuit), TEMPLATE)


def test_named_slots_round_trip():
    circuit = _ansatz(TEMPLATE).build()
    names = [f"theta_{i}" for i in range(8)]
    params = Parameters.all_rotations(circuit).with_names(names)
    assert params.name_of(0) == "theta_0"
    assert params.slot_of("theta_7") == 7
    assert params.slot_of("absent") is None
    assert Parameters(1).name_of(0) is None


def test_declared_slot_no_gate_reads_is_reported_not_rejected():
    circuit = _ansatz(TEMPLATE).build()
    params = Parameters(3)
    params.link(0, 0)
    assert params.unread_slots() == [1, 2]
    assert params.bind(circuit, [0.5, 9.0, 9.0]).gate_count() == circuit.gate_count()


def test_binding_errors_are_prism_errors():
    prepared = _prepared_ansatz()
    with pytest.raises(PrismError):
        prepared.bind([0.1] * 7)
    with pytest.raises(PrismError):
        prepared.bind([float("nan")] + [0.1] * 7)
    with pytest.raises(PrismError):
        Parameters(1).link(0, 1)
    with pytest.raises(PrismError):
        Parameters.all_rotations(_ansatz(TEMPLATE).build()).with_names(["only_one"])


def test_link_to_a_gate_carrying_no_angle_is_rejected():
    circuit = CircuitBuilder(2).h(0).cx(0, 1).build()
    params = Parameters(1)
    params.link(1, 0)
    with pytest.raises(PrismError):
        params.validate(circuit)


def test_param_after_a_non_angle_gate_is_rejected():
    with pytest.raises(PrismError):
        CircuitBuilder(2).h(0).param(0)


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _rotation_matrix(n, theta, factors):
    """exp(-i * theta * P / 2) with qubit 0 in the least significant bit."""
    letters = dict(factors)
    string = np.array([[1.0 + 0j]])
    for q in reversed(range(n)):
        string = np.kron(string, _PAULI[letters.get(q, "I")])
    return math.cos(theta / 2) * np.eye(2**n) - 1j * math.sin(theta / 2) * string


def test_pauli_rotation_matches_its_definition():
    theta = 0.7
    factors = [(0, "X"), (1, "Y"), (2, "Z")]

    plus = CircuitBuilder(3).h(0).h(1).h(2).build()
    rotated = Circuit(3)
    for q in range(3):
        rotated.add_gate(Gate.h(), [q])
    rotated.add_pauli_rotation(theta, factors)

    np.testing.assert_allclose(
        simulate(rotated).seed(42).state_vector(),
        _rotation_matrix(3, theta, factors) @ simulate(plus).seed(42).state_vector(),
        atol=1e-12,
    )


def test_pauli_rotation_lowers_weight_one_and_zz():
    single = Circuit(2)
    single.add_pauli_rotation(0.4, [(1, "Y")])
    assert single.gate_count() == 1
    np.testing.assert_allclose(
        simulate(single).seed(42).state_vector(),
        simulate(CircuitBuilder(2).ry(0.4, 1).build()).seed(42).state_vector(),
        atol=1e-12,
    )

    zz = Circuit(2)
    zz.add_pauli_rotation(0.4, [(0, "Z"), (1, "Z")])
    np.testing.assert_allclose(
        simulate(zz).seed(42).state_vector(),
        simulate(CircuitBuilder(2).rzz(0.4, 0, 1).build()).seed(42).state_vector(),
        atol=1e-12,
    )


def _pauli_ansatz(angles):
    builder = CircuitBuilder(3)
    builder.h(0).h(1).h(2)
    builder.pauli_rotation(angles[0], [(0, "X"), (1, "Y"), (2, "Z")]).param(0)
    builder.pauli_rotation(angles[1], [(0, "Z"), (2, "X")]).param(1)
    return builder


def test_pauli_rotation_binds_through_a_prepared_circuit():
    builder = _pauli_ansatz([0.1, 0.1])
    prepared = PreparedCircuit(builder.build(), builder.parameters())

    for values in ([0.3, 1.1], [-2.0, 0.45]):
        rebuilt = _pauli_ansatz(values).build()
        np.testing.assert_allclose(
            simulate(prepared.bind(values)).seed(42).state_vector(),
            simulate(rebuilt).seed(42).state_vector(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            prepared.run(values, seed=42).probabilities,
            simulate(rebuilt).seed(42).run().probabilities,
            atol=1e-12,
        )


def test_pauli_rotation_gradient_reaches_the_native_gate():
    builder = _pauli_ansatz([0.6, 0.25])
    circuit = builder.build()
    links = builder.parameter_links()
    hamiltonian = [(1.0, [(0, "Z")])]
    value, grad = simulate(circuit).seed(42).expectation_gradient(hamiltonian, links)
    assert grad.shape == (2,)

    eps = 1e-6
    for slot in range(2):
        shifted = [0.6, 0.25]
        shifted[slot] += eps
        up = simulate(_pauli_ansatz(shifted).build()).seed(42).expectation_values(
            [[(0, "Z")]]
        )[0]
        shifted[slot] -= 2 * eps
        down = simulate(_pauli_ansatz(shifted).build()).seed(42).expectation_values(
            [[(0, "Z")]]
        )[0]
        assert math.isclose(grad[slot], (up - down) / (2 * eps), abs_tol=1e-5)
    assert math.isclose(
        value, simulate(circuit).seed(42).expectation_values([[(0, "Z")]])[0], abs_tol=1e-12
    )


def test_pauli_rotation_rejects_bad_factors():
    circuit = Circuit(3)
    with pytest.raises(PrismError):
        circuit.add_pauli_rotation(0.5, [])
    with pytest.raises(PrismError):
        circuit.add_pauli_rotation(0.5, [(0, "X"), (0, "Z")])
    with pytest.raises(PrismError):
        circuit.add_pauli_rotation(0.5, [(0, "X"), (7, "Z")])
    with pytest.raises(PrismError):
        circuit.add_pauli_rotation(0.5, [(0, "Q")])

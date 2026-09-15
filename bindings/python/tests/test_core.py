import math
import importlib

import numpy as np
import pytest

import prism_q
from prism_q import CircuitBuilder, circuits, parse_qasm, run_qasm, simulate

BELL_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


def test_version_is_string():
    assert isinstance(prism_q.__version__, str)
    assert prism_q.__version__.count(".") == 2


def test_bell_state_probabilities():
    bell = CircuitBuilder(2).h(0).cx(0, 1).build()
    out = simulate(bell).seed(42).run()
    probs = out.probabilities
    assert isinstance(probs, np.ndarray)
    assert probs.dtype == np.float64
    assert probs.shape == (4,)
    assert math.isclose(probs[0], 0.5, abs_tol=1e-9)
    assert math.isclose(probs[3], 0.5, abs_tol=1e-9)
    assert math.isclose(probs[1], 0.0, abs_tol=1e-9)


def test_builder_is_chainable_and_counts_gates():
    circuit = CircuitBuilder(3).h(0).cx(0, 1).cx(1, 2).build()
    assert circuit.num_qubits == 3
    assert circuit.gate_count() == 3
    assert circuit.is_clifford_only()


def test_build_is_non_destructive():
    builder = CircuitBuilder(2).h(0).cx(0, 1)
    c1 = builder.build()
    c2 = builder.build()
    assert c1.num_qubits == 2 and c1.gate_count() == 2
    assert c2.num_qubits == 2 and c2.gate_count() == 2
    c3 = builder.t(0).build()
    assert c3.gate_count() == 3


def test_parse_qasm_and_shots_counts():
    circuit = parse_qasm(BELL_QASM)
    result = simulate(circuit).seed(7).shots(1000)
    assert result.num_shots == 1000
    assert result.num_classical_bits == 2
    counts = result.counts()
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 1000


def test_sample_counts():
    circuit = parse_qasm(BELL_QASM)
    cr = simulate(circuit).seed(3).sample_counts(500)
    counts = cr.counts()
    assert sum(counts.values()) == 500
    assert set(counts) <= {"00", "11"}


def test_run_qasm_helper():
    qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];"
    out = run_qasm(qasm, 42)
    probs = out.probabilities
    assert math.isclose(probs[0], 0.5, abs_tol=1e-9)
    assert math.isclose(probs[3], 0.5, abs_tol=1e-9)


def test_statevector_is_complex128():
    sv = simulate(circuits.ghz(3)).seed(1).state_vector()
    assert sv.dtype == np.complex128
    assert sv.shape == (8,)
    assert math.isclose(abs(sv[0]) ** 2, 0.5, abs_tol=1e-9)
    assert math.isclose(abs(sv[7]) ** 2, 0.5, abs_tol=1e-9)


def test_statevector_honours_the_selected_backend():
    # A backend holding no pure state declines rather than being replaced by
    # one that does.
    ghz = circuits.ghz(2)
    sv = simulate(ghz).seed(1).backend(prism_q.BackendKind.stabilizer()).state_vector()
    assert math.isclose(abs(sv[0]) ** 2, 0.5, abs_tol=1e-9)
    with pytest.raises(prism_q.PrismError) as backend:
        simulate(ghz).seed(1).backend(prism_q.BackendKind.density_matrix()).state_vector()
    assert backend.value.kind == "backend_unsupported"

    # A noise model declines for the same reason from the other direction: a
    # mixture has no one pure state to export.
    noise = prism_q.NoiseModel.uniform_depolarizing(ghz, 0.1)
    with pytest.raises(prism_q.PrismError) as mixed:
        simulate(ghz).seed(1).noise(noise).state_vector()
    assert mixed.value.kind


def test_shots_is_a_bool_matrix():
    bell = parse_qasm(BELL_QASM)
    result = simulate(bell).seed(1).shots(16)
    shots = result.shots
    assert shots.dtype == np.bool_
    assert shots.shape == (16, 2)
    # Both classical bits agree in every shot of a Bell pair.
    assert np.array_equal(shots[:, 0], shots[:, 1])


def test_errors_carry_a_kind():
    with pytest.raises(prism_q.PrismError) as excinfo:
        parse_qasm("OPENQASM 3.0;\nqubit[1] q;\nnosuchgate q[0];")
    assert excinfo.value.kind == "unsupported_construct"

    with pytest.raises(prism_q.PrismError) as excinfo:
        simulate(circuits.ghz(2)).seed(1).backend(
            prism_q.BackendKind.density_matrix()
        ).state_vector()
    assert excinfo.value.kind == "backend_unsupported"


def test_marginals():
    bell = CircuitBuilder(2).h(0).cx(0, 1).build()
    marginals = simulate(bell).seed(1).marginals()
    assert len(marginals) == 2
    for p0, p1 in marginals:
        assert math.isclose(p0, 0.5, abs_tol=1e-9)
        assert math.isclose(p1, 0.5, abs_tol=1e-9)


def test_cu_accepts_nested_list_and_numpy():
    xmat_list = [[0, 1], [1, 0]]
    c1 = CircuitBuilder(2).h(0).cu(xmat_list, 0, 1).build()
    xmat_np = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    c2 = CircuitBuilder(2).h(0).cu(xmat_np, 0, 1).build()
    p1 = simulate(c1).seed(1).run().probabilities
    p2 = simulate(c2).seed(1).run().probabilities
    np.testing.assert_allclose(p1, p2, atol=1e-12)
    assert math.isclose(p1[0], 0.5, abs_tol=1e-9)
    assert math.isclose(p1[3], 0.5, abs_tol=1e-9)


@pytest.mark.parametrize(
    "prep, expected_index",
    [
        ([0, 1], 0b111),
        ([0], 0b001),
    ],
)
def test_mcu_toffoli(prep, expected_index):
    xmat = [[0, 1], [1, 0]]
    builder = CircuitBuilder(3)
    for q in prep:
        builder.x(q)
    circuit = builder.mcu(xmat, [0, 1], 2).build()
    probs = simulate(circuit).seed(1).run().probabilities
    assert math.isclose(probs[expected_index], 1.0, abs_tol=1e-9)


def test_invalid_matrix_shape_raises():
    with pytest.raises(prism_q.PrismError):
        prism_q.Gate.cu([[1, 0, 0], [0, 1, 0]])


def test_prebuilt_circuits_smoke():
    assert circuits.qft(4).num_qubits == 4
    assert circuits.qaoa(4, 2, seed=1).num_qubits == 4
    assert circuits.hardware_efficient_ansatz(4, 2, seed=1).num_qubits == 4
    assert circuits.w_state(3).num_qubits == 3
    assert circuits.clifford_t(4, 4, t_fraction=0.2, seed=1).num_qubits == 4


def test_circuits_submodule_is_importable():
    mod = importlib.import_module("prism_q.circuits")
    assert mod.ghz(2).num_qubits == 2


def test_invalid_programmatic_indices_raise_prism_error():
    with pytest.raises(prism_q.PrismError):
        CircuitBuilder(1).h(2)
    with pytest.raises(prism_q.PrismError):
        prism_q.Circuit(1).add_gate(prism_q.Gate.h(), [2])
    with pytest.raises(prism_q.PrismError):
        CircuitBuilder(1, 0).measure(0, 0)


def test_initial_state_evolves_from_the_given_amplitudes():
    # H (cos(pi/8)|0> + sin(pi/8)|1>) has p(0) = (1 + sin(pi/4))/2.
    theta = math.pi / 8
    start = np.array([math.cos(theta), math.sin(theta)], dtype=np.complex128)
    circuit = CircuitBuilder(1).h(0).build()

    probs = simulate(circuit).initial_state(start).seed(42).run().probabilities
    assert math.isclose(probs[0], (1 + 1 / math.sqrt(2)) / 2, abs_tol=1e-12)
    assert math.isclose(probs[1], (1 - 1 / math.sqrt(2)) / 2, abs_tol=1e-12)

    amps = simulate(circuit).initial_state([1.0, 0.0]).seed(42).state_vector()
    assert math.isclose(abs(amps[0]), 1 / math.sqrt(2), abs_tol=1e-12)


def test_initial_state_validation_and_backend_limits():
    circuit = CircuitBuilder(2).cx(0, 1).build()
    with pytest.raises(prism_q.PrismError):
        simulate(circuit).initial_state([1.0, 0.0]).seed(42).run()
    with pytest.raises(prism_q.PrismError):
        simulate(circuit).initial_state([1.0, 1.0, 0.0, 0.0]).seed(42).run()
    with pytest.raises(prism_q.PrismError):
        (
            simulate(circuit)
            .backend(prism_q.BackendKind.stabilizer())
            .initial_state([1.0, 0.0, 0.0, 0.0])
            .seed(42)
            .run()
        )


def test_observable_expectation_matches_weighted_expectation_values():
    circuit = circuits.hardware_efficient_ansatz(6, 2, 42)
    hamiltonian = [
        (0.5, [(0, "Z")]),
        (-1.25, [(1, "Z"), (3, "Z")]),
        (2.0, [(0, "X"), (2, "X")]),
        (0.75, [(1, "Y"), (4, "Y")]),
        (0.25, []),
    ]
    result = simulate(circuit).seed(42).observable_expectation(hamiltonian)
    values = simulate(circuit).seed(42).expectation_values([t for _, t in hamiltonian])
    weighted = sum(c * v for (c, _), v in zip(hamiltonian, values))
    assert result.mean == pytest.approx(weighted, abs=1e-12)
    assert result.variance is not None
    assert result.group_variances is not None
    assert result.variance == pytest.approx(sum(result.group_variances), abs=1e-12)
    assert result.metadata.backend == "Statevector"


def test_observable_expectation_variance_matches_hand_value():
    # |+>: Var(2 Z0) = 4 and Var(X0) = 0 in two commuting groups; forced onto
    # the statevector so the Clifford route cannot serve it.
    from prism_q import BackendKind

    plus = CircuitBuilder(1).h(0).build()
    result = (
        simulate(plus)
        .backend(BackendKind.statevector())
        .seed(42)
        .observable_expectation([(2.0, [(0, "Z")]), (1.0, [(0, "X")])])
    )
    assert result.mean == pytest.approx(1.0, abs=1e-12)
    assert result.variance == pytest.approx(4.0, abs=1e-12)
    assert len(result.group_variances) == 2


def test_observable_expectation_clifford_route_has_no_variance():
    bell = CircuitBuilder(2).h(0).cx(0, 1).build()
    result = (
        simulate(bell)
        .seed(42)
        .observable_expectation([(2.0, [(0, "Z"), (1, "Z")]), (-1.0, [(0, "X"), (1, "X")])])
    )
    assert result.mean == pytest.approx(1.0, abs=1e-12)
    assert result.variance is None
    assert result.group_variances is None


def test_subset_probability_shows_what_marginals_cannot():
    circuit = prism_q.parse_qasm(
        "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncnot q[0], q[1];"
    )
    sim = prism_q.simulate(circuit).seed(42)
    assert np.allclose([p for p, _ in sim.marginals()], [0.5, 0.5])
    joint = prism_q.simulate(circuit).seed(42).probabilities_of([0, 1])
    assert np.allclose(joint, [0.5, 0.0, 0.0, 0.5])

    # A Bell pair is symmetric under a transposition, so the target order shows
    # only on a state that is not.
    asymmetric = prism_q.parse_qasm("OPENQASM 3.0;\nqubit[2] q;\nx q[0];")
    sim = prism_q.simulate(asymmetric).seed(42)
    assert np.allclose(sim.probabilities_of([0, 1]), [0.0, 1.0, 0.0, 0.0])
    swapped = prism_q.simulate(asymmetric).seed(42).probabilities_of([1, 0])
    # qubits[0] is the lowest bit, so naming the pair the other way transposes.
    assert np.allclose(swapped, [0.0, 0.0, 1.0, 0.0])


def test_reduced_density_matrix_and_entropy_of_a_bell_pair():
    circuit = prism_q.parse_qasm(
        "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncnot q[0], q[1];"
    )
    reduced = prism_q.simulate(circuit).seed(42).reduced_density_matrix([0])
    assert reduced.qubits == [0]
    assert reduced.matrix.shape == (2, 2)
    assert np.allclose(reduced.matrix, np.eye(2) * 0.5)
    assert np.isclose(reduced.purity, 0.5)
    assert reduced.metadata.is_exact

    result = prism_q.simulate(circuit).seed(42).entanglement_entropy([0])
    assert np.isclose(result.entropy, math.log(2))
    assert np.allclose(sorted(result.schmidt_values), [2**-0.5, 2**-0.5])


def test_observable_variance_is_the_operator_spread():
    # ry(pi/4)|0> is the +1 eigenstate of (X + Z)/sqrt(2), so the operator has
    # no spread while measuring X and Z in separate groups reads 1/2 each.
    circuit = prism_q.parse_qasm(
        f"OPENQASM 3.0;\nqubit[1] q;\nry({math.pi / 4}) q[0];"
    )
    terms = [(1.0, [(0, "X")]), (1.0, [(0, "Z")])]
    operator = prism_q.simulate(circuit).seed(42).observable_variance(terms)
    grouped = prism_q.simulate(circuit).seed(42).observable_expectation(terms)
    assert np.isclose(operator.variance, 0.0, atol=1e-9)
    assert np.isclose(grouped.variance, 1.0)
    assert np.isclose(operator.mean, grouped.mean)


def test_measurement_map_lists_every_written_bit():
    circuit = prism_q.parse_qasm(
        "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nh q[0];\nc = measure q;"
    )
    assert circuit.measurement_map() == [(0, 0), (1, 1)]

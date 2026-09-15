"""Braket-dialect parsing through the Python surface."""

import math

import numpy as np
import pytest

import prism_q
from prism_q import parse_braket, simulate


def test_result_pragmas_cross_as_dicts():
    program = parse_braket(
        "OPENQASM 3.0;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "#pragma braket result probability all\n"
        "#pragma braket result expectation z(q[0]) @ z(q[1])\n"
    )
    assert program.circuit.gate_count() == 2
    results = program.results
    assert [r["type"] for r in results] == ["probability", "expectation"]
    # `all` and an omitted list both mean every qubit.
    assert results[0]["targets"] is None
    assert results[0]["requires_exact"] is False
    factors = results[1]["observable"]
    assert [f["kind"] for f in factors] == ["z", "z"]
    assert factors[0]["targets"] == [0]


def test_state_vector_result_is_exact_only():
    program = parse_braket(
        "OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n#pragma braket result state_vector\n"
    )
    assert program.results[0]["requires_exact"] is True
    assert program.noise is None


def test_hermitian_observable_carries_its_matrix():
    program = parse_braket(
        "OPENQASM 3.0;\n"
        "qubit[1] q;\n"
        "h q[0];\n"
        "#pragma braket result variance hermitian([[0, -1im], [1im, 0]]) q[0]\n"
    )
    factor = program.results[0]["observable"][0]
    assert factor["kind"] == "hermitian"
    assert factor["matrix"].dtype == np.complex128
    np.testing.assert_allclose(factor["matrix"], [[0.0, -1.0j], [1.0j, 0.0]])


def test_noise_pragmas_build_a_runnable_model():
    # A flip on the control before the cnot would propagate to both qubits and
    # leave the correlation intact, so the channel sits after it.
    program = parse_braket(
        "OPENQASM 3.0;\n"
        "qubit[2] q;\n"
        "bit[2] c;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "#pragma braket noise bit_flip(0.1) q[0]\n"
        "c = measure q;\n"
    )
    noise = program.noise
    assert noise is not None

    def disagreement(model):
        counts = (
            simulate(program.circuit).seed(7).noise(model).sample_counts(2000).counts()
        )
        return sum(n for key, n in counts.items() if key[0] != key[1]) / 2000

    assert 0.05 < disagreement(noise) < 0.16
    clean = prism_q.NoiseModel.uniform_depolarizing(program.circuit, 0.0)
    assert disagreement(clean) == 0.0


def test_braket_angles_are_radians():
    turns = prism_q.parse_qasm("OPENQASM 3.0;\nqubit[1] q;\ngpi(0.25) q[0];")
    radians = parse_braket(
        f"OPENQASM 3.0;\nqubit[1] q;\ngpi({math.tau * 0.25}) q[0];"
    ).circuit
    a = simulate(turns).seed(1).state_vector()
    b = simulate(radians).seed(1).state_vector()
    assert np.allclose(a, b)


def test_a_pragma_needs_the_braket_entry_point():
    source = "OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n#pragma braket result state_vector\n"
    with pytest.raises(prism_q.PrismError) as excinfo:
        prism_q.parse_qasm(source)
    assert excinfo.value.kind == "unsupported_construct"
    assert parse_braket(source).results


def test_malformed_pragmas_report_their_kind():
    with pytest.raises(prism_q.PrismError) as excinfo:
        parse_braket(
            "OPENQASM 3.0;\n"
            "qubit[1] q;\n"
            "h q[0];\n"
            "#pragma braket noise bit_flip(0.9) q[0]\n"
        )
    assert excinfo.value.kind == "invalid_parameter"


def test_evaluated_results_use_braket_basis_order():
    program = parse_braket(
        "OPENQASM 3.0;\n"
        "qubit[2] q;\n"
        "x q[0];\n"
        '#pragma braket result state_vector\n'
        "#pragma braket result probability all\n"
        '#pragma braket result amplitude "10", "01"\n'
        "#pragma braket result density_matrix all\n"
    )
    values = program.evaluate()
    assert [v["type"] for v in values] == [
        "state_vector",
        "probability",
        "amplitude",
        "density_matrix",
    ]
    # Braket writes q[0] as the most significant bit, so `x q[0]` lands at 2.
    assert np.argmax(np.abs(values[0]["value"])) == 2
    assert np.isclose(values[1]["value"][2], 1.0)
    assert np.isclose(abs(values[2]["value"]["10"]), 1.0)
    assert np.isclose(abs(values[2]["value"]["01"]), 0.0)
    assert np.isclose(values[3]["value"][2, 2], 1.0)


def test_expectation_and_variance_of_a_hermitian_observable():
    matrix = np.array([[1.0, 2.0 - 1.0j], [2.0 + 1.0j, -3.0]])
    program = parse_braket(
        "OPENQASM 3.0;\n"
        "qubit[1] q;\n"
        "h q[0];\n"
        "#pragma braket result expectation hermitian([[1, 2-1im], [2+1im, -3]]) q[0]\n"
        "#pragma braket result variance hermitian([[1, 2-1im], [2+1im, -3]]) q[0]\n"
    )
    values = program.evaluate()
    state = simulate(program.circuit).seed(42).state_vector()
    mean = np.real(state.conj() @ matrix @ state)
    second = np.real(state.conj() @ matrix @ matrix @ state)
    assert np.isclose(values[0]["value"][0], mean)
    assert np.isclose(values[1]["value"][0], second - mean * mean)


def test_an_untargeted_observable_reports_one_value_per_qubit():
    program = parse_braket(
        "OPENQASM 3.0;\nqubit[3] q;\nx q[1];\n"
        "#pragma braket result expectation z all\n"
    )
    assert np.allclose(program.evaluate()[0]["value"], [1.0, -1.0, 1.0])


def test_noise_selects_the_density_matrix_without_being_asked():
    program = parse_braket(
        "OPENQASM 3.0;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "#pragma braket noise bit_flip(0.25) q[1]\n"
        "#pragma braket result probability all\n"
    )
    values = program.evaluate()
    assert np.allclose(values[0]["value"], [0.375, 0.125, 0.125, 0.375])


def test_a_sample_request_is_declined():
    program = parse_braket(
        "OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n"
        "#pragma braket result sample z(q[0])\n"
    )
    with pytest.raises(prism_q.PrismError) as excinfo:
        program.evaluate()
    assert excinfo.value.kind == "backend_unsupported"


def test_shots_report_a_sample_series_and_its_statistics():
    program = parse_braket(
        "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncnot q[0], q[1];\n"
        "#pragma braket result sample z(q[0]) @ z(q[1])\n"
        "#pragma braket result expectation z(q[0]) @ z(q[1])\n"
        "#pragma braket result variance z(q[0]) @ z(q[1])\n"
        "#pragma braket result probability all\n"
    )
    values = program.evaluate(shots=2000)
    series = values[0]["value"]
    assert len(series) == 1
    assert series[0].shape == (2000,)
    # Both halves of a Bell pair always agree, so every shot reads +1.
    assert np.allclose(series[0], 1.0)
    assert np.isclose(values[1]["value"][0], 1.0)
    assert np.isclose(values[2]["value"][0], 0.0)
    assert np.allclose(values[3]["value"], [0.5, 0.0, 0.0, 0.5], atol=0.05)


def test_shots_decline_the_requests_that_read_the_state():
    program = parse_braket(
        "OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n"
        "#pragma braket result state_vector\n"
    )
    with pytest.raises(prism_q.PrismError) as excinfo:
        program.evaluate(shots=100)
    assert excinfo.value.kind == "backend_unsupported"


def test_observables_in_conflicting_bases_are_rejected_under_shots():
    program = parse_braket(
        "OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n"
        "#pragma braket result expectation x(q[0])\n"
        "#pragma braket result expectation z(q[0])\n"
    )
    with pytest.raises(prism_q.PrismError):
        program.evaluate(shots=100)

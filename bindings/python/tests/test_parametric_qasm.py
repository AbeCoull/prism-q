import numpy as np
import pytest

from prism_q import (
    BackendKind,
    PreparedCircuit,
    PrismError,
    parse_qasm,
    parse_qasm_parametric,
    simulate,
)

PARAMETRIC = """
OPENQASM 3.0;
input float[64] theta;
input angle phi;
input float[64] unused;
qubit[3] q;
h q[0];
rx(theta) q[0];
cx q[0], q[1];
rz(phi) q[1];
rzz(theta) q[1], q[2];
ry(0.3) q[2];
"""


def test_named_inputs_preserve_declaration_order_and_shared_slots():
    template, params = parse_qasm_parametric(PARAMETRIC)
    assert template.num_qubits == 3
    assert params.num_slots == 3
    assert [params.name_of(i) for i in range(3)] == ["theta", "phi", "unused"]
    assert params.slot_of("phi") == 1
    assert params.slot_of("absent") is None
    assert params.links() == [(1, 0), (3, 1), (4, 0)]
    assert params.unread_slots() == [2]
    assert params.values(template) == [0.0, 0.0, 0.0]
    params.validate(template)


@pytest.mark.parametrize("theta,phi", [(0.41, 1.27), (-0.8, 0.2), (0.0, 0.0)])
def test_bound_and_prepared_inputs_match_numeric_qasm(theta, phi):
    template, params = parse_qasm_parametric(PARAMETRIC)
    values = [theta, phi, 5.0]
    numeric = parse_qasm(
        f"""OPENQASM 3.0;
        qubit[3] q;
        h q[0];
        rx({theta}) q[0];
        cx q[0], q[1];
        rz({phi}) q[1];
        rzz({theta}) q[1], q[2];
        ry(0.3) q[2];
        """
    )
    expected = simulate(numeric).seed(42)
    bound = params.bind(template, values)
    np.testing.assert_allclose(
        simulate(bound).seed(42).state_vector(), expected.state_vector(), atol=1e-12
    )
    prepared = PreparedCircuit(template, params, BackendKind.statevector())
    np.testing.assert_allclose(
        prepared.run(values, seed=42).probabilities,
        expected.run().probabilities,
        atol=1e-12,
    )
    assert params.values(template) == [0.0, 0.0, 0.0]


def test_parametric_parser_accepts_a_program_without_inputs():
    source = "OPENQASM 2.0; qreg q[1]; h q[0];"
    template, params = parse_qasm_parametric(source)
    assert params.num_slots == 0
    np.testing.assert_allclose(
        simulate(params.bind(template, [])).seed(42).state_vector(),
        simulate(parse_qasm(source)).seed(42).state_vector(),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "source,kind",
    [
        ("OPENQASM 3.0; qubit[1] q; rx(", "parse"),
        (
            "OPENQASM 3.0; input float[64] t; qubit[1] q; rx(2 * t) q[0];",
            "unsupported_construct",
        ),
        ("OPENQASM 3.0; input int[32] n; qubit[1] q;", "unsupported_construct"),
    ],
)
def test_parametric_parser_preserves_error_kinds(source, kind):
    with pytest.raises(PrismError) as error:
        parse_qasm_parametric(source)
    assert error.value.kind == kind


def test_regular_parser_still_rejects_unbound_inputs():
    with pytest.raises(PrismError) as error:
        parse_qasm(PARAMETRIC)
    assert error.value.kind == "invalid_parameter"

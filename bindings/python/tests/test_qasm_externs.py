import traceback

import numpy as np
import pytest

from prism_q import (
    PrismError,
    parse_qasm,
    parse_qasm_dynamic,
    parse_qasm_parametric,
    simulate,
    simulate_program,
)


def state(circuit):
    return simulate(circuit).seed(42).state_vector()


def test_a_call_on_constants_folds_at_parse_time():
    calls = []

    def scale(n, x):
        calls.append((n, x))
        return n * x

    circuit = parse_qasm(
        "OPENQASM 3.0; extern scale(int[8], float) -> float; qubit[1] q;"
        " rx(scale(257, pi / 2)) q[0];",
        externs={"scale": scale},
    )
    assert calls == [(1, np.pi / 2)]
    np.testing.assert_allclose(
        state(circuit),
        state(parse_qasm("OPENQASM 3.0; qubit[1] q; rx(pi / 2) q[0];")),
        atol=1e-12,
    )


def test_parametric_parse_takes_externs_beside_inputs():
    template, params = parse_qasm_parametric(
        "OPENQASM 3.0; extern twice(int) -> int; input int n; input float t;"
        " qubit[1] q; for int i in [1:twice(n)] { rx(t) q[0]; }",
        inputs={"n": 1},
        externs={"twice": lambda n: 2 * n},
    )
    assert params.num_slots == 1
    np.testing.assert_allclose(
        state(params.bind(template, [0.3])),
        state(parse_qasm("OPENQASM 3.0; qubit[1] q; rx(0.6) q[0];")),
        atol=1e-12,
    )


def test_a_call_on_a_measured_bit_runs_in_each_shot():
    program = parse_qasm_dynamic(
        """
        OPENQASM 3.0;
        extern flip(bit) -> bool;
        qubit[2] q;
        bit[2] c;
        h q[0];
        c[0] = measure q[0];
        if (flip(c[0])) x q[1];
        c[1] = measure q[1];
        """,
        externs={"flip": lambda bit: not bit},
    )
    shots = simulate_program(program).seed(42).shots(200).shots
    assert (shots[:, 1] != shots[:, 0]).all()
    assert shots[:, 0].any() and not shots[:, 0].all()


def test_an_int_result_for_a_float_is_promoted():
    circuit = parse_qasm(
        "OPENQASM 3.0; extern three() -> float; qubit[1] q; rx(three()) q[0];",
        externs={"three": lambda: 3},
    )
    np.testing.assert_allclose(
        state(circuit),
        state(parse_qasm("OPENQASM 3.0; qubit[1] q; rx(3.0) q[0];")),
        atol=1e-12,
    )


def test_without_externs_every_parse_declines_an_extern():
    source = "OPENQASM 3.0; extern f(int) -> float; qubit[1] q;"
    for parse in (parse_qasm, parse_qasm_parametric, parse_qasm_dynamic):
        with pytest.raises(PrismError) as error:
            parse(source)
        assert error.value.kind == "unsupported_construct"


@pytest.mark.parametrize(
    "function",
    [lambda n: "one", lambda n: 1.5, lambda n: True],
)
def test_a_result_of_the_wrong_kind_is_invalid(function):
    with pytest.raises(PrismError) as error:
        parse_qasm(
            "OPENQASM 3.0; extern f(int) -> int; qubit[1] q; rx(f(1)) q[0];",
            externs={"f": function},
        )
    assert error.value.kind == "invalid_parameter"


def test_an_exception_in_the_callable_names_the_extern():
    def broken(n):
        raise ValueError("no entry")

    with pytest.raises(PrismError) as error:
        parse_qasm(
            "OPENQASM 3.0; extern f(int) -> float; qubit[1] q; rx(f(1)) q[0];",
            externs={"f": broken},
        )
    assert error.value.kind == "invalid_parameter"
    assert "extern `f` raised" in str(error.value)
    assert "no entry" in str(error.value)
    cause = error.value.__cause__
    assert isinstance(cause, ValueError)
    assert cause.args == ("no entry",)
    assert any(frame.name == "broken" for frame in traceback.extract_tb(cause.__traceback__))


def test_a_runtime_exception_surfaces_from_the_run():
    def broken(bit):
        raise RuntimeError("decoder offline")

    program = parse_qasm_dynamic(
        "OPENQASM 3.0; extern f(bit) -> bool; qubit[1] q; bit[1] c;"
        " h q[0]; c[0] = measure q[0]; if (f(c[0])) x q[0];",
        externs={"f": broken},
    )
    for run in (lambda sim: sim.shots(8), lambda sim: sim.sample_counts(8)):
        with pytest.raises(PrismError) as error:
            run(simulate_program(program).seed(42))
        assert error.value.kind == "invalid_parameter"
        assert "decoder offline" in str(error.value)
        assert isinstance(error.value.__cause__, RuntimeError)
        assert any(
            frame.name == "broken"
            for frame in traceback.extract_tb(error.value.__cause__.__traceback__)
        )


def test_a_parametric_parse_chains_the_raised_exception():
    def broken(n):
        raise KeyError(n)

    with pytest.raises(PrismError) as error:
        parse_qasm_parametric(
            "OPENQASM 3.0; extern f(int) -> float; qubit[1] q; rx(f(4)) q[0];",
            externs={"f": broken},
        )
    assert error.value.kind == "invalid_parameter"
    assert isinstance(error.value.__cause__, KeyError)


def test_an_error_with_no_raise_has_no_cause():
    with pytest.raises(PrismError) as error:
        parse_qasm(
            "OPENQASM 3.0; extern f(int) -> int; qubit[1] q; rx(f(1)) q[0];",
            externs={"f": lambda n: "one"},
        )
    assert error.value.__cause__ is None


def test_a_def_call_bounds_a_for_range():
    source = (
        "OPENQASM 3.0; def twice(int n) -> int { return 2 * n; } qubit[1] q;"
        " for int i in [1:twice(5)] { rx(0.1) q[0]; }"
    )
    want = state(parse_qasm("OPENQASM 3.0; qubit[1] q; rx(1.0) q[0];"))
    np.testing.assert_allclose(state(parse_qasm(source)), want, atol=1e-12)
    circuit = parse_qasm_dynamic(source).static_circuit()
    np.testing.assert_allclose(state(circuit), want, atol=1e-12)


@pytest.mark.parametrize(
    ("source", "externs", "kind"),
    [
        ("OPENQASM 3.0; extern f(int) -> float; qubit[1] q;", None, "unsupported_construct"),
        ("OPENQASM 3.0; extern f(int) -> float; qubit[1] q;", {}, "invalid_parameter"),
        ("OPENQASM 3.0; extern f(int) -> float; qubit[1] q;", {"f": 3}, "invalid_parameter"),
        ("OPENQASM 3.0; extern f(int); qubit[1] q;", {"f": abs}, "unsupported_construct"),
        (
            "OPENQASM 3.0; extern f(bit[65]) -> float; qubit[1] q;",
            {"f": abs},
            "unsupported_construct",
        ),
        (
            "OPENQASM 3.0; extern f(int) -> float; qubit[1] q; rx(f(1, 2)) q[0];",
            {"f": abs},
            "parse",
        ),
    ],
)
def test_declines_keep_their_error_kinds(source, externs, kind):
    with pytest.raises(PrismError) as error:
        parse_qasm(source, externs=externs)
    assert error.value.kind == kind

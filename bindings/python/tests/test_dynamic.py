import numpy as np
import pytest

from prism_q import (
    BackendKind,
    DynamicProgramBuilder,
    Gate,
    PrismError,
    parse_qasm,
    parse_qasm_dynamic,
    simulate,
    simulate_program,
)

RUS = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
uint[8] tries = 1;
h q[0];
c[0] = measure q[0];
while (c[0]) {
    reset q[0];
    h q[0];
    c[0] = measure q[0];
    tries += 1;
}
"""


def rus_builder():
    b = DynamicProgramBuilder(1, 1)
    b.declare("tries", "uint[8]", 1)
    b.add_gate(Gate.h(), [0]).add_measure(0, 0)
    b.begin_while("rus", "c[0]")
    b.add_reset(0).add_gate(Gate.h(), [0]).add_measure(0, 0)
    b.assign("tries", "tries + 1")
    b.end()
    return b


def test_parsed_loop_ends_on_zero_and_reports_shots():
    program = parse_qasm_dynamic(RUS)
    assert program.num_qubits == 1
    assert program.variables == ["tries"]
    assert program.static_circuit() is None
    result = simulate_program(program).seed(7).shots(500)
    assert result.num_shots == 500
    assert not result.shots[:, 0].any(), "every shot leaves the loop on a zero"


def test_builder_loop_counts_trials_geometrically():
    b = DynamicProgramBuilder(1, 5)
    b.declare("tries", "uint[4]", 1)
    b.add_gate(Gate.h(), [0]).add_measure(0, 0)
    b.begin_while("rus", "c[0] && tries < 15")
    b.add_reset(0).add_gate(Gate.h(), [0]).add_measure(0, 0)
    b.assign("tries", "tries + 1")
    b.end()
    for bit in range(4):
        b.begin_if(f"((tries >> {bit}) & 1) == 1")
        b.add_gate(Gate.x(), [0]).add_measure(0, bit + 1).add_reset(0)
        b.end()
    program = b.build()
    shots = simulate_program(program).seed(11).shots(4000).shots
    tries = shots[:, 1:] @ np.array([1, 2, 4, 8])
    assert abs(tries.mean() - 2.0) < 0.1


def test_counts_agree_with_shots_and_seed_repeats():
    program = rus_builder().build()
    first = simulate_program(program).seed(3).shots(300)
    again = simulate_program(program).seed(3).shots(300)
    assert np.array_equal(first.shots, again.shots)
    counts = simulate_program(program).seed(3).sample_counts(300)
    assert counts.counts() == first.counts()
    assert counts.num_classical_bits == 1


def test_static_program_runs_as_its_circuit():
    source = "OPENQASM 3.0; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;"
    program = parse_qasm_dynamic(source)
    assert program.static_circuit() is not None
    dynamic = simulate_program(program).seed(5).shots(200).shots
    plain = simulate(parse_qasm(source)).seed(5).shots(200).shots
    assert np.array_equal(dynamic, plain)


def test_runtime_rotation_and_else():
    b = DynamicProgramBuilder(2, 2)
    b.declare("theta", "float", 0.0)
    b.add_gate(Gate.x(), [0]).add_measure(0, 0)
    b.begin_if("c[0]").assign("theta", "pi").begin_else().assign("theta", 0.0).end()
    b.add_rotation("rx", [1], "theta").add_measure(1, 1)
    counts = simulate_program(b.build()).seed(1).sample_counts(64).counts()
    assert counts == {"11": 64}


def test_break_and_continue_shape_the_loop():
    b = DynamicProgramBuilder(1, 1)
    b.declare("n", "int[8]")
    b.begin_while("count", True)
    b.assign("n", "n + 1")
    b.begin_if("n < 3").continue_loop().end()
    b.break_loop()
    b.end()
    b.begin_if("n == 3").add_gate(Gate.x(), [0]).end()
    b.add_measure(0, 0)
    counts = simulate_program(b.build()).seed(2).sample_counts(16).counts()
    assert counts == {"1": 16}


def test_runaway_loop_raises_step_limit_naming_it():
    b = DynamicProgramBuilder(1, 0)
    b.begin_while("forever", True).add_gate(Gate.x(), [0]).end()
    program = b.build()
    with pytest.raises(PrismError, match="forever") as err:
        simulate_program(program).max_steps(50).shots(1)
    assert err.value.kind == "step_limit"


def test_explicit_backend_and_declines():
    program = rus_builder().build()
    result = simulate_program(program).backend(BackendKind.stabilizer()).seed(4).shots(50)
    assert result.metadata.backend == "Stabilizer"
    with pytest.raises(PrismError) as err:
        simulate_program(program).backend(BackendKind.stabilizer_rank()).shots(10)
    assert err.value.kind == "incompatible_backend"


def test_builder_rejects_bad_input():
    b = DynamicProgramBuilder(1, 1)
    with pytest.raises(PrismError):
        b.declare("x", "complex")
    with pytest.raises(PrismError):
        b.assign("missing", 1)
    with pytest.raises(PrismError):
        b.begin_if("nope + 1")
    with pytest.raises(PrismError):
        b.end()
    b.build()
    with pytest.raises(PrismError, match="already built"):
        b.build()


def test_parse_declines_loops_and_points_at_dynamic_entry():
    with pytest.raises(PrismError, match="parse_dynamic") as err:
        parse_qasm(RUS)
    assert err.value.kind == "unsupported_construct"

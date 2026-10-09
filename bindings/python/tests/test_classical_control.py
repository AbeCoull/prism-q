import pytest

import prism_q
from prism_q import ClassicalCondition, CircuitBuilder, Gate, simulate


def test_measure_in_basis_reads_the_eigenvalue():
    plus = CircuitBuilder(1, 1).h(0).measure_in_basis(0, "X", 0).build()
    assert simulate(plus).seed(1).shots(200).counts() == {"0": 200}
    minus = CircuitBuilder(1, 1).x(0).h(0).measure_in_basis(0, "x", 0).build()
    assert simulate(minus).seed(1).shots(200).counts() == {"1": 200}
    plus_i = CircuitBuilder(1, 1).h(0).s(0).measure_in_basis(0, "Y", 0).build()
    assert simulate(plus_i).seed(1).shots(200).counts() == {"0": 200}


def test_measure_pauli_product_reads_the_parity_on_an_extra_qubit():
    builder = CircuitBuilder(2, 2).h(0).cx(0, 1)
    builder.measure_pauli_product([(0, "X"), (1, "X")], 0)
    builder.measure_pauli_product([(0, "Z"), (1, "Z")], 1)
    circuit = builder.build()
    assert circuit.num_qubits == 3
    assert simulate(circuit).seed(1).shots(300).counts() == {"00": 300}


def test_builder_reset():
    circuit = CircuitBuilder(1, 1).x(0).reset(0).measure(0, 0).build()
    assert simulate(circuit).seed(1).shots(50).counts() == {"0": 50}


def test_measurement_arguments_are_checked():
    builder = CircuitBuilder(2, 1)
    with pytest.raises(prism_q.PrismError):
        builder.measure_in_basis(0, "W", 0)
    with pytest.raises(prism_q.PrismError):
        builder.measure_in_basis(2, "X", 0)
    with pytest.raises(prism_q.PrismError):
        builder.measure_in_basis(0, "X", 1)
    with pytest.raises(prism_q.PrismError):
        builder.measure_pauli_product([], 0)
    with pytest.raises(prism_q.PrismError):
        builder.measure_pauli_product([(0, "Z"), (0, "X")], 0)
    with pytest.raises(prism_q.PrismError):
        builder.reset(5)


def test_conditional_gate_applies_on_the_measured_bit():
    circuit = (
        CircuitBuilder(2, 2)
        .x(0)
        .measure(0, 0)
        .conditional(ClassicalCondition.bit(0), Gate.x(), [1])
        .measure(1, 1)
        .build()
    )
    assert simulate(circuit).seed(1).shots(100).counts() == {"11": 100}
    skipped = (
        CircuitBuilder(2, 2)
        .measure(0, 0)
        .conditional(ClassicalCondition.bit(0), Gate.x(), [1])
        .measure(1, 1)
        .build()
    )
    assert simulate(skipped).seed(1).shots(100).counts() == {"00": 100}


def test_guarded_if_else_takes_exactly_one_branch():
    builder = CircuitBuilder(2, 2).h(0).measure(0, 0)
    builder.guarded(ClassicalCondition.bit(0), lambda then: then.x(1), lambda other: other.id(1))
    builder.measure(1, 1)
    counts = simulate(builder.build()).seed(4).shots(2000).counts()
    assert set(counts) == {"00", "11"}


def test_guarded_body_may_measure_and_reset():
    builder = CircuitBuilder(3, 3).x(0).measure(0, 0)

    def body(inner):
        inner.x(1).measure(1, 1).reset(1)

    builder.guarded(ClassicalCondition.bit(0, True), body)
    builder.measure(1, 2)
    assert simulate(builder.build()).seed(1).shots(64).counts() == {"110": 64}


def test_nested_regions_and_register_conditions():
    builder = CircuitBuilder(3, 3).x(0).x(1).measure(0, 0).measure(1, 1)
    builder.guarded(
        ClassicalCondition.register_equals(0, 2, 3),
        lambda outer: outer.guarded(ClassicalCondition.parity([0, 1], False), lambda inner: inner.x(2)),
    )
    builder.measure(2, 2)
    assert simulate(builder.build()).seed(1).shots(32).counts() == {"111": 32}


def test_else_needs_a_body_that_leaves_the_condition_bits_alone():
    builder = CircuitBuilder(2, 2).h(0).measure(0, 0)
    with pytest.raises(prism_q.PrismError):
        builder.guarded(
            ClassicalCondition.bit(0),
            lambda then: then.measure(1, 0),
            lambda other: other.x(1),
        )


def test_guarded_rejects_widening_and_out_of_range_bits():
    builder = CircuitBuilder(2, 1)
    with pytest.raises(prism_q.PrismError):
        builder.guarded(
            ClassicalCondition.bit(0),
            lambda body: body.measure_pauli_product([(0, "Z"), (1, "Z")], 0),
        )
    with pytest.raises(prism_q.PrismError):
        builder.guarded(ClassicalCondition.bit(3), lambda body: body.x(0))
    with pytest.raises(prism_q.PrismError):
        builder.conditional(ClassicalCondition.bit(1), Gate.x(), [0])
    assert builder.build().gate_count() == 0


def test_errors_raised_inside_a_body_propagate():
    builder = CircuitBuilder(1, 1)

    def body(inner):
        raise ValueError("from the body")

    with pytest.raises(ValueError, match="from the body"):
        builder.guarded(ClassicalCondition.bit(0), body)


def test_condition_values():
    one = ClassicalCondition.bit(2)
    assert one.bits == [2]
    assert one.evaluate([False, False, True])
    assert not (~one).evaluate([False, False, True])
    assert one.negate().evaluate([False, False, False])
    register = ClassicalCondition.register_equals(1, 2, 2)
    assert register.bits == [1, 2]
    assert register.evaluate([False, False, True])
    assert ClassicalCondition.register_not_equals(1, 2, 2).evaluate([False, True, False])
    parity = ClassicalCondition.parity([0, 1])
    assert parity.evaluate([True, False])
    assert not parity.evaluate([True, True])
    with pytest.raises(prism_q.PrismError):
        one.evaluate([True])
    with pytest.raises(prism_q.PrismError):
        ClassicalCondition.register_equals(0, 2, 4)
    with pytest.raises(prism_q.PrismError):
        ClassicalCondition.register_equals(0, 65, 0)
    with pytest.raises(prism_q.PrismError):
        ClassicalCondition.parity([])

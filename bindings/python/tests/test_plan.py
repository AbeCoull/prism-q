import pytest

from prism_q import BackendKind, CircuitBuilder, Limits, PrismError, Query, simulate


def test_plan_counts_matches_execution_and_keeps_request():
    circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1).build()
    request = simulate(circuit).seed(42)
    plan = request.plan(Query.counts(100))
    result = request.sample_counts(100)
    assert plan.would_fail is None
    assert plan.backend == result.metadata.backend
    assert plan.rule == result.metadata.reason
    assert request.plan(Query.counts(100)).rule == plan.rule


def test_plan_exposes_resource_error_without_raising_it():
    circuit = CircuitBuilder(3).h(0).cx(0, 1).ry(0.3, 2).build()
    request = simulate(circuit).backend(BackendKind.statevector()).limits(
        Limits(max_statevector_qubits=2)
    )
    plan = request.plan(Query.run())
    assert isinstance(plan.would_fail, PrismError)
    assert plan.would_fail.kind == "resource_limit"
    with pytest.raises(PrismError):
        request.plan(Query.reduced_density_matrix([9]))


def test_gradient_query_has_nested_evaluation_reports():
    circuit, params = CircuitBuilder(1).ry(0.3, 0).param(0).build_parametric()
    query = Query.gradient_shift([(1.0, [(0, "Z")])], params)
    plan = simulate(circuit).plan(query)
    assert plan.rule == "parameter_shift"
    assert len(plan.evaluations) == 3
    assert all(evaluation.would_fail is None for evaluation in plan.evaluations)
    assert plan.peak_memory["kind"] == "unknown"


def test_gradient_query_takes_parameters_not_links():
    _, params = CircuitBuilder(1).ry(0.3, 0).param(0).build_parametric()
    for build in (Query.gradient, Query.gradient_shift):
        with pytest.raises(TypeError):
            build([(1.0, [(0, "Z")])], params.links())


def test_conditional_fallback_error_does_not_reject_primary():
    builder = CircuitBuilder(18)
    for qubit in range(18):
        builder.h(qubit)
    for qubit in range(1, 18):
        builder.cx(qubit - 1, qubit)
    circuit = builder.t(0).build()
    request = simulate(circuit).limits(
        Limits(max_statevector_qubits=20, max_tn_peak_qubits=0, max_tn_slices=1)
    ).require_exact()
    plan = request.plan(Query.marginals())
    assert plan.would_fail is None
    assert plan.rule == "spd_exact"
    fallback = plan.fallbacks[0]
    assert isinstance(fallback["would_fail"], PrismError)
    assert fallback["exactness"] == "exact"
    assert fallback["bound_reported"] is False
    assert fallback["samples"] is None


def test_expectation_query_takes_pauli_strings():
    circuit = CircuitBuilder(2).h(0).cx(0, 1).build()
    request = simulate(circuit).seed(42)
    by_string = request.plan(Query.expectation(["ZZ", "XI"]))
    by_factors = request.plan(Query.expectation([[(0, "Z"), (1, "Z")], [(0, "X")]]))
    assert by_string.backend == by_factors.backend
    assert by_string.rule == by_factors.rule


def test_plan_reports_a_pauli_string_wider_than_the_circuit():
    from prism_q import PauliObservable

    circuit, params = CircuitBuilder(2).ry(0.3, 0).param(0).cx(0, 1).build_parametric()
    request = simulate(circuit).seed(42)
    cases = [
        (Query.expectation(["ZIII"]), lambda: request.expectation(["ZIII"])),
        (
            Query.observable(PauliObservable([(1.0, "ZII")])),
            lambda: request.expectation(PauliObservable([(1.0, "ZII")])),
        ),
        (
            Query.gradient([(1.0, "ZIZ")], params),
            lambda: request.expectation_gradient([(1.0, "ZIZ")], params),
        ),
        (
            Query.gradient_shift([(1.0, "ZIZ")], params),
            lambda: request.expectation_gradient_shift([(1.0, "ZIZ")], params),
        ),
    ]
    for query, run in cases:
        failure = request.plan(query).would_fail
        assert isinstance(failure, PrismError)
        with pytest.raises(PrismError) as raised:
            run()
        assert str(failure) == str(raised.value)
    assert request.plan(Query.expectation(["ZI"])).would_fail is None

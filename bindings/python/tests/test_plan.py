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
    circuit = CircuitBuilder(1).ry(0.3, 0).build()
    query = Query.gradient_shift([(1.0, [(0, "Z")])], [(0, 0)])
    plan = simulate(circuit).plan(query)
    assert plan.rule == "parameter_shift"
    assert len(plan.evaluations) == 3
    assert all(evaluation.would_fail is None for evaluation in plan.evaluations)
    assert plan.peak_memory["kind"] == "unknown"


def test_gradient_query_rejects_overflowing_parameter_slot():
    import struct

    maximum = (1 << (8 * struct.calcsize("P"))) - 1
    with pytest.raises(PrismError):
        Query.gradient([(1.0, [(0, "Z")])], [(0, maximum)])


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

import pytest

import prism_q
from prism_q import BackendKind, CircuitBuilder, simulate


def bell():
    return CircuitBuilder(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1).build()


def entangling_brickwork(num_qubits, layers):
    builder = CircuitBuilder(num_qubits)
    for q in range(num_qubits):
        builder = builder.h(q)
    for layer in range(layers):
        for q in range(layer % 2, num_qubits - 1, 2):
            builder = builder.cx(q, q + 1)
        for q in range(num_qubits):
            builder = builder.t(q)
            builder = builder.h(q)
    return builder.build()


def test_run_reports_an_exact_backend():
    result = simulate(bell()).seed(42).run()
    assert result.metadata.is_exact
    assert result.metadata.fidelity_lower_bound is None
    assert result.metadata.placement in ("host", "device")
    # A Bell pair is Clifford, so automatic dispatch takes the tableau.
    assert result.metadata.backend == "Stabilizer"


def test_shots_and_counts_carry_metadata():
    shots = simulate(bell()).seed(42).shots(64)
    assert shots.metadata.shots == 64
    assert shots.metadata.is_exact

    counts = simulate(bell()).seed(42).sample_counts(64)
    assert counts.metadata.is_exact


# The flag marks the route and the bound reports the run, so a bond the circuit
# never fills reports approximate at fidelity 1.
def test_mps_reports_a_fidelity_bound():
    roomy = (
        simulate(entangling_brickwork(6, 1))
        .backend(BackendKind.mps(64))
        .seed(42)
        .run()
    )
    assert not roomy.metadata.is_exact
    assert roomy.metadata.fidelity_lower_bound == pytest.approx(1.0)

    clamped = (
        simulate(entangling_brickwork(10, 6))
        .backend(BackendKind.mps(2))
        .seed(42)
        .run()
    )
    assert clamped.metadata.fidelity_lower_bound < 1.0
    assert clamped.metadata.backend == "Mps"


def test_require_exact_rejects_an_approximate_backend():
    circuit = entangling_brickwork(6, 2)
    assert simulate(circuit).backend(BackendKind.mps(8)).seed(42).run() is not None
    with pytest.raises(prism_q.PrismError):
        simulate(circuit).backend(BackendKind.mps(8)).seed(42).require_exact().run()


def test_require_exact_leaves_an_exact_route_alone():
    result = simulate(bell()).seed(42).require_exact().run()
    assert result.metadata.is_exact


def test_metadata_repr():
    text = repr(simulate(bell()).seed(42).run().metadata)
    assert "RunMetadata(" in text
    assert "exact" in text

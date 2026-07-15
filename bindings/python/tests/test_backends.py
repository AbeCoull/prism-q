import math

import numpy as np
import pytest

from prism_q import BackendKind, CircuitBuilder, circuits, simulate


def _bell():
    return CircuitBuilder(2).h(0).cx(0, 1).build()


def test_explicit_statevector_backend_matches_auto():
    bell = _bell()
    auto = simulate(bell).seed(1).run().probabilities
    sv = simulate(bell).backend(BackendKind.statevector()).seed(1).run().probabilities
    np.testing.assert_allclose(auto, sv, atol=1e-12)


def test_stabilizer_backend_on_clifford():
    ghz = circuits.ghz(4)
    probs = simulate(ghz).backend(BackendKind.stabilizer()).seed(1).run().probabilities
    assert math.isclose(probs[0], 0.5, abs_tol=1e-9)
    assert math.isclose(probs[-1], 0.5, abs_tol=1e-9)


def test_mps_backend_with_bond_dim():
    bell = _bell()
    probs = simulate(bell).backend(BackendKind.mps(max_bond_dim=8)).seed(1).run().probabilities
    assert math.isclose(probs[0], 0.5, abs_tol=1e-9)
    assert math.isclose(probs[3], 0.5, abs_tol=1e-9)


def test_backendkind_repr_includes_params():
    assert "256" in repr(BackendKind.mps(256))
    assert "Auto" in repr(BackendKind.auto())


def test_sparse_backend_runs():
    ghz = circuits.ghz(3)
    probs = simulate(ghz).backend(BackendKind.sparse()).seed(1).run().probabilities
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)


@pytest.mark.parametrize(
    "backend",
    [
        BackendKind.stochastic_pauli(4000),
        BackendKind.deterministic_pauli(0.0, 4096),
    ],
)
def test_pauli_backends_return_valid_marginals(backend):
    circuit = CircuitBuilder(2).h(0).t(0).cx(0, 1).build()
    marginals = simulate(circuit).backend(backend).seed(1).marginals()
    assert len(marginals) == 2
    for p0, p1 in marginals:
        assert 0.0 <= p0 <= 1.0
        assert math.isclose(p0 + p1, 1.0, abs_tol=1e-6)

import math

import numpy as np
import pytest

from prism_q import BackendKind, CircuitBuilder, PrismError, StabilizerBackend, circuits, simulate


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
        BackendKind.deterministic_pauli_budget(4096),
    ],
)
def test_pauli_backends_return_valid_marginals(backend):
    circuit = CircuitBuilder(2).h(0).t(0).cx(0, 1).build()
    marginals = simulate(circuit).backend(backend).seed(1).marginals()
    assert len(marginals) == 2
    for p0, p1 in marginals:
        assert 0.0 <= p0 <= 1.0
        assert math.isclose(p0 + p1, 1.0, abs_tol=1e-6)


def _clifford_halves(n, bits):
    prefix = CircuitBuilder(n, bits)
    for q in range(n):
        prefix.h(q)
    for q in range(n - 1):
        prefix.cx(q, q + 1)
    suffix = CircuitBuilder(n, bits)
    for q in range(bits):
        suffix.h(q).measure(q, q)
    return prefix.build(), suffix.build()


def test_stabilizer_tableau_round_trip_resumes_the_run():
    prefix, suffix = _clifford_halves(6, 4)

    whole = StabilizerBackend(seed=7)
    whole.run(prefix)
    expected = whole.apply(suffix)

    prep = StabilizerBackend(seed=7)
    prep.run(prefix)
    words, phases = prep.export_tableau()
    assert words.dtype == np.uint64
    assert phases.dtype == np.bool_

    resumed = StabilizerBackend(seed=7)
    resumed.import_tableau(6, words, phases, num_classical_bits=4)
    assert resumed.num_qubits == 6
    assert resumed.apply(suffix) == expected


def test_stabilizer_import_rejects_a_short_word_array():
    prep = StabilizerBackend(seed=7)
    prep.run(circuits.ghz(4))
    words, phases = prep.export_tableau()
    with pytest.raises(PrismError) as excinfo:
        prep.import_tableau(4, words[:-1], phases)
    assert excinfo.value.kind == "invalid_parameter"

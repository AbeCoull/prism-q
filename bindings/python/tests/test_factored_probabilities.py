import numpy as np
import pytest

from prism_q import BackendKind, Limits, PrismError, circuits, simulate


def test_wide_factored_result_needs_no_dense_vector():
    """A 30-qubit dense vector is 8 GB. Reading blocks must not build one."""
    outcome = simulate(circuits.independent_bell_pairs(15)).seed(42).run()

    assert outcome.num_basis_states == 2**30
    blocks = outcome.probabilities_factored()
    assert blocks is not None, "expected a factored result, got a dense one"

    assert len(blocks) == 15
    for qubits, probs in blocks:
        assert len(qubits) == 2
        assert qubits == sorted(qubits)
        assert probs.dtype == np.float64
        assert probs.shape == (4,)
        # A Bell pair: |00> and |11> at one half each.
        np.testing.assert_allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-12)

    covered = sorted(q for qubits, _ in blocks for q in qubits)
    assert covered == list(range(30))


def test_blocks_multiply_out_to_the_dense_distribution():
    outcome = simulate(circuits.independent_bell_pairs(4)).seed(42).run()
    blocks = outcome.probabilities_factored()
    assert blocks is not None

    joint = np.zeros(outcome.num_basis_states)
    for index in range(len(joint)):
        p = 1.0
        for qubits, probs in blocks:
            local = 0
            for bit, qubit in enumerate(qubits):
                local |= ((index >> qubit) & 1) << bit
            p *= probs[local]
        joint[index] = p

    np.testing.assert_allclose(joint, outcome.probabilities, atol=1e-12)
    assert abs(joint.sum() - 1.0) < 1e-12


def test_a_dense_result_reports_no_blocks():
    outcome = (
        simulate(circuits.ghz(4)).backend(BackendKind.statevector()).seed(42).run()
    )
    assert outcome.probabilities_factored() is None
    assert outcome.num_basis_states == 16
    assert outcome.probabilities is not None


def test_num_basis_states_is_none_without_a_distribution():
    # A stabilizer run at a width past the dense cap exposes no distribution.
    outcome = simulate(circuits.ghz(200)).backend(BackendKind.stabilizer()).seed(42).run()
    assert outcome.probabilities is None
    assert outcome.probabilities_factored() is None
    assert outcome.num_basis_states is None


def test_repr_names_the_form():
    factored = simulate(circuits.independent_bell_pairs(4)).seed(42).run()
    assert "factored" in repr(factored)
    dense = simulate(circuits.ghz(4)).backend(BackendKind.statevector()).seed(42).run()
    assert "dense" in repr(dense)


@pytest.mark.parametrize("num_pairs", [4, 8, 15])
def test_block_probabilities_sum_to_one(num_pairs):
    outcome = simulate(circuits.independent_bell_pairs(num_pairs)).seed(42).run()
    blocks = outcome.probabilities_factored()
    assert blocks is not None
    for _, probs in blocks:
        assert abs(float(probs.sum()) - 1.0) < 1e-12


def test_a_narrow_independent_circuit_still_answers_dense():
    """Decomposition needs the widest block several qubits below the register,
    so two Bell pairs stay dense and `probabilities_factored` says so."""
    outcome = simulate(circuits.independent_bell_pairs(2)).seed(42).run()
    assert outcome.probabilities_factored() is None
    assert outcome.num_basis_states == 16
    np.testing.assert_allclose(float(outcome.probabilities.sum()), 1.0, atol=1e-12)


def test_factored_dense_probabilities_respects_probability_limit():
    """A factored run stays usable when `.probabilities` is over `max_probability_qubits`."""
    outcome = (
        simulate(circuits.independent_bell_pairs(5))
        .limits(Limits(max_probability_qubits=4))
        .seed(42)
        .run()
    )
    assert outcome.num_basis_states == 2**10
    assert outcome.probabilities_factored() is not None
    with pytest.raises(PrismError) as exc_info:
        _ = outcome.probabilities
    assert exc_info.value.kind == "resource_limit"


def test_circuit_builder_validation_raises_prism_error():
    with pytest.raises(PrismError):
        circuits.qft(0)
    with pytest.raises(PrismError):
        circuits.qft(256)
    with pytest.raises(PrismError):
        circuits.phase_estimation(1)
    with pytest.raises(PrismError):
        circuits.clifford_t(2, 1, t_fraction=float("nan"))
    with pytest.raises(PrismError):
        circuits.clifford_t(2, 1, t_fraction=1.5)
    assert circuits.random(0, 2, seed=42).num_qubits == 0
    assert circuits.hardware_efficient_ansatz(0, 2, seed=42).num_qubits == 0
    assert circuits.clifford_heavy(0, 2, seed=42).num_qubits == 0
    assert circuits.qaoa(0, 2, seed=42).num_qubits == 0

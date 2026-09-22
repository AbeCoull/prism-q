import math

import numpy as np
import pytest

import prism_q
from prism_q import BackendKind, CircuitBuilder, NoiseChannel, NoiseModel, simulate

SEED = 42
DM_EPS = 1e-12


def test_density_matrix_backend_matches_statevector():
    circuit = CircuitBuilder(3).h(0).ry(0.7, 1).cx(0, 1).rx(1.1, 2).cz(1, 2).t(0).build()
    sv = simulate(circuit).seed(SEED).run().probabilities
    dm = simulate(circuit).backend(BackendKind.density_matrix()).seed(SEED).run().probabilities
    assert dm is not None
    for i, (a, b) in enumerate(zip(dm, sv)):
        assert abs(a - b) < DM_EPS, f"basis state {i}: dm {a} vs statevector {b}"


def test_noiseless_expectation_matches_statevector():
    circuit = CircuitBuilder(3).h(0).ry(0.7, 1).cx(0, 1).rx(1.1, 2).cz(1, 2).t(0).build()
    observables = [
        [(0, "Z")],
        [(2, "X")],
        [(0, "Z"), (1, "Z")],
        [(0, "Y"), (2, "X")],
    ]
    reference = simulate(circuit).seed(SEED).expectation_values(observables)
    values = simulate(circuit).seed(SEED).density_matrix_expectation_values(observables)
    assert len(values) == len(observables)
    for i, (a, b) in enumerate(zip(values, reference)):
        assert abs(a - b) < DM_EPS, f"observable {i}: dm {a} vs statevector {b}"


def test_amplitude_damping_expectation_is_analytic():
    # |1> relaxes to |0> with probability gamma, so <Z> = gamma - (1 - gamma).
    gamma = 0.3
    circuit = CircuitBuilder(1).x(0).build()
    model = NoiseModel.empty(circuit)
    model.add_event(0, NoiseChannel.amplitude_damping(gamma), [0])
    (value,) = (
        simulate(circuit).seed(SEED).noise(model).density_matrix_expectation_values([[(0, "Z")]])
    )
    assert math.isclose(value, 2.0 * gamma - 1.0, abs_tol=DM_EPS)


def test_thermal_relaxation_walks_toward_its_excited_population():
    # <Z> relaxes geometrically toward the steady state 1 - 2 p, so after n
    # intervals it sits at (1 - 2 p) + 2 p (1 - gamma)^n. Forty intervals at
    # gate_time / t1 = 0.25 leave exp(-10), about 3.6e-5 short of the limit,
    # which is why this pins the closed form rather than the limit alone.
    excited = 0.4
    steps = 40
    t1, t2, gate_time = 100.0, 80.0, 25.0
    builder = CircuitBuilder(1)
    for _ in range(steps):
        builder = builder.id(0)
    circuit = builder.build()
    model = NoiseModel.empty(circuit)
    for index in range(steps):
        model.add_event(
            index, NoiseChannel.thermal_relaxation(t1, t2, gate_time, excited), [0]
        )
    (value,) = (
        simulate(circuit).seed(SEED).noise(model).density_matrix_expectation_values([[(0, "Z")]])
    )
    gamma = 1.0 - math.exp(-gate_time / t1)
    expected = (1.0 - 2.0 * excited) + 2.0 * excited * (1.0 - gamma) ** steps
    assert math.isclose(value, expected, abs_tol=1e-9)


def test_depolarizing_expectation_is_analytic():
    # Depolarizing applies X, Y, Z each with p/3; on |+> only Y and Z flip the
    # sign of <X>, giving 1 - 4p/3.
    p = 0.12
    circuit = CircuitBuilder(1).h(0).build()
    model = NoiseModel.empty(circuit)
    model.add_event(0, NoiseChannel.depolarizing(p), [0])
    (value,) = (
        simulate(circuit).seed(SEED).noise(model).density_matrix_expectation_values([[(0, "X")]])
    )
    assert math.isclose(value, 1.0 - 4.0 * p / 3.0, abs_tol=DM_EPS)


def test_bell_parity_under_depolarizing_is_analytic():
    # <Z0 Z1> is 1 on the Bell state; X and Y on qubit 0 flip it, Z does not.
    p = 0.09
    circuit = CircuitBuilder(2).h(0).cx(0, 1).build()
    model = NoiseModel.empty(circuit)
    model.add_event(1, NoiseChannel.depolarizing(p), [0])
    (value,) = (
        simulate(circuit)
        .seed(SEED)
        .noise(model)
        .density_matrix_expectation_values([[(0, "Z"), (1, "Z")]])
    )
    assert math.isclose(value, 1.0 - 4.0 * p / 3.0, abs_tol=DM_EPS)


def test_oversize_circuit_reports_the_qubit_cap():
    circuit = CircuitBuilder(40).h(0).build()
    with pytest.raises(prism_q.PrismError) as excinfo:
        simulate(circuit).seed(SEED).density_matrix_expectation_values([[(0, "Z")]])
    message = str(excinfo.value)
    assert "density_matrix" in message
    assert "exceeding the cap" in message
    # Both variables bind: the 4^n state is allocated as a 2n-qubit statevector.
    assert "PRISM_MAX_DM_QUBITS and PRISM_MAX_SV_QUBITS" in message


def test_noisy_shots_sample_the_exact_distribution():
    circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
    model = NoiseModel.uniform_depolarizing(circuit, 0.01)
    sim = simulate(circuit).backend(BackendKind.density_matrix()).seed(SEED).noise(model)
    counts = sim.shots(4000).counts()
    assert sum(counts.values()) == 4000
    assert set(counts) <= {"00", "01", "10", "11"}
    # Light depolarizing leaks a little weight onto the odd-parity outcomes and
    # leaves the Bell pair dominant.
    assert counts["00"] + counts["11"] > 3 * (counts.get("01", 0) + counts.get("10", 0))


def test_noisy_run_probabilities_are_seed_independent():
    circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
    model = NoiseModel.uniform_depolarizing(circuit, 0.05)
    runs = [
        simulate(circuit)
        .backend(BackendKind.density_matrix())
        .seed(seed)
        .noise(model)
        .run()
        .probabilities
        for seed in (SEED, SEED + 1, SEED + 7)
    ]
    for other in runs[1:]:
        for i, (a, b) in enumerate(zip(other, runs[0])):
            assert abs(a - b) < DM_EPS, f"basis state {i}: {a} vs {b}"


def test_mismatched_noise_model_is_rejected():
    circuit = CircuitBuilder(1).h(0).build()
    other = CircuitBuilder(1).h(0).t(0).build()
    model = NoiseModel.uniform_depolarizing(other, 0.01)
    with pytest.raises(prism_q.PrismError) as excinfo:
        simulate(circuit).seed(SEED).noise(model).density_matrix_expectation_values([[(0, "Z")]])
    assert "2 event slots for 1 instructions" in str(excinfo.value)


def test_expectation_values_rejects_noise_without_a_mixture():
    circuit = CircuitBuilder(1).h(0).build()
    model = NoiseModel.uniform_depolarizing(circuit, 0.01)
    with pytest.raises(prism_q.PrismError):
        simulate(circuit).seed(SEED).noise(model).expectation_values([[(0, "X")]])


def test_unknown_pauli_axis_is_rejected():
    circuit = CircuitBuilder(1).h(0).build()
    with pytest.raises(prism_q.PrismError):
        simulate(circuit).seed(SEED).density_matrix_expectation_values([[(0, "W")]])


def test_initial_density_matrix_continues_an_exported_mixture():
    first = CircuitBuilder(3).h(0).cx(0, 1).cx(1, 2).build()
    second = CircuitBuilder(3).t(0).ry(0.4, 1).cx(2, 0).build()
    whole = CircuitBuilder(3).h(0).cx(0, 1).cx(1, 2).t(0).ry(0.4, 1).cx(2, 0).build()
    dm = BackendKind.density_matrix()
    rho = (
        simulate(first)
        .backend(dm)
        .noise(NoiseModel.uniform_depolarizing(first, 0.02))
        .seed(SEED)
        .reduced_density_matrix([0, 1, 2])
        .matrix
    )
    resumed = (
        simulate(second)
        .backend(dm)
        .noise(NoiseModel.uniform_depolarizing(second, 0.02))
        .initial_density_matrix(rho)
        .seed(SEED)
        .run()
        .probabilities
    )
    exact = (
        simulate(whole)
        .backend(dm)
        .noise(NoiseModel.uniform_depolarizing(whole, 0.02))
        .seed(SEED)
        .run()
        .probabilities
    )
    assert np.allclose(resumed, exact, atol=DM_EPS)


def test_initial_density_matrix_validation_and_backend_limits():
    circuit = CircuitBuilder(1).x(0).build()
    dm = BackendKind.density_matrix()
    mixed = [[0.75, 0.0], [0.0, 0.25]]
    probs = simulate(circuit).backend(dm).initial_density_matrix(mixed).seed(SEED).run().probabilities
    assert abs(probs[0] - 0.25) < DM_EPS and abs(probs[1] - 0.75) < DM_EPS
    with pytest.raises(prism_q.PrismError, match="Hermitian"):
        simulate(circuit).backend(dm).initial_density_matrix([[0.5, 0.25j], [0.25j, 0.5]]).run()
    with pytest.raises(prism_q.PrismError, match="trace"):
        simulate(circuit).backend(dm).initial_density_matrix([[0.6, 0.0], [0.0, 0.6]]).run()
    with pytest.raises(prism_q.PrismError, match="square"):
        simulate(circuit).backend(dm).initial_density_matrix(np.zeros((1, 4), dtype=complex)).run()
    with pytest.raises(prism_q.PrismError, match="Statevector"):
        simulate(circuit).backend(BackendKind.statevector()).initial_density_matrix(mixed).run()

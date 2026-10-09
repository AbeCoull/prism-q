import math

import numpy as np
import pytest

from prism_q import (
    BackendKind,
    CircuitBuilder,
    DeviceCalibration,
    DriftDistribution,
    GateFilter,
    GateTimes,
    NoiseBuilder,
    NoiseChannel,
    NoiseModel,
    PrismError,
    QecNoise,
    QecProgram,
    simulate,
)


def _bell():
    return CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()


def test_leaked_qubit_reads_one_and_heralds():
    circuit = CircuitBuilder(2, 2).x(0).cx(0, 1).x(0).measure_all().build()
    model = NoiseModel.empty(circuit)
    model.add_event(0, NoiseChannel.leakage(1.0), [0])
    result = simulate(circuit).backend(BackendKind.statevector()).seed(42).noise(model).shots(4000)
    shots = result.shots
    assert shots[:, 0].all()
    assert abs(shots[:, 1].mean() - 0.5) < 0.05
    leaked = result.leaked
    assert leaked.shape == (4000, 2)
    assert leaked[:, 0].all() and not leaked[:, 1].any()


def test_models_without_leakage_have_no_herald():
    circuit = _bell()
    result = simulate(circuit).seed(42).noise(NoiseModel.uniform_depolarizing(circuit, 0.01)).shots(16)
    assert result.leaked is None


def test_leakage_channels_build_through_rules():
    circuit = _bell()
    model = (
        NoiseBuilder()
        .after_gates_joint(GateFilter.all().arity(2), NoiseChannel.leakage_transport(0.2))
        .after_gates(GateFilter.all(), NoiseChannel.leakage(0.05))
        .after_gates(GateFilter.all(), NoiseChannel.seepage(0.1))
        .build(circuit)
    )
    model.validate()
    assert not model.is_pauli_only()
    with pytest.raises(PrismError):
        simulate(circuit).backend(BackendKind.density_matrix()).seed(1).noise(model).shots(8)


def test_detuning_ramsey_follows_the_gaussian_envelope():
    sigma, delay, n = 2.0e5, 5.0e-6, 20000
    circuit = CircuitBuilder(1, 1).h(0).id(0).h(0).measure_all().build()
    model = (
        NoiseBuilder()
        .schedule(GateTimes(0.0, 0.0).with_gate("id", delay))
        .quasi_static_detuning(DriftDistribution.independent([sigma]))
        .build(circuit)
    )
    p1 = simulate(circuit).seed(42).noise(model).shots(n).shots[:, 0].mean()
    expected = (1.0 - math.exp(-((sigma * delay) ** 2) / 2.0)) / 2.0
    assert abs(p1 - expected) < 5.0 * math.sqrt(expected * (1 - expected) / n)


def test_drift_distribution_shapes_and_validation():
    drift = DriftDistribution.independent([1.0, 2.0]).with_neighbour_correlation([(0, 1)], 0.5)
    assert drift.num_qubits == 2
    assert np.allclose(drift.covariance(), [[1.0, 1.0], [1.0, 4.0]])
    assert np.allclose(
        DriftDistribution.from_t2_star([1.0]).covariance(), [[2.0]]
    )
    with pytest.raises(PrismError):
        drift.with_neighbour_correlation([(0, 5)], 0.1)
    bad = DriftDistribution.from_covariance([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(PrismError):
        NoiseBuilder().schedule(GateTimes(1e-8, 1e-7)).quasi_static_detuning(bad).build(_bell())
    with pytest.raises(PrismError):
        NoiseBuilder().quasi_static_detuning(drift).build(_bell())


def test_amplitude_drift_and_quasi_static_channel():
    circuit = CircuitBuilder(1, 1).rx(math.pi, 0).measure_all().build()
    model = NoiseBuilder().over_rotation_drift(GateFilter.all().named("rx"), 0.2).build(circuit)
    p0 = 1.0 - simulate(circuit).seed(42).noise(model).shots(20000).shots[:, 0].mean()
    x = math.pi * 0.2
    assert abs(p0 - (1.0 - math.exp(-x * x / 2.0)) / 2.0) < 0.02
    channel = NoiseChannel.quasi_static("x", [(0, 0.1)])
    assert "QuasiStatic" in repr(channel)
    with pytest.raises(PrismError):
        NoiseChannel.quasi_static("w", [(0, 0.1)])


def test_scheduled_calibration_lowering():
    builder = CircuitBuilder(3, 1).x(0)
    for _ in range(40):
        builder = builder.cx(1, 2)
    circuit = builder.barrier([0, 1, 2]).measure(0, 0).build()
    calibration = DeviceCalibration.superconducting_transmon(3)
    assert isinstance(calibration.gate_times(), GateTimes)

    def p1(model):
        run = simulate(circuit).backend(BackendKind.density_matrix()).noise(model).seed(1)
        return run.shots(20000).shots[:, 0].mean()

    plain = p1(calibration.to_noise_model(circuit))
    scheduled = p1(calibration.to_scheduled_noise_model(circuit))
    assert scheduled < plain - 0.05


def test_qec_leakage_heralds():
    qp = QecProgram.from_text("R 0 1\nLEAK(1) 0\nCX 0 1\nM 0 1")
    qp.set_options(shots=2000, seed=42)
    result = qp.run()
    heralds = result.heralds
    assert heralds.shape == (2000, 1)
    assert heralds.all()
    assert result.measurements[:, 0].all()
    assert abs(result.measurements[:, 1].mean() - 0.5) < 0.06

    qp = QecProgram(2)
    qp.noise(QecNoise.leak(0.5), [0, 1])
    qp.noise(QecNoise.seep(0.1), [0])
    qp.noise(QecNoise.leak_transport(0.1), [0, 1])
    qp.measure_z(0)
    qp.set_options(shots=64, seed=1)
    assert qp.run().heralds.shape == (64, 2)
    assert QecProgram.from_text("R 0\nM 0").run().heralds is None

import pytest

import prism_q
from prism_q import (
    CircuitBuilder,
    DeviceCalibration,
    NoiseChannel,
    NoiseModel,
    parse_qasm,
    simulate,
)

BELL_QASM = """
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


def test_uniform_depolarizing_introduces_errors():
    circuit = parse_qasm(BELL_QASM)
    model = NoiseModel.uniform_depolarizing(circuit, 0.1)
    counts = simulate(circuit).seed(2).noise(model).shots(4000).counts()
    assert set(counts) == {"00", "01", "10", "11"}
    leaked = counts.get("01", 0) + counts.get("10", 0)
    assert leaked > 0


def test_empty_model_add_event_and_validate():
    circuit = parse_qasm(BELL_QASM)
    model = NoiseModel.empty(circuit)
    model.add_event(0, NoiseChannel.depolarizing(0.05), [0])
    model.validate()
    assert model.is_pauli_only()
    counts = simulate(circuit).seed(1).noise(model).shots(2000).counts()
    assert sum(counts.values()) == 2000


def test_amplitude_damping_is_not_pauli_only():
    circuit = parse_qasm(BELL_QASM)
    model = NoiseModel.with_amplitude_damping(circuit, 0.1)
    assert not model.is_pauli_only()
    counts = simulate(circuit).seed(1).noise(model).shots(1000).counts()
    assert sum(counts.values()) == 1000


def test_device_calibration_parses_and_lowers():
    circuit = parse_qasm(BELL_QASM)
    calibration = DeviceCalibration.parse(
        "qubit 0 t1=120e-6 t2=80e-6 p01=0.02 p10=0.03\n"
        "qubit 1 t1=95e-6 t2=110e-6\n"
        "gate1q time=35e-9 error=3e-4\n"
        "gate2q time=300e-9 error=8e-3\n"
    )
    assert calibration.num_qubits == 2
    model = calibration.to_noise_model(circuit)
    model.validate()
    assert not model.is_pauli_only()
    counts = simulate(circuit).seed(1).noise(model).shots(1000).counts()
    assert sum(counts.values()) == 1000

    preset = DeviceCalibration.superconducting_transmon(2)
    preset.to_noise_model(circuit).validate()
    with pytest.raises(prism_q.PrismError) as info:
        DeviceCalibration.parse("qubit 0 t1=1e-4 t2=5e-4\n")
    assert "line 1" in str(info.value)


def test_readout_error_applies():
    circuit = parse_qasm(BELL_QASM)
    model = NoiseModel.empty(circuit)
    model.with_readout_error(0.1, 0.1)
    counts = simulate(circuit).seed(1).noise(model).shots(2000).counts()
    assert sum(counts.values()) == 2000


@pytest.mark.parametrize("terminal", ["run", "marginals", "state_vector"])
def test_noise_rejected_on_backends_without_a_mixture(terminal):
    # Auto dispatch never selects the density matrix, so no mixture exists here.
    bell = CircuitBuilder(2).h(0).cx(0, 1).build()
    model = NoiseModel.uniform_depolarizing(bell, 0.01)
    sim = simulate(bell).noise(model)
    with pytest.raises(prism_q.PrismError):
        getattr(sim, terminal)()


def test_custom_kraus_channel_runs():
    circuit = parse_qasm(BELL_QASM)
    identity = [[1.0, 0.0], [0.0, 1.0]]
    channel = NoiseChannel.custom([identity])
    model = NoiseModel.empty(circuit)
    model.add_event(0, channel, [0])
    model.validate()
    assert not model.is_pauli_only()
    counts = simulate(circuit).seed(1).noise(model).shots(1000).counts()
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 1000


def test_custom_kraus_requires_operator():
    with pytest.raises(prism_q.PrismError):
        NoiseChannel.custom([])


def test_add_event_out_of_range_raises():
    circuit = parse_qasm(BELL_QASM)
    model = NoiseModel.empty(circuit)
    with pytest.raises(prism_q.PrismError):
        model.add_event(9999, NoiseChannel.depolarizing(0.05), [0])

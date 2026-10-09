import copy
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

import prism_q
from prism_q import (
    ClassicalCondition,
    CircuitBuilder,
    Gate,
    NoiseChannel,
    NoiseModel,
    Parameters,
    PauliObservable,
    QecBasis,
    QecNoise,
    QecProgram,
    QecRecordRef,
    SaveSpec,
    circuits,
    simulate,
)


def _round_trip(value):
    """Pickle `value` and back, checking the second pickle is byte-identical.

    The payload carries every field with floats as IEEE bits, so equal bytes
    mean an identical value rather than a close one.
    """
    data = pickle.dumps(value)
    restored = pickle.loads(data)
    assert type(restored) is type(value)
    assert pickle.dumps(restored) == data
    return restored


def _kitchen_sink():
    permutation = np.eye(8)[[1, 0, 2, 3, 4, 5, 7, 6]]
    builder = (
        CircuitBuilder(4, 4)
        .h(0)
        .rx(0.1, 1)
        .param(0)
        .pauli_rotation(1 / 3, [(2, "X"), (0, "Y"), (3, "Z")])
        .param(1)
        .cu([[0, 1], [1, 0]], 0, 1)
        .mcu([[1, 0], [0, 1j]], [0, 1], 2)
        .gate(Gate.unitary(permutation), [0, 1, 3])
        .gate(Gate.unitary([[0, 1j], [1j, 0]]), [2])
        .measure(0, 0)
    )
    builder.guarded(
        ClassicalCondition.parity([0], False),
        lambda then: then.x(1).measure(1, 1).reset(1),
        lambda other: other.z(1),
    )
    builder.conditional(ClassicalCondition.register_equals(0, 2, 1), Gate.y(), [2])
    builder.barrier([0, 3]).measure_all()
    circuit = builder.build()
    circuit.add_save(SaveSpec.Probabilities, "after")
    return builder, circuit


def test_circuit_round_trip_is_lossless():
    _, circuit = _kitchen_sink()
    restored = _round_trip(circuit)
    assert restored.num_qubits == circuit.num_qubits
    assert restored.num_classical_bits == circuit.num_classical_bits
    assert restored.gate_count() == circuit.gate_count()
    assert str(restored) == str(circuit)
    assert restored.measurement_map() == circuit.measurement_map()
    assert pickle.dumps(copy.deepcopy(circuit)) == pickle.dumps(circuit)


def test_restored_circuits_simulate_identically():
    for circuit in (
        circuits.qft(5),
        circuits.quantum_volume(5, 4, seed=3),
        circuits.hardware_efficient_ansatz(5, 2, seed=1),
        CircuitBuilder(3).h(0).pauli_rotation(0.123456789, [(0, "X"), (1, "Y"), (2, "Z")]).build(),
    ):
        restored = _round_trip(circuit)
        assert (simulate(restored).state_vector() == simulate(circuit).state_vector()).all()


def test_qasm_text_survives_a_round_trip_bit_for_bit():
    circuit = circuits.random(5, 6, seed=9)
    assert _round_trip(circuit).to_qasm() == circuit.to_qasm()


def test_parameters_keep_links_names_and_the_edit_guard():
    builder, circuit = _kitchen_sink()
    params = builder.parameters().with_names(["theta", "phi"])
    restored = _round_trip(params)
    assert restored.links() == params.links()
    assert restored.num_slots == 2
    assert restored.name_of(1) == "phi"
    assert restored.slot_of("theta") == 0
    template = _round_trip(circuit)
    assert str(restored.bind(template, [0.5, 0.25])) == str(params.bind(circuit, [0.5, 0.25]))

    edited = CircuitBuilder(4, 4).h(0).rz(0.1, 1).pauli_rotation(0.1, [(0, "X"), (1, "Y")]).build()
    for guarded in (params, restored):
        with pytest.raises(prism_q.PrismError, match="edited"):
            guarded.validate(edited)

    loose = Parameters(3)
    loose.link(1, 2)
    restored_loose = _round_trip(loose)
    assert restored_loose.links() == [(1, 2)]
    assert restored_loose.name_of(0) is None


def test_noise_round_trips():
    _, circuit = _kitchen_sink()
    model = NoiseModel.uniform_depolarizing(circuit, 0.01)
    model.add_event(0, NoiseChannel.custom_2q([np.eye(4)]), [0, 1])
    model.add_event(1, NoiseChannel.thermal_relaxation(50e-6, 70e-6, 35e-9, 0.01), [1])
    model.add_event(2, NoiseChannel.custom([np.eye(2)]), [2])
    model.with_readout_error(0.01, 0.02)
    restored = _round_trip(model)
    restored.validate()
    assert restored.is_pauli_only() == model.is_pauli_only()
    for channel in (
        NoiseChannel.pauli(0.1, 0.2, 0.3),
        NoiseChannel.depolarizing(0.1),
        NoiseChannel.amplitude_damping(0.2),
        NoiseChannel.phase_damping(0.3),
        NoiseChannel.two_qubit_depolarizing(0.05),
        NoiseChannel.leakage(0.01),
        NoiseChannel.seepage(0.2),
        NoiseChannel.leakage_transport(0.1),
        NoiseChannel.quasi_static("y", [(0, 0.5), (3, -1.25)]),
    ):
        assert repr(_round_trip(channel)) == repr(channel)


def test_noisy_counts_agree_after_a_round_trip():
    circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
    model = NoiseModel.uniform_depolarizing(circuit, 0.05)
    expected = simulate(circuit).seed(3).noise(model).sample_counts(500).counts()
    restored = simulate(_round_trip(circuit)).seed(3).noise(_round_trip(model))
    assert restored.sample_counts(500).counts() == expected


def test_qec_program_round_trip_keeps_options_and_ops():
    qp = QecProgram.from_text(
        "R 0 1 2\nX_ERROR(0.01) 0 2\nDEPOLARIZE2(0.002) 0 1\nCX 0 1 2 1\nTICK\nMR 1\n"
        "DETECTOR(1, 0.5) rec[-1]\nM 0 2\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    qp.postselect([QecRecordRef.lookback(1)], False)
    body = QecProgram(3)
    body.push_gate(Gate.x(), [2])
    qp.feedforward([QecRecordRef.absolute(0)], True, body)
    qp.set_options(shots=300, seed=8, chunk_size=128, keep_measurements=False)
    restored = _round_trip(qp)
    assert repr(restored) == repr(qp)
    original, replay = qp.run_reference(), restored.run_reference()
    assert (original.detectors == replay.detectors).all()
    assert original.accepted_shots == replay.accepted_shots


def test_small_values_round_trip():
    for value in (
        Gate.h(),
        Gate.rzz(0.25),
        Gate.cphase(0.5),
        Gate.mcu([[0, 1], [1, 0]], 3),
        Gate.unitary(np.eye(4)[[0, 1, 3, 2]]),
        ClassicalCondition.bit(2, False),
        ClassicalCondition.register_not_equals(1, 3, 5),
        QecRecordRef.lookback(3),
        QecRecordRef.absolute(4),
        QecNoise.depolarize2(0.01),
        QecNoise.y_error(0.02),
        QecNoise.pauli_channel_1(0.1, 0.2, 0.3),
        QecNoise.pauli_channel_2([0.001 * (k + 1) for k in range(15)]),
        QecNoise.leak(0.01),
        QecNoise.seep(0.2),
        QecNoise.leak_transport(0.1),
        PauliObservable([(0.5, [(0, "X"), (2, "Z")]), (-1.0, [])]),
    ):
        assert repr(_round_trip(value)) == repr(value)
    assert pickle.loads(pickle.dumps(QecBasis.Y)) == QecBasis.Y
    assert pickle.loads(pickle.dumps(SaveSpec.DensityMatrix)) == SaveSpec.DensityMatrix
    observable = PauliObservable([(0.5, [(1, "Y")]), (0.25, [(0, "Z")])])
    assert _round_trip(observable).terms() == observable.terms()


def test_corrupt_payloads_raise():
    with pytest.raises(prism_q.PrismError, match="version"):
        prism_q.Circuit._from_pickle(b"PQC\x02")
    with pytest.raises(prism_q.PrismError, match="kind"):
        prism_q.Circuit._from_pickle(pickle.loads(pickle.dumps(Gate.h().__reduce__()[1][0])))
    data = Gate.h().__reduce__()[1][0]
    with pytest.raises(prism_q.PrismError):
        Gate._from_pickle(data + b"\x00")
    _, circuit = _kitchen_sink()
    payload = circuit.__reduce__()[1][0]
    with pytest.raises(prism_q.PrismError):
        prism_q.Circuit._from_pickle(payload[:-5])


def _counts_in_a_worker(circuit):
    return simulate(circuit).seed(1).sample_counts(256).counts()


def test_circuits_cross_a_process_boundary():
    circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
    with ProcessPoolExecutor(max_workers=1) as pool:
        counts = pool.submit(_counts_in_a_worker, circuit).result(timeout=120)
    assert counts == _counts_in_a_worker(circuit)


def test_a_record_ref_pickled_under_its_old_name_still_loads():
    record = QecRecordRef.lookback(2)
    payload = pickle.dumps(record, protocol=0)
    assert b"QecRecordRef" in payload
    legacy = payload.replace(b"QecRecordRef", b"RecordRef")
    with pytest.warns(DeprecationWarning):
        restored = pickle.loads(legacy)
    assert pickle.dumps(restored, protocol=0) == payload

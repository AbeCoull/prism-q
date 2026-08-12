import numpy as np

import prism_q
from prism_q import QecBasis, QecNoise, QecProgram, RecordRef


def _repetition_round():
    qp = QecProgram(3)
    qp.set_options(shots=128, seed=42)
    for q in range(3):
        qp.reset(QecBasis.Z, q)
    qp.push_gate(prism_q.Gate.x(), [0])
    r0 = qp.measure_pauli_product([(QecBasis.Z, 0), (QecBasis.Z, 1)])
    r1 = qp.measure_pauli_product([(QecBasis.Z, 1), (QecBasis.Z, 2)])
    qp.detector([RecordRef.absolute(r0)])
    qp.detector_lookback([1])
    m0 = qp.measure_z(0)
    qp.observable_include(0, [RecordRef.absolute(m0)])
    return qp


def test_program_counts():
    qp = _repetition_round()
    assert qp.num_qubits == 3
    assert qp.num_measurements == 3
    assert qp.num_detectors == 2
    assert qp.num_observables == 1


def test_detectors_and_observables_are_bool_arrays():
    qp = _repetition_round()
    res = qp.run()
    det = res.detectors
    obs = res.observables
    assert det.dtype == np.bool_
    assert det.shape == (128, 2)
    assert obs.shape == (128, 1)
    assert det[:, 0].all()
    assert not det[:, 1].any()
    assert obs[:, 0].all()
    assert res.total_shots == 128
    assert res.accepted_shots == 128
    assert res.logical_error_rates() == [1.0]
    assert res.survivor_rate() == 1.0


def test_noise_randomizes_detector():
    qp = QecProgram(2)
    qp.set_options(shots=1024, seed=7)
    qp.reset(QecBasis.Z, 0)
    qp.reset(QecBasis.Z, 1)
    qp.noise(QecNoise.x_error(0.5), [1])
    rr = qp.measure_pauli_product([(QecBasis.Z, 0), (QecBasis.Z, 1)])
    qp.detector([RecordRef.absolute(rr)])
    res = qp.run()
    frac = res.detectors[:, 0].mean()
    assert 0.4 < frac < 0.6


def test_postselect_rejects_shots():
    qp = QecProgram(1)
    qp.set_options(shots=512, seed=1)
    qp.reset(QecBasis.Z, 0)
    qp.push_gate(prism_q.Gate.h(), [0])
    r = qp.measure_z(0)
    qp.postselect([RecordRef.absolute(r)], False)
    res = qp.run()
    assert res.accepted_shots + res.discarded_shots == res.total_shots == 512
    assert 0 < res.accepted_shots < 512


def test_from_text_parses():
    qp = QecProgram.from_text("R 0 1\nM 0\nM 1\nDETECTOR rec[-2]\n")
    assert qp.num_qubits == 2
    assert qp.num_measurements == 2
    assert qp.num_detectors == 1


def test_lookback_zero_raises():
    import pytest

    with pytest.raises(prism_q.PrismError):
        RecordRef.lookback(0)


def test_detector_error_model_matches_program():
    qp = QecProgram(3)
    qp.noise(QecNoise.x_error(0.05), [0, 1, 2])
    r0 = qp.measure_pauli_product([(QecBasis.Z, 0), (QecBasis.Z, 1)])
    r1 = qp.measure_pauli_product([(QecBasis.Z, 1), (QecBasis.Z, 2)])
    qp.detector([RecordRef.absolute(r0)], coords=[0.5, 0.0])
    qp.detector([RecordRef.absolute(r1)])
    m0 = qp.measure_z(0)
    qp.observable_include(0, [RecordRef.absolute(m0)])

    dem = qp.detector_error_model()
    assert dem.num_detectors == qp.num_detectors == 2
    assert dem.num_observables == qp.num_observables == 1
    # X on qubit 0 flips check 0 and the observable, X on qubit 1 both checks,
    # X on qubit 2 check 1.
    assert dem.num_mechanisms == 3
    probs = dem.probabilities()
    assert probs.dtype == np.float64
    assert np.allclose(probs, 0.05)
    det = dem.detector_matrix()
    obs = dem.observable_matrix()
    assert det.dtype == np.bool_ and det.shape == (2, 3)
    assert obs.shape == (1, 3)
    assert det.tolist() == [[True, True, False], [False, True, True]]
    assert obs.tolist() == [[True, False, False]]
    assert dem.detector_coords() == [[0.5, 0.0], []]
    text = dem.to_text()
    assert text.count("error(") == 3
    assert "detector(0.5, 0) D0" in text
    assert "logical_observable L0" in text

    graphlike = dem.decompose_graphlike()
    assert graphlike.num_mechanisms == dem.num_mechanisms
    assert (graphlike.detector_matrix() == dem.detector_matrix()).all()


def _repetition_memory(rounds, p, shots):
    qp = QecProgram(3)
    qp.set_options(shots=shots, seed=42, chunk_size=4096, keep_measurements=False)
    prev = None
    for _ in range(rounds):
        qp.noise(QecNoise.depolarize1(p), [0, 1, 2])
        checks = [
            qp.measure_pauli_product([(QecBasis.Z, q), (QecBasis.Z, q + 1)])
            for q in range(2)
        ]
        if prev is None:
            for record in checks:
                qp.detector([RecordRef.absolute(record)])
        else:
            for record, prior in zip(checks, prev):
                qp.detector([RecordRef.absolute(record), RecordRef.absolute(prior)])
        prev = checks
    readout = [qp.measure_z(q) for q in range(3)]
    for check in range(2):
        qp.detector(
            [
                RecordRef.absolute(readout[check]),
                RecordRef.absolute(readout[check + 1]),
                RecordRef.absolute(prev[check]),
            ]
        )
    qp.observable_include(0, [RecordRef.absolute(readout[0])])
    return qp


def test_decoder_beats_physical_error_rate():
    p = 0.02
    qp = _repetition_memory(3, p, 20_000)
    decoder = prism_q.Decoder(qp.detector_error_model())
    assert decoder.num_detectors == qp.num_detectors == 8
    assert decoder.num_observables == 1

    res = qp.run()
    predicted = decoder.decode(res.detectors)
    assert predicted.dtype == np.bool_
    assert predicted.shape == (res.total_shots, 1)
    failures = int((predicted[:, 0] != res.observables[:, 0]).sum())
    # The fixed-seed golden decode count, pinned in `tests/qec_decoder.rs`.
    assert failures == 23
    assert failures / res.total_shots < p


def test_decoder_rejects_bad_inputs():
    import pytest

    qp = _repetition_memory(1, 0.05, 16)
    dem = qp.detector_error_model()
    decoder = prism_q.Decoder(dem)
    with pytest.raises(prism_q.PrismError):
        decoder.decode(np.zeros((4, 2), dtype=np.bool_))

    hyper = QecProgram(1)
    hyper.noise(QecNoise.x_error(0.1), [0])
    for _ in range(3):
        record = hyper.measure_pauli_product([(QecBasis.Z, 0)])
        hyper.detector([RecordRef.absolute(record)])
    with pytest.raises(prism_q.PrismError, match="decompose_graphlike"):
        prism_q.Decoder(hyper.detector_error_model())

import numpy as np
import pytest

import prism_q
from prism_q import QecBasis, QecNoise, QecProgram, QecRecordRef


def _repetition_round():
    qp = QecProgram(3)
    qp.set_options(shots=128, seed=42)
    for q in range(3):
        qp.reset(QecBasis.Z, q)
    qp.push_gate(prism_q.Gate.x(), [0])
    r0 = qp.measure_pauli_product([(QecBasis.Z, 0), (QecBasis.Z, 1)])
    r1 = qp.measure_pauli_product([(QecBasis.Z, 1), (QecBasis.Z, 2)])
    qp.detector([QecRecordRef.absolute(r0)])
    qp.detector_lookback([1])
    m0 = qp.measure_z(0)
    qp.observable_include(0, [QecRecordRef.absolute(m0)])
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
    qp.detector([QecRecordRef.absolute(rr)])
    res = qp.run()
    frac = res.detectors[:, 0].mean()
    assert 0.4 < frac < 0.6


def test_postselect_rejects_shots():
    qp = QecProgram(1)
    qp.set_options(shots=512, seed=1)
    qp.reset(QecBasis.Z, 0)
    qp.push_gate(prism_q.Gate.h(), [0])
    r = qp.measure_z(0)
    qp.postselect([QecRecordRef.absolute(r)], False)
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
        QecRecordRef.lookback(0)


def test_detector_error_model_matches_program():
    qp = QecProgram(3)
    qp.noise(QecNoise.x_error(0.05), [0, 1, 2])
    r0 = qp.measure_pauli_product([(QecBasis.Z, 0), (QecBasis.Z, 1)])
    r1 = qp.measure_pauli_product([(QecBasis.Z, 1), (QecBasis.Z, 2)])
    qp.detector([QecRecordRef.absolute(r0)], coords=[0.5, 0.0])
    qp.detector([QecRecordRef.absolute(r1)])
    m0 = qp.measure_z(0)
    qp.observable_include(0, [QecRecordRef.absolute(m0)])

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
                qp.detector([QecRecordRef.absolute(record)])
        else:
            for record, prior in zip(checks, prev):
                qp.detector([QecRecordRef.absolute(record), QecRecordRef.absolute(prior)])
        prev = checks
    readout = [qp.measure_z(q) for q in range(3)]
    for check in range(2):
        qp.detector(
            [
                QecRecordRef.absolute(readout[check]),
                QecRecordRef.absolute(readout[check + 1]),
                QecRecordRef.absolute(prev[check]),
            ]
        )
    qp.observable_include(0, [QecRecordRef.absolute(readout[0])])
    return qp


def test_decoder_beats_physical_error_rate():
    p = 0.02
    qp = _repetition_memory(3, p, 20_000)
    decoder = prism_q.UnionFindDecoder(qp.detector_error_model())
    assert decoder.num_detectors == qp.num_detectors == 8
    assert decoder.num_observables == 1

    res = qp.run()
    predicted = decoder.decode(res.detectors)
    assert predicted.dtype == np.bool_
    assert predicted.shape == (res.total_shots, 1)
    failures = int((predicted[:, 0] != res.observables[:, 0]).sum())
    # The fixed-seed golden decode count, pinned in `tests/qec_decoder.rs`.
    assert failures == 27
    assert failures / res.total_shots < p


def test_decoder_rejects_bad_inputs():
    import pytest

    qp = _repetition_memory(1, 0.05, 16)
    dem = qp.detector_error_model()
    decoder = prism_q.UnionFindDecoder(dem)
    with pytest.raises(prism_q.PrismError):
        decoder.decode(np.zeros((4, 2), dtype=np.bool_))

    hyper = QecProgram(1)
    hyper.noise(QecNoise.x_error(0.1), [0])
    for _ in range(3):
        record = hyper.measure_pauli_product([(QecBasis.Z, 0)])
        hyper.detector([QecRecordRef.absolute(record)])
    with pytest.raises(prism_q.PrismError, match="decompose_graphlike"):
        prism_q.UnionFindDecoder(hyper.detector_error_model())


def _corrected_bell_pair(shots=256):
    qp = QecProgram(2)
    qp.set_options(shots=shots, seed=11)
    qp.push_gate(prism_q.Gate.h(), [0])
    qp.push_gate(prism_q.Gate.cx(), [0, 1])
    record = qp.measure_z(0)
    body = QecProgram(2)
    body.push_gate(prism_q.Gate.x(), [1])
    qp.feedforward([QecRecordRef.absolute(record)], True, body)
    qp.measure_z(1)
    return qp


def test_feedforward_corrects_on_the_reference_path():
    res = _corrected_bell_pair().run_reference()
    measurements = res.measurements
    assert measurements.shape == (256, 2)
    assert 0 < measurements[:, 0].sum() < 256
    assert not measurements[:, 1].any()


def test_feedforward_needs_the_reference_path():
    with pytest.raises(prism_q.PrismError, match="run_qec_program_reference"):
        _corrected_bell_pair().run()


def test_feedforward_body_takes_gates_and_resets_only():
    qp = QecProgram(2)
    r = qp.measure_z(0)
    measuring = QecProgram(2)
    measuring.measure_z(1)
    with pytest.raises(prism_q.PrismError):
        qp.feedforward([QecRecordRef.absolute(r)], True, measuring)
    with pytest.raises(prism_q.PrismError):
        qp.feedforward([QecRecordRef.absolute(r)], True, QecProgram(2))
    wide = QecProgram(3)
    wide.push_gate(prism_q.Gate.x(), [2])
    with pytest.raises(prism_q.PrismError):
        qp.feedforward([QecRecordRef.absolute(r)], True, wide)
    resetting = QecProgram(2)
    resetting.reset(QecBasis.Z, 1)
    qp.feedforward([QecRecordRef.lookback(1)], False, resetting)


def _bell_channel_program(channel, shots=4096):
    qp = QecProgram(2)
    qp.set_options(shots=shots, seed=42)
    qp.push_gate(prism_q.Gate.h(), [0])
    qp.push_gate(prism_q.Gate.cx(), [0, 1])
    qp.noise(channel, [0])
    zz = qp.measure_pauli_product([(QecBasis.Z, 0), (QecBasis.Z, 1)])
    xx = qp.measure_pauli_product([(QecBasis.X, 0), (QecBasis.X, 1)])
    qp.detector([QecRecordRef.absolute(zz)])
    qp.detector([QecRecordRef.absolute(xx)])
    return qp


def test_pauli_channels_flip_their_detectors():
    det = _bell_channel_program(QecNoise.y_error(1.0)).run().detectors
    assert det.all()

    channel = QecNoise.pauli_channel_1(0.0, 0.0, 1.0)
    det = _bell_channel_program(channel).run().detectors
    assert not det[:, 0].any() and det[:, 1].all()

    channel = QecNoise.pauli_channel_1(0.1, 0.2, 0.3)
    dem = _bell_channel_program(channel).detector_error_model()
    assert sorted(dem.probabilities().tolist()) == [0.1, 0.2, 0.3]


def test_pauli_channel_2_branches_reach_the_model():
    rates = [0.005 * (k + 1) for k in range(15)]
    qp = QecProgram(2)
    qp.noise(QecNoise.pauli_channel_2(rates), [0, 1])
    qp.detector([QecRecordRef.absolute(qp.measure_z(0))])
    qp.detector([QecRecordRef.absolute(qp.measure_z(1))])
    dem = qp.detector_error_model()
    # Z readout flips on X or Y: first letter on qubit 0, second on qubit 1.
    x_or_y = {1, 2}
    expected = {}
    for k, p in enumerate(rates):
        first, second = (k + 1) // 4, (k + 1) % 4
        key = (first in x_or_y, second in x_or_y)
        if any(key):
            expected[key] = expected.get(key, 0.0) + p
    matrix = dem.detector_matrix()
    got = {
        (bool(matrix[0, m]), bool(matrix[1, m])): p
        for m, p in enumerate(dem.probabilities())
    }
    assert got.keys() == expected.keys()
    for key, p in expected.items():
        assert abs(got[key] - p) < 1e-12


def test_pauli_channel_2_requires_fifteen_rates():
    import pytest

    with pytest.raises((TypeError, ValueError)):
        QecNoise.pauli_channel_2([0.1] * 14)


def test_program_text_round_trips():
    text = "R 0 1\nH 0\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nM 0 1\nDETECTOR(0, 1) rec[-1] rec[-2]\n"
    qp = QecProgram.from_text(text)
    assert qp.to_text() == text
    again = QecProgram.from_text(qp.to_text())
    assert again.num_detectors == qp.num_detectors == 1


def test_detector_error_model_text_round_trips():
    dem = prism_q.DetectorErrorModel.from_text(
        "error(0.1) D0 D1 ^ D2 L0\nrepeat 2 {\n error(0.01) D0\n shift_detectors(0, 1) 1\n}\n"
        "detector(1, 2) D0\n"
    )
    assert dem.num_mechanisms == 3
    assert dem.num_detectors == 3
    assert dem.num_observables == 1
    assert dem.detector_coords()[2] == [1.0, 4.0]
    assert dem.suggested_decompositions()[0] == [([0, 1], []), ([2], [0])]
    assert dem.suggested_decompositions()[1] == []
    again = prism_q.DetectorErrorModel.from_text(dem.to_text())
    assert again.to_text() == dem.to_text()
    assert (again.detector_matrix() == dem.detector_matrix()).all()

    import pytest

    with pytest.raises(prism_q.PrismError):
        prism_q.DetectorErrorModel.from_text("error(2) D0")


def test_memory_generators_are_deterministic_without_noise():
    programs = [
        QecProgram.repetition_memory(3, 2),
        QecProgram.surface_memory(3, 2),
        QecProgram.surface_memory(3, 2, basis=QecBasis.X),
        QecProgram.color_memory(5, 2, QecBasis.Z),
    ]
    for qp in programs:
        qp.set_options(shots=256, seed=42)
        res = qp.run()
        assert not res.detectors.any()
        assert not res.observables.any()
    assert programs[1].num_qubits == 17
    assert programs[3].num_qubits == 19 + 9


def test_memory_generator_noise_and_decoding():
    noise = prism_q.QecCircuitNoise(
        after_clifford_depolarization=0.01, before_measure_flip_probability=0.01
    )
    assert noise.after_reset_flip_probability == 0.0
    assert prism_q.QecCircuitNoise.uniform(0.02).before_round_data_depolarization == 0.02

    rates = []
    for d in (3, 7):
        qp = QecProgram.repetition_memory(d, d, prism_q.QecCircuitNoise.uniform(0.02))
        qp.set_options(shots=20_000, seed=42)
        decoder = prism_q.UnionFindDecoder(qp.detector_error_model().decompose_graphlike())
        res = qp.run()
        predicted = decoder.decode(res.detectors)
        rates.append(float((predicted[:, 0] != res.observables[:, 0]).mean()))
    assert rates[1] < rates[0]

    qp = QecProgram.surface_memory(3, 3, QecBasis.Z, noise)
    again = QecProgram.from_text(qp.to_text())
    assert again.num_detectors == qp.num_detectors
    assert again.detector_error_model().to_text() == qp.detector_error_model().to_text()


def test_memory_generators_reject_bad_parameters():
    import pytest

    with pytest.raises(prism_q.PrismError):
        QecProgram.color_memory(4, 2)
    with pytest.raises(prism_q.PrismError):
        QecProgram.surface_memory(3, 2, basis=QecBasis.Y)
    with pytest.raises(prism_q.PrismError):
        QecProgram.repetition_memory(3, 0)


def _hypergraph_program():
    hyper = QecProgram(1)
    hyper.set_options(shots=256, seed=42)
    hyper.noise(QecNoise.x_error(0.1), [0])
    for _ in range(3):
        record = hyper.measure_pauli_product([(QecBasis.Z, 0)])
        hyper.detector([QecRecordRef.absolute(record)])
    return hyper


def test_matching_decoder_never_worse_than_union_find():
    p = 0.02
    qp = _repetition_memory(3, p, 20_000)
    dem = qp.detector_error_model()
    res = qp.run()
    union_find = prism_q.UnionFindDecoder(dem)
    matching = prism_q.MatchingDecoder(dem)
    assert matching.num_detectors == 8
    assert matching.num_observables == 1

    predicted = matching.decode(res.detectors)
    assert predicted.dtype == np.bool_
    assert predicted.shape == (res.total_shots, 1)
    failures = int((predicted[:, 0] != res.observables[:, 0]).sum())
    rate = matching.logical_error_rate(res.detectors, res.observables)
    assert rate == failures / res.total_shots
    assert rate <= union_find.logical_error_rate(res.detectors, res.observables)
    assert rate < p


def test_matching_decoder_rejects_hypergraph_models():
    import pytest

    with pytest.raises(prism_q.PrismError, match="decompose_graphlike"):
        prism_q.MatchingDecoder(_hypergraph_program().detector_error_model())


def test_bposd_decoder_accepts_hypergraph_models():
    hyper = _hypergraph_program()
    dem = hyper.detector_error_model()
    decoder = prism_q.BpOsdDecoder(dem, osd_method="exhaustive", osd_order=4)
    assert decoder.num_detectors == 3
    res = hyper.run()
    predicted = decoder.decode(res.detectors)
    assert predicted.shape == (res.total_shots, 0)
    assert decoder.logical_error_rate(res.detectors, res.observables) == 0.0


def test_bposd_decoder_matches_union_find_on_repetition_memory():
    p = 0.02
    qp = _repetition_memory(3, p, 20_000)
    dem = qp.detector_error_model()
    res = qp.run()
    union_find = prism_q.UnionFindDecoder(dem).logical_error_rate(res.detectors, res.observables)
    for bp_method in ("min_sum", "product_sum"):
        decoder = prism_q.BpOsdDecoder(dem, bp_method=bp_method, max_iterations=20)
        rate = decoder.logical_error_rate(res.detectors, res.observables)
        assert rate <= union_find


def test_bposd_decoder_rejects_bad_options():
    import pytest

    dem = _repetition_memory(1, 0.05, 16).detector_error_model()
    with pytest.raises(prism_q.PrismError, match="bp_method"):
        prism_q.BpOsdDecoder(dem, bp_method="sum_product")
    with pytest.raises(prism_q.PrismError, match="osd_method"):
        prism_q.BpOsdDecoder(dem, osd_method="osd1")
    with pytest.raises(prism_q.PrismError, match="scaling"):
        prism_q.BpOsdDecoder(dem, min_sum_scaling=0.0)
    with pytest.raises(prism_q.PrismError, match="exhaustive"):
        prism_q.BpOsdDecoder(dem, osd_method="exhaustive", osd_order=40)
    decoder = prism_q.BpOsdDecoder(dem)
    with pytest.raises(prism_q.PrismError):
        decoder.decode(np.zeros((4, 2), dtype=np.bool_))


def test_decoders_take_packed_rows():
    import pytest

    qp = _repetition_memory(3, 0.02, 2_000)
    dem = qp.detector_error_model()
    res = qp.run()
    for decoder in (
        prism_q.UnionFindDecoder(dem),
        prism_q.MatchingDecoder(dem),
        prism_q.BpOsdDecoder(dem),
    ):
        packed = decoder.decode_packed(res.packed_detectors())
        assert packed.dtype == np.uint8
        assert packed.shape == (res.total_shots, 1)
        unpacked = np.unpackbits(packed, axis=1, bitorder="little")[:, :1].astype(bool)
        assert np.array_equal(unpacked, decoder.decode(res.detectors))
        assert decoder.logical_error_rate(
            res.packed_detectors(), res.packed_observables()
        ) == decoder.logical_error_rate(res.detectors, res.observables)
        with pytest.raises(prism_q.PrismError):
            decoder.decode_packed(np.zeros((4, 3), dtype=np.uint8))


def test_expectation_values_reach_python():
    qp = QecProgram(2)
    qp.set_options(shots=64, seed=42)
    qp.push_gate(prism_q.Gate.x(), [0])
    qp.expectation_value([(QecBasis.Z, 0)], 2.0)
    qp.expectation_value([(QecBasis.Z, 0), (QecBasis.Z, 1)])
    assert qp.num_expectation_values == 2
    for result in (qp.run(), qp.run_reference()):
        estimates = result.expectation_values
        assert [e.mean for e in estimates] == [-2.0, -1.0]
        assert all(e.variance == 0.0 for e in estimates)
    assert QecProgram.from_text("EXP_VAL Z0").run().expectation_values[0].mean == 1.0
    assert _repetition_memory(3, 0.02, 16).run().expectation_values is None


def test_wilson_intervals_bracket_the_rates():
    res = _repetition_memory(3, 0.05, 4_000).run()
    low, high = res.survivor_rate_wilson_interval()
    assert low <= res.survivor_rate() <= high
    (low, high), = res.logical_error_rate_wilson_intervals(1.0)
    assert low <= res.logical_error_rates()[0] <= high


def test_renamed_qec_types_keep_deprecated_aliases():
    import pytest

    for old, new in (
        ("Decoder", "UnionFindDecoder"),
        ("QecResult", "QecSampleResult"),
        ("RecordRef", "QecRecordRef"),
    ):
        with pytest.warns(DeprecationWarning, match=new):
            assert getattr(prism_q, old) is getattr(prism_q, new)
    assert isinstance(QecProgram(1).run(), prism_q.QecSampleResult)

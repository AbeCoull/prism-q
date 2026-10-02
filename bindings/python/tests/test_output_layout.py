import numpy as np

from prism_q import QecProgram, parse_qasm, simulate


def _qasm(num_qubits, num_bits, body):
    return parse_qasm(
        "OPENQASM 3.0;\n"
        'include "stdgates.inc";\n'
        f"qubit[{num_qubits}] q;\n"
        f"bit[{num_bits}] c;\n" + body
    )


def _shots_and_counts(circuit, num_shots=8):
    result = simulate(circuit).seed(5).shots(num_shots)
    return result.shots, result.counts()


def test_shot_columns_follow_classical_bit_order():
    circuit = _qasm(3, 3, "x q[0];\n" + "".join(f"c[{i}] = measure q[{i}];\n" for i in range(3)))
    shots, counts = _shots_and_counts(circuit)
    assert shots.dtype == np.bool_
    assert shots.shape == (8, 3)
    assert (shots == np.array([True, False, False])).all()
    assert counts == {"100": 8}


def test_permuted_map_leaves_unused_bits_clear():
    body = "x q[0];\nx q[2];\nc[3] = measure q[0];\nc[0] = measure q[1];\nc[1] = measure q[2];\n"
    shots, counts = _shots_and_counts(_qasm(3, 5, body))
    assert shots.shape == (8, 5)
    assert (shots == np.array([False, True, False, True, False])).all()
    assert counts == {"01010": 8}


def test_bit_written_twice_keeps_the_last_write():
    body = "x q[0];\nc[0] = measure q[0];\nc[0] = measure q[1];\nc[1] = measure q[0];\n"
    shots, counts = _shots_and_counts(_qasm(2, 2, body))
    assert (shots == np.array([False, True])).all()
    assert counts == {"01": 8}


def test_shots_cross_word_boundaries():
    n = 130
    flipped = {0, 63, 64, 65, 127, 128, 129}
    body = "".join(f"x q[{q}];\n" for q in sorted(flipped))
    body += "".join(f"c[{q}] = measure q[{q}];\n" for q in range(n))
    circuit = _qasm(n, n, body)
    shots, counts = _shots_and_counts(circuit, num_shots=70)
    expected = np.array([q in flipped for q in range(n)])
    key = "".join("1" if b else "0" for b in expected)
    assert shots.shape == (70, n)
    assert (shots == expected).all()
    assert counts == {key: 70}
    assert simulate(circuit).seed(5).sample_counts(70).counts() == {key: 70}


def test_random_shots_agree_with_counts():
    n = 70
    body = "".join(f"h q[{q}];\n" for q in range(n))
    body += "".join(f"c[{q}] = measure q[{q}];\n" for q in range(n))
    result = simulate(_qasm(n, n, body)).seed(11).shots(300)
    shots = result.shots
    rows = {}
    for row in shots:
        key = "".join("1" if b else "0" for b in row)
        rows[key] = rows.get(key, 0) + 1
    assert result.counts() == rows


def _qec_flip_program(num_qubits, flipped, noisy):
    lines = [f"X {q}" for q in sorted(flipped)]
    if noisy:
        lines.append("X_ERROR(0.5) " + " ".join(str(q) for q in noisy))
    lines += [f"M {q}" for q in range(num_qubits)]
    lines += [f"DETECTOR rec[-{num_qubits - q}]" for q in range(num_qubits)]
    lines += [f"OBSERVABLE_INCLUDE({k}) rec[-{num_qubits - q}]" for k, q in enumerate(noisy)]
    return QecProgram.from_text("\n".join(lines) + "\n")


def test_qec_records_cross_word_boundaries():
    n = 130
    flipped = {0, 63, 64, 65, 127, 129}
    program = _qec_flip_program(n, flipped, [])
    program.set_options(shots=100, seed=3)
    result = program.run()
    expected = np.array([q in flipped for q in range(n)])
    for records in (result.detectors, result.measurements):
        assert records.dtype == np.bool_
        assert records.shape == (100, n)
        assert (records == expected).all()
    assert result.observables.shape == (100, 0)


def test_qec_random_records_match_measurements():
    n = 70
    noisy = [1, 64, 69]
    program = _qec_flip_program(n, {2, 66}, noisy)
    program.set_options(shots=1000, seed=9)
    result = program.run()
    detectors = result.detectors
    measurements = result.measurements
    observables = result.observables
    assert np.array_equal(detectors, measurements)
    assert np.array_equal(observables, measurements[:, noisy])
    assert observables.sum(axis=0).tolist() == result.logical_errors
    assert measurements[:, 2].all() and measurements[:, 66].all()
    quiet = [q for q in range(n) if q not in noisy and q not in (2, 66)]
    assert not measurements[:, quiet].any()


def test_qec_dropped_measurements_are_empty():
    program = _qec_flip_program(5, {1}, [])
    program.set_options(shots=10, seed=1, keep_measurements=False)
    result = program.run()
    assert result.measurements.shape == (0, 5)
    assert result.detectors.shape == (10, 5)


def test_analytical_observable_records_put_ones_first():
    # A noiseless EXP_VAL program takes the analytical route, whose observable
    # records are stored measurement-major rather than shot-major.
    program = QecProgram.from_text(
        "H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
        "H 1\nM 1\nOBSERVABLE_INCLUDE(1) rec[-1]\nEXP_VAL Z2\n"
    )
    program.set_options(shots=100, seed=4)
    result = program.run()
    observables = result.observables
    assert observables.shape == (100, 2)
    for column, count in enumerate(result.logical_errors):
        assert 0 < count < 100
        assert observables[:count, column].all()
        assert not observables[count:, column].any()

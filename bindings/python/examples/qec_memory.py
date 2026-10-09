"""Run a repetition-code memory experiment and decode it with the built-in decoder."""

from prism_q import UnionFindDecoder, Gate, QecBasis, QecNoise, QecProgram, QecRecordRef


def repetition_memory(distance, rounds, p, shots):
    data = [2 * i for i in range(distance)]
    ancillas = [2 * i + 1 for i in range(distance - 1)]
    qp = QecProgram(2 * distance - 1)
    qp.set_options(shots, seed=42)
    for q in data + ancillas:
        qp.reset(QecBasis.Z, q)

    previous = None
    for _ in range(rounds):
        qp.noise(QecNoise.x_error(p), data)
        for a in ancillas:
            qp.push_gate(Gate.cx(), [a - 1, a])
            qp.push_gate(Gate.cx(), [a + 1, a])
        current = []
        for a in ancillas:
            current.append(qp.measure_z(a))
            qp.reset(QecBasis.Z, a)
        for i, record in enumerate(current):
            refs = [QecRecordRef.absolute(record)]
            if previous is not None:
                refs.append(QecRecordRef.absolute(previous[i]))
            qp.detector(refs)
        previous = current

    final = [qp.measure_z(q) for q in data]
    for i, record in enumerate(previous):
        qp.detector([QecRecordRef.absolute(r) for r in (final[i], final[i + 1], record)])
    qp.observable_include(0, [QecRecordRef.absolute(final[0])])
    return qp


def logical_error_rate(qp):
    model = qp.detector_error_model().decompose_graphlike()
    result = qp.run()
    predicted = UnionFindDecoder(model).decode(result.detectors)
    failures = (predicted[:, 0] != result.observables[:, 0]).sum()
    return failures / result.total_shots, result.logical_error_rates()[0]


for distance in (3, 5, 7):
    qp = repetition_memory(distance, rounds=distance, p=0.05, shots=20_000)
    decoded, raw = logical_error_rate(qp)
    print(f"d={distance}: {qp.num_detectors} detectors, raw {raw:.4f}, decoded {decoded:.4f}")

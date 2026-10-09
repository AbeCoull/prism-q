# A QEC Memory Experiment

Goal: protect one bit in a distance-d repetition code for d rounds of noisy syndrome
extraction, decode the detector record, and report the logical error rate at three
distances.

The code stores a logical 0 in `d` data qubits. Between each pair of data qubits sits an
ancilla that measures their parity `Z_i Z_{i+1}`. Before every round each data qubit
suffers a bit flip with probability `p`. A logical error is a majority of data qubits
flipped at the end, which the decoder has to infer from parity changes alone.

## Build the program

`QecProgram` is a measurement-record program: every `measure_z` returns the index of the
record it writes, and detectors and observables are parities over records. Data qubits
take the even indices and ancillas the odd ones.

```python
from prism_q import Gate, QecBasis, QecNoise, QecProgram, QecRecordRef


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
```

Three kinds of detector appear. In the first round an ancilla outcome is compared with the
noiseless value 0, so the detector is that one record. In later rounds it is the change
since the previous round. After the last round the data qubits are measured directly and
each final detector checks their parity against the last ancilla outcome. Every detector
reads 0 when nothing went wrong.

The observable is data qubit 0. Each ancilla is reset after it is measured, since a
program may not reuse a measured qubit before a reset.

## Sample and decode

`detector_error_model()` derives, from the noise annotations, which detectors and
observables each error flips and with what probability. The built-in union-find decoder
needs the graphlike form, where each mechanism flips at most two detectors.

```python
from prism_q import UnionFindDecoder

qp = repetition_memory(distance=5, rounds=5, p=0.05, shots=20_000)
model = qp.detector_error_model().decompose_graphlike()
print(qp.num_detectors, model.num_mechanisms)   # 24 25

result = qp.run()
predicted = UnionFindDecoder(model).decode(result.detectors)
failures = (predicted[:, 0] != result.observables[:, 0]).sum()
print(result.detectors.shape)                    # (20000, 24)
print(failures / result.total_shots)             # 0.00545
```

`result.detectors` and `result.observables` are `bool` arrays with one row per shot.
`decode` returns the predicted observable flips in the same shape as `observables`, and
a shot fails when the prediction disagrees with what happened.

## Compare distances

```python
for distance in (3, 5, 7):
    qp = repetition_memory(distance, rounds=distance, p=0.05, shots=20_000)
    model = qp.detector_error_model().decompose_graphlike()
    result = qp.run()
    predicted = UnionFindDecoder(model).decode(result.detectors)
    decoded = (predicted[:, 0] != result.observables[:, 0]).mean()
    raw = result.logical_error_rates()[0]
    print(f"d={distance}: raw {raw:.4f}, decoded {decoded:.4f}")
# d=3: raw 0.1362, decoded 0.0235
# d=5: raw 0.2084, decoded 0.0054
# d=7: raw 0.2647, decoded 0.0008
```

`logical_error_rates()` counts shots whose observable flipped with no correction, so the
raw rate grows with distance: more rounds, more chances for qubit 0 to flip. The decoded
rate falls by a factor of four to seven per step in distance, which is what a code
operating below threshold looks like.

`model.to_text()` writes the model in the common detector error model text format, for
an external matching or belief-propagation decoder. The
[Noise and QEC guide](../guides/qec.md) covers the text program format, postselection,
and the other noise channels.

## In Rust

The program builder has the same shape, with `Result` returns. Decoding works on the
packed shot records directly:

```rust
use prism_q::{UnionFindDecoder, parse_qec_program, run_qec_program};

let program = parse_qec_program(
    "R 0 1 2 3 4
     X_ERROR(0.05) 0 2 4
     CX 0 1 2 1 2 3 4 3
     MR 1 3
     DETECTOR rec[-2]
     DETECTOR rec[-1]
     M 0 2 4
     DETECTOR rec[-3] rec[-2] rec[-5]
     DETECTOR rec[-2] rec[-1] rec[-4]
     OBSERVABLE_INCLUDE(0) rec[-3]",
)?;
let model = program.detector_error_model()?.decompose_graphlike()?;
let decoder = UnionFindDecoder::from_model(&model)?;
let result = run_qec_program(&program)?;
let predicted = decoder.decode_packed(&result.detectors)?;
let failures = (0..result.total_shots)
    .filter(|&shot| predicted.get_bit(shot, 0) != result.observables.get_bit(shot, 0))
    .count();
assert!(failures < result.total_shots / 10);
# Ok::<(), prism_q::PrismError>(())
```

That is one round at distance 3 in the text format.
[`examples/qec_memory.rs`](https://github.com/AbeCoull/prism-q/blob/main/examples/qec_memory.rs)
builds the multi-round program with `QecProgram` methods and prints the same table.

Next: [Dynamic Circuits](./dynamic-circuits.md).

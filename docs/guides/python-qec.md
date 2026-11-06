# Quantum Error Correction from Python

`QecProgram` holds a program in the native QEC IR: detectors, logical observables,
postselection and Pauli noise annotations around Clifford gates and measurements. The
[QEC guide](qec.md) describes the model and its runners, and the
[QEC memory tutorial](../tutorials/qec-memory.md) decodes a repetition code at three
distances.

## Memory experiments

```python
from prism_q import QecBasis, QecCircuitNoise, QecProgram

qp = QecProgram.surface_memory(3, 3, QecBasis.Z, QecCircuitNoise.uniform(0.002))
qp.set_options(shots=10_000, seed=42)
res = qp.run()
raw_rate = res.logical_error_rates()[0]
```

`repetition_memory(distance, rounds, noise=None)` builds a Z memory on the repetition
code, `surface_memory(distance, rounds, basis=QecBasis.Z, noise=None)` one on the rotated
surface code, and `color_memory(...)`, with the same arguments, one on the triangular
6.6.6 color code at odd distance. Each carries detectors with coordinates and logical
observable 0. `QecCircuitNoise` sets four rates, all keyword arguments defaulting to 0:
`after_clifford_depolarization`, `before_measure_flip_probability`,
`after_reset_flip_probability` and `before_round_data_depolarization`.
`QecCircuitNoise.uniform(p)` sets all four to `p`, and `noise=None` gives the noiseless
program.

`set_options(shots, seed=42, chunk_size=None, keep_measurements=True,
detector_reference="noiseless")` configures `run()`. `"raw"` reports each detector's
plain parity instead of its flip against the noiseless reference run
([Detector reference](qec.md#detector-reference)).

## Building a program

`QecProgram(num_qubits)` starts empty, and `from_text` parses the text form that
`to_text` writes. Operations append in order:

| Method | Appends |
|--------|---------|
| `push_gate(gate, targets)` | a Clifford gate |
| `reset(basis, qubit)` | a reset in `QecBasis.X`, `Y` or `Z` |
| `measure(basis, qubit)`, `measure_z(qubit)`, `measure_x(qubit)` | a measurement; returns its record index |
| `measure_pauli_product(terms)` | an `MPP` over `(basis, qubit)` terms; returns its record index |
| `detector(records, coords=None)` | a detector over `QecRecordRef` records; returns its index |
| `detector_lookback(distances)` | a detector over records counted back from the latest, 1 being `rec[-1]` |
| `observable_include(observable, records)` | records into a logical observable |
| `postselect(records, expected)` | a filter that discards shots whose parity differs |
| `noise(channel, targets)` | a `QecNoise` annotation |
| `expectation_value(terms, coefficient=1.0)` | an `EXP_VAL` estimate, below |
| `feedforward(records, expected, body)` | a correction, below |

`QecRecordRef.absolute(index)` names a record by its index and `QecRecordRef.lookback(d)`
by its distance back. `QecNoise` has `x_error(p)`, `y_error(p)`, `z_error(p)`,
`depolarize1(p)`, `depolarize2(p)`, `pauli_channel_1(px, py, pz)`,
`pauli_channel_2(probabilities)` over the 15 non-identity pairs, and the leakage channels
`leak(p)`, `seep(p)` and `leak_transport(p)`.

```python
from prism_q import Gate, QecNoise, QecRecordRef

check = QecProgram(3)
check.set_options(shots=1000, seed=7)
check.noise(QecNoise.x_error(0.05), [0, 2])
check.push_gate(Gate.cx(), [0, 1])
check.push_gate(Gate.cx(), [2, 1])
check.measure_z(1)
check.detector_lookback([1])
check.measure_pauli_product([(QecBasis.Z, 0), (QecBasis.Z, 2)])
check.detector_lookback([1, 2])
data = check.measure_z(0)
check.observable_include(0, [QecRecordRef.absolute(data)])
print(check.to_text())
```

The ancilla and the `MPP` measure the same `Z0 Z2` parity, so the second detector never
fires, while the first fires on the 9.5 percent of shots where exactly one data qubit
flipped.

## Results

`run()` returns a `QecSampleResult`. `detectors`, `observables` and `measurements` are
NumPy `bool_` arrays with one row per shot. `logical_error_rates()` and `survivor_rate()`
come with `logical_error_rate_wilson_intervals(z_score)` and
`survivor_rate_wilson_interval(z_score)`, whose default `z_score` gives a two-sided 95
percent interval. `total_shots`, `accepted_shots` and `discarded_shots` count
postselection, `logical_errors` the failures per observable, and `heralds` holds one
column per `LEAK` target, or `None`.

`expectation_value(terms, coefficient=1.0)` appends the text format's `EXP_VAL` op over
`(basis, qubit)` terms. Estimates arrive in op order as
`QecSampleResult.expectation_values`, a list of `QecObservableEstimate` with `mean`,
`variance` and `num_shots`, or `None` for a program without one.
`observable_expectations` has the same shape for the logical observables when a
weighted or analytical T strategy produced them:

```python
prog = QecProgram(2)
prog.set_options(shots=1000, seed=1)
prog.push_gate(Gate.h(), [0])
prog.push_gate(Gate.cx(), [0, 1])
prog.expectation_value([(QecBasis.Z, 0), (QecBasis.Z, 1)])
zz = prog.run().expectation_values[0].mean   # 1.0
```

## Compiling once for repeated sampling

`run()` compiles the program on every call. `compile_sampler()` compiles once and
returns a `QecCompiledSampler` with the program's seed, chunk size and detector
reference:

```python
sampler = qp.compile_sampler()
detectors, observables = sampler.sample(10_000)
batch = sampler.sample_result(1000, seed=7)
```

`sample(shots, *, seed=None, bit_packed=True, separate_observables=True)` returns a
`(detectors, observables)` pair of `(shots, ceil(n / 8))` byte arrays in the
`packed_detectors()` layout below; bool arrays with `bit_packed=False`; or, with
`separate_observables=False`, one array whose columns are the detectors followed by the
observables. It counts every shot, whether or not postselection accepts it.
`sample_result` returns a full `QecSampleResult`. `seed=None` continues one stream that
starts at the program's seed, and an explicit seed draws what `run()` draws at that
seed. Sampling releases the GIL, and calls on one sampler from several threads run one
at a time. Measurement records are dropped unless `compile_sampler(keep_measurements=True)`
asks for them, which `sample_result` then reports. A program that needs the per-shot or
analytical runners (`EXP_VAL`, non-Clifford gates) raises `PrismError` from
`compile_sampler`.

## Detector error models and decoders

`detector_error_model()` derives the program's `DetectorErrorModel`.
`probabilities()` (float64), `detector_matrix()`, and `observable_matrix()` (bool,
detectors or observables by mechanisms) feed check-matrix decoder constructors;
`detector_coords()` holds per-detector coordinates; `decompose_graphlike()` returns the
form matching decoders need (at most two detectors per mechanism); and `to_text()`
writes the common detector error model text format for file-based decoders:

```python
dem = qp.detector_error_model()
H, p, L = dem.detector_matrix(), dem.probabilities(), dem.observable_matrix()
with open("memory_d3.dem", "w") as f:
    f.write(dem.to_text())
```

`UnionFindDecoder` is a built-in union-find decoder over a graphlike model, so a memory
experiment needs no external decoder. `decode` takes the `(shots, num_detectors)` bool
detector array and returns the `(shots, num_observables)` predicted observable flips:

```python
decoder = prism_q.UnionFindDecoder(dem.decompose_graphlike())
predicted = decoder.decode(res.detectors)
failures = (predicted[:, 0] != res.observables[:, 0]).sum()
```

`MatchingDecoder` (exact minimum-weight perfect matching, graphlike models) and
`BpOsdDecoder` (belief propagation with ordered statistics, any model) share that
surface. Every decoder also has `logical_error_rate(detectors, observables)`, the
fraction of shots with any observable mispredicted, which takes bool or packed arrays
as long as both arguments match, and `decode_packed`, which takes the
`packed_detectors()` layout below and returns flips in that layout. `BpOsdDecoder` takes
keyword options `max_iterations`, `bp_method` (`"min_sum"` or `"product_sum"`),
`min_sum_scaling`, `osd_method` (`"osd0"`, `"cs"`, or `"exhaustive"`), and
`osd_order`:

```python
matching = prism_q.MatchingDecoder(dem.decompose_graphlike())
bposd = prism_q.BpOsdDecoder(dem, osd_method="cs", osd_order=10)
rates = [d.logical_error_rate(res.detectors, res.observables) for d in (matching, bposd)]
```

## Feedforward

`feedforward(records, expected, body)` applies the gates and resets of `body`, a
second `QecProgram` holding the correction, only in shots where the parity over
`records` equals `expected`: an adaptive correction keyed on a syndrome. A body of only
Pauli gates (`X`, `Y`, `Z`, identity) runs on the compiled path of `run()`, where the
correction becomes a fixed set of record flips. Any other body breaks the single affine
map that sampler applies to every shot, so the program runs through `run_reference()`,
the per-shot statevector path, at `O(shots * 2^n)`; it suits small programs.

```python
from prism_q import Gate, QecProgram, QecRecordRef

prog = QecProgram(2)
prog.set_options(shots=1000, seed=1)
prog.push_gate(Gate.h(), [0])
prog.push_gate(Gate.cx(), [0, 1])
flag = prog.measure_z(0)
fix = QecProgram(2)
fix.push_gate(Gate.x(), [1])
prog.feedforward([QecRecordRef.absolute(flag)], True, fix)
prog.measure_z(1)
assert not prog.run().measurements[:, 1].any()
assert not prog.run_reference().measurements[:, 1].any()
```

## Packed records

`packed_detectors()`, `packed_observables()`, and `packed_measurements()` return
the same records eight to a byte, as `(shots, ceil(n / 8))` `uint8` arrays: record
`j` of a shot is bit `j % 8` of byte `j // 8`, and the unused high bits of the last
byte are zero. That is an eighth of the bool arrays' memory; `np.unpackbits` restores
them:

```python
import numpy as np

packed = res.packed_detectors()
detectors = np.unpackbits(packed, axis=1, count=qp.num_detectors, bitorder="little")
assert (detectors.astype(bool) == res.detectors).all()
```

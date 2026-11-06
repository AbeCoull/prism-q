# Quantum Error Correction from Python

`QecProgram` exposes the native QEC IR: `reset`, `measure`, `detector`,
`observable_include`, `postselect`, `noise` and `expectation_value`, with `QecBasis`,
`QecNoise`, and `QecRecordRef` as the supporting types. `run()` returns a
`QecSampleResult` carrying detector, observable, and measurement arrays as NumPy
`bool_` matrices, plus `logical_error_rates()` and `survivor_rate()` with
`logical_error_rate_wilson_intervals(z_score)` and
`survivor_rate_wilson_interval(z_score)` (the default `z_score` gives a two-sided 95
percent interval). `QecProgram.from_text` parses the text form and
`QecProgram.from_stim` reads Stim circuit files. The
[Noise and QEC guide](./qec.md) describes the model.

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
res = qp.run()
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

# Native QEC Program IR

`QecProgram` in `src/qec/mod.rs` is a measurement-record IR for QEC workloads
that need detectors, logical observables, postselection, expectation metadata,
and Pauli-noise annotations before sampler lowering. It is separate from
`Circuit` so measurement-record programs do not need to fit final-measurement
OpenQASM semantics.

`QecOp` stores gates, basis measurements, MPP-style Pauli-product
measurements, resets, detector rows, observable includes, expectation-value
metadata, postselection predicates, noise annotations, and tick separators.
Record references can be absolute indices or `rec[-k]` style lookbacks.
Construction validates qubit bounds, gate arity, finite coordinates and
coefficients, finite probabilities, and measurement-record scope. Detector,
observable, and postselection rows can be resolved to absolute measurement
indices for later compilation into packed samplers.

## Parsing

`parse_qec_program` and `QecProgram::from_text` parse the native QEC text
subset used by current benchmark planning: `H`, `S`, `S_DAG`, `T`, `T_DAG`,
`CX`, `CZ`, `R`/`RX`/`RY`, `M`/`MX`/`MY`, `MR` variants, `MPP`, `DETECTOR`,
`OBSERVABLE_INCLUDE`, `POSTSELECT`, `EXP_VAL`, Pauli-noise instructions, `TICK`,
`QUBIT_COORDS`, `SHIFT_COORDS`, and flattened `REPEAT` blocks. The parser
resolves `rec[-k]` references while building the program. Numeric arguments on
basis measurements, such as `M(0.001)`, lower to pre-measurement Pauli flips
that affect the measurement record.

## Lowering

`compile_qec_program_rows` lowers basis measurements and `MPP` records into the
same packed X/Z Pauli row representation used by the compiled sampler internals.
It also carries detector, observable, and postselection rows forward as
absolute measurement-record indices. Detector, observable, and postselection
projection uses `PackedShots::parity_rows`, so the QEC layer reuses the existing
packed parity engine instead of maintaining a second one. This is a
sampler-lowering artifact, not an execution engine. Gate, reset, and noise
execution lives in `run_qec_program`; `EXP_VAL` remains unsupported in the
packed runner until estimator semantics land.

## Execution

`run_qec_program` is the scalable native QEC execution path. It lowers
Clifford-compatible QEC programs into the existing packed compiled sampler, so
sampling uses packed measurement records instead of dense amplitudes. It
supports Clifford gates, basis resets and measurements, `MPP`, detectors,
observables, postselection, Pauli-noise annotations, and optional
raw-measurement retention. Noisy Clifford programs compile `X_ERROR`,
`Z_ERROR`, `DEPOLARIZE1`, and correlated `DEPOLARIZE2` events into packed
sensitivity rows, then XOR those rows into the packed measurement records.
Non-Clifford gates and `EXP_VAL` reject clearly until their production
strategies land.

```admonish note title="V1 reset requirement"
V1 requires a reset before any later gate reuse of a measured qubit, because the
compiled lowering defers measurements to terminal records. `QecOptions::chunk_size`
bounds compiled-runner shot batches. When raw measurements are omitted, chunking avoids
materializing the full measurement record matrix before detector, observable,
postselection, and logical-error accounting.
```

`run_qec_program_reference` is the small-program correctness oracle. It runs one
state-vector simulation per shot and supports Pauli-noise annotations, but it is
not the production performance path.

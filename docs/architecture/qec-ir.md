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
execution lives in `run_qec_program`; `EXP_VAL` has no packed-row
representation, so the row compiler rejects it and `run_qec_program` routes
such programs to the estimator paths described below.

## Execution

`run_qec_program` is the scalable native QEC execution path. It lowers
Clifford-compatible QEC programs into the existing packed compiled sampler, so
sampling uses packed measurement records instead of dense amplitudes. It
supports Clifford gates, basis resets and measurements, `MPP`, detectors,
observables, postselection, Pauli-noise annotations, and optional
raw-measurement retention. Noisy Clifford programs compile `X_ERROR`,
`Z_ERROR`, `DEPOLARIZE1`, and correlated `DEPOLARIZE2` events into packed
sensitivity rows, then XOR those rows into the packed measurement records.
Non-Clifford gates reject clearly on the packed path; programs containing
`EXP_VAL` route to the expectation-value paths below, which accept T gates
via the analytical strategies.

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

## Expectation values

`EXP_VAL(c) P1*...*Pk` estimates `c * <P>` for the Pauli product `P` in the
program's final state and returns one `QecObservableEstimate` per op, in op
order, in `QecSampleResult::expectation_values`. Two placement rules make
"final state" well defined on every path:

- Terminal placement: no gate, measurement, reset, or active noise may follow
  an `EXP_VAL` op. Detector, observable, postselection, and tick metadata may.
- Live qubits only: a term may not reference a qubit that was
  single-qubit-measured after its last reset. Under this rule the Pauli
  commutes with every prior measurement projector, so the sampled
  post-measurement average equals the measurement-stripped pure-state
  expectation the analytical strategies compute. `MPP` does not affect
  liveness: the deferred lowering measures a scratch alias, and the projected
  cross terms cancel exactly.

Estimator paths:

| Path | Selection | `mean` | `variance` |
| --- | --- | --- | --- |
| Reference runner | `run_qec_program` with active noise, or as the detector-split fallback; `run_qec_program_reference` directly | per-shot exact `c * <P>` averaged over accepted shots | unbiased sample variance |
| Analytical ladder (SPD, CAMPS, tensor network) | `run_qec_program` noiseless; `run_qec_program_with_strategy` | exact `c * <P>` on the lowered unitary | `c^2 *` squared SPD truncation weight; 0.0 for CAMPS and tensor network |

Noiseless programs with detectors split into two runs: the packed compiled
sampler executes the program without its `EXP_VAL` ops, producing real sampled
measurement, detector, and observable records, while the analytical ladder
executes it without its detectors, producing the estimates. Detectors are
record metadata with no effect on the state, so the two halves compose into
one result. When either half cannot run (non-Clifford gates on the packed
half, or an analytical failure), the whole program falls back to the
reference runner.

CAMPS evaluates arbitrary X/Y/Z strings by conjugating each letter through the
signed Clifford prefix (`Y = i * X * Z` composes the two inverse-tableau rows).
Postselection composes on both paths: the reference runner averages over
accepted shots, and the analytical combiner conditions via
`<O * Pi> / <Pi>`, whose projector Z-strings live on measured aliases and so
never overlap `EXP_VAL` terms. With noise annotations the reference runner's
per-shot trajectories average to the mixed-state expectation `Tr(rho P)`. The
stabilizer-rerouted SPD path does not support `EXP_VAL`.

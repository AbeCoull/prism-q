# Native QEC Program IR

`QecProgram` in `src/qec/mod.rs` is a measurement-record IR for QEC workloads
that need detectors, logical observables, postselection, expectation metadata,
and Pauli-noise annotations before sampler lowering. It is separate from
`Circuit` so measurement-record programs do not need to fit final-measurement
OpenQASM semantics.

`QecOp` stores gates, basis measurements, MPP-style Pauli-product
measurements, resets, detector rows, observable includes, expectation-value
metadata, postselection predicates, feed-forward corrections, noise
annotations, and tick separators.
Record references can be absolute indices or `rec[-k]` style lookbacks.
Construction validates qubit bounds, gate arity, finite coordinates and
coefficients, finite probabilities, and measurement-record scope. Detector,
observable, and postselection rows can be resolved to absolute measurement
indices for later compilation into packed samplers.

## Feed-forward

`QecOp::Feedforward` conditions a body on the parity of a record list against an
expected value, the shape a detector already has. It reuses the guarded-region
contract instead of adding a second conditional mechanism: the QEC
record space and the classical bit vector are one address space, because the
reference runner writes record `i` to classical bit `i`, so a resolved record
index is directly the bit a `ClassicalCondition::Parity` reads. The op executes
as the guarded instruction `circuit::guarded` picks for its body: a `Region`
through `Backend::apply_region`, or a `Conditional` when the body is one gate.

The body admits gates and resets only. Detectors and observables index the
record space absolutely, so a measurement whose execution depended on a record
would make every later index depend on the shot.

The compiled QEC sampler evaluates a static affine map from random bits to
outcomes, which a record-conditioned branch makes depend on the sample. It
rejects by name and points at `run_qec_program_reference`, as do the deferred
lowering and the density-matrix estimator. The detector-error-model derivation
inherits the deferred lowering's rejection rather than carrying its own. The op
is built through `QecProgram::feedforward`; the native text format does not
spell it.

## Parsing

`parse_qec_program` and `QecProgram::from_text` parse the native QEC text
subset: `H`, `S`, `S_DAG`, `T`, `T_DAG`,
`CX`, `CZ`, `R`/`RX`/`RY`, `M`/`MX`/`MY`, `MR` variants, `MPP`, `DETECTOR`,
`OBSERVABLE_INCLUDE`, `POSTSELECT`, `EXP_VAL`, the Pauli-noise instructions
(`X_ERROR`, `Y_ERROR`, `Z_ERROR`, `DEPOLARIZE1`, `DEPOLARIZE2`, `PAULI_CHANNEL_1`,
`PAULI_CHANNEL_2`), `TICK`,
`QUBIT_COORDS`, `SHIFT_COORDS`, and flattened `REPEAT` blocks. The parser
resolves `rec[-k]` references while building the program. Numeric arguments on
basis measurements, such as `M(0.001)`, lower to pre-measurement Pauli flips
that affect the measurement record.

`QecProgram::to_text` writes the same subset back, and parsing its output
reproduces the op list. Consecutive gates, measurements, resets, and `MPP`
products of one kind share a line, so `MR` comes back as `M` followed by `R`.
Record references print as `rec[-k]`, which the parser resolves to absolute
indices, so a program built with lookbacks reads back with absolute references
to the same records. A leading `QUBIT_COORDS` line keeps the register width
when the highest qubit is idle. `QecOptions` are not part of the text,
zero-probability noise reads back as nothing, and a gate outside the parser's
set or a `FEEDFORWARD` op is an error rather than a lossy write.

## Lowering

`compile_qec_program_rows` lowers basis measurements and `MPP` records into the
same packed X/Z Pauli row representation used by the compiled sampler internals.
It also carries detector, observable, and postselection rows forward as
absolute measurement-record indices. Detector, observable, and postselection
projection uses `PackedShots::parity_rows`, the packed parity engine the compiled
sampler already has. The rows are a lowering artifact, not an execution engine: gate,
reset, and noise execution lives in `run_qec_program`; `EXP_VAL` has no packed-row
representation, so the row compiler rejects it and `run_qec_program` routes
such programs to the estimator paths described below.

## Execution

`run_qec_program` lowers Clifford-compatible programs into the packed
compiled sampler, compiles Pauli-noise annotations into sensitivity rows
XORed into the records, and routes `EXP_VAL` programs to estimator paths.
`run_qec_program_reference` is the per-shot state-vector correctness oracle.
The runner routing, the compiled and noisy sampling paths, the circuit
lowerings, and the result shape are covered in
[QEC program execution](./qec-programs.md).

## Expectation values

`EXP_VAL(c) P1*...*Pk` estimates `c * <P>` for the Pauli product `P` in the
program's final state and returns one `QecObservableEstimate` per op, in op
order, in `QecSampleResult::expectation_values`. The placement rules, the
estimator paths, and the analytical strategy ladder are defined in
[QEC program execution](./qec-programs.md).

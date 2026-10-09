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

## Memory-experiment generators

`src/qec/generators.rs` builds the standard memory experiments as ordinary
programs, so everything downstream (sampling, model derivation, decoding,
text export) treats them like any other. Each generator assembles the op list
and validates it once through `QecProgram::from_ops`; the incremental builders
recount records on every append, which would make a distance-13, 1000-round
program quadratic to build.

| Generator | Qubits | Round | Memory basis | Observable 0 |
| --- | --- | --- | --- | --- |
| `repetition_memory(d, r, noise)` | `d` data on even indices, `d - 1` ZZ ancillas between them | two `CX` layers, then measure and reset the ancillas | Z | data qubit 0 |
| `surface_memory(d, r, basis, noise)` | `d * d` data, then one ancilla per stabilizer | `H` on X ancillas, four `CX` layers, `H`, measure and reset | X or Z | left data column (Z), top row (X) |
| `color_memory(d, r, basis, noise)` | `(3 d^2 + 1) / 4` data, then one ancilla per face | X stabilizers in six `CX` layers, measure and reset, then the Z stabilizers likewise | X or Z | every data qubit |

`QecCircuitNoise` has four rates, each emitted only when nonzero:
`after_clifford_depolarization` adds `DEPOLARIZE1` after each `H` layer and
`DEPOLARIZE2` after each `CX` layer; `before_measure_flip_probability` and
`after_reset_flip_probability` add the flip the basis is sensitive to
(`X_ERROR` around Z operations, `Z_ERROR` around X ones); and
`before_round_data_depolarization` adds `DEPOLARIZE1` on the data at the start
of every round. `TICK` separates the layers.

Detectors compare each stabilizer with its previous round. In the first round
only stabilizers of the memory basis have detectors, and after the final data
readout in the memory basis each such stabilizer gets one more detector, the
product of its data readouts against its last ancilla record. Coordinates are
`(x, t)` for the repetition code with `x` the ancilla's qubit index, `(x, y, t)`
for the surface code with data qubit `(r, c)` at `(2c + 1, 2r + 1)` and each
ancilla at its plaquette centre, and `(x, y, t, c)` for the color code with
`c` the face color, plus 3 for Z stabilizers. The final detectors sit at
`t = rounds`.

The surface schedule visits X plaquettes NW, SW, NE, SE and Z plaquettes NW,
NE, SW, SE. That order measures the stabilizers correctly when X and Z
ancillas interleave (on every shared pair of data qubits, the Z ancilla acts
first on both or on neither), and it lays each hook error, the two-qubit error
an ancilla fault halfway through a plaquette leaves on the data, perpendicular
to the logical operator it could extend. The graphlike distance of the
decomposed model, the fewest mechanisms that flip the observable with no
detector, equals `d` for both bases at `d` = 3 and 5, as it does for the
repetition code.

The color code is the triangular 6.6.6 code: data qubits are the honeycomb
vertices inside a triangle with one boundary of each color, giving hexagons in
the bulk and weight-4 half hexagons on the boundary. A face visits its
vertices in angular order, and a vertex sits at a different angular position
in each of its three faces, so six layers measure every face without two
ancillas meeting one data qubit. X and Z stabilizers are measured in
separate halves of the round on the same ancilla, which needs no interleaving
argument. There are no flag qubits, so an ancilla fault midway through a
hexagon can leave weight-3 damage and the circuit distance can fall below
`d`. The code distance is `d`: under data noise alone the fewest mechanisms
flipping the observable undetected is exactly `d` at `d` = 3 and 5 in both
bases. A bulk data error flips three faces, and from distance 5 on some of
those hyperedges have no graphlike cover, so `decompose_graphlike` rejects the
model and the union-find decoder does not apply; a hypergraph decoder
consumes it as derived.

Measured with the union-find decoder at 100k shots, seed 42, `d` rounds and
uniform noise, failures fall with distance: the repetition code at `p = 0.01`
fails 691, 152, and 21 times at `d` = 3, 5, 7; the surface code at
`p = 0.002` fails 309 and 112 times (Z memory) and 339 and 144 times (X
memory) at `d` = 3 and 5.

The MPP-based repetition and surface memories in the benchmark and test
helpers stay as they are: they measure stabilizers directly with
phenomenological noise, a different program from these circuit-level ones,
and the seeded-record digests and benchmark history pin them.

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

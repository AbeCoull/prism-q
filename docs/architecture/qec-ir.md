# Native QEC Program IR

`QecProgram` in `src/qec/mod.rs` is a measurement-record IR for QEC workloads:
detectors, logical observables, postselection, expectation metadata and Pauli-noise
annotations, held before sampler lowering. It is separate from `Circuit` so these
programs do not have to fit final-measurement OpenQASM semantics.

`QecOp` stores gates, basis measurements, MPP-style Pauli-product measurements,
resets, detector rows, observable includes, expectation-value metadata, postselection
predicates, feed-forward corrections, noise annotations and tick separators. Record
references are absolute indices or `rec[-k]` lookbacks. Construction validates qubit
bounds, gate arity, finite coordinates, coefficients and probabilities, and
record-reference scope; the full op table is on
[QEC program execution](./qec-programs.md#program-ops-and-options). Detector,
observable and postselection rows resolve to absolute measurement indices for
compilation into packed samplers.

## Memory-experiment generators

`src/qec/generators.rs` builds the standard memory experiments as ordinary programs,
so sampling, model derivation, decoding and text export treat them like any other.
Each generator assembles the op list and validates it once through
`QecProgram::from_ops`; the incremental builders recount records on every append,
which would make a distance-13, 1000-round program quadratic to build.

| Generator | Qubits | Round | Memory basis | Observable 0 |
| --- | --- | --- | --- | --- |
| `repetition_memory(d, r, noise)` | `d` data on even indices, `d - 1` ZZ ancillas between them | two `CX` layers, then measure and reset the ancillas | Z | data qubit 0 |
| `surface_memory(d, r, basis, noise)` | `d * d` data, then one ancilla per stabilizer | `H` on X ancillas, four `CX` layers, `H`, measure and reset | X or Z | left data column (Z), top row (X) |
| `color_memory(d, r, basis, noise)` | `(3 d^2 + 1) / 4` data, then one ancilla per face | X stabilizers in six `CX` layers, measure and reset, then the Z stabilizers likewise | X or Z | every data qubit |

`QecCircuitNoise` has four rates, each emitted only when nonzero:

| Rate | Emits |
| --- | --- |
| `after_clifford_depolarization` | `DEPOLARIZE1` after each `H` layer, `DEPOLARIZE2` after each `CX` layer |
| `before_measure_flip_probability` | the flip the basis is sensitive to: `X_ERROR` before Z measurements, `Z_ERROR` before X ones |
| `after_reset_flip_probability` | the same, after resets |
| `before_round_data_depolarization` | `DEPOLARIZE1` on the data at the start of every round |

`TICK` separates the layers.

Detectors compare each stabilizer with its previous round. In the first round only
stabilizers of the memory basis have detectors. After the final data readout in the
memory basis, each such stabilizer gets one more detector: the product of its data
readouts against its last ancilla record, at `t = rounds`. Coordinates:

- repetition: `(x, t)`, `x` the ancilla's qubit index;
- surface: `(x, y, t)`, data qubit `(r, c)` at `(2c + 1, 2r + 1)`, each ancilla at its
  plaquette centre;
- color: `(x, y, t, c)`, `c` the face color, plus 3 for Z stabilizers.

The surface schedule visits X plaquettes NW, SW, NE, SE and Z plaquettes NW, NE, SW,
SE. On every data pair an X and a Z ancilla share, the Z ancilla acts first on both or
on neither, so interleaved ancillas measure the stabilizers correctly. The order also
lays each hook error (the two-qubit error an ancilla fault halfway through a plaquette
leaves on the data) perpendicular to the logical operator it could extend. The
graphlike distance of the decomposed model, the fewest mechanisms that flip the
observable with no detector, equals `d` for both bases at `d` = 3 and 5, as it does
for the repetition code.

The color code is the triangular 6.6.6 code: data qubits are the honeycomb vertices
inside a triangle with one boundary of each color, giving hexagons in the bulk and
weight-4 half hexagons on the boundary. A face visits its vertices in angular order,
and a vertex sits at a different angular position in each of its three faces, so six
layers measure every face without two ancillas meeting one data qubit. X and Z
stabilizers use the same ancilla in separate halves of the round, so no interleaving
argument is needed. There are no flag qubits: an ancilla fault midway through a hexagon
can leave weight-3 damage, and the circuit distance can fall below `d`. The code
distance is `d`; under data noise alone the fewest undetected observable-flipping
mechanisms is exactly `d` at `d` = 3 and 5 in both bases. A bulk data error flips three
faces, and from distance 5 some of those hyperedges have no graphlike cover, so
`decompose_graphlike` rejects the model and union-find does not apply. A hypergraph
decoder consumes the model as derived.

With the union-find decoder at 100k shots, seed 42, `d` rounds and uniform noise:

| Code | `p` | Failures |
| --- | --- | --- |
| repetition | 0.01 | 691, 152, 21 at `d` = 3, 5, 7 |
| surface, Z memory | 0.002 | 309, 112 at `d` = 3, 5 |
| surface, X memory | 0.002 | 339, 144 at `d` = 3, 5 |

The MPP-based repetition and surface memories in the benchmark and test helpers are a
different program: they measure stabilizers directly under phenomenological noise, and
seeded-record digests and benchmark history pin them, so they stay as they are.

## Feed-forward

`QecOp::Feedforward` conditions a body on the parity of a record list against an
expected value, the shape a detector already has. It reuses the guarded-region
contract. The QEC record space and the classical bit vector are one address space
(the reference runner writes record `i` to classical bit `i`), so a resolved record
index is the bit a `ClassicalCondition::Parity` reads. The op executes as the guarded
instruction `circuit::guarded` picks for its body: a `Region` through
`Backend::apply_region`, or a `Conditional` when the body is one gate.

The body admits gates and resets only. Detectors and observables index the record
space absolutely, so a measurement whose execution depended on a record would make
every later index depend on the shot.

The compiled QEC sampler evaluates a static affine map from random bits to outcomes. A
record-conditioned branch breaks that, except when the body is a Pauli product: a Pauli
at a fixed point flips the same later records in whichever shot it fires, like a noise
fault there, and reaches only records produced after it. `run_qec_program` therefore
samples the program without its feed-forwards, applies noise, then walks the
feed-forwards in op order, XORing each one's precomputed flip set into the shots where
its predicate holds (reading records earlier corrections already flipped). The flip
sets come from the forward Pauli frame walk the record noise uses, up to 64
corrections as lanes of one walk. A Pauli on a qubit measured since its last reset
flips nothing and is dropped, as noise there is.

Rejected by name, with a pointer to `run_qec_program_reference`: a body holding any
other gate or a reset, and a feed-forward in a program with leakage annotations. A
feed-forward program also skips the record-free parity route, since its predicates
read records. The deferred lowering's other consumers and the density-matrix estimator
reject the op outright; detector-error-model derivation inherits the deferred
lowering's rejection. Build the op with `QecProgram::feedforward`; the native text
format does not spell it.

## Parsing

`parse_qec_program` and `QecProgram::from_text` parse the native QEC text subset:
`H`, `S`, `S_DAG`, `T`, `T_DAG`, `CX`, `CZ`, `R`/`RX`/`RY`, `M`/`MX`/`MY`, `MR`
variants, `MPP`, `DETECTOR`, `OBSERVABLE_INCLUDE`, `POSTSELECT`, `EXP_VAL`, the
Pauli-noise instructions (`X_ERROR`, `Y_ERROR`, `Z_ERROR`, `DEPOLARIZE1`,
`DEPOLARIZE2`, `PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`), `TICK`, `QUBIT_COORDS`,
`SHIFT_COORDS`, flattened `REPEAT` blocks, and the instructions below, which lower onto
the same ops. `rec[-k]` references resolve while the program is built. A numeric
argument on a basis measurement, such as `M(0.001)`, lowers to a pre-measurement Pauli
flip on that record. Inverted (`!`) targets are rejected; instruction tags such as
`X_ERROR[tag](p)` are ignored.

| Instruction | Lowering |
| --- | --- |
| `SQRT_X` / `SQRT_X_DAG` | `Gate::SX` / `Gate::SXdg`, the `+pi/2` and `-pi/2` X rotations. |
| `SQRT_Y` / `SQRT_Y_DAG` | `S_DAG`, `SQRT_X` (or `SQRT_X_DAG`), `S` on the qubit. |
| `SQRT_Z` / `SQRT_Z_DAG` | `S` / `S_DAG`. |
| `C_XYZ` | `S_DAG`, `H`: the X -> Y -> Z -> X axis cycle. |
| `C_ZYX` | `H`, `S`: the inverse cycle, X -> Z -> Y -> X. |
| `CY a b` | `S_DAG b`, `CX a b`, `S b`. |
| `SWAP a b` | `CX a b`, `CX b a`, `CX a b`. |
| `ISWAP a b` / `ISWAP_DAG a b` | `S a`, `S b` (`S_DAG` for the dagger), `CZ a b`, then the three `CX` of a swap. |
| `CXSWAP a b` | `CX b a`, `CX a b` (a `CX a b` followed by a swap). |
| `SWAPCX a b` | `CX a b`, `CX b a` (a swap followed by `CX a b`). |
| `MXX` / `MYY` / `MZZ a b ...` | A `MeasurePauliProduct` over each pair, one record per pair; no measurement-error argument. |
| `MPAD v ...` | A `Pad` op per value, each one constant record. |
| `E(p)` / `CORRELATED_ERROR(p) P ...` | A `CorrelatedError` with one branch: the Pauli product fires with probability `p`. |
| `ELSE_CORRELATED_ERROR(p) P ...` | Appends a branch to the chain it directly follows, fired only when no earlier branch fired. |
| `HERALDED_ERASE(p) q ...` | A `HeraldedErase` op per target: one record, 1 when the qubit is erased, with a uniformly random Pauli applied on erasure. |

`QecProgram::to_text` writes the same subset back, and parsing its output reproduces
the op list. Consecutive gates, measurements, resets and `MPP` products of one kind
share a line, so `MR` comes back as `M` then `R`, `MZZ` as `MPP`, `SWAP` as three
`CX`, and `SQRT_Y` as `S_DAG`, `SQRT_X`, `S`. Record references print as `rec[-k]`
and parse back to absolute indices, so a program built with lookbacks reads back with
absolute references to the same records. A leading `QUBIT_COORDS` line keeps the
register width when the highest qubit is idle. `QecOptions` are not written,
zero-probability noise reads back as nothing, and a gate outside the parser's set or a
`FEEDFORWARD` op is an error rather than a lossy write.

## Lowering

`compile_qec_program_rows` lowers basis measurements and `MPP` records into packed
X/Z Pauli rows, carries detector, observable and postselection rows as absolute record
indices, and projects them through `PackedShots::parity_rows`. It is a lowering
artifact, not an execution engine; what it rejects and how the rows are laid out is on
[QEC program execution](./qec-programs.md#compiled-rows).

## Execution

`run_qec_program` lowers Clifford-compatible programs into the packed compiled sampler,
compiles Pauli-noise annotations into sensitivity rows XORed into the records, and
routes `EXP_VAL` programs to estimator paths. `run_qec_program_reference` is the
per-shot statevector oracle. Routing, the sampling paths, the lowerings and the result
shape are on [QEC program execution](./qec-programs.md).

## Expectation values

`EXP_VAL(c) P1*...*Pk` estimates `c * <P>` in the program's final state, one
`QecObservableEstimate` per op in op order, in `QecSampleResult::expectation_values`.
Placement rules, estimator paths and the strategy ladder are on
[QEC program execution](./qec-programs.md#expectation-values).

# Parser and Circuit IR

## Parser

Handwritten front end in three stages: a lexer turns source into tokens, a parser turns
tokens into a statement tree, and a walker turns the tree into `Circuit` IR. The tree
holds syntax only, so meaning is decided once, in the walker. A statement ends at its
`;` and a block at its `}`, wherever the newlines fall.

The [OpenQASM guide](../guides/openqasm.md#the-subset) tabulates what parses, what
declines, and which error each decline returns.

## Circuit IR

`Circuit` holds `num_qubits`, `num_classical_bits`, and `Vec<Instruction>`. Instructions
are an enum:

| Variant | Fields | Description |
|---------|--------|-------------|
| `Gate` | `gate`, `targets` | Gate application |
| `Measure` | `qubit`, `classical_bit` | Destructive measurement |
| `Reset` | `qubit` | Return one qubit to the zero state, mid-circuit |
| `Barrier` | `qubits` | Synchronization barrier |
| `Conditional` | `condition`, `gate`, `targets` | Classical-controlled gate |
| `Region` | `Box<GuardedRegion>` | Classical-controlled span of instructions |
| `Save` | `spec`, `qubits`, `label` | Record the state at this point |

Targets are a `SmallVec<[usize; 4]>`, stored inline up to 4 qubits, so a typical gate
allocates nothing.

### Save points

`Save` records the state at its position. Its `qubits` field lists the whole register,
which makes it a fusion barrier: every pass reads an instruction's qubits to decide what
may move past it, so a save blocks reordering both ways with no rule of its own.

No backend implements a save. The runner splits the stream at each save point, hands the
backend the segments between, and reads the state through the export the spec names:
`export_statevector` for `StateVector`, `probabilities` for `Probabilities`,
`reduced_density_matrix` over the whole register for `DensityMatrix`. A backend that
cannot produce that form fails the run with its own message. A route that rewrites the
circuit instead of executing it (the compiled samplers, the stabilizer-rank engine, the
Pauli-propagation engines) declines a circuit carrying one.

Only `Simulate::run` returns the records; every other terminal declines a circuit with
save points rather than discarding what they recorded.

### Guarded regions

`Region` carries a condition and a body that runs, in order, only when the condition
holds against the classical bits as they stand when control reaches it. The body admits
any instruction, measurement and reset included, and nests to `MAX_REGION_DEPTH`. It is
boxed, so `Instruction` stays at 96 bytes.

`GuardedRegion` caches the sorted union of the qubits its body touches, nested bodies
included. Passes read that set instead of re-walking the body: fusion flushes exactly
those qubits at the boundary and is transparent elsewhere, and no pass fuses across the
boundary because a region is not an `Instruction::Gate`.

`fuse_region_bodies` runs the same pipeline over each body as an ordinary instruction
list on the same register, nested bodies included. Fusion stays inside the body, and
fusing it as a unit is sound because it executes as a unit.

`Conditional` is the single-gate lowering of the same construct. `if (c) x q[0];` keeps
that form, so the common guarded gate allocates nothing; anything else becomes a
`Region`. Build either through `circuit::guarded`, which picks the form and returns
`None` for an empty body.

A circuit holding a region runs once per shot: measurement-conditioned execution has no
single distribution to sample, so the compiled samplers reject it and the run replays. A
noiseless run on the host statevector replays once per distinct outcome history instead,
splitting the shots binomially at each measurement until only measurements remain, then
drawing each branch's readout from its state. Routes requiring a unitary circuit
(adjoint gradients, exact expectation values, Pauli propagation, stabilizer-rank
probabilities) reject a region as they reject a bare conditional. A noise model keeps one
event slot per top-level instruction, then a slot for every guarded body instruction and
every conditional's gate; a body slot fires only on the shots its instruction runs. Only
routes holding one state per shot serve such a model.

`Circuit::fold_static_guards` runs once per shots or counts call and resolves guards that
cannot depend on a measurement. A condition reading only bits no earlier measurement
writes depends only on the initial classical state, which every backend zeroes, so the
guard is dropped or inlined, and a circuit whose only guard can never fire returns to
terminal sampling. A circuit with no guard borrows through unchanged.

### Condition language

`ClassicalCondition` is a pure function of a bit slice: `BitIsOne`, `BitIsZero`,
`RegisterEquals` and `RegisterNotEquals` over a contiguous range read as `u64`, and
`Parity` over an arbitrary bit set compared against an expected value. The bit set is
boxed, so the enum stays at the size the register variants set. The language is closed
under negation (`ClassicalCondition::negate`), so `else` lowers without growing the
instruction.

`else` emits the `then` guard and a second guard with the negated condition. `switch`
emits one `RegisterEquals` guard per case label, with the `default` arm nested once per
label to spell the conjunction of their negations. Both lowerings re-read the classical
bits after an earlier body has run, so the parser rejects a source whose body measures
into a bit its own guard reads.

## Gate enum

`Gate` is a `Clone` enum kept at 16 bytes. Simple variants carry parameters inline;
composite variants `Box` their payload.

```admonish warning title="Keep the enum at 16 bytes"
Adding inline data larger than 16 bytes pollutes cache lines and has caused 40-130%
regressions. Check `size_of::<Gate>()` after adding a variant, and `Box` large payloads.
```

| Variant | Data | Size |
|---------|------|------|
| `Id`, `X`, `Y`, `Z`, `H`, `S`, `Sdg`, `T`, `Tdg`, `SX`, `SXdg` | None | 16B |
| `Rx(f64)`, `Ry(f64)`, `Rz(f64)`, `P(f64)`, `Rzz(f64)` | Inline f64 | 16B |
| `Cx`, `Cz`, `Swap` | None | 16B |
| `QftBlock { start: u8, num: u8, inverse: bool, swaps: bool, big_endian: bool }` | Inline range and form flags, the one boxless composite | 16B |
| `Cu(Box<[[Complex64; 2]; 2]>)` | Boxed 2×2 | 16B |
| `Mcu(Box<McuData>)` | Boxed matrix + control count | 16B |
| `Fused(Box<[[Complex64; 2]; 2]>)` | Boxed pre-fused 1q matrix | 16B |
| `Fused2q(Box<[[Complex64; 4]; 4]>)` | Boxed pre-fused 2q matrix | 16B |
| `MultiFused(Box<MultiFusedData>)` | Batched 1q gates for tiled pass | 16B |
| `Multi2q(Box<Multi2qData>)` | Batched 2q gates for tiled pass | 16B |
| `BatchPhase(Box<BatchPhaseData>)` | Batched cphase with shared control | 16B |
| `BatchRzz(Box<BatchRzzData>)` | Batched ZZ rotations | 16B |
| `DiagonalBatch(Box<DiagonalBatchData>)` | Mixed diagonal 1q/2q batch | 16B |
| `PauliRot(Box<PauliRotData>)` | Multi-qubit Pauli rotation, boxed angle plus letters | 16B |
| `MultiPauliRot(Box<MultiPauliRotData>)` | Batched Pauli rotations as masks and angles for tiled pass | 16B |
| `Unitary(Box<UnitaryData>)` | Caller-supplied dense `2^k x 2^k` matrix, boxed with its width | 16B |

`Unitary` is the one variant built from a caller's matrix, through `Gate::unitary`. It
checks `U U^dagger = I` to 1e-9, then lowers to an existing variant where one fits: a
named gate or `Fused` at one qubit; a named two-qubit gate, `Cu` or `Fused2q` at two;
`Mcu` at any width when the matrix is the identity outside its trailing 2x2 block. Only a
dense matrix on three or four qubits reaches the variant, so Clifford recognition and
the controlled-gate kernels keep firing on the forms they know. `Circuit::add_unitary`
adds the one lowering that needs target indices: a diagonal whose phases factor into
one- and two-body terms becomes a `DiagonalBatch`.

Fusion neither builds nor absorbs a `Unitary`. Blocking follows instruction targets, so
the gate is a barrier over its own qubits while runs on other qubits fuse around it.

```admonish note title="Qubit ordering"
`q[0]` is the least significant bit. Applying `x q[0]` produces state index 1, not 2.
```

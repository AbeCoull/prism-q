# Dynamic Programs

A `Circuit` is a finite instruction list. Guarded regions, `else`, `switch`, and
bounded `for` loops all fit in one, because each lowers to a span of the list at parse
time. A program whose instruction count depends on what it measures does not: a
`while` loop that repeats until an outcome comes up, or a counter that a measurement
increments, has no finite list to lower to. `DynamicProgram` holds those programs as a
control-flow graph and runs them once per shot.

The rest of the engine is unchanged. Every block holds an ordinary `Circuit`, every
pass that works on a circuit works on a block, and a program with no runtime control
flow never reaches the graph walker at all.

## The program

| Part | Shape |
|------|-------|
| `BasicBlock` | A `Circuit` over the full register, then a list of `Action`s, then a `Terminator` |
| `Terminator` | `Jump(block)`, `Branch { condition, then, otherwise }`, or `End`. Block 0 is the entry |
| `Action` | `Assign { var, value }` stores into a variable; `Rotation { kind, targets, angle }` applies `rx`, `ry`, `rz`, `p` or `rzz` at an angle computed at runtime |
| `Variable` | A name, a `ClassicalType`, and the value every shot starts from |
| `ClassicalExpr` | Constants, variables, one classical bit, a register of up to 64 bits read as an unsigned integer, and unary and binary operators |

A block runs its circuit first and its actions after. Bits a measurement writes in the
circuit are visible to the actions and the terminator, and the circuit's own guarded
regions read the bits as they stand, so measurement-conditioned code inside a block
keeps the guarded-region form. Variables are never visible to a circuit. That split is
what keeps fusion's dependency model intact: fusion orders instructions by the qubits
they touch, and the only classical state a circuit reads is still written by a
measurement on a qubit the same circuit orders.

`ClassicalType` is `bool`, `int[n]` and `uint[n]` for `n` from 1 to 64, `float`, and
`angle`. A store converts the value to the declared type: integers truncate toward zero
and then wrap to their width, and an angle reduces into `[0, 2 pi)`. Arithmetic on two
integers stays integral, with `/` truncating; a float on either side makes it a float.
Comparisons and the logical operators return a `bool`, and `&&` and `||` short-circuit.
An integer divided by zero fails the run with `InvalidParameter`.

`DynamicProgram::new` validates the whole graph: every block a terminator names, every
variable an expression or assignment reads, and every qubit and classical bit an
instruction or expression touches. It rejects a save point, since a shot result has
nowhere to return one, and a bitwise operator applied to a float.

## Building one

`DynamicProgramBuilder` takes structured control flow. `begin_while` opens a loop as an
empty header block that tests the condition and branches to the body or past the loop;
the body jumps back to the header. `begin_if` opens a branch to a join block, and
`begin_else` either switches the open `if` to its other arm or reopens the `if` that
`end` has just closed. `break_loop` and `continue_loop` jump to the innermost loop's exit
or header and continue in a fresh block that nothing reaches until a later construct
joins into it.

An instruction appended after an action starts a new block, since a block's actions run
after its whole circuit. `build` then threads every jump through empty blocks, merges
each block into the one block that jumps to it alone, and drops what the entry cannot
reach, so fusion sees the longest straight-line circuits the control flow allows.

## Lowering OpenQASM

`openqasm::parse_dynamic` reads everything `openqasm::parse` reads plus `while`,
`break`, `continue`, the full classical operator set, and classical bits read as
values. It walks the same statement tree with the same evaluator, with one extra state:
the graph being built.

Before the walk, a pass over the tree decides which classical variables live at
runtime. A variable does when it is assigned from a measured bit or another runtime
variable, or when it is written under a `while` body or a measured branch that its
declaration sits outside. Iterating that rule to a fixed point settles chains of
assignments. Every other variable folds at parse time exactly as `parse` folds it. A
variable declared and written inside one loop body stays a parse-time value, because
each pass re-declares it, and so does one written only in an unrolled `for`.

The evaluator then streams each statement's instructions into the block being built. An
`if` takes one of three forms:

- Folded at parse time, when its condition reads no bit and no runtime variable.
- A guarded region, when its condition is one the instruction list can carry and
  nothing in its bodies opens a block. This is the form `parse` produces.
- A branch in the graph otherwise: a condition over a runtime variable or a general
  expression, a body that writes a runtime variable, loops, or applies a runtime angle,
  and an `else` whose `if` body measures its own condition, which the guard pair could
  not lower soundly.

A `switch` over a runtime variable, or with an arm that opens a block, becomes a chain
of branches. Because a program with no runtime control flow never opens a block, it
lowers to one block holding the instruction list `parse` returns.

## Running one

`simulate_program(&program).seed(s).shots(n)` returns the same `ShotsResult`, and
`sample_counts` the same `CountsResult`, that a circuit run returns.

A program whose graph is a single block with no actions is its circuit, and runs through
the circuit path unchanged: same routing, same sampling shortcuts, same seeded shots.

Any other program is prepared once. Routing reads every block's instructions together,
with each runtime rotation standing in at a generic angle, so a program whose blocks
hold only Clifford gates runs on the stabilizer tableau and a runtime angle counts as a
non-Clifford gate. Each block is then expanded and fused for the chosen backend, once.
Every shot initializes one backend, resets the variables to their initial values, and
walks the graph from block 0, applying each block's fused circuit, evaluating its
actions against that shot's bits and variables, and following its terminator.

Shot `i` draws from `mix_seed(seed, i)`, the seed the per-shot circuit routes give
shot `i`, and shots split across Rayon workers under the same rule those routes use. The
host statevector keeps one backend per worker and reseeds it per shot, so its buffer is
allocated once per worker rather than once per shot; every other backend is built per
shot.

A shot that runs more blocks than its step bound stops with `PrismError::StepLimit`
naming the loop it was in. The bound is one million blocks by default and
`SimulateProgram::max_steps` sets another. Every block entered counts, the empty loop
header included.

### Backend coverage

Every backend that holds a per-shot state walks a graph through its ordinary
`apply_instructions`, so statevector, stabilizer, sparse, MPS, product state, factored,
factored stabilizer, tensor network, and density matrix all run dynamic programs.
Stabilizer rank, the Pauli-propagation engines, and the distributed
statevector decline with `IncompatibleBackend`: the first two keep no per-shot state, and
the distributed backend runs its ranks in lockstep on one circuit.

### What stays declined

- A runtime value where a parse-time constant belongs: a qubit index, a loop bound, a
  register size, a `def` argument, and the angle of any gate but the five rotations.
- A builtin function (`sin`, `sqrt`, and the rest) over a runtime value. Only operators
  run at runtime.
- A `bit` or `qubit` register declared inside a `while` body, and an `array` a loop or
  measurement writes.
- A measurement straight into a variable; measure into a bit and assign the bit.
- `break` and `continue` inside a `for`, which unrolls at parse time.
- `input` parameters and noise models. A dynamic program has no parameter binding and
  no noise slots.

Writing a classical bit from an expression is also out of reach: bits are written by
measurement only, which no backend relaxes.

## Cost

A dynamic program runs once per shot, so it pays the per-shot cliff every guarded
circuit pays, plus one evaluation per action and terminator. Fusion runs within a block
and never across a block boundary, which is the cost a branch imposes: the loop body and
the code around it fuse separately. `dynamic/rus/{10,16}` price a repeat-until-success
loop at 1000 shots on 10 and 16 qubits.

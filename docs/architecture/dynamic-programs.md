# Dynamic Programs

A `Circuit` is a finite instruction list. Guarded regions, `else`, `switch` and bounded
`for` loops fit in one because each lowers to a span of the list at parse time. A
`while` loop that repeats until an outcome comes up, or a counter a measurement
increments, has no finite list to lower to. `DynamicProgram` holds such programs as a
control-flow graph and walks it per shot, or per outcome history on the host
statevector.

Every block holds an ordinary `Circuit`, every circuit pass works on a block, and a
program with no runtime control flow never reaches the graph walker.

## The program

| Part | Shape |
|------|-------|
| `BasicBlock` | A `Circuit` over the full register, then a list of `Action`s, then a `Terminator` |
| `Terminator` | `Jump(block)`, `Branch { condition, then, otherwise }`, or `End`. Block 0 is the entry |
| `Action` | `Assign { var, value }` stores into a variable; `Rotation { kind, targets, angle }` applies `rx`, `ry`, `rz`, `p` or `rzz` at an angle computed at runtime; `WriteBits { bits, value }` writes the low bits of an integer or bool to classical bits, through `Backend::set_classical_bit` |
| `Variable` | A name, a `ClassicalType`, and the value every shot starts from |
| `ClassicalExpr` | Constants, variables, one classical bit, a register of up to 64 bits read as an unsigned integer, and unary and binary operators |

A block runs its circuit, then its actions. Bits a measurement writes in the circuit
are visible to the actions and the terminator. The circuit's own guarded regions read
the bits as they stand, so measurement-conditioned code inside a block keeps the
guarded-region form. Variables are never visible to a circuit. This keeps fusion's
dependency model intact: fusion orders instructions by the qubits they touch, and the
only classical state a circuit reads is written by a measurement on a qubit the same
circuit orders.

`ClassicalType` is `bool`, `int[n]` and `uint[n]` for `n` from 1 to 64, `float[n]` for
`n` of 16, 32, 64 or 128, and `angle`. A store converts to the declared type. Integers
truncate toward zero and wrap to their width, a 16- or 32-bit float rounds to half or
single precision, and an angle reduces into `[0, 2 pi)`. Arithmetic on two integers
stays integral, `/` truncating; a float on either side makes it a float. `%` and the
bitwise operators take integers only, and construction rejects a float operand.
Comparisons and logical operators return a `bool`, and `&&` and `||` short-circuit.
Integer division by zero fails the run with `InvalidParameter`.

`DynamicProgram::new` validates every block a terminator names, every variable an
expression or assignment reads, and every qubit and classical bit an instruction or
expression touches. It rejects a save point, since a shot result has nowhere to return
one, and a bitwise operator on a float.

## Building one

`DynamicProgramBuilder` takes structured control flow:

- `begin_while` opens an empty header block that tests the condition and branches to
  the body or past the loop; the body jumps back to the header.
- `begin_if` opens a branch to a join block. `begin_else` switches the open `if` to its
  other arm, or reopens the `if` that `end` just closed.
- `break_loop` and `continue_loop` jump to the innermost loop's exit or header and
  continue in a fresh block that nothing reaches until a later construct joins into it.

An instruction appended after an action starts a new block, since a block's actions run
after its whole circuit. `build` threads every jump through empty blocks, merges each
block into the one block that alone jumps to it, and drops what the entry cannot reach.
Fusion then sees the longest straight-line circuits the control flow allows.

## Lowering OpenQASM

`openqasm::parse_dynamic` walks the same statement tree as `openqasm::parse`, with the
same evaluator, and builds the graph as it goes. The
[OpenQASM guide](../guides/openqasm.md#dynamic-programs) lists what it reads beyond
`parse`, and [Dynamic declines](../guides/openqasm.md#dynamic-declines) what it still
rejects.

A pass before the walk decides which classical variables live at runtime. A variable
does when it is assigned from a measured bit or another runtime variable, or written
under a `while` body or a measured branch its declaration sits outside. The rule
iterates to a fixed point, which settles chains of assignments. Every other variable
folds at parse time as `parse` folds it. A variable declared and written inside one loop
body stays a parse-time value, because each pass re-declares it, and so does one
written only in an unrolled `for`.

A `for` still unrolls when a measurement decides a `break` or `continue` in it, but its
passes become blocks in sequence: `continue` jumps to the next pass and `break` past the
last. The pass over the tree marks such a loop. Every variable its body writes from
outside lives at runtime, since a pass may stop before the write.

An exit every path reaches stops the unrolling at parse time, as under `parse`. The
exception is a measured `continue` earlier in the pass that sent some paths on to the
next one; then the exit jumps and the unrolling goes on. A `return` likewise leaves the
code after the loop in place when a measured `break` reached the loop's exit. A measured
`end` jumps to a block that ends the shot, and a `def` holding an `end` runs as a call so
the same holds inside it. An `end` that no branch, `return` or measured exit separates
from the program top still stops the parse there, through every call.

The evaluator streams each statement's instructions into the current block. An `if`
takes one of three forms:

- Folded at parse time, when its condition reads no bit and no runtime variable.
- A guarded region, the form `parse` produces, when the instruction list can carry the
  condition and nothing in the bodies opens a block.
- A branch in the graph otherwise. That covers a condition over a runtime variable or a
  general expression, and a body that writes a runtime variable, loops, or applies a
  runtime angle. It also covers an `else` whose `if` body measures its own condition,
  which the guard pair could not lower soundly.

A `switch` over a runtime variable, or with an arm that opens a block, becomes a chain
of branches. A program with no runtime control flow never opens a block, so it lowers
to one block holding the instruction list `parse` returns.

Runtime subroutines expand into the same graph. Each call copies its classical
arguments into local variables and gets its own exit block. A return assigns the typed
result and jumps to that exit, including from a loop inside the callee; loop exits
cannot cross the call boundary. Locals are initialized each time the call executes,
and every reachable path through a typed call must return. Definitions keep their
lexical constants and cannot see the caller's variables.

Calls in a `while` condition expand into its header, so their effects repeat at each
test. A call on the right of a runtime `&&` or `||` expands into a branch, which keeps
short-circuit evaluation. Static calls keep the instruction-list path.

## Running one

`simulate_program(&program).seed(s).shots(n)` returns the same `ShotsResult`, and
`sample_counts` the same `CountsResult`, as a circuit run.

A program that is a single block with no actions is its circuit and runs through the
circuit path unchanged: same routing, same sampling shortcuts, same seeded shots.

Any other program is prepared once. Routing reads every block's instructions together,
with each runtime rotation standing in at a generic angle. A program whose blocks hold
only Clifford gates runs on the stabilizer tableau, and a runtime angle counts as
non-Clifford. Each block is expanded and fused for the chosen backend once. Every shot
initializes one backend, resets the variables, and walks the graph from block 0. At each
block it applies the fused circuit, evaluates the actions against that shot's bits and
variables, and follows the terminator.

On the host statevector the shots walk the graph together, by the outcome branching
that circuits with mid-circuit measurements use. At each measurement the shots on a
branch split by a binomial draw on the outcome probability, and each non-empty outcome
carries its own state, variables and step count. Variables, loop tests and branch
conditions are functions of the outcome history, so every shot on a branch takes the
same path. A prefix all shots share, the entry block included, evolves once. The
live-state budget and the per-shot fallback past it are the circuit walk's. Only a
block that ends the graph with no actions is cut after its last measurement or drawn
as a terminal readout; anywhere else a later block may still read the state.

On every other backend shot `i` draws from `mix_seed(seed, i)`, the seed the per-shot
circuit routes give shot `i`, and shots split across Rayon workers by the same rule
those routes use. Each backend is built per shot.

A shot that runs more blocks than its step bound stops with `PrismError::StepLimit`
naming the loop it was in. The bound defaults to one million blocks;
`SimulateProgram::max_steps` sets another. Every block entered counts, the empty loop
header included.

### Backend coverage

Every backend that holds a per-shot state walks a graph through its ordinary
`apply_instructions`: statevector, stabilizer, sparse, MPS, product state, factored,
factored stabilizer, tensor network and density matrix. Stabilizer rank and the
Pauli-propagation engines keep no per-shot state, and the distributed statevector runs
its ranks in lockstep on one circuit, so all three decline with `IncompatibleBackend`.

### Noise

`SimulateProgram::noise` takes a `NoiseBuilder`, not a `NoiseModel`. A model is laid
out slot by slot against one circuit, and a program holds one circuit per block, as
the builder's merging decides. The rules compile against each block's circuit, guarded
regions included, and against each runtime rotation as a one-gate circuit.

The compiled noisy sampler and the error chain complex cannot follow control flow, and
the density matrix holds no per-shot state to branch on, so a noisy program runs one
trajectory per shot. Routing reads the summary circuit and every compiled event as a
noisy circuit's trajectory route does. Pauli noise on Clifford blocks stays on the
tableau, other channels move to the statevector family, and a channel the chosen
backend cannot sample is rejected before a shot starts. Blocks run unfused so each
event stays on its instruction. The host statevector stops branching by outcome
history, since noise makes every history its own. Shot `i` draws its noise from the
stream a circuit trajectory gives shot `i`. Readout error flips the finished record, so
a branch reads the bit the measurement wrote, as a guarded region does.

Leak flags and quasi-static draws belong to the shot, not the block. Each shot draws
its quasi-static normals once before its first event, for every source any block or
rotation names. A leaked qubit stays leaked into later blocks until a reset or seepage
returns it. `ShotsResult::leaked` carries the heralds as for a circuit. A program whose
arms are written as guarded regions of one circuit draws the same values in the same
order, so both give the same seeded shots (`tests/integration/noisy_programs.rs`).

Declined with `IncompatibleBackend`: the density matrix, which holds the mixture rather
than a trajectory, and an over-rotation rule (fixed or drifting) matching a runtime
rotation, whose angle exists only once the shot evaluates it.

A program with no runtime control flow builds the rules against its one circuit and
runs as that circuit does under `Simulate::noise`, mixture routes included.

## Cost

Off the host statevector a dynamic program runs once per shot. It pays the per-shot
cliff every guarded circuit pays, plus one evaluation per action and terminator. On the
host statevector it pays one evolution per distinct outcome history. Fusion runs within
a block and never across a boundary, so the loop body and the code around it fuse
separately. `dynamic/rus/{10,16}` price a repeat-until-success loop at 1000 shots on 10
and 16 qubits.

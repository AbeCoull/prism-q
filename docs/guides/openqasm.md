# OpenQASM Support

PRISM-Q parses a practical subset of OpenQASM 3.0, with backward compatibility for common
2.0 syntax. Text is lexed, parsed to a statement tree, and evaluated into the
[Circuit IR](../architecture/ir.md); the tree holds syntax only, so a program's meaning is
decided once, in the evaluator.

[The subset](#the-subset) below states what parses and what declines. A construct that is
valid OpenQASM but outside the subset returns `UnsupportedConstruct` naming it, never a
panic and never a silent drop.

## Parsing and running

```rust
use prism_q::circuit::openqasm;
use prism_q::simulate;

# let qasm_str = "OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];";
let circuit = openqasm::parse(qasm_str).expect("parse error");
let result = simulate(&circuit).seed(42).run().unwrap();
```

`run_qasm(qasm, seed)` parses and simulates in one call.

## Exporting

```rust
use prism_q::circuit::qasm_export;

# let circuit = prism_q::circuits::qft_circuit(4);
let qasm = qasm_export::to_qasm3(&circuit).expect("export error");
```

Export inverts the parser: re-parsing the result gives back the same instruction
stream, with gate matrices agreeing to floating-point round-off and inline angles
(`rx`, `rz`, `rzz`, `p`) surviving exactly. Qubits come out as one `qubit[n] q`
register, classical bits as one `bit[m] c` register, split only where a condition
compares against a register narrower than the whole.

A circuit that has been through [fusion](../architecture/fusion.md) is not
exportable: fused blocks, tiled multi-gate passes, and diagonal batches carry
matrices with no OpenQASM spelling, and `to_qasm3` returns `ExportUnsupported`
naming the instruction index. Export the circuit before fusing it, or the template
a `PreparedCircuit` binds. `QftBlock` and `PauliRot` are the exceptions: export
expands the first to its textbook Hadamard, controlled-phase, and swap sequence
and the second to its CNOT-ladder lowering on the way out. On the CPU statevector, an
expanded QFT parsed back in runs as one block again (see
[QFT recognition](../architecture/fusion.md#qft-recognition)).

## Declarations and measurement

```text
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;          // OpenQASM 3.0 register
bit[3] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0]; // OQ3 measurement
```

OpenQASM 2.0 syntax also works: `qreg q[3];` / `creg c[3];` declarations and
`measure q[0] -> c[0];` measurement.

A `bit` declaration can take its value at once. `bit b = measure q[0];` measures
into it, and `bit[2] f = "00";` or `bit b = 0;` spell out the cleared register it
starts as anyway. A bit string is written most significant bit first. An initializer
with a set bit, and any assignment to a bit (`c = 2;`, `c[0] = 1;`, `c <<= 1;`), returns
`UnsupportedConstruct` under `parse`, since nothing in the instruction list writes a
classical bit; [`parse_dynamic`](#dynamic-programs) writes them.

A name may open with any Unicode letter, so `gate cphase(θ) a, b { ... }` and
`qubit γ;` parse.

A program declares at most 2^24 qubits and 2^24 classical bits, physical `$k` indices
included; past that the parse fails with `PrismError::Parse`.

`output bit[3] c;` declares the register and marks it as the program's result.
Every classical bit is reported either way, so the marking costs nothing and
changes nothing.

### Slices and aliases

An index may be a single position, an inclusive range with an optional step in
the middle, or an explicit set. A sliced register broadcasts exactly as a whole
one does. A negative index counts from the end: `q[-1]` is the last qubit and
`q[1:-1]` runs from the second to the last.

```qasm
OPENQASM 3.0;
qubit[6] q;
h q[0:2];         // q[0], q[1], q[2]
h q[0:2:5];       // q[0], q[2], q[4]
h q[5:-1:4];      // q[5], q[4]
h q[{1, 4}];      // q[1], q[4]
cx q[0:1], q[2:3];
```

`let` names qubits or bits in the order it writes them, `++` joins operands,
and an alias is itself sliceable, so a slice of an alias follows the alias's
order rather than the register's.

```qasm
let ends = q[5] ++ q[0];
let pair = q[0:3];
h ends[0];        // q[5]
h pair[1:2];      // q[1], q[2]
```

### Classical variables

`int`, `uint`, `bool`, `float` and `angle` declarations, with an optional width,
are folded at parse time. The value then reads anywhere a literal would: as an
index, a loop bound, a gate angle, or a condition operand. `const` marks a name
that cannot be assigned again.

```qasm
OPENQASM 3.0;
qubit[4] q;
bit[4] c;
const int width = 4;
float theta = pi / 8;
int cursor = 0;
for int k in [0:width - 1] { rx(theta * k) q[k]; }
cursor += 2;
h q[cursor];
```

Each type holds what the language says it holds. An `int` or `uint` is an exact
integer that wraps at its width, 64 bits when none is given, and `/` or `%` between two
integers truncates toward zero, so `1 / 2` is `0` while `1.0 / 2` is `0.5`. A `float[n]`
is an IEEE 754 float of 16, 32, 64 or 128 bits: `float[16]` and `float[32]` round to half
and single precision, and `float[128]` holds a double, since nothing here computes in
quad precision. A float takes `+ - * /` and `**`; `%` takes integers, and `mod(a, b)` is
the float remainder. An `angle[n]` holds the nearest multiple of `2 pi / 2^n`, taken
modulo `2 pi`. A cast such as `int[8](x)`, `uint[2](c)`, `bool(x)` or `angle[3](t)`
converts the same way, and `a[i]` on an integer reads one of its bits. A float cast to an
integer drops its fraction and wraps at the integer's width, so `int[8](1e30)` is `0`;
past about `1.7e38` it saturates first, and NaN reads as `0`. A float has no bit pattern
to cast to `bit[n]`; cast it to `uint` first. An expression on its own, `a == 2;`, is
evaluated and discarded.

A declaration inside a `for` body binds for that pass only: the scope the body
opened is dropped at the end of each iteration, while a write to a name declared
outside the loop carries on to the next pass and past the loop.

An `if`, `else` or `switch` body opens a scope too, but whether it runs is decided by a
measurement, and a classical variable is folded here at parse time. Such a body may
declare and write its own variables, but a write to one declared outside it returns
`UnsupportedConstruct`: the value would otherwise change whatever the measurement read.
Holding a value that does depend on one needs a control-flow graph, which
[`parse_dynamic`](#dynamic-programs) builds.

An `array` of any of those element types takes a size per dimension, fixed at parse
time, and folds the same way. An element reads as `a[i, j]` or `a[i][j]` wherever a
value would, and takes plain and compound assignment. A write made inside a loop
outlives the pass, which is what filling an array needs.

```qasm
array[float[64], 2, 2] angles = {{0.1, 0.2}, {0.3, 0.4}};
array[int, 4] order;
for int k in [0:3] { order[k] = 3 - k; }
rx(angles[1, 0]) q[order[0]];   // rx(0.3) q[3]
```

Elements of `bit` belong in a `bit[n]` register, so `array[bit, n]` declines. A `def`
body sees the program's `const` arrays and no others, as it sees no other non-constant
global.

### Timing

A simulation has no clock, so timing parses and then has no effect. `delay` is the
identity on the qubits it names, or on every qubit when it names none, and emits no
instruction, so fusion merges across it as if it were absent. A `box`, with or without
a duration, runs its body in place: its edges carry no barrier, since a box only fixes
when its contents run, and the names its body declares go out of scope with it.

```qasm
OPENQASM 3.0;
qubit[2] q;
const duration pulse = 40ns;
duration settle = 2 * pulse + 1us;
stretch slack;
h q[0];
delay[settle] q[0];
delay[slack] q;
rz(settle / pulse) q[1];   // a ratio of durations is a number: 27
box[settle] {
  cx q[0], q[1];
}
```

Literals take `dt`, `ns`, `us` (or `µs`), `ms` and `s`. Durations add, subtract,
scale by a number and divide into a number, all folded at parse time. A ratio needs
both sides in SI units or both in `dt`, since only a backend's sample period relates
the two. A `stretch` is sized by a scheduler, so it may stand wherever a delay takes a
length but is rejected where its value would be read.

## Input parameters

An `input` declaration names a parameter slot. `openqasm::parse_parametric`
returns the template circuit alongside the `Parameters` that binds it, in
declaration order and under the declared names:

```qasm
OPENQASM 3.0;
input float[64] theta;
input float[64] phi;
qubit[2] q;
h q[0];
rx(theta) q[0];
cx q[0], q[1];
rz(phi) q[1];
```

```rust
# use prism_q::circuit::{openqasm, qasm_export::to_qasm3};
# let qasm = "OPENQASM 3.0; input float[64] theta; input float[64] phi; qubit[2] q;
#     h q[0]; rx(theta) q[0]; cx q[0], q[1]; rz(phi) q[1];";
let (template, params) = openqasm::parse_parametric(qasm)?;
let bound = params.bind(&template, &[0.41, 1.27])?;
let text = to_qasm3(&bound)?;   // angles written out, no `input` line
# Ok::<(), prism_q::PrismError>(())
```

Several gates may read one input, which is the weight sharing `Parameters`
already models: `rx(theta) q;` over a register links every gate it produces to
the same slot, and binding writes one angle to each.

`parse` itself rejects a program that declares an input, because it has nowhere
to take the value and a zero would be a quiet wrong answer. Feed those through
`parse_parametric`, or through `PreparedCircuit` for a sweep.

An input binds an angle whole, so it may only be the entire angle argument of a
directly named parametric gate at the top level. `rx(2 * theta)`, an input on a
gate carrying no rotation angle, one reaching a `gate`, `def`, `for`, or guarded
body, and one on a modified gate all return `UnsupportedConstruct` naming the
reason rather than binding something the source did not mean.

## Dynamic programs

`openqasm::parse` returns a `Circuit`, a finite instruction list, so it declines a
`while` loop and a classical value a measurement decides. `openqasm::parse_dynamic`
reads the same language plus those, and returns a `DynamicProgram`: basic blocks of
ordinary circuits joined by branches on classical expressions, run once per shot by
`simulate_program`.

```rust
use prism_q::circuit::openqasm;
use prism_q::simulate_program;

let source = "
    OPENQASM 3.0;
    qubit[1] q;
    bit[1] c;
    uint[8] tries = 1;
    h q[0];
    c[0] = measure q[0];
    while (c[0]) {
        reset q[0];
        h q[0];
        c[0] = measure q[0];
        tries += 1;
        if (tries == 10) break;
    }";
let program = openqasm::parse_dynamic(source)?;
let shots = simulate_program(&program).seed(42).shots(1000)?;
# Ok::<(), prism_q::PrismError>(())
```

On top of what `parse` reads, `parse_dynamic` accepts:

- `while (condition) { ... }`, with `break` and `continue`.
- Any condition or `switch` target over classical bits, where `parse` takes only those
  a bit or register guard spells: `c[0] && c[1]` and `c + 1 == 2` run here.
- A classical bit read as a value (`c[0]`) and a bit register read as an unsigned
  integer (`c`), up to 64 bits.
- A write to a variable under a measured `if` or `switch`, which is evaluated when the
  program runs rather than declined.
- A write to classical bits: `bit[2] f = "11";`, `c = c + 1;`, `c[0] = c[1] ^ c[2];`,
  `c <<= 1;`. A write takes an integer or a bool, and a bit register wider than 64 bits
  takes one only from a parse-time value.
- A runtime value as the angle of `rx`, `ry`, `rz`, `p` or `rzz`.

A variable lives at runtime only when it has to: when a measured bit or another runtime
variable reaches it, or when a loop or a measured branch writes it from outside the
scope it was declared in. Every other variable folds at parse time as it does under
`parse`, so a program `parse` accepts gives the same circuit under `parse_dynamic`, and
runs with the same seeded shots. Runtime `int` and `uint` variables wrap at their
declared width as folded ones do, and runtime floats round to their width. A runtime
`angle` holds a double whatever its width, so `angle[n]` steps apply only to folded values.

A shot that runs more than a million blocks stops with `PrismError::StepLimit` naming
its loop, rather than hanging; `SimulateProgram::max_steps` sets another bound. The
architecture page on [dynamic programs](../architecture/dynamic-programs.md) covers the
block structure, the backends that run it, and what it costs.

`SimulateProgram::noise` takes `NoiseBuilder` rules, which compile against every block
and runtime rotation, and runs one noisy trajectory per shot. The loop above with
amplitude damping after each `h`:

```rust
# use prism_q::circuit::openqasm;
# use prism_q::simulate_program;
# let program = openqasm::parse_dynamic(
#     "OPENQASM 3.0; qubit[1] q; bit[1] c; h q[0]; c[0] = measure q[0];
#      while (c[0]) { reset q[0]; h q[0]; c[0] = measure q[0]; }",
# )?;
use prism_q::{GateFilter, NoiseBuilder, NoiseChannel};

let rules = NoiseBuilder::new().after_gates(
    GateFilter::all().named("h"),
    NoiseChannel::AmplitudeDamping { gamma: 0.4 },
);
let noisy = simulate_program(&program).noise(&rules).seed(42).shots(1000)?;
# Ok::<(), prism_q::PrismError>(())
```

Declined under `parse_dynamic`, all `UnsupportedConstruct`: a runtime value where a
parse-time constant belongs (an index, a loop bound, a `def` argument, the angle of any
other gate), a builtin function over a runtime value, a register declared inside a
`while` body, an `array` a loop or measurement writes, a measurement into a variable
rather than a bit, `break` or `continue` inside a `for`, and `input` parameters.

## Supported gates

- **Standard / aliases**: x, y, z, h, s, sdg, t, tdg, sx, rx, ry, rz, p/phase, cx/CX/cnot,
  cy, cz, cp/cphase, crx, cry, crz, ch, swap, ccx/toffoli, cswap/fredkin, cu, u1, u2,
  u3/u/U.
- **Braket spellings**: i, si, ti, v, vi, phaseshift, cphaseshift, cphaseshift00,
  cphaseshift01, cphaseshift10, cv, ccnot, prx, pswap, xy, xx, yy, zz. Most name a
  gate the list above already carries; `cphaseshift00/01/10`, `pswap`, and `xy` are
  their own matrices.
- **Qiskit / exporter**: sxdg, cs, csdg, csx, ccz, r, xx_plus_yy,
  xx_minus_yy, ecr, iswap, dcx, c3x, c4x, mcx, rccx, rc3x/rcccx.
- **Hardware-native**: gpi, gpi2, ms, syc, sqrt_iswap, sqrt_iswap_inv.

  `gpi`, `gpi2`, and `ms` take angles in turns, matching IonQ's transpiler output.
  Amazon Braket spells the same three gates in radians; parse those with
  `openqasm::parse_with(source, Dialect::Braket)`, which changes the reading of
  that family and nothing else.

- **Pauli rotations**: `r` followed by one Pauli letter per qubit argument.
  `rxx`, `ryy`, and `rzz` are the two-letter cases; `rxyz(0.7) q[0], q[1], q[2];`
  is `exp(-i * 0.7 * (X⊗Y⊗Z) / 2)` with `x` on `q[0]`. `rzz` resolves to the
  native two-qubit rotation and a one-letter name to `rx`/`ry`/`rz`; wider
  strings build the native multi-qubit gate, which the statevector applies in
  one pass and every other backend receives as its CNOT-ladder lowering.

  The wider spelling is a PRISM-Q extension rather than standard OpenQASM, and
  `to_qasm3` emits it so a round trip preserves the gate instead of a lowering
  of it. For output another toolchain reads, run
  `circuit::expand_pauli_rotations` before exporting.

## Amazon Braket programs

`openqasm::parse_braket` reads a program under Braket's dialect and returns a
`BraketProgram`: the circuit, its `input` parameters, the result requests the
`#pragma braket result` lines made, and the noise model the
`#pragma braket noise` lines built. `parse` reads the native dialect, where a
pragma is an unsupported construct; `parse_with` under `Dialect::Braket` reads
the pragmas but returns only the circuit, so what they declared is dropped.
Use `parse_braket` whenever a result request has to survive.

```qasm
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket noise bit_flip(0.1) q[0]
#pragma braket result probability all
#pragma braket result expectation z(q[0]) @ z(q[1])
```

- **Result requests**: `state_vector`, `density_matrix`, `amplitude`,
  `probability`, `expectation`, `variance`, `sample`. Observables are `x`, `y`,
  `z`, `h`, `i` and `hermitian([[...]])`, joined into tensor products with `@`.
  A target list may be `all`, may be omitted (which means the same), or may
  name qubits. `adjoint_gradient` is declined: Braket serves it on SV1 only.
- **Noise channels**: `bit_flip`, `phase_flip`, `pauli_channel`,
  `depolarizing`, `amplitude_damping`, `generalized_amplitude_damping`,
  `phase_damping`, `two_qubit_depolarizing`, `two_qubit_dephasing`, and
  `kraus` with explicit operators. Each attaches after the instruction it
  follows, so a pragma standing before every instruction is an error.
  Probabilities are range-checked against Braket's own bounds.
- **Inline unitaries**: `#pragma braket unitary([[...]]) q[0]` on one or two
  targets. A wider matrix has no gate variant to carry it and says so.
- **Verbatim boxes**: `#pragma braket verbatim` followed by `box { ... }`. The
  body runs as written, verbatim being a directive to a device compiler that a
  simulator has nothing to honour. A `box` without the pragma runs the same way;
  the pragma without a box is rejected.

Matrix entries take Braket's complex notation: a real (`0`, `-1.5`), an
imaginary (`1im`, `-1im`), or their sum (`0.7 + 0.7im`).

### Computing the results

`Simulate::braket_results` evaluates the requests and returns them in Braket's
own conventions, which differ from the native terminals' in one way that matters:
Braket writes qubit 0 as the most significant bit of a basis index, so `x q[0]`
lands at index 2 of a two-qubit result and not at index 1.

```rust
use prism_q::circuit::openqasm;
use prism_q::simulate;

# let qasm_str = "OPENQASM 3.0; qubit[2] q; h q[0]; cnot q[0], q[1];
#     #pragma braket result probability all
#     #pragma braket result expectation z(q[0]) @ z(q[1])";
let program = openqasm::parse_braket(qasm_str).expect("parse error");
let values = simulate(&program.circuit)
    .seed(42)
    .braket_results(&program.results)
    .unwrap();
```

Every `expectation` and `variance` request is served by one traversal: the
observables lower to weighted Pauli sums, the distinct strings across all of them
are evaluated together, and each value is a weighted sum over that evaluation.
`h` expands as `(X + Z)/sqrt(2)` and a Hermitian matrix by its Pauli
decomposition, so no observable needs a second evaluation path.

An observable with no target list is applied to each qubit in parallel and
reports one value per qubit, which is why an expectation value is a list rather
than a number. `variance` is the spread of the operator itself,
`<O^2> - <O>^2`, not the grouped-measurement variance `ObservableExpectation`
reports beside a mean. `sample` has no exact reading and is declined here.

### Shot-based results

`Simulate::braket_results_sampled` answers the same requests from a measurement
record, which is what Braket does above zero shots:

```rust
# use prism_q::{circuit::openqasm, simulate};
# let program = openqasm::parse_braket("OPENQASM 3.0; qubit[2] q; h q[0]; cnot q[0], q[1];
#     #pragma braket result probability all
#     #pragma braket result expectation z(q[0]) @ z(q[1])").unwrap();
let values = simulate(&program.circuit)
    .seed(42)
    .braket_results_sampled(&program.results, 1000)
    .unwrap();
```

Each observable is diagonalized and the rotation carrying it onto the
computational basis is appended to the circuit once, so a single pass serves
every `sample`, `expectation` and `variance` request. `sample` reports the
eigenvalue each shot read, `expectation` its mean and `variance` its population
variance. A Pauli or `h` observable takes a named rotation; an explicit
Hermitian matrix is diagonalized numerically, which needs a gate wide enough to
carry the result and so stops at two qubits.

Two observables reading one qubit in different bases cannot share a measurement
and are rejected rather than answered from whichever rotation was applied first.
The identity is the exception: it reads no basis, so it shares a qubit with
anything. A `probability` request reads the computational basis and so takes its
own unrotated pass whenever a rotation was applied.

`state_vector`, `amplitude` and `density_matrix` report the state itself and are
declined above zero shots, the same way `sample` is declined at zero.

## Physical qubits

A program may name physical qubits directly instead of declaring a register:

```qasm
OPENQASM 3.0;
bit[2] c;
h $0;
cx $0, $1;
c[0] = measure $0;
c[1] = measure $1;
```

The indices are absolute, so the register is as wide as the highest one named
and nothing declares it. A `qubit` or `qreg` declaration in the same program is
rejected: a physical index and a register offset would give `0` two meanings.

## Subroutines

A `def` is inlined at each call. Classical arguments, including `bit` and `bit[n]`,
are passed by value. Under `parse`, a bit parameter can be read but cannot be
written. Under `parse_dynamic`, classical assignments update a copy of the argument.
Qubit parameters may name fixed-width registers, such as `qubit[2] anc`, including
slices supplied by the caller. Their order is preserved. Arguments may not overlap
on a qubit. Signature widths are resolved where the subroutine is declared.

A `-> bit` or `-> bit[n]` result lands on the bits the call is assigned to. The body
returns a measurement directly with `return measure q;`, or ends with `return r;`
for a bit it declared, which it may branch on before returning.
`parse_dynamic` also supports returning a computed bit value or a bit parameter.
A returned local bit that needs a classical initializer uses `parse_dynamic`, which
initializes it on every call, after copying the arguments.

```qasm
OPENQASM 3.0;
qubit[3] q;
bit[3] c;
def mx(qubit a) -> bit {
  h a;
  return measure a;
}
def herald(qubit a, qubit flag) -> bit {
  bit r;
  r = measure a;
  if (r) x flag;
  return r;
}
def fix(bit b, qubit a) {
  if (b) z a;
}
c[0] = mx(q[0]);
c[1] = herald(q[1], q[2]);
fix(c[0], q[2]);
```

The call expands to exactly the instructions the inlined body would, so `c[1] =
herald(q[1], q[2]);` is `c[1] = measure q[1]; if (c[1]) x q[2];`.

Scalar results of type `int`, `uint`, `bool`, `float`, `angle` or `duration` fold at
parse time when their values are known. Arguments and results are converted to their
declared widths before use. The body sees global constants from its declaration
scope, not variables or registers belonging to the caller.
Calls may appear in initializers, assignments, gate arguments and return expressions:

```qasm
qubit q;
def offset(int n) -> int[8] {
  if (n > 0) return n;
  return 0;
}
int n = offset(2) + offset(3);
rx(n) q;
```

A `return` exits the subroutine, including any enclosing loop, `box` or conditional
body. `return;` exits a subroutine with no result. A scalar result may be discarded.
Under `parse`, the path taken by a call must return its declared result. Under
`parse_dynamic`, every reachable path through the call must do so. A missing result
is a `Parse` error.

Under `parse_dynamic`, numerical results and scalar locals may depend on measured
bits. Nested calls keep their own arguments and results. Calls in `while` conditions
run at each test; calls on the right of `&&` or `||` run only when needed.

```qasm
qubit q;
bit[2] c;
def decode(bit b) -> int {
  if (b) return 1;
  return 0;
}
h q;
c[0] = measure q;
int correction = decode(c[0]);
rx(correction * pi) q;
c[1] = measure q;
```

Forms requiring private measurement storage still decline by name: a call that
discards a bit result, a measurement into a bit parameter, and a bit the body declares
without returning. Runtime duration and fixed-width angle values in subroutines,
modifiers on runtime calls, and the `complex` result type also decline.

## The subset

`UnsupportedConstruct` means the program is valid OpenQASM that this parser does not
implement. `Parse` means the text is not accepted as written. The remaining variants name
the specific mistake: `UndefinedRegister`, `InvalidQubit`, `InvalidClassicalBit`,
`GateArity`.

| Construct | Status | What a decline returns |
| --- | --- | --- |
| `OPENQASM 2.x` and `3.x` headers | Parses | Another major version: `UnsupportedConstruct` naming the version. A header that is not a number: `Parse` |
| `include "..."` | Accepted and ignored | Nothing. The standard gates are built in, so an include adds no names; a gate it would have defined declines later by name |
| `qubit`, `qreg`, `bit`, `creg` | Parses | A `bit` initializer with a set bit, or an assignment to a bit: `UnsupportedConstruct` under `parse`, written at runtime under `parse_dynamic`. A bit string whose length differs from the register, or a float written to bits: `Parse` |
| Physical qubits (`$0`) | Parses | A `qubit` or `qreg` declaration in the same program: `UnsupportedConstruct` |
| `int`, `uint`, `bool`, `float`, `angle`, `const` | Parses | Any other type, `complex` included: `UnsupportedConstruct` naming the type. A write under a runtime `if` or `switch` to a variable declared outside it: `UnsupportedConstruct` under `parse`, evaluated at runtime under `parse_dynamic` |
| `array` | Parses; folded at parse time | An element type other than `int`, `uint`, `float`, `angle` or `bool`: `UnsupportedConstruct`. An initializer of the wrong shape, an index out of range, or more than 2^20 elements: `Parse` |
| `box`, `box[d]` | Runs its body in place | A length that is not a duration or is negative: `Parse` |
| `duration`, `stretch`, `delay` | Parses; a delay is the identity | A length that is not a duration (`delay[10]`) or is negative, and a duration where a number belongs: `Parse`. A ratio over a stretch or mixing `dt` with SI units, and `durationof`: `UnsupportedConstruct` |
| `input`, `output` | Parses | `input` of a type other than `float` or `angle`, `output` of a type other than `bit`, or an `input` anywhere but as the whole angle argument of a top-level parametric gate: `UnsupportedConstruct` |
| `measure`, `reset` | Parses | A register measure whose widths disagree: `Parse` |
| `barrier;`, `barrier q;` and `barrier q[0], q[1];` | Parses | A bare `barrier;` spans every qubit declared so far, across registers |
| `nop` | Parses; emits nothing | |
| Casts, `int[8](x)`, `bool(c)` | Parses | A width on `bool`, an integer width outside 1 to 64 bits, a float width other than 16, 32, 64 or 128, or a float cast to `bit`: `Parse` |
| Expression statements, `a == 2;` | Evaluated and discarded; scalar subroutine calls are inlined | See [Subroutines](#subroutines) for call limits |
| `if`, `else`, `else if` | Parses | `else` at the head of a statement: `UnsupportedConstruct`. A condition over classical bits that no bit or register guard spells: `UnsupportedConstruct` under `parse`, evaluated at runtime under `parse_dynamic`. An `else` whose `if` body measures into a bit the condition reads: `Parse` |
| `switch`, `case`, `default` | Parses | An arm that measures into the switched register: `Parse`. More case labels than the region depth bound when a `default` is present, or a target over classical bits that no register guard spells: `UnsupportedConstruct` |
| `for` | Unrolls at parse time | A range in any form but `[start:stop]`, `[start:step:stop]` or `{a,b,c}`: `UnsupportedConstruct` naming what it found. The bounds themselves may be classical variables |
| `while`, `break`, `continue` | Parses under `parse_dynamic` | `UnsupportedConstruct` under `parse`: a loop that exits on a measurement has no finite instruction list. `break` or `continue` inside a `for`: `UnsupportedConstruct` |
| `def` | Inlines its body, with fixed-width qubit registers, bit results, and scalar results; `parse_dynamic` supports runtime numerical results | See [Subroutines](#subroutines) for the forms that decline, all `UnsupportedConstruct`. A result or argument of the wrong width, overlapping qubit arguments, or a missing result: `Parse` |
| `return` | Exits a `def`, including nested branches and loops | Outside a `def`: `UnsupportedConstruct`. Under a measurement-dependent branch: `UnsupportedConstruct` under `parse`, lowered into the graph under `parse_dynamic` |
| `gate` blocks | Parses; an empty body is the identity | |
| A gate length, `x[100ns] q;` | Parses; the length is dropped | |
| `defcalgrammar`, `cal`, `defcal` | Parsed and skipped: a calibration says how a gate is pulsed, not what it does | A call to a gate only a `defcal` names: `UnsupportedConstruct` naming it |
| `extern`, `opaque` | Declines | `UnsupportedConstruct`. A call out of the program has nothing to run against here |
| `ctrl`, `negctrl`, `ctrl(n)`, `negctrl(n)`, `inv`, `pow(k)` | Parses, chainable in any order | See [Other supported constructs](#other-supported-constructs) for the reach of each, and below for the declines |
| `gphase(theta)` | Parses and is carried | |
| `#pragma braket ...`, `pragma braket ...` | Parses under `Dialect::Braket` | Any other dialect, or a `braket` pragma this crate does not read: `UnsupportedConstruct` naming it |
| Any other pragma, and annotations (`@ns.tag ...`) | Ignored, as the language asks of a pass that does not read them | |
| A gate name the crate does not implement | Declines | `UnsupportedConstruct` naming it |
| A gate call at the wrong width | Declines | `GateArity` naming the gate, the arity it wanted, and what it got |

Modifier declines, all `UnsupportedConstruct`: `ctrl @` on a `def` call, which is a
subroutine rather than a gate; `ctrl @` on a body that measures, resets or branches;
a `ctrl @` chain past 255 controls; `pow(k) @` on a call spanning more than four qubits
or carrying an instruction with no matrix; `inv @` on a body that is not a gate sequence;
and a whole-number `pow` past one million repetitions, which returns `Parse` naming the
count instead.

Gate calls, loops and repetitions multiply, so one parse also stops with `Parse` once it
expands past 2^24 units of work: an emitted instruction or classical write, one per
qubit for a runtime rotation, or a statement or loop pass that emits none. Subroutine
argument binding and copying its constant environment also count. A `pow` repetition
is checked against the same bound before it is built.

Every decline above happens at parse time, so a program that parses is one the IR can
hold. The backend it then runs on may still decline the circuit; those limits are on the
[Capabilities](capabilities.md) page, not here.

## Other supported constructs

- Gate modifiers: `ctrl @`, `negctrl @`, `inv @`, `pow(k) @`, chainable and in
  any order. `ctrl(n) @` and `negctrl(n) @` stand for `n` controls. A control consumes one qubit from the front of the argument list,
  in the order the modifiers are written, and `negctrl` differs from `ctrl` only
  in firing on `|0>`.

  A control applies to whatever the gate expanded to rather than to the gate
  name, since `ctrl(U1 U2 ... Un)` is `ctrl(U1) ctrl(U2) ... ctrl(Un)`. That
  reaches a user `gate` body, a gate that lowers to a sequence, and `swap`,
  which becomes the Fredkin lowering rather than three controlled CNOTs. The
  two-qubit gates carried as a matrix (`ecr`, `xy`, `pswap`,
  `cphaseshift00`/`01`/`10`, `ms`, `syc`, `sqrt_iswap`, `sqrt_iswap_inv`,
  `xx_plus_yy`, `xx_minus_yy`) take a control through their eigenbasis, since
  `ctrl(V D V-dagger)` is `V ctrl(D) V-dagger` and a controlled diagonal costs
  one phase per entry whatever the control count. A `def` call declines, being
  a subroutine rather than a gate, and so does a call spanning more than four
  qubits, which is where the dense reduction stops.

  `pow(k)` takes any real `k`. On a single-qubit gate it is the principal
  matrix power, read from the eigenvalues with each angle on `(-pi, pi]`, so
  `pow(0.5) @ x` is `sx` and `pow(1.0/3)` applied three times is the gate again.
  The principal branch is what decides between the two square roots of a gate,
  and it follows the eigenvalue rather than the written angle: `p(1.5*pi)` and
  `p(-0.5*pi)` are the same matrix, so they take the same root. A whole
  number is repetition at any width, and past a million it is rejected rather
  than run; a fraction on a wider call is the principal power of the matrix the
  whole call composes to, up to four qubits. `inv @` and `pow(k) @` commute
  with a control, so a chain reads the same either way.
- `gphase(theta)` multiplies the state by `e^(i theta)`. It is carried rather
  than dropped, being observable through a `state_vector` result and under a
  control, where `ctrl @ gphase(theta) q[0]` is a phase gate on `q[0]`.
- Classical `int`, `uint`, `bool`, `float` and `angle` declarations with
  assignment, and `let` aliases over sliced registers. See
  [Declarations and measurement](#declarations-and-measurement).
- User-defined `gate` blocks.
- Classical `if` conditionals, guarding either a single statement or a braced
  body. A braced body admits any supported statement, `measure` and `reset`
  included, and may nest.
- `else` and `else if` arms, and `switch` with `case` and `default` arms. Both
  lower to guards on the existing condition language rather than new syntax in
  the IR.
- Parity conditions, `if (c[0] ^ c[2])` or `if ((c[0] ^ c[2]) == 0)`.
- Multi-register broadcast, `barrier`, and block comments.
- An expression evaluator over `+ - * / %`, `**`, the comparisons, `&&`, `||`,
  `!`, and `&`, `|`, `^`, `~`, `<<`, `>>` with C precedence, the constants `pi`, `tau`
  and `euler`, and the OpenQASM builtins (`arcsin`, `arccos`, `arctan`,
  `ceiling`, `cos`, `exp`, `floor`, `log`, `mod`, `popcount`, `pow`, `sin`,
  `sqrt`, `tan`). `**` is right associative and binds tighter than unary minus,
  so `-2 ** 2` is `-4`. The constants, `true` and `false` cannot be redeclared
  in the global scope; a gate or `def` parameter may shadow a constant.
  Statements, conditions and expressions nest at most 256 levels, an operator
  chain such as `a + b + c` counting one level per operator, and a deeper source
  fails with `PrismError::Parse`. The bound also covers the gate and `def` bodies
  a call expands, counted together with the blocks around it.

```qasm
bit[2] c;
qubit[3] q;
c[0] = measure q[0];
if (c[0]) {
  x q[1];
  c[1] = measure q[1];
  if (c[1]) { reset q[2]; }
}
```

```admonish warning title="Not supported"
[The subset](#the-subset) has the full list. The two declines worth the reasoning: `else`
is rejected when the `if` body measures into a bit the condition reads, and `switch` when
any arm measures into the switched register. Both lower to a chain of guards that re-read
the classical bits, so such a source could otherwise take two arms of one choice. An
`else` body may write freely, since nothing re-reads after it.

Under `parse`, classical expressions beyond the condition language are outside the
subset, as is `while`; [`parse_dynamic`](#dynamic-programs) reads both.
```

```admonish note title="Qubit ordering"
`q[0]` is the least significant bit, so `x q[0]` produces state index 1, not 2.
```

# OpenQASM Support

PRISM-Q parses a subset of OpenQASM 3.0 and the common 2.0 syntax. Text is lexed, parsed
to a statement tree that holds syntax only, and evaluated into the
[Circuit IR](../architecture/ir.md), so a program's meaning is decided once, in the
evaluator.

[The subset](#the-subset) lists what parses and what declines. Valid OpenQASM outside the
subset returns `UnsupportedConstruct` naming the construct, never a panic or a silent drop.

## Parsing and running

```rust
use prism_q::circuit::openqasm;
use prism_q::simulate;

# let qasm_str = "OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];";
let circuit = openqasm::parse(qasm_str).expect("parse error");
let result = simulate(&circuit).seed(42).run().unwrap();
```

`run_qasm(qasm, seed)` parses and simulates in one call. The other entry points in
`openqasm` differ in what they return:

| Function | Returns |
| --- | --- |
| `parse`, `parse_with(source, dialect)` | A `Circuit` |
| `parse_parametric`, `parse_parametric_with(source, dialect)` | The template `Circuit` and the `Parameters` its `input` declarations name (see [Input parameters](#input-parameters)) |
| `parse_parametric_with_inputs(source, inputs)` | The same, with values supplied for some `input` declarations by name |
| `parse_dynamic`, `parse_dynamic_with_inputs(source, inputs)` | A `DynamicProgram` with loops and runtime classical values (see [Dynamic programs](#dynamic-programs)) |
| `parse_with_externs`, `parse_parametric_with_externs`, `parse_dynamic_with_externs` | The same three, with functions for `extern` declarations (see [Extern functions](#extern-functions)) |
| `parse_braket` | A `BraketProgram`: circuit, parameters, result requests and noise (see [Amazon Braket Programs](./braket.md)) |

`Dialect::Native`, the default, reads `gpi`, `gpi2` and `ms` angles in turns;
`Dialect::Braket` reads them in radians and accepts `#pragma braket` lines.

## Exporting

```rust
use prism_q::circuit::qasm_export;

# let circuit = prism_q::circuits::qft_circuit(4);
let qasm = qasm_export::to_qasm3(&circuit).expect("export error");
```

Re-parsing an export gives back the same instruction stream, with gate matrices equal to
round-off and inline angles (`rx`, `rz`, `rzz`, `p`) exact. Qubits come out as one
`qubit[n] q` register and classical bits as one `bit[m] c`, split only where a condition
compares against a register narrower than the whole.

A circuit that has been through [fusion](../architecture/fusion.md) does not export: fused
blocks, tiled multi-gate passes and diagonal batches carry matrices with no OpenQASM
spelling, and `to_qasm3` returns `ExportUnsupported` naming the instruction index. Export
before fusing, or export the template a `PreparedCircuit` binds. `QftBlock` and `PauliRot`
are the exceptions: the first expands to its textbook Hadamard, controlled-phase and swap
sequence, the second to its CNOT ladder. On the CPU statevector, an expanded QFT parsed
back in runs as one block again (see
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

OpenQASM 2.0 `qreg q[3];`, `creg c[3];` and `measure q[0] -> c[0];` also work.

A `bit` declaration may take a value: `bit b = measure q[0];` measures into it, and
`bit[2] f = "00";` or `bit b = 0;` spell out the cleared register it starts as anyway. Bit
strings are written most significant bit first. Under `parse`, an initializer with a set
bit, or any assignment to a bit (`c = 2;`, `c[0] = 1;`, `c <<= 1;`), returns
`UnsupportedConstruct`, since no instruction in the list writes a classical bit;
[`parse_dynamic`](#dynamic-programs) writes them.

Names may open with any Unicode letter, so `gate cphase(θ) a, b { ... }` and `qubit γ;`
parse. A program declares at most 2^24 qubits and 2^24 classical bits, physical `$k`
indices included; past that the parse fails with `PrismError::Parse`.

`output bit[3] c;` declares the register and marks it as the program's result. The mark
changes nothing, since every classical bit is reported either way.

### Physical qubits

A program may name physical qubits directly instead of declaring a register:

```qasm
OPENQASM 3.0;
bit[2] c;
h $0;
cx $0, $1;
c[0] = measure $0;
c[1] = measure $1;
```

The indices are absolute, so the register is as wide as the highest one named. A `qubit`
or `qreg` declaration in the same program is rejected, since a physical index and a
register offset would give `0` two meanings.

### Slices and aliases

An index may be a single position, an inclusive range with an optional step in the middle,
or an explicit set. A sliced register broadcasts as a whole one does. A negative index
counts from the end: `q[-1]` is the last qubit and `q[1:-1]` runs from the second to the
last.

```qasm
OPENQASM 3.0;
qubit[6] q;
h q[0:2];         // q[0], q[1], q[2]
h q[0:2:5];       // q[0], q[2], q[4]
h q[5:-1:4];      // q[5], q[4]
h q[{1, 4}];      // q[1], q[4]
cx q[0:1], q[2:3];
```

`let` names qubits or bits in the order written, `++` joins operands, and an alias is
itself sliceable, so a slice of an alias follows the alias's order, not the register's.

```qasm
let ends = q[5] ++ q[0];
let pair = q[0:3];
h ends[0];        // q[5]
h pair[1:2];      // q[1], q[2]
```

### Classical variables

`int`, `uint`, `bool`, `float` and `angle` declarations, with an optional width, fold at
parse time, and the value reads anywhere a literal would: an index, a loop bound, a gate
angle, a condition operand. `const` forbids reassignment.

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

An `int` or `uint` is an exact integer that wraps at its width, 64 bits when none is
given, and `/` or `%` between two integers truncates toward zero: `1 / 2` is `0`,
`1.0 / 2` is `0.5`. A `float[n]` is an IEEE 754 float of 16, 32, 64 or 128 bits.
`float[16]` and `float[32]` round to half and single precision, and `float[128]` holds a
double, since nothing here computes in quad precision. Floats take `+ - * /` and `**`;
`%` takes integers, and `mod(a, b)` is the float remainder. An `angle[n]` holds the
nearest multiple of `2 pi / 2^n`, modulo `2 pi`. Casts such as `int[8](x)`, `uint[2](c)`,
`bool(x)` and `angle[3](t)` convert the same way, and `a[i]` on an integer reads one of its
bits. A float cast to an integer drops its fraction and wraps at the integer's width, so
`int[8](1e30)` is `0`; past about `1.7e38` it saturates first, and NaN reads as `0`. A
float has no bit pattern to cast to `bit[n]`; cast it to `uint` first. An expression on
its own, `a == 2;`, is evaluated and discarded.

A declaration inside a `for` body binds for one pass: the body's scope is dropped at the
end of each iteration, while a write to a name declared outside the loop carries on to the
next pass and past the loop.

An `if`, `else` or `switch` body opens a scope too, but a measurement decides whether it
runs, and classical variables fold at parse time. Such a body may declare and write its
own variables. A write to one declared outside it returns `UnsupportedConstruct`, since
the value would otherwise depend on what the measurement read. Holding such a value needs
the control-flow graph [`parse_dynamic`](#dynamic-programs) builds.

An `array` of any of those element types takes a size per dimension, fixed at parse time,
and folds the same way. Elements read as `a[i, j]` or `a[i][j]` wherever a value would and
take plain and compound assignment. A write inside a loop outlives the pass, which
filling an array needs.

```qasm
array[float[64], 2, 2] angles = {{0.1, 0.2}, {0.3, 0.4}};
array[int, 4] order;
for int k in [0:3] { order[k] = 3 - k; }
rx(angles[1, 0]) q[order[0]];   // rx(0.3) q[3]
```

`sizeof(a)` and `sizeof(a, d)` give the length of dimension `d`, `0` by default. A range
in a subscript, `a[1:3]` or `g[0, 1:2]`, names a slice that keeps the dimension, and a
subscript with fewer indices than dimensions names a row. A slice or row reads as an
array where one belongs: an initializer, the right side of an assignment to a slice or a
whole array, and an array argument. `++` joins arrays along their first dimension. The
shapes on both sides of an array assignment must match, and a mismatch is a `Parse`
error. `a[i:j]` on an `int` or `uint` reads those bits as an unsigned value.

```qasm
array[int[8], 2] first = {0, 1};
array[int[8], 3] second = {2, 3, 4};
array[int[8], 5] joined = first ++ second;
second[1:2] = first;              // second == {2, 0, 1}
def fill(mutable array[int[8], #dim = 1] out) {
  for int i in [0:sizeof(out) - 1] { out[i] = i; }
}
fill(joined[2:4]);                // joined == {0, 1, 0, 1, 2}
```

A `def` takes an array as `readonly` or `mutable`, with one size per dimension or a rank,
`#dim = n`. The argument passes by reference, so a `mutable` parameter writes back to the
caller's array or slice. The element type must match, and the shape must match the sizes
or the rank. A compound assignment to a slice, `a[0:1] += b`, declines.

`array[bit, n]`, `array[complex, n]` and `array[duration, n]` decline. A `def` body sees
the program's `const` arrays and its array parameters, as it sees no other non-constant
global.

### Complex numbers

`complex[float[n]]`, and bare `complex` for 64-bit components, fold at parse time like the
other scalars. An imaginary literal is a number followed by `im`, with spaces or tabs
allowed before the suffix. Complex values take `+ - * /` and `**`, which returns the
principal value, and compare with `==` and `!=`. `exp` and `sqrt` take a complex
argument. `real`, `imag`, `abs` and `arg` read one as a real number, and a complex value
reaches a gate angle only through them. A `def` takes and returns `complex` values.

```qasm
complex[float[64]] a = 10.0 + 5.0im;
complex[float[64]] b = a ** (-2.0 - 7.0im);
rz(arg(b)) q[0];
```

A complex value where a real one belongs, and a cast from complex to a real type, are
`Parse` errors. A complex value a measurement or a loop decides declines under
`parse_dynamic`, which has no runtime type for one.

### Timing

A simulation has no clock, so timing parses and has no effect. `delay` is the identity on
the qubits it names, or on every qubit when it names none, and emits no instruction, so
fusion merges across it. A `box`, with or without a duration, runs its body in place with
no barrier at its edges, since a box only fixes when its contents run; names its body
declares go out of scope with it.

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

Literals take `dt`, `ns`, `us` (or `µs`), `ms` and `s`. Durations add, subtract, scale by
a number and divide into a number, all folded at parse time. A ratio needs both sides in
SI units or both in `dt`, since only a backend's sample period relates the two. A
`stretch` is sized by a scheduler: it may stand wherever a delay takes a length and is
rejected where its value would be read.

## Supported gates

- Standard names and aliases: id/i, x, y, z, h, s, sdg, t, tdg, sx, rx, ry, rz, p/phase,
  cx/CX/cnot, cy, cz, cp/cphase, crx, cry, crz, ch, swap, ccx/toffoli, cswap/fredkin, cu,
  u1, u2, u3/u/U.
- Braket spellings: si, ti, v, vi, phaseshift, cphaseshift, cphaseshift00,
  cphaseshift01, cphaseshift10, cv, ccnot, prx, pswap, xy, xx, yy, zz. Most name a gate
  the list above carries; `cphaseshift00/01/10`, `pswap` and `xy` are their own matrices.
- Qiskit and exporter names: sxdg, cs, csdg, csx, ccz, r, xx_plus_yy, xx_minus_yy, ecr,
  iswap, dcx, c3x, c4x, mcx, rccx, rc3x/rcccx.
- Hardware-native: gpi, gpi2, ms, syc, sqrt_iswap, sqrt_iswap_inv. `gpi`, `gpi2` and `ms`
  take angles in turns, matching IonQ's transpiler output. Amazon Braket spells the same
  three gates in radians; `openqasm::parse_with(source, Dialect::Braket)` reads them that
  way and changes nothing else.
- Pauli rotations: `r` followed by one Pauli letter per qubit argument. `rxx`, `ryy` and
  `rzz` are the two-letter cases; `rxyz(0.7) q[0], q[1], q[2];` is
  `exp(-i * 0.7 * (X⊗Y⊗Z) / 2)` with `x` on `q[0]`. `rzz` resolves to the native
  two-qubit rotation and a one-letter name to `rx`/`ry`/`rz`; wider strings build the
  native multi-qubit gate, which the statevector applies in one pass and every other
  backend receives as its CNOT-ladder lowering.

  The wider spelling is a PRISM-Q extension, not standard OpenQASM, and `to_qasm3` emits
  it so a round trip keeps the gate. For output another toolchain reads, run
  `circuit::expand_pauli_rotations` before exporting.

`gphase(theta)` multiplies the state by `e^(i theta)`. It is carried, not dropped: a
`state_vector` result shows it, and under a control `ctrl @ gphase(theta) q[0]` is a
phase gate on `q[0]`.

User `gate` blocks define new gates; an empty body is the identity. A gate applied to
whole registers broadcasts across them, and a gate length such as `x[100ns] q;` parses
and is dropped.

## Gate modifiers

`ctrl @`, `negctrl @`, `inv @` and `pow(k) @` chain in any order. `ctrl(n) @` and
`negctrl(n) @` stand for `n` controls. Each control consumes one qubit from the front of
the argument list, in the order the modifiers are written, and `negctrl` differs from
`ctrl` only in firing on `|0>`.

A control applies to what the gate expands to, not to its name, since
`ctrl(U1 U2 ... Un)` is `ctrl(U1) ctrl(U2) ... ctrl(Un)`. That reaches a user `gate` body,
a gate that lowers to a sequence, and `swap`, which becomes the Fredkin lowering rather
than three controlled CNOTs. The two-qubit gates carried as a matrix (`ecr`, `xy`,
`pswap`, `cphaseshift00`/`01`/`10`, `ms`, `syc`, `sqrt_iswap`, `sqrt_iswap_inv`,
`xx_plus_yy`, `xx_minus_yy`) take a control through their eigenbasis:
`ctrl(V D V-dagger)` is `V ctrl(D) V-dagger`, and a controlled diagonal costs one phase per
entry at any control count. A `def` call declines, being a subroutine, and so does a call
spanning more than four qubits, where the dense reduction stops.

`pow(k)` takes any real `k`. On a single-qubit gate it is the principal matrix power, read
from the eigenvalues with each angle on `(-pi, pi]`, so `pow(0.5) @ x` is `sx` and
`pow(1.0/3)` applied three times is the gate again. The branch follows the eigenvalue, not
the written angle: `p(1.5*pi)` and `p(-0.5*pi)` are the same matrix and take the same
root. A whole number is repetition at any width, rejected past a million; a fraction on a
wider call is the principal power of the matrix the whole call composes to, up to four
qubits. `inv @` and `pow(k) @` commute with a control, so a chain reads the same either
way.

The declines are listed under [The subset](#the-subset).

## Expressions

Expressions take `+ - * / %`, `**`, the comparisons, `&&`, `||`, `!`, and `&`, `|`, `^`,
`~`, `<<`, `>>` with C precedence; the constants `pi`, `tau` and `euler`; and the
OpenQASM builtins `arcsin`, `arccos`, `arctan`, `ceiling`, `cos`, `exp`, `floor`, `log`,
`mod`, `popcount`, `pow`, `sin`, `sqrt` and `tan`. `**` is right associative and binds
tighter than unary minus, so `-2 ** 2` is `-4`. The constants, `true` and `false` cannot
be redeclared in the global scope; a gate or `def` parameter may shadow a constant. Line
and block comments are accepted.

Statements, conditions and expressions nest at most 256 levels, an operator chain such as
`a + b + c` counting one level per operator, and a deeper source fails with
`PrismError::Parse`. The bound also covers the gate and `def` bodies a call expands,
counted together with the blocks around it. Separately, gate and `def` calls nest at most
32 deep inside one another's bodies, which stops a recursive definition with a `Parse`
error naming the gate or subroutine.

## Conditions

An `if` guards a single statement or a braced body. A braced body admits any supported
statement, `measure` and `reset` included, and may nest.

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

`else`, `else if` and `switch` with `case` and `default` arms lower to guards on the
existing condition language rather than new IR syntax. A condition may test a bit, a
register against a value, or a parity such as `if (c[0] ^ c[2])` or
`if ((c[0] ^ c[2]) == 0)`. Under `parse`, anything else over classical bits declines;
[`parse_dynamic`](#dynamic-programs) evaluates it at runtime.

Because the guards re-read the classical bits, `else` is rejected when the `if` body
measures into a bit the condition reads, and `switch` when any arm measures into the
switched register: such a source could take two arms of one choice. An `else` body may
write freely, since nothing re-reads after it.

## Subroutines

A `def` is inlined at each call. Classical arguments, `bit` and `bit[n]` included, pass by
value: under `parse` a bit parameter can be read but not written, and under
`parse_dynamic` assignments update a copy. Qubit parameters may name fixed-width
registers such as `qubit[2] anc`, including slices the caller supplies, in order.
Arguments may not overlap on a qubit. Signature widths resolve where the subroutine is
declared.

A `-> bit` or `-> bit[n]` result lands on the bits the call is assigned to. The body
returns a measurement with `return measure q;`, or ends with `return r;` for a bit it
declared, which it may branch on first. `parse_dynamic` also returns a computed bit value
or a bit parameter, and a returned local bit with a classical initializer needs it: it
initializes the bit on every call, after copying the arguments.

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

The call expands to exactly the inlined body's instructions, so
`c[1] = herald(q[1], q[2]);` is `c[1] = measure q[1]; if (c[1]) x q[2];`.

Scalar results of type `int`, `uint`, `bool`, `float`, `angle` or `duration` fold at parse
time when their values are known. Arguments and results convert to their declared widths
before use. The body sees global constants from its declaration scope, not the caller's
variables or registers. Calls may appear in initializers, assignments, gate arguments and
return expressions:

```qasm
qubit q;
def offset(int n) -> int[8] {
  if (n > 0) return n;
  return 0;
}
int n = offset(2) + offset(3);
rx(n) q;
```

A `return` exits the subroutine from inside any loop, `box` or conditional body; `return;`
exits one with no result, and a scalar result may be discarded. Under `parse` the path a
call takes must return its declared result; under `parse_dynamic` every reachable path
must. A missing result is a `Parse` error.

Under `parse_dynamic`, numerical results and scalar locals may depend on measured bits.
Nested calls keep their own arguments and results. Calls in `while` conditions run at each
test, and calls on the right of `&&` or `||` run only when needed.

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

Forms that need private measurement storage decline by name: a call that discards a bit
result, a measurement into a bit parameter, and a bit the body declares without returning.
So do a measurement returned as a scalar result, a `stretch` returned as a `duration`,
runtime duration, fixed-width angle and `complex` values, and modifiers on runtime calls.
Under `parse`, a scalar result that depends on a measurement and a
call inside a measurement-dependent condition decline too; `parse_dynamic` takes both.
Under `parse_dynamic`, a bit argument or bit result read at runtime declines past 64
bits, the width a runtime value holds.

## Input parameters

An `input` declaration names a parameter slot. `openqasm::parse_parametric` returns the
template circuit with the `Parameters` that bind it, in declaration order and under the
declared names; `parse_parametric_with` does the same under a chosen `Dialect`.

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

Several gates may read one input, the weight sharing `Parameters` already models:
`rx(theta) q;` over a register links every gate it produces to one slot, and binding
writes one angle to each.

`parse` rejects a program that declares an input, since it has no value to take and a zero
would be a quiet wrong answer. Use `parse_parametric`, or `PreparedCircuit` for a sweep.

A gate angle may be affine in one input: `rx(-theta / 2)` and `rz(2 * theta + pi)` link
their gates with that scale and offset, so binding writes `scale * value + offset` and a
gradient picks up the factor `scale`. The template holds each angle at an input of zero.
An input reaches a gate through a `for` body, or as the argument of a `gate` or `def`
whose body uses it in an angle; an `angle` parameter keeps it unwrapped, as a bound
`input angle` is. A `float[32]` or fixed-width `angle` parameter would round it and
declines.

These return `UnsupportedConstruct` naming the reason: an angle over two inputs or not
affine in one (`sin(theta)`, `theta * theta`), two inputs on one gate, an input on a gate
that lowers to a sequence or has no rotation angle, a modified gate or `gphase`, a guarded
body, and an input read as a classical value (an index, a loop bound, a declaration).

`parse_parametric_with_inputs` takes values for some inputs by name. A supplied input
folds as a declaration of its type initialized to the value would, so `input int[8] n;`
reads as a loop bound and `input bit[2] mask;` as bits. A supplied `float` or `angle` is a
constant and takes no slot; the unsupplied ones keep theirs. An unsupplied input of any
other type, a name supplied twice, and a name no `input` declares return
`InvalidParameter`.

```rust
# use prism_q::circuit::{dynamic::ClassicalValue, openqasm};
let qasm = "OPENQASM 3.0; input int[8] n; input float theta; qubit[4] q;
    for int i in [0:n] { rx(theta / 2) q[i]; }";
let (template, params) =
    openqasm::parse_parametric_with_inputs(qasm, &[("n", ClassicalValue::Int(3))])?;
let bound = params.bind(&template, &[0.8])?;
# Ok::<(), prism_q::PrismError>(())
```

## Extern functions

An `extern` declaration names a function the program calls and gives its types. The
caller supplies the function: `Externs::register` binds a Rust closure to the name, and
the `_with_externs` entry points take the registry. Python passes a mapping of callables
as `externs=` to `parse_qasm`, `parse_qasm_parametric` and `parse_qasm_dynamic`.

```rust
use prism_q::circuit::dynamic::ClassicalValue;
use prism_q::circuit::openqasm::{self, Externs};

let mut externs = Externs::new();
externs.register("correction", |args| Ok(ClassicalValue::Float(args[0].to_float() * 0.1)));
let circuit = openqasm::parse_with_externs(
    "OPENQASM 3.0;
     extern correction(int[8]) -> float;
     qubit[1] q;
     rx(correction(3)) q[0];",
    &externs,
)?;
# Ok::<(), prism_q::PrismError>(())
```

Each argument is stored through its declared type before the call, so `int[8]` passes 257
as 1. The function returns a `Bool` for `bool` and `bit`, an `Int` for `int[n]` and
`uint[n]`, and a `Float` for `float[n]` and `angle`, where an `Int` is promoted. The result
is stored through the declared type. A call reads as a value anywhere one belongs, inside `def` bodies too, and
as a statement its value is dropped.

A call whose arguments fold at parse time runs once, at parse time, under every entry
point. Under `parse_dynamic_with_externs`, a call over a measured bit or a runtime variable
lowers to `ClassicalExpr::Extern` and runs when a shot evaluates it, so its value may feed
a condition, an assignment, a bit write or a runtime rotation angle:

```qasm
extern decode(bit, bit) -> bool;
qubit[3] q;
bit[2] s;
s[0] = measure q[1];
s[1] = measure q[2];
if (decode(s[0], s[1])) x q[0];
```

Shots run on several threads, so a Rust function is `Send + Sync`; a Python callable takes
the GIL for each call, and an exception it raises becomes the `__cause__` of the
`PrismError` the caller sees. A call nothing reads may not run, and how often a runtime call runs
is otherwise unspecified, so a function should be pure.

An entry point without `_with_externs`, and a Python parse given no `externs`, declines
every `extern` declaration with `UnsupportedConstruct`. Under the `_with_externs` entry
points, these return `InvalidParameter`: a declaration with no function registered under its name,
a result of the wrong kind, and any error the function returns or a Python callable raises.
A function registered for no declaration is ignored. These return `UnsupportedConstruct`:
an `extern` with no result type, a parameter or result of another type (`bit[n]`,
`angle[n]`, `complex`, `duration`, arrays), a call over an unsupplied `input`, and under
`parse` a call over a measured value. A call with the wrong number of arguments, an
`extern` declared inside a body, and a `gate` or `def` sharing an extern's name return
`Parse`.

`ExternCall::new` builds a call for a `DynamicProgramBuilder` from a name, the parameter
and result types, the function, and the argument expressions. `build` checks the argument
count and the type widths.

## Dynamic programs

`openqasm::parse` returns a `Circuit`, a finite instruction list, so it declines `while`
and any classical value a measurement decides. `openqasm::parse_dynamic` reads both and
returns a `DynamicProgram`: basic blocks of ordinary circuits joined by branches on
classical expressions, run once per shot by `simulate_program`.

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

Beyond what `parse` reads, `parse_dynamic` accepts:

- `while (condition) { ... }`, with `break` and `continue`.
- `break`, `continue` and `end` under a measured `if` or `switch`. A `for` still unrolls,
  and an exit a measurement decides jumps between its passes.
- Any condition or `switch` target over classical bits, where `parse` takes only those a
  bit or register guard spells: `c[0] && c[1]` and `c + 1 == 2` run here.
- A classical bit read as a value (`c[0]`) and a bit register read as an unsigned integer
  (`c`), up to 64 bits.
- A write to a variable under a measured `if` or `switch`, evaluated when the program
  runs.
- A write to classical bits: `bit[2] f = "11";`, `c = c + 1;`, `c[0] = c[1] ^ c[2];`,
  `c <<= 1;`. A write takes an integer or a bool, and a bit register wider than 64 bits
  takes one only from a parse-time value.
- A runtime value as the angle of `rx`, `ry`, `rz`, `p` or `rzz`.

A variable lives at runtime only when a measured bit or another runtime variable reaches
it, or when a loop or a measured branch writes it from outside its scope. Every other
variable folds as under `parse`, so a program `parse` accepts gives the same circuit
under `parse_dynamic` and the same seeded shots. Runtime `int` and `uint` variables wrap
at their declared width, and runtime floats round to their width. A runtime `angle`
holds a double. Fixed-width runtime `angle[n]` declarations and casts return
`UnsupportedConstruct`; their precision is supported only when the value folds at parse time.

A shot that runs more than a million blocks stops with `PrismError::StepLimit` naming its
loop; `SimulateProgram::max_steps` sets another bound. The
[dynamic programs](../architecture/dynamic-programs.md) architecture page covers the block
structure, the backends that run it, and its cost.

`SimulateProgram::noise` takes `NoiseBuilder` rules, compiles them against every block and
runtime rotation, and runs one noisy trajectory per shot. The loop above with amplitude
damping after each `h`:

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

### Dynamic declines

`parse_dynamic` returns `UnsupportedConstruct` for:

- A runtime value where a parse-time constant belongs: a qubit index, a loop bound, a
  register size, a duration, and the angle of any gate but the five rotations above.
- A runtime argument to a `def` written as a gate application, `f(x) q;`. Called as a
  function, `f(x, q)`, the same `def` takes it.
- A slice read as a runtime value.
- A builtin function (`sin`, `sqrt` and the rest) over a runtime value. Only operators
  and [`extern` calls](#extern-functions) run at runtime.
- A `bit` or `qubit` register declared inside a `while` body, and an `array` a loop or
  measurement writes.
- A measurement straight into a variable. Measure into a bit and assign the bit.
- An unsupplied `float` or `angle` input, since a dynamic program has no parameter
  binding. `parse_dynamic_with_inputs` takes a value for every input by name, as
  `parse_parametric_with_inputs` does, and a supplied `bit` input may set bits.

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
| `int`, `uint`, `bool`, `float`, `angle`, `complex`, `const` | Parses | Any other type: `UnsupportedConstruct` naming the type. A complex value where a real one belongs: `Parse`. A write under a runtime `if` or `switch` to a variable declared outside it: `UnsupportedConstruct` under `parse`, evaluated at runtime under `parse_dynamic` |
| `array` | Parses; folded at parse time, with slices, `++`, `sizeof` and array arguments | An element type other than `int`, `uint`, `float`, `angle` or `bool`, or a compound assignment to a slice: `UnsupportedConstruct`. An initializer or assignment of the wrong shape, an index out of range, an index set on an array, or more than 2^20 elements: `Parse` |
| `box`, `box[d]` | Runs its body in place | A length that is not a duration or is negative: `Parse` |
| `duration`, `stretch`, `delay` | Parses; a delay is the identity | A length that is not a duration (`delay[10]`) or is negative, and a duration where a number belongs: `Parse`. A ratio over a stretch or mixing `dt` with SI units, a runtime value in a duration under `parse_dynamic`, and `durationof`: `UnsupportedConstruct` |
| `input`, `output` | Parses; an unsupplied `float` or `angle` input is a slot that gate angles affine in it link to | An unsupplied input of another type: `InvalidParameter`. `output` of a type other than `bit`, two inputs on one gate, or an unsupplied input anywhere but in an angle affine in it: `UnsupportedConstruct` |
| `measure`, `reset` | Parses | A register measure whose widths disagree: `Parse` |
| `barrier;`, `barrier q;` and `barrier q[0], q[1];` | Parses | A bare `barrier;` spans every qubit declared so far, across registers |
| `nop` | Parses; emits nothing | |
| Casts, `int[8](x)`, `bool(c)` | Parses | A width on `bool`, an integer width outside 1 to 64 bits, a float width other than 16, 32, 64 or 128, or a float cast to `bit`: `Parse` |
| Expression statements, `a == 2;` | Evaluated and discarded; scalar subroutine calls are inlined | See [Subroutines](#subroutines) for call limits |
| `if`, `else`, `else if` | Parses | `else` at the head of a statement: `UnsupportedConstruct`. A condition over classical bits that no bit or register guard spells: `UnsupportedConstruct` under `parse`, evaluated at runtime under `parse_dynamic`. An `else` whose `if` body measures into a bit the condition reads: `Parse` |
| `switch`, `case`, `default` | Parses | An arm that measures into the switched register: `Parse`. More case labels than the region depth bound when a `default` is present, or a target over classical bits that no register guard spells: `UnsupportedConstruct` |
| `for` | Unrolls at parse time | A range in any form but `[start:stop]`, `[start:step:stop]` or `{a,b,c}`: `UnsupportedConstruct` naming what it found. The bounds may be classical variables and `def` or `extern` calls |
| `while` | Parses under `parse_dynamic` | `UnsupportedConstruct` under `parse`: a loop that exits on a measurement has no finite instruction list |
| `break`, `continue` | Parses inside `for` and `while` | Outside a loop, or at the top level of a `def` or `gate` body: `Parse`. In a `for` under a measurement-dependent branch: `UnsupportedConstruct` under `parse`, lowered into the graph under `parse_dynamic` |
| `end` | Stops the program; nothing after it runs | Inside a `gate` body, called or not: `Parse`. Under a measurement-dependent branch: `UnsupportedConstruct` under `parse`, lowered into the graph under `parse_dynamic` |
| `def` | Inlines its body, with fixed-width qubit registers, bit results, and scalar results; `parse_dynamic` supports runtime numerical results | See [Subroutines](#subroutines) for the forms that decline, all `UnsupportedConstruct`. A result or argument of the wrong width, overlapping qubit arguments, a missing result, or calls nested past 32 deep: `Parse` |
| `return` | Exits a `def`, including nested branches and loops | Outside a `def`: `UnsupportedConstruct`. Under a measurement-dependent branch: `UnsupportedConstruct` under `parse`, lowered into the graph under `parse_dynamic` |
| `gate` blocks | Parses; an empty body is the identity | Calls nested past 32 deep: `Parse` |
| A gate length, `x[100ns] q;` | Parses; the length is dropped | |
| `defcalgrammar`, `cal`, `defcal` | Parsed and skipped: a calibration says how a gate is pulsed, not what it does | A call to a gate only a `defcal` names: `UnsupportedConstruct` naming it |
| `extern` | Calls a registered function under the `_with_externs` entry points; see [Extern functions](#extern-functions) | Any other entry point: `UnsupportedConstruct`. No registered function, or a result of the wrong kind: `InvalidParameter`. No result type, a type other than `bool`, `bit`, `int[n]`, `uint[n]`, `float[n]` and `angle`, a call over an unsupplied input, or a call over a measured value under `parse`: `UnsupportedConstruct`. The wrong number of arguments: `Parse` |
| `opaque` | Declines | `UnsupportedConstruct` |
| `ctrl`, `negctrl`, `ctrl(n)`, `negctrl(n)`, `inv`, `pow(k)` | Parses, chainable in any order | See [Gate modifiers](#gate-modifiers) for the reach of each, and below for the declines |
| `gphase(theta)` | Parses and is carried | |
| `#pragma braket ...`, `pragma braket ...` | Parses under `Dialect::Braket` | Any other dialect, a `braket` pragma inside a braced body, or one this crate does not read: `UnsupportedConstruct` naming it |
| Any other pragma, and annotations (`@ns.tag ...`) | Ignored, as the language asks of a pass that does not read them | |
| A gate name the crate does not implement | Declines | `UnsupportedConstruct` naming it |
| A gate call at the wrong width | Declines | `GateArity` naming the gate, the arity it wanted, and what it got |

Modifier declines, all `UnsupportedConstruct`: `ctrl @` on a `def` call, which is a
subroutine rather than a gate; `ctrl @` on a body that measures, resets or branches; a
`ctrl @` chain past 255 controls; `pow(k) @` on a call spanning more than four qubits or
carrying an instruction with no matrix; and `inv @` on a body that is not a gate sequence.
A whole-number `pow` past one million repetitions returns `Parse` naming the count.

Gate calls, loops and repetitions multiply, so a parse also stops with `Parse` once it
expands past 2^24 units of work: an emitted instruction or classical write, one per qubit
for a runtime rotation, or a statement or loop pass that emits none. Subroutine argument
binding and copying its constant environment count too, and a `pow` repetition is checked
against the bound before it is built.

Every decline above happens at parse time, so a program that parses is one the IR can
hold. The backend it runs on may still decline the circuit; the
[Capabilities](capabilities.md) page lists those limits.

## Amazon Braket programs

`parse_braket` reads Braket's result, noise, unitary and verbatim pragmas, and
`Simulate::braket_results` answers the requests in Braket's conventions. See
[Amazon Braket Programs](./braket.md).

```admonish note title="Qubit ordering"
`q[0]` is the least significant bit, so `x q[0]` produces state index 1, not 2.
```

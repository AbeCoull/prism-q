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

let circuit = openqasm::parse(qasm_str).expect("parse error");
let result = simulate(&circuit).seed(42).run().unwrap();
```

`run_qasm(qasm, seed)` parses and simulates in one call.

## Exporting

```rust
use prism_q::circuit::qasm_export;

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
and the second to its CNOT-ladder lowering on the way out.

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

`output bit[3] c;` declares the register and marks it as the program's result.
Every classical bit is reported either way, so the marking costs nothing and
changes nothing.

### Slices and aliases

An index may be a single position, an inclusive range with an optional step in
the middle, or an explicit set. A sliced register broadcasts exactly as a whole
one does.

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

A declaration inside a `for` body binds for that pass only: the scope the body
opened is dropped at the end of each iteration.

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
let (template, params) = openqasm::parse_parametric(qasm)?;
let bound = params.bind(&template, &[0.41, 1.27])?;
let text = to_qasm3(&bound)?;   // angles written out, no `input` line
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
  simulator has nothing to honour. A `box` without the pragma is rejected, and
  so is the pragma without a box.

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

## The subset

`UnsupportedConstruct` means the program is valid OpenQASM that this parser does not
implement. `Parse` means the text is not accepted as written. The remaining variants name
the specific mistake: `UndefinedRegister`, `InvalidQubit`, `InvalidClassicalBit`,
`GateArity`.

| Construct | Status | What a decline returns |
| --- | --- | --- |
| `OPENQASM 2.0` and `3.0` headers | Parses | Any other version: `UnsupportedConstruct` naming the version |
| `include "..."` | Accepted and ignored | Nothing. The standard gates are built in, so an include adds no names; a gate it would have defined declines later by name |
| `qubit`, `qreg`, `bit`, `creg` | Parses | |
| Physical qubits (`$0`) | Parses | A `qubit` or `qreg` declaration in the same program: `UnsupportedConstruct` |
| `int`, `uint`, `bool`, `float`, `angle`, `const` | Parses | Any other type, `complex` included: `UnsupportedConstruct` naming the type |
| `array` declarations | Declines | `UnsupportedConstruct` |
| `duration`, `stretch`, `delay` | Declines | `UnsupportedConstruct`. Timing has no meaning here: nothing schedules |
| `input`, `output` | Parses | `input` of a type other than `float` or `angle`, `output` of a type other than `bit`, or an `input` anywhere but as the whole angle argument of a top-level parametric gate: `UnsupportedConstruct` |
| `measure`, `reset` | Parses | A register measure whose widths disagree: `Parse` |
| `barrier;`, `barrier q;` and `barrier q[0], q[1];` | Parses | A bare `barrier;` spans every qubit declared so far, across registers |
| `if`, `else`, `else if` | Parses | `else` at the head of a statement: `UnsupportedConstruct`. An `else` whose `if` body measures into a bit the condition reads: `Parse` |
| `switch`, `case`, `default` | Parses | An arm that measures into the switched register: `Parse`. More case labels than the region depth bound when a `default` is present: `UnsupportedConstruct` |
| `for` | Unrolls at parse time | A range in any form but `[start:stop]`, `[start:step:stop]` or `{a,b,c}`: `UnsupportedConstruct` naming what it found. The bounds themselves may be classical variables |
| `while` | Declines | `UnsupportedConstruct` |
| `def` | Inlines a unitary body at the call site | A classical bit parameter or a return type: `UnsupportedConstruct` |
| `gate` blocks | Parses | |
| `defcal`, `extern`, `opaque`, `box` | Declines | `UnsupportedConstruct` |
| `ctrl`, `negctrl`, `inv`, `pow(k)` | Parses, chainable in any order | See [Other supported constructs](#other-supported-constructs) for the reach of each, and below for the declines |
| `gphase(theta)` | Parses and is carried | |
| `#pragma braket ...` | Parses under `Dialect::Braket` | Any other dialect, or any other pragma: `UnsupportedConstruct` naming the pragma |
| A gate name the crate does not implement | Declines | `UnsupportedConstruct` naming it |
| A gate call at the wrong width | Declines | `GateArity` naming the gate, the arity it wanted, and what it got |

Modifier declines, all `UnsupportedConstruct`: `ctrl @` on a `def` call, which is a
subroutine rather than a gate; `ctrl @` on a body that measures, resets or branches;
a `ctrl @` chain past 255 controls; `pow(k) @` on a call spanning more than four qubits
or carrying an instruction with no matrix; `inv @` on a body that is not a gate sequence;
and a whole-number `pow` past one million repetitions, which returns `Parse` naming the
count instead.

Every decline above happens at parse time, so a program that parses is one the IR can
hold. The backend it then runs on may still decline the circuit; those limits are on the
[Capabilities](capabilities.md) page, not here.

## Other supported constructs

- Gate modifiers: `ctrl @`, `negctrl @`, `inv @`, `pow(k) @`, chainable and in
  any order. A control consumes one qubit from the front of the argument list,
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
  `pow(0.5) @ x` is `sx` and `pow(1/3)` applied three times is the gate again.
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
- An expression evaluator over `+ - * / %` and `**`, the constants `pi`, `tau`
  and `euler`, and the OpenQASM builtins (`arcsin`, `arccos`, `arctan`,
  `ceiling`, `cos`, `exp`, `floor`, `log`, `mod`, `popcount`, `pow`, `sin`,
  `sqrt`, `tan`). `**` is right associative and binds tighter than unary minus,
  so `-2 ** 2` is `-4`.

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

Classical expressions beyond the condition language are outside the subset, as is `while`.
```

```admonish note title="Qubit ordering"
`q[0]` is the least significant bit, so `x q[0]` produces state index 1, not 2.
```

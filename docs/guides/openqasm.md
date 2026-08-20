# OpenQASM Support

PRISM-Q parses a practical subset of OpenQASM 3.0, with backward compatibility for common
2.0 syntax. The parser converts text directly to the [Circuit IR](../architecture/ir.md)
with no intermediate AST.

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
- **Qiskit / exporter**: sxdg, cs, csdg, csx, ccz, r, rzz, rxx, ryy, xx_plus_yy,
  xx_minus_yy, ecr, iswap, dcx, c3x, c4x, mcx, rccx, rc3x/rcccx.
- **Hardware-native**: gpi, gpi2, ms, syc, sqrt_iswap, sqrt_iswap_inv.

## Other supported constructs

- Gate modifiers: `ctrl @`, `inv @`, `pow(k) @`.
- User-defined `gate` blocks.
- Classical `if` conditionals, guarding either a single statement or a braced
  body. A braced body admits any supported statement, `measure` and `reset`
  included, and may nest.
- `else` and `else if` arms, and `switch` with `case` and `default` arms. Both
  lower to guards on the existing condition language rather than new syntax in
  the IR.
- Parity conditions, `if (c[0] ^ c[2])` or `if ((c[0] ^ c[2]) == 0)`.
- Multi-register broadcast, `barrier`, and an expression evaluator with math functions.

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
`while` loops and classical expressions beyond the condition language. A `for` loop
with a compile-time trip count unrolls at parse time; a `def` subroutine inlines at
its call site, but only a unitary one. A construct that parses as valid OpenQASM but
is unsupported returns `UnsupportedConstruct` rather than panicking; see the
[Error Model](../architecture/api-surface.md).

`else` is rejected when the `if` body measures into a bit the condition reads, and
`switch` when any arm measures into the switched register. Both lower to a chain of
guards that re-read the classical bits, so such a source could otherwise take two arms
of one choice. An `else` body may write freely: nothing re-reads after it.
```

```admonish note title="Qubit ordering"
`q[0]` is the least significant bit, so `x q[0]` produces state index 1, not 2.
```

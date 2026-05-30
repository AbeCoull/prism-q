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
- Classical `if` conditionals.
- Multi-register broadcast, `barrier`, and an expression evaluator with math functions.

```admonish warning title="Not supported"
`for` / `while` loops, subroutines, and classical expressions beyond `if`. A construct
that parses as valid OpenQASM but is unsupported returns `UnsupportedConstruct` rather
than panicking; see the [Error Model](../architecture/api-surface.md).
```

```admonish note title="Qubit ordering"
`q[0]` is the least significant bit, so `x q[0]` produces state index 1, not 2.
```

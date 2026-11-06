# Amazon Braket Programs

`openqasm::parse_braket` reads a program under Braket's dialect and returns a
`BraketProgram`: the circuit, its `input` parameters, the result requests from
`#pragma braket result` lines, and the noise model from `#pragma braket noise` lines.
`parse` reads the native dialect, where a pragma is an unsupported construct. `parse_with`
under `Dialect::Braket` reads the pragmas but returns only the circuit, dropping what they
declared. Use `parse_braket` whenever a result request has to survive. Everything else
about the language is on [OpenQASM Support](./openqasm.md).

```qasm
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket noise bit_flip(0.1) q[0]
#pragma braket result probability all
#pragma braket result expectation z(q[0]) @ z(q[1])
```

## Pragmas

- Result requests: `state_vector`, `density_matrix`, `amplitude`, `probability`,
  `expectation`, `variance`, `sample`. Observables are `x`, `y`, `z`, `h`, `i` and
  `hermitian([[...]])`, joined into tensor products with `@`. A target list may be `all`,
  omitted (the same thing), or named qubits. `adjoint_gradient` is declined: Braket
  serves it on SV1 only.
- Noise channels: `bit_flip`, `phase_flip`, `pauli_channel`, `depolarizing`,
  `amplitude_damping`, `generalized_amplitude_damping`, `phase_damping`,
  `two_qubit_depolarizing`, `two_qubit_dephasing`, and `kraus` with explicit operators.
  Each attaches after the instruction it follows, so a pragma before every instruction is
  an error. Probabilities are checked against Braket's own bounds.
- Inline unitaries: `#pragma braket unitary([[...]]) q[0]` on one or two targets. A wider
  matrix has no gate variant to carry it and says so.
- Verbatim boxes: `#pragma braket verbatim` followed by `box { ... }`. The body runs as
  written: verbatim directs a device compiler, and a simulator has nothing to honour. A
  `box` without the pragma runs the same way; the pragma without a box is rejected.

Matrix entries take Braket's complex notation: a real (`0`, `-1.5`), an imaginary (`1im`,
`-1im`), or their sum (`0.7 + 0.7im`). A `braket` pragma stands at the top level; inside
a braced body it returns `UnsupportedConstruct`.

Braket also spells `gpi`, `gpi2` and `ms` in radians where the native dialect reads turns,
and names several gates its own way; [Supported gates](./openqasm.md#supported-gates) lists
them.

## Computing the results

`Simulate::braket_results` evaluates the requests in Braket's conventions, which differ
from the native terminals in one way: Braket writes qubit 0 as the most significant bit of
a basis index, so `x q[0]` lands at index 2 of a two-qubit result, not index 1.

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

One traversal serves every `expectation` and `variance` request: the observables lower to
weighted Pauli sums, the distinct strings across all of them are evaluated together, and
each value is a weighted sum over that evaluation. `h` expands as `(X + Z)/sqrt(2)` and a
Hermitian matrix by its Pauli decomposition, so no observable needs a second path.

An observable with no target list applies to each qubit separately and reports one value
per qubit, so an expectation value is a list rather than a number. `variance` is the
spread of the operator itself, `<O^2> - <O>^2`, not the grouped-measurement variance
`ObservableExpectation` reports beside a mean. `sample` has no exact reading and is
declined here.

## Shot-based results

`Simulate::braket_results_sampled` answers the same requests from a measurement record, as
Braket does above zero shots:

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

Each observable is diagonalized and the rotation onto the computational basis is appended
once, so one pass serves every `sample`, `expectation` and `variance` request. `sample`
reports the eigenvalue each shot read, `expectation` its mean and `variance` its
population variance. Pauli and `h` observables take a named rotation; a Hermitian matrix
is diagonalized numerically, which needs a gate wide enough to carry the result, so it
stops at two qubits.

Two observables reading one qubit in different bases cannot share a measurement and are
rejected. The identity reads no basis, so it shares a qubit with anything. A `probability`
request reads the computational basis and takes its own unrotated pass whenever a rotation
was applied. `state_vector`, `amplitude` and `density_matrix` report the state itself and
are declined above zero shots, as `sample` is at zero.

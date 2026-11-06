# Your First Circuit

Build a circuit with `CircuitBuilder`, or parse it from OpenQASM text. Each example is
in Rust, then Python.

## With the builder

`CircuitBuilder` chains gate calls and runs the result. A Bell pair,
`(|00⟩ + |11⟩) / √2`:

```rust
use prism_q::CircuitBuilder;

let result = CircuitBuilder::new(2)
    .h(0)
    .cx(0, 1)
    .run(42)                       // seed = 42
    .expect("simulation failed");

let probs = result.probabilities.expect("no probabilities");
for i in 0..probs.len() {
    let p = probs.get(i);
    if p > 1e-10 {
        println!("|{i:02b}> = {p:.4}");
    }
}
// |00> = 0.5000
// |11> = 0.5000
```

```python
from prism_q import CircuitBuilder, simulate

bell = CircuitBuilder(2).h(0).cx(0, 1).build()
print(simulate(bell).seed(42).run().probabilities)   # [0.5 0.  0.  0.5]
```

<figure class="circuit-figure">
<div class="circuit-scroll" role="region" aria-label="Bell pair circuit" tabindex="0">
<img src="../diagrams/bell_2.svg" alt="A Hadamard on q[0] followed by a controlled X from q[0] to q[1].">
</div>
<figcaption>H creates a superposition; CX correlates the two qubits. <a href="../diagrams/bell_2.svg">Open full SVG</a>.</figcaption>
</figure>

Measurement defaults to the Z basis. `measure_in_basis(qubit, axis, bit)` measures along
X, Y or Z, and `measure_pauli_product(&terms, bit)` records the parity of a Pauli string
such as `X0 Z1`. Both lower onto gates every backend runs. A basis measurement leaves
the qubit in the Z eigenstate of the recorded bit instead of rotating back, and a Pauli
product accumulates its parity on one extra qubit past the declared register:

```rust
use prism_q::{CircuitBuilder, PauliAxis, PauliTerm};

let circuit = CircuitBuilder::new_with_classical(2, 2)
    .h(0)
    .cx(0, 1)
    .measure_in_basis(0, PauliAxis::X, 0)
    .measure_pauli_product(&[PauliTerm::z(0), PauliTerm::z(1)], 1)
    .build();
assert_eq!(circuit.num_qubits, 3);
```

```python
circuit = (
    CircuitBuilder(2, 2)
    .h(0)
    .cx(0, 1)
    .measure_in_basis(0, "X", 0)
    .measure_pauli_product([(0, "Z"), (1, "Z")], 1)
    .build()
)
assert circuit.num_qubits == 3
```

## From OpenQASM

The same Bell pair in OpenQASM 3.0:

```rust
use prism_q::circuit::openqasm;
use prism_q::simulate;

let qasm = r#"
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    h q[0];
    cx q[0], q[1];
"#;

let circuit = openqasm::parse(qasm).expect("failed to parse QASM");
let result = simulate(&circuit).seed(42).run().expect("simulation failed");
```

```python
from prism_q import parse_qasm

circuit = parse_qasm("""
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
""")
result = simulate(circuit).seed(42).run()
```

`run_qasm(qasm, seed)` parses and simulates in one call, in both languages. The
[OpenQASM Support guide](../guides/openqasm.md) lists the supported subset.

```admonish note title="Qubit ordering"
`q[0]` is the least significant bit, so `x q[0]` produces state index 1, not 2. A state
index formatted with `{i:b}`, as above, reads most significant qubit first. Shot
bitstrings from `bitstring` and Python count keys run the other way, with classical
bit 0 leftmost.
```

Next: sample measurement outcomes in [Shots and Sampling](./shots.md).

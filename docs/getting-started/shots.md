# Shots and Sampling

Probabilities give you the exact distribution. Shots give you sampled measurement
outcomes, the way real hardware reports results. PRISM-Q samples deterministically from a
fixed seed.

## Sampling shots

```rust
use prism_q::circuit::openqasm;
use prism_q::simulate;

let qasm = r#"
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    cx q[0], q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
"#;
let circuit = openqasm::parse(qasm).expect("failed to parse QASM");

let result = simulate(&circuit).seed(42).shots(1024).expect("shots failed");
print!("{result}");   // ShotsResult implements Display
```

The same seed always produces the same samples. Pass `rand::random()` as the seed for
non-deterministic sampling.

## Counts and marginals

For large shot counts, you usually want aggregates rather than raw shots:

```rust
// Frequency histogram: bitstring -> count
let counts = simulate(&circuit).seed(42).sample_counts(100_000).unwrap();

// Per-qubit P(measuring |1>), without the full joint distribution
let marginals = simulate(&circuit).seed(42).marginals().unwrap();
```

```admonish tip title="Sampling scales past the statevector"
`sample_counts` and `shots` route through PRISM-Q's compiled samplers, which propagate
measurements through the circuit instead of materializing the full statevector on every
shot. For Clifford circuits this scales to thousands of qubits. See
[Compiled Samplers](../architecture/samplers.md).
```

## Noisy sampling

Attach a `NoiseModel` to sample under depolarizing or readout noise:

```rust
use prism_q::{simulate, BackendKind, NoiseModel};

let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
let result = simulate(&circuit)
    .backend(BackendKind::Statevector)
    .noise(noise)
    .seed(42)
    .shots(1024)
    .unwrap();
```

The [Noise and QEC guide](../guides/qec.md) covers noise models and detector sampling in
depth.

Next: learn how PRISM-Q picks a representation in [Choosing a Backend](./choosing-a-backend.md).

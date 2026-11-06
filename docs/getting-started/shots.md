# Shots and Sampling

Probabilities are the exact distribution. Shots are sampled outcomes, as hardware reports
them, and a fixed seed makes them deterministic.

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

In Python, `shots` returns a `(shots, bits)` bool array, and `counts()` folds it into a
histogram:

```python
from prism_q import CircuitBuilder, simulate

circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
result = simulate(circuit).seed(42).shots(1024)
print(result.shots.shape)   # (1024, 2)
print(result.counts())      # {'00': ..., '11': ...}
```

Pass `rand::random()` as the seed in Rust, or any fresh integer in Python, for
nondeterministic sampling.

## Counts and marginals

At large shot counts, ask for aggregates instead of raw shots:

```rust
// Frequency histogram: bitstring -> count
let counts = simulate(&circuit).seed(42).sample_counts(100_000).unwrap();

// Per-qubit (P(0), P(1)) pairs, without the full joint distribution
let marginals = simulate(&circuit).seed(42).marginals().unwrap();
```

```python
counts = simulate(circuit).seed(42).sample_counts(100_000).counts()
marginals = simulate(circuit).seed(42).marginals().marginals
```

```admonish tip title="Sampling scales past the statevector"
On a Clifford circuit, `sample_counts` and `shots` route through the compiled samplers,
which propagate measurements through the circuit instead of rebuilding the state for
every shot, so Clifford circuits sample at thousands of qubits. Other circuits with
terminal measurements sample from one final state. See
[Compiled Samplers](../architecture/samplers.md).
```

## Noisy sampling

Attach a `NoiseModel` to sample under depolarizing or readout noise:

```rust
use prism_q::{simulate, BackendKind, NoiseModel};

let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
let result = simulate(&circuit)
    .backend(BackendKind::Statevector)
    .noise(&noise)
    .seed(42)
    .shots(1024)
    .unwrap();
```

```python
from prism_q import BackendKind, NoiseModel

noise = NoiseModel.uniform_depolarizing(circuit, 0.001)
result = (
    simulate(circuit)
    .backend(BackendKind.statevector())
    .noise(noise)
    .seed(42)
    .shots(1024)
)
```

The [Noise guide](../guides/noise.md) covers noise models, and the
[QEC guide](../guides/qec.md) covers detector sampling.

Next: how PRISM-Q picks a representation, in [Choosing a Backend](./choosing-a-backend.md).

# Simulate and Sample

Goal: build a three-qubit GHZ circuit, read its exact distribution, sample it like
hardware would, and run it on a backend of your choosing.

## Build and run

```python
from prism_q import CircuitBuilder, simulate

ghz = CircuitBuilder(3).h(0).cx(0, 1).cx(1, 2).build()
outcome = simulate(ghz).seed(42).run()
print(outcome.probabilities)       # [0.5 0.  0.  0.  0.  0.  0.  0.5]
print(outcome.metadata.backend)    # Stabilizer
```

`probabilities` is a NumPy array over the `2**3` basis states, with qubit 0 as the least
significant bit of the index. Half the weight sits on `|000>` and half on `|111>`.

`metadata.backend` names the engine that ran. Nothing asked for a stabilizer tableau: the
default `BackendKind.auto()` saw only Clifford gates and picked one.

## Sample shots

Hardware returns measurement records, not amplitudes. Add classical bits, measure, and
sample:

```python
measured = CircuitBuilder(3, 3).h(0).cx(0, 1).cx(1, 2).measure_all().build()
counts = simulate(measured).seed(42).sample_counts(1000).counts()
print(sorted(counts.items()))      # [('000', 485), ('111', 515)]

shots = simulate(measured).seed(42).shots(5).shots
print(shots.shape, shots.dtype)    # (5, 3) bool
```

The seed fixes the draw, so the same seed gives the same counts on every run. In a count
key, character `i` is classical bit `i`, so a key reads with bit 0 on the left:

```python
flipped = CircuitBuilder(3, 3).x(0).measure_all().build()
print(simulate(flipped).seed(42).sample_counts(10).counts())   # {'100': 10}
```

## Choose a backend

Pass a `BackendKind` to override the automatic choice. The answer stays the same; the
representation, and so the cost, changes:

```python
from prism_q import BackendKind

for kind in [BackendKind.statevector(), BackendKind.mps(16), BackendKind.sparse()]:
    result = simulate(ghz).backend(kind).seed(42).run()
    print(result.metadata.backend, result.probabilities[[0, 7]])
# Statevector [0.5 0.5]
# Mps [0.5 0.5]
# Sparse [0.5 0.5]
```

A backend that cannot hold the circuit says so instead of guessing. The product-state
backend keeps one state per qubit and declines anything entangling:

```python
from prism_q import PrismError

try:
    simulate(ghz).backend(BackendKind.product_state()).seed(42).run()
except PrismError as exc:
    print(exc.kind)                # incompatible_backend
```

[Choosing a Backend](../getting-started/choosing-a-backend.md) has the dispatch tree and
a table from circuit shape to backend.

## In Rust

```rust
use prism_q::{BackendKind, CircuitBuilder, bitstring, simulate};

let ghz = CircuitBuilder::new(3).h(0).cx(0, 1).cx(1, 2).build();
let outcome = simulate(&ghz).seed(42).run()?;
let probs = outcome.probabilities.expect("no probabilities");
assert!((probs.get(0b000) - 0.5).abs() < 1e-12);

let measured = CircuitBuilder::new_with_classical(3, 3)
    .h(0)
    .cx(0, 1)
    .cx(1, 2)
    .measure_all()
    .build();
let counts = simulate(&measured).seed(42).sample_counts(1000)?;
for (key, n) in &counts.counts {
    println!("{}: {n}", bitstring(key, counts.num_classical_bits));
}

let sv = simulate(&ghz).backend(BackendKind::Statevector).seed(42).run()?;
println!("{:?}", sv.metadata.backend);
# Ok::<(), prism_q::PrismError>(())
```

Next: [Variational Circuits and Gradients](./variational.md).

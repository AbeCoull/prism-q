<div class="prism-hero">

# PRISM-Q

**A Rust quantum circuit simulator built for speed.**

</div>

PRISM-Q matches each circuit to a simulation strategy. It dispatches across nine CPU
backends plus optional CUDA and MPI paths, runs circuits through a multi-pass fusion
pipeline, and uses AVX2, FMA and BMI2 SIMD in the inner loop. Input is OpenQASM 3.0,
with backward-compatible 2.0 syntax. A two-qubit Bell pair and a thousand-qubit Clifford
circuit go through the same entry point.

```rust
use prism_q::CircuitBuilder;

let result = CircuitBuilder::new(2).h(0).cx(0, 1).run(42).unwrap();
let probs = result.probabilities.unwrap();
// |00> = 0.5, |11> = 0.5
```

<div class="prism-cards">
<a class="prism-card" href="./getting-started/install.html"><span class="prism-card-title">Get started</span><span class="prism-card-body">Install the crate, build a circuit, and sample shots.</span></a>
<a class="prism-card" href="./getting-started/choosing-a-backend.html"><span class="prism-card-title">Choose a backend</span><span class="prism-card-body">Statevector, stabilizer, MPS, sparse, and the rest, matched to the circuit.</span></a>
<a class="prism-card" href="./guides/performance.html"><span class="prism-card-title">Performance and SIMD</span><span class="prism-card-body">Fusion passes, cache-resident tiled kernels, and the threading model.</span></a>
<a class="prism-card" href="./architecture/overview.html"><span class="prism-card-title">Architecture</span><span class="prism-card-body">The layers from parser to backends, the dispatch tree, and the gate IR.</span></a>
</div>

## What it does

- Nine CPU backends: statevector, stabilizer, factored stabilizer, sparse, MPS, product
  state, tensor network and dynamic factored split-state, each picked from the
  circuit's structure, plus an exact density matrix selected by name.
- Compiled shot samplers that do not rebuild the statevector for every shot, including
  the noisy and detector/QEC paths.
- Clifford+T engines (stabilizer rank, stochastic and deterministic Pauli propagation)
  for circuits a dense statevector cannot hold.
- An optional CUDA path for statevector, stabilizer and density-matrix execution.
- [Python bindings](./guides/python.md) with NumPy output.

## Where to go next

- [Installation](./getting-started/install.md) and [Your First Circuit](./getting-started/first-circuit.md)
- [Architecture](./architecture/overview.md): backends, dispatch tree, fusion pipeline and SIMD strategy
- [Benchmarks](./benchmarks.md): measured timings on the reference circuit suite
- [Glossary](./glossary.md)
- [API reference](https://docs.rs/prism-q) on docs.rs
- [Source and issues](https://github.com/AbeCoull/prism-q) on GitHub

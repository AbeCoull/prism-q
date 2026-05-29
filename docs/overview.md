<div class="prism-hero">

# PRISM-Q

**A Rust quantum circuit simulator built for speed.**

</div>

PRISM-Q runs quantum circuits fast by matching each one to the right simulation strategy.
It dispatches across eight CPU backends plus an optional CUDA path, optimizes circuits
through a multi-pass fusion pipeline, and uses AVX2, FMA, and BMI2 SIMD in the inner loop.
Input is OpenQASM 3.0, with backward-compatible 2.0 syntax. The same library handles a
two-qubit Bell pair and a thousand-qubit Clifford circuit.

```rust
use prism_q::CircuitBuilder;

let result = CircuitBuilder::new(2).h(0).cx(0, 1).run(42).unwrap();
let probs = result.probabilities.unwrap();
// |00> = 0.5, |11> = 0.5
```

<div class="prism-cards">
<a class="prism-card" href="./getting-started/install.html"><span class="prism-card-title">Get started</span><span class="prism-card-body">Install the crate, build a circuit, and sample shots in a few minutes.</span></a>
<a class="prism-card" href="./getting-started/choosing-a-backend.html"><span class="prism-card-title">Choose a backend</span><span class="prism-card-body">Statevector, stabilizer, MPS, sparse, and more. Match the representation to the circuit.</span></a>
<a class="prism-card" href="./guides/performance.html"><span class="prism-card-title">Performance and SIMD</span><span class="prism-card-body">Fusion passes, cache-resident tiled kernels, and the threading model behind the speed.</span></a>
<a class="prism-card" href="./architecture/overview.html"><span class="prism-card-title">Architecture</span><span class="prism-card-body">The layered design from parser to backends, the dispatch tree, and the gate IR.</span></a>
</div>

## What it does

- **Eight CPU backends** selected automatically per circuit: statevector, stabilizer
  (with factored and filtered variants), sparse, MPS, product state, tensor network, and
  dynamic factored split-state.
- **Compiled shot samplers** that sample without rebuilding the full statevector each
  shot, including noisy and detector/QEC paths.
- **Clifford+T strategies** (stabilizer rank, stochastic and deterministic Pauli
  propagation) for circuits beyond the reach of a dense statevector.
- **Optional CUDA backend** for statevector and stabilizer execution.

## Where to go next

- [Installation](./getting-started/install.md) and [Your First Circuit](./getting-started/first-circuit.md): a hands-on start.
- [Architecture](./architecture/overview.md): backends, dispatch tree, fusion pipeline, and SIMD strategy.
- [Glossary](./glossary.md): terminology used across the documentation.
- [API reference](https://docs.rs/prism-q): generated Rust API documentation.
- [Source and issues](https://github.com/AbeCoull/prism-q): the repository on GitHub.

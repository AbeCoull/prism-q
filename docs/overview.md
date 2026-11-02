# PRISM-Q

An open-source quantum circuit simulator written in Rust.

## Start here

- [Get started](./getting-started/install.md): install the crate and run a circuit.
- [Choose a backend](./getting-started/choosing-a-backend.md): match the simulation
  strategy to the circuit.
- [Performance and SIMD](./guides/performance.md): fusion, tiled kernels, and threading.
- [Architecture](./architecture/overview.md): the parser, circuit IR, and dispatch tree.

## Bell pair

```rust
use prism_q::CircuitBuilder;

let result = CircuitBuilder::new(2).h(0).cx(0, 1).run(42).unwrap();
let probs = result.probabilities.unwrap();
// |00> = 0.5, |11> = 0.5
```

## Simulation

PRISM-Q matches each circuit to a simulation strategy. It dispatches across nine CPU
backends plus optional CUDA and MPI-distributed paths, runs circuits through a
multi-pass gate fusion pipeline, and uses AVX2, FMA and BMI2 SIMD in the inner loop
(NEON on ARM64). Input is OpenQASM 3.0, with backward-compatible 2.0 syntax, and
Python bindings expose the same engine with NumPy output. A two-qubit Bell pair and a
thousand-qubit Clifford circuit go through the same entry point.

- Nine CPU backends: statevector, stabilizer, factored stabilizer, sparse, MPS, product
  state, tensor network (on expectation and marginal queries) and dynamic factored
  split-state, each picked from the circuit's structure, plus an exact density matrix
  selected by name.
- Compiled shot samplers that do not rebuild the statevector for every shot, including
  the noisy and detector/QEC paths.
- Clifford+T engines (stabilizer rank, stochastic and deterministic Pauli propagation)
  for circuits a dense statevector cannot hold.
- An optional CUDA path for statevector, stabilizer and density-matrix execution, and
  an optional [MPI-distributed statevector](./guides/distributed.md) for registers
  larger than one host's memory.
- [Python bindings](./guides/python.md) with NumPy output.

## Reference and tools

- [Interactive atlas](./atlas/index.html): one statevector gate traced from its matrix
  through memory layout and SIMD lanes, and native recordings of each fusion pass
- [Benchmarks](./benchmarks.md): measured timings on the reference circuit suite, and
  the [methodology](./guides/benchmarking.md) behind them
- [Glossary](./glossary.md)
- [API reference](https://docs.rs/prism-q) on docs.rs
- [Source and issues](https://github.com/AbeCoull/prism-q) on GitHub

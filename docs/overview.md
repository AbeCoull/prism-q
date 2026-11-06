# PRISM-Q

An open-source quantum circuit simulator written in Rust, with Python bindings.

## Start here

- [Get started](./getting-started/install.md): install the crate or the wheel and run a
  circuit.
- [Choose a backend](./getting-started/choosing-a-backend.md): match the simulation
  strategy to the circuit.
- [Performance and SIMD](./guides/performance.md): fusion, tiled kernels and threading.
- [Architecture](./architecture/overview.md): the parser, circuit IR and dispatch tree.

## Bell pair

```rust
use prism_q::CircuitBuilder;

let result = CircuitBuilder::new(2).h(0).cx(0, 1).run(42).unwrap();
let probs = result.probabilities.unwrap();
// |00> = 0.5, |11> = 0.5
```

The same circuit from Python:

```python
import prism_q

bell = prism_q.CircuitBuilder(2).h(0).cx(0, 1).build()
print(prism_q.simulate(bell).seed(42).run().probabilities)  # [0.5 0.  0.  0.5]
```

## Simulation

A two-qubit Bell pair and a thousand-qubit Clifford circuit go through the same entry
point, which picks a backend from the circuit's structure. Circuits pass through a
multi-pass gate fusion pipeline, and the inner loops use AVX2, FMA and BMI2 (NEON on
ARM64). Input is OpenQASM 3.0 with backward-compatible 2.0 syntax, and a circuit
exports back to OpenQASM 3.0.

- Nine CPU backends: statevector, stabilizer, factored stabilizer, sparse, MPS, product
  state, tensor network (on expectation and marginal queries) and factored, picked
  automatically, plus an exact density matrix selected by name.
- Compiled shot samplers that do not rebuild the statevector for every shot, including
  the noisy and detector sampling paths.
- Four Clifford+T engines (stabilizer rank, stochastic and deterministic Pauli
  propagation, and Pauli path) for circuits a dense statevector cannot hold.
- Noise models built from rules or a device calibration table, with leakage, drift and
  timed idle noise, and a native QEC program path with detector sampling and decoding.
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

# Tutorials

Each tutorial starts from an empty file and ends at a number you can check, in a few
minutes. Code is Python first, with the Rust equivalent where the two differ in more
than syntax. Every Python block on these pages runs in CI, as does every Rust block.

| Tutorial | You end with |
|----------|--------------|
| [Simulate and Sample](./simulate-and-sample.md) | GHZ probabilities, shot counts, and a backend picked by hand |
| [Variational Circuits and Gradients](./variational.md) | An optimized Ising energy from adjoint gradients, and a batched parameter sweep |
| [Noisy Simulation](./noisy-simulation.md) | GHZ fidelity under gate noise, readout error, and a device calibration |
| [A QEC Memory Experiment](./qec-memory.md) | Decoded logical error rates for a repetition code at three distances |
| [Dynamic Circuits](./dynamic-circuits.md) | Teleportation with mid-circuit measurement and feed-forward |
| [Large and Structured Circuits](./large-circuits.md) | 1000-qubit Clifford sampling, a Clifford+T expectation, and an MPS run with its error bound |

Install the package first:

```bash
pip install prism-q
```

Each tutorial has a runnable script in
[`bindings/python/examples/`](https://github.com/AbeCoull/prism-q/tree/main/bindings/python/examples),
and the Rust programs in
[`examples/`](https://github.com/AbeCoull/prism-q/tree/main/examples) cover the same
ground:

```bash
python bindings/python/examples/qec_memory.py
cargo run --release --example qec_memory
```

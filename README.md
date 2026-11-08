# PRISM-Q: quantum circuit simulator for Rust and Python

PRISM-Q picks a state representation from each circuit's structure: a stabilizer
tableau for Clifford circuits, a dense statevector for general ones, and a sparse map,
matrix product state, product state or factored form where the circuit allows. One
`simulate` call covers all of them, and the result names the backend that ran.

[![Crates.io](https://img.shields.io/crates/v/prism-q?logo=rust)](https://crates.io/crates/prism-q)
[![PyPI](https://img.shields.io/pypi/v/prism-q?logo=pypi&logoColor=white)](https://pypi.org/project/prism-q/)
[![docs.rs](https://img.shields.io/docsrs/prism-q?logo=docsdotrs&logoColor=white)](https://docs.rs/prism-q)
[![CI](https://github.com/AbeCoull/prism-q/actions/workflows/ci.yml/badge.svg)](https://github.com/AbeCoull/prism-q/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)

```bash
pip install prism-q    # Python 3.11+, wheels for Linux, macOS arm64 and Windows
cargo add prism-q      # Rust 1.89+
```

## Quick start

```python
import prism_q

bell = prism_q.CircuitBuilder(2).h(0).cx(0, 1).build()
print(prism_q.simulate(bell).seed(42).run().probabilities)  # [0.5 0.  0.  0.5]

measured = prism_q.CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
print(prism_q.simulate(measured).seed(42).shots(1000).counts())
# {'00': 507, '11': 493}: sampled, so the split moves with the seed
```

```rust
use prism_q::{simulate, CircuitBuilder};

let bell = CircuitBuilder::new(2).h(0).cx(0, 1).build();
let result = simulate(&bell).seed(42).run().unwrap();
println!("{:?}", result.probabilities.unwrap().to_vec()); // [0.5, 0.0, 0.0, 0.5]
```

Count keys are LSB-first: the leftmost character is classical bit 0, and `q[0]` is the
least significant bit of a statevector index. The Python package wraps the same engine
and returns NumPy arrays.

## Backend selection

```python
from prism_q import CircuitBuilder, circuits, simulate

def backend(circuit):
    return simulate(circuit).seed(42).run().metadata.backend

print(backend(circuits.qft(16)))    # Statevector
print(backend(circuits.ghz(1000)))  # Stabilizer

ghz_t = CircuitBuilder(60).h(0)
for q in range(59):
    ghz_t.cx(q, q + 1)
print(backend(ghz_t.t(0).build()))  # Sparse
```

The 1,000-qubit GHZ circuit is all Clifford, so a 2n × 2n bit tableau replaces 2^1000
amplitudes. One T gate rules out the tableau, but only two basis states are ever
nonzero, so the sparse map runs it. Auto dispatch reads structure and the memory cap
without timing alternatives; `.backend(...)` overrides it. The
[decision tree](https://abecoull.github.io/prism-q/architecture/engine.html#auto-dispatch-decision-tree)
shows every route.

| Backend | Best for | Scaling |
| --- | --- | --- |
| Statevector | General circuits | O(2ⁿ) |
| Stabilizer | Clifford only | O(n²) |
| Factored Stabilizer | Clifford with independent blocks | O(n²) per block |
| Sparse | Few live amplitudes | O(k) |
| MPS | Low entanglement or 1D | O(nχ²) |
| Product State | No entanglement | O(n) |
| Tensor Network | Low treewidth, expectation and marginal queries | Depends on contraction order |
| Factored | Partial entanglement | O(2ᵏ) per block of k qubits |
| Density Matrix | Exact noisy evolution, selected by name | O(4ⁿ) |

Four Clifford+T engines (stabilizer rank, stochastic and deterministic Pauli
propagation, Pauli path) sit beside these, with a CUDA path and an MPI-distributed
statevector behind features.

## Features

- OpenQASM 3.0 and the compatible 2.0 subset in, OpenQASM 3.0 out
  ([support table](https://abecoull.github.io/prism-q/guides/openqasm.html)).
- Shots, probabilities, expectation values and marginals from one builder.
- Parameter binding, a reusable fusion plan across bindings, and adjoint or
  parameter-shift gradients.
- Noise models with leakage and drift; QEC programs with detector sampling and
  decoding ([QEC guide](https://abecoull.github.io/prism-q/guides/qec.html)).
- `gpu`: CUDA statevector, density-matrix and stabilizer backends, driver and NVRTC
  loaded at run time ([GPU guide](https://abecoull.github.io/prism-q/guides/gpu.html)).
- `distributed`: one exact statevector sharded over MPI ranks
  ([distributed guide](https://abecoull.github.io/prism-q/guides/distributed.html)).

## Performance

A multi-pass fusion pipeline merges gates before a circuit runs. Statevector kernels use
AVX2, FMA and BMI2 on x86-64 and NEON on ARM64, chosen at run time, and split across
Rayon threads from 15 qubits. Timings are on
[Benchmarks](https://abecoull.github.io/prism-q/benchmarks.html); the harness behind
[Comparative Measurements](https://abecoull.github.io/prism-q/comparison.html) is
committed in [`comparison/`](comparison/README.md).

<!-- comparison-preview:start -->

[![Comparator time over PRISM-Q time per workload](docs/comparison/figures/ratio-heatmap.svg)](docs/comparison/figures/ratio-heatmap.svg)

Time ratios for 64 validated comparisons against 8 comparator configurations on one
desktop: [comparison pages](https://abecoull.github.io/prism-q/comparison.html).

<!-- comparison-preview:end -->

## Documentation

| Task | Page |
| --- | --- |
| Install and first steps | [Installation](https://abecoull.github.io/prism-q/getting-started/install.html) |
| Work through a task end to end | [Tutorials](https://abecoull.github.io/prism-q/tutorials/index.html) |
| Use the simulator from Python | [Python Bindings](https://abecoull.github.io/prism-q/guides/python.html) |
| Look up the Rust API | [docs.rs/prism-q](https://docs.rs/prism-q) |
| Pick or override a backend | [Choosing a Backend](https://abecoull.github.io/prism-q/getting-started/choosing-a-backend.html) |
| Check what runs where | [Capability Matrix](https://abecoull.github.io/prism-q/guides/capabilities.html) |
| Add noise, leakage and drift | [Noise Models](https://abecoull.github.io/prism-q/guides/noise.html) |
| Simulate Clifford+T past the dense limit | [Clifford+T Simulation](https://abecoull.github.io/prism-q/guides/clifford-t.html) |
| Tune threads, tiles and memory caps | [Performance and SIMD](https://abecoull.github.io/prism-q/guides/performance.html) |
| Read how the layers fit together | [Architecture](https://abecoull.github.io/prism-q/architecture/overview.html) |
| Watch one gate go from matrix to SIMD lanes | [Interactive atlas](https://abecoull.github.io/prism-q/atlas/) |

Runnable scripts are in [`bindings/python/examples`](bindings/python/examples) and
[`examples`](examples); open work is on the
[roadmap](https://abecoull.github.io/prism-q/roadmap.html).

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) has the build, test and benchmark workflow. Bug
reports and questions go to [issues](https://github.com/AbeCoull/prism-q/issues) and
[discussions](https://github.com/AbeCoull/prism-q/discussions). To cite PRISM-Q, use
[`CITATION.cff`](CITATION.cff) and name the release.

## License

Dual-licensed under MIT or Apache-2.0, at your option.

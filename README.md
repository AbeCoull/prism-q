# PRISM-Q: High-Performance Quantum Circuit Simulator in Rust

PRISM-Q is an open-source quantum circuit simulator written in Rust, with Python
bindings. It reads OpenQASM 3.0 (with backward-compatible 2.0 syntax), selects a
simulation backend from the circuit's structure, runs a multi-pass gate fusion pipeline,
and executes on SIMD kernels: AVX2, FMA and BMI2 on x86-64, NEON on ARM64. Two optional
features extend the dense path: CUDA acceleration for the statevector, stabilizer
(experimental) and density-matrix backends, and an MPI-distributed statevector for
registers larger than one host's memory.

The simulation methods are a dense statevector, a stabilizer tableau, a sparse amplitude
map, matrix product states (MPS), a product state, a tensor network, an exact density
matrix, and two factored variants that split independent registers. Beside the backends
sit Clifford+T engines (stabilizer rank and Pauli propagation) and a native quantum
error-correction (QEC) program path with detector sampling and decoding. Circuits export
back to OpenQASM 3.0.

[![Crates.io](https://img.shields.io/crates/v/prism-q?logo=rust)](https://crates.io/crates/prism-q)
[![docs.rs](https://img.shields.io/docsrs/prism-q?logo=docsdotrs&logoColor=white)](https://docs.rs/prism-q)
[![PyPI](https://img.shields.io/pypi/v/prism-q?logo=pypi&logoColor=white)](https://pypi.org/project/prism-q/)
[![CI](https://github.com/AbeCoull/prism-q/actions/workflows/ci.yml/badge.svg)](https://github.com/AbeCoull/prism-q/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/AbeCoull/4ea63a3791840048749e67b2484098a3/raw/coverage.json)
![MSRV](https://img.shields.io/crates/msrv/prism-q?logo=rust)
![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)
![OpenQASM](https://img.shields.io/badge/OpenQASM-3.0-purple)

```text
 ██████╗ ██████╗ ██╗███████╗███╗   ███╗       ██████╗
 ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║      ██╔═══██╗
 ██████╔╝██████╔╝██║███████╗██╔████╔██║█████╗██║   ██║
 ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║╚════╝██║▄▄ ██║
 ██║     ██║  ██║██║███████║██║ ╚═╝ ██║      ╚██████╔╝
 ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝       ╚══▀▀═╝
```

- Documentation: <https://abecoull.github.io/prism-q/> (machine-readable index at
  [`llms.txt`](https://abecoull.github.io/prism-q/llms.txt))
- API reference: [docs.rs](https://docs.rs/prism-q)
- Measured timings: [Benchmarks](https://abecoull.github.io/prism-q/benchmarks.html),
  with the method in
  [Benchmark Methodology](https://abecoull.github.io/prism-q/guides/benchmarking.html)
- Reproducible measurements beside Qiskit Aer, qsim, QuEST, Spinoza and RustQIP:
  [Comparative Measurements](https://abecoull.github.io/prism-q/comparison.html)
- Design: [Architecture reference](https://abecoull.github.io/prism-q/architecture/overview.html)
- What each backend supports on CPU and GPU:
  [capability matrix](https://abecoull.github.io/prism-q/guides/capabilities.html)
- Changes by release: [`CHANGELOG.md`](CHANGELOG.md)

## Install

```bash
cargo add prism-q                          # Rayon parallelism and faer SVD (default)
cargo add prism-q --no-default-features    # single-threaded, minimal dependencies
pip install prism-q                        # Python
pip install "prism-q[cuda12]"              # Python, plus NVRTC for the CUDA backends
```

The `gpu` feature builds without a CUDA toolkit and loads the NVIDIA driver and NVRTC at
run time; running it needs a CUDA device, a CUDA 12 driver, and NVRTC from a CUDA 12
toolkit or the `cuda12` Python extra. Build with
`cargo build --release --features "parallel gpu"`. Building from source and pinning a
git revision are covered in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quick start

```rust
use prism_q::run_qasm;

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

let result = run_qasm(qasm, 42).unwrap();
println!("{:?}", result.probabilities);
// Bell state: ~50% |00⟩, ~50% |11⟩
```

The Python bindings in [`bindings/python`](bindings/python) expose simulation, noise,
QEC and the CUDA backends, with NumPy output:

```python
import prism_q

circuit = prism_q.CircuitBuilder(2).h(0).cx(0, 1).build()
print(prism_q.simulate(circuit).seed(42).run().probabilities)  # [0.5, 0, 0, 0.5]
```

Count keys and measurement bits are LSB-first (`q[0]` is the least significant qubit),
the reverse of Qiskit. See the
[Python guide](https://abecoull.github.io/prism-q/guides/python.html).

### Shots

```rust
use prism_q::{bitstring, circuit::openqasm, simulate};

let circuit = openqasm::parse(qasm).unwrap();
let result = simulate(&circuit).seed(42).shots(1024).unwrap();
println!("{result}");
// 00: 512
// 11: 512

let counts = simulate(&circuit)
    .seed(42)
    .sample_counts(1024)
    .unwrap();
for (bits, count) in counts.into_counts() {
    println!("{}: {count}", bitstring(&bits, circuit.num_classical_bits));
}
```

### Expectation values and marginals

```rust
use prism_q::{simulate, CircuitBuilder, PauliTerm};

let bell = CircuitBuilder::new(2).h(0).cx(0, 1).build();

let observables = [
    vec![PauliTerm::z(0), PauliTerm::z(1)],
    vec![PauliTerm::x(0), PauliTerm::x(1)],
];
let values = simulate(&bell).seed(42).expectation_values(&observables).unwrap();
// [1.0, 1.0]: ⟨ZZ⟩ and ⟨XX⟩ on the Bell state.

let marginals = simulate(&bell).seed(42).marginals().unwrap();
// Per-qubit (P(0), P(1)) pairs: [(0.5, 0.5), (0.5, 0.5)].
```

An observable is a product of single-qubit Paulis with identity factors omitted.
Clifford circuits propagate it exactly, and past the statevector memory budget the
selected backend answers from its own representation. To start somewhere other than
|0...0⟩, pass `initial_state` a normalized amplitude vector of length 2^n, qubit 0 in
the least significant bit.

### Parameters and gradients

Mark angles as parameters while building, then rebind without rebuilding.
`PreparedCircuit` reuses one fusion plan across bindings and falls back to a full
fusion pass when a binding changes what fusion would emit.

```rust
use prism_q::{simulate, CircuitBuilder, PauliTerm, PreparedCircuit};

let (template, params) = CircuitBuilder::new(2)
    .ry(0.0, 0)
    .param(0)
    .cx(0, 1)
    .rz(0.0, 1)
    .param(1)
    .build_parametric();

let bound = params.bind(&template, &[0.3, 1.1]).unwrap();
let result = simulate(&bound).seed(42).run().unwrap();

let mut prepared = PreparedCircuit::new(template, params.clone()).unwrap();
let fused = prepared.bind_fused(&[0.4, 1.2]).unwrap();

let hamiltonian = vec![(1.0, vec![PauliTerm::z(0)])];
let g = simulate(&bound)
    .seed(42)
    .expectation_gradient(&hamiltonian, &params)
    .unwrap();
println!("<H> = {}, gradient = {:?}", g.value, g.gradient);
```

`expectation_gradient` uses the adjoint method on the statevector backend.
`expectation_gradient_shift` uses the parameter-shift rule and covers the backends and
circuit shapes the adjoint declines.

## Learn

The [Tutorials](https://abecoull.github.io/prism-q/tutorials/index.html) walk through
sampling, variational gradients, noise, a QEC memory experiment, dynamic circuits and
wide structured circuits, in Python with the Rust equivalents. Runnable scripts sit in
[`bindings/python/examples`](bindings/python/examples) and [`examples`](examples):

```bash
python bindings/python/examples/qec_memory.py
cargo run --release --example qec_memory
```

## Backends and automatic selection

| Backend | Best for | Scaling | Key property |
| --- | --- | --- | --- |
| Statevector | General circuits | O(2ⁿ) | SIMD, tiled L2/L3 kernels, optional CUDA path |
| Stabilizer | Clifford only | O(n²) | SIMD, thousands of qubits |
| Factored Stabilizer | Clifford with independent blocks | O(n²) per cluster | Per-cluster tableaux, dynamic merge and split |
| Sparse | Few live amplitudes | O(k) | HashMap with parallel measurement |
| MPS | Low entanglement or 1D | O(nχ²) | Hybrid faer / Jacobi SVD |
| Product State | No entanglement | O(n) | Per qubit |
| Tensor Network | Low treewidth | Depends on contraction order | Greedy min size heuristic |
| Factored | Partial entanglement | Dynamic | Tracks independent sub-states |
| Density Matrix | Exact noisy evolution | O(4ⁿ) | Explicit dispatch only, reuses statevector kernels |
| Distributed Statevector | Beyond single-host memory | O(2ⁿ) over MPI ranks | `distributed` feature, exact results |

`BackendKind::Auto` is the default: PRISM-Q selects a backend from the circuit's
structure and the host's memory, and the run metadata reports which one ran. The
choice is a shape rule, not a cost model, so an explicit backend can still win for a
circuit the rules misjudge. Circuits with no entangling gates go to Product
State. All-Clifford circuits go to Stabilizer, or to Factored Stabilizer when a large
circuit splits into independent blocks. Circuits past the statevector memory budget go
to Sparse when the state never holds more than 128 basis states and to MPS with bond
dimension 256 otherwise; partially independent circuits go to Factored, and the rest run
on Statevector. Before that tree, a Clifford+T circuit with few T gates can take the
stabilizer-rank sampler for shots and Pauli propagation for marginals.
The budget is half the machine's physical memory, read once and cached;
`PRISM_MAX_SV_QUBITS` overrides it. Density Matrix and Distributed Statevector are never
selected automatically; name them.

To choose a backend yourself, call `.backend(BackendKind::Stabilizer)` (or
`BackendKind::Mps { max_bond_dim: 64 }`, `BackendKind::Sparse`, and so on) on the
`simulate` builder. [Choosing a backend](https://abecoull.github.io/prism-q/getting-started/choosing-a-backend.html)
and the [backends deep dive](https://abecoull.github.io/prism-q/guides/backends.html)
cover when each one wins.

## OpenQASM

The parser accepts the `stdgates.inc` set, common controlled and multi-controlled
variants, Qiskit exporter gates, IonQ and Google/Cirq native gate names, decomposed
multi-instruction gates, IBM legacy u1/u2/u3, and user-defined `gate` declarations.
The `inv @`, `ctrl @` and `pow(k) @` modifiers chain on direct gates.
`qasm_export::to_qasm3` writes a `Circuit` back out as OpenQASM 3.0 that re-parses to
the same instruction stream with inline angles exact;
fused payloads have no OpenQASM spelling and are rejected with an error naming the
instruction.

The [OpenQASM guide](https://abecoull.github.io/prism-q/guides/openqasm.html) lists the
accepted subset. The parser itself, [`src/circuit/openqasm.rs`](src/circuit/openqasm.rs)
(`resolve_gate()` and `resolve_decomposed_gate()`), is the authoritative gate list.

## GPU

The `gpu` feature compiles PTX at runtime through NVRTC for the device's compute
capability. Opt in through the simulation builder:

```rust
use prism_q::{gpu::GpuContext, simulate};

let ctx = GpuContext::new(0)?;
let result = simulate(&circuit).gpu(ctx).seed(42).run()?;
```

The circuit still goes through fusion and subsystem decomposition, and a size crossover
keeps small sub-circuits on the CPU. `simulate(&circuit).gpu_auto(ctx)` runs automatic
dispatch with the device opted in, `BackendKind::StabilizerGpu` runs Clifford circuits
on the device, and `CompiledSampler::with_gpu(ctx)` moves large compiled BTS shot
counts onto it. The [GPU guide](https://abecoull.github.io/prism-q/guides/gpu.html)
covers the kernels, the crossover thresholds and their environment overrides.

## Distributed execution (MPI)

The `distributed` feature shards one exact statevector across `2^p` ranks: the top `p`
qubits select the rank and the rest index a local slice that runs on the same SIMD
kernels and fusion pipeline as the single-host backend. `distributed-mpi` adds the MPI
transport over `rsmpi` and needs a system MPI installation.

```rust
use prism_q::{distributed::DistributedContext, simulate};

let context = DistributedContext::world()?;   // MPI_Init at MPI_THREAD_FUNNELED
let result = simulate(&circuit).distributed(context).seed(42).run()?;
```

Results do not depend on the rank count. The
[distributed guide](https://abecoull.github.io/prism-q/guides/distributed.html) covers
the layout, the qubit relabeling that keeps busy qubits local, the launch scripts, and
the limits (power-of-two ranks, no noise models, CPU only).

## Benchmarks

```bash
cargo bench --bench circuits     --features parallel         # circuit macrobenchmarks
cargo bench --bench bench_driver --features parallel         # gate microbenchmarks
cargo bench --bench bench_gpu    --features "parallel gpu"   # GPU dispatch benchmarks
```

Baselines were taken with `parallel` enabled, so keep it on. Run one `cargo bench` at a
time: concurrent Rayon pools contend for cores and skew results. `RAYON_NUM_THREADS`
caps the thread count. The published timings, the circuit definitions, the warmup and
timing rules, and how to run a like-for-like comparison against another simulator are in
[Benchmark Methodology](https://abecoull.github.io/prism-q/guides/benchmarking.html).
The A/B and regression workflow used for PRs is in [`CONTRIBUTING.md`](CONTRIBUTING.md)
and [`benches/README.md`](benches/README.md).

## Documentation map

| Task | Page |
| --- | --- |
| Install the crate or the wheel | [Installation](https://abecoull.github.io/prism-q/getting-started/install.html) |
| Work through a task end to end | [Tutorials](https://abecoull.github.io/prism-q/tutorials/index.html) |
| Let the dispatcher pick a backend, or override it | [Choosing a Backend](https://abecoull.github.io/prism-q/getting-started/choosing-a-backend.html) |
| Compare the backends' memory and scaling | [Backends Deep Dive](https://abecoull.github.io/prism-q/guides/backends.html) |
| Use the simulator from Python | [Python Bindings](https://abecoull.github.io/prism-q/guides/python.html) |
| Check which OpenQASM constructs parse | [OpenQASM Support](https://abecoull.github.io/prism-q/guides/openqasm.html) |
| Tune threads, tiles and memory caps | [Performance and SIMD](https://abecoull.github.io/prism-q/guides/performance.html) |
| Run on a CUDA device | [GPU Backend](https://abecoull.github.io/prism-q/guides/gpu.html) |
| Run across MPI ranks | [Distributed Statevector](https://abecoull.github.io/prism-q/guides/distributed.html) |
| Simulate Clifford+T circuits past the dense limit | [Clifford+T Simulation](https://abecoull.github.io/prism-q/guides/clifford-t.html) |
| Add noise, detectors and decoding | [Noise and QEC](https://abecoull.github.io/prism-q/guides/qec.html) |
| Reproduce or extend the benchmarks | [Benchmark Methodology](https://abecoull.github.io/prism-q/guides/benchmarking.html) |
| See how it measures beside other simulators | [Comparative Measurements](https://abecoull.github.io/prism-q/comparison.html), harness in [`comparison/`](comparison/README.md) |
| Read how the layers fit together | [Architecture](https://abecoull.github.io/prism-q/architecture/overview.html) |

## Roadmap

- Mid-circuit branching beyond the current `if` form.
- Multi-GPU and distributed GPU execution. A GPU context binds one device and the
  distributed backend is CPU only; sharding one statevector across devices also needs
  peer access, since a host-staged exchange costs far more than the gate it serves.
- ROCm ports of the CUDA statevector and stabilizer kernels.
- Noisy shots on the distributed backend, which rejects noise models today because
  trajectory execution is not lockstep across ranks.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) has the build, test, coverage, profiling and
benchmark workflow. The [architecture reference](docs/architecture/overview.md) covers
the layered design, backend trait, SIMD strategy, fusion pipeline and compiled samplers.

## Citing

[`CITATION.cff`](CITATION.cff) carries the citation metadata; GitHub renders it under
"Cite this repository". Name the release version used, since timings and dispatch
rules change between releases.

## License

Dual-licensed under MIT or Apache-2.0, at your option.

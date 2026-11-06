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

The same circuit in Rust:

```rust
use prism_q::{simulate, CircuitBuilder};

let bell = CircuitBuilder::new(2).h(0).cx(0, 1).build();
let result = simulate(&bell).seed(42).run().unwrap();
println!("{:?}", result.probabilities.unwrap().to_vec()); // [0.5, 0.0, 0.0, 0.5]
```

The Python package wraps the same engine, including noise models, QEC programs and the
CUDA backends, and returns NumPy arrays.

## One API, several representations

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

The 16-qubit QFT runs on a dense statevector. The 1,000-qubit GHZ circuit is all
Clifford, so a 2n × 2n bit tableau holds it in place of 2^1000 amplitudes. One T gate on
a 60-qubit GHZ circuit rules out the tableau, but only two of its 2^60 basis states are
ever nonzero, so it runs on a sparse map.

`BackendKind::Auto` reads the circuit's structure and the statevector memory cap and
does not time the alternatives, so naming a backend can beat it on a circuit it
misjudges. [Simulation Engine and Dispatch](https://abecoull.github.io/prism-q/architecture/engine.html#auto-dispatch-decision-tree)
draws the decision tree and lists the routes outside it: one backend per block for circuits
that split, compiled samplers for Clifford shots, stabilizer rank for Clifford+T shots
with few T gates, and Pauli propagation or tensor contraction for some expectation and
marginal queries. The density matrix and distributed statevector are never picked
automatically, and the GPU runs only when a device context is passed in.

## Performance

A multi-pass fusion pipeline merges adjacent gates into larger kernels before a circuit
runs. The statevector kernels use AVX2, FMA and BMI2 on x86-64 and NEON on ARM64,
picked at run time, and split across Rayon threads from 15 qubits. Optional features add
CUDA execution for the statevector, density-matrix and (experimental) stabilizer
backends, and an MPI-distributed statevector for registers larger than one machine's
memory.

Timings are on [Benchmarks](https://abecoull.github.io/prism-q/benchmarks.html), and the
method on [Benchmark Methodology](https://abecoull.github.io/prism-q/guides/benchmarking.html).
[Comparative Measurements](https://abecoull.github.io/prism-q/comparison.html) runs the
same circuits through other simulators from a committed harness in
[`comparison/`](comparison/README.md), so every number there can be reproduced.

<!-- comparison-preview:start -->

[![Comparator time over PRISM-Q time per workload](docs/comparison/figures/ratio-heatmap.svg)](docs/comparison/figures/ratio-heatmap.svg)

Time ratios for 64 validated comparisons against 8 comparator configurations on one
desktop: [comparison pages](https://abecoull.github.io/prism-q/comparison.html).

<!-- comparison-preview:end -->

## Documentation

| Task | Page |
| --- | --- |
| Install the crate or the wheel | [Installation](https://abecoull.github.io/prism-q/getting-started/install.html) |
| Work through a task end to end | [Tutorials](https://abecoull.github.io/prism-q/tutorials/index.html) |
| Use the simulator from Python | [Python Bindings](https://abecoull.github.io/prism-q/guides/python.html) |
| Look up the Rust API | [docs.rs/prism-q](https://docs.rs/prism-q) |
| Let the dispatcher pick a backend, or override it | [Choosing a Backend](https://abecoull.github.io/prism-q/getting-started/choosing-a-backend.html) |
| Compare the backends' memory and scaling | [Choosing a Backend](https://abecoull.github.io/prism-q/getting-started/choosing-a-backend.html#scaling) |
| Check what runs where, and what is experimental | [Capability Matrix](https://abecoull.github.io/prism-q/guides/capabilities.html) |
| Check which OpenQASM constructs parse | [OpenQASM Support](https://abecoull.github.io/prism-q/guides/openqasm.html) |
| Add noise models, leakage and drift | [Noise Models](https://abecoull.github.io/prism-q/guides/noise.html) |
| Build QEC programs, detectors and decoders | [Quantum Error Correction](https://abecoull.github.io/prism-q/guides/qec.html) |
| Simulate Clifford+T circuits past the dense limit | [Clifford+T Simulation](https://abecoull.github.io/prism-q/guides/clifford-t.html) |
| Tune threads, tiles and memory caps | [Performance and SIMD](https://abecoull.github.io/prism-q/guides/performance.html) |
| Run on a CUDA device | [GPU Backend](https://abecoull.github.io/prism-q/guides/gpu.html) |
| Run across MPI ranks | [Distributed Statevector](https://abecoull.github.io/prism-q/guides/distributed.html) |
| Reproduce or extend the benchmarks | [Benchmark Methodology](https://abecoull.github.io/prism-q/guides/benchmarking.html) |
| Read how the layers fit together | [Architecture](https://abecoull.github.io/prism-q/architecture/overview.html) |
| Watch one gate go from matrix to SIMD lanes | [Interactive atlas](https://abecoull.github.io/prism-q/atlas/) ([source](docs/atlas/README.md)) |

Runnable scripts sit in [`bindings/python/examples`](bindings/python/examples) and
[`examples`](examples). Changes by release are in [`CHANGELOG.md`](CHANGELOG.md), and a
plain-text index of the docs for tools is at
[`llms.txt`](https://abecoull.github.io/prism-q/llms.txt).

## More examples

### Shots from OpenQASM

```rust
use prism_q::{bitstring, circuit::openqasm, simulate};

let qasm = r#"
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    cx q[0], q[1];
    c = measure q;
"#;

let circuit = openqasm::parse(qasm).unwrap();
let result = simulate(&circuit).seed(42).shots(1024).unwrap();
println!("{result}");
// 00: 517
// 11: 507

let counts = simulate(&circuit)
    .seed(42)
    .sample_counts(1024)
    .unwrap();
for (bits, count) in counts.into_counts() {
    println!("{}: {count}", bitstring(&bits, circuit.num_classical_bits));
}
```

Count keys and measurement bits are LSB-first: the leftmost character is classical bit
0, and `q[0]` is the least significant qubit of a statevector index. This is the reverse
of Qiskit's order. `run_qasm(source, seed)` parses and runs in one call.

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
Clifford circuits propagate it exactly; past the statevector memory budget the selected
backend answers from its own representation. To start from a state other than |0...0⟩,
pass `initial_state` a normalized amplitude vector of length 2^n, qubit 0 in the least
significant bit.

### Parameters and gradients

Mark angles as parameters while building, then rebind without rebuilding.
`PreparedCircuit` reuses one fusion plan across bindings and reruns full fusion when a
binding changes what fusion would emit.

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

The [Tutorials](https://abecoull.github.io/prism-q/tutorials/index.html) cover sampling,
variational gradients, noise, a QEC memory experiment, dynamic circuits and wide
structured circuits, in Python with the Rust equivalents:

```bash
python bindings/python/examples/qec_memory.py
cargo run --release --example qec_memory
```

## Backends

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

The distributed statevector is an optional path outside these nine: the `distributed`
feature shards one exact statevector over MPI ranks. Beside them sit four Clifford+T
engines (stabilizer rank, stochastic and deterministic Pauli propagation, Pauli path)
and a native quantum error-correction program path with detector sampling and decoding.
The [backends reference](https://abecoull.github.io/prism-q/architecture/backends.html) covers
memory layout and when each one wins.

The statevector memory budget is half the machine's physical memory, read once and
cached, which is about 29 qubits on a 16 GiB host. `Limits` overrides it or any single
cap, per request with `.limits(...)` or per process with `set_default_limits`; the
`PRISM_MAX_*` variables set the same caps from the environment.

To pick a backend yourself, call `.backend(BackendKind::Stabilizer)` (or
`BackendKind::Mps { max_bond_dim: 64 }`, `BackendKind::Sparse`, and so on) on the
`simulate` builder.

## OpenQASM

The parser reads OpenQASM 3.0 and the compatible subset of 2.0: the `stdgates.inc` set,
common controlled and multi-controlled variants, Qiskit exporter gates, IonQ and
Google/Cirq native gate names, decomposed multi-instruction gates, IBM legacy u1/u2/u3,
and user-defined `gate` declarations. The `inv @`, `ctrl @` and `pow(k) @` modifiers
chain on direct gates. `qasm_export::to_qasm3` writes a `Circuit` back out as OpenQASM
3.0 that re-parses to the same instruction stream, with inline angles exact; fused
payloads have no OpenQASM spelling and return an error naming the instruction.

The [OpenQASM guide](https://abecoull.github.io/prism-q/guides/openqasm.html) lists the
accepted subset and what each unsupported construct returns. The authoritative gate list
is `resolve_gate()` and `resolve_decomposed_gate()` in
[`src/circuit/openqasm.rs`](src/circuit/openqasm.rs).

## GPU

The `gpu` feature builds without a CUDA toolkit and loads the NVIDIA driver and NVRTC
at run time, compiling PTX for the device's compute capability. Running it needs a CUDA
device, a CUDA 12 driver, and NVRTC from a CUDA 12 toolkit or the `cuda12` Python extra.
Build with `cargo build --release --features "parallel gpu"`, then opt in through the
simulation builder:

```rust,ignore
use prism_q::{gpu::GpuContext, simulate};

let ctx = GpuContext::new(0)?;
let result = simulate(&circuit).gpu(ctx).seed(42).run()?;
```

Fusion and subsystem decomposition still run, and a size crossover keeps small
sub-circuits on the CPU. `simulate(&circuit).gpu_auto(ctx)` runs automatic
dispatch with the device opted in, `BackendKind::StabilizerGpu` runs Clifford circuits
on the device, and `CompiledSampler::with_gpu(ctx)` moves large compiled BTS shot
counts onto it. The [GPU guide](https://abecoull.github.io/prism-q/guides/gpu.html)
covers the kernels, the crossover thresholds and their environment overrides.

## Distributed execution (MPI)

The `distributed` feature shards one exact statevector across `2^p` ranks: the top `p`
qubits select the rank and the rest index a local slice that runs on the same SIMD
kernels and fusion pipeline as the single-host backend. `distributed-mpi` adds the MPI
transport over `rsmpi` and needs a system MPI installation.

```rust,ignore
use prism_q::{distributed::DistributedContext, simulate};

let context = DistributedContext::world()?;   // MPI_Init at MPI_THREAD_FUNNELED
let result = simulate(&circuit).distributed(context).seed(42).run()?;
```

Results do not depend on the rank count. The
[distributed guide](https://abecoull.github.io/prism-q/guides/distributed.html) covers
the layout, the qubit relabeling that keeps busy qubits local, lockstep noisy shots, the
launch scripts, and the limits (power-of-two ranks, CPU only).

## Benchmarks

```bash
cargo bench --bench circuits     --features parallel         # circuit macrobenchmarks
cargo bench --bench bench_driver --features parallel         # gate microbenchmarks
cargo bench --bench bench_gpu    --features "parallel gpu"   # GPU dispatch benchmarks
```

Baselines were taken with `parallel` enabled, so keep it on. Run one `cargo bench` at a
time, since concurrent Rayon pools contend for cores. `RAYON_NUM_THREADS` caps the
thread count. Circuit definitions, warmup and timing rules, and like-for-like runs
against another simulator are in
[Benchmark Methodology](https://abecoull.github.io/prism-q/guides/benchmarking.html);
the A/B and regression workflow for PRs is in [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`benches/README.md`](benches/README.md).

## Roadmap

Open work is grouped under QEC and noise, stabilizer and Clifford+T simulation, large
structured circuits, input and interop, the GPU, CPU reach, and running beyond one
device. The [roadmap](https://abecoull.github.io/prism-q/roadmap.html) tags each item
with what it waits on and lists hardware that would unblock some of them.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) has the build, test, coverage, profiling and
benchmark workflow, and the [architecture reference](docs/architecture/overview.md)
covers the layers, backend trait, SIMD strategy, fusion pipeline and compiled samplers.
Bug reports and questions go to [issues](https://github.com/AbeCoull/prism-q/issues)
and [discussions](https://github.com/AbeCoull/prism-q/discussions).

## Citing

[`CITATION.cff`](CITATION.cff) holds the citation metadata, which GitHub shows under
"Cite this repository". Name the release used, since timings and dispatch rules change
between releases.

## License

Dual-licensed under MIT or Apache-2.0, at your option.

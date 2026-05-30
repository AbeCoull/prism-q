# PRISM-Q

```text
 ██████╗ ██████╗ ██╗███████╗███╗   ███╗       ██████╗
 ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║      ██╔═══██╗
 ██████╔╝██████╔╝██║███████╗██╔████╔██║█████╗██║   ██║
 ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║╚════╝██║▄▄ ██║
 ██║     ██║  ██║██║███████║██║ ╚═╝ ██║      ╚██████╔╝
 ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝       ╚══▀▀═╝
```

[![CI](https://github.com/AbeCoull/prism-q/actions/workflows/ci.yml/badge.svg)](https://github.com/AbeCoull/prism-q/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/AbeCoull/4ea63a3791840048749e67b2484098a3/raw/coverage.json)
![Rust](https://img.shields.io/badge/rust-1.75%2B-orange?logo=rust)
![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)
![OpenQASM](https://img.shields.io/badge/OpenQASM-3.0-purple)

PRISM-Q is a Rust quantum circuit simulator built for speed. It dispatches across
multiple specialized backends, runs a multi pass fusion pipeline, and uses AVX2, FMA,
and BMI2 SIMD kernels in the inner loop. CPU kernels are the default path, with
optional CUDA support for statevector and experimental stabilizer workloads. Input is
OpenQASM 3.0 with backward compatible 2.0 syntax.

## Documentation

Full documentation is published at <https://abecoull.github.io/prism-q/>. The generated
API reference is on [docs.rs](https://docs.rs/prism-q).

## Installation

Add PRISM-Q to a Rust project:

```bash
cargo add prism-q
```

Rayon parallelism and the faer SVD path are on by default. For a single-threaded,
minimal-dependency build, opt out:

```bash
cargo add prism-q --no-default-features
```

For CUDA support, install CUDA Toolkit 12.x or newer, then build with:

```bash
cargo build --release --features "parallel gpu"
```

Building from source or pinning to a git revision is covered in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

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

### Shot-based sampling

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

### Backend dispatch

```rust
use prism_q::{circuit::openqasm, simulate, BackendKind};

let circuit = openqasm::parse(qasm).unwrap();

// Auto picks the optimal backend based on circuit properties.
let auto = simulate(&circuit).seed(42).run().unwrap();

// Or choose explicitly.
let stab = simulate(&circuit)
    .backend(BackendKind::Stabilizer)
    .seed(42)
    .run()
    .unwrap();
let mps = simulate(&circuit)
    .backend(BackendKind::Mps { max_bond_dim: 64 })
    .seed(42)
    .run()
    .unwrap();
let sparse = simulate(&circuit)
    .backend(BackendKind::Sparse)
    .seed(42)
    .run()
    .unwrap();
```

### Programmatic circuit construction

```rust
use prism_q::CircuitBuilder;

let result = CircuitBuilder::new(3)
    .h(0)
    .cx(0, 1)
    .cx(1, 2)
    .run(42)
    .unwrap();
```

`CircuitBuilder` chains gate, control, and execution methods. For lower-level access,
use `Circuit` directly:

```rust
use prism_q::{simulate, Circuit, gates::Gate};

let mut c = Circuit::new(3, 0);
c.add_gate(Gate::H, &[0]);
c.add_gate(Gate::Cx, &[0, 1]);
c.add_gate(Gate::Cx, &[1, 2]);
let result = simulate(&c).seed(42).run().unwrap();
```

## Backends

| Backend | Best for | Scaling | Key property |
| --- | --- | --- | --- |
| **Statevector** | General circuits | O(2ⁿ) | Full SIMD, tiled L2/L3 kernels, optional CUDA path |
| **Stabilizer** | Clifford only | O(n²) | SIMD optimized, scales to thousands of qubits |
| **Sparse** | Few live amplitudes | O(k) | HashMap with parallel measurement |
| **MPS** | Low entanglement or 1D | O(nχ²) | Hybrid faer / Jacobi SVD |
| **Product State** | No entanglement | O(n) | Per qubit, instant |
| **Tensor Network** | Low treewidth | Depends on contraction order | Greedy min size heuristic |
| **Factored** | Partial entanglement | Dynamic | Tracks independent sub-states |

`BackendKind::Auto` selects at dispatch time. Non-entangling circuits go to Product
State, all-Clifford circuits go to Stabilizer, large circuits fall through to MPS with
bond dimension 256 once they exceed the statevector memory budget, and everything else
runs on Statevector. The memory budget is dynamic, derived from available RAM at
dispatch time, and can be overridden with `PRISM_MAX_SV_QUBITS`.

## Gates and OpenQASM support

Covers the standard OpenQASM `stdgates.inc` set, common controlled and multi-controlled
variants, Qiskit exporter gates, IonQ and Google/Cirq native gate names, decomposed
multi-instruction gates, and IBM legacy u1/u2/u3 syntax. Modifiers `inv @`, `ctrl @`,
`pow(k) @` chain arbitrarily for direct gates, and user-defined `gate` declarations are
supported.

The authoritative list of supported gate keywords, language features, and modifiers
lives in the parser at [`src/circuit/openqasm.rs`](src/circuit/openqasm.rs). See
`resolve_gate()` and `resolve_decomposed_gate()`. Smoke tests in
[`tests/smoke_openqasm.rs`](tests/smoke_openqasm.rs) exercise each feature end to end.

## Build and test

```bash
cargo build --release
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings -D clippy::undocumented_unsafe_blocks
cargo fmt --check
cargo doc --no-deps --all-features
```

For Rayon parallelism on larger circuits:

```bash
cargo build --release --features parallel
```

Thread count defaults to logical cores. Set `RAYON_NUM_THREADS` to override.

## GPU backend (optional)

The `gpu` feature enables CUDA paths for the statevector and (experimentally) the
stabilizer backend. Requires CUDA Toolkit 12.x or newer and a CUDA capable device.
PTX is compiled at runtime via NVRTC against the device's compute capability.

```bash
cargo build --release --features "parallel gpu"
```

Opt in through the simulation builder. The circuit still goes through fusion and
independent subsystem decomposition, and a size aware crossover keeps small sub
circuits on the CPU:

```rust
use prism_q::{gpu::GpuContext, simulate};

let ctx = GpuContext::new(0)?;
let result = simulate(&circuit).gpu(ctx).seed(42).run()?;
```

`BackendKind::StabilizerGpu` runs Clifford circuits on the device, and
`CompiledSampler::with_gpu(ctx)` accelerates large shot counts for compiled BTS
sampling. Crossover thresholds are conservative by default and can be tuned through
`PRISM_GPU_MIN_QUBITS`, `PRISM_STABILIZER_GPU_MIN_QUBITS`, and
`PRISM_GPU_BTS_MIN_SHOTS`.

`BackendKind::Auto` does not yet route to GPU. See
[`docs/guides/gpu.md`](docs/guides/gpu.md) for kernel design, crossover analysis,
and the full set of tuning knobs.

## Coverage

Requires `rustup component add llvm-tools-preview` and `cargo install cargo-llvm-cov`.

```bash
cargo llvm-cov --all-features                # terminal summary
cargo llvm-cov --all-features --html --open  # browseable HTML report
```

CI generates coverage on every push and PR, and updates the badge automatically.

## Benchmarks

```bash
cargo bench --bench circuits     --features parallel         # circuit macrobenchmarks
cargo bench --bench bench_driver --features parallel         # gate microbenchmarks
cargo bench --bench bench_gpu    --features "parallel gpu"   # GPU dispatch benchmarks
```

Always use `--features parallel`; baselines were taken with Rayon enabled. Do not run
two `cargo bench` invocations concurrently on the same machine — Rayon thread pools
contend for cores and skew results.

Baseline capture, regression checks, and the markdown table workflow used in PRs live
in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Profiling

Needs `cargo install flamegraph`:

```bash
./scripts/flamegraph.sh "qft_textbook/16"              # unix
.\scripts\flamegraph.ps1 "qft_textbook/16"             # windows
```

SVGs land in `bench_results/` (gitignored).

## Roadmap

- Density matrix backend: mixed state simulation for noise and decoherence modeling.
- GPU auto dispatch: thread a GPU context into `BackendKind::Auto` so large circuits
  route to GPU without an explicit `BackendKind::StatevectorGpu`. Crossover and
  decomposition already work through the explicit variant.
- Expanded classical control: mid circuit branching beyond the current `if` form, and
  parameterized circuit reuse for variational workloads.
- Distributed statevector: multi node sharding for circuits beyond single host memory.

## Architecture

See the [architecture reference](docs/architecture/overview.md) for the full picture:
layered design, backend trait contract, SIMD strategy, fusion pipeline, and compiled
samplers. The published docs site is at <https://abecoull.github.io/prism-q/>.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the build, test, and benchmark workflow.
The pull request template at
[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) captures the
required checklist.

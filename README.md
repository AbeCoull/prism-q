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

A quantum circuit simulator written in Rust attempting to run circuits quickly.

Automatic dispatch across multiple simulation strategies picks the engine that best
fits each circuit's structure. CPU kernels use SIMD with an optional CUDA path for the
statevector backend. Input is OpenQASM 3.0, with backward-compatible 2.0 syntax.

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
use prism_q::{circuit::openqasm, run_shots};

let circuit = openqasm::parse(qasm).unwrap();
let result = run_shots(&circuit, 1024, 42).unwrap();
println!("{result}");
// 00: 512
// 11: 512
```

### Backend dispatch

```rust
use prism_q::{circuit::openqasm, run_with, BackendKind};

let circuit = openqasm::parse(qasm).unwrap();

// Auto picks the optimal backend based on circuit properties.
let auto   = run_with(BackendKind::Auto, &circuit, 42).unwrap();

// Or choose explicitly.
let stab   = run_with(BackendKind::Stabilizer, &circuit, 42).unwrap();
let mps    = run_with(BackendKind::Mps { max_bond_dim: 64 }, &circuit, 42).unwrap();
let sparse = run_with(BackendKind::Sparse, &circuit, 42).unwrap();
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
use prism_q::{Circuit, sim, gates::Gate};

let mut c = Circuit::new(3, 0);
c.add_gate(Gate::H, &[0]);
c.add_gate(Gate::Cx, &[0, 1]);
c.add_gate(Gate::Cx, &[1, 2]);
let result = sim::run(&c, 42).unwrap();
```

## Backends

| Backend | Best for | Scaling | Key property |
| --- | --- | --- | --- |
| **Statevector** | General circuits | O(2^n) | Full SIMD, tiled L2/L3 kernels, optional CUDA path |
| **Stabilizer** | Clifford-only | O(n^2) | SIMD-optimized, scales to thousands of qubits |
| **Sparse** | Few live amplitudes | O(k) | HashMap with parallel measurement |
| **MPS** | Low-entanglement or 1D | O(n chi^2) | Hybrid faer / Jacobi SVD |
| **Product State** | No entanglement | O(n) | Per-qubit, instant |
| **Tensor Network** | Low treewidth | Contraction-dependent | Greedy min-size heuristic |
| **Factored** | Partial entanglement | Dynamic | Tracks independent sub-states |

`BackendKind::Auto` selects at dispatch time. Non-entangling circuits go to Product
State, all-Clifford circuits go to Stabilizer, large circuits fall through to MPS with
bond dimension 256 once they exceed the statevector memory budget, and everything else
runs on Statevector. The memory budget is dynamic, derived from available RAM at
dispatch time, and can be overridden with `PRISM_MAX_SV_QUBITS`.

## Gates and OpenQASM support

Covers the standard OpenQASM `stdgates.inc` set, common controlled and multi-controlled
variants, decomposed multi-instruction gates, and IBM legacy u1/u2/u3 syntax. Modifiers
`inv @`, `ctrl @`, `pow(k) @` chain arbitrarily, and user-defined `gate` declarations
are supported.

The authoritative list of supported gate keywords, language features, and modifiers
lives in the parser at [`src/circuit/openqasm.rs`](src/circuit/openqasm.rs). See
`resolve_gate()` and `resolve_decomposed_gate()`. Smoke tests in
[`tests/smoke_openqasm.rs`](tests/smoke_openqasm.rs) exercise each feature end to end.

## Build and test

```bash
cargo build --release
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
cargo doc --no-deps --all-features
```

For Rayon parallelism on larger circuits:

```bash
cargo build --release --features parallel
```

Thread count defaults to logical cores. Set `RAYON_NUM_THREADS` to override.

## GPU backend (optional)

The `gpu` feature enables a CUDA statevector path.

```bash
cargo build --release --features "parallel gpu"
cargo test  --features "parallel gpu" --test golden_gpu
```

Requires the CUDA toolkit (12.x or newer) and a CUDA-capable device. PTX is compiled at
runtime via NVRTC against the device's compute capability. Every `Gate` variant is
covered by a dedicated kernel, including batched kernels for `BatchPhase`, `BatchRzz`,
`DiagonalBatch`, and both diagonal and non-diagonal `MultiFused`. Golden tests in
[`tests/golden_gpu.rs`](tests/golden_gpu.rs) verify amplitude equivalence against the
CPU statevector within 1e-10.

`BackendKind::Auto` does not yet route to GPU. Opt in explicitly. The recommended
entry point is `BackendKind::StatevectorGpu`, which inherits the fusion pipeline plus
independent-subsystem decomposition and applies a size-aware crossover (default: GPU
only for ≥14 qubit sub-circuits, overridable via `PRISM_GPU_MIN_QUBITS`):

```rust
use prism_q::{gpu::GpuContext, run_with, BackendKind};

let ctx = GpuContext::new(0)?;
let result = run_with(BackendKind::StatevectorGpu { context: ctx }, &circuit, 42)?;
```

For kernel-level experiments where every gate must hit the device, use the low-level
`StatevectorBackend::new(seed).with_gpu(ctx)` builder instead — this bypasses the
dispatch crossover by design.

See [`docs/architecture.md`](docs/architecture.md) for the kernel design and crossover
analysis.

## Coverage

Requires `rustup component add llvm-tools-preview` and `cargo install cargo-llvm-cov`.

```bash
cargo llvm-cov --all-features                # terminal summary
cargo llvm-cov --all-features --html --open  # browseable HTML report
```

CI generates coverage on every push and PR, and updates the badge automatically.

## Benchmarks

```bash
cargo bench --bench circuits     --features parallel   # circuit macrobenchmarks
cargo bench --bench bench_driver --features parallel   # gate microbenchmarks
cargo bench --features "parallel,bench-fast"           # quick smoke test
```

Always use `--features parallel`. Baselines were taken with Rayon enabled. Never run
two `cargo bench` invocations at the same time on the same machine. Rayon thread pools
fight for cores and produce large swings in results.

### Regression checks

```bash
# Save a baseline.
cargo bench --features parallel
./scripts/bench_check.sh save --name "before"         # unix
.\scripts\bench_check.ps1 save -Name "before"         # windows

# Make changes, bench again.
cargo bench --features parallel

# Compare (exits 1 on regression).
./scripts/bench_check.sh compare --baseline "before"
.\scripts\bench_check.ps1 compare -Baseline "before"

# Markdown table for PRs.
./scripts/bench_check.sh table --baseline "before"
.\scripts\bench_check.ps1 table -Baseline "before"
```

## Profiling

Needs `cargo install flamegraph`:

```bash
./scripts/flamegraph.sh "qft_textbook/16"              # unix
.\scripts\flamegraph.ps1 "qft_textbook/16"             # windows
```

SVGs land in `bench_results/` (gitignored).

## Roadmap

- Expanded OpenQASM 3.0: `reset` instruction, `for` loop unrolling, `def` subroutines.
- Expectation values: `<psi|O|psi>` for Pauli strings (VQE and QAOA).
- Density matrix backend: mixed-state simulation for noise and decoherence modeling.
- GPU auto-dispatch: thread a GPU context into `BackendKind::Auto` so large circuits
  route to GPU without an explicit `BackendKind::StatevectorGpu`. Crossover and
  decomposition already work through the explicit variant.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full picture: layered design,
backend trait contract, SIMD strategy, fusion pipeline, compiled samplers, and how to
add a new backend.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the build, test, and benchmark workflow.
The pull request template at
[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) captures the
required checklist.

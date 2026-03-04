# PRISM-Q

```text
 ██████╗ ██████╗ ██╗███████╗███╗   ███╗       ██████╗
 ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║      ██╔═══██╗
 ██████╔╝██████╔╝██║███████╗██╔████╔██║█████╗██║   ██║
 ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║╚════╝██║▄▄ ██║
 ██║     ██║  ██║██║███████║██║ ╚═╝ ██║      ╚██████╔╝
 ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝       ╚══▀▀═╝
```

![Rust](https://img.shields.io/badge/rust-1.75%2B-orange?logo=rust)
![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)
![Tests](https://img.shields.io/badge/tests-687_passing-brightgreen)
![OpenQASM](https://img.shields.io/badge/OpenQASM-3.0-purple)

A performance oriented quantum circuit simulator written in Rust.

## Highlights

- **simulation backends** with automatic selection based on circuit structure
- **OpenQASM 3.0** parser with backward-compatible 2.0 support
- **Gate modifiers** — `inv @`, `ctrl @`, `pow(k) @` with arbitrary chaining
- **SIMD-optimized kernels** — AVX2, FMA, and BMI2 with runtime feature detection
- **Seven-stage fusion pipeline** — gate cancellation, 1q/2q fusion, commutation-aware reordering, multi-gate batching, controlled-phase batching with PEXT lookup tables
- **Shot-based sampling** with deterministic or random seeding
- **Rayon parallelism** at 14+ qubits with auto-tuned chunking
- **Subsystem decomposition** — independent qubit groups simulated in parallel and merged via Kronecker product

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

// Auto picks the optimal backend based on circuit properties
let auto   = run_with(BackendKind::Auto, &circuit, 42).unwrap();

// Or choose explicitly
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

`CircuitBuilder` supports fluent method chaining with 44 gate/control/execution methods. For lower-level access, use `Circuit` directly:

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
| ------- | -------- | ------- | ------------ |
| **Statevector** | General circuits | O(2^n) | Full SIMD, tiled L2/L3 kernels |
| **Stabilizer** | Clifford-only | O(n^2) | SIMD-optimized, handles 5000+ qubits |
| **Sparse** | Few live amplitudes | O(k) | HashMap with parallel measurement |
| **MPS** | Low-entanglement / 1D | O(n chi^2) | Hybrid faer/Jacobi SVD |
| **Product State** | No entanglement | O(n) | Per-qubit, instant |
| **Tensor Network** | Low treewidth | Contraction-dependent | Greedy min-size heuristic |
| **Factored** | Partial entanglement | Dynamic | Tracks independent sub-states |

`BackendKind::Auto` selects at dispatch time: non-entangling circuits go to Product State, all-Clifford to Stabilizer, large circuits (>28q) to MPS, and everything else to Statevector.

## Gates & OpenQASM support

15 single-qubit gates, 4 two-qubit gates, 6 controlled variants, 2 multi-controlled gates, 8 decomposed multi-instruction gates, and 3 IBM legacy gates — plus `inv @`, `ctrl @`, `pow(k) @` modifiers with arbitrary chaining and user-defined `gate` definitions.

The complete list of supported gate keywords, language features, and modifiers is defined in the parser at [`src/circuit/openqasm.rs`](src/circuit/openqasm.rs) (see `resolve_gate()` and `resolve_decomposed_gate()`). Smoke tests in [`tests/smoke_openqasm.rs`](tests/smoke_openqasm.rs) exercise each feature end-to-end.

## Build & test

```bash
cargo build --release
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

## Parallelism

Rayon threading kicks in at 14+ qubits:

```bash
cargo build --release --features parallel
```

Defaults to logical core count. Set `RAYON_NUM_THREADS` to override.

## Benchmarks

```bash
# full suite
cargo bench --bench circuits --features parallel

# gate-level microbenchmarks
cargo bench --bench bench_driver --features parallel

# quick smoke test (fewer iterations)
cargo bench --features "parallel,bench-fast"
```

### Regression checks

```bash
# save baseline
cargo bench --features parallel
./scripts/bench_check.sh save --name "before"         # unix
.\scripts\bench_check.ps1 save -Name "before"         # windows

# make changes, bench again
cargo bench --features parallel

# compare (exits 1 on regression)
./scripts/bench_check.sh compare --baseline "before"
.\scripts\bench_check.ps1 compare -Baseline "before"

# markdown table for PRs
./scripts/bench_check.sh table --baseline "before"
.\scripts\bench_check.ps1 table -Baseline "before"
```

## Profiling

Needs `cargo install flamegraph`:

```bash
./scripts/flamegraph.sh "qft_textbook/16"             # unix
.\scripts\flamegraph.ps1 "qft_textbook/16"             # windows
```

SVGs go to `bench_results/` (gitignored).

## Roadmap

- **Expanded OpenQASM 3.0** — `reset` instruction, `for` loop unrolling, `def` subroutines
- **Expectation values** — `<psi|O|psi>` for Pauli strings (VQE/QAOA support)
- **Density matrix backend** — mixed-state simulation for noise and decoherence modeling
- **GPU acceleration** — CUDA/Vulkan compute backend for large qubit counts

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full picture — layered design, backend trait contract, SIMD strategy, and how to add a new backend.

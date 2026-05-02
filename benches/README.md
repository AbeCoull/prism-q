# PRISM-Q Benchmarks

## Framework

Criterion.rs with HTML reports. Two benchmark binaries:

- **bench_driver**: Microbenchmarks for individual gate kernels, measurement, and end-to-end QASM.
- **circuits**: Macrobenchmarks for circuit family sweeps across qubit counts and depths.

## Benchmark categories

### Microbenchmarks (bench_driver)

| Group | What it measures |
|-------|-----------------|
| `single_qubit_gates` | H, Rx, T gate kernels across qubit counts (4–20) |
| `two_qubit_gates` | CX, CZ, SWAP kernels across qubit counts (4–20) |
| `measurement` | Measure-after-superposition across qubit counts |
| `e2e_qasm` | Full parse + simulate from OpenQASM string |

### Macrobenchmarks (circuits)

| Group | What it measures |
|-------|-----------------|
| `qubit_sweep/random_d10` | Seeded random circuits, depth 10, 4–20 qubits |
| `qubit_sweep/qft_like` | QFT-structured circuits, 4–16 qubits |
| `qubit_sweep/hea_l5` | Hardware-efficient ansatz, 5 layers, 4–20 qubits |
| `qubit_sweep/clifford_d10` | Clifford-heavy circuits, depth 10, 4–20 qubits |
| `depth_sweep/12q_random` | 12-qubit random circuits, depth 5–100 |
| `entanglement_structure` | Sparse vs dense entanglement, 16 qubits |

## Circuit families

All seeded with `0xDEAD_BEEF` for reproducibility.

- **Random**: Mix of single-qubit gates + 50% CX on even-odd pairs per layer.
- **QFT-like**: Hadamard + controlled rotations (approximated with Rz + CX decomposition).
- **Hardware-efficient ansatz**: Ry/Rz layers + linear CX chain.
- **Clifford-heavy**: H, S, X, Y, Z + CX only.
- **Sparse entanglement**: H on all qubits + single CX(0, n-1) per layer.
- **Dense entanglement**: H on all qubits + linear CX chain per layer.

## Running benchmarks

```bash
# Full suite
cargo bench

# Specific benchmark
cargo bench --bench bench_driver -- "single_qubit_gates/h_gate/16"

# Quick smoke (fast settings)
cargo bench --bench bench_driver -- --warm-up-time 1 --measurement-time 3 --sample-size 10

# Save baseline
cargo bench -- --save-baseline my_baseline

# Compare against baseline
cargo bench -- --baseline my_baseline
```

## Baseline workflow

```bash
# Save (Unix)
./scripts/bench_baseline.sh

# Save (Windows)
.\scripts\bench_baseline.ps1

# Compare (Unix)
./scripts/bench_compare.sh

# Compare (Windows)
.\scripts\bench_compare.ps1

# Custom threshold
REGRESSION_THRESHOLD=10 ./scripts/bench_compare.sh
```

## CI regression gate

Pull requests run a focused benchmark gate after lint and tests pass. The job
checks out the base commit and PR head on the same runner, runs
`scripts/bench_ci.sh` in both worktrees, saves the base results with
`scripts/bench_check.sh`, then fails if any matching benchmark regresses beyond
the configured threshold.

The CI subset uses `CI_BENCH_FEATURES=parallel,bench-fast` and covers larger
CPU-only parameter points that are already present on the base branch. The
filters are intentionally narrow so GitHub hosted runner noise from tiny
parameter sweeps does not dominate the gate. It is a regression gate, not a
replacement for the full local benchmark suite required for performance
sensitive changes.

Representative CI workloads:

| Filter | Coverage |
|--------|----------|
| `statevector/scalability_d5/22` | Dense statevector scaling at a larger qubit count |
| `statevector/qft_textbook/22` | Structured controlled phase and swap workload |
| `statevector/qpe_t_gate/22q` | Phase estimation with non-Clifford gates |
| `stabilizer/scaling/1000` | Large Clifford stabilizer backend path |
| `auto/qft_textbook/22` | Auto dispatch on a structured dense circuit |
| `compiled_sampler/noiseless/noiseless_1000q_10k` | Compiled shot sampling path |

The script builds the `circuits` benchmark binary once with
`cargo bench --no-run`, then runs the compiled executable directly with
Criterion filters. This keeps Cargo setup out of the timed regression subset.

Local reproduction:

```bash
CI_BENCH_FEATURES=parallel,bench-fast ./scripts/bench_ci.sh
./scripts/bench_check.sh save --name ci-base

# After applying changes:
CI_BENCH_FEATURES=parallel,bench-fast ./scripts/bench_ci.sh
./scripts/bench_check.sh table --baseline ci-base
./scripts/bench_check.sh compare --baseline ci-base
```

## Regression detection

Default threshold: **5%** per benchmark (configurable via `REGRESSION_THRESHOLD`).

`bench_check.*` reads Criterion JSON from `target/criterion/`, compares matching
benchmark means, and exits with code 1 when any benchmark exceeds the threshold.
The older `bench_compare.*` wrappers still parse Criterion console output for
quick baseline checks.

## Reproducibility checklist

- [ ] Fixed RNG seed (`0xDEAD_BEEF` for circuit generation, `42` for simulation)
- [ ] Criterion defaults: 5s warm-up, 5s measurement, 100 samples
- [ ] Document CPU model, OS, Rust version, RUSTFLAGS
- [ ] Disable CPU frequency scaling if possible (`performance` governor on Linux)
- [ ] Close background applications
- [ ] Consider CPU pinning (`taskset` on Linux) for reduced variance

## Output

Criterion stores results in `target/criterion/`. Each benchmark gets:
- `estimates.json`: statistical estimates (mean, median, std dev)
- `benchmark.json`: configuration
- HTML reports in `target/criterion/report/`

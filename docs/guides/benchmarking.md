# Benchmark Methodology

How PRISM-Q's numbers are produced, what they include, and how to reproduce them or
run a like-for-like comparison against another quantum circuit simulator. The
[Benchmarks](../benchmarks.md) page holds the published timings; this page holds the
rules behind them. The mechanisms being measured (fusion, tiling, SIMD dispatch,
threading) are in [Performance and SIMD](./performance.md).

## Two kinds of measurement

| Purpose | Tool | Output |
|---------|------|--------|
| Published wall-clock timings per circuit family | `examples/bench_suite.rs` | Rewrites `docs/benchmarks.md` |
| Per-kernel and per-family rows held to a regression gate | Criterion targets under `benches/` and `scripts/bench_ab.sh` | Criterion estimates and an A/B report |

The suite answers "how long does this circuit take end to end on this host". The
Criterion rows answer "did this change move this code path", which needs the
adjacent-binary A/B described below rather than two runs minutes apart.

## The published suite

`cargo run --release --features parallel --example bench_suite` builds four circuit
families from the `prism_q::circuits` generators the library ships, times each size,
and rewrites the Benchmarks page with a Setup section naming the date, CPU, thread
count, compiler and crate version of that run.

| Family | Generator | Circuit | Sizes |
|--------|-----------|---------|-------|
| GHZ | `ghz_circuit(n)` | H on qubit 0, then a CX chain | 24, 28, 256, 1024, 4096 |
| QFT | `qft_circuit(n)` | Textbook quantum Fourier transform; the generator emits one `QftBlock`, which the CPU statevector runs as its FFT path and other backends expand | 16, 20, 24, 26, 28 |
| HEA | `hardware_efficient_ansatz(n, 5, seed)` | 5 layers of random Ry and Rz on every qubit followed by a linear CX chain | 16, 20, 24, 26, 28 |
| QV | `quantum_volume_circuit(n, n, seed)` | `n` layers, each a random pairing of qubits with a random SU(4) on each pair, decomposed into CX and single-qubit rotations | 16, 20, 24 |

Rules the generator fixes, so a rerun on another host differs only in the host:

- Circuit seed `0xDEAD_BEEF` for the random families; simulation seed `42`.
- Backend `Auto`. GHZ is Clifford and lands on the stabilizer backend, which is why it
  runs to 4096 qubits; the dense families land on the statevector backend up to the
  memory cap. A size that fails to allocate is skipped, not reported.
- The timed region is the full `simulate(&circuit).seed(42).run()` call: the fusion
  pass, gate application, and the probability extraction at the end. Circuit
  construction happens once per size, outside the timer.
- Warmup and repeat counts follow the size, because a 28-qubit run costs tens of
  seconds: up to 18 qubits, 2 warmup runs then 7 timed; 19 to 22, 1 then 5; 23 to 25,
  1 then 3; 26 and above, no warmup and 1 timed run. The reported value is the median
  of the timed runs.
- Build: the `release` profile, which sets `opt-level = 3`, fat LTO,
  `codegen-units = 1` and `panic = "abort"`, with the `parallel` feature.
- Threads: every logical core, unless `RAYON_NUM_THREADS` caps the Rayon pool. The
  page records both the host's logical core count and the pool width used.
- Precision: amplitudes are `Complex<f64>` on every backend. There is no single
  precision mode.

A number on that page describes the crate version and host named in its Setup
section. Dispatch rules and kernels change between releases, so compare numbers only
within one run.

## Criterion targets

| Target | Rows | Feature flags |
|--------|------|---------------|
| `bench_driver` | Single-qubit (H, Rx, T) and two-qubit (CX, CZ, SWAP) kernels across qubit counts, kernel variants by control and target placement, measurement and reset, end-to-end OpenQASM parse and simulate | `parallel` |
| `circuits` | Circuit families (random, QFT, HEA, QAOA, Trotter, QV, W state, Clifford, depth and width sweeps) per backend, plus compiled samplers, prepared parameter sweeps, marginals and reduced density matrices | `parallel`; some groups need `bench-internal` |
| `bench_shots_perf` | Shot and count sampling, packed QEC runner | `parallel` |
| `qec_decoder`, `qec_t_strategies` | Union-find decoding of sampled detector batches; Clifford+T sampling strategies | `parallel` |
| `svd_bench` | The MPS decomposition kernels | `parallel` |
| `bench_gpu` | The CPU sweeps on the device path, Pauli expectations, statevector readback | `parallel gpu`; skipped without a device |
| `bench_distributed` | Rank exchanges and sampling over the thread loopback transport | `parallel distributed bench-internal` |

`cargo bench --bench <target> -- --list` prints a target's rows. The group-by-group
description, including which rows the development host cannot resolve to the gate, is
in `benches/README.md`.

Which category each need maps to:

| Question | Rows |
|----------|------|
| Single-qubit gate cost | `bench_driver` `single_qubit_gates` |
| Controlled and two-qubit gate cost | `bench_driver` `two_qubit_gates`, `two_qubit_gate_kernels`, `controlled_gates` |
| Dense multi-qubit unitaries | `circuits` `density_matrix/unitary_layers`, QV rows (`statevector/qv`) |
| Random circuits | `statevector/random_d10`, `sparse/random_d10`, `mps/random_d10`, `tn/random_d10` |
| GHZ | `bench_suite` GHZ family; `stabilizer/scaling` |
| QFT | `statevector/qft_textbook`, `statevector/qft_like`, `gpu/qft_textbook` |
| Quantum Volume | `statevector/qv` |
| Shallow and wide | `statevector/scalability_d5`, `auto/scalability_d5`, `tn/scalar_hea_l2` (20 to 50 qubits) |
| Deep | `statevector/depth_sweep_12q` (depth 5 to 100) |
| Fused versus unfused | `density_matrix/fused_layers` and `rzz_layers_fused` against `unitary_layers`; the `PRISM_NO_REORDER` and `PRISM_NO_QFT_BLOCK` flags in [Performance and SIMD](./performance.md#tuning-environment-variables) switch individual passes off for any row |
| Thread scaling | Any row rerun under `RAYON_NUM_THREADS=1,2,4,...`; no target sweeps threads on its own |
| Memory scaling | The qubit ladders inside each family; the memory caps are in [Backends](../architecture/backends.md#memory-budget) |
| CPU versus GPU | `bench_gpu` groups beside their CPU twins, or `examples/bench_gpu_vs_cpu.rs` for a per-gate comparison with fusion disabled on both sides |
| Distributed execution | `bench_distributed`, and `scripts/test-mpi.sh --timed` for real ranks across hosts |

## Reading a number

Criterion reports the mean of `PRISM_BENCH_SAMPLES` samples (default 30) per row. The
`bench-fast` feature drops to 10 samples for triage and is what the CI gate runs; a
`bench-fast` table is triage, not a claim. `PRISM_BENCH_PLOTS` turns the HTML report
on; it is off by default because rendering cost more than measuring.

Sample count sets the precision of one run's mean. It does not remove drift between
runs: on the reference host, back-to-back runs of identical code moved up to about 10%
either way, from rebuilds landing between runs, code layout, and host load. Two
`cargo bench` invocations are therefore not a comparison. `scripts/bench_ab.sh` builds
the reference and working-tree binaries side by side, checks the tree did not move
between the builds, runs the two binaries back to back, and reports each row against
a same-code control pair. A delta inside its own control spread is noise, and the
report says so rather than rounding it to a win.

```bash
./scripts/bench_ab.sh --filter '^statevector/qft_textbook/' --ref main
```

`REGRESSION_THRESHOLD` (default 5, in percent) is the gate every PR that touches a hot
path must clear; CI runs the same A/B against the PR base at the `bench-fast` tier.
Run one benchmark process at a time, with no other load on the host: competing Rayon
pools and a busy GPU have both produced swings larger than the gate.

## Comparative measurements

Comparative performance measurements against commonly used quantum simulators are
intended to make performance characteristics reproducible and transparent across
representative workloads, not to rank projects. Each simulator makes different
trade-offs in precision, optimization, output and scope, and a measurement is only
meaningful where those are held equal and stated. The
[Comparative Measurements](../comparison.md) page carries the numbers, generated by the
harness in `comparison/` under the controls below; this page carries the rules.

- Same program. Hand every simulator one gate list and replay it through each native
  API, which is what the harness in `comparison/` does; parsing OpenQASM into the other
  simulator is the alternative, and then its importer's transpilation sits inside its
  measurement. The QFT generator's `QftBlock` has no gate-level spelling, so the shared
  list is the textbook sequence and PRISM-Q runs it as ordinary gates too; its block FFT
  path is measured on the Benchmarks page, not in the comparison. Record the gate count
  after each simulator's own optimization, since fusion changes what runs.
- Same output. The suite times `run()`, which ends with a `2^n` probability vector.
  Time the other simulator to the same observable, or time both to the final state
  only; mixing the two moves a 26-qubit row by the cost of a full pass over the state.
- Same precision. PRISM-Q is `f64` complex throughout. A simulator run in single
  precision moves half the memory traffic, which is a different measurement rather
  than a worse or better one.
- Same thread budget. Pin both with their own variable (`RAYON_NUM_THREADS` here,
  `OMP_NUM_THREADS` or an API call there) and report the count. Report the physical
  and logical core counts of the host.
- Same build. Release builds on both sides, with the compiler and flags recorded.
  For PRISM-Q that is the `release` profile above and `rustc --version`.
- Same discipline. Warm up, repeat, report the median and the spread, run on an idle
  host, and run one process at a time. Report peak resident memory from the OS
  (`/usr/bin/time -v`, or the Windows peak working set) when memory is part of the
  claim.
- Versions. Name the release of each simulator and the commit when a release is not
  enough.

A results table that names the version, hardware, compiler, qubit count, circuit,
shot count, thread count, timing and memory for every row can be reproduced by its
readers. Where a workload favors one simulator's design, say which design choice
explains it; a difference without its mechanism is not yet a finding.

## Profiling

```bash
./scripts/flamegraph.sh "qft_textbook/16"     # unix, needs `cargo install flamegraph`
.\scripts\flamegraph.ps1 "qft_textbook/16"    # windows
```

The script header lists the platform profiler it needs and where the SVG lands.
The `bench` profile keeps function names and line tables, so the flamegraph resolves
kernel names while the build stays close to release.

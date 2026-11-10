# Benchmark Methodology

[Performance](../performance/index.md) holds the published timings, and
[Performance and SIMD](./performance.md) the mechanisms being measured.

## Two kinds of measurement

| Purpose | Tool | Output |
|---------|------|--------|
| Published wall-clock timings per circuit family | `tools/bench_suite.rs` | Rewrites `docs/performance/suite.json` |
| Kernel and circuit family rows held to a regression gate | Criterion targets under `benches/` and `scripts/bench_ab.sh` | Criterion estimates and an A/B report |
| Instruction counts on a shared CI runner | The `iai` target under callgrind | Counts per row, compared against the base commit |

The suite answers how long a circuit takes end to end on one host. The Criterion rows
answer whether a change moved a code path, which takes the A/B described under
[Reading a number](#reading-a-number).

## The published suite

`cargo run --release --features parallel --example bench_suite` builds four circuit
families from the `prism_q::circuits` generators, times each size, and rewrites
`docs/performance/suite.json` with the run's date, CPU, thread count, compiler and crate
version. `comparison/matrix.py report` folds it into the Performance page.

| Family | Generator | Circuit | Sizes |
|--------|-----------|---------|-------|
| GHZ | `ghz_circuit(n)` | H on qubit 0, then a CX chain | 24, 28, 256, 1024, 4096 |
| QFT | `qft_circuit(n)` | Textbook quantum Fourier transform; the generator emits one `QftBlock`, which the CPU and GPU statevectors run as an FFT and other backends expand | 16, 20, 24, 26, 28 |
| HEA | `hardware_efficient_ansatz(n, 5, seed)` | 5 layers of random Ry and Rz on every qubit followed by a linear CX chain | 16, 20, 24, 26, 28 |
| QV | `quantum_volume_circuit(n, n, seed)` | `n` layers, each a random pairing of qubits with a random SU(4) on each pair, decomposed into CX and single-qubit rotations | 16, 20, 24 |

Fixed by the generator, so a rerun on another host differs only in the host:

- Circuit seed `0xDEAD_BEEF` for the random families; simulation seed `42`.
- Backend `Auto`. GHZ is Clifford and lands on the stabilizer backend, which is why it
  runs to 4096 qubits; the dense families land on the statevector backend up to the
  memory cap. A size that fails to allocate is skipped, not reported.
- The timed region is the full `simulate(&circuit).seed(42).run()` call: fusion, gate
  application and the final probability extraction. Circuit construction happens once
  per size, outside the timer.
- Warmup and repeats scale with size, since a 28-qubit run takes tens of seconds: up to
  18 qubits, 2 warmup runs then 7 timed; 19 to 22, 1 then 5; 23 to 25, 1 then 3; 26 and
  above, no warmup and 1 timed run. The reported value is the median of the timed runs.
- Build: the `release` profile (`opt-level = 3`, fat LTO, `codegen-units = 1`,
  `panic = "abort"`) with the `parallel` feature.
- Threads: every logical core, unless `RAYON_NUM_THREADS` caps the Rayon pool. The page
  records the host's logical core count and the pool width used.
- Precision: `Complex<f64>` amplitudes on every backend, with no single precision mode.

The page's Setup section names the crate version and host. Dispatch rules and kernels
change between releases, so compare numbers only within one run.

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
| `iai` | Not Criterion: eight single-thread kernel rows counted in instructions under callgrind, Linux only (elsewhere it only builds) | `parallel` |

`cargo bench --bench <target> -- --list` prints a target's rows. `benches/README.md`
describes each group, the rows the development host cannot resolve to the gate, and the
`iai` instruction-count gate.

Rows by question:

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
| CPU versus GPU | `bench_gpu` groups beside their CPU twins, or `tools/bench_gpu_vs_cpu.rs` for a per-gate comparison with fusion disabled on both sides |
| Distributed execution | `bench_distributed`, and `scripts/test-mpi.sh --timed` for real ranks across hosts |

## Reading a number

Criterion reports the mean of `PRISM_BENCH_SAMPLES` samples (default 30) per row. The
`bench-fast` feature drops to 10 samples and is what the CI gate runs; a `bench-fast`
table is triage, not a claim. `PRISM_BENCH_PLOTS` turns the HTML report on; it is off by
default because rendering cost more than measuring.

Sample count sets the precision of one run's mean, not the drift between runs: on the
reference host, back-to-back runs of identical code moved up to about 10% either way,
from rebuilds landing between runs, code layout and host load. Two `cargo bench`
invocations are not a comparison. `scripts/bench_ab.sh` builds the reference
and working-tree binaries side by side, checks the tree did not change between the
builds, runs the two back to back, and reports each row against a same-code control
pair. A delta inside its own control spread is noise, and the report says so.

```bash
./scripts/bench_ab.sh --filter '^statevector/qft_textbook/' --ref main
```

`REGRESSION_THRESHOLD` (default 5, in percent) is the gate every PR touching a hot path
must clear; CI runs the same A/B against the PR base at the `bench-fast` tier. Run one
benchmark process at a time on an otherwise idle host: competing Rayon pools and a busy
GPU have both produced swings larger than the gate.

## Comparative measurements

Simulators trade off precision, optimization, output and scope differently, so a
measurement against another one means something only where those are held equal and
stated. [Performance](../performance/index.md) carries the numbers, generated by the
harness in `comparison/` under these controls:

- Same program. Hand every simulator one gate list and replay it through each native
  API, as the harness does. Parsing OpenQASM into the other simulator instead puts its
  importer's transpilation inside its measurement. The QFT generator's `QftBlock` has
  no gate-level spelling, so the shared list is the textbook sequence. PRISM-Q folds
  that exact sequence back into its block FFT, as it does for any QFT that arrives as
  gates, so the comparison times the path a user's expanded QFT takes;
  `PRISM_NO_QFT_BLOCK` restores gate-by-gate replay. Record the gate count after each
  simulator's own optimization, since fusion changes what runs.
- Same output. The suite times `run()`, which ends with a `2^n` probability vector.
  Time the other simulator to the same observable, or time both to the final state
  only; mixing the two moves a 26-qubit row by the cost of a full pass over the state.
- Same precision. PRISM-Q is `f64` complex throughout. A single precision run moves half
  the memory traffic, which makes it a different measurement.
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

Where a workload favors one simulator's design, name the design choice that explains it.
A difference without its mechanism is not yet a finding.

### The benchmark matrix

`comparison/matrix.py` applies those controls to the published comparison. PRISM-Q runs
with automatic dispatch and, for the dispatch rows, with each backend forced; every peer
runs at its defaults. [Performance](../performance/index.md) lists each simulator with its
version and precision.

| Family | Sizes | Circuit |
| --- | --- | --- |
| `rcs` | 20 to 26 qubits, depth 10, 20 | Random rotations, CX brickwork |
| `qft` | 20 to 26 | Textbook QFT on a random product state |
| `qv` | 16 to 24 | Quantum volume |
| `clifford` | 100 to 10,000 | Random Clifford brickwork |
| `lowent` | 50 to 200, depth 4, 8 | Shallow brickwork |
| `disjoint` | 40 to 200 | Independent 10-qubit components |
| `sampling` | 16 to 24, 1 to 1M shots | rcs depth 10, measured |
| `midcircuit` | 12, 18 | Measure, feed forward, reset |

- Defaults everywhere, nothing tuned per row. Seed `0xDEADBEEF`; one gate list per
  workload, stored in `circuits.jsonl.gz`.
- Timed call: run the circuit, return probabilities or shots; format conversion is
  untimed. One process per measurement, one cold call, then up to five warm calls; the
  median warm call is reported. Simulator order is shuffled per row (seed 42).
- Failures are recorded with their reason and never retried. Re-measured rows move the
  replaced records to `superseded.jsonl`.
- Dense outputs: total variation distance to an fp64 reference, at most 1e-8 (fp32:
  1e-5). Sampled outputs: bit marginals and pair parities against exact values from an
  independent light-cone reference, exact binomial tests at family-wise alpha 1e-3;
  under 100 shots, deterministic bits only.
- Not covered: GPU, 16 threads, ISA variants, noise, simulators without a Windows wheel
  for Python 3.14, and dense sizes past 26 qubits (the `full` profile).

## Profiling

```bash
./scripts/flamegraph.sh "qft_textbook/16"     # unix, needs `cargo install flamegraph`
.\scripts\flamegraph.ps1 "qft_textbook/16"    # windows
```

The script header lists the platform profiler it needs and where the SVG lands. The
`bench` profile keeps function names and line tables, so the flamegraph resolves kernel
names while the build stays close to release.

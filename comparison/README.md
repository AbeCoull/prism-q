# Cross-simulator comparison

Performance measurements against other quantum simulators, built so a reader
without the originating machine can regenerate the numbers and check them. The
aim is reproducible numbers, not a ranking.

`matrix.py` runs the benchmark matrix: eight workload families (dense random
brickwork, QFT, quantum volume, wide Clifford, shallow wide, disconnected,
sampling-heavy, mid-circuit), thread scaling, PRISM-Q's automatic dispatch
against every forced backend, an accuracy study under a bond cap, tensor network
expectation values on wide shallow circuits, deep MPS sampling, a bond-cap sweep on
wider registers, and high-shot stabilizer sampling that returns statistics only. Its page
is [Performance](https://abecoull.github.io/prism-q/performance/index.html).

## Setup

```bash
python -m venv comparison/.venv
comparison/.venv/Scripts/pip install -r comparison/requirements.txt    # bin/pip on unix
# Stim from source with its AVX2 extension, replacing the wheel (needs a C++20 compiler)
comparison/.venv/Scripts/python comparison/build_stim.py

# Rust comparators (Spinoza needs nightly). Build from inside the crate so its
# rust-toolchain.toml and .cargo/config.toml (target-cpu=x86-64-v3) apply.
rustup toolchain install nightly
(cd comparison/peers && cargo build --release)

# QuEST comparator (CMake 3.21+, a C++17 compiler with OpenMP; fetches QuEST v4.3.0)
cmake -S comparison/quest -B comparison/quest/build -G "Visual Studio 17 2022" -A x64   # or -G Ninja
cmake --build comparison/quest/build --config Release
```

`matrix.py run` builds `compare_runner` itself; pass `--no-build` when it is
current. It then runs `matrix.py smoke`: every simulator the selected profile names
must be installed, report its version, and match PRISM-Q on a tiny circuit in each
output mode it serves on the profile, at its top thread count there; a dense probability
peer also matches PRISM-Q on one circuit per gate it declares. `run` refuses to start while a named peer
fails; `--skip-peer NAME` (repeatable, a base name skips every variant) leaves it out
of the run and is recorded in the manifest.

## Benchmark matrix

```bash
comparison/.venv/Scripts/python comparison/matrix.py list --profile reference      # what would run, and what is skipped and why
comparison/.venv/Scripts/python comparison/matrix.py smoke --profile reference     # every named peer installed and agreeing with PRISM-Q
comparison/.venv/Scripts/python comparison/matrix.py run --profile smoke --out comparison/results/my-smoke
comparison/.venv/Scripts/python comparison/matrix.py run --profile reference --out comparison/results/my-run
comparison/.venv/Scripts/python comparison/matrix.py report comparison/results/my-run   --docs docs/performance
```

| Profile | Contents | Wall time on the reference host |
| --- | --- | --- |
| `smoke` | every family and adapter at a small size, a few seconds per row | about 5 minutes |
| `reference` | the published subset: 78 workloads, 363 measurements and 23 not-applicable cells | about three hours |
| `full` | adds 28-qubit dense rows, QuEST at 26 qubits, Spinoza and qip at every size, threads up to 16 | most of a day; 32 GB RAM |

`--groups headline,dispatch`, `--match qft` and `--simulators prismq,qsim`
narrow a profile. `--append` adds to an existing results directory, and
`--replace` with it moves the records being re-measured to `superseded.jsonl`
once each row's new records are written. `--drop-other-simulators` also moves a
re-measured row's records of simulators its profile no longer lists, and
`--retire TEXT` moves records whose workload id contains `TEXT`, for a workload
whose definition changed. A record's identity is its group, workload, simulator
and thread count; where two records share one, the newest run wins in every
table, ratio and page row. A run without `--append` clears the directory's
earlier results first. Run a long profile from a scheduled task or another
detached shell on an otherwise idle machine, never beside a build.

Dense outputs are checked against Aer's fp64 vector. Where QuEST also ran on the
row, Aer serves as the reference only within 1e-8 of QuEST; otherwise the row is
left unvalidated with the distance as the reason.

Expectation-value rows (`expval`) read Z on single qubits and ZZ on adjacent pairs;
each value is checked against the exact cone reference within 1e-8 (1e-5 for fp32),
against a simulator of another family where the cone is too wide, and the wide
bond-cap sweep reports the largest error without a pass or fail. High-shot rows
(`stats`) return only the checked statistics and the shot total, so 10 million shots
on a wide register never materialize a histogram; they take the same binomial tests
as sampled rows. quimb's tensor network finds a fresh contraction path inside every
timed call; kahypar has no wheel on the reference host, so cotengra falls back to
greedy and labels, and the manifest records which it used.

The harness is formatted with `ruff format` and checked with `ruff check` under
`comparison/ruff.toml`. Its tests run with
`comparison/.venv/Scripts/python -m unittest discover -s comparison/tests`.

### Layout

| File | Role |
| --- | --- |
| `harness/workloads.py` | canonical workload generators and the gate list format |
| `harness/profiles.py` | the matrices: which workloads, simulators, thread counts, repetition policy |
| `harness/runners.py` | simulator registry, applicability rules, one process per measurement, peak memory, failure classification |
| `harness/worker.py` | Aer, qsim, Stim and quimb behind the same `exec` protocol as `tools/compare_runner.rs` |
| `harness/reference.py` | independent exact reference for measurement statistics (past-cone extraction, branching NumPy statevector) |
| `harness/validate.py` | TVD for dense outputs, Bonferroni-corrected exact binomial tests and a shot-total check for sampled and statistics-only outputs, per-value tolerance for expectation values |
| `harness/stats.py` | medians, IQR, seeded bootstrap intervals for medians and ratios |
| `harness/aggregate.py`, `pages.py` | CSV and JSON tables, and the Performance page with its chart data, all from the raw files |

### Results directory

| File | Contents |
| --- | --- |
| `manifest.json` | run id, profile, host (CPU, flags, affinity, clock, power plan, GPU, RAM, OS), toolchains, package versions, environment, timed-region and policy statements, SHA-256 of the runner binaries and harness scripts, and every supplementary run, each with its own host, toolchain, version and statement blocks |
| `observations.jsonl` | one record per simulator, workload and thread count: status (`ok`, `not-applicable` for a row the simulator cannot run, `skipped` for one the profile leaves out, `timeout`, `oom`, `crash`, `error`) and reason, phases, warm samples, statistics, peak memory, resolved backend and route, validation result, seed reproducibility |
| `superseded.jsonl` | records a supplementary run re-measured or retired, each tagged with the run that superseded it |
| `outputs/` | gzipped outputs of the simulators a row validates against (Aer and QuEST vectors, sampled counts), named by each record's `output_file`, so a rerun of part of a row can be checked against the stored reference; not committed |
| `fusion.json` | PRISM-Q's fused instruction counts for the dense headline circuits, from `matrix.py fusion-counts` |
| `samples.csv`, `summary.csv`, `derived.json` | written by `aggregate`: long-form samples, one row per observation, and the derived ratio, dispatch, scaling, sampling and seed tables |
| `circuits.jsonl.gz` | every gate list, one JSON line each with its SHA-256 |

### Environment variables

| Variable | Default |
| --- | --- |
| `PRISMQ_CMP_RUNNER` | `target/release/examples/compare_runner` |
| `PRISMQ_CMP_PEERS` | `comparison/peers/target/release/peers` |
| `PRISMQ_CMP_QUEST` | `comparison/quest/build/[Release/]quest_runner` |
| `PRISMQ_CMP_PYTHON` | the interpreter running `matrix.py` |

The overrides let a worktree reuse comparator builds from another checkout.

### Not measured

GPU paths (the comparators' GPU builds are Linux-only on this host; PRISM-Q's
`gpu` feature would need `--features "parallel gpu"` and a CUDA toolkit), 16
threads (the reference host has 8), ISA variants (PRISM-Q has no switch to force
its scalar kernels), Qulacs (no wheel for Python 3.14 on Windows), noise, and
sampling or mid-circuit rows on QuEST, Spinoza and qip (their adapters implement
the dense output only). The methodology page lists each with its reason.

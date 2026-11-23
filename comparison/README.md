# Cross-simulator comparison

Performance measurements against other quantum simulators, built so a reader
without the originating machine can regenerate the numbers and check them. The
aim is reproducible numbers, not a ranking.

`matrix.py` runs the benchmark matrix: eight workload families (dense random
brickwork, QFT, quantum volume, wide Clifford, shallow wide, disconnected,
sampling-heavy, mid-circuit), a circuit zoo (GHZ, Bernstein-Vazirani, QAOA on a
random 3-regular graph, a hardware-efficient ansatz, a Trotterized Ising chain, a
W state and a ripple-carry adder), scaling ladders that take each family as wide
as the host allows, thread scaling, PRISM-Q's automatic dispatch against every
forced backend, an accuracy study under a bond cap, tensor network expectation
values on wide shallow circuits, deep MPS sampling, a bond-cap sweep on wider
registers, and high-shot stabilizer sampling that returns statistics only. Its page
is [Performance](https://abecoull.github.io/prism-q/performance/index.html).

## Setup

```bash
python -m venv comparison/.venv
comparison/.venv/Scripts/pip install -r comparison/requirements.txt    # bin/pip on unix
# Stim from source with its AVX2 extension, replacing the wheel (needs a C++20 compiler;
# on Windows, Visual Studio Build Tools 2022 with the C++ workload)
comparison/.venv/Scripts/python comparison/build_stim.py

# Rust comparators (Spinoza needs nightly). Build from inside the crate so its
# rust-toolchain.toml and .cargo/config.toml (target-cpu=x86-64-v3) apply.
rustup toolchain install nightly
(cd comparison/peers && cargo build --release)

# QuEST comparator (CMake 3.21+, a C++17 compiler with OpenMP; fetches QuEST v4.3.0).
# Build Tools 2022 ships CMake under Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin.
cmake -S comparison/quest -B comparison/quest/build -G "Visual Studio 17 2022" -A x64   # or -G Ninja
cmake --build comparison/quest/build --config Release
```

`matrix.py run` builds `compare_runner` itself, and a second copy with the `gpu`
feature under `target/gpu/` when the profile names `prismq:gpu`; pass `--no-build`
when both are current. The GPU copy needs no CUDA toolkit to build. On a host
without a CUDA device its smoke check fails, so pass `--skip-peer prismq:gpu`.

`run` then runs `matrix.py smoke`: every simulator the selected profile names must be
installed, report its version, and match PRISM-Q on a tiny circuit in each output mode
it serves on the profile, at its top thread count there; a dense probability peer also
matches PRISM-Q on one circuit per gate it declares. `run` refuses to start while a
named peer fails; `--skip-peer NAME` (repeatable, a base name skips every variant)
leaves it out of the run and is recorded in the manifest.

## Benchmark matrix

```bash
comparison/.venv/Scripts/python comparison/matrix.py list --profile reference      # what would run, and what is skipped and why
comparison/.venv/Scripts/python comparison/matrix.py smoke --profile reference     # every named peer installed and agreeing with PRISM-Q
comparison/.venv/Scripts/python comparison/matrix.py run --profile smoke --out comparison/results/my-smoke
comparison/.venv/Scripts/python comparison/matrix.py run --profile reference --out comparison/results/my-run
comparison/.venv/Scripts/python comparison/matrix.py report comparison/results/my-run   --docs docs/performance
```

| Profile | Contents | Wall time |
| --- | --- | --- |
| `smoke` | every family, zoo circuit and adapter at a small size, a short ladder per method, a few seconds per row | about 10 minutes |
| `reference` | the published run: 156 workloads, 750 measurements and 29 not-applicable cells, sized for 24 threads and 32 GB | about 6 hours, run detached |
| `full` | adds every peer at every dense size, QV at 26 qubits and more thread counts | longer than `reference` |

A scaling ladder (group `scaling`) is a run of rows sharing one `ladder` key, from
narrow to wide. Each simulator climbs it until a rung times out, is refused for
memory, or takes longer than the ladder's `ceiling_s` for its cold call; the record
of that rung carries `ceiling` (`timeout`, `memory` or `budget`) and the wider rungs
are recorded as `skipped` with `beyond_ceiling`; a harness memory refusal also counts
as a `memory` ceiling, and `--append` keeps ceilings already recorded. The cold call is
timed inside every runner, the native peers included, so process start and the output
write stay out of the cutoff. The page shows the widest rung each simulator ran and what
stopped it, PRISM-Q on the CPU beside fp64 peers, with fp32, 1-thread and GPU results
apart. Dense ladders climb from 20 to 28 qubits in steps of 2, so a simulator that
clears 28 shows the top of the ladder rather than a ceiling. They force PRISM-Q's
statevector, so every rung is exact, with
`PRISM_MAX_SV_QUBITS` set to the width the harness admits for a dense peer (80% of RAM)
in place of PRISM-Q's default of half of physical memory, and add `prismq:gpu` on CUDA
device 0. Where no fp64 peer reached a dense rung, the two PRISM-Q statevectors are
checked against each other and marked as such. The stabilizer ladders also run PRISM-Q
at 1 thread beside single-threaded Stim.

`circuits.jsonl.gz` stores the generator name and arguments of every workload the
harness builds, and the text only of the lists exported from `prism_q::circuits`;
loading rebuilds each list and checks it against its recorded SHA-256.

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
timed call; kahypar has no Windows wheel, so cotengra falls back to greedy and
labels there, and the manifest records which it used.

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
| `observations.jsonl` | one record per simulator, workload and thread count: status (`ok`, `not-applicable` for a row the simulator cannot run, `skipped` for one the profile leaves out or one past a ladder ceiling, `timeout`, `oom`, `crash`, `error`) and reason, phases, warm samples, statistics, peak memory, resolved backend, route and placement, validation result, seed reproducibility, and for a ladder row its `ladder` key and any `ceiling` |
| `superseded.jsonl` | records a supplementary run re-measured or retired, each tagged with the run that superseded it |
| `outputs/` | gzipped outputs of the simulators a row validates against (Aer and QuEST vectors, sampled counts), named by each record's `output_file`, so a rerun of part of a row can be checked against the stored reference; not committed |
| `fusion.json` | PRISM-Q's fused instruction counts for the dense headline circuits, from `matrix.py fusion-counts` |
| `samples.csv`, `summary.csv`, `derived.json` | written by `aggregate`: long-form samples, one row per observation, and the derived ratio, dispatch, scaling, sampling and seed tables |
| `circuits.jsonl.gz` | every gate list, one JSON line each with its SHA-256 and either its generator recipe or its text |

### Environment variables

| Variable | Default |
| --- | --- |
| `PRISMQ_CMP_RUNNER` | `target/release/examples/compare_runner` |
| `PRISMQ_CMP_RUNNER_GPU` | `target/gpu/release/examples/compare_runner`, built with `parallel gpu` |
| `PRISMQ_CMP_PEERS` | `comparison/peers/target/release/peers` |
| `PRISMQ_CMP_QUEST` | `comparison/quest/build/[Release/]quest_runner` |
| `PRISMQ_CMP_PYTHON` | the interpreter running `matrix.py` |

The overrides let a worktree reuse comparator builds from another checkout.

### Not measured

Any peer GPU path: the Aer and qsim GPU builds need Linux, and QuEST, which
builds with CUDA under MSVC, was built CPU-only, so PRISM-Q's GPU statevector on the
dense ladders has no GPU peer. Also ISA variants (PRISM-Q has no switch to
force its scalar kernels), Qulacs (no wheel for Python 3.14 on Windows), noise,
sampling or mid-circuit rows on QuEST, Spinoza and qip (their adapters implement
the dense output only), and Aer fp32 on rows with CP or RZZ (it exits with an access
violation on CP when fusion is on and on RZZ). Spinoza has no ZZ rotation, so its
adapter applies RZZ as two phase gates and a controlled phase. The methodology page
lists each with its reason.

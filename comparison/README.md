# Cross-simulator comparison

Performance measurements against other quantum simulators, built so a reader
without the originating machine can regenerate the numbers and check them. The
aim is reproducible numbers, not a ranking.

Two harnesses share the adapters. `matrix.py` is the benchmark matrix: eight
workload families (dense random brickwork, QFT, quantum volume, wide Clifford,
shallow wide, disconnected, sampling-heavy, mid-circuit), thread scaling,
PRISM-Q's automatic dispatch against every forced backend, and an accuracy study
under a bond cap. Its page is
[Performance](https://abecoull.github.io/prism-q/performance/index.html).
`run_comparison.py` is the earlier probability-vector comparison of four
families at 16 to 24 qubits, with a `--check` mode against its committed
reference.

## Benchmark matrix

```bash
comparison/.venv/Scripts/python comparison/matrix.py list --profile reference      # what would run, and what is skipped and why
comparison/.venv/Scripts/python comparison/matrix.py run --profile smoke --out comparison/results/my-smoke
comparison/.venv/Scripts/python comparison/matrix.py run --profile reference --out comparison/results/my-run
comparison/.venv/Scripts/python comparison/matrix.py report comparison/results/my-run   --docs docs/performance
```

| Profile | Contents | Wall time on the reference host |
| --- | --- | --- |
| `smoke` | every family and adapter at a small size, a few seconds per row | about 3 minutes |
| `reference` | the published subset: 52 workloads, 269 measurements and 6 not-applicable cells | about 90 minutes |
| `full` | adds 28-qubit dense rows, QuEST at 26 qubits, Spinoza and qip at every size, threads up to 16 | most of a day; 32 GB RAM |

`--groups headline,dispatch` and `--match qft` narrow a profile. `--append`
adds to an existing results directory, and `--replace` with it moves the records
being re-measured to `superseded.jsonl`. `--retire TEXT` also moves records whose
workload id contains `TEXT`, for a workload whose definition changed. Run a long
profile from a scheduled task or another detached shell on an otherwise idle
machine, never beside a build.

The harness is formatted with `ruff format` and checked with `ruff check` under
`comparison/ruff.toml`. Its tests run with
`comparison/.venv/Scripts/python -m unittest discover -s comparison/tests`.

### Layout

| File | Role |
| --- | --- |
| `harness/workloads.py` | canonical workload generators and the gate list format |
| `harness/profiles.py` | the matrices: which workloads, simulators, thread counts, repetition policy |
| `harness/runners.py` | simulator registry, applicability rules, one process per measurement, peak memory, failure classification |
| `harness/worker.py` | Aer, qsim and Stim behind the same `exec` protocol as `tools/compare_runner.rs` |
| `harness/reference.py` | independent exact reference for measurement statistics (past-cone extraction, branching NumPy statevector) |
| `harness/validate.py` | TVD for dense outputs, Bonferroni-corrected exact binomial tests for sampled outputs |
| `harness/stats.py` | medians, IQR, seeded bootstrap intervals for medians and ratios |
| `harness/aggregate.py`, `figures.py`, `pages.py` | CSV and JSON tables, SVG figures, and the documentation pages, all from the raw files |

### Results directory

| File | Contents |
| --- | --- |
| `manifest.json` | run id, profile, host (CPU, flags, affinity, clock, power plan, GPU, RAM, OS), toolchains, package versions, environment, timed-region and policy statements, SHA-256 of the runner binaries and harness scripts, and every supplementary run |
| `observations.jsonl` | one record per simulator, workload and thread count: status (`ok`, `not-applicable`, `timeout`, `oom`, `crash`, `error`) and reason, phases, warm samples, statistics, peak memory, resolved backend and route, validation result, seed reproducibility |
| `superseded.jsonl` | records a supplementary run re-measured or retired, each tagged with the run that superseded it |
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

## Probability-vector comparison

Every `run_comparison.py` run emits a machine-readable file carrying the
circuits it ran (by content hash), the host, the version of everything involved,
and an equivalence check showing the simulators computed the same distribution.
The controls behind it are in
[Benchmark Methodology](https://abecoull.github.io/prism-q/guides/benchmarking.html).

### Simulators

| Adapter | Simulator | How it runs | Threads |
| --- | --- | --- | --- |
| `prismq` | PRISM-Q, `BackendKind::Auto` | `tools/compare_runner.rs`, built by the harness | `RAYON_NUM_THREADS` |
| `aer-statevector`, `aer-automatic` | Qiskit Aer | in-process through `qiskit_aer.AerSimulator` | `max_parallel_threads` |
| `qsim` | qsim through qsimcirq | in-process through `qsimcirq.QSimSimulator` | `QSimOptions.cpu_threads` |
| `quest` | QuEST, OpenMP build | `quest/quest_runner`, built with CMake | `OMP_NUM_THREADS` |
| `spinoza` | Spinoza | `peers/peers spinoza`, built on nightly Rust | `Config.threads` |
| `qip` | RustQIP (`qip`) | `peers/peers qip`, built on nightly Rust | `RAYON_NUM_THREADS` |

Every simulator keeps its own optimization on (gate fusion in PRISM-Q, Aer and
qsim), since that is what its users get. All run in double precision except
qsim, which offers only single precision. The equivalence tolerance passes
float32 rounding and fails a wrong answer, and the measured distance is printed
per row.

### Setup

```bash
python -m venv comparison/.venv
comparison/.venv/Scripts/pip install -r comparison/requirements.txt    # bin/pip on unix

# Rust comparators (Spinoza needs nightly; the crate pins it)
rustup toolchain install nightly
cargo build --release --manifest-path comparison/peers/Cargo.toml

# QuEST comparator (CMake 3.21+, a C++17 compiler with OpenMP; fetches QuEST v4.3.0)
cmake -S comparison/quest -B comparison/quest/build -G "Visual Studio 17 2022" -A x64   # or -G Ninja
cmake --build comparison/quest/build --config Release
```

`run_comparison.py` builds `compare_runner` itself; pass `--no-build` when it is
current. A comparator whose executable is missing fails the run with the path it
expected, so a partial setup cannot silently narrow the run. Pass `--simulators`
to run a subset.

### Running

```bash
comparison/.venv/Scripts/python comparison/run_comparison.py \
  --qubits 16,20,24 --iterations 5 \
  --out comparison/results/run.json --markdown comparison/results/run.md
```

To check a run against a committed reference:

```bash
comparison/.venv/Scripts/python comparison/run_comparison.py --check comparison/results/reference-<host>.json
```

`--check` exits non-zero when a circuit hash differs, an output diverges, or a
verdict flips. It checks ratios and verdicts; absolute times do not transfer
across hardware.

### What is measured

On every side the timer covers executing the circuit and materializing the full
`2^n` probability vector. Building the circuit from the gate list is outside it.

| Rule | Why |
| --- | --- |
| One shared gate list per circuit | `compare_runner export` writes the suite circuit as `h`, `cx`, `swap`, `ry`, `rz` and `cp` lines; every adapter replays those through its native API. No simulator parses OpenQASM and none gets a private transpilation pass. |
| Matched thread counts | Every simulator is pinned to `--threads` through its own control. |
| Comparators at their documented defaults | Fusion stays on where a simulator has it; nothing is tuned per row. |
| Median of N iterations after a warmup | Reported with min, p25 and p75, never a single sample. `--iteration-budget-s` caps the time on one simulator and circuit: the equivalence pass doubles as a probe, a comparator that cannot fit N samples in the budget gets as many as fit (at least two), and the report names every row measured with fewer. A 120 s budget cuts a full run from about an hour to about fifteen minutes on the reference host, losing precision only on rows far from the band. |
| A 10% band | Cross-process timing noise on one host is of that order, so a ratio within the band is reported as within band rather than as a direction. |
| Equivalence checked in the same run | Total variation distance against `aer-statevector`. A row that diverges keeps its timing in the table but is not counted. |
| Per-call overhead recorded | Driving Aer or qsim from Python costs a fixed amount per call, measured on a one-gate circuit and printed, so rows under 16 qubits are shown but left out of the summary. |

PRISM-Q is measured under `BackendKind::Auto`, the default. Aer is measured
twice, forced to `statevector` and on `automatic`, which separates the part of a
gap that comes from dispatch (Clifford circuits take Aer's stabilizer method
under `automatic`, as they take PRISM-Q's stabilizer backend) from the part that
comes from the kernels.

The comparison is CPU against CPU on one host; it covers no simulator's CUDA,
MPI or distributed paths.

### Corpus

The four families of the `bench_suite` run on the
[Performance](https://abecoull.github.io/prism-q/performance/index.html) page, from the `prism_q::circuits` generators with circuit seed `0xDEAD_BEEF`:

| Family | Circuit |
| --- | --- |
| `ghz` | H on qubit 0, then a CX chain |
| `qft` | Textbook quantum Fourier transform: H, controlled phases, final swaps (the generator's `QftBlock`, expanded; PRISM-Q folds the exact expansion back into its block FFT, as it does for a QFT parsed from OpenQASM) |
| `hea` | Hardware-efficient ansatz: 5 layers of Ry and Rz on every qubit, then a CX chain |
| `qv` | Quantum volume: `n` layers of random pairings, each pair a random SU(4) as CX and rotations |

The results file stores each gate list's SHA-256. A circuit is never dropped for
how it performs; a size the generator cannot produce is listed with the reason.

### Results file

One JSON file per run, `schema_version` 1. The fields that matter for
verification:

| Field | Contents |
| --- | --- |
| `provenance` | CPU, cores, RAM, OS, rustc for PRISM-Q and for the peers, the C++ compiler, cargo features, RUSTFLAGS, commit and dirty flag, Python and package versions, every simulator's version, thread counts |
| `corpus` | Families, gate set, circuit seed, per-circuit SHA-256, and every skip with its reason |
| `run.timed_region` | Prose statement of what the timer covers |
| `results[].equivalence` | Total variation distance per simulator against the reference and whether it passed |
| `results[].ratio_vs_prismq` | Comparator median over PRISM-Q median; above 1.00 means PRISM-Q finished sooner |
| `results[].verdict` | `prismq-sooner`, `within-band`, or `comparator-sooner`; absent where the comparator did not run or diverged |
| `summary` | Counts per comparator over the headline rows, the median and range of ratios, every equivalence failure and every error |

`harness/report.py` renders this to Markdown, and `--docs <path>` writes the
documentation page from the same data. Both are generated, never hand edited,
and print rows where PRISM-Q is slower the same way as rows where it is faster.

### Generating a reference

Reference results are committed per host so a verifier has something to diff
against. Generate one on a quiet machine, never alongside `cargo bench`:

```bash
comparison/.venv/Scripts/python comparison/run_comparison.py --iterations 5 --qubits 16,20,24 \
  --out comparison/results/reference-<host>.json \
  --markdown comparison/results/reference-<host>.md
```

Commit a reference only when its run passed the equivalence checks.

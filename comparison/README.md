# Cross-simulator comparison

Comparative performance measurements against commonly used quantum simulators,
built so a reader with no access to the originating machine can regenerate the
numbers and check them. The results are intended to make performance
characteristics reproducible and transparent across representative workloads,
not to rank projects.

Every run emits a machine-readable file carrying the circuits it ran (by content
hash), the host, the version of everything involved, and an equivalence check
showing the simulators computed the same distribution. The rendered page is
[Comparative Measurements](https://abecoull.github.io/prism-q/comparison.html)
in the documentation; the controls behind it are in
[Benchmark Methodology](https://abecoull.github.io/prism-q/guides/benchmarking.html).

## Simulators

| Adapter | Simulator | How it runs | Threads |
| --- | --- | --- | --- |
| `prismq` | PRISM-Q, `BackendKind::Auto` | `examples/compare_runner.rs`, built by the harness | `RAYON_NUM_THREADS` |
| `aer-statevector`, `aer-automatic` | Qiskit Aer | in-process through `qiskit_aer.AerSimulator` | `max_parallel_threads` |
| `qsim` | qsim through qsimcirq | in-process through `qsimcirq.QSimSimulator` | `QSimOptions.cpu_threads` |
| `quest` | QuEST, OpenMP build | `quest/quest_runner`, built with CMake | `OMP_NUM_THREADS` |
| `spinoza` | Spinoza | `peers/peers spinoza`, built on nightly Rust | `Config.threads` |
| `qip` | RustQIP (`qip`) | `peers/peers qip`, built on nightly Rust | `RAYON_NUM_THREADS` |

Every simulator keeps its own optimization on (gate fusion in PRISM-Q, Aer and
qsim), since that is what its users get. All run in double precision except
qsim, whose amplitudes are single precision because that is the only precision
it offers; the equivalence tolerance is set to pass float32 rounding and fail a
wrong answer, and the measured distance is printed per row.

## Setup

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
expected, so a partial setup is reported rather than silently narrowed; pass
`--simulators` to run a subset.

## Running

```bash
comparison/.venv/Scripts/python comparison/run_comparison.py \
  --qubits 16,20,24 --iterations 5 \
  --out comparison/results/run.json --markdown comparison/results/run.md
```

To check a run against a committed reference:

```bash
comparison/.venv/Scripts/python comparison/run_comparison.py --check comparison/results/reference-<host>.json
```

`--check` exits non-zero when a claim stops holding: a circuit hash that differs,
an output that diverges, or a verdict that flips. Absolute times are not checked
because they do not transfer across hardware; ratios and verdicts are.

## What is measured

The timed region is the same on every side: execute the circuit and materialize
the full `2^n` probability vector. Building the circuit from the gate list is
outside it for every simulator.

| Rule | Why |
| --- | --- |
| One shared gate list per circuit | `compare_runner export` writes the suite circuit as `h`, `cx`, `swap`, `ry`, `rz` and `cp` lines; every adapter replays those through its native API. No simulator parses OpenQASM and none gets a private transpilation pass. |
| Matched thread counts | Every simulator is pinned to `--threads` through its own control. |
| Comparators at their documented defaults | Fusion stays on where a simulator has it; nothing is tuned per row. |
| Median of N iterations after a warmup | Reported with min, p25 and p75. Never a single sample. |
| A 10% band | Cross-process timing noise on one host is of that order, so a ratio within the band is reported as within band rather than as a direction. |
| Equivalence checked in the same run | Total variation distance against `aer-statevector`. A row that diverges keeps its timing in the table but is not counted. |
| Per-call overhead recorded | Driving Aer or qsim from Python costs a fixed amount per call, measured on a one-gate circuit and printed, so rows under 16 qubits are shown but left out of the summary. |

PRISM-Q is measured under `BackendKind::Auto`, the label a user gets by default.
Aer is measured twice, forced to `statevector` and on `automatic`, so a reader
can separate the part of a gap that comes from dispatch (Clifford circuits take
Aer's stabilizer method under `automatic`, as they take PRISM-Q's stabilizer
backend) from the part that comes from the kernels.

Nothing here covers the CUDA, MPI or distributed paths of any simulator. The
comparison is CPU against CPU on one host.

## Corpus

The four families of the published [Benchmarks](https://abecoull.github.io/prism-q/benchmarks.html)
page, from the `prism_q::circuits` generators with circuit seed `0xDEAD_BEEF`:

| Family | Circuit |
| --- | --- |
| `ghz` | H on qubit 0, then a CX chain |
| `qft` | Textbook quantum Fourier transform: H, controlled phases, final swaps (the generator's `QftBlock`, expanded; PRISM-Q replays the expansion like every other simulator, so its block FFT path is not part of the comparison) |
| `hea` | Hardware-efficient ansatz: 5 layers of Ry and Rz on every qubit, then a CX chain |
| `qv` | Quantum volume: `n` layers of random pairings, each pair a random SU(4) as CX and rotations |

The gate list is hashed with SHA-256 and the hash is stored in the results file.
A circuit is never dropped for how it performs; a size the generator cannot
produce is listed with the reason.

## Results file

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
and print rows where PRISM-Q is slower the same as rows where it is faster.

## Generating a reference

Reference results are committed per host so a verifier has something to diff
against. Generate one on a quiet machine, with nothing else running, never
alongside `cargo bench`:

```bash
comparison/.venv/Scripts/python comparison/run_comparison.py --iterations 5 --qubits 16,20,24 \
  --out comparison/results/reference-<host>.json \
  --markdown comparison/results/reference-<host>.md \
  --docs docs/comparison.md
```

A reference is only meaningful when the run it came from passed its equivalence
checks. Do not commit one that did not.

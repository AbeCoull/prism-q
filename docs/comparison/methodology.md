# Benchmark Matrix: Methodology

This page states how the [comparative measurements](../comparison.md) were taken, so a
reader can reproduce them or argue with them. The harness is `comparison/matrix.py` and
the modules under `comparison/harness/`; every table and figure on the [results
page](./results.md) is generated from the raw files a run writes.

## Questions the matrix answers

1. On general dense circuits (random brickwork, QFT, quantum volume), how does PRISM-Q's
   time per call compare with established statevector simulators at matched precision and
   threads?
2. On structured circuits that a dense statevector cannot hold (wide Clifford, shallow
   wide, disconnected), how does PRISM-Q's choice of representation compare with the
   specialized method another simulator offers for the same structure?
3. Where does the time go: process start, circuit construction, the first call, or the
   steady-state call?
4. How do the simulators scale with thread count and with qubit count, and what does each
   one's memory footprint look like?
5. Does PRISM-Q's automatic dispatch pick the fastest backend it has, and what does the
   decision itself cost?
6. Where an approximation is in play (a bond-dimension cap), what accuracy does each
   runtime buy?

## What is compared

| Adapter | Simulator and configuration | Precision |
| --- | --- | --- |
| `prismq` | PRISM-Q, `BackendKind::Auto`, `--features parallel`, release profile | fp64 |
| `prismq:<backend>` | PRISM-Q with one backend forced (dispatch and accuracy studies only) | fp64 |
| `aer-statevector` | Qiskit Aer, `method="statevector"`, defaults otherwise (fusion on) | fp64 |
| `aer-statevector-f32` | Qiskit Aer, `method="statevector"`, `precision="single"` | fp32 |
| `aer-automatic` | Qiskit Aer, `method="automatic"` | per method |
| `aer-mps`, `aer-mps:<chi>` | Qiskit Aer, `method="matrix_product_state"`, optional bond cap | fp64 |
| `stim` | Stim, `Circuit.compile_sampler(seed=42).sample(shots)`; Clifford rows only, single-threaded | exact (tableau) |
| `qsim` | qsim through qsimcirq, `QSimOptions` defaults | fp32 (its only precision) |
| `quest` | QuEST v4.3.0, OpenMP build | fp64 |
| `spinoza` | Spinoza, `Config.threads` pinned | fp64 |
| `qip` | RustQIP, `LocalBuilder<f64>`, parallel feature | fp64 |

Each simulator runs at its documented defaults, with its own optimizations on, because
that is what its users get. Nothing is tuned per row. fp32 results are printed in their
own tables and their own heatmap panel: a single-precision run does half the memory
traffic of a double-precision one, so a ratio across precisions measures the precision as
much as the simulator. `aer-statevector-f32` exists so the fp32 group has a statevector
beside qsim. PRISM-Q has no single-precision path, so it has no entry in that group, and
its fp64 time is printed beside the fp32 tables for scale only.

The Clifford rows compare PRISM-Q, under automatic dispatch, with Stim, both as tableau
simulators. The shallow and disconnected rows compare PRISM-Q with Aer's matrix product
state method, built for that structure, and with `aer-automatic`, which shows what Aer's
own dispatcher picks. These are not statevector comparisons, and the tables label each
simulator's representation.

## Workloads and why each is here

| Family | Sizes in the reference run | What it measures |
| --- | --- | --- |
| Dense random brickwork (`rcs`) | 20, 24, 26 qubits; depth 10 and 20 | General dense evolution at a fixed gate mix: Rz Ry Rz on every qubit per layer, CX on alternating neighbour pairs. Compute and memory bandwidth on a full statevector. |
| QFT | 20, 24, 26 | A structured circuit with long-range controlled phases, applied to a seeded random product state (Ry then Rz on every qubit). On \|0...0> every controlled phase has its control in \|0>, so the output is uniform whatever the phase angles and a dropped gate would pass validation; the product input makes every phase change the output. Every simulator receives the same expanded textbook sequence; PRISM-Q can recognize a textbook QFT and run it on a block FFT, and the run metadata does not record whether it did. |
| Quantum volume (`qv`) | 16, 20, 24 | `n` layers; each layer pairs the qubits by a random permutation and applies a random SU(4) per pair, decomposed into CX, Ry and Rz (the `prism_q::circuits::quantum_volume_circuit` construction). Two-qubit gates on arbitrary pairs, so locality-based tricks do not apply. |
| Clifford brickwork (`clifford`) | 100, 1000, 10000 qubits; depth 8; 1000 shots | Tableau simulation, PRISM-Q against Stim. A dense statevector is impossible past about 30 qubits. |
| Shallow wide (`lowent`) | 50, 100, 200 qubits; depth 4 and 8; 1000 shots | Low entanglement by construction: the bond dimension across any cut is at most 2^ceil(d/2), so an exact MPS fits. |
| Disconnected (`disjoint`) | 40, 100, 200 qubits; 10-qubit components; 1000 shots | Independent 10-qubit components, each a depth-10 brickwork. `contiguous` places component k on qubits 10k to 10k+9; `strided` (40 qubits) interleaves the components, so a method that relies on qubit order sees long-range gates. |
| Sampling (`sampling`) | 16, 20, 24 qubits; 1 to 1,000,000 shots; all or half the qubits measured | Separates the one-time simulation cost from the per-shot sampling cost. |
| Mid-circuit (`midcircuit`) | 12 qubits at 1000 and 10,000 shots, 18 qubits at 1000; 3 rounds | Repeated parity checks: data rotations and entanglers, each ancilla measured, a conditional X fed forward from the outcome, the ancilla reset. Exercises measurement, reset and classical control inside the circuit. PRISM-Q only in the reference run: Aer simulates these shot by shot and was left out for its run time; qsim's adapter has no reset or feed-forward. |

Random families draw from NumPy's PCG64 with seed `0xDEADBEEF`; QFT and QV come from the
`prism_q::circuits` generators through `compare_runner export`, the same circuits as the
[Benchmarks](../benchmarks.md) page. Every gate list is written once, hashed with SHA-256,
and handed unchanged to every simulator through its native API. No simulator parses
OpenQASM and none gets a transpilation pass of its own. The lists are saved with their
SHA-256 in `circuits.jsonl.gz` beside the results.

## Timed regions

| Phase | What it covers | Where it comes from |
| --- | --- | --- |
| Process start and imports | From process launch to a ready simulator object | Python workers time their own imports; for the Rust runner it is process wall time minus everything below |
| Circuit build | Gate list text to the simulator's circuit object | Timed in every runner, never inside the call |
| Cold call | The first execution in the process | Includes first-touch allocation and any one-time setup |
| Warm call | Each subsequent execution; the headline number is the median of these | Same process, same circuit object |
| Plan | PRISM-Q's dispatch decision alone, the median of 21 calls to `Simulate::plan` | Rust runner |

For dense rows a call is: execute the circuit and materialize the full 2^n probability
vector (for qsim, the squared magnitudes of its state vector, computed inside the call).
For sampled rows a call is: execute the circuit and return the simulator's own shot result
(PRISM-Q's count map, Aer's `Result`, qsim's measurement array, Stim's sample array).
Stim's timed call includes `compile_sampler`, as PRISM-Q's includes its own compilation;
Stim returns per-shot bits while PRISM-Q returns a histogram, and the histogram of Stim's
bits is built after the timer. Conversion to the harness's file format happens after the
timer stops, for every simulator. Aer's own `time_taken` for each call is recorded too, so
the Python-side share of an Aer call can be read off. QuEST, Spinoza and qip are driven
through their older two-command protocol, which gives the same warm-call region but no
phase split.

## Repetition and stopping policy

Declared per profile before the run, and stored with every observation:

- one cold call, then up to `iterations` warm calls in the same process (5 in the
  reference profile, 3 in smoke);
- stop early once the warm calls have used the row's budget (30 s in the reference
  profile, 15 s for dispatch rows), but never below `min_iterations` (3; 2 for the
  18-qubit mid-circuit row);
- a call that outlives the row's timeout is killed and recorded as a timeout, never
  retried.

Rows are never dropped for how they perform and outliers are never removed. Every sample
is in `samples.csv`. A row is re-measured whole, never one simulator at a time, so every
ratio pairs two observations from the same pass. A supplementary run that re-measures a
row moves the earlier records to `superseded.jsonl` beside the observations, and the
manifest lists every supplementary run with its command, load check, code hashes and the
number of records it moved. The first six supplementary runs of the reference run predate
this rule and deleted the records they replaced; their counts are in the manifest. The
dense rows were then re-measured twice: a pass that started on a busy host (its load check
is in the manifest) and the published pass, with both earlier sets in `superseded.jsonl`.

## Statistics

Each cell reports the median warm call with its p25 and p75. The 95% interval on a median,
and on a ratio of two medians, is a percentile bootstrap (2000 resamples, seed 42) over
the warm samples. With five samples that interval cannot leave the sample range, so it
shows how spread the samples were rather than a precise bound. A ratio is the comparator's
median over PRISM-Q's, so 2.00x means the comparator's median call took twice as long.
Cross-process timing on one shared desktop varies by up to about 10%, so ratios closer to
1.00x than that are not distinguishable from each other on this host.

Measurement noise and workload variation are reported separately. Within a row, the spread
is the IQR of repeated calls on one circuit. Across rows, the seed study reruns the
20-qubit random brickwork and QV families on three circuit seeds; when the spread of the
per-seed medians is no larger than the within-seed IQR, the seed does not move the timing.

## Correctness

Every output is checked in the same run that timed it.

- Dense outputs: total variation distance (TVD) against the row's fp64 reference vector,
  `aer-statevector`'s (QuEST's where Aer did not run), with a tolerance of 1e-8 for fp64
  and 1e-5 for fp32. A wrong circuit lands near 1e-1. The dispatch rows run
  `aer-statevector` beside PRISM-Q's backends for this purpose. The reference's own record
  is marked as the reference rather than counted as a pass.
- Sampled outputs: the counts are reduced to single-bit marginals P(b = 1) and
  adjacent-pair parities P(a xor b = 1) (all of them up to 128 bits, a seeded sample of
  128 of each beyond). Each statistic is tested against its exact value with a two-sided
  exact binomial test, Bonferroni-corrected so a correct simulator has a 1e-3 chance of
  any false alarm across the whole set. A statistic the reference proves deterministic
  must match exactly. Below 100 shots only the deterministic statistics are checked and
  the output counts as not validated; its timing is still reported. The sampled rows of
  the reference run predate the exact test and were checked with a normal-approximation
  z-test, which over-rejects rare outcomes and so errs towards a false failure; each
  record names the test it went through.
- The exact values come from `comparison/harness/reference.py`, which shares no code with
  any simulator under test. The distribution of a few classical bits depends only on their
  causal past: the gates that touch a qubit in the cone, the measurements and resets of
  those qubits, and the measurements whose bits feed a conditional inside it. The
  reference extracts that cone and evaluates it with a small NumPy statevector that
  branches on every measurement. On local circuits the cone stays narrow at any width,
  which gives exact statistics for the 10,000-qubit Clifford and 200-qubit shallow rows. A
  statistic whose cone exceeds 20 qubits, or holds more than 14 measurements and resets,
  falls back to a two-sample z-test against another simulator's counts (never another
  PRISM-Q backend for a PRISM-Q record), and the results say so.
- Seed reproducibility: PRISM-Q, Aer and Stim are reseeded on every call, so every call
  must return the same output; the hash of each call's output is compared. qsim is seeded
  once per process, so its calls draw fresh samples and the check does not apply.
- The accuracy study deliberately runs truncated MPS configurations; their distance from
  the exact vector is the measurement, so they are not counted as validation failures.

## Fairness controls

- One gate list per workload, replayed natively by every simulator, with its hash in every
  record.
- Thread counts matched through each simulator's own control: `RAYON_NUM_THREADS` for
  PRISM-Q, Spinoza and qip, `OMP_NUM_THREADS` for QuEST, `max_parallel_threads` for Aer,
  `QSimOptions.cpu_threads` for qsim.
- One process per measurement, so a simulator's allocator state, caches and peak memory
  belong to the workload that caused them.
- The order of simulators within each workload is shuffled with a fixed seed (42) and
  recorded, so slow drift on the host does not always land on the same simulator; a 0.5 s
  pause separates measurements.
- The results tables keep fp64 and fp32 comparators in separate tables. The heatmap and
  the landing summary do show fp32 ratios, against PRISM-Q's fp64 time, in their own
  labelled block. Approximate and exact runs are never compared on time alone.
- Peak memory is the largest working set over the measuring process and its children (a
  Python virtual environment's launcher runs the interpreter as a child), sampled every 20
  ms and once after exit.

## Not covered, and why

| Not measured | Reason |
| --- | --- |
| GPU paths | The comparators' GPU builds are Linux-only here (the Aer GPU wheel, qsim's CUDA build), and PRISM-Q's GPU feature was out of scope for this CPU run. The host has a GTX 1080 Ti; commands are in `comparison/README.md`. |
| 16 threads | The host has 8 logical threads. The `full` profile lists 16 for hosts that have them. |
| ISA variants | PRISM-Q selects its kernels by runtime CPU detection and has no switch to force a scalar path; its AVX-512 tier needs hardware this Skylake host lacks. |
| 28-qubit dense rows, QuEST at 26 qubits, Spinoza and qip past 20 | Cost: a single QuEST call at 26 qubits runs for minutes. All are in the `full` profile. |
| Qulacs | No wheel for Python 3.14 on Windows at the time of the run. |
| QuEST, Spinoza and qip sampling and mid-circuit rows | Their adapters implement the dense output only. |
| Noise | Noise semantics differ enough between simulators that a like-for-like noisy row needs its own design. |

## Limits to generalization

One host, one operating system, one compiler per simulator, a desktop with an interactive
session running beside the benchmark. Absolute times do not transfer to other hardware;
the ratios hold only to the extent that the bottleneck (memory bandwidth, core count,
cache size) is the same on the reader's machine. The sizes stop where a dense statevector
stops fitting in 32 GB. The large-structure families are constructed to have the structure
a specialized method exploits, so they show what that method costs when it applies, not
how often it applies in a reader's workload.

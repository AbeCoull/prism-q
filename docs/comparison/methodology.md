# Benchmark Matrix: Methodology

Harness: `comparison/matrix.py` and `comparison/harness/`. The
[results page](./results.md) is generated from the raw run files.

## Simulators

| Adapter | Configuration | Precision |
| --- | --- | --- |
| `prismq` | PRISM-Q, automatic dispatch, `--features parallel` | fp64 |
| `prismq:<backend>` | PRISM-Q, one backend forced | fp64 |
| `aer-statevector`, `-f32` | Qiskit Aer statevector, defaults | fp64, fp32 |
| `aer-mps`, `aer-automatic` | Qiskit Aer MPS; Aer's own method choice | fp64 |
| `stim` | Stim compiled sampler, single-threaded | exact |
| `qsim` | qsimcirq, defaults | fp32 |
| `quest`, `spinoza`, `qip` | QuEST v4.3.0, Spinoza, RustQIP | fp64 |

Defaults everywhere, nothing tuned per row. fp32 results get their own columns.

## Workloads

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

Seed `0xDEADBEEF`. One gate list per workload, replayed through each simulator's native
API, stored in `circuits.jsonl.gz`.

## Timing

- Timed call: run the circuit, return probabilities or shots. Format conversion is untimed.
- One process per measurement; peak memory is that process's.
- One cold call, then up to five warm calls; the median warm call is reported.
- Simulator order shuffled per row (seed 42).
- Failures are recorded with their reason, never retried.
- Rows are re-measured whole; replaced records go to `superseded.jsonl`.

## Validation

- Dense: total variation distance to an fp64 reference, at most 1e-8 (fp32: 1e-5).
- Sampled: bit marginals and pair parities against exact values from an independent
  light-cone reference, exact binomial tests, family-wise alpha 1e-3. Under 100 shots,
  deterministic bits only.

## Not covered

GPU, 16 threads, ISA variants, noise, Qulacs (no Windows wheel for Python 3.14), and dense
sizes past 26 qubits (in the `full` profile).

## Rerunning

```bash
comparison/.venv/Scripts/python comparison/matrix.py run --profile reference --out comparison/results/my-run
comparison/.venv/Scripts/python comparison/matrix.py fusion-counts comparison/results/my-run
comparison/.venv/Scripts/python comparison/matrix.py report comparison/results/my-run
```

About 90 minutes on the host above; `--profile smoke` takes about 2. Comparator builds are
in `comparison/README.md`.

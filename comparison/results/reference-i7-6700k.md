# Comparative measurements

Comparative performance measurements against commonly used quantum simulators. These results are intended to make performance characteristics reproducible and transparent across representative workloads, not to rank projects: each simulator makes different trade-offs, and a ratio here describes one workload on one host under the controls listed below.

Every simulator replays the same gate list for each circuit, built from the `prism_q::circuits` generators and hashed so a rerun can prove it did the same work. The timed region is: execute the circuit and materialize the full 2^n probability vector; building the circuit from the gate list is excluded on every side. Ratios are the comparator's median over PRISM-Q's median, so a ratio above 1.00x means PRISM-Q finished sooner and below 1.00x means the comparator did; a ratio within 10% of 1.00x is reported as within band, because cross-process timing noise on one host is of that order.

## Host and versions

| Field | Value |
| --- | --- |
| CPU | Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz |
| Cores | 4 physical, 8 logical |
| RAM | 31.9 GB |
| OS | Windows 10 |
| rustc (PRISM-Q) | 1.97.0, features `parallel`, profile release |
| rustc (Spinoza, qip) | 1.99.0-nightly, profile release |
| C++ compiler (QuEST) | MSVC 19.38.33145.0 |
| Python | 3.14.0 |
| Commit | d43f14b46620500419fa980c1cf159ee5a2c1e25 |
| Threads | 8 on every simulator |
| Iterations | 5 timed per circuit after one warmup |

| Simulator | Version | Settings |
| --- | --- | --- |
| PRISM-Q (auto dispatch) | 0.33.0 | BackendKind::Auto, fusion on, double precision, RAYON_NUM_THREADS pinned |
| Qiskit Aer (statevector) | 0.17.2 | method=statevector, fusion on (default), double precision, max_parallel_threads pinned |
| Qiskit Aer (automatic) | 0.17.2 | method=automatic, fusion on (default), double precision, max_parallel_threads pinned |
| qsim (qsimcirq) | 0.22.1 | QSimOptions defaults (max_fused_gate_size 2), cpu_threads pinned; single precision, the only precision qsim offers |
| QuEST (OpenMP) | v4.3.0 | static library, double precision, OpenMP on, OMP_NUM_THREADS pinned |
| Spinoza | 0.5.1 (git f900971) | default double feature, Config.threads pinned |
| RustQIP (qip) | 1.5.0 | LocalBuilder<f64>, parallel feature, RAYON_NUM_THREADS pinned |

Per-call overhead of driving a comparator from Python on a one-gate circuit, recorded so small rows can be read correctly: aer-statevector 464 us, aer-automatic 477 us, qsim 133 us.

## Results

| Circuit | Qubits | Gates | PRISM-Q | aer-statevector | ratio | aer-automatic | ratio | qsim | ratio | quest | ratio | spinoza | ratio | qip | ratio | max TVD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ghz | 16 | 16 | 34 us | 3.0 ms | 88.68x | 1.4 ms | 42.90x | 2.9 ms | 86.72x | 2.4 ms | 72.11x | 1.4 ms | 42.54x | 7.3 ms | 216.67x | 1.7e-08 |
| qft | 16 | 144 | 2.4 ms | 25.1 ms | 10.36x | 25.1 ms | 10.36x | 12.3 ms | 5.06x | 13.1 ms | 5.41x | 10.3 ms | 4.27x | 126.1 ms | 52.06x | 1.9e-15 |
| hea | 16 | 235 | 2.5 ms | 24.5 ms | 9.64x | 24.2 ms | 9.49x | 16.1 ms | 6.32x | 23.2 ms | 9.12x | 92.8 ms | 36.46x | 50.4 ms | 19.82x | 5.1e-07 |
| qv | 16 | 1024 | 5.5 ms | 79.5 ms | 14.51x | 81.5 ms | 14.88x | 55.8 ms | 10.19x | 99.6 ms | 18.18x | 315.7 ms | 57.64x | 211.1 ms | 38.54x | 1.2e-06 |
| ghz | 20 | 20 | 3.6 ms | 23.1 ms | 6.48x | 4.7 ms | 1.32x | 46.5 ms | 13.04x | 33.6 ms | 9.42x | 18.7 ms | 5.24x | 125.6 ms | 35.22x | 1.7e-08 |
| qft | 20 | 220 | 45.4 ms | 132.4 ms | 2.92x | 141.0 ms | 3.11x | 97.8 ms | 2.15x | 176.3 ms | 3.89x | 198.6 ms | 4.38x | 2.94 s | 64.83x | 2.2e-15 |
| hea | 20 | 295 | 36.8 ms | 178.6 ms | 4.85x | 181.0 ms | 4.91x | 94.5 ms | 2.56x | 508.1 ms | 13.80x | 1.59 s | 43.28x | 1.02 s | 27.61x | 6.2e-07 |
| qv | 20 | 1600 | 72.0 ms | 365.6 ms | 5.08x | 358.0 ms | 4.97x | 152.2 ms | 2.11x | 1.84 s | 25.59x | 6.42 s | 89.18x | 4.53 s | 62.91x | 1.6e-06 |
| ghz | 24 | 24 | 46.8 ms | 440.8 ms | 9.43x | 19.6 ms | 0.42x | 602.8 ms | 12.89x | 539.4 ms | 11.54x | 387.9 ms | 8.30x | 2.18 s | 46.60x | 1.7e-08 |
| qft | 24 | 312 | 1.02 s | 2.56 s | 2.50x | 2.42 s | 2.37x | 3.64 s | 3.56x | 4.31 s | 4.21x | 4.54 s | 4.43x | 58.13 s | 56.81x | 2.6e-15 |
| hea | 24 | 355 | 649.1 ms | 1.87 s | 2.88x | 1.84 s | 2.84x | 1.75 s | 2.70x | 7.68 s | 11.82x | 24.10 s | 37.12x | 16.29 s | 25.09x | 6.6e-07 |
| qv | 24 | 2304 | 1.50 s | 5.70 s | 3.79x | 5.58 s | 3.72x | 3.82 s | 2.54x | 50.68 s | 33.74x | 130.26 s | 86.70x | 102.82 s | 68.44x | 1.8e-06 |

Reading the ratios: the comparators differ in design, and the design explains most of a gap. QuEST and Spinoza apply every gate as its own pass over the state and carry no gate fusion, so their time grows with the gate count; PRISM-Q, Aer and qsim fuse gates before execution, which pays most on the deep families (HEA, QV). Under `automatic`, Aer routes a Clifford circuit (GHZ) to its stabilizer method, as PRISM-Q routes it to its stabilizer backend, so that row compares two tableau simulations plus the dense read-out rather than two statevector runs. The GHZ rows at every size are dominated by materializing the `2^n` probability vector, not by the gates.

## Summary

Counted over circuits with at least 16 qubits. A row where the comparator did not run is not counted.

| Comparator | PRISM-Q sooner | Within band | Comparator sooner | Median ratio | Range |
| --- | --- | --- | --- | --- | --- |
| aer-statevector | 12 | 0 | 0 | 5.78x | 2.50x to 88.68x |
| aer-automatic | 11 | 0 | 1 | 4.32x | 0.42x to 42.90x |
| qsim | 12 | 0 | 0 | 4.31x | 2.11x to 86.72x |
| quest | 12 | 0 | 0 | 11.68x | 3.89x to 72.11x |
| spinoza | 12 | 0 | 0 | 36.79x | 4.27x to 89.18x |
| qip | 12 | 0 | 0 | 49.33x | 19.82x to 216.67x |

## Equivalence

Every simulator reproduced the reference probability vector (aer-statevector) to within 1e-05 total variation distance on every circuit. The tolerance separates a wrong answer, which lands near 1e-1, from rounding; the max TVD column shows the measured distance, and a value near 1e-6 is the single-precision comparator.

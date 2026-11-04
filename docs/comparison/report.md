# Benchmark Matrix: Report

Reading of the [results](./results.md). Ratio = comparator time / PRISM-Q time, 8 threads.

## Findings

- **Dense circuits:** 5.85x to 7.30x against Aer, 2.41x to 3.67x against qsim (fp32).
  PRISM-Q fuses the 24-qubit, 835-gate brickwork into 7 instructions (`fusion.json`).
- **QFT, 26 qubits:** 1.11 s PRISM-Q, 14.1 s Aer, 18.2 s qsim. Input is a random product
  state; on |0...0> the phases have no effect, so validation could not see them.
- **Clifford against Stim:** 0.66x at 100 qubits, 1.47x at 1000, 1.23x at 10,000. Stim is
  single-threaded; no 1-thread PRISM-Q row was run.
- **Shallow wide against Aer MPS:** 45.7x to 49.6x at depth 4, 9.99x to 12.5x at depth 8.
- **Disconnected components:** 107x at 40 qubits, 2.52x and 2.29x at 100 and 200, where
  PRISM-Q switches to a per-shot route. Aer automatic ran out of memory on all ten rows.
- **Sampling, 20 qubits, 1M shots:** 144 ms PRISM-Q, 2.29 s Aer, 4.32 s qsim.
- **Dispatch:** at most 1.05x the fastest valid forced backend on all seven rows.
- **Bond cap:** both MPS reach the same accuracy per cap; PRISM-Q 8.1 to 82.9 ms, Aer 1.03
  to 1.30 s.

## Correctness

79 of 81 dense and 76 of 76 sampled outputs passed. Failures, both 20-qubit QFT:
Spinoza's multithreaded path (0.86 from the reference; correct on one thread), and a
forced 64-bond MPS (truncation, 1.3e-7).

## Limits

- One desktop (i7-6700K, Windows 10); dense ratios depend on memory bandwidth.
- Five warm calls per cell; gaps under 10% are noise here.
- Aer fp32 crashed on the QFT rows on this host.
- qsim at defaults, no tuned run.
- Structured families are built for their method; they show cost, not frequency.
- No GPU, distributed or 16-thread runs.

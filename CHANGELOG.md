# Changelog

All notable changes to PRISM-Q will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-10

First release. This is a hobby project born out of wanting to see how fast a quantum circuit
simulator could get in Rust. It's been a fun ride and there's still a lot to do, but it's
still v0, there will be bugs. If you run into something, open an issue on GitHub and I'll
get to it. Contributions are also very welcome.

### Backends

Eight simulation backends, each suited to different circuit shapes:

- **Statevector** -- full 2^n state vector with AVX2/FMA/BMI2 SIMD kernels and Rayon
  parallelism at 14+ qubits. Deferred measurement normalization.
- **Stabilizer** -- Aaronson-Gottesman tableau with word-group gate batching, type-grouped
  masks, and SIMD rowmul. Scales to thousands of qubits for Clifford-only circuits.
- **Factored** -- dynamic split state that starts as n independent qubits and merges
  sub-states on demand when entangling gates connect groups.
- **Sparse** -- HashMap based, O(k) in the number of nonzero amplitudes.
- **MPS** -- matrix product state with hybrid faer/Jacobi SVD. Configurable bond dimension.
- **Product State** -- per qubit storage, O(n). Nonentangling circuits only.
- **Tensor Network** -- deferred contraction with greedy min size heuristic.
- **Compiled Sampler** -- Heisenberg picture parity tracking for Clifford circuits. O(n) per
  sample without ever building the state vector.

Automatic backend dispatch picks the right one based on circuit structure.

### Parser

OpenQASM 3.0 parser. Gate modifiers
(`ctrl @`, `inv @`, `pow(k) @`) compose and resolve at parse time. User-defined gates,
classical `if` control flow, multi-register broadcast, and a recursive descent expression
evaluator with 13 math functions.

34 gate types supported including the IBM basis set (U1/U2/U3), multi-controlled unitaries,
and native Rzz/Rxx/Ryy rotations.

### Fusion

12-pass fusion pipeline that rewrites circuits before execution:

- Self inverse cancellation (non-adjacent CX/CZ/SWAP pairs)
- Rzz synthesis and BatchRzz grouping with LUT kernels
- Single qubit fusion and commutation-aware reorder through CX/CZ
- Recancel and refuse after reorder to catch newly exposed opportunities
- MultiFused batching with per-qubit accumulation across 2q boundaries
- Controlled-phase batching with BMI2 PEXT + LUT
- Post-phase 1q re-batching

Returns `Cow<Circuit>` so there's zero cost when nothing fires.

### Noise

General noise model with quantum trajectory (Monte Carlo wavefunction) execution.
Channels: Pauli, depolarizing, amplitude damping, phase damping, two-qubit depolarizing,
and readout error. Pauli noise on Clifford circuits routes to the compiled sampler.

### Everything else

- Subsystem decomposition via union-find for independent qubit groups
- Shot based sampling with deterministic seeding (`run_shots`, `run_counts`)
- Circuit builders for QFT, random, HEA, QPE, Clifford, GHZ, and QAOA
- aarch64 NEON fallbacks for all SIMD kernels

[0.1.0]: https://github.com/AbeCoull/prism-q/releases/tag/v0.1.0

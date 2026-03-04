# Changelog

All notable changes to PRISM-Q will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-26

Initial release.

### Added

- **Seven simulation backends:**
  - Statevector (full state-vector, AVX2/FMA SIMD, Rayon parallelism at 14q+)
  - Stabilizer (Aaronson-Gottesman tableau, O(n^2), gate batching by word-group)
  - Sparse (hash-map state, O(k) for k non-zero amplitudes)
  - MPS (matrix product state, hybrid faer/Jacobi SVD, configurable bond dimension)
  - Product State (per-qubit O(n), non-entangling circuits only)
  - Tensor Network (deferred contraction, greedy min-size heuristic)
  - Factored (dynamic split-state, automatic sub-state merging)

- **OpenQASM 3.0 parser** with backward-compatible OpenQASM 2.0 support:
  - `qubit[n]`/`bit[n]` declarations (OQ3) and `qreg`/`creg` (OQ2)
  - Gate modifiers: `ctrl @`, `inv @`, `pow(k) @` (chainable)
  - User-defined gates (`gate` keyword) with parameter expressions
  - Classical `if` control flow (OQ2 `if(creg==val)` and OQ3 `if(c[i])`)
  - Multi-register broadcast (`h q;` applies to all qubits)
  - Recursive descent expression evaluator (13 math functions, `pi`/`tau`/`e`)

- **Expanded gate set:** H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg, Rx, Ry, Rz, P, CX, CY, CZ, CH, CRX, CRY, CRZ, CSX, CCX, CCZ, SWAP, CSWAP, RXX, RYY, RZZ, ECR, iSWAP, DCX, U1, U2, U3

- **7-pass fusion optimizer:**
  - Self-inverse pair cancellation (CX, CZ, SWAP)
  - Single-qubit gate fusion (matrix multiplication)
  - Commutation-aware reorder (diagonal gates through CX/CZ)
  - 2-qubit CX fusion (20q+)
  - MultiFused batching with three-tier tiled kernel (L2/L3/individual)
  - Controlled-phase batching with BMI2 PEXT + LUT
  - Post-phase 1q batching (18q+)

- **Automatic backend dispatch** (`BackendKind::Auto`):
  - Non-entangling circuits -> Product State
  - All-Clifford circuits -> Stabilizer
  - Large circuits (>28q) -> MPS
  - Default -> Statevector

- **Subsystem decomposition** for circuits with independent qubit groups (union-find analysis, per-block execution, Kronecker product merge)

- **Shot-based sampling** (`run_shots`, `run_shots_with`, `run_shots_random`) with deterministic and random seeding

- **SIMD kernels:** AVX2+FMA for 1q/2q gates, diagonal gates, complex scaling, batch phase (BMI2 PEXT), rowmul XOR. Runtime feature detection with scalar fallbacks.

- **Reusable circuit builders:** `qft_circuit`, `random_circuit`, `hea_circuit`, `phase_estimation_circuit`, `clifford_circuit`, `ghz_circuit`, `qaoa_circuit`

- **675 tests** (unit, fusion correctness, cross-backend golden, parser smoke, SIMD)

[0.1.0]: https://github.com/AbeCoull/prism-q/releases/tag/v0.1.0

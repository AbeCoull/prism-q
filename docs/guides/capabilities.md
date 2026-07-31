# Capability and Support Matrix

This page records which CPU and GPU architectures each PRISM-Q backend supports,
and where distributed execution stands. CPU backends are written in portable Rust
and run on every supported architecture; SIMD acceleration (AVX2/FMA/BMI2 on
x86-64, NEON on ARM64) is selected at runtime where a kernel exists, otherwise a
scalar path is used.

## Legend

| Mark | Meaning |
| --- | --- |
| Yes | Supported |
| SIMD | Supported with a dedicated SIMD-accelerated kernel on this architecture |
| Scalar | Runs, but without a dedicated SIMD kernel (portable fallback) |
| No | Not available for this backend |
| Planned | Not implemented yet; on the roadmap |

## Backend support by architecture

The nine CPU backends implement the `Backend` trait; the distributed statevector
backend is a tenth, feature-gated implementation covered by the Distributed
column. `Planned` marks only work the roadmap carries: a ROCm port of the
existing CUDA kernels. Backends without a CUDA kernel have nothing to port, so
their ROCm cell is `No`, and the roadmap carries no distributed execution for
any backend other than the statevector.

| Backend | x86-64 | AVX2/FMA/BMI2 | ARM64 | NEON | CUDA (NVIDIA) | ROCm (AMD) | Distributed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Statevector | Yes | SIMD | Yes | SIMD | Yes | Planned | Yes |
| Stabilizer | Yes | SIMD | Yes | SIMD | Yes | Planned | No |
| Factored Stabilizer | Yes | SIMD | Yes | SIMD | No | No | No |
| Sparse | Yes | Scalar | Yes | Scalar | No | No | No |
| MPS | Yes | SIMD | Yes | SIMD | No | No | No |
| Product State | Yes | Scalar | Yes | Scalar | No | No | No |
| Tensor Network | Yes | Scalar | Yes | Scalar | No | No | No |
| Factored | Yes | SIMD | Yes | SIMD | No | No | No |
| Density Matrix | Yes | SIMD | Yes | SIMD | No | No | No |

The Clifford+T engines below are not `Backend` implementations; they serve
probability, shot, and observable queries through their own routes (see
[Clifford+T Simulation](./clifford-t.md)).

| Engine | x86-64 | AVX2/FMA/BMI2 | ARM64 | NEON | CUDA (NVIDIA) | ROCm (AMD) | Distributed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stabilizer Rank | Yes | SIMD | Yes | SIMD | No | No | No |
| Stochastic Pauli | Yes | Scalar | Yes | Scalar | No | No | No |
| Deterministic Pauli | Yes | Scalar | Yes | Scalar | No | No | No |

Notes:

- **AVX2/FMA/BMI2** is the x86-64 SIMD tier. The active tier is chosen at runtime
  (AVX2+FMA, then FMA, then SSE2 baseline). See
  [Threading, SIMD, and Memory Layout](../architecture/threading-simd.md).
- **NEON** is the ARM64 SIMD tier. Backends marked `SIMD` carry a NEON kernel that
  mirrors the x86-64 path; the rest fall back to scalar code on ARM64.
- **CUDA** covers the optional `gpu` feature. Only the statevector and stabilizer
  paths have device kernels; every other backend runs on CPU. See the
  [GPU Backend](./gpu.md) guide.
- **Distributed** covers the optional `distributed` and `distributed-mpi` features.
  The statevector backend splits the state across MPI ranks with exact results,
  including gates, measurement, reset, and multi-shot sampling without gathering
  the dense state. Use `simulate(&circuit).distributed(context)`.

## Shot and observable queries above the dense cap

`simulate(...).shots(n)`, `.sample_counts(n)`, and `.expectation_values(...)`
answer from a dense `2^n` vector unless the backend carries its own path. The
dense route is capped by system memory (roughly 29 qubits on a 16 GiB host; see
`PRISM_MAX_SV_QUBITS`). Backends marked `Native` below answer without it and are
bounded only by their own representation.

| Backend | Shots and counts | Expectation values |
| --- | --- | --- |
| Sparse | Native, CDF over the stored amplitudes | Native, `O(k)` over the amplitude map |
| MPS | Native, sequential conditional sampling | Native, one chain contraction per observable |
| Factored | Native, one draw per sub-state | Native, product over the blocks |
| Product State | Native, one Bernoulli draw per qubit | Native, one closed-form factor per qubit |
| Statevector | Dense (streams from amplitudes, no probability vector) | Dense |
| Stabilizer, Factored Stabilizer | Compiled Clifford sampler | Sparse Pauli Dynamics, exact |
| Stochastic / Deterministic Pauli | Not applicable | Native Pauli propagation |
| Tensor Network, Density Matrix | Dense | Rejected, naming the backend |

Native sampling is deterministic from the seed alone: the same seed and shot
count reproduce the same bitstrings. It is not shot-for-shot identical to the
dense route, which consumes its randomness on a different schedule; the
distributions agree.

Backends without an observable path return `BackendUnsupported` naming
themselves, so a rejected request says which engine could not serve it rather
than blaming the route that selected it.

## Not yet supported

| Target | Status | Notes |
| --- | --- | --- |
| ROCm (AMD GPU) | Planned | No AMD device kernels; the GPU path is CUDA-only |
| Distributed GPU | Planned | No multi-node GPU execution |
| Multi-GPU | Planned | A GPU context binds a single device |
| Distributed noisy shots | Planned | Noise models are rejected on the distributed backend; trajectory execution is not lockstep across ranks |

These targets are listed so the matrix reflects the roadmap rather than hiding
the gaps.

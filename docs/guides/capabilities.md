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

| Backend | x86-64 | AVX2/FMA/BMI2 | ARM64 | NEON | CUDA (NVIDIA) | ROCm (AMD) | Distributed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Statevector | Yes | SIMD | Yes | SIMD | Yes | Planned | Planned |
| Stabilizer | Yes | SIMD | Yes | SIMD | Yes | Planned | Planned |
| Factored Stabilizer | Yes | SIMD | Yes | SIMD | No | Planned | Planned |
| Sparse | Yes | Scalar | Yes | Scalar | No | Planned | Planned |
| MPS | Yes | SIMD | Yes | SIMD | No | Planned | Planned |
| Product State | Yes | Scalar | Yes | Scalar | No | Planned | Planned |
| Tensor Network | Yes | Scalar | Yes | Scalar | No | Planned | Planned |
| Factored | Yes | SIMD | Yes | SIMD | No | Planned | Planned |
| Stabilizer Rank | Yes | SIMD | Yes | SIMD | No | Planned | Planned |
| Stochastic Pauli | Yes | Scalar | Yes | Scalar | No | Planned | Planned |
| Deterministic Pauli | Yes | Scalar | Yes | Scalar | No | Planned | Planned |

Notes:

- **AVX2/FMA/BMI2** is the x86-64 SIMD tier. The active tier is chosen at runtime
  (AVX2+FMA, then FMA, then SSE2 baseline). See
  [Threading, SIMD, and Memory Layout](../architecture/threading-simd.md).
- **NEON** is the ARM64 SIMD tier. Backends marked `SIMD` carry a NEON kernel that
  mirrors the x86-64 path; the rest fall back to scalar code on ARM64.
- **CUDA** covers the optional `gpu` feature. Only the statevector and stabilizer
  paths have device kernels; every other backend runs on CPU. See the
  [GPU Backend](./gpu.md) guide.

## Not yet supported

| Target | Status | Notes |
| --- | --- | --- |
| ROCm (AMD GPU) | Planned | No AMD device kernels; the GPU path is CUDA-only |
| Distributed CPU | Planned | No multi-node execution |
| Distributed GPU | Planned | No multi-node GPU execution |
| Multi-GPU | Planned | A GPU context binds a single device |

These targets are listed so the matrix reflects the roadmap rather than hiding
the gaps. Distributed execution is not a code path PRISM-Q exposes today.

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
  the dense state. Use `simulate(&circuit).distributed(context)`. A run can start
  from an injected amplitude vector through `.initial_state(...)`: every rank
  receives the full `2^n` vector and keeps only its own slice.

## Dense multi-qubit unitaries

`Gate::unitary` takes a caller-supplied `2^k x 2^k` matrix and lowers it into
the existing gate set wherever one of those variants carries it; only a dense
matrix on three or four qubits becomes `Gate::Unitary`, which fewer backends
execute. A lowered matrix is an ordinary gate and every backend treats it as
one, so the table below covers the dense form alone.

| Backend | Dense `Gate::Unitary` |
| --- | --- |
| Statevector (CPU) | Native, one gather-scatter pass over the state |
| Statevector (CUDA) | Declines, no device kernel |
| Factored | Native, on the merged block holding the targets |
| Density Matrix (CPU) | Native, `U rho U^dagger` on the doubled register |
| Density Matrix (CUDA) | Declines, no device kernel |
| MPS | Native, through the same swap network a multi-controlled gate uses |
| Tensor Network | Native, as one `k`-leg tensor |
| Sparse | Declines, the gate is dense by construction |
| Product State | Declines, it entangles its targets |
| Stabilizer, Factored Stabilizer | Decline, not Clifford |
| Stabilizer Rank, Stochastic Pauli, Deterministic Pauli | Decline, no Clifford+T lowering |
| Distributed statevector | Native once the targets are relabelled local; declines when they cannot be, since no exchange path carries a dense `k`-qubit gate |

Every decline is a `BackendUnsupported` naming the backend and the gate, so a
route that cannot serve the matrix says so rather than dropping it. The gate
also has no OpenQASM spelling: exporting a circuit that holds one raises
`ExportUnsupported`.

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
| Distributed Statevector | Native, rank-local CDF plus one scalar per rank | Native, rank-local sandwich plus one `Allreduce` |
| Statevector | Dense (streams from amplitudes, no probability vector) | Dense |
| Stabilizer, Factored Stabilizer | Compiled Clifford sampler | Sparse Pauli Dynamics, exact |
| Stochastic / Deterministic Pauli | Not applicable | Native Pauli propagation |
| Tensor Network | Dense | Native, one doubled-network contraction per observable |
| Density Matrix | Dense | Native, `Tr(rho P)` per observable |

Native sampling is deterministic from the seed alone: the same seed and shot
count reproduce the same bitstrings. It is not shot-for-shot identical to the
dense route, which consumes its randomness on a different schedule; the
distributions agree.

Every backend the crate ships has an observable path. The trait default is a
`BackendUnsupported` naming the backend, so one added later that omits the path
declines loudly, and the rejection says which engine could not serve the request
rather than blaming the route that selected it.

`simulate(...).marginals()` reads per-qubit Z expectations rather than a
distribution when the resolved backend has an observable path and the circuit
routes straight to it. Every backend has one, and the dense output cap does not
apply on that route; a Clifford circuit with measurements reads its marginals
off the tableau at any width, and the dense statevector reads them off its own
amplitudes rather than building the `2^n` distribution to sum. It falls back to
the dense distribution on a circuit that splits into independent blocks unless
those blocks run as product states, and under a noise model, where the mixture
is read densely and the density-matrix memory limit applies instead.

Under a noise model `run()` and `marginals()` answer from the exact mixture, so
both reject a model carrying readout error instead of serving one: readout acts
on the measurement record rather than the state, and it is indexed by classical
bit where a marginal is indexed by qubit. `shots` and `sample_counts` are the
terminals that apply it.

`simulate(...).run()` is the one terminal that needs the whole distribution, so
on the distributed backend it rejects a register past the dense cap up front
rather than running first and answering with no distribution.

## What kind of answer a result is

Every result type carries a `metadata` field describing how it was produced:
the engine automatic dispatch resolved to, whether that engine can discard state
weight, where the state lived, and the shot count for a sampled result.

| Field | Reads |
| --- | --- |
| `backend` | The engine that answered, after `Auto` routing |
| `exactness` | `Exact`, or `Approximate` with a fidelity lower bound when the engine reports one |
| `placement` | `Host` or `Device` |
| `shots` | Shots drawn, `None` for an analytic result |

`Approximate` marks the route rather than the run. An MPS at bond 256 on a
circuit that never fills a bond truncates nothing and still reports
`Approximate`, with a bound of 1.0: the variant answers whether the answer could
have been approximated, the bound answers whether it was.

The bound describes the normalized state, which is what every read returns: a
truncating MPS does not renormalize its chain, but expectation values, shot
sampling, the probability vector and the exported statevector all rescale on
read, so the probabilities sum to 1 at any bond cap. The discarded weight the
bound reports is error in that state, not weight missing from it.

`Auto` sends a circuit past the statevector cap to an MPS at a bounded bond
dimension, which is the only route those circuits have. It is taken by default
and the result says so. `simulate(...).require_exact()` rejects that route
instead, with an error naming the engine it would have used.

`Simulate::expectation_values_reported` returns the values with a standard error
per value on a route that estimates rather than evaluates. An evaluated route
reports `Exact` and no interval, so a caller distinguishes "converged" from "not
estimated" without comparing a float against zero.

## Declared limits

A limit either declines by name or has no form to write. Nothing here truncates an
input and answers anyway.

| Limit | What you get |
| --- | --- |
| Kraus sets reach two qubits | No error. `Kraus2q` is the widest set in the channel enum and three or more qubits has no variant, so a wider set cannot be written. A wider interaction is modelled by composing the channels the enum does carry, or by the density matrix directly |
| Readout error acts on the measurement record | `InvalidParameter` from `run`, `marginals`, `expectation_values` and `observable_expectation`, naming the terminals that do apply it. A marginal is indexed by qubit and readout by classical bit, so there is nothing to apply it to |
| `EXP_VAL` in a QEC program must be terminal and live | `InvalidParameter` naming the op that followed it, or the qubit measured since its last reset |
| `EXP_VAL` on the reference QEC runner reaches 64 qubits | `IncompatibleBackend` naming the cap and the width the program needs. The Pauli-mask reduction is one word per shot; statevector memory binds long before this does |
| Analytical conditional expectation reaches 12 postselection rows | `BackendUnsupported` naming the count and the cap, and pointing at the reference runner, since the expansion is `2^rows` Pauli evaluations |
| Save points are returned by `run` alone | `IncompatibleBackend` naming the terminal and the number of save points. A shot loop runs the circuit many times and a marginal reduces it, so neither has a place to put one record per point |
| A save needs a route that holds a state | `IncompatibleBackend` naming the save and the route. The compiled samplers reorder measurements, the stabilizer-rank engine carries a weighted sum of branches, and the Pauli-propagation engines carry no state vector at all |
| A save has no OpenQASM spelling | `ExportUnsupported` naming the save and its label. The subset has no save syntax to export into |

Backend width and memory caps are separate and live with the terminals that raise
them; see [Shot and observable queries above the dense cap](#shot-and-observable-queries-above-the-dense-cap).

## Not yet supported

| Target | Status | Notes |
| --- | --- | --- |
| ROCm (AMD GPU) | Planned | No AMD device kernels; the GPU path is CUDA-only |
| Distributed GPU | Planned | No multi-node GPU execution |
| Multi-GPU | Planned | A GPU context binds a single device; sharding one statevector across devices also needs peer access between them to stay ahead of the host path |
| Distributed noisy shots | Planned | Noise models are rejected on the distributed backend; trajectory execution is not lockstep across ranks |

These targets are listed so the matrix reflects the roadmap rather than hiding
the gaps.

## Compatibility

The minimum supported Rust version is 1.87.0. It is pinned in three places that are
updated together: `rust-version` in `Cargo.toml`, `msrv` in `clippy.toml`, and the
`CI_MSRV` job that builds against exactly that toolchain. Raising it is a minor bump
while the crate is below 1.0.

Four feature flags are part of the surface: `parallel`, on by default, and `gpu`,
`distributed` and `distributed-mpi`, each off. `bench-fast` and `bench-internal` are
not: they exist to shape benchmark runs, they gate items no caller should reach for,
and they may change or disappear in any release.

The crate and the Python wheel carry one version. A release bumps the crate, then
writes that same version into the bindings manifest and tags the wheel `py-v<version>`,
so a wheel and a crate that share a number were built from one commit. Below 1.0 the
minor position is the compatibility boundary, because Cargo reads `0.32.0` as
`^0.32.0`: `0.32` to `0.33` already signals a break to every downstream caret
requirement, so the release tooling resolves a breaking change to a minor bump rather
than to 1.0.0.

What a version promises is the surface tabulated in
[API surface](../architecture/api-surface.md) and the Python package. Which backend
`Auto` picks for a given circuit, where the fusion thresholds sit, how fast a kernel
runs, and what the benchmark rows are named all move underneath that promise without a
bump, because they are implementation rather than interface. Public enums are
`#[non_exhaustive]` apart from a handful that mirror a closed set, so a new variant is
additive and a `match` on one outside the crate keeps a wildcard arm. That page names
the exceptions.

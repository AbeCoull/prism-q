# Capability and Support Matrix

Which CPU and GPU architectures each backend supports, what each answers past the dense
memory cap, and which limits decline by name. CPU backends are portable Rust. A SIMD
kernel (AVX2/FMA/BMI2 on x86-64, NEON on ARM64) is selected at runtime where one exists,
and a scalar path runs otherwise.

## Legend

| Mark | Meaning |
| --- | --- |
| Yes | Supported |
| SIMD | Supported with a dedicated SIMD-accelerated kernel on this architecture |
| Scalar | Runs on the portable path, with no dedicated SIMD kernel |
| No | Not available for this backend |
| Roadmap | Not implemented yet; the [roadmap](../roadmap.md) carries it and says what it waits on |

## Backend support by architecture

The nine CPU backends implement the `Backend` trait; the feature-gated distributed
statevector is a tenth, covered by the Distributed column. `Roadmap` marks only work the
roadmap carries: here a ROCm port of the existing CUDA kernels, and under
[Not yet supported](#not-yet-supported) multi-GPU and distributed GPU execution. Backends
without a CUDA kernel have nothing to port, so their ROCm cell is `No`, and the roadmap
carries no distributed execution for any backend but the statevector.

| Backend | x86-64 | AVX2/FMA/BMI2 | ARM64 | NEON | CUDA (NVIDIA) | ROCm (AMD) | Distributed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Statevector | Yes | SIMD | Yes | SIMD | Yes | Roadmap | Yes |
| Stabilizer | Yes | SIMD | Yes | SIMD | Yes | Roadmap | No |
| Factored Stabilizer | Yes | SIMD | Yes | SIMD | No | No | No |
| Sparse | Yes | Scalar | Yes | Scalar | No | No | No |
| MPS | Yes | SIMD | Yes | SIMD | No | No | No |
| Product State | Yes | Scalar | Yes | Scalar | No | No | No |
| Tensor Network | Yes | Scalar | Yes | Scalar | No | No | No |
| Factored | Yes | SIMD | Yes | SIMD | No | No | No |
| Density Matrix | Yes | SIMD | Yes | SIMD | Yes | No | No |

The four Clifford+T engines are not `Backend` implementations; they serve probability,
shot and observable queries through their own routes (see
[Clifford+T Simulation](./clifford-t.md)).

| Engine | x86-64 | AVX2/FMA/BMI2 | ARM64 | NEON | CUDA (NVIDIA) | ROCm (AMD) | Distributed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stabilizer Rank | Yes | SIMD | Yes | SIMD | No | No | No |
| Stochastic Pauli | Yes | Scalar | Yes | Scalar | No | No | No |
| Deterministic Pauli | Yes | Scalar | Yes | Scalar | No | No | No |
| Pauli Path | Yes | Scalar | Yes | Scalar | No | No | No |

Notes:

- **AVX2/FMA/BMI2**: the x86-64 tier is chosen at runtime, AVX2+FMA, then FMA, then the
  SSE2 baseline, with an AVX-512 tier first when `PRISM_AVX512=1` is set on a CPU that
  has it. See [Threading, SIMD, and Memory Layout](../architecture/threading-simd.md).
- **NEON**: backends marked `SIMD` carry a NEON kernel that mirrors the x86-64 path; the
  rest run scalar code on ARM64.
- **CUDA**: the optional `gpu` feature. The statevector, stabilizer and density matrix
  have device kernels, and every other backend runs on CPU. The density matrix runs on
  the device only when selected explicitly (`BackendKind::DensityMatrixGpu`). See the
  [GPU Backend](./gpu.md) guide.
- **Distributed**: the optional `distributed` and `distributed-mpi` features. The
  statevector splits the state across MPI ranks with exact results, gates, measurement,
  reset and multi-shot sampling included, without gathering the dense state. Noisy shots
  run every trajectory channel in lockstep, one shot at a time on every rank. Use
  `simulate(&circuit).distributed(context)`. `.initial_state(...)` starts a run from an
  injected amplitude vector: every rank receives the full `2^n` vector and keeps only its
  own slice.

## Dense multi-qubit unitaries

`Gate::unitary` takes a caller-supplied `2^k x 2^k` matrix and lowers it into the existing
gate set wherever a variant carries it. Only a dense matrix on three or four qubits
becomes `Gate::Unitary`, which fewer backends execute; a lowered matrix is an ordinary
gate, so the table covers the dense form alone.

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
| Stabilizer Rank, Stochastic Pauli, Deterministic Pauli, Pauli Path | Decline, no Clifford+T lowering |
| Distributed statevector | Native once the targets are relabelled local; declines when they cannot be, since no exchange path carries a dense `k`-qubit gate |

Every decline is a `BackendUnsupported` naming the backend and the gate. The gate has no
OpenQASM spelling either: exporting a circuit that holds one raises `ExportUnsupported`.

## Shot and observable queries above the dense cap

`simulate(...).shots(n)`, `.sample_counts(n)` and `.expectation_values(...)` answer from a
dense `2^n` vector unless the backend has its own path. System memory caps the dense route
(roughly 29 qubits on a 16 GiB host; see `Limits::max_statevector_qubits` or
`PRISM_MAX_SV_QUBITS`). Backends marked `Native` answer without it and are bounded only by
their own representation. Which engine `Auto` picks for each terminal is in
[routes the tree does not show](../architecture/engine.md#routes-the-tree-does-not-show).

| Backend | Shots and counts | Expectation values |
| --- | --- | --- |
| Sparse | Native, CDF over the stored amplitudes | Native, `O(k)` over the amplitude map |
| MPS | Native, sequential conditional sampling | Native, one contraction over each observable's span between shared identity environments |
| Factored | Native, one draw per sub-state | Native, product over the blocks |
| Product State | Native, one Bernoulli draw per qubit | Native, one closed-form factor per qubit |
| Distributed Statevector | Native, a CDF on each rank plus one scalar per rank | Native, a sandwich on each rank plus one `Allreduce` |
| Statevector | Dense (streams from amplitudes, no probability vector) | Dense |
| Stabilizer, Factored Stabilizer | Compiled Clifford sampler | Sparse Pauli Dynamics, exact |
| Stochastic, Deterministic Pauli, Pauli Path | Not applicable | Native Pauli propagation |
| Tensor Network | Dense below `PRISM_MAX_PROB_QUBITS`, then native conditional sampling, one doubled-network contraction per qubit per shot | Native, one doubled-network contraction per observable over its backward light cone |
| Density Matrix | Dense | Native, `Tr(rho P)` per observable |

Native sampling is deterministic from the seed alone: the same seed and shot count give
the same bitstrings at any thread count. It does not match the dense route shot for shot,
since that route consumes its randomness on a different schedule; the distributions agree.

Every backend the crate ships has an observable path. One added later without it declines
with `BackendUnsupported` naming the engine, not the route that selected it.

`simulate(...).marginals()` escapes the dense output cap when the circuit routes straight
to an observable path, so a Clifford circuit with measurements reads its marginals off the
tableau at any width. The cases that fall back to the dense distribution are listed with
the [routing rules](../architecture/engine.md#routes-the-tree-does-not-show).

Under a noise model `run()` and `marginals()` answer from the exact mixture, so both
reject a model that carries readout error: readout acts on the measurement record, not
the state, and is indexed by classical bit where a marginal is indexed by qubit. `shots`
and `sample_counts` apply it.

Expectation terminals need a unitary circuit, except on `BackendKind::DensityMatrix` and
its device sibling, which evolve a reset as a channel, with or without a noise model. The
run honours `.limits()`, `.require_exact()` and a start state, and reports the same
metadata as the circuit without its resets. A measurement, conditional or `Region` is
still rejected there.

`simulate(...).run()` is the one terminal that needs the whole distribution, so on the
distributed backend it rejects a register past the dense cap before running.

## What kind of answer a result is

Every result type carries a `metadata` field describing how it was produced:

| Field | Reads |
| --- | --- |
| `backend` | The engine that answered, after `Auto` routing |
| `exactness` | `Exact`, or `Approximate` with a fidelity lower bound when the engine reports one |
| `placement` | `Host` or `Device` |
| `shots` | Shots drawn, `None` for an analytic result |
| `bond` | Peak bond dimension against the cap, `None` unless the MPS ran |
| `blocks` | The engine of each independent block when `backend` is `Decomposed`, ordered by each block's lowest qubit; empty otherwise |

`Approximate` marks the route, not the run. An MPS at bond 256 on a circuit that never
fills a bond truncates nothing and still reports `Approximate`, with a bound of 1.0: the
variant says whether the answer could have been approximated, the bound whether it was.

For an MPS run, `bond` carries `peak`, the widest bond any cut kept over the run, and
`cap`, the configured maximum; `saturated()` reads `peak >= cap`. A run whose peak stayed
under the cap truncated nothing on the cap's account.

The bound describes the normalized state, which every read returns. A truncating MPS does
not renormalize its chain, but expectation values, shot sampling, the probability vector
and the exported statevector all rescale on read, so probabilities sum to 1 at any bond
cap. The discarded weight the bound reports is error in that state, not weight missing
from it. How the MPS computes the bound, and how a decomposed run merges its blocks'
claims, is on
[Backends](../architecture/backends.md#what-a-backend-reports-about-its-own-result).

Two tensor-network kinds decide what happens at the tensor-network memory cap. Under
`BackendKind::TensorNetwork` a contraction over the cap is sliced: legs are fixed, the
network is contracted once per assignment and summed, and the answer stays `Exact`. Every
slice repeats the whole contraction, so a query over the cap can take many times longer
than one that fits. `BackendKind::TensorNetworkBounded { tolerance }` first factors an
intermediate and keeps the new bond only as far as discarding `tolerance` of that cut's
squared weight allows, then slices whatever is still over the cap. It reports
`Approximate`, with a bound of 1 minus the summed per-cut discarded weights, and
`require_exact()` rejects it by name.

Past the statevector cap `Auto` takes an MPS at a bounded bond dimension, unless an
expectation or marginals terminal finds the exact tensor route cheaper. The MPS route is
taken by default and the result says so, since refusing it would leave an oversize circuit
that is not sparse with no route at all. `simulate(...).require_exact()` rejects it with
an error naming the engine it would have used; the
[engine page](../architecture/engine.md#result-provenance) says when that check runs.

`Simulate::expectation_values_reported` returns a standard error per value on a route that
estimates rather than evaluates. An evaluated route reports `Exact` and no interval, so a
caller can tell "converged" from "not estimated" without comparing a float against zero.

## Declared limits

A limit either declines by name or has no form to write; nothing here truncates an input
and answers anyway.

| Limit | What it returns |
| --- | --- |
| Kraus sets reach two qubits | No error. `Kraus2q` is the widest set in the channel enum and three or more qubits has no variant, so a wider set cannot be written. Model a wider interaction by composing the channels the enum carries, or on the density matrix directly |
| Readout error acts on the measurement record | `InvalidParameter` from `run`, `marginals`, `expectation_values` and `observable_expectation`, naming the terminals that do apply it. A marginal is indexed by qubit and readout by classical bit, so there is nothing to apply it to |
| `EXP_VAL` in a QEC program must be terminal and live | `InvalidParameter` naming the op that followed it, or the qubit measured since its last reset |
| `EXP_VAL` on the reference QEC runner reaches 64 qubits | `IncompatibleBackend` naming the cap and the width the program needs. The Pauli-mask reduction is one word per shot; statevector memory binds long before this does |
| Analytical conditional expectation reaches 12 postselection rows | `BackendUnsupported` naming the count and the cap, and pointing at the reference runner, since the expansion is `2^rows` Pauli evaluations |
| A start mixture runs on the density matrix | `IncompatibleBackend` from `initial_density_matrix` on every other backend, `Auto` included, since automatic dispatch never selects the density matrix. The `4^n` buffer is checked before any allocation, Hermitian to 1e-12 per entry and unit trace to 1e-9, with `InvalidParameter` naming the failed check. Positive semidefiniteness is not checked, so a mixture with a negative eigenvalue evolves without complaint |
| Save points are returned by `run` alone | `IncompatibleBackend` naming the terminal and the number of save points. A shot loop runs the circuit many times and a marginal reduces it, so neither has a place to put one record per point |
| A save needs a route that holds a state | `IncompatibleBackend` naming the save and the route. The compiled samplers reorder measurements, the stabilizer-rank engine carries a weighted sum of branches, and the Pauli-propagation engines carry no state vector at all |
| A save has no OpenQASM spelling | `ExportUnsupported` naming the save and its label. The subset has no save syntax to export into |
| Noise inside a guarded body needs one state per shot | `IncompatibleBackend` from the density matrix (any guard), the compiled noisy sampler (`compiled noisy sampler`) and `ErrorChainComplex` (`error chain complex`). Trajectories and the stabilizer replay fire a body's events on the shots that take the branch, and so does the noisy shot route on a Clifford circuit whose guards it rewrites to controlled Paulis |
| The error chain complex takes no guarded gates | `IncompatibleBackend` from `ErrorChainComplex` (`error chain complex`) for a circuit with a conditional. Whether the gate runs is decided per shot, and an error on its guard bit can decide it, so which measurements an earlier error reaches is not fixed. `compile_measurements` and `compile_noisy` decline conditionals for the same reason; the shot routes rewrite a guarded Pauli on a bit, a parity, or a register comparison reading one measured bit into Paulis controlled by the measured qubits and sample it compiled |
| The error chain complex needs named Clifford gates and single-qubit Pauli channels | `IncompatibleBackend` from `ErrorChainComplex` (`error chain complex`) for a gate outside the named Clifford set, which can turn one Pauli error into a sum of them (a rotation at a Clifford angle is not recognized), or for a damping, Kraus or two-qubit channel, which has no single error column. `noisy_marginals_analytical` returns the same errors |
| Schedule rules need a static schedule | `IncompatibleBackend` from `NoiseBuilder::build` when an idle, scheduled idle, detuning or before-measurement rule meets a circuit holding a region, whose duration depends on the branch |

Backend width and memory caps live with the terminals that raise them; see
[Shot and observable queries above the dense cap](#shot-and-observable-queries-above-the-dense-cap).
A cap overrun raises `PrismError::ResourceLimit`, whose `knob` names the `Limits` field
that raises the cap and whose `limit_source` says whether detection, the environment, the
process default or the request set it.

## Not yet supported

| Target | Status | Notes |
| --- | --- | --- |
| ROCm (AMD GPU) | Roadmap, needs hardware | No AMD device kernels; the GPU path is CUDA-only |
| Distributed GPU | Roadmap, research | No multi-node GPU execution; it follows multi-GPU on one host |
| Multi-GPU | Roadmap, needs hardware | A GPU context binds a single device; sharding one statevector across devices also needs peer access between them to stay ahead of the host path |

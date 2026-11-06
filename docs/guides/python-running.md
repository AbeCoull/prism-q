# Running Simulations

## Running a simulation

`simulate(circuit)` returns a `Simulation`: configure it with `.seed()`, `.backend()`,
and `.noise()`, then call a terminal method. The default seed is 42. A seed reproduces a
result within a release; [Seeded output](compatibility.md#seeded-output) lists the
releases that changed what a seed draws.

```python
from prism_q import BackendKind, simulate

sim = simulate(circuit).seed(7).backend(BackendKind.statevector())
outcome = sim.run()
```

| Terminal | Returns | Honors `.noise()` | Honors `.initial_state()` |
|----------|---------|-------------------|---------------------------|
| `run()` | `RunOutcome`: classical bits and the full probability array | density matrix only | yes |
| `shots(n)` | `ShotsResult`: per-shot measurement records | yes | without `.noise()` |
| `sample_counts(n)` | `CountsResult`: frequency histogram | yes | without `.noise()` |
| `marginals()` | `MarginalsResult`: per-qubit `(p0, p1)` with `.metadata`; indexes and iterates as the list of pairs | density matrix only | yes |
| `state_vector()` | `complex128` amplitudes | no | yes |
| `probabilities_of(qubits)` | `float64` joint distribution over a subset, `qubits[0]` the lowest bit | density matrix only | yes |
| `reduced_density_matrix(qubits)` | `ReducedDensityMatrix`: `.matrix` over a subset with `qubits[0]` the lowest bit, `.purity`, `.metadata` | density matrix only | yes |
| `entanglement_entropy(subsystem)` | `EntropyResult`: von Neumann entropy of the cut in nats, with `.schmidt_values` | density matrix only | yes |
| `expectation(obs)` | `ExpectationResult` for a list of observables: `.values` per observable, `.std_errors` from a sampling route, `.metadata` naming the backend; `ObservableExpectation` for a `PauliObservable`: the weighted mean with its variance, group variances and standard error | density matrix only | yes |
| `observable_variance(obs)` | `ObservableVariance`: `<H^2> - <H>^2` beside the mean | density matrix only | yes |
| `expectation_gradient(h, params)` | `ExpectationGradient`: `.value` and `.gradient` via the adjoint method | no | no |
| `expectation_gradient_shift(h, params)` | `ExpectationGradient` via the parameter-shift rule | density matrix or `pauli_path()` | yes |
| `overlap(other)` | `OverlapResult`: `.fidelity`, `\|<a\|b>\|^2` against a second `Simulation`, with `.left` and `.right` metadata | no | yes |

[Advanced State Handling](python-state.md#entropy-overlap-and-reduced-states) has
examples of the entropy, overlap and reduced state terminals. Read
`expectation().metadata` when the route matters: under `auto()` a wide shallow circuit
may be answered by tensor contraction instead of the state vector, and only the metadata
says which ran. `expectation_gradient_shift()` gives the same gradient as
`expectation_gradient()` at two extra circuit runs per parameter, and is the only route
on a backend with no adjoint pass or with a noise model attached; under noise each
evaluation reads the exact mixture, and the shift stays exact because the channels do
not depend on the shifted angle. `overlap()` takes a second `Simulation`, so each side
keeps its own backend, seed and start state; both circuits must have the same width and
be unitary.

`shots()` and `sample_counts()` average trajectories on any backend holding a per-shot
pure state. Rows marked "density matrix only" read the exact mixed state and need
`.backend(BackendKind.density_matrix())`, which auto dispatch never selects. The mixture
is evolved once and every terminal reads it, so probabilities are seed independent and
observables carry no sampling error. With `.noise()` attached, that route rejects
mid-circuit measurement and classical conditioning, since the mixture holds every
measurement branch at once.
Without a noise model the density matrix backend runs them as the statevector does: each
measurement samples and collapses, so `run()` returns one branch's probabilities and
`sample_counts()` draws a fresh branch per shot.

Terminals that cannot honor a model raise `PrismError` naming the reason.
`state_vector()` honors `.backend(...)` and declines on a backend holding no pure state
rather than substituting one.

`ShotsResult` and `CountsResult` both expose `counts()`, a dict keyed by bitstring.
`ShotsResult.shots` is a `bool` array of shape `(shots, bits)`; call `.tolist()` for
nested lists.

### Amazon Braket programs

`parse_braket(source)` reads Braket's dialect and returns a `BraketProgram`: its
`circuit`, `parameters`, `noise` model, and the `results` its `#pragma braket result`
lines request. `evaluate()` returns one dict per request with `type` and `value`, in
Braket's basis order, where qubit 0 is the most significant bit:

```python
from prism_q import parse_braket

source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result probability all
#pragma braket result expectation z(q[0]) @ z(q[1])
"""
program = parse_braket(source)
exact = program.evaluate()
sampled = program.evaluate(shots=1000)
```

At the default `shots=0` every value is exact and `sample` is declined. Above zero the
values come from a measurement record, and `state_vector`, `amplitude` and
`density_matrix` are declined. The [Amazon Braket guide](braket.md) covers the pragmas.

### Memory caps

A run over a cap raises `PrismError` with kind `"resource_limit"`, whose attributes
[Errors](python-names.md#errors) lists. Caps derive from a memory budget of half of
physical memory. `Limits` sets the budget or any single cap, per request with
`Simulation.limits()` or for the rest of the process with `set_default_limits()`. A
request value beats the process default, which beats the `PRISM_MAX_*` environment
variables; a new default applies from the next call:

```python
from prism_q import Limits, set_default_limits

tenant = Limits(memory_budget_bytes=4 << 30)
outcome = simulate(circuits.ghz(20)).limits(tenant).seed(42).run()

set_default_limits(Limits(max_statevector_qubits=26))
set_default_limits(Limits())  # back to the environment and detection
```

Every `Limits` field is a keyword argument that defaults to `None`, meaning the layer
below decides: `memory_budget_bytes`; the qubit caps `max_statevector_qubits`,
`max_density_matrix_qubits`, `max_probability_qubits` (a dense probability array),
`max_export_qubits` (a dense state vector export), `max_sparse_qubits`,
`max_factored_merge_qubits`, `max_stabilizer_cluster_qubits`,
`max_mps_workspace_qubits` and `max_tn_peak_qubits`; `max_tn_slices` (default 1024),
the slices a tensor network contraction may sum over; and `auto_mps_bond_dim` (default
256), the bond cap of the MPS that automatic dispatch falls back to.
`default_limits()` returns the process default.

### Distributions too wide to write down

A circuit whose qubits fall into independent groups is answered per group, and
`RunOutcome` builds the dense vector only when `probabilities` is read. Fifteen
independent Bell pairs span 30 qubits: 8 GB dense, or 15 arrays of four entries.

```python
outcome = simulate(circuits.independent_bell_pairs(15)).seed(42).run()

outcome.num_basis_states            # 2 ** 30, and nothing was materialized
for qubits, probs in outcome.probabilities_factored():
    print(qubits, probs)            # [0, 1] [0.5 0. 0. 0.5], ...
```

Each block is `(qubits, probs)` with `qubits` ascending and `probs` indexed by those
qubits packed in order, `qubits[0]` least significant. A basis state's probability is
the product of one entry per block, which is what `probabilities` computes.

`probabilities_factored()` returns `None` for a dense distribution, the common case: the
decomposed route needs the widest group several qubits narrower than the register, so
two Bell pairs stay dense. `num_basis_states` is `None` when the backend exposed no
distribution.

## Result metadata

`RunOutcome`, `ShotsResult`, and `CountsResult` carry a `metadata` object describing the
route that produced them.

```python
circuit = CircuitBuilder(2).h(0).rx(0.3, 1).cx(0, 1).build()
result = simulate(circuit).seed(42).run()
print(result.metadata.backend)               # 'Statevector'
print(result.metadata.engine)                # None unless samplers share the backend
print(result.metadata.is_exact)              # True
print(result.metadata.fidelity_lower_bound)  # None when exact
print(result.metadata.placement)             # 'host' or 'device'
print(result.metadata.bond)                  # None unless the MPS ran
print(result.metadata.blocks)                # per-block engines when 'Decomposed'
```

`is_exact` is False when the engine can discard state weight or estimate by sampling. It
marks the route, not the run: an MPS whose bond dimension the circuit never fills
reports `is_exact == False` with `fidelity_lower_bound == 1.0`. The flag says the answer
could have been approximated; the bound says whether it was.

An MPS run also reports `bond`, a `BondReport`: `peak` (the widest bond any cut kept),
`cap` (the configured maximum), and `saturated` (`peak >= cap`). A run whose peak stayed
under the cap truncated nothing on the cap's account, whatever the exactness label says.

Automatic dispatch sends a circuit past the statevector cap to the sparse map when its
state never holds more than 128 basis states, and to an MPS with a bounded bond otherwise.
The MPS route is the default and the result says so; `.require_exact()` rejects it with a
`PrismError` naming the engine it would have used.

```python
big_circuit = circuits.hardware_efficient_ansatz(40, 2)
simulate(big_circuit).seed(42).require_exact().marginals()  # raises
```

## Inspecting a terminal before it runs

`Simulation.plan(Query)` reports the route a terminal would take and leaves the request
ready to run. The query can change the route: an expectation may use an exact scalar
tensor contraction where a run over the full state would pick a bounded MPS.

```python
from prism_q import CircuitBuilder, Query, simulate

circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1).build()
request = simulate(circuit).seed(42)
plan = request.plan(Query.counts(1000))
assert plan.would_fail is None
result = request.sample_counts(1000)
assert result.metadata.reason == plan.rule
```

`Plan.backend`, `engine`, `placement`, and `rule` describe the selected route.
`RunMetadata.reason` gives its snake_case decision name after execution. A bounded SPD
attempt or soft GPU initialization can take a runtime fallback, listed in
`Plan.fallbacks`; the result names the fallback when it happens. A fallback's own
`would_fail` says nothing about the primary route.

`Plan.exactness` is `"exact"`, `"approximate"`, or `"estimated"`. `bound_reported` says
whether an approximate route reports a bound, and `samples` gives an estimator's sample
count. `require_exact()` rejects an approximate fallback only if execution needs it.

`would_fail` is a `PrismError` instance for a predictable resource or capability
failure; malformed query inputs raise `PrismError` directly. `alternatives` lists
explicit backend selections rejected by the validation execution uses, without ranking
accepted backends by speed.

`peak_memory` holds `bytes` and `kind`. Known bounds cover the state and terminal buffer
payload, not circuit or fusion storage, allocator overhead, or process memory. Other
routes return `{"bytes": None, "kind": "unknown"}`; state size alone is not reported as
a peak bound. Inspection may run bounded sparse support or tensor contraction planning
probes but does not execute the terminal.

Query constructors are `run()`, `shots(n)`, `counts(n)`, `marginals()`,
`expectation(observables)`, `observable(hamiltonian)`, `state_vector()`,
`reduced_density_matrix(qubits)`, and `entropy(qubits)`. The gradient constructors,
`gradient(hamiltonian, parameters)` and `gradient_shift(hamiltonian, parameters)`,
take the same Hamiltonian and `(instruction, slot)` links as the execution terminals.
A shift plan's `evaluations` lists its baseline, then the plus and minus evaluations in
link order. The top-level backend names the baseline; child reports keep any different
route chosen for another evaluation, including prepared backend reuse.

## Selecting a backend

`BackendKind.auto()`, the default, picks a backend from circuit structure; an explicit
one overrides it.

| Constructor | Notes |
|-------------|-------|
| `auto()` | Structure-driven dispatch. See [Choosing a Backend](../getting-started/choosing-a-backend.md). |
| `statevector()` | Dense amplitudes, the general path |
| `stabilizer()`, `factored_stabilizer()` | Clifford-only circuits |
| `stabilizer_rank()` | Clifford+T, requires at least one T gate |
| `sparse()` | Sparse states, for circuits that stay concentrated |
| `product_state()` | No entangling gates |
| `factored()` | Partially independent subsystems |
| `tensor_network()` | Contraction over a network |
| `tensor_network_bounded(tolerance)` | The tensor network with bond truncation, discarding at most `tolerance` of each cut's squared weight; approximate |
| `mps(max_bond_dim=256)` | Approximate, truncates at the bond dimension |
| `density_matrix()` | Exact mixed states, never chosen by `auto()` |
| `stochastic_pauli(num_samples=1000)` | Sampled Pauli propagation |
| `deterministic_pauli(epsilon=0.0, max_terms=65536)` | Truncated Pauli propagation |
| `deterministic_pauli_budget(max_terms=65536)` | Pauli propagation holding a fixed term count |
| `pauli_path(epsilon=1e-8, max_terms=65536)` | Heisenberg Pauli propagation through the attached noise model, for expectation values only; never chosen by `auto()`, and `max_terms=0` keeps every term |
| `auto_gpu(context)`, `statevector_gpu(context)`, `stabilizer_gpu(context)`, `density_matrix_gpu(context)` | CUDA device paths, see [GPU backends](#gpu-backends) |

The density matrix backend stores `4^n` amplitudes, so its qubit ceiling is about half
the statevector cap; exceeding it raises `PrismError` naming the cap.
`statevector_distributed(context)` reaches the sharded backend
([Distributed backend](#distributed-backend)).

## GPU backends

The GPU constructors take a `GpuContext`, an opaque handle to one CUDA device and its
compiled kernels. Construction compiles the kernel module, so build one and share it
across simulations.

```python,ignore
from prism_q import BackendKind, GpuContext, circuits, simulate

context = GpuContext(0)
outcome = simulate(circuits.qft(16)).backend(BackendKind.auto_gpu(context)).seed(42).run()
print(outcome.probabilities)
```

A missing or unusable device raises `PrismError` from `GpuContext(device_id)`; there is
no fallback at that point. After construction, routing is soft, as in the Rust API:
`statevector_gpu` keeps circuits below its crossover on the host, `auto_gpu` routes each
block on its own, and a block whose device allocation fails runs on the host instead of
erroring, so a host result is not a failure signal. `density_matrix_gpu(context)` is the
exception: it has no crossover and no host fallback, `auto_gpu` never selects it, and a
width whose `4^n` buffer exceeds free device memory raises before allocating.
`context.device_name` reports the device's product name.

The macOS wheel and a default source build carry no CUDA support. There the constructors
still exist and `GpuContext(...)` raises `PrismError` naming the missing build feature,
not an `AttributeError`. The [GPU guide](gpu.md) covers driver and NVRTC loading, the
kernel cache, and the crossover defaults with the environment variables that move them.

`gpu_info()` opens a device and reports its name, or the reason it cannot be used:

```python
info = prism_q.gpu_info()
print(info)   # GpuInfo(available=True, device="NVIDIA GeForce GTX 1080 Ti")
```

Two cheaper predicates stop short of compiling the kernels:

```python
GpuContext.is_supported()   # was this build compiled with CUDA support
GpuContext.is_available()   # ... and does the driver open a device
```

## Distributed backend

The distributed statevector shards the dense state across MPI ranks. Python reaches it
through a `DistributedContext`, which attaches to an MPI that is already running. The
extension never calls `MPI_Init` or `MPI_Finalize`: mpi4py does both, at import and at
interpreter exit, and a handle that finalized MPI when dropped would break every later
MPI call in the process.

```python,ignore
from mpi4py import MPI  # starts MPI; import before touching the context
from prism_q import BackendKind, DistributedContext, circuits, simulate

context = DistributedContext()
outcome = (
    simulate(circuits.qft(24))
    .backend(BackendKind.statevector_distributed(context))
    .seed(42)
    .run()
)
print(context.rank, context.size, outcome.probabilities[:4])
```

Run it with `mpiexec -n 4 python script.py`. The rank count must be a power of two, and
every rank needs enough local qubits after the shard bits are taken; the
[distributed guide](distributed.md) has the memory layout, the tuning variables and what
runs sharded.

The contract is SPMD, and enforced. Four ranks are four interpreters running the same
source, and all of them enter every collective inside the backend, so a script that
branches before the call

```python,ignore
if comm.rank == 0:
    result = simulate(circuit).backend(...).run()   # deadlocks
```

hangs the job: rank 0 blocks in a collective the others never enter. Every rank calls
`run` with the same circuit and seed; only the handling of the returned value may branch
on rank. Two checks, one collective each at run entry, turn the common violations into
errors instead of hangs: ranks disagreeing about the register shape, the seed, or the
tuning knobs, and ranks handed different circuits.

Two cases raise on purpose.

A world of one rank raises. MPI-2 and later let a singleton `MPI_Init` succeed, so a
script launched without `mpiexec` would get a correct answer at the speed of one host,
with no sign that nothing was distributed.
`BackendKind.statevector_distributed(context, allow_single_rank=True)` allows it.

Constructing the context without MPI running raises instead of starting MPI, so a
forgotten `from mpi4py import MPI` is reported where it happened.

The thread level mpi4py negotiated must be at least `MPI_THREAD_FUNNELED`: Rayon workers
run beside MPI in the same process, and every MPI call stays on the thread that
constructed the context. mpi4py requests `MPI_THREAD_MULTIPLE` by default; lowering it
(`mpi4py.rc.thread_level = 'single'` or `mpi4py.rc.threads = False`) makes the
constructor raise. Below `MPI_THREAD_MULTIPLE`, a simulation call from a Python thread
other than the constructing one raises before it reaches MPI, and at
`MPI_THREAD_FUNNELED` the constructor itself raises off the thread that imported mpi4py.

The published wheels have no MPI support: `mpi-sys` runs bindgen against a system MPI at
build time, and the extension must link the same MPI implementation and ABI as mpi4py
and the launcher. Two implementations in one process corrupt state silently, so this is
a source build:

```bash
maturin develop --manifest-path bindings/python/Cargo.toml --features distributed-mpi
```

`DistributedContext.is_supported()` reports MPI support as `GpuContext.is_supported()`
does CUDA.

Sub-communicators, per-rank device placement, and distributed paths other than the
statevector backend are out of scope: the context is the world communicator.

## Running many small circuits

`run_batch` takes a list and crosses into Rust once, reusing one backend across circuits
of the same width that draw no randomness.

```python
from prism_q import circuits, run_batch

batch = [circuits.hardware_efficient_ansatz(4, 2, seed=s) for s in range(3)]
outcomes = run_batch(batch, seed=42)
```

Results match running each circuit alone with the same seed; a failing batch raises the
first failure in list order. Up to 16 qubits the circuits split across cores, one per
core, with the GIL released: a 200-point sweep of a two-layer ansatz ran 3.2x to
4.8x faster than a loop on a four-core host from 4 to 12 qubits, and 5.5x and 1.9x faster
than the batch run one circuit at a time at 14 and 16 qubits. From 17 qubits up each
circuit uses every core itself, and the batch saves only the crossing into Rust, about
2.4 microseconds a call.

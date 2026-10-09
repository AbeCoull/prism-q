# Python Bindings

`prism-q` ships Python bindings built with PyO3. They are a thin wrapper over the
Rust crate: the compiled extension is `prism_q._prism_q` and the pure-Python
`prism_q` package re-exports it. Simulation runs in Rust with the GIL released,
so the wrapper adds no per-gate overhead.

Wheels are `abi3` for Python 3.11 and newer, so one wheel per platform covers
every supported interpreter. PyPI carries wheels for Linux x86_64 and aarch64
(manylinux), macOS arm64 and Windows x64. Other platforms build from the source
distribution, which needs a Rust toolchain.

## Install

```bash
pip install prism-q
```

NumPy is the only runtime dependency. The Linux and Windows wheels also carry the
CUDA paths (see [GPU backends](#gpu-backends)), which need an NVIDIA driver and
NVRTC, the CUDA runtime compiler. The `cuda12` extra installs NVRTC from PyPI:

```bash
pip install "prism-q[cuda12]"
```

Neither is loaded at import, so on a host without them the package runs on the CPU
as usual and only `GpuContext(...)` raises. The macOS wheel has no CUDA paths.

Building from a source checkout needs [maturin](https://www.maturin.rs/):

```bash
pip install maturin
maturin develop --manifest-path bindings/python/Cargo.toml
```

The bindings enable the `parallel` feature by default. `gpu` and `distributed-mpi`
are optional. `--features gpu` adds the CUDA paths to a source build and needs no
CUDA toolkit to compile. The sharded statevector (see
[Distributed](#distributed-backend)) is in no wheel, so `distributed-mpi` always
means a source build.

## Quick start

```python
from prism_q import CircuitBuilder, simulate

circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
counts = simulate(circuit).seed(42).shots(1000).counts()
print(counts)          # {'00': 507, '11': 493}
```

```admonish warning title="q[0] is the least significant qubit"
`x q[0]` produces state index 1, not 2. In a counts key, character `i` is
classical bit `i` with bit 0 leftmost, so keys read reversed relative to Qiskit.
A Bell pair gives `'00'` and `'11'`, which look the same either way, but
`CircuitBuilder(2, 2).x(0).measure_all()` gives `'10'`, not `'01'`.
```

## Building circuits

`CircuitBuilder` is a fluent API. Every gate method returns the builder, and
`build()` produces the `Circuit` that simulation consumes.

```python
from prism_q import CircuitBuilder

circuit = (
    CircuitBuilder(3, 3)
    .h(0)
    .cx(0, 1)
    .rz(0.5, 2)
    .cphase(0.25, 1, 2)
    .measure_all()
    .build()
)
```

| Group | Methods |
|-------|---------|
| Single qubit | `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg` |
| Rotations | `rx(theta, q)`, `ry(theta, q)`, `rz(theta, q)`, `p(theta, q)` |
| Two qubit | `cx(control, target)`, `cz(q0, q1)`, `swap(q0, q1)`, `rzz(theta, q0, q1)`, `cphase(theta, control, target)` |
| Multi-qubit rotation | `pauli_rotation(theta, factors)` |
| Arbitrary unitary | `cu(matrix, control, target)`, `mcu(matrix, controls, target)`, `gate(gate, targets)` |
| Non-unitary | `measure(qubit, bit)`, `measure_all()`, `barrier(qubits)` |
| Parameters | `param(slot)`, `parameters()`, `parameter_links()` |

`pauli_rotation(theta, factors)` appends `exp(-i * theta * P / 2)` for the Pauli
string `P` given as `(qubit, axis)` factors with `axis` one of `"X"`, `"Y"`,
`"Z"`; identity factors are omitted. A weight-1 string lowers to `rx`, `ry`, or
`rz` and a two-qubit `ZZ` string to `rzz`, so fusion and Clifford recognition
keep firing on them. `Circuit.add_pauli_rotation` is the imperative spelling.

```python
builder.pauli_rotation(0.4, [(0, "X"), (1, "Y"), (3, "Z")]).param(0)
```

`cu` and `mcu` take a 2x2 matrix as nested Python sequences of complex numbers.
Out-of-range qubits raise `PrismError` at build time rather than at simulation
time.

Three other routes produce a `Circuit`:

```python
from prism_q import Circuit, circuits, parse_qasm

manual = Circuit(2, 2)                    # imperative, add_gate / add_measure / add_reset
ghz = circuits.ghz(10)                    # pre-built corpus
parsed = parse_qasm(qasm_source)          # OpenQASM 3.0, with 2.0 accepted
```

The `circuits` submodule mirrors the Rust builders documented in
[Circuit Builders](../reference/builders.md): `qft`, `ghz`, `w_state`, `random`,
`hardware_efficient_ansatz`, `clifford_heavy`, `clifford_random_pairs`, `qaoa`,
`single_qubit_rotation`, `clifford_t`, `quantum_volume`, `cz_chain`,
`phase_estimation`, `independent_bell_pairs`, `independent_random_blocks`, and
`local_clifford_blocks`. Seeded builders default to seed 42.

## Running a simulation

`simulate(circuit)` returns a `Simulation` you configure with `.seed()`,
`.backend()`, and `.noise()`, then finish with a terminal method. The default
seed is 42.

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
| `marginals()` | `list[tuple[float, float]]`, per-qubit `(p0, p1)` | density matrix only | yes |
| `state_vector()` | `complex128` amplitudes | no | yes |
| `probabilities_of(qubits)` | `float64` joint distribution over a subset, `qubits[0]` the lowest bit | density matrix only | yes |
| `reduced_density_matrix(qubits)` | `ReducedDensityMatrix`: `.matrix` over a subset with `qubits[0]` the lowest bit, `.purity`, `.metadata` | density matrix only | yes |
| `entanglement_entropy(subsystem)` | `EntropyResult`: von Neumann and Renyi-2 entropy of the cut | density matrix only | yes |
| `expectation_values(obs)` | `list[float]`, `⟨ψ\|P\|ψ⟩` per observable | density matrix only | yes |
| `expectation_values_reported(obs)` | `ExpectationResult`: the same values with `.metadata` naming the backend that served them | density matrix only | yes |
| `observable_variance(obs)` | `ObservableVariance`: `<H^2> - <H>^2` beside the mean | density matrix only | yes |
| `observable_expectation(h)` | `ObservableExpectation`: the weighted mean with its variance, group variances and standard error | density matrix only | yes |
| `density_matrix_expectation_values(obs)` | `list[float]`, exact `Tr(rho P)` | yes | no |
| `expectation_gradient(h, params)` | `(value, gradient)` via the adjoint method | no | no |
| `expectation_gradient_shift(h, params)` | `(value, gradient)` via the parameter-shift rule | no | no |
| `overlap(other)` | `OverlapResult`: `\|<a\|b>\|^2` against a second seeded builder, with one `.metadata` per side | no | yes |

Call `expectation_values_reported()` instead of `expectation_values()` when the
route matters: under `auto()` a wide shallow circuit can be answered by a tensor
contraction rather than by the state vector, and only the metadata says which ran. `expectation_gradient_shift()` computes the same gradient as
`expectation_gradient()` at two extra circuit runs per parameter, and is the
only route on a backend with no adjoint pass. `overlap()` takes a second seeded
builder, so each side keeps its own backend, seed and start state; both circuits
must declare the same width and both must be unitary.

`shots()` and `sample_counts()` average trajectories on any backend holding a
per-shot pure state. Every row marked "density matrix only" reads the exact mixed state
instead, so it needs `.backend(BackendKind.density_matrix())`; auto dispatch never
selects it. There the mixture is evolved once and every terminal reads that one
evolution, so the probabilities are seed independent and the observables carry no
sampling error. With `.noise()` attached, circuits with mid-circuit measurement or
classical conditioning are rejected on that route, since the mixture holds every
measurement branch at once; `density_matrix_expectation_values()` rejects them too.
Without a noise model the density-matrix backend runs them the way the statevector
does: each measurement samples an outcome and collapses onto it, so `run()` returns
the probabilities of one branch and `sample_counts()` draws a fresh branch per shot.

Terminals that cannot honor a model raise `PrismError` naming the reason.
`state_vector()` honors `.backend(...)` and declines
on a backend holding no pure state, rather than substituting one that does;
`density_matrix_expectation_values()` always uses the density-matrix backend.

`ShotsResult` and `CountsResult` both expose `counts()`, returning a dict keyed
by bitstring. `ShotsResult.shots` is a `bool` array of shape `(shots, bits)`
rather than a list of lists, so a caller indexing it as a nested list needs
`.tolist()`.

### Amazon Braket programs

`parse_braket(source)` reads a program under Braket's dialect and returns a
`BraketProgram`: its `circuit`, `parameters`, `noise` model, and the `results`
its `#pragma braket result` lines requested. `evaluate()` runs it and returns
one dict per request carrying `type` and `value`, in Braket's own basis order
where qubit 0 is the most significant bit:

```python
from prism_q import parse_braket

program = parse_braket(source)
exact = program.evaluate()
sampled = program.evaluate(shots=1000)
```

At the default `shots=0` every value is exact and `sample` is declined. Above
zero the values come from a measurement record, and `state_vector`, `amplitude`
and `density_matrix` are declined in turn. See the
[OpenQASM guide](openqasm.md) for the pragma surface.

`PrismError` carries a `kind` string naming the variant it came from, so a
caller can branch on the failure without matching its message. A run over a memory
cap raises kind `"resource_limit"` with the numbers attached: `resource` names the
unit (`"qubits"`, `"amplitudes"`, `"entries"`, `"elements"` or
`"bytes of device memory"`), `required` and `limit` are integers in that unit, and
`env_var` names the variable that overrides the cap, or is `None`.

### Distributions too wide to write down

A circuit whose qubits fall into independent groups is answered per group, and
`RunOutcome` keeps it that way: the dense vector is built only when
`probabilities` is read. Fifteen independent Bell pairs span 30 qubits, whose
dense form is 8 GB, and the blocks are 15 arrays of four entries.

```python
outcome = simulate(circuits.independent_bell_pairs(15)).seed(42).run()

outcome.num_basis_states            # 2 ** 30, and nothing was materialized
for qubits, probs in outcome.probabilities_factored():
    print(qubits, probs)            # [0, 1] [0.5 0. 0. 0.5], ...
```

Each block is `(qubits, probs)` with `qubits` ascending, and `probs` indexed by
those qubits packed in that order with `qubits[0]` in the least significant bit.
The probability of a basis state is the product of one entry per block, which is
what `probabilities` computes.

`probabilities_factored()` returns `None` when the run produced a dense
distribution, which is the common case: the decomposed route needs the widest
group several qubits narrower than the register, so two Bell pairs stay dense.
`num_basis_states` is `None` when the backend exposed no distribution at all.

## Result metadata

`RunOutcome`, `ShotsResult`, and `CountsResult` each carry a `metadata` object
describing how the result was produced.

```python
result = simulate(circuit).seed(42).run()
print(result.metadata.backend)               # 'Statevector'
print(result.metadata.engine)                # None unless samplers share the backend
print(result.metadata.is_exact)              # True
print(result.metadata.fidelity_lower_bound)  # None when exact
print(result.metadata.placement)             # 'host' or 'device'
print(result.metadata.bond)                  # None unless the MPS ran
```

`is_exact` is False when the engine that ran can discard state weight or
estimate by sampling. It marks the route, not the run: an MPS whose bond
dimension the circuit never fills reports `is_exact == False` with
`fidelity_lower_bound == 1.0`, so the flag answers whether the answer could have
been approximated and the bound answers whether it was.

An MPS run also reports `bond`, with `peak` (the widest bond any cut kept over the
run), `cap` (the configured maximum), and `saturated` (`peak >= cap`). Saturation is
the signal that the cap bound the run: a run whose peak stayed under the cap
truncated nothing on the cap's account, whatever the exactness label says.

Automatic dispatch sends a circuit past the statevector cap to the sparse map when its
state never holds more than 128 basis states and to a bounded-bond MPS otherwise. The
MPS route is taken by default and the result says so. `.require_exact()` rejects it
instead, raising `PrismError` naming the engine it would have used.

```python
simulate(big_circuit).seed(42).require_exact().marginals()  # raises
```

## Starting from a state other than |0...0>

`.initial_state(amplitudes)` replaces the default all-zero start. It takes any
sequence of complex numbers, a `complex128` NumPy array included, indexed with
qubit 0 in the least significant bit.

```python
import math
import numpy as np
from prism_q import CircuitBuilder, simulate

theta = math.pi / 8
start = np.array([math.cos(theta), math.sin(theta)], dtype=np.complex128)
circuit = CircuitBuilder(1).h(0).build()
probs = simulate(circuit).initial_state(start).seed(42).run().probabilities
```

The vector must have `2 ** num_qubits` entries and unit norm. A wrong length, a
non-finite entry, or a norm off unity raises `PrismError`; an unnormalized
vector is rejected rather than rescaled, so a mistake surfaces instead of
becoming a silent factor on every amplitude.

A start state also narrows the route. Auto dispatch reads circuit structure, and
its shortcuts (tableau, product state, subsystem decomposition, Pauli
propagation) are only valid from |0...0>: a Clifford circuit produces a
stabilizer state only when its input is one. So `auto()` resolves to the
statevector, the GPU and distributed statevectors and `density_matrix()` are the
only other backends that accept one, and every other choice raises `PrismError`
naming itself. Every terminal whose table row says so carries it;
`expectation_gradient()` and `density_matrix_expectation_values()` reject it, as
do `shots()` and `sample_counts()` with a noise model attached, since trajectory
replay reinitializes a pure state per shot. To evolve a start
state under noise, read the exact mixture with `run()`, `marginals()`, or
`expectation_values()` on `density_matrix()`.

## Starting from a mixture

`.initial_density_matrix(rho)` starts the density-matrix backend from a mixed
state. It takes a square `complex128` NumPy array, or a sequence of rows, in
the layout `reduced_density_matrix()` over the whole register returns:
row-major `2 ** n` by `2 ** n` with qubit 0 the least significant bit of both
indices. A noisy run can be paused, its mixture stored, and resumed later:

```python
from prism_q import BackendKind, NoiseModel, simulate

dm = BackendKind.density_matrix()
rho = (
    simulate(first)
    .backend(dm)
    .noise(NoiseModel.uniform_depolarizing(first, 0.02))
    .reduced_density_matrix(range(first.num_qubits))
    .matrix
)
probs = (
    simulate(second)
    .backend(dm)
    .noise(NoiseModel.uniform_depolarizing(second, 0.02))
    .initial_density_matrix(rho)
    .run()
    .probabilities
)
```

The matrix must have `4 ** num_qubits` entries, be Hermitian to 1e-12 per
entry, and have unit trace to 1e-9; one failing a check raises `PrismError`
naming it. Positive semidefiniteness is not checked. Only `density_matrix()`
and its GPU sibling accept a mixture, so every other backend, `auto()`
included, raises `PrismError` naming itself. The terminal table above applies
unchanged, and setting a mixture clears an earlier `.initial_state()`, as that
call clears a mixture.

## Handing a Clifford state to a second run

`StabilizerBackend` is a tableau held across calls. `run(circuit)` resets it to
the circuit's width and runs the circuit; `export_tableau()` returns the rows as
a `uint64` array of bit-packed words and a `bool` array of row signs;
`import_tableau(num_qubits, words, phases, num_classical_bits)` starts a backend
from that pair; and `apply(circuit)` runs a circuit on the held state without
resetting it. The pair is the raw tableau, so it is the checkpoint format for a
Clifford prefix that is expensive to replay.

```python
from prism_q import StabilizerBackend

prep = StabilizerBackend(seed=42)
prep.run(clifford_prefix)
words, phases = prep.export_tableau()

resumed = StabilizerBackend(seed=42)
resumed.import_tableau(clifford_prefix.num_qubits, words, phases, num_classical_bits=8)
bits = resumed.apply(measurement_suffix)
```

The import checks the lengths and that each destabilizer anticommutes with its
stabilizer partner, nothing more; the intended input is an export. The random
stream restarts from the importing backend's seed, so a resumed run and an
uninterrupted one draw the same outcomes only when neither drew before the
split.

## Selecting a backend

`BackendKind.auto()` is the default and picks a backend from circuit structure.
Pass an explicit one to override it.

| Constructor | Notes |
|-------------|-------|
| `auto()` | Structure-driven dispatch. See [Choosing a Backend](../getting-started/choosing-a-backend.md). |
| `statevector()` | Dense amplitudes, the general-purpose path |
| `stabilizer()`, `factored_stabilizer()` | Clifford-only circuits |
| `stabilizer_rank()` | Clifford+T, requires at least one T gate |
| `sparse()` | Sparse states, for circuits that stay concentrated |
| `product_state()` | No entangling gates |
| `factored()` | Partially independent subsystems |
| `tensor_network()` | Contraction over a network |
| `mps(max_bond_dim=256)` | Approximate, truncates at the bond dimension |
| `density_matrix()` | Exact mixed states, never chosen by `auto()` |
| `stochastic_pauli(num_samples=1000)` | Sampled Pauli propagation |
| `deterministic_pauli(epsilon=0.0, max_terms=65536)` | Truncated Pauli propagation |
| `deterministic_pauli_budget(max_terms=65536)` | Pauli propagation holding a fixed term count |
| `auto_gpu(context)`, `statevector_gpu(context)`, `stabilizer_gpu(context)`, `density_matrix_gpu(context)` | CUDA device paths, see [GPU backends](#gpu-backends) |

The density-matrix backend stores `4^n` amplitudes, so its qubit ceiling is
about half the statevector cap; exceeding it raises `PrismError` naming the cap.
`statevector_distributed(context)` reaches the sharded backend; the extension attaches
to a running MPI rather than calling `MPI_Init` itself, which is what makes it safe
beside mpi4py.

## GPU backends

The GPU constructors take a `GpuContext`, an opaque handle to one CUDA device
and its compiled kernels. Build it once and reuse it: construction compiles the
kernel module, and passing the same handle to several simulations shares that
work.

```python,ignore
from prism_q import BackendKind, GpuContext, circuits, simulate

context = GpuContext(0)
outcome = simulate(circuits.qft(16)).backend(BackendKind.auto_gpu(context)).seed(42).run()
print(outcome.probabilities)
```

`GpuContext(device_id)` is where a missing or unusable device is reported, and
it raises `PrismError` rather than falling back. Past construction, routing is
soft by design and matches the Rust API: `statevector_gpu` runs circuits below
the crossover (`PRISM_GPU_MIN_QUBITS`, default 14) on the host, `auto_gpu`
routes each block independently, and a block whose device allocation fails
degrades to the host rather than erroring, so host results are not a failure
signal.

`stabilizer_gpu` sets its crossover at 100000 qubits
(`PRISM_STABILIZER_GPU_MIN_QUBITS`), so it runs on the host tableau unless that
override is lowered. The device tableau is correct; the default stays high until
benchmarks justify lowering it.

`density_matrix_gpu(context)` holds the exact mixed state on the device and is the
one device kind with no crossover and no host fallback: `auto_gpu` never selects it,
every noisy terminal that `density_matrix()` serves answers from the device buffer,
and a width whose `4^n` buffer does not fit in free device memory raises `PrismError`
before anything is allocated (13 qubits on an 11 GiB card).

The Linux and Windows wheels are built with CUDA support; the macOS wheel and a
default source build are not. Without it the constructors still exist and
`GpuContext(...)` raises `PrismError` naming the missing build feature, so code
written against the GPU API fails with a message rather than an `AttributeError`.

A CUDA build loads the NVIDIA driver when the first `GpuContext` opens, not at
import, and NVRTC only when no cached kernel image matches the device, so a host
with a warm cache never opens NVRTC. The driver must support CUDA 12.0 or newer.
NVRTC comes from a CUDA 12 toolkit on the loader path (`PATH` on Windows, the
`ld.so` search path on Linux), and otherwise from the `nvidia-cuda-nvrtc-cu12`
package that the `cuda12` extra installs. When either is missing, `GpuContext(...)`
raises `PrismError` naming it.

The kernels compile to SASS for the device's own architecture, which any CUDA 12
driver loads, so the newest NVRTC the extra installs also runs on older 12.x
drivers. Only a device newer than the NVRTC falls back to PTX, which the driver
compiles itself and which needs a driver at least as new as the NVRTC;
`GpuContext(...)` names both versions when they disagree.

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

The distributed statevector shards the dense state across MPI ranks. It is
reachable from Python through a `DistributedContext`, which attaches to an MPI
that is already running. This extension never calls `MPI_Init` or
`MPI_Finalize`: mpi4py does both, at its own import and at interpreter exit, and
a handle whose refcount drop finalized MPI would make every later MPI call in
the process erroneous.

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

Run it with `mpiexec -n 4 python script.py`. The rank count must be a power of
two, and every rank needs enough local qubits after the shard bits are taken
(`PRISM_DIST_MIN_LOCAL_QUBITS`).

The contract is SPMD, and it is enforced rather than assumed. Four ranks are
four interpreters running the same source, and every collective inside the
backend is entered by all of them, so a script that branches before the call

```python,ignore
if comm.rank == 0:
    result = simulate(circuit).backend(...).run()   # deadlocks
```

hangs the job: rank 0 blocks in a collective the others never enter. Every rank
calls `run` with the same circuit and seed; only what you do with the returned
value may branch on rank. Two cross-checks turn the common violations into
errors instead of hangs, each costing one collective at run entry: ranks
disagreeing about the register shape, the seed, or the tuning knobs are rejected,
and so are ranks handed different circuits.

Two surfaces are deliberately loud rather than convenient.

A world of one rank raises. MPI-2 and later make a singleton `MPI_Init` succeed,
so a script launched without `mpiexec` would otherwise get a correct answer from
one rank at single-host speed with no signal that nothing was distributed. Pass
`allow_single_rank=True` when that is what you meant.

Constructing the context without MPI running raises rather than starting MPI, so
a forgotten `from mpi4py import MPI` is reported at the point it happened.

The thread level mpi4py negotiated must be at least `MPI_THREAD_FUNNELED`: Rayon
workers run beside MPI in the same process, and every MPI call stays on the thread
that constructed the context. mpi4py requests `MPI_THREAD_MULTIPLE` by default; a
script that lowers it (`mpi4py.rc.thread_level = 'single'` or
`mpi4py.rc.threads = False`) gets an error from the constructor rather than a run
that mixes threads with a single-threaded MPI. Below `MPI_THREAD_MULTIPLE`, a
simulation call from a Python thread other than the one that constructed the
context raises before it reaches MPI, and at `MPI_THREAD_FUNNELED` the constructor
itself raises off the thread that imported mpi4py.

The published wheels have no MPI support: `mpi-sys` runs bindgen and needs a
system MPI at build time, and the extension has to link the same MPI
implementation and ABI as mpi4py and as the launcher. Mixing two implementations
in one process corrupts rather than failing loudly, which is why this is a
from-source path:

```bash
maturin develop --manifest-path bindings/python/Cargo.toml --features distributed-mpi
```

`DistributedContext.is_supported()` says whether a build has MPI support, the
same way `GpuContext.is_supported()` does for CUDA.

Sub-communicators, per-rank device placement, and any distributed path other
than the statevector backend are out of scope: the context is the world
communicator and nothing else.

## Noise

Build a `NoiseModel` from a circuit, then attach it. A model is sized to the
circuit it was built from, and using it with a different circuit raises
`PrismError`.

```python
from prism_q import NoiseChannel, NoiseModel, simulate

model = NoiseModel.uniform_depolarizing(circuit, 0.01)
counts = simulate(circuit).seed(42).noise(model).sample_counts(4000).counts()
```

`NoiseModel.uniform_depolarizing(circuit, p)` and
`NoiseModel.with_amplitude_damping(circuit, gamma)` cover the common cases.
`NoiseBuilder` attaches channels by rule and validates the resulting model at
`build(circuit)`, before simulation:

```python
from prism_q import GateFilter, NoiseBuilder

noise_rules = (
    NoiseBuilder()
    .after_gates(GateFilter.all().arity(1), NoiseChannel.depolarizing(0.001))
    .after_gates_joint(
        GateFilter.all().named("cx"), NoiseChannel.two_qubit_depolarizing(0.01)
    )
    .uniform_readout_error(0.01, 0.02)
)
rule_model = noise_rules.build(circuit)
```

Fluent methods update the same builder or filter. Adding a rule copies its filter
and channel, so later filter edits leave the rule unchanged; `build` keeps the
builder reusable. `GateFilter()` and `GateFilter.all()` start unrestricted.
`named`, `arity`, `on_qubits` and `on_targets` combine restrictions: `on_qubits`
selects an unordered set of targets for noise, while `on_targets([0, 1])` matches
the complete ordered target list, excluding `cx(1, 0)`. Build against the original
circuit because fusion changes gate names. Gate rules skip conditional instructions.

| Rule | Effect |
|------|--------|
| `after_gates(filter, channel)` | One single-qubit event per matching target |
| `after_gates_joint(filter, channel)` | One event on the complete target list when its arity matches the channel |
| `crosstalk(filter, coupling, channel)` | Spectator noise along undirected coupling edges; two-qubit channels put the gate target first |
| `over_rotation(filter, relative)` | An extra `relative * theta` rotation after `rx`, `ry`, `rz` or `p` |
| `on_idle_qubits(channel)` | One single-qubit event per idle qubit per greedy circuit layer |
| `after_resets(channel)` | One single-qubit event on each reset qubit |
| `before_measurements(channel)` | Damage the measured state, including outcomes used by feed-forward |
| `readout_error(bit, p01, p10)` | Override the uniform readout rates on one classical bit |

Rules emit events in registration order within each instruction slot. Idle events
attach to the highest-indexed instruction in their layer. Pre-measurement events
attach to the preceding instruction, so a measurement at instruction zero needs a
barrier prepended. Readout error changes reported bits after sampling and leaves
the quantum state intact.

For explicit instruction slots, start from `NoiseModel.empty(circuit)` and attach
events:

```python
model = NoiseModel.empty(circuit)
model.add_event(0, NoiseChannel.amplitude_damping(0.05), [0])
model.add_event(1, NoiseChannel.two_qubit_depolarizing(0.02), [0, 1])
model.with_readout_error(0.01, 0.01)
model.validate()
```

A device calibration table lowers onto a circuit in one call. `DeviceCalibration.parse`
reads the text form described in the [Noise and QEC guide](./qec.md), and the presets
carry illustrative magnitudes for a technology class rather than a measured device:

```python
from prism_q import DeviceCalibration

calibration = DeviceCalibration.superconducting_transmon(circuit.num_qubits)
model = calibration.to_noise_model(circuit)
```

Channels are `pauli(px, py, pz)`, `depolarizing(p)`, `amplitude_damping(gamma)`,
`phase_damping(gamma)`, `thermal_relaxation(t1, t2, gate_time, excited_population=0.0)`,
`two_qubit_depolarizing(p)`, and `custom(kraus)` for an explicit list of 2x2
Kraus operators. `validate()` checks probabilities and Kraus completeness;
`is_pauli_only()` reports whether the model holds only single-qubit Pauli
channels and no readout error. A model carrying readout error or
`two_qubit_depolarizing` answers `False` there and still runs on the stabilizer
samplers, which apply readout to the measurement record and sample the pair
channel as one joint draw over its 15 branches.

## Expectation values

An observable is a list of `(qubit, axis)` factors with `axis` one of `"X"`,
`"Y"`, `"Z"`. Identity factors are omitted, so `[(0, "Z"), (2, "X")]` means
`Z0 ⊗ I1 ⊗ X2`. Both expectation terminals take a list of observables and return
one float each.

```python
observables = [[(0, "Z")], [(0, "Z"), (1, "Z")]]
values = simulate(circuit).seed(42).expectation_values(observables)
```

`expectation_values` requires a unitary circuit and gives `⟨ψ|P|ψ⟩`.
`density_matrix_expectation_values` evolves the density matrix through the
circuit and any attached noise model and gives exact `Tr(rho P)`, with
measurements read off the final mixed state without collapse. It is the
zero-variance analogue of averaging over trajectories:

```python
model = NoiseModel.empty(circuit)
model.add_event(0, NoiseChannel.amplitude_damping(0.3), [0])
exact = simulate(circuit).seed(42).noise(model).density_matrix_expectation_values(
    [[(0, "Z")]]
)
```

```admonish note title="Compare with a tolerance"
Analytic means are not bit-stable across separate invocations. Hash-ordered term
accumulation can move the last ulp, so compare against `1e-12` rather than
asserting exact equality.
```

## Mid-circuit saves

A save point records the state where it sits, so a circuit can be inspected part
way through without being cut in two and run twice.

```python
from prism_q import CircuitBuilder, Gate, SaveSpec, simulate

builder = CircuitBuilder(4)
for q in range(4):
    builder.h(q)
circuit = builder.build()
circuit.add_save(SaveSpec.StateVector, "after_hadamards")
circuit.add_gate(Gate.cx(), [0, 1])

outcome = simulate(circuit).seed(42).run()
for record in outcome.saves:
    print(record["label"], record["kind"], record["value"].shape)
```

Each record is a dictionary with `label`, `kind`, and `value`. `SaveSpec.StateVector`
and `SaveSpec.DensityMatrix` come back as `complex128` arrays, the density matrix flat
and row major over `2**n` rows; `SaveSpec.Probabilities` comes back as `float64`.
Records arrive in the order their points were reached, and labels need not be unique.

A save is a barrier across the whole register, so no gate is fused or reordered across
it. Only `run` returns the records: `shots`, `marginals` and the rest decline a circuit
carrying a save point rather than running it and dropping what it recorded. The same
goes for routes that hold no state to read, and for OpenQASM export, which has no save
syntax to write.

## Running many small circuits

`run_batch` takes a list and crosses into Rust once, holding one backend across
circuits of the same width that draw no randomness.

```python
from prism_q import run_batch

outcomes = run_batch(circuits, seed=42)
```

Results match running each circuit alone with the same seed, and a failing batch raises
the first failure in list order. Up to 16 qubits the circuits split across cores with the
GIL released, one circuit per core: a 200-point sweep of a two-layer ansatz ran 3.2x to
4.8x faster than a loop on a four-core host from 4 to 12 qubits, and 5.5x and 1.9x faster
than the batch run one circuit at a time at 14 and 16 qubits. From 17 qubits up each
circuit uses every core itself, and the batch saves only the crossing into Rust, about
2.4 microseconds a call.

## Parameter sweeps

A variational loop rebinds angles while the gate sequence stays fixed.
`Parameters` names the slots those angles land in, and `PreparedCircuit` holds
the circuit across bindings so fusion and backend selection are settled once
rather than per point.

```python
from prism_q import CircuitBuilder, PreparedCircuit

builder = CircuitBuilder(4)
for q in range(4):
    builder.ry(0.1, q).param(q)
for q in range(3):
    builder.cx(q, q + 1)

prepared = PreparedCircuit(builder.build(), builder.parameters())
for values in [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]:
    outcome = prepared.run(values, seed=42)
```

`run(values, seed=42)` returns the same `RunOutcome` as `simulate(...).run()`.
`expectation_values(values, observables)` and
`observable_expectation(values, hamiltonian)` return what the `Simulation`
terminals of the same name return on the bound circuit, without planning
dispatch or fusing again. `bind(values)` returns the bound `Circuit` instead, for
handing to `simulate(...)` with different options or to any other consumer of a
circuit.

`run_many`, `expectation_values_many` and `observable_expectation_many` take a
`(points, num_slots)` array and cross into Rust once. Up to 16 qubits the rows
split across cores, as `run_batch` does, and every result matches the single
call with the same seed.

```python
energies = prepared.observable_expectation_many(points, hamiltonian, seed=42)
```

A term list is parsed on every call, and the grouping of its terms into
qubit-wise-commuting sets is recomputed with it. `PauliObservable` parses the
list once and keeps the grouping after its first use. Every argument that takes
a Hamiltonian accepts one in place of the list, the gradients included:

```python
from prism_q import PauliObservable

h = PauliObservable(hamiltonian)
for values in points:
    mean = prepared.observable_expectation(values, h, seed=42).mean
```

Automatic dispatch reads the template, so build it at angles representative of
the sweep. A template whose rotations are all zero reads as Clifford and settles
on a backend that then rejects the bound circuit. Pass an explicit backend as
the third argument to decide it yourself:

```python
prepared = PreparedCircuit(circuit, params, BackendKind.statevector())
```

`reuses_fusion_plan` reports whether the recorded fused structure is being
replayed. Results agree either way; only the speed differs.

Build a `Parameters` three ways. `builder.parameters()` returns the set recorded
by `param(slot)`; `Parameters.all_rotations(circuit)` gives every bindable gate
its own slot in circuit order; `Parameters(n)` plus `link(instruction, slot)`
declares the slots up front. Several gates may share a slot, in which case
binding writes one angle to each.

`parse_qasm_parametric` returns a template and the named slots declared by OpenQASM
`input` statements, in declaration order:

```python
from prism_q import BackendKind, PreparedCircuit, parse_qasm_parametric

template, parameters = parse_qasm_parametric("""
OPENQASM 3.0;
input float[64] theta;
qubit[2] q;
ry(theta) q[0];
cx q[0], q[1];
ry(theta) q[1];
""")
assert parameters.slot_of("theta") == 0
qasm_sweep = PreparedCircuit(template, parameters, BackendKind.statevector())
outcomes = qasm_sweep.run_many([[0.2], [0.7]], seed=42)
```

Input angles start at zero, so bind the template before simulation and choose an
explicit backend when preparing it. Each input must occupy a whole angle argument
of a supported gate at the top level; expressions such as `2 * theta` and inputs in
control-flow bodies raise `PrismError`. `parse_qasm` continues to reject unbound inputs.

| Method | Returns |
|--------|---------|
| `bind(template, values)` | `template` with the linked angles overwritten |
| `values(circuit)` | the angle each slot currently holds |
| `validate(circuit)` | nothing; raises if a link no longer points at a bindable gate |
| `with_names(names)` | a copy naming the slots, matching OpenQASM `input` declarations |
| `name_of(slot)`, `slot_of(name)` | the name and slot of a named set, else `None` |
| `unread_slots()` | declared slots no instruction reads, whose values are discarded |

A wrong-length value vector, a non-finite angle, and a link pointing at a gate
that carries no angle all raise `PrismError`.

## Gradients

`expectation_gradient` computes `⟨H⟩` and its exact gradient with respect to the
bound parameters by the adjoint method, at a cost independent of the
parameter count. Mark parameters with `param(slot)` while building, then
pass `parameter_links()` through.

```python
import numpy as np
from prism_q import CircuitBuilder, simulate

builder = CircuitBuilder(2)
builder.ry(0.3, 0).param(0)
builder.cx(0, 1)
builder.rz(0.7, 1).param(1)
circuit, links = builder.build(), builder.parameter_links()

hamiltonian = [(1.0, [(0, "Z")]), (0.5, [(0, "Z"), (1, "Z")])]
value, gradient = simulate(circuit).seed(42).expectation_gradient(hamiltonian, links)
```

A Hamiltonian term is `(coefficient, observable)`. Several gates may share a
slot, in which case their gradients accumulate. `param()` rejects anything
but a differentiable gate (`rx`, `ry`, `rz`, `rzz`, `p`, `pauli_rotation`), and
the circuit must be unitary.

## Quantum error correction

`QecProgram` exposes the native QEC IR: `reset`, `measure`, `detector`,
`observable_include`, `postselect`, and `noise`, with `QecBasis`, `QecNoise`,
and `RecordRef` as the supporting types. `run()` returns a `QecResult` carrying
detector, observable, and measurement arrays as NumPy `bool_` matrices, plus
`logical_error_rates()` and `survivor_rate()`. Programs can also be parsed from
text with `QecProgram.from_text`. See the [Noise and QEC guide](./qec.md) for the
model itself.

`detector_error_model()` derives the program's `DetectorErrorModel` for
decoding: `probabilities()` (float64), `detector_matrix()`, and
`observable_matrix()` (bool, detectors or observables by mechanisms) feed
check-matrix decoder constructors directly, `detector_coords()` carries the
per-detector coordinates, `decompose_graphlike()` returns the form matching
decoders need (at most two detectors per mechanism), and `to_text()` writes
the common detector error model text format for file-based decoders:

```python
dem = qp.detector_error_model()
H, p, L = dem.detector_matrix(), dem.probabilities(), dem.observable_matrix()
with open("memory_d3.dem", "w") as f:
    f.write(dem.to_text())
```

`Decoder` runs the built-in union-find decoder over a graphlike model, so a
memory experiment's logical error rate needs no external decoder. `decode`
takes the `(shots, num_detectors)` bool detector array and returns the
`(shots, num_observables)` predicted observable flips:

```python
decoder = prism_q.Decoder(dem.decompose_graphlike())
res = qp.run()
predicted = decoder.decode(res.detectors)
failures = (predicted[:, 0] != res.observables[:, 0]).sum()
```

`packed_detectors()`, `packed_observables()`, and `packed_measurements()` return
the same records eight to a byte, as `(shots, ceil(n / 8))` `uint8` arrays: record
`j` of a shot is bit `j % 8` of byte `j // 8`, and the unused high bits of the
last byte are zero. A million shots then hold an eighth of the memory of the
bool arrays, and `np.unpackbits` restores them:

```python
import numpy as np

packed = res.packed_detectors()
detectors = np.unpackbits(packed, axis=1, count=qp.num_detectors, bitorder="little")
assert (detectors.astype(bool) == res.detectors).all()
```

## Errors and typing

Every failure surfaces as `prism_q.PrismError`, carrying the message from the
Rust error. Backend limits, unsupported operations, and invalid arguments all
raise it:

```python
import prism_q

try:
    simulate(huge).density_matrix_expectation_values([[(0, "Z")]])
except prism_q.PrismError as exc:
    print(exc)   # backend `density_matrix` is incompatible: circuit has ... qubits
```

The package ships type stubs (`prism_q/_prism_q.pyi`) and a `py.typed` marker,
so mypy and Pyright see the full surface.

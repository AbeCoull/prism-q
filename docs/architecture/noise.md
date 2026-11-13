# Noise Compilation

How a noise model becomes events, and which engine samples them. The user-facing
version is the [noise guide](../guides/noise.md), and the native QEC path is on
[QEC programs](./qec-programs.md).

## Noise models (`src/sim/noise.rs`)

`NoiseModel` holds one list of noise events per instruction slot. Every noisy engine
supports Pauli and depolarizing channels, and every one but `noisy_marginals_analytical`
supports two-qubit depolarizing. Amplitude damping, phase damping, thermal relaxation,
and one- and two-qubit custom Kraus operators need the trajectory engine. Readout error
acts on the measurement record rather than the state, so the route is chosen on the
channels alone and each engine applies readout itself.

`NoiseBuilder` (`src/sim/noise_builder.rs`) compiles declarative rules into the same
slot vector. The rules cover rates by gate type and by qubit, idle decoherence against
circuit layers, crosstalk through a coupling map, and coherent over-rotation
proportional to a rotation gate's angle. They also cover reset error, error before
measurement, readout per bit, idle relaxation and detuning timed by an as-soon-as-possible
schedule of gate durations, and amplitude drift per shot. Every rule is evaluated once at
build time, so nothing reaches a loop over shots or instructions. The drift rules emit
fixed weights over draws the trajectory makes per shot. A timed rule places each qubit's
events for a layer just before that qubit's next instruction, since instruction order
can list a later layer's gate on the qubit ahead of an earlier layer's last instruction.

Custom Kraus sets must be trace preserving, `sum_k Kdagger_k K_k = I` to 1e-9: the exact
route applies a set literally while the trajectory route normalizes its branch
probabilities. `ThermalRelaxation` is generalized amplitude damping composed with pure
dephasing on both routes. Populations decay as `exp(-gate_time/t1)` and coherences as
`exp(-gate_time/t2)` toward a steady state of `excited_population`. A mixture of reset
and `Z` would match the coherence decay only for `t2 <= t1`.

Two-qubit Kraus operators are indexed `K[t][t']` with `t = 2*bit(q0) + bit(q1)`, the
packing `Gate::matrix_4x4` uses. The density matrix compiles the set into a 16x16 block
superoperator (`apply_2q_kraus`, which `apply_2q_depolarizing` lowers onto). The
trajectory engine draws a branch from `Tr(Kdagger K rho)` over
`Backend::reduced_density_matrix_2q` and applies the normalized operator as a `Fused2q`.
The host statevector, sparse, factored, MPS and distributed statevector backends answer
that reduction, the distributed one through `Backend::reduced_density_matrix`. The
tableau backends, the product state and the tensor network decline it.

`Auto` counts a two-qubit Kraus event as entangling when it picks the trajectory
backend, so one-qubit gates carrying a crosstalk channel run on the statevector (or the
sparse map or MPS past the dense cap), not the product state. Routing asks the
trajectory backend for `Backend::supports_two_qubit_kraus` before the first shot. An
explicit kind that declines the channel, a device statevector included, is named at
dispatch, and `Auto` with a device keeps such a model on the host statevector.

## Noisy compiled sampler

Backward Pauli propagation through the circuit gives each noise location an X-flip and a
Z-flip sensitivity row. At sample time Bernoulli draws decide which channels fire, and
their sensitivity rows XOR into the sample.

With the `gpu` feature and a context attached via `with_gpu`, the sampler can sample,
apply noise, and reduce counts or marginals on the device. Device noise masks come from
a stream seeded on the device, so GPU output matches CPU output statistically, not bit
for bit. Device counts are limited to 512 measurements (8 packed words); larger circuits
fall back to the CPU reduction.

## Engine routing and the observable contract

Noisy sampling is reachable through `simulate(...).noise(...).shots(n)` and
`.sample_counts(n)` on trajectory backends. The density matrix also serves `run`,
`marginals` and `expectation_values`
([noise across the terminals](./engine.md#noise-across-the-terminals)).
`run_shots_with_noise` (`src/sim/mod.rs`) routes Pauli-only noise on Clifford circuits
to the compiled family when the backend is `Auto` or a stabilizer kind and the
measurements are terminal or rewrite to terminal (below). Every other accepted
combination runs the trajectory engine over the resolved backend. Within the compiled
family, `run_shots_noisy` (`src/sim/noise.rs`) picks one engine per call:

| Engine | Selected when | Limitations |
|---|---|---|
| Brute-force replay (`run_shots_noisy_brute_with`) | Dynamic circuits the rewrite declines | Tableau replay per shot, O(shots) simulations; non-Clifford circuits error here (the public entry point routes them to the trajectory engine instead) |
| Pauli frame | Shallow circuits: gate count / qubits < 3, or < 5 at >= 200 qubits | Clifford, terminal measurements, no event in a guarded body |
| Compiled Pauli (`NoisyCompiledSampler`) | Remaining Clifford circuits | Clifford, terminal measurements after the rewrite |

All engines sample the same distribution of measurement records for the models and
circuits they accept. They consume independent RNG streams, so one seed gives different
shots with matching marginals, correlators and histograms. The frame and compiled
engines and the noiseless compiled sampler all resolve to
`ResolvedBackend::CompiledStabilizer`, and `RunMetadata::engine` names which one ran.

`defer_noisy_circuit` (`src/sim/compiled/mod.rs`) rewrites a circuit with resets,
mid-circuit measurements or guarded Pauli gates to terminal measurements. A reset moves
its qubit to a fresh alias. A measurement whose qubit nothing touches before its next
reset moves to the end unchanged; one whose qubit a later gate, guard or measurement
uses becomes a `CX` onto a fresh alias measured at the end, so the qubit stays live and
the copy only ever controls. A guarded Pauli `P` on qubit `t` becomes `P`
controlled by the alias of each record its condition reads, plus an unconditional `P`
when the condition is inverted. That holds for a single bit, a `Parity`, or a register
comparison reading at most one written bit. A comparison reading several written bits is
an AND of records, which no controlled Pauli applies, so its Pauli stays out of the
circuit as a guarded entry. A guarded body holds Pauli gates only. An event on a measured
alias drops. The noiseless shot route takes the same rewrite at two shots or more.

An event in a guarded body fires only on shots whose guard holds, which no fixed linear
map expresses. The compiled sampler draws every shot's records under the unguarded
events. Each guarded entry then lands its precomputed record flips in execution order:
an event draws at its rate on every shot and lands where its guard holds, and a guarded
Pauli lands on every shot its guard holds. When a guarded Pauli is among the entries,
they land on measurement rows, so a guard is a few word operations per 64 shots; readout
comes last. Readout error on a bit a guard reads is not left to the end: the rewrite
places an `X` of that rate on each alias measured into the bit, which the guard's
controls and the record then both see. A rate equal both ways fires that `X` on every
shot. Unequal rates make it a guarded entry instead, drawn at the rate the record's value
picks after the earlier entries land. Such a circuit always takes the compiled sampler,
never the frame sampler.

## Slots for guarded instructions

Slots `0..n` follow the `n` top-level instructions and fire every shot, a guarded
instruction's included. Body slots come after: each region lays out its body's slots,
then its nested guards', and a conditional takes one slot for its gate. A body slot
fires only on the shots its instruction runs. `NoiseBuilder` gate rules put a guarded
gate's noise in its body slot, and rules that read a schedule decline a circuit holding
a region. `compile_noisy` and the error chain complex reject body slots that carry an
event, and the density matrix rejects any guard.

## Trajectories

The trajectory engine (`src/sim/trajectory.rs`) covers everything the compiled family
rejects (non-Pauli channels, non-Clifford gates, dynamic circuits the rewrite declines)
at the cost of one state evolution per shot. On the distributed statevector the
trajectories run serially in shot order on every rank, so the ranks issue their
collectives in lockstep.

Pauli noise on the host statevector skips the per-shot cost when all measurements are
terminal, there is no reset, and no live event follows the first measurement. Every
shot's error pattern is drawn first from its own seed, and shots with the same pattern
share one evolution. Each shot then draws its record from its group's final state on its
own stream. The distribution is right but the shots differ, seed for seed, from the
per-shot path. When distinct patterns exceed nine tenths of the shots, the run falls back
to one trajectory per shot.

Amplitude and phase damping on the host statevector share a prefix of gates: everything
before the first instruction that is not a gate or barrier, or whose events include a
channel other than Pauli, depolarizing or damping. Until a shot draws below a damping
`gamma` or a Pauli threshold, its state is the one every such shot holds. The run walks
the shared state once, and each shot copies it where it leaves, redraws its stream that
far and finishes alone. The records match the per-shot path seed for seed. A walk that
reads `P(1)` past one by more than its headroom falls back to per-shot trajectories, as
do leakage and quasi-static channels.

## Two-qubit channels

A two-qubit channel is one joint draw over its 15 non-identity Pauli products, never two
single-qubit draws: both letters move with probability `0.6p`, where independent draws
give `(0.8p)^2`. The compiled sampler stores the propagated X and Z components of both
targets, four rows an event, in a table run as a second pass after the single-qubit
rows. A pair whose targets fall in different subsystem blocks takes the monolithic
compile.

A non-empty pair table keeps noise application on the host, because the device noise
kernel would draw the two qubits independently. The noiseless parity sample and the bit
transpose still run on the device.

## Readout and validation

Readout error flips a classical bit when a measurement writes it, so a condition reads
the noisy bit and each measurement into the bit draws its own flip. The state is never
touched. When no condition reads a bit carrying readout error, only each bit's last
write reaches the record, and flipping the finished record draws the same distribution.
Every route then keeps doing that, with the seeded shots it always drew. A trajectory
whose conditions read such a bit, the brute-force stabilizer replay included, flips
each bit on the shot's noise stream right after the measurement instead.

The frame and compiled samplers apply readout error to the packed measurement record
after the reference outcomes are folded in, so a set bit is a measured one. Records are
thinned at `max(p01, p10)`, then a candidate is accepted at the rate its live bit
selects. Brute-force replay flips each shot's unpacked record on that shot's own stream,
as the trajectory engine does.

Every entry point that draws shots calls `NoiseModel::validate_for` against the circuit
before allocating state. It checks one event slot per instruction, channel parameters
in range, distinct targets on a two-qubit channel, and every target inside the register.
`noisy_marginals_analytical` is the exception, having no per-shot state to allocate.

## Leakage and quasi-static drift (`src/sim/trajectory.rs`, `src/qec/leakage.rs`)

Two channel families carry state from one event to the next within a shot: leakage
(`Leakage`, `Seepage`, `LeakageTransport`) keeps a leak flag per qubit, and
`QuasiStatic` reads standard normal draws made once at the start of the shot. Neither is
a CPTP map applied event by event, so the density matrix and the Pauli path estimator
decline them and they run on the trajectory engines only. A model holding neither takes
the trajectory loop unchanged, which `MemorySpec::of` decides once per run.

### A leaked qubit in a trajectory

When a qubit leaks, the engine resets it and flips it to `|1>`, then leaves it alone
until it returns. Gates naming it are skipped, other channels skip events naming it, and
a measurement of it reports 1. The reset is one trajectory of the channel
`rho -> |0><0| (x) tr_q rho` (see `Backend::reset`), so averaged over shots the rest of
the register holds exactly `tr_q rho`. After the reset the qubit is an unentangled basis
state:

- Measuring it is a deterministic outcome that collapses nothing else, so the ordinary
  measurement path reports 1, and a classical condition reading the bit sees the same 1.
- A multi-qubit gate it blocks is replaced by one uniformly random Pauli (`I`, `X`, `Y`,
  `Z` at a quarter each) on each unleaked target: complete depolarization.
- Seepage returns the qubit to a uniformly random basis state with one coin and at most
  one `X`.

`ShotsResult::leaked` records, per shot, which qubits leaked at any point: the erasure
herald.

### Quasi-static draws

`QuasiStatic { axis, weights }` rotates its qubit by `sum_k w_k z_k`, where `z` holds
one independent standard normal per source index, drawn by Box-Muller from the shot's
noise stream before its first event. `NoiseBuilder` lowers a `DriftDistribution` with
covariance `Sigma` through its Cholesky factor `L`. Qubit `q`'s offset is row `q` of `L`
against `z`, so the offsets have covariance `L L^T = Sigma` exactly, and detuning scales
the row by each scheduled layer's duration. A singular `Sigma` (perfect correlation)
keeps zero columns where the factor has no variance left.

### Leakage on the compiled QEC path

`QecLeakageSampler` walks the deferred circuit forward per unit of 8192 shots, holding a
packed leak row per alias next to the frame rows. A Pauli frame cannot skip a gate, and
does not need to. A qubit that leaks gets a uniformly random frame Pauli, and every
two-qubit gate meeting a leaked qubit scrambles the frames of both its qubits on the
shots where either is leaked. Since `tr_ab(U rho U^dag) = tr_ab(rho)` for any `U` on the
pair, the rest of the register sees what a trajectory that skips the gate sees. Pauli
annotations replay in program order with the leakage ones, and a firing that names a
leaked qubit is dropped. The leaked qubit's own record is forced to 1 after the frame
flips land.

The result matches trajectory leakage in distribution on a Clifford program. `LEAK`
targets add herald columns, `QecSampleResult::heralds`. Forcing a bit is not a linear
flip, so these programs bypass the record noise and parity projection samplers.

## Error chain complex (`src/sim/homological.rs`)

`ErrorChainComplex` is a GF(2) chain complex over the circuit's noise locations. The
kernel of its boundary map gives the error cycles syndrome measurements cannot detect.
`noisy_marginals_analytical` gives closed-form marginals from the parity matrix and noise
rates, with no Monte Carlo sampling.

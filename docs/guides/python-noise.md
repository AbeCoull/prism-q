# Noise and Shots

This page covers the Python calls; [Noise Models](noise.md) describes what each rule and
channel does, and the [Noisy Simulation tutorial](../tutorials/noisy-simulation.md) works
through a GHZ state under gate noise, readout error and a device calibration.

## Noise

A `NoiseModel` is built from a circuit and sized to it; attaching it to a different
circuit raises `PrismError`.

```python
from prism_q import CircuitBuilder, NoiseChannel, NoiseModel, simulate

circuit = CircuitBuilder(2, 2).h(0).cx(0, 1).measure_all().build()
model = NoiseModel.uniform_depolarizing(circuit, 0.01)
counts = simulate(circuit).seed(42).noise(model).sample_counts(4000).counts()
```

`NoiseModel.uniform_depolarizing(circuit, p)` and
`NoiseModel.with_amplitude_damping(circuit, gamma)` cover the common cases.
`NoiseBuilder` attaches channels by rule and validates the model at `build(circuit)`:

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

Fluent methods update the same builder or filter. A rule copies its filter and channel,
so later filter edits leave it unchanged, and `build` leaves the builder reusable.
`GateFilter()` and `GateFilter.all()` start unrestricted; `named`, `arity`, `on_qubits`
and `on_targets` add restrictions. `on_qubits` selects an unordered set of targets,
while `on_targets([0, 1])` matches the full ordered target list and excludes
`cx(1, 0)`. Build against the original circuit, since fusion changes gate names.

The rules are `after_gates`, `after_gates_joint`, `crosstalk`, `over_rotation`,
`on_idle_qubits`, `after_resets`, `before_measurements`, `readout_error` and
`uniform_readout_error`, with the effects and event ordering listed under
[Rules](noise.md#rules).

For explicit slots, start from `NoiseModel.empty(circuit)` and add events:

```python
model = NoiseModel.empty(circuit)
model.add_event(0, NoiseChannel.amplitude_damping(0.05), [0])
model.add_event(1, NoiseChannel.two_qubit_depolarizing(0.02), [0, 1])
model.with_readout_error(0.01, 0.01)
model.validate()
```

A device calibration table lowers onto a circuit in one call. `DeviceCalibration.parse`
reads the text form in [Device calibration](noise.md#device-calibration); the presets
carry illustrative magnitudes for a technology class, not a measured device:

```python
from prism_q import DeviceCalibration

calibration = DeviceCalibration.superconducting_transmon(circuit.num_qubits)
model = calibration.to_noise_model(circuit)
```

Channels are `pauli(px, py, pz)`, `depolarizing(p)`, `amplitude_damping(gamma)`,
`phase_damping(gamma)`, `thermal_relaxation(t1, t2, gate_time, excited_population=0.0)`,
`two_qubit_depolarizing(p)`, `custom(kraus)` for an explicit list of 2x2 Kraus operators,
`custom_2q(kraus)` for 4x4 operators on two qubits, indexed with the first target as the
high bit, the leakage channels below, and `quasi_static(axis, weights)`, a rotation about
`"x"`, `"y"` or `"z"` by `sum(w * z[source])` over `(source, w)` pairs, with `z` standard
normal draws made once per shot; the drift rules below lower onto it. `num_qubits` is the
number of targets an event with the channel names. `validate()` checks probabilities and
Kraus completeness. `is_pauli_only()` is true only for single-qubit Pauli channels with no
readout error; a model with readout error or `two_qubit_depolarizing` reports `False` yet
still runs on the stabilizer samplers, which apply readout to the record and draw the pair
channel jointly over its 15 branches.

A two-qubit Kraus channel models correlated noise, here `ZZ` dephasing:

```python
import numpy as np

p = 0.02
zz_dephasing = NoiseChannel.custom_2q(
    [np.sqrt(1 - p) * np.eye(4), np.sqrt(p) * np.diag([1, -1, -1, 1])]
)
model = NoiseModel.empty(circuit)
model.add_event(1, zz_dephasing, [0, 1])
model.validate()
```

It runs exactly on the density matrix and by trajectories on the statevector, sparse,
factored, and MPS backends; the stabilizer samplers decline it.

### Leakage

`leakage(p)`, `seepage(p)` and the two-qubit `leakage_transport(p)` give each trajectory
a leak flag per qubit. A leaked qubit ignores gates, a gate pairing it with others
applies a uniformly random Pauli to them, measuring it reports 1, and a reset or a
seepage (to a uniformly random basis state) clears it. `ShotsResult.leaked`, the erasure
herald, is a `(num_shots, num_qubits)` bool array marking qubits that leaked at any
point of the shot, or `None` without leakage:

```python
leaky = (
    NoiseBuilder()
    .after_gates_joint(GateFilter.all().arity(2), NoiseChannel.leakage_transport(0.1))
    .after_gates(GateFilter.all(), NoiseChannel.leakage(0.01))
    .after_gates(GateFilter.all(), NoiseChannel.seepage(0.1))
    .build(circuit)
)
result = simulate(circuit).backend(BackendKind.statevector()).seed(42).noise(leaky).shots(1000)
assert result.leaked.shape == (1000, circuit.num_qubits)
```

Leakage needs a per-shot trajectory, so the density matrix backend rejects it. QEC
programs take `QecNoise.leak(p)`, `seep(p)` and `leak_transport(p)` (text form
`LEAK(p)`, `SEEP(p)`, `LEAK_TRANSPORT(p)`), sampled as heralded erasures with one
`QecSampleResult.heralds` column per `LEAK` target. [Leakage](noise.md#leakage) and
[Leakage as heralded erasure](qec.md#leakage-as-heralded-erasure) describe the two
models.

### Timed and correlated noise

`NoiseBuilder.schedule(GateTimes(one_qubit, two_qubit))` schedules the circuit as soon
as possible, durations in seconds; `with_gate(name, t)`, `with_pair(a, b, t)`,
`with_measure(t)` and `with_reset(t)` refine it, and an `"id"` entry turns identity
gates into delays. These rules read the schedule or draw once per shot:

| Rule | Effect |
|------|--------|
| `scheduled_idle(coherence)` | Thermal relaxation over each qubit's idle time in every layer, from its `(t1, t2)`; replaces `on_idle_qubits` |
| `quasi_static_detuning(drift)` | A per-shot frequency offset per qubit in rad/s, applied as a Z phase over each layer's duration |
| `over_rotation_drift(filter, sigma)` | One per-shot fractional angle error shared by matching `rx`, `ry`, `rz`, `p` gates |
| `over_rotation_drift_per_qubit(filter, drift)` | One per-shot fractional angle error per target qubit |

A `DriftDistribution` is a zero-mean Gaussian over per-qubit offsets, built by
`independent(sigmas)`, `from_t2_star(t2_star)` (widths `sqrt(2) / t2_star`), or
`from_covariance(matrix)`; `with_neighbour_correlation(coupling, rho)` correlates the
two qubits of each coupling edge. Drawn offsets have exactly the requested covariance,
and a matrix that is not positive semidefinite raises at `build`:

```python
drift = DriftDistribution.from_t2_star([40e-6, 55e-6]).with_neighbour_correlation(
    [(0, 1)], 0.5
)
timed = (
    NoiseBuilder()
    .schedule(GateTimes(35e-9, 300e-9).with_measure(1e-6))
    .scheduled_idle([(100e-6, 80e-6), (90e-6, 70e-6)])
    .quasi_static_detuning(drift)
    .build(circuit)
)
counts = simulate(circuit).seed(42).noise(timed).sample_counts(1000).counts()
```

Drift makes every shot a different unitary, so models with it run per-shot
trajectories; models without it keep the shared evolution paths.
`DeviceCalibration.to_scheduled_noise_model(circuit)` is `to_noise_model` plus
`scheduled_idle` timed by the table's own `gate_times()`.

## Compiled sampling

`simulate(circuit).shots(n)` compiles a Clifford circuit's measurements into a parity
sampler on every call. `CompiledSampler` keeps the compiled form across draws and adds
reductions that stream shots in bounded chunks instead of holding them:

```python
from prism_q import CircuitBuilder, CompiledSampler

ghz = CircuitBuilder(3, 3).h(0).cx(0, 1).cx(1, 2).measure_all().build()
sampler = CompiledSampler(ghz, seed=7)
shots = sampler.sample(1000)                    # (1000, 3) bool
counts = sampler.sample_counts(10**9)           # closed form at small rank
marginals = sampler.marginals(10**6)            # P(record reads 1)
parities = sampler.parity_expectations([[0, 1], [0, 1, 2]], 10**6)
correlators = sampler.correlators([(0, 2)], 10**6)
```

Columns, count keys, and indices are measurement records in circuit order: record `j`
is the `j`-th measurement, and `circuit.measurement_map()[j]` names its qubit and
classical bit; `shots()` reports classical bits instead. Calls continue one seeded
stream, so successive calls draw fresh shots and a new sampler with the same seed
replays them. `rank` counts the independent random bits behind the record, and
`exact_counts()` enumerates all `2 ** rank` outcomes up to rank 25. `sample_packed(n)`
returns records eight to a byte in the `QecSampleResult.packed_measurements()` layout.

`parity_expectations(rows, n)` estimates `<(-1)^parity>` for each row of record indices,
the expectation of the Z-type Pauli product those records measure; `correlators(pairs,
n)` is the two-record case.

`noise=model` compiles a Pauli noise model, readout error included, into the sampler;
`rank` and `exact_counts()` are then `None`. The circuit must be Clifford with terminal
measurements and no reset or classical condition; other input raises `PrismError`
naming the reason.

```python
from prism_q import ErrorChainComplex, NoiseModel, noisy_marginals_analytical

model = NoiseModel.uniform_depolarizing(ghz, 0.01)
noisy = CompiledSampler(ghz, seed=7, noise=model)
estimate = noisy.marginals(10**6)
exact = noisy_marginals_analytical(ghz, model)
logical_classes = ErrorChainComplex(ghz, model).homology_dim
```

`noisy_marginals_analytical(circuit, model)` returns the exact probability that each
classical bit reads 1 under Pauli noise, without sampling or a rank limit.
`ErrorChainComplex(circuit, model)` exposes the GF(2) chain complex behind it:
`homology_dim` counts the independent logical error classes, `boundary_dim` the
independent errors that leave the state unchanged and that no final measurement
detects, and `noisy_marginals(noiseless)` maps noiseless record marginals to noisy
ones.

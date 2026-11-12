# Noise Models

A `NoiseModel` attaches channels to the instructions of one circuit. The trajectory
backends sample it shot by shot, the density matrix evolves it exactly, and on a
Clifford circuit with Pauli noise the [compiled samplers](../architecture/samplers.md)
draw it with no state per shot. [Noise internals](../architecture/noise.md) covers how
each route runs a model, the [Noisy Simulation tutorial](../tutorials/noisy-simulation.md)
walks through one, and [Noise and Shots](python-noise.md) is the Python version of this
page.

## Attaching a model

```rust
use prism_q::{simulate, BackendKind, NoiseModel};

# let circuit = prism_q::CircuitBuilder::new_with_classical(3, 3).h(0).cx(0, 1).cx(1, 2).measure_all().build();
let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
let result = simulate(&circuit)
    .backend(BackendKind::Statevector)
    .noise(&noise)
    .seed(42)
    .shots(1024)
    .unwrap();
```

A model holds a list of `NoiseEvent { channel, qubits }` per instruction slot, so it is
sized to the circuit it was built from, and attaching it to another circuit is an error.
Channels are `Pauli`, `Depolarizing`, `AmplitudeDamping`, `PhaseDamping`,
`ThermalRelaxation`, `TwoQubitDepolarizing`, `Custom` and `Custom2q` Kraus sets, the
leakage channels below, and `QuasiStatic`. Readout error sits beside the events and flips
reported bits after sampling. On a Clifford circuit the noisy compiled sampler propagates
each event's sensitivity row once and XORs the fired rows into every shot.

## Rules

`NoiseBuilder` places channels by rule and checks the result at `build(&circuit)`.
`GateFilter::all()` matches every gate; `named`, `arity`, `on_qubits` and `on_targets`
narrow it. `on_qubits` is an unordered set, while `on_targets([0, 1])` matches the ordered
target list and so skips `cx(1, 0)`.

```rust
use prism_q::{GateFilter, NoiseBuilder, NoiseChannel};

# let circuit = prism_q::CircuitBuilder::new_with_classical(3, 3).h(0).cx(0, 1).cx(1, 2).measure_all().build();
let noise = NoiseBuilder::new()
    .after_gates(GateFilter::all().arity(1), NoiseChannel::Depolarizing { p: 1e-3 })
    .after_gates_joint(
        GateFilter::all().named("cx"),
        NoiseChannel::TwoQubitDepolarizing { p: 1e-2 },
    )
    .crosstalk(
        GateFilter::all().named("cx"),
        [(0, 1), (1, 2)],
        NoiseChannel::Depolarizing { p: 1e-4 },
    )
    .uniform_readout_error(0.01, 0.02)
    .build(&circuit)?;
# Ok::<(), prism_q::PrismError>(())
```

| Rule | Effect |
|------|--------|
| `after_gates(filter, channel)` | One single-qubit event per matching target |
| `after_gates_joint(filter, channel)` | One event on the whole target list when its arity matches the channel |
| `crosstalk(filter, coupling, channel)` | Spectator noise along undirected coupling edges; a two-qubit channel takes the gate target first |
| `over_rotation(filter, relative)` | An extra `relative * theta` rotation after `rx`, `ry`, `rz` or `p` |
| `on_idle_qubits(channel)` | One event per idle qubit per greedy circuit layer |
| `after_resets(channel)` | One event on each reset qubit |
| `before_measurements(channel)` | Damage to the measured state, which feedforward also reads |
| `readout_error(bit, p01, p10)` | Readout rates for one classical bit, overriding `uniform_readout_error`; the bit flips when a measurement writes it, so a condition reads the noisy value |

Events in one instruction slot follow rule order. Idle events attach to the
highest-indexed instruction of their layer and pre-measurement events to the instruction
before the measurement, so a measurement at instruction zero needs a barrier ahead of it.
Gate rules skip conditional instructions. Build against the circuit as written, since
fusion renames gates.

## Device calibration

A `DeviceCalibration` holds what a calibration page reports: `t1`, `t2` and readout rates
per qubit, a duration and depolarizing error per gate family, and optional two-qubit
entries per pair. `to_noise_model(&circuit)` lowers it onto a circuit. After every gate,
each target gets a `ThermalRelaxation` channel with its own `t1` and `t2` over the family's
duration, and a nonzero family error adds `Depolarizing` on a one-qubit target or
`TwoQubitDepolarizing` on the pair. Each measurement takes its classical bit's readout
error from the measured qubit. Gates on more than two qubits are rejected; decompose them
first.

The text form has one record per line, `key=value` fields in any order, and `#`
comments:

| Record | Fields | Notes |
| --- | --- | --- |
| `qubit <n> ...` | `t1`, `t2` in seconds; `p01`, `p10` (default 0) | one per qubit, indices `0..N` in any order |
| `gate1q ...` | `time` in seconds, `error` | exactly once |
| `gate2q ...` | `time`, `error` | exactly once |
| `gate2q <a> <b> ...` | `time`, `error` | optional; replaces the family on that pair in either order |

Values are checked on read (`t2 <= 2 t1`, probabilities in `[0, 1]`, durations
positive), and an error names the line and the field.

```rust
use prism_q::{simulate, BackendKind, DeviceCalibration};

# let circuit = prism_q::CircuitBuilder::new_with_classical(2, 2).h(0).cx(0, 1).measure_all().build();
let calibration = DeviceCalibration::parse(
    "qubit 0 t1=120e-6 t2=80e-6 p01=0.02 p10=0.03
     qubit 1 t1=95e-6 t2=110e-6 p01=0.01 p10=0.02
     gate1q time=35e-9 error=3e-4
     gate2q time=300e-9 error=8e-3
     gate2q 0 1 time=250e-9 error=5e-3",
)
.unwrap();
let noise = calibration.to_noise_model(&circuit).unwrap();
let result = simulate(&circuit)
    .backend(BackendKind::DensityMatrix)
    .noise(&noise)
    .seed(42)
    .shots(1024)
    .unwrap();
```

`prism_q::sim::calibration::presets` has `superconducting_transmon(n)`, `trapped_ion(n)`
and `neutral_atom(n)`: illustrative magnitudes for a technology class, the same on every
qubit. A specific device needs the text form.

`to_noise_model` charges decoherence only to the qubits a gate touches.
`to_scheduled_noise_model(&circuit)` also decays the qubits that wait, using the table's
gate durations. The circuit is laid out as soon as possible in the greedy layers
`Circuit::depth` counts, each layer as long as its longest gate, and at the end of each
layer every qubit relaxes with its own `t1` and `t2` over the part of the layer it spent
idle. `gate_times()` returns those durations as a `GateTimes` for a builder.

## Leakage

A leakage channel moves a qubit out of the computational subspace. The trajectory
engines carry it as a flag per qubit:

| Channel | Effect |
| --- | --- |
| `Leakage { p }` | An unleaked qubit leaks with probability `p` |
| `Seepage { p }` | A leaked qubit returns with probability `p`, to a uniformly random basis state |
| `LeakageTransport { p }` | Two-qubit: when exactly one qubit is leaked, the other leaks with probability `p` |

Gates on a leaked qubit do not act, and a gate pairing it with unleaked qubits applies a
uniformly random Pauli to each of those instead. Measuring a leaked qubit reports 1, a
reset clears the flag, and every other channel skips it. `ShotsResult::leaked` holds, per
shot, the qubits that leaked at any point: an erasure herald for a decoder.

```rust
use prism_q::{simulate, BackendKind, GateFilter, NoiseBuilder, NoiseChannel};

# let circuit = prism_q::CircuitBuilder::new_with_classical(3, 3).h(0).cx(0, 1).cx(1, 2).measure_all().build();
let noise = NoiseBuilder::new()
    .after_gates_joint(GateFilter::all().arity(2), NoiseChannel::LeakageTransport { p: 0.1 })
    .after_gates(GateFilter::all(), NoiseChannel::Leakage { p: 1e-3 })
    .after_gates(GateFilter::all(), NoiseChannel::Seepage { p: 0.05 })
    .build(&circuit)
    .unwrap();
let result = simulate(&circuit)
    .backend(BackendKind::Statevector)
    .noise(&noise)
    .seed(42)
    .shots(1024)
    .unwrap();
let heralds = result.leaked.as_ref().unwrap();
assert_eq!(heralds[0].len(), 3);
```

A transport event reads the flags when it fires, so declare it before the leakage rule on
the same gates to spread only the leakage the gate itself saw. The density matrix rejects
leakage, since a mixture evolved event by event has no flag to carry. QEC programs model
leakage as heralded erasure instead; see [QEC](qec.md#leakage-as-heralded-erasure).

## Correlated and time-dependent noise

`NoiseBuilder::schedule(GateTimes)` times the circuit for the rules that act over elapsed
time. `GateTimes::new(one_qubit, two_qubit)` takes family durations in seconds;
`with_gate(name, t)` overrides one gate (an `"id"` entry turns identities into delays),
`with_pair(a, b, t)` one qubit pair, and `with_measure` and `with_reset` the instructions
that are not gates.

- `scheduled_idle(coherence)` relaxes each qubit from its `(t1, t2)` over the idle part
  of every layer. It replaces `on_idle_qubits`; declaring both is an error.
- `quasi_static_detuning(drift)` draws one frequency offset per qubit per shot, in
  radians per second, and turns each qubit about `Z` by its offset times every layer's
  duration. A qubit left for time `t` then dephases on average as `exp(-(sigma t)^2 / 2)`.
- `over_rotation_drift(filter, sigma)` draws one fractional angle error per shot, shared
  by every matching `rx`, `ry`, `rz` or `p` gate; `over_rotation_drift_per_qubit(filter,
  drift)` draws one per qubit.

A `DriftDistribution` is a zero-mean Gaussian over per-qubit offsets:
`independent(sigmas)`, `from_t2_star(t2_star)` for widths `sqrt(2) / t2_star`, a full
`from_covariance(matrix)`, or `with_neighbour_correlation(coupling, rho)` to correlate the
qubits of each coupling edge. Each shot draws standard normals and correlates them through
the covariance's Cholesky factor. `build` rejects a covariance that is not positive
semidefinite.

```rust
use prism_q::{DriftDistribution, GateFilter, GateTimes, NoiseBuilder};

# let circuit = prism_q::CircuitBuilder::new_with_classical(3, 3).rx(0.4, 0).cx(0, 1).cx(1, 2).measure_all().build();
let detuning = DriftDistribution::from_t2_star([40e-6, 55e-6, 30e-6])
    .with_neighbour_correlation([(0, 1), (1, 2)], 0.3);
let noise = NoiseBuilder::new()
    .schedule(GateTimes::new(35e-9, 300e-9).with_measure(1e-6))
    .scheduled_idle([(100e-6, 80e-6); 3])
    .quasi_static_detuning(detuning)
    .over_rotation_drift(GateFilter::all().named("rx"), 0.01)
    .build(&circuit)
    .unwrap();
```

The drift rules lower onto `NoiseChannel::QuasiStatic` events, which make each shot a
different unitary, so they run on the per-shot trajectory engines only. A model without
them keeps the shared evolution paths.

## Exact noisy marginals

On a Clifford circuit with single-qubit Pauli noise, `noisy_marginals_analytical` returns
the probability that each classical bit reads 1, in closed form from the parity matrix
and the noise rates, with no sampling and no rank limit. Readout error is part of that
closed form: a reported 1 is a measured 1 that did not flip or a measured 0 that did. Any
other channel, and any channel naming two qubits, is rejected.

`ErrorChainComplex::build` exposes the GF(2) chain complex behind it, over noise
locations, measurements and stabilizers. `homology_dim` counts the independent logical
error classes, `boundary_dim` the errors that leave the state unchanged and that no final
measurement detects, and `noisy_marginals(noiseless)` maps noiseless record marginals to
noisy ones.

```rust
use prism_q::{CircuitBuilder, ErrorChainComplex, NoiseModel, noisy_marginals_analytical};

let ghz = CircuitBuilder::new_with_classical(3, 3).h(0).cx(0, 1).cx(1, 2).measure_all().build();
let noise = NoiseModel::uniform_depolarizing(&ghz, 0.01);
let marginals = noisy_marginals_analytical(&ghz, &noise, 42)?;
assert!(marginals.iter().all(|p| (p - 0.5).abs() < 1e-9));
let complex = ErrorChainComplex::build(&ghz, &noise, 42)?;
let logical_classes = complex.homology_dim();
# Ok::<(), prism_q::PrismError>(())
```

## Gates compiled from pulses

The `unstable-pulse` feature adds `prism_q::pulse`, which integrates a pulse schedule on
a model of one to three transmons once and hands the result to the gate level. It is
closed-system only and outside the [stability promise](compatibility.md#feature-flags):
its items may change in any release.

```bash
cargo add prism-q --features unstable-pulse
```

```rust,ignore
use prism_q::pulse::{compile, Device, Envelope, Schedule, Transmon};
use prism_q::{Circuit, Gate};

let device = Device::new([Transmon { frequency: 5e9, anharmonicity: -300e6, levels: 3 }]);
let mut schedule = Schedule::new(1e-9);
let drag = Envelope::Drag { amp: 50e6, sigma: 5e-9, beta: 1.0 / (2.0 * std::f64::consts::PI * 300e6) };
schedule.play(0, 0.0, 20e-9, drag);
let gate = compile(&device, &schedule, 16)?;
let mut circuit = Circuit::new(1, 0);
circuit.add_unitary(gate.unitary(), &[0])?;
let residual = gate.error_after(&Gate::X);
```

Frequencies and amplitudes are in Hz and times in seconds. The three outputs:

- `unitary()`: the nearest unitary to the propagator restricted to `|0>` and `|1>` on
  every site, packed for `Gate::unitary` and `Circuit::add_unitary` with site 0 on the
  first target.
- `leakage()`: per site, the probability of leaving `{|0>, |1>}` from `|0>` and from
  `|1>`. These can parameterize the `Leakage` channel above.
- `error_after(ideal)`: the coherent difference between the pulse and the gate it stands
  for, as a one-operator `Custom` or `Kraus2q` channel to attach after that gate with a
  `NoiseBuilder`.

`evolve` returns the populations after each sample period instead, for Rabi or leakage
curves.

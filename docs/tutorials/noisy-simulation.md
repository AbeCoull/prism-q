# Noisy Simulation

Goal: watch a GHZ state degrade under gate noise and readout error, first from rules you
write and then from a device calibration table, and compare sampled and exact answers.

The figure of merit throughout is the GHZ fraction: the share of shots reading `000` or
`111`.

```python
from prism_q import CircuitBuilder, simulate

circuit = CircuitBuilder(3, 3).h(0).cx(0, 1).cx(1, 2).measure_all().build()


def ghz_fraction(counts):
    return (counts.get("000", 0) + counts.get("111", 0)) / sum(counts.values())


ideal = simulate(circuit).seed(42).sample_counts(10_000).counts()
print(ghz_fraction(ideal))                    # 1.0
```

## Noise from rules

`NoiseBuilder` attaches channels by rule. Each rule pairs a `GateFilter`, which picks the
gates, with a `NoiseChannel`. `build(circuit)` lowers the rules onto that circuit and
validates the result.

```python
from prism_q import GateFilter, NoiseBuilder, NoiseChannel

noise = (
    NoiseBuilder()
    .after_gates(GateFilter.all().arity(1), NoiseChannel.depolarizing(0.001))
    .after_gates_joint(GateFilter.all().named("cx"), NoiseChannel.two_qubit_depolarizing(0.02))
    .uniform_readout_error(0.01, 0.02)
    .build(circuit)
)
noisy = simulate(circuit).seed(42).noise(noise).sample_counts(10_000).counts()
print(round(ghz_fraction(noisy), 4))          # 0.9273
```

`after_gates` adds a one-qubit channel on each target of a matching gate.
`after_gates_joint` adds one channel across all of a gate's targets, here a two-qubit
depolarizing channel after every `cx`. `uniform_readout_error(p01, p10)` flips reported
bits after sampling: `p01` is the chance a measured 0 reads as 1. The
[Python guide](../guides/python.md#noise) lists the other rules: crosstalk,
over-rotation, idle noise, and noise after resets or before measurement.

A model is tied to the circuit it was built from. Build a new one for a different circuit.

## Noise from a device calibration

`DeviceCalibration` reads what a device's calibration page reports: `t1`, `t2` and
readout rates per qubit, a duration and error per gate family, and optional per-pair
overrides. `to_noise_model` turns it into thermal relaxation over each gate's duration,
depolarizing error per gate, and readout error per measured qubit.

```python
from prism_q import DeviceCalibration

calibration = DeviceCalibration.parse(
    """
    qubit 0 t1=120e-6 t2=80e-6 p01=0.02 p10=0.03
    qubit 1 t1=95e-6 t2=110e-6 p01=0.01 p10=0.02
    qubit 2 t1=60e-6 t2=50e-6 p01=0.02 p10=0.04
    gate1q time=35e-9 error=3e-4
    gate2q time=300e-9 error=8e-3
    gate2q 1 2 time=450e-9 error=2e-2
    """
)
device = calibration.to_noise_model(circuit)
sampled = simulate(circuit).seed(42).noise(device).sample_counts(10_000).counts()
print(round(ghz_fraction(sampled), 4))        # 0.9028
```

Times are in seconds. The last line overrides the two-qubit family on the pair `(1, 2)`,
in either order. `DeviceCalibration.superconducting_transmon(n)`, `trapped_ion(n)` and
`neutral_atom(n)` are presets with illustrative magnitudes for a technology class, not a
measured device.

## Exact answers from the density matrix

Sampling carries shot noise. The density-matrix backend evolves the mixed state itself,
so `run()` returns exact probabilities under the model. It never runs by default, since it
stores `4**n` entries; select it by name. Measurements and readout error are left off here
to read the state before measurement:

```python
from prism_q import BackendKind

unmeasured = CircuitBuilder(3).h(0).cx(0, 1).cx(1, 2).build()
exact = (
    simulate(unmeasured)
    .backend(BackendKind.density_matrix())
    .noise(calibration.to_noise_model(unmeasured))
    .seed(42)
    .run()
    .probabilities
)
print(round(exact[0] + exact[7], 4))          # 0.9712
```

The gap between this and the sampled 0.90 is mostly readout error, which acts only on the
reported bits.

## In Rust

```rust
use prism_q::{CircuitBuilder, GateFilter, NoiseBuilder, NoiseChannel, simulate};

let circuit = CircuitBuilder::new_with_classical(3, 3)
    .h(0)
    .cx(0, 1)
    .cx(1, 2)
    .measure_all()
    .build();
let noise = NoiseBuilder::new()
    .after_gates(GateFilter::all().arity(1), NoiseChannel::Depolarizing { p: 0.001 })
    .after_gates_joint(
        GateFilter::all().named("cx"),
        NoiseChannel::TwoQubitDepolarizing { p: 0.02 },
    )
    .uniform_readout_error(0.01, 0.02)
    .build(&circuit)?;
let counts = simulate(&circuit).noise(&noise).seed(42).sample_counts(10_000)?;
let good = counts.counts.get(&vec![0b000]).unwrap_or(&0)
    + counts.counts.get(&vec![0b111]).unwrap_or(&0);
assert!(good > 9_000);
# Ok::<(), prism_q::PrismError>(())
```

Rust count keys are packed words with classical bit 0 in the lowest bit, so `000` and
`111` are `vec![0b000]` and `vec![0b111]`.
[`examples/noise.rs`](https://github.com/AbeCoull/prism-q/blob/main/examples/noise.rs)
adds the calibration and density-matrix steps.

Next: [A QEC Memory Experiment](./qec-memory.md).

# Noise and QEC

PRISM-Q models noise and quantum error correction through the
[compiled samplers](../architecture/samplers.md) and the
[native QEC program IR](../architecture/qec-ir.md), without a dense statevector per shot.

## Noisy shot sampling

Attach a `NoiseModel` and sample:

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

`NoiseModel` holds a list of `NoiseEvent { channel, qubits }` per instruction, each channel
a `NoiseChannel`: Pauli, depolarizing, readout error, amplitude damping and others. On
Clifford circuits the noisy compiled sampler propagates noise sensitivity rows and XORs the
fired channels into each sample, with no per-shot state evolution.

## Device calibration import

A `DeviceCalibration` holds what a calibration page reports: `t1`, `t2` and readout rates
per qubit, a duration and depolarizing error per gate family, and optional two-qubit
entries per pair. `to_noise_model(&circuit)` lowers it onto a circuit. After every gate,
each target gets a `ThermalRelaxation` channel with its own `t1` and `t2` over the family's
duration; a nonzero family error adds `Depolarizing` on a one-qubit target or
`TwoQubitDepolarizing` on the pair; each measurement takes its classical bit's readout
error from the measured qubit. Gates on more than two qubits are rejected, so decompose
them first.

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
qubit, not a measured device. Write the text form for a specific device.

`to_noise_model` charges decoherence only to the qubits a gate touches.
`to_scheduled_noise_model(&circuit)` also decays the qubits that wait, using the table's
gate durations: the circuit is laid out as soon as possible in the greedy layers
`Circuit::depth` counts, each layer as long as its longest gate, and at the end of each
layer every qubit relaxes with its own `t1` and `t2` over the part of the layer it spent
idle. `gate_times()` returns those durations as a `GateTimes` for a builder.

## Leakage

A leakage channel moves a qubit out of the computational subspace. The trajectory
engines carry it as a per-qubit flag:

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
leakage, since a mixture evolved event by event has no flag to carry.

## Correlated and time-dependent noise

`NoiseBuilder::schedule(GateTimes)` times the circuit for the rules that act over elapsed
time. `GateTimes::new(one_qubit, two_qubit)` takes family durations in seconds;
`with_gate(name, t)` overrides one gate (an `"id"` entry turns identities into delays),
`with_pair(a, b, t)` one qubit pair, and `with_measure` and `with_reset` the non-gate
instructions.

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
the covariance's Cholesky factor; a covariance that is not positive semidefinite is
rejected at `build`.

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
them keeps the shared-evolution fast paths.

## Detector sampling

`compile_detector_sampler` compiles a Clifford circuit with measurement and reset reuse,
the shape of repeated syndrome extraction, into a packed sampler, and derives detector and
observable records as parity rows over the measurement record. Reset reuse becomes fresh
qubit aliases, so no shot replays the tableau.

## Native QEC programs

For detectors, logical observables, postselection and Pauli-noise annotations as
first-class constructs, use the native QEC program IR instead of a `Circuit`. Its text
format follows the Stim circuit language for the instructions listed below; `LEAK`,
`SEEP`, `LEAK_TRANSPORT` and `EXP_VAL` are additions, and any other instruction is a
parse error.

```rust
use prism_q::{parse_qec_program, run_qec_program};

let qec_text = "
    R 0 1 2 3 4
    X_ERROR(0.01) 0 2 4
    CX 0 1 2 1 2 3 4 3
    MR 1 3
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    M 0 2 4
    OBSERVABLE_INCLUDE(0) rec[-1]
";
let program = parse_qec_program(qec_text).unwrap();
let result = run_qec_program(&program).unwrap();
```

`run_qec_program` lowers Clifford-compatible programs into the packed compiled sampler;
`run_qec_program_reference` is the per-shot statevector oracle for validating small
programs. `program.to_text()` writes the same text format back, so a program built in code
can be saved and parsed again.

```admonish info title="What QEC programs support"
Clifford gates (including the `SQRT_*`, `C_XYZ` / `C_ZYX`, `CY`, `SWAP`, `ISWAP`,
`CXSWAP`, and `SWAPCX` spellings), basis resets and measurements, `MPP` and
`MXX` / `MYY` / `MZZ` Pauli-product measurements, `MPAD` constant records,
detectors, observables, postselection, `X_ERROR` / `Y_ERROR` / `Z_ERROR` /
`DEPOLARIZE1` / `DEPOLARIZE2` / `PAULI_CHANNEL_1` / `PAULI_CHANNEL_2` noise,
`E` / `CORRELATED_ERROR` chains with `ELSE_CORRELATED_ERROR`, `HERALDED_ERASE`,
`LEAK` / `SEEP` / `LEAK_TRANSPORT` leakage, and terminal `EXP_VAL` final-state
expectation estimates.
A noiseless `EXP_VAL` uses the analytical T strategies, with any detector records
still sampled by the packed runner. A noisy one is computed exactly on the density
matrix when it fits, and falls to the per-shot reference runner when the program
carries measurement records or postselection or exceeds the density-matrix cap.
The packed sampling path rejects non-Clifford gates.
The [QEC IR reference](../architecture/qec-ir.md) has the full grammar;
[QEC program execution](../architecture/qec-programs.md) has the runner routing, the
V1 reset requirement, and the `EXP_VAL` placement rules.
```

### Reading Stim circuit files

`parse_stim_circuit` (or `QecProgram::from_stim`, `QecProgram.from_stim` in Python)
reads `.stim` circuit text, such as the surface, repetition and color code memory
circuits other QEC tooling generates, into the same program:

```rust
use prism_q::{parse_stim_circuit, run_qec_program};

let program = parse_stim_circuit("
    R 0 1 2
    X_ERROR(0.01) 0 1 2
    CX 0 1 2 1
    MR 1
    DETECTOR(1, 0) rec[-1]
    M !0 2
    OBSERVABLE_INCLUDE(0) rec[-2]
").unwrap();
let result = run_qec_program(&program).unwrap();
```

It accepts the instruction set above apart from the native additions (`T`, `T_DAG`,
`LEAK`, `SEEP`, `LEAK_TRANSPORT`, `POSTSELECT`, `EXP_VAL`), and adds three Stim
behaviours the native parser rejects or ignores:

- inverted measurement targets (`M !0`, `MR !0`, `MXX !0 1`, `MPP !X0*Z1`), lowered as
  a Pauli on either side of the measurement so only the record flips; the second Pauli
  waits for the qubit's next use and is dropped at a reset, so a qubit measured once
  stays on the compiled sampler;
- record-controlled Paulis (`CX rec[-1] 2`, `CY rec[-1] 2`, `CZ rec[-1] 2`), lowered to
  feed-forward corrections;
- `SHIFT_COORDS`, which offsets later `DETECTOR` coordinates.

Any other instruction, `sweep[k]` targets, and noise arguments on `MPP`, the pair
measurements and `MPAD` are parse errors naming the line. `M(p)` flips the qubit before
measuring, so it matches Stim's record-only flip when the qubit is reset or not measured
again; generated memory circuits spell measurement noise as a separate `X_ERROR`.

### Detector reference

A detector reports 1 when its records disagree with the noiseless reference run of the
program, so a noiseless program never fires and a detector event always means noise
flipped it. Observables and logical error counts follow the same convention. The
reference run takes 0 for every random measurement outcome, including the hidden collapse
of a reset. A detector whose noiseless value is random is still XORed with that run's
value: it fires in about half the shots, and `detector_error_model` rejects the program.
Postselection and heralds read the raw records, and programs with non-Clifford gates
report raw parities.

`QecOptions::detector_reference` selects the convention. `QecDetectorReference::Raw`
(`detector_reference="raw"` in Python) reports the plain parity of each detector's
records instead:

```rust
use prism_q::{QecDetectorReference, QecOptions, parse_qec_program, run_qec_program};

let mut program = parse_qec_program(
    "R 0
     X 0
     M 0
     DETECTOR rec[-1]
     OBSERVABLE_INCLUDE(0) rec[-1]",
)?;
assert_eq!(run_qec_program(&program)?.logical_error_rates(), vec![0.0]);

program.set_options(QecOptions {
    detector_reference: QecDetectorReference::Raw,
    ..QecOptions::default()
});
assert_eq!(run_qec_program(&program)?.logical_error_rates(), vec![1.0]);
# Ok::<(), prism_q::PrismError>(())
```

### Compiling once for repeated sampling

`run_qec_program` compiles the program on every call. `QecProgram::compile_sampler`
compiles once and returns a `QecCompiledSampler` whose `sample` call only samples, so many
small batches or a sweep over seeds pay for one compile:

```rust
use prism_q::parse_qec_program;

let program = parse_qec_program(
    "R 0 1 2
     X_ERROR(0.01) 0 2
     CX 0 1 2 1
     M 0 1 2
     DETECTOR rec[-2]
     OBSERVABLE_INCLUDE(0) rec[-1]",
)?;
let mut sampler = program.compile_sampler()?;
let first = sampler.sample(100, None)?;
let seeded = sampler.sample(100, Some(7))?;
# Ok::<(), prism_q::PrismError>(())
```

`sample(shots, Some(seed))` draws exactly the shots `run_qec_program` draws with that seed
in the program's options. `sample(shots, None)` continues one stream: the first call uses
the program's seed and each later call derives a fresh seed from the previous one, so two
samplers compiled from the same program replay the same sequence. The sampler is `Send`
and `sample` takes `&mut self`, so share one behind a lock or compile one per thread.
Programs that need the per-shot or analytical runners (`EXP_VAL`, non-Clifford gates)
return an error from `compile_sampler`.

In Python, `QecProgram.compile_sampler()` returns a `QecCompiledSampler`.
`sample(shots, *, seed=None, bit_packed=True, separate_observables=True)` returns a
`(detectors, observables)` pair of `(shots, ceil(n / 8))` little-endian byte arrays; bool
arrays with `bit_packed=False`; or, with `separate_observables=False`, one array whose
columns are the detectors followed by the observables. `sample_result` returns a full
`QecSampleResult`. Sampling releases the GIL, and calls on one sampler from several
threads run one at a time. Measurement records are dropped unless
`compile_sampler(keep_measurements=True)` asks for them, which `sample_result` then
reports.

### Leakage as heralded erasure

`LEAK(p) q ...` leaks each target with probability `p`, `SEEP(p) q ...` returns each
leaked target to a random basis state with probability `p`, and
`LEAK_TRANSPORT(p) a b ...` spreads leakage across each pair at rate `p`.
`run_qec_program` samples them as erasures: a leaking qubit gets a uniformly random Pauli
frame, a two-qubit gate meeting a leaked qubit fully depolarizes both its qubits, other
noise annotations naming a leaked qubit are skipped, and the leaked qubit's own
measurement reports 1. On a Clifford program this matches the trajectory leakage model
above in distribution.

Each `LEAK` target adds one column to `QecSampleResult::heralds`, set when that qubit is
leaked just after the annotation; `LEAK(0)` reads the flags without changing them. An
`MPP` over a leaked qubit reports a random parity. The detector error model and the
reference runner have no leak flag and reject these programs, and the memory-experiment
generators below emit no leakage.

```rust
use prism_q::{QecProgram, run_qec_program};

let program = QecProgram::from_text(
    "R 0 1 2
     CX 0 1
     LEAK(0.01) 0 1
     CX 1 2
     LEAK_TRANSPORT(0.1) 1 2
     M 0 1 2
     DETECTOR rec[-1] rec[-2]",
)?;
let result = run_qec_program(&program)?;
let heralds = result.heralds.as_ref().unwrap();
assert_eq!(heralds.num_measurements(), 2);
# Ok::<(), prism_q::PrismError>(())
```

## Memory experiments

`QecProgram::repetition_memory`, `surface_memory` and `color_memory` build the standard
memory experiments with circuit-level noise from a `QecCircuitNoise`: depolarization after
every Clifford gate, a flip before every measurement and after every reset, and
depolarization of the data at the start of each round. Each program carries detectors
with coordinates and logical observable 0:

```rust
use prism_q::{QecBasis, QecCircuitNoise, QecProgram, UnionFindDecoder, run_qec_program};

let noise = QecCircuitNoise::uniform(0.002);
let program = QecProgram::surface_memory(3, 3, QecBasis::Z, &noise)?;
let model = program.detector_error_model()?.decompose_graphlike()?;
let decoder = UnionFindDecoder::from_model(&model)?;
let result = run_qec_program(&program)?;
let predicted = decoder.decode_packed(&result.detectors)?;
let failures = (0..result.total_shots)
    .filter(|&shot| predicted.get_bit(shot, 0) != result.observables.get_bit(shot, 0))
    .count();
# Ok::<(), prism_q::PrismError>(())
```

The repetition code protects a Z memory; the rotated surface and triangular 6.6.6 color
codes take an X or Z memory basis. A zero rate adds no annotation, so
`QecCircuitNoise::default()` gives the noiseless program, whose detectors and observable
never fire. In Python the generators are static methods of `QecProgram` with an optional
`QecCircuitNoise`. The [QEC IR reference](../architecture/qec-ir.md) has the layouts and
schedules.

## Detector error model export

Matching and belief-propagation decoders read an error model, not raw detector samples.
`QecProgram::detector_error_model` derives one from the program's noise annotations,
detectors and observables, and `to_text` writes it in the detector error model text
format external decoders read:

```rust,no_run
# let program = prism_q::parse_qec_program(
#     "R 0 1 2 3 4
#      X_ERROR(0.01) 0 2 4
#      CX 0 1 2 1 2 3 4 3
#      MR 1 3
#      DETECTOR rec[-2]
#      DETECTOR rec[-1]
#      M 0 2 4
#      OBSERVABLE_INCLUDE(0) rec[-1]",
# )?;
let model = program.detector_error_model().unwrap();
std::fs::write("memory_d3.dem", model.to_text()).unwrap();
# Ok::<(), prism_q::PrismError>(())
```

Each mechanism carries a probability and the detector and observable indices it flips;
detector coordinates pass through from the program. In Python the model also exposes
`probabilities()`, `detector_matrix()` and `observable_matrix()`, the check-matrix triple
in-process decoder libraries accept. Matching decoders need at most two detectors per
mechanism: `decompose_graphlike` splits each hypergraph mechanism across existing
graphlike ones and returns an error when no split exists, and `to_text` writes the splits
as `^`-separated components. Every detector and observable must be deterministic in the
noiseless program, since a random one has no error mechanism to describe it;
`detector_error_model` names it in the error.

`DetectorErrorModel::from_text` reads the same format, `repeat` blocks and
`shift_detectors` included, so a model written by another tool can drive the in-crate
decoder:

```rust
use prism_q::{DetectorErrorModel, UnionFindDecoder};

let model = DetectorErrorModel::from_text(
    "error(0.1) D0
     repeat 2 {
         error(0.1) D0 D1
         shift_detectors 1
     }
     error(0.1) D0 L0",
)?;
assert_eq!(model.num_detectors(), 3);
let decoder = UnionFindDecoder::from_model(&model)?;
# Ok::<(), prism_q::PrismError>(())
```

[QEC program execution](../architecture/qec-programs.md) has the derivation semantics and
the grammar in both directions.

## Decoding

`UnionFindDecoder` decodes sampled detectors against a graphlike model in-process, so a
memory experiment's logical error rate needs no external tool:

```rust
use prism_q::{UnionFindDecoder, run_qec_program};

# let program = prism_q::parse_qec_program(
#     "R 0 1 2 3 4
#      X_ERROR(0.01) 0 2 4
#      CX 0 1 2 1 2 3 4 3
#      MR 1 3
#      DETECTOR rec[-2]
#      DETECTOR rec[-1]
#      M 0 2 4
#      OBSERVABLE_INCLUDE(0) rec[-1]",
# )?;
let model = program.detector_error_model()?.decompose_graphlike()?;
let decoder = UnionFindDecoder::from_model(&model)?;
let result = run_qec_program(&program)?;
let predicted = decoder.decode_packed(&result.detectors)?;
let failures = (0..result.total_shots)
    .filter(|&shot| predicted.get_bit(shot, 0) != result.observables.get_bit(shot, 0))
    .count();
# Ok::<(), prism_q::PrismError>(())
```

The decoder is weighted union-find with peeling: edges weigh `ln((1-p)/p)`, one-detector
mechanisms are boundary edges, and mechanisms that flip no detector set a floor under the
achievable logical error rate. Construction rejects hypergraph models and points to
`decompose_graphlike`. Decoding is deterministic and allocation-free per shot, and large
batches decode in parallel.

Two more decoders take the same model and expose the same `decode_packed` and
`logical_error_rate` calls:

- `MatchingDecoder` is exact minimum-weight perfect matching over the same graphlike
  model. It finds the least-weight correction for every shot, so it fails no more often
  than union-find on hard noise, at a higher per-shot cost.
- `BpOsdDecoder` runs belief propagation with ordered-statistics post-processing on the
  check matrix and accepts hypergraph models directly, which color codes and qLDPC codes
  need. `BpOsdOptions` picks the BP rule (`MinSum { scaling }` or `ProductSum`), the
  iteration cap, and the OSD search (`Zero`, `CombinationSweep { order }`, or
  `Exhaustive { order }`).

```rust
use prism_q::{BpOsdDecoder, BpOsdOptions, MatchingDecoder, OsdMethod, run_qec_program};

# let program = prism_q::parse_qec_program(
#     "R 0 1 2 3 4
#      X_ERROR(0.01) 0 2 4
#      CX 0 1 2 1 2 3 4 3
#      MR 1 3
#      DETECTOR rec[-2]
#      DETECTOR rec[-1]
#      M 0 2 4
#      OBSERVABLE_INCLUDE(0) rec[-1]",
# )?;
let result = run_qec_program(&program)?;
let model = program.detector_error_model()?;
let matching = MatchingDecoder::from_model(&model.decompose_graphlike()?)?;
let matching_rate = matching.logical_error_rate(&result.detectors, &result.observables)?;

let options = BpOsdOptions {
    osd_method: OsdMethod::CombinationSweep { order: 10 },
    ..BpOsdOptions::default()
};
let bposd = BpOsdDecoder::with_options(&model, options)?;
let bposd_rate = bposd.logical_error_rate(&result.detectors, &result.observables)?;
# Ok::<(), prism_q::PrismError>(())
```

The decoding section of [QEC program execution](../architecture/qec-programs.md) covers
each algorithm, its complexity, and its validation against brute force and the exact ML
rate.

## Error chain complex

`ErrorChainComplex` models the GF(2) chain complex over noise locations and identifies
undetectable error cycles. `noisy_marginals_analytical` computes marginals in closed form
from the parity matrix and noise rates, with no Monte Carlo. Readout error is part of that
closed form: a reported 1 is either a measured 1 that did not flip or a measured 0 that
did. Any non-Pauli channel, and any channel naming two qubits, is rejected.

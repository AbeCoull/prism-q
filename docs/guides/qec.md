# Noise and QEC

PRISM-Q models noise and quantum error correction without falling back to a dense
statevector per shot, through the [compiled samplers](../architecture/samplers.md) and the
[native QEC program IR](../architecture/qec-ir.md).

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

`NoiseModel` carries a list of `NoiseEvent { channel, qubits }` per instruction, where
the channel is a `NoiseChannel`: Pauli, depolarizing, readout error, amplitude damping
and the rest. For Clifford circuits, the noisy compiled sampler propagates noise
sensitivity rows and XORs fired channels into each sample, with no per-shot state
evolution.

## Device calibration import

A `DeviceCalibration` holds what a device's calibration page reports: `t1`, `t2`,
and readout rates per qubit, a duration and depolarizing error per gate family, and
optional per-pair two-qubit entries. `to_noise_model(&circuit)` lowers it onto a
circuit: after every gate each target gets a `ThermalRelaxation` channel with its own
`t1` and `t2` over the family's duration, a nonzero family error adds a `Depolarizing`
channel on a one-qubit target or a `TwoQubitDepolarizing` channel on the pair, and each
measurement sets its classical bit's readout error from the measured qubit. Gates on
more than two qubits are rejected, so decompose them first.

The text form has one record per line, `key=value` fields in any order, and `#`
comments:

| Record | Fields | Notes |
| --- | --- | --- |
| `qubit <n> ...` | `t1`, `t2` in seconds; `p01`, `p10` (default 0) | one per qubit, indices `0..N` in any order |
| `gate1q ...` | `time` in seconds, `error` | exactly once |
| `gate2q ...` | `time`, `error` | exactly once |
| `gate2q <a> <b> ...` | `time`, `error` | optional; replaces the family on that pair in either order |

Values are checked as they are read (`t2 <= 2 t1`, probabilities in `[0, 1]`, durations
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

`prism_q::sim::calibration::presets` has `superconducting_transmon(n)`,
`trapped_ion(n)`, and `neutral_atom(n)`. Their numbers are illustrative magnitudes for
a technology class, the same on every qubit, for exploring how such a model behaves;
they are not a measured device. Write the text form for a specific device.

## Detector sampling

For repeated syndrome extraction, `compile_detector_sampler` compiles a Clifford circuit
with measurement and reset reuse into a packed sampler, then derives detector and
observable records as parity rows over the measurement record. Reset reuse becomes fresh
qubit aliases, so there is no per-shot tableau replay.

## Native QEC programs

When you need detectors, logical observables, postselection, and Pauli-noise annotations
as first-class constructs, use the native QEC program IR rather than a `Circuit`:

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

`run_qec_program` lowers Clifford-compatible programs into the packed compiled sampler.
`run_qec_program_reference` is the per-shot statevector oracle for validating small
programs. `program.to_text()` writes a program back in the same text format, so a
program built in code can be saved and parsed again.

```admonish info title="What QEC programs support"
Clifford gates, basis resets and measurements, `MPP` Pauli-product measurements,
detectors, observables, postselection, `X_ERROR` / `Y_ERROR` / `Z_ERROR` /
`DEPOLARIZE1` / `DEPOLARIZE2` / `PAULI_CHANNEL_1` / `PAULI_CHANNEL_2` noise, and
terminal `EXP_VAL` final-state expectation estimates.
A noiseless `EXP_VAL` uses the analytical T strategies, with any detector records
still sampled by the packed runner. A noisy one is estimated exactly on the density
matrix when it fits, and falls to the per-shot reference runner when the program
carries measurement records or postselection or exceeds the density-matrix cap.
Non-Clifford gates are rejected on the packed sampling path.
See the [QEC IR reference](../architecture/qec-ir.md) for the full
grammar, and [QEC program execution](../architecture/qec-programs.md) for the
runner routing, the V1 reset requirement, and the `EXP_VAL` placement rules.
```

## Memory experiments

`QecProgram::repetition_memory`, `surface_memory`, and `color_memory` build
the standard memory experiments with circuit-level noise from a
`QecCircuitNoise`: depolarization after every Clifford gate, a flip before
every measurement and after every reset, and depolarization of the data at
the start of each round. Each program carries detectors with coordinates and
logical observable 0, ready for sampling, model derivation, and decoding:

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

The repetition code protects a Z memory; the rotated surface and triangular
6.6.6 color codes take an X or Z memory basis. A zero rate adds no annotation,
so `QecCircuitNoise::default()` gives the noiseless program, whose detectors
and observable never fire. In Python the same generators are static methods
of `QecProgram`, with an optional `QecCircuitNoise`. See the
[QEC IR reference](../architecture/qec-ir.md) for the layouts and schedules.

## Detector error model export

Matching and belief-propagation decoders consume an error model, not raw
detector samples. `QecProgram::detector_error_model` derives one from the
program's noise annotations, detectors, and observables, and `to_text` renders
it in the common detector error model text format that external decoders read:

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

Each mechanism carries a probability and the detector and observable indices
it flips; detector coordinates pass through from the program. In Python the
model also exposes `probabilities()`, `detector_matrix()`, and
`observable_matrix()`, the check-matrix triple that in-process decoder
libraries accept directly. Matching decoders need at most two detectors per
mechanism: `decompose_graphlike` returns that form, splitting each hypergraph
mechanism across existing graphlike ones and erroring loudly when no split
exists. `to_text` writes those splits as `^`-separated components.

`DetectorErrorModel::from_text` reads the same format, `repeat` blocks and
`shift_detectors` included, so a model written by another tool can drive the
in-crate decoder:

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

See [QEC program execution](../architecture/qec-programs.md) for the
derivation semantics and the grammar both directions cover.

## Decoding

`UnionFindDecoder` decodes sampled detectors against a graphlike model
in-process, so the logical error rate of a memory experiment never leaves the
tool:

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

The decoder is weighted union-find with peeling: edges weigh `ln((1-p)/p)`,
one-detector mechanisms are boundary edges, and mechanisms flipping no
detector bound the achievable logical error rate from below. Construction
rejects hypergraph models with a pointer to `decompose_graphlike`. Decoding is
deterministic and allocation-free per shot; large batches decode in parallel.
See the decoding section of
[QEC program execution](../architecture/qec-programs.md) for the growth and
peeling semantics and the validation against the exact ML rate.

## Error chain complex

`ErrorChainComplex` models the GF(2) chain complex over noise locations, identifying
undetectable error cycles. `noisy_marginals_analytical` computes marginals in closed form
from the parity matrix and noise rates, with no Monte Carlo. Readout error is part of that
closed form, a reported 1 being either a measured 1 that did not flip or a measured 0 that
did. It rejects a channel the complex cannot hold, which is any non-Pauli channel and any
channel naming two qubits.

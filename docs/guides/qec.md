# Quantum Error Correction

Two routes sample error correction experiments without a dense statevector per shot. A
Clifford `Circuit` with reset reuse compiles into a detector sampler, and the
[native QEC program IR](../architecture/qec-programs.md) adds detectors, logical observables,
postselection and noise annotations as operations. Noise on ordinary circuits
is on [Noise Models](noise.md). The [QEC memory tutorial](../tutorials/qec-memory.md)
runs a decoded experiment end to end, and [QEC from Python](python-qec.md) covers the
Python surface.

## Detector sampling

`compile_detector_sampler(&circuit, detector_rows, observable_rows, seed)` compiles a
Clifford circuit with measurement and reset reuse, the shape of repeated syndrome
extraction, into a packed sampler. Each detector and observable is a row of measurement
record indices, in circuit measurement order, and samples as the parity of those records.
Reset reuse becomes fresh qubit aliases, so no shot replays the tableau.

```rust
use prism_q::{CircuitBuilder, compile_detector_sampler};

let mut builder = CircuitBuilder::new_with_classical(3, 4);
builder.h(0).cx(0, 2);
for round in 0..2 {
    builder.cx(0, 1).cx(2, 1).measure(1, round).reset(1);
}
let circuit = builder.measure(0, 2).measure(2, 3).build();

let mut sampler =
    compile_detector_sampler(&circuit, vec![vec![0], vec![0, 1]], vec![vec![2, 3]], 42)?;
let batch = sampler.sample_packed(1000)?;
assert!((0..1000).all(|shot| !batch.detectors.get_bit(shot, 1)));
assert!((0..1000).all(|shot| !batch.observables.get_bit(shot, 0)));
# Ok::<(), prism_q::PrismError>(())
```

Qubit 1 checks the `ZZ` parity of qubits 0 and 2 twice. The data qubits read random but
equal values, so the observable over both data records never fires, and neither does the
detector comparing the two rounds. `sample_packed` draws the measurements once and derives
both parity sets from the same shots; `sample_detector_counts` streams a histogram.

## Native QEC programs

A `QecProgram` holds detectors, logical observables, postselection and Pauli noise
annotations as operations of their own, where a `Circuit` has none. Its text
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

`run_qec_program` lowers Clifford programs into the packed compiled sampler;
`run_qec_program_reference` is the per-shot statevector oracle for validating small
programs. `program.to_text()` writes the same text format back, so a program built in code
can be saved and parsed again.

```admonish info title="What QEC programs support"
Clifford gates (including the `SQRT_*`, `C_XYZ` / `C_ZYX`, `CY`, `SWAP`, `ISWAP`,
`CXSWAP`, and `SWAPCX` spellings), basis resets and measurements, `MPP` and
`MXX` / `MYY` / `MZZ` Pauli product measurements, `MPAD` constant records,
detectors, observables, postselection, `X_ERROR` / `Y_ERROR` / `Z_ERROR` /
`DEPOLARIZE1` / `DEPOLARIZE2` / `PAULI_CHANNEL_1` / `PAULI_CHANNEL_2` noise,
`E` / `CORRELATED_ERROR` chains with `ELSE_CORRELATED_ERROR`, `HERALDED_ERASE`,
`LEAK` / `SEEP` / `LEAK_TRANSPORT` leakage, and terminal `EXP_VAL` final-state
expectation estimates.
A noiseless `EXP_VAL` runs under a `QecTStrategy`: `Auto` tries exact light-cone SPD,
then CAMPS, then a tensor network scalar, and `run_qec_program_with_strategy` picks
`Spd`, `Camps` or `Reference` directly. Detector records are still sampled by the packed
runner. A noisy `EXP_VAL` is computed exactly on the density
matrix when it fits, and falls to the per-shot reference runner when the program
carries measurement records or postselection or exceeds the density matrix cap.
The packed sampling path rejects non-Clifford gates.
[QEC programs](../architecture/qec-programs.md) has the full grammar, the runner
routing, the V1 reset requirement, and the `EXP_VAL` placement rules.
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
`LEAK`, `SEEP`, `LEAK_TRANSPORT`, `POSTSELECT`, `EXP_VAL`, `RECORD_FLIP`), and adds
four Stim behaviours the native parser rejects, ignores or reads differently:

- inverted measurement targets (`M !0`, `MR !0`, `MXX !0 1`, `MPP !X0*Z1`), lowered as
  a Pauli on either side of the measurement so only the record flips; the second Pauli
  waits for the qubit's next use and is dropped at a reset, so a qubit measured once
  stays on the compiled sampler;
- record-controlled Paulis (`CX rec[-1] 2`, `CY rec[-1] 2`, `CZ rec[-1] 2`), lowered to
  feed-forward corrections;
- `SHIFT_COORDS`, which offsets later `DETECTOR` coordinates;
- measurement error on `M(p)`, `MR(p)` and the X and Y variants, which flips only the
  record with probability `p` and leaves the qubit alone, so `M(0.2) 0` then `M 0`
  gives a second record that is always 0. Native text reads `M(p)` as a Pauli flip
  before the measurement, which differs only when the qubit is measured again without a
  reset.

Any other instruction, `sweep[k]` targets, and noise arguments on `MPP`, the pair
measurements and `MPAD` are parse errors naming the line. A record flip lowers to
`QecOp::RecordFlip`, which every runner and `detector_error_model` honour; `to_text`
writes it as the native `RECORD_FLIP(p)` line after the measurement.

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

### Leakage as heralded erasure

`LEAK(p) q ...` leaks each target with probability `p`, `SEEP(p) q ...` returns each
leaked target to a random basis state with probability `p`, and
`LEAK_TRANSPORT(p) a b ...` spreads leakage across each pair at rate `p`.
`run_qec_program` samples them as erasures: a leaking qubit gets a uniformly random Pauli
frame, a two-qubit gate meeting a leaked qubit fully depolarizes both its qubits, other
noise annotations naming a leaked qubit are skipped, and the leaked qubit's own
measurement reports 1. On a Clifford program this matches the trajectory leakage model
in [Noise Models](noise.md#leakage) in distribution.

Each `LEAK` target adds one column to `QecSampleResult::heralds`, set when that qubit is
leaked just after the annotation; `LEAK(0)` reads the flags without changing them. An
`MPP` over a leaked qubit reports a random parity. The detector error model and the
reference runner have no leak flag and reject these programs, and the memory experiment
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
never fire. [QEC programs](../architecture/qec-programs.md#memory-experiment-generators) has the
layouts and schedules.

## Detector error model export

Matching and belief propagation decoders read an error model, not raw detector samples.
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
detector coordinates pass through from the program. Matching decoders need at most two
detectors per mechanism: `decompose_graphlike` splits each hypergraph mechanism across
existing graphlike ones and returns an error when no split exists, and `to_text` writes
the splits as components separated by `^`. Every detector and observable must be
deterministic in the noiseless program, since a random one has no error mechanism to
describe it; `detector_error_model` names it in the error.

`DetectorErrorModel::from_text` reads the same format, `repeat` blocks and
`shift_detectors` included, so a model written by another tool can drive the decoders in
the crate:

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

[QEC decoding](../architecture/qec-decoding.md) has the derivation semantics and the
grammar in both directions.

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
`decompose_graphlike`. Decoding is deterministic and allocates nothing per shot, and large
batches decode in parallel.

Two more decoders take the same model and expose the same `decode_packed` and
`logical_error_rate` calls:

- `MatchingDecoder` is exact minimum-weight perfect matching over the same graphlike
  model. It finds the least-weight correction for every shot, so it fails no more often
  than union-find on hard noise, at a higher per-shot cost.
- `BpOsdDecoder` runs belief propagation with ordered statistics decoding on the check
  matrix and accepts hypergraph models directly, which color codes and qLDPC codes
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

[QEC decoding](../architecture/qec-decoding.md#decoders) covers each algorithm, its
complexity, and its validation against brute force and the exact ML rate.

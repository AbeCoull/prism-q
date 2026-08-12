# Noise and QEC

PRISM-Q models noise and quantum error correction without falling back to a dense
statevector per shot. The machinery is the [compiled samplers](../architecture/samplers.md)
and the [native QEC program IR](../architecture/qec-ir.md); this guide shows how they fit
together.

## Noisy shot sampling

Attach a `NoiseModel` and sample:

```rust
use prism_q::{simulate, BackendKind, NoiseModel};

let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
let result = simulate(&circuit)
    .backend(BackendKind::Statevector)
    .noise(noise)
    .seed(42)
    .shots(1024)
    .unwrap();
```

`NoiseModel` carries per-instruction depolarizing channels (`NoiseOp { qubit, px, py, pz }`)
and supports readout error and amplitude damping. For Clifford circuits, the noisy
compiled sampler propagates noise sensitivity rows and XORs fired channels into each
sample, avoiding per-shot state evolution entirely.

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

let program = parse_qec_program(qec_text).unwrap();
let result = run_qec_program(&program).unwrap();
```

`run_qec_program` lowers Clifford-compatible programs into the packed compiled sampler.
`run_qec_program_reference` is the per-shot statevector oracle for validating small
programs.

```admonish info title="What QEC programs support"
Clifford gates, basis resets and measurements, `MPP` Pauli-product measurements,
detectors, observables, postselection, `X_ERROR` / `Z_ERROR` / `DEPOLARIZE1` /
`DEPOLARIZE2` noise, and terminal `EXP_VAL` final-state expectation estimates
(noiseless programs use the analytical T strategies, with any detector records
still sampled by the packed runner; noisy programs use the per-shot reference
runner). Non-Clifford gates are rejected on the packed sampling path.
See the [QEC IR reference](../architecture/qec-ir.md) for the full
grammar, and [QEC program execution](../architecture/qec-programs.md) for the
runner routing, the V1 reset requirement, and the `EXP_VAL` placement rules.
```

## Detector error model export

Matching and belief-propagation decoders consume an error model, not raw
detector samples. `QecProgram::detector_error_model` derives one from the
program's noise annotations, detectors, and observables, and `to_text` renders
it in the common detector error model text format that external decoders read:

```rust
let model = program.detector_error_model().unwrap();
std::fs::write("memory_d3.dem", model.to_text()).unwrap();
```

Each mechanism carries a probability and the detector and observable indices
it flips; detector coordinates pass through from the program. In Python the
model also exposes `probabilities()`, `detector_matrix()`, and
`observable_matrix()`, the check-matrix triple that in-process decoder
libraries accept directly. Matching decoders need at most two detectors per
mechanism: `decompose_graphlike` returns that form, splitting each hypergraph
mechanism across existing graphlike ones and erroring loudly when no split
exists. See [QEC program execution](../architecture/qec-programs.md) for the
derivation semantics and the emitted grammar.

## Decoding

`UnionFindDecoder` decodes sampled detectors against a graphlike model
in-process, so the logical error rate of a memory experiment never leaves the
tool:

```rust
use prism_q::{UnionFindDecoder, run_qec_program};

let model = program.detector_error_model()?.decompose_graphlike()?;
let decoder = UnionFindDecoder::from_model(&model)?;
let result = run_qec_program(&program)?;
let predicted = decoder.decode_packed(&result.detectors)?;
let failures = (0..result.total_shots)
    .filter(|&shot| predicted.get_bit(shot, 0) != result.observables.get_bit(shot, 0))
    .count();
```

The decoder is weighted union-find with peeling: edges weigh `ln((1-p)/p)`,
one-detector mechanisms are boundary edges, and mechanisms flipping no
detector bound the achievable logical error rate from below. Construction
rejects hypergraph models with a pointer to `decompose_graphlike`. Decoding is
deterministic and allocation-free per shot; large batches decode in parallel.
See the decoding section of
[QEC program execution](../architecture/qec-programs.md) for the growth and
peeling semantics and the validation against the exact ML rate.

## Homological sampling

`run_shots_homological` and `ErrorChainComplex` model the GF(2) chain complex over noise
locations, identifying undetectable error cycles. `noisy_marginals_analytical` computes
marginals in closed form from the parity matrix and noise rates, with no Monte Carlo.

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
detectors, observables, postselection, and `X_ERROR` / `Z_ERROR` / `DEPOLARIZE1` /
`DEPOLARIZE2` noise. Non-Clifford gates and `EXP_VAL` are rejected until their production
strategies land. See the [QEC IR reference](../architecture/qec-ir.md) for the full
grammar and the V1 reset requirement.
```

## Homological sampling

`run_shots_homological` and `ErrorChainComplex` model the GF(2) chain complex over noise
locations, identifying undetectable error cycles. `noisy_marginals_analytical` computes
marginals in closed form from the parity matrix and noise rates, with no Monte Carlo.

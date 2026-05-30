# Error Model and Public API

## Error model

All public APIs return `Result<T, PrismError>`. Error variants:

| Variant | Category | Description |
|---------|----------|-------------|
| `Parse` | Parsing | OpenQASM parse error with line number |
| `UnsupportedConstruct` | Parsing | Valid OpenQASM not supported by PRISM-Q |
| `UndefinedRegister` | Parsing | Reference to undeclared register |
| `InvalidQubit` | Validation | Qubit index exceeds register size |
| `InvalidClassicalBit` | Validation | Classical bit index exceeds register |
| `GateArity` | Validation | Wrong number of qubits for gate |
| `InvalidParameter` | Validation | Invalid gate parameter (NaN, etc.) |
| `BackendUnsupported` | Runtime | Backend can't perform requested operation |
| `IncompatibleBackend` | Runtime | Backend incompatible with circuit |

```admonish note
No panics on user input. `debug_assert!` is used for internal invariants only.
```

## Public API surface

Top-level re-exports from `src/lib.rs`. The full generated documentation is on
[docs.rs](https://docs.rs/prism-q).

**Simulation:**
`simulate`, `run_on`, `run_qasm`, `bitstring`

**Compiled sampling:**
`compile_measurements`, `compile_forward`, `compile_detector_sampler`, `compile_noisy`, `run_shots_compiled`, `run_shots_noisy`, `run_shots_homological`, `noisy_marginals_analytical`

**Native QEC:**
`parse_qec_program`, `compile_qec_program_rows`, `run_qec_program`, `run_qec_program_reference`, `QecProgram`, `QecOp`, `QecOptions`, `QecSampleResult`, `QecBasis`, `QecPauli`, `QecRecordRef`, `QecNoise`, `QecMeasurementRow`, `QecCompiledRows`

**Clifford+T:**
`run_stabilizer_rank`, `run_stabilizer_rank_approx`, `stabilizer_overlap_sq`, `stabilizer_inner_product`, `run_spp`, `run_spp_observable`, `run_spd`, `run_spd_observable`, `run_spd_observable_light_cone`

**Types:**
`Circuit`, `CircuitBuilder`, `Instruction`, `ClassicalCondition`, `Gate`, `BackendKind`, `RunOutcome`, `CountsResult`, `MarginalsResult`, `Probabilities`, `FactoredBlock`, `ShotsResult`, `PrismError`, `Result`

**Backends:**
`StatevectorBackend`, `StabilizerBackend`, `SparseBackend`, `MpsBackend`, `ProductStateBackend`, `TensorNetworkBackend`, `FactoredBackend`

**Accumulators:**
`ShotAccumulator`, `HistogramAccumulator`, `MarginalsAccumulator`, `PauliExpectationAccumulator`, `CorrelatorAccumulator`, `NullAccumulator`, `PackedShots`, `ShotLayout`

**Data types:**
`CompiledSampler`, `CompiledDetectorSampler`, `DetectorSampleBatch`, `NoisyCompiledSampler`, `HomologicalSampler`, `ErrorChainComplex`, `NoiseModel`, `NoiseOp`, `QecProgram`, `QecOp`, `QecSampleResult`, `StabRankResult`, `SppResult`, `SpdResult`, `SparseParity`, `ParityStats`, `PauliVec`, `MultiFusedData`, `BatchPhaseData`, `McuData`, `Multi2qData`

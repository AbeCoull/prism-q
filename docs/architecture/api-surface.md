# Error Model and Public API

## Error model

Fallible public APIs return `Result<T, PrismError>`. Error variants:

| Variant | Category | Description |
|---------|----------|-------------|
| `Parse` | Parsing | OpenQASM parse error with line number |
| `UnsupportedConstruct` | Parsing | Valid OpenQASM not supported by PRISM-Q |
| `UndefinedRegister` | Parsing | Reference to undeclared register |
| `InvalidQubit` | Validation | Qubit index exceeds register size |
| `InvalidClassicalBit` | Validation | Classical bit index exceeds register |
| `GateArity` | Validation | Wrong number of qubits for gate |
| `InvalidParameter` | Validation | Invalid gate parameter (NaN, etc.) |
| `ExportUnsupported` | Export | Instruction with no OpenQASM 3.0 spelling |
| `BackendUnsupported` | Runtime | Backend can't perform requested operation |
| `IncompatibleBackend` | Runtime | Backend incompatible with circuit |

```admonish note
Invalid input data (QASM text, incompatible backend) returns `PrismError`. API misuse
(out-of-range indices, wrong-variant accessors) panics, and each such method documents
the condition under `# Panics`. `debug_assert!` is used for internal invariants only.
```

## Public API surface

Top-level re-exports from `src/lib.rs`. The full generated documentation is on
[docs.rs](https://docs.rs/prism-q).

**Simulation:**
`simulate`, `Simulate`, `Unseeded`, `Seeded`, `run_on`, `run_on_state`, `run_qasm`,
`run_expectation_values`, `bitstring`

**Gradients:**
`run_expectation_gradient`, `run_expectation_gradient_shift`, `ExpectationGradient`

**Parameters and binding:**
`Parameters`, `ParamLink`, `PreparedCircuit`. One parameter model serves both
consumers: the gradient path reads the links, and binding writes through them.

**Compiled sampling:**
`compile_measurements`, `compile_forward`, `compile_detector_sampler`, `compile_noisy`,
`run_shots_compiled`, `run_shots_noisy`, `run_shots_homological`,
`noisy_marginals_analytical`, `density_matrix_expectation_values`; with the `gpu`
feature: `run_shots_compiled_with_gpu`, `DevicePackedShots`

**Native QEC:**
`parse_qec_program`, `compile_qec_program_rows`, `run_qec_program`,
`run_qec_program_reference`, `run_qec_program_with_strategy`,
`run_qec_program_spd_rerouted`, `QecProgram`, `QecOp`, `QecOptions`, `QecSampleResult`,
`QecBasis`, `QecPauli`, `QecRecordRef`, `QecNoise`, `QecMeasurementRow`,
`QecCompiledRows`, `QecObservableEstimate`, `QecObservableReroute`, `QecTStrategy`,
`DetectorErrorModel`, `ErrorMechanism`, `UnionFindDecoder`

**Clifford+T:**
`run_stabilizer_rank`, `run_stabilizer_rank_approx`, `stabilizer_overlap_sq`,
`stabilizer_inner_product`, `StabRankResult`, `run_spp`, `run_spp_observable`,
`run_spd`, `run_spd_observable`, `run_spd_observable_light_cone`, `inverse_light_cone`,
`PauliAxis`, `PauliTerm`, `SppResult`, `SppObservableResult`, `SpdResult`,
`SpdObservableResult`

**Types:**
`Circuit`, `CircuitBuilder`, `Instruction`, `ClassicalCondition`, `SvgOptions`,
`TextOptions`, `Gate`, `GeneratorKind`, `BackendKind`, `RunOutcome`, `CountsResult`,
`MarginalsResult`, `Probabilities`, `FactoredBlock`, `ShotsResult`, `PrismError`,
`Result`, `MultiFusedData`, `BatchPhaseData`, `McuData`, `Multi2qData`

**Backends:**
`StatevectorBackend`, `StabilizerBackend`, `SparseBackend`, `MpsBackend`,
`ProductStateBackend`, `TensorNetworkBackend`, `FactoredBackend`,
`FactoredStabilizerBackend`; with the `distributed` feature:
`DistributedStatevectorBackend`, `DistributedContext`, `RankComm`, `SerialComm`; with
the `distributed-mpi` feature: `MpiComm`

**Accumulators:**
`ShotAccumulator`, `HistogramAccumulator`, `MarginalsAccumulator`,
`PauliExpectationAccumulator`, `CorrelatorAccumulator`, `NullAccumulator`,
`PackedShots`, `ShotLayout`, `ParityStats`

**Data types:**
`CompiledSampler`, `CompiledDetectorSampler`, `DetectorSampleBatch`,
`NoisyCompiledSampler`, `NoiseChannel`, `NoiseEvent`, `NoiseModel`, `ReadoutError`,
`HomologicalSampler`, `ErrorChainComplex`

Not re-exported at the root but part of the documented surface: the `Backend` trait and
`BasisSamples` at `prism_q::backend`, the density matrix backend at
`prism_q::backend::density_matrix`, and the accumulator chunk-size helpers
(`default_chunk_size`, `optimal_chunk_size`) at `prism_q::sim::compiled`.

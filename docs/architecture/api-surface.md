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
| `StepLimit` | Runtime | A dynamic program shot ran more blocks than its step bound; names the loop |
| `ResourceLimit` | Runtime | An allocation over a memory cap; the boxed `ResourceLimit` carries the operation, the `ResourceKind` unit, `required` and `limit`, and the environment variable that overrides the cap |

```admonish note
Invalid input data (QASM text, incompatible backend) returns `PrismError`. API misuse
(out-of-range indices, wrong-variant accessors) panics, and each such method documents
the condition under `# Panics`. `debug_assert!` is used for internal invariants only.
```

## Public API surface

Top-level re-exports from `src/lib.rs`. The full generated documentation is on
[docs.rs](https://docs.rs/prism-q).

| Area | Items |
| --- | --- |
| Simulation | `simulate`, `Simulate`, `Unseeded`, `Seeded`, `run_on`, `run_on_state`, `run_qasm`, `PauliObservable`, `ObservableExpectation`, `ObservableVariance`, `ExpectationResult`, `OverlapResult`, `bitstring` |
| Route inspection | `Query`, `Plan`, `PlanFallback`, `PlannedExactness`, `RejectReason`, `RouteRule`, `MemoryEstimate`, `MemoryEstimateKind` |
| Limits | `Limits`, `LimitKnob`, `LimitSource`, `default_limits`, `set_default_limits` |
| Dynamic programs | `DynamicProgram`, `DynamicProgramBuilder`, `simulate_program`, `SimulateProgram` |
| Gradients | `ExpectationGradient`, returned by `Simulate::expectation_gradient` and `Simulate::expectation_gradient_shift` |
| Parameters | `Parameters`, `ParamLink`, `PreparedCircuit` |
| Compiled sampling | `compile_measurements`, `compile_forward`, `compile_detector_sampler`, `compile_noisy`, `run_shots_compiled`, `run_shots_noisy`, `noisy_marginals_analytical`, `density_matrix_expectation_values`, `CompiledSampler`, `CompiledDetectorSampler`, `DetectorSampleBatch`; with `gpu`: `run_shots_compiled_with_gpu`, `DevicePackedShots` |
| Accumulators | `ShotAccumulator`, `HistogramAccumulator`, `MarginalsAccumulator`, `PauliExpectationAccumulator`, `CorrelatorAccumulator`, `NullAccumulator`, `PackedShots`, `ShotLayout`, `ParityStats` |
| Noise | `NoiseModel`, `NoiseBuilder`, `NoiseChannel`, `NoiseEvent`, `GateFilter`, `ReadoutError`, `DriftDistribution`, `NoisyCompiledSampler`, `DeviceCalibration`, `QubitCalibration`, `GateCalibration`, `GateTimes`, `ErrorChainComplex` |
| Native QEC | `parse_qec_program`, `compile_qec_program_rows`, `run_qec_program`, `run_qec_program_reference`, `run_qec_program_with_strategy`, `run_qec_program_spd_rerouted`, `QecProgram`, `QecOp`, `QecOptions`, `QecSampleResult`, `QecBasis`, `QecPauli`, `QecRecordRef`, `QecNoise`, `QecCorrelatedBranch`, `QecCircuitNoise`, `QecDetectorReference`, `QecMeasurementRow`, `QecCompiledRows`, `QecCompiledSampler`, `QecObservableEstimate`, `QecObservableReroute`, `QecTStrategy` |
| QEC decoding | `DetectorErrorModel`, `ErrorMechanism`, `UnionFindDecoder`, `MatchingDecoder`, `BpOsdDecoder`, `BpOsdOptions`, `BpMethod`, `OsdMethod` |
| Clifford+T | `run_stabilizer_rank`, `run_stabilizer_rank_approx`, `stabilizer_overlap_sq`, `stabilizer_inner_product`, `StabRankResult`, `run_spp`, `run_spp_observable`, `run_spd`, `run_spd_with`, `run_spd_observable`, `run_spd_observable_budgeted`, `run_spd_observable_light_cone`, `inverse_light_cone`, `PauliAxis`, `PauliTerm`, `SppResult`, `SppObservableResult`, `SpdResult`, `SpdObservableResult`, `SpdTruncation` |
| Circuits and gates | `Circuit`, `CircuitBuilder`, `TrotterOrder`, `Instruction`, `ClassicalCondition`, `SaveSpec`, `SaveRecord`, `SavedValue`, `SvgOptions`, `TextOptions`, `Gate`, `GeneratorKind`, `McuData`, `UnitaryData` |
| Results and provenance | `BackendKind`, `RunOutcome`, `CountsResult`, `MarginalsResult`, `ShotsResult`, `ReducedDensityMatrix`, `EntropyResult`, `Probabilities`, `FactoredBlock`, `RunMetadata`, `BondReport`, `Engine`, `Exactness`, `Placement`, `ResolvedBackend` |
| Errors | `PrismError`, `ResourceLimit`, `ResourceKind`, `Result` |
| Backends | `StatevectorBackend`, `StabilizerBackend`, `SparseBackend`, `MpsBackend`, `ProductStateBackend`, `TensorNetworkBackend`, `FactoredBackend`, `FactoredStabilizerBackend`, `DensityMatrixBackend`; with `distributed`: `DistributedStatevectorBackend`, `DistributedContext`, `RankComm`, `SerialComm`; with `distributed-mpi`: `MpiComm` |
| Threading | with `parallel`: `ThreadPool`, a Rayon pool the caller supplies, which simulation runs inside instead of sizing the process-wide one; see [Threading, SIMD, and Memory Layout](./threading-simd.md) |

Not re-exported at the root but part of the documented surface: the `Backend` trait and
`BasisSamples` at `prism_q::backend`, `run_batch` at `prism_q::sim`, `parse_dynamic` and
the IR types under `prism_q::circuit::openqasm` and `prism_q::circuit::dynamic`, and the
accumulator chunk size helpers (`default_chunk_size`, `optimal_chunk_size`) at
`prism_q::sim::compiled`. `Backend` is sealed: only the crate implements it, so a new
method is additive. Items behind the `bench-internal` feature (`bench_variant`, the sweep
profile helpers, `QecProfiledSampler`) are benchmark tooling, not public API.

The complete item list is checked in as [`public-api.txt`](public-api.txt), generated by
`cargo public-api` with `--features "parallel gpu distributed"`. CI regenerates it and
fails on a difference, so every surface change shows in review as a diff to that file.

### Save points

`Circuit::add_save` appends an `Instruction::Save` recording a `SaveSpec`, and
`Simulate::run` returns one `SaveRecord` per point in `RunOutcome::saves`, carrying a
`SavedValue`. See [Circuit IR](./ir.md) for what a save does to fusion and which routes
decline one.

### Batches

`prism_q::sim::run_batch` runs a list of circuits, holding one backend across those of
the same width that draw no randomness. Results match running each circuit alone with
the same seed. Under `parallel`, batches of circuits up to 16 qubits split across cores,
one circuit per core with each run single threaded; wider batches run one circuit at a
time on the kernels' own threads.

### State diagnostics

- `Simulate::reduced_density_matrix` returns a `ReducedDensityMatrix` (row major, side
  `2^k`, `qubits[0]` the lowest bit of the row index) with `purity` for `Tr(rho^2)`.
- `Simulate::entanglement_entropy` returns an `EntropyResult`: the von Neumann entropy in
  nats and the descending Schmidt spectrum, `None` where the backend holds the entropy
  without the spectrum.
- `Simulate::overlap` takes a second seeded builder over a circuit of the same width and
  returns an `OverlapResult`: the squared inner product and the provenance of both runs.

All three require a unitary circuit, and under `BackendKind::Auto` a route that cannot
answer falls back to the statevector. [Backends](./backends.md#state-diagnostics)
tabulates which backends answer each.

### Prepared circuits

`PreparedCircuit` carries `run`, `expectation_values` and `observable_expectation`, each
answering as the `Simulate` terminal of that name does on the bound circuit, plus a
`_many` form of each that takes a list of bindings and splits it across cores where
`run_batch` would. See [Parameters, Binding and Gradients](./parameters.md).

## Growth of the public enums

Public enums are `#[non_exhaustive]` unless named below. A new gate, backend, engine
label, error variant, noise channel, truncation policy, dialect, result request or
result value is an additive release, so a `match` from outside the crate keeps a
wildcard arm. `RunMetadata` is `#[non_exhaustive]` too: a new field is additive, and
outside code reads it rather than building it.

`Instruction` stays exhaustive. It is the circuit IR, and a new instruction kind changes
what every consumer handles, so adding one is a breaking change.

Enums mirroring a closed mathematical set stay exhaustive as well, since a wildcard arm
the caller can never reach buys nothing: `PauliAxis`, `MpsPauliAxis` and `QecBasis` are
X, Y and Z, `GeneratorKind` and `DiagEntry` are gate-algebra shapes, and
`QftTextbookStep` is a fixed decomposition.

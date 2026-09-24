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
`run_expectation_values`, `run_observable_expectation`, `PauliObservable`,
`ObservableExpectation`, `ObservableVariance`, `ExpectationResult`, `OverlapResult`,
`bitstring`

**Save points:** `Circuit::add_save` appends an `Instruction::Save` recording a
`SaveSpec`, and `Simulate::run` returns one `SaveRecord` per point in `RunOutcome::saves`,
carrying a `SavedValue`. All three types are re-exported. See [Circuit IR](./ir.md) for
what a save does to fusion and which routes decline one.

**Batches:** `prism_q::sim::run_batch` runs a list of circuits, holding one backend
across those of the same width that draw no randomness. Results match running each circuit alone with the
same seed. Up to 16 qubits, under `parallel`, the circuits split across cores: a
200-point sweep at 4 to 12 qubits, where each run is single-threaded inside, ran 3.2x to
4.8x faster than a loop on a four-core host, and one circuit per core still ran 5.5x and
1.9x faster than the kernels' own threads at 14 and 16 qubits. At 18 qubits the split took
10% longer, so wider batches run one circuit at a time. On one thread the held backend is the only
saving, and it read within about 10% of a loop.

**State diagnostics:** `Simulate::reduced_density_matrix` returns a
`ReducedDensityMatrix` (row major, side `2^k`, `qubits[0]` the lowest bit of the row
index) with a `purity` method for `Tr(rho^2)`; `Simulate::entanglement_entropy` returns
an `EntropyResult` carrying the von Neumann entropy in nats and the descending Schmidt
spectrum, which is `None` where the backend holds the entropy without the spectrum
behind it; `Simulate::overlap` takes a second seeded builder over a circuit of the same width and
returns an `OverlapResult` carrying the squared inner product and the provenance of both
runs. All three require a unitary circuit, and under `BackendKind::Auto` a route that
cannot answer falls back to the statevector. Which backends answer each is tabulated in
[Backends](./backends.md).

**Gradients:**
`run_expectation_gradient`, `run_expectation_gradient_shift`, `ExpectationGradient`

**Parameters and binding:**
`Parameters`, `ParamLink`, `PreparedCircuit`. The gradient path reads the links and
binding writes through them. `PreparedCircuit` carries `run`, `expectation_values` and
`observable_expectation`, each answering what the `Simulate` terminal of that name
answers on the bound circuit, plus a `_many` form of each that takes a list of bindings
and splits it across cores where `run_batch` would.

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
`run_spd`, `run_spd_with`, `run_spd_observable`, `run_spd_observable_budgeted`,
`run_spd_observable_light_cone`, `inverse_light_cone`, `PauliAxis`, `PauliTerm`,
`SppResult`, `SppObservableResult`, `SpdResult`, `SpdObservableResult`,
`SpdTruncation`

**Types:**
`Circuit`, `CircuitBuilder`, `Instruction`, `ClassicalCondition`, `SvgOptions`,
`TextOptions`, `Gate`, `GeneratorKind`, `BackendKind`, `RunOutcome`, `CountsResult`,
`MarginalsResult`, `ReducedDensityMatrix`, `EntropyResult`, `Probabilities`,
`FactoredBlock`, `ShotsResult`, `PrismError`, `Result`, `MultiFusedData`,
`BatchPhaseData`, `McuData`, `Multi2qData`, `UnitaryData`, `RunMetadata`, `BondReport`,
`Engine`, `Exactness`, `Placement`, `ResolvedBackend`

**Backends:**
`StatevectorBackend`, `StabilizerBackend`, `SparseBackend`, `MpsBackend`,
`ProductStateBackend`, `TensorNetworkBackend`, `FactoredBackend`,
`FactoredStabilizerBackend`, `DensityMatrixBackend`; with the `distributed` feature:
`DistributedStatevectorBackend`, `DistributedContext`, `RankComm`, `SerialComm`; with
the `distributed-mpi` feature: `MpiComm`

**Threading:** with the `parallel` feature: `ThreadPool`, a caller-supplied Rayon pool
that simulation runs inside instead of sizing the process-wide one. See
[Threading, SIMD, and Memory Layout](./threading-simd.md).

**Accumulators:**
`ShotAccumulator`, `HistogramAccumulator`, `MarginalsAccumulator`,
`PauliExpectationAccumulator`, `CorrelatorAccumulator`, `NullAccumulator`,
`PackedShots`, `ShotLayout`, `ParityStats`

**Data types:**
`CompiledSampler`, `CompiledDetectorSampler`, `DetectorSampleBatch`,
`NoisyCompiledSampler`, `NoiseChannel`, `NoiseEvent`, `NoiseModel`, `NoiseBuilder`,
`GateFilter`, `ReadoutError`, `DeviceCalibration`, `QubitCalibration`, `GateCalibration`,
`HomologicalSampler`, `ErrorChainComplex`

Not re-exported at the root but part of the documented surface: the `Backend` trait and
`BasisSamples` at `prism_q::backend`, `run_batch` at `prism_q::sim`, and the accumulator
chunk-size helpers (`default_chunk_size`, `optimal_chunk_size`) at
`prism_q::sim::compiled`.

## Growth of the public enums

Public enums are `#[non_exhaustive]` unless named below. A new gate, backend, engine
label, error variant, noise channel, truncation policy, dialect, result request or
result value is an additive release, so a `match` on one from outside the crate keeps a
wildcard arm. `RunMetadata` is `#[non_exhaustive]` for the same reason: a new field on
it is additive, and code outside the crate reads it rather than building it.

`Instruction` stays exhaustive. It is the circuit IR, and a new instruction kind
changes what every consumer has to handle, so adding one is a breaking change.

So do the enums that mirror a closed mathematical set, where a wildcard arm the caller
can never reach buys nothing: `PauliAxis`, `MpsPauliAxis` and `QecBasis` are X, Y and Z,
`GeneratorKind` and `DiagEntry` are gate-algebra shapes, and `QftTextbookStep` is a
fixed decomposition.

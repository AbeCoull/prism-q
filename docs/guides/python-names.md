# Rust and Python Names

Python names follow the Rust ones, written in Python's own conventions: a Rust
constructor such as `CircuitBuilder::new_with_classical` is the class call, a builder
consumed by value is a method chain on a mutable object, and a result field is a
property. Two names differ for the reader's sake: `simulate()` returns a `Simulation`
where Rust has the `Simulate` typestate builder (and `simulate_program()` a
`ProgramSimulation`), and the `circuits` generators drop the `_circuit` suffix their
module already states. `bindings/python/parity.toml` records, for every Rust function,
method and result field in the user-facing surface, the Python name that reaches it or
the reason it stays Rust-only, and a test keeps that record complete as the Rust API
grows.

Three classes have taken their Rust names: `Decoder` is now `UnionFindDecoder`,
`QecResult` is `QecSampleResult`, and `RecordRef` is `QecRecordRef`. The old names
still resolve, with a `DeprecationWarning`, until the next minor release, and pickles
written under the old names load.

Free functions with a Python counterpart map as follows, with Rust paths relative to
the crate root. The manifest gives the reason for each function left out.

<!-- parity-table:start -->
| Rust | Python |
|------|--------|
| `circuit::guarded` | `CircuitBuilder.guarded` |
| `circuit::openqasm::parse` | `parse_qasm` |
| `circuit::openqasm::parse_braket` | `parse_braket` |
| `circuit::openqasm::parse_dynamic` | `parse_qasm_dynamic` |
| `circuit::openqasm::parse_parametric` | `parse_qasm_parametric` |
| `circuit::qasm_export::to_qasm3` | `Circuit.to_qasm` |
| `circuits::clifford_heavy_circuit` | `circuits.clifford_heavy` |
| `circuits::clifford_random_pairs` | `circuits.clifford_random_pairs` |
| `circuits::clifford_t_circuit` | `circuits.clifford_t` |
| `circuits::cz_chain_circuit` | `circuits.cz_chain` |
| `circuits::ghz_circuit` | `circuits.ghz` |
| `circuits::hardware_efficient_ansatz` | `circuits.hardware_efficient_ansatz` |
| `circuits::independent_bell_pairs` | `circuits.independent_bell_pairs` |
| `circuits::independent_random_blocks` | `circuits.independent_random_blocks` |
| `circuits::local_clifford_blocks` | `circuits.local_clifford_blocks` |
| `circuits::phase_estimation_circuit` | `circuits.phase_estimation` |
| `circuits::qaoa_circuit` | `circuits.qaoa` |
| `circuits::qft_circuit` | `circuits.qft` |
| `circuits::quantum_volume_circuit` | `circuits.quantum_volume` |
| `circuits::random_circuit` | `circuits.random` |
| `circuits::single_qubit_rotation_circuit` | `circuits.single_qubit_rotation` |
| `circuits::w_state_circuit` | `circuits.w_state` |
| `compile_measurements` | `CompiledSampler` |
| `compile_noisy` | `CompiledSampler` |
| `default_limits` | `default_limits` |
| `density_matrix_expectation_values` | `Simulation.density_matrix_expectation_values` |
| `gpu::is_available` | `GpuContext.is_available` |
| `noisy_marginals_analytical` | `noisy_marginals_analytical` |
| `parse_qec_program` | `QecProgram.from_text` |
| `run_expectation_gradient` | `Simulation.expectation_gradient` |
| `run_expectation_gradient_shift` | `Simulation.expectation_gradient_shift` |
| `run_expectation_values` | `Simulation.expectation_values` |
| `run_observable_expectation` | `Simulation.observable_expectation` |
| `run_qasm` | `run_qasm` |
| `run_qec_program` | `QecProgram.run` |
| `run_qec_program_reference` | `QecProgram.run_reference` |
| `run_shots_compiled` | `Simulation.shots` |
| `run_shots_noisy` | `Simulation.shots` |
| `run_spd` | `BackendKind.deterministic_pauli` |
| `run_spd_with` | `BackendKind.deterministic_pauli_budget` |
| `run_spp` | `BackendKind.stochastic_pauli` |
| `run_stabilizer_rank` | `BackendKind.stabilizer_rank` |
| `set_default_limits` | `set_default_limits` |
| `sim::calibration::presets::neutral_atom` | `DeviceCalibration.neutral_atom` |
| `sim::calibration::presets::superconducting_transmon` | `DeviceCalibration.superconducting_transmon` |
| `sim::calibration::presets::trapped_ion` | `DeviceCalibration.trapped_ion` |
| `sim::run_batch` | `run_batch` |
| `sim::unified_pauli::run_pauli_path_observable` | `BackendKind.pauli_path` |
| `simulate` | `simulate` |
| `simulate_program` | `simulate_program` |
<!-- parity-table:end -->

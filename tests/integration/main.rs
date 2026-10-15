//! Integration tests built as one binary, so the crate's generic code is instantiated and
//! linked once. A file that changes process-wide state (an environment cap, the global
//! Rayon pool, the default limits) stays a separate `tests/<name>.rs` binary instead.

#[path = "../common/mod.rs"]
mod common;
#[path = "../conformance_corpus/mod.rs"]
mod conformance_corpus;
#[path = "../qec_common/mod.rs"]
mod qec_common;

mod backend_matrix;
mod batch_execution;
mod bench_fixture_checks;
mod bench_fixture_routing;
mod braket_pragmas;
mod braket_results;
mod braket_sampling;
mod conformance_matrix;
mod degenerate_inputs;
mod dense_unitary;
mod density_matrix_correctness;
mod density_matrix_fold_contract;
mod device_calibration;
mod differential_wide;
mod dynamic_programs;
mod expectation_gradient;
mod expectation_gradient_noisy;
mod expectation_values;
mod factored_correctness;
mod fusion_correctness;
mod fuzz_replay;
mod gate_matrix;
mod gate_modifiers;
mod golden_small_circuits;
mod guarded_regions;
mod initial_state;
mod leakage_and_drift;
mod measurement_matrix;
mod mid_circuit_saves;
mod mps_correctness;
mod native_sampling;
mod noise_construction;
mod noise_validation;
mod observable_variance;
mod openqasm_goldens;
mod parameter_binding;
mod pauli_joint_observable;
mod pauli_path_correctness;
mod pauli_rot_correctness;
mod plan_api;
mod product_correctness;
mod qasm_classical;
mod qasm_subset;
mod qec_decoder;
mod qec_dem_export;
mod qec_dem_import;
mod qec_e2e_d3;
mod qec_exp_val;
mod qec_feedforward;
mod qec_ir;
mod qec_leakage;
mod qec_memory_generators;
mod qec_noisy_validation;
mod qec_observable_reroute;
mod qec_pauli_channels;
mod qec_seeded_records;
mod qec_t_correctness;
mod qec_text_roundtrip;
mod qft_expansion_regression;
mod qft_recognition;
mod reduced_density_matrix;
mod reset_channel;
mod result_metadata;
mod sampling_contract;
mod schmidt_values;
mod smoke_openqasm;
mod sparse_correctness;
mod sparse_width_ceiling;
mod stabilizer_correctness;
mod stabilizer_noise_routes;
mod stabilizer_tableau_import;
mod state_vector_terminal;
mod subset_probabilities;
mod tensor_route;
mod tensornetwork_correctness;
mod trajectory_correctness;

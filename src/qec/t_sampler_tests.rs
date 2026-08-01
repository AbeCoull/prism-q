use super::*;

const TEST_SEED: u64 = 0xDEAD_BEEF;

struct AutoHookGuard;

impl AutoHookGuard {
    fn force_spd_nonexact_and_camps_failure() -> Self {
        super::auto_test_hooks::set_force_spd_nonexact(true);
        super::auto_test_hooks::set_force_camps_failure(true);
        Self
    }
}

impl Drop for AutoHookGuard {
    fn drop(&mut self) {
        super::auto_test_hooks::set_force_spd_nonexact(false);
        super::auto_test_hooks::set_force_camps_failure(false);
    }
}

fn options(shots: usize) -> QecOptions {
    QecOptions {
        shots,
        seed: TEST_SEED,
        chunk_size: Some(128),
        keep_measurements: false,
    }
}

fn h_t_h_program(shots: usize) -> QecProgram {
    let mut program = QecProgram::with_options(1, options(shots));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.push_gate(Gate::H, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
}

fn entangled_xor_program(shots: usize) -> QecProgram {
    let mut program = QecProgram::with_options(3, options(shots));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.push_gate(Gate::Cx, &[1, 2]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    for qubit in 0..3 {
        program.push_gate(Gate::H, &[qubit]).unwrap();
    }
    let mut records = Vec::new();
    for qubit in 0..3 {
        records.push(QecRecordRef::absolute(program.measure_z(qubit).unwrap()));
    }
    program.observable_include(0, &records).unwrap();
    program
}

fn postselected_program(shots: usize) -> QecProgram {
    let mut program = QecProgram::with_options(1, options(shots));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.reset(QecBasis::Z, 0).unwrap();
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    let m1 = program.measure_z(0).unwrap();
    program
        .postselect(&[QecRecordRef::absolute(m0)], false)
        .unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m1)])
        .unwrap();
    program
}

fn deterministic_t_zero_program(shots: usize) -> QecProgram {
    let mut program = QecProgram::with_options(1, options(shots));
    program.push_gate(Gate::T, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
}

fn deterministic_t_one_program(shots: usize) -> QecProgram {
    let mut program = QecProgram::with_options(1, options(shots));
    program.push_gate(Gate::X, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
}

fn estimates(result: &QecSampleResult) -> &[QecObservableEstimate] {
    result
        .observable_expectations
        .as_ref()
        .expect("analytical result must populate observable expectations")
}

fn assert_estimates_close(actual: &QecSampleResult, expected: &QecSampleResult) {
    let actual_estimates = estimates(actual);
    let expected_estimates = estimates(expected);
    assert_eq!(actual_estimates.len(), expected_estimates.len());
    for (idx, (actual, expected)) in actual_estimates
        .iter()
        .zip(expected_estimates.iter())
        .enumerate()
    {
        assert!(
            (actual.mean - expected.mean).abs() < 1e-10,
            "observable {idx}: tensor-network mean {} differs from expected {}",
            actual.mean,
            expected.mean
        );
        assert_eq!(actual.variance, 0.0);
    }
    assert_eq!(actual.accepted_shots, expected.accepted_shots);
    assert_eq!(actual.discarded_shots, expected.discarded_shots);
    assert_eq!(actual.logical_errors, expected.logical_errors);
}

#[test]
fn tensor_network_observable_matches_spd_on_small_non_truncated_programs() {
    let fixtures = vec![
        h_t_h_program(512),
        entangled_xor_program(512),
        postselected_program(512),
    ];
    for program in fixtures {
        let spd = run_qec_program_spd(&program).unwrap();
        assert!(analytical_result_has_no_truncation(&spd));
        let tensor_network = run_qec_program_tensor_network_observable(&program).unwrap();
        assert_estimates_close(&tensor_network, &spd);
    }
}

#[test]
fn tensor_network_observable_matches_reference_on_deterministic_t_fixtures() {
    let fixtures = vec![
        deterministic_t_zero_program(128),
        deterministic_t_one_program(128),
    ];
    for program in fixtures {
        let reference = run_qec_program_reference(&program).unwrap();
        let tensor_network = run_qec_program_tensor_network_observable(&program).unwrap();
        assert_eq!(tensor_network.accepted_shots, reference.accepted_shots);
        assert_eq!(tensor_network.discarded_shots, reference.discarded_shots);
        assert_eq!(tensor_network.logical_errors, reference.logical_errors);
    }
}

#[test]
fn auto_uses_tensor_network_when_spd_nonexact_and_camps_fails() {
    let program = h_t_h_program(256);
    let expected = run_qec_program_tensor_network_observable(&program).unwrap();
    let _guard = AutoHookGuard::force_spd_nonexact_and_camps_failure();
    let actual = run_qec_program_auto(&program).unwrap();
    assert_estimates_close(&actual, &expected);
}

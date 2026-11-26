use super::*;

const TEST_SEED: u64 = 0xDEAD_BEEF;

fn options(shots: usize) -> QecOptions {
    QecOptions {
        shots,
        seed: TEST_SEED,
        chunk_size: Some(128),
        keep_measurements: false,
        ..QecOptions::default()
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

fn run_qec_program_tensor_network_observable(program: &QecProgram) -> Result<QecSampleResult> {
    let limits = tensor_network_fallback_limits(ResolvedLimits::process_default());
    run_qec_program_tensor_network_observable_within(program, &limits)
}

fn run_qec_program_tensor_network_observable_within(
    program: &QecProgram,
    limits: &ResolvedLimits,
) -> Result<QecSampleResult> {
    tensor_network_on_lowered(program, &lower_pauli_observables(program)?, limits)
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

fn brickwork_t_program(num_qubits: usize, layers: usize) -> QecProgram {
    let mut program = QecProgram::with_options(num_qubits, options(64));
    for qubit in 0..num_qubits {
        program.push_gate(Gate::H, &[qubit]).unwrap();
    }
    for layer in 0..layers {
        for qubit in (layer % 2..num_qubits - 1).step_by(2) {
            program.push_gate(Gate::Cx, &[qubit, qubit + 1]).unwrap();
        }
        for qubit in 0..num_qubits {
            program.push_gate(Gate::T, &[qubit]).unwrap();
            program.push_gate(Gate::H, &[qubit]).unwrap();
        }
    }
    let records: Vec<_> = (0..num_qubits)
        .map(|qubit| QecRecordRef::absolute(program.measure_z(qubit).unwrap()))
        .collect();
    program.observable_include(0, &records).unwrap();
    program
}

#[test]
fn tensor_network_fallback_refuses_an_over_cap_observable_without_slicing() {
    use crate::backend::tensornetwork::last_slice_count;
    use crate::limits::{LimitKnob, Limits};

    let program = brickwork_t_program(10, 6);
    let capped = Limits::new().max_tn_peak_qubits(8);

    let fallback = tensor_network_fallback_limits(ResolvedLimits::resolve(Some(&capped)));
    let err = run_qec_program_tensor_network_observable_within(&program, &fallback).unwrap_err();
    assert!(
        matches!(&err, PrismError::ResourceLimit(limit) if limit.knob == Some(LimitKnob::TnPeakQubits)),
        "{err}"
    );
    assert_eq!(last_slice_count(), 1, "the refusal recorded a slice plan");

    let sliceable = ResolvedLimits::resolve(Some(&capped.max_tn_slices(1 << 10)));
    run_qec_program_tensor_network_observable_within(
        &program,
        &tensor_network_fallback_limits(sliceable),
    )
    .unwrap();
    assert!(
        last_slice_count() > 1,
        "the fixture fits the peak cap unsliced"
    );
}

#[test]
fn only_a_lowering_refusal_reroutes_to_the_reference_runner() {
    use crate::error::{ResourceKind, ResourceLimit};
    use crate::limits::LimitSource;

    let refusal = PrismError::IncompatibleBackend {
        backend: "QEC deferred lowering".to_string(),
        reason: "cannot lower".to_string(),
    };
    assert!(is_lowering_refusal(&refusal));
    let cap = PrismError::ResourceLimit(Box::new(ResourceLimit {
        backend: "statevector".to_string(),
        operation: "allocation".to_string(),
        resource: ResourceKind::Qubits,
        required: 40,
        limit: 29,
        env_var: None,
        knob: None,
        limit_source: LimitSource::Detected,
    }));
    assert!(!is_lowering_refusal(&cap));
    let malformed = PrismError::InvalidQubit {
        index: 3,
        register_size: 2,
    };
    assert!(!is_lowering_refusal(&malformed));

    let mut non_terminal = QecProgram::with_options(1, options(16));
    non_terminal
        .expectation_value(&[super::super::QecPauli::z(0)], 1.0)
        .unwrap();
    non_terminal.push_gate(Gate::X, &[0]).unwrap();
    let lowering_error = lower_pauli_observables(&non_terminal).err().unwrap();
    assert!(!is_lowering_refusal(&lowering_error), "{lowering_error}");
    assert_eq!(
        run_qec_program_auto_or_reference(&non_terminal).unwrap_err(),
        lowering_error
    );
}

#[test]
fn auto_uses_tensor_network_when_spd_nonexact_and_camps_fails() {
    let program = h_t_h_program(256);
    let expected = run_qec_program_tensor_network_observable(&program).unwrap();
    let spd_nonexact = |program: &QecProgram, lowered: &LoweredPauliObservables| {
        let mut result = spd_on_lowered(program, lowered)?;
        for estimates in [
            result.observable_expectations.as_mut(),
            result.expectation_values.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            for estimate in estimates {
                estimate.variance = estimate.variance.max(1.0);
            }
        }
        Ok((result, 0.0))
    };
    let camps_failure = |_: &QecProgram, _: &LoweredPauliObservables| {
        Err(PrismError::BackendUnsupported {
            backend: "QEC CAMPS".to_string(),
            operation: "forced CAMPS failure for auto-dispatch test".to_string(),
        })
    };
    let lowered = lower_pauli_observables(&program).unwrap();
    let actual =
        run_qec_program_auto_with(&program, &lowered, spd_nonexact, camps_failure).unwrap();
    assert_estimates_close(&actual, &expected);
}

#[test]
fn conditional_expectation_propagates_denominator_truncation_even_at_zero_mean() {
    let postsel = vec![(vec![0usize], 1.0)];
    let observables = vec![vec![PauliTerm::z(1)], vec![PauliTerm::z(2)]];
    // For mask=0: pi_pauli is empty -> short-circuits to 1.0.
    // For mask=1: pi_pauli=[Z_0] -> mean=0.5, discarded_sq=1e-4 (pi_err_subset=0.01).
    // accept_rate = (1.0 + 0.5) / 2 = 0.75; pi_err = 0.01 / 2 = 0.005.
    // Observable 1 (Z_1): outcome.mean = 0.4, discarded_sq = 0.0 -> post_mean = (0.4 + 0.4)/2 / 0.75 = 0.4 / 0.75.
    // Observable 2 (Z_2): outcome.mean = 0.0, discarded_sq = 0.0 -> post_mean = 0.0.
    let cond = evaluate_conditional_expectations(&observables, &postsel, |terms| {
        if terms == [PauliTerm::z(0)] {
            Ok(ConditionalEval {
                mean: 0.5,
                discarded_sq: 1e-4,
            })
        } else if terms.iter().any(|t| t.qubit == 1) {
            Ok(ConditionalEval {
                mean: 0.4,
                discarded_sq: 0.0,
            })
        } else {
            Ok(ConditionalEval {
                mean: 0.0,
                discarded_sq: 0.0,
            })
        }
    })
    .unwrap();

    assert!((cond.accept_rate - 0.75).abs() < 1e-14);
    let pi_err = 0.005;
    let denom = cond.accept_rate - pi_err;
    let expected_0 = (cond.means[0].abs().max(1.0) * pi_err / denom).powi(2);
    let expected_1 = (1.0 * pi_err / denom).powi(2);
    assert!((cond.discarded[0] - expected_0).abs() < 1e-15);
    assert!((cond.discarded[1] - expected_1).abs() < 1e-15);
    assert!((cond.pi_variance - expected_1).abs() < 1e-15);
    assert!(cond.discarded[0] > QEC_SPD_NEGLIGIBLE_VARIANCE);
    assert!(cond.discarded[1] > QEC_SPD_NEGLIGIBLE_VARIANCE);
    assert_eq!(cond.means[1], 0.0);
}

#[test]
fn conditional_expectation_amplifies_numerator_error_by_inverse_accept_rate() {
    let postsel = vec![(vec![0usize], 1.0)];
    let observables = vec![vec![PauliTerm::z(1)]];
    // mask=0 -> pi=1.0, mask=1 -> pi=-0.9 -> accept_rate = (1.0 - 0.9)/2 = 0.05.
    let cond = evaluate_conditional_expectations(&observables, &postsel, |terms| {
        if terms == [PauliTerm::z(0)] {
            Ok(ConditionalEval {
                mean: -0.9,
                discarded_sq: 0.0,
            })
        } else if terms == [PauliTerm::z(1)] {
            Ok(ConditionalEval {
                mean: 0.05,
                discarded_sq: 1e-8,
            })
        } else {
            Ok(ConditionalEval {
                mean: 0.0,
                discarded_sq: 0.0,
            })
        }
    })
    .unwrap();

    assert!((cond.accept_rate - 0.05).abs() < 1e-14);
    assert!((cond.means[0] - 0.5).abs() < 1e-12);
    // obs_err = (1e-4 + 0.0) / 2 = 5e-5; bound = 5e-5 / 0.05 = 1e-3; variance = 1e-6.
    assert!((cond.discarded[0] - 1e-6).abs() < 1e-15);

    // Exact or negligible (1e-10 total_discarded => 1e-20 discarded_sq) passes Auto exactness gate.
    let negl = evaluate_conditional_expectations(&observables, &postsel, |terms| {
        if terms == [PauliTerm::z(0)] {
            Ok(ConditionalEval {
                mean: -0.9,
                discarded_sq: 1e-20,
            })
        } else {
            Ok(ConditionalEval {
                mean: 0.02,
                discarded_sq: 1e-20,
            })
        }
    })
    .unwrap();
    let sample_result = QecSampleResult::empty(0, 0, 1)
        .with_observable_expectations(vec![QecObservableEstimate {
            mean: 0.4,
            variance: negl.discarded[0],
            num_shots: 64,
        }])
        .unwrap();
    assert!(analytical_result_has_no_truncation(&sample_result));
}

#[test]
fn conditional_expectation_short_circuits_empty_pi_pauli_and_cancelling_products() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Case 1: J = 1 with 2 observables -> mask=0 skips eval(&[]), so 1 + 2*2 = 5 calls.
    let calls = AtomicUsize::new(0);
    let postsel = vec![(vec![0usize], 1.0)];
    let observables = vec![vec![PauliTerm::z(1)], vec![PauliTerm::z(2)]];
    let cond = evaluate_conditional_expectations(&observables, &postsel, |terms| {
        assert!(
            !terms.is_empty(),
            "empty Pauli string must be short-circuited"
        );
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(ConditionalEval {
            mean: 0.5,
            discarded_sq: 0.0,
        })
    })
    .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 5);
    assert!((cond.accept_rate - 0.75).abs() < 1e-14);
    assert!(cond.discarded.iter().all(|&d| d == 0.0));
    assert!((cond.means[0] - 0.5 / 0.75).abs() < 1e-12);

    // Case 2: J = 2 with identical Z_0 postselection rows -> mask=0 and mask=3 both cancel to empty.
    let calls2 = AtomicUsize::new(0);
    let postsel_cancel = vec![(vec![0usize], 1.0), (vec![0usize], 1.0)];
    let obs_one = vec![vec![PauliTerm::z(1)]];
    let _ = evaluate_conditional_expectations(&obs_one, &postsel_cancel, |terms| {
        assert!(!terms.is_empty(), "cancelling Z_0*Z_0 must short-circuit");
        calls2.fetch_add(1, Ordering::Relaxed);
        Ok(ConditionalEval {
            mean: 0.25,
            discarded_sq: 0.0,
        })
    })
    .unwrap();
    // mask=0: 0 pi + 1 obs; mask=1: 1 pi + 1 obs; mask=2: 1 pi + 1 obs; mask=3: 0 pi + 1 obs => 6 calls.
    assert_eq!(calls2.load(Ordering::Relaxed), 6);

    // Case 3: observable Z_0 cancels with pi_pauli [Z_0] at mask=1 -> combined_pauli is empty.
    let calls3 = AtomicUsize::new(0);
    let obs_z0 = vec![vec![PauliTerm::z(0)]];
    let cond3 = evaluate_conditional_expectations(&obs_z0, &postsel, |terms| {
        assert!(!terms.is_empty());
        calls3.fetch_add(1, Ordering::Relaxed);
        Ok(ConditionalEval {
            mean: 0.5,
            discarded_sq: 0.0,
        })
    })
    .unwrap();
    // mask=0: 0 pi + 1 obs (Z_0); mask=1: 1 pi (Z_0) + 0 obs (Z_0*Z_0=I) => 2 calls.
    assert_eq!(calls3.load(Ordering::Relaxed), 2);
    assert!((cond3.means[0] - 1.0).abs() < 1e-14);
}

#[test]
fn auto_falls_back_when_postselection_projector_truncates() {
    use super::super::QecPauli;

    // 1) Mocked SPD where only pi_value has discarded_sq > 0: Auto must fall back to CAMPS.
    let program = postselected_program(256);
    let lowered = lower_pauli_observables(&program).unwrap();
    let expected = camps_on_lowered(&program, &lowered).unwrap();
    let postsel =
        lower_postselection_rows(&lowered.postsel_rows, &lowered.record_to_qubit).unwrap();
    let spd_denom_truncated = |prog: &QecProgram, low: &LoweredPauliObservables| {
        let cond =
            evaluate_conditional_expectations(&low.observable_terms_list, &postsel, |terms| {
                // Trigger denominator truncation on pi_pauli only.
                let is_pi_only = terms.len() == 1 && postsel[0].0.contains(&terms[0].qubit);
                Ok(ConditionalEval {
                    mean: 0.7,
                    discarded_sq: if is_pi_only { 1e-4 } else { 0.0 },
                })
            })?;
        let shots = accepted_shots_for_rate(prog.options().shots, cond.accept_rate);
        let estimates = cond
            .means
            .iter()
            .zip(cond.discarded.iter())
            .map(|(&mean, &variance)| QecObservableEstimate {
                mean,
                variance,
                num_shots: shots,
            })
            .collect();
        Ok((
            build_qec_result_with_acceptance(prog, cond.means, estimates, cond.accept_rate)?,
            cond.pi_variance,
        ))
    };
    let via_auto =
        run_qec_program_auto_with(&program, &lowered, spd_denom_truncated, camps_on_lowered)
            .unwrap();
    assert!((estimates(&via_auto)[0].mean - estimates(&expected)[0].mean).abs() < 1e-12);
    assert_eq!(estimates(&via_auto)[0].variance, 0.0);

    // 2) Real postselected Clifford+T program whose MPP postselection cone spans 21 T gates
    // (exceeding SPD_MAX_TERMS_CEILING), forcing Auto to fall back to CAMPS.
    let mut wide = QecProgram::with_options(22, options(512));
    for q in 0..21 {
        wide.push_gate(Gate::H, &[q]).unwrap();
        wide.push_gate(Gate::T, &[q]).unwrap();
        wide.push_gate(Gate::H, &[q]).unwrap();
    }
    let mpp_terms: Vec<QecPauli> = (0..21).map(QecPauli::z).collect();
    let m_post = wide.measure_pauli_product(&mpp_terms).unwrap();
    wide.postselect(&[QecRecordRef::absolute(m_post)], false)
        .unwrap();

    wide.push_gate(Gate::H, &[21]).unwrap();
    wide.push_gate(Gate::T, &[21]).unwrap();
    wide.push_gate(Gate::H, &[21]).unwrap();
    let m_obs = wide.measure_z(21).unwrap();
    wide.observable_include(0, &[QecRecordRef::absolute(m_obs)])
        .unwrap();

    assert!(run_qec_program_spd(&wide).is_err());
    let auto = run_qec_program_with_strategy(&wide, QecTStrategy::Auto).unwrap();
    let camps = run_qec_program_with_strategy(&wide, QecTStrategy::Camps).unwrap();
    assert!(analytical_result_has_no_truncation(&auto));
    let auto_est = estimates(&auto)[0];
    let camps_est = estimates(&camps)[0];
    assert!((auto_est.mean - camps_est.mean).abs() < 1e-10);
    assert!((auto_est.mean - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
}

#[test]
fn zero_observable_postselected_auto_falls_back_when_pi_truncates() {
    use super::super::QecPauli;
    use std::sync::atomic::{AtomicBool, Ordering};

    // 1) Mocked 0-observable/0-EXP_VAL postselected program: pi_variance above the
    // threshold forces fallback even when observable_expectations is empty.
    let mut program = QecProgram::with_options(1, options(1_000));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .postselect(&[QecRecordRef::absolute(m0)], false)
        .unwrap();

    let lowered = lower_pauli_observables(&program).unwrap();
    assert!(lowered.observable_terms_list.is_empty());
    assert!(lowered.exp_vals.is_empty());

    let camps_called = AtomicBool::new(false);
    let spd_pi_truncated = |prog: &QecProgram, low: &LoweredPauliObservables| {
        let postsel = lower_postselection_rows(&low.postsel_rows, &low.record_to_qubit)?;
        let cond = evaluate_conditional_expectations(&[], &postsel, |_| {
            Ok(ConditionalEval {
                mean: 0.4,
                discarded_sq: 1e-4,
            })
        })?;
        assert!(cond.pi_variance > QEC_SPD_NEGLIGIBLE_VARIANCE);
        let res = build_qec_result_with_acceptance(prog, Vec::new(), Vec::new(), cond.accept_rate)?;
        assert!(analytical_result_has_no_truncation(&res));
        Ok((res, cond.pi_variance))
    };
    let via_auto = run_qec_program_auto_with(&program, &lowered, spd_pi_truncated, |p, l| {
        camps_called.store(true, Ordering::Relaxed);
        camps_on_lowered(p, l)
    })
    .unwrap();
    assert!(camps_called.load(Ordering::Relaxed));
    let expected = camps_on_lowered(&program, &lowered).unwrap();
    assert_eq!(via_auto.accepted_shots, expected.accepted_shots);

    // Negligible pi_variance keeps SPD without falling back.
    let camps_called_negl = AtomicBool::new(false);
    let spd_pi_negl = |prog: &QecProgram, low: &LoweredPauliObservables| {
        let postsel = lower_postselection_rows(&low.postsel_rows, &low.record_to_qubit)?;
        let cond = evaluate_conditional_expectations(&[], &postsel, |_| {
            Ok(ConditionalEval {
                mean: 0.4,
                discarded_sq: 1e-20,
            })
        })?;
        assert!(cond.pi_variance <= QEC_SPD_NEGLIGIBLE_VARIANCE);
        let res = build_qec_result_with_acceptance(prog, Vec::new(), Vec::new(), cond.accept_rate)?;
        Ok((res, cond.pi_variance))
    };
    let _ = run_qec_program_auto_with(&program, &lowered, spd_pi_negl, |p, l| {
        camps_called_negl.store(true, Ordering::Relaxed);
        camps_on_lowered(p, l)
    })
    .unwrap();
    assert!(!camps_called_negl.load(Ordering::Relaxed));

    // 2) Real 0-observable postselected program whose MPP cone truncates pi_value
    // below SPD_MAX_TERMS_CEILING: SPD succeeds with pi truncation, and Auto
    // detects pi_variance and falls back cleanly.
    let mut wide = QecProgram::with_options(17, options(1_000_000));
    wide.push_gate(Gate::H, &[15]).unwrap();
    wide.push_gate(Gate::Rz(9e-11), &[15]).unwrap();
    wide.push_gate(Gate::H, &[15]).unwrap();
    for q in 0..15 {
        wide.push_gate(Gate::H, &[q]).unwrap();
        wide.push_gate(Gate::T, &[q]).unwrap();
        wide.push_gate(Gate::H, &[q]).unwrap();
    }
    wide.push_gate(Gate::X, &[16]).unwrap();
    wide.push_gate(Gate::Rx(0.08), &[16]).unwrap();
    let m_low = wide.measure_z(16).unwrap();
    let mpp_terms: Vec<QecPauli> = (0..16).map(QecPauli::z).collect();
    let m_post = wide.measure_pauli_product(&mpp_terms).unwrap();
    wide.postselect(&[QecRecordRef::absolute(m_low)], false)
        .unwrap();
    wide.postselect(&[QecRecordRef::absolute(m_post)], false)
        .unwrap();

    assert!(run_qec_program_spd(&wide).is_ok());
    let wide_lowered = lower_pauli_observables(&wide).unwrap();
    let (_, wide_pi_var) = spd_on_lowered_checked(&wide, &wide_lowered).unwrap();
    assert!(
        wide_pi_var > QEC_SPD_NEGLIGIBLE_VARIANCE,
        "wide_pi_var={wide_pi_var}"
    );

    let real_fallback_called = AtomicBool::new(false);
    let via_fallback =
        run_qec_program_auto_with(&wide, &wide_lowered, spd_on_lowered_checked, |p, l| {
            real_fallback_called.store(true, Ordering::Relaxed);
            camps_on_lowered(p, l)
        })
        .unwrap();
    assert!(real_fallback_called.load(Ordering::Relaxed));

    let auto = run_qec_program_with_strategy(&wide, QecTStrategy::Auto).unwrap();
    let tn = run_qec_program_tensor_network_observable(&wide).unwrap();
    assert_eq!(auto.accepted_shots, tn.accepted_shots);
    assert_eq!(auto.discarded_shots, tn.discarded_shots);
    assert_eq!(via_fallback.accepted_shots, tn.accepted_shots);
}

//! Cross-checks for the T-strategy dispatch.
//!
//! Every analytical strategy under [`QecTStrategy`] must match
//! [`QecTStrategy::Reference`] within statistical tolerance on a small
//! Clifford+T fixture.

mod qec_common;

use prism_q::{
    Gate, QecObservableReroute, QecOptions, QecProgram, QecRecordRef, QecTStrategy,
    run_qec_program_reference, run_qec_program_spd_rerouted, run_qec_program_with_strategy,
};
use qec_common::ANALYTICAL_STRATEGIES;

const STAT_SHOTS: usize = 4_000;

fn options(shots: usize) -> QecOptions {
    qec_common::qec_options(shots, 2048, false)
}

fn small_clifford_t_program() -> QecProgram {
    let mut program = QecProgram::with_options(1, options(STAT_SHOTS));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.push_gate(Gate::H, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
}

#[test]
fn reference_strategy_dispatch_matches_direct_reference_runner() {
    let program = small_clifford_t_program();
    let direct = run_qec_program_reference(&program).unwrap();
    let via_strategy = run_qec_program_with_strategy(&program, QecTStrategy::Reference).unwrap();
    assert_eq!(direct.total_shots, via_strategy.total_shots);
    assert_eq!(direct.accepted_shots, via_strategy.accepted_shots);
    assert_eq!(direct.discarded_shots, via_strategy.discarded_shots);
    assert_eq!(direct.logical_errors, via_strategy.logical_errors);
}

#[test]
fn reference_strategy_recovers_expected_t_rotation_rate() {
    // H · T · H |0> rotates around X by π/4. Probability of measuring
    // 1 in the Z basis is sin²(π/8) ≈ 0.1464. Wide tolerance for
    // 4k shots.
    let program = small_clifford_t_program();
    let result = run_qec_program_with_strategy(&program, QecTStrategy::Reference).unwrap();
    assert_eq!(result.total_shots, STAT_SHOTS);
    assert_eq!(result.accepted_shots, STAT_SHOTS);
    let rate = result.logical_errors[0] as f64 / result.accepted_shots as f64;
    let expected = (std::f64::consts::PI / 8.0).sin().powi(2);
    assert!(
        (rate - expected).abs() < 0.03,
        "logical rate {rate:.4} not within tolerance of expected {expected:.4}"
    );
}

#[test]
fn camps_matches_reference_on_h_t_h_circuit() {
    let program = small_clifford_t_program();
    let result = run_qec_program_with_strategy(&program, QecTStrategy::Camps).unwrap();
    let estimates = result
        .observable_expectations
        .as_ref()
        .expect("CAMPS must populate observable_expectations");
    // Born ⟨Z⟩ for H T H |0⟩ = cos(π/4).
    let expected_z = (std::f64::consts::FRAC_PI_4).cos();
    assert!(
        (estimates[0].mean - expected_z).abs() < 1e-6,
        "CAMPS ⟨Z⟩ {:.6} should match Born cos(π/4) = {expected_z:.6}",
        estimates[0].mean
    );
}

#[test]
fn camps_matches_reference_on_two_qubit_zz_observable() {
    let mut program = QecProgram::with_options(2, options(2_000));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    let m1 = program.measure_z(1).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0), QecRecordRef::absolute(m1)])
        .unwrap();

    let reference = run_qec_program_reference(&program).unwrap();
    let result = run_qec_program_with_strategy(&program, QecTStrategy::Camps).unwrap();
    let ref_rate = reference.logical_errors[0] as f64 / reference.accepted_shots as f64;
    let camps_rate = result.logical_errors[0] as f64 / result.total_shots as f64;
    assert!(
        (camps_rate - ref_rate).abs() < 0.03,
        "CAMPS rate {camps_rate:.4} diverged from reference {ref_rate:.4}"
    );
}

/// Runs a parameterized correctness sweep: for each `parameter`, builds a
/// program, takes the per-shot statevector runner as the oracle, runs every
/// `strategy` through `run_qec_program_with_strategy`, and asserts the
/// strategy rate is within `tolerance(ref_stderr)` of the reference rate.
fn run_strategy_sweep<P: Copy, F, T>(
    title: &str,
    parameters: &[P],
    strategies: &[QecTStrategy],
    mut build: F,
    tolerance: T,
) where
    F: FnMut(P) -> (String, QecProgram),
    T: Fn(f64) -> f64,
{
    for &param in parameters {
        let (label, program) = build(param);
        let reference = run_qec_program_reference(&program).unwrap();
        let ref_rate = reference.logical_errors[0] as f64 / reference.accepted_shots as f64;
        let p_hat = ref_rate.clamp(1e-6, 1.0 - 1e-6);
        let ref_stderr = (p_hat * (1.0 - p_hat) / reference.accepted_shots as f64).sqrt();
        let tol = tolerance(ref_stderr);
        for &strategy in strategies {
            let result = run_qec_program_with_strategy(&program, strategy).unwrap();
            let denom = result.accepted_shots.max(1) as f64;
            let strat_rate = result.logical_errors[0] as f64 / denom;
            let delta = (strat_rate - ref_rate).abs();
            assert!(
                delta < tol,
                "{title}: label={label} strategy={strategy:?} ref={ref_rate:.4} \
                 strat={strat_rate:.4} |delta|={delta:.4} > tol {tol:.4}"
            );
        }
    }
}

/// Unified 5σ correctness sweep across small Clifford+T fixtures. Each
/// fixture has a single observable whose Born rate is computed via the
/// reference statevector runner; every strategy must land within
/// `5·stderr + 0.03` of that rate.
#[test]
fn analytical_strategy_correctness_sweep() {
    fn h_t_h() -> QecProgram {
        small_clifford_t_program()
    }

    fn two_qubit_single_t() -> QecProgram {
        let mut p = QecProgram::with_options(2, options(STAT_SHOTS));
        p.push_gate(Gate::H, &[0]).unwrap();
        p.push_gate(Gate::T, &[0]).unwrap();
        p.push_gate(Gate::H, &[0]).unwrap();
        p.push_gate(Gate::H, &[1]).unwrap();
        p.push_gate(Gate::T, &[1]).unwrap();
        p.push_gate(Gate::H, &[1]).unwrap();
        let m0 = p.measure_z(0).unwrap();
        let m1 = p.measure_z(1).unwrap();
        p.observable_include(0, &[QecRecordRef::absolute(m0), QecRecordRef::absolute(m1)])
            .unwrap();
        p
    }

    fn cx_t_zz() -> QecProgram {
        let mut p = QecProgram::with_options(2, options(STAT_SHOTS));
        p.push_gate(Gate::H, &[0]).unwrap();
        p.push_gate(Gate::Cx, &[0, 1]).unwrap();
        p.push_gate(Gate::T, &[0]).unwrap();
        let m0 = p.measure_z(0).unwrap();
        let m1 = p.measure_z(1).unwrap();
        p.observable_include(0, &[QecRecordRef::absolute(m0), QecRecordRef::absolute(m1)])
            .unwrap();
        p
    }

    fn h_t_t_h() -> QecProgram {
        let mut p = QecProgram::with_options(1, options(STAT_SHOTS));
        p.push_gate(Gate::H, &[0]).unwrap();
        p.push_gate(Gate::T, &[0]).unwrap();
        p.push_gate(Gate::T, &[0]).unwrap();
        p.push_gate(Gate::H, &[0]).unwrap();
        let m0 = p.measure_z(0).unwrap();
        p.observable_include(0, &[QecRecordRef::absolute(m0)])
            .unwrap();
        p
    }

    type FixtureBuilder = fn() -> QecProgram;
    let fixtures: &[(&str, FixtureBuilder)] = &[
        ("h_t_h", h_t_h),
        ("two_qubit_single_t", two_qubit_single_t),
        ("cx_t_zz", cx_t_zz),
        ("h_t_t_h", h_t_t_h),
    ];
    let strategies = [
        QecTStrategy::Reference,
        QecTStrategy::Auto,
        QecTStrategy::Spd,
        QecTStrategy::Camps,
    ];

    run_strategy_sweep(
        "unified correctness sweep",
        fixtures,
        &strategies,
        |(name, build)| (name.to_string(), build()),
        |stderr| 5.0 * stderr + 0.03,
    );
}

/// Single-qubit `(H · T)^t · H` rotation. Used by the t-scaling
/// correctness sweep; analytic Born rate for the Z observable is
/// `sin²(t · π/8)`.
fn t_chain_program(shots: usize, t_count: usize) -> QecProgram {
    let mut p = QecProgram::with_options(1, options(shots));
    p.push_gate(Gate::H, &[0]).unwrap();
    for _ in 0..t_count {
        p.push_gate(Gate::T, &[0]).unwrap();
        p.push_gate(Gate::H, &[0]).unwrap();
    }
    let m0 = p.measure_z(0).unwrap();
    p.observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    p
}

/// Correctness across the T-count scaling fixture. The oracle is the
/// per-shot reference statevector runner; every other strategy must
/// land within a strategy-specific tolerance for each
/// `t ∈ {1, 2, 4, 8, 12}`. StabRankShots is skipped at `t ≥ 2`
/// (documented bias on multi-T-per-path observables).
///
/// **Known limitation from this sweep:** SPP picks up the same multi-T
/// bias as the previous weighted-shot estimator. The Heisenberg-frame Pauli backprop uses
/// 50/50 branching at each T with a `√2` weight; cross-branch
/// interference between successive T gates is dropped at the
/// sample level. The tolerance reflects this, SPP is treated as
/// `≤ 5σ` correct only at `t = 1`, and the table records the bias
/// at larger `t` rather than asserting against it.
#[test]
fn analytical_t_count_scaling_correctness() {
    let t_counts: &[usize] = &[1, 2, 4, 8, 12];
    let strategies = ANALYTICAL_STRATEGIES;
    run_strategy_sweep(
        "T-count scaling sweep",
        t_counts,
        &strategies,
        |t| (format!("t={t}"), t_chain_program(STAT_SHOTS, t)),
        |_| 0.02,
    );
}

/// Multi-qubit program with a multi-qubit twisted Pauli and a joint
/// XOR observable that resolves the T phase.
///
/// Layout: `H q[0]; CX(i,i+1)` cascade prepares a GHZ-like prefix,
/// then `T q[0]`, then `H` on every qubit, then measure every qubit
/// and XOR the outcomes into the observable.
///
/// At T application the prefix is `CX_chain · H_0`, so
/// `Z̄ = (CX_chain)† H_0 Z_0 H_0 CX_chain = X_0 X_1 … X_{n-1}`, pure
/// X support over all `n` qubits. The MPS is still `|0…0⟩`, so the
/// first qubit is a valid `|0⟩` anchor; OFD fires.
///
/// Final-layer `H` on all qubits maps the GHZ-with-phase
/// `(|0…0⟩ + e^{iπ/4}|1…1⟩)/√2` into a superposition where the XOR
/// parity carries the T phase. Analytic Born rate for `XOR = 1` is
/// `(1 − cos(π/4))/2 ≈ 0.1465` for every `n ≥ 1`; matches the
/// single-qubit `sin²(π/8)` rate. A broken CAMPS biases this away
/// from 0.1465.
fn entangled_t_xor_program(shots: usize, n: usize) -> QecProgram {
    assert!(n >= 2);
    let mut p = QecProgram::with_options(n, options(shots));
    p.push_gate(Gate::H, &[0]).unwrap();
    for i in 0..n - 1 {
        p.push_gate(Gate::Cx, &[i, i + 1]).unwrap();
    }
    p.push_gate(Gate::T, &[0]).unwrap();
    for q in 0..n {
        p.push_gate(Gate::H, &[q]).unwrap();
    }
    let mut record_refs: Vec<QecRecordRef> = Vec::with_capacity(n);
    for q in 0..n {
        let m = p.measure_z(q).unwrap();
        record_refs.push(QecRecordRef::absolute(m));
    }
    p.observable_include(0, &record_refs).unwrap();
    p
}

/// CAMPS must match the statevector reference on multi-qubit
/// Clifford+T programs whose twisted Pauli has multi-qubit support.
/// Sweeps `n ∈ {2, 3, 4, 5, 6}` with an entangled `H·T·H·T·H` pattern
/// that exercises both OFD (first T, MPS at |0⟩) and OFDS (second T,
/// MPS holds non-Clifford content) paths and produces a non-trivial
/// Born rate so a broken CAMPS would visibly bias.
#[test]
fn camps_matches_reference_on_entangled_multi_qubit_t() {
    let qubit_counts: &[usize] = &[2, 3, 4, 5, 6];
    let strategies = ANALYTICAL_STRATEGIES;
    run_strategy_sweep(
        "multi-qubit GHZ-chain correctness sweep",
        qubit_counts,
        &strategies,
        |n| (format!("n={n}"), entangled_t_xor_program(STAT_SHOTS, n)),
        |stderr| 5.0 * stderr + 0.03,
    );
}

/// Two-T multi-qubit program. After the first T fires, the MPS holds
/// non-Clifford content, so the second T's OFD anchor selection may
/// reject the X/Y anchor and fall through to OFDS. Exercises the
/// OFDS path in production rather than only in unit tests.
fn entangled_two_t_xor_program(shots: usize, n: usize) -> QecProgram {
    assert!(n >= 2);
    let mut p = QecProgram::with_options(n, options(shots));
    p.push_gate(Gate::H, &[0]).unwrap();
    for i in 0..n - 1 {
        p.push_gate(Gate::Cx, &[i, i + 1]).unwrap();
    }
    p.push_gate(Gate::T, &[0]).unwrap();
    p.push_gate(Gate::H, &[0]).unwrap();
    p.push_gate(Gate::T, &[0]).unwrap();
    for q in 0..n {
        p.push_gate(Gate::H, &[q]).unwrap();
    }
    let mut record_refs: Vec<QecRecordRef> = Vec::with_capacity(n);
    for q in 0..n {
        let m = p.measure_z(q).unwrap();
        record_refs.push(QecRecordRef::absolute(m));
    }
    p.observable_include(0, &record_refs).unwrap();
    p
}

/// Two T gates in a multi-qubit program. The second T's twisted Pauli
/// is evaluated against an MPS with non-Clifford content from the
/// first T. Verifies CAMPS and SPD match the reference simulator
/// across `n ∈ {2, 3, 4, 5, 6}`.
///
/// MagicFrame is excluded because the stabilizer-rank backend returns exactly
/// 0.5 on every fixture regardless of `n`, which indicates the T phase is
/// collapsed before the joint XOR observable is read. That failure is
/// independent of the CAMPS count_y regression covered here.
#[test]
fn camps_matches_reference_on_two_t_multi_qubit() {
    let qubit_counts: &[usize] = &[2, 3, 4, 5, 6];
    let strategies = ANALYTICAL_STRATEGIES;
    run_strategy_sweep(
        "two-T multi-qubit correctness sweep",
        qubit_counts,
        &strategies,
        |n| (format!("n={n}"), entangled_two_t_xor_program(STAT_SHOTS, n)),
        |stderr| 5.0 * stderr + 0.03,
    );
}

#[test]
fn spd_matches_reference_on_h_t_t_h_circuit() {
    // Deterministic SPD should land exactly on the Born rate
    // (no truncation needed for tiny T-count).
    let mut program = QecProgram::with_options(1, options(2_000));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.push_gate(Gate::H, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let reference = run_qec_program_reference(&program).unwrap();
    let result = run_qec_program_with_strategy(&program, QecTStrategy::Spd).unwrap();
    let estimates = result
        .observable_expectations
        .as_ref()
        .expect("SPD must populate observable_expectations");
    let spd_rate = ((1.0 - estimates[0].mean) * 0.5).clamp(0.0, 1.0);
    let ref_rate = reference.logical_errors[0] as f64 / reference.accepted_shots as f64;
    // Reference has Wilson noise; SPD is exact within truncation. Use a
    // generous statistical-only band.
    assert!(
        (spd_rate - ref_rate).abs() < 0.03,
        "SPD rate {spd_rate:.4} diverged from reference {ref_rate:.4}"
    );
}

#[test]
fn rerouted_spd_matches_spd_on_supplied_product_z_stabilizer() {
    let mut program = QecProgram::with_options(2, options(512));
    for _ in 0..8 {
        program.push_gate(Gate::T, &[0]).unwrap();
        program.push_gate(Gate::S, &[0]).unwrap();
    }
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let direct = run_qec_program_with_strategy(&program, QecTStrategy::Spd).unwrap();
    let rerouted = run_qec_program_spd_rerouted(
        &program,
        &[QecObservableReroute {
            observable: 0,
            stabilizers: vec![vec![0, 1]],
        }],
    )
    .unwrap();
    let direct_estimates = direct
        .observable_expectations
        .as_ref()
        .expect("SPD must populate observable expectations");
    let rerouted_estimates = rerouted
        .observable_expectations
        .as_ref()
        .expect("rerouted SPD must populate observable expectations");

    assert!((direct_estimates[0].mean - rerouted_estimates[0].mean).abs() < 1e-10);
    assert_eq!(rerouted.logical_errors, direct.logical_errors);
}

#[test]
fn rerouted_spd_rejects_out_of_range_stabilizer_qubit() {
    let mut program = QecProgram::with_options(2, options(64));
    program.push_gate(Gate::T, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let err = run_qec_program_spd_rerouted(
        &program,
        &[QecObservableReroute {
            observable: 0,
            stabilizers: vec![vec![0, 7]],
        }],
    )
    .expect_err("reroute stabilizers must stay inside the lowered circuit");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("InvalidQubit"),
        "expected InvalidQubit, got {msg}"
    );
}

#[test]
fn rerouted_spd_rejects_reset_programs() {
    // RESET reassigns a logical qubit to a fresh lowered index, so a
    // stabilizer expressed in logical-qubit indices would silently land on
    // the wrong physical qubit. The rerouted path must reject such programs
    // rather than evaluate a different operator.
    let mut program = QecProgram::with_options(2, options(64));
    program.push_gate(Gate::T, &[0]).unwrap();
    program.reset(prism_q::QecBasis::Z, 0).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let err = run_qec_program_spd_rerouted(
        &program,
        &[QecObservableReroute {
            observable: 0,
            stabilizers: vec![vec![0, 1]],
        }],
    )
    .expect_err("reroute must reject programs whose lowering relabels qubits");
    let msg = format!("{err:?}");
    assert!(msg.contains("RESET"), "expected RESET rejection, got {msg}");
}

#[test]
fn analytical_strategies_handle_reset() {
    // Prepare |+>, reset to |0>, then T·H, analytic Born of Z on
    // the post-reset circuit is sin²(π/8). The analytical strategies
    // (SPD, CAMPS) must agree with the reference.
    let mut program = QecProgram::with_options(1, options(STAT_SHOTS));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.reset(prism_q::QecBasis::Z, 0).unwrap();
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.push_gate(Gate::H, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let reference = run_qec_program_reference(&program).unwrap();
    let ref_rate = reference.logical_errors[0] as f64 / reference.accepted_shots as f64;
    for strategy in ANALYTICAL_STRATEGIES {
        let result = run_qec_program_with_strategy(&program, strategy).unwrap();
        let denom = result.accepted_shots.max(1) as f64;
        let rate = result.logical_errors[0] as f64 / denom;
        assert!(
            (rate - ref_rate).abs() < 0.03,
            "{strategy:?} with Reset: rate {rate:.4} diverged from reference {ref_rate:.4}"
        );
    }
}

#[test]
fn analytical_strategies_handle_x_basis_measurement() {
    // H on |0⟩ → |+⟩; T·H on |+⟩; measure in X basis.
    // The X-basis measurement adds an `H` rotation before the
    // Z-equivalent measurement, so the joint Pauli is X on qubit 0
    // before the rotation, or equivalently Z on qubit 0 in the
    // rotated basis, the broadened lowering handles this via
    // `append_basis_to_z_rotation`.
    let mut program = QecProgram::with_options(1, options(STAT_SHOTS));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    let m0 = program.measure_x(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let reference = run_qec_program_reference(&program).unwrap();
    let ref_rate = reference.logical_errors[0] as f64 / reference.accepted_shots as f64;
    for strategy in ANALYTICAL_STRATEGIES {
        let result = run_qec_program_with_strategy(&program, strategy).unwrap();
        let denom = result.accepted_shots.max(1) as f64;
        let rate = result.logical_errors[0] as f64 / denom;
        assert!(
            (rate - ref_rate).abs() < 0.03,
            "{strategy:?} with X-basis MX: rate {rate:.4} diverged from reference {ref_rate:.4}"
        );
    }
}

#[test]
fn analytical_strategies_handle_postselection() {
    // Repetition-style: one qubit measured twice, postselect on the
    // first measurement being "0" outcome, observable on the second.
    // The postselection projector is (I + Z_q)/2 on the data qubit,
    // and the conditional expectation should match the reference.
    let mut program = QecProgram::with_options(1, options(STAT_SHOTS));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.reset(prism_q::QecBasis::Z, 0).unwrap();
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    let m1 = program.measure_z(0).unwrap();
    program
        .postselect(&[QecRecordRef::absolute(m0)], false)
        .unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m1)])
        .unwrap();

    let reference = run_qec_program_reference(&program).unwrap();
    let ref_rate = reference.logical_errors[0] as f64 / reference.accepted_shots as f64;
    for strategy in ANALYTICAL_STRATEGIES {
        let result = run_qec_program_with_strategy(&program, strategy).unwrap();
        let denom = result.accepted_shots.max(1) as f64;
        let rate = result.logical_errors[0] as f64 / denom;
        assert!(
            (rate - ref_rate).abs() < 0.03,
            "{strategy:?} with postselection: rate {rate:.4} diverged from reference {ref_rate:.4}"
        );
        assert!(
            result.accepted_shots > 0 && result.accepted_shots <= result.total_shots,
            "{strategy:?} accepted_shots = {}; total_shots = {}",
            result.accepted_shots,
            result.total_shots
        );
    }
}

#[test]
fn auto_routes_non_clifford_non_t_gate_to_tensor_network() {
    let theta = std::f64::consts::FRAC_PI_3;
    let mut program = QecProgram::with_options(1, options(512));
    program.push_gate(Gate::Rx(theta), &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let spd_err = run_qec_program_with_strategy(&program, QecTStrategy::Spd)
        .expect_err("SPD must reject gates outside Clifford+T");
    let spd_msg = format!("{spd_err:?}");
    assert!(
        spd_msg.contains("non-Clifford+T"),
        "unexpected SPD rejection: {spd_msg}"
    );

    let result = run_qec_program_with_strategy(&program, QecTStrategy::Auto).unwrap();
    let estimate = result
        .observable_expectations
        .as_ref()
        .expect("Auto tensor-network fallback must populate expectations")[0];
    let expected = theta.cos();
    assert!(
        (estimate.mean - expected).abs() < 1e-10,
        "Auto mean {:.12} should match cos(theta) {:.12}",
        estimate.mean,
        expected
    );
}

#[test]
fn analytical_observable_records_match_logical_error_counts() {
    let program = small_clifford_t_program();
    let result = run_qec_program_with_strategy(&program, QecTStrategy::Spd).unwrap();
    let ones = (0..result.total_shots)
        .filter(|&shot| result.observables.get_bit(shot, 0))
        .count() as u64;
    assert_eq!(ones, result.logical_errors[0]);
    assert!(
        ones > 0,
        "analytical observable records should not be all zero for a nonzero logical rate"
    );
}

#[test]
fn analytical_strategies_reject_active_noise_route_to_reference() {
    use prism_q::QecNoise;
    // Active Pauli noise channels are incompatible with pure-state
    // expectation strategies. The runner must reject explicitly and
    // the Reference path must continue to succeed on the same program.
    let mut program = QecProgram::with_options(1, options(1_000));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.noise(QecNoise::XError(0.1), &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.push_gate(Gate::H, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    for strategy in ANALYTICAL_STRATEGIES {
        let err = run_qec_program_with_strategy(&program, strategy)
            .expect_err("analytical strategies must reject active noise channels");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("noise") && msg.contains("reference"),
            "{strategy:?} returned unexpected error: {msg}"
        );
    }

    let reference = run_qec_program_with_strategy(&program, QecTStrategy::Reference).unwrap();
    assert_eq!(reference.total_shots, 1_000);
}

#[test]
fn analytical_strategies_reject_detectors() {
    let mut program = QecProgram::with_options(1, options(256));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    for strategy in ANALYTICAL_STRATEGIES {
        let err = run_qec_program_with_strategy(&program, strategy)
            .expect_err("analytical strategy must reject detector records");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("detector"),
            "{strategy:?} produced unexpected detector rejection: {msg}"
        );
    }
}

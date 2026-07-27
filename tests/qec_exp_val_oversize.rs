//! Oversize guard for the noisy `EXP_VAL` density-matrix route.
//!
//! Isolated in its own test binary: it overrides `PRISM_MAX_DM_QUBITS`, which
//! `max_density_matrix_qubits()` caches per process, so it must not share a
//! process with tests that expect the real memory-derived cap.

mod qec_common;

use prism_q::{Gate, QecNoise, QecPauli, QecProgram};

#[test]
fn noisy_exp_val_beyond_the_density_matrix_cap_keeps_the_reference_path() {
    // SAFETY: single test in this binary; the variable is set before any cap
    // query and no other thread is running.
    unsafe { std::env::set_var("PRISM_MAX_DM_QUBITS", "1") };

    let mut program = QecProgram::with_options(2, qec_common::qec_options(512, 2048, false));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.noise(QecNoise::XError(0.3), &[0]).unwrap();
    program
        .expectation_value(&[QecPauli::z(0), QecPauli::z(1)], 1.0)
        .unwrap();

    let routed =
        qec_common::assert_matches_reference_runner(&program, "over the density-matrix cap");
    assert!(
        qec_common::estimates(&routed)[0].variance > 0.0,
        "a program over the density-matrix cap must keep the sampled reference path"
    );
}

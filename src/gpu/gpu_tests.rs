use super::*;
use crate::error::PrismError;

#[test]
fn stub_context_reports_available_false() {
    let ctx = GpuContext::stub_for_tests();
    assert!(ctx.device().is_stub());
}

#[test]
fn state_new_on_stub_returns_unsupported() {
    let ctx = GpuContext::stub_for_tests();
    assert!(matches!(
        GpuState::new(ctx, 4).unwrap_err(),
        PrismError::BackendUnsupported { .. }
    ));
}

#[test]
fn tableau_new_on_stub_returns_unsupported() {
    let ctx = GpuContext::stub_for_tests();
    assert!(matches!(
        GpuTableau::new(ctx, 4).unwrap_err(),
        PrismError::BackendUnsupported { .. }
    ));
}

#[test]
fn min_qubits_default_when_env_unset() {
    // The value is cached per process and the env var may override it, so only
    // plausibility is checked.
    let n = min_qubits();
    assert!(
        (1..=32).contains(&n),
        "implausible gpu crossover threshold: {n}"
    );
}

#[test]
fn stub_vram_available_rejects_cleanly() {
    let ctx = GpuContext::stub_for_tests();
    assert!(matches!(
        ctx.vram_available().unwrap_err(),
        PrismError::BackendUnsupported { .. }
    ));
}

#[test]
fn stub_fits_statevector_rejects_cleanly() {
    let ctx = GpuContext::stub_for_tests();
    assert!(matches!(
        ctx.fits_statevector(4).unwrap_err(),
        PrismError::BackendUnsupported { .. }
    ));
}

#[test]
fn fits_statevector_rejects_overflowing_qubit_counts() {
    let ctx = GpuContext::stub_for_tests();
    // usize::BITS - 4 boundary: `1 << 60` times 16 bytes is already 16 EiB
    // which no GPU has. The function clamps these to `Ok(false)` before
    // touching the device, so even the stub context returns cleanly.
    assert!(!ctx.fits_statevector(128).unwrap());
}

#[test]
fn state_new_rejects_overflowing_qubit_counts() {
    let ctx = GpuContext::stub_for_tests();
    // 63 is the silent case: `2 << 63` wraps to a zero-length buffer that
    // allocates fine and is then written past. 62 already fails at
    // allocation. The bound runs before allocation, so the stub reaches it.
    assert!(matches!(
        GpuState::new(ctx, 63).unwrap_err(),
        PrismError::InvalidParameter { .. }
    ));
}

#[test]
fn fits_statevector_with_scratch_rejects_cleanly_on_stub() {
    let ctx = GpuContext::stub_for_tests();
    assert!(matches!(
        ctx.fits_statevector_with_scratch(4).unwrap_err(),
        PrismError::BackendUnsupported { .. }
    ));
}

#[test]
fn fits_statevector_with_scratch_clamps_overflow_before_device() {
    let ctx = GpuContext::stub_for_tests();
    assert!(!ctx.fits_statevector_with_scratch(128).unwrap());
}

#[test]
fn fits_tableau_rejects_cleanly_on_stub() {
    let ctx = GpuContext::stub_for_tests();
    assert!(matches!(
        ctx.fits_tableau(64).unwrap_err(),
        PrismError::BackendUnsupported { .. }
    ));
}

#[test]
fn fits_tableau_clamps_overflow_before_device() {
    let ctx = GpuContext::stub_for_tests();
    assert!(!ctx.fits_tableau(usize::MAX / 2).unwrap());
}

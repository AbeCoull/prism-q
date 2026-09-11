//! Process-global environment overrides for the test binaries that pin a cap
//! or a kernel switch, plus the two shapes a capped rejection takes.

use std::sync::Once;

use prism_q::PrismError;

/// Set every `(variable, value)` pair once per process, before any cap query.
///
/// Caps are cached per process, so a binary that calls this must not also
/// expect the real cap, and every call site in the binary must pass the same
/// pairs: only the first call writes.
pub fn set_once(pairs: &[(&str, &str)]) {
    static SET: Once = Once::new();
    SET.call_once(|| {
        for (variable, value) in pairs {
            // SAFETY: written exactly once behind this `Once` and before the
            // binary's first cap query, so no thread reads a cap mid-write.
            unsafe { std::env::set_var(variable, value) };
        }
    });
}

pub fn assert_cap_rejection(err: PrismError, backend: &str) {
    match err {
        PrismError::IncompatibleBackend {
            backend: named,
            reason,
        } => {
            assert_eq!(named, backend, "wrong backend named: {reason}");
            assert!(
                reason.contains("exceeding the cap"),
                "expected a cap rejection, got {reason}"
            );
        }
        other => panic!("expected a clean cap error, got {other:?}"),
    }
}

/// The `reason` of an `IncompatibleBackend` raised by `backend`, for tests that
/// go on to assert which cap and which width the message names.
pub fn incompatible_reason(err: PrismError, backend: &str) -> String {
    match err {
        PrismError::IncompatibleBackend {
            backend: named,
            reason,
        } => {
            assert_eq!(named, backend);
            reason
        }
        other => panic!("expected IncompatibleBackend, got {other:?}"),
    }
}

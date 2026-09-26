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
        PrismError::ResourceLimit(data) => {
            let prism_q::ResourceLimit {
                backend: named,
                required,
                limit,
                ..
            } = prism_q::ResourceLimit::clone(&data);
            assert_eq!(named, backend, "wrong backend named");
            assert!(required > limit, "{required} is within the cap of {limit}");
        }
        other => panic!("expected a clean cap error, got {other:?}"),
    }
}

/// The message of a `ResourceLimit` raised by `backend`, for tests that go on to
/// assert which cap and which width it names.
pub fn cap_message(err: PrismError, backend: &str) -> String {
    let message = err.to_string();
    match err {
        PrismError::ResourceLimit(data) => {
            let prism_q::ResourceLimit { backend: named, .. } =
                prism_q::ResourceLimit::clone(&data);
            assert_eq!(named, backend);
            message
        }
        other => panic!("expected ResourceLimit, got {other:?}"),
    }
}

/// The `(operation, required, limit)` of a `ResourceLimit` raised by `backend`.
pub fn cap_fields(err: PrismError, backend: &str) -> (String, u128, u128) {
    match err {
        PrismError::ResourceLimit(data) => {
            let prism_q::ResourceLimit {
                backend: named,
                operation,
                required,
                limit,
                ..
            } = prism_q::ResourceLimit::clone(&data);
            assert_eq!(named, backend);
            (operation, required, limit)
        }
        other => panic!("expected ResourceLimit, got {other:?}"),
    }
}

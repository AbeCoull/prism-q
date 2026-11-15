//! Device acquisition shared by the GPU test binaries. A missing or unusable device
//! fails the test; `PRISM_ALLOW_NO_GPU=1` lets a host without one skip instead.

use std::sync::Arc;

use prism_q::gpu::GpuContext;

/// Device 0, or `None` when it does not open and `PRISM_ALLOW_NO_GPU` is set.
pub fn context() -> Option<Arc<GpuContext>> {
    match GpuContext::new(0) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            assert!(
                std::env::var_os("PRISM_ALLOW_NO_GPU").is_some(),
                "no usable GPU ({e}). Set PRISM_ALLOW_NO_GPU=1 to let the GPU suites \
                 skip on a host without one."
            );
            eprintln!("SKIP: no usable GPU ({e}); PRISM_ALLOW_NO_GPU is set");
            None
        }
    }
}

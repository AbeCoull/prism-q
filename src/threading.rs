//! Caller-supplied Rayon pool, for embedding PRISM-Q in an application that
//! owns the process-wide pool.

use crate::error::{PrismError, Result};

/// A bounded Rayon pool that PRISM-Q runs inside, leaving the process-wide pool
/// untouched.
///
/// Simulation entry points size the global Rayon pool on first use, which an
/// embedding application may not want. Work run through [`ThreadPool::install`]
/// uses this pool instead, and the global pool is neither built nor resized, so
/// an application that installed its own keeps it.
///
/// Pool width is part of what a result depends on. Dense unitary evolution,
/// seeded terminal sampling, and the stabilizer tableau are bitwise at any
/// width, parallel reductions move by about 1e-12, and compiled (BTS) shot
/// payloads differ between widths because the batched sampler splits shots by
/// pool width. A narrower pool is therefore not a slower route to the same
/// bytes for every result; the per-path contract is the determinism section of
/// the threading architecture page.
///
/// # Examples
///
/// ```
/// use prism_q::{ThreadPool, run_qasm};
///
/// let qasm = r#"
///     OPENQASM 3.0;
///     include "stdgates.inc";
///     qubit[2] q;
///     h q[0];
///     cx q[0], q[1];
/// "#;
///
/// let pool = ThreadPool::with_threads(2).expect("build pool");
/// let result = pool.install(|| run_qasm(qasm, 42)).expect("simulation");
/// let probs = result.probabilities.expect("no probabilities").to_vec();
/// assert!((probs[0] - 0.5).abs() < 1e-10);
/// ```
pub struct ThreadPool {
    inner: rayon::ThreadPool,
}

impl ThreadPool {
    /// Build a pool of `threads` workers, or of the Rayon default width when
    /// `threads` is 0.
    ///
    /// # Errors
    ///
    /// Returns [`PrismError::InvalidParameter`] when the operating system
    /// refuses to spawn the workers.
    pub fn with_threads(threads: usize) -> Result<Self> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map(|inner| Self { inner })
            .map_err(|e| PrismError::InvalidParameter {
                message: format!("could not build a Rayon pool of {threads} threads: {e}"),
            })
    }

    /// Run `op` on this pool, blocking the calling thread until it returns.
    ///
    /// Every Rayon kernel PRISM-Q reaches from `op` runs on these workers. Work
    /// that escapes `op`, such as a simulation handed to another thread, falls
    /// back to the global pool.
    pub fn install<T: Send>(&self, op: impl FnOnce() -> T + Send) -> T {
        self.inner.install(op)
    }

    /// Worker count, which is what `rayon::current_num_threads` reports inside
    /// [`install`](Self::install). Resolves the default width taken by
    /// `with_threads(0)`.
    pub fn num_threads(&self) -> usize {
        self.inner.current_num_threads()
    }
}

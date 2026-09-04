//! Distributed context, transport, and thresholds.
//!
//! [`DistributedContext`] wraps a shared [`RankComm`] transport. Backend modules
//! define their own state partitioning and use this module for rank access.
//!
//! Available contexts:
//! - [`DistributedContext::serial`]: one rank for tests and runs without MPI.
//! - `DistributedContext::world`: the MPI world communicator (requires the
//!   `distributed-mpi` feature and an MPI launcher).
//!
//! Tuning thresholds are cached after the first environment variable read.

pub mod comm;
#[cfg(any(test, feature = "bench-internal"))]
pub mod loopback;

use std::sync::Arc;

pub use comm::{RankComm, SerialComm};

#[cfg(feature = "distributed-mpi")]
pub use comm::MpiComm;

/// Default minimum local qubit count below which distribution is not worthwhile.
///
/// Small slices per rank spend more time in communication than computation.
pub const MIN_LOCAL_QUBITS_DEFAULT: usize = 10;

/// Minimum local qubits per rank, tunable via `PRISM_DIST_MIN_LOCAL_QUBITS`.
///
/// An unparseable or out-of-range value warns on stderr and uses the default.
pub fn min_local_qubits() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_usize_knob(
            "PRISM_DIST_MIN_LOCAL_QUBITS",
            std::env::var("PRISM_DIST_MIN_LOCAL_QUBITS").ok(),
            MIN_LOCAL_QUBITS_DEFAULT,
            1,
        )
    })
}

/// Maximum number of amplitudes exchanged per message for a global one qubit
/// gate. Chunking bounds the receive buffer to this value.
///
/// Tunable via `PRISM_DIST_EXCHANGE_CHUNK`. The default (`usize::MAX`) keeps the
/// original one message behavior, so there is no change unless set.
pub const EXCHANGE_CHUNK_DEFAULT: usize = usize::MAX;

/// Chunk size in amplitudes for tiled global one qubit exchange.
///
/// An unparseable or out-of-range value warns on stderr and uses the default.
pub fn exchange_chunk() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_usize_knob(
            "PRISM_DIST_EXCHANGE_CHUNK",
            std::env::var("PRISM_DIST_EXCHANGE_CHUNK").ok(),
            EXCHANGE_CHUNK_DEFAULT,
            1,
        )
    })
}

/// Whether the distributed backend relabels qubits to keep busy qubits local.
///
/// On by default. Relabeling turns SWAP gates into zero-communication map
/// updates and moves global qubits into local positions with a half-slice
/// exchange before non-diagonal gates touch them. Set `PRISM_DIST_RELABEL=0`
/// to disable and force direct per-gate exchange.
///
/// Accepts `0`/`false` and `1`/`true`; anything else warns on stderr and uses
/// the default.
pub fn relabel_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_bool_knob(
            "PRISM_DIST_RELABEL",
            std::env::var("PRISM_DIST_RELABEL").ok(),
            true,
        )
    })
}

/// Read a count knob, warning on stderr and falling back to `default` when the
/// value does not parse or falls below `min`.
///
/// Warning rather than erroring is the knob contract: every reader is a cached
/// initializer on an infallible path, so an invalid value must not take down a
/// run that would otherwise be correct with the default.
fn parse_usize_knob(var: &str, raw: Option<String>, default: usize, min: usize) -> usize {
    let Some(raw) = raw else {
        return default;
    };
    match raw.trim().parse::<usize>() {
        Ok(n) if n >= min => n,
        Ok(n) => {
            eprintln!("warning: {var}={n} is below the minimum of {min}; using {default}.");
            default
        }
        Err(_) => {
            eprintln!("warning: {var}={raw:?} is not a count; using {default}.");
            default
        }
    }
}

/// Read a flag knob. See [`parse_usize_knob`] for why an invalid value warns.
fn parse_bool_knob(var: &str, raw: Option<String>, default: bool) -> bool {
    let Some(raw) = raw else {
        return default;
    };
    match raw.trim() {
        "0" => false,
        "1" => true,
        other if other.eq_ignore_ascii_case("false") => false,
        other if other.eq_ignore_ascii_case("true") => true,
        other => {
            eprintln!("warning: {var}={other:?} is not a flag; using {default}.");
            default
        }
    }
}

/// Shared handle to a rank transport for distributed simulation.
pub struct DistributedContext {
    comm: Arc<dyn RankComm>,
}

impl std::fmt::Debug for DistributedContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedContext")
            .field("rank", &self.rank())
            .field("size", &self.size())
            .finish()
    }
}

impl DistributedContext {
    /// Build a context from any [`RankComm`] implementation.
    pub fn from_comm(comm: Arc<dyn RankComm>) -> Arc<Self> {
        Arc::new(Self { comm })
    }

    /// Single rank. Used by tests and runs without MPI.
    pub fn serial() -> Arc<Self> {
        Self::from_comm(Arc::new(SerialComm))
    }

    /// Initialize MPI and capture the world communicator.
    ///
    /// Returns `None` when MPI cannot be initialized.
    #[cfg(feature = "distributed-mpi")]
    pub fn world() -> Option<Arc<Self>> {
        MpiComm::world().map(|c| Self::from_comm(Arc::new(c)))
    }

    /// Attach to an MPI another component has already initialized, taking no
    /// ownership of its lifetime. See [`MpiComm::attach_world`] for why an
    /// embedding interpreter needs this rather than [`DistributedContext::world`].
    ///
    /// Returns `None` when MPI is not initialized.
    #[cfg(feature = "distributed-mpi")]
    pub fn attached_world() -> Option<Arc<Self>> {
        MpiComm::attach_world().map(|c| Self::from_comm(Arc::new(c)))
    }

    /// Index of the calling rank.
    pub fn rank(&self) -> usize {
        self.comm.rank()
    }

    /// Total number of ranks.
    pub fn size(&self) -> usize {
        self.comm.size()
    }

    pub(crate) fn comm(&self) -> &Arc<dyn RankComm> {
        &self.comm
    }
}

#[cfg(test)]
mod knob_tests {
    use super::*;

    // One case per parse site. The knob functions cache per process, so the
    // parsers are exercised directly rather than through the environment.
    #[test]
    fn min_local_qubits_knob_rejects_invalid_values() {
        let d = MIN_LOCAL_QUBITS_DEFAULT;
        let parse =
            |raw: &str| parse_usize_knob("PRISM_DIST_MIN_LOCAL_QUBITS", Some(raw.into()), d, 1);
        assert_eq!(parse("4"), 4);
        assert_eq!(parse(" 4 "), 4);
        assert_eq!(parse("abc"), d, "unparseable falls back");
        assert_eq!(parse("-1"), d, "negative falls back");
        assert_eq!(parse("0"), d, "below the minimum falls back");
        assert_eq!(
            parse_usize_knob("PRISM_DIST_MIN_LOCAL_QUBITS", None, d, 1),
            d
        );
    }

    #[test]
    fn exchange_chunk_knob_rejects_invalid_values() {
        let d = EXCHANGE_CHUNK_DEFAULT;
        let parse =
            |raw: &str| parse_usize_knob("PRISM_DIST_EXCHANGE_CHUNK", Some(raw.into()), d, 1);
        assert_eq!(parse("4096"), 4096);
        assert_eq!(parse("4k"), d, "unparseable falls back");
        assert_eq!(parse("0"), d, "below the minimum falls back");
        assert_eq!(parse_usize_knob("PRISM_DIST_EXCHANGE_CHUNK", None, d, 1), d);
    }

    #[test]
    fn relabel_knob_rejects_invalid_values() {
        let parse = |raw: &str| parse_bool_knob("PRISM_DIST_RELABEL", Some(raw.into()), true);
        assert!(!parse("0"));
        assert!(!parse("false"));
        assert!(!parse("FALSE"));
        assert!(parse("1"));
        assert!(parse("true"));
        assert!(parse("yes"), "unrecognized falls back to the default");
        assert!(parse(""), "empty falls back to the default");
        assert!(parse_bool_knob("PRISM_DIST_RELABEL", None, true));
    }
}

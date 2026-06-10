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

use std::sync::Arc;

pub use comm::{RankComm, SerialComm};

#[cfg(feature = "distributed-mpi")]
pub use comm::MpiComm;

/// Default minimum local qubit count below which distribution is not worthwhile.
///
/// Small slices per rank spend more time in communication than computation.
pub const MIN_LOCAL_QUBITS_DEFAULT: usize = 10;

/// Minimum local qubits per rank, tunable via `PRISM_DIST_MIN_LOCAL_QUBITS`.
pub fn min_local_qubits() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| env_usize_or("PRISM_DIST_MIN_LOCAL_QUBITS", MIN_LOCAL_QUBITS_DEFAULT, 1))
}

/// Maximum number of amplitudes exchanged per message for a global one qubit
/// gate. Chunking bounds the receive buffer to this value.
///
/// Tunable via `PRISM_DIST_EXCHANGE_CHUNK`. The default (`usize::MAX`) keeps the
/// original one message behavior, so there is no change unless set.
pub const EXCHANGE_CHUNK_DEFAULT: usize = usize::MAX;

/// Chunk size in amplitudes for tiled global one qubit exchange.
pub fn exchange_chunk() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| env_usize_or("PRISM_DIST_EXCHANGE_CHUNK", EXCHANGE_CHUNK_DEFAULT, 1))
}

/// Whether the distributed backend relabels qubits to keep busy qubits local.
///
/// On by default. Relabeling turns SWAP gates into zero-communication map
/// updates and moves global qubits into local positions with a half-slice
/// exchange before non-diagonal gates touch them. Set `PRISM_DIST_RELABEL=0`
/// to disable and force direct per-gate exchange.
pub fn relabel_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("PRISM_DIST_RELABEL")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    })
}

#[inline]
fn env_usize_or(var: &str, default: usize, min: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|n| n.max(min))
        .unwrap_or(default)
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

//! Rank transport for distributed simulation: the [`RankComm`] trait, the single rank
//! [`SerialComm`], and `MpiComm` over `rsmpi` behind the `distributed-mpi` feature.

use num_complex::Complex64;

#[cfg(feature = "distributed-mpi")]
use crate::error::{PrismError, Result};
#[cfg(feature = "distributed-mpi")]
use mpi::environment::Threading;
#[cfg(any(test, feature = "distributed-mpi"))]
use std::thread::ThreadId;

/// Collective and peer operations across a rank set.
///
/// The amplitude exchange routines treat `Complex64` as two contiguous `f64`
/// values, matching the `#[repr(C)]` layout of `num_complex::Complex`.
pub trait RankComm: std::fmt::Debug + Send + Sync {
    /// Index of the calling rank, in `0..size()`.
    fn rank(&self) -> usize;

    /// Total number of ranks. Always a power of two for the distributed backend.
    fn size(&self) -> usize;

    /// Concatenate every rank's `local` block in ascending rank order.
    ///
    /// The returned vector has length `size() * local.len()` and is identical
    /// on every rank.
    fn allgather_c64(&self, local: &[Complex64]) -> Vec<Complex64>;

    /// `f64` version of [`allgather_c64`](RankComm::allgather_c64).
    fn allgather_f64(&self, local: &[f64]) -> Vec<f64>;

    /// Concatenate blocks of differing length in ascending rank order.
    ///
    /// `counts[r]` is the length of rank `r`'s block, so `counts[rank()]`
    /// equals `local.len()` and every rank passes the same `counts`. The
    /// returned vector has length `counts.iter().sum()` and is identical on
    /// every rank.
    fn allgatherv_u64(&self, local: &[u64], counts: &[usize]) -> Vec<u64>;

    /// `u64` version of [`allgather_c64`](RankComm::allgather_c64).
    fn allgather_u64(&self, local: &[u64]) -> Vec<u64> {
        self.allgatherv_u64(local, &vec![local.len(); self.size()])
    }

    /// Sum a scalar across all ranks; every rank receives the total.
    fn allreduce_sum_f64(&self, value: f64) -> f64;

    /// Exchange equal length amplitude blocks with `partner`.
    ///
    /// On return, `recv` holds `partner`'s `send` block. `send` and `recv` must
    /// have the same length.
    fn sendrecv_c64(&self, partner: usize, send: &[Complex64], recv: &mut [Complex64]);

    /// Block until all ranks reach this point.
    fn barrier(&self);
}

/// Single rank. All collectives are identity operations.
#[derive(Debug, Default, Clone, Copy)]
pub struct SerialComm;

impl RankComm for SerialComm {
    #[inline]
    fn rank(&self) -> usize {
        0
    }

    #[inline]
    fn size(&self) -> usize {
        1
    }

    #[inline]
    fn allgather_c64(&self, local: &[Complex64]) -> Vec<Complex64> {
        local.to_vec()
    }

    #[inline]
    fn allgather_f64(&self, local: &[f64]) -> Vec<f64> {
        local.to_vec()
    }

    #[inline]
    fn allgatherv_u64(&self, local: &[u64], _counts: &[usize]) -> Vec<u64> {
        local.to_vec()
    }

    #[inline]
    fn allreduce_sum_f64(&self, value: f64) -> f64 {
        value
    }

    #[inline]
    fn sendrecv_c64(&self, _partner: usize, send: &[Complex64], recv: &mut [Complex64]) {
        debug_assert_eq!(send.len(), recv.len());
        recv.copy_from_slice(send);
    }

    #[inline]
    fn barrier(&self) {}
}

/// `Complex64` is two adjacent `f64` values. Assert the layout used by MPI.
#[cfg(feature = "distributed-mpi")]
const _: () = assert!(std::mem::size_of::<Complex64>() == 2 * std::mem::size_of::<f64>());

#[cfg(feature = "distributed-mpi")]
#[inline]
fn as_f64(slice: &[Complex64]) -> &[f64] {
    // SAFETY: Complex64 is repr(C) over two f64 values with no padding.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f64, slice.len() * 2) }
}

#[cfg(feature = "distributed-mpi")]
#[inline]
fn as_f64_mut(slice: &mut [Complex64]) -> &mut [f64] {
    // SAFETY: same layout as `as_f64`; the mutable borrow is exclusive.
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f64, slice.len() * 2) }
}

/// Thread level requested from `MPI_Init_thread`. Rayon workers touch only
/// memory; every MPI call is made from the thread that constructed the comm.
#[cfg(feature = "distributed-mpi")]
const REQUIRED_THREADING: Threading = Threading::Funneled;

/// Reject an MPI whose provided thread level is below [`REQUIRED_THREADING`].
#[cfg(feature = "distributed-mpi")]
fn check_threading(provided: Threading) -> Result<()> {
    if provided >= REQUIRED_THREADING {
        return Ok(());
    }
    Err(PrismError::IncompatibleBackend {
        backend: "distributed".into(),
        reason: format!(
            "MPI thread level {provided:?} is below {REQUIRED_THREADING:?}, the minimum for \
             running Rayon workers beside MPI"
        ),
    })
}

/// The thread every MPI call must come from at `provided`: the calling one below
/// `MPI_THREAD_MULTIPLE`, any thread (`None`) at it.
#[cfg(feature = "distributed-mpi")]
fn owner_for(provided: Threading) -> Option<ThreadId> {
    (provided < Threading::Multiple).then(|| std::thread::current().id())
}

#[cfg(any(test, feature = "distributed-mpi"))]
#[inline]
fn assert_owner_thread(owner: Option<ThreadId>) {
    if let Some(owner) = owner {
        let here = std::thread::current().id();
        assert!(
            here == owner,
            "MPI call from thread {here:?}, but the MPI thread level confines calls to \
             thread {owner:?}, which constructed the comm"
        );
    }
}

/// MPI transport over `rsmpi`.
///
/// Requires the `distributed-mpi` feature, a system MPI install, and an MPI
/// launcher. MPI must run at `MPI_THREAD_FUNNELED` or higher: [`MpiComm::world`]
/// requests that level and both constructors return an error when the provided
/// level is lower. Drop a comm from [`MpiComm::world`] on the thread that
/// constructed it, since the drop runs `MPI_Finalize`.
///
/// # Panics
///
/// Below `MPI_THREAD_MULTIPLE`, every collective and exchange panics before
/// reaching MPI when called from a thread other than the one that constructed
/// the comm.
#[cfg(feature = "distributed-mpi")]
pub struct MpiComm {
    /// Held only by a comm that ran `MPI_Init` itself, because dropping a
    /// `Universe` runs `MPI_Finalize`. A comm attached to an MPI another
    /// component owns leaves that lifetime alone.
    _universe: Option<mpi::environment::Universe>,
    owner: Option<ThreadId>,
    rank: usize,
    size: usize,
}

#[cfg(feature = "distributed-mpi")]
impl std::fmt::Debug for MpiComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MpiComm")
            .field("rank", &self.rank)
            .field("size", &self.size)
            .finish()
    }
}

#[cfg(feature = "distributed-mpi")]
impl MpiComm {
    /// Initialize MPI at `MPI_THREAD_FUNNELED` and capture the world communicator.
    ///
    /// # Errors
    ///
    /// Fails when MPI is already initialized (`mpi::initialize_with_threading`
    /// declines rather than attaching; use [`MpiComm::attach_world`] in a
    /// process where something else owns MPI) and when the provided thread
    /// level is below the requested one.
    pub fn world() -> Result<Self> {
        let (universe, provided) = mpi::environment::initialize_with_threading(REQUIRED_THREADING)
            .ok_or_else(|| PrismError::IncompatibleBackend {
                backend: "distributed".into(),
                reason: "MPI is already initialized; attach to it with `MpiComm::attach_world`"
                    .into(),
            })?;
        check_threading(provided)?;
        Ok(Self::from_parts(Some(universe), provided))
    }

    /// Attach to an MPI another component has already initialized.
    ///
    /// The returned comm holds no `Universe`, so dropping it does not call
    /// `MPI_Finalize`. In an interpreter where mpi4py calls `MPI_Init_thread` at
    /// import and registers `MPI_Finalize` at exit, finalizing from a dropped
    /// handle would make every later MPI call in the process erroneous.
    ///
    /// Returns `Ok(None)` when MPI is not initialized, and an error when the
    /// owner initialized it below `MPI_THREAD_FUNNELED`, or at that level on
    /// another thread.
    pub fn attach_world() -> Result<Option<Self>> {
        if !mpi::environment::is_initialized() {
            return Ok(None);
        }
        let provided = mpi::environment::threading_support();
        check_threading(provided)?;
        if provided < Threading::Serialized && !is_thread_main() {
            return Err(PrismError::IncompatibleBackend {
                backend: "distributed".into(),
                reason: "MPI runs at MPI_THREAD_FUNNELED, which confines MPI calls to the \
                         thread that initialized it; attach from that thread"
                    .into(),
            });
        }
        Ok(Some(Self::from_parts(None, provided)))
    }

    fn from_parts(universe: Option<mpi::environment::Universe>, provided: Threading) -> Self {
        use mpi::traits::Communicator;
        let world = mpi::topology::SimpleCommunicator::world();
        Self {
            _universe: universe,
            owner: owner_for(provided),
            rank: world.rank() as usize,
            size: world.size() as usize,
        }
    }

    /// The world communicator, fetched per call rather than stored: Open MPI
    /// types `MPI_Comm` as a pointer, so a stored one would make the comm
    /// neither `Send` nor `Sync`.
    fn world_comm(&self) -> mpi::topology::SimpleCommunicator {
        assert_owner_thread(self.owner);
        mpi::topology::SimpleCommunicator::world()
    }
}

#[cfg(feature = "distributed-mpi")]
fn is_thread_main() -> bool {
    let mut flag: std::os::raw::c_int = 0;
    // SAFETY: MPI is initialized, and the standard allows `MPI_Is_thread_main`
    // from any thread at any level. rsmpi exposes no safe wrapper for it.
    unsafe { mpi::ffi::MPI_Is_thread_main(&mut flag) };
    flag != 0
}

#[cfg(feature = "distributed-mpi")]
impl RankComm for MpiComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn allgather_c64(&self, local: &[Complex64]) -> Vec<Complex64> {
        use mpi::traits::CommunicatorCollectives;
        let mut out = vec![Complex64::new(0.0, 0.0); local.len() * self.size];
        self.world_comm()
            .all_gather_into(as_f64(local), as_f64_mut(&mut out));
        out
    }

    fn allgather_f64(&self, local: &[f64]) -> Vec<f64> {
        use mpi::traits::CommunicatorCollectives;
        let mut out = vec![0.0_f64; local.len() * self.size];
        self.world_comm().all_gather_into(local, &mut out);
        out
    }

    fn allgatherv_u64(&self, local: &[u64], counts: &[usize]) -> Vec<u64> {
        use mpi::datatype::PartitionMut;
        use mpi::traits::CommunicatorCollectives;
        debug_assert_eq!(counts[self.rank], local.len());
        let counts: Vec<mpi::Count> = counts.iter().map(|&c| c as mpi::Count).collect();
        let mut displs = Vec::with_capacity(counts.len());
        let mut total: mpi::Count = 0;
        for &c in &counts {
            displs.push(total);
            total += c;
        }
        let mut out = vec![0_u64; total as usize];
        self.world_comm()
            .all_gather_varcount_into(local, &mut PartitionMut::new(&mut out[..], counts, displs));
        out
    }

    fn allreduce_sum_f64(&self, value: f64) -> f64 {
        use mpi::traits::CommunicatorCollectives;
        let mut out = 0.0_f64;
        self.world_comm().all_reduce_into(
            &value,
            &mut out,
            mpi::collective::SystemOperation::sum(),
        );
        out
    }

    fn sendrecv_c64(&self, partner: usize, send: &[Complex64], recv: &mut [Complex64]) {
        use mpi::point_to_point as p2p;
        use mpi::traits::Communicator;
        debug_assert_eq!(send.len(), recv.len());
        let world = self.world_comm();
        let peer = world.process_at_rank(partner as i32);
        p2p::send_receive_into(as_f64(send), &peer, as_f64_mut(recv), &peer);
    }

    fn barrier(&self) {
        use mpi::traits::CommunicatorCollectives;
        self.world_comm().barrier();
    }
}

#[cfg(test)]
mod owner_thread_tests {
    use super::assert_owner_thread;

    #[test]
    fn owner_thread_check_admits_the_owner_and_rejects_other_threads() {
        let owner = Some(std::thread::current().id());
        assert_owner_thread(owner);
        assert_owner_thread(None);
        std::thread::spawn(|| assert_owner_thread(None))
            .join()
            .expect("an unconfined comm admits every thread");
        let panic = std::thread::spawn(move || assert_owner_thread(owner))
            .join()
            .expect_err("a confined comm rejects other threads");
        let message = panic
            .downcast_ref::<String>()
            .expect("formatted panic message");
        assert!(message.contains("constructed the comm"), "{message}");
    }
}

#[cfg(all(test, feature = "distributed-mpi"))]
mod threading_tests {
    use super::*;

    // The check leans on the crate's `Ord`, which compares the raw MPI constants,
    // so the ordering the MPI standard guarantees is pinned alongside it.
    #[test]
    fn thread_level_check_rejects_only_single() {
        assert!(Threading::Single < Threading::Funneled);
        assert!(Threading::Funneled < Threading::Serialized);
        assert!(Threading::Serialized < Threading::Multiple);
        assert!(check_threading(Threading::Single).is_err());
        for level in [
            Threading::Funneled,
            Threading::Serialized,
            Threading::Multiple,
        ] {
            assert!(check_threading(level).is_ok(), "{level:?}");
        }
    }

    #[test]
    fn only_multiple_leaves_calls_unconfined() {
        let here = Some(std::thread::current().id());
        assert_eq!(owner_for(Threading::Funneled), here);
        assert_eq!(owner_for(Threading::Serialized), here);
        assert_eq!(owner_for(Threading::Multiple), None);
    }
}

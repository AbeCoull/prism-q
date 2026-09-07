//! Rank transport for distributed simulation.
//!
//! [`RankComm`] abstracts the collective and peer operations used by distributed
//! backends. It is independent of state representation.
//!
//! [`SerialComm`] is the single rank implementation used by tests. `MpiComm`
//! uses `rsmpi` behind the `distributed-mpi` feature and requires `mpiexec`.

use num_complex::Complex64;

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

/// Reinterpret `Complex64` as flat `f64` values for MPI calls.
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

/// MPI transport over `rsmpi`.
///
/// Requires the `distributed-mpi` feature, a system MPI install, and an MPI
/// launcher.
#[cfg(feature = "distributed-mpi")]
pub struct MpiComm {
    /// Held only by a comm that ran `MPI_Init` itself, because dropping a
    /// `Universe` runs `MPI_Finalize`. A comm attached to an MPI another
    /// component owns leaves that lifetime alone.
    _universe: Option<mpi::environment::Universe>,
    world: mpi::topology::SimpleCommunicator,
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
    /// Initialize MPI and capture the world communicator.
    ///
    /// Returns `None` when MPI initialization fails, which includes MPI already
    /// being initialized: `mpi::initialize` declines rather than attaching. Use
    /// [`MpiComm::attach_world`] in a process where something else owns MPI.
    pub fn world() -> Option<Self> {
        let universe = mpi::initialize()?;
        let world = universe.world();
        Some(Self::from_parts(Some(universe), world))
    }

    /// Attach to an MPI another component has already initialized.
    ///
    /// The returned comm holds no `Universe`, so dropping it does not call
    /// `MPI_Finalize`. That is the point of the constructor: in an interpreter
    /// where mpi4py calls `MPI_Init_thread` at import and registers
    /// `MPI_Finalize` at exit, finalizing from a dropped handle would make
    /// every later MPI call in the process erroneous.
    ///
    /// Returns `None` when MPI is not initialized.
    pub fn attach_world() -> Option<Self> {
        if !mpi::environment::is_initialized() {
            return None;
        }
        Some(Self::from_parts(
            None,
            mpi::topology::SimpleCommunicator::world(),
        ))
    }

    fn from_parts(
        universe: Option<mpi::environment::Universe>,
        world: mpi::topology::SimpleCommunicator,
    ) -> Self {
        use mpi::traits::Communicator;
        let rank = world.rank() as usize;
        let size = world.size() as usize;
        Self {
            _universe: universe,
            world,
            rank,
            size,
        }
    }
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
        self.world
            .all_gather_into(as_f64(local), as_f64_mut(&mut out));
        out
    }

    fn allgather_f64(&self, local: &[f64]) -> Vec<f64> {
        use mpi::traits::CommunicatorCollectives;
        let mut out = vec![0.0_f64; local.len() * self.size];
        self.world.all_gather_into(local, &mut out);
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
        self.world
            .all_gather_varcount_into(local, &mut PartitionMut::new(&mut out[..], counts, displs));
        out
    }

    fn allreduce_sum_f64(&self, value: f64) -> f64 {
        use mpi::traits::CommunicatorCollectives;
        let mut out = 0.0_f64;
        self.world
            .all_reduce_into(&value, &mut out, mpi::collective::SystemOperation::sum());
        out
    }

    fn sendrecv_c64(&self, partner: usize, send: &[Complex64], recv: &mut [Complex64]) {
        use mpi::point_to_point as p2p;
        use mpi::traits::Communicator;
        debug_assert_eq!(send.len(), recv.len());
        let peer = self.world.process_at_rank(partner as i32);
        p2p::send_receive_into(as_f64(send), &peer, as_f64_mut(recv), &peer);
    }

    fn barrier(&self) {
        use mpi::traits::CommunicatorCollectives;
        self.world.barrier();
    }
}

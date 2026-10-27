//! Thread-backed [`RankComm`] for driving multi-rank code without an MPI
//! launcher. Available to tests and, behind `bench-internal`, to benchmarks.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use num_complex::Complex64;

use super::comm::{MPI_MAX_COUNT, allgather_pieces, allgatherv_pieces, sendrecv_pieces};
use super::{DistributedContext, RankComm};

/// Debug SIMD kernels overflow the default thread stack at the widths the
/// loopback suite runs.
const RANK_STACK_BYTES: usize = 64 * 1024 * 1024;

struct LoopbackShared {
    size: usize,
    state: Mutex<LoopbackState>,
    cv: Condvar,
    /// Point to point mailboxes indexed by `sender * size + receiver`, each
    /// under its own lock so exchanges between disjoint pairs never contend
    /// with each other or with a collective in flight.
    mailboxes: Vec<Mailbox>,
    /// Largest element count one transport call may carry, `c64` counting two
    /// per amplitude as under MPI. A call past it panics, as rsmpi does at the
    /// C `int` limit.
    count_limit: usize,
    /// Largest amplitude count one rank passed to a single `sendrecv_c64`.
    max_exchange: AtomicUsize,
}

/// Collective state: every rank's contribution plus the barrier counter.
struct LoopbackState {
    generation: u64,
    arrived: usize,
    cslots: Vec<Vec<Complex64>>,
    fslots: Vec<Vec<f64>>,
    uslots: Vec<Vec<u64>>,
    scalars: Vec<f64>,
    /// Largest block one rank passed to an `f64` or `c64` allgather. Shot
    /// sampling tests assert this stays at one element, proving no dense
    /// gather happened. The `u64` gather carries fingerprints and sampled
    /// indices, whose size is set by the shot count, and is not counted.
    max_gather_block: usize,
}

struct Mailbox {
    queue: Mutex<MailboxQueue>,
    cv: Condvar,
}

/// Pending messages in FIFO order, matching MPI sendrecv order, plus the
/// buffers of delivered messages. Steady-state traffic reuses those instead
/// of allocating per message, so the transport stays at memcpy cost.
struct MailboxQueue {
    pending: VecDeque<Vec<Complex64>>,
    spare: Vec<Vec<Complex64>>,
}

impl Mailbox {
    fn new() -> Self {
        Self {
            queue: Mutex::new(MailboxQueue {
                pending: VecDeque::new(),
                spare: Vec::new(),
            }),
            cv: Condvar::new(),
        }
    }

    fn send(&self, msg: &[Complex64]) {
        let mut q = self.queue.lock().unwrap();
        let mut buf = q.spare.pop().unwrap_or_default();
        buf.clear();
        buf.extend_from_slice(msg);
        q.pending.push_back(buf);
        self.cv.notify_one();
    }

    fn recv(&self, out: &mut [Complex64]) {
        let mut q = self.queue.lock().unwrap();
        loop {
            if let Some(buf) = q.pending.pop_front() {
                out.copy_from_slice(&buf);
                q.spare.push(buf);
                return;
            }
            q = self.cv.wait(q).unwrap();
        }
    }
}

impl LoopbackShared {
    fn new(size: usize, count_limit: usize) -> Arc<Self> {
        Arc::new(Self {
            size,
            state: Mutex::new(LoopbackState {
                generation: 0,
                arrived: 0,
                cslots: vec![Vec::new(); size],
                fslots: vec![Vec::new(); size],
                uslots: vec![Vec::new(); size],
                scalars: vec![0.0; size],
                max_gather_block: 0,
            }),
            cv: Condvar::new(),
            mailboxes: (0..size * size).map(|_| Mailbox::new()).collect(),
            count_limit,
            max_exchange: AtomicUsize::new(0),
        })
    }

    fn barrier(&self) {
        let mut st = self.state.lock().unwrap();
        let arrival_generation = st.generation;
        st.arrived += 1;
        if st.arrived == self.size {
            st.arrived = 0;
            st.generation = st.generation.wrapping_add(1);
            self.cv.notify_all();
        } else {
            while st.generation == arrival_generation {
                st = self.cv.wait(st).unwrap();
            }
        }
    }
}

#[derive(Clone)]
struct LoopbackComm {
    shared: Arc<LoopbackShared>,
    rank: usize,
}

impl std::fmt::Debug for LoopbackComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopbackComm")
            .field("rank", &self.rank)
            .field("size", &self.shared.size)
            .finish()
    }
}

impl LoopbackComm {
    fn check_count(&self, count: usize) {
        let limit = self.shared.count_limit;
        assert!(
            count <= limit,
            "transport call of {count} elements exceeds the count limit {limit}"
        );
    }

    /// One gather call: post `block` to this rank's slot and copy every slot
    /// into `out` in rank order.
    fn gather_into<T: Copy>(
        &self,
        block: &[T],
        out: &mut [T],
        slots: fn(&mut LoopbackState) -> &mut Vec<Vec<T>>,
    ) {
        {
            let mut st = self.shared.state.lock().unwrap();
            let slot = &mut slots(&mut st)[self.rank];
            slot.clear();
            slot.extend_from_slice(block);
        }
        self.shared.barrier();
        {
            let mut st = self.shared.state.lock().unwrap();
            let mut at = 0;
            for slot in slots(&mut st).iter() {
                out[at..at + slot.len()].copy_from_slice(slot);
                at += slot.len();
            }
        }
        self.shared.barrier();
    }

    fn record_gather_block(&self, len: usize) {
        let mut st = self.shared.state.lock().unwrap();
        st.max_gather_block = st.max_gather_block.max(len);
    }
}

impl RankComm for LoopbackComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.shared.size
    }

    fn allgather_c64(&self, local: &[Complex64]) -> Vec<Complex64> {
        self.record_gather_block(local.len());
        let limit = self.shared.count_limit / 2;
        allgather_pieces(local, self.shared.size, limit, |block, out| {
            self.check_count(2 * out.len());
            self.gather_into(block, out, |st| &mut st.cslots);
        })
    }

    fn allgather_f64(&self, local: &[f64]) -> Vec<f64> {
        self.record_gather_block(local.len());
        let limit = self.shared.count_limit;
        allgather_pieces(local, self.shared.size, limit, |block, out| {
            self.check_count(out.len());
            self.gather_into(block, out, |st| &mut st.fslots);
        })
    }

    fn allgatherv_u64(&self, local: &[u64], counts: &[usize]) -> Vec<u64> {
        debug_assert_eq!(counts[self.rank], local.len());
        let limit = self.shared.count_limit;
        allgatherv_pieces(local, counts, self.rank, limit, |block, _, out| {
            self.check_count(out.len());
            self.gather_into(block, out, |st| &mut st.uslots);
        })
    }

    fn allreduce_sum_f64(&self, value: f64) -> f64 {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.scalars[self.rank] = value;
        }
        self.shared.barrier();
        let sum = self.shared.state.lock().unwrap().scalars.iter().sum();
        self.shared.barrier();
        sum
    }

    fn sendrecv_c64(&self, partner: usize, send: &[Complex64], recv: &mut [Complex64]) {
        debug_assert_eq!(send.len(), recv.len());
        self.shared
            .max_exchange
            .fetch_max(send.len(), Ordering::Relaxed);
        let size = self.shared.size;
        sendrecv_pieces(send, recv, self.shared.count_limit / 2, |s, r| {
            self.check_count(2 * s.len());
            // Ranks that skip an exchange do not block because their partner skips it too.
            self.shared.mailboxes[self.rank * size + partner].send(s);
            self.shared.mailboxes[partner * size + self.rank].recv(r);
        });
    }

    fn barrier(&self) {
        self.shared.barrier();
    }
}

/// Run `f` once per simulated rank on its own thread and collect the results in
/// rank order.
///
/// `f` receives a context whose transport reaches every other rank, so it may
/// issue the same collectives real ranks do. Every rank must call the same
/// sequence, as under MPI.
pub fn run_ranks<T, F>(size: usize, f: F) -> Vec<T>
where
    F: Fn(Arc<DistributedContext>) -> T + Sync,
    T: Send,
{
    run_ranks_max_gather(size, f).0
}

/// [`run_ranks`] plus the largest block any rank passed to an allgather, which
/// distinguishes a dense gather from a scalar reduction.
pub fn run_ranks_max_gather<T, F>(size: usize, f: F) -> (Vec<T>, usize)
where
    F: Fn(Arc<DistributedContext>) -> T + Sync,
    T: Send,
{
    let shared = LoopbackShared::new(size, MPI_MAX_COUNT);
    let results = run_shared(&shared, f);
    let max_gather = shared.state.lock().unwrap().max_gather_block;
    (results, max_gather)
}

/// [`run_ranks`] under a transport that panics on any call carrying more than
/// `count_limit` elements, plus the largest amplitude count any rank passed to
/// one `sendrecv_c64`.
#[cfg(test)]
pub(crate) fn run_ranks_max_exchange<T, F>(size: usize, count_limit: usize, f: F) -> (Vec<T>, usize)
where
    F: Fn(Arc<DistributedContext>) -> T + Sync,
    T: Send,
{
    let shared = LoopbackShared::new(size, count_limit);
    let results = run_shared(&shared, f);
    (results, shared.max_exchange.load(Ordering::Relaxed))
}

fn run_shared<T, F>(shared: &Arc<LoopbackShared>, f: F) -> Vec<T>
where
    F: Fn(Arc<DistributedContext>) -> T + Sync,
    T: Send,
{
    let size = shared.size;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..size)
            .map(|rank| {
                let comm = LoopbackComm {
                    shared: shared.clone(),
                    rank,
                };
                let f = &f;
                std::thread::Builder::new()
                    .stack_size(RANK_STACK_BYTES)
                    .spawn_scoped(scope, move || {
                        f(DistributedContext::from_comm(Arc::new(comm)))
                    })
                    .expect("spawn rank thread")
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread panicked"))
            .collect()
    })
}

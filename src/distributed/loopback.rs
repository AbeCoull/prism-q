//! Thread-backed [`RankComm`] for driving multi-rank code without an MPI
//! launcher. Available to tests and, behind `bench-internal`, to benchmarks.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use num_complex::Complex64;

use super::{DistributedContext, RankComm};

/// Debug SIMD kernels overflow the default thread stack at the widths the
/// loopback suite runs.
const RANK_STACK_BYTES: usize = 64 * 1024 * 1024;

struct LoopbackShared {
    size: usize,
    state: Mutex<LoopbackState>,
    cv: Condvar,
}

struct LoopbackState {
    generation: u64,
    arrived: usize,
    cslots: Vec<Vec<Complex64>>,
    fslots: Vec<f64>,
    reduce: Vec<f64>,
    /// Largest block one rank passed to an allgather. Shot sampling tests
    /// assert this stays at one element, proving no dense gather happened.
    max_gather_block: usize,
    /// Mailboxes indexed by `sender * size + receiver`. FIFO order matches MPI
    /// sendrecv order and stays separate from collective barriers.
    mailbox: Vec<VecDeque<Vec<Complex64>>>,
}

impl LoopbackShared {
    fn new(size: usize) -> Arc<Self> {
        Arc::new(Self {
            size,
            state: Mutex::new(LoopbackState {
                generation: 0,
                arrived: 0,
                cslots: vec![Vec::new(); size],
                fslots: vec![0.0; size],
                reduce: Vec::new(),
                max_gather_block: 0,
                mailbox: (0..size * size).map(|_| VecDeque::new()).collect(),
            }),
            cv: Condvar::new(),
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

impl RankComm for LoopbackComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.shared.size
    }

    fn allgather_c64(&self, local: &[Complex64]) -> Vec<Complex64> {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.max_gather_block = st.max_gather_block.max(local.len());
            st.cslots[self.rank] = local.to_vec();
        }
        self.shared.barrier();
        let out = {
            let st = self.shared.state.lock().unwrap();
            st.cslots.iter().flat_map(|s| s.iter().copied()).collect()
        };
        self.shared.barrier();
        out
    }

    fn allgather_f64(&self, local: &[f64]) -> Vec<f64> {
        let as_c: Vec<Complex64> = local.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        self.allgather_c64(&as_c).iter().map(|c| c.re).collect()
    }

    fn allreduce_sum_f64(&self, value: f64) -> f64 {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.fslots[self.rank] = value;
        }
        self.shared.barrier();
        let sum = {
            let st = self.shared.state.lock().unwrap();
            st.fslots.iter().sum()
        };
        self.shared.barrier();
        sum
    }

    fn allreduce_sum_f64_slice(&self, values: &mut [f64]) {
        {
            let mut st = self.shared.state.lock().unwrap();
            if st.reduce.len() != values.len() {
                st.reduce = vec![0.0; values.len()];
            }
            for (acc, &v) in st.reduce.iter_mut().zip(values.iter()) {
                *acc += v;
            }
        }
        self.shared.barrier();
        {
            let st = self.shared.state.lock().unwrap();
            values.copy_from_slice(&st.reduce);
        }
        self.shared.barrier();
        if self.rank == 0 {
            let mut st = self.shared.state.lock().unwrap();
            st.reduce.clear();
        }
        self.shared.barrier();
    }

    fn sendrecv_c64(&self, partner: usize, send: &[Complex64], recv: &mut [Complex64]) {
        debug_assert_eq!(send.len(), recv.len());
        let size = self.shared.size;
        let mut st = self.shared.state.lock().unwrap();
        // Send to partner, then wait for partner to send back. Ranks that skip
        // an exchange do not block because their partner skips it too.
        st.mailbox[self.rank * size + partner].push_back(send.to_vec());
        self.shared.cv.notify_all();
        let inbox = partner * size + self.rank;
        loop {
            if let Some(msg) = st.mailbox[inbox].pop_front() {
                recv.copy_from_slice(&msg);
                return;
            }
            st = self.shared.cv.wait(st).unwrap();
        }
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
    let shared = LoopbackShared::new(size);
    let results = std::thread::scope(|scope| {
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
    });
    let max_gather = shared.state.lock().unwrap().max_gather_block;
    (results, max_gather)
}

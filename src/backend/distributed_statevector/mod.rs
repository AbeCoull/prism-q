//! Distributed state vector backend.
//!
//! Splits the `2^n` amplitude vector across `P = 2^p` ranks. The low `n - p`
//! qubits index the local slice. The top `p` qubits select the rank. Each rank
//! stores `2^(n - p)` amplitudes in an inner [`StatevectorBackend`].
//!
//! # Memory layout
//!
//! Global index: `rank * 2^(n - p) + local_index`. If `q < n - p`, qubit `q` is
//! bit `q` of `local_index`; otherwise it is bit `q - (n - p)` of the rank id.
//! Qubit 0 is the least significant bit. `|0...0>` is index 0 on rank 0.
//!
//! # Gate support
//!
//! Implemented: local gates, rank bit one qubit gates, two qubit gates, controlled
//! gates across rank bits, `probabilities`, and `export_statevector`. A global
//! control is constant on a rank, so it gates the whole slice with no
//! communication. Diagonal controlled gates never communicate. With one rank
//! ([`SerialComm`](crate::distributed::SerialComm)), every qubit is local.
//!
//! Fusion runs in every mode. Local fused gates dispatch to the inner SIMD
//! kernels. Fused or batched gates that span rank bits are decomposed into the
//! paths above. A general two qubit gate over one global qubit needs one
//! pairwise exchange; over two global qubits it runs a two step butterfly
//! across the group of four ranks that share the other rank bits.
//!
//! Once a rank resolves its global qubit bits, the remaining gate is local and
//! dispatches to the inner backend. The only manual amplitude loops combine the
//! received buffers after communication.
//!
//! Measurement, reset, and classical conditionals are supported. Measurement
//! probabilities are summed with `Allreduce`. Each rank uses the same seeded RNG,
//! so ranks agree without exchanging the draw. Reset runs one trajectory of the
//! reset channel, as [`Backend::reset`] specifies: sample the outcome, collapse
//! onto it, and apply X when it is 1.
//!
//! Per-qubit probabilities and Pauli expectation values answer from rank-local
//! sums plus one `Allreduce`, at any register width. `probabilities` and
//! `export_statevector` are the only queries that gather, so they carry the
//! dense output cap; `Simulate::run` rejects a beyond-cap register before the
//! run rather than returning no distribution after it.
//!
//! # When to prefer this backend
//!
//! - Amplitude vectors too large for one host, split across MPI ranks.
//!   Requested via `BackendKind::StatevectorDistributed`; Auto never selects it.
//! - Evaluating qubit routing strategies through the exchange counters, on one
//!   host via the serial or loopback transports.
//!
//! # When NOT to use this backend
//!
//! - Circuits that fit one host; the inner statevector backend does the same
//!   work without collectives.
//! - Per-shot noise trajectories; `run_shots_with_noise` rejects the backend
//!   (see the `backend` module docs).
//!
//! # Qubit relabeling
//!
//! At more than one rank the backend keeps a circuit-to-physical qubit map
//! (on by default, see [`crate::distributed::relabel_enabled`]). SWAP becomes a
//! map update: no amplitudes move and no rank communicates, at any local or
//! global split. Before a gate applies non-diagonal action to a qubit in a rank
//! bit position, the qubit is relabeled into a local position by exchanging the
//! half slice whose local bit differs from the rank bit, evicting the least
//! recently used local qubit. The gate and every later gate on that qubit then
//! run on the inner SIMD kernels with no further communication, until the qubit
//! is evicted again. Diagonal action and control bits stay free on global
//! qubits, so they never trigger a relabel.
//!
//! Gate targets and the qubit indices inside batched gate data (`MultiFused`,
//! `Multi2q`, `BatchPhase`, `BatchRzz`, `DiagonalBatch`) are translated through
//! the map at apply time. `probabilities` and `export_statevector` reorder the
//! gathered vector back to circuit qubit order; measurement, reset, and
//! `qubit_probability` translate the qubit index. The direct per-gate exchange
//! paths remain for relabeling disabled and for instructions whose qubits
//! cannot all be made local (no eviction victim).
//!
//! Relabeling wins whenever gate activity has qubit locality: SWAP networks,
//! repeated gates on the same qubits, and working sets that fit the local
//! positions. The known adverse pattern is a cyclic scan, a gate wall over
//! more hot qubits than local positions repeated layer after layer, which
//! defeats least recently used eviction and can exceed direct exchange volume.
//! Lookahead epoch planning addresses that case; until then
//! `PRISM_DIST_RELABEL=0` restores direct exchange.
//!
//! # Communication cost
//!
//! Only gates that are not diagonal and touch a global target communicate. With
//! relabeling, a global SWAP costs nothing and the first non-diagonal gate on a
//! global qubit costs a half-slice relabel exchange that also makes later gates
//! on that qubit local. On the direct paths, a global one qubit gate, or a two
//! qubit gate over one global qubit, costs one pairwise exchange of the local
//! slice; a two qubit gate over two global qubits costs two, one per rank bit.
//! Every direct exchange and the relabel exchange are tiled by
//! [`crate::distributed::exchange_chunk`], which bounds the transfer buffers.
//!
//! [`DistributedStatevectorBackend::exchange_messages`] and
//! [`DistributedStatevectorBackend::exchange_amplitudes`] expose rank local
//! communication volume. Use these counters to evaluate qubit reordering and
//! routing, since one host cannot measure real network latency.
//!
//! # Shot sampling
//!
//! Circuits whose measurements are terminal sample shots without gathering the
//! dense state or probability vector on any rank; communication scales with
//! the rank count and shot count, never with the state size. See
//! [`DistributedStatevectorBackend::sample_state_indices`] for the algorithm,
//! which [`Backend::sample_basis_states`] exposes at the trait level. Circuits
//! with mid-circuit measurements fall back to one lockstep run per shot.
//!
//! Not implemented yet: lookahead epoch planning that batches several relabels
//! into one exchange.

#[cfg(test)]
mod tests;
#[cfg(any(test, feature = "bench-internal"))]
pub mod tiled;

use std::borrow::Cow;
use std::sync::Arc;

use num_complex::Complex64;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::backend::simd;
#[cfg(feature = "parallel")]
use crate::backend::statevector::SendPtr;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::{
    Backend, BasisSamples, dense_probability_len, dense_statevector_len, measurement_inv_norm,
};
#[cfg(feature = "parallel")]
use crate::backend::{
    MIN_PAR_ELEMS, MIN_PAR_REDUCE_ELEMS, PARALLEL_THRESHOLD_QUBITS, chunk_min_len,
};
use crate::circuit::{Instruction, SmallVec, smallvec};
use crate::distributed::DistributedContext;
use crate::error::{PrismError, Result};
use crate::gates::{DiagEntry, Gate, is_diagonal_2x2};
use crate::sim::unified_pauli::PauliTerm;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const BACKEND_NAME: &str = "distributed_statevector";

/// Shard length at which the combine loops below fan out to Rayon: the same
/// `2^14` amplitudes the inner backend uses for its own kernels.
#[cfg(feature = "parallel")]
const PAR_SHARD_LEN: usize = 1 << PARALLEL_THRESHOLD_QUBITS;

fn scale_shard(state: &mut [Complex64], factor: Complex64) {
    #[cfg(feature = "parallel")]
    if state.len() >= PAR_SHARD_LEN {
        state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .for_each(|tile| simd::scale_complex_slice(tile, factor));
        return;
    }
    simd::scale_complex_slice(state, factor);
}

/// Scale the amplitudes whose index has every bit of `mask` set.
fn scale_shard_masked(state: &mut [Complex64], mask: usize, factor: Complex64) {
    let tile = |base: usize, tile: &mut [Complex64]| {
        for (k, amp) in tile.iter_mut().enumerate() {
            if (base + k) & mask == mask {
                *amp *= factor;
            }
        }
    };
    #[cfg(feature = "parallel")]
    if state.len() >= PAR_SHARD_LEN {
        state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(t, chunk)| tile(t * MIN_PAR_ELEMS, chunk));
        return;
    }
    tile(0, state);
}

fn zero_shard(state: &mut [Complex64]) {
    #[cfg(feature = "parallel")]
    if state.len() >= PAR_SHARD_LEN {
        state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .for_each(simd::zero_slice);
        return;
    }
    simd::zero_slice(state);
}

/// `dst[i] = c_self * dst[i] + c_remote * remote[i]` over a received block.
fn combine_shard(
    dst: &mut [Complex64],
    remote: &[Complex64],
    c_self: Complex64,
    c_remote: Complex64,
) {
    #[cfg(feature = "parallel")]
    if dst.len() >= PAR_SHARD_LEN {
        dst.par_chunks_mut(MIN_PAR_ELEMS)
            .zip(remote.par_chunks(MIN_PAR_ELEMS))
            .for_each(|(d, r)| simd::combine_global_half(d, r, c_self, c_remote));
        return;
    }
    simd::combine_global_half(dst, remote, c_self, c_remote);
}

/// `pack[k] = state[index_of(k)]` for every `k`.
fn gather_indexed(
    pack: &mut [Complex64],
    state: &[Complex64],
    index_of: impl Fn(usize) -> usize + Sync,
) {
    #[cfg(feature = "parallel")]
    if pack.len() >= PAR_SHARD_LEN {
        pack.par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(t, tile)| {
                let base = t * MIN_PAR_ELEMS;
                for (k, slot) in tile.iter_mut().enumerate() {
                    *slot = state[index_of(base + k)];
                }
            });
        return;
    }
    for (k, slot) in pack.iter_mut().enumerate() {
        *slot = state[index_of(k)];
    }
}

/// `state[index_of(k)] = f(state[index_of(k)], recv[k])` for every `k`.
/// `index_of` must be injective on `0..recv.len()`: the parallel arm relies on
/// it to keep the tasks' writes disjoint.
fn scatter_indexed(
    state: &mut [Complex64],
    recv: &[Complex64],
    index_of: impl Fn(usize) -> usize + Sync,
    f: impl Fn(Complex64, Complex64) -> Complex64 + Sync,
) {
    #[cfg(feature = "parallel")]
    if recv.len() >= PAR_SHARD_LEN {
        let ptr = SendPtr(state.as_mut_ptr());
        recv.par_chunks(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(t, tile)| {
                let base = t * MIN_PAR_ELEMS;
                for (k, &r) in tile.iter().enumerate() {
                    let i = index_of(base + k);
                    // SAFETY: `index_of` is injective and maps into `state`, so
                    // each index is read and written by exactly one task and no
                    // two tasks touch the same amplitude.
                    unsafe { ptr.store(i, f(ptr.load(i), r)) };
                }
            });
        return;
    }
    for (k, &r) in recv.iter().enumerate() {
        let i = index_of(k);
        state[i] = f(state[i], r);
    }
}

/// Visit the `(lo, hi)` halves of every `2^(local_q + 1)` block of `state`
/// together with the same halves of `recv`, as four equal-length tiles whose
/// `k`-th elements share a basis index apart from the `local_q` bit.
fn for_each_pair_tile<F>(state: &mut [Complex64], recv: &mut [Complex64], local_q: usize, f: F)
where
    F: Fn(&mut [Complex64], &mut [Complex64], &mut [Complex64], &mut [Complex64]) + Sync,
{
    let half = 1usize << local_q;
    let block = half << 1;
    #[cfg(feature = "parallel")]
    if state.len() >= PAR_SHARD_LEN {
        if state.len() / block >= 4 {
            state
                .par_chunks_mut(block)
                .zip(recv.par_chunks_mut(block))
                .with_min_len(chunk_min_len(block))
                .for_each(|(s, r)| {
                    let (lo, hi) = s.split_at_mut(half);
                    let (rlo, rhi) = r.split_at_mut(half);
                    f(lo, hi, rlo, rhi);
                });
        } else {
            for (s, r) in state.chunks_mut(block).zip(recv.chunks_mut(block)) {
                let (lo, hi) = s.split_at_mut(half);
                let (rlo, rhi) = r.split_at_mut(half);
                lo.par_chunks_mut(MIN_PAR_ELEMS)
                    .zip(hi.par_chunks_mut(MIN_PAR_ELEMS))
                    .zip(rlo.par_chunks_mut(MIN_PAR_ELEMS))
                    .zip(rhi.par_chunks_mut(MIN_PAR_ELEMS))
                    .for_each(|(((lo, hi), rlo), rhi)| f(lo, hi, rlo, rhi));
            }
        }
        return;
    }
    for (s, r) in state.chunks_mut(block).zip(recv.chunks_mut(block)) {
        let (lo, hi) = s.split_at_mut(half);
        let (rlo, rhi) = r.split_at_mut(half);
        f(lo, hi, rlo, rhi);
    }
}

/// Butterfly step over a received block: `pack[i] = forward[0] * state[i] +
/// forward[1] * remote[i]` is the partial sum forwarded to the next partner,
/// then `state[i] = keep[0] * state[i] + keep[1] * remote[i]`.
fn butterfly_shard(
    state: &mut [Complex64],
    remote: &[Complex64],
    pack: &mut [Complex64],
    keep: [Complex64; 2],
    forward: [Complex64; 2],
) {
    let tile = |state: &mut [Complex64], remote: &[Complex64], pack: &mut [Complex64]| {
        for ((s, &r), p) in state.iter_mut().zip(remote).zip(pack.iter_mut()) {
            let own = *s;
            *p = forward[0] * own + forward[1] * r;
            *s = keep[0] * own + keep[1] * r;
        }
    };
    #[cfg(feature = "parallel")]
    if state.len() >= PAR_SHARD_LEN {
        state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .zip(remote.par_chunks(MIN_PAR_ELEMS))
            .zip(pack.par_chunks_mut(MIN_PAR_ELEMS))
            .for_each(|((s, r), p)| tile(s, r, p));
        return;
    }
    tile(state, remote, pack);
}

/// `dst[i] += src[i]` over a received block.
fn add_shard(dst: &mut [Complex64], src: &[Complex64]) {
    let tile = |dst: &mut [Complex64], src: &[Complex64]| {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d += s;
        }
    };
    #[cfg(feature = "parallel")]
    if dst.len() >= PAR_SHARD_LEN {
        dst.par_chunks_mut(MIN_PAR_ELEMS)
            .zip(src.par_chunks(MIN_PAR_ELEMS))
            .for_each(|(d, s)| tile(d, s));
        return;
    }
    tile(dst, src);
}

/// `sum |a|^2` over the `qubit == outcome` half of every block of a shard,
/// parallel above `MIN_PAR_REDUCE_ELEMS` like `state_norm_sqr`.
fn half_norm_sqr(state: &[Complex64], qubit: usize, outcome: bool) -> f64 {
    fn select(block: &[Complex64], half: usize, outcome: bool) -> &[Complex64] {
        if outcome {
            &block[half..]
        } else {
            &block[..half]
        }
    }
    let half = 1usize << qubit;
    let block = half << 1;
    #[cfg(feature = "parallel")]
    if state.len() >= MIN_PAR_REDUCE_ELEMS {
        if state.len() / block >= 4 {
            return state
                .par_chunks(block)
                .with_min_len(chunk_min_len(block))
                .map(|b| simd::norm_sqr_sum(select(b, half, outcome)))
                .sum();
        }
        return state
            .chunks(block)
            .map(|b| {
                select(b, half, outcome)
                    .par_chunks(MIN_PAR_ELEMS)
                    .map(simd::norm_sqr_sum)
                    .sum::<f64>()
            })
            .sum();
    }
    state
        .chunks(block)
        .map(|b| simd::norm_sqr_sum(select(b, half, outcome)))
        .sum()
}

/// Visit every circuit qubit an instruction touches: the instruction targets
/// plus qubit indices stored inside batched gate data. Indices may repeat.
fn for_each_gate_qubit(gate: &Gate, targets: &[usize], mut f: impl FnMut(usize)) {
    for &q in targets {
        f(q);
    }
    match gate {
        Gate::BatchPhase(data) => {
            for &(target, _) in &data.phases {
                f(target);
            }
        }
        Gate::BatchRzz(data) => {
            for &(q0, q1, _) in &data.edges {
                f(q0);
                f(q1);
            }
        }
        Gate::MultiFused(data) => {
            for &(q, _) in &data.gates {
                f(q);
            }
        }
        Gate::Multi2q(data) => {
            for &(q0, q1, _) in &data.gates {
                f(q0);
                f(q1);
            }
        }
        Gate::DiagonalBatch(data) => {
            for entry in &data.entries {
                match *entry {
                    DiagEntry::Phase1q { qubit, .. } => f(qubit),
                    DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } => {
                        f(q0);
                        f(q1);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Circuit qubits that must occupy local positions for the gate to apply
/// without a per-gate amplitude exchange. Diagonal action and control bits
/// are free on global qubits, so only non-diagonal application targets count.
fn required_local_qubits(gate: &Gate, targets: &[usize]) -> SmallVec<[usize; 8]> {
    let mut req: SmallVec<[usize; 8]> = SmallVec::new();
    fn push(req: &mut SmallVec<[usize; 8]>, q: usize) {
        if !req.contains(&q) {
            req.push(q);
        }
    }
    match gate {
        Gate::Cx => push(&mut req, targets[1]),
        Gate::Cz
        | Gate::Swap
        | Gate::Rzz(_)
        | Gate::BatchPhase(_)
        | Gate::BatchRzz(_)
        | Gate::DiagonalBatch(_) => {}
        Gate::Cu(_) | Gate::Mcu(_) => {
            if gate.controlled_phase().is_none() {
                let (target, mat) = match gate {
                    Gate::Mcu(data) => (targets[data.num_controls as usize], &data.mat),
                    Gate::Cu(mat) => (targets[1], &**mat),
                    _ => unreachable!("outer match arm is Cu | Mcu"),
                };
                if !is_diagonal_2x2(mat) {
                    push(&mut req, target);
                }
            }
        }
        Gate::Fused2q(_) => {
            push(&mut req, targets[0]);
            push(&mut req, targets[1]);
        }
        Gate::Multi2q(data) => {
            for &(q0, q1, _) in &data.gates {
                push(&mut req, q0);
                push(&mut req, q1);
            }
        }
        Gate::MultiFused(data) => {
            for &(q, ref mat) in &data.gates {
                if !is_diagonal_2x2(mat) {
                    push(&mut req, q);
                }
            }
        }
        g if g.num_qubits() == 1 && !g.is_diagonal_1q() => {
            push(&mut req, targets[0]);
        }
        _ => {}
    }
    req
}

/// Distributed state vector backend over an `Arc`-shared [`DistributedContext`].
pub struct DistributedStatevectorBackend {
    context: Arc<DistributedContext>,
    inner: StatevectorBackend,
    num_qubits: usize,
    global_qubits: usize,
    /// Receive buffer for the direct exchange paths. Grows to the largest tile
    /// requested and never shrinks; callers take a `[..len]` view.
    recv: Vec<Complex64>,
    seed: u64,
    /// Max amplitudes exchanged per message on the direct exchange paths.
    /// Tiling bounds `recv` and `pack` to one tile; the two qubit paths round
    /// the tile down to whole pair blocks.
    exchange_chunk: usize,
    /// Count of `sendrecv` messages issued by this rank, and the total
    /// amplitudes exchanged. Reorder and routing passes should minimize these
    /// counters.
    exchange_messages: u64,
    exchange_amplitudes: u64,
    /// RNG for measurement decisions, seeded identically on every rank and
    /// advanced in lockstep. Outcomes are derived from `Allreduce`d global
    /// probabilities, so all ranks agree without exchanging the draw.
    meas_rng: ChaCha8Rng,
    /// Circuit qubit to physical position. Positions below `local_qubits()`
    /// index the local slice; the rest are rank bits. Identity until a SWAP or
    /// a relabel exchange moves a qubit.
    qubit_map: Vec<usize>,
    /// Physical position to circuit qubit. Inverse of `qubit_map`.
    phys_map: Vec<usize>,
    /// Fast path flag: true while `qubit_map` is the identity.
    map_identity: bool,
    /// Whether gates relabel global qubits into local positions instead of
    /// exchanging amplitudes per gate.
    relabel: bool,
    /// Instruction tick at which each circuit qubit was last referenced.
    /// Drives least recently used eviction for relabel victims.
    last_used: Vec<u64>,
    tick: u64,
    /// Send-side buffer for the indexed exchanges and the forwarded partial of
    /// the two global butterfly. Grow-only like `recv`.
    pack: Vec<Complex64>,
    /// Armed by `init`, spent by the first instruction batch, which is where the
    /// circuit fingerprint is cross-checked.
    circuit_check_pending: bool,
}

impl DistributedStatevectorBackend {
    /// Create a backend bound to the given rank context and RNG seed.
    pub fn new(context: Arc<DistributedContext>, seed: u64) -> Self {
        Self {
            context,
            inner: StatevectorBackend::new(seed),
            num_qubits: 0,
            global_qubits: 0,
            recv: Vec::new(),
            seed,
            exchange_chunk: crate::distributed::exchange_chunk(),
            exchange_messages: 0,
            exchange_amplitudes: 0,
            meas_rng: ChaCha8Rng::seed_from_u64(seed),
            qubit_map: Vec::new(),
            phys_map: Vec::new(),
            map_identity: true,
            relabel: crate::distributed::relabel_enabled(),
            last_used: Vec::new(),
            tick: 0,
            pack: Vec::new(),
            circuit_check_pending: true,
        }
    }

    /// Override the exchange chunk size in amplitudes. Tests use this to cover
    /// the tiled path without using the process environment.
    #[cfg(test)]
    pub(crate) fn set_exchange_chunk(&mut self, chunk: usize) {
        self.exchange_chunk = chunk.max(1);
    }

    /// Enable or disable qubit relabeling for this backend instance, overriding
    /// the `PRISM_DIST_RELABEL` default. With relabeling off, every gate on a
    /// global qubit uses the direct per-gate exchange paths.
    pub fn set_relabel(&mut self, enabled: bool) {
        self.relabel = enabled;
    }

    /// Number of `sendrecv` messages this rank has issued since `init`.
    ///
    /// Cost proxy for this backend. One host cannot measure real network
    /// latency, so routing changes are evaluated against this count. Counts
    /// gate and relabel exchanges; the query paths take `&self` and cannot
    /// record theirs.
    pub fn exchange_messages(&self) -> u64 {
        self.exchange_messages
    }

    /// Total amplitudes this rank has sent across all exchanges since `init`.
    pub fn exchange_amplitudes(&self) -> u64 {
        self.exchange_amplitudes
    }

    /// Record a pairwise exchange of `amplitudes` for the cost counters.
    #[inline]
    fn count_exchange(&mut self, amplitudes: usize) {
        self.exchange_messages += 1;
        self.exchange_amplitudes += amplitudes as u64;
    }

    /// Grow `recv` to at least `len` amplitudes. Never shrinks, so paths that
    /// alternate between chunk and slice lengths reuse one allocation and skip
    /// the refill.
    #[inline]
    fn ensure_recv(&mut self, len: usize) {
        if self.recv.len() < len {
            self.recv.resize(len, Complex64::new(0.0, 0.0));
        }
    }

    /// `pack` counterpart of [`Self::ensure_recv`].
    #[inline]
    fn ensure_pack(&mut self, len: usize) {
        if self.pack.len() < len {
            self.pack.resize(len, Complex64::new(0.0, 0.0));
        }
    }

    /// Exchange chunk rounded down to whole `2^block_bits` blocks, at least one
    /// block and at most `len`, so a tile holds complete pair blocks.
    #[inline]
    fn block_chunk(&self, block_bits: usize, len: usize) -> usize {
        let block = 1usize << block_bits;
        ((self.exchange_chunk / block).max(1) * block).min(len)
    }

    #[inline]
    fn local_qubits(&self) -> usize {
        self.num_qubits - self.global_qubits
    }

    #[inline]
    fn is_single_rank(&self) -> bool {
        self.context.size() == 1
    }

    /// Bit position within the rank id for global qubit `q` (`q >= local`).
    #[inline]
    fn global_bit(&self, q: usize) -> usize {
        q - self.local_qubits()
    }

    /// Whether this rank holds the `|1>` half of global qubit `q`.
    #[inline]
    fn rank_bit_set(&self, q: usize) -> bool {
        (self.context.rank() >> self.global_bit(q)) & 1 == 1
    }

    /// Advance the instruction tick and mark every circuit qubit the
    /// instruction references. Marked qubits are exempt from eviction until the
    /// next instruction. Identical on every rank because the instruction stream
    /// is identical.
    fn touch_instruction(&mut self, gate: &Gate, targets: &[usize]) {
        self.tick += 1;
        let tick = self.tick;
        for_each_gate_qubit(gate, targets, |q| self.last_used[q] = tick);
    }

    fn refresh_map_identity(&mut self) {
        self.map_identity = self.qubit_map.iter().enumerate().all(|(q, &p)| q == p);
    }

    /// Apply SWAP as a pure relabeling: exchange the two circuit qubits' map
    /// entries. No amplitudes move and no rank communicates.
    fn swap_circuit_qubits(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let pa = self.qubit_map[a];
        let pb = self.qubit_map[b];
        self.qubit_map.swap(a, b);
        self.phys_map.swap(pa, pb);
        self.refresh_map_identity();
    }

    /// Local position holding the least recently used circuit qubit that the
    /// current instruction does not reference. `None` when every local qubit is
    /// referenced this tick.
    fn pick_victim(&self) -> Option<usize> {
        let local = self.local_qubits();
        let mut best: Option<(u64, usize)> = None;
        for pos in 0..local {
            let used = self.last_used[self.phys_map[pos]];
            if used == self.tick {
                continue;
            }
            match best {
                Some((b, _)) if used >= b => {}
                _ => best = Some((used, pos)),
            }
        }
        best.map(|(_, pos)| pos)
    }

    /// Bring each requested circuit qubit into a local position. Best effort:
    /// stops when no eviction victim remains, leaving the rest to the direct
    /// exchange paths. Each relabel costs one half-slice exchange.
    fn make_local(&mut self, req: &[usize]) {
        for &q in req {
            let pos = self.qubit_map[q];
            if pos < self.local_qubits() {
                continue;
            }
            let Some(victim) = self.pick_victim() else {
                return;
            };
            self.relabel_swap(victim, pos);
        }
    }

    /// Exchange the moving half of a local/global SWAP with the partner rank.
    /// Only amplitudes whose local bit differs from this rank's bit of the
    /// global position move, so each rank exchanges half its slice. Both ranks
    /// enumerate their moving halves in ascending index order, which the
    /// single-bit XOR relation between the two sets preserves, so the k-th
    /// received amplitude lands at the k-th moving index. Tiled by
    /// `exchange_chunk` like the direct global exchange. Pure data movement:
    /// the qubit map is untouched.
    fn half_slice_swap(&mut self, local_pos: usize, global_pos: usize) {
        let partner = self.context.rank() ^ (1usize << self.global_bit(global_pos));
        let gbit = self.rank_bit_set(global_pos);
        let stride = 1usize << local_pos;
        let fixed = if gbit { 0 } else { stride };
        let moving = self.inner.state.len() / 2;
        let chunk = self.exchange_chunk.min(moving).max(1);
        self.ensure_pack(chunk);
        self.ensure_recv(chunk);
        let index_of =
            |flat: usize| ((flat >> local_pos) << (local_pos + 1)) | fixed | (flat & (stride - 1));
        let mut off = 0;
        while off < moving {
            let count = (off + chunk).min(moving) - off;
            gather_indexed(&mut self.pack[..count], &self.inner.state, |k| {
                index_of(off + k)
            });
            self.count_exchange(count);
            self.context
                .comm()
                .sendrecv_c64(partner, &self.pack[..count], &mut self.recv[..count]);
            scatter_indexed(
                &mut self.inner.state,
                &self.recv[..count],
                |k| index_of(off + k),
                |_, remote| remote,
            );
            off += count;
        }
    }

    /// Physically swap the qubits at a local and a global position, then update
    /// the map.
    fn relabel_swap(&mut self, local_pos: usize, global_pos: usize) {
        self.half_slice_swap(local_pos, global_pos);

        let local_q = self.phys_map[local_pos];
        let global_q = self.phys_map[global_pos];
        self.qubit_map[local_q] = global_pos;
        self.qubit_map[global_q] = local_pos;
        self.phys_map.swap(local_pos, global_pos);
        self.refresh_map_identity();
    }

    /// Physically swap the qubits at positions `a < b` and update the map.
    /// When both positions are local, this runs the inner SWAP kernel. When one
    /// position is local and one is global, this reuses the half-slice relabel
    /// exchange. When both positions are global, this exchanges full slices
    /// between rank pairs whose two bits differ.
    fn swap_physical_positions(&mut self, a: usize, b: usize) {
        debug_assert!(
            a < b,
            "positions must be ordered: branch selection assumes a < b"
        );
        let local = self.local_qubits();
        if b < local {
            self.inner
                .apply(&Instruction::Gate {
                    gate: Gate::Swap,
                    targets: smallvec![a, b],
                })
                .expect("local SWAP cannot fail");
        } else if a < local {
            self.relabel_swap(a, b);
            return;
        } else {
            self.swap_global_slices(a, b);
        }
        let qa = self.phys_map[a];
        let qb = self.phys_map[b];
        self.qubit_map[qa] = b;
        self.qubit_map[qb] = a;
        self.phys_map.swap(a, b);
        self.refresh_map_identity();
    }

    /// Physically reorder the state until every circuit qubit occupies its own
    /// position. Runs in lockstep on every rank because the maps are identical.
    /// Each misplaced qubit costs at most one exchange. The identity map
    /// returns without work.
    fn restore_identity_map(&mut self) {
        while !self.map_identity {
            let Some(pos) = (0..self.num_qubits).find(|&p| self.phys_map[p] != p) else {
                break;
            };
            let src = self.qubit_map[pos];
            self.swap_physical_positions(pos, src);
        }
    }

    /// Translate an instruction into physical positions: map the targets and
    /// rewrite qubit indices stored inside batched gate data. Borrows the gate
    /// unchanged while the map is the identity.
    fn to_physical<'g>(
        &self,
        gate: &'g Gate,
        targets: &[usize],
    ) -> (Cow<'g, Gate>, SmallVec<[usize; 4]>) {
        if self.map_identity {
            return (Cow::Borrowed(gate), targets.into());
        }
        let ptargets: SmallVec<[usize; 4]> = targets.iter().map(|&q| self.qubit_map[q]).collect();
        // A payload whose every index maps to itself is borrowed unchanged:
        // once any relabel leaves the map non-identity, a per-application deep
        // clone of every batched payload is the steady state otherwise.
        let fixed = |q: usize| self.qubit_map[q] == q;
        let pgate = match gate {
            Gate::MultiFused(data) => {
                if data.gates.iter().all(|&(q, _)| fixed(q)) {
                    Cow::Borrowed(gate)
                } else {
                    let mut data = data.clone();
                    for entry in &mut data.gates {
                        entry.0 = self.qubit_map[entry.0];
                    }
                    Cow::Owned(Gate::MultiFused(data))
                }
            }
            Gate::Multi2q(data) => {
                if data.gates.iter().all(|&(q0, q1, _)| fixed(q0) && fixed(q1)) {
                    Cow::Borrowed(gate)
                } else {
                    let mut data = data.clone();
                    for entry in &mut data.gates {
                        entry.0 = self.qubit_map[entry.0];
                        entry.1 = self.qubit_map[entry.1];
                    }
                    Cow::Owned(Gate::Multi2q(data))
                }
            }
            Gate::BatchPhase(data) => {
                if data.phases.iter().all(|&(q, _)| fixed(q)) {
                    Cow::Borrowed(gate)
                } else {
                    let mut data = data.clone();
                    for entry in &mut data.phases {
                        entry.0 = self.qubit_map[entry.0];
                    }
                    Cow::Owned(Gate::BatchPhase(data))
                }
            }
            Gate::BatchRzz(data) => {
                if data.edges.iter().all(|&(q0, q1, _)| fixed(q0) && fixed(q1)) {
                    Cow::Borrowed(gate)
                } else {
                    let mut data = data.clone();
                    for entry in &mut data.edges {
                        entry.0 = self.qubit_map[entry.0];
                        entry.1 = self.qubit_map[entry.1];
                    }
                    Cow::Owned(Gate::BatchRzz(data))
                }
            }
            Gate::DiagonalBatch(data) => {
                let entry_fixed = |entry: &DiagEntry| match *entry {
                    DiagEntry::Phase1q { qubit, .. } => fixed(qubit),
                    DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } => {
                        fixed(q0) && fixed(q1)
                    }
                };
                if data.entries.iter().all(entry_fixed) {
                    Cow::Borrowed(gate)
                } else {
                    let mut data = data.clone();
                    for entry in &mut data.entries {
                        match entry {
                            DiagEntry::Phase1q { qubit, .. } => *qubit = self.qubit_map[*qubit],
                            DiagEntry::Phase2q { q0, q1, .. }
                            | DiagEntry::Parity2q { q0, q1, .. } => {
                                *q0 = self.qubit_map[*q0];
                                *q1 = self.qubit_map[*q1];
                            }
                        }
                    }
                    Cow::Owned(Gate::DiagonalBatch(data))
                }
            }
            _ => Cow::Borrowed(gate),
        };
        (pgate, ptargets)
    }

    /// Whether every physical position the translated instruction touches,
    /// including indices inside batched gate data, is below the local boundary.
    fn instruction_qubits_local(&self, gate: &Gate, targets: &[usize]) -> bool {
        let local = self.local_qubits();
        let mut all = true;
        for_each_gate_qubit(gate, targets, |q| all &= q < local);
        all
    }

    /// Fingerprint of every setting the collective sequence assumes is shared.
    /// Rank id and rank count are excluded; they legitimately differ.
    fn config_fingerprint(&self, num_qubits: usize, num_classical_bits: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::hash::DefaultHasher::new();
        (
            self.seed,
            self.exchange_chunk,
            self.relabel,
            crate::distributed::min_local_qubits(),
            num_qubits,
            num_classical_bits,
        )
            .hash(&mut hasher);
        hasher.finish()
    }

    /// Fold the instruction stream into the value ranks compare.
    ///
    /// Hashes the `Debug` rendering rather than walking the structure. `Gate`
    /// has variants carrying dense matrices and no fingerprint of its own, and
    /// `f64`'s `Debug` round-trips, so ranks differing in any field of any
    /// instruction hash differently and a new variant is covered without being
    /// listed here. Runs once per multi-rank run, not per gate.
    fn circuit_fingerprint(instructions: &[Instruction]) -> u64 {
        use std::fmt::Write as _;
        use std::hash::Hasher;

        struct HashSink<'a>(&'a mut std::hash::DefaultHasher);
        impl std::fmt::Write for HashSink<'_> {
            fn write_str(&mut self, s: &str) -> std::fmt::Result {
                self.0.write(s.as_bytes());
                Ok(())
            }
        }

        let mut hasher = std::hash::DefaultHasher::new();
        let _ = write!(HashSink(&mut hasher), "{instructions:?}");
        hasher.finish()
    }

    /// Reject a run whose ranks were handed different circuits.
    ///
    /// [`Self::check_config_agreement`] compares seed, register shape, and the
    /// tuning knobs, all of which two ranks can agree on while still executing
    /// different gate streams. That desynchronizes the exchange sequence and
    /// hangs the job at the first collective the two streams disagree about.
    ///
    /// One collective, on the first instruction batch of a run. It cannot catch
    /// a rank that never reaches the run at all: that one hangs in this
    /// allgather instead of a later one.
    fn check_circuit_agreement(&self, instructions: &[Instruction]) -> Result<()> {
        let local = Self::circuit_fingerprint(instructions);
        let all = self.context.comm().allgather_u64(&[local]);
        match all.iter().position(|&other| other != local) {
            None => Ok(()),
            Some(other) => Err(PrismError::BackendUnsupported {
                backend: BACKEND_NAME.to_string(),
                operation: format!(
                    "the circuit on rank {} differs from rank {other}: every rank enters every \
                     collective, so every rank must run the same circuit",
                    self.context.rank()
                ),
            }),
        }
    }

    /// Reject a run whose ranks disagree about anything the collective sequence
    /// depends on.
    ///
    /// Without this the mismatch surfaces as a hang (a diverging collective
    /// order) or as silently wrong amplitudes (measurement branches drawn from
    /// different seeds), both far from the setting that caused them.
    fn check_config_agreement(&self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        let local = self.config_fingerprint(num_qubits, num_classical_bits);
        let all = self.context.comm().allgather_u64(&[local]);
        match all.iter().position(|&other| other != local) {
            None => Ok(()),
            Some(other) => Err(PrismError::BackendUnsupported {
                backend: BACKEND_NAME.to_string(),
                operation: format!(
                    "configuration on rank {} differs from rank {other}: seed, relabel mode, \
                     exchange chunk, local qubit floor, and register shape must be identical \
                     on every rank",
                    self.context.rank()
                ),
            }),
        }
    }

    /// Shared prologue of `init` and `init_from_amplitudes`: agree the register
    /// shape across ranks, check the rank count against it, and reset the qubit
    /// map to the identity. Returns the local qubit count for the shard the
    /// caller loads next.
    fn prepare_shard(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<usize> {
        let size = self.context.size();
        // Before the local validations: those read `num_qubits`, so ranks given
        // different circuits could disagree about whether to reject and leave
        // one side alone at the next collective.
        if size > 1 {
            self.check_config_agreement(num_qubits, num_classical_bits)?;
        }
        self.circuit_check_pending = true;
        if !size.is_power_of_two() {
            return Err(PrismError::BackendUnsupported {
                backend: BACKEND_NAME.to_string(),
                operation: format!("rank count {size} is not a power of two"),
            });
        }
        let p = size.trailing_zeros() as usize;
        let min_local = crate::distributed::min_local_qubits();
        if size > 1 && num_qubits < p + min_local {
            return Err(PrismError::BackendUnsupported {
                backend: BACKEND_NAME.to_string(),
                operation: format!(
                    "{num_qubits} qubits across {size} ranks leaves fewer than \
                     {min_local} local qubits per rank"
                ),
            });
        }

        self.num_qubits = num_qubits;
        self.global_qubits = p;
        self.meas_rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.exchange_messages = 0;
        self.exchange_amplitudes = 0;
        self.qubit_map = (0..num_qubits).collect();
        self.phys_map = (0..num_qubits).collect();
        self.map_identity = true;
        self.last_used = vec![0; num_qubits];
        self.tick = 0;
        Ok(num_qubits - p)
    }

    /// Translate a circuit-qubit bit mask into physical positions.
    fn to_physical_mask(&self, mask: usize) -> usize {
        if self.map_identity {
            return mask;
        }
        let mut out = 0usize;
        for (q, &pos) in self.qubit_map.iter().enumerate() {
            out |= ((mask >> q) & 1) << pos;
        }
        out
    }

    /// Reorder a gathered dense vector from physical to circuit qubit order.
    fn unpermuted<T: Copy + Default>(&self, phys: Vec<T>) -> Vec<T> {
        if self.map_identity {
            return phys;
        }
        let mut out = vec![T::default(); phys.len()];
        for (c, slot) in out.iter_mut().enumerate() {
            let mut p = 0usize;
            for (q, &pos) in self.qubit_map.iter().enumerate() {
                p |= ((c >> q) & 1) << pos;
            }
            *slot = phys[p];
        }
        out
    }

    /// Apply a one qubit gate whose target is stored in the rank id.
    ///
    /// Exchange with the partner rank, then write this rank's half of the 2x2
    /// result. The combine is elementwise, so the exchange is tiled in chunks of
    /// [`crate::distributed::exchange_chunk`] amplitudes, bounding the receive
    /// buffer to `chunk` instead of a full slice copy. The default chunk is
    /// the whole slice (single message), so behavior is unchanged unless tuned.
    fn apply_global_1q(&mut self, target: usize, mat: [[Complex64; 2]; 2]) {
        let partner = self.context.rank() ^ (1usize << self.global_bit(target));
        let (c_self, c_remote) = if self.rank_bit_set(target) {
            (mat[1][1], mat[1][0])
        } else {
            (mat[0][0], mat[0][1])
        };
        let len = self.inner.state.len();
        let chunk = self.exchange_chunk.min(len).max(1);
        self.ensure_recv(chunk);
        let mut off = 0;
        while off < len {
            let end = (off + chunk).min(len);
            self.count_exchange(end - off);
            let recv = &mut self.recv[..end - off];
            self.context
                .comm()
                .sendrecv_c64(partner, &self.inner.state[off..end], recv);
            combine_shard(&mut self.inner.state[off..end], recv, c_self, c_remote);
            off = end;
        }
    }

    /// Apply a diagonal one qubit gate whose target is stored in the rank id.
    ///
    /// The rank bit is constant across the local slice, so this only scales the
    /// slice by `d0` or `d1`.
    fn apply_global_diagonal_1q(&mut self, target: usize, d0: Complex64, d1: Complex64) {
        let factor = if self.rank_bit_set(target) { d1 } else { d0 };
        scale_shard(&mut self.inner.state, factor);
    }

    /// Apply a 2x2 matrix to a local target qubit, gated by a set of local
    /// control qubits (all must be 1). The whole operation is local, so it
    /// dispatches to the inner backend's SIMD and parallel controlled kernels.
    fn apply_local_controlled_1q(
        &mut self,
        local_controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) {
        let gate = match local_controls.len() {
            0 => {
                self.inner
                    .apply_1q_matrix(target, &mat)
                    .expect("local 1q matrix");
                return;
            }
            1 => Gate::cu(mat),
            n => Gate::mcu(mat, n as u8),
        };
        let mut targets: SmallVec<[usize; 4]> = local_controls.iter().copied().collect();
        targets.push(target);
        self.inner
            .apply(&Instruction::Gate { gate, targets })
            .expect("local controlled 1q");
    }

    /// Apply a 2x2 matrix to a global target qubit, gated by local control
    /// qubits (all must be 1). Only the control-selected sublattice is
    /// consumed, so with `k` controls each rank packs and exchanges `len / 2^k`
    /// amplitudes instead of the full slice. Both ranks enumerate the same
    /// sublattice in ascending index order, so the exchange stays aligned.
    /// Tiled by `exchange_chunk` like the direct global exchange.
    fn apply_global_controlled_1q(
        &mut self,
        local_controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) {
        if local_controls.is_empty() {
            self.apply_global_1q(target, mat);
            return;
        }
        let partner = self.context.rank() ^ (1usize << self.global_bit(target));
        let (c_self, c_remote) = if self.rank_bit_set(target) {
            (mat[1][1], mat[1][0])
        } else {
            (mat[0][0], mat[0][1])
        };

        let mut ctrl_pos: SmallVec<[usize; 4]> = local_controls.iter().copied().collect();
        ctrl_pos.sort_unstable();
        let index_of = |flat: usize| {
            let mut i = flat;
            for &p in &ctrl_pos {
                let low = i & ((1usize << p) - 1);
                i = ((i >> p) << (p + 1)) | (1usize << p) | low;
            }
            i
        };
        let moving = self.inner.state.len() >> ctrl_pos.len();
        let chunk = self.exchange_chunk.min(moving).max(1);
        self.ensure_pack(chunk);
        self.ensure_recv(chunk);
        let mut off = 0;
        while off < moving {
            let count = (off + chunk).min(moving) - off;
            gather_indexed(&mut self.pack[..count], &self.inner.state, |k| {
                index_of(off + k)
            });
            self.count_exchange(count);
            self.context
                .comm()
                .sendrecv_c64(partner, &self.pack[..count], &mut self.recv[..count]);
            scatter_indexed(
                &mut self.inner.state,
                &self.recv[..count],
                |k| index_of(off + k),
                |own, remote| c_self * own + c_remote * remote,
            );
            off += count;
        }
    }

    /// Apply a controlled gate (one target, zero or more controls) whose qubit
    /// set may span local and global qubits. Covers Cx, Cu, and Mcu uniformly.
    /// A diagonal target matrix needs no communication regardless of the split.
    fn apply_controlled_dist(
        &mut self,
        controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) {
        let local = self.local_qubits();
        let mut local_controls: SmallVec<[usize; 4]> = SmallVec::new();
        for &c in controls {
            if c < local {
                local_controls.push(c);
            } else if !self.rank_bit_set(c) {
                // A zero global control disables the gate on this rank.
                return;
            }
        }

        if target < local {
            self.apply_local_controlled_1q(&local_controls, target, mat);
        } else if is_diagonal_2x2(&mat) {
            // A global diagonal target contributes its rank bit: scale the
            // control-selected sublattice by the selected diagonal entry.
            let d = if self.rank_bit_set(target) {
                mat[1][1]
            } else {
                mat[0][0]
            };
            if local_controls.is_empty() {
                scale_shard(&mut self.inner.state, d);
                return;
            }
            let ctrl_mask: usize = local_controls.iter().map(|&c| 1usize << c).sum();
            scale_shard_masked(&mut self.inner.state, ctrl_mask, d);
        } else {
            self.apply_global_controlled_1q(&local_controls, target, mat);
        }
    }

    /// Apply a controlled diagonal gate `diag(1, phase)` on the all ones corner
    /// of its qubit set. Covers Cz, controlled phase, and diagonal Mcu with no
    /// communication: a global qubit contributes a constant rank bit, and
    /// local qubits restrict which slice indices receive the phase.
    ///
    /// The residual on local qubits is another controlled phase gate, so it uses
    /// the inner backend kernels.
    fn apply_controlled_phase_dist(&mut self, qubits: &[usize], phase: Complex64) {
        let local = self.local_qubits();
        let mut local_qubits: SmallVec<[usize; 8]> = SmallVec::new();
        for &q in qubits {
            if q < local {
                local_qubits.push(q);
            } else if !self.rank_bit_set(q) {
                // A zero global corner bit makes the gate inactive on this rank.
                return;
            }
        }
        self.apply_local_corner_phase(&local_qubits, phase);
    }

    /// Apply `phase` on the all ones corner through the inner backend.
    fn apply_local_corner_phase(&mut self, local_qubits: &[usize], phase: Complex64) {
        let z = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        match local_qubits.len() {
            0 => scale_shard(&mut self.inner.state, phase),
            1 => self
                .inner
                .apply_1q_matrix(local_qubits[0], &[[one, z], [z, phase]])
                .expect("local diagonal phase"),
            n => {
                let mat = [[one, z], [z, phase]];
                let gate = if n == 2 {
                    Gate::cu(mat)
                } else {
                    Gate::mcu(mat, (n - 1) as u8)
                };
                self.inner
                    .apply(&Instruction::Gate {
                        gate,
                        targets: local_qubits.iter().copied().collect(),
                    })
                    .expect("local controlled phase");
            }
        }
    }

    /// Apply `Rzz(theta)` across any local or global split. Rzz is diagonal,
    /// `phase = exp(-i theta/2)` when the two qubit bits agree and
    /// `exp(i theta/2)` when they differ, so no communication is needed: a
    /// global qubit contributes a constant rank bit to the parity.
    fn apply_rzz_dist(&mut self, q0: usize, q1: usize, theta: f64) {
        let phase_same = Complex64::from_polar(1.0, -theta / 2.0);
        let phase_diff = Complex64::from_polar(1.0, theta / 2.0);
        self.apply_rzz_phases_dist(q0, q1, phase_same, phase_diff);
    }

    /// Apply a parity diagonal two qubit phase. Shared by `Rzz` and
    /// `Parity2q`; it needs no communication.
    fn apply_rzz_phases_dist(
        &mut self,
        q0: usize,
        q1: usize,
        phase_same: Complex64,
        phase_diff: Complex64,
    ) {
        let local = self.local_qubits();

        match (q0 < local, q1 < local) {
            (true, true) => {
                // Both qubits are local, so use the inner diagonal batch kernel.
                use crate::gates::{DiagEntry, DiagonalBatchData};
                let entry = DiagEntry::Parity2q {
                    q0,
                    q1,
                    same: phase_same,
                    diff: phase_diff,
                };
                self.inner
                    .apply(&Instruction::Gate {
                        gate: Gate::DiagonalBatch(Box::new(DiagonalBatchData {
                            entries: vec![entry],
                        })),
                        targets: smallvec![q0, q1],
                    })
                    .expect("local parity diagonal");
            }
            (false, false) => {
                let parity =
                    ((self.rank_bit_set(q0) as usize) ^ (self.rank_bit_set(q1) as usize)) & 1;
                let factor = [phase_same, phase_diff][parity];
                scale_shard(&mut self.inner.state, factor);
            }
            (true, false) | (false, true) => {
                // One global qubit is fixed on this rank. The residual is a
                // diagonal one qubit gate.
                let (local_q, global_q) = if q0 < local { (q0, q1) } else { (q1, q0) };
                let gbit = self.rank_bit_set(global_q) as usize;
                // Local bit 0 uses parity gbit. Local bit 1 uses gbit ^ 1.
                let d0 = [phase_same, phase_diff][gbit];
                let d1 = [phase_same, phase_diff][gbit ^ 1];
                let z = Complex64::new(0.0, 0.0);
                self.inner
                    .apply_1q_matrix(local_q, &[[d0, z], [z, d1]])
                    .expect("local parity residual");
            }
        }
    }

    /// Apply `SWAP(a, b)` across any local or global split.
    ///
    /// Local pairs delegate to the inner kernel. With a global qubit, only the
    /// `|01>` and `|10>` amplitudes move.
    fn apply_swap_dist(&mut self, a: usize, b: usize) {
        let local = self.local_qubits();
        match (a < local, b < local) {
            (true, true) => {
                self.inner
                    .apply(&Instruction::Gate {
                        gate: Gate::Swap,
                        targets: smallvec![a, b],
                    })
                    .expect("local swap");
            }
            (false, false) => self.swap_global_slices(a, b),
            (true, false) | (false, true) => {
                let (local_q, global_q) = if a < local { (a, b) } else { (b, a) };
                self.half_slice_swap(local_q, global_q);
            }
        }
    }

    /// Exchange whole slices between the rank pairs whose bits at global
    /// positions `a` and `b` differ; a rank with equal bits holds fixed points
    /// of the SWAP and skips. The copy is elementwise, so the exchange streams
    /// in `exchange_chunk` tiles.
    fn swap_global_slices(&mut self, a: usize, b: usize) {
        if self.rank_bit_set(a) == self.rank_bit_set(b) {
            return;
        }
        let partner =
            self.context.rank() ^ (1usize << self.global_bit(a)) ^ (1usize << self.global_bit(b));
        let len = self.inner.state.len();
        let chunk = self.exchange_chunk.min(len).max(1);
        self.ensure_recv(chunk);
        let mut off = 0;
        while off < len {
            let end = (off + chunk).min(len);
            self.count_exchange(end - off);
            let recv = &mut self.recv[..end - off];
            self.context
                .comm()
                .sendrecv_c64(partner, &self.inner.state[off..end], recv);
            self.inner.state[off..end].copy_from_slice(recv);
            off = end;
        }
    }

    /// Apply a general 4x4 two qubit unitary across any local or global split.
    ///
    /// `mat` uses basis index `2*b0 + b1`, with `q0` as the high bit. Local
    /// pairs delegate to the inner kernel. One global qubit needs one exchange;
    /// two global qubits gather the rank group that shares both rank bits.
    fn apply_2q_dist(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        let local = self.local_qubits();
        match (q0 < local, q1 < local) {
            (true, true) => self.apply_local_fused_2q(q0, q1, mat),
            (true, false) | (false, true) => self.apply_2q_one_global(q0, q1, mat),
            (false, false) => self.apply_2q_two_global(q0, q1, mat),
        }
    }

    /// Apply a fully local 4x4 gate through the inner backend's tiled kernel.
    fn apply_local_fused_2q(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        self.inner.apply_fused_2q(q0, q1, mat);
    }

    /// One qubit is local and one is global. Exchange with the partner rank,
    /// then recompute each amplitude from the four inputs of the 2x2 block.
    /// Each pair block of `2^(local_q + 1)` amplitudes is self-contained, so the
    /// exchange streams in tiles of whole blocks.
    fn apply_2q_one_global(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        let local = self.local_qubits();
        let (local_q, global_q, global_is_q0) = if q0 < local {
            (q0, q1, false)
        } else {
            (q1, q0, true)
        };
        let partner = self.context.rank() ^ (1usize << self.global_bit(global_q));
        let len = self.inner.state.len();
        let chunk = self.block_chunk(local_q + 1, len);
        self.ensure_recv(chunk);

        let g = self.rank_bit_set(global_q) as usize;
        // Basis index in `mat` is `2*b_q0 + b_q1`.
        let basis = |gbit: usize, lbit: usize| -> usize {
            if global_is_q0 {
                (gbit << 1) | lbit
            } else {
                (lbit << 1) | gbit
            }
        };
        // Columns in input order: own lo, own hi, partner lo, partner hi.
        let cols = [basis(g, 0), basis(g, 1), basis(1 - g, 0), basis(1 - g, 1)];
        let coeffs = |row: usize| cols.map(|c| mat[row][c]);
        let (m_lo, m_hi) = (coeffs(basis(g, 0)), coeffs(basis(g, 1)));
        let mut off = 0;
        while off < len {
            let end = (off + chunk).min(len);
            self.count_exchange(end - off);
            let recv = &mut self.recv[..end - off];
            self.context
                .comm()
                .sendrecv_c64(partner, &self.inner.state[off..end], recv);
            // Both outputs of a pair read both inputs, so each pair is finished
            // before either slot is written.
            for_each_pair_tile(
                &mut self.inner.state[off..end],
                recv,
                local_q,
                |lo, hi, rlo, rhi| {
                    for (((s0, s1), &r0), &r1) in
                        lo.iter_mut().zip(hi).zip(rlo.iter()).zip(rhi.iter())
                    {
                        let (own0, own1) = (*s0, *s1);
                        *s0 = m_lo[0] * own0 + m_lo[1] * own1 + m_lo[2] * r0 + m_lo[3] * r1;
                        *s1 = m_hi[0] * own0 + m_hi[1] * own1 + m_hi[2] * r0 + m_hi[3] * r1;
                    }
                },
            );
            off = end;
        }
    }

    /// Apply a run of two qubit gates that all pair a local qubit with the same
    /// global qubit. One exchange serves the whole run: after it this rank holds
    /// both halves of the pair subspace, so each entry updates the local slice
    /// and the mirrored partner copy together, exactly as the partner does with
    /// the roles flipped. Sums run in canonical basis order so both ranks
    /// produce bit-identical copies and stay in lockstep without further
    /// communication. Every entry's pair block fits inside a tile of whole
    /// blocks of the widest entry, so the exchange streams in such tiles and
    /// the run is applied tile by tile.
    fn apply_2q_run_one_global(
        &mut self,
        global_q: usize,
        entries: &[(usize, usize, [[Complex64; 4]; 4])],
    ) {
        let partner = self.context.rank() ^ (1usize << self.global_bit(global_q));
        let local_of = |q0: usize, q1: usize| if q0 == global_q { q1 } else { q0 };
        let widest = entries
            .iter()
            .map(|&(q0, q1, _)| local_of(q0, q1))
            .max()
            .expect("a run has at least one entry");
        let len = self.inner.state.len();
        let chunk = self.block_chunk(widest + 1, len);
        self.ensure_recv(chunk);

        let g = self.rank_bit_set(global_q) as usize;
        let zero = Complex64::new(0.0, 0.0);
        let mut off = 0;
        while off < len {
            let end = (off + chunk).min(len);
            self.count_exchange(end - off);
            let recv = &mut self.recv[..end - off];
            self.context
                .comm()
                .sendrecv_c64(partner, &self.inner.state[off..end], recv);
            for &(q0, q1, ref mat) in entries {
                let (local_q, global_is_q0) = (local_of(q0, q1), q0 == global_q);
                let basis = |gbit: usize, lbit: usize| -> usize {
                    if global_is_q0 {
                        (gbit << 1) | lbit
                    } else {
                        (lbit << 1) | gbit
                    }
                };
                // Slot order: state lo, state hi, recv lo, recv hi.
                let slots = [basis(g, 0), basis(g, 1), basis(1 - g, 0), basis(1 - g, 1)];
                for_each_pair_tile(
                    &mut self.inner.state[off..end],
                    recv,
                    local_q,
                    |lo, hi, rlo, rhi| {
                        for (((s0, s1), r0), r1) in lo.iter_mut().zip(hi).zip(rlo).zip(rhi) {
                            let mut by_col = [zero; 4];
                            by_col[slots[0]] = *s0;
                            by_col[slots[1]] = *s1;
                            by_col[slots[2]] = *r0;
                            by_col[slots[3]] = *r1;
                            let mut outs = [zero; 4];
                            for (out, &row) in outs.iter_mut().zip(slots.iter()) {
                                for (c, &amp) in by_col.iter().enumerate() {
                                    *out += mat[row][c] * amp;
                                }
                            }
                            *s0 = outs[0];
                            *s1 = outs[1];
                            *r0 = outs[2];
                            *r1 = outs[3];
                        }
                    },
                );
            }
            off = end;
        }
    }

    /// Both qubits global. The four `(q0, q1)` slices live on four ranks that
    /// share every other rank bit. Two pairwise exchanges of one slice each
    /// replace a gather of the other three. Write `a(c0, c1)` for the slice on
    /// the rank whose bits are `(c0, c1)`, `b(c0, c1) = 2 c0 + c1` for the basis
    /// index, `m = mat[b(g0, g1)]` for this rank's row, and `m' = mat[b(g0, 1 - g1)]`
    /// for the row of the partner across `q1`:
    ///
    /// 1. Exchange with `rank ^ bit(q0)`: send `a(g0, g1)`, receive `a(1 - g0, g1)`.
    ///    Form in place `p = m[b(g0, g1)] a(g0, g1) + m[b(1 - g0, g1)] a(1 - g0, g1)`
    ///    and, into the pack buffer, the partial the `q1` partner needs,
    ///    `u = m'[b(g0, g1)] a(g0, g1) + m'[b(1 - g0, g1)] a(1 - g0, g1)`.
    /// 2. Exchange with `rank ^ bit(q1)`: send `u`, receive the mirrored partial
    ///    `t = m[b(g0, 1 - g1)] a(g0, 1 - g1) + m[b(1 - g0, 1 - g1)] a(1 - g0, 1 - g1)`.
    ///    The output row is `p + t`.
    ///
    /// Both steps are elementwise, so each `exchange_chunk` tile runs through
    /// both exchanges before the next tile starts; the transient buffers are
    /// two tiles rather than three slices. Volume is `2 len` amplitudes per rank
    /// against `3 len` for the gather. The association `(own + q0 partner) +
    /// (q1 partner + diagonal partner)` differs from the gather's left to right
    /// sum by rounding only.
    fn apply_2q_two_global(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        let rank = self.context.rank();
        let bit0 = 1usize << self.global_bit(q0);
        let bit1 = 1usize << self.global_bit(q1);
        let g0 = (rank & bit0 != 0) as usize;
        let g1 = (rank & bit1 != 0) as usize;
        let basis = |c0: usize, c1: usize| (c0 << 1) | c1;
        let (row, forward_row) = (basis(g0, g1), basis(g0, 1 - g1));
        let (own, across) = (basis(g0, g1), basis(1 - g0, g1));
        let keep = [mat[row][own], mat[row][across]];
        let forward = [mat[forward_row][own], mat[forward_row][across]];

        let len = self.inner.state.len();
        let chunk = self.exchange_chunk.min(len).max(1);
        self.ensure_recv(chunk);
        self.ensure_pack(chunk);
        let mut off = 0;
        while off < len {
            let end = (off + chunk).min(len);
            let count = end - off;
            self.count_exchange(count);
            self.context.comm().sendrecv_c64(
                rank ^ bit0,
                &self.inner.state[off..end],
                &mut self.recv[..count],
            );
            butterfly_shard(
                &mut self.inner.state[off..end],
                &self.recv[..count],
                &mut self.pack[..count],
                keep,
                forward,
            );
            self.count_exchange(count);
            self.context.comm().sendrecv_c64(
                rank ^ bit1,
                &self.pack[..count],
                &mut self.recv[..count],
            );
            add_shard(&mut self.inner.state[off..end], &self.recv[..count]);
            off = end;
        }
    }

    /// Dispatch a gate that spans at least one global qubit.
    fn apply_global_multi_qubit(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Cx => {
                self.apply_controlled_dist(&targets[..1], targets[1], Gate::X.matrix_2x2());
                Ok(())
            }
            Gate::Cz => {
                self.apply_controlled_phase_dist(
                    &[targets[0], targets[1]],
                    -Complex64::new(1.0, 0.0),
                );
                Ok(())
            }
            Gate::Swap => {
                self.apply_swap_dist(targets[0], targets[1]);
                Ok(())
            }
            Gate::Rzz(theta) => {
                self.apply_rzz_dist(targets[0], targets[1], *theta);
                Ok(())
            }
            Gate::Cu(mat) => {
                if let Some(phase) = gate.controlled_phase() {
                    self.apply_controlled_phase_dist(&[targets[0], targets[1]], phase);
                } else {
                    self.apply_controlled_dist(&targets[..1], targets[1], **mat);
                }
                Ok(())
            }
            Gate::Mcu(data) => {
                let num_ctrl = data.num_controls as usize;
                let controls = &targets[..num_ctrl];
                let target = targets[num_ctrl];
                if let Some(phase) = gate.controlled_phase() {
                    let mut corner: Vec<usize> = controls.to_vec();
                    corner.push(target);
                    self.apply_controlled_phase_dist(&corner, phase);
                } else {
                    self.apply_controlled_dist(controls, target, data.mat);
                }
                Ok(())
            }
            Gate::Fused2q(mat) => {
                self.apply_2q_dist(targets[0], targets[1], mat);
                Ok(())
            }
            Gate::Multi2q(data) => {
                // Consecutive entries sharing one global qubit have the same
                // partner (a CNOT star onto one qubit); one exchange serves
                // the whole run.
                let local = self.local_qubits();
                let one_global_on = |entry: &(usize, usize, [[Complex64; 4]; 4])| {
                    let (q0, q1, _) = *entry;
                    match (q0 < local, q1 < local) {
                        (true, false) => Some(q1),
                        (false, true) => Some(q0),
                        _ => None,
                    }
                };
                let mut i = 0;
                while i < data.gates.len() {
                    let Some(g) = one_global_on(&data.gates[i]) else {
                        let (q0, q1, ref mat) = data.gates[i];
                        self.apply_2q_dist(q0, q1, mat);
                        i += 1;
                        continue;
                    };
                    let mut end = i + 1;
                    while end < data.gates.len() && one_global_on(&data.gates[end]) == Some(g) {
                        end += 1;
                    }
                    if end - i == 1 {
                        let (q0, q1, ref mat) = data.gates[i];
                        self.apply_2q_one_global(q0, q1, mat);
                    } else {
                        self.apply_2q_run_one_global(g, &data.gates[i..end]);
                    }
                    i = end;
                }
                Ok(())
            }
            Gate::MultiFused(data) => {
                for &(q, ref mat) in &data.gates {
                    if q < self.local_qubits() {
                        self.inner.apply_1q_matrix(q, mat).expect("local 1q matrix");
                    } else if is_diagonal_2x2(mat) {
                        self.apply_global_diagonal_1q(q, mat[0][0], mat[1][1]);
                    } else {
                        self.apply_global_1q(q, *mat);
                    }
                }
                Ok(())
            }
            Gate::BatchPhase(data) => {
                let control = targets[0];
                for &(target, phase) in &data.phases {
                    self.apply_controlled_phase_dist(&[control, target], phase);
                }
                Ok(())
            }
            Gate::BatchRzz(data) => {
                for &(q0, q1, theta) in &data.edges {
                    self.apply_rzz_dist(q0, q1, theta);
                }
                Ok(())
            }
            Gate::DiagonalBatch(data) => {
                for entry in &data.entries {
                    self.apply_diag_entry_dist(entry);
                }
                Ok(())
            }
            _ => Err(self.unsupported("gate spanning a global qubit")),
        }
    }

    /// Apply a single [`DiagEntry`] across any local or global split.
    fn apply_diag_entry_dist(&mut self, entry: &crate::gates::DiagEntry) {
        use crate::gates::DiagEntry;
        match *entry {
            DiagEntry::Phase1q { qubit, d0, d1 } => {
                if qubit < self.local_qubits() {
                    self.inner
                        .apply_1q_matrix(
                            qubit,
                            &[
                                [d0, Complex64::new(0.0, 0.0)],
                                [Complex64::new(0.0, 0.0), d1],
                            ],
                        )
                        .expect("local diagonal 1q");
                } else {
                    self.apply_global_diagonal_1q(qubit, d0, d1);
                }
            }
            DiagEntry::Phase2q { q0, q1, phase } => {
                self.apply_controlled_phase_dist(&[q0, q1], phase);
            }
            DiagEntry::Parity2q {
                q0, q1, same, diff, ..
            } => {
                // These are the parity phases for Rzz(theta).
                self.apply_rzz_phases_dist(q0, q1, same, diff);
            }
        }
    }

    /// Total scaled weight of the `qubit == outcome` subspace across ranks.
    fn prob_outcome_global(&self, qubit: usize, outcome: bool) -> f64 {
        let norm_sq = self.inner.pending_norm * self.inner.pending_norm;
        let local_prob = if qubit < self.local_qubits() {
            half_norm_sqr(&self.inner.state, qubit, outcome)
        } else if self.rank_bit_set(qubit) == outcome {
            crate::backend::state_norm_sqr(&self.inner.state)
        } else {
            0.0
        };
        self.context.comm().allreduce_sum_f64(local_prob) * norm_sq
    }

    /// Total weight of the `qubit == 1` subspace across all ranks. Used by
    /// measurement and as `P(qubit = 1)`.
    fn prob_one_global(&self, qubit: usize) -> f64 {
        self.prob_outcome_global(qubit, true)
    }

    /// Measure `qubit`, collapse the state, and record the bit. Deterministic
    /// across ranks: the outcome is drawn from the lockstep `meas_rng` against an
    /// `Allreduce`d probability, so every rank collapses to the same branch.
    fn measure_dist(&mut self, qubit: usize, classical_bit: usize) {
        let qubit = self.physical_qubit(qubit);
        let prob_one = self.prob_one_global(qubit);
        let outcome = self.meas_rng.random::<f64>() < prob_one;
        self.inner.classical_bits[classical_bit] = outcome;
        self.collapse(qubit, outcome);
        self.inner.pending_norm *= measurement_inv_norm(outcome, prob_one);
    }

    /// Physical position of a circuit qubit. Identity before `init` runs the
    /// map setup or while no relabeling has occurred.
    #[inline]
    fn physical_qubit(&self, qubit: usize) -> usize {
        if self.map_identity {
            qubit
        } else {
            self.qubit_map[qubit]
        }
    }

    /// Sample `num_shots` computational basis indices in circuit qubit order
    /// without gathering the dense state or probability vector on any rank.
    ///
    /// Relabeled qubits are first restored to their circuit positions with
    /// bounded exchanges, so each rank owns a contiguous slice in circuit
    /// order. Each rank then builds a cumulative distribution for its local
    /// slice. One gather shares a single mass value from each rank. Every rank
    /// assigns each shot to an owning rank from the same seeded draw stream, so
    /// every rank knows the owner sequence. Each owner samples its local
    /// distribution for its shots, one variable-count gather concatenates the
    /// owned indices in rank order, and each rank scatters them back into shot
    /// order. Buffers scale with the rank count and shot count, not the global
    /// state size.
    ///
    /// Collective: every rank must call this with identical `num_shots` and
    /// `seed`. The result is identical on every rank and reproduces the dense
    /// sampling path draw for draw, independent of the rank count, except
    /// when accumulated rounding differences move a draw across an interval
    /// edge in the cumulative distribution.
    pub fn sample_state_indices(&mut self, num_shots: usize, seed: u64) -> Result<Vec<u64>> {
        if num_shots == 0 {
            return Ok(Vec::new());
        }
        self.restore_identity_map();
        debug_assert!(self.map_identity);

        // The rank-local CDF is a working buffer half the size of the slice
        // this rank already holds, so the dense output cap does not gate it.
        let mut local_cdf = self.inner.host_probability_vector();
        let mut acc = 0.0f64;
        for p in &mut local_cdf {
            acc += *p;
            *p = acc;
        }

        let masses = self.context.comm().allgather_f64(&[acc]);
        let mut rank_cdf = Vec::with_capacity(masses.len());
        let mut total = 0.0f64;
        for &m in &masses {
            total += m;
            rank_cdf.push(total);
        }
        if let Some(last) = rank_cdf.last_mut() {
            *last = 1.0;
        }

        let rank = self.context.rank();
        let size = self.context.size();
        let local_qubits = self.local_qubits();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut owners = Vec::with_capacity(num_shots);
        let mut counts = vec![0usize; size];
        let mut owned = Vec::new();
        for _ in 0..num_shots {
            let r: f64 = rng.random();
            // First rank whose cumulative mass reaches r. The strict
            // comparison matches the dense binary search at exact boundary
            // hits and never selects an empty interval.
            let owner = rank_cdf.partition_point(|&c| c < r);
            owners.push(owner);
            counts[owner] += 1;
            if owner != rank {
                continue;
            }
            let residual = if owner == 0 {
                r
            } else {
                r - rank_cdf[owner - 1]
            };
            let local_idx = crate::sim::shots::sample_from_cdf(&local_cdf, residual);
            owned.push(((rank as u64) << local_qubits) | local_idx as u64);
        }

        let gathered = self.context.comm().allgatherv_u64(&owned, &counts);
        let mut next = vec![0usize; size];
        for r in 1..size {
            next[r] = next[r - 1] + counts[r - 1];
        }
        let indices = owners
            .iter()
            .map(|&owner| {
                let i = next[owner];
                next[owner] += 1;
                gathered[i]
            })
            .collect();
        Ok(indices)
    }

    /// Zero the amplitudes inconsistent with `qubit == outcome`.
    fn collapse(&mut self, qubit: usize, outcome: bool) {
        if qubit < self.local_qubits() {
            fn dropped(block: &mut [Complex64], half: usize, outcome: bool) -> &mut [Complex64] {
                let (lo, hi) = block.split_at_mut(half);
                if outcome { lo } else { hi }
            }
            let half = 1usize << qubit;
            let block_size = half << 1;
            #[cfg(feature = "parallel")]
            if self.inner.state.len() >= PAR_SHARD_LEN {
                if self.inner.state.len() / block_size >= 4 {
                    self.inner
                        .state
                        .par_chunks_mut(block_size)
                        .with_min_len(chunk_min_len(block_size))
                        .for_each(|block| simd::zero_slice(dropped(block, half, outcome)));
                } else {
                    for block in self.inner.state.chunks_mut(block_size) {
                        dropped(block, half, outcome)
                            .par_chunks_mut(MIN_PAR_ELEMS)
                            .for_each(simd::zero_slice);
                    }
                }
                return;
            }
            for block in self.inner.state.chunks_mut(block_size) {
                simd::zero_slice(dropped(block, half, outcome));
            }
        } else if self.rank_bit_set(qubit) != outcome {
            // This rank holds the eliminated branch entirely.
            zero_shard(&mut self.inner.state);
        }
    }

    /// Reset `qubit` to `|0>` as one trajectory of the reset channel: sample
    /// the outcome, collapse onto it, then apply X when it is 1. The draw
    /// comes from the rank-replicated measurement stream, so every rank
    /// selects the same branch.
    fn reset_dist(&mut self, qubit: usize) -> Result<()> {
        let physical = self.physical_qubit(qubit);
        let prob_one = self.prob_one_global(physical);
        let outcome = self.meas_rng.random::<f64>() < prob_one;
        self.collapse(physical, outcome);
        self.inner.pending_norm *= measurement_inv_norm(outcome, prob_one);
        if outcome {
            self.apply_gate(&Gate::X, &[qubit])?;
        }
        Ok(())
    }

    /// Route a gate to the local fast path or the distributed paths.
    ///
    /// With relabeling on, SWAP becomes a map update, and qubits that need
    /// non-diagonal application are moved into local positions first, so the
    /// per-gate exchange paths below only fire when no eviction victim exists
    /// or relabeling is disabled.
    fn apply_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        if self.global_qubits == 0 {
            self.inner.dispatch_gate(gate, targets);
            return Ok(());
        }
        if self.relabel {
            self.touch_instruction(gate, targets);
            if matches!(gate, Gate::Swap) {
                self.swap_circuit_qubits(targets[0], targets[1]);
                return Ok(());
            }
            let req = required_local_qubits(gate, targets);
            if !req.is_empty() {
                self.make_local(&req);
            }
        }
        if matches!(gate, Gate::QftBlock { .. }) && !self.map_identity {
            return Err(self.unsupported("QftBlock with a permuted qubit map"));
        }
        let (pgate, ptargets) = self.to_physical(gate, targets);
        if self.instruction_qubits_local(&pgate, &ptargets) {
            self.inner.dispatch_gate(pgate.as_ref(), &ptargets);
            return Ok(());
        }
        let pgate = pgate.as_ref();
        if pgate.num_qubits() == 1 {
            let target = ptargets[0];
            let mat = pgate.matrix_2x2();
            if pgate.is_diagonal_1q() {
                self.apply_global_diagonal_1q(target, mat[0][0], mat[1][1]);
            } else {
                self.apply_global_1q(target, mat);
            }
            return Ok(());
        }
        self.apply_global_multi_qubit(pgate, &ptargets)
    }

    fn unsupported(&self, operation: &str) -> PrismError {
        PrismError::BackendUnsupported {
            backend: BACKEND_NAME.to_string(),
            operation: operation.to_string(),
        }
    }
}

impl Backend for DistributedStatevectorBackend {
    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::Distributed
    }

    fn supports_fused_gates(&self) -> bool {
        // Fusion runs in every mode. Fully local fused gates dispatch to the
        // inner backend's tiled SIMD kernels; fused or batched gates that span a
        // rank bit are decomposed into primitives at apply time.
        true
    }

    fn supports_qft_block(&self) -> bool {
        self.is_single_rank() && self.inner.supports_qft_block()
    }

    fn supports_pauli_rotation(&self) -> bool {
        self.is_single_rank() && self.inner.supports_pauli_rotation()
    }

    fn apply_instructions(&mut self, instructions: &[Instruction]) -> Result<()> {
        if std::mem::take(&mut self.circuit_check_pending) && self.context.size() > 1 {
            self.check_circuit_agreement(instructions)?;
        }
        for instruction in instructions {
            self.apply(instruction)?;
        }
        Ok(())
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        let local_qubits = self.prepare_shard(num_qubits, num_classical_bits)?;
        self.inner.init(local_qubits, num_classical_bits)?;

        // inner.init seeds index 0 on every rank; only rank 0 owns |0...0>.
        if self.context.rank() != 0 {
            if let Some(amp) = self.inner.state.get_mut(0) {
                *amp = Complex64::new(0.0, 0.0);
            }
        }
        Ok(())
    }

    fn supports_initial_state(&self) -> bool {
        true
    }

    /// Load this rank's shard from the full `2^n` vector.
    ///
    /// Every rank receives the whole vector and keeps the `2^(n - p)` amplitudes
    /// from `rank * 2^(n - p)`, the identity layout `init` establishes; a map
    /// left permuted by an earlier relabeled run is reset, not written into.
    /// Collective: every rank must call it with an identical vector.
    fn init_from_amplitudes(
        &mut self,
        amplitudes: Vec<Complex64>,
        num_classical_bits: usize,
    ) -> Result<()> {
        crate::backend::validate_initial_amplitudes(&amplitudes)?;
        let num_qubits = amplitudes.len().trailing_zeros() as usize;
        let local_qubits = self.prepare_shard(num_qubits, num_classical_bits)?;
        if self.is_single_rank() {
            return self.inner.init_from_state(amplitudes, num_classical_bits);
        }
        self.inner.init(local_qubits, num_classical_bits)?;
        let len = 1usize << local_qubits;
        let start = self.context.rank() * len;
        self.inner
            .state
            .copy_from_slice(&amplitudes[start..start + len]);
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            // Measurement routes through the distributed path even at a single
            // rank, so `meas_rng` is the sole measurement RNG and one seed
            // draws one outcome stream. Outcomes then agree across rank counts
            // except where the `Allreduce` association order moves a summed
            // probability across the drawn value, the caveat
            // `sample_state_indices` documents for the same reason.
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.measure_dist(*qubit, *classical_bit);
                Ok(())
            }
            Instruction::Reset { qubit } => self.reset_dist(*qubit),
            Instruction::Barrier { .. } => Ok(()),
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(self.inner.classical_results()) {
                    self.apply_gate(gate, targets)
                } else {
                    Ok(())
                }
            }
            Instruction::Gate { gate, targets } => self.apply_gate(gate, targets),
            // Every rank draws measurement outcomes from the same seeded RNG
            // against an `Allreduce`d probability, so every rank holds the same
            // classical bits and takes the same branch without a consensus
            // exchange.
            Instruction::Region(region) => self.apply_region(region),
        }
    }

    fn classical_results(&self) -> &[bool] {
        self.inner.classical_results()
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let local = self.inner.probabilities()?;
        if self.global_qubits == 0 {
            return Ok(local);
        }
        dense_probability_len(BACKEND_NAME, self.num_qubits)?;
        let gathered = self.context.comm().allgather_f64(&local);
        Ok(self.unpermuted(gathered))
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let local = self.inner.export_statevector()?;
        if self.global_qubits == 0 {
            return Ok(local);
        }
        dense_statevector_len(BACKEND_NAME, "statevector export", self.num_qubits)?;
        let gathered = self.context.comm().allgather_c64(&local);
        Ok(self.unpermuted(gathered))
    }

    fn qubit_probability(&self, qubit: usize) -> Result<f64> {
        Ok(self.prob_one_global(self.physical_qubit(qubit)))
    }

    fn supports_native_sampling(&self) -> bool {
        true
    }

    /// Trait-level entry to [`DistributedStatevectorBackend::sample_state_indices`],
    /// so a caller holding a `dyn Backend` gets the same rank-local draw the
    /// shot route takes instead of falling back to the dense vector.
    ///
    /// Collective: every rank must call it with identical `num_shots` and
    /// `seed`.
    fn sample_basis_states(&mut self, num_shots: usize, seed: u64) -> Result<BasisSamples> {
        let indices = self.sample_state_indices(num_shots, seed)?;
        let mut samples = BasisSamples::new(num_shots, self.num_qubits);
        for (shot, &index) in indices.iter().enumerate() {
            samples.set_index(shot, index as usize);
        }
        Ok(samples)
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// Evaluate each observable on the sharded state with no dense gather.
    ///
    /// A Z factor on a rank bit is a constant sign for the whole slice, so an
    /// observable whose X and Y factors are all local costs one `Allreduce` and
    /// no transfer. X and Y factors on rank bits displace the bra by the same
    /// rank offset for every amplitude, so however many there are they name one
    /// partner rank, and one slice exchange covers them. That is the direct
    /// route rather than a relabel because relabeling mutates the state, which
    /// a `&self` query cannot do.
    ///
    /// Collective: every rank must call it with identical observables.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        let comm = self.context.comm();
        let norm = comm.allreduce_sum_f64(crate::backend::state_norm_sqr(&self.inner.state));
        let local_qubits = self.local_qubits();
        let local_mask = (1usize << local_qubits) - 1;
        let mut recv: Vec<Complex64> = Vec::new();

        let mut values = Vec::with_capacity(observables.len());
        for observable in observables {
            let (xmask, zmask, num_y) = crate::sim::pauli_masks(observable, self.num_qubits)?;
            let xphys = self.to_physical_mask(xmask);
            let zphys = self.to_physical_mask(zmask);
            let partner_bits = xphys >> local_qubits;
            if partner_bits != 0 {
                recv.resize(self.inner.state.len(), Complex64::new(0.0, 0.0));
                comm.sendrecv_c64(
                    self.context.rank() ^ partner_bits,
                    &self.inner.state,
                    &mut recv,
                );
            }
            let bra: &[Complex64] = if partner_bits == 0 {
                &self.inner.state
            } else {
                &recv
            };
            let sandwich = crate::sim::pauli_sandwich(
                bra,
                &self.inner.state,
                xphys & local_mask,
                zphys & local_mask,
                num_y,
            );
            let rank_parity = (self.context.rank() & (zphys >> local_qubits)).count_ones() & 1;
            let signed = if rank_parity == 1 {
                -sandwich.re
            } else {
                sandwich.re
            };
            // The total is real, so summing the per-rank real parts loses
            // nothing even where a rank's own term is not.
            let total = comm.allreduce_sum_f64(signed);
            values.push(if norm == 0.0 { 0.0 } else { total / norm });
        }
        Ok(values)
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.reset_dist(qubit)
    }

    /// Apply a 2x2 matrix to one circuit qubit across the rank split, on the same
    /// route `apply_gate` takes for a one-qubit gate: relabel a non-diagonal
    /// target into a local position when a victim exists, apply locally when the
    /// physical position is local, otherwise exchange with the partner rank.
    ///
    /// Collective when the target is global, so every rank must call it with the
    /// same qubit.
    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        if self.global_qubits == 0 {
            return self.inner.apply_1q_matrix(qubit, matrix);
        }

        let diagonal = is_diagonal_2x2(matrix);
        if self.relabel {
            self.tick += 1;
            self.last_used[qubit] = self.tick;
            if !diagonal {
                self.make_local(&[qubit]);
            }
        }

        let target = self.physical_qubit(qubit);
        if target < self.local_qubits() {
            return self.inner.apply_1q_matrix(target, matrix);
        }
        if diagonal {
            self.apply_global_diagonal_1q(target, matrix[0][0], matrix[1][1]);
        } else {
            self.apply_global_1q(target, *matrix);
        }
        Ok(())
    }
}

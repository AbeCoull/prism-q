//! Dense kernels over factored sub-state slices: branch norms, collapse and
//! extraction, Kronecker merges, and the batch-phase kernels.

use num_complex::Complex64;

use crate::backend::{is_phase_one, simd};

#[cfg(feature = "parallel")]
use crate::backend::{
    MIN_PAR_ELEMS, PARALLEL_THRESHOLD_QUBITS, chunk_min_len as par_chunk_min_len,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Shortest contiguous run a split or single-qubit merge copies under Rayon.
#[cfg(feature = "parallel")]
const MIN_PAR_RUN: usize = 1 << (PARALLEL_THRESHOLD_QUBITS - 1);

/// Probability that bit `local` of a normalized sub-state reads one.
pub(super) fn branch_norm_sqr(state: &[Complex64], local: usize) -> f64 {
    let half = 1usize << local;

    #[cfg(feature = "parallel")]
    if state.len() >= 1 << PARALLEL_THRESHOLD_QUBITS {
        if half >= MIN_PAR_ELEMS {
            let chunks_per_half = half / MIN_PAR_ELEMS;
            return state
                .par_chunks(MIN_PAR_ELEMS)
                .enumerate()
                .filter(|(c, _)| (c / chunks_per_half) & 1 == 1)
                .map(|(_, chunk)| simd::norm_sqr_sum(chunk))
                .sum();
        }
        return state
            .par_chunks((half << 1).max(MIN_PAR_ELEMS))
            .map(|chunk| branch_norm_sqr_seq(chunk, half))
            .sum();
    }
    branch_norm_sqr_seq(state, half)
}

#[inline(always)]
fn branch_norm_sqr_seq(state: &[Complex64], half: usize) -> f64 {
    if half < 4 {
        return state
            .iter()
            .enumerate()
            .filter(|(i, _)| i & half != 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();
    }
    state
        .chunks_exact(half << 1)
        .map(|block| simd::norm_sqr_sum(&block[half..]))
        .sum()
}

/// Scale the `outcome` branch of bit `local` by `scale` onto the branch
/// `lands_on_one` names and zero the other.
pub(super) fn collapse_in_place(
    state: &mut [Complex64],
    local: usize,
    outcome: bool,
    lands_on_one: bool,
    scale: f64,
) {
    let half = 1usize << local;
    let factor = Complex64::new(scale, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    for block in state.chunks_exact_mut(half << 1) {
        let (lo, hi) = block.split_at_mut(half);
        match (outcome, lands_on_one) {
            (false, _) => {
                simd::scale_complex_slice(lo, factor);
                hi.fill(zero);
            }
            (true, true) => {
                simd::scale_complex_slice(hi, factor);
                lo.fill(zero);
            }
            (true, false) => {
                simd::scale_complex_to_slice(lo, hi, factor);
                hi.fill(zero);
            }
        }
    }
}

/// Keep the amplitudes whose bit `local` equals `outcome`, scaled by `scale`,
/// compacted in place into a sub-state one qubit narrower. Bits above `local`
/// shift down by one; capacity is kept for a later merge to grow back into.
pub(super) fn extract_branch(state: &mut Vec<Complex64>, local: usize, outcome: bool, scale: f64) {
    let half = 1usize << local;
    let offset = if outcome { half } else { 0 };
    let len = state.len() >> 1;

    if half < 4 {
        let low = half - 1;
        for j in 0..len {
            state[j] = state[((j & !low) << 1) | offset | (j & low)] * scale;
        }
    } else {
        let factor = Complex64::new(scale, 0.0);
        // Run `c` moves from `2c * half + offset` down to `c * half`. Past the
        // first run the two ranges are disjoint, and ascending order reads
        // every run before anything lands on it.
        for c in 0..len / half {
            let src = 2 * c * half + offset;
            let dst = c * half;
            if src == dst {
                scale_run_in_place(&mut state[..half], factor);
            } else {
                let (lo, hi) = state.split_at_mut(src);
                scale_run(&mut lo[dst..dst + half], &hi[..half], factor);
            }
        }
    }
    state.truncate(len);
}

/// Tensor a one-qubit state into bit `pos` of `state` in place, shifting the
/// bits at and above `pos` up by one.
pub(super) fn insert_qubit(state: &mut Vec<Complex64>, pos: usize, single: [Complex64; 2]) {
    let len = state.len();
    let half = 1usize << pos;
    state.resize(len << 1, Complex64::new(0.0, 0.0));

    if half < 4 {
        let low = half - 1;
        // Descending: index `j` writes at or above `j`, so no unread source
        // is overwritten.
        for j in (0..len).rev() {
            let amp = state[j];
            let base = ((j & !low) << 1) | (j & low);
            state[base | half] = amp * single[1];
            state[base] = amp * single[0];
        }
        return;
    }
    for c in (0..len / half).rev() {
        if c == 0 {
            let (lo, hi) = state.split_at_mut(half);
            scale_run(&mut hi[..half], lo, single[1]);
            scale_run_in_place(lo, single[0]);
        } else {
            let (lo, hi) = state.split_at_mut(2 * c * half);
            let src = &lo[c * half..(c + 1) * half];
            let (dst0, dst1) = hi[..2 * half].split_at_mut(half);
            scale_run(dst0, src, single[0]);
            scale_run(dst1, src, single[1]);
        }
    }
}

#[inline(always)]
fn scale_run(dst: &mut [Complex64], src: &[Complex64], factor: Complex64) {
    #[cfg(feature = "parallel")]
    if dst.len() >= MIN_PAR_RUN {
        dst.par_chunks_mut(MIN_PAR_ELEMS)
            .zip(src.par_chunks(MIN_PAR_ELEMS))
            .for_each(|(d, s)| simd::scale_complex_to_slice(d, s, factor));
        return;
    }
    simd::scale_complex_to_slice(dst, src, factor);
}

#[inline(always)]
fn scale_run_in_place(run: &mut [Complex64], factor: Complex64) {
    #[cfg(feature = "parallel")]
    if run.len() >= MIN_PAR_RUN {
        run.par_chunks_mut(MIN_PAR_ELEMS)
            .for_each(|chunk| simd::scale_complex_slice(chunk, factor));
        return;
    }
    simd::scale_complex_slice(run, factor);
}

/// Kronecker product where `low_state` occupies the low `low_n` bits of the
/// merged index and `high_state` occupies the upper `high_n` bits.
///
/// `merged[h * low_dim + l] = low_state[l] * high_state[h]`. Inner loop is a
/// contiguous SIMD scaled copy, parallelized across `h` for large merges.
pub(super) fn kron_low_high(
    low_state: &[Complex64],
    high_state: &[Complex64],
    low_n: usize,
    high_n: usize,
) -> Vec<Complex64> {
    let low_dim = 1usize << low_n;
    let high_dim = 1usize << high_n;
    let total_dim = low_dim * high_dim;
    let mut merged = vec![Complex64::new(0.0, 0.0); total_dim];

    #[cfg(feature = "parallel")]
    if (low_n + high_n) >= PARALLEL_THRESHOLD_QUBITS {
        merged
            .par_chunks_mut(low_dim)
            .with_min_len(par_chunk_min_len(low_dim))
            .enumerate()
            .for_each(|(h, chunk)| {
                simd::scale_complex_to_slice(chunk, low_state, high_state[h]);
            });
        return merged;
    }

    for (h, chunk) in merged.chunks_mut(low_dim).enumerate() {
        simd::scale_complex_to_slice(chunk, low_state, high_state[h]);
    }
    merged
}

/// General Kronecker product where dst and src bits are interleaved across
/// the merged index. Per-element scatter, no SIMD inner loop.
pub(super) fn kron_scatter(
    dst_state: &[Complex64],
    src_state: &[Complex64],
    dst_bit_positions: &[usize],
    src_bit_positions: &[usize],
    total_dim: usize,
) -> Vec<Complex64> {
    let mut merged = vec![Complex64::new(0.0, 0.0); total_dim];

    for (merged_idx, amp) in merged.iter_mut().enumerate() {
        let mut dst_local = 0usize;
        for (local_bit, &merged_bit) in dst_bit_positions.iter().enumerate() {
            dst_local |= ((merged_idx >> merged_bit) & 1) << local_bit;
        }
        let mut src_local = 0usize;
        for (local_bit, &merged_bit) in src_bit_positions.iter().enumerate() {
            src_local |= ((merged_idx >> merged_bit) & 1) << local_bit;
        }
        *amp = dst_state[dst_local] * src_state[src_local];
    }
    merged
}

/// Scalar batch-phase kernel for sub-state slices.
///
/// For sub-states in the factored backend (typically < 14 qubits), the
/// per-element phase accumulation is fast enough without a LUT.
#[inline(always)]
pub(super) fn apply_batch_phase_seq(
    state: &mut [Complex64],
    num_qubits: usize,
    control: usize,
    phases: &[(usize, Complex64)],
) {
    let ctrl_mask = 1usize << control;
    let one = Complex64::new(1.0, 0.0);
    let n = 1usize << num_qubits;

    for (i, amp) in state.iter_mut().enumerate().take(n) {
        if (i & ctrl_mask) == 0 {
            continue;
        }
        let mut combined = one;
        for &(tgt, phase) in phases {
            if (i >> tgt) & 1 != 0 {
                combined *= phase;
            }
        }
        if !is_phase_one(combined) {
            *amp *= combined;
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
pub(super) fn par_apply_batch_phase(
    state: &mut [Complex64],
    control: usize,
    phases: &[(usize, Complex64)],
) {
    let ctrl_half = 1usize << control;
    let block_size = ctrl_half << 1;
    let one = Complex64::new(1.0, 0.0);

    state
        .par_chunks_mut(block_size)
        .with_min_len(par_chunk_min_len(block_size))
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let block_start = chunk_idx * block_size;
            let (_, hi) = chunk.split_at_mut(ctrl_half);

            for (local_i, amp) in hi.iter_mut().enumerate() {
                let i = block_start + ctrl_half + local_i;
                let mut combined = one;
                for &(tgt, phase) in phases {
                    if (i >> tgt) & 1 != 0 {
                        combined *= phase;
                    }
                }
                if !is_phase_one(combined) {
                    *amp *= combined;
                }
            }
        });
}

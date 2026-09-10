//! Reduced density matrix helpers shared by the backends: the width cap, the
//! index tables of a partial trace, the trace normalization, and the trace of
//! a dense amplitude vector.

use num_complex::Complex64;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::statevector::insert_zero_bit;
#[cfg(feature = "parallel")]
use super::{MIN_PAR_ITERS, PARALLEL_THRESHOLD_QUBITS};
use crate::error::Result;

/// Side `2^k` of the reduced density matrix over `k` qubits, once its `4^k`
/// entries are priced as a `2k`-qubit statevector against the dense export
/// cap.
pub(crate) fn reduced_density_side(backend: &str, k: usize) -> Result<usize> {
    let width = 2 * k;
    if width > super::schmidt::export_cap() {
        return Err(super::schmidt::export_cap_exceeded(
            backend,
            format!(
                "reduced density matrix on {k} qubits, which is the size of a statevector for \
                 {width} qubits"
            ),
        ));
    }
    Ok(1usize << k)
}

/// Basis index bits of each row index `t` in `0..2^k`: bit `i` of `t` lands
/// on qubit `subsystem[i]`.
pub(crate) fn row_offsets(subsystem: &[usize]) -> Vec<usize> {
    (0..1usize << subsystem.len())
        .map(|t| {
            subsystem
                .iter()
                .enumerate()
                .fold(0, |off, (i, &q)| off | (((t >> i) & 1) << q))
        })
        .collect()
}

/// Basis index with the traced index `e` spread over the qubits outside
/// `subsystem` and zeros on the subsystem's own bits, `ascending` being the
/// subsystem sorted.
#[inline(always)]
pub(crate) fn traced_base(e: usize, ascending: &[usize]) -> usize {
    ascending
        .iter()
        .fold(e, |base, &q| insert_zero_bit(base, q))
}

/// Scale `rho` so its trace is 1, whatever norm the state it was read from
/// carried.
pub(crate) fn normalize_trace(rho: &mut [Complex64], dim: usize) {
    let trace: f64 = (0..dim).map(|t| rho[t * dim + t].re).sum();
    let scale = 1.0 / trace;
    for entry in rho.iter_mut() {
        *entry *= scale;
    }
}

/// Traced indices per fold job: about the multiply-adds of [`MIN_PAR_ITERS`]
/// two-qubit groups each, and no more than about four jobs per thread, since
/// every job zeroes and later merges its own `dim * dim` accumulator.
#[cfg(feature = "parallel")]
pub(crate) fn fold_min_len(groups: usize, work_per_index: usize) -> usize {
    let jobs = 4 * rayon::current_num_threads();
    (MIN_PAR_ITERS * 16 / work_per_index)
        .max(groups / jobs)
        .max(1)
}

/// Traced indices per result row under which `dense_reduced_density` stripes
/// the rows instead of folding over the traced index. Measured at 20 qubits
/// on 8 threads: at 16 (k = 8) the stripe read 38 ms against the fold's 54,
/// at 256 (k = 6) the fold read 10 ms against the stripe's 15.
#[cfg(feature = "parallel")]
const STRIPE_RATIO: usize = 64;

/// Result rows per stripe job: about four jobs per thread.
#[cfg(feature = "parallel")]
pub(crate) fn stripe_rows(dim: usize) -> usize {
    (dim / (4 * rayon::current_num_threads())).max(1)
}

/// Add the contribution of the traced index at `base` to `rows`, the rows
/// `first..first + rows.len() / dim` of the lower triangle, gathering the
/// amplitudes through `amps` (of length `dim`) up to the last row needed.
#[inline(always)]
fn accumulate_rows(
    state: &[Complex64],
    offsets: &[usize],
    base: usize,
    amps: &mut [Complex64],
    rows: &mut [Complex64],
    first: usize,
) {
    let dim = amps.len();
    let count = rows.len() / dim;
    for (amp, &off) in amps[..first + count].iter_mut().zip(offsets) {
        *amp = state[base | off];
    }
    for (r, row) in rows.chunks_exact_mut(dim).enumerate() {
        let t = first + r;
        let a = amps[t];
        for (entry, b) in row[..=t].iter_mut().zip(&amps[..=t]) {
            *entry += a * b.conj();
        }
    }
}

/// `rho[t * dim + t'] = sum_e psi[idx(t, e)] * conj(psi[idx(t', e)])`, where
/// bit `i` of `t` is bit `subsystem[i]` of the basis index and `e` runs over
/// the other `n - k` qubits. The lower triangle is summed and mirrored, so the
/// answer is Hermitian by construction. Whatever norm `state` carries stays in
/// the answer.
///
/// Above [`PARALLEL_THRESHOLD_QUBITS`](super::PARALLEL_THRESHOLD_QUBITS) the
/// work is split one of two ways. A fold over the traced index gives each job
/// its own `dim * dim` accumulator to allocate, zero and merge; a stripe of
/// result rows gathers the `dim` amplitudes of every traced index again,
/// `groups * dim` loads per job. The stripe is the cheaper side once the
/// traced indices number under [`STRIPE_RATIO`] times the side.
pub(crate) fn dense_reduced_density(
    state: &[Complex64],
    num_qubits: usize,
    subsystem: &[usize],
) -> Vec<Complex64> {
    debug_assert_eq!(state.len(), 1usize << num_qubits);
    let k = subsystem.len();
    let dim = 1usize << k;
    let offsets = row_offsets(subsystem);
    let mut ascending = subsystem.to_vec();
    ascending.sort_unstable();
    let groups = state.len() >> k;

    let zero = Complex64::new(0.0, 0.0);
    let mut rho = vec![zero; dim * dim];

    #[cfg(feature = "parallel")]
    if num_qubits >= PARALLEL_THRESHOLD_QUBITS {
        if groups < STRIPE_RATIO * dim {
            let rows = stripe_rows(dim);
            rho.par_chunks_mut(rows * dim)
                .enumerate()
                .for_each(|(stripe, out)| {
                    let mut amps = vec![zero; dim];
                    for e in 0..groups {
                        let base = traced_base(e, &ascending);
                        accumulate_rows(state, &offsets, base, &mut amps, out, stripe * rows);
                    }
                });
        } else {
            let fresh = || (vec![zero; dim * dim], vec![zero; dim]);
            let accumulate = |(mut acc, mut amps): (Vec<Complex64>, Vec<Complex64>), e: usize| {
                let base = traced_base(e, &ascending);
                accumulate_rows(state, &offsets, base, &mut amps, &mut acc, 0);
                (acc, amps)
            };
            let add = |(mut a, amps): (Vec<Complex64>, Vec<Complex64>),
                       (b, _): (Vec<Complex64>, Vec<Complex64>)| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                (a, amps)
            };
            rho = (0..groups)
                .into_par_iter()
                .with_min_len(fold_min_len(groups, dim * (dim + 1) / 2))
                .fold(fresh, accumulate)
                .reduce(fresh, add)
                .0;
        }
        mirror_lower_triangle(&mut rho, dim);
        return rho;
    }

    let mut amps = vec![zero; dim];
    for e in 0..groups {
        let base = traced_base(e, &ascending);
        accumulate_rows(state, &offsets, base, &mut amps, &mut rho, 0);
    }
    mirror_lower_triangle(&mut rho, dim);
    rho
}

fn mirror_lower_triangle(rho: &mut [Complex64], dim: usize) {
    for t in 0..dim {
        for tp in 0..t {
            rho[tp * dim + t] = rho[t * dim + tp].conj();
        }
    }
}

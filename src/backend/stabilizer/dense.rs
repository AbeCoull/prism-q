//! Dense probabilities and amplitudes of the stabilizer state held in rows
//! `n..2n` of a tableau, for the stabilizer and factored stabilizer backends.

use num_complex::Complex64;
use smallvec::SmallVec;

use super::kernels::{rowmul_words, xor_words};
use crate::backend::{
    NORM_CLAMP_MIN, dense_probability_len, dense_statevector_len, reserve_dense_output,
};
use crate::error::Result;

/// Support of the stabilizer state in rows `n..2n` of `xz`, as its lowest basis
/// state and the X-parts of the `k` generators that span it: the support is the
/// `2^k` states `seed ^ span(xparts)`, each with probability `2^-k`.
///
/// Rows are `stride` words, `nw` X words then `nw` Z words; bit `q` of a basis
/// index is qubit `q`. O(n^3/64) from the two eliminations.
fn tableau_support(
    xz: &[u64],
    phase: &[bool],
    n: usize,
    nw: usize,
    stride: usize,
) -> (usize, Vec<usize>) {
    let (stab_x, stab_z, stab_phase, diag) = gauss_eliminate_x(xz, phase, n, nw, stride);
    let seed = solve_diagonal_seed(&stab_z, &stab_phase, &diag, nw, n);

    let mut non_diag_set = vec![true; n];
    for &di in &diag {
        non_diag_set[di] = false;
    }

    let coset_gens: Vec<usize> = (0..n)
        .filter(|&i| non_diag_set[i])
        .map(|i| {
            let mut x = 0usize;
            #[allow(clippy::needless_range_loop)]
            for w in 0..nw {
                let shift = w * 64;
                if shift < usize::BITS as usize {
                    x |= (stab_x[i * nw + w] as usize) << shift;
                }
            }
            x
        })
        .collect();

    (min_coset_member(seed, &coset_gens), coset_gens)
}

/// Lowest index in `seed ^ span(gens)`. Anchoring the dense export's phase at the
/// lowest support index makes it independent of how the generators are written,
/// so tableaux that differ only in generator choice export the same vector, and
/// a product of clusters exports the same vector as one tableau of the whole.
fn min_coset_member(seed: usize, gens: &[usize]) -> usize {
    let mut by_top_bit = [0usize; usize::BITS as usize];
    for &g in gens {
        let mut v = g;
        while v != 0 {
            let top = v.ilog2() as usize;
            if by_top_bit[top] == 0 {
                by_top_bit[top] = v;
                break;
            }
            v ^= by_top_bit[top];
        }
    }
    let mut min = seed;
    for top in (0..by_top_bit.len()).rev() {
        if (min >> top) & 1 == 1 {
            min ^= by_top_bit[top];
        }
    }
    min
}

/// Build the `2^n` probabilities by Gray-code enumeration of the support coset.
///
/// Fails with the `backend` name when the output exceeds the probability cap or
/// cannot be allocated.
pub(crate) fn dense_probabilities(
    xz: &[u64],
    phase: &[bool],
    n: usize,
    nw: usize,
    stride: usize,
    backend: &str,
) -> Result<Vec<f64>> {
    let dim = dense_probability_len(backend, n)?;
    let mut probs = Vec::new();
    reserve_dense_output(&mut probs, dim, backend, "probabilities")?;

    let (seed, coset_gens) = tableau_support(xz, phase, n, nw, stride);
    let k = coset_gens.len();
    let amplitude_sq = 1.0 / (1u64 << k) as f64;

    if k == n {
        probs.resize(dim, amplitude_sq);
        return Ok(probs);
    }
    probs.resize(dim, 0.0);

    let coset_size = 1usize << k;
    let mut current = seed;
    probs[current] = amplitude_sq;

    for i in 1..coset_size {
        let bit = i.trailing_zeros() as usize;
        current ^= coset_gens[bit];
        probs[current] = amplitude_sq;
    }

    Ok(probs)
}

/// Build the `2^n` amplitudes by projecting the support seed through each
/// generator (see [`project_generators`]). The amplitude at the lowest support
/// index comes out real and positive, which fixes the global phase.
///
/// Fails with `backend` and `operation` when the output exceeds the export cap
/// or cannot be allocated.
pub(crate) fn dense_statevector(
    xz: &[u64],
    phase: &[bool],
    n: usize,
    nw: usize,
    stride: usize,
    backend: &str,
    operation: &str,
) -> Result<Vec<Complex64>> {
    let dim = dense_statevector_len(backend, operation, n)?;
    let mut state = Vec::new();
    reserve_dense_output(&mut state, dim, backend, operation)?;
    let mut visited_gen = Vec::new();
    reserve_dense_output(&mut visited_gen, dim, backend, operation)?;

    let (seed, _) = tableau_support(xz, phase, n, nw, stride);
    state.resize(dim, Complex64::new(0.0, 0.0));
    state[seed] = Complex64::new(1.0, 0.0);
    visited_gen.resize(dim, 0u32);

    project_generators(&mut state, &mut visited_gen, xz, phase, n, nw, stride);

    Ok(state)
}

/// Gaussian-eliminate the stabilizer X-part to separate diagonal (Z-only)
/// from non-diagonal generators.
///
/// Returns (stab_x, stab_z, stab_phase, diag_indices).
/// stab_x and stab_z are flat arrays with stride `nw` (row i at offset `i * nw`).
#[allow(clippy::type_complexity)]
fn gauss_eliminate_x(
    xz: &[u64],
    phase: &[bool],
    n: usize,
    nw: usize,
    stride: usize,
) -> (Vec<u64>, Vec<u64>, Vec<bool>, Vec<usize>) {
    let mut stab_x = vec![0u64; n * nw];
    let mut stab_z = vec![0u64; n * nw];
    let mut stab_phase = vec![false; n];

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let src = (i + n) * stride;
        let dst = i * nw;
        stab_x[dst..dst + nw].copy_from_slice(&xz[src..src + nw]);
        stab_z[dst..dst + nw].copy_from_slice(&xz[src + nw..src + nw + nw]);
        stab_phase[i] = phase[i + n];
    }

    let mut remaining: Vec<usize> = (0..n).collect();

    for col in 0..n {
        let w = col / 64;
        let b = col % 64;
        let mut pivot_idx = None;
        for (ri, &row) in remaining.iter().enumerate() {
            if (stab_x[row * nw + w] >> b) & 1 == 1 {
                pivot_idx = Some(ri);
                break;
            }
        }

        if let Some(ri) = pivot_idx {
            let pr = remaining.swap_remove(ri);
            let pr_off = pr * nw;

            for row in 0..n {
                if row == pr {
                    continue;
                }
                let row_off = row * nw;
                if (stab_x[row_off + w] >> b) & 1 == 1 {
                    let initial_sum = if stab_phase[pr] { 2u64 } else { 0 }
                        + if stab_phase[row] { 2u64 } else { 0 };
                    // SAFETY: row != pr, so [row_off..row_off+nw] and
                    // [pr_off..pr_off+nw] are non-overlapping regions.
                    let (dst_x, dst_z, src_x, src_z) = unsafe {
                        let xp = stab_x.as_mut_ptr();
                        let zp = stab_z.as_mut_ptr();
                        (
                            std::slice::from_raw_parts_mut(xp.add(row_off), nw),
                            std::slice::from_raw_parts_mut(zp.add(row_off), nw),
                            std::slice::from_raw_parts(xp.add(pr_off) as *const u64, nw),
                            std::slice::from_raw_parts(zp.add(pr_off) as *const u64, nw),
                        )
                    };
                    let sum = rowmul_words(dst_x, dst_z, src_x, src_z, initial_sum);
                    stab_phase[row] = (sum & 3) >= 2;
                }
            }
        }
    }

    (stab_x, stab_z, stab_phase, remaining)
}

fn solve_diagonal_seed(
    stab_z: &[u64],
    stab_phase: &[bool],
    diag: &[usize],
    nw: usize,
    n: usize,
) -> usize {
    let d = diag.len();
    if d == 0 {
        return 0;
    }

    let mut z_rows: Vec<u64> = Vec::with_capacity(d * nw);
    let mut phases: Vec<bool> = Vec::with_capacity(d);
    for &di in diag {
        z_rows.extend_from_slice(&stab_z[di * nw..(di + 1) * nw]);
        phases.push(stab_phase[di]);
    }

    let mut pivot_col = vec![usize::MAX; d];
    let mut available_cols: Vec<usize> = (0..n).collect();

    for row in 0..d {
        let row_off = row * nw;
        let mut found = None;
        for (ci, &col) in available_cols.iter().enumerate() {
            if (z_rows[row_off + col / 64] >> (col % 64)) & 1 == 1 {
                found = Some(ci);
                break;
            }
        }

        if let Some(ci) = found {
            let col = available_cols.swap_remove(ci);
            pivot_col[row] = col;
            let w = col / 64;
            let b = col % 64;

            let pivot_z: SmallVec<[u64; 16]> = SmallVec::from_slice(&z_rows[row_off..row_off + nw]);
            let pivot_phase = phases[row];

            #[allow(clippy::needless_range_loop)]
            for other in 0..d {
                if other == row {
                    continue;
                }
                let other_off = other * nw;
                if (z_rows[other_off + w] >> b) & 1 == 1 {
                    // SAFETY: other_off..other_off+nw and pivot_z are non-overlapping
                    // valid regions of nw u64s. pivot_z was cloned from z_rows at
                    // row_off (row != other), so the regions do not alias.
                    unsafe {
                        xor_words(z_rows.as_mut_ptr().add(other_off), pivot_z.as_ptr(), nw);
                    }
                    phases[other] ^= pivot_phase;
                }
            }
        }
    }

    let mut seed = 0usize;
    for row in 0..d {
        if pivot_col[row] != usize::MAX && phases[row] {
            seed |= 1 << pivot_col[row];
        }
    }
    seed
}

/// Project a support seed through the `n` stabilizer generators and normalize,
/// building the dense statevector in place.
///
/// `sv` holds the seed basis state and has `2^n` entries; `visited_gen` is a
/// scratch buffer of the same length. Rows `n..2n` of `xz` are the generators,
/// bit-packed as `nw` X words then `nw` Z words per `stride`-word row.
///
/// AG convention: `g = (-1)^r * i^m * prod_j X_j^{x_j} Z_j^{z_j}`, where
/// `m = popcount(x_bits & z_bits)` counts the implicit i-factor the Y-type
/// qubits contribute, so `g|y> = (-1)^{r + dot(z,y)} * i^m * |y ^ x_bits>`.
/// The projectors `(I + g_i)/2` commute, so generator order is irrelevant.
fn project_generators(
    sv: &mut [Complex64],
    visited_gen: &mut [u32],
    xz: &[u64],
    phase: &[bool],
    n: usize,
    nw: usize,
    stride: usize,
) {
    let dim = sv.len();
    let zero = Complex64::new(0.0, 0.0);
    let powers_of_i = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, -1.0),
    ];

    let mut current_gen = 0u32;
    for i in 0..n {
        let row = i + n;
        let base = row * stride;

        let mut x_bits = 0usize;
        let mut z_bits = 0usize;
        for w in 0..nw {
            let shift = w * 64;
            if shift < usize::BITS as usize {
                x_bits |= (xz[base + w] as usize) << shift;
                z_bits |= (xz[base + nw + w] as usize) << shift;
            }
        }
        let r = phase[row];

        let m = (x_bits & z_bits).count_ones() as usize;
        let i_factor = powers_of_i[m & 3];
        let base_sign = if r { -1.0 } else { 1.0 };

        if x_bits == 0 {
            for (y, s) in sv.iter_mut().enumerate() {
                let dot_parity = (z_bits & y).count_ones() & 1;
                let phase_val = if dot_parity == 0 {
                    base_sign
                } else {
                    -base_sign
                };
                if phase_val < 0.0 {
                    *s = zero;
                }
            }
        } else {
            current_gen += 1;
            for y in 0..dim {
                if visited_gen[y] == current_gen {
                    continue;
                }
                let partner = y ^ x_bits;
                visited_gen[partner] = current_gen;

                let a = sv[y];
                let b = sv[partner];

                let dot_y = (z_bits & y).count_ones() & 1;
                let real_y = if dot_y == 0 { base_sign } else { -base_sign };
                let gy_phase = i_factor * real_y;

                let dot_p = (z_bits & partner).count_ones() & 1;
                let real_p = if dot_p == 0 { base_sign } else { -base_sign };
                let gp_phase = i_factor * real_p;

                sv[y] = (a + b * gp_phase) * 0.5;
                sv[partner] = (b + a * gy_phase) * 0.5;
            }
        }
    }

    let norm_sq: f64 = sv.iter().map(Complex64::norm_sqr).sum();
    if norm_sq > NORM_CLAMP_MIN {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for amp in sv {
            *amp *= inv_norm;
        }
    }
}

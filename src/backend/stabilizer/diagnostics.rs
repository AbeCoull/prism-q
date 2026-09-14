//! Subsystem diagnostics on a stabilizer tableau: the entanglement entropy
//! from the rank of the generators restricted to one side of a cut, and the
//! reduced density matrix from the projector onto the generators supported
//! inside the subsystem.

use num_complex::Complex64;

use crate::backend::dense_statevector_len;
use crate::error::Result;

/// `i^p`, the coefficient a signed Pauli carries in the `i^p X^x Z^z` form
/// used here.
#[inline]
fn i_power(p: u8) -> Complex64 {
    match p & 3 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

/// X and Z words of stabilizer generator `i`, rows `n..2n` of the tableau.
#[inline]
fn generator(xz: &[u64], n: usize, nw: usize, i: usize) -> (&[u64], &[u64]) {
    let base = (n + i) * 2 * nw;
    (&xz[base..base + nw], &xz[base + nw..base + 2 * nw])
}

#[inline]
fn bit(words: &[u64], q: usize) -> bool {
    words[q / 64] >> (q % 64) & 1 == 1
}

#[inline]
fn set_bit(words: &mut [u64], column: usize) {
    words[column / 64] |= 1 << (column % 64);
}

/// `subsystem` or its complement, whichever names fewer qubits. The state a
/// tableau holds is pure, so both sides of a cut carry the same entropy.
fn smaller_side(n: usize, subsystem: &[usize]) -> Vec<usize> {
    if 2 * subsystem.len() <= n {
        return subsystem.to_vec();
    }
    let mut inside = vec![false; n];
    for &q in subsystem {
        inside[q] = true;
    }
    (0..n).filter(|&q| !inside[q]).collect()
}

/// Forward GF(2) elimination over `cols` columns of a `count`-row bit matrix,
/// returning the rank.
fn eliminate(rows: &mut [u64], count: usize, words: usize, cols: usize) -> usize {
    let mut rank = 0;
    for column in 0..cols {
        let (word, mask) = (column / 64, 1u64 << (column % 64));
        let Some(pivot) = (rank..count).find(|&r| rows[r * words + word] & mask != 0) else {
            continue;
        };
        for w in 0..words {
            rows.swap(rank * words + w, pivot * words + w);
        }
        for r in rank + 1..count {
            if rows[r * words + word] & mask == 0 {
                continue;
            }
            for w in 0..words {
                rows[r * words + w] ^= rows[rank * words + w];
            }
        }
        rank += 1;
    }
    rank
}

/// Rank of the cut at `subsystem`: `rank(G restricted to A) - |A|`, one
/// elimination of `n` rows by `2|A|` bits, read on the smaller side.
///
/// The entanglement entropy is this times `ln 2` and the Schmidt spectrum is
/// [`flat_spectrum`] of it, since a stabilizer cut has equal weights.
pub(crate) fn subsystem_rank(xz: &[u64], n: usize, nw: usize, subsystem: &[usize]) -> usize {
    let side = smaller_side(n, subsystem);
    let k = side.len();
    let words = (2 * k).div_ceil(64);
    let mut rows = vec![0u64; n * words];
    for i in 0..n {
        let (x, z) = generator(xz, n, nw, i);
        let row = &mut rows[i * words..(i + 1) * words];
        for (column, &q) in side.iter().enumerate() {
            if bit(x, q) {
                set_bit(row, column);
            }
            if bit(z, q) {
                set_bit(row, k + column);
            }
        }
    }
    let rank = eliminate(&mut rows, n, words, 2 * k);
    debug_assert!(
        rank >= k,
        "a cut of a pure stabilizer state has rank at least the subsystem size"
    );
    rank - k
}

/// The Schmidt spectrum of a stabilizer cut of rank `r`: `2^r` copies of
/// `2^(-r/2)`, the whole of it, since every weight is equal.
///
/// Priced as an `r`-qubit statevector against the dense export cap. The
/// marginal on the same cut is priced at `2k` qubits for `r <= k`, so a cut
/// whose marginal this backend would build has a spectrum it will also build.
pub(crate) fn flat_spectrum(backend: &str, rank: usize) -> Result<Vec<f64>> {
    let count = dense_statevector_len(backend, "Schmidt values", rank)?;
    Ok(vec![(count as f64).sqrt().recip(); count])
}

/// Product of two signed Paulis over the subsystem's bit order:
/// `U(x1, z1) U(x2, z2)` is `(-1)^popcount(z1 & x2) U(x1 ^ x2, z1 ^ z2)`, so
/// the exponent of `i` picks up twice that parity.
#[inline]
fn multiply(a: (usize, usize, u8), b: (usize, usize, u8)) -> (usize, usize, u8) {
    let cross = 2 * ((a.1 & b.0).count_ones() as u8 & 1);
    (a.0 ^ b.0, a.1 ^ b.1, (a.2 + b.2 + cross) & 3)
}

/// Add `i^p X^x Z^z` to `rho`: one entry per column, at row `y ^ x`.
#[inline]
fn accumulate(rho: &mut [Complex64], dim: usize, (x, z, p): (usize, usize, u8)) {
    let coeff = i_power(p);
    for y in 0..dim {
        let sign = if (z & y).count_ones() & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        rho[(y ^ x) * dim + y] += coeff * sign;
    }
}

/// The stabilizer generators supported entirely inside `subsystem`, each as
/// `(x, z, p)` over the subsystem's own bit order for the operator
/// `i^p X^x Z^z`.
///
/// They are the rows left standing after a forward GF(2) elimination over the
/// columns of the complement, which is why the phase travels with the rows:
/// every pivot step multiplies one Pauli into another and the sign of that
/// product depends on the whole string, complement bits included.
fn interior_generators(
    xz: &[u64],
    phase: &[bool],
    n: usize,
    nw: usize,
    subsystem: &[usize],
) -> Vec<(usize, usize, u8)> {
    let stride = 2 * nw;
    let mut rows = vec![0u64; n * stride];
    let mut pow = vec![0u8; n];
    for i in 0..n {
        let (x, z) = generator(xz, n, nw, i);
        let row = &mut rows[i * stride..(i + 1) * stride];
        row[..nw].copy_from_slice(x);
        row[nw..].copy_from_slice(z);
        let y_count: u32 = x.iter().zip(z).map(|(a, b)| (a & b).count_ones()).sum();
        pow[i] = ((2 * u32::from(phase[n + i]) + y_count) & 3) as u8;
    }

    let mut inside = vec![false; n];
    for &q in subsystem {
        inside[q] = true;
    }
    let mut rank = 0;
    for q in (0..n).filter(|&q| !inside[q]) {
        for half in [0, nw] {
            let (word, mask) = (half + q / 64, 1u64 << (q % 64));
            let Some(pivot) = (rank..n).find(|&r| rows[r * stride + word] & mask != 0) else {
                continue;
            };
            for w in 0..stride {
                rows.swap(rank * stride + w, pivot * stride + w);
            }
            pow.swap(rank, pivot);
            let (head, tail) = rows.split_at_mut((rank + 1) * stride);
            let pivot_row = &head[rank * stride..];
            for (offset, target) in tail.chunks_exact_mut(stride).enumerate() {
                if target[word] & mask == 0 {
                    continue;
                }
                let cross: u32 = (0..nw)
                    .map(|w| (target[nw + w] & pivot_row[w]).count_ones())
                    .sum();
                let r = rank + 1 + offset;
                pow[r] = (pow[r] + pow[rank] + 2 * (cross & 1) as u8) & 3;
                for (entry, &pivot_word) in target.iter_mut().zip(pivot_row) {
                    *entry ^= pivot_word;
                }
            }
            rank += 1;
        }
    }

    (rank..n)
        .map(|r| {
            let row = &rows[r * stride..(r + 1) * stride];
            let (mut x, mut z) = (0usize, 0usize);
            for (i, &q) in subsystem.iter().enumerate() {
                if bit(&row[..nw], q) {
                    x |= 1 << i;
                }
                if bit(&row[nw..], q) {
                    z |= 1 << i;
                }
            }
            (x, z, pow[r])
        })
        .collect()
}

/// Reduced density matrix of `subsystem` on the state the tableau stabilizes,
/// row major with side `dim = 2^k` and bit `i` of the row index the state of
/// `subsystem[i]`.
///
/// `rho_A` is `2^-k` times the sum over the subgroup supported inside the
/// subsystem, enumerated in Gray-code order so each of its `2^m` elements
/// costs one Pauli product and `dim` accumulations. `m` is at most `k`, so the
/// whole walk stays inside the `4^k` the answer already occupies, and the
/// `2^n` vector is never expanded.
pub(crate) fn subsystem_density(
    xz: &[u64],
    phase: &[bool],
    n: usize,
    nw: usize,
    subsystem: &[usize],
    dim: usize,
) -> Vec<Complex64> {
    let generators = interior_generators(xz, phase, n, nw, subsystem);
    let mut rho = vec![Complex64::new(0.0, 0.0); dim * dim];
    let mut element = (0usize, 0usize, 0u8);
    accumulate(&mut rho, dim, element);
    for t in 1..1usize << generators.len() {
        element = multiply(element, generators[t.trailing_zeros() as usize]);
        accumulate(&mut rho, dim, element);
    }
    let scale = 1.0 / dim as f64;
    for entry in &mut rho {
        *entry *= scale;
    }
    rho
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::backend::stabilizer::StabilizerBackend;
    use crate::circuit::Circuit;
    use crate::gates::Gate;

    fn tableau(circuit: &Circuit) -> StabilizerBackend {
        let mut backend = StabilizerBackend::new(42);
        crate::sim::run_on(&mut backend, circuit).unwrap();
        backend
    }

    fn density(backend: &StabilizerBackend, subsystem: &[usize]) -> Vec<Complex64> {
        let (xz, phase) = backend.raw_tableau();
        subsystem_density(
            xz,
            phase,
            backend.num_qubits(),
            backend.num_qubits().div_ceil(64),
            subsystem,
            1 << subsystem.len(),
        )
    }

    fn assert_entries(actual: &[Complex64], expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).norm() < 1e-15, "entry {i} reads {a} against {e}");
        }
    }

    // `h; s` stabilizes +Y, whose generator carries the implicit factor of i
    // that the X and Z bits of one qubit both being set stands for. Reading it
    // as a plain XZ product would flip the sign of the off-diagonal.
    #[test]
    fn a_y_generator_gives_the_plus_i_eigenstate() {
        let mut circuit = Circuit::new(1, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::S, &[0]);
        let half = Complex64::new(0.5, 0.0);
        assert_entries(
            &density(&tableau(&circuit), &[0]),
            &[
                half,
                Complex64::new(0.0, -0.5),
                Complex64::new(0.0, 0.5),
                half,
            ],
        );
    }

    // `x` stabilizes -Z, so the projector lands on |1> rather than |0>.
    #[test]
    fn a_negative_generator_moves_the_projector() {
        let mut circuit = Circuit::new(1, 0);
        circuit.add_gate(Gate::X, &[0]);
        let (zero, one) = (Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
        assert_entries(&density(&tableau(&circuit), &[0]), &[zero, zero, zero, one]);
    }

    // Naming the whole register leaves no complement to eliminate, so every
    // generator is interior and the sum over the group is the pure state's own
    // projector: the Bell pair's four corners at 1/2 and nothing between them.
    #[test]
    fn the_whole_register_gives_the_pure_state_projector() {
        let mut circuit = Circuit::new(2, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        let backend = tableau(&circuit);
        let (half, zero) = (Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0));
        assert_entries(
            &density(&backend, &[0, 1]),
            &[
                half, zero, zero, half, zero, zero, zero, zero, zero, zero, zero, zero, half, zero,
                zero, half,
            ],
        );
        let (xz, _) = backend.raw_tableau();
        assert_eq!(subsystem_rank(xz, 2, 1, &[0]), 1);
    }
}

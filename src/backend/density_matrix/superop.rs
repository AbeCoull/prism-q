//! Deferred block superoperators for the exact noisy walk: Kraus set compilers, the
//! channels trailing a two-qubit gate, and the floors that decide when a fold pays.

use num_complex::Complex64;

use super::{DensityMatrixBackend, matrix_1q};
use crate::gates::{Gate, mat_mul_4x4};

/// Compile a one-qubit Kraus set into the 4x4 superoperator acting on the
/// `(row-bit, col-bit)` block of `rho`, where block index `i = 2*a + b` orders
/// `(row-bit a, col-bit b)`:
/// `S[2a+b][2a'+b'] = sum_k K_k[a][a'] * conj(K_k[b][b'])`.
///
/// A single-element set `[U]` gives the unitary sandwich `U rho U^dagger`.
pub(super) fn block_superoperator(kraus: &[[[Complex64; 2]; 2]]) -> [[Complex64; 4]; 4] {
    let mut s = [[Complex64::new(0.0, 0.0); 4]; 4];
    for k in kraus {
        for a in 0..2 {
            for b in 0..2 {
                for ap in 0..2 {
                    for bp in 0..2 {
                        s[2 * a + b][2 * ap + bp] += k[a][ap] * k[b][bp].conj();
                    }
                }
            }
        }
    }
    s
}

/// Compile a two-qubit Kraus set into the 16x16 block superoperator
/// [`DensityMatrixBackend::apply_2q_kraus`] sweeps:
/// `S[4*tr+tc][4*trp+tcp] = sum_k K_k[tr][trp] * conj(K_k[tc][tcp])`.
pub(super) fn kraus_2q_superoperator(kraus: &[[[Complex64; 4]; 4]]) -> [[Complex64; 16]; 16] {
    let mut s = [[Complex64::new(0.0, 0.0); 16]; 16];
    for k in kraus {
        for tr in 0..4 {
            for trp in 0..4 {
                let kr = k[tr][trp];
                if kr == Complex64::new(0.0, 0.0) {
                    continue;
                }
                for tc in 0..4 {
                    for tcp in 0..4 {
                        s[4 * tr + tc][4 * trp + tcp] += kr * k[tc][tcp].conj();
                    }
                }
            }
        }
    }
    s
}

/// The gate's 4x4 matrix over `(targets[0], targets[1])`, or `None` for a
/// variant [`Gate::matrix_4x4`] does not cover.
fn matrix_2q(gate: &Gate) -> Option<[[Complex64; 4]; 4]> {
    match gate {
        Gate::Rzz(_) | Gate::Cx | Gate::Cz | Gate::Swap | Gate::Cu(_) | Gate::Fused2q(_) => {
            Some(gate.matrix_4x4())
        }
        Gate::PauliRot(data) if data.axes.len() == 2 => Some(gate.matrix_4x4()),
        _ => None,
    }
}

/// `A (x) B` over the 16x16 block index `4 * tr + tc`, `A` on the first
/// qubit's `(row-bit, col-bit)` pair and `B` on the second's.
fn kron_block_superoperators(
    a: &[[Complex64; 4]; 4],
    b: &[[Complex64; 4]; 4],
) -> [[Complex64; 16]; 16] {
    let pair = |t: usize, u: usize, shift: usize| 2 * ((t >> shift) & 1) + ((u >> shift) & 1);
    std::array::from_fn(|row| {
        let (tr, tc) = (row >> 2, row & 3);
        std::array::from_fn(|col| {
            let (trp, tcp) = (col >> 2, col & 3);
            a[pair(tr, tc, 1)][pair(trp, tcp, 1)] * b[pair(tr, tc, 0)][pair(trp, tcp, 0)]
        })
    })
}

fn mat_mul_16x16(a: &[[Complex64; 16]; 16], b: &[[Complex64; 16]; 16]) -> [[Complex64; 16]; 16] {
    std::array::from_fn(|r| std::array::from_fn(|c| (0..16).map(|k| a[r][k] * b[k][c]).sum()))
}

const IDENTITY_4X4: [[Complex64; 4]; 4] = {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, o]]
};

/// Narrowest register the exact noisy walk defers one-qubit maps on. Below it
/// the native one-qubit and two-qubit kernels beat the composed sweeps.
pub(crate) const MIN_DEFERRED_QUBITS: usize = 7;

/// Fewest pending maps and trailing channels a two-qubit gate must absorb for its
/// dense 16x16 sweep to beat the native kernels, by where the buffer sits. Past the
/// last-level cache every sweep streams from memory and costs about the same, so one
/// map pays. Resident in a quarter of it or more, the sweep is compute-bound and costs
/// about two one-qubit block sweeps: on an i7-6700K (8 MB L3) a one-map fold at 9
/// qubits read 10% slower than a CX or controlled phase plus its block sweep, and a
/// one-map `Rzz` fold 11% faster. Smaller still, composing the 16x16 map is itself a
/// sweep's worth of work and three maps pay for it; four lost the three-map `Rzz`
/// folds at 7 qubits.
const MIN_FOLDED_MAPS_PAST_CACHE: usize = 1;
const MIN_FOLDED_MAPS_IN_CACHE: usize = 2;
const MIN_FOLDED_MAPS_SMALL: usize = 3;

/// Last-level cache assumed when the OS reports none: the 8 MB L3 the floors above were
/// measured against.
const ASSUMED_LLC_BYTES: usize = 8 << 20;

/// The fold floor for an `num_qubits` buffer on this host, see
/// [`MIN_FOLDED_MAPS_PAST_CACHE`].
fn min_folded_maps(num_qubits: usize) -> usize {
    let bytes = size_of::<Complex64>() << (2 * num_qubits);
    let llc = crate::backend::cache::last_level_cache().map_or(ASSUMED_LLC_BYTES, |c| c.bytes);
    if bytes >= llc {
        MIN_FOLDED_MAPS_PAST_CACHE
    } else if bytes >= llc / 4 {
        MIN_FOLDED_MAPS_IN_CACHE
    } else {
        MIN_FOLDED_MAPS_SMALL
    }
}

/// One-qubit block superoperators the exact noisy walk holds back per qubit.
///
/// Gates and one-qubit channels on a qubit compose here rather than sweeping
/// the buffer one at a time, whichever instruction the channel trails. A
/// two-qubit gate folds both its qubits' pending maps and its own trailing
/// channels on the pair, one- or two-qubit, into one 16x16 sweep, and any
/// other instruction on a held qubit flushes it first. Every sweep costs about
/// the same on a buffer past the caches, so a CX with relaxation on both
/// targets and a pair depolarizing channel drops from five sweeps to one.
pub(crate) struct DeferredSuperoperators {
    pending: Vec<Option<[[Complex64; 4]; 4]>>,
    min_maps: usize,
}

/// A channel trailing a two-qubit gate, lowered for
/// [`DeferredSuperoperators::apply_2q`].
pub(crate) enum PairChannel {
    /// Block superoperator on one of the gate's two qubits.
    Block(usize, Box<[[Complex64; 4]; 4]>),
    /// 16x16 superoperator over the gate's pair, indexed for `(q0, q1)` as
    /// [`DensityMatrixBackend::apply_2q_kraus`] compiles it.
    Pair(Box<[[Complex64; 16]; 16]>),
}

impl PairChannel {
    pub(crate) fn block(qubit: usize, kraus: &[[[Complex64; 2]; 2]]) -> Self {
        Self::Block(qubit, Box::new(block_superoperator(kraus)))
    }

    /// A two-qubit Kraus set given on `(q1, q0)` when `swapped`.
    pub(crate) fn kraus_2q(kraus: &[[[Complex64; 4]; 4]], swapped: bool) -> Self {
        let s = kraus_2q_superoperator(kraus);
        if !swapped {
            return Self::Pair(Box::new(s));
        }
        let swap = |i: usize| {
            let t = |t: usize| ((t & 1) << 1) | (t >> 1);
            4 * t(i >> 2) + t(i & 3)
        };
        Self::Pair(Box::new(std::array::from_fn(|r| {
            std::array::from_fn(|c| s[swap(r)][swap(c)])
        })))
    }

    /// Symmetric two-qubit depolarizing, the 16x16 form of
    /// [`DensityMatrixBackend::apply_2q_depolarizing`]'s closed form.
    pub(crate) fn depolarizing_2q(p: f64) -> Self {
        let alpha = Complex64::new(1.0 - 16.0 * p / 15.0, 0.0);
        let beta = Complex64::new(4.0 * p / 15.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        Self::Pair(Box::new(std::array::from_fn(|r| {
            std::array::from_fn(|c| {
                let mut e = if r == c { alpha } else { zero };
                if r % 5 == 0 && c % 5 == 0 {
                    e += beta;
                }
                e
            })
        })))
    }
}

impl DeferredSuperoperators {
    pub(crate) fn new(num_qubits: usize) -> Self {
        Self {
            pending: vec![None; num_qubits],
            min_maps: min_folded_maps(num_qubits),
        }
    }

    /// Compose `gate` onto `qubit`'s pending map. False, with nothing held,
    /// when `gate` has no 2x2 matrix.
    pub(crate) fn defer_gate(&mut self, gate: &Gate, qubit: usize) -> bool {
        let Some(mat) = matrix_1q(gate) else {
            return false;
        };
        self.defer_channel(qubit, &[mat]);
        true
    }

    /// Compose the Kraus set onto `qubit`'s pending map.
    pub(crate) fn defer_channel(&mut self, qubit: usize, kraus: &[[[Complex64; 2]; 2]]) {
        let mut s = block_superoperator(kraus);
        if let Some(prior) = &self.pending[qubit] {
            s = mat_mul_4x4(&s, prior);
        }
        self.pending[qubit] = Some(s);
    }

    /// Apply `gate` on `(q0, q1)` in one sweep, preceded by both qubits'
    /// pending maps and followed by `channels` in order, each on `q0`, `q1`,
    /// or the pair. False, with nothing applied, when `gate` has no 4x4 matrix
    /// or the fold would absorb fewer pending maps and channels than the
    /// register width calls for; one-qubit channels count once per qubit.
    pub(crate) fn apply_2q(
        &mut self,
        dm: &mut DensityMatrixBackend,
        gate: &Gate,
        q0: usize,
        q1: usize,
        channels: &[PairChannel],
    ) -> bool {
        let Some(g) = matrix_2q(gate) else {
            return false;
        };
        let on = |qubit: usize| {
            channels
                .iter()
                .any(|c| matches!(c, PairChannel::Block(q, _) if *q == qubit))
        };
        let maps = usize::from(self.pending[q0].is_some())
            + usize::from(self.pending[q1].is_some())
            + usize::from(on(q0))
            + usize::from(on(q1))
            + channels
                .iter()
                .filter(|c| matches!(c, PairChannel::Pair(_)))
                .count();
        if maps < self.min_maps {
            return false;
        }

        let before = kron_block_superoperators(
            self.pending[q0].as_ref().unwrap_or(&IDENTITY_4X4),
            self.pending[q1].as_ref().unwrap_or(&IDENTITY_4X4),
        );
        let gate_map: [[Complex64; 16]; 16] = std::array::from_fn(|row| {
            std::array::from_fn(|col| g[row >> 2][col >> 2] * g[row & 3][col & 3].conj())
        });
        let mut s = mat_mul_16x16(&gate_map, &before);
        let (mut after0, mut after1) = (None, None);
        let close = |s: &mut [[Complex64; 16]; 16], a0: &mut Option<_>, a1: &mut Option<_>| {
            if a0.is_some() || a1.is_some() {
                let after = kron_block_superoperators(
                    a0.as_ref().unwrap_or(&IDENTITY_4X4),
                    a1.as_ref().unwrap_or(&IDENTITY_4X4),
                );
                *s = mat_mul_16x16(&after, s);
                (*a0, *a1) = (None, None);
            }
        };
        for channel in channels {
            match channel {
                PairChannel::Block(qubit, b) => {
                    let target = if *qubit == q0 {
                        &mut after0
                    } else {
                        &mut after1
                    };
                    *target = Some(match target {
                        Some(prior) => mat_mul_4x4(b, prior),
                        None => **b,
                    });
                }
                PairChannel::Pair(p) => {
                    close(&mut s, &mut after0, &mut after1);
                    s = mat_mul_16x16(p, &s);
                }
            }
        }
        close(&mut s, &mut after0, &mut after1);

        self.pending[q0] = None;
        self.pending[q1] = None;
        dm.apply_2q_superoperator(q0, q1, &s);
        true
    }

    /// Sweep every pending map into the buffer.
    pub(crate) fn flush(&mut self, dm: &mut DensityMatrixBackend) {
        for (qubit, pending) in self.pending.iter_mut().enumerate() {
            if let Some(s) = pending.take() {
                dm.apply_block_superoperator_infallible(qubit, &s);
            }
        }
    }

    /// Sweep the pending maps of `qubits` into the buffer.
    pub(crate) fn flush_qubits(&mut self, dm: &mut DensityMatrixBackend, qubits: &[usize]) {
        for &qubit in qubits {
            if let Some(s) = self.pending[qubit].take() {
                dm.apply_block_superoperator_infallible(qubit, &s);
            }
        }
    }
}

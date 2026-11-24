//! Clifford propagation passes feeding the compiled sampler: backward
//! Heisenberg propagation of measured Z operators (bit-packed, batched) and
//! column-major forward stabilizer simulation.

use super::accumulator::transpose_64x64_dispatch;
use super::{PauliVec, flip_bit, get_bit, incompatible, set_bit, xor_words};
use crate::backend::stabilizer::kernels::rowmul_words_cached;
#[cfg(target_arch = "x86_64")]
use crate::backend::word_ops::{WordTier, word_tier};
use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::gates::Gate;

/// Heisenberg-picture backward propagation of a Pauli through a Clifford gate,
/// `P <- U†·P·U`, up to sign: the frame carries no phase, so `HYH = -Y` lands
/// as `Y`.
#[inline]
pub(crate) fn propagate_backward(pauli: &mut PauliVec, gate: &Gate, targets: &[usize]) {
    propagate_backward_words(&mut pauli.x, &mut pauli.z, gate, targets);
}

/// [`propagate_backward`] on a Pauli held as bare `x` and `z` word slices.
pub(crate) fn propagate_backward_words(
    x: &mut [u64],
    z: &mut [u64],
    gate: &Gate,
    targets: &[usize],
) {
    match gate {
        Gate::H => {
            let q = targets[0];
            let xb = get_bit(x, q);
            let zb = get_bit(z, q);
            set_bit(x, q, zb);
            set_bit(z, q, xb);
        }
        Gate::S | Gate::Sdg => {
            let q = targets[0];
            if get_bit(x, q) {
                flip_bit(z, q);
            }
        }
        Gate::SX | Gate::SXdg => {
            let q = targets[0];
            if get_bit(z, q) {
                flip_bit(x, q);
            }
        }
        Gate::Cx => {
            let ctrl = targets[0];
            let tgt = targets[1];
            if get_bit(x, ctrl) {
                flip_bit(x, tgt);
            }
            if get_bit(z, tgt) {
                flip_bit(z, ctrl);
            }
        }
        Gate::Cz => {
            let q0 = targets[0];
            let q1 = targets[1];
            let x0 = get_bit(x, q0);
            let x1 = get_bit(x, q1);
            if x1 {
                flip_bit(z, q0);
            }
            if x0 {
                flip_bit(z, q1);
            }
        }
        Gate::Swap => {
            let q0 = targets[0];
            let q1 = targets[1];
            let x0 = get_bit(x, q0);
            let x1 = get_bit(x, q1);
            set_bit(x, q0, x1);
            set_bit(x, q1, x0);
            let z0 = get_bit(z, q0);
            let z1 = get_bit(z, q1);
            set_bit(z, q0, z1);
            set_bit(z, q1, z0);
        }
        _ => {}
    }
}

/// Batch backward-propagate Z_q through all gates for every measurement simultaneously.
///
/// Transposed representation: each qubit's m measurement x/z bits are packed into
/// m/64 u64 words, so gate propagation is bulk bitwise ops (64× fewer operations).
///
/// Returns (PauliVec, classical_bit, sign) per measurement.
pub(super) fn build_measurement_rows(circuit: &Circuit) -> Vec<(PauliVec, usize, bool)> {
    let n = circuit.num_qubits;
    let num_qubit_words = n.div_ceil(64);

    let measurements: Vec<(usize, usize)> = circuit
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Measure {
                qubit,
                classical_bit,
            } => Some((*qubit, *classical_bit)),
            _ => None,
        })
        .collect();

    let m = measurements.len();
    if m == 0 {
        return Vec::new();
    }

    let m_words = m.div_ceil(64);

    let mut x_packed: Vec<Vec<u64>> = vec![vec![0u64; m_words]; n];
    let mut z_packed: Vec<Vec<u64>> = vec![vec![0u64; m_words]; n];
    let mut sign_packed: Vec<u64> = vec![0u64; m_words];

    for (meas_idx, &(qubit, _)) in measurements.iter().enumerate() {
        z_packed[qubit][meas_idx / 64] |= 1u64 << (meas_idx % 64);
    }

    let gates_reversed = circuit
        .instructions
        .iter()
        .rev()
        .filter_map(|inst| match inst {
            Instruction::Gate { gate, targets } => Some((gate, targets.as_slice())),
            Instruction::Conditional { gate, targets, .. } => Some((gate, targets.as_slice())),
            _ => None,
        });

    for (gate, targets) in gates_reversed {
        batch_propagate_backward(
            &mut x_packed,
            &mut z_packed,
            &mut sign_packed,
            gate,
            targets,
            m_words,
        );
    }

    let mut rows: Vec<(PauliVec, usize, bool)> = Vec::with_capacity(m);
    for (meas_idx, &(_, classical_bit)) in measurements.iter().enumerate() {
        let mut pauli = PauliVec::new(num_qubit_words);
        for q in 0..n {
            if x_packed[q][meas_idx / 64] >> (meas_idx % 64) & 1 != 0 {
                pauli.x[q / 64] |= 1u64 << (q % 64);
            }
            if z_packed[q][meas_idx / 64] >> (meas_idx % 64) & 1 != 0 {
                pauli.z[q / 64] |= 1u64 << (q % 64);
            }
        }
        let sign = (sign_packed[meas_idx / 64] >> (meas_idx % 64)) & 1 != 0;
        rows.push((pauli, classical_bit, sign));
    }

    rows
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_h_avx2(xq: &mut [u64], zq: &mut [u64], sign: &mut [u64], m_words: usize) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let chunks = m_words / 4;
        let xp = xq.as_mut_ptr() as *mut __m256i;
        let zp = zq.as_mut_ptr() as *mut __m256i;
        let sp = sign.as_mut_ptr() as *mut __m256i;
        for i in 0..chunks {
            let xv = _mm256_loadu_si256(xp.add(i));
            let zv = _mm256_loadu_si256(zp.add(i));
            let sv = _mm256_loadu_si256(sp.add(i));
            _mm256_storeu_si256(sp.add(i), _mm256_xor_si256(sv, _mm256_and_si256(xv, zv)));
            _mm256_storeu_si256(xp.add(i), zv);
            _mm256_storeu_si256(zp.add(i), xv);
        }
        let tail = chunks * 4;
        for w in tail..m_words {
            *sign.get_unchecked_mut(w) ^= *xq.get_unchecked(w) & *zq.get_unchecked(w);
        }
        for w in tail..m_words {
            let tmp = *xq.get_unchecked(w);
            *xq.get_unchecked_mut(w) = *zq.get_unchecked(w);
            *zq.get_unchecked_mut(w) = tmp;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_s_avx2(
    xq: &mut [u64],
    zq: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
    negate_z: bool,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let chunks = m_words / 4;
        let xp = xq.as_ptr() as *const __m256i;
        let zp = zq.as_mut_ptr() as *mut __m256i;
        let sp = sign.as_mut_ptr() as *mut __m256i;
        if negate_z {
            for i in 0..chunks {
                let xv = _mm256_loadu_si256(xp.add(i));
                let zv = _mm256_loadu_si256(zp.add(i));
                let sv = _mm256_loadu_si256(sp.add(i));
                _mm256_storeu_si256(sp.add(i), _mm256_xor_si256(sv, _mm256_andnot_si256(zv, xv)));
                _mm256_storeu_si256(zp.add(i), _mm256_xor_si256(zv, xv));
            }
            let tail = chunks * 4;
            for w in tail..m_words {
                *sign.get_unchecked_mut(w) ^= *xq.get_unchecked(w) & !*zq.get_unchecked(w);
                *zq.get_unchecked_mut(w) ^= *xq.get_unchecked(w);
            }
        } else {
            for i in 0..chunks {
                let xv = _mm256_loadu_si256(xp.add(i));
                let zv = _mm256_loadu_si256(zp.add(i));
                let sv = _mm256_loadu_si256(sp.add(i));
                _mm256_storeu_si256(sp.add(i), _mm256_xor_si256(sv, _mm256_and_si256(xv, zv)));
                _mm256_storeu_si256(zp.add(i), _mm256_xor_si256(zv, xv));
            }
            let tail = chunks * 4;
            for w in tail..m_words {
                *sign.get_unchecked_mut(w) ^= *xq.get_unchecked(w) & *zq.get_unchecked(w);
                *zq.get_unchecked_mut(w) ^= *xq.get_unchecked(w);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_sign_xor2_avx2(dst: &mut [u64], a: &[u64], b: &[u64], m_words: usize) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let chunks = m_words / 4;
        let dp = dst.as_mut_ptr() as *mut __m256i;
        let ap = a.as_ptr() as *const __m256i;
        let bp = b.as_ptr() as *const __m256i;
        for i in 0..chunks {
            let dv = _mm256_loadu_si256(dp.add(i));
            let av = _mm256_loadu_si256(ap.add(i));
            let bv = _mm256_loadu_si256(bp.add(i));
            _mm256_storeu_si256(dp.add(i), _mm256_xor_si256(dv, _mm256_xor_si256(av, bv)));
        }
        let tail = chunks * 4;
        for w in tail..m_words {
            *dst.get_unchecked_mut(w) ^= *a.get_unchecked(w) ^ *b.get_unchecked(w);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_sx_avx2(
    xq: &mut [u64],
    zq: &[u64],
    sign: &mut [u64],
    m_words: usize,
    negate_x: bool,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let chunks = m_words / 4;
        let xp = xq.as_mut_ptr() as *mut __m256i;
        let zp = zq.as_ptr() as *const __m256i;
        let sp = sign.as_mut_ptr() as *mut __m256i;
        if negate_x {
            for i in 0..chunks {
                let xv = _mm256_loadu_si256(xp.add(i));
                let zv = _mm256_loadu_si256(zp.add(i));
                let sv = _mm256_loadu_si256(sp.add(i));
                _mm256_storeu_si256(sp.add(i), _mm256_xor_si256(sv, _mm256_andnot_si256(xv, zv)));
                _mm256_storeu_si256(xp.add(i), _mm256_xor_si256(xv, zv));
            }
            let tail = chunks * 4;
            for w in tail..m_words {
                *sign.get_unchecked_mut(w) ^= !*xq.get_unchecked(w) & *zq.get_unchecked(w);
                *xq.get_unchecked_mut(w) ^= *zq.get_unchecked(w);
            }
        } else {
            for i in 0..chunks {
                let xv = _mm256_loadu_si256(xp.add(i));
                let zv = _mm256_loadu_si256(zp.add(i));
                let sv = _mm256_loadu_si256(sp.add(i));
                _mm256_storeu_si256(sp.add(i), _mm256_xor_si256(sv, _mm256_and_si256(xv, zv)));
                _mm256_storeu_si256(xp.add(i), _mm256_xor_si256(xv, zv));
            }
            let tail = chunks * 4;
            for w in tail..m_words {
                *sign.get_unchecked_mut(w) ^= *xq.get_unchecked(w) & *zq.get_unchecked(w);
                *xq.get_unchecked_mut(w) ^= *zq.get_unchecked(w);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_cx_avx2(
    x_ctrl: &[u64],
    z_ctrl: &mut [u64],
    x_tgt: &mut [u64],
    z_tgt: &[u64],
    sign: &mut [u64],
    m_words: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let chunks = m_words / 4;
        let xcp = x_ctrl.as_ptr() as *const __m256i;
        let zcp = z_ctrl.as_mut_ptr() as *mut __m256i;
        let xtp = x_tgt.as_mut_ptr() as *mut __m256i;
        let ztp = z_tgt.as_ptr() as *const __m256i;
        let sp = sign.as_mut_ptr() as *mut __m256i;
        for i in 0..chunks {
            let xc = _mm256_loadu_si256(xcp.add(i));
            let zc = _mm256_loadu_si256(zcp.add(i));
            let xt = _mm256_loadu_si256(xtp.add(i));
            let zt = _mm256_loadu_si256(ztp.add(i));
            let sv = _mm256_loadu_si256(sp.add(i));
            let xnor = _mm256_andnot_si256(_mm256_xor_si256(zc, xt), _mm256_set1_epi64x(-1));
            let flip = _mm256_and_si256(_mm256_and_si256(xc, zt), xnor);
            _mm256_storeu_si256(sp.add(i), _mm256_xor_si256(sv, flip));
            _mm256_storeu_si256(xtp.add(i), _mm256_xor_si256(xt, xc));
            _mm256_storeu_si256(zcp.add(i), _mm256_xor_si256(zc, zt));
        }
        let tail = chunks * 4;
        for w in tail..m_words {
            let xc = *x_ctrl.get_unchecked(w);
            let zc = *z_ctrl.get_unchecked(w);
            let xt = *x_tgt.get_unchecked(w);
            let zt = *z_tgt.get_unchecked(w);
            *sign.get_unchecked_mut(w) ^= xc & zt & !(zc ^ xt);
            *x_tgt.get_unchecked_mut(w) = xt ^ xc;
            *z_ctrl.get_unchecked_mut(w) = zc ^ zt;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_cz_avx2(
    x0: &[u64],
    z0: &mut [u64],
    x1: &[u64],
    z1: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let chunks = m_words / 4;
        let x0p = x0.as_ptr() as *const __m256i;
        let z0p = z0.as_mut_ptr() as *mut __m256i;
        let x1p = x1.as_ptr() as *const __m256i;
        let z1p = z1.as_mut_ptr() as *mut __m256i;
        let sp = sign.as_mut_ptr() as *mut __m256i;
        for i in 0..chunks {
            let xv0 = _mm256_loadu_si256(x0p.add(i));
            let zv0 = _mm256_loadu_si256(z0p.add(i));
            let xv1 = _mm256_loadu_si256(x1p.add(i));
            let zv1 = _mm256_loadu_si256(z1p.add(i));
            let sv = _mm256_loadu_si256(sp.add(i));
            let zxor = _mm256_xor_si256(zv0, zv1);
            let flip = _mm256_and_si256(_mm256_and_si256(xv0, xv1), zxor);
            _mm256_storeu_si256(sp.add(i), _mm256_xor_si256(sv, flip));
            _mm256_storeu_si256(z0p.add(i), _mm256_xor_si256(zv0, xv1));
            _mm256_storeu_si256(z1p.add(i), _mm256_xor_si256(zv1, xv0));
        }
        let tail = chunks * 4;
        for w in tail..m_words {
            let xv0 = *x0.get_unchecked(w);
            let zv0 = *z0.get_unchecked(w);
            let xv1 = *x1.get_unchecked(w);
            let zv1 = *z1.get_unchecked(w);
            *sign.get_unchecked_mut(w) ^= xv0 & xv1 & (zv0 ^ zv1);
            *z0.get_unchecked_mut(w) = zv0 ^ xv1;
            *z1.get_unchecked_mut(w) = zv1 ^ xv0;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_swap_avx2(
    x0: &mut [u64],
    z0: &mut [u64],
    x1: &mut [u64],
    z1: &mut [u64],
    m_words: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let chunks = m_words / 4;
        let x0p = x0.as_mut_ptr() as *mut __m256i;
        let z0p = z0.as_mut_ptr() as *mut __m256i;
        let x1p = x1.as_mut_ptr() as *mut __m256i;
        let z1p = z1.as_mut_ptr() as *mut __m256i;
        for i in 0..chunks {
            let xv0 = _mm256_loadu_si256(x0p.add(i));
            let xv1 = _mm256_loadu_si256(x1p.add(i));
            _mm256_storeu_si256(x0p.add(i), xv1);
            _mm256_storeu_si256(x1p.add(i), xv0);
            let zv0 = _mm256_loadu_si256(z0p.add(i));
            let zv1 = _mm256_loadu_si256(z1p.add(i));
            _mm256_storeu_si256(z0p.add(i), zv1);
            _mm256_storeu_si256(z1p.add(i), zv0);
        }
        let tail = chunks * 4;
        for w in tail..m_words {
            let tmp = *x0.get_unchecked(w);
            *x0.get_unchecked_mut(w) = *x1.get_unchecked(w);
            *x1.get_unchecked_mut(w) = tmp;
            let tmp = *z0.get_unchecked(w);
            *z0.get_unchecked_mut(w) = *z1.get_unchecked(w);
            *z1.get_unchecked_mut(w) = tmp;
        }
    }
}

/// Visit `$m_words` in eight-word steps, binding the word offset and a lane mask that is
/// all ones except on the last partial step.
#[cfg(target_arch = "x86_64")]
macro_rules! for_each_word_oct {
    ($m_words:expr, |$off:ident, $k:ident| $body:block) => {{
        let m_words: usize = $m_words;
        let full = m_words & !7;
        let mut $off = 0usize;
        while $off < full {
            let $k: std::arch::x86_64::__mmask8 = 0xFF;
            $body
            $off += 8;
        }
        if $off < m_words {
            let $k: std::arch::x86_64::__mmask8 = (1u8 << (m_words - $off)) - 1;
            $body
        }
    }};
}

/// Load eight words at `p`, or the lanes `k` selects when it is not all ones.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn load_words(p: *const u64, k: std::arch::x86_64::__mmask8) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        if k == 0xFF {
            _mm512_loadu_si512(p as *const __m512i)
        } else {
            _mm512_maskz_loadu_epi64(k, p as *const i64)
        }
    }
}

/// Store eight words at `p`, or the lanes `k` selects when it is not all ones.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn store_words(p: *mut u64, k: std::arch::x86_64::__mmask8, v: std::arch::x86_64::__m512i) {
    use std::arch::x86_64::*;
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        if k == 0xFF {
            _mm512_storeu_si512(p as *mut __m512i, v);
        } else {
            _mm512_mask_storeu_epi64(p as *mut i64, k, v);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batch_propagate_h_avx512(
    xq: &mut [u64],
    zq: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let (xp, zp, sp) = (xq.as_mut_ptr(), zq.as_mut_ptr(), sign.as_mut_ptr());
        for_each_word_oct!(m_words, |off, k| {
            let xv = load_words(xp.add(off), k);
            let zv = load_words(zp.add(off), k);
            let sv = load_words(sp.add(off), k);
            store_words(
                sp.add(off),
                k,
                _mm512_ternarylogic_epi64::<0x78>(sv, xv, zv),
            );
            store_words(xp.add(off), k, zv);
            store_words(zp.add(off), k, xv);
        });
    }
}

/// `S` when `negate_z`, `Sdg` otherwise: the sign takes `x & !z` or `x & z`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batch_propagate_s_avx512(
    xq: &mut [u64],
    zq: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
    negate_z: bool,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let (xp, zp, sp) = (xq.as_ptr(), zq.as_mut_ptr(), sign.as_mut_ptr());
        if negate_z {
            for_each_word_oct!(m_words, |off, k| {
                let xv = load_words(xp.add(off), k);
                let zv = load_words(zp.add(off), k);
                let sv = load_words(sp.add(off), k);
                store_words(
                    sp.add(off),
                    k,
                    _mm512_ternarylogic_epi64::<0xB4>(sv, xv, zv),
                );
                store_words(zp.add(off), k, _mm512_xor_si512(zv, xv));
            });
        } else {
            for_each_word_oct!(m_words, |off, k| {
                let xv = load_words(xp.add(off), k);
                let zv = load_words(zp.add(off), k);
                let sv = load_words(sp.add(off), k);
                store_words(
                    sp.add(off),
                    k,
                    _mm512_ternarylogic_epi64::<0x78>(sv, xv, zv),
                );
                store_words(zp.add(off), k, _mm512_xor_si512(zv, xv));
            });
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batch_propagate_sign_xor2_avx512(dst: &mut [u64], a: &[u64], b: &[u64], m_words: usize) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let (dp, ap, bp) = (dst.as_mut_ptr(), a.as_ptr(), b.as_ptr());
        for_each_word_oct!(m_words, |off, k| {
            let dv = load_words(dp.add(off), k);
            let av = load_words(ap.add(off), k);
            let bv = load_words(bp.add(off), k);
            store_words(
                dp.add(off),
                k,
                _mm512_ternarylogic_epi64::<0x96>(dv, av, bv),
            );
        });
    }
}

/// `SXdg` when `negate_x`, `SX` otherwise: the sign takes `!x & z` or `x & z`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batch_propagate_sx_avx512(
    xq: &mut [u64],
    zq: &[u64],
    sign: &mut [u64],
    m_words: usize,
    negate_x: bool,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let (xp, zp, sp) = (xq.as_mut_ptr(), zq.as_ptr(), sign.as_mut_ptr());
        if negate_x {
            for_each_word_oct!(m_words, |off, k| {
                let xv = load_words(xp.add(off), k);
                let zv = load_words(zp.add(off), k);
                let sv = load_words(sp.add(off), k);
                store_words(
                    sp.add(off),
                    k,
                    _mm512_ternarylogic_epi64::<0xD2>(sv, xv, zv),
                );
                store_words(xp.add(off), k, _mm512_xor_si512(xv, zv));
            });
        } else {
            for_each_word_oct!(m_words, |off, k| {
                let xv = load_words(xp.add(off), k);
                let zv = load_words(zp.add(off), k);
                let sv = load_words(sp.add(off), k);
                store_words(
                    sp.add(off),
                    k,
                    _mm512_ternarylogic_epi64::<0x78>(sv, xv, zv),
                );
                store_words(xp.add(off), k, _mm512_xor_si512(xv, zv));
            });
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batch_propagate_cx_avx512(
    x_ctrl: &[u64],
    z_ctrl: &mut [u64],
    x_tgt: &mut [u64],
    z_tgt: &[u64],
    sign: &mut [u64],
    m_words: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let (xcp, zcp) = (x_ctrl.as_ptr(), z_ctrl.as_mut_ptr());
        let (xtp, ztp) = (x_tgt.as_mut_ptr(), z_tgt.as_ptr());
        let sp = sign.as_mut_ptr();
        for_each_word_oct!(m_words, |off, k| {
            let xc = load_words(xcp.add(off), k);
            let zc = load_words(zcp.add(off), k);
            let xt = load_words(xtp.add(off), k);
            let zt = load_words(ztp.add(off), k);
            let sv = load_words(sp.add(off), k);
            let xc_eq = _mm512_ternarylogic_epi64::<0x90>(xc, zc, xt);
            store_words(
                sp.add(off),
                k,
                _mm512_ternarylogic_epi64::<0x78>(sv, xc_eq, zt),
            );
            store_words(xtp.add(off), k, _mm512_xor_si512(xt, xc));
            store_words(zcp.add(off), k, _mm512_xor_si512(zc, zt));
        });
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batch_propagate_cz_avx512(
    x0: &[u64],
    z0: &mut [u64],
    x1: &[u64],
    z1: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let (x0p, z0p, x1p, z1p) = (x0.as_ptr(), z0.as_mut_ptr(), x1.as_ptr(), z1.as_mut_ptr());
        let sp = sign.as_mut_ptr();
        for_each_word_oct!(m_words, |off, k| {
            let xv0 = load_words(x0p.add(off), k);
            let zv0 = load_words(z0p.add(off), k);
            let xv1 = load_words(x1p.add(off), k);
            let zv1 = load_words(z1p.add(off), k);
            let sv = load_words(sp.add(off), k);
            let x0_zne = _mm512_ternarylogic_epi64::<0x60>(xv0, zv0, zv1);
            store_words(
                sp.add(off),
                k,
                _mm512_ternarylogic_epi64::<0x78>(sv, x0_zne, xv1),
            );
            store_words(z0p.add(off), k, _mm512_xor_si512(zv0, xv1));
            store_words(z1p.add(off), k, _mm512_xor_si512(zv1, xv0));
        });
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn batch_propagate_swap_avx512(
    x0: &mut [u64],
    z0: &mut [u64],
    x1: &mut [u64],
    z1: &mut [u64],
    m_words: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let (x0p, z0p, x1p, z1p) = (
            x0.as_mut_ptr(),
            z0.as_mut_ptr(),
            x1.as_mut_ptr(),
            z1.as_mut_ptr(),
        );
        for_each_word_oct!(m_words, |off, k| {
            let xv0 = load_words(x0p.add(off), k);
            let xv1 = load_words(x1p.add(off), k);
            store_words(x0p.add(off), k, xv1);
            store_words(x1p.add(off), k, xv0);
            let zv0 = load_words(z0p.add(off), k);
            let zv1 = load_words(z1p.add(off), k);
            store_words(z0p.add(off), k, zv1);
            store_words(z1p.add(off), k, zv0);
        });
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_h_neon(xq: &mut [u64], zq: &mut [u64], sign: &mut [u64], m_words: usize) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;
        let chunks = m_words / 2;
        let xp = xq.as_mut_ptr();
        let zp = zq.as_mut_ptr();
        let sp = sign.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 2;
            let xv = vld1q_u64(xp.add(off));
            let zv = vld1q_u64(zp.add(off));
            let sv = vld1q_u64(sp.add(off));
            vst1q_u64(sp.add(off), veorq_u64(sv, vandq_u64(xv, zv)));
            vst1q_u64(xp.add(off), zv);
            vst1q_u64(zp.add(off), xv);
        }
        let tail = chunks * 2;
        for w in tail..m_words {
            *sign.get_unchecked_mut(w) ^= *xq.get_unchecked(w) & *zq.get_unchecked(w);
        }
        for w in tail..m_words {
            let tmp = *xq.get_unchecked(w);
            *xq.get_unchecked_mut(w) = *zq.get_unchecked(w);
            *zq.get_unchecked_mut(w) = tmp;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_s_neon(
    xq: &mut [u64],
    zq: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
    negate_z: bool,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;
        let chunks = m_words / 2;
        let xp = xq.as_ptr();
        let zp = zq.as_mut_ptr();
        let sp = sign.as_mut_ptr();
        if negate_z {
            for i in 0..chunks {
                let off = i * 2;
                let xv = vld1q_u64(xp.add(off));
                let zv = vld1q_u64(zp.add(off));
                let sv = vld1q_u64(sp.add(off));
                vst1q_u64(sp.add(off), veorq_u64(sv, vbicq_u64(xv, zv)));
                vst1q_u64(zp.add(off), veorq_u64(zv, xv));
            }
            let tail = chunks * 2;
            for w in tail..m_words {
                *sign.get_unchecked_mut(w) ^= *xq.get_unchecked(w) & !*zq.get_unchecked(w);
                *zq.get_unchecked_mut(w) ^= *xq.get_unchecked(w);
            }
        } else {
            for i in 0..chunks {
                let off = i * 2;
                let xv = vld1q_u64(xp.add(off));
                let zv = vld1q_u64(zp.add(off));
                let sv = vld1q_u64(sp.add(off));
                vst1q_u64(sp.add(off), veorq_u64(sv, vandq_u64(xv, zv)));
                vst1q_u64(zp.add(off), veorq_u64(zv, xv));
            }
            let tail = chunks * 2;
            for w in tail..m_words {
                *sign.get_unchecked_mut(w) ^= *xq.get_unchecked(w) & *zq.get_unchecked(w);
                *zq.get_unchecked_mut(w) ^= *xq.get_unchecked(w);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_sign_xor2_neon(dst: &mut [u64], a: &[u64], b: &[u64], m_words: usize) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;
        let chunks = m_words / 2;
        let dp = dst.as_mut_ptr();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        for i in 0..chunks {
            let off = i * 2;
            let dv = vld1q_u64(dp.add(off));
            let av = vld1q_u64(ap.add(off));
            let bv = vld1q_u64(bp.add(off));
            vst1q_u64(dp.add(off), veorq_u64(dv, veorq_u64(av, bv)));
        }
        let tail = chunks * 2;
        for w in tail..m_words {
            *dst.get_unchecked_mut(w) ^= *a.get_unchecked(w) ^ *b.get_unchecked(w);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_sx_neon(
    xq: &mut [u64],
    zq: &[u64],
    sign: &mut [u64],
    m_words: usize,
    negate_x: bool,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;
        let chunks = m_words / 2;
        let xp = xq.as_mut_ptr();
        let zp = zq.as_ptr();
        let sp = sign.as_mut_ptr();
        if negate_x {
            for i in 0..chunks {
                let off = i * 2;
                let xv = vld1q_u64(xp.add(off));
                let zv = vld1q_u64(zp.add(off));
                let sv = vld1q_u64(sp.add(off));
                vst1q_u64(sp.add(off), veorq_u64(sv, vbicq_u64(zv, xv)));
                vst1q_u64(xp.add(off), veorq_u64(xv, zv));
            }
            let tail = chunks * 2;
            for w in tail..m_words {
                *sign.get_unchecked_mut(w) ^= !*xq.get_unchecked(w) & *zq.get_unchecked(w);
                *xq.get_unchecked_mut(w) ^= *zq.get_unchecked(w);
            }
        } else {
            for i in 0..chunks {
                let off = i * 2;
                let xv = vld1q_u64(xp.add(off));
                let zv = vld1q_u64(zp.add(off));
                let sv = vld1q_u64(sp.add(off));
                vst1q_u64(sp.add(off), veorq_u64(sv, vandq_u64(xv, zv)));
                vst1q_u64(xp.add(off), veorq_u64(xv, zv));
            }
            let tail = chunks * 2;
            for w in tail..m_words {
                *sign.get_unchecked_mut(w) ^= *xq.get_unchecked(w) & *zq.get_unchecked(w);
                *xq.get_unchecked_mut(w) ^= *zq.get_unchecked(w);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_cx_neon(
    x_ctrl: &[u64],
    z_ctrl: &mut [u64],
    x_tgt: &mut [u64],
    z_tgt: &[u64],
    sign: &mut [u64],
    m_words: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;
        let chunks = m_words / 2;
        let xcp = x_ctrl.as_ptr();
        let zcp = z_ctrl.as_mut_ptr();
        let xtp = x_tgt.as_mut_ptr();
        let ztp = z_tgt.as_ptr();
        let sp = sign.as_mut_ptr();
        let ones = vdupq_n_u64(!0u64);
        for i in 0..chunks {
            let off = i * 2;
            let xc = vld1q_u64(xcp.add(off));
            let zc = vld1q_u64(zcp.add(off));
            let xt = vld1q_u64(xtp.add(off));
            let zt = vld1q_u64(ztp.add(off));
            let sv = vld1q_u64(sp.add(off));
            let xnor = veorq_u64(veorq_u64(zc, xt), ones);
            let flip = vandq_u64(vandq_u64(xc, zt), xnor);
            vst1q_u64(sp.add(off), veorq_u64(sv, flip));
            vst1q_u64(xtp.add(off), veorq_u64(xt, xc));
            vst1q_u64(zcp.add(off), veorq_u64(zc, zt));
        }
        let tail = chunks * 2;
        for w in tail..m_words {
            let xc = *x_ctrl.get_unchecked(w);
            let zc = *z_ctrl.get_unchecked(w);
            let xt = *x_tgt.get_unchecked(w);
            let zt = *z_tgt.get_unchecked(w);
            *sign.get_unchecked_mut(w) ^= xc & zt & !(zc ^ xt);
            *x_tgt.get_unchecked_mut(w) = xt ^ xc;
            *z_ctrl.get_unchecked_mut(w) = zc ^ zt;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_cz_neon(
    x0: &[u64],
    z0: &mut [u64],
    x1: &[u64],
    z1: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;
        let chunks = m_words / 2;
        let x0p = x0.as_ptr();
        let z0p = z0.as_mut_ptr();
        let x1p = x1.as_ptr();
        let z1p = z1.as_mut_ptr();
        let sp = sign.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 2;
            let xv0 = vld1q_u64(x0p.add(off));
            let zv0 = vld1q_u64(z0p.add(off));
            let xv1 = vld1q_u64(x1p.add(off));
            let zv1 = vld1q_u64(z1p.add(off));
            let sv = vld1q_u64(sp.add(off));
            let zxor = veorq_u64(zv0, zv1);
            let flip = vandq_u64(vandq_u64(xv0, xv1), zxor);
            vst1q_u64(sp.add(off), veorq_u64(sv, flip));
            vst1q_u64(z0p.add(off), veorq_u64(zv0, xv1));
            vst1q_u64(z1p.add(off), veorq_u64(zv1, xv0));
        }
        let tail = chunks * 2;
        for w in tail..m_words {
            let xv0 = *x0.get_unchecked(w);
            let zv0 = *z0.get_unchecked(w);
            let xv1 = *x1.get_unchecked(w);
            let zv1 = *z1.get_unchecked(w);
            *sign.get_unchecked_mut(w) ^= xv0 & xv1 & (zv0 ^ zv1);
            *z0.get_unchecked_mut(w) = zv0 ^ xv1;
            *z1.get_unchecked_mut(w) = zv1 ^ xv0;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_swap_neon(
    x0: &mut [u64],
    z0: &mut [u64],
    x1: &mut [u64],
    z1: &mut [u64],
    m_words: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;
        let chunks = m_words / 2;
        let x0p = x0.as_mut_ptr();
        let z0p = z0.as_mut_ptr();
        let x1p = x1.as_mut_ptr();
        let z1p = z1.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 2;
            let xv0 = vld1q_u64(x0p.add(off));
            let xv1 = vld1q_u64(x1p.add(off));
            vst1q_u64(x0p.add(off), xv1);
            vst1q_u64(x1p.add(off), xv0);
            let zv0 = vld1q_u64(z0p.add(off));
            let zv1 = vld1q_u64(z1p.add(off));
            vst1q_u64(z0p.add(off), zv1);
            vst1q_u64(z1p.add(off), zv0);
        }
        let tail = chunks * 2;
        for w in tail..m_words {
            let tmp = *x0.get_unchecked(w);
            *x0.get_unchecked_mut(w) = *x1.get_unchecked(w);
            *x1.get_unchecked_mut(w) = tmp;
            let tmp = *z0.get_unchecked(w);
            *z0.get_unchecked_mut(w) = *z1.get_unchecked(w);
            *z1.get_unchecked_mut(w) = tmp;
        }
    }
}

#[inline(always)]
pub(crate) fn batch_propagate_backward(
    x: &mut [Vec<u64>],
    z: &mut [Vec<u64>],
    sign: &mut [u64],
    gate: &Gate,
    targets: &[usize],
    m_words: usize,
) {
    match *targets {
        [q] => propagate_rows_1q(gate, &mut x[q], &mut z[q], sign, m_words),
        [a, b] => {
            let (x0, x1) = pair_mut(x, a, b);
            let (z0, z1) = pair_mut(z, a, b);
            propagate_rows_2q(gate, x0, z0, x1, z1, sign, m_words);
        }
        _ => {}
    }
}

/// [`batch_propagate_backward`] over rows packed end to end: slot `s` holds words
/// `s * words..(s + 1) * words` of `x` and `z`.
#[inline(always)]
pub(crate) fn batch_propagate_backward_flat(
    x: &mut [u64],
    z: &mut [u64],
    sign: &mut [u64],
    words: usize,
    gate: &Gate,
    slots: &[usize],
) {
    match *slots {
        [q] => propagate_rows_1q(
            gate,
            &mut x[q * words..(q + 1) * words],
            &mut z[q * words..(q + 1) * words],
            sign,
            words,
        ),
        [a, b] => {
            let (x0, x1) = pair_rows_mut(x, a, b, words);
            let (z0, z1) = pair_rows_mut(z, a, b, words);
            propagate_rows_2q(gate, x0, z0, x1, z1, sign, words);
        }
        _ => {}
    }
}

fn pair_mut<T>(v: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    debug_assert_ne!(a, b);
    if a < b {
        let (lo, hi) = v.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = v.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

/// Rows `a` and `b` of `v`, `words` words each, as two disjoint slices.
pub(crate) fn pair_rows_mut(
    v: &mut [u64],
    a: usize,
    b: usize,
    words: usize,
) -> (&mut [u64], &mut [u64]) {
    debug_assert_ne!(a, b);
    if a < b {
        let (lo, hi) = v.split_at_mut(b * words);
        (&mut lo[a * words..(a + 1) * words], &mut hi[..words])
    } else {
        let (lo, hi) = v.split_at_mut(a * words);
        (&mut hi[..words], &mut lo[b * words..(b + 1) * words])
    }
}

/// Run the first SIMD kernel whose flag is set and return, or fall through to the
/// scalar loop after the call. Each kernel takes slices of at least `m_words` words.
macro_rules! simd_dispatch {
    ($avx512:ident: $a:expr, $avx2:ident: $b:expr, $neon:ident: $c:expr $(,)?) => {
        #[cfg(target_arch = "x86_64")]
        if $avx512 {
            // SAFETY: AVX-512F detected, slices are disjoint and valid with m_words elements
            unsafe { $a };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if $avx2 {
            // SAFETY: AVX2 detected, slices are disjoint and valid with m_words elements
            unsafe { $b };
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if $neon {
            // SAFETY: NEON is baseline on aarch64, slices are disjoint and valid with m_words elements
            unsafe { $c };
            return;
        }
    };
}

#[inline(always)]
fn propagate_rows_1q(
    gate: &Gate,
    xq: &mut [u64],
    zq: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
) {
    #[cfg(target_arch = "x86_64")]
    let tier = word_tier();
    #[cfg(target_arch = "x86_64")]
    let use_avx512 = m_words >= 8 && tier == WordTier::Avx512;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = m_words >= 4 && tier != WordTier::Scalar;
    #[cfg(target_arch = "aarch64")]
    let use_neon = m_words >= 2;

    match gate {
        Gate::H => {
            simd_dispatch!(
                use_avx512: batch_propagate_h_avx512(xq, zq, sign, m_words),
                use_avx2: batch_propagate_h_avx2(xq, zq, sign, m_words),
                use_neon: batch_propagate_h_neon(xq, zq, sign, m_words),
            );
            for w in 0..m_words {
                sign[w] ^= xq[w] & zq[w];
            }
            xq[..m_words].swap_with_slice(&mut zq[..m_words]);
        }
        Gate::S => {
            simd_dispatch!(
                use_avx512: batch_propagate_s_avx512(xq, zq, sign, m_words, true),
                use_avx2: batch_propagate_s_avx2(xq, zq, sign, m_words, true),
                use_neon: batch_propagate_s_neon(xq, zq, sign, m_words, true),
            );
            for w in 0..m_words {
                sign[w] ^= xq[w] & !zq[w];
                zq[w] ^= xq[w];
            }
        }
        Gate::Sdg => {
            simd_dispatch!(
                use_avx512: batch_propagate_s_avx512(xq, zq, sign, m_words, false),
                use_avx2: batch_propagate_s_avx2(xq, zq, sign, m_words, false),
                use_neon: batch_propagate_s_neon(xq, zq, sign, m_words, false),
            );
            for w in 0..m_words {
                sign[w] ^= xq[w] & zq[w];
                zq[w] ^= xq[w];
            }
        }
        Gate::X => xor_words(&mut sign[..m_words], &zq[..m_words]),
        Gate::Y => {
            simd_dispatch!(
                use_avx512: batch_propagate_sign_xor2_avx512(sign, xq, zq, m_words),
                use_avx2: batch_propagate_sign_xor2_avx2(sign, xq, zq, m_words),
                use_neon: batch_propagate_sign_xor2_neon(sign, xq, zq, m_words),
            );
            for w in 0..m_words {
                sign[w] ^= xq[w] ^ zq[w];
            }
        }
        Gate::Z => xor_words(&mut sign[..m_words], &xq[..m_words]),
        Gate::SX => {
            simd_dispatch!(
                use_avx512: batch_propagate_sx_avx512(xq, zq, sign, m_words, false),
                use_avx2: batch_propagate_sx_avx2(xq, zq, sign, m_words, false),
                use_neon: batch_propagate_sx_neon(xq, zq, sign, m_words, false),
            );
            for w in 0..m_words {
                sign[w] ^= xq[w] & zq[w];
                xq[w] ^= zq[w];
            }
        }
        Gate::SXdg => {
            simd_dispatch!(
                use_avx512: batch_propagate_sx_avx512(xq, zq, sign, m_words, true),
                use_avx2: batch_propagate_sx_avx2(xq, zq, sign, m_words, true),
                use_neon: batch_propagate_sx_neon(xq, zq, sign, m_words, true),
            );
            for w in 0..m_words {
                sign[w] ^= !xq[w] & zq[w];
                xq[w] ^= zq[w];
            }
        }
        _ => {}
    }
}

#[inline(always)]
fn propagate_rows_2q(
    gate: &Gate,
    x0: &mut [u64],
    z0: &mut [u64],
    x1: &mut [u64],
    z1: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
) {
    #[cfg(target_arch = "x86_64")]
    let tier = word_tier();
    #[cfg(target_arch = "x86_64")]
    let use_avx512 = m_words >= 8 && tier == WordTier::Avx512;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = m_words >= 4 && tier != WordTier::Scalar;
    #[cfg(target_arch = "aarch64")]
    let use_neon = m_words >= 2;

    match gate {
        Gate::Cx => {
            simd_dispatch!(
                use_avx512: batch_propagate_cx_avx512(x0, z0, x1, z1, sign, m_words),
                use_avx2: batch_propagate_cx_avx2(x0, z0, x1, z1, sign, m_words),
                use_neon: batch_propagate_cx_neon(x0, z0, x1, z1, sign, m_words),
            );
            for w in 0..m_words {
                sign[w] ^= x0[w] & z1[w] & !(z0[w] ^ x1[w]);
                x1[w] ^= x0[w];
                z0[w] ^= z1[w];
            }
        }
        Gate::Cz => {
            simd_dispatch!(
                use_avx512: batch_propagate_cz_avx512(x0, z0, x1, z1, sign, m_words),
                use_avx2: batch_propagate_cz_avx2(x0, z0, x1, z1, sign, m_words),
                use_neon: batch_propagate_cz_neon(x0, z0, x1, z1, sign, m_words),
            );
            for w in 0..m_words {
                sign[w] ^= x0[w] & x1[w] & (z0[w] ^ z1[w]);
                z0[w] ^= x1[w];
                z1[w] ^= x0[w];
            }
        }
        Gate::Swap => {
            simd_dispatch!(
                use_avx512: batch_propagate_swap_avx512(x0, z0, x1, z1, m_words),
                use_avx2: batch_propagate_swap_avx2(x0, z0, x1, z1, m_words),
                use_neon: batch_propagate_swap_neon(x0, z0, x1, z1, m_words),
            );
            x0[..m_words].swap_with_slice(&mut x1[..m_words]);
            z0[..m_words].swap_with_slice(&mut z1[..m_words]);
        }
        _ => {}
    }
}

/// Multiply the row `src` into the row `dst`, both laid out `[x(nw) | z(nw)]`,
/// returning the phase of the product.
#[inline(always)]
pub(super) fn rowmul_phase(
    src: &[u64],
    dst: &mut [u64],
    nw: usize,
    src_phase: bool,
    dst_phase: bool,
) -> bool {
    let initial_sum = if src_phase { 2u64 } else { 0 } + if dst_phase { 2u64 } else { 0 };
    let (src_x, src_z) = src.split_at(nw);
    let (dst_x, dst_z) = dst.split_at_mut(nw);
    (rowmul_words_cached(dst_x, dst_z, src_x, src_z, initial_sum) & 3) >= 2
}

/// Whether `pauli` anticommutes with a tableau row laid out as `[x(nw) | z(nw)]`.
#[inline(always)]
fn anticommutes(pauli: &PauliVec, row: &[u64], nw: usize) -> bool {
    let (row_x, row_z) = row.split_at(nw);
    let inner = pauli.x[..nw]
        .iter()
        .zip(&pauli.z[..nw])
        .zip(row_x.iter().zip(row_z))
        .fold(0u64, |acc, ((&px, &pz), (&rx, &rz))| {
            acc ^ (px & rz) ^ (pz & rx)
        });
    inner.count_ones() % 2 == 1
}

/// Compute reference measurement outcomes by simulating measurements of the
/// propagated Paulis on |0⟩^n using the Aaronson-Gottesman measurement protocol.
/// O(m x n^2/64), avoids re-simulating all T circuit gates through the stabilizer.
#[allow(clippy::needless_range_loop)]
pub(super) fn compute_reference_bits(
    measurement_rows: &[(PauliVec, usize, bool)],
    n: usize,
) -> Vec<bool> {
    let m = measurement_rows.len();
    let nw = n.div_ceil(64);
    let stride = 2 * nw;

    let mut xz = vec![0u64; 2 * n * stride];
    let mut phase = vec![false; 2 * n];

    for i in 0..n {
        xz[i * stride + i / 64] |= 1u64 << (i % 64);
        xz[(n + i) * stride + nw + i / 64] |= 1u64 << (i % 64);
    }

    let mut ref_bits = vec![false; m];
    let mut p_data = vec![0u64; stride];
    let mut scratch = vec![0u64; stride];

    for (meas_idx, (pauli, _, _)) in measurement_rows.iter().enumerate() {
        let mut anti_idx = None;
        for g in n..2 * n {
            if anticommutes(pauli, &xz[g * stride..(g + 1) * stride], nw) {
                anti_idx = Some(g);
                break;
            }
        }

        match anti_idx {
            Some(p) => {
                p_data.copy_from_slice(&xz[p * stride..(p + 1) * stride]);
                let p_phase = phase[p];

                for r in 0..2 * n {
                    if r == p {
                        continue;
                    }
                    let r_base = r * stride;
                    if anticommutes(pauli, &xz[r_base..r_base + stride], nw) {
                        phase[r] = rowmul_phase(
                            &p_data,
                            &mut xz[r_base..r_base + stride],
                            nw,
                            p_phase,
                            phase[r],
                        );
                    }
                }

                let dest_idx = p - n;
                let dest_base = dest_idx * stride;
                xz.copy_within(p * stride..(p + 1) * stride, dest_base);
                phase[dest_idx] = p_phase;

                let p_base = p * stride;
                for w in 0..nw {
                    xz[p_base + w] = pauli.x[w];
                    xz[p_base + nw + w] = pauli.z[w];
                }
                phase[p] = false;

                ref_bits[meas_idx] = false;
            }
            None => {
                scratch.fill(0);
                let mut scratch_phase = false;

                for g in 0..n {
                    if anticommutes(pauli, &xz[g * stride..(g + 1) * stride], nw) {
                        let s_base = (g + n) * stride;
                        let s_phase = phase[g + n];
                        scratch_phase = rowmul_phase(
                            &xz[s_base..s_base + stride],
                            &mut scratch,
                            nw,
                            s_phase,
                            scratch_phase,
                        );
                    }
                }

                ref_bits[meas_idx] = scratch_phase;
            }
        }
    }

    ref_bits
}

/// Forward conjugation `G P G†` is the backward step by `G†`.
pub(super) fn forward_as_backward_gate(gate: &Gate) -> Result<&Gate> {
    match gate {
        Gate::S => Ok(&Gate::Sdg),
        Gate::Sdg => Ok(&Gate::S),
        Gate::SX => Ok(&Gate::SXdg),
        Gate::SXdg => Ok(&Gate::SX),
        Gate::H | Gate::X | Gate::Y | Gate::Z | Gate::Id | Gate::Cx | Gate::Cz | Gate::Swap => {
            Ok(gate)
        }
        _ => Err(incompatible(format!(
            "unsupported gate {:?} in column-major forward sim",
            gate
        ))),
    }
}

/// [`batch_propagate_backward_flat`] restricted to words `lo..hi` of each column and of
/// `phase`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn propagate_col_words(
    x: &mut [u64],
    z: &mut [u64],
    phase: &mut [u64],
    row_words: usize,
    gate: &Gate,
    cols: &[usize],
    lo: usize,
    hi: usize,
) {
    let sign = &mut phase[lo..hi];
    match *cols {
        [q] => {
            let base = q * row_words;
            propagate_rows_1q(
                gate,
                &mut x[base + lo..base + hi],
                &mut z[base + lo..base + hi],
                sign,
                hi - lo,
            );
        }
        [a, b] => {
            let (x0, x1) = pair_rows_mut(x, a, b, row_words);
            let (z0, z1) = pair_rows_mut(z, a, b, row_words);
            propagate_rows_2q(
                gate,
                &mut x0[lo..hi],
                &mut z0[lo..hi],
                &mut x1[lo..hi],
                &mut z1[lo..hi],
                sign,
                hi - lo,
            );
        }
        _ => {}
    }
}

/// Column-major forward stabilizer simulation.
///
/// Stores x_cols[qubit][row_word] and z_cols[qubit][row_word] so gate
/// application is O(row_words) vector XOR instead of O(2n) per-row bit ops.
///
/// Returns (xz_rowmajor, phase, nw) transposed back to row-major.
pub(super) fn colmajor_forward_sim(
    n: usize,
    instructions: &[Instruction],
) -> Result<(Vec<u64>, Vec<bool>, usize)> {
    let total_rows = 2 * n;
    let nw = n.div_ceil(64);
    let row_words = total_rows.div_ceil(64);

    let mut x_cols = vec![0u64; n * row_words];
    let mut z_cols = vec![0u64; n * row_words];
    let mut phase: Vec<u64> = vec![0u64; row_words];

    for q in 0..n {
        x_cols[q * row_words + q / 64] |= 1u64 << (q % 64);
        let stab_row = n + q;
        z_cols[q * row_words + stab_row / 64] |= 1u64 << (stab_row % 64);
    }

    // Qubits at or above `touched` still hold their initial rows, so every column is
    // zero outside destabilizer rows [0, touched) and stabilizer rows [n, n + touched).
    let mut touched = 0usize;
    let stab_start = n / 64;
    for inst in instructions {
        let (gate, targets) = match inst {
            Instruction::Gate { gate, targets } => (gate, targets.as_slice()),
            Instruction::Conditional { gate, targets, .. } => (gate, targets.as_slice()),
            _ => continue,
        };
        let gate = forward_as_backward_gate(gate)?;
        for &t in targets {
            touched = touched.max(t + 1);
        }
        let destab_end = touched.div_ceil(64);
        let stab_end = (n + touched).div_ceil(64);
        if stab_start <= destab_end {
            propagate_col_words(
                &mut x_cols,
                &mut z_cols,
                &mut phase,
                row_words,
                gate,
                targets,
                0,
                stab_end,
            );
        } else {
            propagate_col_words(
                &mut x_cols,
                &mut z_cols,
                &mut phase,
                row_words,
                gate,
                targets,
                0,
                destab_end,
            );
            propagate_col_words(
                &mut x_cols,
                &mut z_cols,
                &mut phase,
                row_words,
                gate,
                targets,
                stab_start,
                stab_end,
            );
        }
    }

    let stride = 2 * nw;
    let mut xz = vec![0u64; total_rows * stride];
    let mut phase_vec = vec![false; total_rows];

    let mut block = [0u64; 64];
    for (cols, offset) in [(x_cols, 0), (z_cols, nw)] {
        for qw in 0..nw {
            let qubits = &cols[qw * 64 * row_words..n.min(qw * 64 + 64) * row_words];
            let count = qubits.len() / row_words;
            for rw in 0..row_words {
                let mut any = 0u64;
                for (b, col) in qubits.chunks_exact(row_words).enumerate() {
                    block[b] = col[rw];
                    any |= col[rw];
                }
                if any == 0 {
                    continue;
                }
                block[count..].fill(0);
                transpose_64x64_dispatch(&mut block);
                let rows = (total_rows - rw * 64).min(64);
                for (i, &word) in block[..rows].iter().enumerate() {
                    xz[(rw * 64 + i) * stride + offset + qw] = word;
                }
            }
        }
    }

    for (rw, &pw) in phase.iter().enumerate().take(row_words) {
        let mut pbits = pw;
        while pbits != 0 {
            let bit = pbits.trailing_zeros() as usize;
            let row = rw * 64 + bit;
            if row < total_rows {
                phase_vec[row] = true;
            }
            pbits &= pbits - 1;
        }
    }

    Ok((xz, phase_vec, nw))
}

#[cfg(test)]
#[path = "propagation_tests.rs"]
mod tests;

// For every `a != b` among up to four rows of up to two words: no panic, and the rows
// returned are rows `a` and `b`, in bounds and disjoint.
#[cfg(kani)]
mod pair_proofs {
    use super::{pair_mut, pair_rows_mut};

    fn any_below(n: usize) -> usize {
        let v: usize = kani::any();
        kani::assume(v < n);
        v
    }

    fn word_offset(base: *const u64, p: *const u64) -> usize {
        (p as usize - base as usize) / size_of::<u64>()
    }

    #[kani::proof]
    fn pair_mut_returns_disjoint_rows_in_bounds() {
        let mut v = [0u64; 4];
        let rows = any_below(5);
        let (a, b) = (any_below(rows), any_below(rows));
        kani::assume(a != b);
        let base = v.as_ptr();
        let (ra, rb) = pair_mut(&mut v[..rows], a, b);
        let (oa, ob) = (word_offset(base, ra), word_offset(base, rb));
        assert_eq!((oa, ob), (a, b));
        assert!(oa < rows && ob < rows && oa != ob);
    }

    #[kani::proof]
    fn pair_rows_mut_returns_disjoint_rows_in_bounds() {
        let mut v = [0u64; 8];
        let (rows, words) = (any_below(5), any_below(3));
        let (a, b) = (any_below(rows), any_below(rows));
        kani::assume(a != b);
        let base = v.as_ptr();
        let (ra, rb) = pair_rows_mut(&mut v[..rows * words], a, b, words);
        let (oa, ob) = (
            word_offset(base, ra.as_ptr()),
            word_offset(base, rb.as_ptr()),
        );
        assert_eq!((oa, ra.len()), (a * words, words));
        assert_eq!((ob, rb.len()), (b * words, words));
        assert!(oa + words <= rows * words && ob + words <= rows * words);
        assert!(oa + words <= ob || ob + words <= oa);
    }
}

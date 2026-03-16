use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::ShotsResult;
use rand::RngCore;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone)]
pub struct PauliVec {
    pub x: Vec<u64>,
    pub z: Vec<u64>,
}

impl PauliVec {
    pub fn new(num_words: usize) -> Self {
        Self {
            x: vec![0u64; num_words],
            z: vec![0u64; num_words],
        }
    }

    pub fn z_on_qubit(num_words: usize, qubit: usize) -> Self {
        let mut pv = Self::new(num_words);
        pv.z[qubit / 64] |= 1u64 << (qubit % 64);
        pv
    }
}

#[inline(always)]
fn get_bit(words: &[u64], qubit: usize) -> bool {
    (words[qubit / 64] >> (qubit % 64)) & 1 != 0
}

#[inline(always)]
fn set_bit(words: &mut [u64], qubit: usize, val: bool) {
    let word = qubit / 64;
    let bit = qubit % 64;
    if val {
        words[word] |= 1u64 << bit;
    } else {
        words[word] &= !(1u64 << bit);
    }
}

#[inline(always)]
fn flip_bit(words: &mut [u64], qubit: usize) {
    words[qubit / 64] ^= 1u64 << (qubit % 64);
}

/// Heisenberg-picture backward propagation of a Pauli through a Clifford gate.
///
/// Given P, computes U†·P·U where U is the gate.
pub fn propagate_backward(pauli: &mut PauliVec, gate: &Gate, targets: &[usize]) {
    match gate {
        Gate::H => {
            let q = targets[0];
            let xb = get_bit(&pauli.x, q);
            let zb = get_bit(&pauli.z, q);
            set_bit(&mut pauli.x, q, zb);
            set_bit(&mut pauli.z, q, xb);
        }
        Gate::S => {
            let q = targets[0];
            if get_bit(&pauli.x, q) {
                flip_bit(&mut pauli.z, q);
            }
        }
        Gate::Sdg => {
            let q = targets[0];
            if get_bit(&pauli.x, q) {
                flip_bit(&mut pauli.z, q);
            }
        }
        Gate::X => {}
        Gate::Y => {}
        Gate::Z => {}
        Gate::SX => {
            let q = targets[0];
            if get_bit(&pauli.z, q) {
                flip_bit(&mut pauli.x, q);
            }
        }
        Gate::SXdg => {
            let q = targets[0];
            if get_bit(&pauli.z, q) {
                flip_bit(&mut pauli.x, q);
            }
        }
        Gate::Cx => {
            let ctrl = targets[0];
            let tgt = targets[1];
            if get_bit(&pauli.x, ctrl) {
                flip_bit(&mut pauli.x, tgt);
            }
            if get_bit(&pauli.z, tgt) {
                flip_bit(&mut pauli.z, ctrl);
            }
        }
        Gate::Cz => {
            let q0 = targets[0];
            let q1 = targets[1];
            let x0 = get_bit(&pauli.x, q0);
            let x1 = get_bit(&pauli.x, q1);
            if x1 {
                flip_bit(&mut pauli.z, q0);
            }
            if x0 {
                flip_bit(&mut pauli.z, q1);
            }
        }
        Gate::Swap => {
            let q0 = targets[0];
            let q1 = targets[1];
            let x0 = get_bit(&pauli.x, q0);
            let x1 = get_bit(&pauli.x, q1);
            set_bit(&mut pauli.x, q0, x1);
            set_bit(&mut pauli.x, q1, x0);
            let z0 = get_bit(&pauli.z, q0);
            let z1 = get_bit(&pauli.z, q1);
            set_bit(&mut pauli.z, q0, z1);
            set_bit(&mut pauli.z, q1, z0);
        }
        Gate::Id => {}
        _ => {}
    }
}

/// Batch backward-propagate Z_q through all gates for every measurement simultaneously.
///
/// Transposed representation: each qubit's m measurement x/z bits are packed into
/// m/64 u64 words, so gate propagation is bulk bitwise ops (64× fewer operations).
///
/// Returns (PauliVec, classical_bit, sign) per measurement.
fn build_measurement_rows(circuit: &Circuit) -> Vec<(PauliVec, usize, bool)> {
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

    let gate_instructions: Vec<(&Gate, &[usize])> = circuit
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Gate { gate, targets } => Some((gate, targets.as_slice())),
            Instruction::Conditional { gate, targets, .. } => Some((gate, targets.as_slice())),
            _ => None,
        })
        .collect();

    for &(gate, targets) in gate_instructions.iter().rev() {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_s_avx2(
    xq: &mut [u64],
    zq: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
    negate_z: bool,
) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_sign_xor_avx2(dst: &mut [u64], src: &[u64], m_words: usize) {
    use std::arch::x86_64::*;
    let chunks = m_words / 4;
    let dp = dst.as_mut_ptr() as *mut __m256i;
    let sp = src.as_ptr() as *const __m256i;
    for i in 0..chunks {
        let dv = _mm256_loadu_si256(dp.add(i));
        let sv = _mm256_loadu_si256(sp.add(i));
        _mm256_storeu_si256(dp.add(i), _mm256_xor_si256(dv, sv));
    }
    let tail = chunks * 4;
    for w in tail..m_words {
        *dst.get_unchecked_mut(w) ^= *src.get_unchecked(w);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_sign_xor2_avx2(dst: &mut [u64], a: &[u64], b: &[u64], m_words: usize) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_propagate_sx_avx2(
    xq: &mut [u64],
    zq: &[u64],
    sign: &mut [u64],
    m_words: usize,
    negate_x: bool,
) {
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
        let xnor = _mm256_andnot_si256(_mm256_xor_si256(zv0, zv1), _mm256_set1_epi64x(-1));
        let flip = _mm256_and_si256(_mm256_and_si256(xv0, xv1), xnor);
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
        *sign.get_unchecked_mut(w) ^= xv0 & xv1 & !(zv0 ^ zv1);
        *z0.get_unchecked_mut(w) = zv0 ^ xv1;
        *z1.get_unchecked_mut(w) = zv1 ^ xv0;
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

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_h_neon(xq: &mut [u64], zq: &mut [u64], sign: &mut [u64], m_words: usize) {
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

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_s_neon(
    xq: &mut [u64],
    zq: &mut [u64],
    sign: &mut [u64],
    m_words: usize,
    negate_z: bool,
) {
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

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_sign_xor_neon(dst: &mut [u64], src: &[u64], m_words: usize) {
    use std::arch::aarch64::*;
    let chunks = m_words / 2;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    for i in 0..chunks {
        let off = i * 2;
        let dv = vld1q_u64(dp.add(off));
        let sv = vld1q_u64(sp.add(off));
        vst1q_u64(dp.add(off), veorq_u64(dv, sv));
    }
    let tail = chunks * 2;
    for w in tail..m_words {
        *dst.get_unchecked_mut(w) ^= *src.get_unchecked(w);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_sign_xor2_neon(dst: &mut [u64], a: &[u64], b: &[u64], m_words: usize) {
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

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn batch_propagate_sx_neon(
    xq: &mut [u64],
    zq: &[u64],
    sign: &mut [u64],
    m_words: usize,
    negate_x: bool,
) {
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
    use std::arch::aarch64::*;
    let chunks = m_words / 2;
    let x0p = x0.as_ptr();
    let z0p = z0.as_mut_ptr();
    let x1p = x1.as_ptr();
    let z1p = z1.as_mut_ptr();
    let sp = sign.as_mut_ptr();
    let ones = vdupq_n_u64(!0u64);
    for i in 0..chunks {
        let off = i * 2;
        let xv0 = vld1q_u64(x0p.add(off));
        let zv0 = vld1q_u64(z0p.add(off));
        let xv1 = vld1q_u64(x1p.add(off));
        let zv1 = vld1q_u64(z1p.add(off));
        let sv = vld1q_u64(sp.add(off));
        let xnor = veorq_u64(veorq_u64(zv0, zv1), ones);
        let flip = vandq_u64(vandq_u64(xv0, xv1), xnor);
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
        *sign.get_unchecked_mut(w) ^= xv0 & xv1 & !(zv0 ^ zv1);
        *z0.get_unchecked_mut(w) = zv0 ^ xv1;
        *z1.get_unchecked_mut(w) = zv1 ^ xv0;
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

#[inline(always)]
pub(crate) fn batch_propagate_backward(
    x: &mut [Vec<u64>],
    z: &mut [Vec<u64>],
    sign: &mut [u64],
    gate: &Gate,
    targets: &[usize],
    m_words: usize,
) {
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = m_words >= 4 && is_x86_feature_detected!("avx2");
    #[cfg(target_arch = "aarch64")]
    let use_neon = m_words >= 2;

    match gate {
        Gate::H => {
            let q = targets[0];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: AVX2 detected, slices are valid with m_words elements
                unsafe { batch_propagate_h_avx2(&mut x[q], &mut z[q], sign, m_words) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                // SAFETY: NEON is baseline on aarch64, slices are valid with m_words elements
                unsafe { batch_propagate_h_neon(&mut x[q], &mut z[q], sign, m_words) };
                return;
            }
            for w in 0..m_words {
                sign[w] ^= x[q][w] & z[q][w];
            }
            std::mem::swap(&mut x[q], &mut z[q]);
        }
        Gate::S => {
            let q = targets[0];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: AVX2 detected, slices are valid with m_words elements
                unsafe { batch_propagate_s_avx2(&mut x[q], &mut z[q], sign, m_words, true) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                // SAFETY: NEON is baseline on aarch64, slices are valid with m_words elements
                unsafe { batch_propagate_s_neon(&mut x[q], &mut z[q], sign, m_words, true) };
                return;
            }
            for w in 0..m_words {
                sign[w] ^= x[q][w] & !z[q][w];
                z[q][w] ^= x[q][w];
            }
        }
        Gate::Sdg => {
            let q = targets[0];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: AVX2 detected, slices are valid with m_words elements
                unsafe { batch_propagate_s_avx2(&mut x[q], &mut z[q], sign, m_words, false) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                // SAFETY: NEON is baseline on aarch64, slices are valid with m_words elements
                unsafe { batch_propagate_s_neon(&mut x[q], &mut z[q], sign, m_words, false) };
                return;
            }
            for w in 0..m_words {
                sign[w] ^= x[q][w] & z[q][w];
                z[q][w] ^= x[q][w];
            }
        }
        Gate::X => {
            let q = targets[0];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: AVX2 detected, slices are valid with m_words elements
                unsafe { batch_propagate_sign_xor_avx2(sign, &z[q], m_words) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                // SAFETY: NEON is baseline on aarch64, slices are valid with m_words elements
                unsafe { batch_propagate_sign_xor_neon(sign, &z[q], m_words) };
                return;
            }
            for w in 0..m_words {
                sign[w] ^= z[q][w];
            }
        }
        Gate::Y => {
            let q = targets[0];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: AVX2 detected, slices are valid with m_words elements
                unsafe { batch_propagate_sign_xor2_avx2(sign, &x[q], &z[q], m_words) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                // SAFETY: NEON is baseline on aarch64, slices are valid with m_words elements
                unsafe { batch_propagate_sign_xor2_neon(sign, &x[q], &z[q], m_words) };
                return;
            }
            for w in 0..m_words {
                sign[w] ^= x[q][w] ^ z[q][w];
            }
        }
        Gate::Z => {
            let q = targets[0];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: AVX2 detected, slices are valid with m_words elements
                unsafe { batch_propagate_sign_xor_avx2(sign, &x[q], m_words) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                // SAFETY: NEON is baseline on aarch64, slices are valid with m_words elements
                unsafe { batch_propagate_sign_xor_neon(sign, &x[q], m_words) };
                return;
            }
            for w in 0..m_words {
                sign[w] ^= x[q][w];
            }
        }
        Gate::Id => {}
        Gate::SX => {
            let q = targets[0];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: AVX2 detected, slices are valid with m_words elements
                unsafe { batch_propagate_sx_avx2(&mut x[q], &z[q], sign, m_words, true) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                // SAFETY: NEON is baseline on aarch64, slices are valid with m_words elements
                unsafe { batch_propagate_sx_neon(&mut x[q], &z[q], sign, m_words, true) };
                return;
            }
            for w in 0..m_words {
                sign[w] ^= !x[q][w] & z[q][w];
                x[q][w] ^= z[q][w];
            }
        }
        Gate::SXdg => {
            let q = targets[0];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: AVX2 detected, slices are valid with m_words elements
                unsafe { batch_propagate_sx_avx2(&mut x[q], &z[q], sign, m_words, false) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                // SAFETY: NEON is baseline on aarch64, slices are valid with m_words elements
                unsafe { batch_propagate_sx_neon(&mut x[q], &z[q], sign, m_words, false) };
                return;
            }
            for w in 0..m_words {
                sign[w] ^= x[q][w] & z[q][w];
                x[q][w] ^= z[q][w];
            }
        }
        Gate::Cx => {
            let ctrl = targets[0];
            let tgt = targets[1];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                let (x_ctrl_sl, x_tgt_sl) = if ctrl < tgt {
                    let (lo, hi) = x.split_at_mut(tgt);
                    (&lo[ctrl][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = x.split_at_mut(ctrl);
                    (&hi[0][..], &mut lo[tgt][..])
                };
                let (z_ctrl_sl, z_tgt_sl) = if ctrl < tgt {
                    let (lo, hi) = z.split_at_mut(tgt);
                    (&mut lo[ctrl][..], &hi[0][..])
                } else {
                    let (lo, hi) = z.split_at_mut(ctrl);
                    (&mut hi[0][..], &lo[tgt][..])
                };
                // SAFETY: AVX2 detected, slices are valid, ctrl != tgt
                unsafe {
                    batch_propagate_cx_avx2(x_ctrl_sl, z_ctrl_sl, x_tgt_sl, z_tgt_sl, sign, m_words)
                };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                let (x_ctrl_sl, x_tgt_sl) = if ctrl < tgt {
                    let (lo, hi) = x.split_at_mut(tgt);
                    (&lo[ctrl][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = x.split_at_mut(ctrl);
                    (&hi[0][..], &mut lo[tgt][..])
                };
                let (z_ctrl_sl, z_tgt_sl) = if ctrl < tgt {
                    let (lo, hi) = z.split_at_mut(tgt);
                    (&mut lo[ctrl][..], &hi[0][..])
                } else {
                    let (lo, hi) = z.split_at_mut(ctrl);
                    (&mut hi[0][..], &lo[tgt][..])
                };
                // SAFETY: NEON is baseline on aarch64, slices are valid, ctrl != tgt
                unsafe {
                    batch_propagate_cx_neon(x_ctrl_sl, z_ctrl_sl, x_tgt_sl, z_tgt_sl, sign, m_words)
                };
                return;
            }
            for w in 0..m_words {
                sign[w] ^= x[ctrl][w] & z[tgt][w] & !(z[ctrl][w] ^ x[tgt][w]);
                x[tgt][w] ^= x[ctrl][w];
                z[ctrl][w] ^= z[tgt][w];
            }
        }
        Gate::Cz => {
            let q0 = targets[0];
            let q1 = targets[1];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                let (x0_sl, x1_sl) = if q0 < q1 {
                    let (lo, hi) = x.split_at_mut(q1);
                    (&lo[q0][..], &hi[0][..])
                } else {
                    let (lo, hi) = x.split_at_mut(q0);
                    (&hi[0][..], &lo[q1][..])
                };
                let (z0_sl, z1_sl) = if q0 < q1 {
                    let (lo, hi) = z.split_at_mut(q1);
                    (&mut lo[q0][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = z.split_at_mut(q0);
                    (&mut hi[0][..], &mut lo[q1][..])
                };
                // SAFETY: AVX2 detected, slices are valid, q0 != q1
                unsafe { batch_propagate_cz_avx2(x0_sl, z0_sl, x1_sl, z1_sl, sign, m_words) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                let (x0_sl, x1_sl) = if q0 < q1 {
                    let (lo, hi) = x.split_at_mut(q1);
                    (&lo[q0][..], &hi[0][..])
                } else {
                    let (lo, hi) = x.split_at_mut(q0);
                    (&hi[0][..], &lo[q1][..])
                };
                let (z0_sl, z1_sl) = if q0 < q1 {
                    let (lo, hi) = z.split_at_mut(q1);
                    (&mut lo[q0][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = z.split_at_mut(q0);
                    (&mut hi[0][..], &mut lo[q1][..])
                };
                // SAFETY: NEON is baseline on aarch64, slices are valid, q0 != q1
                unsafe { batch_propagate_cz_neon(x0_sl, z0_sl, x1_sl, z1_sl, sign, m_words) };
                return;
            }
            for w in 0..m_words {
                sign[w] ^= x[q0][w] & x[q1][w] & !(z[q0][w] ^ z[q1][w]);
                z[q0][w] ^= x[q1][w];
                z[q1][w] ^= x[q0][w];
            }
        }
        Gate::Swap => {
            let q0 = targets[0];
            let q1 = targets[1];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                let (x0_sl, x1_sl) = if q0 < q1 {
                    let (lo, hi) = x.split_at_mut(q1);
                    (&mut lo[q0][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = x.split_at_mut(q0);
                    (&mut hi[0][..], &mut lo[q1][..])
                };
                let (z0_sl, z1_sl) = if q0 < q1 {
                    let (lo, hi) = z.split_at_mut(q1);
                    (&mut lo[q0][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = z.split_at_mut(q0);
                    (&mut hi[0][..], &mut lo[q1][..])
                };
                // SAFETY: AVX2 detected, slices are valid, q0 != q1
                unsafe { batch_propagate_swap_avx2(x0_sl, z0_sl, x1_sl, z1_sl, m_words) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                let (x0_sl, x1_sl) = if q0 < q1 {
                    let (lo, hi) = x.split_at_mut(q1);
                    (&mut lo[q0][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = x.split_at_mut(q0);
                    (&mut hi[0][..], &mut lo[q1][..])
                };
                let (z0_sl, z1_sl) = if q0 < q1 {
                    let (lo, hi) = z.split_at_mut(q1);
                    (&mut lo[q0][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = z.split_at_mut(q0);
                    (&mut hi[0][..], &mut lo[q1][..])
                };
                // SAFETY: NEON is baseline on aarch64, slices are valid, q0 != q1
                unsafe { batch_propagate_swap_neon(x0_sl, z0_sl, x1_sl, z1_sl, m_words) };
                return;
            }
            for w in 0..m_words {
                let tmp_x = x[q0][w];
                x[q0][w] = x[q1][w];
                x[q1][w] = tmp_x;
                let tmp_z = z[q0][w];
                z[q0][w] = z[q1][w];
                z[q1][w] = tmp_z;
            }
        }
        _ => {}
    }
}

fn gaussian_eliminate(rows: &mut [Vec<u64>], num_cols: usize) -> (usize, Vec<usize>) {
    let num_rows = rows.len();
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut current_row = 0;

    for col in 0..num_cols {
        let word = col / 64;
        let bit = col % 64;
        let mask = 1u64 << bit;

        let pivot = rows[current_row..num_rows]
            .iter()
            .position(|row| row[word] & mask != 0)
            .map(|i| i + current_row);

        let pivot_row = match pivot {
            Some(r) => r,
            None => continue,
        };

        if pivot_row != current_row {
            rows.swap(pivot_row, current_row);
        }

        let (top, rest) = rows.split_at_mut(current_row + 1);
        let pivot_data = &top[current_row];
        for row in rest.iter_mut() {
            if row[word] & mask != 0 {
                for w in 0..row.len() {
                    row[w] ^= pivot_data[w];
                }
            }
        }

        pivot_cols.push(col);
        current_row += 1;
    }

    (current_row, pivot_cols)
}

const LUT_GROUP_SIZE: usize = 8;
const LUT_MIN_RANK: usize = 8;

struct FlipLut {
    data: Vec<u64>,
    m_words: usize,
    num_full_groups: usize,
    remainder_size: usize,
}

impl FlipLut {
    fn build(flip_rows: &[Vec<u64>], m_words: usize) -> Self {
        let rank = flip_rows.len();
        let num_full_groups = rank / LUT_GROUP_SIZE;
        let remainder_size = rank % LUT_GROUP_SIZE;
        let total_groups = num_full_groups + usize::from(remainder_size > 0);
        let entries_per_group = 1 << LUT_GROUP_SIZE;

        let mut data = vec![0u64; total_groups * entries_per_group * m_words];

        for g in 0..total_groups {
            let group_start = g * LUT_GROUP_SIZE;
            let k = if g < num_full_groups {
                LUT_GROUP_SIZE
            } else {
                remainder_size
            };
            let lut_offset = g * entries_per_group * m_words;

            for byte in 1..(1usize << k) {
                let lowest = byte & byte.wrapping_neg();
                let row_idx = group_start + lowest.trailing_zeros() as usize;
                let prev = byte ^ lowest;

                let dst_start = lut_offset + byte * m_words;
                let src_start = lut_offset + prev * m_words;

                for w in 0..m_words {
                    data[dst_start + w] = data[src_start + w] ^ flip_rows[row_idx][w];
                }
            }
        }

        Self {
            data,
            m_words,
            num_full_groups,
            remainder_size,
        }
    }

    #[inline(always)]
    fn lookup(&self, group: usize, byte: usize) -> &[u64] {
        let offset = (group * (1 << LUT_GROUP_SIZE) + byte) * self.m_words;
        &self.data[offset..offset + self.m_words]
    }
}

pub struct CompiledSampler {
    /// r rows, each packed as m_words u64s. Row j stores which measurements
    /// flip when random bit j is 1.
    flip_rows: Vec<Vec<u64>>,
    ref_bits_packed: Vec<u64>,
    rank: usize,
    num_measurements: usize,
    rng: ChaCha8Rng,
    lut: Option<FlipLut>,
}

fn pack_bools(bools: &[bool]) -> Vec<u64> {
    let n_words = bools.len().div_ceil(64);
    let mut packed = vec![0u64; n_words];
    for (i, &b) in bools.iter().enumerate() {
        if b {
            packed[i / 64] |= 1u64 << (i % 64);
        }
    }
    packed
}

impl CompiledSampler {
    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    pub fn sample(&mut self) -> Vec<bool> {
        let num_meas_words = self.num_measurements.div_ceil(64);
        let mut accum = vec![0u64; num_meas_words];
        self.sample_into(&mut accum);
        self.unpack_result(&accum)
    }

    pub(crate) fn sample_bulk_words(&mut self, num_shots: usize) -> (Vec<u64>, usize) {
        let m_words = self.num_measurements.div_ceil(64);
        if num_shots == 0 || self.num_measurements == 0 || self.rank == 0 {
            return (vec![0u64; num_shots * m_words], m_words);
        }

        if let Some(lut) = &self.lut {
            let total_groups = lut.num_full_groups + usize::from(lut.remainder_size > 0);
            let bytes_per_shot = total_groups;
            let total_bytes = num_shots * bytes_per_shot;
            let mut rand_bytes: Vec<u8> = vec![0u8; total_bytes];
            {
                let full_chunks = total_bytes / 8;
                let tail = full_chunks * 8;
                for i in 0..full_chunks {
                    let r = self.rng.next_u64();
                    rand_bytes[i * 8..(i + 1) * 8].copy_from_slice(&r.to_le_bytes());
                }
                if tail < total_bytes {
                    let r = self.rng.next_u64();
                    let bytes = r.to_le_bytes();
                    rand_bytes[tail..total_bytes].copy_from_slice(&bytes[..total_bytes - tail]);
                }
                if lut.remainder_size > 0 {
                    let remainder_mask = (1u8 << lut.remainder_size) - 1;
                    let last_group = lut.num_full_groups;
                    for s in 0..num_shots {
                        rand_bytes[s * bytes_per_shot + last_group] &= remainder_mask;
                    }
                }
            }

            let mut accum: Vec<u64> = vec![0u64; num_shots * m_words];

            let max_batch = if m_words > 0 {
                (256 * 1024 / (m_words * 8)).max(64)
            } else {
                num_shots
            };

            #[cfg(feature = "parallel")]
            const PAR_SHOT_THRESHOLD: usize = 256;

            #[cfg(feature = "parallel")]
            if num_shots >= PAR_SHOT_THRESHOLD {
                use rayon::prelude::*;
                let shots_per_chunk =
                    (num_shots.div_ceil(rayon::current_num_threads())).max(max_batch);
                let chunk_m = shots_per_chunk * m_words;
                accum
                    .par_chunks_mut(chunk_m)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let chunk_shots = chunk.len() / m_words;
                        let chunk_start = ci * shots_per_chunk;
                        for tile_start in (0..chunk_shots).step_by(max_batch) {
                            let tile_end = (tile_start + max_batch).min(chunk_shots);
                            for g in 0..total_groups {
                                for s in tile_start..tile_end {
                                    let gs = chunk_start + s;
                                    let byte = rand_bytes[gs * bytes_per_shot + g] as usize;
                                    let entry = lut.lookup(g, byte);
                                    let base = s * m_words;
                                    xor_words(&mut chunk[base..base + m_words], entry);
                                }
                            }
                        }
                    });
            } else {
                for tile_start in (0..num_shots).step_by(max_batch) {
                    let tile_end = (tile_start + max_batch).min(num_shots);
                    for g in 0..total_groups {
                        for s in tile_start..tile_end {
                            let byte = rand_bytes[s * bytes_per_shot + g] as usize;
                            let entry = lut.lookup(g, byte);
                            let shot_base = s * m_words;
                            xor_words(&mut accum[shot_base..shot_base + m_words], entry);
                        }
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            {
                for tile_start in (0..num_shots).step_by(max_batch) {
                    let tile_end = (tile_start + max_batch).min(num_shots);
                    for g in 0..total_groups {
                        for s in tile_start..tile_end {
                            let byte = rand_bytes[s * bytes_per_shot + g] as usize;
                            let entry = lut.lookup(g, byte);
                            let shot_base = s * m_words;
                            xor_words(&mut accum[shot_base..shot_base + m_words], entry);
                        }
                    }
                }
            }

            (accum, m_words)
        } else {
            let mut accum = vec![0u64; num_shots * m_words];
            for s in 0..num_shots {
                let shot_base = s * m_words;
                let shot_accum = &mut accum[shot_base..shot_base + m_words];
                for j in 0..self.rank {
                    let bit = self.rng.next_u32() & 1;
                    if bit != 0 {
                        let row = &self.flip_rows[j];
                        xor_words(shot_accum, row);
                    }
                }
            }
            (accum, m_words)
        }
    }

    pub(crate) fn ref_bits_packed(&self) -> &[u64] {
        &self.ref_bits_packed
    }

    pub fn sample_bulk(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        if num_shots == 0 || self.num_measurements == 0 {
            return vec![Vec::new(); num_shots];
        }

        let m_words = self.num_measurements.div_ceil(64);

        if self.rank == 0 {
            let result = self.unpack_result_static();
            return vec![result; num_shots];
        }

        if self.lut.is_some() {
            self.sample_bulk_grouped(num_shots, m_words)
        } else {
            self.sample_bulk_sequential(num_shots, m_words)
        }
    }

    fn sample_bulk_grouped(&mut self, num_shots: usize, m_words: usize) -> Vec<Vec<bool>> {
        let lut = self.lut.as_ref().unwrap();
        let total_groups = lut.num_full_groups + usize::from(lut.remainder_size > 0);

        let bytes_per_shot = total_groups;
        let total_bytes = num_shots * bytes_per_shot;
        let mut rand_bytes: Vec<u8> = vec![0u8; total_bytes];
        {
            let full_chunks = total_bytes / 8;
            let tail = full_chunks * 8;
            for i in 0..full_chunks {
                let r = self.rng.next_u64();
                rand_bytes[i * 8..(i + 1) * 8].copy_from_slice(&r.to_le_bytes());
            }
            if tail < total_bytes {
                let r = self.rng.next_u64();
                let bytes = r.to_le_bytes();
                rand_bytes[tail..total_bytes].copy_from_slice(&bytes[..total_bytes - tail]);
            }
            if lut.remainder_size > 0 {
                let remainder_mask = (1u8 << lut.remainder_size) - 1;
                let last_group = lut.num_full_groups;
                for s in 0..num_shots {
                    rand_bytes[s * bytes_per_shot + last_group] &= remainder_mask;
                }
            }
        }

        let mut accum: Vec<u64> = vec![0u64; num_shots * m_words];

        // Tile shots so accumulators fit in L2 (~256KB). Each shot uses m_words × 8 bytes.
        let max_batch = if m_words > 0 {
            (256 * 1024 / (m_words * 8)).max(64)
        } else {
            num_shots
        };

        // Group-major loop with tiling: LUT group stays L1-hot within each tile
        for tile_start in (0..num_shots).step_by(max_batch) {
            let tile_end = (tile_start + max_batch).min(num_shots);
            for g in 0..total_groups {
                for s in tile_start..tile_end {
                    let byte = rand_bytes[s * bytes_per_shot + g] as usize;
                    let entry = lut.lookup(g, byte);
                    let shot_base = s * m_words;
                    xor_words(&mut accum[shot_base..shot_base + m_words], entry);
                }
            }
        }

        let mut shots = Vec::with_capacity(num_shots);
        for s in 0..num_shots {
            let shot_base = s * m_words;
            shots.push(self.unpack_result(&accum[shot_base..shot_base + m_words]));
        }
        shots
    }

    fn sample_bulk_sequential(&mut self, num_shots: usize, m_words: usize) -> Vec<Vec<bool>> {
        let mut accum = vec![0u64; m_words];
        let mut shots = Vec::with_capacity(num_shots);

        for _ in 0..num_shots {
            accum.fill(0);
            self.sample_into(&mut accum);
            shots.push(self.unpack_result(&accum));
        }

        shots
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) fn sample_into_raw(&mut self, accum: &mut [u64]) {
        self.sample_into(accum);
    }

    #[inline(always)]
    fn sample_into(&mut self, accum: &mut [u64]) {
        if self.rank == 0 {
            return;
        }

        if let Some(lut) = &self.lut {
            let mut rand_buf = 0u64;
            let mut rand_pos = 8usize;

            for g in 0..lut.num_full_groups {
                if rand_pos >= 8 {
                    rand_buf = self.rng.next_u64();
                    rand_pos = 0;
                }
                let byte = ((rand_buf >> (rand_pos * 8)) & 0xFF) as usize;
                rand_pos += 1;
                let entry = lut.lookup(g, byte);
                xor_words(accum, entry);
            }
            if lut.remainder_size > 0 {
                if rand_pos >= 8 {
                    rand_buf = self.rng.next_u64();
                }
                let mask = (1u64 << lut.remainder_size) - 1;
                let byte = (rand_buf & mask) as usize;
                let entry = lut.lookup(lut.num_full_groups, byte);
                xor_words(accum, entry);
            }
        } else {
            for j in 0..self.rank {
                let bit = self.rng.next_u32() & 1;
                if bit != 0 {
                    let row = &self.flip_rows[j];
                    xor_words(accum, row);
                }
            }
        }
    }

    #[inline(always)]
    fn unpack_result(&self, accum: &[u64]) -> Vec<bool> {
        let mut result = Vec::with_capacity(self.num_measurements);
        for m in 0..self.num_measurements {
            let w = m / 64;
            let ref_word = if w < self.ref_bits_packed.len() {
                self.ref_bits_packed[w]
            } else {
                0
            };
            let bit = ((accum[w] ^ ref_word) >> (m % 64)) & 1 != 0;
            result.push(bit);
        }
        result
    }

    pub(crate) fn apply_ref_bits(&self, accum: &mut [u64]) {
        xor_words(accum, &self.ref_bits_packed);
    }

    fn unpack_result_static(&self) -> Vec<bool> {
        let mut result = Vec::with_capacity(self.num_measurements);
        for m in 0..self.num_measurements {
            let w = m / 64;
            let bit = if w < self.ref_bits_packed.len() {
                (self.ref_bits_packed[w] >> (m % 64)) & 1 != 0
            } else {
                false
            };
            result.push(bit);
        }
        result
    }
}

/// AVX2 rowmul: src (read-only) multiplied into dst (modified in-place).
/// Both buffers are laid out as [x_words(nw) | z_words(nw)].
/// Returns the accumulated Pauli phase sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rowmul_avx2(src_ptr: *const u64, dst_ptr: *mut u64, nw: usize, initial_sum: u64) -> u64 {
    use std::arch::x86_64::*;
    let chunks = nw / 4;
    let mut sum = initial_sum;
    let src_x = src_ptr;
    let src_z = src_ptr.add(nw);
    let dst_x = dst_ptr;
    let dst_z = dst_ptr.add(nw);

    for i in 0..chunks {
        let off = i * 4;
        let x1 = _mm256_loadu_si256(src_x.add(off) as *const __m256i);
        let z1 = _mm256_loadu_si256(src_z.add(off) as *const __m256i);
        let x2 = _mm256_loadu_si256(dst_x.add(off) as *const __m256i);
        let z2 = _mm256_loadu_si256(dst_z.add(off) as *const __m256i);

        let new_x = _mm256_xor_si256(x1, x2);
        let new_z = _mm256_xor_si256(z1, z2);
        _mm256_storeu_si256(dst_x.add(off) as *mut __m256i, new_x);
        _mm256_storeu_si256(dst_z.add(off) as *mut __m256i, new_z);

        let any = _mm256_or_si256(_mm256_or_si256(x1, z1), _mm256_or_si256(x2, z2));
        if _mm256_testz_si256(any, any) == 0 {
            let x1z1 = _mm256_and_si256(x1, z1);
            let x2z2 = _mm256_and_si256(x2, z2);

            let nonzero = _mm256_and_si256(
                _mm256_and_si256(_mm256_or_si256(new_x, new_z), _mm256_or_si256(x1, z1)),
                _mm256_or_si256(x2, z2),
            );
            // pos = (x1&z1&!x2&z2) | (x1&!z1&x2&z2) | (!x1&z1&x2&!z2)
            let pos = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_and_si256(x1z1, _mm256_andnot_si256(x2, z2)),
                    _mm256_and_si256(_mm256_andnot_si256(z1, x1), x2z2),
                ),
                _mm256_and_si256(_mm256_andnot_si256(x1, z1), _mm256_andnot_si256(z2, x2)),
            );

            let nz_arr: [u64; 4] = std::mem::transmute(nonzero);
            let pos_arr: [u64; 4] = std::mem::transmute(pos);
            let mut nz_count = 0u64;
            let mut pos_count = 0u64;
            for j in 0..4 {
                nz_count += nz_arr[j].count_ones() as u64;
                pos_count += pos_arr[j].count_ones() as u64;
            }
            sum = sum.wrapping_add(2 * pos_count);
            sum = sum.wrapping_sub(nz_count);
        }
    }

    let tail = chunks * 4;
    for w in tail..nw {
        let x1 = *src_x.add(w);
        let z1 = *src_z.add(w);
        let x2 = *dst_x.add(w);
        let z2 = *dst_z.add(w);
        let new_x = x1 ^ x2;
        let new_z = z1 ^ z2;
        *dst_x.add(w) = new_x;
        *dst_z.add(w) = new_z;
        if (x1 | z1 | x2 | z2) != 0 {
            let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
            let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
            sum = sum.wrapping_add(2 * pos.count_ones() as u64);
            sum = sum.wrapping_sub(nonzero.count_ones() as u64);
        }
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn rowmul_neon(src_ptr: *const u64, dst_ptr: *mut u64, nw: usize, initial_sum: u64) -> u64 {
    use std::arch::aarch64::*;
    let chunks = nw / 2;
    let mut sum = initial_sum;
    let src_x = src_ptr;
    let src_z = src_ptr.add(nw);
    let dst_x = dst_ptr;
    let dst_z = dst_ptr.add(nw);

    for i in 0..chunks {
        let off = i * 2;
        let x1 = vld1q_u64(src_x.add(off));
        let z1 = vld1q_u64(src_z.add(off));
        let x2 = vld1q_u64(dst_x.add(off));
        let z2 = vld1q_u64(dst_z.add(off));

        let new_x = veorq_u64(x1, x2);
        let new_z = veorq_u64(z1, z2);
        vst1q_u64(dst_x.add(off), new_x);
        vst1q_u64(dst_z.add(off), new_z);

        let any = vorrq_u64(vorrq_u64(x1, z1), vorrq_u64(x2, z2));
        let any_arr: [u64; 2] = std::mem::transmute(any);
        if (any_arr[0] | any_arr[1]) != 0 {
            let x1z1 = vandq_u64(x1, z1);
            let x2z2 = vandq_u64(x2, z2);

            let nonzero = vandq_u64(
                vandq_u64(vorrq_u64(new_x, new_z), vorrq_u64(x1, z1)),
                vorrq_u64(x2, z2),
            );
            // pos = (x1&z1&!x2&z2) | (x1&!z1&x2&z2) | (!x1&z1&x2&!z2)
            let pos = vorrq_u64(
                vorrq_u64(
                    vandq_u64(x1z1, vbicq_u64(z2, x2)),
                    vandq_u64(vbicq_u64(x1, z1), x2z2),
                ),
                vandq_u64(vbicq_u64(z1, x1), vbicq_u64(x2, z2)),
            );

            let nz_arr: [u64; 2] = std::mem::transmute(nonzero);
            let pos_arr: [u64; 2] = std::mem::transmute(pos);
            let mut nz_count = 0u64;
            let mut pos_count = 0u64;
            for j in 0..2 {
                nz_count += nz_arr[j].count_ones() as u64;
                pos_count += pos_arr[j].count_ones() as u64;
            }
            sum = sum.wrapping_add(2 * pos_count);
            sum = sum.wrapping_sub(nz_count);
        }
    }

    let tail = chunks * 2;
    for w in tail..nw {
        let x1 = *src_x.add(w);
        let z1 = *src_z.add(w);
        let x2 = *dst_x.add(w);
        let z2 = *dst_z.add(w);
        let new_x = x1 ^ x2;
        let new_z = z1 ^ z2;
        *dst_x.add(w) = new_x;
        *dst_z.add(w) = new_z;
        if (x1 | z1 | x2 | z2) != 0 {
            let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
            let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
            sum = sum.wrapping_add(2 * pos.count_ones() as u64);
            sum = sum.wrapping_sub(nonzero.count_ones() as u64);
        }
    }
    sum
}

/// Rowmul: multiply src into xz[r_base..], returning the resulting phase.
#[inline(always)]
fn rowmul_phase(
    src: &[u64],
    xz: &mut [u64],
    r_base: usize,
    nw: usize,
    src_phase: bool,
    dst_phase: bool,
) -> bool {
    let initial_sum = if src_phase { 2u64 } else { 0 } + if dst_phase { 2u64 } else { 0 };

    #[cfg(target_arch = "x86_64")]
    if nw >= 4 && is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected, src has 2*nw elements, xz[r_base..] has 2*nw elements
        let sum =
            unsafe { rowmul_avx2(src.as_ptr(), xz.as_mut_ptr().add(r_base), nw, initial_sum) };
        return (sum & 3) >= 2;
    }

    #[cfg(target_arch = "aarch64")]
    if nw >= 2 {
        // SAFETY: NEON is baseline on aarch64, src has 2*nw elements, xz[r_base..] has 2*nw elements
        let sum =
            unsafe { rowmul_neon(src.as_ptr(), xz.as_mut_ptr().add(r_base), nw, initial_sum) };
        return (sum & 3) >= 2;
    }

    let mut sum = initial_sum;
    for w in 0..nw {
        let x1 = src[w];
        let z1 = src[nw + w];
        let x2 = xz[r_base + w];
        let z2 = xz[r_base + nw + w];
        let new_x = x1 ^ x2;
        let new_z = z1 ^ z2;
        xz[r_base + w] = new_x;
        xz[r_base + nw + w] = new_z;
        if (x1 | z1 | x2 | z2) != 0 {
            let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
            let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
            sum = sum.wrapping_add(2 * pos.count_ones() as u64);
            sum = sum.wrapping_sub(nonzero.count_ones() as u64);
        }
    }
    (sum & 3) >= 2
}

/// Rowmul variant for scratch buffer: multiply src (from xz) into scratch.
#[inline(always)]
fn rowmul_phase_into(
    xz: &[u64],
    s_base: usize,
    scratch: &mut [u64],
    nw: usize,
    src_phase: bool,
    dst_phase: bool,
) -> bool {
    let initial_sum = if src_phase { 2u64 } else { 0 } + if dst_phase { 2u64 } else { 0 };

    #[cfg(target_arch = "x86_64")]
    if nw >= 4 && is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected, xz[s_base..] is src, scratch is dst
        let sum = unsafe {
            rowmul_avx2(
                xz.as_ptr().add(s_base),
                scratch.as_mut_ptr(),
                nw,
                initial_sum,
            )
        };
        return (sum & 3) >= 2;
    }

    #[cfg(target_arch = "aarch64")]
    if nw >= 2 {
        // SAFETY: NEON is baseline on aarch64, xz[s_base..] is src, scratch is dst
        let sum = unsafe {
            rowmul_neon(
                xz.as_ptr().add(s_base),
                scratch.as_mut_ptr(),
                nw,
                initial_sum,
            )
        };
        return (sum & 3) >= 2;
    }

    let mut sum = initial_sum;
    for w in 0..nw {
        let x1 = xz[s_base + w];
        let z1 = xz[s_base + nw + w];
        let x2 = scratch[w];
        let z2 = scratch[nw + w];
        let new_x = x1 ^ x2;
        let new_z = z1 ^ z2;
        scratch[w] = new_x;
        scratch[nw + w] = new_z;
        if (x1 | z1 | x2 | z2) != 0 {
            let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
            let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
            sum = sum.wrapping_add(2 * pos.count_ones() as u64);
            sum = sum.wrapping_sub(nonzero.count_ones() as u64);
        }
    }
    (sum & 3) >= 2
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_words_avx2(dst: &mut [u64], src: &[u64]) {
    use std::arch::x86_64::*;
    let len = dst.len().min(src.len());
    let chunks = len / 4;
    let dp = dst.as_mut_ptr() as *mut __m256i;
    let sp = src.as_ptr() as *const __m256i;
    for i in 0..chunks {
        let d = _mm256_loadu_si256(dp.add(i));
        let s = _mm256_loadu_si256(sp.add(i));
        _mm256_storeu_si256(dp.add(i), _mm256_xor_si256(d, s));
    }
    let tail = chunks * 4;
    for i in tail..len {
        *dst.get_unchecked_mut(i) ^= *src.get_unchecked(i);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn xor_words_neon(dst: &mut [u64], src: &[u64]) {
    use std::arch::aarch64::*;
    let len = dst.len().min(src.len());
    let chunks = len / 2;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    for i in 0..chunks {
        let off = i * 2;
        let d = vld1q_u64(dp.add(off));
        let s = vld1q_u64(sp.add(off));
        vst1q_u64(dp.add(off), veorq_u64(d, s));
    }
    let tail = chunks * 2;
    for i in tail..len {
        *dst.get_unchecked_mut(i) ^= *src.get_unchecked(i);
    }
}

#[inline(always)]
pub(crate) fn xor_words(dst: &mut [u64], src: &[u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && dst.len() >= 4 {
            // SAFETY: AVX2 detected, pointers are valid u64 slices
            unsafe {
                xor_words_avx2(dst, src);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if dst.len() >= 2 {
            // SAFETY: NEON is baseline on aarch64, pointers are valid u64 slices
            unsafe {
                xor_words_neon(dst, src);
            }
            return;
        }
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}

/// Compute reference measurement outcomes by simulating measurements of the
/// propagated Paulis on |0⟩^n using the Aaronson-Gottesman measurement protocol.
/// O(m × n²/64) — avoids re-simulating all T circuit gates through the stabilizer.
#[allow(clippy::needless_range_loop)]
fn compute_reference_bits(measurement_rows: &[(PauliVec, usize, bool)], n: usize) -> Vec<bool> {
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

    for (meas_idx, (pauli, _, _)) in measurement_rows.iter().enumerate() {
        let mut anti_idx = None;
        for g in n..2 * n {
            let base = g * stride;
            let mut inner = 0u64;
            for w in 0..nw {
                inner ^= (pauli.x[w] & xz[base + nw + w]) ^ (pauli.z[w] & xz[base + w]);
            }
            if inner.count_ones() % 2 == 1 {
                anti_idx = Some(g);
                break;
            }
        }

        match anti_idx {
            Some(p) => {
                // Random measurement. Pick outcome = 0.
                // For all rows (0..2n) that anti-commute with P (except p): rowmul(row, p)
                let p_data: Vec<u64> = xz[p * stride..(p + 1) * stride].to_vec();
                let p_phase = phase[p];

                for r in 0..2 * n {
                    if r == p {
                        continue;
                    }
                    let r_base = r * stride;
                    let mut inner = 0u64;
                    for w in 0..nw {
                        inner ^= (pauli.x[w] & xz[r_base + nw + w]) ^ (pauli.z[w] & xz[r_base + w]);
                    }
                    if inner.count_ones() % 2 == 1 {
                        phase[r] = rowmul_phase(&p_data, &mut xz, r_base, nw, p_phase, phase[r]);
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
                // Deterministic: P commutes with all stabilizers.
                // Accumulate rowmul of stabilizer[g] for each destabilizer[g]
                // that anti-commutes with P.
                let mut scratch = vec![0u64; stride];
                let mut scratch_phase = false;

                for g in 0..n {
                    let d_base = g * stride;
                    let mut inner = 0u64;
                    for w in 0..nw {
                        inner ^= (pauli.x[w] & xz[d_base + nw + w]) ^ (pauli.z[w] & xz[d_base + w]);
                    }
                    if inner.count_ones() % 2 == 1 {
                        let s_base = (g + n) * stride;
                        let s_phase = phase[g + n];
                        scratch_phase = rowmul_phase_into(
                            &xz,
                            s_base,
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

/// Forward-compile a Clifford circuit's measurements into a fast sampler.
///
/// SGI-optimized stabilizer processes gates forward; dependency tracking during
/// measurement extracts reference bits and the flip matrix in one pass.
///
/// Compilation: O(T × active_avg × nw) gates + O(m × n × (nw + r_words)) measurements.
/// Per-shot: O(r·m/64) where r = rank.
/// Column-major forward stabilizer simulation.
///
/// Stores x_cols[qubit][row_word] and z_cols[qubit][row_word] so gate
/// application is O(row_words) vector XOR instead of O(2n) per-row bit ops.
///
/// Returns (xz_rowmajor, phase, nw) transposed back to row-major.
fn colmajor_forward_sim(
    n: usize,
    instructions: &[Instruction],
) -> Result<(Vec<u64>, Vec<bool>, usize)> {
    let total_rows = 2 * n;
    let nw = n.div_ceil(64);
    let row_words = total_rows.div_ceil(64);

    let mut x_cols: Vec<Vec<u64>> = vec![vec![0u64; row_words]; n];
    let mut z_cols: Vec<Vec<u64>> = vec![vec![0u64; row_words]; n];
    let mut phase: Vec<u64> = vec![0u64; row_words];

    for q in 0..n {
        x_cols[q][q / 64] |= 1u64 << (q % 64);
        let stab_row = n + q;
        z_cols[q][stab_row / 64] |= 1u64 << (stab_row % 64);
    }

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = row_words >= 4 && is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;
    #[cfg(target_arch = "aarch64")]
    let use_neon = row_words >= 2;

    for inst in instructions {
        let (gate, targets) = match inst {
            Instruction::Gate { gate, targets } => (gate, targets.as_slice()),
            Instruction::Conditional { gate, targets, .. } => (gate, targets.as_slice()),
            _ => continue,
        };

        match gate {
            Gate::H => {
                let q = targets[0];
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 detected, slices have row_words elements
                    unsafe {
                        batch_propagate_h_avx2(
                            &mut x_cols[q],
                            &mut z_cols[q],
                            &mut phase,
                            row_words,
                        )
                    };
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if use_neon {
                        // SAFETY: NEON is baseline on aarch64
                        unsafe {
                            batch_propagate_h_neon(
                                &mut x_cols[q],
                                &mut z_cols[q],
                                &mut phase,
                                row_words,
                            )
                        };
                        continue;
                    }
                    for w in 0..row_words {
                        phase[w] ^= x_cols[q][w] & z_cols[q][w];
                    }
                    std::mem::swap(&mut x_cols[q], &mut z_cols[q]);
                }
            }
            Gate::S => {
                let q = targets[0];
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 detected
                    unsafe {
                        batch_propagate_s_avx2(
                            &mut x_cols[q],
                            &mut z_cols[q],
                            &mut phase,
                            row_words,
                            false,
                        )
                    };
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if use_neon {
                        // SAFETY: NEON is baseline on aarch64
                        unsafe {
                            batch_propagate_s_neon(
                                &mut x_cols[q],
                                &mut z_cols[q],
                                &mut phase,
                                row_words,
                                false,
                            )
                        };
                        continue;
                    }
                    for w in 0..row_words {
                        phase[w] ^= x_cols[q][w] & z_cols[q][w];
                        z_cols[q][w] ^= x_cols[q][w];
                    }
                }
            }
            Gate::Sdg => {
                let q = targets[0];
                // Forward Sdg: z ^= x (first), then p ^= x & z_new
                // Different order from backward — no reusable SIMD function
                for w in 0..row_words {
                    z_cols[q][w] ^= x_cols[q][w];
                    phase[w] ^= x_cols[q][w] & z_cols[q][w];
                }
            }
            Gate::X => {
                let q = targets[0];
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 detected
                    unsafe {
                        batch_propagate_sign_xor_avx2(&mut phase, &z_cols[q], row_words)
                    };
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if use_neon {
                        // SAFETY: NEON is baseline on aarch64
                        unsafe { batch_propagate_sign_xor_neon(&mut phase, &z_cols[q], row_words) };
                        continue;
                    }
                    for w in 0..row_words {
                        phase[w] ^= z_cols[q][w];
                    }
                }
            }
            Gate::Y => {
                let q = targets[0];
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 detected
                    unsafe {
                        batch_propagate_sign_xor2_avx2(
                            &mut phase, &x_cols[q], &z_cols[q], row_words,
                        )
                    };
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if use_neon {
                        // SAFETY: NEON is baseline on aarch64
                        unsafe {
                            batch_propagate_sign_xor2_neon(
                                &mut phase, &x_cols[q], &z_cols[q], row_words,
                            )
                        };
                        continue;
                    }
                    for w in 0..row_words {
                        phase[w] ^= x_cols[q][w] ^ z_cols[q][w];
                    }
                }
            }
            Gate::Z => {
                let q = targets[0];
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 detected
                    unsafe {
                        batch_propagate_sign_xor_avx2(&mut phase, &x_cols[q], row_words)
                    };
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if use_neon {
                        // SAFETY: NEON is baseline on aarch64
                        unsafe { batch_propagate_sign_xor_neon(&mut phase, &x_cols[q], row_words) };
                        continue;
                    }
                    for w in 0..row_words {
                        phase[w] ^= x_cols[q][w];
                    }
                }
            }
            Gate::Id => {}
            Gate::SX => {
                let q = targets[0];
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 detected
                    unsafe {
                        batch_propagate_sx_avx2(
                            &mut x_cols[q],
                            &z_cols[q],
                            &mut phase,
                            row_words,
                            true,
                        )
                    };
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if use_neon {
                        // SAFETY: NEON is baseline on aarch64
                        unsafe {
                            batch_propagate_sx_neon(
                                &mut x_cols[q],
                                &z_cols[q],
                                &mut phase,
                                row_words,
                                true,
                            )
                        };
                        continue;
                    }
                    for w in 0..row_words {
                        phase[w] ^= !x_cols[q][w] & z_cols[q][w];
                        x_cols[q][w] ^= z_cols[q][w];
                    }
                }
            }
            Gate::SXdg => {
                let q = targets[0];
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 detected
                    unsafe {
                        batch_propagate_sx_avx2(
                            &mut x_cols[q],
                            &z_cols[q],
                            &mut phase,
                            row_words,
                            false,
                        )
                    };
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if use_neon {
                        // SAFETY: NEON is baseline on aarch64
                        unsafe {
                            batch_propagate_sx_neon(
                                &mut x_cols[q],
                                &z_cols[q],
                                &mut phase,
                                row_words,
                                false,
                            )
                        };
                        continue;
                    }
                    for w in 0..row_words {
                        phase[w] ^= x_cols[q][w] & z_cols[q][w];
                        x_cols[q][w] ^= z_cols[q][w];
                    }
                }
            }
            Gate::Cx => {
                let ctrl = targets[0];
                let tgt = targets[1];
                let (xc_sl, xt_sl) = if ctrl < tgt {
                    let (lo, hi) = x_cols.split_at_mut(tgt);
                    (&lo[ctrl][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = x_cols.split_at_mut(ctrl);
                    (&hi[0][..], &mut lo[tgt][..])
                };
                let (zc_sl, zt_sl) = if ctrl < tgt {
                    let (lo, hi) = z_cols.split_at_mut(tgt);
                    (&mut lo[ctrl][..], &hi[0][..])
                } else {
                    let (lo, hi) = z_cols.split_at_mut(ctrl);
                    (&mut hi[0][..], &lo[tgt][..])
                };
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 detected, ctrl != tgt
                    unsafe {
                        batch_propagate_cx_avx2(xc_sl, zc_sl, xt_sl, zt_sl, &mut phase, row_words)
                    };
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if use_neon {
                        // SAFETY: NEON is baseline on aarch64, ctrl != tgt
                        unsafe {
                            batch_propagate_cx_neon(
                                xc_sl, zc_sl, xt_sl, zt_sl, &mut phase, row_words,
                            )
                        };
                        continue;
                    }
                    for w in 0..row_words {
                        phase[w] ^= xc_sl[w] & zt_sl[w] & !(zc_sl[w] ^ xt_sl[w]);
                        xt_sl[w] ^= xc_sl[w];
                        zc_sl[w] ^= zt_sl[w];
                    }
                }
            }
            Gate::Cz => {
                let q0 = targets[0];
                let q1 = targets[1];
                let (x0_sl, x1_sl) = if q0 < q1 {
                    let (lo, hi) = x_cols.split_at_mut(q1);
                    (&lo[q0][..], &hi[0][..])
                } else {
                    let (lo, hi) = x_cols.split_at_mut(q0);
                    (&hi[0][..], &lo[q1][..])
                };
                let (z0_sl, z1_sl) = if q0 < q1 {
                    let (lo, hi) = z_cols.split_at_mut(q1);
                    (&mut lo[q0][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = z_cols.split_at_mut(q0);
                    (&mut hi[0][..], &mut lo[q1][..])
                };
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 detected, q0 != q1
                    unsafe {
                        batch_propagate_cz_avx2(x0_sl, z0_sl, x1_sl, z1_sl, &mut phase, row_words)
                    };
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if use_neon {
                        // SAFETY: NEON is baseline on aarch64, q0 != q1
                        unsafe {
                            batch_propagate_cz_neon(
                                x0_sl, z0_sl, x1_sl, z1_sl, &mut phase, row_words,
                            )
                        };
                        continue;
                    }
                    for w in 0..row_words {
                        phase[w] ^= x0_sl[w] & x1_sl[w] & !(z0_sl[w] ^ z1_sl[w]);
                        z0_sl[w] ^= x1_sl[w];
                        z1_sl[w] ^= x0_sl[w];
                    }
                }
            }
            Gate::Swap => {
                let q0 = targets[0];
                let q1 = targets[1];
                let (x0_sl, x1_sl) = if q0 < q1 {
                    let (lo, hi) = x_cols.split_at_mut(q1);
                    (&mut lo[q0][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = x_cols.split_at_mut(q0);
                    (&mut hi[0][..], &mut lo[q1][..])
                };
                let (z0_sl, z1_sl) = if q0 < q1 {
                    let (lo, hi) = z_cols.split_at_mut(q1);
                    (&mut lo[q0][..], &mut hi[0][..])
                } else {
                    let (lo, hi) = z_cols.split_at_mut(q0);
                    (&mut hi[0][..], &mut lo[q1][..])
                };
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2 detected, q0 != q1
                    unsafe {
                        batch_propagate_swap_avx2(x0_sl, z0_sl, x1_sl, z1_sl, row_words)
                    };
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if use_neon {
                        // SAFETY: NEON is baseline on aarch64, q0 != q1
                        unsafe { batch_propagate_swap_neon(x0_sl, z0_sl, x1_sl, z1_sl, row_words) };
                        continue;
                    }
                    for w in 0..row_words {
                        std::mem::swap(&mut x0_sl[w], &mut x1_sl[w]);
                        std::mem::swap(&mut z0_sl[w], &mut z1_sl[w]);
                    }
                }
            }
            _ => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "CompiledSampler".to_string(),
                    reason: format!("unsupported gate {:?} in column-major forward sim", gate),
                });
            }
        }
    }

    let stride = 2 * nw;
    let mut xz = vec![0u64; total_rows * stride];
    let mut phase_vec = vec![false; total_rows];

    for q in 0..n {
        let qw = q / 64;
        let qb = q % 64;
        let qm = 1u64 << qb;
        for rw in 0..row_words {
            let mut xbits = x_cols[q][rw];
            while xbits != 0 {
                let bit = xbits.trailing_zeros() as usize;
                let row = rw * 64 + bit;
                if row < total_rows {
                    xz[row * stride + qw] |= qm;
                }
                xbits &= xbits - 1;
            }
            let mut zbits = z_cols[q][rw];
            while zbits != 0 {
                let bit = zbits.trailing_zeros() as usize;
                let row = rw * 64 + bit;
                if row < total_rows {
                    xz[row * stride + nw + qw] |= qm;
                }
                zbits &= zbits - 1;
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

pub fn compile_forward(circuit: &Circuit, seed: u64) -> Result<CompiledSampler> {
    if !circuit.is_clifford_only() {
        return Err(PrismError::IncompatibleBackend {
            backend: "CompiledSampler".to_string(),
            reason: "circuit contains non-Clifford gates".to_string(),
        });
    }

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

    let num_measurements = measurements.len();
    if num_measurements == 0 {
        return Ok(CompiledSampler {
            flip_rows: Vec::new(),
            ref_bits_packed: Vec::new(),
            rank: 0,
            num_measurements: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            lut: None,
        });
    }

    let n = circuit.num_qubits;

    let (mut xz, mut phase, nw) = colmajor_forward_sim(n, &circuit.instructions)?;
    let stride = 2 * nw;
    let m = num_measurements;
    let m_words = m.div_ceil(64);

    let rank_words = m_words;
    let total_rows = 2 * n;
    let mut gen_dep: Vec<Vec<u64>> = vec![vec![0u64; rank_words]; total_rows + 1];
    let mut ref_bits: Vec<bool> = vec![false; m];
    let mut rank = 0usize;

    let mut flip_rows: Vec<Vec<u64>> = Vec::with_capacity(m);
    let mut p_data: Vec<u64> = vec![0u64; stride];
    let mut p_dep: Vec<u64> = vec![0u64; rank_words];
    let mut scratch: Vec<u64> = vec![0u64; stride];
    let scratch_idx = total_rows;

    for (meas_idx, &(qubit, _)) in measurements.iter().enumerate() {
        let word = qubit / 64;
        let bit_mask = 1u64 << (qubit % 64);

        let mut p: Option<usize> = None;
        for i in n..2 * n {
            if xz[i * stride + word] & bit_mask != 0 {
                p = Some(i);
                break;
            }
        }

        if let Some(p_row) = p {
            // Random measurement — this is the k-th random degree of freedom
            let k = rank;
            rank += 1;
            flip_rows.push(vec![0u64; m_words]);

            flip_rows[k][meas_idx / 64] |= 1u64 << (meas_idx % 64);

            let p_base = p_row * stride;
            p_data.copy_from_slice(&xz[p_base..p_base + stride]);
            let p_phase = phase[p_row];
            p_dep.copy_from_slice(&gen_dep[p_row][..rank_words]);

            for r in 0..total_rows {
                if r == p_row {
                    continue;
                }
                if xz[r * stride + word] & bit_mask == 0 {
                    continue;
                }

                let r_base = r * stride;
                phase[r] = rowmul_phase(&p_data, &mut xz, r_base, nw, p_phase, phase[r]);
                xor_words(&mut gen_dep[r][..rank_words], &p_dep[..rank_words]);
            }

            let dest_idx = p_row - n;
            let dest_base = dest_idx * stride;
            xz.copy_within(p_row * stride..p_row * stride + stride, dest_base);
            phase[dest_idx] = p_phase;
            gen_dep[dest_idx][..rank_words].copy_from_slice(&p_dep);

            let p_base = p_row * stride;
            xz[p_base..p_base + stride].fill(0);
            xz[p_base + nw + word] |= bit_mask;
            phase[p_row] = false;

            gen_dep[p_row][..rank_words].fill(0);
            gen_dep[p_row][k / 64] |= 1u64 << (k % 64);

            ref_bits[meas_idx] = false;
        } else {
            scratch[..stride].fill(0);
            let mut scratch_phase = false;
            gen_dep[scratch_idx][..rank_words].fill(0);

            for g in 0..n {
                let d_base = g * stride;
                if xz[d_base + word] & bit_mask == 0 {
                    continue;
                }

                let s_base = (g + n) * stride;
                let s_phase = phase[g + n];
                scratch_phase =
                    rowmul_phase_into(&xz, s_base, &mut scratch, nw, s_phase, scratch_phase);

                let (lo, hi) = gen_dep.split_at_mut(scratch_idx);
                for (dst, &src) in hi[0][..rank_words].iter_mut().zip(&lo[g + n][..rank_words]) {
                    *dst ^= src;
                }
            }

            ref_bits[meas_idx] = scratch_phase;

            for (w, &dep_word) in gen_dep[scratch_idx][..rank_words].iter().enumerate() {
                let mut bits = dep_word;
                while bits != 0 {
                    let bit_pos = bits.trailing_zeros() as usize;
                    let k = w * 64 + bit_pos;
                    if k < rank {
                        flip_rows[k][meas_idx / 64] |= 1u64 << (meas_idx % 64);
                    }
                    bits &= bits - 1;
                }
            }
        }
    }

    let num_meas_words = m_words;
    let lut = if rank >= LUT_MIN_RANK {
        Some(FlipLut::build(&flip_rows, num_meas_words))
    } else {
        None
    };

    Ok(CompiledSampler {
        flip_rows,
        ref_bits_packed: pack_bools(&ref_bits),
        rank,
        num_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed),
        lut,
    })
}

fn compile_measurements_filtered(
    circuit: &Circuit,
    blocks: &[Vec<usize>],
    seed: u64,
) -> Result<CompiledSampler> {
    let num_global_measurements: usize = circuit
        .instructions
        .iter()
        .filter(|i| matches!(i, Instruction::Measure { .. }))
        .count();

    if num_global_measurements == 0 {
        return Ok(CompiledSampler {
            flip_rows: Vec::new(),
            ref_bits_packed: Vec::new(),
            rank: 0,
            num_measurements: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            lut: None,
        });
    }

    let mut qubit_to_block: Vec<usize> = vec![0; circuit.num_qubits];
    for (bi, block) in blocks.iter().enumerate() {
        for &q in block {
            qubit_to_block[q] = bi;
        }
    }

    let mut block_samplers: Vec<CompiledSampler> = Vec::with_capacity(blocks.len());
    for (bi, block) in blocks.iter().enumerate() {
        let (sub_circuit, _qubit_map, _classical_map) = circuit.extract_subcircuit(block);
        let block_seed = seed.wrapping_add(bi as u64 * 0x1234_5678);
        block_samplers.push(compile_measurements(&sub_circuit, block_seed)?);
    }

    let mut meas_map: Vec<(usize, usize)> = Vec::with_capacity(num_global_measurements);
    let mut block_meas_count: Vec<usize> = vec![0; blocks.len()];
    for inst in &circuit.instructions {
        if let Instruction::Measure { qubit, .. } = inst {
            let bi = qubit_to_block[*qubit];
            let local_idx = block_meas_count[bi];
            block_meas_count[bi] += 1;
            meas_map.push((bi, local_idx));
        }
    }

    let m_words = num_global_measurements.div_ceil(64);
    let total_rank: usize = block_samplers.iter().map(|s| s.rank).sum();

    let mut flip_rows: Vec<Vec<u64>> = Vec::with_capacity(total_rank);
    let mut ref_bits_packed: Vec<u64> = vec![0u64; num_global_measurements.div_ceil(64)];

    for (gi, &(bi, li)) in meas_map.iter().enumerate() {
        let src = &block_samplers[bi].ref_bits_packed;
        let bit = (src[li / 64] >> (li % 64)) & 1;
        if bit != 0 {
            ref_bits_packed[gi / 64] |= 1u64 << (gi % 64);
        }
    }

    let mut local_to_global: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
    for (gi, &(bi, _li)) in meas_map.iter().enumerate() {
        local_to_global[bi].push(gi);
    }

    for (bi, sampler) in block_samplers.iter().enumerate() {
        let mapping = &local_to_global[bi];
        for local_row in &sampler.flip_rows {
            let mut global_row = vec![0u64; m_words];
            for (lm, &gi) in mapping.iter().enumerate() {
                if (local_row[lm / 64] >> (lm % 64)) & 1 != 0 {
                    global_row[gi / 64] |= 1u64 << (gi % 64);
                }
            }
            flip_rows.push(global_row);
        }
    }

    let lut = if total_rank >= LUT_MIN_RANK {
        Some(FlipLut::build(&flip_rows, m_words))
    } else {
        None
    };

    Ok(CompiledSampler {
        flip_rows,
        ref_bits_packed,
        rank: total_rank,
        num_measurements: num_global_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed),
        lut,
    })
}

/// Compile a Clifford circuit's measurements into a fast sampler.
///
/// Selects forward (SGI stabilizer + dependency tracking) or backward (Pauli
/// propagation + Gaussian elimination) based on circuit depth. Forward wins
/// for deep circuits (gate_count >= 5×measurements).
pub fn compile_measurements(circuit: &Circuit, seed: u64) -> Result<CompiledSampler> {
    if !circuit.is_clifford_only() {
        return Err(PrismError::IncompatibleBackend {
            backend: "CompiledSampler".to_string(),
            reason: "circuit contains non-Clifford gates".to_string(),
        });
    }

    if circuit.num_qubits >= 4 {
        let blocks = circuit.independent_subsystems();
        if blocks.len() > 1 {
            let max_block = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
            if max_block < circuit.num_qubits {
                return compile_measurements_filtered(circuit, &blocks, seed);
            }
        }
    }

    if circuit.num_qubits >= 64 {
        return compile_forward(circuit, seed);
    }

    let measurement_rows = build_measurement_rows(circuit);
    let num_measurements = measurement_rows.len();

    if num_measurements == 0 {
        return Ok(CompiledSampler {
            flip_rows: Vec::new(),
            ref_bits_packed: Vec::new(),
            rank: 0,
            num_measurements: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            lut: None,
        });
    }

    let n = circuit.num_qubits;

    let x_rows: Vec<Vec<u64>> = measurement_rows
        .iter()
        .map(|(p, _, _)| p.x.clone())
        .collect();
    let signs: Vec<bool> = measurement_rows.iter().map(|(_, _, s)| *s).collect();

    let mut x_copy = x_rows.clone();
    let (rank, pivot_cols) = gaussian_eliminate(&mut x_copy, n);

    let gate_count = circuit
        .instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate { .. } | Instruction::Conditional { .. }
            )
        })
        .count();

    let ref_bits: Vec<bool> = if gate_count > 2 * num_measurements {
        let mini_outcomes = compute_reference_bits(&measurement_rows, n);
        mini_outcomes
            .iter()
            .zip(signs.iter())
            .map(|(&outcome, &sign)| outcome ^ sign)
            .collect()
    } else {
        use crate::backend::stabilizer::StabilizerBackend;
        use crate::backend::Backend;
        let mut stab = StabilizerBackend::new(seed);
        stab.init(circuit.num_qubits, circuit.num_classical_bits)?;
        stab.apply_instructions(&circuit.instructions)?;
        let ref_classical = stab.classical_results().to_vec();
        let classical_bit_order: Vec<usize> = measurement_rows.iter().map(|(_, c, _)| *c).collect();
        classical_bit_order
            .iter()
            .map(|&cbit| {
                if cbit < ref_classical.len() {
                    ref_classical[cbit]
                } else {
                    false
                }
            })
            .collect()
    };

    let num_meas_words = num_measurements.div_ceil(64);
    let mut flip_rows: Vec<Vec<u64>> = vec![vec![0u64; num_meas_words]; rank];

    for (j, &pcol) in pivot_cols.iter().enumerate() {
        for (i, x_row) in x_rows.iter().enumerate() {
            if get_bit(x_row, pcol) {
                flip_rows[j][i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    let lut = if rank >= LUT_MIN_RANK {
        Some(FlipLut::build(&flip_rows, num_meas_words))
    } else {
        None
    };

    Ok(CompiledSampler {
        flip_rows,
        ref_bits_packed: pack_bools(&ref_bits),
        rank,
        num_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed),
        lut,
    })
}

pub fn run_shots_compiled(circuit: &Circuit, num_shots: usize, seed: u64) -> Result<ShotsResult> {
    let mut sampler = compile_measurements(circuit, seed)?;
    let shots = sampler.sample_bulk(num_shots);
    Ok(ShotsResult {
        shots,
        probabilities: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits;
    use crate::sim::BackendKind;

    #[test]
    fn ghz_rank_is_one() {
        let mut c = circuits::ghz_circuit(10);
        c.num_classical_bits = 10;
        for i in 0..10 {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(sampler.rank(), 1, "GHZ-10 should have rank 1");
    }

    #[test]
    fn bell_pairs_rank() {
        let n_pairs = 5;
        let mut c = circuits::independent_bell_pairs(n_pairs);
        let n = c.num_qubits;
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(sampler.rank(), n_pairs, "5 Bell pairs should have rank 5");
    }

    #[test]
    fn random_clifford_rank_is_n() {
        let n = 10;
        let mut c = circuits::clifford_heavy_circuit(n, 50, 42);
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(
            sampler.rank(),
            n,
            "Random Clifford 10q d50 should have rank {n}"
        );
    }

    #[test]
    fn non_clifford_rejected() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::Rx(0.5), &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);
        let result = compile_measurements(&c, 42);
        assert!(result.is_err());
    }

    #[test]
    fn no_measurements_rank_zero() {
        let c = circuits::ghz_circuit(5);
        let sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(sampler.rank(), 0);
        assert_eq!(sampler.num_measurements(), 0);
    }

    #[test]
    fn identity_circuit_all_zeros() {
        let mut c = Circuit::new(4, 4);
        for i in 0..4 {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(sampler.rank(), 0, "Identity circuit should have rank 0");

        let mut sampler = sampler;
        for _ in 0..100 {
            let outcome = sampler.sample();
            assert!(outcome.iter().all(|&b| !b), "All outcomes should be 0");
        }
    }

    #[test]
    fn single_h_measure() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        let sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(sampler.rank(), 1, "H+measure should have rank 1");
    }

    #[test]
    fn ghz_distribution() {
        let n = 10;
        let mut c = circuits::ghz_circuit(n);
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }
        let result = run_shots_compiled(&c, 10_000, 42).unwrap();
        let counts = result.counts();

        let all_zero: Vec<bool> = vec![false; n];
        let all_one: Vec<bool> = vec![true; n];

        let n_zero = counts.get(&all_zero).copied().unwrap_or(0);
        let n_one = counts.get(&all_one).copied().unwrap_or(0);

        assert_eq!(
            counts.len(),
            2,
            "GHZ should produce exactly 2 outcomes, got {}",
            counts.len()
        );
        assert!(
            n_zero + n_one == 10_000,
            "All shots should be all-0 or all-1"
        );
        let ratio = n_zero as f64 / 10_000.0;
        assert!(
            (0.45..=0.55).contains(&ratio),
            "Expected ~50/50, got {ratio:.3}"
        );
    }

    #[test]
    fn bell_pairs_always_agree() {
        let n_pairs = 5;
        let mut c = circuits::independent_bell_pairs(n_pairs);
        let n = c.num_qubits;
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }
        let result = run_shots_compiled(&c, 10_000, 42).unwrap();

        for shot in &result.shots {
            for p in 0..n_pairs {
                assert_eq!(
                    shot[2 * p],
                    shot[2 * p + 1],
                    "Bell pair {p} qubits disagree"
                );
            }
        }
    }

    #[test]
    fn random_clifford_marginals() {
        let n = 10;
        let mut c = circuits::clifford_heavy_circuit(n, 10, 42);
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }
        let compiled = run_shots_compiled(&c, 50_000, 42).unwrap();
        let reference =
            crate::sim::run_shots_with(BackendKind::Stabilizer, &c, 50_000, 42).unwrap();

        for q in 0..n {
            let compiled_ones: usize = compiled.shots.iter().filter(|s| s[q]).count();
            let ref_ones: usize = reference.shots.iter().filter(|s| s[q]).count();
            let compiled_frac = compiled_ones as f64 / 50_000.0;
            let ref_frac = ref_ones as f64 / 50_000.0;
            assert!(
                (compiled_frac - ref_frac).abs() < 0.03,
                "Qubit {q} marginal mismatch: compiled={compiled_frac:.4} ref={ref_frac:.4}"
            );
        }
    }

    #[test]
    fn lut_grouped_sampling_50q() {
        let n = 50;
        let mut c = circuits::clifford_heavy_circuit(n, 10, 42);
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert!(sampler.rank() >= 40, "50q should have high rank for LUT");
        assert!(sampler.lut.is_some(), "rank >= 8 should build LUT");

        let compiled = run_shots_compiled(&c, 5_000, 42).unwrap();
        let reference = crate::sim::run_shots_with(BackendKind::Stabilizer, &c, 5_000, 42).unwrap();

        for q in 0..n {
            let compiled_ones: usize = compiled.shots.iter().filter(|s| s[q]).count();
            let ref_ones: usize = reference.shots.iter().filter(|s| s[q]).count();
            let compiled_frac = compiled_ones as f64 / 5_000.0;
            let ref_frac = ref_ones as f64 / 5_000.0;
            assert!(
                (compiled_frac - ref_frac).abs() < 0.05,
                "q{q} marginal mismatch: compiled={compiled_frac:.4} ref={ref_frac:.4}"
            );
        }
    }

    #[test]
    fn forward_ghz_rank_and_distribution() {
        let n = 10;
        let mut c = circuits::ghz_circuit(n);
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }
        let sampler = compile_forward(&c, 42).unwrap();
        assert_eq!(sampler.rank(), 1, "Forward GHZ-10 should have rank 1");

        let mut sampler = compile_forward(&c, 42).unwrap();
        let shots = sampler.sample_bulk(10_000);
        let all_zero: Vec<bool> = vec![false; n];
        let all_one: Vec<bool> = vec![true; n];
        let n_zero = shots.iter().filter(|s| *s == &all_zero).count();
        let n_one = shots.iter().filter(|s| *s == &all_one).count();
        assert_eq!(
            n_zero + n_one,
            10_000,
            "GHZ should produce only all-0 or all-1"
        );
        let ratio = n_zero as f64 / 10_000.0;
        assert!(
            (0.45..=0.55).contains(&ratio),
            "Expected ~50/50, got {ratio:.3}"
        );
    }

    #[test]
    fn forward_bell_pairs_agree() {
        let n_pairs = 5;
        let mut c = circuits::independent_bell_pairs(n_pairs);
        let n = c.num_qubits;
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }
        let sampler = compile_forward(&c, 42).unwrap();
        assert_eq!(
            sampler.rank(),
            n_pairs,
            "Forward 5 Bell pairs should have rank 5"
        );

        let mut sampler = compile_forward(&c, 42).unwrap();
        let shots = sampler.sample_bulk(10_000);
        for shot in &shots {
            for p in 0..n_pairs {
                assert_eq!(
                    shot[2 * p],
                    shot[2 * p + 1],
                    "Bell pair {p} qubits disagree"
                );
            }
        }
    }

    #[test]
    fn forward_identity_all_zeros() {
        let mut c = Circuit::new(4, 4);
        for i in 0..4 {
            c.add_measure(i, i);
        }
        let sampler = compile_forward(&c, 42).unwrap();
        assert_eq!(sampler.rank(), 0, "Forward identity should have rank 0");

        let mut sampler = sampler;
        for _ in 0..100 {
            let outcome = sampler.sample();
            assert!(outcome.iter().all(|&b| !b), "All outcomes should be 0");
        }
    }

    #[test]
    fn forward_single_h_measure() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        let sampler = compile_forward(&c, 42).unwrap();
        assert_eq!(sampler.rank(), 1, "Forward H+measure should have rank 1");
    }

    #[test]
    fn forward_x_measure_always_one() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::X, &[0]);
        c.add_measure(0, 0);
        let mut sampler = compile_forward(&c, 42).unwrap();
        assert_eq!(
            sampler.rank(),
            0,
            "X+measure should have rank 0 (deterministic)"
        );
        for _ in 0..100 {
            let outcome = sampler.sample();
            assert!(outcome[0], "X(q0) + measure should always give 1");
        }
    }

    #[test]
    fn forward_random_clifford_marginals() {
        let n = 10;
        let mut c = circuits::clifford_heavy_circuit(n, 10, 42);
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }

        let mut forward = compile_forward(&c, 42).unwrap();
        let mut backward = compile_measurements(&c, 42).unwrap();

        assert_eq!(forward.rank(), backward.rank(), "Ranks must match");

        let fwd_shots = forward.sample_bulk(50_000);
        let bwd_shots = backward.sample_bulk(50_000);

        for q in 0..n {
            let fwd_ones: usize = fwd_shots.iter().filter(|s| s[q]).count();
            let bwd_ones: usize = bwd_shots.iter().filter(|s| s[q]).count();
            let fwd_frac = fwd_ones as f64 / 50_000.0;
            let bwd_frac = bwd_ones as f64 / 50_000.0;
            assert!(
                (fwd_frac - bwd_frac).abs() < 0.03,
                "Qubit {q} marginal mismatch: forward={fwd_frac:.4} backward={bwd_frac:.4}"
            );
        }
    }

    #[test]
    fn forward_clifford_50q_marginals() {
        let n = 50;
        let mut c = circuits::clifford_heavy_circuit(n, 10, 42);
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }

        let mut forward = compile_forward(&c, 42).unwrap();
        let reference = crate::sim::run_shots_with(BackendKind::Stabilizer, &c, 5_000, 42).unwrap();

        let fwd_shots = forward.sample_bulk(5_000);

        for q in 0..n {
            let fwd_ones: usize = fwd_shots.iter().filter(|s| s[q]).count();
            let ref_ones: usize = reference.shots.iter().filter(|s| s[q]).count();
            let fwd_frac = fwd_ones as f64 / 5_000.0;
            let ref_frac = ref_ones as f64 / 5_000.0;
            assert!(
                (fwd_frac - ref_frac).abs() < 0.05,
                "q{q} marginal mismatch: forward={fwd_frac:.4} ref={ref_frac:.4}"
            );
        }
    }

    #[test]
    fn rank_analysis_across_circuit_types() {
        let sizes = [10, 50, 100, 200];

        for &n in &sizes {
            let mut c = circuits::ghz_circuit(n);
            c.num_classical_bits = n;
            for i in 0..n {
                c.add_measure(i, i);
            }
            let sampler = compile_measurements(&c, 42).unwrap();
            assert_eq!(sampler.rank(), 1, "GHZ-{n} should have rank 1");
        }

        for &n in &sizes {
            let pairs = n / 2;
            let mut c = circuits::independent_bell_pairs(pairs);
            let nq = c.num_qubits;
            c.num_classical_bits = nq;
            for i in 0..nq {
                c.add_measure(i, i);
            }
            let sampler = compile_measurements(&c, 42).unwrap();
            assert_eq!(
                sampler.rank(),
                pairs,
                "Bell-{pairs} should have rank {pairs}"
            );
        }

        for &n in &sizes {
            let mut c = circuits::clifford_heavy_circuit(n, 50, 42);
            c.num_classical_bits = n;
            for i in 0..n {
                c.add_measure(i, i);
            }
            let sampler = compile_measurements(&c, 42).unwrap();
            assert!(
                sampler.rank() >= n - 1,
                "Random Clifford {n}q d50 should have rank ~{n}, got {}",
                sampler.rank()
            );
        }

        for &n in &sizes {
            let mut c = Circuit::new(n, n);
            for i in 0..n {
                c.add_gate(Gate::H, &[i]);
                c.add_measure(i, i);
            }
            let sampler = compile_measurements(&c, 42).unwrap();
            assert_eq!(
                sampler.rank(),
                n,
                "Product H-measure {n}q should have rank {n} (independent random bits)"
            );
        }
    }

    #[test]
    fn filtered_bell_pairs_matches_monolithic() {
        let n_pairs = 50;
        let n = 2 * n_pairs;
        let mut c = circuits::independent_bell_pairs(n_pairs);
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }

        let blocks = c.independent_subsystems();
        assert_eq!(
            blocks.len(),
            n_pairs,
            "Bell pairs should decompose into {n_pairs} blocks"
        );

        let mut sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(
            sampler.rank(),
            n_pairs,
            "Bell pairs rank should be {n_pairs}"
        );

        let shots = sampler.sample_bulk(10_000);
        for shot in &shots {
            for p in 0..n_pairs {
                assert_eq!(
                    shot[2 * p],
                    shot[2 * p + 1],
                    "Bell pair {p}: qubits must agree"
                );
            }
        }

        let ones: usize = shots.iter().filter(|s| s[0]).count();
        let frac = ones as f64 / shots.len() as f64;
        assert!(
            (frac - 0.5).abs() < 0.05,
            "Bell pair first qubit should be ~50/50, got {frac:.3}"
        );
    }

    #[test]
    fn filtered_product_h_matches_monolithic() {
        let n = 100;
        let mut c = Circuit::new(n, n);
        for i in 0..n {
            c.add_gate(Gate::H, &[i]);
            c.add_measure(i, i);
        }

        let blocks = c.independent_subsystems();
        assert_eq!(
            blocks.len(),
            n,
            "Product H should decompose into {n} blocks"
        );

        let mut sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(sampler.rank(), n);

        let shots = sampler.sample_bulk(5_000);
        for q in 0..n {
            let ones: usize = shots.iter().filter(|s| s[q]).count();
            let frac = ones as f64 / shots.len() as f64;
            assert!(
                (frac - 0.5).abs() < 0.06,
                "Qubit {q} should be ~50/50, got {frac:.3}"
            );
        }
    }
}

use super::{flip_bit, get_bit, set_bit, PauliVec};
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

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
pub(super) fn rowmul_phase(
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
pub(super) fn rowmul_phase_into(
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

/// Compute reference measurement outcomes by simulating measurements of the
/// propagated Paulis on |0⟩^n using the Aaronson-Gottesman measurement protocol.
/// O(m × n²/64) — avoids re-simulating all T circuit gates through the stabilizer.
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
pub(super) fn colmajor_forward_sim(
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

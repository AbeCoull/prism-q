use std::hash::{Hash, Hasher};

use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::ShotsResult;
use rand::RngCore;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, PartialEq, Eq)]
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

    #[inline(always)]
    pub fn is_diagonal(&self) -> bool {
        self.x.iter().all(|&w| w == 0)
    }

    #[inline(always)]
    pub fn has_x_or_y(&self, qubit: usize) -> bool {
        get_bit(&self.x, qubit)
    }
}

impl Hash for PauliVec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.z.hash(state);
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
pub(crate) fn flip_bit(words: &mut [u64], qubit: usize) {
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

#[derive(Debug, Clone)]
pub struct SparseParity {
    pub col_indices: Vec<u32>,
    pub row_offsets: Vec<u32>,
    pub num_rows: usize,
}

impl SparseParity {
    pub fn from_flip_rows(flip_rows: &[Vec<u64>], num_measurements: usize) -> Self {
        let num_rows = num_measurements;
        let rank = flip_rows.len();
        let mut row_offsets = Vec::with_capacity(num_rows + 1);
        let mut col_indices = Vec::new();

        for m in 0..num_rows {
            row_offsets.push(col_indices.len() as u32);
            let w = m / 64;
            let bit = m % 64;
            for (j, row) in flip_rows.iter().enumerate().take(rank) {
                if (row[w] >> bit) & 1 != 0 {
                    col_indices.push(j as u32);
                }
            }
        }
        row_offsets.push(col_indices.len() as u32);

        Self {
            col_indices,
            row_offsets,
            num_rows,
        }
    }

    #[inline(always)]
    pub fn row_weight(&self, row: usize) -> usize {
        (self.row_offsets[row + 1] - self.row_offsets[row]) as usize
    }

    pub fn row_cols(&self, row: usize) -> &[u32] {
        let start = self.row_offsets[row] as usize;
        let end = self.row_offsets[row + 1] as usize;
        &self.col_indices[start..end]
    }

    pub fn build_xor_dag(&self) -> XorDag {
        let n = self.num_rows;
        let mut entries: Vec<XorDagEntry> = Vec::with_capacity(n);

        for m in 0..n {
            let cols = self.row_cols(m);
            let weight = cols.len();

            let mut best_parent = None;
            let mut best_residual_weight = weight;

            for p in 0..m {
                let parent_cols = self.row_cols(p);
                let sym_diff_size = symmetric_difference_size(cols, parent_cols);
                if sym_diff_size < best_residual_weight {
                    best_residual_weight = sym_diff_size;
                    best_parent = Some(p);
                }
            }

            if let Some(p) = best_parent {
                if best_residual_weight < weight {
                    let parent_cols = self.row_cols(p);
                    let residual = symmetric_difference(cols, parent_cols);
                    entries.push(XorDagEntry {
                        parent: Some(p),
                        residual_cols: residual,
                    });
                } else {
                    entries.push(XorDagEntry {
                        parent: None,
                        residual_cols: cols.to_vec(),
                    });
                }
            } else {
                entries.push(XorDagEntry {
                    parent: None,
                    residual_cols: cols.to_vec(),
                });
            }
        }

        let original_weight: usize = (0..n).map(|m| self.row_weight(m)).sum();
        let dag_weight: usize = entries.iter().map(|e| e.residual_cols.len()).sum();

        XorDag {
            entries,
            original_weight,
            dag_weight,
        }
    }

    pub fn stats(&self) -> ParityStats {
        if self.num_rows == 0 {
            return ParityStats {
                min_weight: 0,
                max_weight: 0,
                mean_weight: 0.0,
                total_weight: 0,
                num_deterministic: 0,
            };
        }
        let mut min_w = usize::MAX;
        let mut max_w = 0usize;
        let mut total = 0usize;
        let mut num_det = 0usize;
        for r in 0..self.num_rows {
            let w = self.row_weight(r);
            min_w = min_w.min(w);
            max_w = max_w.max(w);
            total += w;
            if w == 0 {
                num_det += 1;
            }
        }
        ParityStats {
            min_weight: min_w,
            max_weight: max_w,
            mean_weight: total as f64 / self.num_rows as f64,
            total_weight: total,
            num_deterministic: num_det,
        }
    }

    pub fn find_blocks(&self, rank: usize) -> Option<Vec<Vec<usize>>> {
        if self.num_rows <= 1 || rank == 0 {
            return None;
        }

        let mut col_to_rows: Vec<Vec<usize>> = vec![Vec::new(); rank];
        for m in 0..self.num_rows {
            for &c in self.row_cols(m) {
                col_to_rows[c as usize].push(m);
            }
        }

        let mut parent: Vec<usize> = (0..self.num_rows).collect();
        let mut size: Vec<usize> = vec![1; self.num_rows];

        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut root = x;
            while parent[root] != root {
                root = parent[root];
            }
            let mut cur = x;
            while parent[cur] != root {
                let next = parent[cur];
                parent[cur] = root;
                cur = next;
            }
            root
        }

        fn union(parent: &mut [usize], size: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra == rb {
                return;
            }
            if size[ra] < size[rb] {
                parent[ra] = rb;
                size[rb] += size[ra];
            } else {
                parent[rb] = ra;
                size[ra] += size[rb];
            }
        }

        for rows in &col_to_rows {
            for i in 1..rows.len() {
                union(&mut parent, &mut size, rows[0], rows[i]);
            }
        }

        let mut block_map: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for m in 0..self.num_rows {
            let root = find(&mut parent, m);
            block_map.entry(root).or_default().push(m);
        }

        let blocks: Vec<Vec<usize>> = block_map.into_values().collect();
        if blocks.len() <= 1 {
            return None;
        }

        Some(blocks)
    }

    pub fn compile_detection_events(&self, pairs: &[(usize, usize)]) -> SparseParity {
        let num_events = pairs.len();
        let mut row_offsets = Vec::with_capacity(num_events + 1);
        let mut col_indices = Vec::new();

        for &(m_a, m_b) in pairs {
            row_offsets.push(col_indices.len() as u32);
            let cols_a = self.row_cols(m_a);
            let cols_b = self.row_cols(m_b);
            let sym_diff = symmetric_difference(cols_a, cols_b);
            col_indices.extend_from_slice(&sym_diff);
        }
        row_offsets.push(col_indices.len() as u32);

        SparseParity {
            col_indices,
            row_offsets,
            num_rows: num_events,
        }
    }
}

#[derive(Debug, Clone)]
pub struct XorDagEntry {
    pub parent: Option<usize>,
    pub residual_cols: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct XorDag {
    pub entries: Vec<XorDagEntry>,
    pub original_weight: usize,
    pub dag_weight: usize,
}

fn symmetric_difference_size(a: &[u32], b: &[u32]) -> usize {
    let mut count = 0;
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                count += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                count += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    count + (a.len() - i) + (b.len() - j)
}

fn symmetric_difference(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

#[derive(Debug, Clone)]
pub struct ParityStats {
    pub min_weight: usize,
    pub max_weight: usize,
    pub mean_weight: f64,
    pub total_weight: usize,
    pub num_deterministic: usize,
}

pub struct ParityBlock {
    pub meas_indices: Vec<usize>,
    pub sparse: SparseParity,
    pub block_rank: usize,
    pub ref_bits_packed: Vec<u64>,
}

pub struct ParityBlocks {
    pub blocks: Vec<ParityBlock>,
}

impl ParityBlocks {
    fn build(
        global_sparse: &SparseParity,
        block_meas: Vec<Vec<usize>>,
        rank: usize,
        ref_bits: &[u64],
    ) -> Self {
        let mut blocks = Vec::with_capacity(block_meas.len());
        for meas_indices in block_meas {
            let mut col_set: Vec<u32> = Vec::new();
            for &m in &meas_indices {
                for &c in global_sparse.row_cols(m) {
                    col_set.push(c);
                }
            }
            col_set.sort_unstable();
            col_set.dedup();
            let block_rank = col_set.len();

            let mut col_remap = vec![0u32; rank];
            for (new_idx, &old_idx) in col_set.iter().enumerate() {
                col_remap[old_idx as usize] = new_idx as u32;
            }

            let num_rows = meas_indices.len();
            let mut row_offsets = Vec::with_capacity(num_rows + 1);
            let mut col_indices = Vec::new();
            for &m in &meas_indices {
                row_offsets.push(col_indices.len() as u32);
                for &c in global_sparse.row_cols(m) {
                    col_indices.push(col_remap[c as usize]);
                }
            }
            row_offsets.push(col_indices.len() as u32);

            let sparse = SparseParity {
                col_indices,
                row_offsets,
                num_rows,
            };

            let ref_words = num_rows.div_ceil(64);
            let mut block_ref = vec![0u64; ref_words];
            for (local_m, &global_m) in meas_indices.iter().enumerate() {
                let ref_bit = (ref_bits[global_m / 64] >> (global_m % 64)) & 1;
                if ref_bit != 0 {
                    block_ref[local_m / 64] |= 1u64 << (local_m % 64);
                }
            }

            blocks.push(ParityBlock {
                meas_indices,
                sparse,
                block_rank,
                ref_bits_packed: block_ref,
            });
        }
        ParityBlocks { blocks }
    }
}

pub struct CompiledSampler {
    flip_rows: Vec<Vec<u64>>,
    ref_bits_packed: Vec<u64>,
    rank: usize,
    num_measurements: usize,
    rng: ChaCha8Rng,
    lut: Option<FlipLut>,
    sparse: Option<SparseParity>,
    xor_dag: Option<XorDag>,
    parity_blocks: Option<ParityBlocks>,
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

    pub fn sparse(&self) -> Option<&SparseParity> {
        self.sparse.as_ref()
    }

    pub fn parity_stats(&self) -> Option<ParityStats> {
        self.sparse.as_ref().map(|s| s.stats())
    }

    pub fn sample(&mut self) -> Vec<bool> {
        let num_meas_words = self.num_measurements.div_ceil(64);
        let mut accum = vec![0u64; num_meas_words];
        self.sample_into(&mut accum);
        self.unpack_result(&accum)
    }

    pub(crate) fn sample_bulk_words_shot_major(&mut self, num_shots: usize) -> (Vec<u64>, usize) {
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

    fn bts_weight_threshold(&self, num_shots: usize) -> usize {
        let s_words = num_shots.div_ceil(64);
        let m_words = self.num_measurements.div_ceil(64);
        let lut_groups = self.rank.div_ceil(LUT_GROUP_SIZE);
        if s_words == 0 || self.num_measurements == 0 {
            return 0;
        }
        (num_shots as u64 * lut_groups as u64 * m_words as u64
            / (s_words as u64 * self.num_measurements as u64)) as usize
    }

    fn should_use_bts(&self, num_shots: usize) -> bool {
        if let Some(sparse) = &self.sparse {
            if self.rank == 0 {
                return false;
            }
            let m_words = self.num_measurements.div_ceil(64) as u64;
            let lut_groups = (self.rank.div_ceil(LUT_GROUP_SIZE)) as u64;

            let lut_alloc_bytes = num_shots as u64 * (lut_groups + m_words * 8);
            if lut_alloc_bytes > MAX_LUT_ALLOC_BYTES {
                return true;
            }

            let s_words = num_shots.div_ceil(64);
            let stats = sparse.stats();
            let bts_work = stats.total_weight as u64 * s_words as u64;
            let lut_work = num_shots as u64 * lut_groups * m_words;
            bts_work < lut_work
        } else {
            false
        }
    }

    #[allow(dead_code)]
    fn partition_measurements(&self, num_shots: usize) -> (Vec<usize>, Vec<usize>) {
        let threshold = self.bts_weight_threshold(num_shots);
        let sparse = match &self.sparse {
            Some(s) => s,
            None => return (Vec::new(), (0..self.num_measurements).collect()),
        };
        let mut light = Vec::new();
        let mut heavy = Vec::new();
        for m in 0..self.num_measurements {
            if sparse.row_weight(m) <= threshold {
                light.push(m);
            } else {
                heavy.push(m);
            }
        }
        (light, heavy)
    }

    pub(crate) fn ref_bits_packed(&self) -> &[u64] {
        &self.ref_bits_packed
    }

    pub fn sample_bulk(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        self.sample_bulk_packed(num_shots).to_shots()
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

    pub fn sample_bulk_packed(&mut self, num_shots: usize) -> PackedShots {
        let m_words = self.num_measurements.div_ceil(64);
        let s_words = num_shots.div_ceil(64);
        if num_shots == 0 || self.num_measurements == 0 {
            return PackedShots {
                data: Vec::new(),
                num_shots,
                num_measurements: self.num_measurements,
                m_words,
                s_words,
                layout: ShotLayout::ShotMajor,
            };
        }
        if self.rank == 0 {
            let mut data = vec![0u64; num_shots * m_words];
            for s in 0..num_shots {
                let base = s * m_words;
                data[base..base + m_words].copy_from_slice(&self.ref_bits_packed);
            }
            return PackedShots {
                data,
                num_shots,
                num_measurements: self.num_measurements,
                m_words,
                s_words,
                layout: ShotLayout::ShotMajor,
            };
        }

        if self.should_use_bts(num_shots) {
            return self.sample_bulk_packed_bts(num_shots, m_words, s_words);
        }

        let (mut data, _) = self.sample_bulk_words_shot_major(num_shots);
        for s in 0..num_shots {
            let base = s * m_words;
            xor_words(&mut data[base..base + m_words], &self.ref_bits_packed);
        }
        PackedShots {
            data,
            num_shots,
            num_measurements: self.num_measurements,
            m_words,
            s_words,
            layout: ShotLayout::ShotMajor,
        }
    }

    fn sample_bulk_packed_bts(
        &mut self,
        num_shots: usize,
        m_words: usize,
        s_words: usize,
    ) -> PackedShots {
        let num_meas = self.num_measurements;

        if let Some(pb) = &self.parity_blocks {
            let block_seeds: Vec<u64> = (0..pb.blocks.len())
                .map(|_| self.rng.next_u64())
                .collect();

            #[cfg(feature = "parallel")]
            let block_results: Vec<(Vec<u64>, &[usize])> = {
                use rayon::prelude::*;
                pb.blocks
                    .par_iter()
                    .zip(block_seeds.par_iter())
                    .map(|(block, &seed)| {
                        let mut block_rng = ChaCha8Rng::seed_from_u64(seed);
                        let data = sample_bts_meas_major(
                            &block.sparse,
                            num_shots,
                            &block.ref_bits_packed,
                            &mut block_rng,
                            block.block_rank,
                        );
                        (data, block.meas_indices.as_slice())
                    })
                    .collect()
            };

            #[cfg(not(feature = "parallel"))]
            let block_results: Vec<(Vec<u64>, &[usize])> = pb
                .blocks
                .iter()
                .zip(block_seeds.iter())
                .map(|(block, &seed)| {
                    let mut block_rng = ChaCha8Rng::seed_from_u64(seed);
                    let data = sample_bts_meas_major(
                        &block.sparse,
                        num_shots,
                        &block.ref_bits_packed,
                        &mut block_rng,
                        block.block_rank,
                    );
                    (data, block.meas_indices.as_slice())
                })
                .collect();

            let mut meas_major = vec![0u64; num_meas * s_words];
            for (block_data, meas_indices) in &block_results {
                for (local_m, &global_m) in meas_indices.iter().enumerate() {
                    let src = &block_data[local_m * s_words..(local_m + 1) * s_words];
                    let dst = &mut meas_major[global_m * s_words..(global_m + 1) * s_words];
                    dst.copy_from_slice(src);
                }
            }

            return PackedShots {
                data: meas_major,
                num_shots,
                num_measurements: num_meas,
                m_words,
                s_words,
                layout: ShotLayout::MeasMajor,
            };
        }

        let sparse = self.sparse.as_ref().unwrap();

        if num_shots <= BTS_BATCH_SHOTS {
            let data = bts_single_pass(
                sparse,
                self.xor_dag.as_ref(),
                num_shots,
                &self.ref_bits_packed,
                &mut self.rng,
                self.rank,
            );
            return PackedShots {
                data,
                num_shots,
                num_measurements: num_meas,
                m_words,
                s_words,
                layout: ShotLayout::MeasMajor,
            };
        }

        let data = bts_batched(
            sparse,
            self.xor_dag.as_ref(),
            num_shots,
            s_words,
            &self.ref_bits_packed,
            &mut self.rng,
            self.rank,
        );
        PackedShots {
            data,
            num_shots,
            num_measurements: num_meas,
            m_words,
            s_words,
            layout: ShotLayout::MeasMajor,
        }
    }

    pub fn sample_streaming<F>(&mut self, total_shots: usize, batch_size: usize, mut callback: F)
    where
        F: FnMut(&PackedShots),
    {
        let mut remaining = total_shots;
        while remaining > 0 {
            let this_batch = remaining.min(batch_size);
            let packed = self.sample_bulk_packed(this_batch);
            callback(&packed);
            remaining -= this_batch;
        }
    }

    pub fn sample_counts_streaming(
        &mut self,
        total_shots: usize,
        batch_size: usize,
    ) -> std::collections::HashMap<Vec<u64>, u64> {
        use std::collections::HashMap;
        let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
        self.sample_streaming(total_shots, batch_size, |packed| {
            for (key, count) in packed.counts() {
                *counts.entry(key).or_insert(0) += count;
            }
        });
        counts
    }

    pub fn sample_detection_events(
        &mut self,
        pairs: &[(usize, usize)],
        num_shots: usize,
    ) -> PackedShots {
        let sparse = self.sparse.as_ref().expect("sparse parity required");
        let det_sparse = sparse.compile_detection_events(pairs);
        let num_events = det_sparse.num_rows;
        let m_words = num_events.div_ceil(64);
        let s_words = num_shots.div_ceil(64);

        if num_events == 0 || num_shots == 0 || self.rank == 0 {
            return PackedShots {
                data: vec![0u64; num_events * s_words],
                num_shots,
                num_measurements: num_events,
                m_words,
                s_words,
                layout: ShotLayout::MeasMajor,
            };
        }

        let det_ref = vec![0u64; m_words];

        let data = if num_shots > BTS_BATCH_SHOTS {
            bts_batched(
                &det_sparse,
                None,
                num_shots,
                s_words,
                &det_ref,
                &mut self.rng,
                self.rank,
            )
        } else {
            sample_bts_meas_major(
                &det_sparse,
                num_shots,
                &det_ref,
                &mut self.rng,
                self.rank,
            )
        };

        PackedShots {
            data,
            num_shots,
            num_measurements: num_events,
            m_words,
            s_words,
            layout: ShotLayout::MeasMajor,
        }
    }

    pub fn exact_counts(&self) -> Option<std::collections::HashMap<Vec<u64>, u64>> {
        if self.rank > MAX_RANK_FOR_GRAY_CODE {
            return None;
        }
        let sparse = self.sparse.as_ref()?;
        Some(gray_code_exact_counts(
            sparse,
            self.rank,
            &self.ref_bits_packed,
            self.num_measurements,
        ))
    }

    pub fn marginal_probabilities(&self) -> Vec<f64> {
        let mut probs = vec![0.5f64; self.num_measurements];
        if let Some(sparse) = &self.sparse {
            for (m, p) in probs.iter_mut().enumerate() {
                if sparse.row_weight(m) == 0 {
                    let ref_bit = (self.ref_bits_packed[m / 64] >> (m % 64)) & 1;
                    *p = ref_bit as f64;
                }
            }
        } else {
            for (m, p) in probs.iter_mut().enumerate() {
                let mut depends_on_random = false;
                for row in &self.flip_rows {
                    let w = m / 64;
                    if w < row.len() && (row[w] >> (m % 64)) & 1 != 0 {
                        depends_on_random = true;
                        break;
                    }
                }
                if !depends_on_random {
                    let ref_bit = (self.ref_bits_packed[m / 64] >> (m % 64)) & 1;
                    *p = ref_bit as f64;
                }
            }
        }
        probs
    }

    pub fn parity_report(&self) -> String {
        let mut report = format!(
            "CompiledSampler: {} measurements, rank {}, {} flip rows\n",
            self.num_measurements,
            self.rank,
            self.flip_rows.len()
        );
        if let Some(sparse) = &self.sparse {
            let stats = sparse.stats();
            report.push_str(&format!(
                "Parity matrix: {} rows, total weight {}\n\
                 Weight range: {} to {}, mean {:.1}\n\
                 Deterministic measurements: {}\n",
                sparse.num_rows,
                stats.total_weight,
                stats.min_weight,
                stats.max_weight,
                stats.mean_weight,
                stats.num_deterministic,
            ));
            let mut histogram = [0usize; 8];
            for m in 0..sparse.num_rows {
                let w = sparse.row_weight(m);
                let bucket = w.min(7);
                histogram[bucket] += 1;
            }
            report.push_str("Weight histogram: ");
            for (i, &count) in histogram.iter().enumerate() {
                if count > 0 {
                    if i < 7 {
                        report.push_str(&format!("w{}={} ", i, count));
                    } else {
                        report.push_str(&format!("w7+={} ", count));
                    }
                }
            }
            report.push('\n');
        } else {
            report.push_str("No sparse parity matrix available\n");
        }
        report
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotLayout {
    ShotMajor,
    MeasMajor,
}

#[derive(Debug, Clone)]
pub struct PackedShots {
    data: Vec<u64>,
    num_shots: usize,
    num_measurements: usize,
    m_words: usize,
    s_words: usize,
    layout: ShotLayout,
}

impl PackedShots {
    #[allow(dead_code)]
    pub(crate) fn empty(
        num_shots: usize,
        num_measurements: usize,
        m_words: usize,
        s_words: usize,
    ) -> Self {
        Self {
            data: Vec::new(),
            num_shots,
            num_measurements,
            m_words,
            s_words,
            layout: ShotLayout::ShotMajor,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn from_shot_major(
        data: Vec<u64>,
        num_shots: usize,
        num_measurements: usize,
        m_words: usize,
    ) -> Self {
        Self {
            data,
            num_shots,
            num_measurements,
            m_words,
            s_words: num_shots.div_ceil(64),
            layout: ShotLayout::ShotMajor,
        }
    }

    pub fn num_shots(&self) -> usize {
        self.num_shots
    }

    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    pub fn layout(&self) -> ShotLayout {
        self.layout
    }

    #[allow(dead_code)]
    pub(crate) fn data_mut(&mut self) -> &mut [u64] {
        &mut self.data
    }

    #[inline(always)]
    pub fn get_bit(&self, shot: usize, measurement: usize) -> bool {
        match self.layout {
            ShotLayout::ShotMajor => {
                let base = shot * self.m_words;
                let w = measurement / 64;
                (self.data[base + w] >> (measurement % 64)) & 1 != 0
            }
            ShotLayout::MeasMajor => {
                let base = measurement * self.s_words;
                let w = shot / 64;
                (self.data[base + w] >> (shot % 64)) & 1 != 0
            }
        }
    }

    pub fn shot_words(&self, shot: usize) -> &[u64] {
        assert!(
            self.layout == ShotLayout::ShotMajor,
            "shot_words requires ShotMajor layout"
        );
        let base = shot * self.m_words;
        &self.data[base..base + self.m_words]
    }

    pub fn to_shots(&self) -> Vec<Vec<bool>> {
        let mut shots = Vec::with_capacity(self.num_shots);
        for s in 0..self.num_shots {
            let mut shot = Vec::with_capacity(self.num_measurements);
            for m in 0..self.num_measurements {
                shot.push(self.get_bit(s, m));
            }
            shots.push(shot);
        }
        shots
    }

    pub fn counts(&self) -> std::collections::HashMap<Vec<u64>, u64> {
        use std::collections::HashMap;
        let mut map = HashMap::new();

        match self.layout {
            ShotLayout::ShotMajor => {
                for s in 0..self.num_shots {
                    let base = s * self.m_words;
                    let words = self.data[base..base + self.m_words].to_vec();
                    *map.entry(words).or_insert(0) += 1;
                }
            }
            ShotLayout::MeasMajor => {
                let batch_size = 64;
                let mut shot_buf = vec![0u64; batch_size * self.m_words];
                for batch_start in (0..self.num_shots).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(self.num_shots);
                    let batch_len = batch_end - batch_start;
                    shot_buf[..batch_len * self.m_words].fill(0);

                    let sw_base = batch_start / 64;
                    let bit_off = batch_start % 64;

                    for m in 0..self.num_measurements {
                        let mw = m / 64;
                        let mbit = m % 64;
                        let meas_row = &self.data[m * self.s_words..];
                        let word = meas_row[sw_base];
                        let shifted = word >> bit_off;
                        for s in 0..batch_len {
                            if (shifted >> s) & 1 != 0 {
                                shot_buf[s * self.m_words + mw] |= 1u64 << mbit;
                            }
                        }
                    }

                    for s in 0..batch_len {
                        let base = s * self.m_words;
                        let words = shot_buf[base..base + self.m_words].to_vec();
                        *map.entry(words).or_insert(0) += 1;
                    }
                }
            }
        }
        map
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

fn bts_single_pass(
    sparse: &SparseParity,
    xor_dag: Option<&XorDag>,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut ChaCha8Rng,
    rank: usize,
) -> Vec<u64> {
    if let Some(dag) = xor_dag {
        sample_bts_meas_major_dag(sparse, dag, num_shots, ref_bits, rng, rank)
    } else {
        sample_bts_meas_major(sparse, num_shots, ref_bits, rng, rank)
    }
}

fn bts_batched(
    sparse: &SparseParity,
    xor_dag: Option<&XorDag>,
    num_shots: usize,
    total_s_words: usize,
    ref_bits: &[u64],
    rng: &mut ChaCha8Rng,
    rank: usize,
) -> Vec<u64> {
    let num_meas = sparse.num_rows;

    #[cfg(feature = "parallel")]
    {
        let num_threads = rayon::current_num_threads();
        if num_threads > 1 {
            let shots_per_thread =
                (num_shots.div_ceil(num_threads) / 64) * 64;
            if shots_per_thread >= 64 {
                let base_seed = rng.next_u64();

                let chunks: Vec<(usize, usize)> = (0..num_threads)
                    .map(|t| {
                        let start = t * shots_per_thread;
                        let end = ((t + 1) * shots_per_thread).min(num_shots);
                        (start, end)
                    })
                    .filter(|(s, e)| s < e)
                    .collect();

                let mut output = vec![0u64; num_meas * total_s_words];
                let out_ptr = output.as_mut_ptr();

                {
                    use rayon::prelude::*;
                    let out_slice = &mut output[..];
                    let total_sw = total_s_words;
                    let nm = num_meas;

                    chunks
                        .into_par_iter()
                        .enumerate()
                        .map(|(t, (shot_start, shot_end))| {
                            let chunk_shots = shot_end - shot_start;
                            let word_offset = shot_start / 64;
                            let mut thread_rng = ChaCha8Rng::seed_from_u64(base_seed);
                            thread_rng.set_stream(t as u64 + 1);

                            let mut results = Vec::new();
                            let mut chunk_done = 0usize;
                            while chunk_done < chunk_shots {
                                let batch_shots =
                                    (chunk_shots - chunk_done).min(BTS_BATCH_SHOTS);
                                let batch_offset = word_offset + chunk_done / 64;
                                let batch_data = sample_bts_meas_major(
                                    sparse,
                                    batch_shots,
                                    ref_bits,
                                    &mut thread_rng,
                                    rank,
                                );
                                results.push((batch_data, batch_shots, batch_offset));
                                chunk_done += batch_shots;
                            }
                            results
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .for_each(|thread_results| {
                            for (batch_data, batch_shots, batch_offset) in thread_results {
                                let batch_s_words = batch_shots.div_ceil(64);
                                for m in 0..nm {
                                    let src = &batch_data
                                        [m * batch_s_words..(m + 1) * batch_s_words];
                                    let dst_start = m * total_sw + batch_offset;
                                    out_slice[dst_start..dst_start + batch_s_words]
                                        .copy_from_slice(src);
                                }
                            }
                        });
                }

                let _ = out_ptr;
                return output;
            }
        }
    }

    let mut output = vec![0u64; num_meas * total_s_words];
    let mut shots_done = 0usize;

    while shots_done < num_shots {
        let batch_shots = (num_shots - shots_done).min(BTS_BATCH_SHOTS);
        let batch_s_words = batch_shots.div_ceil(64);
        let word_offset = shots_done / 64;

        let batch_data = bts_single_pass(sparse, xor_dag, batch_shots, ref_bits, rng, rank);

        for m in 0..num_meas {
            let src = &batch_data[m * batch_s_words..(m + 1) * batch_s_words];
            let dst_start = m * total_s_words + word_offset;
            output[dst_start..dst_start + batch_s_words].copy_from_slice(src);
        }

        shots_done += batch_shots;
    }

    output
}

fn sample_bts_meas_major(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut ChaCha8Rng,
    rank: usize,
) -> Vec<u64> {
    let num_meas = sparse.num_rows;
    let s_words = num_shots.div_ceil(64);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && num_shots >= 256 {
            // SAFETY: AVX2 detected, all pointer arithmetic bounded by allocation sizes
            return unsafe { sample_bts_meas_major_avx2(sparse, num_shots, ref_bits, rng, rank) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if num_shots >= 128 {
            // SAFETY: NEON is baseline on aarch64, pointers are valid
            return unsafe { sample_bts_meas_major_neon(sparse, num_shots, ref_bits, rng, rank) };
        }
    }

    let mut meas_major = vec![0u64; num_meas * s_words];
    let mut random_bits = vec![0u64; rank];

    for batch in 0..s_words {
        for r in random_bits.iter_mut().take(rank) {
            *r = rng.next_u64();
        }
        if batch == s_words - 1 {
            let rem = num_shots % 64;
            if rem != 0 {
                let mask = (1u64 << rem) - 1;
                for r in random_bits.iter_mut().take(rank) {
                    *r &= mask;
                }
            }
        }

        for m in 0..num_meas {
            let cols = sparse.row_cols(m);
            let acc = match cols.len() {
                0 => 0u64,
                1 => random_bits[cols[0] as usize],
                2 => random_bits[cols[0] as usize] ^ random_bits[cols[1] as usize],
                3 => {
                    random_bits[cols[0] as usize]
                        ^ random_bits[cols[1] as usize]
                        ^ random_bits[cols[2] as usize]
                }
                _ => {
                    let mut a =
                        random_bits[cols[0] as usize] ^ random_bits[cols[1] as usize];
                    for &c in &cols[2..] {
                        a ^= random_bits[c as usize];
                    }
                    a
                }
            };
            meas_major[m * s_words + batch] = acc;
        }
    }

    apply_ref_bits_meas_major(&mut meas_major, ref_bits, num_meas, s_words);
    meas_major
}

fn gray_code_exact_counts(
    sparse: &SparseParity,
    rank: usize,
    ref_bits: &[u64],
    num_measurements: usize,
) -> std::collections::HashMap<Vec<u64>, u64> {
    use std::collections::HashMap;

    let m_words = num_measurements.div_ceil(64);
    let mut meas_vec = ref_bits[..m_words].to_vec();
    let total: u64 = 1u64 << rank;

    let mut col_words: Vec<Vec<u64>> = Vec::with_capacity(rank);
    for col in 0..rank {
        let mut cw = vec![0u64; m_words];
        for m in 0..num_measurements {
            let start = sparse.row_offsets[m] as usize;
            let end = sparse.row_offsets[m + 1] as usize;
            for &c in &sparse.col_indices[start..end] {
                if c as usize == col {
                    cw[m / 64] |= 1u64 << (m % 64);
                }
            }
        }
        col_words.push(cw);
    }

    let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
    *counts.entry(meas_vec.clone()).or_insert(0) += 1;

    for step in 1..total {
        let bit_to_flip = step.trailing_zeros() as usize;
        let col = &col_words[bit_to_flip];
        for (mw, cw) in meas_vec.iter_mut().zip(col.iter()) {
            *mw ^= cw;
        }
        *counts.entry(meas_vec.clone()).or_insert(0) += 1;
    }

    counts
}

fn apply_ref_bits_meas_major(
    meas_major: &mut [u64],
    ref_bits: &[u64],
    num_meas: usize,
    s_words: usize,
) {
    for m in 0..num_meas {
        let ref_bit = (ref_bits[m / 64] >> (m % 64)) & 1;
        if ref_bit != 0 {
            let row = &mut meas_major[m * s_words..(m + 1) * s_words];
            for w in row.iter_mut() {
                *w ^= !0u64;
            }
        }
    }
}

fn sample_bts_meas_major_dag(
    sparse: &SparseParity,
    dag: &XorDag,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut ChaCha8Rng,
    rank: usize,
) -> Vec<u64> {
    let num_meas = sparse.num_rows;
    let s_words = num_shots.div_ceil(64);

    let mut meas_major = vec![0u64; num_meas * s_words];
    let mut random_bits = vec![0u64; rank];

    for batch in 0..s_words {
        for r in random_bits.iter_mut().take(rank) {
            *r = rng.next_u64();
        }
        if batch == s_words - 1 {
            let rem = num_shots % 64;
            if rem != 0 {
                let mask = (1u64 << rem) - 1;
                for r in random_bits.iter_mut().take(rank) {
                    *r &= mask;
                }
            }
        }

        for (m, entry) in dag.entries.iter().enumerate() {
            let mut acc = if let Some(p) = entry.parent {
                meas_major[p * s_words + batch]
            } else {
                0u64
            };
            for &c in &entry.residual_cols {
                acc ^= random_bits[c as usize];
            }
            meas_major[m * s_words + batch] = acc;
        }
    }

    apply_ref_bits_meas_major(&mut meas_major, ref_bits, num_meas, s_words);
    meas_major
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sample_bts_meas_major_avx2(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut ChaCha8Rng,
    rank: usize,
) -> Vec<u64> {
    use std::arch::x86_64::*;

    let num_meas = sparse.num_rows;
    let s_words = num_shots.div_ceil(64);
    let s_quads = num_shots.div_ceil(256);

    let mut meas_major = vec![0u64; num_meas * s_words];
    let mut random_avx: Vec<__m256i> = vec![_mm256_setzero_si256(); rank];
    let mut rng_buf = vec![0u64; rank * 4];

    for quad in 0..s_quads {
        let base_sw = quad * 4;
        let words_this_quad = (s_words - base_sw).min(4);

        for j in 0..rank {
            for w in 0..4 {
                rng_buf[j * 4 + w] = rng.next_u64();
            }
        }

        let last_quad = quad == s_quads - 1;
        if last_quad {
            let rem = num_shots % 256;
            if rem != 0 {
                let full_words = rem / 64;
                let tail_bits = rem % 64;
                for j in 0..rank {
                    for w in (full_words + usize::from(tail_bits > 0))..4 {
                        rng_buf[j * 4 + w] = 0;
                    }
                    if tail_bits > 0 {
                        rng_buf[j * 4 + full_words] &= (1u64 << tail_bits) - 1;
                    }
                }
            }
        }

        for (j, avx) in random_avx.iter_mut().enumerate().take(rank) {
            *avx = _mm256_loadu_si256(rng_buf[j * 4..].as_ptr() as *const __m256i);
        }

        for m in 0..num_meas {
            let cols = sparse.row_cols(m);
            let acc = match cols.len() {
                0 => _mm256_setzero_si256(),
                1 => random_avx[cols[0] as usize],
                2 => _mm256_xor_si256(
                    random_avx[cols[0] as usize],
                    random_avx[cols[1] as usize],
                ),
                3 => _mm256_xor_si256(
                    _mm256_xor_si256(
                        random_avx[cols[0] as usize],
                        random_avx[cols[1] as usize],
                    ),
                    random_avx[cols[2] as usize],
                ),
                4 => _mm256_xor_si256(
                    _mm256_xor_si256(
                        random_avx[cols[0] as usize],
                        random_avx[cols[1] as usize],
                    ),
                    _mm256_xor_si256(
                        random_avx[cols[2] as usize],
                        random_avx[cols[3] as usize],
                    ),
                ),
                _ => {
                    let mut a = _mm256_xor_si256(
                        random_avx[cols[0] as usize],
                        random_avx[cols[1] as usize],
                    );
                    for &c in &cols[2..] {
                        a = _mm256_xor_si256(a, random_avx[c as usize]);
                    }
                    a
                }
            };

            let out_ptr = meas_major[m * s_words + base_sw..].as_mut_ptr();
            if words_this_quad == 4 {
                _mm256_storeu_si256(out_ptr as *mut __m256i, acc);
            } else {
                let mut tmp = [0u64; 4];
                _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
                for (w, &val) in tmp.iter().enumerate().take(words_this_quad) {
                    *out_ptr.add(w) = val;
                }
            }
        }
    }

    apply_ref_bits_meas_major(&mut meas_major, ref_bits, num_meas, s_words);
    meas_major
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
unsafe fn sample_bts_meas_major_avx2(
    _sparse: &SparseParity,
    _num_shots: usize,
    _ref_bits: &[u64],
    _rng: &mut ChaCha8Rng,
    _rank: usize,
) -> Vec<u64> {
    unreachable!()
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn sample_bts_meas_major_neon(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut ChaCha8Rng,
    rank: usize,
) -> Vec<u64> {
    use std::arch::aarch64::*;

    let num_meas = sparse.num_rows;
    let s_words = num_shots.div_ceil(64);
    let s_pairs = num_shots.div_ceil(128);

    let mut meas_major = vec![0u64; num_meas * s_words];
    let mut random_neon: Vec<uint64x2_t> = vec![vdupq_n_u64(0); rank];
    let mut rng_buf = vec![0u64; rank * 2];

    for pair in 0..s_pairs {
        let base_sw = pair * 2;
        let words_this_pair = (s_words - base_sw).min(2);

        for j in 0..rank {
            rng_buf[j * 2] = rng.next_u64();
            rng_buf[j * 2 + 1] = rng.next_u64();
        }

        let last_pair = pair == s_pairs - 1;
        if last_pair {
            let rem = num_shots % 128;
            if rem != 0 {
                let full_words = rem / 64;
                let tail_bits = rem % 64;
                for j in 0..rank {
                    if full_words + usize::from(tail_bits > 0) < 2 {
                        rng_buf[j * 2 + 1] = 0;
                    }
                    if tail_bits > 0 {
                        rng_buf[j * 2 + full_words] &= (1u64 << tail_bits) - 1;
                    }
                }
            }
        }

        for (j, nval) in random_neon.iter_mut().enumerate().take(rank) {
            *nval = vld1q_u64(rng_buf[j * 2..].as_ptr());
        }

        for m in 0..num_meas {
            let cols = sparse.row_cols(m);
            let acc = match cols.len() {
                0 => vdupq_n_u64(0),
                1 => random_neon[cols[0] as usize],
                2 => veorq_u64(
                    random_neon[cols[0] as usize],
                    random_neon[cols[1] as usize],
                ),
                3 => veorq_u64(
                    veorq_u64(
                        random_neon[cols[0] as usize],
                        random_neon[cols[1] as usize],
                    ),
                    random_neon[cols[2] as usize],
                ),
                _ => {
                    let mut a = veorq_u64(
                        random_neon[cols[0] as usize],
                        random_neon[cols[1] as usize],
                    );
                    for &c in &cols[2..] {
                        a = veorq_u64(a, random_neon[c as usize]);
                    }
                    a
                }
            };

            let out_ptr = meas_major[m * s_words + base_sw..].as_mut_ptr();
            if words_this_pair == 2 {
                vst1q_u64(out_ptr, acc);
            } else {
                *out_ptr = vgetq_lane_u64(acc, 0);
            }
        }
    }

    apply_ref_bits_meas_major(&mut meas_major, ref_bits, num_meas, s_words);
    meas_major
}

#[cfg(not(target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn sample_bts_meas_major_neon(
    _sparse: &SparseParity,
    _num_shots: usize,
    _ref_bits: &[u64],
    _rng: &mut ChaCha8Rng,
    _rank: usize,
) -> Vec<u64> {
    unreachable!()
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

fn row_weight(row: &[u64]) -> u32 {
    row.iter().map(|w| w.count_ones()).sum()
}

const MAX_MEASUREMENTS_FOR_DAG: usize = 2000;
const MIN_DAG_REDUCTION_PCT: usize = 20;
const MIN_MEAN_WEIGHT_FOR_DAG: usize = 3;

fn build_xor_dag_if_useful(sparse: &SparseParity) -> Option<XorDag> {
    if sparse.num_rows <= 1 || sparse.num_rows > MAX_MEASUREMENTS_FOR_DAG {
        return None;
    }
    let stats = sparse.stats();
    if stats.mean_weight < MIN_MEAN_WEIGHT_FOR_DAG as f64 {
        return None;
    }
    let dag = sparse.build_xor_dag();
    if dag.original_weight == 0 {
        return None;
    }
    let saved = dag.original_weight - dag.dag_weight;
    let reduction_pct = 100 * saved / dag.original_weight;
    if reduction_pct >= MIN_DAG_REDUCTION_PCT {
        Some(dag)
    } else {
        None
    }
}

const MAX_RANK_FOR_GRAY_CODE: usize = 25;
const MAX_LUT_ALLOC_BYTES: u64 = 256 * 1024 * 1024;
const BTS_BATCH_SHOTS: usize = 65536;
const MIN_BLOCKS_FOR_PARALLEL: usize = 2;
const MIN_BLOCK_MEASUREMENTS: usize = 2;

fn build_parity_blocks_if_useful(
    sparse: &SparseParity,
    rank: usize,
    ref_bits: &[u64],
) -> Option<ParityBlocks> {
    let block_meas = sparse.find_blocks(rank)?;

    if block_meas.len() < MIN_BLOCKS_FOR_PARALLEL {
        return None;
    }
    if block_meas
        .iter()
        .any(|b| b.len() < MIN_BLOCK_MEASUREMENTS)
    {
        return None;
    }

    Some(ParityBlocks::build(sparse, block_meas, rank, ref_bits))
}

const MAX_RANK_FOR_WEIGHT_MIN: usize = 500;

fn minimize_flip_row_weight(flip_rows: &mut [Vec<u64>]) -> (usize, usize) {
    let rank = flip_rows.len();
    let total: usize = flip_rows.iter().map(|r| row_weight(r) as usize).sum();
    if rank <= 1 || rank > MAX_RANK_FOR_WEIGHT_MIN {
        return (total, total);
    }

    let before = total;
    let mut weights: Vec<u32> = flip_rows.iter().map(|r| row_weight(r)).collect();

    let mut improved = true;
    while improved {
        improved = false;
        for i in 0..rank {
            let wi = weights[i];
            if wi == 0 {
                continue;
            }
            let mut best_j = usize::MAX;
            let mut best_w = wi;
            for j in 0..rank {
                if j == i {
                    continue;
                }
                let xor_w: u32 = flip_rows[i]
                    .iter()
                    .zip(flip_rows[j].iter())
                    .map(|(&a, &b)| (a ^ b).count_ones())
                    .sum();
                if xor_w < best_w {
                    best_w = xor_w;
                    best_j = j;
                }
            }
            if best_j != usize::MAX {
                for w in 0..flip_rows[i].len() {
                    flip_rows[i][w] ^= flip_rows[best_j][w];
                }
                weights[i] = best_w;
                improved = true;
            }
        }
    }

    let after: usize = weights.iter().map(|&w| w as usize).sum();
    (before, after)
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
            sparse: None,
            xor_dag: None,
            parity_blocks: None,
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
    minimize_flip_row_weight(&mut flip_rows);

    let lut = if rank >= LUT_MIN_RANK {
        Some(FlipLut::build(&flip_rows, num_meas_words))
    } else {
        None
    };

    let sparse = SparseParity::from_flip_rows(&flip_rows, num_measurements);
    let xor_dag = build_xor_dag_if_useful(&sparse);
    let ref_bits_packed = pack_bools(&ref_bits);
    let parity_blocks = build_parity_blocks_if_useful(&sparse, rank, &ref_bits_packed);

    Ok(CompiledSampler {
        flip_rows,
        ref_bits_packed,
        rank,
        num_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed),
        lut,
        sparse: Some(sparse),
        xor_dag,
        parity_blocks,
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
            sparse: None,
            xor_dag: None,
            parity_blocks: None,
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

    minimize_flip_row_weight(&mut flip_rows);

    let lut = if total_rank >= LUT_MIN_RANK {
        Some(FlipLut::build(&flip_rows, m_words))
    } else {
        None
    };

    let sparse = SparseParity::from_flip_rows(&flip_rows, num_global_measurements);
    let xor_dag = build_xor_dag_if_useful(&sparse);
    let parity_blocks = build_parity_blocks_if_useful(&sparse, total_rank, &ref_bits_packed);

    Ok(CompiledSampler {
        flip_rows,
        ref_bits_packed,
        rank: total_rank,
        num_measurements: num_global_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed),
        lut,
        sparse: Some(sparse),
        xor_dag,
        parity_blocks,
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
            sparse: None,
            xor_dag: None,
            parity_blocks: None,
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

    minimize_flip_row_weight(&mut flip_rows);

    let lut = if rank >= LUT_MIN_RANK {
        Some(FlipLut::build(&flip_rows, num_meas_words))
    } else {
        None
    };

    let sparse = SparseParity::from_flip_rows(&flip_rows, num_measurements);
    let xor_dag = build_xor_dag_if_useful(&sparse);
    let ref_bits_packed = pack_bools(&ref_bits);
    let parity_blocks = build_parity_blocks_if_useful(&sparse, rank, &ref_bits_packed);

    Ok(CompiledSampler {
        flip_rows,
        ref_bits_packed,
        rank,
        num_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed),
        lut,
        sparse: Some(sparse),
        xor_dag,
        parity_blocks,
    })
}

pub fn run_shots_compiled(circuit: &Circuit, num_shots: usize, seed: u64) -> Result<ShotsResult> {
    let mut sampler = compile_measurements(circuit, seed)?;
    let packed = sampler.sample_bulk_packed(num_shots);
    Ok(ShotsResult {
        shots: packed.to_shots(),
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

    #[test]
    fn packed_shots_roundtrip_ghz() {
        let mut c = circuits::ghz_circuit(10);
        c.num_classical_bits = 10;
        for i in 0..10 {
            c.add_measure(i, i);
        }
        let mut sampler = compile_forward(&c, 42).unwrap();
        let packed = sampler.sample_bulk_packed(1000);
        assert_eq!(packed.num_shots(), 1000);
        assert_eq!(packed.num_measurements(), 10);

        let unpacked = packed.to_shots();
        assert_eq!(unpacked.len(), 1000);
        for shot in &unpacked {
            assert_eq!(shot.len(), 10);
            let first = shot[0];
            assert!(shot.iter().all(|&b| b == first));
        }
    }

    #[test]
    fn packed_shots_matches_sample_bulk() {
        let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
        c.num_classical_bits = 20;
        for i in 0..20 {
            c.add_measure(i, i);
        }

        let mut sampler1 = compile_forward(&c, 42).unwrap();
        let mut sampler2 = compile_forward(&c, 42).unwrap();

        let num_shots = 5000;
        let bulk = sampler1.sample_bulk(num_shots);
        let packed = sampler2.sample_bulk_packed(num_shots);
        let unpacked = packed.to_shots();

        assert_eq!(bulk.len(), unpacked.len());
        assert_eq!(bulk[0].len(), unpacked[0].len());

        let n = bulk[0].len();
        for q in 0..n {
            let freq1: usize = bulk.iter().filter(|s| s[q]).count();
            let freq2: usize = unpacked.iter().filter(|s| s[q]).count();
            let p1 = freq1 as f64 / num_shots as f64;
            let p2 = freq2 as f64 / num_shots as f64;
            assert!(
                (p1 - p2).abs() < 0.05,
                "qubit {q}: bulk={p1:.3}, packed={p2:.3}"
            );
        }
    }

    #[test]
    fn packed_shots_get_bit() {
        let mut c = circuits::ghz_circuit(4);
        c.num_classical_bits = 4;
        for i in 0..4 {
            c.add_measure(i, i);
        }
        let mut sampler = compile_forward(&c, 42).unwrap();
        let packed = sampler.sample_bulk_packed(100);

        for s in 0..100 {
            let first = packed.get_bit(s, 0);
            for m in 1..4 {
                assert_eq!(packed.get_bit(s, m), first);
            }
        }
    }

    #[test]
    fn packed_shots_counts() {
        let mut c = circuits::ghz_circuit(4);
        c.num_classical_bits = 4;
        for i in 0..4 {
            c.add_measure(i, i);
        }
        let mut sampler = compile_forward(&c, 42).unwrap();
        let packed = sampler.sample_bulk_packed(10_000);
        let counts = packed.counts();

        assert_eq!(counts.len(), 2);
        let total: u64 = counts.values().sum();
        assert_eq!(total, 10_000);
    }

    #[test]
    fn sparse_parity_ghz() {
        let mut c = circuits::ghz_circuit(4);
        c.num_classical_bits = 4;
        for i in 0..4 {
            c.add_measure(i, i);
        }
        let sampler = compile_forward(&c, 42).unwrap();

        let sp = sampler.sparse().expect("sparse should be Some");
        assert_eq!(sp.num_rows, 4);

        let stats = sp.stats();
        assert!(stats.total_weight > 0);
        assert!(stats.min_weight <= stats.max_weight);

        for m in 0..4 {
            let cols = sp.row_cols(m);
            assert_eq!(cols.len(), sp.row_weight(m));
        }
    }

    #[test]
    fn sparse_parity_matches_flip_rows() {
        let mut c = circuits::ghz_circuit(8);
        c.num_classical_bits = 8;
        for i in 0..8 {
            c.add_measure(i, i);
        }
        let sampler = compile_forward(&c, 42).unwrap();

        let sp = sampler.sparse().unwrap();
        let stats = sampler.parity_stats().unwrap();
        assert_eq!(stats.min_weight, sp.stats().min_weight);
        assert_eq!(stats.max_weight, sp.stats().max_weight);
    }

    #[test]
    fn sparse_parity_empty_circuit() {
        let c = Circuit::new(2, 0);
        let sampler = compile_forward(&c, 42).unwrap();
        assert!(sampler.sparse().is_none());
    }

    #[test]
    fn bts_meas_major_ghz_counts() {
        let mut c = circuits::ghz_circuit(8);
        c.num_classical_bits = 8;
        for i in 0..8 {
            c.add_measure(i, i);
        }
        let mut sampler = compile_forward(&c, 42).unwrap();
        let packed = sampler.sample_bulk_packed(10_000);

        assert_eq!(packed.layout(), ShotLayout::MeasMajor);

        let counts = packed.counts();
        assert_eq!(counts.len(), 2, "GHZ should have exactly 2 outcomes");
        let total: u64 = counts.values().sum();
        assert_eq!(total, 10_000);
    }

    #[test]
    fn bts_meas_major_get_bit_consistency() {
        let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
        c.num_classical_bits = 20;
        for i in 0..20 {
            c.add_measure(i, i);
        }
        let mut sampler = compile_forward(&c, 42).unwrap();
        let packed = sampler.sample_bulk_packed(500);

        let shots = packed.to_shots();
        for (s, shot) in shots.iter().enumerate() {
            for (m, &val) in shot.iter().enumerate().take(20) {
                assert_eq!(packed.get_bit(s, m), val, "Mismatch at shot={s} meas={m}");
            }
        }
    }

    #[test]
    fn bts_meas_major_marginals_match_stabilizer() {
        let mut c = circuits::clifford_heavy_circuit(50, 5, 42);
        c.num_classical_bits = 50;
        for i in 0..50 {
            c.add_measure(i, i);
        }
        let mut sampler = compile_forward(&c, 42).unwrap();
        let packed = sampler.sample_bulk_packed(5_000);

        let reference = crate::sim::run_shots_with(BackendKind::Stabilizer, &c, 5_000, 42).unwrap();

        for q in 0..50 {
            let bts_ones: usize = (0..5_000).filter(|&s| packed.get_bit(s, q)).count();
            let ref_ones: usize = reference.shots.iter().filter(|s| s[q]).count();
            let bts_frac = bts_ones as f64 / 5_000.0;
            let ref_frac = ref_ones as f64 / 5_000.0;
            assert!(
                (bts_frac - ref_frac).abs() < 0.05,
                "q{q}: bts={bts_frac:.3} ref={ref_frac:.3}"
            );
        }
    }

    #[test]
    fn streaming_counts_ghz() {
        let mut c = circuits::ghz_circuit(10);
        c.num_classical_bits = 10;
        for i in 0..10 {
            c.add_measure(i, i);
        }
        let mut sampler = compile_forward(&c, 42).unwrap();
        let counts = sampler.sample_counts_streaming(10_000, 1_000);

        assert_eq!(counts.len(), 2, "GHZ should produce exactly 2 outcomes");
        let total: u64 = counts.values().sum();
        assert_eq!(total, 10_000);
    }

    #[test]
    fn marginal_probabilities_ghz() {
        let mut c = circuits::ghz_circuit(8);
        c.num_classical_bits = 8;
        for i in 0..8 {
            c.add_measure(i, i);
        }
        let sampler = compile_forward(&c, 42).unwrap();
        let probs = sampler.marginal_probabilities();
        assert_eq!(probs.len(), 8);
        for &p in &probs {
            assert!((p - 0.5).abs() < 1e-10, "GHZ marginals should be 0.5");
        }
    }

    #[test]
    fn marginal_probabilities_x_all_ones() {
        let mut c = Circuit::new(4, 4);
        for i in 0..4 {
            c.add_gate(Gate::X, &[i]);
            c.add_measure(i, i);
        }
        let sampler = compile_forward(&c, 42).unwrap();
        let probs = sampler.marginal_probabilities();
        for &p in &probs {
            assert!(
                (p - 1.0).abs() < 1e-10,
                "X then measure should be deterministic 1"
            );
        }
    }

    #[test]
    fn parity_report_not_empty() {
        let mut c = circuits::ghz_circuit(8);
        c.num_classical_bits = 8;
        for i in 0..8 {
            c.add_measure(i, i);
        }
        let sampler = compile_forward(&c, 42).unwrap();
        let report = sampler.parity_report();
        assert!(report.contains("measurements"));
        assert!(report.contains("rank"));
        assert!(report.contains("Weight"));
    }

    #[test]
    fn weight_minimization_reduces_weight() {
        let mut rows: Vec<Vec<u64>> = vec![vec![0b1111], vec![0b1110], vec![0b1100]];
        let (before, after) = minimize_flip_row_weight(&mut rows);
        assert!(
            after <= before,
            "weight should not increase: {} -> {}",
            before,
            after
        );
        assert!(
            after < before,
            "weight should decrease for reducible rows: {} -> {}",
            before,
            after
        );
        assert_eq!(before, 4 + 3 + 2);
        assert_eq!(after, 1 + 1 + 2);
    }

    #[test]
    fn weight_minimization_preserves_sampling() {
        let mut c = circuits::clifford_random_pairs(16, 20, 42);
        c.num_classical_bits = 16;
        for i in 0..16 {
            c.add_measure(i, i);
        }
        let mut sampler = compile_forward(&c, 42).unwrap();
        let counts = sampler.sample_bulk_packed(50_000).counts();
        let total: u64 = counts.values().sum();
        assert_eq!(total, 50_000);
        assert!(counts.len() > 1);
    }

    #[test]
    fn xor_dag_reduces_weight() {
        let sp = SparseParity {
            col_indices: vec![0, 1, 0, 1, 2, 1, 2],
            row_offsets: vec![0, 2, 5, 7],
            num_rows: 3,
        };
        let dag = sp.build_xor_dag();
        assert!(
            dag.dag_weight < dag.original_weight,
            "DAG weight {} should be less than original {}",
            dag.dag_weight,
            dag.original_weight
        );
        assert_eq!(dag.original_weight, 2 + 3 + 2);
        assert!(dag.entries[1].parent.is_some() || dag.entries[2].parent.is_some());
    }

    #[test]
    fn xor_dag_bts_correctness() {
        let mut c = circuits::clifford_random_pairs(16, 20, 42);
        c.num_classical_bits = 16;
        for i in 0..16 {
            c.add_measure(i, i);
        }
        let mut sampler = compile_forward(&c, 42).unwrap();
        let packed = sampler.sample_bulk_packed(10_000);
        let counts = packed.counts();
        let total: u64 = counts.values().sum();
        assert_eq!(total, 10_000);
        assert!(counts.len() > 1);
    }

    #[test]
    fn block_detection_independent_pairs() {
        let mut c = circuits::independent_bell_pairs(4);
        c.num_classical_bits = 8;
        for i in 0..8 {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        let sparse = sampler.sparse.as_ref().unwrap();
        let blocks = sparse.find_blocks(sampler.rank);
        assert!(blocks.is_some());
        let blocks = blocks.unwrap();
        assert_eq!(blocks.len(), 4);
        for b in &blocks {
            assert_eq!(b.len(), 2);
        }
    }

    #[test]
    fn block_detection_single_block() {
        let mut c = circuits::clifford_random_pairs(8, 20, 42);
        c.num_classical_bits = 8;
        for i in 0..8 {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert!(sampler.parity_blocks.is_none());
    }

    #[test]
    fn block_parallel_bts_correctness() {
        let mut c = circuits::independent_bell_pairs(8);
        c.num_classical_bits = 16;
        for i in 0..16 {
            c.add_measure(i, i);
        }

        let mut sampler_block = compile_measurements(&c, 42).unwrap();
        assert!(sampler_block.parity_blocks.is_some());

        let packed = sampler_block.sample_bulk_packed(10_000);
        let counts = packed.counts();
        let total: u64 = counts.values().sum();
        assert_eq!(total, 10_000);

        let shots = packed.to_shots();
        for shot in &shots {
            for pair in 0..8 {
                let b0 = shot[2 * pair];
                let b1 = shot[2 * pair + 1];
                assert_eq!(b0, b1, "Bell pair {pair} must be correlated");
            }
        }
    }

    #[test]
    fn gray_code_exact_counts_bell_pair() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);
        let sampler = compile_measurements(&c, 42).unwrap();
        let counts = sampler.exact_counts().unwrap();
        let total: u64 = counts.values().sum();
        assert!(total.is_power_of_two());
        assert_eq!(counts.len(), 2);
        let half = total / 2;
        for &v in counts.values() {
            assert_eq!(v, half);
        }
    }

    #[test]
    fn gray_code_exact_counts_ghz() {
        let mut c = Circuit::new(4, 4);
        c.add_gate(Gate::H, &[0]);
        for i in 0..3 {
            c.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for i in 0..4 {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        let counts = sampler.exact_counts().unwrap();
        assert_eq!(counts.len(), 2);
        let total: u64 = counts.values().sum();
        let half = total / 2;
        for &v in counts.values() {
            assert_eq!(v, half);
        }
    }

    #[test]
    fn gray_code_matches_sampling() {
        let mut c = circuits::clifford_random_pairs(8, 10, 42);
        c.num_classical_bits = 8;
        for i in 0..8 {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        let exact = sampler.exact_counts().unwrap();
        let total: u64 = exact.values().sum();
        let exact_probs: std::collections::HashMap<Vec<u64>, f64> = exact
            .iter()
            .map(|(k, &v)| (k.clone(), v as f64 / total as f64))
            .collect();

        let mut sampler2 = compile_measurements(&c, 123).unwrap();
        let packed = sampler2.sample_bulk_packed(100_000);
        let sample_counts = packed.counts();
        for (outcome, &exact_p) in &exact_probs {
            if exact_p > 0.01 {
                let sampled = *sample_counts.get(outcome).unwrap_or(&0) as f64 / 100_000.0;
                let diff = (sampled - exact_p).abs();
                assert!(
                    diff < 0.02,
                    "outcome {outcome:?}: exact={exact_p:.4}, sampled={sampled:.4}"
                );
            }
        }
    }

    #[test]
    fn bts_batched_correctness() {
        let mut c = Circuit::new(4, 4);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Cx, &[2, 3]);
        for i in 0..4 {
            c.add_measure(i, i);
        }

        let num_shots = BTS_BATCH_SHOTS * 3 + 100;
        let mut sampler = compile_measurements(&c, 42).unwrap();
        assert!(sampler.should_use_bts(num_shots));

        let packed = sampler.sample_bulk_packed(num_shots);
        assert_eq!(packed.num_shots, num_shots);
        let counts = packed.counts();
        let total: u64 = counts.values().sum();
        assert_eq!(total, num_shots as u64);

        let shots = packed.to_shots();
        for shot in &shots {
            assert_eq!(shot[0], shot[1], "Bell pair 0 must be correlated");
            assert_eq!(shot[2], shot[3], "Bell pair 1 must be correlated");
        }
    }

    #[test]
    fn memory_aware_bts_dispatch() {
        let mut c = circuits::clifford_random_pairs(100, 20, 42);
        c.num_classical_bits = 100;
        for i in 0..100 {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert!(sampler.should_use_bts(100_000_000));
    }

    #[test]
    fn detection_event_parity_matrix() {
        let mut c = Circuit::new(2, 4);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);
        c.add_measure(0, 2);
        c.add_measure(1, 3);

        let sampler = compile_measurements(&c, 42).unwrap();
        let sparse = sampler.sparse.as_ref().unwrap();

        let det = sparse.compile_detection_events(&[(2, 0), (3, 1)]);
        assert_eq!(det.num_rows, 2);
        for m in 0..2 {
            assert_eq!(det.row_weight(m), 0, "same-stabilizer detection event must be deterministic");
        }
    }

    #[test]
    fn detection_event_sampling() {
        let mut c = Circuit::new(4, 8);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Cx, &[2, 3]);
        for i in 0..4 {
            c.add_measure(i, i);
        }
        for i in 0..4 {
            c.add_measure(i, i + 4);
        }

        let mut sampler = compile_measurements(&c, 42).unwrap();
        let packed = sampler.sample_detection_events(
            &[(4, 0), (5, 1), (6, 2), (7, 3)],
            10_000,
        );
        assert_eq!(packed.num_shots, 10_000);
        assert_eq!(packed.num_measurements, 4);

        let shots = packed.to_shots();
        for shot in &shots {
            for (i, &val) in shot.iter().enumerate().take(4) {
                assert!(!val, "detection event {i} must be 0");
            }
        }
    }
}

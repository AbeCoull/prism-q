//! Pauli rotation kernels: pair rotations, parity phases and tiled multi-rotation passes.

use super::*;

/// Shortest run of contiguous pairs handed to the SIMD pair kernel. Below it the
/// AVX2 loop runs at most one 256-bit iteration per call, so the per-run dispatch
/// outweighs it and the scalar loop takes the block.
const MIN_PAULI_ROT_SIMD_RUN: usize = 4;

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_popcnt() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| is_x86_feature_detected!("popcnt"))
}

/// The pair arithmetic of `exp(-i θ P / 2)` for [`pauli_rot_pair_halves`].
///
/// `m_lo` and `m_hi` are the parity-even cross coefficients toward the pivot-clear
/// and pivot-set side; `even` and `odd` hold the pair matrix
/// `[[c, ±m_lo], [±m_hi, c]]` for each parity with its SIMD tier chosen once.
struct PauliRotPair {
    c: f64,
    m_lo: Complex64,
    m_hi: Complex64,
    #[cfg(target_arch = "x86_64")]
    imag: bool,
    even: simd::PreparedGate1q,
    odd: simd::PreparedGate1q,
}

impl PauliRotPair {
    /// The pair arithmetic for a rotation by `theta` whose string has `num_y` Y letters.
    ///
    /// `m_hi` points toward the pivot-set side of a pair; the opposite direction
    /// differs by the parity of `xmask & zmask`, which is the Y count.
    fn new(theta: f64, num_y: u32) -> Self {
        let (s, c) = (theta / 2.0).sin_cos();
        let m_hi = match num_y % 4 {
            0 => Complex64::new(0.0, -s),
            1 => Complex64::new(s, 0.0),
            2 => Complex64::new(0.0, s),
            _ => Complex64::new(-s, 0.0),
        };
        let imag = num_y.is_multiple_of(2);
        let m_lo = if imag { m_hi } else { -m_hi };
        let diag = Complex64::new(c, 0.0);
        Self {
            c,
            m_lo,
            m_hi,
            #[cfg(target_arch = "x86_64")]
            imag,
            even: simd::PreparedGate1q::new(&[[diag, m_lo], [m_hi, diag]]),
            odd: simd::PreparedGate1q::new(&[[diag, -m_lo], [-m_hi, diag]]),
        }
    }
}

/// Mix the pairs `(lo[i], hi[i ^ xlow])` of one Pauli-rotation pivot block.
///
/// `base` is the global index of `lo[0]` and must be a multiple of `lo.len()`,
/// so `(base + i) & zmask` reads the pair's parity sign directly; `flip` (0 or 1)
/// is added to that parity. Pairs whose index bits below the lowest X or Z letter
/// vary form contiguous runs on both sides with one parity, and runs of at least
/// [`MIN_PAULI_ROT_SIMD_RUN`] go through the SIMD pair kernel.
#[inline(always)]
fn pauli_rot_pair_halves(
    lo: &mut [Complex64],
    hi: &mut [Complex64],
    base: usize,
    xlow: usize,
    zmask: usize,
    pair: &PauliRotPair,
    flip: u32,
) {
    let run = 1usize << (((xlow | zmask) & (lo.len() - 1)) | lo.len()).trailing_zeros();
    if run >= MIN_PAULI_ROT_SIMD_RUN {
        for start in (0..lo.len()).step_by(run) {
            let k = start ^ xlow;
            let prepared = if ((base + start) & zmask).count_ones() & 1 == flip {
                &pair.even
            } else {
                &pair.odd
            };
            prepared.apply(&mut lo[start..start + run], &mut hi[k..k + run]);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if lo.len() >= 2 && simd::has_avx2_fma() && has_popcnt() {
        let x0 = (xlow & 1) != 0;
        // SAFETY: AVX2, FMA, and POPCNT detected, and `lo.len() >= 2` is a power of two
        // matching `hi.len()` with `xlow < lo.len()`.
        unsafe {
            match (x0, pair.imag) {
                (false, false) => pauli_rot_pair_halves_avx2fma::<false, false>(
                    lo, hi, base, xlow, zmask, pair, flip,
                ),
                (false, true) => pauli_rot_pair_halves_avx2fma::<false, true>(
                    lo, hi, base, xlow, zmask, pair, flip,
                ),
                (true, false) => pauli_rot_pair_halves_avx2fma::<true, false>(
                    lo, hi, base, xlow, zmask, pair, flip,
                ),
                (true, true) => pauli_rot_pair_halves_avx2fma::<true, true>(
                    lo, hi, base, xlow, zmask, pair, flip,
                ),
            }
        }
        return;
    }

    let (c, m_lo, m_hi) = (pair.c, pair.m_lo, pair.m_hi);
    for (i, amp_lo) in lo.iter_mut().enumerate() {
        let k = i ^ xlow;
        let a = *amp_lo;
        let b = hi[k];
        if ((base + i) & zmask).count_ones() & 1 == flip {
            *amp_lo = a * c + b * m_lo;
            hi[k] = b * c + a * m_hi;
        } else {
            *amp_lo = a * c - b * m_lo;
            hi[k] = b * c - a * m_hi;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,popcnt")]
unsafe fn pauli_rot_pair_halves_avx2fma<const X0: bool, const IMAG: bool>(
    lo: &mut [Complex64],
    hi: &mut [Complex64],
    base: usize,
    xlow: usize,
    zmask: usize,
    pair: &PauliRotPair,
    flip: u32,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(lo.len() >= 2 && lo.len().is_power_of_two() && lo.len() == hi.len());
        let len = lo.len();
        let lo_ptr = lo.as_mut_ptr() as *mut f64;
        let hi_ptr = hi.as_mut_ptr() as *mut f64;
        let x_hi = xlow & !1;
        let z0 = (zmask & 1) != 0;

        let c_vec = _mm256_set1_pd(pair.c);
        let z0_flip = if z0 {
            _mm256_setr_pd(0.0, 0.0, -0.0, -0.0)
        } else {
            _mm256_setzero_pd()
        };
        let z0_flip_hi = if X0 && z0 {
            _mm256_setr_pd(-0.0, -0.0, 0.0, 0.0)
        } else {
            z0_flip
        };

        let i_mask = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
        let im_s_vec = _mm256_set1_pd(pair.m_hi.im);
        let m_lo_vec = _mm256_set1_pd(pair.m_lo.re);
        let m_hi_vec = _mm256_set1_pd(pair.m_hi.re);

        let mut i = 0;
        while i < len {
            let k_base = i ^ x_hi;
            let p_lo = lo_ptr.add(i * 2);
            let p_hi = hi_ptr.add(k_base * 2);

            let a = _mm256_loadu_pd(p_lo);
            let b_raw = _mm256_loadu_pd(p_hi);
            let (b, a_for_hi) = if X0 {
                (
                    _mm256_permute2f128_pd(b_raw, b_raw, 0x01),
                    _mm256_permute2f128_pd(a, a, 0x01),
                )
            } else {
                (b_raw, a)
            };

            let p0 = (((base + i) & zmask).count_ones() & 1) ^ flip;
            let s0_all = _mm256_castsi256_pd(_mm256_set1_epi64x((p0 as i64) << 63));
            let sign_lo = _mm256_xor_pd(s0_all, z0_flip);
            let sign_hi = _mm256_xor_pd(s0_all, z0_flip_hi);

            let (new_a, new_b) = if IMAG {
                let b_swap = _mm256_permute_pd(b, 0b0101);
                let a_swap = _mm256_permute_pd(a_for_hi, 0b0101);
                (
                    _mm256_fmadd_pd(
                        im_s_vec,
                        _mm256_xor_pd(b_swap, _mm256_xor_pd(sign_lo, i_mask)),
                        _mm256_mul_pd(c_vec, a),
                    ),
                    _mm256_fmadd_pd(
                        im_s_vec,
                        _mm256_xor_pd(a_swap, _mm256_xor_pd(sign_hi, i_mask)),
                        _mm256_mul_pd(c_vec, b_raw),
                    ),
                )
            } else {
                (
                    _mm256_fmadd_pd(m_lo_vec, _mm256_xor_pd(b, sign_lo), _mm256_mul_pd(c_vec, a)),
                    _mm256_fmadd_pd(
                        m_hi_vec,
                        _mm256_xor_pd(a_for_hi, sign_hi),
                        _mm256_mul_pd(c_vec, b_raw),
                    ),
                )
            };

            _mm256_storeu_pd(p_lo, new_a);
            _mm256_storeu_pd(p_hi, new_b);
            i += 2;
        }
    }
}

/// Apply `PauliRot` when `xmask == 1` (`pivot == 0`), where each pair `(a_k, b_k)`
/// occupies adjacent complex elements in `slice`.
#[inline(always)]
fn apply_pauli_rot_pivot0_slice(
    slice: &mut [Complex64],
    base: usize,
    zmask: usize,
    pair: &PauliRotPair,
    flip: u32,
) {
    debug_assert!(slice.len().is_multiple_of(2) && base.is_multiple_of(2));

    #[cfg(target_arch = "x86_64")]
    if slice.len() >= 2 && simd::has_avx2_fma() && has_popcnt() {
        // SAFETY: AVX2, FMA, and POPCNT detected, and `slice.len() >= 2` is even.
        unsafe {
            if pair.imag {
                apply_pauli_rot_pivot0_avx2fma::<true>(slice, base, zmask, pair, flip);
            } else {
                apply_pauli_rot_pivot0_avx2fma::<false>(slice, base, zmask, pair, flip);
            }
        }
        return;
    }

    let (c, m_lo, m_hi) = (pair.c, pair.m_lo, pair.m_hi);
    let (chunks, _) = slice.as_chunks_mut::<2>();
    for (j, [amp_a, amp_b]) in chunks.iter_mut().enumerate() {
        let a = *amp_a;
        let b = *amp_b;
        if ((base + j * 2) & zmask).count_ones() & 1 == flip {
            *amp_a = a * c + b * m_lo;
            *amp_b = b * c + a * m_hi;
        } else {
            *amp_a = a * c - b * m_lo;
            *amp_b = b * c - a * m_hi;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,popcnt")]
unsafe fn apply_pauli_rot_pivot0_avx2fma<const IMAG: bool>(
    slice: &mut [Complex64],
    base: usize,
    zmask: usize,
    pair: &PauliRotPair,
    flip: u32,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let len = slice.len();
        let ptr = slice.as_mut_ptr() as *mut f64;
        let c_vec = _mm256_set1_pd(pair.c);
        let i_mask = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
        let im_s_vec = _mm256_set1_pd(pair.m_hi.im);
        let m_vec = _mm256_setr_pd(pair.m_lo.re, pair.m_lo.re, pair.m_hi.re, pair.m_hi.re);

        let mut j = 0;
        while j < len {
            let p = ptr.add(j * 2);
            let v = _mm256_loadu_pd(p);
            let swap_ab = _mm256_permute2f128_pd(v, v, 0x01);
            let parity = (((base + j) & zmask).count_ones() & 1) ^ flip;
            let s_all = _mm256_castsi256_pd(_mm256_set1_epi64x((parity as i64) << 63));
            let new_v = if IMAG {
                let swap_rot = _mm256_permute_pd(swap_ab, 0b0101);
                _mm256_fmadd_pd(
                    im_s_vec,
                    _mm256_xor_pd(swap_rot, _mm256_xor_pd(s_all, i_mask)),
                    _mm256_mul_pd(c_vec, v),
                )
            } else {
                _mm256_fmadd_pd(
                    m_vec,
                    _mm256_xor_pd(swap_ab, s_all),
                    _mm256_mul_pd(c_vec, v),
                )
            };
            _mm256_storeu_pd(p, new_v);
            j += 2;
        }
    }
}

/// Multiply each amplitude in `slice` by `even` or `odd` according to `((base + j) & zmask)`.
///
/// Requires `even.re == odd.re` and `even.im == -odd.im`, `slice.len()` a power of two,
/// and `base` a multiple of `slice.len()`.
#[inline(always)]
fn apply_parity_phase_slice(
    slice: &mut [Complex64],
    base: usize,
    zmask: usize,
    even: Complex64,
    odd: Complex64,
) {
    debug_assert!(slice.len().is_power_of_two() && base.is_multiple_of(slice.len()));

    if slice.len() < 4 {
        let phases = [even, odd];
        for (j, amp) in slice.iter_mut().enumerate() {
            *amp *= phases[(((base + j) & zmask).count_ones() & 1) as usize];
        }
        return;
    }

    if zmask == 0 {
        simd::scale_complex_slice(slice, even);
        return;
    }

    let run = 1usize << zmask.trailing_zeros();
    if run >= slice.len() {
        let phase = if ((base & zmask).count_ones() & 1) == 0 {
            even
        } else {
            odd
        };
        simd::scale_complex_slice(slice, phase);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if has_popcnt() {
        match simd::simd_tier() {
            simd::SimdTier::Avx512 => {
                // SAFETY: AVX-512F and POPCNT detected, and `slice` has power-of-two
                // length >= 4 with `run < slice.len()`.
                unsafe {
                    apply_parity_phase_avx512(slice, base, zmask, run, even, odd);
                }
                return;
            }
            simd::SimdTier::Avx2Fma => {
                // SAFETY: AVX2, FMA, and POPCNT detected, and `slice` has power-of-two
                // length >= 4 with `run < slice.len()`.
                unsafe {
                    apply_parity_phase_avx2fma(slice, base, zmask, run, even, odd);
                }
                return;
            }
            simd::SimdTier::Fma | simd::SimdTier::Sse2 => {}
        }
    }

    let phases = [even, odd];
    for (j, amp) in slice.iter_mut().enumerate() {
        *amp *= phases[(((base + j) & zmask).count_ones() & 1) as usize];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,popcnt")]
unsafe fn apply_parity_phase_avx512(
    slice: &mut [Complex64],
    base: usize,
    zmask: usize,
    run: usize,
    even: Complex64,
    odd: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let len = slice.len();
        let ptr = slice.as_mut_ptr() as *mut f64;
        let c_rr = _mm512_set1_pd(even.re);

        if run >= 4 {
            let mut j = 0;
            while j < len {
                let p = ((base + j) & zmask).count_ones() & 1;
                let ii = _mm512_set1_pd(if p == 0 { even.im } else { odd.im });
                let mut k = j;
                while k < j + run {
                    let off = k * 2;
                    let v = _mm512_loadu_pd(ptr.add(off));
                    _mm512_storeu_pd(ptr.add(off), simd::complex_mul_avx512(c_rr, ii, v));
                    k += 4;
                }
                j += run;
            }
        } else {
            let im_for = |d: usize| if d == 0 { even.im } else { odd.im };
            let im0 = im_for(0);
            let im1 = im_for(zmask & 1);
            let im2 = im_for((zmask >> 1) & 1);
            let im3 = im_for((zmask & 1) ^ ((zmask >> 1) & 1));
            let ii_base = _mm512_setr_pd(im0, im0, im1, im1, im2, im2, im3, im3);
            let ii_base_i = _mm512_castpd_si512(ii_base);

            let mut j = 0;
            while j < len {
                let p0 = ((base + j) & zmask).count_ones() & 1;
                let flip_mask = _mm512_set1_epi64((p0 as i64) << 63);
                let ii = _mm512_castsi512_pd(_mm512_xor_si512(ii_base_i, flip_mask));
                let off = j * 2;
                let v = _mm512_loadu_pd(ptr.add(off));
                _mm512_storeu_pd(ptr.add(off), simd::complex_mul_avx512(c_rr, ii, v));
                j += 4;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,popcnt")]
unsafe fn apply_parity_phase_avx2fma(
    slice: &mut [Complex64],
    base: usize,
    zmask: usize,
    run: usize,
    even: Complex64,
    odd: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let len = slice.len();
        let ptr = slice.as_mut_ptr() as *mut f64;
        let c_rr = _mm256_set1_pd(even.re);

        if run >= 4 {
            let mut j = 0;
            while j < len {
                let p = ((base + j) & zmask).count_ones() & 1;
                let ii = _mm256_set1_pd(if p == 0 { even.im } else { odd.im });
                let mut k = j;
                while k < j + run {
                    let off = k * 2;
                    let v0 = _mm256_loadu_pd(ptr.add(off));
                    let v1 = _mm256_loadu_pd(ptr.add(off + 4));
                    _mm256_storeu_pd(ptr.add(off), simd::complex_mul_avx2fma(c_rr, ii, v0));
                    _mm256_storeu_pd(ptr.add(off + 4), simd::complex_mul_avx2fma(c_rr, ii, v1));
                    k += 4;
                }
                j += run;
            }
        } else {
            let ii_base = if (zmask & 1) != 0 {
                _mm256_setr_pd(even.im, even.im, odd.im, odd.im)
            } else {
                _mm256_set1_pd(even.im)
            };
            let z1_bit = (zmask >> 1) & 1;
            let mut j = 0;
            while j < len {
                let p0 = ((base + j) & zmask).count_ones() & 1;
                let p2 = p0 ^ (z1_bit as u32);
                let m0 = _mm256_castsi256_pd(_mm256_set1_epi64x((p0 as i64) << 63));
                let m2 = _mm256_castsi256_pd(_mm256_set1_epi64x((p2 as i64) << 63));
                let ii0 = _mm256_xor_pd(ii_base, m0);
                let ii1 = _mm256_xor_pd(ii_base, m2);
                let off = j * 2;
                let v0 = _mm256_loadu_pd(ptr.add(off));
                let v1 = _mm256_loadu_pd(ptr.add(off + 4));
                _mm256_storeu_pd(ptr.add(off), simd::complex_mul_avx2fma(c_rr, ii0, v0));
                _mm256_storeu_pd(ptr.add(off + 4), simd::complex_mul_avx2fma(c_rr, ii1, v1));
                j += 4;
            }
        }
    }
}

/// How one `MultiPauliRot` rotation acts inside a tile.
#[allow(clippy::large_enum_variant)]
enum TiledAction {
    /// A diagonal string: the phase for an even and for an odd Z parity.
    Parity([Complex64; 2]),
    Pairs(PauliRotPair),
}

/// One rotation of a `MultiPauliRot` batch in tile coordinates.
///
/// `xmask` and `zmask` cover tile bits. `zrest` keeps the Z letters on qubits
/// outside the tile: their parity is fixed across a tile and, when odd, negates
/// both cross terms or swaps the two phases.
pub(super) struct TiledPauliRot {
    xmask: usize,
    zmask: usize,
    zrest: usize,
    action: TiledAction,
}

impl TiledPauliRot {
    /// Move a rotation's global masks into the tile `tile_bit` describes: the tile
    /// bit a qubit lands on, or `None` for a qubit outside the tile, which must
    /// carry no X or Y letter.
    pub(super) fn new(
        xmask: usize,
        zmask: usize,
        theta: f64,
        tile_bit: impl Fn(usize) -> Option<usize>,
    ) -> Self {
        let (mut tile_x, mut tile_z, mut zrest) = (0usize, 0usize, 0usize);
        let mut letters = xmask | zmask;
        while letters != 0 {
            let q = letters.trailing_zeros() as usize;
            letters &= letters - 1;
            match tile_bit(q) {
                Some(bit) => {
                    tile_x |= (xmask >> q & 1) << bit;
                    tile_z |= (zmask >> q & 1) << bit;
                }
                None => {
                    debug_assert_eq!(xmask >> q & 1, 0, "X or Y letter outside the tile");
                    zrest |= 1 << q;
                }
            }
        }
        let action = if xmask == 0 {
            let (s, c) = (theta / 2.0).sin_cos();
            TiledAction::Parity([Complex64::new(c, -s), Complex64::new(c, s)])
        } else {
            TiledAction::Pairs(PauliRotPair::new(theta, (xmask & zmask).count_ones()))
        };
        Self {
            xmask: tile_x,
            zmask: tile_z,
            zrest,
            action,
        }
    }

    /// Apply to `tile`, whose first amplitude has global index `base` once the
    /// tile bits are cleared.
    #[inline(always)]
    pub(super) fn apply(&self, tile: &mut [Complex64], base: usize) {
        let flip = (base & self.zrest).count_ones() & 1;
        match &self.action {
            TiledAction::Parity([even, odd]) => {
                let (e, o) = if flip == 1 {
                    (*odd, *even)
                } else {
                    (*even, *odd)
                };
                apply_parity_phase_slice(tile, 0, self.zmask, e, o);
            }
            TiledAction::Pairs(pair) => {
                let pivot = usize::BITS as usize - 1 - self.xmask.leading_zeros() as usize;
                if pivot == 0 {
                    apply_pauli_rot_pivot0_slice(tile, 0, self.zmask, pair, flip);
                    return;
                }
                let half = 1usize << pivot;
                let block = half << 1;
                let xlow = self.xmask ^ half;
                for (k, chunk) in tile.chunks_mut(block).enumerate() {
                    let (lo, hi) = chunk.split_at_mut(half);
                    pauli_rot_pair_halves(lo, hi, k * block, xlow, self.zmask, pair, flip);
                }
            }
        }
    }
}

impl StatevectorBackend {
    /// Apply `exp(-i θ P / 2)` in one pass, `P` given as one letter per target.
    ///
    /// `P|j> = i^{num_y} (-1)^{popcount(j & zmask)} |j ^ xmask>`, so the gate
    /// mixes each pair `(j, j ^ xmask)` with `cos(θ/2)` on the diagonal and a
    /// sign-resolved `-i sin(θ/2) i^{num_y}` cross term. A Z-only string has
    /// `xmask == 0` and reduces to parity phases on the diagonal.
    #[inline(always)]
    pub(in crate::backend::statevector) fn apply_pauli_rot(
        &mut self,
        targets: &[usize],
        theta: f64,
        axes: &[PauliAxis],
    ) {
        let (xmask, zmask, num_y) = pauli_rot_masks(targets, axes);
        self.apply_pauli_rot_masks(xmask, zmask, num_y, theta);
    }

    #[inline(always)]
    fn apply_pauli_rot_masks(&mut self, xmask: usize, zmask: usize, num_y: u32, theta: f64) {
        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        if xmask == 0 {
            let (s, c) = (theta / 2.0).sin_cos();
            self.apply_parity_phase(zmask, Complex64::new(c, -s), Complex64::new(c, s));
            return;
        }

        let pair = PauliRotPair::new(theta, num_y);
        let pivot = usize::BITS as usize - 1 - xmask.leading_zeros() as usize;
        let xlow = xmask ^ (1usize << pivot);

        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            self.apply_pauli_rot_pairs_par(pivot, xlow, zmask, &pair);
            return;
        }

        if pivot == 0 {
            apply_pauli_rot_pivot0_slice(&mut self.state, 0, zmask, &pair, 0);
            return;
        }

        let half = 1usize << pivot;
        let block = half << 1;
        for (chunk_idx, chunk) in self.state.chunks_mut(block).enumerate() {
            let (lo, hi) = chunk.split_at_mut(half);
            pauli_rot_pair_halves(lo, hi, chunk_idx * block, xlow, zmask, &pair, 0);
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_pauli_rot_pairs_par(
        &mut self,
        pivot: usize,
        xlow: usize,
        zmask: usize,
        pair: &PauliRotPair,
    ) {
        if pivot == 0 {
            self.state
                .par_chunks_mut(MIN_PAR_ELEMS)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    apply_pauli_rot_pivot0_slice(chunk, chunk_idx * MIN_PAR_ELEMS, zmask, pair, 0);
                });
            return;
        }

        let half = 1usize << pivot;
        let block = half << 1;
        let num_blocks = self.state.len() / block;

        if num_blocks >= 4 {
            self.state
                .par_chunks_mut(block)
                .with_min_len(chunk_min_len(block))
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let (lo, hi) = chunk.split_at_mut(half);
                    pauli_rot_pair_halves(lo, hi, chunk_idx * block, xlow, zmask, pair, 0);
                });
        } else {
            // Tiles are a power of two above `xlow`, so `i ^ xlow` stays
            // inside its tile and each (lo, hi) tile pair is disjoint.
            let tile = MIN_PAR_ELEMS.max((xlow + 1).next_power_of_two());
            for (chunk_idx, chunk) in self.state.chunks_mut(block).enumerate() {
                let chunk_base = chunk_idx * block;
                let (lo, hi) = chunk.split_at_mut(half);
                lo.par_chunks_mut(tile)
                    .zip(hi.par_chunks_mut(tile))
                    .enumerate()
                    .for_each(|(t, (lo_t, hi_t))| {
                        pauli_rot_pair_halves(
                            lo_t,
                            hi_t,
                            chunk_base + t * tile,
                            xlow,
                            zmask,
                            pair,
                            0,
                        );
                    });
            }
        }
    }

    /// Multiply each amplitude by `even` or `odd` from the parity of its
    /// index bits under `zmask`.
    #[inline(always)]
    fn apply_parity_phase(&mut self, zmask: usize, even: Complex64, odd: Complex64) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            self.apply_parity_phase_par(zmask, even, odd);
            return;
        }

        apply_parity_phase_slice(&mut self.state, 0, zmask, even, odd);
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_parity_phase_par(&mut self, zmask: usize, even: Complex64, odd: Complex64) {
        self.state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                apply_parity_phase_slice(chunk, chunk_idx * MIN_PAR_ELEMS, zmask, even, odd);
            });
    }

    /// Apply `(xmask, zmask, theta)` Pauli rotations in order in one cache-tiled pass.
    ///
    /// Strings whose X and Y letters all sit below the tile bits run on the state in
    /// place; others gather the subcube holding their X and Y qubits, as
    /// [`apply_multi_2q`](Self::apply_multi_2q) does. Z letters never need the tile.
    /// A batch no tile holds takes one pass per rotation.
    pub(in crate::backend::statevector) fn apply_multi_pauli_rot(
        &mut self,
        rotations: &[(usize, usize, f64)],
    ) {
        let xall = rotations.iter().fold(0usize, |acc, &(x, _, _)| acc | x);
        let xqubits = (0..self.num_qubits).filter(|&q| xall >> q & 1 == 1);
        if let Some(plan) = qubit_subcube_plan(xqubits, self.num_qubits) {
            let tiled: Vec<TiledPauliRot> = rotations
                .iter()
                .map(|&(x, z, theta)| {
                    TiledPauliRot::new(x, z, theta, |q| {
                        Some(plan.map[q]).filter(|&bit| bit != usize::MAX)
                    })
                })
                .collect();
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            let rest = plan.rest(self.num_qubits);
            let subcubes = 1usize << (self.num_qubits - plan.tile_bits);
            #[cfg(feature = "parallel")]
            if self.num_qubits >= kernel_parallel_qubits() {
                let ptr = SendPtr(self.state.as_mut_ptr());
                (0..subcubes).into_par_iter().for_each(|outer| {
                    // SAFETY: `ptr` holds 2^num_qubits amplitudes and distinct
                    // `outer` values address disjoint subcubes, so no two tasks
                    // touch the same amplitude.
                    unsafe {
                        apply_pauli_rot_subcube(ptr.as_complex_ptr(), &plan, &rest, outer, &tiled)
                    };
                });
                return;
            }
            let state = self.state.as_mut_ptr();
            for outer in 0..subcubes {
                // SAFETY: `state` holds 2^num_qubits amplitudes and each `outer`
                // addresses its own subcube, applied one at a time here.
                unsafe { apply_pauli_rot_subcube(state, &plan, &rest, outer, &tiled) };
            }
            return;
        }

        let tile_bits = multi_2q_tile_bits_for(self.num_qubits).min(self.num_qubits);
        if xall >> tile_bits != 0 {
            for &(x, z, theta) in rotations {
                self.apply_pauli_rot_masks(x, z, (x & z).count_ones(), theta);
            }
            return;
        }
        let tiled: Vec<TiledPauliRot> = rotations
            .iter()
            .map(|&(x, z, theta)| TiledPauliRot::new(x, z, theta, |q| (q < tile_bits).then_some(q)))
            .collect();
        let tile_len = 1usize << tile_bits;

        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            self.state
                .par_chunks_mut(tile_len)
                .with_min_len(chunk_min_len(tile_len))
                .enumerate()
                .for_each(|(k, tile)| {
                    for rotation in &tiled {
                        rotation.apply(tile, k * tile_len);
                    }
                });
            return;
        }

        for (k, tile) in self.state.chunks_mut(tile_len).enumerate() {
            for rotation in &tiled {
                rotation.apply(tile, k * tile_len);
            }
        }
    }
}

//! QFT engine: twiddle caches, bit reversal, radix-2 and radix-4 butterflies and the
//! gathered high-stage passes.

use super::*;

type QftTwiddleTable = Arc<[Complex64]>;

struct CachedQftTwiddles {
    table: QftTwiddleTable,
    last_used: AtomicU64,
}

// Hits take the read lock only: the LRU stamp and the tick are atomics so a
// lookup never needs exclusive access. Misses build the table outside any lock
// and take the write lock to insert and evict. Relaxed suffices on the atomics
// because the write guard excludes every reader: the guard's release/acquire
// pair orders each stamp before the eviction scan that reads it.
struct QftTwiddleCache {
    entries: Vec<Option<CachedQftTwiddles>>,
    bytes: usize,
    limit_bytes: usize,
    tick: AtomicU64,
    /// Holds the conjugate tables the inverse transform reads.
    inverse: bool,
}

impl QftTwiddleCache {
    fn new(limit_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            limit_bytes,
            tick: AtomicU64::new(0),
            inverse: false,
        }
    }

    #[inline]
    fn next_tick(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::Relaxed).wrapping_add(1)
    }
}

const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

const QFT_TWIDDLE_CACHE_DEFAULT_LIMIT_MB: usize = 256;

// Soft LRU cap for QFT twiddles. One oversized table may occupy the cache
// alone, which keeps large repeated QFT runs from rebuilding twiddles every
// sample. `PRISM_QFT_TWIDDLE_CACHE_LIMIT_MB=0` disables caching.
#[inline]
fn qft_twiddle_cache_limit_bytes() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::env_knobs::usize_knob(
            "PRISM_QFT_TWIDDLE_CACHE_LIMIT_MB",
            QFT_TWIDDLE_CACHE_DEFAULT_LIMIT_MB,
            0,
        )
        .saturating_mul(1024 * 1024)
    })
}

#[inline(always)]
fn qft_twiddle_table_bytes(table_len: usize) -> usize {
    table_len.saturating_mul(std::mem::size_of::<Complex64>())
}

/// DIF twiddles `exp(2 pi i k / 2^n) / sqrt(2)`, conjugated for `inverse`.
fn qft_twiddles_scaled(n: usize, inverse: bool) -> QftTwiddleTable {
    static FORWARD: OnceLock<RwLock<QftTwiddleCache>> = OnceLock::new();
    static INVERSE: OnceLock<RwLock<QftTwiddleCache>> = OnceLock::new();
    let cache = if inverse {
        INVERSE.get_or_init(|| {
            RwLock::new(QftTwiddleCache {
                inverse: true,
                ..QftTwiddleCache::new(qft_twiddle_cache_limit_bytes())
            })
        })
    } else {
        FORWARD.get_or_init(|| RwLock::new(QftTwiddleCache::new(qft_twiddle_cache_limit_bytes())))
    };
    cached_qft_twiddles(cache, n)
}

fn cached_qft_twiddles(cache: &RwLock<QftTwiddleCache>, n: usize) -> QftTwiddleTable {
    let (cache_limit, inverse) = {
        let guard = cache.read().unwrap();
        if let Some(Some(entry)) = guard.entries.get(n) {
            entry
                .last_used
                .fetch_max(guard.next_tick(), Ordering::Relaxed);
            return Arc::clone(&entry.table);
        }
        (guard.limit_bytes, guard.inverse)
    };

    let qft_size = 1usize << n;
    let mut twiddles = Vec::with_capacity(qft_size / 2);
    for k in 0..qft_size / 2 {
        let angle = std::f64::consts::TAU * k as f64 / qft_size as f64;
        let w = Complex64::from_polar(INV_SQRT2, angle);
        twiddles.push(if inverse { w.conj() } else { w });
    }
    let twiddles: Arc<[Complex64]> = twiddles.into();
    let table_bytes = qft_twiddle_table_bytes(twiddles.len());
    if cache_limit == 0 {
        return twiddles;
    }

    let mut guard = cache.write().unwrap();
    let tick = guard.next_tick();
    if guard.entries.len() <= n {
        guard.entries.resize_with(n + 1, || None);
    }
    if let Some(existing) = &guard.entries[n] {
        existing.last_used.fetch_max(tick, Ordering::Relaxed);
        return Arc::clone(&existing.table);
    }

    while guard.bytes.saturating_add(table_bytes) > cache_limit {
        let Some((idx, _)) = guard
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| {
                entry
                    .as_ref()
                    .map(|entry| (idx, entry.last_used.load(Ordering::Relaxed)))
            })
            .min_by_key(|&(_, last_used)| last_used)
        else {
            break;
        };
        if let Some(evicted) = guard.entries[idx].take() {
            guard.bytes = guard
                .bytes
                .saturating_sub(qft_twiddle_table_bytes(evicted.table.len()));
        }
    }

    guard.entries[n] = Some(CachedQftTwiddles {
        table: Arc::clone(&twiddles),
        last_used: AtomicU64::new(tick),
    });
    guard.bytes = guard.bytes.saturating_add(table_bytes);
    twiddles
}

/// Fewest low qubits [`StatevectorBackend::apply_qft_block`] runs together in one tile.
const QFT_MIN_TILE_BITS: usize = 13;

/// Most low qubits per tile: twice the tile budget, 512 KiB on the x86 cores measured.
#[inline(always)]
pub(crate) fn qft_max_tile_bits() -> usize {
    (2 * cache::tile_budget_bytes() / size_of::<Complex64>()).ilog2() as usize
}

/// Tile width for an `n`-qubit QFT over a `state_bits`-qubit state: as wide as leaves
/// the pool its tile count ([`cache::min_parallel_tiles`]), and above the minimum one
/// bit narrower where that leaves an even count of stages above the tile, which then
/// all run as radix-4 pairs instead of paying a radix-2 pass.
pub(crate) fn qft_tile_bits(n: usize, state_bits: usize) -> usize {
    qft_tile_bits_for(n, qft_max_tile_bits(), qft_min_tiles_log2(n, state_bits))
}

/// Log2 of the tiles each `2^n` chunk of a `state_bits`-qubit state owes the pool.
pub(crate) fn qft_min_tiles_log2(n: usize, state_bits: usize) -> usize {
    (cache::min_parallel_tiles().ilog2() as usize).saturating_sub(state_bits - n)
}

/// Tile width before the radix-4 pairing adjustment of [`qft_tile_bits_for`], which
/// the gathered high-stage passes do not need.
pub(crate) fn qft_widest_tile_bits_for(
    n: usize,
    max_tile_bits: usize,
    min_tiles_log2: usize,
) -> usize {
    max_tile_bits
        .min(n.saturating_sub(min_tiles_log2))
        .max(QFT_MIN_TILE_BITS.min(n))
}

fn qft_tile_bits_for(n: usize, max_tile_bits: usize, min_tiles_log2: usize) -> usize {
    let widest = qft_widest_tile_bits_for(n, max_tile_bits, min_tiles_log2);
    if widest > QFT_MIN_TILE_BITS && (n - widest) % 2 == 1 {
        widest - 1
    } else {
        widest
    }
}

/// Smallest QFT the gathered high-stage passes run on, where a whole-register transform
/// on an 8-thread host first leaves [`QFT_GATHER_MIN_STAGES`] stages above its tile.
/// Provisional until swept.
pub(crate) const QFT_GATHER_MIN_QUBITS: usize = 16;

/// Fewest stages above the tile that take the gathered passes. From three, one gathered
/// pass replaces at least two radix-2 or radix-4 passes; at two or fewer both take one.
pub(crate) const QFT_GATHER_MIN_STAGES: usize = 3;

/// Fewest amplitudes per gathered row: four cache lines, so a strided row read is a
/// burst rather than a lone line.
const QFT_GATHER_MIN_ROW: usize = 16;

/// Amplitudes one gathered group holds: half the tile budget, so the buffers of two SMT
/// siblings share one L2.
#[inline(always)]
fn qft_gather_len() -> usize {
    cache::tile_budget_bytes() / 2 / size_of::<Complex64>()
}

/// Most stages one gathered pass runs: 9 on a 256 KiB tile budget.
#[inline(always)]
pub(crate) fn qft_gather_max_bits() -> usize {
    (qft_gather_len() / QFT_GATHER_MIN_ROW).ilog2() as usize
}

/// Unit phases `exp(2 pi i x / 2^total_bits)` for `x < 2^total_bits`, conjugated for
/// `inverse`, as a product of three tables. The low table spans a gathered group, so a
/// phase below that length is one read.
struct QftGatherPhases {
    low: Box<[Complex64]>,
    mid: Box<[Complex64]>,
    high: Box<[Complex64]>,
    low_bits: usize,
    mid_bits: usize,
}

impl QftGatherPhases {
    fn new(total_bits: usize, inverse: bool) -> Self {
        let low_bits = (qft_gather_len().ilog2() as usize).min(total_bits);
        let rest = total_bits - low_bits;
        let mid_bits = rest.div_ceil(2);
        let scale = std::f64::consts::TAU / (1u64 << total_bits) as f64;
        let table = |bits: usize, shift: usize| -> Box<[Complex64]> {
            (0..1usize << bits)
                .map(|x| {
                    let w = Complex64::from_polar(1.0, (x << shift) as f64 * scale);
                    if inverse { w.conj() } else { w }
                })
                .collect()
        };
        Self {
            low: table(low_bits, 0),
            mid: table(mid_bits, low_bits),
            high: table(rest - mid_bits, low_bits + mid_bits),
            low_bits,
            mid_bits,
        }
    }

    #[inline(always)]
    fn phase(&self, x: usize) -> Complex64 {
        let mid = (x >> self.low_bits) & ((1 << self.mid_bits) - 1);
        self.low[x & (self.low.len() - 1)]
            * self.mid[mid]
            * self.high[x >> (self.low_bits + self.mid_bits)]
    }
}

fn qft_gather_phases(total_bits: usize, inverse: bool) -> &'static QftGatherPhases {
    const WIDTHS: usize = usize::BITS as usize;
    static FORWARD: [OnceLock<QftGatherPhases>; WIDTHS] = [const { OnceLock::new() }; WIDTHS];
    static INVERSE: [OnceLock<QftGatherPhases>; WIDTHS] = [const { OnceLock::new() }; WIDTHS];
    let slots = if inverse { &INVERSE } else { &FORWARD };
    slots[total_bits].get_or_init(|| QftGatherPhases::new(total_bits, inverse))
}

/// Per-stride QFT twiddles for the in-tile stages: entries `[s, 2s)` hold
/// `exp(2 pi i k / 2s) / sqrt(2)` for each power of two `s`, conjugated for `inverse`.
/// They are bit-identical to the whole-state table's entries `k * total / 2s`, which a
/// small stride would read one page apart.
fn qft_compact_twiddles(inverse: bool) -> &'static [Complex64] {
    static FORWARD: OnceLock<Box<[Complex64]>> = OnceLock::new();
    static INVERSE: OnceLock<Box<[Complex64]>> = OnceLock::new();
    let build = || {
        let mut table = vec![Complex64::new(0.0, 0.0); 1 << qft_max_tile_bits()];
        let mut s = 1;
        while s < table.len() {
            for k in 0..s {
                let angle = std::f64::consts::TAU * k as f64 / (2 * s) as f64;
                let w = Complex64::from_polar(INV_SQRT2, angle);
                table[s + k] = if inverse { w.conj() } else { w };
            }
            s <<= 1;
        }
        table.into_boxed_slice()
    };
    if inverse {
        INVERSE.get_or_init(build)
    } else {
        FORWARD.get_or_init(build)
    }
}

#[inline(always)]
pub(crate) fn reverse_low_bits(x: usize, bits: usize) -> usize {
    if bits == 0 {
        return 0;
    }
    x.reverse_bits() >> (usize::BITS as usize - bits)
}

/// Bits per side of the square tiles the bit reversal transposes: 32 rows of 32
/// amplitudes, so a tile pair takes 32 KiB.
const BITREV_TILE_BITS: usize = 5;

/// Move tile `b` of a `bits`-bit state to tile `reverse(b)` and back, viewing an index
/// as `(a, b, c)` with `a` the high and `c` the low [`BITREV_TILE_BITS`] bits.
/// Amplitude `(a, b, c)` moves to `(rev(c), rev(b), rev(a))`, so every row read or
/// written is a contiguous run. Out of line, so its 32 KiB of tiles live only while a
/// task runs instead of in every frame of Rayon's recursive split.
///
/// # Safety
///
/// `state` must be valid for `1 << bits` amplitudes, `bits >= 2 * BITREV_TILE_BITS`,
/// and no other thread may access tile `b` or its reverse during the call.
#[inline(never)]
unsafe fn bit_reverse_tile_pair(state: *mut Complex64, bits: usize, b: usize) {
    const Q: usize = BITREV_TILE_BITS;
    const SIDE: usize = 1 << Q;
    let rb = reverse_low_bits(b, bits - 2 * Q);
    let row_stride = 1usize << (bits - Q);
    let rev: [usize; SIDE] = std::array::from_fn(|x| reverse_low_bits(x, Q));
    let mut tile_b = [Complex64::new(0.0, 0.0); SIDE * SIDE];
    let mut tile_rb = [Complex64::new(0.0, 0.0); SIDE * SIDE];
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        for a in 0..SIDE {
            let row = a * row_stride;
            std::ptr::copy_nonoverlapping(
                state.add(row + (b << Q)),
                tile_b.as_mut_ptr().add(a * SIDE),
                SIDE,
            );
            if rb != b {
                std::ptr::copy_nonoverlapping(
                    state.add(row + (rb << Q)),
                    tile_rb.as_mut_ptr().add(a * SIDE),
                    SIDE,
                );
            }
        }
        for a in 0..SIDE {
            let row = a * row_stride;
            let out_rb = state.add(row + (rb << Q));
            for c in 0..SIDE {
                *out_rb.add(c) = tile_b[rev[c] * SIDE + rev[a]];
            }
            if rb != b {
                let out_b = state.add(row + (b << Q));
                for c in 0..SIDE {
                    *out_b.add(c) = tile_rb[rev[c] * SIDE + rev[a]];
                }
            }
        }
    }
}

#[inline]
fn apply_bit_reverse_permutation(state: &mut [Complex64], bits: usize) {
    let len = state.len();
    debug_assert_eq!(len, 1usize << bits);
    if len <= 2 {
        return;
    }
    if bits < 2 * BITREV_TILE_BITS {
        for i in 0..len {
            let j = reverse_low_bits(i, bits);
            if j > i {
                state.swap(i, j);
            }
        }
        return;
    }

    let middle = bits - 2 * BITREV_TILE_BITS;
    let owns_pair = move |&b: &usize| reverse_low_bits(b, middle) >= b;

    #[cfg(feature = "parallel")]
    if bits >= kernel_parallel_qubits() {
        let ptr = SendPtr(state.as_mut_ptr());
        (0..1usize << middle)
            .into_par_iter()
            .filter(owns_pair)
            .for_each(move |b| {
                // SAFETY: reversal of the middle bits is an involution, so the pairs
                // `{b, reverse(b)}` with `b <= reverse(b)` partition the tiles and each
                // task owns its pair. Tiles are disjoint sets of rows of the state.
                unsafe { bit_reverse_tile_pair(ptr.as_complex_ptr(), bits, b) }
            });
        return;
    }

    let ptr = state.as_mut_ptr();
    for b in (0..1usize << middle).filter(owns_pair) {
        // SAFETY: `ptr` spans the whole state, borrowed mutably for this loop alone.
        unsafe { bit_reverse_tile_pair(ptr, bits, b) }
    }
}

/// Reverse the low `bits` index bits within every `2^bits` chunk of `state`.
fn apply_bit_reverse_chunks(state: &mut [Complex64], bits: usize) {
    #[cfg(test)]
    crate::backend::statevector::sweeps::tick();
    let chunk = 1usize << bits;
    if state.len() == chunk {
        apply_bit_reverse_permutation(state, bits);
        return;
    }
    #[cfg(feature = "parallel")]
    if state.len() >= 1usize << kernel_parallel_qubits() {
        state
            .par_chunks_mut(chunk.max(MIN_PAR_ELEMS))
            .for_each(|task| {
                for c in task.chunks_mut(chunk) {
                    apply_bit_reverse_permutation(c, bits);
                }
            });
        return;
    }
    for c in state.chunks_mut(chunk) {
        apply_bit_reverse_permutation(c, bits);
    }
}

/// Run one FFT stage in parallel.
///
/// Large group counts split by group. High-stride stages split inside each
/// group so all cores still get work.
#[cfg(feature = "parallel")]
#[inline]
fn fft_stage_par(
    state: &mut [Complex64],
    stride: usize,
    block_size: usize,
    twiddle_step: usize,
    twiddles_scaled: &[Complex64],
) {
    let total = state.len();
    let num_groups = total / block_size;

    if num_groups >= 4 && block_size >= MIN_PAR_ELEMS {
        // Many large groups: split state by group.
        state.par_chunks_mut(block_size).for_each(|group| {
            let (lo, hi) = group.split_at_mut(stride);
            apply_radix2_span(lo, hi, 0, twiddle_step, twiddles_scaled);
        });
    } else if block_size < MIN_PAR_ELEMS {
        // Small groups: bundle work so each Rayon job has enough elements.
        let task_size = MIN_PAR_ELEMS;
        state.par_chunks_mut(task_size).for_each(|task_chunk| {
            let usable = task_chunk.len() - (task_chunk.len() % block_size);
            run_radix2_stage_seq(
                &mut task_chunk[..usable],
                stride,
                twiddles_scaled,
                twiddle_step,
            );
        });
    } else {
        // Few groups: split each large group into zipped chunks.
        const MIN_PAR_PAIRS: usize = MIN_PAR_ELEMS / 2;
        for group_start in (0..total).step_by(block_size) {
            let group = &mut state[group_start..group_start + block_size];
            let (lo, hi) = group.split_at_mut(stride);
            let sub = MIN_PAR_PAIRS.min(stride).max(1);
            lo.par_chunks_mut(sub)
                .zip(hi.par_chunks_mut(sub))
                .enumerate()
                .for_each(|(chunk_idx, (lo_c, hi_c))| {
                    let k_base = chunk_idx * sub;
                    apply_radix2_span(lo_c, hi_c, k_base, twiddle_step, twiddles_scaled);
                });
        }
    }
}

/// Scalar radix-2 DIF butterfly. `w` includes the `1/sqrt(2)` factor.
#[inline(always)]
fn radix2_butterfly_values(a: Complex64, b: Complex64, w: Complex64) -> (Complex64, Complex64) {
    let lo = Complex64::new((a.re + b.re) * INV_SQRT2, (a.im + b.im) * INV_SQRT2);
    let hi = (a - b) * w;
    (lo, hi)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn radix2_span_avx512(
    pl: *mut f64,
    ph: *mut f64,
    pairs: usize,
    k_base: usize,
    twiddles: *const Complex64,
    twiddle_step: usize,
) -> usize {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm512_add_pd, _mm512_loadu_pd, _mm512_movedup_pd, _mm512_mul_pd, _mm512_permute_pd,
            _mm512_set1_pd, _mm512_storeu_pd, _mm512_sub_pd,
        };
        let inv_sqrt2_v = _mm512_set1_pd(INV_SQRT2);
        let end = pairs & !3;
        for off in (0..end).step_by(4) {
            let a = _mm512_loadu_pd(pl.add(off * 2));
            let b = _mm512_loadu_pd(ph.add(off * 2));
            let w = twiddle_quad_avx512(twiddles, k_base + off, twiddle_step);
            let w_rr = _mm512_movedup_pd(w);
            let w_ii = _mm512_permute_pd::<0xFF>(w);
            _mm512_storeu_pd(
                pl.add(off * 2),
                _mm512_mul_pd(_mm512_add_pd(a, b), inv_sqrt2_v),
            );
            _mm512_storeu_pd(
                ph.add(off * 2),
                simd::complex_mul_avx512(w_rr, w_ii, _mm512_sub_pd(a, b)),
            );
        }
        end
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn radix2_span_avx2fma(
    pl: *mut f64,
    ph: *mut f64,
    start: usize,
    pairs: usize,
    k_base: usize,
    twiddles: *const Complex64,
    twiddle_step: usize,
) -> usize {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_movedup_pd, _mm256_mul_pd, _mm256_permute_pd,
            _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
        };
        let end = pairs & !1;
        if start >= end {
            return start;
        }
        let inv_sqrt2_v = _mm256_set1_pd(INV_SQRT2);
        for off in (start..end).step_by(2) {
            let a = _mm256_loadu_pd(pl.add(off * 2));
            let b = _mm256_loadu_pd(ph.add(off * 2));
            let w = twiddle_pair_avx2(twiddles, k_base + off, twiddle_step);
            let w_rr = _mm256_movedup_pd(w);
            let w_ii = _mm256_permute_pd(w, 0b1111);
            _mm256_storeu_pd(
                pl.add(off * 2),
                _mm256_mul_pd(_mm256_add_pd(a, b), inv_sqrt2_v),
            );
            _mm256_storeu_pd(
                ph.add(off * 2),
                simd::complex_mul_avx2fma(w_rr, w_ii, _mm256_sub_pd(a, b)),
            );
        }
        end
    }
}

#[cfg(any(feature = "parallel", test))]
#[inline]
fn apply_radix2_span(
    lo: &mut [Complex64],
    hi: &mut [Complex64],
    k_base: usize,
    twiddle_step: usize,
    twiddles_scaled: &[Complex64],
) {
    let pairs = lo.len();
    assert_eq!(pairs, hi.len());
    if pairs == 0 {
        return;
    }
    assert!((k_base + pairs - 1) * twiddle_step < twiddles_scaled.len());

    #[allow(unused_mut)]
    let mut done = 0usize;
    #[cfg(target_arch = "x86_64")]
    {
        let tier = simd::simd_tier();
        let pl = lo.as_mut_ptr() as *mut f64;
        let ph = hi.as_mut_ptr() as *mut f64;
        let twiddles = twiddles_scaled.as_ptr();
        if tier == simd::SimdTier::Avx512 && pairs >= 4 {
            // SAFETY: AVX-512F detected. `lo` and `hi` hold `pairs` disjoint amplitudes, and
            // `twiddles` is bounded by the assertion above.
            done = unsafe { radix2_span_avx512(pl, ph, pairs, k_base, twiddles, twiddle_step) };
        }
        if matches!(tier, simd::SimdTier::Avx512 | simd::SimdTier::Avx2Fma) && pairs - done >= 2 {
            // SAFETY: AVX2 and FMA detected. `lo` and `hi` hold `pairs` disjoint amplitudes,
            // and `twiddles` is bounded by the assertion above.
            done =
                unsafe { radix2_span_avx2fma(pl, ph, done, pairs, k_base, twiddles, twiddle_step) };
        }
    }
    for j in done..pairs {
        let w = twiddles_scaled[(k_base + j) * twiddle_step];
        let (out_lo, out_hi) = radix2_butterfly_values(lo[j], hi[j], w);
        lo[j] = out_lo;
        hi[j] = out_hi;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn radix2_stride1_avx512(base: *mut f64, groups: usize, w: Complex64) -> usize {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm512_add_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_set1_pd, _mm512_shuffle_f64x2,
            _mm512_storeu_pd, _mm512_sub_pd,
        };
        let inv_sqrt2_v = _mm512_set1_pd(INV_SQRT2);
        let (w_rr, w_ii) = (_mm512_set1_pd(w.re), _mm512_set1_pd(w.im));
        let end = groups & !3;
        for g in (0..end).step_by(4) {
            let p = base.add(g * 4);
            let v0 = _mm512_loadu_pd(p);
            let v1 = _mm512_loadu_pd(p.add(8));
            let a = _mm512_shuffle_f64x2::<0x88>(v0, v1);
            let b = _mm512_shuffle_f64x2::<0xDD>(v0, v1);
            let out_lo = _mm512_mul_pd(_mm512_add_pd(a, b), inv_sqrt2_v);
            let out_hi = simd::complex_mul_avx512(w_rr, w_ii, _mm512_sub_pd(a, b));
            let t0 = _mm512_shuffle_f64x2::<0x44>(out_lo, out_hi);
            let t1 = _mm512_shuffle_f64x2::<0xEE>(out_lo, out_hi);
            _mm512_storeu_pd(p, _mm512_shuffle_f64x2::<0xD8>(t0, t0));
            _mm512_storeu_pd(p.add(8), _mm512_shuffle_f64x2::<0xD8>(t1, t1));
        }
        end
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn radix2_stride1_avx2fma(
    base: *mut f64,
    start: usize,
    groups: usize,
    w: Complex64,
) -> usize {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute2f128_pd, _mm256_set1_pd,
            _mm256_storeu_pd, _mm256_sub_pd,
        };
        let end = groups & !1;
        if start >= end {
            return start;
        }
        let inv_sqrt2_v = _mm256_set1_pd(INV_SQRT2);
        let (w_rr, w_ii) = (_mm256_set1_pd(w.re), _mm256_set1_pd(w.im));
        for g in (start..end).step_by(2) {
            let p = base.add(g * 4);
            let v0 = _mm256_loadu_pd(p);
            let v1 = _mm256_loadu_pd(p.add(4));
            let a = _mm256_permute2f128_pd::<0x20>(v0, v1);
            let b = _mm256_permute2f128_pd::<0x31>(v0, v1);
            let out_lo = _mm256_mul_pd(_mm256_add_pd(a, b), inv_sqrt2_v);
            let out_hi = simd::complex_mul_avx2fma(w_rr, w_ii, _mm256_sub_pd(a, b));
            _mm256_storeu_pd(p, _mm256_permute2f128_pd::<0x20>(out_lo, out_hi));
            _mm256_storeu_pd(p.add(4), _mm256_permute2f128_pd::<0x31>(out_lo, out_hi));
        }
        end
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn radix2_stride2_avx512(
    base: *mut f64,
    groups: usize,
    w0: Complex64,
    w1: Complex64,
) -> usize {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm512_add_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_set1_pd, _mm512_setr_pd,
            _mm512_shuffle_f64x2, _mm512_storeu_pd, _mm512_sub_pd,
        };
        let inv_sqrt2_v = _mm512_set1_pd(INV_SQRT2);
        let w_rr = _mm512_setr_pd(w0.re, w0.re, w1.re, w1.re, w0.re, w0.re, w1.re, w1.re);
        let w_ii = _mm512_setr_pd(w0.im, w0.im, w1.im, w1.im, w0.im, w0.im, w1.im, w1.im);
        let end = groups & !1;
        for g in (0..end).step_by(2) {
            let p = base.add(g * 8);
            let v0 = _mm512_loadu_pd(p);
            let v1 = _mm512_loadu_pd(p.add(8));
            let a = _mm512_shuffle_f64x2::<0x44>(v0, v1);
            let b = _mm512_shuffle_f64x2::<0xEE>(v0, v1);
            let out_lo = _mm512_mul_pd(_mm512_add_pd(a, b), inv_sqrt2_v);
            let out_hi = simd::complex_mul_avx512(w_rr, w_ii, _mm512_sub_pd(a, b));
            _mm512_storeu_pd(p, _mm512_shuffle_f64x2::<0x44>(out_lo, out_hi));
            _mm512_storeu_pd(p.add(8), _mm512_shuffle_f64x2::<0xEE>(out_lo, out_hi));
        }
        end
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn radix2_stride2_avx2fma(
    base: *mut f64,
    start: usize,
    groups: usize,
    w0: Complex64,
    w1: Complex64,
) -> usize {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set_pd, _mm256_set1_pd,
            _mm256_storeu_pd, _mm256_sub_pd,
        };
        let inv_sqrt2_v = _mm256_set1_pd(INV_SQRT2);
        let w_rr = _mm256_set_pd(w1.re, w1.re, w0.re, w0.re);
        let w_ii = _mm256_set_pd(w1.im, w1.im, w0.im, w0.im);
        for g in start..groups {
            let p = base.add(g * 8);
            let a = _mm256_loadu_pd(p);
            let b = _mm256_loadu_pd(p.add(4));
            _mm256_storeu_pd(p, _mm256_mul_pd(_mm256_add_pd(a, b), inv_sqrt2_v));
            _mm256_storeu_pd(
                p.add(4),
                simd::complex_mul_avx2fma(w_rr, w_ii, _mm256_sub_pd(a, b)),
            );
        }
        groups
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn radix2_groups_avx512(
    base: *mut f64,
    groups: usize,
    stride: usize,
    twiddles_scaled: &[Complex64],
    twiddle_step: usize,
) {
    assert!(stride >= 4);
    if groups == 0 {
        return;
    }
    assert!((stride - 1) * twiddle_step < twiddles_scaled.len());
    let twiddles = twiddles_scaled.as_ptr();
    // SAFETY: same contract as the enclosing unsafe fn; twiddle bounds are asserted above.
    unsafe {
        for g in 0..groups {
            let pl = base.add(g * stride * 4);
            let ph = pl.add(stride * 2);
            let done = radix2_span_avx512(pl, ph, stride, 0, twiddles, twiddle_step);
            let pl_c = pl as *mut Complex64;
            let ph_c = ph as *mut Complex64;
            for k in done..stride {
                let w = *twiddles.add(k * twiddle_step);
                let (lo, hi) = radix2_butterfly_values(*pl_c.add(k), *ph_c.add(k), w);
                *pl_c.add(k) = lo;
                *ph_c.add(k) = hi;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn radix2_groups_avx2fma(
    base: *mut f64,
    groups: usize,
    stride: usize,
    twiddles_scaled: &[Complex64],
    twiddle_step: usize,
) {
    assert!(stride >= 2);
    if groups == 0 {
        return;
    }
    assert!((stride - 1) * twiddle_step < twiddles_scaled.len());
    let twiddles = twiddles_scaled.as_ptr();
    // SAFETY: same contract as the enclosing unsafe fn; twiddle bounds are asserted above.
    unsafe {
        for g in 0..groups {
            let pl = base.add(g * stride * 4);
            let ph = pl.add(stride * 2);
            let done = radix2_span_avx2fma(pl, ph, 0, stride, 0, twiddles, twiddle_step);
            let pl_c = pl as *mut Complex64;
            let ph_c = ph as *mut Complex64;
            for k in done..stride {
                let w = *twiddles.add(k * twiddle_step);
                let (lo, hi) = radix2_butterfly_values(*pl_c.add(k), *ph_c.add(k), w);
                *pl_c.add(k) = lo;
                *ph_c.add(k) = hi;
            }
        }
    }
}

/// Apply one full radix-2 DIF stage in place to `state` at the given stride.
/// Butterfly `k` takes `twiddles_scaled[k * twiddle_step]`: step `total / (2 * stride)`
/// on the whole-state table, 1 on a per-stride run.
#[inline]
fn run_radix2_stage_seq(
    state: &mut [Complex64],
    stride: usize,
    twiddles_scaled: &[Complex64],
    twiddle_step: usize,
) {
    if state.is_empty() || stride == 0 {
        return;
    }
    let block_size = stride << 1;
    let total = state.len();
    debug_assert_eq!(total % block_size, 0);
    let groups = total / block_size;
    if groups == 0 {
        return;
    }
    assert!((stride - 1) * twiddle_step < twiddles_scaled.len());

    if stride == 1 {
        #[allow(unused_mut)]
        let mut g = 0usize;
        let w = twiddles_scaled[0];
        #[cfg(target_arch = "x86_64")]
        {
            let tier = simd::simd_tier();
            let base = state.as_mut_ptr() as *mut f64;
            if tier == simd::SimdTier::Avx512 && groups >= 4 {
                // SAFETY: AVX-512F detected. `state` holds `groups` pairs of amplitudes.
                g = unsafe { radix2_stride1_avx512(base, groups, w) };
            }
            if matches!(tier, simd::SimdTier::Avx512 | simd::SimdTier::Avx2Fma) && groups - g >= 2 {
                // SAFETY: AVX2 and FMA detected. `state` holds `groups` pairs of amplitudes.
                g = unsafe { radix2_stride1_avx2fma(base, g, groups, w) };
            }
        }
        while g < groups {
            let i0 = g << 1;
            let (lo, hi) = radix2_butterfly_values(state[i0], state[i0 + 1], w);
            state[i0] = lo;
            state[i0 + 1] = hi;
            g += 1;
        }
        return;
    }

    if stride == 2 {
        #[allow(unused_mut)]
        let mut g = 0usize;
        let (w0, w1) = (twiddles_scaled[0], twiddles_scaled[twiddle_step]);
        #[cfg(target_arch = "x86_64")]
        {
            let tier = simd::simd_tier();
            let base = state.as_mut_ptr() as *mut f64;
            if tier == simd::SimdTier::Avx512 && groups >= 2 {
                // SAFETY: AVX-512F detected. `state` holds `groups` 4-amplitude groups.
                g = unsafe { radix2_stride2_avx512(base, groups, w0, w1) };
            }
            if matches!(tier, simd::SimdTier::Avx512 | simd::SimdTier::Avx2Fma) && g < groups {
                // SAFETY: AVX2 and FMA detected. `state` holds `groups` 4-amplitude groups.
                g = unsafe { radix2_stride2_avx2fma(base, g, groups, w0, w1) };
            }
        }
        while g < groups {
            let i0 = g << 2;
            let (lo0, hi0) = radix2_butterfly_values(state[i0], state[i0 + 2], w0);
            state[i0] = lo0;
            state[i0 + 2] = hi0;
            let (lo1, hi1) = radix2_butterfly_values(state[i0 + 1], state[i0 + 3], w1);
            state[i0 + 1] = lo1;
            state[i0 + 3] = hi1;
            g += 1;
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let tier = simd::simd_tier();
        if tier == simd::SimdTier::Avx512 && stride >= 4 {
            // SAFETY: AVX-512F detected. `state` holds `groups` contiguous blocks of `2 * stride`
            // amplitudes, and twiddle bounds are asserted above and inside `radix2_groups_avx512`.
            unsafe {
                radix2_groups_avx512(
                    state.as_mut_ptr() as *mut f64,
                    groups,
                    stride,
                    twiddles_scaled,
                    twiddle_step,
                );
            }
            return;
        }
        if matches!(tier, simd::SimdTier::Avx512 | simd::SimdTier::Avx2Fma) {
            // SAFETY: AVX2 and FMA detected. `state` holds `groups` contiguous blocks of
            // `2 * stride` amplitudes, and twiddle bounds are asserted above and inside
            // `radix2_groups_avx2fma`.
            unsafe {
                radix2_groups_avx2fma(
                    state.as_mut_ptr() as *mut f64,
                    groups,
                    stride,
                    twiddles_scaled,
                    twiddle_step,
                );
            }
            return;
        }
    }

    let mut group_start = 0;
    while group_start < total {
        for k in 0..stride {
            let i0 = group_start + k;
            let i1 = i0 + stride;
            let w = twiddles_scaled[k * twiddle_step];
            let (lo, hi) = radix2_butterfly_values(state[i0], state[i1], w);
            state[i0] = lo;
            state[i1] = hi;
        }
        group_start += block_size;
    }
}

/// Scalar radix-4 butterfly math. Twiddles include the `1/sqrt(2)` factor.
#[inline(always)]
fn radix4_butterfly_values(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    d: Complex64,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let t_ac_sum = Complex64::new((a.re + c.re) * INV_SQRT2, (a.im + c.im) * INV_SQRT2);
    let t_ac_diff = (a - c) * w_2s_k;
    let t_bd_sum = Complex64::new((b.re + d.re) * INV_SQRT2, (b.im + d.im) * INV_SQRT2);
    let t_bd_diff = (b - d) * w_2s_kps;

    let oa = Complex64::new(
        (t_ac_sum.re + t_bd_sum.re) * INV_SQRT2,
        (t_ac_sum.im + t_bd_sum.im) * INV_SQRT2,
    );
    let ob = (t_ac_sum - t_bd_sum) * w_s_k;
    let oc = Complex64::new(
        (t_ac_diff.re + t_bd_diff.re) * INV_SQRT2,
        (t_ac_diff.im + t_bd_diff.im) * INV_SQRT2,
    );
    let od = (t_ac_diff - t_bd_diff) * w_s_k;
    (oa, ob, oc, od)
}

/// Apply one fused radix-4 butterfly at state index `i`.
#[inline(always)]
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn radix4_butterfly_scalar(
    state: &mut [Complex64],
    i: usize,
    inner_stride: usize,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
) {
    let s = inner_stride;
    let (oa, ob, oc, od) = radix4_butterfly_values(
        state[i],
        state[i + s],
        state[i + 2 * s],
        state[i + 3 * s],
        w_2s_k,
        w_2s_kps,
        w_s_k,
    );
    state[i] = oa;
    state[i + s] = ob;
    state[i + 2 * s] = oc;
    state[i + 3 * s] = od;
}

/// Apply one radix-4 butterfly across four disjoint quarter slices.
#[inline(always)]
#[cfg_attr(
    any(target_arch = "aarch64", not(feature = "parallel")),
    allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
fn radix4_butterfly_quartet_scalar(
    qa: &mut [Complex64],
    qb: &mut [Complex64],
    qc: &mut [Complex64],
    qd: &mut [Complex64],
    j: usize,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
) {
    let (oa, ob, oc, od) =
        radix4_butterfly_values(qa[j], qb[j], qc[j], qd[j], w_2s_k, w_2s_kps, w_s_k);
    qa[j] = oa;
    qb[j] = ob;
    qc[j] = oc;
    qd[j] = od;
}

/// FMA variant for one radix-4 butterfly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn radix4_butterfly_fma(
    base_ptr: *mut f64,
    inner_stride: usize,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd, _mm_sub_pd,
        };
        let s = inner_stride;
        let inv_sqrt2_v = _mm_set1_pd(std::f64::consts::FRAC_1_SQRT_2);

        let a_ptr = base_ptr;
        let b_ptr = base_ptr.add(s * 2);
        let c_ptr = base_ptr.add(2 * s * 2);
        let d_ptr = base_ptr.add(3 * s * 2);

        let a = _mm_loadu_pd(a_ptr);
        let b = _mm_loadu_pd(b_ptr);
        let c = _mm_loadu_pd(c_ptr);
        let d = _mm_loadu_pd(d_ptr);

        let w_2s_k_rr = _mm_set1_pd(w_2s_k.re);
        let w_2s_k_ii = _mm_set1_pd(w_2s_k.im);
        let w_2s_kps_rr = _mm_set1_pd(w_2s_kps.re);
        let w_2s_kps_ii = _mm_set1_pd(w_2s_kps.im);
        let w_s_k_rr = _mm_set1_pd(w_s_k.re);
        let w_s_k_ii = _mm_set1_pd(w_s_k.im);

        let t_ac_sum = _mm_mul_pd(_mm_add_pd(a, c), inv_sqrt2_v);
        let t_ac_diff = simd::complex_mul_fma(w_2s_k_rr, w_2s_k_ii, _mm_sub_pd(a, c));
        let t_bd_sum = _mm_mul_pd(_mm_add_pd(b, d), inv_sqrt2_v);
        let t_bd_diff = simd::complex_mul_fma(w_2s_kps_rr, w_2s_kps_ii, _mm_sub_pd(b, d));

        _mm_storeu_pd(
            a_ptr,
            _mm_mul_pd(_mm_add_pd(t_ac_sum, t_bd_sum), inv_sqrt2_v),
        );
        _mm_storeu_pd(
            b_ptr,
            simd::complex_mul_fma(w_s_k_rr, w_s_k_ii, _mm_sub_pd(t_ac_sum, t_bd_sum)),
        );
        _mm_storeu_pd(
            c_ptr,
            _mm_mul_pd(_mm_add_pd(t_ac_diff, t_bd_diff), inv_sqrt2_v),
        );
        _mm_storeu_pd(
            d_ptr,
            simd::complex_mul_fma(w_s_k_rr, w_s_k_ii, _mm_sub_pd(t_ac_diff, t_bd_diff)),
        );
    }
}

/// AVX2-256 SIMD variant: processes TWO radix-4 butterflies per call (at
/// consecutive `k` and `k+1`, four complex amplitudes each, eight elements
/// total). Each load/store is one 256-bit op covering two adjacent complex
/// values (which are 32 bytes contiguous in memory). Twiddles are loaded
/// scalar-by-scalar since the inner twiddle step is 2.
#[cfg(all(target_arch = "x86_64", test))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn radix4_butterfly_pair_avx2fma(
    base_ptr: *mut f64,
    inner_stride: usize,
    w_2s_k0: Complex64,
    w_2s_k1: Complex64,
    w_2s_kps0: Complex64,
    w_2s_kps1: Complex64,
    w_s_k0: Complex64,
    w_s_k1: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set_pd, _mm256_set1_pd,
            _mm256_storeu_pd, _mm256_sub_pd,
        };
        let s = inner_stride;
        let inv_sqrt2_v = _mm256_set1_pd(std::f64::consts::FRAC_1_SQRT_2);

        let a_ptr = base_ptr;
        let b_ptr = base_ptr.add(s * 2);
        let c_ptr = base_ptr.add(2 * s * 2);
        let d_ptr = base_ptr.add(3 * s * 2);

        let a = _mm256_loadu_pd(a_ptr);
        let b = _mm256_loadu_pd(b_ptr);
        let c = _mm256_loadu_pd(c_ptr);
        let d = _mm256_loadu_pd(d_ptr);

        // c_rr layout: [w0.re, w0.re, w1.re, w1.re], broadcasts each twiddle's
        // real part to the two lanes corresponding to that butterfly's complex.
        let w_2s_k_rr = _mm256_set_pd(w_2s_k1.re, w_2s_k1.re, w_2s_k0.re, w_2s_k0.re);
        let w_2s_k_ii = _mm256_set_pd(w_2s_k1.im, w_2s_k1.im, w_2s_k0.im, w_2s_k0.im);
        let w_2s_kps_rr = _mm256_set_pd(w_2s_kps1.re, w_2s_kps1.re, w_2s_kps0.re, w_2s_kps0.re);
        let w_2s_kps_ii = _mm256_set_pd(w_2s_kps1.im, w_2s_kps1.im, w_2s_kps0.im, w_2s_kps0.im);
        let w_s_k_rr = _mm256_set_pd(w_s_k1.re, w_s_k1.re, w_s_k0.re, w_s_k0.re);
        let w_s_k_ii = _mm256_set_pd(w_s_k1.im, w_s_k1.im, w_s_k0.im, w_s_k0.im);

        let t_ac_sum = _mm256_mul_pd(_mm256_add_pd(a, c), inv_sqrt2_v);
        let t_ac_diff = simd::complex_mul_avx2fma(w_2s_k_rr, w_2s_k_ii, _mm256_sub_pd(a, c));
        let t_bd_sum = _mm256_mul_pd(_mm256_add_pd(b, d), inv_sqrt2_v);
        let t_bd_diff = simd::complex_mul_avx2fma(w_2s_kps_rr, w_2s_kps_ii, _mm256_sub_pd(b, d));

        _mm256_storeu_pd(
            a_ptr,
            _mm256_mul_pd(_mm256_add_pd(t_ac_sum, t_bd_sum), inv_sqrt2_v),
        );
        _mm256_storeu_pd(
            b_ptr,
            simd::complex_mul_avx2fma(w_s_k_rr, w_s_k_ii, _mm256_sub_pd(t_ac_sum, t_bd_sum)),
        );
        _mm256_storeu_pd(
            c_ptr,
            _mm256_mul_pd(_mm256_add_pd(t_ac_diff, t_bd_diff), inv_sqrt2_v),
        );
        _mm256_storeu_pd(
            d_ptr,
            simd::complex_mul_avx2fma(w_s_k_rr, w_s_k_ii, _mm256_sub_pd(t_ac_diff, t_bd_diff)),
        );
    }
}

#[cfg(all(target_arch = "x86_64", test))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn radix4_butterfly_pair_slices_avx2fma(
    a_ptr: *mut f64,
    b_ptr: *mut f64,
    c_ptr: *mut f64,
    d_ptr: *mut f64,
    w_2s_k0: Complex64,
    w_2s_k1: Complex64,
    w_2s_kps0: Complex64,
    w_2s_kps1: Complex64,
    w_s_k0: Complex64,
    w_s_k1: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set_pd, _mm256_set1_pd,
            _mm256_storeu_pd, _mm256_sub_pd,
        };
        let inv_sqrt2_v = _mm256_set1_pd(std::f64::consts::FRAC_1_SQRT_2);

        let a = _mm256_loadu_pd(a_ptr);
        let b = _mm256_loadu_pd(b_ptr);
        let c = _mm256_loadu_pd(c_ptr);
        let d = _mm256_loadu_pd(d_ptr);

        let w_2s_k_rr = _mm256_set_pd(w_2s_k1.re, w_2s_k1.re, w_2s_k0.re, w_2s_k0.re);
        let w_2s_k_ii = _mm256_set_pd(w_2s_k1.im, w_2s_k1.im, w_2s_k0.im, w_2s_k0.im);
        let w_2s_kps_rr = _mm256_set_pd(w_2s_kps1.re, w_2s_kps1.re, w_2s_kps0.re, w_2s_kps0.re);
        let w_2s_kps_ii = _mm256_set_pd(w_2s_kps1.im, w_2s_kps1.im, w_2s_kps0.im, w_2s_kps0.im);
        let w_s_k_rr = _mm256_set_pd(w_s_k1.re, w_s_k1.re, w_s_k0.re, w_s_k0.re);
        let w_s_k_ii = _mm256_set_pd(w_s_k1.im, w_s_k1.im, w_s_k0.im, w_s_k0.im);

        let t_ac_sum = _mm256_mul_pd(_mm256_add_pd(a, c), inv_sqrt2_v);
        let t_ac_diff = simd::complex_mul_avx2fma(w_2s_k_rr, w_2s_k_ii, _mm256_sub_pd(a, c));
        let t_bd_sum = _mm256_mul_pd(_mm256_add_pd(b, d), inv_sqrt2_v);
        let t_bd_diff = simd::complex_mul_avx2fma(w_2s_kps_rr, w_2s_kps_ii, _mm256_sub_pd(b, d));

        _mm256_storeu_pd(
            a_ptr,
            _mm256_mul_pd(_mm256_add_pd(t_ac_sum, t_bd_sum), inv_sqrt2_v),
        );
        _mm256_storeu_pd(
            b_ptr,
            simd::complex_mul_avx2fma(w_s_k_rr, w_s_k_ii, _mm256_sub_pd(t_ac_sum, t_bd_sum)),
        );
        _mm256_storeu_pd(
            c_ptr,
            _mm256_mul_pd(_mm256_add_pd(t_ac_diff, t_bd_diff), inv_sqrt2_v),
        );
        _mm256_storeu_pd(
            d_ptr,
            simd::complex_mul_avx2fma(w_s_k_rr, w_s_k_ii, _mm256_sub_pd(t_ac_diff, t_bd_diff)),
        );
    }
}

/// Radix-4 DIF butterflies at two consecutive offsets with each twiddle pair packed in one
/// register, as [`twiddle_pair_avx2`] returns it.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn radix4_butterfly_pair_packed_avx2fma(
    a_ptr: *mut f64,
    b_ptr: *mut f64,
    c_ptr: *mut f64,
    d_ptr: *mut f64,
    w_2s_k: std::arch::x86_64::__m256d,
    w_2s_kps: std::arch::x86_64::__m256d,
    w_s_k: std::arch::x86_64::__m256d,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_movedup_pd, _mm256_mul_pd, _mm256_permute_pd,
            _mm256_set1_pd, _mm256_storeu_pd, _mm256_sub_pd,
        };
        macro_rules! parts {
            ($w:expr) => {
                (_mm256_movedup_pd($w), _mm256_permute_pd($w, 0b1111))
            };
        }
        let inv_sqrt2_v = _mm256_set1_pd(std::f64::consts::FRAC_1_SQRT_2);
        let (w_2s_k_rr, w_2s_k_ii) = parts!(w_2s_k);
        let (w_2s_kps_rr, w_2s_kps_ii) = parts!(w_2s_kps);
        let (w_s_k_rr, w_s_k_ii) = parts!(w_s_k);

        let a = _mm256_loadu_pd(a_ptr);
        let b = _mm256_loadu_pd(b_ptr);
        let c = _mm256_loadu_pd(c_ptr);
        let d = _mm256_loadu_pd(d_ptr);

        let t_ac_sum = _mm256_mul_pd(_mm256_add_pd(a, c), inv_sqrt2_v);
        let t_ac_diff = simd::complex_mul_avx2fma(w_2s_k_rr, w_2s_k_ii, _mm256_sub_pd(a, c));
        let t_bd_sum = _mm256_mul_pd(_mm256_add_pd(b, d), inv_sqrt2_v);
        let t_bd_diff = simd::complex_mul_avx2fma(w_2s_kps_rr, w_2s_kps_ii, _mm256_sub_pd(b, d));

        _mm256_storeu_pd(
            a_ptr,
            _mm256_mul_pd(_mm256_add_pd(t_ac_sum, t_bd_sum), inv_sqrt2_v),
        );
        _mm256_storeu_pd(
            b_ptr,
            simd::complex_mul_avx2fma(w_s_k_rr, w_s_k_ii, _mm256_sub_pd(t_ac_sum, t_bd_sum)),
        );
        _mm256_storeu_pd(
            c_ptr,
            _mm256_mul_pd(_mm256_add_pd(t_ac_diff, t_bd_diff), inv_sqrt2_v),
        );
        _mm256_storeu_pd(
            d_ptr,
            simd::complex_mul_avx2fma(w_s_k_rr, w_s_k_ii, _mm256_sub_pd(t_ac_diff, t_bd_diff)),
        );
    }
}

/// Pack twiddles `k * step` and `(k + 1) * step` into one 256-bit register.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn twiddle_pair_avx2(
    twiddles: *const Complex64,
    k: usize,
    step: usize,
) -> std::arch::x86_64::__m256d {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{_mm_loadu_pd, _mm256_loadu_pd, _mm256_set_m128d};
        if step == 1 {
            _mm256_loadu_pd(twiddles.add(k).cast())
        } else {
            _mm256_set_m128d(
                _mm_loadu_pd(twiddles.add((k + 1) * step).cast()),
                _mm_loadu_pd(twiddles.add(k * step).cast()),
            )
        }
    }
}

/// The pair loop of [`apply_radix4_groups`] over `groups` radix-4 groups of `s` offsets
/// starting at `base`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn radix4_groups_avx2fma(
    base: *mut f64,
    groups: usize,
    s: usize,
    k_base: usize,
    twiddles_scaled: &[Complex64],
    step_outer: usize,
    step_inner: usize,
) {
    assert!(s >= 2 && s.is_multiple_of(2));
    if groups == 0 {
        return;
    }
    let len = twiddles_scaled.len();
    assert!((k_base + 2 * s - 1) * step_outer < len && (k_base + s - 1) * step_inner < len);
    let twiddles = twiddles_scaled.as_ptr();
    // SAFETY: the amplitudes follow the enclosing unsafe fn's contract, and the last pair
    // reads twiddles `(k_base + 2s - 1) * step_outer` and `(k_base + s - 1) * step_inner`,
    // which the assertion bounds.
    unsafe {
        for g in 0..groups {
            let group_off = g * s * 4;
            for k_offset in (0..s).step_by(2) {
                let k = k_base + k_offset;
                let a = base.add((group_off + k_offset) * 2);
                radix4_butterfly_pair_packed_avx2fma(
                    a,
                    a.add(s * 2),
                    a.add(2 * s * 2),
                    a.add(3 * s * 2),
                    twiddle_pair_avx2(twiddles, k, step_outer),
                    twiddle_pair_avx2(twiddles, k + s, step_outer),
                    twiddle_pair_avx2(twiddles, k, step_inner),
                );
            }
        }
    }
}

/// The quad loop of [`apply_radix4_groups`] over `groups` radix-4 groups of `s` offsets
/// starting at `base`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn radix4_groups_avx512(
    base: *mut f64,
    groups: usize,
    s: usize,
    k_base: usize,
    twiddles_scaled: &[Complex64],
    step_outer: usize,
    step_inner: usize,
) {
    let len = twiddles_scaled.len();
    assert!((k_base + 2 * s - 1) * step_outer < len && (k_base + s - 1) * step_inner < len);
    let twiddles = twiddles_scaled.as_ptr();
    // SAFETY: the amplitudes follow the enclosing unsafe fn's contract, and the last quad
    // reads twiddles `(k_base + 2s - 1) * step_outer` and `(k_base + s - 1) * step_inner`,
    // which the assertion bounds. A checked per-element gather took 31% of a profiled
    // 22-qubit AVX-512 QFT.
    unsafe {
        for g in 0..groups {
            let group_off = g * s * 4;
            for k_offset in (0..s).step_by(4) {
                let k = k_base + k_offset;
                let a = base.add((group_off + k_offset) * 2);
                radix4_butterfly_quad_packed_avx512(
                    a,
                    a.add(s * 2),
                    a.add(2 * s * 2),
                    a.add(3 * s * 2),
                    twiddle_quad_avx512(twiddles, k, step_outer),
                    twiddle_quad_avx512(twiddles, k + s, step_outer),
                    twiddle_quad_avx512(twiddles, k, step_inner),
                );
            }
        }
    }
}

/// Pack twiddles `(k + j) * step` for `j < 4` one per lane pair. Step 1 is one load, step
/// 2 two loads and a shuffle, and a wider step four separate reads.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn twiddle_quad_avx512(
    twiddles: *const Complex64,
    k: usize,
    step: usize,
) -> std::arch::x86_64::__m512d {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{_mm512_loadu_pd, _mm512_setr_pd, _mm512_shuffle_f64x2};
        match step {
            1 => _mm512_loadu_pd(twiddles.add(k).cast()),
            2 => _mm512_shuffle_f64x2::<0xD8>(
                _mm512_loadu_pd(twiddles.add(2 * k).cast()),
                _mm512_loadu_pd(twiddles.add(2 * k + 3).cast()),
            ),
            _ => {
                let (w0, w1) = (*twiddles.add(k * step), *twiddles.add((k + 1) * step));
                let (w2, w3) = (*twiddles.add((k + 2) * step), *twiddles.add((k + 3) * step));
                _mm512_setr_pd(w0.re, w0.im, w1.re, w1.im, w2.re, w2.im, w3.re, w3.im)
            }
        }
    }
}

/// Radix-4 DIF butterflies at four consecutive offsets of four rows, `a` to `d`. Twiddle
/// `j` of each array belongs to offset `j`. Each lane runs the AVX2 pair kernel's
/// arithmetic, so the two agree bitwise.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
unsafe fn radix4_butterfly_quad_avx512(
    a_ptr: *mut f64,
    b_ptr: *mut f64,
    c_ptr: *mut f64,
    d_ptr: *mut f64,
    w_2s_k: &[Complex64; 4],
    w_2s_kps: &[Complex64; 4],
    w_s_k: &[Complex64; 4],
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::_mm512_loadu_pd;
        radix4_butterfly_quad_packed_avx512(
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
            _mm512_loadu_pd(w_2s_k.as_ptr() as *const f64),
            _mm512_loadu_pd(w_2s_kps.as_ptr() as *const f64),
            _mm512_loadu_pd(w_s_k.as_ptr() as *const f64),
        );
    }
}

/// [`radix4_butterfly_quad_avx512`] with each twiddle quad packed in one register, as
/// [`twiddle_quad_avx512`] returns it.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn radix4_butterfly_quad_packed_avx512(
    a_ptr: *mut f64,
    b_ptr: *mut f64,
    c_ptr: *mut f64,
    d_ptr: *mut f64,
    w_2s_k: std::arch::x86_64::__m512d,
    w_2s_kps: std::arch::x86_64::__m512d,
    w_s_k: std::arch::x86_64::__m512d,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm512_add_pd, _mm512_loadu_pd, _mm512_movedup_pd, _mm512_mul_pd, _mm512_permute_pd,
            _mm512_set1_pd, _mm512_storeu_pd, _mm512_sub_pd,
        };
        macro_rules! parts {
            ($w:expr) => {
                (_mm512_movedup_pd($w), _mm512_permute_pd::<0xFF>($w))
            };
        }
        let inv_sqrt2_v = _mm512_set1_pd(std::f64::consts::FRAC_1_SQRT_2);
        let (w_2s_k_rr, w_2s_k_ii) = parts!(w_2s_k);
        let (w_2s_kps_rr, w_2s_kps_ii) = parts!(w_2s_kps);
        let (w_s_k_rr, w_s_k_ii) = parts!(w_s_k);

        let a = _mm512_loadu_pd(a_ptr);
        let b = _mm512_loadu_pd(b_ptr);
        let c = _mm512_loadu_pd(c_ptr);
        let d = _mm512_loadu_pd(d_ptr);

        let t_ac_sum = _mm512_mul_pd(_mm512_add_pd(a, c), inv_sqrt2_v);
        let t_ac_diff = simd::complex_mul_avx512(w_2s_k_rr, w_2s_k_ii, _mm512_sub_pd(a, c));
        let t_bd_sum = _mm512_mul_pd(_mm512_add_pd(b, d), inv_sqrt2_v);
        let t_bd_diff = simd::complex_mul_avx512(w_2s_kps_rr, w_2s_kps_ii, _mm512_sub_pd(b, d));

        _mm512_storeu_pd(
            a_ptr,
            _mm512_mul_pd(_mm512_add_pd(t_ac_sum, t_bd_sum), inv_sqrt2_v),
        );
        _mm512_storeu_pd(
            b_ptr,
            simd::complex_mul_avx512(w_s_k_rr, w_s_k_ii, _mm512_sub_pd(t_ac_sum, t_bd_sum)),
        );
        _mm512_storeu_pd(
            c_ptr,
            _mm512_mul_pd(_mm512_add_pd(t_ac_diff, t_bd_diff), inv_sqrt2_v),
        );
        _mm512_storeu_pd(
            d_ptr,
            simd::complex_mul_avx512(w_s_k_rr, w_s_k_ii, _mm512_sub_pd(t_ac_diff, t_bd_diff)),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_twiddle_parts(
    w: Complex64,
) -> (
    std::arch::aarch64::float64x2_t,
    std::arch::aarch64::float64x2_t,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::{vcombine_f64, vdup_n_f64, vdupq_n_f64};
        (
            vdupq_n_f64(w.re),
            vcombine_f64(vdup_n_f64(-w.im), vdup_n_f64(w.im)),
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn radix4_butterfly_neon(
    base_ptr: *mut f64,
    inner_stride: usize,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::{
            vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64,
        };
        let s = inner_stride;
        let inv_sqrt2_v = vdupq_n_f64(std::f64::consts::FRAC_1_SQRT_2);

        let a_ptr = base_ptr;
        let b_ptr = base_ptr.add(s * 2);
        let c_ptr = base_ptr.add(2 * s * 2);
        let d_ptr = base_ptr.add(3 * s * 2);

        let a = vld1q_f64(a_ptr);
        let b = vld1q_f64(b_ptr);
        let c = vld1q_f64(c_ptr);
        let d = vld1q_f64(d_ptr);

        let (w_2s_k_rr, w_2s_k_ii_as) = neon_twiddle_parts(w_2s_k);
        let (w_2s_kps_rr, w_2s_kps_ii_as) = neon_twiddle_parts(w_2s_kps);
        let (w_s_k_rr, w_s_k_ii_as) = neon_twiddle_parts(w_s_k);

        let t_ac_sum = vmulq_f64(vaddq_f64(a, c), inv_sqrt2_v);
        let t_ac_diff = simd::complex_mul_neon(w_2s_k_rr, w_2s_k_ii_as, vsubq_f64(a, c));
        let t_bd_sum = vmulq_f64(vaddq_f64(b, d), inv_sqrt2_v);
        let t_bd_diff = simd::complex_mul_neon(w_2s_kps_rr, w_2s_kps_ii_as, vsubq_f64(b, d));

        vst1q_f64(a_ptr, vmulq_f64(vaddq_f64(t_ac_sum, t_bd_sum), inv_sqrt2_v));
        vst1q_f64(
            b_ptr,
            simd::complex_mul_neon(w_s_k_rr, w_s_k_ii_as, vsubq_f64(t_ac_sum, t_bd_sum)),
        );
        vst1q_f64(
            c_ptr,
            vmulq_f64(vaddq_f64(t_ac_diff, t_bd_diff), inv_sqrt2_v),
        );
        vst1q_f64(
            d_ptr,
            simd::complex_mul_neon(w_s_k_rr, w_s_k_ii_as, vsubq_f64(t_ac_diff, t_bd_diff)),
        );
    }
}

// Memory safety only: at stride one the butterfly touches the four amplitudes of an
// eight-double group and nothing outside it. Inputs are bounded to the unit interval
// so the solver is not asked to find an overflow or NaN in the arithmetic.
#[cfg(all(target_arch = "aarch64", kani))]
mod radix4_butterfly_neon_proofs {
    use num_complex::Complex64;

    fn any_unit() -> f64 {
        let x: f64 = kani::any();
        kani::assume(x.is_finite() && x.abs() <= 1.0);
        x
    }

    fn any_complex() -> Complex64 {
        Complex64::new(any_unit(), any_unit())
    }

    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(core::arch::aarch64::vfmaq_f64, crate::backend::neon_model::fmaq_f64)]
    #[kani::stub(core::arch::aarch64::vaddq_f64, crate::backend::neon_model::addq_f64)]
    #[kani::stub(core::arch::aarch64::vsubq_f64, crate::backend::neon_model::subq_f64)]
    #[kani::stub(core::arch::aarch64::vmulq_f64, crate::backend::neon_model::mulq_f64)]
    fn radix4_butterfly_neon_stays_in_bounds() {
        let mut group: [f64; 8] = std::array::from_fn(|_| any_unit());
        // SAFETY: the group holds four complex amplitudes at stride one, the shape the
        // kernel reads and writes; NEON is baseline on aarch64.
        unsafe {
            super::radix4_butterfly_neon(
                group.as_mut_ptr(),
                1,
                any_complex(),
                any_complex(),
                any_complex(),
            )
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
unsafe fn radix4_butterfly_slices_neon(
    a_ptr: *mut f64,
    b_ptr: *mut f64,
    c_ptr: *mut f64,
    d_ptr: *mut f64,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::{
            vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64,
        };
        let inv_sqrt2_v = vdupq_n_f64(std::f64::consts::FRAC_1_SQRT_2);

        let a = vld1q_f64(a_ptr);
        let b = vld1q_f64(b_ptr);
        let c = vld1q_f64(c_ptr);
        let d = vld1q_f64(d_ptr);

        let (w_2s_k_rr, w_2s_k_ii_as) = neon_twiddle_parts(w_2s_k);
        let (w_2s_kps_rr, w_2s_kps_ii_as) = neon_twiddle_parts(w_2s_kps);
        let (w_s_k_rr, w_s_k_ii_as) = neon_twiddle_parts(w_s_k);

        let t_ac_sum = vmulq_f64(vaddq_f64(a, c), inv_sqrt2_v);
        let t_ac_diff = simd::complex_mul_neon(w_2s_k_rr, w_2s_k_ii_as, vsubq_f64(a, c));
        let t_bd_sum = vmulq_f64(vaddq_f64(b, d), inv_sqrt2_v);
        let t_bd_diff = simd::complex_mul_neon(w_2s_kps_rr, w_2s_kps_ii_as, vsubq_f64(b, d));

        vst1q_f64(a_ptr, vmulq_f64(vaddq_f64(t_ac_sum, t_bd_sum), inv_sqrt2_v));
        vst1q_f64(
            b_ptr,
            simd::complex_mul_neon(w_s_k_rr, w_s_k_ii_as, vsubq_f64(t_ac_sum, t_bd_sum)),
        );
        vst1q_f64(
            c_ptr,
            vmulq_f64(vaddq_f64(t_ac_diff, t_bd_diff), inv_sqrt2_v),
        );
        vst1q_f64(
            d_ptr,
            simd::complex_mul_neon(w_s_k_rr, w_s_k_ii_as, vsubq_f64(t_ac_diff, t_bd_diff)),
        );
    }
}

#[inline]
fn apply_radix4_groups(
    group: &mut [Complex64],
    s: usize,
    k_base: usize,
    step_outer: usize,
    step_inner: usize,
    twiddles_scaled: &[Complex64],
) {
    debug_assert_eq!(group.len() % (s * 4), 0);
    let groups = group.len() / (s * 4);

    #[cfg(target_arch = "x86_64")]
    {
        let tier = simd::simd_tier();
        if tier == simd::SimdTier::Avx512 && s >= 4 {
            // SAFETY: AVX-512F detected. Quads of adjacent offsets stay inside the same
            // radix-4 group because s is a power of two of at least four. The safe
            // alternative is the scalar radix-4 loop below, which the AVX2 path already
            // replaces for the measured QFT speedup.
            unsafe {
                radix4_groups_avx512(
                    group.as_mut_ptr() as *mut f64,
                    groups,
                    s,
                    k_base,
                    twiddles_scaled,
                    step_outer,
                    step_inner,
                )
            };
            return;
        }
        if matches!(tier, simd::SimdTier::Avx512 | simd::SimdTier::Avx2Fma) && s >= 2 {
            // SAFETY: AVX2 and FMA detected. Adjacent pairs stay inside the same radix-4
            // group because s is a power of two of at least two. The safe alternative is
            // the scalar radix-4 loop below; this SIMD path is kept because the measured
            // QFT speedup depends on avoiding scalar Complex64 arithmetic and per-pair
            // bounds checks in the fused stage-pair hot loop.
            unsafe {
                radix4_groups_avx2fma(
                    group.as_mut_ptr() as *mut f64,
                    groups,
                    s,
                    k_base,
                    twiddles_scaled,
                    step_outer,
                    step_inner,
                );
            }
            return;
        }
        if simd::has_fma() {
            // SAFETY: has_fma() is runtime checked. The safe alternative is
            // the scalar radix-4 loop below; this SIMD path is kept because the
            // measured QFT speedup depends on avoiding scalar Complex64
            // arithmetic in the fused stage-pair hot loop.
            unsafe {
                let base_ptr = group.as_mut_ptr() as *mut f64;
                for g in 0..groups {
                    let group_off = g * s * 4;
                    for k_offset in 0..s {
                        let k = k_base + k_offset;
                        let w_2s_k = twiddles_scaled[k * step_outer];
                        let w_2s_kps = twiddles_scaled[(k + s) * step_outer];
                        let w_s_k = twiddles_scaled[k * step_inner];
                        let pair_base = base_ptr.add((group_off + k_offset) * 2);
                        radix4_butterfly_fma(pair_base, s, w_2s_k, w_2s_kps, w_s_k);
                    }
                }
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64. The loop maps each radix-4
        // butterfly to four in-bounds amplitudes inside `group`; the safe
        // alternative is the scalar fallback below. This SIMD counterpart is
        // required so Apple Silicon does not pay scalar Complex64 overhead in
        // the same measured QFT hot loop as x86.
        unsafe {
            let base_ptr = group.as_mut_ptr() as *mut f64;
            for g in 0..groups {
                let group_off = g * s * 4;
                for k_offset in 0..s {
                    let k = k_base + k_offset;
                    let w_2s_k = twiddles_scaled[k * step_outer];
                    let w_2s_kps = twiddles_scaled[(k + s) * step_outer];
                    let w_s_k = twiddles_scaled[k * step_inner];
                    let pair_base = base_ptr.add((group_off + k_offset) * 2);
                    radix4_butterfly_neon(pair_base, s, w_2s_k, w_2s_kps, w_s_k);
                }
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for g in 0..groups {
            let base = g * s * 4;
            for k_offset in 0..s {
                let k = k_base + k_offset;
                let w_2s_k = twiddles_scaled[k * step_outer];
                let w_2s_kps = twiddles_scaled[(k + s) * step_outer];
                let w_s_k = twiddles_scaled[k * step_inner];
                radix4_butterfly_scalar(group, base + k_offset, s, w_2s_k, w_2s_kps, w_s_k);
            }
        }
    }
}

/// Run two DIF stages at inner stride `s` within `slice`, reading [`qft_compact_twiddles`].
#[inline]
fn fft_stage_pair_in_slice(slice: &mut [Complex64], s: usize, compact: &[Complex64]) {
    apply_radix4_groups(slice, s, 0, 1, 2, &compact[s << 1..s << 2]);
}

/// Offsets per quarter that one task of [`fft_stage_pair_chunked`] covers.
#[cfg(feature = "parallel")]
const QFT_PAIR_CHUNK: usize = 1024;
/// Radix-4 groups that share one gathered twiddle chunk within a task.
#[cfg(feature = "parallel")]
const QFT_PAIR_GROUPS_PER_TASK: usize = 16;

/// Run a radix-4 stage pair as tasks of one twiddle chunk over a batch of groups.
///
/// Below the top pair a stride's twiddles sit `step_outer` and `step_inner` entries
/// apart in the whole-state table, up to a page each. A task gathers its chunk once
/// and reuses it across the batch, and splitting by chunk keeps every thread busy when
/// a stage has fewer groups than threads. `s` must be a multiple of the chunk.
#[cfg(feature = "parallel")]
fn fft_stage_pair_chunked(
    state: &mut [Complex64],
    s: usize,
    step_outer: usize,
    step_inner: usize,
    twiddles_scaled: &[Complex64],
) {
    const CHUNK: usize = QFT_PAIR_CHUNK;
    debug_assert_eq!(s % CHUNK, 0);
    let chunks = s / CHUNK;
    let groups = state.len() / (s << 2);
    let batch = QFT_PAIR_GROUPS_PER_TASK.min(groups);
    let batches = groups / batch;
    let ptr = SendPtr(state.as_mut_ptr());
    (0..chunks * batches).into_par_iter().for_each(move |task| {
        let k0 = task / batches * CHUNK;
        let first_group = task % batches * batch;
        // SAFETY: a task owns offsets `[k0, k0 + CHUNK)` of the four quarters of each
        // group in its batch. Tasks differ in chunk or batch, so no two share a run.
        unsafe {
            fft_stage_pair_chunk(
                ptr.as_complex_ptr(),
                s,
                k0,
                first_group..first_group + batch,
                step_outer,
                step_inner,
                twiddles_scaled,
            );
        }
    });
}

/// Gather the twiddles for offsets `[k0, k0 + QFT_PAIR_CHUNK)` and apply them to
/// `groups`. Out of line, so its 48 KiB of twiddles live only while a task runs
/// instead of in every frame of Rayon's recursive split.
///
/// # Safety
///
/// `state` must be valid for every group in `groups`, and no other thread may access
/// those offsets of any quarter of these groups during the call.
#[cfg(feature = "parallel")]
#[inline(never)]
unsafe fn fft_stage_pair_chunk(
    state: *mut Complex64,
    s: usize,
    k0: usize,
    groups: std::ops::Range<usize>,
    step_outer: usize,
    step_inner: usize,
    twiddles_scaled: &[Complex64],
) {
    const CHUNK: usize = QFT_PAIR_CHUNK;
    let w_outer: [Complex64; CHUNK] =
        std::array::from_fn(|j| twiddles_scaled[(k0 + j) * step_outer]);
    let w_outer_s: [Complex64; CHUNK] =
        std::array::from_fn(|j| twiddles_scaled[(k0 + j + s) * step_outer]);
    let w_inner: [Complex64; CHUNK] =
        std::array::from_fn(|j| twiddles_scaled[(k0 + j) * step_inner]);
    for g in groups {
        let base = g * (s << 2) + k0;
        // SAFETY: same contract as the enclosing unsafe fn.
        let (qa, qb, qc, qd) = unsafe {
            (
                std::slice::from_raw_parts_mut(state.add(base), CHUNK),
                std::slice::from_raw_parts_mut(state.add(base + s), CHUNK),
                std::slice::from_raw_parts_mut(state.add(base + 2 * s), CHUNK),
                std::slice::from_raw_parts_mut(state.add(base + 3 * s), CHUNK),
            )
        };
        radix4_runs(qa, qb, qc, qd, &w_outer, &w_outer_s, &w_inner);
    }
}

#[cfg(all(target_arch = "x86_64", feature = "parallel"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn radix4_runs_avx2fma(
    pa: *mut f64,
    pb: *mut f64,
    pc: *mut f64,
    pd: *mut f64,
    start: usize,
    end: usize,
    w_outer: *const Complex64,
    w_outer_s: *const Complex64,
    w_inner: *const Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::_mm256_loadu_pd;
        for j in (start..end).step_by(2) {
            radix4_butterfly_pair_packed_avx2fma(
                pa.add(j * 2),
                pb.add(j * 2),
                pc.add(j * 2),
                pd.add(j * 2),
                _mm256_loadu_pd(w_outer.add(j).cast()),
                _mm256_loadu_pd(w_outer_s.add(j).cast()),
                _mm256_loadu_pd(w_inner.add(j).cast()),
            );
        }
    }
}

/// Apply radix-4 butterflies at matching offsets of four quarter slices, with offset
/// `j` taking index `j` of each twiddle slice.
#[cfg(feature = "parallel")]
#[inline(always)]
fn radix4_runs(
    qa: &mut [Complex64],
    qb: &mut [Complex64],
    qc: &mut [Complex64],
    qd: &mut [Complex64],
    w_outer: &[Complex64],
    w_outer_s: &[Complex64],
    w_inner: &[Complex64],
) {
    let n = qa.len();

    #[cfg(target_arch = "x86_64")]
    let tier = simd::simd_tier();
    #[cfg(target_arch = "x86_64")]
    if matches!(tier, simd::SimdTier::Avx512 | simd::SimdTier::Avx2Fma) && n >= 2 {
        assert!(
            qb.len() >= n
                && qc.len() >= n
                && qd.len() >= n
                && w_outer.len() >= n
                && w_outer_s.len() >= n
                && w_inner.len() >= n
        );
        let mut j = 0usize;
        // SAFETY: AVX2, and AVX-512F for the quad loop, detected. Each call reads and
        // writes four or two adjacent amplitudes and twiddles at offset `j` of disjoint
        // slices of length at least `n`, bounded by the assertion above.
        unsafe {
            let (pa, pb, pc, pd) = (
                qa.as_mut_ptr() as *mut f64,
                qb.as_mut_ptr() as *mut f64,
                qc.as_mut_ptr() as *mut f64,
                qd.as_mut_ptr() as *mut f64,
            );
            if tier == simd::SimdTier::Avx512 {
                while j + 4 <= n {
                    radix4_butterfly_quad_avx512(
                        pa.add(j * 2),
                        pb.add(j * 2),
                        pc.add(j * 2),
                        pd.add(j * 2),
                        w_outer[j..].first_chunk().expect("four twiddles"),
                        w_outer_s[j..].first_chunk().expect("four twiddles"),
                        w_inner[j..].first_chunk().expect("four twiddles"),
                    );
                    j += 4;
                }
            }
            let end_pair = n & !1;
            if j < end_pair {
                radix4_runs_avx2fma(
                    pa,
                    pb,
                    pc,
                    pd,
                    j,
                    end_pair,
                    w_outer.as_ptr(),
                    w_outer_s.as_ptr(),
                    w_inner.as_ptr(),
                );
                j = end_pair;
            }
        }
        if j < n {
            radix4_butterfly_quartet_scalar(
                qa,
                qb,
                qc,
                qd,
                j,
                w_outer[j],
                w_outer_s[j],
                w_inner[j],
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64. Each call reads and writes the amplitude at
    // offset `j < n` of four disjoint slices of length `n`.
    unsafe {
        let (pa, pb, pc, pd) = (
            qa.as_mut_ptr() as *mut f64,
            qb.as_mut_ptr() as *mut f64,
            qc.as_mut_ptr() as *mut f64,
            qd.as_mut_ptr() as *mut f64,
        );
        for j in 0..n {
            radix4_butterfly_slices_neon(
                pa.add(j * 2),
                pb.add(j * 2),
                pc.add(j * 2),
                pd.add(j * 2),
                w_outer[j],
                w_outer_s[j],
                w_inner[j],
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    for j in 0..n {
        radix4_butterfly_quartet_scalar(qa, qb, qc, qd, j, w_outer[j], w_outer_s[j], w_inner[j]);
    }
}

/// Run two DIF FFT stages as one radix-4 pass over every `2 * twiddles_scaled.len()`
/// amplitude chunk of `state`.
///
/// `inner_stride = 2^(outer_stage - 1)`. Falls back to a sequential loop when
/// the `parallel` feature is off.
#[inline]
fn fft_stage_pair_par(state: &mut [Complex64], inner_stride: usize, twiddles_scaled: &[Complex64]) {
    let s = inner_stride;
    let fft_len = 2 * twiddles_scaled.len();
    let step_outer = fft_len / (s << 2);
    let step_inner = fft_len / (s << 1);

    #[cfg(feature = "parallel")]
    fft_stage_pair_chunked(state, s, step_outer, step_inner, twiddles_scaled);

    #[cfg(not(feature = "parallel"))]
    apply_radix4_groups(state, s, 0, step_outer, step_inner, twiddles_scaled);
}

/// Run the DIF stages above `tile_bits` of every `2^n` chunk as gathered passes of at
/// most [`qft_gather_max_bits`] stages each, split evenly.
fn qft_gather_high_stages(state: &mut [Complex64], n: usize, tile_bits: usize, inverse: bool) {
    let passes = (n - tile_bits).div_ceil(qft_gather_max_bits());
    let mut top = n;
    for left in (1..=passes).rev() {
        let bits = (top - tile_bits).div_ceil(left);
        qft_gather_pass(state, QftGatherPass::new(top - bits, bits, inverse));
        top -= bits;
    }
}

thread_local! {
    static QFT_GATHER_BUF: std::cell::RefCell<Vec<Complex64>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn with_qft_gather_buf<R>(len: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    QFT_GATHER_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < len {
            buf.resize(len, Complex64::new(0.0, 0.0));
        }
        f(&mut buf[..len])
    })
}

fn qft_gather_pass(state: &mut [Complex64], pass: QftGatherPass) {
    #[cfg(test)]
    crate::backend::statevector::sweeps::tick();
    let groups = state.len() >> (pass.bits + pass.row_len.ilog2() as usize);
    let len = pass.row_len << pass.bits;

    #[cfg(feature = "parallel")]
    if state.len() >= 1usize << kernel_parallel_qubits() {
        let ptr = SendPtr(state.as_mut_ptr());
        (0..groups).into_par_iter().for_each(move |g| {
            with_qft_gather_buf(len, |buf| {
                // SAFETY: a group owns `row_len` offsets of each of its rows, and the
                // groups tile the state, so no two tasks share an amplitude.
                unsafe { pass.run_group(ptr.as_complex_ptr(), g, buf) }
            });
        });
        return;
    }

    let ptr = state.as_mut_ptr();
    with_qft_gather_buf(len, |buf| {
        for g in 0..groups {
            // SAFETY: `ptr` spans the whole state, borrowed mutably for this loop alone.
            unsafe { pass.run_group(ptr, g, buf) }
        }
    });
}

/// One gathered pass of the QFT's DIF network: stages `lo..lo + bits` of every
/// `2^(lo + bits)` chunk, a group of `2^bits` rows at a time.
///
/// A group copies, for `row_len` consecutive offsets `L` below `lo`, the amplitudes at
/// every value of the pass bits into a row each. The stage at local stride `2^u`
/// twiddles position `k 2^lo + L` by `exp(2 pi i (k 2^lo + L) / 2^(lo + u + 1))`, the
/// local DIF twiddle for `k` times a factor in `L` alone. Both inputs of every later
/// butterfly carry the same factor, so the rows run a plain `2^bits`-point DIF on
/// [`qft_compact_twiddles`] and the factors collect into one phase
/// `exp(2 pi i L rev(r) / 2^(lo + bits))` on output row `r`, applied on the way back.
#[derive(Clone, Copy)]
struct QftGatherPass {
    lo: usize,
    bits: usize,
    row_len: usize,
    compact: &'static [Complex64],
    phases: &'static QftGatherPhases,
}

impl QftGatherPass {
    fn new(lo: usize, bits: usize, inverse: bool) -> Self {
        Self {
            lo,
            bits,
            row_len: (qft_gather_len() >> bits).min(1 << lo),
            compact: qft_compact_twiddles(inverse),
            phases: qft_gather_phases(lo + bits, inverse),
        }
    }

    /// Gather group `g` into `buf`, run its stages, and scatter it back phased.
    ///
    /// # Safety
    ///
    /// `state` must be valid for every amplitude of group `g`, and no other thread may
    /// access them during the call.
    #[inline]
    unsafe fn run_group(&self, state: *mut Complex64, g: usize, buf: &mut [Complex64]) {
        let blocks_log2 = self.lo - self.row_len.ilog2() as usize;
        let col0 = (g & ((1 << blocks_log2) - 1)) * self.row_len;
        let base = ((g >> blocks_log2) << (self.lo + self.bits)) + col0;
        for (r, row) in buf.chunks_exact_mut(self.row_len).enumerate() {
            // SAFETY: same contract as the enclosing unsafe fn.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    state.add(base + (r << self.lo)),
                    row.as_mut_ptr(),
                    self.row_len,
                );
            }
        }
        qft_dif_rows(buf, self.row_len, self.bits, self.compact);
        for (r, row) in buf.chunks_exact(self.row_len).enumerate() {
            let rev = reverse_low_bits(r, self.bits);
            // SAFETY: same contract as the enclosing unsafe fn.
            let out = unsafe {
                std::slice::from_raw_parts_mut(state.add(base + (r << self.lo)), self.row_len)
            };
            qft_scatter_row(
                out,
                row,
                self.phases.phase(col0 * rev),
                &self.phases.low,
                rev,
            );
        }
    }
}

/// Run a `2^bits`-point DIF down the rows of `buf`, every column of `row_len` an
/// independent transform, in the stage order of the in-tile stages.
fn qft_dif_rows(buf: &mut [Complex64], row_len: usize, bits: usize, compact: &[Complex64]) {
    let mut top = bits;
    if bits % 2 == 1 {
        top -= 1;
        let s = 1usize << top;
        for block in buf.chunks_exact_mut(2 * s * row_len) {
            let (a, b) = block.split_at_mut(s * row_len);
            for (k, (ra, rb)) in a
                .chunks_exact_mut(row_len)
                .zip(b.chunks_exact_mut(row_len))
                .enumerate()
            {
                radix2_rows(ra, rb, compact[s + k]);
            }
        }
    }
    while top > 0 {
        let s = 1usize << (top - 2);
        let quarter = s * row_len;
        for block in buf.chunks_exact_mut(4 * quarter) {
            let (a, rest) = block.split_at_mut(quarter);
            let (b, rest) = rest.split_at_mut(quarter);
            let (c, d) = rest.split_at_mut(quarter);
            let rows = a
                .chunks_exact_mut(row_len)
                .zip(b.chunks_exact_mut(row_len))
                .zip(c.chunks_exact_mut(row_len))
                .zip(d.chunks_exact_mut(row_len));
            for (k, (((ra, rb), rc), rd)) in rows.enumerate() {
                radix4_rows(
                    ra,
                    rb,
                    rc,
                    rd,
                    compact[2 * s + k],
                    compact[3 * s + k],
                    compact[s + k],
                );
            }
        }
        top -= 2;
    }
}

/// Radix-2 DIF butterflies between two rows of one even length sharing twiddle `w`.
#[inline(always)]
fn radix2_rows(lo: &mut [Complex64], hi: &mut [Complex64], w: Complex64) {
    debug_assert_eq!(lo.len(), hi.len());
    #[cfg(target_arch = "x86_64")]
    match simd::simd_tier() {
        simd::SimdTier::Avx512 => {
            // SAFETY: AVX-512F detected. The rows share one even length.
            unsafe { radix2_rows_avx512(lo, hi, w) };
            return;
        }
        simd::SimdTier::Avx2Fma => {
            // SAFETY: AVX2 detected. The rows share one even length.
            unsafe { radix2_rows_avx2fma(lo, hi, w) };
            return;
        }
        simd::SimdTier::Fma | simd::SimdTier::Sse2 => {}
    }
    for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
        (*a, *b) = radix2_butterfly_values(*a, *b, w);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn radix2_rows_avx2fma(lo: &mut [Complex64], hi: &mut [Complex64], w: Complex64) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
            _mm256_sub_pd,
        };
        let inv_sqrt2_v = _mm256_set1_pd(std::f64::consts::FRAC_1_SQRT_2);
        let (w_rr, w_ii) = (_mm256_set1_pd(w.re), _mm256_set1_pd(w.im));
        let (pl, ph) = (lo.as_mut_ptr() as *mut f64, hi.as_mut_ptr() as *mut f64);
        for off in (0..2 * lo.len()).step_by(4) {
            let a = _mm256_loadu_pd(pl.add(off));
            let b = _mm256_loadu_pd(ph.add(off));
            _mm256_storeu_pd(pl.add(off), _mm256_mul_pd(_mm256_add_pd(a, b), inv_sqrt2_v));
            _mm256_storeu_pd(
                ph.add(off),
                simd::complex_mul_avx2fma(w_rr, w_ii, _mm256_sub_pd(a, b)),
            );
        }
    }
}

/// Four amplitudes per step; a row length that leaves two runs one masked step.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn radix2_rows_avx512(lo: &mut [Complex64], hi: &mut [Complex64], w: Complex64) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use simd::{load_pd, store_pd};
        use std::arch::x86_64::{
            __mmask8, _mm512_add_pd, _mm512_mul_pd, _mm512_set1_pd, _mm512_sub_pd,
        };
        let inv_sqrt2_v = _mm512_set1_pd(std::f64::consts::FRAC_1_SQRT_2);
        let (w_rr, w_ii) = (_mm512_set1_pd(w.re), _mm512_set1_pd(w.im));
        let (pl, ph) = (lo.as_mut_ptr() as *mut f64, hi.as_mut_ptr() as *mut f64);
        simd::for_each_quad!(lo.len(), |off, k| {
            let a = load_pd(pl.add(off), k);
            let b = load_pd(ph.add(off), k);
            store_pd(
                pl.add(off),
                k,
                _mm512_mul_pd(_mm512_add_pd(a, b), inv_sqrt2_v),
            );
            store_pd(
                ph.add(off),
                k,
                simd::complex_mul_avx512(w_rr, w_ii, _mm512_sub_pd(a, b)),
            );
        });
    }
}

/// Radix-4 DIF butterflies across four rows of one even length sharing one twiddle
/// triple, in the argument order of [`radix4_butterfly_values`].
#[inline(always)]
fn radix4_rows(
    qa: &mut [Complex64],
    qb: &mut [Complex64],
    qc: &mut [Complex64],
    qd: &mut [Complex64],
    w_outer: Complex64,
    w_outer_s: Complex64,
    w_inner: Complex64,
) {
    let n = qa.len();
    debug_assert!(qb.len() == n && qc.len() == n && qd.len() == n);

    #[cfg(target_arch = "x86_64")]
    match simd::simd_tier() {
        simd::SimdTier::Avx512 => {
            // SAFETY: AVX-512F detected. The four rows share one even length.
            unsafe { radix4_rows_avx512(qa, qb, qc, qd, w_outer, w_outer_s, w_inner) };
            return;
        }
        simd::SimdTier::Avx2Fma => {
            // SAFETY: AVX2 detected. The four rows share one even length.
            unsafe { radix4_rows_avx2fma(qa, qb, qc, qd, w_outer, w_outer_s, w_inner) };
            return;
        }
        simd::SimdTier::Fma | simd::SimdTier::Sse2 => {}
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64. Each call reads and writes the amplitude at
    // offset `j < n` of four disjoint rows of length `n`.
    unsafe {
        let (pa, pb, pc, pd) = (
            qa.as_mut_ptr() as *mut f64,
            qb.as_mut_ptr() as *mut f64,
            qc.as_mut_ptr() as *mut f64,
            qd.as_mut_ptr() as *mut f64,
        );
        for j in 0..n {
            radix4_butterfly_slices_neon(
                pa.add(j * 2),
                pb.add(j * 2),
                pc.add(j * 2),
                pd.add(j * 2),
                w_outer,
                w_outer_s,
                w_inner,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    for j in 0..n {
        radix4_butterfly_quartet_scalar(qa, qb, qc, qd, j, w_outer, w_outer_s, w_inner);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn radix4_rows_avx2fma(
    qa: &mut [Complex64],
    qb: &mut [Complex64],
    qc: &mut [Complex64],
    qd: &mut [Complex64],
    w_outer: Complex64,
    w_outer_s: Complex64,
    w_inner: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
            _mm256_sub_pd,
        };
        let inv_sqrt2_v = _mm256_set1_pd(std::f64::consts::FRAC_1_SQRT_2);
        let (wo_rr, wo_ii) = (_mm256_set1_pd(w_outer.re), _mm256_set1_pd(w_outer.im));
        let (ws_rr, ws_ii) = (_mm256_set1_pd(w_outer_s.re), _mm256_set1_pd(w_outer_s.im));
        let (wi_rr, wi_ii) = (_mm256_set1_pd(w_inner.re), _mm256_set1_pd(w_inner.im));
        let (pa, pb, pc, pd) = (
            qa.as_mut_ptr() as *mut f64,
            qb.as_mut_ptr() as *mut f64,
            qc.as_mut_ptr() as *mut f64,
            qd.as_mut_ptr() as *mut f64,
        );
        for off in (0..2 * qa.len()).step_by(4) {
            let a = _mm256_loadu_pd(pa.add(off));
            let b = _mm256_loadu_pd(pb.add(off));
            let c = _mm256_loadu_pd(pc.add(off));
            let d = _mm256_loadu_pd(pd.add(off));

            let t_ac_sum = _mm256_mul_pd(_mm256_add_pd(a, c), inv_sqrt2_v);
            let t_ac_diff = simd::complex_mul_avx2fma(wo_rr, wo_ii, _mm256_sub_pd(a, c));
            let t_bd_sum = _mm256_mul_pd(_mm256_add_pd(b, d), inv_sqrt2_v);
            let t_bd_diff = simd::complex_mul_avx2fma(ws_rr, ws_ii, _mm256_sub_pd(b, d));

            _mm256_storeu_pd(
                pa.add(off),
                _mm256_mul_pd(_mm256_add_pd(t_ac_sum, t_bd_sum), inv_sqrt2_v),
            );
            _mm256_storeu_pd(
                pb.add(off),
                simd::complex_mul_avx2fma(wi_rr, wi_ii, _mm256_sub_pd(t_ac_sum, t_bd_sum)),
            );
            _mm256_storeu_pd(
                pc.add(off),
                _mm256_mul_pd(_mm256_add_pd(t_ac_diff, t_bd_diff), inv_sqrt2_v),
            );
            _mm256_storeu_pd(
                pd.add(off),
                simd::complex_mul_avx2fma(wi_rr, wi_ii, _mm256_sub_pd(t_ac_diff, t_bd_diff)),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn radix4_rows_avx512(
    qa: &mut [Complex64],
    qb: &mut [Complex64],
    qc: &mut [Complex64],
    qd: &mut [Complex64],
    w_outer: Complex64,
    w_outer_s: Complex64,
    w_inner: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use simd::{load_pd, store_pd};
        use std::arch::x86_64::{
            __mmask8, _mm512_add_pd, _mm512_mul_pd, _mm512_set1_pd, _mm512_sub_pd,
        };
        let inv_sqrt2_v = _mm512_set1_pd(std::f64::consts::FRAC_1_SQRT_2);
        let (wo_rr, wo_ii) = (_mm512_set1_pd(w_outer.re), _mm512_set1_pd(w_outer.im));
        let (ws_rr, ws_ii) = (_mm512_set1_pd(w_outer_s.re), _mm512_set1_pd(w_outer_s.im));
        let (wi_rr, wi_ii) = (_mm512_set1_pd(w_inner.re), _mm512_set1_pd(w_inner.im));
        let (pa, pb, pc, pd) = (
            qa.as_mut_ptr() as *mut f64,
            qb.as_mut_ptr() as *mut f64,
            qc.as_mut_ptr() as *mut f64,
            qd.as_mut_ptr() as *mut f64,
        );
        simd::for_each_quad!(qa.len(), |off, k| {
            let a = load_pd(pa.add(off), k);
            let b = load_pd(pb.add(off), k);
            let c = load_pd(pc.add(off), k);
            let d = load_pd(pd.add(off), k);

            let t_ac_sum = _mm512_mul_pd(_mm512_add_pd(a, c), inv_sqrt2_v);
            let t_ac_diff = simd::complex_mul_avx512(wo_rr, wo_ii, _mm512_sub_pd(a, c));
            let t_bd_sum = _mm512_mul_pd(_mm512_add_pd(b, d), inv_sqrt2_v);
            let t_bd_diff = simd::complex_mul_avx512(ws_rr, ws_ii, _mm512_sub_pd(b, d));

            store_pd(
                pa.add(off),
                k,
                _mm512_mul_pd(_mm512_add_pd(t_ac_sum, t_bd_sum), inv_sqrt2_v),
            );
            store_pd(
                pb.add(off),
                k,
                simd::complex_mul_avx512(wi_rr, wi_ii, _mm512_sub_pd(t_ac_sum, t_bd_sum)),
            );
            store_pd(
                pc.add(off),
                k,
                _mm512_mul_pd(_mm512_add_pd(t_ac_diff, t_bd_diff), inv_sqrt2_v),
            );
            store_pd(
                pd.add(off),
                k,
                simd::complex_mul_avx512(wi_rr, wi_ii, _mm512_sub_pd(t_ac_diff, t_bd_diff)),
            );
        });
    }
}

/// Write `row` into `out` with amplitude `j` times `anchor * low[j * step]`.
#[inline(always)]
fn qft_scatter_row(
    out: &mut [Complex64],
    row: &[Complex64],
    anchor: Complex64,
    low: &[Complex64],
    step: usize,
) {
    debug_assert_eq!(out.len(), row.len());
    #[cfg(target_arch = "x86_64")]
    match simd::simd_tier() {
        simd::SimdTier::Avx512 => {
            // SAFETY: AVX-512F detected. `out` and `row` share one even length.
            unsafe { qft_scatter_row_avx512(out, row, anchor, low, step) };
            return;
        }
        simd::SimdTier::Avx2Fma => {
            // SAFETY: AVX2 detected. `out` and `row` share one even length.
            unsafe { qft_scatter_row_avx2fma(out, row, anchor, low, step) };
            return;
        }
        simd::SimdTier::Fma | simd::SimdTier::Sse2 => {}
    }
    for (j, (o, &a)) in out.iter_mut().zip(row).enumerate() {
        *o = a * (anchor * low[j * step]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn qft_scatter_row_avx2fma(
    out: &mut [Complex64],
    row: &[Complex64],
    anchor: Complex64,
    low: &[Complex64],
    step: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm256_loadu_pd, _mm256_movedup_pd, _mm256_permute_pd, _mm256_set_pd, _mm256_set1_pd,
            _mm256_storeu_pd,
        };
        let (a_rr, a_ii) = (_mm256_set1_pd(anchor.re), _mm256_set1_pd(anchor.im));
        let (src, dst) = (row.as_ptr() as *const f64, out.as_mut_ptr() as *mut f64);
        let mut x = 0;
        for off in (0..2 * row.len()).step_by(4) {
            let (p0, p1) = (low[x], low[x + step]);
            let w =
                simd::complex_mul_avx2fma(a_rr, a_ii, _mm256_set_pd(p1.im, p1.re, p0.im, p0.re));
            let v = _mm256_loadu_pd(src.add(off));
            _mm256_storeu_pd(
                dst.add(off),
                simd::complex_mul_avx2fma(_mm256_movedup_pd(w), _mm256_permute_pd(w, 0b1111), v),
            );
            x += 2 * step;
        }
    }
}

/// Four amplitudes per step. A two-amplitude tail reads only the two phases it needs
/// and stores under a lane mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn qft_scatter_row_avx512(
    out: &mut [Complex64],
    row: &[Complex64],
    anchor: Complex64,
    low: &[Complex64],
    step: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use simd::{load_pd, store_pd};
        use std::arch::x86_64::{
            __mmask8, _mm512_movedup_pd, _mm512_permute_pd, _mm512_set_pd, _mm512_set1_pd,
        };
        let (a_rr, a_ii) = (_mm512_set1_pd(anchor.re), _mm512_set1_pd(anchor.im));
        let (src, dst) = (row.as_ptr() as *const f64, out.as_mut_ptr() as *mut f64);
        let zero = Complex64::new(0.0, 0.0);
        simd::for_each_quad!(row.len(), |off, k| {
            let x = off / 2 * step;
            let (p0, p1) = (low[x], low[x + step]);
            let (p2, p3) = if k == 0xFF {
                (low[x + 2 * step], low[x + 3 * step])
            } else {
                (zero, zero)
            };
            let w = simd::complex_mul_avx512(
                a_rr,
                a_ii,
                _mm512_set_pd(p3.im, p3.re, p2.im, p2.re, p1.im, p1.re, p0.im, p0.re),
            );
            let v = load_pd(src.add(off), k);
            store_pd(
                dst.add(off),
                k,
                simd::complex_mul_avx512(_mm512_movedup_pd(w), _mm512_permute_pd::<0xFF>(w), v),
            );
        });
    }
}

impl StatevectorBackend {
    /// Apply a QFT on qubits `0..num`, a tiled DIF FFT over every `2^num` amplitude
    /// chunk, in the [`Gate::QftBlock`] form the flags name.
    ///
    /// With `D` the DIF network and `P` the chunk bit reversal, the forward swap form is
    /// `P D` and the forward no-swap form `D`. Conjugate twiddles turn `D` into
    /// `conj(D)`, giving `P conj(D)` for the inverse swap form and `P conj(D) P` for the
    /// inverse no-swap form. Mirroring the labels conjugates each form by `P`.
    pub(in crate::backend::statevector) fn apply_qft_block(
        &mut self,
        num: usize,
        inverse: bool,
        swaps: bool,
        big_endian: bool,
    ) {
        let gather = num >= QFT_GATHER_MIN_QUBITS;
        self.apply_qft_block_with(num, inverse, swaps, big_endian, gather);
    }

    /// [`Self::apply_qft_block`] with the high-stage route chosen: `gather` runs the
    /// stages above the tile as gathered passes when at least [`QFT_GATHER_MIN_STAGES`]
    /// of them sit there, and otherwise they run as full-state radix-4 pairs.
    fn apply_qft_block_with(
        &mut self,
        num: usize,
        inverse: bool,
        swaps: bool,
        big_endian: bool,
        gather: bool,
    ) {
        assert!(num >= 1 && num <= self.num_qubits);
        let reverse_input = big_endian != (inverse && !swaps);
        let reverse_output = big_endian != (swaps || inverse);

        let n = num;

        if reverse_input {
            apply_bit_reverse_chunks(&mut self.state, n);
        }

        // Cache-tiled DIF FFT:
        //   - High-stride stages run as gathered passes or full-state passes.
        //   - Low-stride stages run together inside each cache-sized tile.
        let widest = qft_widest_tile_bits_for(
            n,
            qft_max_tile_bits(),
            qft_min_tiles_log2(n, self.num_qubits),
        );
        let tile_bits = if gather && n - widest >= QFT_GATHER_MIN_STAGES {
            qft_gather_high_stages(&mut self.state, n, widest, inverse);
            widest
        } else {
            let tile_bits = qft_tile_bits(n, self.num_qubits);
            self.qft_high_stage_pairs(n, tile_bits, inverse);
            tile_bits
        };

        // Phase 2: low-stride stages on cache-resident tiles.
        let tile_size = 1usize << tile_bits;

        let compact = qft_compact_twiddles(inverse);
        let apply_low_stages_in_tile = |tile: &mut [Complex64]| {
            let mut stage_top = tile_bits;
            let mut stages_left = tile_bits;
            if stages_left % 2 == 1 {
                stage_top -= 1;
                let stride = 1usize << stage_top;
                run_radix2_stage_seq(tile, stride, &compact[stride..stride << 1], 1);
                stages_left -= 1;
            }
            while stages_left > 0 {
                let inner_stride = 1usize << (stage_top - 2);
                fft_stage_pair_in_slice(tile, inner_stride, compact);
                stage_top -= 2;
                stages_left -= 2;
            }
        };

        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        #[cfg(feature = "parallel")]
        let low_done_parallel =
            if self.num_qubits >= kernel_parallel_qubits() && self.state.len() / tile_size >= 4 {
                self.state
                    .par_chunks_mut(tile_size)
                    .for_each(apply_low_stages_in_tile);
                true
            } else {
                false
            };
        #[cfg(not(feature = "parallel"))]
        let low_done_parallel = false;

        if !low_done_parallel {
            for tile in self.state.chunks_mut(tile_size) {
                apply_low_stages_in_tile(tile);
            }
        }

        if reverse_output {
            apply_bit_reverse_chunks(&mut self.state, n);
        }
    }

    /// Run the DIF stages above `tile_bits` as full-state passes: radix-4 pairs, after
    /// one radix-2 stage when the count is odd.
    fn qft_high_stage_pairs(&mut self, n: usize, tile_bits: usize, inverse: bool) {
        let fft_len = 1usize << n;
        let twiddles_scaled = qft_twiddles_scaled(n, inverse);
        let twiddles_scaled = twiddles_scaled.as_ref();

        let high_stride_count = n - tile_bits;
        let mut stage_top = n;
        let mut stages_left = high_stride_count;

        let run_radix2_stage = |state: &mut Vec<Complex64>, stage: usize| {
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            let stride = 1usize << stage;
            let block_size = stride << 1;
            let twiddle_step = fft_len / block_size;

            #[cfg(feature = "parallel")]
            if state.len() >= 1usize << kernel_parallel_qubits() {
                fft_stage_par(state, stride, block_size, twiddle_step, twiddles_scaled);
                return;
            }

            run_radix2_stage_seq(state, stride, twiddles_scaled, twiddle_step);
        };

        if stages_left % 2 == 1 {
            stage_top -= 1;
            run_radix2_stage(&mut self.state, stage_top);
            stages_left -= 1;
        }

        while stages_left > 0 {
            // Fuse pair (stage_top - 1, stage_top - 2): outer stride
            // = 2^(stage_top - 1), inner stride = 2^(stage_top - 2).
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            let inner_stride = 1usize << (stage_top - 2);
            fft_stage_pair_par(&mut self.state, inner_stride, twiddles_scaled);
            stage_top -= 2;
            stages_left -= 2;
        }
    }

    #[inline]
    pub(in crate::backend::statevector) fn apply_qft_block_textbook(
        &mut self,
        start: usize,
        num: usize,
        inverse: bool,
        swaps: bool,
        big_endian: bool,
    ) {
        assert!(start + num <= self.num_qubits);
        assert!(num >= 1);

        let h = Gate::H.matrix_2x2();
        for step in qft_textbook_steps(start, num, inverse, swaps, big_endian) {
            match step {
                QftTextbookStep::Hadamard(q) => self.apply_single_gate(q, h),
                QftTextbookStep::CPhase {
                    control,
                    target,
                    theta,
                } => self.apply_cu_phase(control, target, Complex64::from_polar(1.0, theta)),
                QftTextbookStep::Swap(a, b) => self.apply_swap(a, b),
            }
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod qft_gather_tests;

//! Tile geometry and budget for the `Multi2q` and `MultiPauliRot` batches.

use num_complex::Complex64;

/// A `Multi2q` batch runs inside one tile of `2^tile_bits` amplitudes: the
/// lowest bits of the index plus up to `tile_bits - MULTI_2Q_LOW_BITS` gathered
/// high qubits, so a batch is tileable when its gates touch at most that many
/// distinct qubits at or above [`multi_2q_low_bits`]. The tile size is per
/// platform, see [`multi_2q_tile_bits`]; [`MULTI_2Q_HIGH_BUDGET`] is the
/// widest budget any platform reaches and sizes the inline qubit lists.
pub(crate) const MULTI_2Q_MAX_TILE_BITS: usize = 18;
/// Smallest tile the knob accepts, 128 KB: half the derived floor, so a sweep can
/// try it.
pub(crate) const MULTI_2Q_MIN_TILE_BITS: usize = 13;
/// Fewest low bits a tile keeps contiguous: 64-amplitude runs, 1 KB each.
pub(crate) const MULTI_2Q_MIN_LOW_BITS: usize = 6;
/// Most distinct high qubits one `Multi2q` batch may span on any platform.
pub(crate) const MULTI_2Q_HIGH_BUDGET: usize = MULTI_2Q_MAX_TILE_BITS - MULTI_2Q_MIN_LOW_BITS;

/// Qubits from this index up count against a batch's high-qubit budget: the
/// tile keeps `2^low_bits` amplitudes contiguous per run and gathers the rest.
/// Default 6, the shortest run, since every bit taken from the run is one
/// more high qubit a pass over the state can serve: at 24 qubits a quantum
/// volume circuit went from 88 passes to 54 and ran 10% faster against 10.
/// `PRISM_MULTI_2Q_LOW_BITS` overrides it, clamped to
/// `MULTI_2Q_MIN_LOW_BITS..tile_bits`.
pub(crate) fn multi_2q_low_bits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::env_knobs::usize_override("PRISM_MULTI_2Q_LOW_BITS", 1)
            .unwrap_or(6)
            .clamp(MULTI_2Q_MIN_LOW_BITS, multi_2q_tile_bits() - 1)
    })
}

/// Log2 of the `Multi2q` tile in amplitudes: the tile budget from the cache topology
/// (2^14, 256 KB, on the x86 cores measured), or `PRISM_MULTI_2Q_TILE_BITS` clamped
/// to `MULTI_2Q_MIN_TILE_BITS..=MULTI_2Q_MAX_TILE_BITS`.
///
/// The tile has to sit in the core's private cache: on an i7-6700K (256 KB
/// of L2 per core, 8 MB of L3 shared by four) a 1 MB tile ran quantum volume
/// at 22 and 24 qubits 2x slower and a 2 MB tile 3x, while 512 KB read flat.
/// A wider tile takes more of a random-pair layer per pass over the state,
/// so on a core with a larger private cache the knob is worth a sweep.
pub(crate) fn multi_2q_tile_bits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::env_knobs::usize_override("PRISM_MULTI_2Q_TILE_BITS", 1)
            .unwrap_or_else(|| {
                (crate::backend::cache::tile_budget_bytes() / size_of::<Complex64>()).ilog2()
                    as usize
            })
            .clamp(MULTI_2Q_MIN_TILE_BITS, MULTI_2Q_MAX_TILE_BITS)
    })
}

/// [`multi_2q_tile_bits`] for a `num_qubits` statevector: cut, down to
/// [`MULTI_2Q_MIN_TILE_BITS`], so a parallel pass gathers at least
/// `cache::min_parallel_tiles` subcubes instead of leaving workers idle at the widths
/// just past the parallel threshold.
#[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
pub(crate) fn multi_2q_tile_bits_for(num_qubits: usize) -> usize {
    let bits = multi_2q_tile_bits();
    #[cfg(feature = "parallel")]
    if num_qubits >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
        let pool_bits = num_qubits - crate::backend::cache::min_parallel_tiles().ilog2() as usize;
        return bits.min(pool_bits).max(MULTI_2Q_MIN_TILE_BITS);
    }
    bits
}

/// Most distinct high qubits one `Multi2q` batch may span on a `num_qubits` state.
pub(crate) fn multi_2q_high_budget_for(num_qubits: usize) -> usize {
    multi_2q_tile_bits_for(num_qubits) - multi_2q_low_bits()
}

/// The high qubits `high` grows to if the pair joins a batch on a `num_qubits`
/// state, or `None` when the pair would take it past [`multi_2q_high_budget_for`].
#[inline]
pub(crate) fn multi_2q_join(
    high: &[usize],
    q0: usize,
    q1: usize,
    num_qubits: usize,
) -> Option<smallvec::SmallVec<[usize; MULTI_2Q_HIGH_BUDGET]>> {
    pauli_rot_join(high, [q0, q1], num_qubits)
}

/// [`multi_2q_join`] for the X and Y qubits of a Pauli string, which a
/// `MultiPauliRot` batch needs inside its tile.
pub(crate) fn pauli_rot_join(
    high: &[usize],
    qubits: impl IntoIterator<Item = usize>,
    num_qubits: usize,
) -> Option<smallvec::SmallVec<[usize; MULTI_2Q_HIGH_BUDGET]>> {
    let budget = multi_2q_high_budget_for(num_qubits);
    let low_bits = multi_2q_low_bits();
    let mut joined: smallvec::SmallVec<[usize; MULTI_2Q_HIGH_BUDGET]> =
        high.iter().copied().collect();
    for q in qubits {
        if q >= low_bits && !joined.contains(&q) {
            if joined.len() == budget {
                return None;
            }
            joined.push(q);
        }
    }
    Some(joined)
}

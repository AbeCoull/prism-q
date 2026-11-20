//! Pauli rotation kernels: pair rotations, parity phases and tiled multi-rotation passes.

use super::*;

/// Shortest run of contiguous pairs handed to the SIMD pair kernel. Below it the
/// AVX2 loop runs at most one 256-bit iteration per call, so the per-run dispatch
/// outweighs it and the scalar loop takes the block.
const MIN_PAULI_ROT_SIMD_RUN: usize = 4;

/// The pair arithmetic of `exp(-i θ P / 2)` for [`pauli_rot_pair_halves`].
///
/// `m_lo` and `m_hi` are the parity-even cross coefficients toward the pivot-clear
/// and pivot-set side; `even` and `odd` hold the pair matrix
/// `[[c, ±m_lo], [±m_hi, c]]` for each parity with its SIMD tier chosen once.
struct PauliRotPair {
    c: f64,
    m_lo: Complex64,
    m_hi: Complex64,
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
        let m_lo = if num_y % 2 == 1 { -m_hi } else { m_hi };
        let diag = Complex64::new(c, 0.0);
        Self {
            c,
            m_lo,
            m_hi,
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
                let phases = if flip == 1 {
                    [*odd, *even]
                } else {
                    [*even, *odd]
                };
                for (i, amp) in tile.iter_mut().enumerate() {
                    *amp *= phases[((i & self.zmask).count_ones() & 1) as usize];
                }
            }
            TiledAction::Pairs(pair) => {
                let pivot = usize::BITS as usize - 1 - self.xmask.leading_zeros() as usize;
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
        let half = 1usize << pivot;
        let block = half << 1;
        let xlow = xmask ^ half;

        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            self.apply_pauli_rot_pairs_par(pivot, xlow, zmask, &pair);
            return;
        }

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

        let phases = [even, odd];
        for (i, amp) in self.state.iter_mut().enumerate() {
            *amp *= phases[((i & zmask).count_ones() & 1) as usize];
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_parity_phase_par(&mut self, zmask: usize, even: Complex64, odd: Complex64) {
        let phases = [even, odd];
        self.state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base = chunk_idx * MIN_PAR_ELEMS;
                for (j, amp) in chunk.iter_mut().enumerate() {
                    *amp *= phases[(((base + j) & zmask).count_ones() & 1) as usize];
                }
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

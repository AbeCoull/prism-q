//! The [`NoisyCompiledSampler`]: Pauli noise on a Clifford circuit as flips of a compiled
//! measurement sensitivity map.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::Result;
use crate::sim::ShotsResult;
use crate::sim::compiled::{
    PackedShots, batch_propagate_backward, compile_measurements, xor_words,
};

#[cfg(feature = "gpu")]
use super::gpu_cache::GpuNoiseCache;
use super::{NoiseEvent, NoiseModel};

/// Rows a two-qubit event stores: the propagated X and Z components of each
/// of its two targets, in the order `x0, z0, x1, z1`. The 15 branches are
/// drawn jointly at sample time and read these four rows, rather than being
/// precomputed as 15 rows of their own.
const PAIR_ROWS: usize = 4;

pub(super) struct FlatNoiseSensitivity {
    pub(super) x_data: Vec<u64>,
    pub(super) z_data: Vec<u64>,
    pub(super) probs: Vec<[f64; 3]>,
    /// [`PAIR_ROWS`] rows of `m_words` per two-qubit event. Kept apart from
    /// the single-qubit rows, which the LUT build, the device CSR transpose
    /// and the threshold table all index positionally.
    pair_data: Vec<u64>,
    pair_probs: Vec<f64>,
    pub(super) m_words: usize,
}

impl FlatNoiseSensitivity {
    fn new(m_words: usize, capacity: usize) -> Self {
        Self {
            x_data: Vec::with_capacity(capacity * m_words),
            z_data: Vec::with_capacity(capacity * m_words),
            probs: Vec::with_capacity(capacity),
            pair_data: Vec::new(),
            pair_probs: Vec::new(),
            m_words,
        }
    }

    fn push(&mut self, x_flip: &[u64], z_flip: &[u64], px: f64, py: f64, pz: f64) {
        self.x_data.extend_from_slice(x_flip);
        self.z_data.extend_from_slice(z_flip);
        self.probs.push([px, py, pz]);
    }

    fn push_pair(&mut self, x0: &[u64], z0: &[u64], x1: &[u64], z1: &[u64], p: f64) {
        for row in [x0, z0, x1, z1] {
            self.pair_data.extend_from_slice(row);
        }
        self.pair_probs.push(p);
    }

    /// The device noise kernel derives each thread's bit from
    /// `(seed, event, batch)` for the one measurement row it owns, so a branch
    /// drawn there would decorrelate the two qubits. A non-empty pair table
    /// therefore keeps noise application on the host.
    #[inline(always)]
    pub(super) fn has_pairs(&self) -> bool {
        !self.pair_probs.is_empty()
    }

    /// One row of pair event `idx`, indexed as [`PAIR_ROWS`] describes.
    #[inline(always)]
    fn pair_row(&self, idx: usize, row: usize) -> &[u64] {
        let off = (idx * PAIR_ROWS + row) * self.m_words;
        &self.pair_data[off..off + self.m_words]
    }

    #[inline(always)]
    pub(super) fn len(&self) -> usize {
        self.probs.len()
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.probs.is_empty()
    }

    #[inline(always)]
    fn x_flip(&self, idx: usize) -> &[u64] {
        let off = idx * self.m_words;
        &self.x_data[off..off + self.m_words]
    }

    #[inline(always)]
    fn z_flip(&self, idx: usize) -> &[u64] {
        let off = idx * self.m_words;
        &self.z_data[off..off + self.m_words]
    }
}

/// XOR the measurements one Pauli letter flips into `shot`. An X error
/// anticommutes with the propagated Z component, a Z error with the X
/// component, and a Y error with both.
#[inline(always)]
fn xor_pair_letter(shot: &mut [u64], x_row: &[u64], z_row: &[u64], letter: usize) {
    match letter {
        1 => xor_words(shot, z_row),
        2 => {
            xor_words(shot, x_row);
            xor_words(shot, z_row);
        }
        3 => xor_words(shot, x_row),
        _ => {}
    }
}

/// `u` is uniform on `[0, 1)` conditioned on the event having fired.
#[inline(always)]
fn xor_pair_branch(events: &FlatNoiseSensitivity, idx: usize, u: f64, shot: &mut [u64]) {
    let (l0, l1) = pair_branch(u);
    xor_pair_letter(shot, events.pair_row(idx, 0), events.pair_row(idx, 1), l0);
    xor_pair_letter(shot, events.pair_row(idx, 2), events.pair_row(idx, 3), l1);
}

/// The Pauli letter on each target of one uniformly drawn two-qubit branch,
/// as 0 for identity through 3 for Z, from a uniform draw on `[0, 1)`.
///
/// The identity product is not a branch: a firing shot picks one of the 15
/// others, so the index runs 1 through 15 and every firing shot moves.
#[inline(always)]
pub(in crate::sim) fn pair_branch(u: f64) -> (usize, usize) {
    let idx = 1 + ((u * 15.0) as usize).min(14);
    (idx >> 2, idx & 3)
}

#[inline(always)]
pub(super) fn geometric_sample(rng: &mut ChaCha8Rng, ln_1mp: f64) -> usize {
    let u: f64 = 1.0 - rand::RngExt::random::<f64>(rng);
    (u.ln() / ln_1mp) as usize
}

/// [`geometric_sample`] on the sampler's own generator. `ln_1mp` is `ln(1 - p)`,
/// precomputed by the caller.
#[inline(always)]
pub(crate) fn geometric_sample_xoshiro(
    rng: &mut crate::sim::compiled::rng::Xoshiro256PlusPlus,
    ln_1mp: f64,
) -> usize {
    // `next_f64` returns [0, 1); shift to (0, 1] so `ln` stays finite.
    let u: f64 = 1.0 - rng.next_f64();
    (u.ln() / ln_1mp) as usize
}

/// RNG source for [`fill_one_event`]. Bundles uniform and geometric draws
/// onto a single trait so the shared body can use one `&mut` borrow instead
/// of two concurrent closures over the underlying generator.
#[cfg(feature = "gpu")]
trait EventRng {
    fn uniform(&mut self) -> f64;
    fn geom(&mut self, ln_1mp: f64) -> usize;
}

#[cfg(feature = "gpu")]
impl EventRng for ChaCha8Rng {
    #[inline(always)]
    fn uniform(&mut self) -> f64 {
        rand::RngExt::random::<f64>(self)
    }
    #[inline(always)]
    fn geom(&mut self, ln_1mp: f64) -> usize {
        geometric_sample(self, ln_1mp)
    }
}

#[cfg(feature = "gpu")]
impl EventRng for crate::sim::compiled::rng::Xoshiro256PlusPlus {
    #[inline(always)]
    fn uniform(&mut self) -> f64 {
        self.next_f64()
    }
    #[inline(always)]
    fn geom(&mut self, ln_1mp: f64) -> usize {
        geometric_sample_xoshiro(self, ln_1mp)
    }
}

/// Fill one event's portion of the noise masks. Shared by the serial master
/// ChaCha path and the parallel per-event Xoshiro path, so the bernoulli
/// and geometric branches live in exactly one place.
#[cfg(feature = "gpu")]
#[inline]
fn fill_one_event(
    x_slot: &mut [u64],
    z_slot: &mut [u64],
    probs: [f64; 3],
    chunk_shots: usize,
    rng: &mut impl EventRng,
) {
    let [px, py, pz] = probs;
    let p_event = px + py + pz;
    if p_event == 0.0 {
        return;
    }
    let pxy = px + py;

    // High-p or tiny chunk: a per-shot bernoulli sweep dominates geometric
    // skipping overhead because most shots flip anyway.
    if p_event >= 0.5 || chunk_shots < 32 {
        for s in 0..chunk_shots {
            let r = rng.uniform();
            let slot_idx = s / 64;
            let bit = 1u64 << (s % 64);
            if r < px {
                z_slot[slot_idx] |= bit;
            } else if r < pxy {
                z_slot[slot_idx] |= bit;
                x_slot[slot_idx] |= bit;
            } else if r < p_event {
                x_slot[slot_idx] |= bit;
            }
        }
        return;
    }

    // Low-p: skip inactive shots via geometric sampling. `px_frac` / `pxy_frac`
    // rescale to the conditional distribution given an event fired.
    let ln_1mp = (-p_event).ln_1p();
    let px_frac = px / p_event;
    let pxy_frac = pxy / p_event;
    let mut pos = rng.geom(ln_1mp);
    while pos < chunk_shots {
        let r = rng.uniform();
        let slot_idx = pos / 64;
        let bit = 1u64 << (pos % 64);
        if r < px_frac {
            z_slot[slot_idx] |= bit;
        } else if r < pxy_frac {
            z_slot[slot_idx] |= bit;
            x_slot[slot_idx] |= bit;
        } else {
            x_slot[slot_idx] |= bit;
        }
        pos += 1 + rng.geom(ln_1mp);
    }
}

const NOISE_LUT_K: usize = 8;
const NOISE_LUT_MIN_EVENTS: usize = 16;
const NOISE_LUT_TILE: usize = 4096;

pub(super) fn unpack_and_remap(
    accum: &[u64],
    m_words: usize,
    num_shots: usize,
    classical_bit_order: &[usize],
    num_classical: usize,
) -> Vec<Vec<bool>> {
    fn unpack_one(src: &[u64], classical_bit_order: &[usize], num_classical: usize) -> Vec<bool> {
        let mut out = vec![false; num_classical];
        for (mi, &cbit) in classical_bit_order.iter().enumerate() {
            if cbit < num_classical {
                out[cbit] = (src[mi / 64] >> (mi % 64)) & 1 != 0;
            }
        }
        out
    }

    #[cfg(feature = "parallel")]
    if num_shots >= 256 {
        use rayon::prelude::*;
        return accum
            .par_chunks(m_words)
            .map(|src| unpack_one(src, classical_bit_order, num_classical))
            .collect();
    }

    #[cfg(not(feature = "parallel"))]
    let _ = num_shots;

    accum
        .chunks(m_words)
        .map(|src| unpack_one(src, classical_bit_order, num_classical))
        .collect()
}

fn unpack_and_remap_packed(
    packed: &PackedShots,
    num_shots: usize,
    classical_bit_order: &[usize],
    num_classical: usize,
) -> Vec<Vec<bool>> {
    fn unpack_one_packed(
        packed: &PackedShots,
        shot: usize,
        classical_bit_order: &[usize],
        num_classical: usize,
    ) -> Vec<bool> {
        let mut out = vec![false; num_classical];
        for (mi, &cbit) in classical_bit_order.iter().enumerate() {
            if cbit < num_classical {
                out[cbit] = packed.get_bit(shot, mi);
            }
        }
        out
    }

    #[cfg(feature = "parallel")]
    if num_shots >= 256 {
        use rayon::prelude::*;
        return (0..num_shots)
            .into_par_iter()
            .map(|s| unpack_one_packed(packed, s, classical_bit_order, num_classical))
            .collect();
    }

    #[cfg(not(feature = "parallel"))]
    let _ = num_shots;

    (0..packed.num_shots())
        .map(|s| unpack_one_packed(packed, s, classical_bit_order, num_classical))
        .collect()
}

/// Readout error on one measurement record, the index the packed shot buffers
/// carry. Only records carrying an entry that can flip a bit appear.
pub(super) struct RecordReadout {
    pub(super) record: usize,
    pub(super) p01: f64,
    pub(super) p10: f64,
}

pub(super) fn record_readout(circuit: &Circuit, noise: &NoiseModel) -> Vec<RecordReadout> {
    circuit
        .classical_bit_order()
        .into_iter()
        .enumerate()
        .filter_map(|(record, cbit)| {
            let err = noise.readout.get(cbit)?.as_ref()?;
            (!err.is_inert()).then_some(RecordReadout {
                record,
                p01: err.p01,
                p10: err.p10,
            })
        })
        .collect()
}

/// Flip readout errors into a shot-major packed record, in place.
///
/// Runs after the reference bits are folded in, so a set bit is a measured one
/// and takes `p10`. Asymmetric rates rule out a single flip mask: candidates
/// are thinned at `max(p01, p10)`, then accepted at the rate the live bit
/// selects. A record with no readout entry draws nothing.
pub(super) fn apply_readout_packed(
    accum: &mut [u64],
    num_shots: usize,
    m_words: usize,
    readout: &[RecordReadout],
    rng: &mut ChaCha8Rng,
) {
    for entry in readout {
        let p_max = entry.p01.max(entry.p10);
        let word = entry.record / 64;
        let bit = 1u64 << (entry.record % 64);

        if p_max >= 0.5 || num_shots < 32 {
            for s in 0..num_shots {
                let slot = &mut accum[s * m_words + word];
                let p = if *slot & bit != 0 {
                    entry.p10
                } else {
                    entry.p01
                };
                let r: f64 = rand::RngExt::random(rng);
                if r < p {
                    *slot ^= bit;
                }
            }
            continue;
        }

        let ln_1mp = (-p_max).ln_1p();
        let accept_set = entry.p10 / p_max;
        let accept_clear = entry.p01 / p_max;
        let mut pos = geometric_sample(rng, ln_1mp);
        while pos < num_shots {
            let slot = &mut accum[pos * m_words + word];
            let accept = if *slot & bit != 0 {
                accept_set
            } else {
                accept_clear
            };
            let r: f64 = rand::RngExt::random(rng);
            if r < accept {
                *slot ^= bit;
            }
            pos += 1 + geometric_sample(rng, ln_1mp);
        }
    }
}

pub(super) struct NoiseFlipLut {
    data: Vec<u64>,
    m_words: usize,
    num_full_groups: usize,
    remainder_size: usize,
}

impl NoiseFlipLut {
    fn build_from_flat(flat_data: &[u64], num_rows: usize, m_words: usize) -> Self {
        let num_full_groups = num_rows / NOISE_LUT_K;
        let remainder_size = num_rows % NOISE_LUT_K;
        let total_groups = num_full_groups + usize::from(remainder_size > 0);
        let entries = 1 << NOISE_LUT_K;

        let mut data = vec![0u64; total_groups * entries * m_words];

        for g in 0..total_groups {
            let group_start = g * NOISE_LUT_K;
            let k = if g < num_full_groups {
                NOISE_LUT_K
            } else {
                remainder_size
            };
            let lut_off = g * entries * m_words;

            for byte in 1..(1usize << k) {
                let lowest = byte & byte.wrapping_neg();
                let row_idx = group_start + lowest.trailing_zeros() as usize;
                let prev = byte ^ lowest;

                let dst = lut_off + byte * m_words;
                let src = lut_off + prev * m_words;
                let row_off = row_idx * m_words;

                for w in 0..m_words {
                    data[dst + w] = data[src + w] ^ flat_data[row_off + w];
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
    fn total_groups(&self) -> usize {
        self.num_full_groups + usize::from(self.remainder_size > 0)
    }

    #[inline(always)]
    fn apply_masked(&self, accum: &mut [u64], mask: &[u64]) {
        for g in 0..self.total_groups() {
            let byte = ((mask[g / 8] >> ((g % 8) * 8)) & 0xFF) as usize;
            if byte != 0 {
                let offset = (g * (1 << NOISE_LUT_K) + byte) * self.m_words;
                xor_words(accum, &self.data[offset..offset + self.m_words]);
            }
        }
    }
}

fn build_noise_luts(events: &FlatNoiseSensitivity) -> (Option<NoiseFlipLut>, Option<NoiseFlipLut>) {
    let ne = events.len();
    if ne < NOISE_LUT_MIN_EVENTS {
        return (None, None);
    }

    let avg_p: f64 = events
        .probs
        .iter()
        .map(|[px, py, pz]| px + py + pz)
        .sum::<f64>()
        / ne as f64;
    if avg_p < 0.05 {
        return (None, None);
    }

    let mw = events.m_words;
    let z_lut = NoiseFlipLut::build_from_flat(&events.z_data, ne, mw);
    let x_lut = NoiseFlipLut::build_from_flat(&events.x_data, ne, mw);
    (Some(z_lut), Some(x_lut))
}

/// Compiled sampler for Clifford circuits with Pauli noise.
///
/// Reuses the noiseless circuit's compiled parity map and applies the Pauli
/// noise per batch of shots. With a GPU context attached (`with_gpu`, `gpu`
/// feature), parity sampling may take the GPU BTS path. Materialized shots return
/// to the host in shot-major form; counts and marginals can keep the packed
/// measurement-major buffer on the device through noise and reduction. `try_`
/// methods propagate GPU errors; their plain twins fall back to the CPU path.
pub struct NoisyCompiledSampler {
    pub(super) noiseless: crate::sim::compiled::CompiledSampler,
    pub(super) events: FlatNoiseSensitivity,
    /// Readout error per measurement record, applied once the reference bits
    /// and noise are folded in: on the device ahead of the counts and
    /// marginals reductions, on the host for materialized shots.
    pub(super) readout: Vec<RecordReadout>,
    pub(super) num_measurements: usize,
    pub(super) rng: ChaCha8Rng,
    pub(super) z_lut: Option<NoiseFlipLut>,
    x_lut: Option<NoiseFlipLut>,
    #[cfg(feature = "gpu")]
    gpu_noise_cache: Option<GpuNoiseCache>,
}

impl NoisyCompiledSampler {
    /// Opt in to GPU-accelerated sampling for the underlying compiled
    /// noiseless sampler.
    ///
    /// Small shot batches may continue to use the CPU path when the compiled
    /// sampler stays below the GPU BTS threshold. On the device path, noise
    /// masks are generated by a device-seeded RNG stream when the compiled
    /// event probabilities permit, otherwise on the host and uploaded; counts
    /// and marginals then apply them and any readout error, and reduce on the
    /// device. Readout error keeps materialized shots on the CPU path. Device
    /// draws use a different RNG stream than the CPU path, so GPU output
    /// matches CPU output statistically, not shot for shot.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, context: std::sync::Arc<crate::gpu::GpuContext>) -> Self {
        self.noiseless = self.noiseless.with_gpu(context);
        self.gpu_noise_cache = None;
        self
    }

    pub(super) fn sample_bulk_words_shot_major_cpu(
        &mut self,
        num_shots: usize,
    ) -> (Vec<u64>, usize) {
        let (mut accum, m_words) = self.noiseless.sample_bulk_words_shot_major(num_shots);
        self.apply_noise_bulk(&mut accum, num_shots, m_words);

        let ref_bits_packed = self.noiseless.ref_bits_packed();

        #[cfg(feature = "parallel")]
        if num_shots >= 256 {
            use rayon::prelude::*;
            accum
                .par_chunks_mut(m_words)
                .with_min_len(crate::backend::chunk_min_len(m_words))
                .for_each(|shot| xor_words(shot, ref_bits_packed));
        } else {
            for s in 0..num_shots {
                let shot_base = s * m_words;
                xor_words(&mut accum[shot_base..shot_base + m_words], ref_bits_packed);
            }
        }

        #[cfg(not(feature = "parallel"))]
        for s in 0..num_shots {
            let shot_base = s * m_words;
            xor_words(&mut accum[shot_base..shot_base + m_words], ref_bits_packed);
        }

        apply_readout_packed(&mut accum, num_shots, m_words, &self.readout, &mut self.rng);

        (accum, m_words)
    }

    #[cfg(feature = "gpu")]
    fn try_sample_bulk_packed_with_gpu_context(&mut self, num_shots: usize) -> Result<PackedShots> {
        let m_words = self.num_measurements.div_ceil(64);
        if num_shots == 0 || self.num_measurements == 0 {
            return Ok(PackedShots::from_shot_major(
                vec![0u64; num_shots * m_words],
                num_shots,
                self.num_measurements,
            ));
        }

        // Prefer the GPU shot-major path: BTS plus on-device bit-transpose
        // into shot-major layout eliminates the host `into_shot_major_data()`
        // transpose that otherwise dominates noisy sampling at large shot
        // counts. Falls back to the default sampler (which may itself dispatch
        // to the GPU meas-major BTS) when the shot-major path does not apply.
        let mut accum = match self.noiseless.try_sample_bulk_shot_major_gpu(num_shots) {
            Some(Ok(data)) => data,
            Some(Err(e)) => return Err(e),
            None => {
                let (accum, _m_words) = self.sample_bulk_words_shot_major_cpu(num_shots);
                return Ok(PackedShots::from_shot_major(
                    accum,
                    num_shots,
                    self.num_measurements,
                ));
            }
        };
        self.apply_noise_bulk(&mut accum, num_shots, m_words);
        Ok(PackedShots::from_shot_major(
            accum,
            num_shots,
            self.num_measurements,
        ))
    }

    #[cfg(feature = "gpu")]
    fn take_gpu_noise_cache(&mut self) -> Result<GpuNoiseCache> {
        if let Some(cache) = self.gpu_noise_cache.take() {
            return Ok(cache);
        }
        let context = self
            .noiseless
            .gpu_context()
            .expect("gpu noise cache requested without gpu_context");
        GpuNoiseCache::new(context, &self.events, &self.readout)
    }

    #[cfg(feature = "gpu")]
    fn fill_noise_masks_event_major(
        events: &FlatNoiseSensitivity,
        rng: &mut ChaCha8Rng,
        x_masks: &mut [u64],
        z_masks: &mut [u64],
        chunk_shots: usize,
        chunk_s_words: usize,
    ) {
        x_masks.fill(0);
        z_masks.fill(0);

        // Events touch disjoint per-event slots of `x_masks` / `z_masks`, so
        // they parallelise cleanly once each worker has its own RNG stream.
        // Seed one Xoshiro per event from the master ChaCha so the draws are
        // independent of Rayon's thread count while matching the serial
        // reference statistically.
        #[cfg(feature = "parallel")]
        {
            use rand::Rng;
            use rayon::prelude::*;
            // Rayon split + per-event seed draw has to be dominated by real
            // work; below this threshold the serial master-rng loop wins.
            // Tuned against the noisy-sampling bench at 1M shots where
            // num_events is small and the tile is also small.
            const MIN_PAR_EVENTS: usize = 64;
            const MIN_PAR_SLOTS: usize = 32_768;
            let work_slots = events.len().saturating_mul(chunk_s_words);
            if events.len() >= MIN_PAR_EVENTS
                && work_slots >= MIN_PAR_SLOTS
                && x_masks.len() == events.len() * chunk_s_words
                && z_masks.len() == events.len() * chunk_s_words
            {
                let thread_seeds: Vec<[u64; 4]> = (0..events.len())
                    .map(|_| {
                        [
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                        ]
                    })
                    .collect();

                x_masks
                    .par_chunks_mut(chunk_s_words)
                    .zip(z_masks.par_chunks_mut(chunk_s_words))
                    .zip(thread_seeds.par_iter())
                    .enumerate()
                    .for_each(|(i, ((x_slot, z_slot), seed))| {
                        let mut trng =
                            crate::sim::compiled::rng::Xoshiro256PlusPlus::from_seeds(*seed);
                        fill_one_event(x_slot, z_slot, events.probs[i], chunk_shots, &mut trng);
                    });
                return;
            }
        }

        for (i, &probs) in events.probs.iter().enumerate() {
            let base = i * chunk_s_words;
            let end = base + chunk_s_words;
            fill_one_event(
                &mut x_masks[base..end],
                &mut z_masks[base..end],
                probs,
                chunk_shots,
                rng,
            );
        }
    }

    #[cfg(feature = "gpu")]
    fn apply_noise_device_meas_major(
        &mut self,
        packed: &mut crate::sim::compiled::DevicePackedShots,
    ) -> Result<()> {
        if self.events.is_empty() || packed.num_shots() == 0 || self.num_measurements == 0 {
            return Ok(());
        }

        let num_events = self.events.len();
        let mut cache = self.take_gpu_noise_cache()?;
        let total_s_words = packed.s_words();
        let context = packed.context().clone();
        let num_shots = packed.num_shots();

        // Fused path: the kernel seeds a per-(event, batch) xoshiro stream
        // from a master seed drawn on the host, then generates and XORs the
        // noise masks in one launch per tile. Drops the per-tile mask fill
        // and ~25-50 MB PCIe upload that the fallback branch pays at large
        // shot counts. The fallback runs only when compiled event
        // probabilities fail the `u64` threshold check.
        let use_fused = cache.event_thresholds_valid;
        let master_seed: u64 = if use_fused {
            rand::Rng::next_u64(&mut self.rng)
        } else {
            0
        };

        let mut shot_start = 0;
        while shot_start < num_shots {
            let chunk_shots = (shot_start + NOISE_LUT_TILE).min(num_shots) - shot_start;
            let chunk_s_words = chunk_shots.div_ceil(64);
            let word_offset = shot_start / 64;

            if use_fused {
                crate::gpu::kernels::bts::generate_and_apply_noise_masks_meas_major_by_row(
                    &context,
                    crate::gpu::kernels::bts::NoiseDeviceGenApplyByRow {
                        meas_major: packed.data_mut(),
                        num_meas: self.num_measurements,
                        s_words: total_s_words,
                        word_offset,
                        chunk_s_words,
                        row_event_offsets: &cache.row_event_offsets_dev,
                        row_event_entries: &cache.row_event_entries_dev,
                        event_thresholds: &cache.event_thresholds_dev,
                        master_seed,
                        batch_offset: word_offset as u64,
                    },
                )?;
            } else {
                let mask_len = num_events * chunk_s_words;
                cache.ensure_mask_capacity(mask_len)?;
                Self::fill_noise_masks_event_major(
                    &self.events,
                    &mut self.rng,
                    &mut cache.x_masks_host[..mask_len],
                    &mut cache.z_masks_host[..mask_len],
                    chunk_shots,
                    chunk_s_words,
                );
                if mask_len > 0 {
                    cache
                        .x_masks_dev
                        .copy_from_host(cache.context.device(), &cache.x_masks_host[..mask_len])?;
                    cache
                        .z_masks_dev
                        .copy_from_host(cache.context.device(), &cache.z_masks_host[..mask_len])?;
                    let base = crate::gpu::kernels::bts::NoiseApplyBase {
                        meas_major: packed.data_mut(),
                        num_meas: self.num_measurements,
                        s_words: total_s_words,
                        word_offset,
                        chunk_s_words,
                        num_events,
                        x_row_offsets: &cache.x_row_offsets_dev,
                        x_row_indices: &cache.x_row_indices_dev,
                        z_row_offsets: &cache.z_row_offsets_dev,
                        z_row_indices: &cache.z_row_indices_dev,
                    };
                    crate::gpu::kernels::bts::apply_noise_masks_meas_major(
                        &context,
                        crate::gpu::kernels::bts::NoiseMaskApply {
                            base,
                            x_masks: &cache.x_masks_dev,
                            z_masks: &cache.z_masks_dev,
                        },
                    )?;
                }
            }

            shot_start += chunk_shots;
        }

        self.gpu_noise_cache = Some(cache);
        Ok(())
    }

    /// Flip readout error into the device record, after
    /// [`Self::apply_noise_device_meas_major`].
    ///
    /// Counts need the flips per shot, so marginals take the same per-bit draws
    /// rather than folding the rates into the reduced probabilities: both stay
    /// shot frequencies, as on the host path.
    #[cfg(feature = "gpu")]
    fn apply_readout_device_meas_major(
        &mut self,
        packed: &mut crate::sim::compiled::DevicePackedShots,
    ) -> Result<()> {
        if self.readout.is_empty() || packed.num_shots() == 0 {
            return Ok(());
        }

        let cache = self.take_gpu_noise_cache()?;
        let master_seed = rand::Rng::next_u64(&mut self.rng);
        let context = packed.context().clone();
        let num_shots = packed.num_shots();
        crate::gpu::kernels::bts::apply_readout_meas_major(
            &context,
            packed.data_mut(),
            num_shots,
            &cache.readout_dev,
            self.readout.len(),
            master_seed,
        )?;
        self.gpu_noise_cache = Some(cache);
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn try_sample_marginals_gpu(&mut self, total_shots: usize) -> Option<Result<Vec<f64>>> {
        if self.events.has_pairs() || !self.noiseless.should_use_gpu_bts(total_shots) {
            return None;
        }

        let mut packed = match self.noiseless.sample_bulk_packed_device(total_shots) {
            Ok(packed) => packed,
            Err(e) => return Some(Err(e)),
        };
        if let Err(e) = self.apply_noise_device_meas_major(&mut packed) {
            return Some(Err(e));
        }
        if let Err(e) = self.apply_readout_device_meas_major(&mut packed) {
            return Some(Err(e));
        }
        Some(packed.marginals())
    }

    #[cfg(feature = "gpu")]
    fn try_sample_counts_gpu(
        &mut self,
        total_shots: usize,
    ) -> Option<Result<std::collections::HashMap<Vec<u64>, u64>>> {
        if self.events.has_pairs() || !self.noiseless.should_use_gpu_bts(total_shots) {
            return None;
        }

        let mut packed = match self.noiseless.sample_bulk_packed_device(total_shots) {
            Ok(packed) => packed,
            Err(e) => return Some(Err(e)),
        };
        if let Err(e) = self.apply_noise_device_meas_major(&mut packed) {
            return Some(Err(e));
        }
        if let Err(e) = self.apply_readout_device_meas_major(&mut packed) {
            return Some(Err(e));
        }
        Some(packed.counts_with_rank_hint(self.num_measurements))
    }

    /// Sample `num_shots` noisy outcomes, packed shot-major.
    ///
    /// With a GPU context attached, parity sampling may run on the device; the
    /// packed output is materialized on the host either way.
    pub fn sample_bulk_packed(&mut self, num_shots: usize) -> PackedShots {
        match self.try_sample_bulk_packed(num_shots) {
            Ok(packed) => packed,
            Err(_) => self.sample_bulk_packed_cpu(num_shots),
        }
    }

    pub fn try_sample_bulk_packed(&mut self, num_shots: usize) -> Result<PackedShots> {
        #[cfg(feature = "gpu")]
        if self.noiseless.has_gpu_context() && self.readout.is_empty() {
            return self.try_sample_bulk_packed_with_gpu_context(num_shots);
        }

        Ok(self.sample_bulk_packed_cpu(num_shots))
    }

    fn sample_bulk_packed_cpu(&mut self, num_shots: usize) -> PackedShots {
        let m_words = self.num_measurements.div_ceil(64);
        if num_shots == 0 || self.num_measurements == 0 {
            return PackedShots::from_shot_major(
                vec![0u64; num_shots * m_words],
                num_shots,
                self.num_measurements,
            );
        }

        let (accum, _m_words) = self.sample_bulk_words_shot_major_cpu(num_shots);
        PackedShots::from_shot_major(accum, num_shots, self.num_measurements)
    }

    /// Stream noisy shots into `acc` at the default chunk size, without
    /// materializing the full shot matrix.
    pub fn sample_chunked<A: crate::sim::compiled::ShotAccumulator>(
        &mut self,
        total_shots: usize,
        acc: &mut A,
    ) {
        let chunk_size = crate::sim::compiled::default_chunk_size(self.num_measurements);
        self.sample_chunked_with_size(total_shots, chunk_size, acc);
    }

    /// Stream noisy shots into `acc` with an explicit chunk size.
    ///
    /// With a GPU context attached, each chunk may use GPU BTS sampling for the
    /// noiseless parity stage.
    pub fn sample_chunked_with_size<A: crate::sim::compiled::ShotAccumulator>(
        &mut self,
        total_shots: usize,
        chunk_size: usize,
        acc: &mut A,
    ) {
        #[cfg(feature = "gpu")]
        if self.noiseless.has_gpu_context()
            && self.readout.is_empty()
            && self
                .noiseless
                .should_use_gpu_bts(total_shots.min(chunk_size.max(1)))
        {
            crate::sim::compiled::for_each_chunk(total_shots, chunk_size, |this_batch| {
                let packed = match self.try_sample_bulk_packed_with_gpu_context(this_batch) {
                    Ok(packed) => packed,
                    Err(_) => self.sample_bulk_packed_cpu(this_batch),
                };
                acc.accumulate(&packed);
            });
            return;
        }

        let m_words = self.num_measurements.div_ceil(64);
        let mut accum_buf: Vec<u64> = Vec::new();
        let mut rand_buf: Vec<u8> = Vec::new();
        let ref_bits_packed = self.noiseless.ref_bits_packed().to_vec();
        crate::sim::compiled::for_each_chunk(total_shots, chunk_size, |this_batch| {
            self.noiseless.sample_bulk_words_shot_major_reuse(
                &mut accum_buf,
                &mut rand_buf,
                this_batch,
            );

            self.apply_noise_bulk(&mut accum_buf, this_batch, m_words);

            crate::sim::compiled::xor_mask_shot_major(
                &mut accum_buf,
                this_batch,
                m_words,
                &ref_bits_packed,
            );

            apply_readout_packed(
                &mut accum_buf,
                this_batch,
                m_words,
                &self.readout,
                &mut self.rng,
            );

            let data = std::mem::take(&mut accum_buf);
            let packed = PackedShots::from_shot_major(data, this_batch, self.num_measurements);
            acc.accumulate(&packed);
            accum_buf = packed.into_data();
        });
    }

    /// Sample noisy outcomes and return exact packed counts.
    ///
    /// A GPU failure falls back to the CPU sampler; [`Self::try_sample_counts`]
    /// surfaces it instead.
    pub fn sample_counts(
        &mut self,
        total_shots: usize,
    ) -> std::collections::HashMap<Vec<u64>, u64> {
        match self.try_sample_counts(total_shots) {
            Ok(counts) => counts,
            Err(_) => self.sample_counts_cpu(total_shots),
        }
    }

    pub fn try_sample_counts(
        &mut self,
        total_shots: usize,
    ) -> Result<std::collections::HashMap<Vec<u64>, u64>> {
        #[cfg(feature = "gpu")]
        if let Some(counts) = self.try_sample_counts_gpu(total_shots) {
            return counts;
        }

        Ok(self.sample_counts_cpu(total_shots))
    }

    fn sample_counts_cpu(
        &mut self,
        total_shots: usize,
    ) -> std::collections::HashMap<Vec<u64>, u64> {
        crate::sim::compiled::counts_from_chunks(|acc| self.sample_chunked(total_shots, acc))
    }

    /// Sample noisy outcomes and return per-measurement `P(bit = 1)`.
    pub fn sample_marginals(&mut self, total_shots: usize) -> Vec<f64> {
        match self.try_sample_marginals(total_shots) {
            Ok(marginals) => marginals,
            Err(_) => self.sample_marginals_cpu(total_shots),
        }
    }

    pub fn try_sample_marginals(&mut self, total_shots: usize) -> Result<Vec<f64>> {
        #[cfg(feature = "gpu")]
        if let Some(marginals) = self.try_sample_marginals_gpu(total_shots) {
            return marginals;
        }

        Ok(self.sample_marginals_cpu(total_shots))
    }

    fn sample_marginals_cpu(&mut self, total_shots: usize) -> Vec<f64> {
        crate::sim::compiled::marginals_from_chunks(self.num_measurements, |acc| {
            self.sample_chunked(total_shots, acc)
        })
    }

    #[cfg(test)]
    pub(super) fn sample_bulk(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        self.sample_bulk_packed(num_shots).to_shots()
    }

    fn apply_noise_bulk(&mut self, accum: &mut [u64], num_shots: usize, m_words: usize) {
        if !self.events.is_empty() {
            if self.z_lut.is_some() {
                self.apply_noise_bulk_grouped(accum, num_shots, m_words);
            } else {
                self.apply_noise_bulk_scalar(accum, num_shots, m_words);
            }
        }

        // Two-qubit events run as a second pass over their own table, so the
        // single-qubit pass above keeps the row order the LUT and the device
        // buffers were built against.
        if self.events.has_pairs() {
            self.apply_pair_noise_bulk(accum, num_shots, m_words);
        }
    }

    fn apply_noise_bulk_scalar(&mut self, accum: &mut [u64], num_shots: usize, m_words: usize) {
        #[cfg(feature = "parallel")]
        if num_shots >= 4096 {
            self.apply_range_par(accum, num_shots, m_words, Self::apply_noise_range);
            return;
        }

        self.apply_noise_bulk_scalar_seq(accum, num_shots, m_words);
    }

    fn apply_noise_bulk_scalar_seq(&mut self, accum: &mut [u64], num_shots: usize, m_words: usize) {
        Self::apply_noise_range(&self.events, accum, 0, num_shots, m_words, &mut self.rng);
    }

    fn apply_pair_noise_bulk(&mut self, accum: &mut [u64], num_shots: usize, m_words: usize) {
        #[cfg(feature = "parallel")]
        if num_shots >= 4096 {
            self.apply_range_par(accum, num_shots, m_words, Self::apply_pair_noise_range);
            return;
        }

        Self::apply_pair_noise_range(&self.events, accum, 0, num_shots, m_words, &mut self.rng);
    }

    #[cfg(feature = "parallel")]
    fn apply_range_par(
        &mut self,
        accum: &mut [u64],
        num_shots: usize,
        m_words: usize,
        apply: fn(&FlatNoiseSensitivity, &mut [u64], usize, usize, usize, &mut ChaCha8Rng),
    ) {
        use rayon::prelude::*;

        let num_threads = rayon::current_num_threads().max(1);
        let shots_per_thread = (num_shots.div_ceil(num_threads) + 63) & !63;

        let seeds: Vec<u64> = (0..num_threads)
            .map(|_| rand::RngExt::random(&mut self.rng))
            .collect();

        let events = &self.events;

        accum
            .par_chunks_mut(shots_per_thread * m_words)
            .enumerate()
            .for_each(|(tid, chunk)| {
                let chunk_shots = chunk.len() / m_words;
                if chunk_shots == 0 {
                    return;
                }
                let mut rng = ChaCha8Rng::seed_from_u64(seeds[tid]);
                apply(events, chunk, 0, chunk_shots, m_words, &mut rng);
            });
    }

    fn apply_noise_range(
        events: &FlatNoiseSensitivity,
        accum: &mut [u64],
        start: usize,
        end: usize,
        m_words: usize,
        rng: &mut ChaCha8Rng,
    ) {
        let ne = events.len();
        let num_shots = end - start;

        for i in 0..ne {
            let [px, py, pz] = events.probs[i];
            let p_event = px + py + pz;
            if p_event == 0.0 {
                continue;
            }

            if p_event >= 0.5 || num_shots < 32 {
                for s in start..end {
                    let r: f64 = rand::RngExt::random(rng);
                    if r < px {
                        let b = s * m_words;
                        xor_words(&mut accum[b..b + m_words], events.z_flip(i));
                    } else if r < px + py {
                        let b = s * m_words;
                        xor_words(&mut accum[b..b + m_words], events.x_flip(i));
                        xor_words(&mut accum[b..b + m_words], events.z_flip(i));
                    } else if r < p_event {
                        let b = s * m_words;
                        xor_words(&mut accum[b..b + m_words], events.x_flip(i));
                    }
                }
            } else {
                let ln_1mp = (-p_event).ln_1p();
                let px_frac = px / p_event;
                let pxy_frac = (px + py) / p_event;

                let mut pos = start + geometric_sample(rng, ln_1mp);
                while pos < end {
                    let r: f64 = rand::RngExt::random(rng);
                    let b = pos * m_words;
                    if r < px_frac {
                        xor_words(&mut accum[b..b + m_words], events.z_flip(i));
                    } else if r < pxy_frac {
                        xor_words(&mut accum[b..b + m_words], events.x_flip(i));
                        xor_words(&mut accum[b..b + m_words], events.z_flip(i));
                    } else {
                        xor_words(&mut accum[b..b + m_words], events.x_flip(i));
                    }
                    pos += 1 + geometric_sample(rng, ln_1mp);
                }
            }
        }
    }

    /// Draw the two-qubit events over `[start, end)`. One branch of 15 per
    /// firing shot, thinned exactly as the single-qubit pass thins its own
    /// events.
    fn apply_pair_noise_range(
        events: &FlatNoiseSensitivity,
        accum: &mut [u64],
        start: usize,
        end: usize,
        m_words: usize,
        rng: &mut ChaCha8Rng,
    ) {
        let num_shots = end - start;

        for i in 0..events.pair_probs.len() {
            let p = events.pair_probs[i];
            if p == 0.0 {
                continue;
            }

            if p >= 0.5 || num_shots < 32 {
                for s in start..end {
                    let r: f64 = rand::RngExt::random(rng);
                    if r < p {
                        let b = s * m_words;
                        xor_pair_branch(events, i, r / p, &mut accum[b..b + m_words]);
                    }
                }
            } else {
                let ln_1mp = (-p).ln_1p();
                let mut pos = start + geometric_sample(rng, ln_1mp);
                while pos < end {
                    let r: f64 = rand::RngExt::random(rng);
                    let b = pos * m_words;
                    xor_pair_branch(events, i, r, &mut accum[b..b + m_words]);
                    pos += 1 + geometric_sample(rng, ln_1mp);
                }
            }
        }
    }

    fn apply_noise_bulk_grouped(&mut self, accum: &mut [u64], num_shots: usize, m_words: usize) {
        let num_events = self.events.len();
        let e_words = num_events.div_ceil(64);

        for tile_start in (0..num_shots).step_by(NOISE_LUT_TILE) {
            let tile_end = (tile_start + NOISE_LUT_TILE).min(num_shots);
            let tile_n = tile_end - tile_start;

            let mut z_mask = vec![0u64; tile_n * e_words];
            let mut x_mask = vec![0u64; tile_n * e_words];

            for i in 0..num_events {
                let [px, py, pz] = self.events.probs[i];
                let p_event = px + py + pz;
                if p_event == 0.0 {
                    continue;
                }

                let ew = i / 64;
                let eb = 1u64 << (i % 64);

                if p_event >= 0.5 || tile_n < 32 {
                    for s in 0..tile_n {
                        let r: f64 = rand::RngExt::random(&mut self.rng);
                        if r < px {
                            z_mask[s * e_words + ew] |= eb;
                        } else if r < px + py {
                            z_mask[s * e_words + ew] |= eb;
                            x_mask[s * e_words + ew] |= eb;
                        } else if r < p_event {
                            x_mask[s * e_words + ew] |= eb;
                        }
                    }
                } else {
                    let ln_1mp = (-p_event).ln_1p();
                    let px_frac = px / p_event;
                    let pxy_frac = (px + py) / p_event;

                    let mut pos = geometric_sample(&mut self.rng, ln_1mp);
                    while pos < tile_n {
                        let r: f64 = rand::RngExt::random(&mut self.rng);
                        if r < px_frac {
                            z_mask[pos * e_words + ew] |= eb;
                        } else if r < pxy_frac {
                            z_mask[pos * e_words + ew] |= eb;
                            x_mask[pos * e_words + ew] |= eb;
                        } else {
                            x_mask[pos * e_words + ew] |= eb;
                        }
                        pos += 1 + geometric_sample(&mut self.rng, ln_1mp);
                    }
                }
            }

            let z_lut = self.z_lut.as_ref().unwrap();
            let x_lut = self.x_lut.as_ref().unwrap();

            for s in 0..tile_n {
                let shot_base = (tile_start + s) * m_words;
                let mask_base = s * e_words;
                let shot_accum = &mut accum[shot_base..shot_base + m_words];

                z_lut.apply_masked(shot_accum, &z_mask[mask_base..mask_base + e_words]);
                x_lut.apply_masked(shot_accum, &x_mask[mask_base..mask_base + e_words]);
            }
        }
    }
}

/// Compile a Clifford circuit with Pauli noise into a reusable noisy sampler.
///
/// Requires the same circuit subset as [`crate::compile_measurements`],
/// namely terminal measurements with no resets or classical conditionals.
pub fn compile_noisy(
    circuit: &Circuit,
    noise: &NoiseModel,
    seed: u64,
) -> Result<NoisyCompiledSampler> {
    noise.validate_for(circuit)?;
    compile_validated_noisy(circuit, noise, seed)
}

/// [`compile_noisy`] for a model already checked by [`NoiseModel::validate_for`].
pub(super) fn compile_validated_noisy(
    circuit: &Circuit,
    noise: &NoiseModel,
    seed: u64,
) -> Result<NoisyCompiledSampler> {
    noise.ensure_only_pauli_channels()?;
    noise.ensure_no_guarded_events(circuit, "compiled noisy sampler")?;
    if circuit.num_qubits >= 4 {
        let blocks = circuit.independent_subsystems();
        if blocks.len() > 1 {
            let max_block = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
            if max_block < circuit.num_qubits && pairs_stay_in_one_block(circuit, noise, &blocks) {
                return compile_noisy_filtered(circuit, noise, &blocks, seed);
            }
        }
    }

    compile_noisy_monolithic(circuit, noise, seed)
}

/// True when any measurement is still sensitive to a Pauli on `q` at this
/// point in the backward sweep.
fn sensitive(x_packed: &[Vec<u64>], z_packed: &[Vec<u64>], q: usize) -> bool {
    x_packed[q].iter().any(|&w| w != 0) || z_packed[q].iter().any(|&w| w != 0)
}

/// Scatter a block-local measurement mask into the global measurement order,
/// which is what the sensitivity table is indexed by.
fn scatter_to_global(local: &[u64], bm_list: &[usize], out: &mut [u64]) {
    out.fill(0);
    for (local_mi, &global_mi) in bm_list.iter().enumerate() {
        if (local[local_mi / 64] >> (local_mi % 64)) & 1 != 0 {
            out[global_mi / 64] |= 1u64 << (global_mi % 64);
        }
    }
}

/// True when both qubits of every two-qubit event sit in one subsystem block.
///
/// The filtered compile sweeps one block at a time and holds only that
/// block's propagated masks, so a pair reaching across blocks has nowhere to
/// read its second qubit from and would silently run at the rate of its first
/// qubit alone. Such models take the monolithic compile, which propagates the
/// whole register at once. A zero-rate pair stores no row and constrains
/// nothing.
pub(super) fn pairs_stay_in_one_block(
    circuit: &Circuit,
    noise: &NoiseModel,
    blocks: &[Vec<usize>],
) -> bool {
    let mut qubit_to_block = vec![usize::MAX; circuit.num_qubits];
    for (bi, block) in blocks.iter().enumerate() {
        for &q in block {
            qubit_to_block[q] = bi;
        }
    }

    noise
        .after_gate
        .iter()
        .flatten()
        .filter_map(NoiseEvent::pauli_pair)
        .filter(|&(_, _, p)| p > 0.0)
        .all(|(q0, q1, _)| qubit_to_block[q0] == qubit_to_block[q1])
}

pub(super) fn compile_noisy_filtered(
    circuit: &Circuit,
    noise: &NoiseModel,
    blocks: &[Vec<usize>],
    seed: u64,
) -> Result<NoisyCompiledSampler> {
    let noiseless = compile_measurements(circuit, seed)?;

    let measurement_qubits: Vec<usize> = circuit
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Measure { qubit, .. } => Some(*qubit),
            _ => None,
        })
        .collect();
    let num_measurements = measurement_qubits.len();
    let m_words = num_measurements.div_ceil(64);

    if num_measurements == 0 {
        return Ok(NoisyCompiledSampler {
            noiseless,
            events: FlatNoiseSensitivity::new(1, 0),
            readout: Vec::new(),
            num_measurements: 0,
            rng: ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xA01CE)),
            z_lut: None,
            x_lut: None,
            #[cfg(feature = "gpu")]
            gpu_noise_cache: None,
        });
    }

    let mut qubit_to_block = vec![0usize; circuit.num_qubits];
    let mut qubit_to_local = vec![0usize; circuit.num_qubits];
    for (bi, block) in blocks.iter().enumerate() {
        for (li, &q) in block.iter().enumerate() {
            qubit_to_block[q] = bi;
            qubit_to_local[q] = li;
        }
    }

    let mut block_meas: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
    for (mi, &q) in measurement_qubits.iter().enumerate() {
        block_meas[qubit_to_block[q]].push(mi);
    }

    let total_noise_events: usize = noise.after_gate.iter().map(|ops| ops.len()).sum();
    let mut events = FlatNoiseSensitivity::new(m_words, total_noise_events);
    let mut global_x_buf = vec![0u64; m_words];
    let mut global_z_buf = vec![0u64; m_words];
    let mut pair_x_buf = vec![0u64; m_words];
    let mut pair_z_buf = vec![0u64; m_words];

    for (bi, block) in blocks.iter().enumerate() {
        let bm_list = &block_meas[bi];
        if bm_list.is_empty() {
            continue;
        }

        let bn = block.len();
        let bm = bm_list.len();
        let bm_words = bm.div_ceil(64);

        let mut x_packed: Vec<Vec<u64>> = vec![vec![0u64; bm_words]; bn];
        let mut z_packed: Vec<Vec<u64>> = vec![vec![0u64; bm_words]; bn];
        let mut sign_packed: Vec<u64> = vec![0u64; bm_words];

        // Gates never couple blocks, so an event reaches only the measurements
        // of the block holding its own qubits, whichever instruction's slot
        // carries it.
        let mut local_mi = bm;
        for (instr_idx, inst) in circuit.instructions.iter().enumerate().rev() {
            for event in &noise.after_gate[instr_idx] {
                if let Some((q0, q1, p)) = event.pauli_pair() {
                    if p == 0.0 || qubit_to_block[q0] != bi {
                        continue;
                    }
                    let (l0, l1) = (qubit_to_local[q0], qubit_to_local[q1]);
                    if !sensitive(&x_packed, &z_packed, l0) && !sensitive(&x_packed, &z_packed, l1)
                    {
                        continue;
                    }
                    scatter_to_global(&x_packed[l0], bm_list, &mut global_x_buf);
                    scatter_to_global(&z_packed[l0], bm_list, &mut global_z_buf);
                    scatter_to_global(&x_packed[l1], bm_list, &mut pair_x_buf);
                    scatter_to_global(&z_packed[l1], bm_list, &mut pair_z_buf);
                    events.push_pair(&global_x_buf, &global_z_buf, &pair_x_buf, &pair_z_buf, p);
                    continue;
                }

                let q = event.qubit();
                if qubit_to_block[q] != bi {
                    continue;
                }
                let (px, py, pz) = event.pauli_probs();
                let local_q = qubit_to_local[q];

                if sensitive(&x_packed, &z_packed, local_q) {
                    scatter_to_global(&x_packed[local_q], bm_list, &mut global_x_buf);
                    scatter_to_global(&z_packed[local_q], bm_list, &mut global_z_buf);
                    events.push(&global_x_buf, &global_z_buf, px, py, pz);
                }
            }

            match inst {
                Instruction::Gate { gate, targets }
                | Instruction::Conditional { gate, targets, .. }
                    if targets.iter().all(|&t| qubit_to_block[t] == bi) =>
                {
                    let local_targets: SmallVec<[usize; 4]> =
                        targets.iter().map(|&t| qubit_to_local[t]).collect();
                    batch_propagate_backward(
                        &mut x_packed,
                        &mut z_packed,
                        &mut sign_packed,
                        gate,
                        local_targets.as_slice(),
                        bm_words,
                    );
                }
                Instruction::Measure { qubit, .. } if qubit_to_block[*qubit] == bi => {
                    local_mi -= 1;
                    z_packed[qubit_to_local[*qubit]][local_mi / 64] |= 1u64 << (local_mi % 64);
                }
                _ => {}
            }
        }
    }

    let (z_lut, x_lut) = build_noise_luts(&events);

    Ok(NoisyCompiledSampler {
        noiseless,
        events,
        readout: record_readout(circuit, noise),
        num_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xCAFE_BABE)),
        z_lut,
        x_lut,
        #[cfg(feature = "gpu")]
        gpu_noise_cache: None,
    })
}

pub(super) fn compile_noisy_monolithic(
    circuit: &Circuit,
    noise: &NoiseModel,
    seed: u64,
) -> Result<NoisyCompiledSampler> {
    let noiseless = compile_measurements(circuit, seed)?;

    let n = circuit.num_qubits;

    let num_measurements = circuit
        .instructions
        .iter()
        .filter(|inst| matches!(inst, Instruction::Measure { .. }))
        .count();
    let m_words = num_measurements.div_ceil(64);

    if num_measurements == 0 {
        return Ok(NoisyCompiledSampler {
            noiseless,
            events: FlatNoiseSensitivity::new(m_words, 0),
            readout: Vec::new(),
            num_measurements: 0,
            rng: ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xA01CE)),
            z_lut: None,
            x_lut: None,
            #[cfg(feature = "gpu")]
            gpu_noise_cache: None,
        });
    }

    let mut x_packed: Vec<Vec<u64>> = vec![vec![0u64; m_words]; n];
    let mut z_packed: Vec<Vec<u64>> = vec![vec![0u64; m_words]; n];
    let mut sign_packed: Vec<u64> = vec![0u64; m_words];

    let total_noise_events: usize = noise.after_gate.iter().map(|ops| ops.len()).sum();
    let mut events = FlatNoiseSensitivity::new(m_words, total_noise_events);

    // A measurement turns sensitive only once the backward sweep reaches it,
    // so an event in a slot between two measurements reaches the later one
    // alone.
    let mut mi = num_measurements;
    for (instr_idx, inst) in circuit.instructions.iter().enumerate().rev() {
        for event in &noise.after_gate[instr_idx] {
            if let Some((q0, q1, p)) = event.pauli_pair() {
                if p == 0.0 {
                    continue;
                }
                if sensitive(&x_packed, &z_packed, q0) || sensitive(&x_packed, &z_packed, q1) {
                    events.push_pair(
                        &x_packed[q0],
                        &z_packed[q0],
                        &x_packed[q1],
                        &z_packed[q1],
                        p,
                    );
                }
                continue;
            }

            let (px, py, pz) = event.pauli_probs();
            let q = event.qubit();

            if sensitive(&x_packed, &z_packed, q) {
                events.push(&x_packed[q], &z_packed[q], px, py, pz);
            }
        }

        match inst {
            Instruction::Gate { gate, targets }
            | Instruction::Conditional { gate, targets, .. } => {
                batch_propagate_backward(
                    &mut x_packed,
                    &mut z_packed,
                    &mut sign_packed,
                    gate,
                    targets.as_slice(),
                    m_words,
                );
            }
            Instruction::Measure { qubit, .. } => {
                mi -= 1;
                z_packed[*qubit][mi / 64] |= 1u64 << (mi % 64);
            }
            _ => {}
        }
    }

    let (z_lut, x_lut) = build_noise_luts(&events);

    Ok(NoisyCompiledSampler {
        noiseless,
        events,
        readout: record_readout(circuit, noise),
        num_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xCAFE_BABE)),
        z_lut,
        x_lut,
        #[cfg(feature = "gpu")]
        gpu_noise_cache: None,
    })
}

pub(super) fn finish_noisy_compiled_run(
    mut sampler: NoisyCompiledSampler,
    circuit: &Circuit,
    num_shots: usize,
) -> Result<ShotsResult> {
    let classical_bit_order = circuit.classical_bit_order();
    let num_classical = circuit.num_classical_bits;

    let packed = sampler.try_sample_bulk_packed(num_shots)?;
    let shots = unpack_and_remap_packed(&packed, num_shots, &classical_bit_order, num_classical);

    Ok(
        ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(
            crate::sim::RunMetadata::exact(crate::sim::ResolvedBackend::CompiledStabilizer)
                .with_engine(crate::sim::Engine::NoisyCompiledSampler)
                .with_reason(crate::sim::RouteRule::CompiledCliffordSampler),
        ),
    )
}

pub(super) fn run_shots_noisy_compiled(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let sampler = compile_validated_noisy(circuit, noise, seed)?;
    finish_noisy_compiled_run(sampler, circuit, num_shots)
}

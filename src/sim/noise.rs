use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::smallvec;

use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::Result;
use crate::gates::Gate;
use crate::sim::compiled::{
    batch_propagate_backward, compile_measurements, xor_words, PackedShots,
};
use crate::sim::ShotsResult;

/// A single-qubit (or two-qubit, for `TwoQubitDepolarizing`) noise channel.
///
/// Channels are applied by the trajectory engine after each gate whose
/// `NoiseModel::after_gate` entry contains a matching `NoiseEvent`. For
/// Pauli-only noise on Clifford circuits, the compiled stabilizer sampler
/// is used instead, see [`NoiseModel::is_pauli_only`].
#[derive(Debug, Clone)]
pub enum NoiseChannel {
    /// Independent Pauli X/Y/Z error with given per-branch probabilities.
    Pauli { px: f64, py: f64, pz: f64 },
    /// Symmetric depolarizing: `I` with probability `1-p`, each of `X/Y/Z`
    /// with probability `p/3`.
    Depolarizing { p: f64 },
    /// T1 relaxation: `|1⟩ → |0⟩` with amplitude `sqrt(gamma)`.
    AmplitudeDamping { gamma: f64 },
    /// Pure dephasing: phase randomisation with amplitude `sqrt(gamma)` on the
    /// `|1⟩` state. Populations are preserved.
    PhaseDamping { gamma: f64 },
    /// Combined T1 + T2 relaxation over a gate of duration `gate_time`.
    /// Sampled as a population reset (prob. `1 - exp(-gate_time/t1)`) followed
    /// by a pure-dephasing branch when `t2 < 2·t1`.
    ThermalRelaxation { t1: f64, t2: f64, gate_time: f64 },
    /// Symmetric two-qubit depolarizing: each of the 15 non-identity
    /// two-qubit Paulis occurs with probability `p/15`.
    TwoQubitDepolarizing { p: f64 },
    /// General single-qubit channel described by a set of Kraus operators.
    ///
    /// **Restriction: supported Kraus class.** Each operator `K_k` must
    /// satisfy that `K_k† K_k` is diagonal. This class is rich enough to
    /// cover all diagonal channels (amplitude / phase damping, dephasing)
    /// plus the Pauli channels (bit flip, phase flip, bit-phase flip, and
    /// arbitrary mixtures of `X`, `Y`, `Z`), which is every standard
    /// single-qubit noise channel in the literature.
    ///
    /// **Rejected Kraus class.** General dense operators such as
    /// `[[a, b], [0, 0]]` whose `K_k† K_k` has non-zero off-diagonal entries
    /// require access to the qubit's coherence `⟨0|ρ|1⟩`, which the
    /// trajectory engine cannot extract from a pure state without cloning
    /// the backend. If you pass such an operator, the trajectory engine
    /// returns [`PrismError::BackendUnsupported`] at runtime.
    ///
    /// **Workarounds for dense Kraus ops:** decompose the channel into a
    /// unitary dilation plus an ancilla measurement, or use a density-matrix
    /// simulator (not provided by PRISM-Q).
    ///
    /// [`PrismError::BackendUnsupported`]: crate::error::PrismError::BackendUnsupported
    Custom { kraus: Vec<[[Complex64; 2]; 2]> },
}

impl NoiseChannel {
    pub fn as_pauli(&self) -> Option<(f64, f64, f64)> {
        match self {
            NoiseChannel::Pauli { px, py, pz } => Some((*px, *py, *pz)),
            NoiseChannel::Depolarizing { p } => {
                let pp = p / 3.0;
                Some((pp, pp, pp))
            }
            _ => None,
        }
    }

    pub fn is_pauli(&self) -> bool {
        matches!(
            self,
            NoiseChannel::Pauli { .. } | NoiseChannel::Depolarizing { .. }
        )
    }
}

#[derive(Debug, Clone)]
pub struct NoiseEvent {
    pub channel: NoiseChannel,
    pub qubits: SmallVec<[usize; 2]>,
}

impl NoiseEvent {
    pub fn pauli(qubit: usize, px: f64, py: f64, pz: f64) -> Self {
        Self {
            channel: NoiseChannel::Pauli { px, py, pz },
            qubits: smallvec![qubit],
        }
    }

    pub fn qubit(&self) -> usize {
        self.qubits[0]
    }

    /// Return the per-Pauli probabilities for a Pauli/Depolarizing channel.
    ///
    /// # Panics
    ///
    /// Panics if the channel is not Pauli or Depolarizing. Callers on the
    /// compiled stabilizer sampler path must guard with
    /// [`NoiseModel::ensure_pauli_only`] before invoking this method.
    pub fn pauli_probs(&self) -> (f64, f64, f64) {
        self.channel
            .as_pauli()
            .expect("pauli_probs called on non-Pauli channel; caller must use ensure_pauli_only")
    }
}

#[derive(Debug, Clone)]
pub struct ReadoutError {
    pub p01: f64,
    pub p10: f64,
}

pub struct NoiseModel {
    pub after_gate: Vec<Vec<NoiseEvent>>,
    pub readout: Vec<Option<ReadoutError>>,
}

impl NoiseModel {
    pub fn uniform_depolarizing(circuit: &Circuit, p: f64) -> Self {
        let px = p / 3.0;
        let py = p / 3.0;
        let pz = p / 3.0;

        let mut after_gate = Vec::with_capacity(circuit.instructions.len());
        for instr in &circuit.instructions {
            match instr {
                Instruction::Gate { targets, .. } => {
                    let ops: Vec<NoiseEvent> = targets
                        .iter()
                        .map(|&q| NoiseEvent::pauli(q, px, py, pz))
                        .collect();
                    after_gate.push(ops);
                }
                _ => {
                    after_gate.push(Vec::new());
                }
            }
        }

        Self {
            after_gate,
            readout: vec![None; circuit.num_classical_bits],
        }
    }

    pub fn with_amplitude_damping(circuit: &Circuit, gamma: f64) -> Self {
        let mut after_gate = Vec::with_capacity(circuit.instructions.len());
        for instr in &circuit.instructions {
            match instr {
                Instruction::Gate { targets, .. } => {
                    let ops: Vec<NoiseEvent> = targets
                        .iter()
                        .map(|&q| NoiseEvent {
                            channel: NoiseChannel::AmplitudeDamping { gamma },
                            qubits: smallvec![q],
                        })
                        .collect();
                    after_gate.push(ops);
                }
                _ => {
                    after_gate.push(Vec::new());
                }
            }
        }

        Self {
            after_gate,
            readout: vec![None; circuit.num_classical_bits],
        }
    }

    pub fn with_readout_error(&mut self, p01: f64, p10: f64) -> &mut Self {
        self.readout.fill(Some(ReadoutError { p01, p10 }));
        self
    }

    pub fn is_pauli_only(&self) -> bool {
        self.after_gate
            .iter()
            .flat_map(|events| events.iter())
            .all(|e| e.channel.is_pauli())
            && self.readout.iter().all(|r| r.is_none())
    }

    pub fn has_noise(&self) -> bool {
        self.after_gate.iter().any(|events| !events.is_empty())
            || self.readout.iter().any(|r| r.is_some())
    }

    /// Return an error if the noise model contains any non-Pauli channels
    /// or readout errors. Used as a precondition check by the compiled
    /// sampler path, which can only handle Pauli noise on Clifford circuits.
    pub fn ensure_pauli_only(&self) -> Result<()> {
        if !self.is_pauli_only() {
            return Err(crate::error::PrismError::IncompatibleBackend {
                backend: "compiled stabilizer sampler".into(),
                reason: "non-Pauli noise channels (amplitude damping, phase damping, \
                         thermal relaxation, custom Kraus) or readout errors require \
                         the trajectory engine; use a non-stabilizer backend or a \
                         Pauli-only noise model"
                    .into(),
            });
        }
        Ok(())
    }
}

struct FlatNoiseSensitivity {
    x_data: Vec<u64>,
    z_data: Vec<u64>,
    probs: Vec<[f64; 3]>,
    m_words: usize,
}

impl FlatNoiseSensitivity {
    fn new(m_words: usize, capacity: usize) -> Self {
        Self {
            x_data: Vec::with_capacity(capacity * m_words),
            z_data: Vec::with_capacity(capacity * m_words),
            probs: Vec::with_capacity(capacity),
            m_words,
        }
    }

    fn push(&mut self, x_flip: &[u64], z_flip: &[u64], px: f64, py: f64, pz: f64) {
        self.x_data.extend_from_slice(x_flip);
        self.z_data.extend_from_slice(z_flip);
        self.probs.push([px, py, pz]);
    }

    #[inline(always)]
    fn len(&self) -> usize {
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

#[inline(always)]
fn geometric_sample(rng: &mut ChaCha8Rng, ln_1mp: f64) -> usize {
    let u: f64 = 1.0 - rand::Rng::random::<f64>(rng);
    (u.ln() / ln_1mp) as usize
}

#[cfg(feature = "gpu")]
fn geometric_sample_xoshiro(
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
    #[inline]
    fn uniform(&mut self) -> f64 {
        rand::Rng::random::<f64>(self)
    }
    #[inline]
    fn geom(&mut self, ln_1mp: f64) -> usize {
        geometric_sample(self, ln_1mp)
    }
}

#[cfg(feature = "gpu")]
impl EventRng for crate::sim::compiled::rng::Xoshiro256PlusPlus {
    #[inline]
    fn uniform(&mut self) -> f64 {
        self.next_f64()
    }
    #[inline]
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
    rng: &mut dyn EventRng,
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
    let ln_1mp = (1.0 - p_event).ln();
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

fn unpack_and_remap(
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

struct NoiseFlipLut {
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

/// Pack `(px, py, pz)` per event into three `u64` thresholds for the fused
/// noise kernel. Each threshold is `p * 2^64` rounded down, so a uniform
/// `u64` random compares below it with probability `p`. Returns the flat
/// buffer along with `false` when any event's probabilities are out of range
/// (negative, NaN, or sum greater than 1). The caller then falls back to the
/// host mask path.
#[cfg(feature = "gpu")]
fn flatten_event_thresholds_u64(probs: &[[f64; 3]]) -> (Vec<u64>, bool) {
    // Nearest f64 below 2^64. Scaling by 2^64 exactly rounds to 2^64 which
    // doesn't fit in a u64; this preserves `u64_rand < threshold` at the
    // intended probability.
    const SCALE: f64 = 18446744073709549568.0_f64;
    const PROB_SUM_SLACK: f64 = 1e-12;

    let mut out = Vec::with_capacity(probs.len() * 3);
    let mut valid = true;
    for &[px, py, pz] in probs {
        let finite = px.is_finite() && py.is_finite() && pz.is_finite();
        let sum = px + py + pz;
        if !finite || px < 0.0 || py < 0.0 || pz < 0.0 || sum > 1.0 + PROB_SUM_SLACK {
            valid = false;
        }
        let to_u64 = |p: f64| -> u64 {
            if !p.is_finite() || p <= 0.0 {
                0
            } else if p >= 1.0 {
                u64::MAX
            } else {
                (p * SCALE) as u64
            }
        };
        out.push(to_u64(px));
        out.push(to_u64(px + py));
        out.push(to_u64(sum));
    }
    (out, valid)
}

/// Transpose the event-major `(x_data, z_data)` bitmaps in a
/// `FlatNoiseSensitivity` into a row-major CSR. Each row's entry list names
/// the events that touch it, packed as `event << 2 | flag`: flag bit 0 for
/// an X contribution, bit 1 for Z, both set for a Y contribution. Lets the
/// per-(row, batch) kernel accumulate XORs in a register without atomics.
/// Paired with `bts_generate_and_apply_noise_meas_major_by_row`.
#[cfg(feature = "gpu")]
fn transpose_events_to_row_major(
    events: &FlatNoiseSensitivity,
    num_meas: usize,
) -> (Vec<u32>, Vec<u32>) {
    let num_events = events.len();
    let m_words = events.m_words;

    // First pass: count per-row entries.
    let mut row_counts = vec![0u32; num_meas];
    for e in 0..num_events {
        let x = &events.x_data[e * m_words..(e + 1) * m_words];
        let z = &events.z_data[e * m_words..(e + 1) * m_words];
        for w in 0..m_words {
            let union = x[w] | z[w];
            let mut bits = union;
            while bits != 0 {
                let row = w * 64 + bits.trailing_zeros() as usize;
                if row < num_meas {
                    row_counts[row] += 1;
                }
                bits &= bits - 1;
            }
        }
    }

    let mut offsets = Vec::with_capacity(num_meas + 1);
    offsets.push(0u32);
    let mut running = 0u32;
    for c in &row_counts {
        running += c;
        offsets.push(running);
    }
    let total = running as usize;

    // Second pass: write entries. Use a fresh cursor array because
    // `row_counts` is still needed to validate; cheaper to just reuse it as
    // a write cursor starting at `offsets[row]`.
    let mut entries = vec![0u32; total];
    let mut cursor: Vec<u32> = offsets[..num_meas].to_vec();
    for e in 0..num_events {
        let x = &events.x_data[e * m_words..(e + 1) * m_words];
        let z = &events.z_data[e * m_words..(e + 1) * m_words];
        for w in 0..m_words {
            let union = x[w] | z[w];
            let mut bits = union;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let row = w * 64 + b;
                if row >= num_meas {
                    continue;
                }
                let bit_mask = 1u64 << b;
                let flag: u32 = (if x[w] & bit_mask != 0 { 1 } else { 0 })
                    | (if z[w] & bit_mask != 0 { 2 } else { 0 });
                let slot = cursor[row] as usize;
                entries[slot] = ((e as u32) << 2) | flag;
                cursor[row] += 1;
            }
        }
    }

    (offsets, entries)
}

#[cfg(feature = "gpu")]
fn flatten_event_rows_to_csr(
    flat_data: &[u64],
    num_rows: usize,
    m_words: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut row_offsets = Vec::with_capacity(num_rows + 1);
    let mut row_indices = Vec::new();

    for row in 0..num_rows {
        row_offsets.push(row_indices.len() as u32);
        let base = row * m_words;
        for word in 0..m_words {
            let mut bits = flat_data[base + word];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                row_indices.push((word * 64 + bit) as u32);
                bits &= bits - 1;
            }
        }
    }
    row_offsets.push(row_indices.len() as u32);

    (row_offsets, row_indices)
}

#[cfg(feature = "gpu")]
struct GpuNoiseCache {
    context: std::sync::Arc<crate::gpu::GpuContext>,
    x_row_offsets_dev: crate::gpu::GpuBuffer<u32>,
    x_row_indices_dev: crate::gpu::GpuBuffer<u32>,
    z_row_offsets_dev: crate::gpu::GpuBuffer<u32>,
    z_row_indices_dev: crate::gpu::GpuBuffer<u32>,
    x_masks_dev: crate::gpu::GpuBuffer<u64>,
    z_masks_dev: crate::gpu::GpuBuffer<u64>,
    x_masks_host: Vec<u64>,
    z_masks_host: Vec<u64>,
    /// Precomputed per-event thresholds for the fused generator. Three u64s
    /// per event hold `[px, px+py, px+py+pz]` scaled to `2^64`, so the
    /// kernel's inner loop replaces a floating-point multiply and branch per
    /// shot with a `u64 <` compare. Stays resident across sampling calls.
    event_thresholds_dev: crate::gpu::GpuBuffer<u64>,
    /// True when every event's probabilities can be represented as a `u64`
    /// threshold. A NaN or a sum greater than 1 flips this to false and
    /// routes the caller to the host mask path.
    event_thresholds_valid: bool,
    /// Row-major event CSR for the per-(row, batch) fused kernel. Each entry
    /// packs `event << 2 | flag`, where bit 0 means the event contributes X
    /// to this row and bit 1 means Z. The row-major layout lets one thread
    /// accumulate every XOR for its output word in a register instead of
    /// broadcasting `(event, batch)` writes across rows with atomics.
    row_event_offsets_dev: crate::gpu::GpuBuffer<u32>,
    row_event_entries_dev: crate::gpu::GpuBuffer<u32>,
}

#[cfg(feature = "gpu")]
impl GpuNoiseCache {
    fn new(
        context: std::sync::Arc<crate::gpu::GpuContext>,
        events: &FlatNoiseSensitivity,
    ) -> Result<Self> {
        let device = context.device();
        let num_events = events.len();
        let (x_row_offsets, x_row_indices) =
            flatten_event_rows_to_csr(&events.x_data, num_events, events.m_words);
        let (z_row_offsets, z_row_indices) =
            flatten_event_rows_to_csr(&events.z_data, num_events, events.m_words);

        let mut x_row_offsets_dev =
            crate::gpu::GpuBuffer::<u32>::alloc_zeros(device, x_row_offsets.len().max(1))?;
        let mut x_row_indices_dev =
            crate::gpu::GpuBuffer::<u32>::alloc_zeros(device, x_row_indices.len().max(1))?;
        let mut z_row_offsets_dev =
            crate::gpu::GpuBuffer::<u32>::alloc_zeros(device, z_row_offsets.len().max(1))?;
        let mut z_row_indices_dev =
            crate::gpu::GpuBuffer::<u32>::alloc_zeros(device, z_row_indices.len().max(1))?;

        if !x_row_offsets.is_empty() {
            x_row_offsets_dev.copy_from_host(device, &x_row_offsets)?;
        }
        if !x_row_indices.is_empty() {
            x_row_indices_dev.copy_from_host(device, &x_row_indices)?;
        }
        if !z_row_offsets.is_empty() {
            z_row_offsets_dev.copy_from_host(device, &z_row_offsets)?;
        }
        if !z_row_indices.is_empty() {
            z_row_indices_dev.copy_from_host(device, &z_row_indices)?;
        }

        let (thresholds_host, thresholds_valid) = flatten_event_thresholds_u64(&events.probs);
        let mut event_thresholds_dev =
            crate::gpu::GpuBuffer::<u64>::alloc_zeros(device, thresholds_host.len().max(1))?;
        if !thresholds_host.is_empty() {
            event_thresholds_dev.copy_from_host(device, &thresholds_host)?;
        }

        // Row-major transpose for the per-(row, batch) fused kernel. Cheap
        // bit-scan over events.x_data | events.z_data at cache init time.
        let (row_event_offsets_host, row_event_entries_host) =
            transpose_events_to_row_major(events, events.m_words * 64);
        let mut row_event_offsets_dev =
            crate::gpu::GpuBuffer::<u32>::alloc_zeros(device, row_event_offsets_host.len().max(1))?;
        let mut row_event_entries_dev =
            crate::gpu::GpuBuffer::<u32>::alloc_zeros(device, row_event_entries_host.len().max(1))?;
        if !row_event_offsets_host.is_empty() {
            row_event_offsets_dev.copy_from_host(device, &row_event_offsets_host)?;
        }
        if !row_event_entries_host.is_empty() {
            row_event_entries_dev.copy_from_host(device, &row_event_entries_host)?;
        }

        Ok(Self {
            context: context.clone(),
            x_row_offsets_dev,
            x_row_indices_dev,
            z_row_offsets_dev,
            z_row_indices_dev,
            x_masks_dev: crate::gpu::GpuBuffer::<u64>::alloc_zeros(device, 1)?,
            z_masks_dev: crate::gpu::GpuBuffer::<u64>::alloc_zeros(device, 1)?,
            x_masks_host: Vec::new(),
            z_masks_host: Vec::new(),
            event_thresholds_dev,
            event_thresholds_valid: thresholds_valid,
            row_event_offsets_dev,
            row_event_entries_dev,
        })
    }

    fn ensure_mask_capacity(&mut self, len: usize) -> Result<()> {
        let needed = len.max(1);
        let device = self.context.device();
        if self.x_masks_dev.len() < needed {
            self.x_masks_dev = crate::gpu::GpuBuffer::<u64>::alloc_zeros(device, needed)?;
        }
        if self.z_masks_dev.len() < needed {
            self.z_masks_dev = crate::gpu::GpuBuffer::<u64>::alloc_zeros(device, needed)?;
        }
        if self.x_masks_host.len() < needed {
            self.x_masks_host.resize(needed, 0);
        }
        if self.z_masks_host.len() < needed {
            self.z_masks_host.resize(needed, 0);
        }
        Ok(())
    }
}

/// Compiled sampler for Clifford circuits with Pauli noise.
///
/// The sampler reuses the compiled measurement parity map for the noiseless
/// circuit, then applies the requested Pauli noise model across batches of
/// shots. When a GPU context is attached with [`Self::with_gpu`], the
/// underlying parity sampling stage may use the GPU BTS path. Materialized
/// shot output still returns to the CPU in shot-major form, while exact
/// counts and marginals can keep the packed measurement-major buffer on the
/// device through noise application and reduction.
pub struct NoisyCompiledSampler {
    noiseless: crate::sim::compiled::CompiledSampler,
    events: FlatNoiseSensitivity,
    num_measurements: usize,
    rng: ChaCha8Rng,
    z_lut: Option<NoiseFlipLut>,
    x_lut: Option<NoiseFlipLut>,
    #[cfg(feature = "gpu")]
    gpu_noise_cache: Option<GpuNoiseCache>,
}

impl NoisyCompiledSampler {
    /// Opt in to GPU-accelerated sampling for the underlying compiled
    /// noiseless sampler.
    ///
    /// Small shot batches may continue to use the CPU path when the compiled
    /// sampler stays below the GPU BTS threshold. Noise masks are still
    /// generated on the CPU; counts and marginals can then apply them and
    /// reduce on the device.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, context: std::sync::Arc<crate::gpu::GpuContext>) -> Self {
        self.noiseless = self.noiseless.with_gpu(context);
        self.gpu_noise_cache = None;
        self
    }

    fn sample_bulk_words_shot_major_cpu(&mut self, num_shots: usize) -> (Vec<u64>, usize) {
        let (mut accum, m_words) = self.noiseless.sample_bulk_words_shot_major(num_shots);
        self.apply_noise_bulk(&mut accum, num_shots, m_words);

        let ref_bits_packed = self.noiseless.ref_bits_packed();

        #[cfg(feature = "parallel")]
        if num_shots >= 256 {
            use rayon::prelude::*;
            accum
                .par_chunks_mut(m_words)
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
        GpuNoiseCache::new(context, &self.events)
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
            use rand::RngCore;
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
            rand::RngCore::next_u64(&mut self.rng)
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

    #[cfg(feature = "gpu")]
    fn try_sample_marginals_gpu(&mut self, total_shots: usize) -> Option<Result<Vec<f64>>> {
        if !self.noiseless.should_use_gpu_bts(total_shots) {
            return None;
        }

        let mut packed = match self.noiseless.sample_bulk_packed_device(total_shots) {
            Ok(packed) => packed,
            Err(e) => return Some(Err(e)),
        };
        if let Err(e) = self.apply_noise_device_meas_major(&mut packed) {
            return Some(Err(e));
        }
        Some(packed.marginals())
    }

    #[cfg(feature = "gpu")]
    fn try_sample_counts_gpu(
        &mut self,
        total_shots: usize,
    ) -> Option<Result<std::collections::HashMap<Vec<u64>, u64>>> {
        if !self.noiseless.should_use_gpu_bts(total_shots) {
            return None;
        }

        let mut packed = match self.noiseless.sample_bulk_packed_device(total_shots) {
            Ok(packed) => packed,
            Err(e) => return Some(Err(e)),
        };
        if let Err(e) = self.apply_noise_device_meas_major(&mut packed) {
            return Some(Err(e));
        }
        Some(packed.counts_with_rank_hint(self.num_measurements))
    }

    /// Sample `num_shots` noisy outcomes and return them in packed form.
    ///
    /// The returned [`PackedShots`] value uses shot-major layout. With a GPU
    /// context attached, the underlying parity sampling stage may run on the
    /// GPU before noise is applied and the packed output is materialized on
    /// the CPU.
    pub fn sample_bulk_packed(&mut self, num_shots: usize) -> PackedShots {
        match self.try_sample_bulk_packed(num_shots) {
            Ok(packed) => packed,
            Err(_) => self.sample_bulk_packed_cpu(num_shots),
        }
    }

    pub fn try_sample_bulk_packed(&mut self, num_shots: usize) -> Result<PackedShots> {
        #[cfg(feature = "gpu")]
        if self.noiseless.has_gpu_context() {
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

    /// Stream noisy shots into `acc` using the default chunk size.
    ///
    /// This avoids materializing the full shot matrix when the caller only
    /// needs derived aggregates such as counts or marginals.
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
    /// When a GPU context is attached, each chunk may use GPU BTS sampling for
    /// the noiseless parity stage before noise is applied.
    pub fn sample_chunked_with_size<A: crate::sim::compiled::ShotAccumulator>(
        &mut self,
        total_shots: usize,
        chunk_size: usize,
        acc: &mut A,
    ) {
        #[cfg(feature = "gpu")]
        if self.noiseless.has_gpu_context()
            && self
                .noiseless
                .should_use_gpu_bts(total_shots.min(chunk_size.max(1)))
        {
            let mut remaining = total_shots;
            while remaining > 0 {
                let this_batch = remaining.min(chunk_size);
                let packed = match self.try_sample_bulk_packed_with_gpu_context(this_batch) {
                    Ok(packed) => packed,
                    Err(_) => self.sample_bulk_packed_cpu(this_batch),
                };
                acc.accumulate(&packed);
                remaining -= this_batch;
            }
            return;
        }

        let m_words = self.num_measurements.div_ceil(64);
        let mut accum_buf: Vec<u64> = Vec::new();
        let mut rand_buf: Vec<u8> = Vec::new();
        let ref_bits_packed = self.noiseless.ref_bits_packed().to_vec();
        let mut remaining = total_shots;
        while remaining > 0 {
            let this_batch = remaining.min(chunk_size);
            self.noiseless.sample_bulk_words_shot_major_reuse(
                &mut accum_buf,
                &mut rand_buf,
                this_batch,
            );

            self.apply_noise_bulk(&mut accum_buf, this_batch, m_words);

            #[cfg(feature = "parallel")]
            if this_batch >= 256 {
                use rayon::prelude::*;
                accum_buf[..this_batch * m_words]
                    .par_chunks_mut(m_words)
                    .for_each(|shot| xor_words(shot, &ref_bits_packed));
            } else {
                for s in 0..this_batch {
                    let shot_base = s * m_words;
                    xor_words(
                        &mut accum_buf[shot_base..shot_base + m_words],
                        &ref_bits_packed,
                    );
                }
            }

            #[cfg(not(feature = "parallel"))]
            for s in 0..this_batch {
                let shot_base = s * m_words;
                xor_words(
                    &mut accum_buf[shot_base..shot_base + m_words],
                    &ref_bits_packed,
                );
            }

            let data = std::mem::take(&mut accum_buf);
            let packed = PackedShots::from_shot_major(data, this_batch, self.num_measurements);
            acc.accumulate(&packed);
            accum_buf = packed.into_data();
            remaining -= this_batch;
        }
    }

    /// Sample noisy outcomes and return exact packed counts.
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
        let mut acc = crate::sim::compiled::HistogramAccumulator::new();
        self.sample_chunked(total_shots, &mut acc);
        acc.into_counts()
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
        let mut acc = crate::sim::compiled::MarginalsAccumulator::new(self.num_measurements);
        self.sample_chunked(total_shots, &mut acc);
        acc.marginals()
    }

    #[cfg(test)]
    fn sample_bulk(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        self.sample_bulk_packed(num_shots).to_shots()
    }

    fn apply_noise_bulk(&mut self, accum: &mut [u64], num_shots: usize, m_words: usize) {
        if self.events.is_empty() {
            return;
        }

        if self.z_lut.is_some() {
            self.apply_noise_bulk_grouped(accum, num_shots, m_words);
        } else {
            self.apply_noise_bulk_scalar(accum, num_shots, m_words);
        }
    }

    fn apply_noise_bulk_scalar(&mut self, accum: &mut [u64], num_shots: usize, m_words: usize) {
        #[cfg(feature = "parallel")]
        if num_shots >= 4096 {
            self.apply_noise_bulk_scalar_par(accum, num_shots, m_words);
            return;
        }

        self.apply_noise_bulk_scalar_seq(accum, num_shots, m_words);
    }

    fn apply_noise_bulk_scalar_seq(&mut self, accum: &mut [u64], num_shots: usize, m_words: usize) {
        Self::apply_noise_range(&self.events, accum, 0, num_shots, m_words, &mut self.rng);
    }

    #[cfg(feature = "parallel")]
    fn apply_noise_bulk_scalar_par(&mut self, accum: &mut [u64], num_shots: usize, m_words: usize) {
        use rayon::prelude::*;

        let num_threads = rayon::current_num_threads().max(1);
        let shots_per_thread = (num_shots.div_ceil(num_threads) + 63) & !63;

        let seeds: Vec<u64> = (0..num_threads)
            .map(|_| rand::Rng::random(&mut self.rng))
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
                Self::apply_noise_range(events, chunk, 0, chunk_shots, m_words, &mut rng);
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
                    let r: f64 = rand::Rng::random(rng);
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
                let ln_1mp = (1.0 - p_event).ln();
                let px_frac = px / p_event;
                let pxy_frac = (px + py) / p_event;

                let mut pos = start + geometric_sample(rng, ln_1mp);
                while pos < end {
                    let r: f64 = rand::Rng::random(rng);
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
                        let r: f64 = rand::Rng::random(&mut self.rng);
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
                    let ln_1mp = (1.0 - p_event).ln();
                    let px_frac = px / p_event;
                    let pxy_frac = (px + py) / p_event;

                    let mut pos = geometric_sample(&mut self.rng, ln_1mp);
                    while pos < tile_n {
                        let r: f64 = rand::Rng::random(&mut self.rng);
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
    noise.ensure_pauli_only()?;
    if circuit.num_qubits >= 4 {
        let blocks = circuit.independent_subsystems();
        if blocks.len() > 1 {
            let max_block = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
            if max_block < circuit.num_qubits {
                return compile_noisy_filtered(circuit, noise, &blocks, seed);
            }
        }
    }

    compile_noisy_monolithic(circuit, noise, seed)
}

fn compile_noisy_filtered(
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

        for (local_mi, &global_mi) in bm_list.iter().enumerate() {
            let q = measurement_qubits[global_mi];
            let local_q = qubit_to_local[q];
            z_packed[local_q][local_mi / 64] |= 1u64 << (local_mi % 64);
        }

        let block_gates: Vec<(usize, &Gate, SmallVec<[usize; 4]>)> = circuit
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(idx, inst)| match inst {
                Instruction::Gate { gate, targets } => {
                    if targets.iter().all(|&t| qubit_to_block[t] == bi) {
                        let local_targets: SmallVec<[usize; 4]> =
                            targets.iter().map(|&t| qubit_to_local[t]).collect();
                        Some((idx, gate, local_targets))
                    } else {
                        None
                    }
                }
                Instruction::Conditional { gate, targets, .. } => {
                    if targets.iter().all(|&t| qubit_to_block[t] == bi) {
                        let local_targets: SmallVec<[usize; 4]> =
                            targets.iter().map(|&t| qubit_to_local[t]).collect();
                        Some((idx, gate, local_targets))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();

        for (instr_idx, gate, local_targets) in block_gates.iter().rev() {
            let noise_ops = &noise.after_gate[*instr_idx];
            if !noise_ops.is_empty() {
                for event in noise_ops {
                    let (px, py, pz) = event.pauli_probs();
                    let local_q = qubit_to_local[event.qubit()];

                    let has_any = x_packed[local_q].iter().any(|&w| w != 0)
                        || z_packed[local_q].iter().any(|&w| w != 0);
                    if has_any {
                        global_x_buf.fill(0);
                        global_z_buf.fill(0);
                        for (local_mi, &global_mi) in bm_list.iter().enumerate() {
                            if (x_packed[local_q][local_mi / 64] >> (local_mi % 64)) & 1 != 0 {
                                global_x_buf[global_mi / 64] |= 1u64 << (global_mi % 64);
                            }
                            if (z_packed[local_q][local_mi / 64] >> (local_mi % 64)) & 1 != 0 {
                                global_z_buf[global_mi / 64] |= 1u64 << (global_mi % 64);
                            }
                        }
                        events.push(&global_x_buf, &global_z_buf, px, py, pz);
                    }
                }
            }

            batch_propagate_backward(
                &mut x_packed,
                &mut z_packed,
                &mut sign_packed,
                gate,
                local_targets.as_slice(),
                bm_words,
            );
        }
    }

    let (z_lut, x_lut) = build_noise_luts(&events);

    Ok(NoisyCompiledSampler {
        noiseless,
        events,

        num_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xCAFE_BABE)),
        z_lut,
        x_lut,
        #[cfg(feature = "gpu")]
        gpu_noise_cache: None,
    })
}

fn compile_noisy_monolithic(
    circuit: &Circuit,
    noise: &NoiseModel,
    seed: u64,
) -> Result<NoisyCompiledSampler> {
    let noiseless = compile_measurements(circuit, seed)?;

    let n = circuit.num_qubits;

    let gate_indices: Vec<(usize, &Gate, &[usize])> = circuit
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(idx, inst)| match inst {
            Instruction::Gate { gate, targets } => Some((idx, gate, targets.as_slice())),
            Instruction::Conditional { gate, targets, .. } => Some((idx, gate, targets.as_slice())),
            _ => None,
        })
        .collect();

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
            events: FlatNoiseSensitivity::new(m_words, 0),

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

    for (mi, &q) in measurement_qubits.iter().enumerate() {
        z_packed[q][mi / 64] |= 1u64 << (mi % 64);
    }

    let total_noise_events: usize = noise.after_gate.iter().map(|ops| ops.len()).sum();
    let mut events = FlatNoiseSensitivity::new(m_words, total_noise_events);

    for &(instr_idx, gate, targets) in gate_indices.iter().rev() {
        let noise_ops = &noise.after_gate[instr_idx];
        if !noise_ops.is_empty() {
            for event in noise_ops {
                let (px, py, pz) = event.pauli_probs();
                let q = event.qubit();

                let has_any =
                    x_packed[q].iter().any(|&w| w != 0) || z_packed[q].iter().any(|&w| w != 0);
                if has_any {
                    events.push(&x_packed[q], &z_packed[q], px, py, pz);
                }
            }
        }

        batch_propagate_backward(
            &mut x_packed,
            &mut z_packed,
            &mut sign_packed,
            gate,
            targets,
            m_words,
        );
    }

    let (z_lut, x_lut) = build_noise_luts(&events);

    Ok(NoisyCompiledSampler {
        noiseless,
        events,

        num_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xCAFE_BABE)),
        z_lut,
        x_lut,
        #[cfg(feature = "gpu")]
        gpu_noise_cache: None,
    })
}

const FRAME_BATCH_SIZE: usize = 256;

struct ReferenceInfo {
    outcomes: Vec<bool>,
    is_random: Vec<bool>,
    random_x_support: Vec<Vec<usize>>,
}

fn reference_simulation(circuit: &Circuit, seed: u64) -> Result<ReferenceInfo> {
    let mut stab = StabilizerBackend::new(seed);
    stab.init(circuit.num_qubits, circuit.num_classical_bits)?;

    let meas_info: Vec<(usize, usize, usize)> = circuit
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| match inst {
            Instruction::Measure {
                qubit,
                classical_bit,
            } => Some((i, *qubit, *classical_bit)),
            _ => None,
        })
        .collect();

    let num_meas = meas_info.len();
    if num_meas == 0 {
        return Ok(ReferenceInfo {
            outcomes: Vec::new(),
            is_random: Vec::new(),
            random_x_support: Vec::new(),
        });
    }

    let first_meas_idx = meas_info[0].0;
    let all_at_end = meas_info
        .iter()
        .enumerate()
        .all(|(i, &(inst_idx, _, _))| inst_idx == first_meas_idx + i);

    if all_at_end {
        stab.apply_gates_only(&circuit.instructions[..first_meas_idx])?;

        let measurements: Vec<(usize, usize)> = meas_info
            .iter()
            .map(|&(_, qubit, classical_bit)| (qubit, classical_bit))
            .collect();
        let (is_random, random_x_support, outcomes) = stab.batch_measure_ref_info(&measurements);

        return Ok(ReferenceInfo {
            outcomes,
            is_random,
            random_x_support,
        });
    }

    let mut is_random = Vec::with_capacity(num_meas);
    let mut random_x_support: Vec<Vec<usize>> = Vec::with_capacity(num_meas);

    let mut seg_start = 0usize;
    for &(meas_inst_idx, qubit, classical_bit) in &meas_info {
        if seg_start < meas_inst_idx {
            stab.apply_gates_only(&circuit.instructions[seg_start..meas_inst_idx])?;
        }

        let (meas_random, support) = stab.apply_measure_with_info(qubit, classical_bit);
        is_random.push(meas_random);
        random_x_support.push(support);
        seg_start = meas_inst_idx + 1;
    }

    if seg_start < circuit.instructions.len() {
        stab.apply_gates_only(&circuit.instructions[seg_start..])?;
    }

    let outcomes: Vec<bool> = meas_info
        .iter()
        .map(|&(_, _, cbit)| stab.classical_results()[cbit])
        .collect();

    Ok(ReferenceInfo {
        outcomes,
        is_random,
        random_x_support,
    })
}

#[inline(always)]
fn apply_gate_to_frame(
    gate: &Gate,
    targets: &[usize],
    x_frame: &mut [Vec<u64>],
    z_frame: &mut [Vec<u64>],
    bw: usize,
) {
    match gate {
        Gate::H => {
            let q = targets[0];
            std::mem::swap(&mut x_frame[q], &mut z_frame[q]);
        }
        Gate::S | Gate::Sdg => {
            let q = targets[0];
            for w in 0..bw {
                z_frame[q][w] ^= x_frame[q][w];
            }
        }
        Gate::SX | Gate::SXdg => {
            let q = targets[0];
            for w in 0..bw {
                x_frame[q][w] ^= z_frame[q][w];
            }
        }
        Gate::X | Gate::Y | Gate::Z | Gate::Id => {}
        Gate::Cx => {
            let ctrl = targets[0];
            let tgt = targets[1];
            for w in 0..bw {
                x_frame[tgt][w] ^= x_frame[ctrl][w];
                z_frame[ctrl][w] ^= z_frame[tgt][w];
            }
        }
        Gate::Cz => {
            let q0 = targets[0];
            let q1 = targets[1];
            for w in 0..bw {
                z_frame[q0][w] ^= x_frame[q1][w];
                z_frame[q1][w] ^= x_frame[q0][w];
            }
        }
        Gate::Swap => {
            let q0 = targets[0];
            let q1 = targets[1];
            x_frame.swap(q0, q1);
            z_frame.swap(q0, q1);
        }
        _ => {
            debug_assert!(
                false,
                "apply_gate_to_frame: unhandled Clifford gate {:?}",
                gate
            );
        }
    }
}

fn run_shots_noisy_frame(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let n = circuit.num_qubits;
    let num_classical = circuit.num_classical_bits;

    let ref_info = reference_simulation(circuit, seed)?;

    let classical_bit_order: Vec<usize> = circuit
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Measure { classical_bit, .. } => Some(*classical_bit),
            _ => None,
        })
        .collect();
    let num_measurements = classical_bit_order.len();
    let m_words = num_measurements.div_ceil(64);

    let mut all_packed = vec![0u64; num_shots * m_words];
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xFAAB_E001));

    for batch_start in (0..num_shots).step_by(FRAME_BATCH_SIZE) {
        let batch_end = (batch_start + FRAME_BATCH_SIZE).min(num_shots);
        let batch_n = batch_end - batch_start;
        let bw = batch_n.div_ceil(64);

        let mut x_frame: Vec<Vec<u64>> = vec![vec![0u64; bw]; n];
        let mut z_frame: Vec<Vec<u64>> = vec![vec![0u64; bw]; n];

        let mut meas_idx = 0usize;

        for (idx, inst) in circuit.instructions.iter().enumerate() {
            match inst {
                Instruction::Gate { gate, targets }
                | Instruction::Conditional { gate, targets, .. } => {
                    apply_gate_to_frame(gate, targets.as_slice(), &mut x_frame, &mut z_frame, bw);

                    for event in &noise.after_gate[idx] {
                        let (px, py, pz) = event.pauli_probs();
                        let q = event.qubit();
                        let p_event = px + py + pz;
                        if p_event == 0.0 {
                            continue;
                        }

                        let px_frac = px / p_event;
                        let pxy_frac = (px + py) / p_event;

                        if p_event < 0.5 && batch_n >= 32 {
                            let ln_1mp = (1.0 - p_event).ln();
                            let mut pos = geometric_sample(&mut rng, ln_1mp);
                            while pos < batch_n {
                                let r: f64 = rand::Rng::random(&mut rng);
                                let bit = 1u64 << (pos % 64);
                                let w = pos / 64;
                                if r < px_frac {
                                    x_frame[q][w] ^= bit;
                                } else if r < pxy_frac {
                                    x_frame[q][w] ^= bit;
                                    z_frame[q][w] ^= bit;
                                } else {
                                    z_frame[q][w] ^= bit;
                                }
                                pos += 1 + geometric_sample(&mut rng, ln_1mp);
                            }
                        } else {
                            for s in 0..batch_n {
                                let r: f64 = rand::Rng::random(&mut rng);
                                if r < px {
                                    x_frame[q][s / 64] ^= 1u64 << (s % 64);
                                } else if r < px + py {
                                    x_frame[q][s / 64] ^= 1u64 << (s % 64);
                                    z_frame[q][s / 64] ^= 1u64 << (s % 64);
                                } else if r < p_event {
                                    z_frame[q][s / 64] ^= 1u64 << (s % 64);
                                }
                            }
                        }
                    }
                }
                Instruction::Measure {
                    qubit,
                    classical_bit: _,
                } => {
                    if ref_info.is_random[meas_idx] {
                        let support = &ref_info.random_x_support[meas_idx];
                        #[allow(clippy::needless_range_loop)]
                        for w in 0..bw {
                            let random_word: u64 = rand::Rng::random(&mut rng);
                            let mask = if w == bw - 1 && batch_n % 64 != 0 {
                                random_word & ((1u64 << (batch_n % 64)) - 1)
                            } else {
                                random_word
                            };
                            if mask != 0 {
                                for &q in support {
                                    x_frame[q][w] ^= mask;
                                }
                            }
                        }
                    }

                    let ref_bit = ref_info.outcomes[meas_idx];
                    let mi_word = meas_idx / 64;
                    let mi_bit = meas_idx % 64;
                    #[allow(clippy::needless_range_loop)]
                    for w in 0..bw {
                        let frame_word = x_frame[*qubit][w];
                        let effective = if ref_bit { !frame_word } else { frame_word };
                        let num_bits = if w == bw - 1 && batch_n % 64 != 0 {
                            batch_n % 64
                        } else {
                            64
                        };
                        let mask = if num_bits == 64 {
                            effective
                        } else {
                            effective & ((1u64 << num_bits) - 1)
                        };
                        let mut bits = mask;
                        while bits != 0 {
                            let s = bits.trailing_zeros() as usize;
                            let gs = batch_start + w * 64 + s;
                            all_packed[gs * m_words + mi_word] |= 1u64 << mi_bit;
                            bits &= bits - 1;
                        }
                    }

                    meas_idx += 1;
                }
                _ => {}
            }
        }
    }

    let shots = unpack_and_remap(
        &all_packed,
        m_words,
        num_shots,
        &classical_bit_order,
        num_classical,
    );

    Ok(ShotsResult {
        shots,
        num_classical_bits: circuit.num_classical_bits,
    })
}

/// Sample a Pauli-noisy Clifford circuit through the compiled noisy sampler,
/// frame sampler, or homological path as appropriate.
///
/// Circuits with resets or classical conditionals fall back to brute-force
/// stabilizer simulation so the public API stays correct.
pub fn run_shots_noisy(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    noise.ensure_pauli_only()?;
    if !circuit.is_clifford_only() {
        return run_shots_noisy_brute_with(
            |s| Box::new(StabilizerBackend::new(s)),
            circuit,
            noise,
            num_shots,
            seed,
        );
    }

    if !super::supports_compiled_measurement_sampling(circuit) {
        return run_shots_noisy_brute_with(
            |s| Box::new(StabilizerBackend::new(s)),
            circuit,
            noise,
            num_shots,
            seed,
        );
    }

    // Try homological sampler for high shot counts, O(r_quantum + 1) per shot
    // when syndrome rank ≤ 20. Falls back to compiled/frame if rank too high.
    if num_shots >= 1000 {
        if let Ok(sampler) = super::homological::HomologicalSampler::compile(circuit, noise, seed) {
            return super::homological::run_shots_homological_inner(sampler, circuit, num_shots);
        }
    }

    let n = circuit.num_qubits;
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

    let depth_ratio = gate_count as f64 / n.max(1) as f64;
    let use_frame = depth_ratio < 3.0 || (n >= 200 && depth_ratio < 5.0);

    if use_frame {
        run_shots_noisy_frame(circuit, noise, num_shots, seed)
    } else {
        run_shots_noisy_compiled(circuit, noise, num_shots, seed)
    }
}

fn finish_noisy_compiled_run(
    mut sampler: NoisyCompiledSampler,
    circuit: &Circuit,
    num_shots: usize,
) -> Result<ShotsResult> {
    let classical_bit_order: Vec<usize> = circuit
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Measure { classical_bit, .. } => Some(*classical_bit),
            _ => None,
        })
        .collect();
    let num_classical = circuit.num_classical_bits;

    let packed = sampler.try_sample_bulk_packed(num_shots)?;
    let shots = unpack_and_remap_packed(&packed, num_shots, &classical_bit_order, num_classical);

    Ok(ShotsResult {
        shots,
        num_classical_bits: circuit.num_classical_bits,
    })
}

fn run_shots_noisy_compiled(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let sampler = compile_noisy(circuit, noise, seed)?;
    finish_noisy_compiled_run(sampler, circuit, num_shots)
}

#[cfg(feature = "gpu")]
pub(crate) fn run_shots_noisy_with_gpu(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
    context: std::sync::Arc<crate::gpu::GpuContext>,
) -> Result<ShotsResult> {
    noise.ensure_pauli_only()?;
    if !circuit.is_clifford_only() {
        return run_shots_noisy_brute_with(
            |s| Box::new(StabilizerBackend::new(s)),
            circuit,
            noise,
            num_shots,
            seed,
        );
    }

    if !super::supports_compiled_measurement_sampling(circuit) {
        return run_shots_noisy_brute_with(
            |s| Box::new(StabilizerBackend::new(s)),
            circuit,
            noise,
            num_shots,
            seed,
        );
    }

    if num_shots >= 1000 {
        if let Ok(sampler) = super::homological::HomologicalSampler::compile(circuit, noise, seed) {
            return super::homological::run_shots_homological_inner(sampler, circuit, num_shots);
        }
    }

    let n = circuit.num_qubits;
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

    let depth_ratio = gate_count as f64 / n.max(1) as f64;
    let use_frame = depth_ratio < 3.0 || (n >= 200 && depth_ratio < 5.0);
    if use_frame {
        return run_shots_noisy_frame(circuit, noise, num_shots, seed);
    }

    let sampler = compile_noisy(circuit, noise, seed)?.with_gpu(context);
    finish_noisy_compiled_run(sampler, circuit, num_shots)
}

pub(crate) fn run_shots_noisy_brute_with(
    backend_factory: impl Fn(u64) -> Box<dyn Backend>,
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let mut shots = Vec::with_capacity(num_shots);

    for i in 0..num_shots {
        let shot_seed = seed.wrapping_add(i as u64);
        let mut rng = ChaCha8Rng::seed_from_u64(shot_seed);
        let mut backend = backend_factory(shot_seed);
        backend.init(circuit.num_qubits, circuit.num_classical_bits)?;

        for (idx, instr) in circuit.instructions.iter().enumerate() {
            backend.apply(instr)?;

            let noise_events = &noise.after_gate[idx];
            for event in noise_events {
                let (px, py, pz) = event.pauli_probs();
                let q = event.qubit();
                let r: f64 = rand::Rng::random(&mut rng);
                if r < px {
                    backend.apply(&Instruction::Gate {
                        gate: Gate::X,
                        targets: SmallVec::from_elem(q, 1),
                    })?;
                } else if r < px + py {
                    backend.apply(&Instruction::Gate {
                        gate: Gate::Y,
                        targets: SmallVec::from_elem(q, 1),
                    })?;
                } else if r < px + py + pz {
                    backend.apply(&Instruction::Gate {
                        gate: Gate::Z,
                        targets: SmallVec::from_elem(q, 1),
                    })?;
                }
            }
        }

        shots.push(backend.classical_results().to_vec());
    }

    Ok(ShotsResult {
        shots,
        num_classical_bits: circuit.num_classical_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits;

    #[test]
    fn noisy_ghz_produces_varied_outcomes() {
        let n = 10;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let result = run_shots_noisy(&circuit, &noise, 1000, 42).unwrap();

        assert_eq!(result.shots.len(), 1000);
        assert_eq!(result.shots[0].len(), n);

        let all_zero: Vec<bool> = vec![false; n];
        let all_one: Vec<bool> = vec![true; n];
        let num_00 = result.shots.iter().filter(|s| **s == all_zero).count();
        let num_11 = result.shots.iter().filter(|s| **s == all_one).count();
        let num_other = 1000 - num_00 - num_11;

        assert!(num_other > 0, "noise should produce non-GHZ outcomes");
        assert!(num_00 > 100, "should still have many |00...0> outcomes");
        assert!(num_11 > 100, "should still have many |11...1> outcomes");
    }

    #[test]
    fn zero_noise_matches_noiseless() {
        let n = 5;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        let result = run_shots_noisy(&circuit, &noise, 100, 42).unwrap();

        for shot in &result.shots {
            let all_same = shot.iter().all(|&b| b == shot[0]);
            assert!(all_same, "GHZ with zero noise must be all-0 or all-1");
        }
    }

    #[test]
    fn noise_model_length_matches_circuit() {
        let n = 10;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        assert_eq!(noise.after_gate.len(), circuit.instructions.len());
    }

    #[test]
    fn compiled_noisy_stats_match_brute_force() {
        let n = 10;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let num_shots = 10000;

        let brute = run_shots_noisy_brute_with(
            |s| Box::new(StabilizerBackend::new(s)),
            &circuit,
            &noise,
            num_shots,
            42,
        )
        .unwrap();
        let compiled = run_shots_noisy_compiled(&circuit, &noise, num_shots, 42).unwrap();

        let count_all_same = |shots: &[Vec<bool>]| -> usize {
            shots
                .iter()
                .filter(|s| s.iter().all(|&b| b == s[0]))
                .count()
        };

        let brute_coherent = count_all_same(&brute.shots);
        let compiled_coherent = count_all_same(&compiled.shots);

        let brute_frac = brute_coherent as f64 / num_shots as f64;
        let compiled_frac = compiled_coherent as f64 / num_shots as f64;

        assert!(
            (brute_frac - compiled_frac).abs() < 0.05,
            "coherent fraction should be similar: brute={brute_frac:.3}, compiled={compiled_frac:.3}"
        );

        let count_errors = |shots: &[Vec<bool>]| -> usize {
            shots
                .iter()
                .filter(|s| !s.iter().all(|&b| b == s[0]))
                .count()
        };

        let brute_errors = count_errors(&brute.shots);
        let compiled_errors = count_errors(&compiled.shots);

        assert!(
            brute_errors > 0 && compiled_errors > 0,
            "both should produce errors"
        );
    }

    #[test]
    fn compiled_noisy_clifford_produces_noise() {
        let n = 20;
        let circuit_base = circuits::clifford_heavy_circuit(n, 10, 42);
        let mut circuit = circuit_base;
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let result = run_shots_noisy(&circuit, &noise, 100, 42).unwrap();

        assert_eq!(result.shots.len(), 100);
        assert_eq!(result.shots[0].len(), n);

        let unique: std::collections::HashSet<Vec<bool>> = result.shots.iter().cloned().collect();
        assert!(unique.len() > 1, "noise should produce varied outcomes");
    }

    #[test]
    fn compile_noisy_rejects_reset_circuits() {
        let mut circuit = Circuit::new(1, 1);
        circuit.add_reset(0);
        circuit.add_measure(0, 0);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        assert!(compile_noisy(&circuit, &noise, 42).is_err());
    }

    #[test]
    fn compile_noisy_rejects_conditionals() {
        let mut circuit = Circuit::new(2, 2);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_measure(0, 0);
        circuit.instructions.push(Instruction::Conditional {
            condition: crate::circuit::ClassicalCondition::BitIsOne(0),
            gate: Gate::X,
            targets: crate::circuit::smallvec![1],
        });
        circuit.add_measure(1, 1);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        assert!(compile_noisy(&circuit, &noise, 42).is_err());
    }

    #[test]
    fn run_shots_noisy_handles_reset_circuits() {
        let mut circuit = Circuit::new(1, 1);
        circuit.add_gate(Gate::X, &[0]);
        circuit.add_reset(0);
        circuit.add_measure(0, 0);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        let result = run_shots_noisy(&circuit, &noise, 32, 42).unwrap();
        assert!(result.shots.iter().all(|shot| !shot[0]));
    }

    #[test]
    fn run_shots_noisy_handles_conditionals() {
        let mut circuit = Circuit::new(2, 2);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_measure(0, 0);
        circuit.instructions.push(Instruction::Conditional {
            condition: crate::circuit::ClassicalCondition::BitIsOne(0),
            gate: Gate::X,
            targets: crate::circuit::smallvec![1],
        });
        circuit.add_measure(1, 1);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        let result = run_shots_noisy(&circuit, &noise, 256, 42).unwrap();
        assert!(result.shots.iter().all(|shot| shot[0] == shot[1]));
    }

    #[test]
    fn frame_ghz_100q_produces_varied_outcomes() {
        let n = 100;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let result = run_shots_noisy_frame(&circuit, &noise, 1000, 42).unwrap();

        assert_eq!(result.shots.len(), 1000);
        assert_eq!(result.shots[0].len(), n);

        let all_zero: Vec<bool> = vec![false; n];
        let all_one: Vec<bool> = vec![true; n];
        let num_00 = result.shots.iter().filter(|s| **s == all_zero).count();
        let num_11 = result.shots.iter().filter(|s| **s == all_one).count();
        let num_other = 1000 - num_00 - num_11;

        assert!(num_other > 0, "noise should produce non-GHZ outcomes");
        assert!(num_00 > 50, "should still have many |00...0> outcomes");
        assert!(num_11 > 50, "should still have many |11...1> outcomes");
    }

    #[test]
    fn frame_zero_noise_matches_noiseless_100q() {
        let n = 100;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        let result = run_shots_noisy_frame(&circuit, &noise, 100, 42).unwrap();

        for shot in &result.shots {
            let all_same = shot.iter().all(|&b| b == shot[0]);
            assert!(all_same, "GHZ with zero noise must be all-0 or all-1");
        }
    }

    #[test]
    fn frame_stats_match_compiled_ghz() {
        let n = 100;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let num_shots = 5000;

        let frame = run_shots_noisy_frame(&circuit, &noise, num_shots, 42).unwrap();
        let compiled = run_shots_noisy_compiled(&circuit, &noise, num_shots, 42).unwrap();

        let count_coherent = |shots: &[Vec<bool>]| -> usize {
            shots
                .iter()
                .filter(|s| s.iter().all(|&b| b == s[0]))
                .count()
        };

        let frame_coh = count_coherent(&frame.shots) as f64 / num_shots as f64;
        let compiled_coh = count_coherent(&compiled.shots) as f64 / num_shots as f64;

        assert!(
            (frame_coh - compiled_coh).abs() < 0.05,
            "coherent fraction should be similar: frame={frame_coh:.3}, compiled={compiled_coh:.3}"
        );
    }

    #[test]
    fn frame_clifford_100q_produces_noise() {
        let n = 100;
        let circuit_base = circuits::clifford_heavy_circuit(n, 10, 42);
        let mut circuit = circuit_base;
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let result = run_shots_noisy_frame(&circuit, &noise, 100, 42).unwrap();

        assert_eq!(result.shots.len(), 100);
        assert_eq!(result.shots[0].len(), n);

        let unique: std::collections::HashSet<Vec<bool>> = result.shots.iter().cloned().collect();
        assert!(unique.len() > 1, "noise should produce varied outcomes");
    }

    #[test]
    fn filtered_noisy_bell_pairs_matches_monolithic() {
        let n_pairs = 50;
        let n = n_pairs * 2;
        let mut circuit = circuits::independent_bell_pairs(n_pairs);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let seed = 42u64;

        let filtered =
            compile_noisy_filtered(&circuit, &noise, &circuit.independent_subsystems(), seed)
                .unwrap();
        let monolithic = compile_noisy_monolithic(&circuit, &noise, seed).unwrap();

        assert_eq!(filtered.num_measurements, monolithic.num_measurements);
        assert_eq!(filtered.events.len(), monolithic.events.len());

        let mut filtered = filtered;
        let mut monolithic = monolithic;
        let num_shots = 10_000;
        let shots_f = filtered.sample_bulk(num_shots);
        let shots_m = monolithic.sample_bulk(num_shots);

        assert_eq!(shots_f.len(), num_shots);
        assert_eq!(shots_m.len(), num_shots);

        let mut agree_f = 0usize;
        let mut agree_m = 0usize;
        for shot in &shots_f {
            for pair in shot.chunks(2) {
                if pair[0] == pair[1] {
                    agree_f += 1;
                }
            }
        }
        for shot in &shots_m {
            for pair in shot.chunks(2) {
                if pair[0] == pair[1] {
                    agree_m += 1;
                }
            }
        }

        let total_pairs = num_shots * n_pairs;
        let agree_rate_f = agree_f as f64 / total_pairs as f64;
        let agree_rate_m = agree_m as f64 / total_pairs as f64;
        assert!(
            agree_rate_f > 0.95,
            "filtered agreement rate {agree_rate_f:.4} should be >0.95 with low noise"
        );
        assert!(
            agree_rate_m > 0.95,
            "monolithic agreement rate {agree_rate_m:.4} should be >0.95 with low noise"
        );
        assert!(
            (agree_rate_f - agree_rate_m).abs() < 0.02,
            "filtered ({agree_rate_f:.4}) and monolithic ({agree_rate_m:.4}) should have similar agreement rates"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn compiled_noisy_with_stub_gpu_matches_cpu_below_threshold() {
        let n = 12;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let shots = 20_000;

        let mut cpu = compile_noisy(&circuit, &noise, 42).unwrap();
        let cpu_marginals = cpu.sample_marginals(shots);

        let mut gpu = compile_noisy(&circuit, &noise, 42)
            .unwrap()
            .with_gpu(crate::gpu::GpuContext::stub_for_tests());
        let gpu_marginals = gpu.sample_marginals(shots);

        for (idx, (cpu_p1, gpu_p1)) in cpu_marginals.iter().zip(gpu_marginals.iter()).enumerate() {
            assert!(
                (cpu_p1 - gpu_p1).abs() < 0.03,
                "marginal[{idx}] diverged too much: cpu={cpu_p1}, gpu={gpu_p1}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn compiled_noisy_with_stub_gpu_low_rank_above_threshold_uses_cpu_fallback() {
        let shots = crate::gpu::bts_min_shots().max(1);
        let n = 12;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let mut gpu = compile_noisy(&circuit, &noise, 42)
            .unwrap()
            .with_gpu(crate::gpu::GpuContext::stub_for_tests());

        assert!(!gpu.noiseless.should_use_gpu_bts(shots));
        let counts = gpu.sample_counts(shots);
        let total: u64 = counts.values().sum();
        assert_eq!(total, shots as u64);
    }
}

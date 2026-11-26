//! Device copies of the noisy compiled sampler's event tables and thresholds.

use crate::error::Result;

use super::compiled::{FlatNoiseSensitivity, RecordReadout};

/// Pack `(px, py, pz)` per event into three `u64` thresholds for the device
/// noise mask generator. Each threshold is `p * 2^64` rounded down, so a uniform
/// `u64` random compares below it with probability `p`. Returns the flat
/// buffer along with `false` when any event's probabilities are out of range
/// (negative, NaN, or sum greater than 1). The caller then falls back to the
/// host mask path.
fn flatten_event_thresholds_u64(probs: &[[f64; 3]]) -> (Vec<u64>, bool) {
    const PROB_SUM_SLACK: f64 = 1e-12;

    let mut out = Vec::with_capacity(probs.len() * 3);
    let mut valid = true;
    for &[px, py, pz] in probs {
        let finite = px.is_finite() && py.is_finite() && pz.is_finite();
        let sum = px + py + pz;
        if !finite || px < 0.0 || py < 0.0 || pz < 0.0 || sum > 1.0 + PROB_SUM_SLACK {
            valid = false;
        }
        out.push(prob_threshold_u64(px));
        out.push(prob_threshold_u64(px + py));
        out.push(prob_threshold_u64(sum));
    }
    (out, valid)
}

/// `p * 2^64` rounded down, saturating at 0 and `u64::MAX`.
fn prob_threshold_u64(p: f64) -> u64 {
    // Nearest f64 below 2^64. Scaling by 2^64 exactly rounds to 2^64 which
    // doesn't fit in a u64; this preserves `u64_rand < threshold` at the
    // intended probability.
    const SCALE: f64 = 18446744073709549568.0_f64;
    if !p.is_finite() || p <= 0.0 {
        0
    } else if p >= 1.0 {
        u64::MAX
    } else {
        (p * SCALE) as u64
    }
}

/// Pack each readout entry as `[record, p01, p10]` for the readout kernel, the
/// rates as [`prob_threshold_u64`] thresholds.
fn flatten_readout_thresholds_u64(readout: &[RecordReadout]) -> Vec<u64> {
    readout
        .iter()
        .flat_map(|r| {
            [
                r.record as u64,
                prob_threshold_u64(r.p01),
                prob_threshold_u64(r.p10),
            ]
        })
        .collect()
}

/// Transpose the event-major `(x_data, z_data)` bitmaps in a
/// `FlatNoiseSensitivity` into a row-major CSR. Each row's entry list names
/// the events that touch it, packed as `event << 2 | flag`: flag bit 0 for
/// an X contribution, bit 1 for Z, both set for a Y contribution. Lets the
/// per-(row, batch) kernel accumulate XORs in a register without atomics.
/// Paired with `bts_apply_noise_row_major`.
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

    // Second pass: write entries through a cursor starting at `offsets[row]`.
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

pub(super) struct GpuNoiseCache {
    pub(super) context: std::sync::Arc<crate::gpu::GpuContext>,
    pub(super) x_row_offsets_dev: crate::gpu::GpuBuffer<u32>,
    pub(super) x_row_indices_dev: crate::gpu::GpuBuffer<u32>,
    pub(super) z_row_offsets_dev: crate::gpu::GpuBuffer<u32>,
    pub(super) z_row_indices_dev: crate::gpu::GpuBuffer<u32>,
    pub(super) x_masks_dev: crate::gpu::GpuBuffer<u64>,
    pub(super) z_masks_dev: crate::gpu::GpuBuffer<u64>,
    pub(super) x_masks_host: Vec<u64>,
    pub(super) z_masks_host: Vec<u64>,
    /// Precomputed per-event thresholds for `bts_generate_noise_masks`. Three
    /// u64s per event hold `[px, px+py, px+py+pz]` scaled to `2^64`, so the
    /// kernel's inner loop replaces a floating-point multiply and branch per
    /// shot with a `u64 <` compare. Stays resident across sampling calls.
    pub(super) event_thresholds_dev: crate::gpu::GpuBuffer<u64>,
    /// True when every event's probabilities can be represented as a `u64`
    /// threshold. A NaN or a sum greater than 1 flips this to false and
    /// routes the caller to the host mask path.
    pub(super) event_thresholds_valid: bool,
    /// Row-major event CSR for `bts_apply_noise_row_major`. Each entry packs
    /// `event << 2 | flag`, where bit 0 means the event contributes X to this
    /// row and bit 1 means Z. The row-major layout lets one thread accumulate
    /// every XOR for its output word in a register instead of broadcasting
    /// `(event, batch)` writes across rows with atomics.
    pub(super) row_event_offsets_dev: crate::gpu::GpuBuffer<u32>,
    pub(super) row_event_entries_dev: crate::gpu::GpuBuffer<u32>,
    /// Readout entries packed by [`flatten_readout_thresholds_u64`].
    pub(super) readout_dev: crate::gpu::GpuBuffer<u64>,
}

impl GpuNoiseCache {
    pub(super) fn new(
        context: std::sync::Arc<crate::gpu::GpuContext>,
        events: &FlatNoiseSensitivity,
        readout: &[RecordReadout],
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

        // Row-major transpose for bts_apply_noise_row_major. Cheap
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

        let readout_host = flatten_readout_thresholds_u64(readout);
        let mut readout_dev =
            crate::gpu::GpuBuffer::<u64>::alloc_zeros(device, readout_host.len().max(1))?;
        if !readout_host.is_empty() {
            readout_dev.copy_from_host(device, &readout_host)?;
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
            readout_dev,
        })
    }

    pub(super) fn ensure_device_mask_capacity(&mut self, len: usize) -> Result<()> {
        let needed = len.max(1);
        let device = self.context.device();
        if self.x_masks_dev.len() < needed {
            self.x_masks_dev = crate::gpu::GpuBuffer::<u64>::alloc_zeros(device, needed)?;
        }
        if self.z_masks_dev.len() < needed {
            self.z_masks_dev = crate::gpu::GpuBuffer::<u64>::alloc_zeros(device, needed)?;
        }
        Ok(())
    }

    pub(super) fn ensure_mask_capacity(&mut self, len: usize) -> Result<()> {
        self.ensure_device_mask_capacity(len)?;
        let needed = len.max(1);
        if self.x_masks_host.len() < needed {
            self.x_masks_host.resize(needed, 0);
        }
        if self.z_masks_host.len() < needed {
            self.z_masks_host.resize(needed, 0);
        }
        Ok(())
    }
}

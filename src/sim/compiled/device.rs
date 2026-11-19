//! Packed shots that stay on the GPU after the device BTS sampler draws them.

use crate::error::{PrismError, Result};

use super::{PackedShots, ShotLayout};

/// Device-resident packed shots emitted by the GPU BTS path.
///
/// The payload stays in measurement-major layout on the device until an
/// explicit host copy via [`Self::to_host`]. Marginals and exact counts can
/// reduce on the device without transferring the shot matrix.
#[derive(Debug)]
pub struct DevicePackedShots {
    context: std::sync::Arc<crate::gpu::GpuContext>,
    data: crate::gpu::GpuBuffer<u64>,
    num_shots: usize,
    num_measurements: usize,
    m_words: usize,
    s_words: usize,
    layout: ShotLayout,
    rank: usize,
}

impl DevicePackedShots {
    /// Wrap the measurement-major shots a device sampler wrote into `data`.
    pub(super) fn new(
        context: std::sync::Arc<crate::gpu::GpuContext>,
        data: crate::gpu::GpuBuffer<u64>,
        num_shots: usize,
        num_measurements: usize,
        rank: usize,
    ) -> Self {
        Self {
            context,
            data,
            num_shots,
            num_measurements,
            m_words: num_measurements.div_ceil(64),
            s_words: num_shots.div_ceil(64),
            layout: ShotLayout::MeasMajor,
            rank,
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

    pub(crate) fn context(&self) -> &std::sync::Arc<crate::gpu::GpuContext> {
        &self.context
    }

    /// Words per shot-major row: `ceil(num_measurements / 64)`.
    pub fn m_words(&self) -> usize {
        self.m_words
    }

    /// Words per measurement-major row: `ceil(num_shots / 64)`.
    pub fn s_words(&self) -> usize {
        self.s_words
    }

    pub(crate) fn data_mut(&mut self) -> &mut crate::gpu::GpuBuffer<u64> {
        &mut self.data
    }

    /// Copy the full packed payload back to host memory.
    pub fn to_host(&self) -> Result<PackedShots> {
        let len = match self.layout {
            ShotLayout::ShotMajor => self.num_shots * self.m_words,
            ShotLayout::MeasMajor => self.num_measurements * self.s_words,
        };
        let mut host = vec![0u64; len];
        if len > 0 {
            self.data
                .copy_to_host(self.context.device(), &mut host)
                .map_err(|e| PrismError::BackendUnsupported {
                    backend: "gpu".to_string(),
                    operation: format!("copy device packed shots to host: {e}"),
                })?;
        }

        Ok(match self.layout {
            ShotLayout::ShotMajor => {
                PackedShots::from_shot_major(host, self.num_shots, self.num_measurements)
            }
            ShotLayout::MeasMajor => {
                PackedShots::from_meas_major(host, self.num_shots, self.num_measurements)
            }
        })
    }

    /// Return per-measurement one-counts without copying the full shot matrix.
    pub fn marginal_counts(&self) -> Result<Vec<u64>> {
        match self.layout {
            ShotLayout::MeasMajor => crate::gpu::kernels::bts::count_meas_major_marginals(
                &self.context,
                &self.data,
                self.num_measurements,
                self.num_shots,
                self.s_words,
            ),
            ShotLayout::ShotMajor => Ok(self.to_host()?.counts().into_iter().fold(
                vec![0u64; self.num_measurements],
                |mut acc, (key, count)| {
                    for m in 0..self.num_measurements {
                        if (key[m / 64] >> (m % 64)) & 1 != 0 {
                            acc[m] += count;
                        }
                    }
                    acc
                },
            )),
        }
    }

    pub fn marginals(&self) -> Result<Vec<f64>> {
        if self.num_shots == 0 {
            return Ok(vec![0.0; self.num_measurements]);
        }
        Ok(self
            .marginal_counts()?
            .into_iter()
            .map(|count| count as f64 / self.num_shots as f64)
            .collect())
    }

    /// Return exact counts, using a GPU reduction when it reduces transfer.
    pub fn counts(&self) -> Result<std::collections::HashMap<Vec<u64>, u64>> {
        self.counts_with_rank_hint(self.rank)
    }

    pub(crate) fn counts_with_rank_hint(
        &self,
        rank_hint: usize,
    ) -> Result<std::collections::HashMap<Vec<u64>, u64>> {
        match self.layout {
            ShotLayout::MeasMajor => {
                if let Some(counts) = crate::gpu::kernels::bts::try_count_meas_major(
                    &self.context,
                    &self.data,
                    self.num_measurements,
                    self.num_shots,
                    self.m_words,
                    self.s_words,
                    rank_hint,
                )? {
                    return Ok(counts);
                }
            }
            ShotLayout::ShotMajor => {
                let raw_transfer_bytes = self
                    .num_shots
                    .saturating_mul(self.m_words)
                    .saturating_mul(std::mem::size_of::<u64>());
                if let Some(counts) = crate::gpu::kernels::bts::try_count_shot_major(
                    &self.context,
                    &self.data,
                    self.num_shots,
                    self.m_words,
                    rank_hint,
                    raw_transfer_bytes,
                )? {
                    return Ok(counts);
                }
            }
        }
        Ok(self.to_host()?.counts())
    }
}

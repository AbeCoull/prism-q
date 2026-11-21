//! Terminal shot sampling on the device statevector: block totals, a host scan of them,
//! and a walk that maps each sorted threshold to one state index.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::error::Result;
use crate::gpu::{GpuContext, GpuState};

use super::super::{ensure_exact, ensure_scratch, launch_counted};

/// Amplitudes per sampling block, as a power of two; mirrors `SAMPLE_BLOCK_LOG` in the
/// CUDA template.
const SAMPLE_BLOCK_LOG: u32 = 12;
/// Threads per sampling CTA; mirrors `SAMPLE_THREADS`.
const SAMPLE_THREADS: u32 = 256;

/// Whether walking `num_shots` on the device moves fewer bytes than the `2^n` probability
/// readback the host sampler takes: 12 bytes a shot (offset up, index down) and 8 a block
/// total, against 8 an amplitude.
pub(crate) fn walk_moves_fewer_bytes(num_qubits: usize, num_shots: usize) -> bool {
    let dim = 1u128 << num_qubits;
    let blocks = dim.div_ceil(1 << SAMPLE_BLOCK_LOG);
    12 * num_shots as u128 + 8 * blocks < 8 * dim
}

/// The state index of each ascending uniform in `sorted`, drawn against the weights
/// `|a_i|^2` scaled by their total, so `pending_norm` drops out.
///
/// An index is the first whose cumulative weight exceeds `u * total`, never one of zero
/// weight. `None` when a weight is NaN or infinite, when the weights sum to zero, when the
/// register is wider than 32 qubits or when the uniforms come within a CTA of `u32::MAX`;
/// the caller then samples the probability vector on the host.
pub(crate) fn sample_sorted(
    ctx: &GpuContext,
    state: &GpuState,
    sorted: &[f64],
) -> Result<Option<Vec<u32>>> {
    let n = state.num_qubits();
    if n > 32 || sorted.len() > (u32::MAX - SAMPLE_THREADS) as usize {
        return Ok(None);
    }
    if sorted.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let dim: u64 = 1u64 << n;
    let num_blocks = dim.div_ceil(1u64 << SAMPLE_BLOCK_LOG) as usize;
    let device = ctx.device();
    let stream = device.stream()?;
    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;

    let sums_buf = ensure_exact(&mut scratch.sample_sums, device, num_blocks)?;
    {
        let func = device.function("sample_block_sums")?;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(state.buffer().raw())
            .arg(&dim)
            .arg(sums_buf.raw_mut());
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (SAMPLE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: signature matches; one CTA per block of 2^SAMPLE_BLOCK_LOG amplitudes,
        // indices at or past `dim` read as zero weight, and `sums_buf` holds exactly
        // `num_blocks` values.
        unsafe { launch_counted(builder, cfg, "sample_block_sums", device, 1) }?;
    }
    let mut sums = vec![0.0_f64; num_blocks];
    sums_buf.copy_to_host(device, &mut sums)?;
    let Some(plan) = plan_walk(&sums, sorted) else {
        return Ok(None);
    };

    let jobs = ensure_scratch(&mut scratch.sample_jobs, device, &plan.jobs)?;
    let offsets = ensure_scratch(&mut scratch.sample_offsets, device, &plan.offsets)?;
    let out = ensure_exact(&mut scratch.sample_out, device, sorted.len())?;
    {
        let func = device.function("sample_walk")?;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(state.buffer().raw())
            .arg(&dim)
            .arg(jobs.raw())
            .arg(offsets.raw())
            .arg(out.raw_mut());
        let cfg = LaunchConfig {
            grid_dim: ((plan.jobs.len() / 3) as u32, 1, 1),
            block_dim: (SAMPLE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: signature matches; one CTA per `(block, lo, hi)` triple of `jobs`, every
        // block below `num_blocks`, every `hi` at most `sorted.len()`, which is the length
        // of `offsets` and of `out`.
        unsafe { launch_counted(builder, cfg, "sample_walk", device, 1) }?;
    }
    let mut indices = vec![0_u32; sorted.len()];
    out.copy_to_host(device, &mut indices)?;
    Ok(Some(indices))
}

/// Most thresholds one walk CTA takes. A block owning more is split across several
/// CTAs, each repeating the block's scan, so a narrow register with many shots still
/// fills the device.
const MAX_JOB_SHOTS: usize = 4096;

/// The walk's jobs, `(block, lo, hi)` flattened, and each threshold less its block's
/// base, in the order of the sorted uniforms.
#[derive(Debug, PartialEq)]
pub(super) struct WalkPlan {
    pub(super) jobs: Vec<u32>,
    pub(super) offsets: Vec<f64>,
}

/// Assign each threshold `u * total` to the block whose running sum spans it: block `b`
/// owns `[base_b, base_b + sums[b])`, scanned left to right. A threshold at or above
/// `total` goes to the last block with weight at an infinite offset, which the walk reads
/// as that block's last weight. `None` when the total is zero or not finite.
pub(super) fn plan_walk(sums: &[f64], sorted: &[f64]) -> Option<WalkPlan> {
    let total: f64 = sums.iter().fold(0.0, |acc, &s| acc + s);
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let mut ranges: Vec<(u32, usize, usize)> = Vec::new();
    let mut offsets = Vec::with_capacity(sorted.len());
    let mut next_shot = 0usize;
    let mut base = 0.0_f64;
    let mut last = 0u32;
    for (block, &sum) in sums.iter().enumerate() {
        if sum == 0.0 {
            continue;
        }
        let end = base + sum;
        let lo = next_shot;
        while let Some(&u) = sorted.get(next_shot) {
            let v = u * total;
            if v >= end {
                break;
            }
            offsets.push(v - base);
            next_shot += 1;
        }
        if next_shot > lo {
            ranges.push((block as u32, lo, next_shot));
        }
        last = block as u32;
        base = end;
    }
    if next_shot < sorted.len() {
        match ranges.last_mut() {
            Some((block, _, hi)) if *block == last => *hi = sorted.len(),
            _ => ranges.push((last, next_shot, sorted.len())),
        }
        offsets.resize(sorted.len(), f64::INFINITY);
    }
    let jobs = ranges
        .into_iter()
        .flat_map(|(block, lo, hi)| {
            (lo..hi)
                .step_by(MAX_JOB_SHOTS)
                .map(move |start| [block, start as u32, hi.min(start + MAX_JOB_SHOTS) as u32])
        })
        .flatten()
        .collect();
    Some(WalkPlan { jobs, offsets })
}

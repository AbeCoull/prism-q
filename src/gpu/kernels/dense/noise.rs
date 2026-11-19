//! Trajectory noise sites decided on the device.

use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};

use crate::error::Result;
use crate::gpu::{GpuContext, GpuState};

use super::super::{check_qubit, ensure_capacity, grid_for, launch_counted, linear_cfg};
use super::{BLOCK_SIZE, KrausSetArg, NOISE_MAX_KRAUS};

/// The decide kernel of one trajectory noise site and its arguments.
enum NoiseDecide<'a> {
    Damping {
        gamma: f64,
        keep: f64,
        moves: i32,
        r: f64,
    },
    Thermal {
        branches: &'a crate::sim::trajectory::ThermalBranches,
        r: f64,
    },
    Kraus {
        set: &'a KrausSetArg,
        r_unit: f64,
    },
}

/// Amplitude damping (`jump_moves_population`) or phase damping on `qubit`, decided on
/// the device from the draw `r`.
pub(crate) fn noise_damping(
    ctx: &GpuContext,
    state: &mut GpuState,
    qubit: usize,
    gamma: f64,
    keep: f64,
    jump_moves_population: bool,
    r: f64,
) -> Result<()> {
    let moves = i32::from(jump_moves_population);
    noise_site(
        ctx,
        state,
        qubit,
        NoiseDecide::Damping {
            gamma,
            keep,
            moves,
            r,
        },
    )
}

/// Thermal relaxation on `qubit`, decided on the device from the draw `r`.
pub(crate) fn noise_thermal(
    ctx: &GpuContext,
    state: &mut GpuState,
    qubit: usize,
    branches: &crate::sim::trajectory::ThermalBranches,
    r: f64,
) -> Result<()> {
    noise_site(ctx, state, qubit, NoiseDecide::Thermal { branches, r })
}

/// A one-qubit Kraus channel on `qubit`, decided on the device from the uniform draw
/// `r_unit`. Needs at most [`NOISE_MAX_KRAUS`] operators, each with its kernel settled.
pub(crate) fn noise_kraus(
    ctx: &GpuContext,
    state: &mut GpuState,
    qubit: usize,
    kraus: &[crate::sim::trajectory::Kraus1q],
    r_unit: f64,
) -> Result<()> {
    assert!(kraus.len() <= NOISE_MAX_KRAUS, "too many Kraus operators");
    let mut set = KrausSetArg {
        v: [0.0; 16 * NOISE_MAX_KRAUS],
        diagonal: [0; NOISE_MAX_KRAUS],
        count: kraus.len() as i32,
    };
    for (k, kraus) in kraus.iter().enumerate() {
        let diagonal = kraus.diagonal.expect("Kraus operator kernel unsettled");
        set.diagonal[k] = i32::from(diagonal);
        let entries = kraus.op.iter().chain(&kraus.effect).flatten();
        for (pair, z) in set.v[16 * k..16 * (k + 1)]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(entries)
        {
            pair[0] = z.re;
            pair[1] = z.im;
        }
    }
    noise_site(ctx, state, qubit, NoiseDecide::Kraus { set: &set, r_unit })
}

/// One trajectory noise site on `qubit` in two launches and no readback: a
/// `noise_site_*` kernel that runs stage 1 of [`measure_prob_one`] (or of
/// [`reduced_density_matrix_1q`] for a Kraus channel) and decides the branch in the block
/// that finishes last, then `apply_gate_1q_from_buffer`. The pending norm is read from the
/// measure log while it holds it, and is not changed: each branch matrix carries its own
/// normalization.
///
/// [`measure_prob_one`]: super::measure_prob_one
/// [`reduced_density_matrix_1q`]: super::reduced_density_matrix_1q
fn noise_site(
    ctx: &GpuContext,
    state: &mut GpuState,
    qubit: usize,
    decide: NoiseDecide<'_>,
) -> Result<()> {
    let n = state.num_qubits();
    check_qubit(n, qubit)?;
    let pairs: u64 = 1u64 << (n - 1);
    let (count, columns) = match decide {
        NoiseDecide::Kraus { .. } => (pairs, 4u32),
        _ => (1u64 << n, 1u32),
    };
    let elems_per_block = 2u64 * BLOCK_SIZE as u64;
    let num_blocks = count.div_ceil(elems_per_block).max(1) as u32;
    let shared_bytes = columns * BLOCK_SIZE * std::mem::size_of::<f64>() as u32;
    let device = ctx.device();
    let stream = device.stream()?;

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    ensure_capacity(
        &mut scratch.measure_partials,
        device,
        (columns * num_blocks) as usize,
    )?;
    ensure_capacity(&mut scratch.noise_decision, device, 9)?;
    ensure_capacity(&mut scratch.noise_ticket, device, 1)?;
    let partials = scratch.measure_partials.as_mut().unwrap();
    let decision = scratch.noise_decision.as_mut().unwrap();
    let ticket = scratch.noise_ticket.as_mut().unwrap();

    let norm_log: u64 = match (&state.measure_log, state.norm_in_log) {
        (Some(log), true) => log.raw().device_ptr(stream).0,
        _ => 0,
    };
    let norm = state.pending_norm;
    let eps = crate::sim::trajectory::JUMP_EPSILON;
    let qubit_i = qubit as i32;
    {
        let name = match decide {
            NoiseDecide::Damping { .. } => "noise_site_damping",
            NoiseDecide::Thermal { .. } => "noise_site_thermal",
            NoiseDecide::Kraus { .. } => "noise_site_kraus",
        };
        let func = device.function(name)?;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(state.buffer().raw())
            .arg(&count)
            .arg(&qubit_i)
            .arg(partials.raw_mut())
            .arg(ticket.raw_mut())
            .arg(decision.raw_mut())
            .arg(&norm_log)
            .arg(&norm);
        match &decide {
            NoiseDecide::Damping {
                gamma,
                keep,
                moves,
                r,
            } => {
                builder.arg(gamma).arg(keep).arg(moves).arg(r);
            }
            NoiseDecide::Thermal { branches, r } => {
                builder
                    .arg(&branches.relax)
                    .arg(&branches.dephase)
                    .arg(&branches.excite)
                    .arg(&branches.excited)
                    .arg(&branches.decay_keep)
                    .arg(&branches.dephase_keep)
                    .arg(&branches.hot[0])
                    .arg(&branches.hot[1])
                    .arg(&branches.cold_scale)
                    .arg(&branches.cold_keep)
                    .arg(r);
            }
            NoiseDecide::Kraus { set, r_unit } => {
                builder.arg(*set).arg(r_unit);
            }
        }
        builder.arg(&eps);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        // SAFETY: signature matches kernel; num_blocks * 2*BLOCK_SIZE covers `count`
        // elements (or pairs), the partials hold `columns` f64s per block, the ticket is
        // zero between launches, and only thread 0 of the last block writes the nine f64s
        // of `decision`. `norm_log` is null or the live measure log.
        unsafe { launch_counted(builder, cfg, name, device, 1) }?;
    }

    let func = device.function("apply_gate_1q_from_buffer")?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer_mut().raw_mut())
        .arg(&pairs)
        .arg(&qubit_i)
        .arg(decision.raw());
    // SAFETY: signature matches kernel; the grid covers `pairs` and each thread writes
    // only its own amplitude pair. `decision` is written by the launch before, in stream
    // order.
    unsafe {
        launch_counted(
            builder,
            linear_cfg(BLOCK_SIZE, grid_for(pairs, BLOCK_SIZE)),
            "apply_gate_1q_from_buffer",
            device,
            1,
        )
    }
}

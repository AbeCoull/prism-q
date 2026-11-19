//! Device tableau path of [`StabilizerBackend`]: attaching a GPU tableau, the batched
//! Clifford queue, and measurement, reset and readback on the device.

use std::sync::Arc;

use super::{StabilizerBackend, kernels};
use crate::backend::{Backend, dense_probability_len};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::gpu::{GpuContext, GpuTableau};

/// Most collapses a device measurement run launches between two readbacks of the
/// random mask. The chunk starts at one and doubles, so a run that one collapse settles
/// wastes one target of launches and an all-random run pays `log2` of this plus
/// `M / 256` readbacks.
const MAX_COLLAPSE_CHUNK: usize = 256;

impl StabilizerBackend {
    /// Commit to GPU mode on a live device tableau: the host rows are dropped
    /// and the SGI counters zeroed, since neither describes the tableau while
    /// it lives on the device.
    pub(super) fn attach_device_tableau(
        &mut self,
        tableau: GpuTableau,
        n: usize,
        nw: usize,
        num_classical_bits: usize,
    ) {
        self.n = n;
        self.num_words = nw;
        self.xz.clear();
        self.phase.clear();
        self.qubit_active = Vec::new();
        self.total_weight = 0;
        self.sgi_max_active = 0;
        self.lazy_destab = false;
        self.gate_row_start = 0;
        self.gpu_tableau = Some(tableau);
        self.pending_gpu_ops.clear();
        self.gpu_batch_scratch.clear();
        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
    }

    /// Opt into GPU acceleration using the given shared execution context.
    ///
    /// When set, [`Backend::init`] allocates a device-resident tableau instead of a
    /// host `Vec<u64>` and gate application routes through GPU kernels.
    pub fn with_gpu(mut self, context: Arc<GpuContext>) -> Self {
        self.gpu_context = Some(context);
        self
    }

    /// Opt into GPU acceleration but fall back to the host tableau if the device
    /// allocation fails at [`Backend::init`], instead of erroring. Mirrors
    /// [`StatevectorBackend::with_gpu_auto`](crate::backend::statevector::StatevectorBackend::with_gpu_auto);
    /// used by the `Auto` GPU dispatch path, where an unfit or racing device must
    /// degrade to CPU rather than surface a failure.
    pub fn with_gpu_auto(mut self, context: Arc<GpuContext>) -> Self {
        self.gpu_context = Some(context);
        self.gpu_soft = true;
        self
    }

    /// GPU measurement. Flush queued Cliffords, then run the
    /// Aaronson-Gottesman pivot search plus cascade entirely on-device.
    /// Only two small host roundtrips remain: the pivot sentinel (i32) and,
    /// on the deterministic branch, the outcome byte (u8).
    pub(super) fn apply_measure_gpu(&mut self, qubit: usize, classical_bit: usize) -> Result<()> {
        use crate::gpu::kernels::stabilizer as k;
        use rand::RngExt;
        self.flush_gpu_ops()?;
        let ctx = self
            .gpu_context
            .as_ref()
            .expect("apply_measure_gpu called without gpu_context")
            .clone();
        // Find the pivot first so the RNG is only advanced on the random
        // branch, matching the CPU `measure_random` draw pattern.
        let pivot = {
            let Some(tableau) = self.gpu_tableau.as_mut() else {
                unreachable!("apply_measure_gpu called without gpu_tableau")
            };
            k::launch_measure_find_pivot(&ctx, tableau, qubit)?
        };
        let outcome = if let Some(pivot_row) = pivot {
            let random_outcome: bool = self.rng.random();
            let Some(tableau) = self.gpu_tableau.as_mut() else {
                unreachable!("gpu_tableau was Some above the RNG draw")
            };
            k::launch_measure_cascade(&ctx, tableau, qubit, pivot_row)?;
            k::launch_measure_fixup(&ctx, tableau, qubit, pivot_row, random_outcome)?;
            random_outcome
        } else {
            let Some(tableau) = self.gpu_tableau.as_mut() else {
                unreachable!("gpu_tableau was Some in the pivot scope above")
            };
            k::launch_measure_deterministic(&ctx, tableau, qubit)?
        };
        self.classical_bits[classical_bit] = outcome;
        Ok(())
    }

    /// GPU reset. Mirrors the CPU strategy: measure on-device into a scratch
    /// classical slot, then queue an X on the Clifford batch when the
    /// outcome is 1 so the next flush flips the qubit.
    pub(super) fn apply_reset_gpu(&mut self, qubit: usize) -> Result<()> {
        use crate::gpu::kernels::stabilizer::op;
        let prev_len = self.classical_bits.len();
        self.classical_bits.push(false);
        let scratch = prev_len;
        self.apply_measure_gpu(qubit, scratch)?;
        let outcome = self.classical_bits[scratch];
        self.classical_bits.truncate(prev_len);
        if outcome {
            self.queue_1q_gpu(op::X, qubit);
        }
        Ok(())
    }

    /// Device instruction loop: a `CX(k, k + 1)` chain of at least `MIN_CX_CHAIN` takes
    /// the chain kernel and a measurement run of at least `MIN_MEASURES_FOR_BATCH` goes
    /// through [`Self::apply_measure_run_gpu`], the same spans the host batches;
    /// everything else takes `apply`.
    pub(super) fn apply_instructions_gpu(&mut self, instructions: &[Instruction]) -> Result<()> {
        use kernels::batch::{MIN_CX_CHAIN, MIN_MEASURES_FOR_BATCH, cx_chain_len};
        let mut idx = 0;
        while idx < instructions.len() {
            if let Instruction::Gate {
                gate: Gate::Cx,
                targets,
            } = &instructions[idx]
            {
                let chain = cx_chain_len(&instructions[idx..]);
                if chain >= MIN_CX_CHAIN {
                    self.flush_gpu_ops()?;
                    let ctx = self
                        .gpu_context
                        .as_ref()
                        .expect("apply_instructions_gpu called without gpu_context");
                    let Some(tableau) = self.gpu_tableau.as_mut() else {
                        unreachable!("apply_instructions_gpu called without gpu_tableau")
                    };
                    let start = targets[0];
                    crate::gpu::kernels::stabilizer::launch_cx_chain(
                        ctx,
                        tableau,
                        start,
                        start + chain,
                    )?;
                    idx += chain;
                    continue;
                }
            }
            if matches!(instructions[idx], Instruction::Measure { .. })
                && matches!(
                    instructions.get(idx + MIN_MEASURES_FOR_BATCH - 1),
                    Some(Instruction::Measure { .. })
                )
            {
                let run = self.measure_run_len(&instructions[idx..]);
                if run >= MIN_MEASURES_FOR_BATCH {
                    self.apply_measure_run_gpu(&instructions[idx..idx + run])?;
                    idx += run;
                    continue;
                }
            }
            self.apply(&instructions[idx])?;
            idx += 1;
        }
        self.flush_gpu_ops()
    }

    /// Measure a run of distinct qubits into distinct classical bits on the device.
    ///
    /// Collapses chain through stream order with their state in device cells, so the
    /// host reads back only the random mask between chunks of collapses. A target whose
    /// mask bit is clear stays deterministic for the rest of the run and is resolved at
    /// the end: from the stabilizer signs in two launches when the last mask is empty
    /// (every stabilizer is diagonal), otherwise by the serial kernel. The random bits
    /// are drawn up front from a clone of the RNG and taken in collapse order, and the
    /// RNG then advances by the number taken, so outcomes match the host run bit for
    /// bit under one seed.
    fn apply_measure_run_gpu(&mut self, run: &[Instruction]) -> Result<()> {
        use crate::gpu::kernels::stabilizer as k;
        use rand::RngExt;
        self.flush_gpu_ops()?;
        let ctx = self
            .gpu_context
            .as_ref()
            .expect("apply_measure_run_gpu called without gpu_context")
            .clone();
        let Some(tableau) = self.gpu_tableau.as_mut() else {
            unreachable!("apply_measure_run_gpu called without gpu_tableau")
        };
        let num_meas = run.len();
        let layout = k::MeasureRunLayout::new(num_meas);
        let sentinel = (2 * self.n) as u32;
        let mut words = vec![0u32; layout.len()];
        let mut cbits = Vec::with_capacity(num_meas);
        for (slot, inst) in words[layout.targets()].iter_mut().zip(run) {
            let Instruction::Measure {
                qubit,
                classical_bit,
            } = inst
            else {
                unreachable!("measure_run_len returns only measure runs")
            };
            *slot = *qubit as u32;
            cbits.push(*classical_bit);
        }
        words[layout.pivots()].fill(sentinel);
        let rand = &mut words[layout.rand()];
        match self.forced_outcome {
            Some(true) => rand.fill(u32::MAX),
            Some(false) => {}
            None => {
                let mut draws = self.rng.clone();
                for i in 0..num_meas {
                    if draws.random::<bool>() {
                        rand[i / 32] |= 1 << (i % 32);
                    }
                }
            }
        }
        k::upload_measure_run(&ctx, tableau, layout, &words[..layout.upload_len()])?;

        let targets = &words[layout.targets()];
        let mut mask = vec![0u64; self.num_words];
        let mut remaining: Vec<usize> = (0..num_meas).collect();
        let mut chunk = 1;
        loop {
            k::launch_run_random_mask(&ctx, tableau, &mut mask)?;
            remaining.retain(|&m| {
                let q = targets[m] as usize;
                (mask[q / 64] >> (q % 64)) & 1 != 0
            });
            if remaining.is_empty() {
                break;
            }
            let take = chunk.min(remaining.len());
            for &m in &remaining[..take] {
                k::launch_run_collapse(&ctx, tableau, layout, m)?;
            }
            remaining.drain(..take);
            chunk = (chunk * 2).min(MAX_COLLAPSE_CHUNK);
        }

        if mask.iter().all(|&w| w == 0) {
            k::launch_run_diagonal_finish(&ctx, tableau, layout)?;
            k::read_measure_run(&ctx, tableau, &mut words)?;
        } else {
            k::read_measure_run(&ctx, tableau, &mut words)?;
            let deterministic: Vec<usize> = (0..num_meas)
                .filter(|&m| words[layout.pivots()][m] == sentinel)
                .collect();
            if !deterministic.is_empty() {
                for &m in &deterministic {
                    k::launch_run_deterministic(&ctx, tableau, layout, m)?;
                }
                k::read_measure_run(&ctx, tableau, &mut words)?;
            }
        }

        for (&cbit, &outcome) in cbits.iter().zip(&words[layout.outcomes()]) {
            self.classical_bits[cbit] = outcome != 0;
        }
        if self.forced_outcome.is_none() {
            for _ in 0..words[0] {
                let _: bool = self.rng.random();
            }
        }
        Ok(())
    }

    /// Copy the device tableau back to host and replay any queued Clifford
    /// ops onto the copied buffers.
    ///
    /// Exists so `&self` read paths (probabilities, export_tableau,
    /// export_statevector) can observe post-flush state without mutating the
    /// device tableau or the queue. The queue is left intact so the next
    /// `&mut self` entry point flushes normally. If the queue is empty the
    /// copied buffers are returned unchanged.
    pub(super) fn copy_device_tableau_with_pending(&self) -> Result<(Vec<u64>, Vec<bool>)> {
        use crate::gpu::kernels::stabilizer::{CLIFOP_STRIDE, op};
        let tableau = self
            .gpu_tableau
            .as_ref()
            .expect("copy_device_tableau_with_pending called without gpu_tableau");
        let (xz, phase) = tableau.copy_to_host()?;
        if self.pending_gpu_ops.is_empty() {
            return Ok((xz, phase));
        }
        let mut cpu = StabilizerBackend::new(0);
        cpu.n = self.n;
        cpu.num_words = self.num_words;
        cpu.xz = xz;
        cpu.phase = phase;
        for chunk in self.pending_gpu_ops.as_chunks::<CLIFOP_STRIDE>().0 {
            let opcode = chunk[0];
            let a = chunk[1] as usize;
            let b = chunk[2] as usize;
            match opcode {
                op::H => cpu.dispatch_gate(&Gate::H, &[a])?,
                op::S => cpu.dispatch_gate(&Gate::S, &[a])?,
                op::SDG => cpu.dispatch_gate(&Gate::Sdg, &[a])?,
                op::X => cpu.dispatch_gate(&Gate::X, &[a])?,
                op::Y => cpu.dispatch_gate(&Gate::Y, &[a])?,
                op::Z => cpu.dispatch_gate(&Gate::Z, &[a])?,
                op::SX => cpu.dispatch_gate(&Gate::SX, &[a])?,
                op::SXDG => cpu.dispatch_gate(&Gate::SXdg, &[a])?,
                op::CX => cpu.dispatch_gate(&Gate::Cx, &[a, b])?,
                op::CZ => cpu.dispatch_gate(&Gate::Cz, &[a, b])?,
                op::SWAP => cpu.dispatch_gate(&Gate::Swap, &[a, b])?,
                _ => {
                    return Err(PrismError::BackendUnsupported {
                        backend: self.name().to_string(),
                        operation: format!("unknown queued Clifford opcode {opcode}"),
                    });
                }
            }
        }
        Ok((cpu.xz, cpu.phase))
    }

    /// GPU probabilities. Check the dense output width first, then copy the
    /// tableau back, replaying queued Clifford ops that have not been flushed to
    /// the device so the result matches a fully-flushed state.
    pub(super) fn probabilities_gpu(&self) -> Result<Vec<f64>> {
        dense_probability_len(self.name(), self.n, &self.limits)?;
        let (xz, phase) = self.copy_device_tableau_with_pending()?;
        self.dense_probabilities(&xz, &phase)
    }

    fn queue_1q_gpu(&mut self, opcode: u32, target: usize) {
        self.pending_gpu_ops
            .extend_from_slice(&[opcode, target as u32, 0, 0]);
    }

    fn queue_2q_gpu(&mut self, opcode: u32, a: usize, b: usize) {
        self.pending_gpu_ops
            .extend_from_slice(&[opcode, a as u32, b as u32, 0]);
    }

    /// Drain and apply the queued Clifford op list in a single kernel launch.
    ///
    /// Must be called before any code path that reads the device tableau or
    /// hands control back to the CPU algorithms (copy-back, probabilities,
    /// measurement, reset, the testing-only rowmul helper, statevector
    /// export). Cheap no-op when the queue is empty.
    pub(super) fn flush_gpu_ops(&mut self) -> Result<()> {
        if self.pending_gpu_ops.is_empty() {
            return Ok(());
        }
        let ctx = self
            .gpu_context
            .as_ref()
            .expect("flush_gpu_ops called without gpu_context")
            .clone();
        let tableau = self
            .gpu_tableau
            .as_mut()
            .expect("flush_gpu_ops called without gpu_tableau");
        crate::gpu::kernels::stabilizer::launch_clifford_batch(
            &ctx,
            tableau,
            &self.pending_gpu_ops,
            &mut self.gpu_batch_scratch,
        )?;
        self.pending_gpu_ops.clear();
        Ok(())
    }

    /// Dispatch a Clifford gate for batched GPU execution. Only callable when
    /// `gpu_tableau` is Some and a `gpu_context` is attached; the public
    /// `apply` guard enforces both invariants before this is called.
    ///
    /// Each call appends onto `pending_gpu_ops`; no kernel is launched until
    /// the next `flush_gpu_ops`. Non-Clifford gates fail loudly and leave the
    /// queue intact so diagnostic state is preserved.
    pub(super) fn dispatch_gate_gpu(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        use crate::gpu::kernels::stabilizer::op;
        match gate {
            Gate::Id => Ok(()),
            Gate::H => {
                self.queue_1q_gpu(op::H, targets[0]);
                Ok(())
            }
            Gate::S => {
                self.queue_1q_gpu(op::S, targets[0]);
                Ok(())
            }
            Gate::Sdg => {
                self.queue_1q_gpu(op::SDG, targets[0]);
                Ok(())
            }
            Gate::X => {
                self.queue_1q_gpu(op::X, targets[0]);
                Ok(())
            }
            Gate::Y => {
                self.queue_1q_gpu(op::Y, targets[0]);
                Ok(())
            }
            Gate::Z => {
                self.queue_1q_gpu(op::Z, targets[0]);
                Ok(())
            }
            Gate::SX => {
                self.queue_1q_gpu(op::SX, targets[0]);
                Ok(())
            }
            Gate::SXdg => {
                self.queue_1q_gpu(op::SXDG, targets[0]);
                Ok(())
            }
            Gate::Cx => {
                self.queue_2q_gpu(op::CX, targets[0], targets[1]);
                Ok(())
            }
            Gate::Cz => {
                self.queue_2q_gpu(op::CZ, targets[0], targets[1]);
                Ok(())
            }
            Gate::Swap => {
                self.queue_2q_gpu(op::SWAP, targets[0], targets[1]);
                Ok(())
            }
            Gate::T
            | Gate::Tdg
            | Gate::Rx(_)
            | Gate::Ry(_)
            | Gate::Rz(_)
            | Gate::P(_)
            | Gate::Rzz(_)
            | Gate::Cu(_)
            | Gate::Mcu(_)
            | Gate::Fused(_)
            | Gate::BatchPhase(_)
            | Gate::BatchRzz(_)
            | Gate::DiagonalBatch(_)
            | Gate::MultiFused(_)
            | Gate::Fused2q(_)
            | Gate::Multi2q(_)
            | Gate::QftBlock { .. }
            | Gate::PauliRot(_)
            | Gate::MultiPauliRot(_)
            | Gate::Unitary(_) => Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "non-Clifford gate `{}` (stabilizer backend supports Clifford gates only)",
                    gate.name()
                ),
            }),
        }
    }
}

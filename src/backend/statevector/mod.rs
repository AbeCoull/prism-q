//! Full state-vector simulation backend.
//!
//! Stores the complete 2^n amplitude vector and applies gates via direct
//! index manipulation. This is the reference backend,,
//! but memory-limited to ~25-30 qubits on typical hardware.
//!
//! # Memory layout
//!
//! State is a contiguous `Vec<Complex64>` of length 2^n, indexed by computational
//! basis state in standard binary order (qubit 0 = least significant bit).
//!
//! # Hot-path design
//!
//! - Single-qubit gates: iterate 2^(n-1) pairs with stride 2^target.
//! - CX/CZ/SWAP: specialized routines avoid materializing a 4×4 matrix.
//! - All gate kernels are `#[inline(always)]` to enable LTO to inline across
//!   the dispatch boundary.
//! - No heap allocation during gate application.
//!
//! # Threading strategy
//!
//! The pair-iteration loops are embarrassingly parallel. When the `parallel`
//! feature is enabled and the qubit count meets `PARALLEL_THRESHOLD_QUBITS`
//! (default: 14), each kernel dispatches to a Rayon-parallelized variant
//! using `par_chunks_mut` / `split_at_mut`. All parallel code is fully safe
//! (no `unsafe`). The sequential path is unchanged when the feature is off
//! or the circuit is below threshold.
//!
//! # SIMD strategy
//!
//! The single-qubit gate kernel uses explicit SIMD intrinsics via the shared
//! `simd` module. Complex64 (2×f64 = 128 bits) maps to one `__m128d` register.
//! Matrix entries are precomputed as broadcast pairs `[re, re]` / `[im, im]`.
//! Runtime dispatch: AVX2+FMA (256-bit) > FMA (128-bit) > SSE2 > scalar.
//!
//! Two-qubit gate and measurement parallel inner loops use SIMD bulk helpers
//! (`negate_slice`, `swap_slices`, `norm_sqr_sum`, `zero_slice`)
//! that dispatch to AVX2 implementations when available.

pub(crate) mod kernels;
#[cfg(test)]
mod tests;

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "gpu")]
use std::sync::Arc;

use crate::backend::simd;
use crate::backend::Backend;
use crate::circuit::Instruction;
use crate::error::Result;
use crate::gates::Gate;

#[cfg(feature = "gpu")]
use crate::gpu::{GpuContext, GpuState};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
pub(crate) use super::{MIN_PAR_ELEMS, PARALLEL_THRESHOLD_QUBITS};

#[cfg(feature = "gpu")]
fn reduced_density_matrix_from_state(state: &[Complex64], qubit: usize) -> [[Complex64; 2]; 2] {
    let half = 1usize << qubit;
    let block_size = half << 1;
    let mut p0 = 0.0f64;
    let mut p1 = 0.0f64;
    let mut r = Complex64::new(0.0, 0.0);

    for block in state.chunks(block_size) {
        let (lo, hi) = block.split_at(half);
        for i in 0..half {
            let a0 = lo[i];
            let a1 = hi[i];
            p0 += a0.norm_sqr();
            p1 += a1.norm_sqr();
            r += a1 * a0.conj();
        }
    }

    [
        [Complex64::new(p0, 0.0), r.conj()],
        [r, Complex64::new(p1, 0.0)],
    ]
}

/// Insert a zero bit at `bit_pos`, shifting all higher bits left by one.
///
/// Maps a compact iteration index to a state-vector index with a gap at
/// `bit_pos`. Chaining multiple calls (in ascending `bit_pos` order) creates
/// gaps at all controlled-gate qubit positions.
#[inline(always)]
pub(crate) fn insert_zero_bit(val: usize, bit_pos: usize) -> usize {
    let lo = val & ((1 << bit_pos) - 1);
    let hi = val >> bit_pos;
    (hi << (bit_pos + 1)) | lo
}

/// Wrapper to send a raw pointer across Rayon threads.
///
/// SAFETY: Callers must ensure no data races — each thread must access
/// disjoint elements. The mask-based index bijection guarantees this for
/// controlled-gate kernels.
#[cfg(feature = "parallel")]
#[derive(Copy, Clone)]
pub(crate) struct SendPtr(pub(crate) *mut Complex64);

#[cfg(feature = "parallel")]
unsafe impl Send for SendPtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for SendPtr {}

#[cfg(feature = "parallel")]
impl SendPtr {
    #[inline(always)]
    pub(crate) unsafe fn load(self, idx: usize) -> Complex64 {
        *self.0.add(idx)
    }

    #[inline(always)]
    pub(crate) unsafe fn store(self, idx: usize, val: Complex64) {
        *self.0.add(idx) = val;
    }

    #[inline(always)]
    pub(crate) fn as_f64_ptr(self) -> *mut f64 {
        self.0 as *mut f64
    }
}

/// Full state-vector backend.
pub struct StatevectorBackend {
    pub(crate) num_qubits: usize,
    pub(crate) state: Vec<Complex64>,
    pub(crate) classical_bits: Vec<bool>,
    pub(crate) rng: ChaCha8Rng,
    pub(crate) pending_norm: f64,
    #[cfg(feature = "gpu")]
    gpu_context: Option<Arc<GpuContext>>,
    #[cfg(feature = "gpu")]
    gpu_state: Option<GpuState>,
}

impl StatevectorBackend {
    /// Create a new statevector backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            state: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            pending_norm: 1.0,
            #[cfg(feature = "gpu")]
            gpu_context: None,
            #[cfg(feature = "gpu")]
            gpu_state: None,
        }
    }

    /// Opt into GPU acceleration using the given shared execution context.
    ///
    /// When set, [`Backend::init`] allocates a device-resident state instead of a host
    /// `Vec<Complex64>` and gate application routes through GPU kernels.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, context: Arc<GpuContext>) -> Self {
        self.gpu_context = Some(context);
        self
    }

    #[cfg(feature = "gpu")]
    fn apply_gpu(&mut self, instruction: &Instruction) -> Result<()> {
        debug_assert!(
            self.gpu_state.is_some(),
            "apply_gpu called without gpu_state (callers must check self.gpu_state.is_some() first)"
        );
        // Unconditional GPU dispatch: every instruction routes to a kernel once
        // `gpu_state` is Some. This path is the explicit opt-in surface
        // (`StatevectorBackend::with_gpu(ctx)`) and intentionally skips the
        // dispatch-level crossover — users calling `with_gpu` directly are asking
        // for kernel-level behavior. For size-aware crossover + decomposition-
        // aware routing, enter via `sim::run_with(BackendKind::StatevectorGpu
        // { context })`, which honors `gpu_min_qubits()` per sub-block.
        //
        // Caveat: Multi2q launches one kernel per sub-gate (rare in practice).
        match instruction {
            Instruction::Gate { gate, targets } => self.dispatch_gate_gpu(gate, targets),
            Instruction::Measure {
                qubit,
                classical_bit,
            } => self.apply_measure_gpu(*qubit, *classical_bit),
            Instruction::Reset { qubit } => self.apply_reset_gpu(*qubit),
            Instruction::Barrier { .. } => Ok(()),
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(&self.classical_bits) {
                    self.dispatch_gate_gpu(gate, targets)
                } else {
                    Ok(())
                }
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn dispatch_gate_gpu(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        use crate::gpu::kernels::dense as k;

        let gpu = self
            .gpu_state
            .as_mut()
            .expect("dispatch_gate_gpu called without gpu_state");
        let ctx = gpu.context().clone();

        match gate {
            Gate::Rzz(theta) => k::launch_apply_rzz(&ctx, gpu, targets[0], targets[1], *theta),
            Gate::Cx => k::launch_apply_cx(&ctx, gpu, targets[0], targets[1]),
            Gate::Cz => k::launch_apply_cz(&ctx, gpu, targets[0], targets[1]),
            Gate::Swap => k::launch_apply_swap(&ctx, gpu, targets[0], targets[1]),
            Gate::Cu(mat) => {
                if let Some(phase) = gate.controlled_phase() {
                    k::launch_apply_cu_phase(&ctx, gpu, targets[0], targets[1], phase)
                } else {
                    k::launch_apply_cu(&ctx, gpu, targets[0], targets[1], **mat)
                }
            }
            Gate::Mcu(data) => {
                let num_ctrl = data.num_controls as usize;
                let controls = &targets[..num_ctrl];
                let target = targets[num_ctrl];
                if let Some(phase) = gate.controlled_phase() {
                    k::launch_apply_mcu_phase(&ctx, gpu, controls, target, phase)
                } else {
                    k::launch_apply_mcu(&ctx, gpu, controls, target, data.mat)
                }
            }
            Gate::BatchPhase(data) => {
                let control = targets[0];
                k::launch_apply_batch_phase(&ctx, gpu, control, &data.phases)
            }
            Gate::BatchRzz(data) => k::launch_apply_batch_rzz(&ctx, gpu, &data.edges),
            Gate::DiagonalBatch(data) => k::launch_apply_diagonal_batch(&ctx, gpu, &data.entries),
            Gate::MultiFused(data) => {
                if data.all_diagonal {
                    k::launch_apply_multi_fused_diagonal(&ctx, gpu, &data.gates)
                } else {
                    k::launch_apply_multi_fused_nondiag(&ctx, gpu, &data.gates)
                }
            }
            Gate::Fused2q(mat) => k::launch_apply_fused_2q(&ctx, gpu, targets[0], targets[1], mat),
            Gate::Multi2q(data) => {
                for &(q0, q1, mat) in &data.gates {
                    k::launch_apply_fused_2q(&ctx, gpu, q0, q1, &mat)?;
                }
                Ok(())
            }
            _ => {
                let mat = gate.matrix_2x2();
                if gate.is_diagonal_1q() {
                    k::launch_apply_diagonal_1q(&ctx, gpu, targets[0], mat[0][0], mat[1][1])
                } else {
                    k::launch_apply_gate_1q(&ctx, gpu, targets[0], mat)
                }
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn apply_measure_gpu(&mut self, qubit: usize, classical_bit: usize) -> Result<()> {
        use crate::gpu::kernels::dense as k;
        use rand::Rng;

        let gpu = self
            .gpu_state
            .as_mut()
            .expect("apply_measure_gpu called without gpu_state");
        let ctx = gpu.context().clone();

        let prob_one = k::measure_prob_one(&ctx, gpu, qubit)?;
        let u: f64 = self.rng.random();
        let outcome = u < prob_one;
        self.classical_bits[classical_bit] = outcome;

        k::measure_collapse(&ctx, gpu, qubit, outcome)?;

        let inv_sqrt = crate::backend::measurement_inv_norm(outcome, prob_one);
        gpu.set_pending_norm(gpu.pending_norm() * inv_sqrt);
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn apply_reset_gpu(&mut self, qubit: usize) -> Result<()> {
        use crate::gpu::kernels::dense as k;

        let gpu = self
            .gpu_state
            .as_mut()
            .expect("apply_reset_gpu called without gpu_state");
        let ctx = gpu.context().clone();

        // Match CPU semantics (src/backend/statevector/kernels.rs::apply_reset):
        // deterministic projection onto |0⟩ irrespective of the current amplitude on |1⟩.
        let prob_one = k::measure_prob_one(&ctx, gpu, qubit)?;
        let prob_zero = (1.0 - prob_one).clamp(0.0, 1.0);

        k::measure_collapse(&ctx, gpu, qubit, false)?;

        if prob_zero > crate::backend::NORM_CLAMP_MIN {
            let inv_norm = 1.0 / prob_zero.sqrt();
            gpu.set_pending_norm(gpu.pending_norm() * inv_norm);
        } else {
            // |0⟩ half was empty — reinitialise to |0…0⟩ (CPU does the same).
            k::launch_set_initial_state(&ctx, gpu)?;
            gpu.set_pending_norm(1.0);
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn apply_1q_matrix_gpu(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        use crate::gpu::kernels::dense as k;

        let gpu = self
            .gpu_state
            .as_mut()
            .expect("apply_1q_matrix_gpu called without gpu_state");
        let ctx = gpu.context().clone();

        let is_diagonal = matrix[0][1].norm() < 1e-14 && matrix[1][0].norm() < 1e-14;
        if is_diagonal {
            k::launch_apply_diagonal_1q(&ctx, gpu, qubit, matrix[0][0], matrix[1][1])
        } else {
            k::launch_apply_gate_1q(&ctx, gpu, qubit, *matrix)
        }
    }

    /// Read-only access to the raw amplitude vector.
    ///
    /// After measurements, amplitudes may be un-normalized due to deferred
    /// normalization. Call [`export_statevector`](Self::export_statevector)
    /// for a properly normalized copy.
    pub fn state_vector(&self) -> &[Complex64] {
        &self.state
    }

    /// Initialize the backend from a pre-computed state vector.
    ///
    /// Accepts ownership of the amplitude vector, bypassing the default |0...0⟩
    /// initialization. The vector length must be a power of 2.
    pub fn init_from_state(
        &mut self,
        state: Vec<Complex64>,
        num_classical_bits: usize,
    ) -> crate::error::Result<()> {
        #[cfg(feature = "parallel")]
        crate::backend::init_thread_pool();

        let dim = state.len();
        if !dim.is_power_of_two() || dim < 2 {
            return Err(crate::error::PrismError::InvalidParameter {
                message: format!(
                    "state vector length must be a power of 2 and >= 2, got {}",
                    dim
                ),
            });
        }
        self.num_qubits = dim.trailing_zeros() as usize;
        self.state = state;
        self.pending_norm = 1.0;
        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    #[inline(always)]
    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) {
        match gate {
            Gate::Rzz(theta) => self.apply_rzz(targets[0], targets[1], *theta),
            Gate::Cx => self.apply_cx(targets[0], targets[1]),
            Gate::Cz => self.apply_cz(targets[0], targets[1]),
            Gate::Swap => self.apply_swap(targets[0], targets[1]),
            Gate::Cu(mat) => {
                if let Some(phase) = gate.controlled_phase() {
                    self.apply_cu_phase(targets[0], targets[1], phase);
                } else {
                    self.apply_cu(targets[0], targets[1], **mat);
                }
            }
            Gate::Mcu(data) => {
                let num_ctrl = data.num_controls as usize;
                if let Some(phase) = gate.controlled_phase() {
                    self.apply_mcu_phase(&targets[..num_ctrl], targets[num_ctrl], phase);
                } else {
                    self.apply_mcu(&targets[..num_ctrl], targets[num_ctrl], data.mat);
                }
            }
            Gate::BatchPhase(data) => {
                self.apply_batch_phase(targets[0], &data.phases);
            }
            Gate::BatchRzz(data) => {
                self.apply_batch_rzz(&data.edges);
            }
            Gate::DiagonalBatch(data) => {
                self.apply_diagonal_batch(&data.entries);
            }
            Gate::MultiFused(data) => {
                if data.all_diagonal {
                    self.apply_multi_1q_diagonal(&data.gates);
                } else {
                    self.apply_multi_1q(&data.gates);
                }
            }
            Gate::Fused2q(mat) => {
                self.apply_fused_2q(targets[0], targets[1], mat);
            }
            Gate::Multi2q(data) => {
                self.apply_multi_2q(&data.gates);
            }
            _ => {
                let mat = gate.matrix_2x2();
                if gate.is_diagonal_1q() {
                    self.apply_diagonal_gate(targets[0], mat[0][0], mat[1][1]);
                } else {
                    self.apply_single_gate(targets[0], mat);
                }
            }
        }
    }
}

impl Backend for StatevectorBackend {
    fn name(&self) -> &'static str {
        "statevector"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        #[cfg(feature = "parallel")]
        crate::backend::init_thread_pool();

        self.num_qubits = num_qubits;
        self.pending_norm = 1.0;
        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);

        #[cfg(feature = "gpu")]
        if let Some(ctx) = self.gpu_context.clone() {
            self.state.clear();
            self.gpu_state = Some(GpuState::new(ctx, num_qubits)?);
            return Ok(());
        }

        let dim = 1usize << num_qubits;
        if self.state.len() == dim {
            self.state.fill(Complex64::new(0.0, 0.0));
        } else {
            self.state = vec![Complex64::new(0.0, 0.0); dim];
        }
        self.state[0] = Complex64::new(1.0, 0.0);
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        #[cfg(feature = "gpu")]
        if self.gpu_state.is_some() {
            return self.apply_gpu(instruction);
        }
        match instruction {
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets),
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.apply_measure(*qubit, *classical_bit);
            }
            Instruction::Reset { qubit } => {
                self.apply_reset(*qubit);
            }
            Instruction::Barrier { .. } => {}
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(&self.classical_bits) {
                    self.dispatch_gate(gate, targets);
                }
            }
        }
        Ok(())
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_state.as_ref() {
            return gpu.probabilities();
        }
        let norm_sq = self.pending_norm * self.pending_norm;
        let dim = self.state.len();
        let mut probs = vec![0.0_f64; dim];

        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            let src_chunks = self.state.par_chunks(MIN_PAR_ELEMS);
            let dst_chunks = probs.par_chunks_mut(MIN_PAR_ELEMS);
            if norm_sq == 1.0 {
                src_chunks.zip(dst_chunks).for_each(|(s, d)| {
                    simd::norm_sqr_to_slice(s, d);
                });
            } else {
                src_chunks.zip(dst_chunks).for_each(|(s, d)| {
                    simd::norm_sqr_to_slice_scaled(s, d, norm_sq);
                });
            }
            return Ok(probs);
        }

        if norm_sq == 1.0 {
            simd::norm_sqr_to_slice(&self.state, &mut probs);
        } else {
            simd::norm_sqr_to_slice_scaled(&self.state, &mut probs, norm_sq);
        }
        Ok(probs)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn export_statevector(&self) -> crate::error::Result<Vec<Complex64>> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_state.as_ref() {
            return gpu.export_statevector();
        }
        if self.pending_norm == 1.0 {
            return Ok(self.state.clone());
        }
        let s = Complex64::new(self.pending_norm, 0.0);
        Ok(self.state.iter().map(|&c| c * s).collect())
    }

    fn qubit_probability(&self, qubit: usize) -> crate::error::Result<f64> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_state.as_ref() {
            return crate::gpu::kernels::dense::measure_prob_one(gpu.context(), gpu, qubit);
        }
        Ok(self.qubit_probability_one(qubit))
    }

    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_state.as_ref() {
            let state = gpu.export_statevector()?;
            return Ok(reduced_density_matrix_from_state(&state, qubit));
        }
        Ok(self.reduced_density_matrix_one(qubit))
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        #[cfg(feature = "gpu")]
        if self.gpu_state.is_some() {
            return self.apply_reset_gpu(qubit);
        }
        self.apply_reset(qubit);
        Ok(())
    }

    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        #[cfg(feature = "gpu")]
        if self.gpu_state.is_some() {
            return self.apply_1q_matrix_gpu(qubit, matrix);
        }
        let is_diagonal = matrix[0][1].norm() < 1e-14 && matrix[1][0].norm() < 1e-14;
        if is_diagonal {
            self.apply_diagonal_gate(qubit, matrix[0][0], matrix[1][1]);
        } else {
            self.apply_single_gate(qubit, *matrix);
        }
        Ok(())
    }
}

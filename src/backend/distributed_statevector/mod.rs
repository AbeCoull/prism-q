//! Distributed state vector backend.
//!
//! Partitions the `2^n` amplitude vector across `P = 2^p` ranks. The low
//! `n - p` qubits are *local* (indexing each rank's contiguous slice); the top
//! `p` qubits are *global* (selecting the rank id). Each rank holds a local
//! slice of length `2^(n-p)` inside an inner [`StatevectorBackend`]. Gates on
//! local qubits reuse the existing SIMD kernels.
//!
//! # Memory layout
//!
//! The global amplitude index is `rank * 2^(n-p) + local_index`. Qubit `q` with
//! `q < n - p` is bit `q` of `local_index`; qubit `q >= n - p` is bit
//! `q - (n - p)` of the rank id. Qubit 0 remains the least significant bit, so
//! `|0...0>` lives at `local_index = 0` on rank 0.
//!
//! # Status
//!
//! Implemented: local gates, one qubit gates on rank bits, global diagonal
//! scaling, and gathers for `probabilities` and `export_statevector`. With a
//! single rank ([`SerialComm`]), every qubit is local.
//!
//! Not implemented yet: gates over multiple qubits that span rank bits,
//! measurement, reset, conditionals, and sampling.

#[cfg(test)]
mod tests;

use std::sync::Arc;

use num_complex::Complex64;

use crate::backend::simd;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::{dense_probability_len, dense_statevector_len, Backend};
use crate::circuit::Instruction;
use crate::distributed::DistributedContext;
use crate::error::{PrismError, Result};

const BACKEND_NAME: &str = "distributed_statevector";

/// Distributed state vector backend over an [`Arc<DistributedContext>`].
pub struct DistributedStatevectorBackend {
    context: Arc<DistributedContext>,
    inner: StatevectorBackend,
    num_qubits: usize,
    global_qubits: usize,
    recv: Vec<Complex64>,
}

impl DistributedStatevectorBackend {
    /// Create a backend bound to the given rank context and RNG seed.
    pub fn new(context: Arc<DistributedContext>, seed: u64) -> Self {
        Self {
            context,
            inner: StatevectorBackend::new(seed),
            num_qubits: 0,
            global_qubits: 0,
            recv: Vec::new(),
        }
    }

    #[inline]
    fn local_qubits(&self) -> usize {
        self.num_qubits - self.global_qubits
    }

    #[inline]
    fn is_single_rank(&self) -> bool {
        self.context.size() == 1
    }

    #[inline]
    fn targets_are_local(&self, targets: &[usize]) -> bool {
        let local = self.local_qubits();
        targets.iter().all(|&q| q < local)
    }

    /// Bit position within the rank id for global qubit `q` (`q >= local`).
    #[inline]
    fn global_bit(&self, q: usize) -> usize {
        q - self.local_qubits()
    }

    /// Whether this rank holds the `|1>` half of global qubit `q`.
    #[inline]
    fn rank_bit_set(&self, q: usize) -> bool {
        (self.context.rank() >> self.global_bit(q)) & 1 == 1
    }

    /// Apply a one qubit gate whose target is stored in the rank id.
    ///
    /// Exchange with the partner rank, then write this rank's half of the 2x2
    /// result.
    fn apply_global_1q(&mut self, target: usize, mat: [[Complex64; 2]; 2]) {
        let partner = self.context.rank() ^ (1usize << self.global_bit(target));
        let len = self.inner.state.len();
        if self.recv.len() != len {
            self.recv.resize(len, Complex64::new(0.0, 0.0));
        }
        self.context
            .comm()
            .sendrecv_c64(partner, &self.inner.state, &mut self.recv);

        let (c_self, c_remote) = if self.rank_bit_set(target) {
            (mat[1][1], mat[1][0])
        } else {
            (mat[0][0], mat[0][1])
        };
        simd::combine_global_half(&mut self.inner.state, &self.recv, c_self, c_remote);
    }

    /// Apply a diagonal one qubit gate whose target is stored in the rank id.
    ///
    /// The rank bit is constant across the local slice, so this only scales the
    /// slice by `d0` or `d1`.
    fn apply_global_diagonal_1q(&mut self, target: usize, d0: Complex64, d1: Complex64) {
        let factor = if self.rank_bit_set(target) { d1 } else { d0 };
        simd::scale_complex_slice(&mut self.inner.state, factor);
    }

    fn unsupported(&self, operation: &str) -> PrismError {
        PrismError::BackendUnsupported {
            backend: BACKEND_NAME.to_string(),
            operation: operation.to_string(),
        }
    }
}

impl Backend for DistributedStatevectorBackend {
    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    fn supports_fused_gates(&self) -> bool {
        // Fusion assumes every target indexes the local slice.
        self.is_single_rank()
    }

    fn supports_qft_block(&self) -> bool {
        self.is_single_rank() && self.inner.supports_qft_block()
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        let size = self.context.size();
        if !size.is_power_of_two() {
            return Err(PrismError::BackendUnsupported {
                backend: BACKEND_NAME.to_string(),
                operation: format!("rank count {size} is not a power of two"),
            });
        }
        let p = size.trailing_zeros() as usize;
        let min_local = crate::distributed::min_local_qubits();
        if size > 1 && num_qubits < p + min_local {
            return Err(PrismError::BackendUnsupported {
                backend: BACKEND_NAME.to_string(),
                operation: format!(
                    "{num_qubits} qubits across {size} ranks leaves fewer than \
                     {min_local} local qubits per rank"
                ),
            });
        }

        self.num_qubits = num_qubits;
        self.global_qubits = p;
        let local_qubits = num_qubits - p;
        self.inner.init(local_qubits, num_classical_bits)?;

        // inner.init seeds index 0 on every rank; only rank 0 owns |0...0>.
        if self.context.rank() != 0 {
            if let Some(amp) = self.inner.state.get_mut(0) {
                *amp = Complex64::new(0.0, 0.0);
            }
        }
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        if self.global_qubits == 0 {
            return self.inner.apply(instruction);
        }
        match instruction {
            Instruction::Gate { gate, targets } => {
                if self.targets_are_local(targets) {
                    return self.inner.apply(instruction);
                }
                if gate.num_qubits() == 1 {
                    let target = targets[0];
                    let mat = gate.matrix_2x2();
                    if gate.is_diagonal_1q() {
                        self.apply_global_diagonal_1q(target, mat[0][0], mat[1][1]);
                    } else {
                        self.apply_global_1q(target, mat);
                    }
                    Ok(())
                } else {
                    Err(self.unsupported("multi-qubit gate spanning a global qubit"))
                }
            }
            Instruction::Barrier { .. } => Ok(()),
            Instruction::Measure { .. } => Err(self.unsupported("distributed measurement")),
            Instruction::Reset { .. } => Err(self.unsupported("distributed reset")),
            Instruction::Conditional { .. } => {
                Err(self.unsupported("distributed classical conditional"))
            }
        }
    }

    fn classical_results(&self) -> &[bool] {
        self.inner.classical_results()
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let local = self.inner.probabilities()?;
        if self.global_qubits == 0 {
            return Ok(local);
        }
        dense_probability_len(BACKEND_NAME, self.num_qubits)?;
        Ok(self.context.comm().allgather_f64(&local))
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let local = self.inner.export_statevector()?;
        if self.global_qubits == 0 {
            return Ok(local);
        }
        dense_statevector_len(BACKEND_NAME, "statevector export", self.num_qubits)?;
        Ok(self.context.comm().allgather_c64(&local))
    }
}

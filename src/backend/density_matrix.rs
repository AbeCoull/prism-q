//! Exact density-matrix backend.
//!
//! Stores the full density operator `rho` for `n` qubits as a `2^(2n)` amplitude
//! buffer laid out row-major: buffer index `(r << n) | c` holds `<r|rho|c>`. That
//! layout is isomorphic to a `2n`-qubit statevector whose high `n` qubits index
//! the ket (row `r`) and whose low `n` qubits index the bra (column `c`). Gate
//! application therefore reuses the validated statevector kernels directly: a
//! unitary `U` on the ket register yields the left product `U rho`, and the same
//! `U` applied to the bra register on a conjugated buffer yields the right product
//! `rho U^dagger`, giving `U rho U^dagger` with no gate math of its own.
//!
//! Memory is `16 * 4^n` bytes (`4^n` `Complex64` entries), so the practical
//! ceiling is about 14 qubits on a 16 GiB host and 15 on a 32 GiB host. CPU only.
//! This backend is explicit-dispatch only and is never chosen by `Auto`.
//!
//! This is the initial core: exact unitary evolution, basis-state probabilities,
//! and the one-qubit reduced density matrix. Measurement, reset, conditionals,
//! and exact noise channels arrive in later milestones. Fusion is disabled
//! (`supports_fused_gates` returns `false`) so every instruction reaching the
//! backend is a primitive whose qubits live in the instruction targets.

use num_complex::Complex64;
use smallvec::SmallVec;

use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;

/// Exact density-matrix simulator. See the module docs for the state layout.
pub struct DensityMatrixBackend {
    num_qubits: usize,
    classical_bits: Vec<bool>,
    sv: StatevectorBackend,
}

impl DensityMatrixBackend {
    /// Create a new density-matrix backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            classical_bits: Vec::new(),
            sv: StatevectorBackend::new(seed),
        }
    }

    /// Purity `Tr(rho^2)`, equal to `1` for a pure state and less otherwise.
    pub fn purity(&self) -> f64 {
        self.sv.state.iter().map(Complex64::norm_sqr).sum()
    }

    #[inline]
    fn dim(&self) -> usize {
        1usize << self.num_qubits
    }

    fn conjugate_buffer(&mut self) {
        for amp in self.sv.state.iter_mut() {
            *amp = amp.conj();
        }
    }

    /// Evolve `rho -> U rho U^dagger` for the unitary `gate` on `targets`.
    ///
    /// `U` on the ket register (targets offset by `n`) is the left product
    /// `U rho`; the same `U` on the bra register (original targets) sandwiched
    /// between two whole-buffer conjugations applies `conj(U)`, giving the right
    /// product `rho U^dagger`.
    fn apply_unitary(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        let n = self.num_qubits;
        let ket_targets: SmallVec<[usize; 4]> = targets.iter().map(|&t| t + n).collect();
        self.sv.apply(&Instruction::Gate {
            gate: gate.clone(),
            targets: ket_targets,
        })?;

        self.conjugate_buffer();
        let bra_targets: SmallVec<[usize; 4]> = targets.iter().copied().collect();
        self.sv.apply(&Instruction::Gate {
            gate: gate.clone(),
            targets: bra_targets,
        })?;
        self.conjugate_buffer();
        Ok(())
    }
}

impl Backend for DensityMatrixBackend {
    fn name(&self) -> &'static str {
        "density_matrix"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.classical_bits = vec![false; num_classical_bits];
        self.sv.init(2 * num_qubits, 0)
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.apply_unitary(gate, targets),
            Instruction::Barrier { .. } => Ok(()),
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Conditional { .. } => Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: "measurement, reset, and conditionals (later density-matrix milestone)"
                    .to_string(),
            }),
        }
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let d = self.dim();
        let mut probs = vec![0.0f64; d];
        for (k, p) in probs.iter_mut().enumerate() {
            *p = self.sv.state[k * d + k].re.max(0.0);
        }
        Ok(probs)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_fused_gates(&self) -> bool {
        false
    }

    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        let n = self.num_qubits;
        let d = self.dim();
        let bit = 1usize << qubit;
        let others = 1usize << (n - 1);
        let mut r00 = Complex64::new(0.0, 0.0);
        let mut r01 = Complex64::new(0.0, 0.0);
        let mut r10 = Complex64::new(0.0, 0.0);
        let mut r11 = Complex64::new(0.0, 0.0);
        for m in 0..others {
            let base = (m & (bit - 1)) | ((m >> qubit) << (qubit + 1));
            let i1 = base | bit;
            r00 += self.sv.state[base * d + base];
            r01 += self.sv.state[base * d + i1];
            r10 += self.sv.state[i1 * d + base];
            r11 += self.sv.state[i1 * d + i1];
        }
        Ok([[r00, r01], [r10, r11]])
    }
}

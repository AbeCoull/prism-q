//! Gate definitions and matrix representations.
//!
//! Gates are represented as an enum for fast dispatch without trait-object overhead
//! in the simulation hot path. Matrix representations use stack-allocated arrays
//! to avoid heap allocation during gate application.
//!
//! # Hot-path design notes
//! - `Gate` methods take `&self` — the enum is 16 bytes (Box indirection for `Fused`).
//! - `matrix_2x2` returns `[[Complex64; 2]; 2]` on the stack.
//! - Two-qubit gates (CX, CZ, SWAP) have dedicated application routines in
//!   backends rather than materializing a 4×4 matrix.

use num_complex::Complex64;
use smallvec::SmallVec;
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::fmt;

/// Threshold for detecting near-zero matrix elements (norm_sqr).
///
/// Used in `preserves_sparsity()` to test if off-diagonal or diagonal entries
/// are effectively zero, indicating a permutation/diagonal gate structure.
const NEAR_ZERO_NORM_SQ: f64 = 1e-24;

/// Threshold for detecting identity-like matrices (element norm).
///
/// Used in `is_diagonal_1q()` for fused gate diagonal detection and in
/// `controlled_phase()` for phase-gate structure recognition.
const IDENTITY_EPS: f64 = 1e-12;

/// Quantum gate identifier.
///
/// Covers the v0 supported gate set. Most variants are data-free or carry an `f64`
/// parameter inline. The `Fused` variant uses `Box` to keep the enum at 16 bytes.
#[derive(Debug, Clone, PartialEq)]
pub enum Gate {
    /// Identity.
    Id,
    /// Pauli-X (bit flip).
    X,
    /// Pauli-Y.
    Y,
    /// Pauli-Z (phase flip).
    Z,
    /// Hadamard.
    H,
    /// S gate (√Z).
    S,
    /// S† gate.
    Sdg,
    /// T gate (π/8).
    T,
    /// T† gate.
    Tdg,
    /// √X gate.
    SX,
    /// √X† gate.
    SXdg,

    /// Rotation about X-axis by angle (radians).
    Rx(f64),
    /// Rotation about Y-axis by angle (radians).
    Ry(f64),
    /// Rotation about Z-axis by angle (radians).
    Rz(f64),
    /// Phase gate `[[1,0],[0,e^{iθ}]]`.
    P(f64),

    /// ZZ rotation: diag(e^{-iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{-iθ/2}).
    /// Qubit order: [q0, q1] (symmetric).
    Rzz(f64),

    /// Controlled-X (CNOT). Qubit order: [control, target].
    Cx,
    /// Controlled-Z. Qubit order: [q0, q1] (symmetric).
    Cz,
    /// SWAP. Qubit order: [q0, q1] (symmetric).
    Swap,

    /// Controlled-unitary. Applies the boxed 2×2 matrix to the target qubit
    /// only when the control qubit is |1⟩. Qubit order: [control, target].
    /// Boxed to keep `Gate` at 16 bytes.
    Cu(Box<[[Complex64; 2]; 2]>),

    /// Multi-controlled unitary. Applies the 2×2 matrix to the target qubit
    /// only when all control qubits are |1⟩. Qubit order:
    /// `[ctrl_0, ctrl_1, ..., ctrl_{k-1}, target]`.
    /// Boxed to keep `Gate` at 16 bytes.
    Mcu(Box<McuData>),

    /// Pre-fused single-qubit unitary (product of consecutive gates on the same target).
    /// Boxed to keep `Gate` at 16 bytes for cache-friendly instruction streams.
    Fused(Box<[[Complex64; 2]; 2]>),

    /// Batched controlled-phase: multiple cphase gates sharing a control qubit,
    /// fused into a single pass over the statevector. Created by the cphase
    /// fusion pass. Targets: `[control]`. The `BatchPhaseData` holds per-target
    /// phases. Boxed to keep `Gate` at 16 bytes.
    BatchPhase(Box<BatchPhaseData>),

    /// Batched ZZ rotations: multiple Rzz gates fused into a single pass.
    /// Created by the batch-Rzz fusion pass. The `BatchRzzData` holds per-edge
    /// angles. Boxed to keep `Gate` at 16 bytes.
    BatchRzz(Box<BatchRzzData>),

    /// Batched diagonal gates: a contiguous run of diagonal 1q and 2q gates
    /// collapsed into a single state-vector sweep with a precomputed phase LUT.
    /// Subsumes BatchPhase and BatchRzz for mixed diagonal runs. Created by the
    /// diagonal batch fusion pass. Boxed to keep `Gate` at 16 bytes.
    DiagonalBatch(Box<DiagonalBatchData>),

    /// Multiple single-qubit gates on distinct qubits, batched for a single
    /// tiled pass over the statevector. Created by the multi-gate fusion pass.
    /// Boxed to keep `Gate` at 16 bytes.
    MultiFused(Box<MultiFusedData>),

    /// Pre-fused two-qubit unitary (4×4 matrix). Created by the 2q fusion pass
    /// which absorbs adjacent single-qubit gates into a two-qubit gate.
    /// Boxed to keep `Gate` at 16 bytes.
    Fused2q(Box<[[Complex64; 4]; 4]>),

    /// Multiple two-qubit gates batched for a single tiled pass over the
    /// statevector. Created by the multi-2q fusion pass. Each entry stores
    /// `(q0, q1, 4×4 matrix)`. Boxed to keep `Gate` at 16 bytes.
    Multi2q(Box<Multi2qData>),
}

/// Data for a multi-controlled unitary gate.
#[derive(Debug, Clone, PartialEq)]
pub struct McuData {
    /// 2×2 unitary applied to the target qubit.
    pub mat: [[Complex64; 2]; 2],
    /// Number of control qubits (≥ 2).
    pub num_controls: u8,
}

/// Data for a batched controlled-phase gate.
///
/// Multiple cphase gates sharing a control qubit are fused into one pass.
/// Each entry is `(target_qubit, phase)`. The control qubit is stored in the
/// instruction's `targets[0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchPhaseData {
    pub phases: SmallVec<[(usize, Complex64); 8]>,
}

/// Data for batched ZZ rotations.
///
/// Multiple Rzz gates batched into a single pass over the statevector.
/// Each entry is `(qubit_0, qubit_1, theta)`. All qubits are stored in the
/// instruction's `targets`.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchRzzData {
    pub edges: Vec<(usize, usize, f64)>,
}

/// An individual diagonal phase contribution in a [`DiagonalBatchData`].
#[derive(Debug, Clone, PartialEq)]
pub enum DiagEntry {
    /// Diagonal on a single qubit: `state[i] *= d0` when bit 0, `*= d1` when bit 1.
    Phase1q {
        qubit: usize,
        d0: Complex64,
        d1: Complex64,
    },
    /// Phase on a qubit pair: `state[i] *= phase` when both bits are set (CZ/CPhase).
    Phase2q {
        q0: usize,
        q1: usize,
        phase: Complex64,
    },
    /// Parity-dependent phase (Rzz): `state[i] *= same` when parity is even,
    /// `state[i] *= diff` when parity is odd.
    Parity2q {
        q0: usize,
        q1: usize,
        same: Complex64,
        diff: Complex64,
    },
}

impl DiagEntry {
    pub fn as_1q_matrix(&self) -> Option<(usize, [[Complex64; 2]; 2])> {
        match *self {
            DiagEntry::Phase1q { qubit, d0, d1 } => {
                let z = Complex64::new(0.0, 0.0);
                Some((qubit, [[d0, z], [z, d1]]))
            }
            _ => None,
        }
    }

    pub fn as_2q_matrix(&self) -> Option<(usize, usize, [[Complex64; 4]; 4])> {
        let z = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        match *self {
            DiagEntry::Phase2q { q0, q1, phase } => Some((
                q0,
                q1,
                [
                    [one, z, z, z],
                    [z, one, z, z],
                    [z, z, one, z],
                    [z, z, z, phase],
                ],
            )),
            DiagEntry::Parity2q {
                q0, q1, same, diff, ..
            } => Some((
                q0,
                q1,
                [
                    [same, z, z, z],
                    [z, diff, z, z],
                    [z, z, diff, z],
                    [z, z, z, same],
                ],
            )),
            _ => None,
        }
    }
}

/// Data for a batched diagonal gate pass.
///
/// A contiguous run of diagonal gates collapsed into a precomputed phase LUT.
/// The `entries` describe individual phase contributions; the kernel extracts
/// unique qubits, builds a LUT indexed by their bits, and applies in one sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagonalBatchData {
    pub entries: Vec<DiagEntry>,
}

/// Data for multi-gate single-pass fusion.
///
/// Batches consecutive single-qubit gates on distinct qubits into one tiled
/// pass over the statevector. Each entry is `(target_qubit, 2×2 matrix)`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiFusedData {
    pub gates: Vec<(usize, [[Complex64; 2]; 2])>,
    pub all_diagonal: bool,
}

/// Data for multi-2q tiled pass fusion.
///
/// Batches consecutive two-qubit gates into a single cache-tiled pass over the
/// statevector. Each entry is `(q0, q1, 4×4 matrix)`. Gate order is preserved.
#[derive(Debug, Clone, PartialEq)]
pub struct Multi2qData {
    pub gates: Vec<(usize, usize, [[Complex64; 4]; 4])>,
}

/// Kronecker product of two 2×2 matrices: A ⊗ B → 4×4.
///
/// Result indices: `(i*2+j, k*2+l) = A[i][k] * B[j][l]`
/// where i,k index A (targets\[0\]) and j,l index B (targets\[1\]).
#[inline]
pub(crate) fn kron_2x2(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 4]; 4] {
    let mut result = [[Complex64::new(0.0, 0.0); 4]; 4];
    for i in 0..2 {
        for k in 0..2 {
            let aik = a[i][k];
            for j in 0..2 {
                for l in 0..2 {
                    result[i * 2 + j][k * 2 + l] = aik * b[j][l];
                }
            }
        }
    }
    result
}

/// Product of two 4×4 matrices: A · B.
#[inline]
pub(crate) fn mat_mul_4x4(a: &[[Complex64; 4]; 4], b: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let mut result = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = zero;
            for k in 0..4 {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    result
}

/// Conjugate-transpose of a 4×4 matrix (U†).
fn adjoint_4x4(m: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let mut result = [[Complex64::new(0.0, 0.0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = m[j][i].conj();
        }
    }
    result
}

/// Conjugate-transpose of a 2×2 matrix (U†).
fn adjoint_2x2(m: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [m[0][0].conj(), m[1][0].conj()],
        [m[0][1].conj(), m[1][1].conj()],
    ]
}

/// Product of two 2×2 matrices: A · B.
#[inline]
pub(crate) fn mat_mul_2x2(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

impl Gate {
    /// Number of qubits this gate acts on.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        match self {
            Gate::Rzz(_) | Gate::Cx | Gate::Cz | Gate::Swap | Gate::Cu(_) | Gate::Fused2q(_) => 2,
            Gate::Mcu(data) => data.num_controls as usize + 1,
            Gate::BatchPhase(data) => 1 + data.phases.len(),
            Gate::BatchRzz(data) => {
                let mut count = 0;
                let mut seen = [false; 64];
                for &(q0, q1, _) in &data.edges {
                    if !seen[q0] {
                        seen[q0] = true;
                        count += 1;
                    }
                    if !seen[q1] {
                        seen[q1] = true;
                        count += 1;
                    }
                }
                count
            }
            Gate::DiagonalBatch(data) => {
                let mut count = 0;
                let mut seen = [false; 64];
                for e in &data.entries {
                    let qs = match e {
                        DiagEntry::Phase1q { qubit, .. } => [*qubit, usize::MAX],
                        DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } => {
                            [*q0, *q1]
                        }
                    };
                    for &q in &qs {
                        if q < 64 && !seen[q] {
                            seen[q] = true;
                            count += 1;
                        }
                    }
                }
                count
            }
            Gate::MultiFused(data) => data.gates.len(),
            Gate::Multi2q(data) => {
                let mut count = 0;
                let mut seen = [false; 64];
                for &(q0, q1, _) in &data.gates {
                    if !seen[q0] {
                        seen[q0] = true;
                        count += 1;
                    }
                    if !seen[q1] {
                        seen[q1] = true;
                        count += 1;
                    }
                }
                count
            }
            _ => 1,
        }
    }

    /// Returns the 2×2 unitary matrix for single-qubit gates.
    ///
    /// # Panics
    /// Panics if called on a multi-qubit or batch gate (`Cx`, `Cz`, `Swap`,
    /// `Cu`, `Mcu`, `BatchPhase`, `MultiFused`, `Fused2q`, `Multi2q`).
    #[inline]
    pub fn matrix_2x2(&self) -> [[Complex64; 2]; 2] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let neg_i = Complex64::new(0.0, -1.0);
        let h = Complex64::new(FRAC_1_SQRT_2, 0.0);

        match self {
            Gate::Id => [[one, zero], [zero, one]],
            Gate::X => [[zero, one], [one, zero]],
            Gate::Y => [[zero, neg_i], [i, zero]],
            Gate::Z => [[one, zero], [zero, -one]],
            Gate::H => [[h, h], [h, -h]],
            Gate::S => [[one, zero], [zero, i]],
            Gate::Sdg => [[one, zero], [zero, neg_i]],
            Gate::T => {
                let phase = Complex64::from_polar(1.0, PI / 4.0);
                [[one, zero], [zero, phase]]
            }
            Gate::Tdg => {
                let phase = Complex64::from_polar(1.0, -PI / 4.0);
                [[one, zero], [zero, phase]]
            }
            Gate::SX => {
                let half = Complex64::new(0.5, 0.0);
                let half_i = Complex64::new(0.0, 0.5);
                [
                    [half + half_i, half - half_i],
                    [half - half_i, half + half_i],
                ]
            }
            Gate::SXdg => {
                let half = Complex64::new(0.5, 0.0);
                let half_i = Complex64::new(0.0, 0.5);
                [
                    [half - half_i, half + half_i],
                    [half + half_i, half - half_i],
                ]
            }
            Gate::Rx(theta) => {
                let c = Complex64::new((theta / 2.0).cos(), 0.0);
                let s = Complex64::new(0.0, -(theta / 2.0).sin());
                [[c, s], [s, c]]
            }
            Gate::Ry(theta) => {
                let c = Complex64::new((theta / 2.0).cos(), 0.0);
                let s = Complex64::new((theta / 2.0).sin(), 0.0);
                [[c, -s], [s, c]]
            }
            Gate::Rz(theta) => {
                let e_neg = Complex64::from_polar(1.0, -theta / 2.0);
                let e_pos = Complex64::from_polar(1.0, theta / 2.0);
                [[e_neg, zero], [zero, e_pos]]
            }
            Gate::P(theta) => {
                let phase = Complex64::from_polar(1.0, *theta);
                [[one, zero], [zero, phase]]
            }
            Gate::Fused(mat) => **mat,
            Gate::Rzz(_)
            | Gate::Cx
            | Gate::Cz
            | Gate::Swap
            | Gate::Cu(_)
            | Gate::Mcu(_)
            | Gate::BatchPhase(_)
            | Gate::BatchRzz(_)
            | Gate::DiagonalBatch(_)
            | Gate::MultiFused(_)
            | Gate::Fused2q(_)
            | Gate::Multi2q(_) => {
                panic!(
                    "matrix_2x2 called on {}-qubit gate `{}`; use dedicated backend routine",
                    self.num_qubits(),
                    self.name()
                )
            }
        }
    }

    /// Returns the 4×4 unitary matrix for two-qubit gates.
    ///
    /// Matrix indices follow the convention: row/col `i*2+j` where `i` indexes
    /// `targets[0]` and `j` indexes `targets[1]`.
    ///
    /// # Panics
    /// Panics on gates other than `Cx`, `Cz`, `Swap`, `Cu`, or `Fused2q`.
    pub fn matrix_4x4(&self) -> [[Complex64; 4]; 4] {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let m = Complex64::new(-1.0, 0.0);
        match self {
            Gate::Rzz(theta) => {
                let ps = Complex64::from_polar(1.0, -theta / 2.0);
                let pd = Complex64::from_polar(1.0, theta / 2.0);
                [[ps, z, z, z], [z, pd, z, z], [z, z, pd, z], [z, z, z, ps]]
            }
            Gate::Cx => [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]],
            Gate::Cz => [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, m]],
            Gate::Swap => [[o, z, z, z], [z, z, o, z], [z, o, z, z], [z, z, z, o]],
            Gate::Cu(mat) => [
                [o, z, z, z],
                [z, o, z, z],
                [z, z, mat[0][0], mat[0][1]],
                [z, z, mat[1][0], mat[1][1]],
            ],
            Gate::Fused2q(mat) => **mat,
            _ => panic!(
                "matrix_4x4 called on non-standard-2q gate `{}`",
                self.name()
            ),
        }
    }

    /// Human-readable gate name (for errors, logs, and OpenQASM round-tripping).
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            Gate::Id => "id",
            Gate::X => "x",
            Gate::Y => "y",
            Gate::Z => "z",
            Gate::H => "h",
            Gate::S => "s",
            Gate::Sdg => "sdg",
            Gate::T => "t",
            Gate::Tdg => "tdg",
            Gate::SX => "sx",
            Gate::SXdg => "sxdg",
            Gate::Rx(_) => "rx",
            Gate::Ry(_) => "ry",
            Gate::Rz(_) => "rz",
            Gate::P(_) => "p",
            Gate::Rzz(_) => "rzz",
            Gate::Cx => "cx",
            Gate::Cz => "cz",
            Gate::Swap => "swap",
            Gate::Cu(_) => "cu",
            Gate::Mcu(_) => "mcu",
            Gate::Fused(_) => "fused",
            Gate::BatchPhase(_) => "batch_phase",
            Gate::BatchRzz(_) => "batch_rzz",
            Gate::DiagonalBatch(_) => "diagonal_batch",
            Gate::MultiFused(_) => "multi_fused",
            Gate::Fused2q(_) => "fused_2q",
            Gate::Multi2q(_) => "multi_2q",
        }
    }

    /// Compute the inverse (adjoint) of this gate.
    pub fn inverse(&self) -> Gate {
        match self {
            Gate::Id | Gate::X | Gate::Y | Gate::Z | Gate::H => self.clone(),
            Gate::S => Gate::Sdg,
            Gate::Sdg => Gate::S,
            Gate::T => Gate::Tdg,
            Gate::Tdg => Gate::T,
            Gate::SX => Gate::SXdg,
            Gate::SXdg => Gate::SX,
            Gate::Rx(theta) => Gate::Rx(-theta),
            Gate::Ry(theta) => Gate::Ry(-theta),
            Gate::Rz(theta) => Gate::Rz(-theta),
            Gate::P(theta) => Gate::P(-theta),
            Gate::Rzz(theta) => Gate::Rzz(-theta),
            Gate::Cx | Gate::Cz | Gate::Swap => self.clone(),
            Gate::Cu(mat) => Gate::cu(adjoint_2x2(mat)),
            Gate::Mcu(data) => Gate::mcu(adjoint_2x2(&data.mat), data.num_controls),
            Gate::Fused(mat) => Gate::Fused(Box::new(adjoint_2x2(mat))),
            Gate::BatchPhase(data) => Gate::BatchPhase(Box::new(BatchPhaseData {
                phases: data.phases.iter().map(|&(q, p)| (q, p.conj())).collect(),
            })),
            Gate::BatchRzz(data) => Gate::BatchRzz(Box::new(BatchRzzData {
                edges: data
                    .edges
                    .iter()
                    .map(|&(q0, q1, theta)| (q0, q1, -theta))
                    .collect(),
            })),
            Gate::DiagonalBatch(data) => Gate::DiagonalBatch(Box::new(DiagonalBatchData {
                entries: data
                    .entries
                    .iter()
                    .map(|e| match e {
                        DiagEntry::Phase1q { qubit, d0, d1 } => DiagEntry::Phase1q {
                            qubit: *qubit,
                            d0: d0.conj(),
                            d1: d1.conj(),
                        },
                        DiagEntry::Phase2q { q0, q1, phase } => DiagEntry::Phase2q {
                            q0: *q0,
                            q1: *q1,
                            phase: phase.conj(),
                        },
                        DiagEntry::Parity2q { q0, q1, same, diff } => DiagEntry::Parity2q {
                            q0: *q0,
                            q1: *q1,
                            same: same.conj(),
                            diff: diff.conj(),
                        },
                    })
                    .collect(),
            })),
            Gate::MultiFused(data) => Gate::MultiFused(Box::new(MultiFusedData {
                gates: data
                    .gates
                    .iter()
                    .map(|&(target, mat)| (target, adjoint_2x2(&mat)))
                    .collect(),
                all_diagonal: data.all_diagonal,
            })),
            Gate::Fused2q(mat) => Gate::Fused2q(Box::new(adjoint_4x4(mat))),
            Gate::Multi2q(data) => Gate::Multi2q(Box::new(Multi2qData {
                gates: data
                    .gates
                    .iter()
                    .rev()
                    .map(|&(q0, q1, ref mat)| (q0, q1, adjoint_4x4(mat)))
                    .collect(),
            })),
        }
    }

    /// Compute integer power of a single-qubit gate.
    ///
    /// Returns the gate raised to the `k`-th power. Negative `k` inverts first.
    /// Only valid for single-qubit gates.
    pub fn matrix_power(&self, k: i64) -> Gate {
        debug_assert_eq!(
            self.num_qubits(),
            1,
            "matrix_power only for single-qubit gates"
        );
        if k == 0 {
            return Gate::Id;
        }
        if k == 1 {
            return self.clone();
        }
        let base = if k < 0 { self.inverse() } else { self.clone() };
        let n = k.unsigned_abs() as usize;
        if n == 1 {
            return base;
        }
        let base_mat = base.matrix_2x2();
        let mut acc = base_mat;
        for _ in 1..n {
            acc = mat_mul_2x2(&base_mat, &acc);
        }
        Gate::Fused(Box::new(acc))
    }

    /// Create a single-controlled unitary gate with the given 2x2 matrix.
    pub fn cu(mat: [[Complex64; 2]; 2]) -> Gate {
        Gate::Cu(Box::new(mat))
    }

    /// Create a multi-controlled unitary gate with `num_controls` control qubits.
    pub fn mcu(mat: [[Complex64; 2]; 2], num_controls: u8) -> Gate {
        Gate::Mcu(Box::new(McuData { mat, num_controls }))
    }

    /// Create a controlled-phase gate CPhase(θ) = Cu(\[\[1,0\],\[0,e^{iθ}\]\]).
    ///
    /// Applies phase e^{iθ} to |11⟩ and identity to all other basis states.
    pub fn cphase(theta: f64) -> Gate {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let phase = Complex64::from_polar(1.0, theta);
        Gate::cu([[one, zero], [zero, phase]])
    }

    /// Returns the phase if this is a controlled-phase gate (Cu/Mcu with
    /// diagonal matrix `[[1,0],[0,e^{iθ}]]`).
    ///
    /// Used by backends to dispatch to optimized phase-only kernels that
    /// touch half the memory of the generic controlled-unitary kernel.
    #[inline]
    pub fn controlled_phase(&self) -> Option<Complex64> {
        let mat = match self {
            Gate::Cu(mat) => &**mat,
            Gate::Mcu(data) => &data.mat,
            _ => return None,
        };
        if (mat[0][0].re - 1.0).abs() < IDENTITY_EPS
            && mat[0][0].im.abs() < IDENTITY_EPS
            && mat[0][1].norm() < IDENTITY_EPS
            && mat[1][0].norm() < IDENTITY_EPS
            && (mat[1][1].norm() - 1.0).abs() < IDENTITY_EPS
        {
            Some(mat[1][1])
        } else {
            None
        }
    }

    /// True if this is a diagonal single-qubit gate (matrix is `[[a,0],[0,b]]`).
    ///
    /// Diagonal gates commute with CX on the control qubit and with CZ on
    /// either qubit. Used by the commutation-aware reordering pass.
    #[inline]
    pub fn is_diagonal_1q(&self) -> bool {
        match self {
            Gate::Id
            | Gate::Z
            | Gate::S
            | Gate::Sdg
            | Gate::T
            | Gate::Tdg
            | Gate::Rz(_)
            | Gate::P(_) => true,
            Gate::Fused(m) => m[0][1].norm() < IDENTITY_EPS && m[1][0].norm() < IDENTITY_EPS,
            _ => false,
        }
    }

    /// True if this is a self-inverse two-qubit gate (applying it twice = identity).
    #[inline]
    pub fn is_self_inverse_2q(&self) -> bool {
        matches!(self, Gate::Cx | Gate::Cz | Gate::Swap)
    }

    /// True if this gate maps computational basis states to computational basis
    /// states (with at most a phase). Such gates preserve the number of non-zero
    /// amplitudes, making the sparse backend optimal (O(1) memory for |0...0⟩).
    ///
    /// Includes diagonal gates (Z, S, T, Rz, P, CZ) and permutation gates
    /// (X, Y, CX, SWAP). Excludes superposition-creating gates (H, Rx, Ry, SX).
    #[inline]
    pub fn preserves_sparsity(&self) -> bool {
        match self {
            Gate::Id | Gate::X | Gate::Y | Gate::Z => true,
            Gate::S | Gate::Sdg | Gate::T | Gate::Tdg => true,
            Gate::Rz(_) | Gate::P(_) => true,
            Gate::Rzz(_) | Gate::Cx | Gate::Cz | Gate::Swap => true,
            Gate::Cu(mat) | Gate::Fused(mat) => {
                let is_diag = mat[0][1].norm_sqr() < NEAR_ZERO_NORM_SQ
                    && mat[1][0].norm_sqr() < NEAR_ZERO_NORM_SQ;
                let is_antidiag = mat[0][0].norm_sqr() < NEAR_ZERO_NORM_SQ
                    && mat[1][1].norm_sqr() < NEAR_ZERO_NORM_SQ;
                is_diag || is_antidiag
            }
            Gate::Mcu(data) => {
                let m = &data.mat;
                let is_diag = m[0][1].norm_sqr() < NEAR_ZERO_NORM_SQ
                    && m[1][0].norm_sqr() < NEAR_ZERO_NORM_SQ;
                let is_antidiag = m[0][0].norm_sqr() < NEAR_ZERO_NORM_SQ
                    && m[1][1].norm_sqr() < NEAR_ZERO_NORM_SQ;
                is_diag || is_antidiag
            }
            Gate::BatchPhase(_) | Gate::BatchRzz(_) | Gate::DiagonalBatch(_) => true,
            _ => false,
        }
    }

    /// Try to recognize a 2x2 unitary matrix as a named gate (up to global phase).
    ///
    /// Used by the fusion pass to emit named gate variants instead of opaque
    /// `Gate::Fused` matrices, enabling downstream passes (e.g. `clifford_prefix_split`)
    /// to identify Clifford gates that arose from fusion (e.g. T·T → S).
    pub fn recognize_matrix(mat: &[[Complex64; 2]; 2]) -> Option<Gate> {
        const EPS: f64 = 1e-10;

        // Check each candidate gate. For each, compute the global phase ratio
        // mat[i][j] / ref[i][j] using the first non-zero entry, then verify
        // all other entries match under that same phase.
        let candidates: &[Gate] = &[
            Gate::H,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::S,
            Gate::Sdg,
            Gate::T,
            Gate::Tdg,
            Gate::SX,
            Gate::SXdg,
        ];

        for candidate in candidates {
            let ref_mat = candidate.matrix_2x2();
            if matrices_equal_up_to_phase(mat, &ref_mat, EPS) {
                return Some(candidate.clone());
            }
        }

        // Identity check: all off-diagonal zero, diagonal entries equal
        if mat[0][1].norm_sqr() < EPS
            && mat[1][0].norm_sqr() < EPS
            && (mat[0][0] - mat[1][1]).norm_sqr() < EPS
            && mat[0][0].norm_sqr() > EPS
        {
            return Some(Gate::Id);
        }

        None
    }

    /// True if this gate is a Clifford gate (relevant for stabilizer backend).
    #[inline]
    pub fn is_clifford(&self) -> bool {
        matches!(
            self,
            Gate::Id
                | Gate::X
                | Gate::Y
                | Gate::Z
                | Gate::H
                | Gate::S
                | Gate::Sdg
                | Gate::SX
                | Gate::SXdg
                | Gate::Cx
                | Gate::Cz
                | Gate::Swap
        )
    }
}

/// Check if two 2x2 unitary matrices are equal up to a global phase factor.
fn matrices_equal_up_to_phase(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2], eps: f64) -> bool {
    // Find the first non-zero entry in b to determine the phase ratio
    let mut phase = None;
    for i in 0..2 {
        for j in 0..2 {
            if b[i][j].norm_sqr() > eps {
                if a[i][j].norm_sqr() < eps {
                    return false;
                }
                phase = Some(a[i][j] / b[i][j]);
                break;
            }
        }
        if phase.is_some() {
            break;
        }
    }

    let phase = match phase {
        Some(p) => p,
        None => return true, // Both are zero matrices
    };

    // Verify all entries match under the same phase
    for i in 0..2 {
        for j in 0..2 {
            let expected = phase * b[i][j];
            if (a[i][j] - expected).norm_sqr() > eps {
                return false;
            }
        }
    }
    true
}

fn format_angle(theta: f64) -> String {
    const FRACTIONS: &[(f64, &str)] = &[
        (1.0, "π"),
        (-1.0, "-π"),
        (0.5, "π/2"),
        (-0.5, "-π/2"),
        (0.25, "π/4"),
        (-0.25, "-π/4"),
        (1.0 / 3.0, "π/3"),
        (-1.0 / 3.0, "-π/3"),
        (2.0 / 3.0, "2π/3"),
        (-2.0 / 3.0, "-2π/3"),
        (1.0 / 6.0, "π/6"),
        (-1.0 / 6.0, "-π/6"),
        (5.0 / 6.0, "5π/6"),
        (-5.0 / 6.0, "-5π/6"),
        (1.0 / 8.0, "π/8"),
        (-1.0 / 8.0, "-π/8"),
        (3.0 / 8.0, "3π/8"),
        (-3.0 / 8.0, "-3π/8"),
        (1.5, "3π/2"),
        (-1.5, "-3π/2"),
        (2.0, "2π"),
        (-2.0, "-2π"),
    ];
    let ratio = theta / std::f64::consts::PI;
    for &(frac, label) in FRACTIONS {
        if (ratio - frac).abs() < 1e-10 {
            return label.to_string();
        }
    }
    format!("{:.4}", theta)
}

impl fmt::Display for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gate::Id => f.write_str("I"),
            Gate::X => f.write_str("X"),
            Gate::Y => f.write_str("Y"),
            Gate::Z => f.write_str("Z"),
            Gate::H => f.write_str("H"),
            Gate::S => f.write_str("S"),
            Gate::Sdg => f.write_str("Sdg"),
            Gate::T => f.write_str("T"),
            Gate::Tdg => f.write_str("Tdg"),
            Gate::SX => f.write_str("SX"),
            Gate::SXdg => f.write_str("SXdg"),
            Gate::Rx(t) => write!(f, "Rx({})", format_angle(*t)),
            Gate::Ry(t) => write!(f, "Ry({})", format_angle(*t)),
            Gate::Rz(t) => write!(f, "Rz({})", format_angle(*t)),
            Gate::P(t) => write!(f, "P({})", format_angle(*t)),
            Gate::Rzz(t) => write!(f, "Rzz({})", format_angle(*t)),
            Gate::Cx => f.write_str("CX"),
            Gate::Cz => f.write_str("CZ"),
            Gate::Swap => f.write_str("SWAP"),
            Gate::Cu(_) => f.write_str("CU"),
            Gate::Mcu(data) => write!(f, "MCU({}ctrl)", data.num_controls),
            Gate::Fused(_) => f.write_str("U"),
            Gate::Fused2q(_) => f.write_str("U2"),
            Gate::MultiFused(data) => write!(f, "MF[{}]", data.gates.len()),
            Gate::BatchPhase(data) => write!(f, "BP[{}]", data.phases.len()),
            Gate::BatchRzz(data) => write!(f, "BZZ[{}]", data.edges.len()),
            Gate::DiagonalBatch(data) => write!(f, "BD[{}]", data.entries.len()),
            Gate::Multi2q(data) => write!(f, "M2[{}]", data.gates.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_angle_pi_fractions() {
        assert_eq!(format_angle(std::f64::consts::PI), "π");
        assert_eq!(format_angle(std::f64::consts::FRAC_PI_2), "π/2");
        assert_eq!(format_angle(std::f64::consts::FRAC_PI_4), "π/4");
        assert_eq!(format_angle(-std::f64::consts::FRAC_PI_4), "-π/4");
        assert_eq!(format_angle(std::f64::consts::PI / 3.0), "π/3");
        assert_eq!(format_angle(0.123), "0.1230");
    }

    #[test]
    fn display_labels() {
        assert_eq!(Gate::H.to_string(), "H");
        assert_eq!(Gate::Cx.to_string(), "CX");
        assert_eq!(Gate::Rx(std::f64::consts::FRAC_PI_2).to_string(), "Rx(π/2)");
        assert_eq!(Gate::Rz(0.5).to_string(), "Rz(0.5000)");
        assert_eq!(Gate::Id.to_string(), "I");
        assert_eq!(Gate::Swap.to_string(), "SWAP");
    }

    #[test]
    fn test_gate_arity() {
        assert_eq!(Gate::H.num_qubits(), 1);
        assert_eq!(Gate::Rx(0.5).num_qubits(), 1);
        assert_eq!(Gate::Cx.num_qubits(), 2);
        assert_eq!(Gate::Swap.num_qubits(), 2);
    }

    #[test]
    fn test_h_matrix_is_unitary() {
        let m = Gate::H.matrix_2x2();
        // H * H = I
        let mut product = [[Complex64::new(0.0, 0.0); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for (k, row) in m.iter().enumerate() {
                    product[i][j] += m[i][k] * row[j];
                }
            }
        }
        let eps = 1e-12;
        assert!((product[0][0].re - 1.0).abs() < eps);
        assert!(product[0][0].im.abs() < eps);
        assert!(product[0][1].norm() < eps);
        assert!(product[1][0].norm() < eps);
        assert!((product[1][1].re - 1.0).abs() < eps);
    }

    #[test]
    fn test_rx_pi_equals_neg_i_x() {
        let rx = Gate::Rx(std::f64::consts::PI).matrix_2x2();
        // Rx(π) = -i·X  (up to global phase)
        // |Rx(π)[0][1]| should be 1
        assert!((rx[0][1].norm() - 1.0).abs() < 1e-12);
        assert!((rx[1][0].norm() - 1.0).abs() < 1e-12);
        assert!(rx[0][0].norm() < 1e-12);
        assert!(rx[1][1].norm() < 1e-12);
    }

    #[test]
    fn test_clifford_classification() {
        assert!(Gate::H.is_clifford());
        assert!(Gate::S.is_clifford());
        assert!(Gate::Cx.is_clifford());
        assert!(!Gate::T.is_clifford());
        assert!(!Gate::Rx(0.5).is_clifford());
        assert!(!Gate::Cu(Box::new([[Complex64::new(1.0, 0.0); 2]; 2])).is_clifford());
    }

    #[test]
    fn test_preserves_sparsity() {
        // Diagonal and permutation gates preserve sparsity
        assert!(Gate::Id.preserves_sparsity());
        assert!(Gate::X.preserves_sparsity());
        assert!(Gate::Y.preserves_sparsity());
        assert!(Gate::Z.preserves_sparsity());
        assert!(Gate::S.preserves_sparsity());
        assert!(Gate::T.preserves_sparsity());
        assert!(Gate::Rz(1.0).preserves_sparsity());
        assert!(Gate::P(0.5).preserves_sparsity());
        assert!(Gate::Cx.preserves_sparsity());
        assert!(Gate::Cz.preserves_sparsity());
        assert!(Gate::Swap.preserves_sparsity());

        // Superposition-creating gates do NOT preserve sparsity
        assert!(!Gate::H.preserves_sparsity());
        assert!(!Gate::Rx(0.5).preserves_sparsity());
        assert!(!Gate::Ry(0.5).preserves_sparsity());
        assert!(!Gate::SX.preserves_sparsity());
        assert!(!Gate::SXdg.preserves_sparsity());

        // Cu with diagonal matrix preserves sparsity
        let diag = Box::new([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
        ]);
        assert!(Gate::Cu(diag).preserves_sparsity());

        // Cu with H-like matrix does NOT preserve sparsity
        let h_mat = Box::new(Gate::H.matrix_2x2());
        assert!(!Gate::Cu(h_mat).preserves_sparsity());
    }

    #[test]
    fn test_cu_arity() {
        let mat = Gate::H.matrix_2x2();
        assert_eq!(Gate::Cu(Box::new(mat)).num_qubits(), 2);
    }

    fn assert_mat_close(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2], eps: f64) {
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a[i][j] - b[i][j]).norm() < eps,
                    "mat[{i}][{j}]: expected {:?}, got {:?}",
                    b[i][j],
                    a[i][j]
                );
            }
        }
    }

    #[test]
    fn test_inverse_self_inverse() {
        assert_eq!(Gate::H.inverse(), Gate::H);
        assert_eq!(Gate::X.inverse(), Gate::X);
        assert_eq!(Gate::Y.inverse(), Gate::Y);
        assert_eq!(Gate::Z.inverse(), Gate::Z);
        assert_eq!(Gate::Id.inverse(), Gate::Id);
        assert_eq!(Gate::Cx.inverse(), Gate::Cx);
        assert_eq!(Gate::Cz.inverse(), Gate::Cz);
        assert_eq!(Gate::Swap.inverse(), Gate::Swap);
    }

    #[test]
    fn test_inverse_adjoint_pairs() {
        assert_eq!(Gate::S.inverse(), Gate::Sdg);
        assert_eq!(Gate::Sdg.inverse(), Gate::S);
        assert_eq!(Gate::T.inverse(), Gate::Tdg);
        assert_eq!(Gate::Tdg.inverse(), Gate::T);
    }

    #[test]
    fn test_inverse_parametric() {
        assert_eq!(Gate::Rx(0.5).inverse(), Gate::Rx(-0.5));
        assert_eq!(Gate::Ry(1.0).inverse(), Gate::Ry(-1.0));
        assert_eq!(Gate::Rz(PI).inverse(), Gate::Rz(-PI));
    }

    #[test]
    fn test_inverse_fused_is_adjoint() {
        let s_mat = Gate::S.matrix_2x2();
        let fused = Gate::Fused(Box::new(s_mat));
        let inv = fused.inverse();
        if let Gate::Fused(inv_mat) = &inv {
            assert_mat_close(inv_mat, &Gate::Sdg.matrix_2x2(), 1e-12);
        } else {
            panic!("expected Fused");
        }
    }

    #[test]
    fn test_inverse_cu() {
        let rz_mat = Gate::Rz(0.5).matrix_2x2();
        let cu = Gate::Cu(Box::new(rz_mat));
        let inv = cu.inverse();
        if let Gate::Cu(inv_mat) = &inv {
            let expected = Gate::Rz(-0.5).matrix_2x2();
            assert_mat_close(inv_mat, &expected, 1e-12);
        } else {
            panic!("expected Cu");
        }
    }

    #[test]
    fn test_matrix_power_zero() {
        assert_eq!(Gate::X.matrix_power(0), Gate::Id);
        assert_eq!(Gate::Rz(0.5).matrix_power(0), Gate::Id);
    }

    #[test]
    fn test_matrix_power_one() {
        assert_eq!(Gate::X.matrix_power(1), Gate::X);
        assert_eq!(Gate::H.matrix_power(1), Gate::H);
    }

    #[test]
    fn test_matrix_power_x_squared() {
        let x2 = Gate::X.matrix_power(2);
        if let Gate::Fused(mat) = &x2 {
            assert_mat_close(mat, &Gate::Id.matrix_2x2(), 1e-12);
        } else {
            panic!("expected Fused");
        }
    }

    #[test]
    fn test_matrix_power_t_squared_is_s() {
        let t2 = Gate::T.matrix_power(2);
        if let Gate::Fused(mat) = &t2 {
            assert_mat_close(mat, &Gate::S.matrix_2x2(), 1e-12);
        } else {
            panic!("expected Fused");
        }
    }

    #[test]
    fn test_matrix_power_negative() {
        let t_inv2 = Gate::T.matrix_power(-2);
        if let Gate::Fused(mat) = &t_inv2 {
            assert_mat_close(mat, &Gate::Sdg.matrix_2x2(), 1e-12);
        } else {
            panic!("expected Fused");
        }
    }

    #[test]
    fn test_mcu_arity() {
        let mat = Gate::H.matrix_2x2();
        let mcu2 = Gate::Mcu(Box::new(McuData {
            mat,
            num_controls: 2,
        }));
        assert_eq!(mcu2.num_qubits(), 3);
        let mcu3 = Gate::Mcu(Box::new(McuData {
            mat,
            num_controls: 3,
        }));
        assert_eq!(mcu3.num_qubits(), 4);
    }

    #[test]
    fn test_mcu_not_clifford() {
        let mat = Gate::X.matrix_2x2();
        let mcu = Gate::Mcu(Box::new(McuData {
            mat,
            num_controls: 2,
        }));
        assert!(!mcu.is_clifford());
    }

    #[test]
    fn test_mcu_inverse() {
        let rz_mat = Gate::Rz(0.5).matrix_2x2();
        let mcu = Gate::Mcu(Box::new(McuData {
            mat: rz_mat,
            num_controls: 2,
        }));
        let inv = mcu.inverse();
        if let Gate::Mcu(inv_data) = &inv {
            let expected = Gate::Rz(-0.5).matrix_2x2();
            assert_mat_close(&inv_data.mat, &expected, 1e-12);
            assert_eq!(inv_data.num_controls, 2);
        } else {
            panic!("expected Mcu");
        }
    }

    #[test]
    fn test_mcu_name() {
        let mat = Gate::H.matrix_2x2();
        let mcu = Gate::Mcu(Box::new(McuData {
            mat,
            num_controls: 2,
        }));
        assert_eq!(mcu.name(), "mcu");
    }

    #[test]
    fn test_cphase_constructor() {
        let g = Gate::cphase(PI / 4.0);
        assert_eq!(g.num_qubits(), 2);
        assert_eq!(g.name(), "cu");
        if let Gate::Cu(mat) = &g {
            let one = Complex64::new(1.0, 0.0);
            assert!((mat[0][0] - one).norm() < 1e-14);
            assert!(mat[0][1].norm() < 1e-14);
            assert!(mat[1][0].norm() < 1e-14);
            let expected = Complex64::from_polar(1.0, PI / 4.0);
            assert!((mat[1][1] - expected).norm() < 1e-14);
        } else {
            panic!("expected Cu");
        }
    }

    #[test]
    fn test_controlled_phase_detection() {
        let cp = Gate::cphase(0.5);
        assert!(cp.controlled_phase().is_some());
        let phase = cp.controlled_phase().unwrap();
        let expected = Complex64::from_polar(1.0, 0.5);
        assert!((phase - expected).norm() < 1e-14);

        // Non-diagonal Cu should not be detected
        let h_mat = Gate::H.matrix_2x2();
        let cu_h = Gate::Cu(Box::new(h_mat));
        assert!(cu_h.controlled_phase().is_none());

        // CZ is Cu([[1,0],[0,-1]]) — should be detected (phase = -1)
        let z_mat = Gate::Z.matrix_2x2();
        let cu_z = Gate::Cu(Box::new(z_mat));
        assert!(cu_z.controlled_phase().is_some());
        let z_phase = cu_z.controlled_phase().unwrap();
        assert!((z_phase.re - (-1.0)).abs() < 1e-14);

        // Rz-based Cu is diagonal but mat[0][0] != 1 — should NOT be detected
        let rz_mat = Gate::Rz(0.5).matrix_2x2();
        let cu_rz = Gate::Cu(Box::new(rz_mat));
        assert!(cu_rz.controlled_phase().is_none());

        // Non-Cu gates should return None
        assert!(Gate::H.controlled_phase().is_none());
        assert!(Gate::Cx.controlled_phase().is_none());
    }

    #[test]
    fn test_controlled_phase_mcu() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let phase = Complex64::from_polar(1.0, 0.7);
        let mcu = Gate::Mcu(Box::new(McuData {
            mat: [[one, zero], [zero, phase]],
            num_controls: 2,
        }));
        assert!(mcu.controlled_phase().is_some());
        assert!((mcu.controlled_phase().unwrap() - phase).norm() < 1e-14);
    }

    #[test]
    fn test_sx_matrix_is_sqrt_x() {
        let sx = Gate::SX.matrix_2x2();
        let sx2 = mat_mul_2x2(&sx, &sx);
        assert_mat_close(&sx2, &Gate::X.matrix_2x2(), 1e-12);
    }

    #[test]
    fn test_sxdg_is_sx_inverse() {
        let sx = Gate::SX.matrix_2x2();
        let sxdg = Gate::SXdg.matrix_2x2();
        let product = mat_mul_2x2(&sx, &sxdg);
        assert_mat_close(&product, &Gate::Id.matrix_2x2(), 1e-12);
    }

    #[test]
    fn test_p_gate_matrix() {
        let p = Gate::P(PI / 4.0).matrix_2x2();
        let t = Gate::T.matrix_2x2();
        assert_mat_close(&p, &t, 1e-12);
    }

    #[test]
    fn test_sx_is_clifford() {
        assert!(Gate::SX.is_clifford());
        assert!(Gate::SXdg.is_clifford());
    }

    #[test]
    fn test_p_inverse() {
        assert_eq!(Gate::P(0.5).inverse(), Gate::P(-0.5));
    }

    #[test]
    fn test_sx_inverse_pair() {
        assert_eq!(Gate::SX.inverse(), Gate::SXdg);
        assert_eq!(Gate::SXdg.inverse(), Gate::SX);
    }

    #[test]
    fn test_is_diagonal_1q() {
        assert!(Gate::Id.is_diagonal_1q());
        assert!(Gate::Z.is_diagonal_1q());
        assert!(Gate::S.is_diagonal_1q());
        assert!(Gate::Sdg.is_diagonal_1q());
        assert!(Gate::T.is_diagonal_1q());
        assert!(Gate::Tdg.is_diagonal_1q());
        assert!(Gate::Rz(0.5).is_diagonal_1q());
        assert!(Gate::P(0.5).is_diagonal_1q());
        assert!(!Gate::H.is_diagonal_1q());
        assert!(!Gate::X.is_diagonal_1q());
        assert!(!Gate::Y.is_diagonal_1q());
        assert!(!Gate::Rx(0.5).is_diagonal_1q());
        assert!(!Gate::Ry(0.5).is_diagonal_1q());
        assert!(!Gate::SX.is_diagonal_1q());
        assert!(!Gate::Cx.is_diagonal_1q());

        let diag_fused = Gate::Fused(Box::new(Gate::T.matrix_2x2()));
        assert!(diag_fused.is_diagonal_1q());
        let nondiag_fused = Gate::Fused(Box::new(Gate::H.matrix_2x2()));
        assert!(!nondiag_fused.is_diagonal_1q());
    }

    #[test]
    fn test_is_self_inverse_2q() {
        assert!(Gate::Cx.is_self_inverse_2q());
        assert!(Gate::Cz.is_self_inverse_2q());
        assert!(Gate::Swap.is_self_inverse_2q());
        assert!(!Gate::H.is_self_inverse_2q());
        assert!(!Gate::T.is_self_inverse_2q());
        let mat = Gate::H.matrix_2x2();
        assert!(!Gate::Cu(Box::new(mat)).is_self_inverse_2q());
    }

    #[test]
    fn test_gate_enum_size() {
        assert_eq!(
            std::mem::size_of::<Gate>(),
            16,
            "Gate enum must stay at 16 bytes"
        );
    }

    #[test]
    fn test_recognize_named_gates() {
        for gate in &[
            Gate::H,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::S,
            Gate::Sdg,
            Gate::T,
            Gate::Tdg,
            Gate::SX,
            Gate::SXdg,
        ] {
            let mat = gate.matrix_2x2();
            let recognized = Gate::recognize_matrix(&mat);
            assert_eq!(
                recognized.as_ref(),
                Some(gate),
                "failed to recognize {:?}",
                gate.name()
            );
        }
    }

    #[test]
    fn test_recognize_identity() {
        let id = Gate::Id.matrix_2x2();
        assert_eq!(Gate::recognize_matrix(&id), Some(Gate::Id));
    }

    #[test]
    fn test_recognize_t_squared_is_s() {
        let t = Gate::T.matrix_2x2();
        let tt = mat_mul_2x2(&t, &t);
        assert_eq!(Gate::recognize_matrix(&tt), Some(Gate::S));
    }

    #[test]
    fn test_recognize_s_squared_is_z() {
        let s = Gate::S.matrix_2x2();
        let ss = mat_mul_2x2(&s, &s);
        assert_eq!(Gate::recognize_matrix(&ss), Some(Gate::Z));
    }

    #[test]
    fn test_recognize_h_squared_is_identity() {
        let h = Gate::H.matrix_2x2();
        let hh = mat_mul_2x2(&h, &h);
        assert_eq!(Gate::recognize_matrix(&hh), Some(Gate::Id));
    }

    #[test]
    fn test_recognize_t_fourth_is_z() {
        let t = Gate::T.matrix_2x2();
        let t2 = mat_mul_2x2(&t, &t);
        let t4 = mat_mul_2x2(&t2, &t2);
        assert_eq!(Gate::recognize_matrix(&t4), Some(Gate::Z));
    }

    #[test]
    fn test_recognize_non_clifford_returns_none() {
        let rx = Gate::Rx(0.7).matrix_2x2();
        assert_eq!(Gate::recognize_matrix(&rx), None);
        let ry = Gate::Ry(1.3).matrix_2x2();
        assert_eq!(Gate::recognize_matrix(&ry), None);
    }

    #[test]
    fn test_recognize_global_phase_invariance() {
        let phase = Complex64::from_polar(1.0, 0.42);
        let h = Gate::H.matrix_2x2();
        let phased = [
            [h[0][0] * phase, h[0][1] * phase],
            [h[1][0] * phase, h[1][1] * phase],
        ];
        assert_eq!(Gate::recognize_matrix(&phased), Some(Gate::H));
    }
}

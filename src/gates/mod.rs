//! Gate definitions and matrix representations.
//!
//! Gates are represented as an enum for fast dispatch without trait-object overhead
//! in the simulation hot path. Matrix representations use stack-allocated arrays
//! to avoid heap allocation during gate application.
//!
//! # Hot-path design notes
//! - `Gate` methods take `&self`, the enum is 16 bytes with large payloads boxed.
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
/// parameter inline. Variants with larger payloads (matrices, batch data) box them
/// to keep the enum at 16 bytes for cache-friendly instruction streams.
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
    Cu(Box<[[Complex64; 2]; 2]>),

    /// Multi-controlled unitary. Applies the 2×2 matrix to the target qubit
    /// only when all control qubits are |1⟩. Qubit order:
    /// `[ctrl_0, ctrl_1, ..., ctrl_{k-1}, target]`.
    Mcu(Box<McuData>),

    /// Pre-fused single-qubit unitary (product of consecutive gates on the same target).
    Fused(Box<[[Complex64; 2]; 2]>),

    /// Batched controlled-phase: multiple cphase gates sharing a control qubit,
    /// fused into a single pass over the statevector. Created by the cphase
    /// fusion pass. Targets: `[control]`. The `BatchPhaseData` holds per-target
    /// phases.
    BatchPhase(Box<BatchPhaseData>),

    /// Batched ZZ rotations: multiple Rzz gates fused into a single pass.
    /// Created by the batch-Rzz fusion pass. The `BatchRzzData` holds per-edge
    /// angles.
    BatchRzz(Box<BatchRzzData>),

    /// Batched diagonal gates: a contiguous run of diagonal 1q and 2q gates
    /// collapsed into a single state-vector sweep with a precomputed phase LUT.
    /// Subsumes BatchPhase and BatchRzz for mixed diagonal runs. Created by the
    /// diagonal batch fusion pass.
    DiagonalBatch(Box<DiagonalBatchData>),

    /// Multiple single-qubit gates on distinct qubits, batched for a single
    /// tiled pass over the statevector. Created by the multi-gate fusion pass.
    MultiFused(Box<MultiFusedData>),

    /// Pre-fused two-qubit unitary (4×4 matrix). Created by the 2q fusion pass
    /// which absorbs adjacent single-qubit gates into a two-qubit gate.
    Fused2q(Box<[[Complex64; 4]; 4]>),

    /// Multiple two-qubit gates batched for a single tiled pass over the
    /// statevector. Created by the multi-2q fusion pass. Each entry stores
    /// `(q0, q1, 4×4 matrix)`.
    Multi2q(Box<Multi2qData>),

    /// Quantum Fourier Transform on `start..start+num`.
    ///
    /// The CPU statevector backend has a fast whole-state FFT path. Subrange
    /// blocks and non-native backends expand to textbook H, cphase, and swap
    /// gates before execution.
    /// Boxless: `(u8, u8)` fits within the 16-byte enum slot.
    QftBlock { start: u8, num: u8 },
}

/// Analytic differentiation generator for a parametric gate.
///
/// Returned by [`Gate::pauli_generator`] and consumed by the adjoint gradient
/// engine. `Gate` stays 16 bytes: this is produced on demand, never stored in
/// the enum. Each rotation variant is `exp(-i θ/2 G)` with the named Pauli
/// generator `G` acting on the gate's `targets` (in order); `Phase` is the
/// non-Pauli phase gate `diag(1, e^{iθ})` whose generator is the projector
/// `|1⟩⟨1|` on `targets[0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorKind {
    /// `Rx(θ)`: generator `X` on `targets[0]`.
    RotX,
    /// `Ry(θ)`: generator `Y` on `targets[0]`.
    RotY,
    /// `Rz(θ)`: generator `Z` on `targets[0]`.
    RotZ,
    /// `Rzz(θ)`: generator `Z⊗Z` on `targets[0]` and `targets[1]`.
    RotZz,
    /// `P(θ)`: generator is the projector `|1⟩⟨1|` on `targets[0]`.
    Phase,
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

impl BatchPhaseData {
    /// Entry cap the fusion pass splits on, matching the kernel group tables.
    /// Targets are distinct within one payload, so a repeated `(control, target)`
    /// pair folds into a single entry rather than adding one.
    pub const MAX_PHASES: usize = 40;
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

impl BatchRzzData {
    /// Edge cap the fusion pass splits on, matching the kernel group tables.
    pub const MAX_EDGES: usize = 32;
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
    /// Return the qubit and dense 2×2 matrix for a [`DiagEntry::Phase1q`]
    /// entry, or `None` for two-qubit entries.
    pub fn as_1q_matrix(&self) -> Option<(usize, [[Complex64; 2]; 2])> {
        match *self {
            DiagEntry::Phase1q { qubit, d0, d1 } => {
                let z = Complex64::new(0.0, 0.0);
                Some((qubit, [[d0, z], [z, d1]]))
            }
            _ => None,
        }
    }

    /// Return the qubit pair and dense 4×4 matrix for a [`DiagEntry::Phase2q`]
    /// or [`DiagEntry::Parity2q`] entry, or `None` for single-qubit entries.
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

fn adjoint_4x4(m: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let mut result = [[Complex64::new(0.0, 0.0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = m[j][i].conj();
        }
    }
    result
}

fn adjoint_2x2(m: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [m[0][0].conj(), m[1][0].conj()],
        [m[0][1].conj(), m[1][1].conj()],
    ]
}

#[inline]
fn count_unique_qubits<I: IntoIterator<Item = usize>>(iter: I) -> usize {
    let mut seen: SmallVec<[usize; 8]> = SmallVec::new();
    for q in iter {
        if !seen.contains(&q) {
            seen.push(q);
        }
    }
    seen.len()
}

#[inline]
fn push_unique_qubit(seen: &mut SmallVec<[usize; 8]>, qubit: usize) {
    if !seen.contains(&qubit) {
        seen.push(qubit);
    }
}

/// Combined phase factor `entries` apply to the basis state `index`.
#[inline]
pub(crate) fn diag_entries_phase(index: usize, entries: &[DiagEntry]) -> Complex64 {
    let mut combined = Complex64::new(1.0, 0.0);
    for entry in entries {
        match entry {
            DiagEntry::Phase1q { qubit, d0, d1 } => {
                combined *= if (index >> qubit) & 1 == 1 { *d1 } else { *d0 };
            }
            DiagEntry::Phase2q { q0, q1, phase } => {
                if (index >> q0) & 1 == 1 && (index >> q1) & 1 == 1 {
                    combined *= phase;
                }
            }
            DiagEntry::Parity2q { q0, q1, same, diff } => {
                let parity = ((index >> q0) ^ (index >> q1)) & 1;
                combined *= if parity == 0 { *same } else { *diff };
            }
        }
    }
    combined
}

#[inline]
fn count_unique_diag_qubits(entries: &[DiagEntry]) -> usize {
    let mut seen: SmallVec<[usize; 8]> = SmallVec::new();
    for entry in entries {
        match entry {
            DiagEntry::Phase1q { qubit, .. } => push_unique_qubit(&mut seen, *qubit),
            DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } => {
                push_unique_qubit(&mut seen, *q0);
                push_unique_qubit(&mut seen, *q1);
            }
        }
    }
    seen.len()
}

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
            Gate::QftBlock { num, .. } => *num as usize,
            Gate::BatchRzz(data) => {
                count_unique_qubits(data.edges.iter().flat_map(|&(q0, q1, _)| [q0, q1]))
            }
            Gate::DiagonalBatch(data) => count_unique_diag_qubits(&data.entries),
            Gate::MultiFused(data) => data.gates.len(),
            Gate::Multi2q(data) => {
                count_unique_qubits(data.gates.iter().flat_map(|&(q0, q1, _)| [q0, q1]))
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
            | Gate::QftBlock { .. }
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
            Gate::QftBlock { .. } => "qft_block",
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
            Gate::QftBlock { .. } => {
                panic!(
                    "Gate::QftBlock has no in-place inverse. Run \
                     circuit::expand_qft_blocks before applying `inv @` or any \
                     transform that calls Gate::inverse()."
                )
            }
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

    /// Return the analytic differentiation generator for a parametric gate,
    /// or `None` if the gate has no defined generator (all non-parametric
    /// gates, and parametric gates whose angle is not stored inline as `f64`,
    /// e.g. controlled unitaries built from a boxed matrix). Used by the
    /// adjoint gradient engine to decide which instructions are differentiable.
    #[inline]
    pub fn pauli_generator(&self) -> Option<GeneratorKind> {
        match self {
            Gate::Rx(_) => Some(GeneratorKind::RotX),
            Gate::Ry(_) => Some(GeneratorKind::RotY),
            Gate::Rz(_) => Some(GeneratorKind::RotZ),
            Gate::Rzz(_) => Some(GeneratorKind::RotZz),
            Gate::P(_) => Some(GeneratorKind::Phase),
            _ => None,
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
            Gate::Fused(m) => is_diagonal_2x2(m),
            _ => false,
        }
    }

    /// True if this gate can be absorbed into a `DiagonalBatch`.
    #[inline]
    pub(crate) fn is_diag_batchable(&self) -> bool {
        match self {
            Gate::Cz | Gate::Rzz(_) => true,
            _ if self.is_diagonal_1q() => true,
            _ if self.controlled_phase().is_some() => true,
            _ => false,
        }
    }

    /// The `DiagEntry` values equivalent to this gate applied on `targets`.
    ///
    /// Only valid for gates where `is_diag_batchable()` returns true.
    pub(crate) fn diag_entries(&self, targets: &[usize]) -> SmallVec<[DiagEntry; 2]> {
        match self {
            Gate::Cz => {
                smallvec::smallvec![DiagEntry::Phase2q {
                    q0: targets[0],
                    q1: targets[1],
                    phase: Complex64::new(-1.0, 0.0),
                }]
            }
            Gate::Rzz(theta) => {
                let half = theta / 2.0;
                let same = Complex64::new((-half).cos(), (-half).sin()); // e^{-iθ/2}
                let diff = Complex64::new(half.cos(), half.sin()); // e^{iθ/2}
                smallvec::smallvec![DiagEntry::Parity2q {
                    q0: targets[0],
                    q1: targets[1],
                    same,
                    diff,
                }]
            }
            _ if self.controlled_phase().is_some() => {
                let phase = self.controlled_phase().unwrap();
                smallvec::smallvec![DiagEntry::Phase2q {
                    q0: targets[0],
                    q1: targets[1],
                    phase,
                }]
            }
            _ => {
                let mat = self.matrix_2x2();
                smallvec::smallvec![DiagEntry::Phase1q {
                    qubit: targets[0],
                    d0: mat[0][0],
                    d1: mat[1][1],
                }]
            }
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

/// Whether a 2x2 matrix is diagonal (both off-diagonal norms below `IDENTITY_EPS`).
#[inline]
pub(crate) fn is_diagonal_2x2(mat: &[[Complex64; 2]; 2]) -> bool {
    mat[0][1].norm() < IDENTITY_EPS && mat[1][0].norm() < IDENTITY_EPS
}

/// Whether a 4x4 matrix is diagonal (all off-diagonal norms below `IDENTITY_EPS`).
#[inline]
pub(crate) fn is_diagonal_4x4(mat: &[[Complex64; 4]; 4]) -> bool {
    for (r, row) in mat.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            if r != c && value.norm() >= IDENTITY_EPS {
                return false;
            }
        }
    }
    true
}

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
            Gate::QftBlock { start, num } => write!(f, "QFT[{}..{}]", start, start + num),
            Gate::BatchRzz(data) => write!(f, "BZZ[{}]", data.edges.len()),
            Gate::DiagonalBatch(data) => write!(f, "BD[{}]", data.entries.len()),
            Gate::Multi2q(data) => write!(f, "M2[{}]", data.gates.len()),
        }
    }
}

#[cfg(test)]
mod tests;

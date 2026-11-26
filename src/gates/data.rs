//! Payload types the boxed [`Gate`](super::Gate) variants carry.

use super::{is_diagonal_2x2, mat_mul_2x2};
use crate::sim::unified_pauli::PauliAxis;
use num_complex::Complex64;
use smallvec::SmallVec;

/// Data for a dense multi-qubit unitary.
///
/// Fields are private: the kernels read `num_qubits` and the matrix length as
/// one fact, and a disagreement between them is an out-of-bounds read rather
/// than a wrong answer. [`Gate::unitary`](super::Gate::unitary) is the only constructor.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitaryData {
    pub(super) mat: Vec<Complex64>,
    pub(super) num_qubits: u8,
}

impl UnitaryData {
    /// Row-major `2^k x 2^k` entries, `targets[0]` the most significant index
    /// bit.
    pub fn matrix(&self) -> &[Complex64] {
        &self.mat
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits as usize
    }

    pub(super) fn adjoint(&self) -> UnitaryData {
        let dim = 1usize << self.num_qubits;
        let mut mat = vec![Complex64::new(0.0, 0.0); dim * dim];
        for row in 0..dim {
            for column in 0..dim {
                mat[row * dim + column] = self.mat[column * dim + row].conj();
            }
        }
        UnitaryData {
            mat,
            num_qubits: self.num_qubits,
        }
    }

    pub(crate) fn conjugated(&self) -> UnitaryData {
        UnitaryData {
            mat: self.mat.iter().map(Complex64::conj).collect(),
            num_qubits: self.num_qubits,
        }
    }
}

/// Analytic differentiation generator for a parametric gate, returned by
/// [`Gate::pauli_generator`](super::Gate::pauli_generator).
///
/// Each rotation variant is `exp(-i θ/2 G)` with the named Pauli
/// generator `G` acting on the gate's `targets` (in order); `Phase` is the
/// non-Pauli phase gate `diag(1, e^{iθ})` whose generator is the projector
/// `|1⟩⟨1|` on `targets[0]`. The borrow is of the gate the generator was read
/// from, so a caller holding one cannot pair it with another gate's letters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorKind<'a> {
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
    /// `PauliRot(θ)`: generator is the Pauli string whose letter at index `i`
    /// acts on `targets[i]`.
    RotPauli(&'a [PauliAxis]),
}

/// Data for a multi-qubit Pauli rotation `exp(-i θ P / 2)`.
///
/// `axes[i]` is the letter of `P` on the owning instruction's `targets[i]`.
/// Fields are private so every construction path runs the recognizing
/// lowering in [`crate::Circuit::add_pauli_rotation`].
#[derive(Debug, Clone, PartialEq)]
pub struct PauliRotData {
    pub(crate) theta: f64,
    pub(crate) axes: Vec<PauliAxis>,
}

impl PauliRotData {
    /// Rotation angle in radians.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Overwrite the rotation angle, leaving the Pauli string alone. The
    /// recognizing lowering runs on the string, so rebinding an angle cannot
    /// invalidate it.
    pub fn set_theta(&mut self, theta: f64) {
        self.theta = theta;
    }

    /// Pauli letters aligned with the instruction targets.
    pub fn axes(&self) -> &[PauliAxis] {
        &self.axes
    }
}

/// Pack the Pauli string of a rotation into `(xmask, zmask, num_y)`, where
/// `xmask` covers X and Y letters and `zmask` covers Z and Y, and `axes[i]`
/// acts on `targets[i]`.
///
/// The native kernel and the adjoint gradient both derive their masks here, so
/// a change to the letter convention cannot move one without the other.
#[inline]
pub(crate) fn pauli_rot_masks(targets: &[usize], axes: &[PauliAxis]) -> (usize, usize, u32) {
    let mut xmask = 0usize;
    let mut zmask = 0usize;
    let mut num_y = 0u32;
    for (&q, axis) in targets.iter().zip(axes) {
        let bit = 1usize << q;
        match axis {
            PauliAxis::X => xmask |= bit,
            PauliAxis::Z => zmask |= bit,
            PauliAxis::Y => {
                xmask |= bit;
                zmask |= bit;
                num_y += 1;
            }
        }
    }
    (xmask, zmask, num_y)
}

/// Data for a `MultiPauliRot` batch: `(xmask, zmask, theta)` per rotation in circuit
/// order. Bit `q` of `xmask` marks an X or Y letter on qubit `q` and bit `q` of `zmask`
/// a Z or Y, so the Y count is `(xmask & zmask).count_ones()`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiPauliRotData {
    pub(crate) rotations: Vec<(usize, usize, f64)>,
}

impl MultiPauliRotData {
    pub fn rotations(&self) -> &[(usize, usize, f64)] {
        &self.rotations
    }

    /// Every qubit any rotation touches, as a bit mask.
    pub(crate) fn support(&self) -> usize {
        self.rotations.iter().fold(0, |acc, &(x, z, _)| acc | x | z)
    }

    /// Call `f` with each rotation as an angle, ascending targets, and their letters.
    #[cfg(feature = "gpu")]
    pub(crate) fn for_each_rotation(&self, mut f: impl FnMut(f64, &[usize], &[PauliAxis])) {
        let mut targets: SmallVec<[usize; 8]> = SmallVec::new();
        let mut axes: SmallVec<[PauliAxis; 8]> = SmallVec::new();
        for &(xmask, zmask, theta) in &self.rotations {
            targets.clear();
            axes.clear();
            let mut rest = xmask | zmask;
            while rest != 0 {
                let q = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                targets.push(q);
                axes.push(match (xmask >> q & 1, zmask >> q & 1) {
                    (1, 1) => PauliAxis::Y,
                    (1, _) => PauliAxis::X,
                    _ => PauliAxis::Z,
                });
            }
            f(theta, &targets, &axes);
        }
    }
}

/// Data for a multi-controlled unitary gate.
#[derive(Debug, Clone, PartialEq)]
pub struct McuData {
    pub(crate) mat: [[Complex64; 2]; 2],
    pub(crate) num_controls: u8,
}

impl McuData {
    /// `mat` applied to the target when all `num_controls` controls are set.
    ///
    /// # Panics
    ///
    /// Panics if `num_controls` is below 2; [`Gate::mcu`](super::Gate::mcu) takes one control and
    /// returns the controlled 2x2 form instead.
    pub fn new(mat: [[Complex64; 2]; 2], num_controls: u8) -> Self {
        assert!(num_controls >= 2, "McuData needs at least 2 controls");
        Self { mat, num_controls }
    }

    /// 2x2 unitary applied to the target qubit.
    pub fn mat(&self) -> &[[Complex64; 2]; 2] {
        &self.mat
    }

    pub fn num_controls(&self) -> u8 {
        self.num_controls
    }
}

/// Data for a batched controlled-phase gate: `(target_qubit, phase)` entries, with
/// the shared control in the instruction's `targets[0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchPhaseData {
    pub(crate) phases: SmallVec<[(usize, Complex64); 8]>,
}

impl BatchPhaseData {
    /// # Panics
    ///
    /// Panics if there are more than [`Self::MAX_PHASES`] entries.
    pub fn new(phases: impl IntoIterator<Item = (usize, Complex64)>) -> Self {
        let phases: SmallVec<[(usize, Complex64); 8]> = phases.into_iter().collect();
        assert!(
            phases.len() <= Self::MAX_PHASES,
            "BatchPhaseData holds at most {} entries",
            Self::MAX_PHASES
        );
        Self { phases }
    }

    pub fn phases(&self) -> &[(usize, Complex64)] {
        &self.phases
    }

    /// Entry cap the fusion pass splits on, matching the kernel group tables.
    /// Targets are distinct within one payload, so a repeated `(control, target)`
    /// pair folds into a single entry rather than adding one.
    pub const MAX_PHASES: usize = 40;
}

/// Data for batched ZZ rotations: `(qubit_0, qubit_1, theta)` entries, with every
/// qubit also in the instruction's `targets`.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchRzzData {
    pub(crate) edges: Vec<(usize, usize, f64)>,
}

impl BatchRzzData {
    /// # Panics
    ///
    /// Panics if there are more than [`Self::MAX_EDGES`] edges.
    pub fn new(edges: Vec<(usize, usize, f64)>) -> Self {
        assert!(
            edges.len() <= Self::MAX_EDGES,
            "BatchRzzData holds at most {} edges",
            Self::MAX_EDGES
        );
        Self { edges }
    }

    pub fn edges(&self) -> &[(usize, usize, f64)] {
        &self.edges
    }

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
    pub(crate) fn as_1q_matrix(&self) -> Option<(usize, [[Complex64; 2]; 2])> {
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
    pub(crate) fn as_2q_matrix(&self) -> Option<(usize, usize, [[Complex64; 4]; 4])> {
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

/// Data for a batched diagonal gate; the kernel builds a phase LUT indexed by the
/// bits of the distinct qubits in `entries`.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagonalBatchData {
    pub(crate) entries: Vec<DiagEntry>,
}

impl DiagonalBatchData {
    pub fn new(entries: Vec<DiagEntry>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &[DiagEntry] {
        &self.entries
    }

    /// Widest qubit group one kernel lookup table covers. Qubits joined by a two-qubit
    /// entry share a group, so the fusion pass closes a batch before a component outgrows it.
    pub(crate) const MAX_QUBITS_PER_GROUP: usize = 10;

    /// Most lookup tables one kernel sweep combines.
    pub(crate) const MAX_GROUPS: usize = 4;
}

/// Data for a `MultiFused` batch of `(target_qubit, 2×2 matrix)` entries.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiFusedData {
    pub(crate) gates: Vec<(usize, [[Complex64; 2]; 2])>,
    pub(crate) all_diagonal: bool,
}

impl MultiFusedData {
    /// Batch `(target_qubit, 2×2 matrix)` entries, computing the diagonal flag the
    /// kernels branch on. Entries on one target fold into one matrix in list order, so
    /// the batch holds each target once and its entries commute.
    pub fn new(gates: Vec<(usize, [[Complex64; 2]; 2])>) -> Self {
        Self::fold_with_sources::<()>(gates, &[]).0
    }

    /// Fold `(target_qubit, 2×2 matrix)` entries by first-seen target qubit, grouping
    /// matching `sources` in the same order when `sources` is non-empty.
    pub(crate) fn fold_with_sources<S: Copy>(
        gates: Vec<(usize, [[Complex64; 2]; 2])>,
        sources: &[S],
    ) -> (Self, Vec<Vec<S>>) {
        let mut folded: Vec<(usize, [[Complex64; 2]; 2])> = Vec::with_capacity(gates.len());
        let mut grouped: Vec<Vec<S>> = if sources.is_empty() {
            Vec::new()
        } else {
            assert_eq!(
                gates.len(),
                sources.len(),
                "gates and sources length mismatch"
            );
            Vec::with_capacity(gates.len())
        };
        for (idx, (target, mat)) in gates.into_iter().enumerate() {
            match folded.iter().position(|(q, _)| *q == target) {
                Some(pos) => {
                    folded[pos].1 = mat_mul_2x2(&mat, &folded[pos].1);
                    if !sources.is_empty() {
                        grouped[pos].push(sources[idx]);
                    }
                }
                None => {
                    folded.push((target, mat));
                    if !sources.is_empty() {
                        grouped.push(vec![sources[idx]]);
                    }
                }
            }
        }
        let all_diagonal = folded.iter().all(|(_, m)| is_diagonal_2x2(m));
        (
            Self {
                gates: folded,
                all_diagonal,
            },
            grouped,
        )
    }

    pub fn gates(&self) -> &[(usize, [[Complex64; 2]; 2])] {
        &self.gates
    }

    /// Whether every matrix is diagonal, so the batch applies as a phase pass.
    pub fn all_diagonal(&self) -> bool {
        self.all_diagonal
    }
}

/// Data for a `Multi2q` batch of `(q0, q1, 4×4 matrix)` entries in circuit order.
#[derive(Debug, Clone, PartialEq)]
pub struct Multi2qData {
    pub(crate) gates: Vec<(usize, usize, [[Complex64; 4]; 4])>,
}

impl Multi2qData {
    pub fn new(gates: Vec<(usize, usize, [[Complex64; 4]; 4])>) -> Self {
        Self { gates }
    }

    pub fn gates(&self) -> &[(usize, usize, [[Complex64; 4]; 4])] {
        &self.gates
    }
}

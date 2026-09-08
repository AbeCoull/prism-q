//! Lowering of the gate forms the parser and fusion emit (`Fused` matrices,
//! `Cu`, and `Rz` spellings of `T`) into the Clifford, Z-rotation, and `Rzz`
//! forms the Clifford+T engines consume. Runs once per circuit, ahead of the run.

use num_complex::Complex64;
use std::f64::consts::FRAC_PI_4;

use crate::circuit::{SmallVec, pauli_rotation_lowering};
use crate::gates::{Gate, is_diagonal_2x2};
use crate::sim::unified_pauli::PauliAxis;

/// Tolerance, in units of pi/4, for an angle to count as on the grid.
const GRID_EPS: f64 = 1e-9;

/// Norm below which a matrix entry counts as zero in the Euler decomposition.
const ZERO_EPS: f64 = 1e-12;

/// Number of eighth turns in `theta`, modulo 8, when `theta` is within
/// `GRID_EPS` of a multiple of pi/4.
pub(crate) fn eighth_turns(theta: f64) -> Option<u8> {
    let k = theta / FRAC_PI_4;
    let rounded = k.round();
    ((k - rounded).abs() <= GRID_EPS).then(|| rounded.rem_euclid(8.0) as u8)
}

/// Gates the Pauli engines branch on or conjugate through without lowering.
pub(crate) fn is_pauli_native(gate: &Gate) -> bool {
    gate.is_clifford()
        || matches!(
            gate,
            Gate::T | Gate::Tdg | Gate::Rz(_) | Gate::P(_) | Gate::Rzz(_)
        )
}

/// Rewrite `gate` into Clifford gates, Z rotations, and `Rzz`, emitting each
/// piece through `emit`. Native forms pass through unchanged; `Rx`, `Ry`, and
/// `PauliRot` take the CNOT-ladder lowering; a `Fused` matrix becomes the named
/// gate or `Rz` it equals up to phase, else its ZYZ Euler triple; a `Cu` lowers
/// when its target is diagonal or a Pauli up to phase. The error text names the
/// rejected form and is the caller's `BackendUnsupported` operation.
pub(crate) fn lower_to_pauli_forms(
    gate: &Gate,
    targets: &[usize],
    emit: &mut impl FnMut(Gate, &[usize]),
) -> Result<(), String> {
    match gate {
        native if is_pauli_native(native) => emit(native.clone(), targets),
        Gate::Rx(theta) => pauli_rotation_lowering(*theta, targets, &[PauliAxis::X], emit),
        Gate::Ry(theta) => pauli_rotation_lowering(*theta, targets, &[PauliAxis::Y], emit),
        Gate::PauliRot(data) => pauli_rotation_lowering(data.theta(), targets, data.axes(), emit),
        Gate::Fused(mat) => lower_1q_matrix(mat, targets[0], emit),
        Gate::Cu(mat) => lower_controlled(mat, targets[0], targets[1], emit)?,
        other => {
            return Err(format!(
                "gate `{}` is neither Clifford nor a supported Pauli rotation",
                other.name()
            ));
        }
    }
    Ok(())
}

/// Rewrite `gate` into Clifford gates plus `T` and `Tdg`, emitting through
/// `emit`, and return the T count. Every Z rotation must land on the pi/4 grid;
/// the first one that does not is the rejection, named after `gate`.
pub(crate) fn lower_to_clifford_t(
    gate: &Gate,
    targets: &[usize],
    emit: &mut impl FnMut(Gate, &[usize]),
) -> Result<usize, String> {
    let mut t_count = 0usize;
    let mut off_grid = None;
    lower_to_pauli_forms(gate, targets, &mut |lowered, tgts| match lowered {
        Gate::T | Gate::Tdg => {
            t_count += 1;
            emit(lowered, tgts);
        }
        Gate::Rz(theta) | Gate::P(theta) => match eighth_turns(theta) {
            Some(k) => {
                t_count += usize::from(k & 1);
                emit_eighth_turns(k, tgts[0], emit);
            }
            None => off_grid = Some(theta),
        },
        Gate::Rzz(theta) => match eighth_turns(theta) {
            Some(k) => {
                t_count += usize::from(k & 1);
                emit_zz_eighth_turns(k, tgts[0], tgts[1], emit);
            }
            None => off_grid = Some(theta),
        },
        clifford => emit(clifford, tgts),
    })?;
    match off_grid {
        Some(theta) => Err(format!(
            "gate `{}` lowers to a Z rotation by {theta} rad, off the pi/4 grid of Clifford+T",
            gate.name()
        )),
        None => Ok(t_count),
    }
}

/// T count of `gate` inside the Clifford+T set: 0 for a Clifford, one per
/// rotation at an odd multiple of pi/4 after lowering, `None` outside the set.
pub(crate) fn clifford_t_count(gate: &Gate) -> Option<usize> {
    if gate.is_clifford() {
        return Some(0);
    }
    if matches!(gate, Gate::T | Gate::Tdg) {
        return Some(1);
    }
    let targets: SmallVec<[usize; 4]> = (0..gate.num_qubits()).collect();
    lower_to_clifford_t(gate, &targets, &mut |_, _| {}).ok()
}

fn lower_1q_matrix(mat: &[[Complex64; 2]; 2], qubit: usize, emit: &mut impl FnMut(Gate, &[usize])) {
    match Gate::recognize_matrix_up_to_phase(mat) {
        Some((Gate::Rz(theta), _)) => emit_z_rotation(theta, qubit, emit),
        Some((named, _)) => emit(named, &[qubit]),
        None => {
            let (beta, gamma, delta) = zyz_angles(mat);
            emit_z_rotation(delta, qubit, emit);
            pauli_rotation_lowering(gamma, &[qubit], &[PauliAxis::Y], &mut *emit);
            emit_z_rotation(beta, qubit, emit);
        }
    }
}

/// Euler angles `(beta, gamma, delta)` with `mat = c · Rz(beta) Ry(gamma) Rz(delta)`
/// for some unit scalar `c`. Dividing out the determinant phase leaves an SU(2)
/// matrix `[[a, -conj(b)], [b, conj(a)]]` with `a = cos(gamma/2) e^{-i(beta+delta)/2}`
/// and `b = sin(gamma/2) e^{i(beta-delta)/2}`, so the half-angles read off `a`
/// and `b` directly; the two square roots of the determinant differ by a sign
/// on both, which shifts `delta` by a full turn and nothing else. A vanishing
/// `a` or `b` leaves one angle free, and it is set to zero.
fn zyz_angles(mat: &[[Complex64; 2]; 2]) -> (f64, f64, f64) {
    let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    let unphase = Complex64::from_polar(1.0, -det.arg() / 2.0);
    let a = mat[0][0] * unphase;
    let b = mat[1][0] * unphase;
    let gamma = 2.0 * b.norm().atan2(a.norm());
    if a.norm() < ZERO_EPS {
        return (2.0 * b.arg(), gamma, 0.0);
    }
    if b.norm() < ZERO_EPS {
        return (0.0, gamma, -2.0 * a.arg());
    }
    (b.arg() - a.arg(), gamma, -b.arg() - a.arg())
}

/// A Pauli target `c · X`, `c · Y`, or `c · Z` is `Cx`, `Sdg · Cx · S` on the
/// target, or `Cz`, followed by `P(arg c)` on the control. A diagonal target
/// `diag(a, b)` is `P(arg a)` on the control times `CPhase(arg b - arg a)`, and
/// `CPhase(phi)` is `P(phi/2)` on both qubits with `Rzz(-phi/2)`, up to phase.
fn lower_controlled(
    mat: &[[Complex64; 2]; 2],
    control: usize,
    target: usize,
    emit: &mut impl FnMut(Gate, &[usize]),
) -> Result<(), String> {
    let pauli = Gate::recognize_matrix_up_to_phase(mat)
        .filter(|(named, _)| matches!(named, Gate::X | Gate::Y | Gate::Z));
    if let Some((named, phase)) = pauli {
        match named {
            Gate::X => emit(Gate::Cx, &[control, target]),
            Gate::Y => {
                emit(Gate::Sdg, &[target]);
                emit(Gate::Cx, &[control, target]);
                emit(Gate::S, &[target]);
            }
            _ => emit(Gate::Cz, &[control, target]),
        }
        emit_z_rotation(phase.arg(), control, emit);
        return Ok(());
    }
    if !is_diagonal_2x2(mat) {
        return Err(
            "gate `cu` has a target that is neither diagonal nor a Pauli up to phase".to_string(),
        );
    }
    let (phi_a, phi_b) = (mat[0][0].arg(), mat[1][1].arg());
    let half = (phi_b - phi_a) / 2.0;
    emit_z_rotation(phi_a + half, control, emit);
    emit_z_rotation(half, target, emit);
    emit_zz_rotation(-half, control, target, emit);
    Ok(())
}

/// Emit `P(theta)` on `qubit`, equal to `Rz(theta)` up to phase: the named
/// gates when `theta` is on the pi/4 grid, so every engine takes them through
/// its Clifford or T path, and `Rz(theta)` otherwise.
fn emit_z_rotation(theta: f64, qubit: usize, emit: &mut impl FnMut(Gate, &[usize])) {
    match eighth_turns(theta) {
        Some(k) => emit_eighth_turns(k, qubit, emit),
        None => emit(Gate::Rz(theta), &[qubit]),
    }
}

fn emit_zz_rotation(theta: f64, a: usize, b: usize, emit: &mut impl FnMut(Gate, &[usize])) {
    match eighth_turns(theta) {
        Some(k) => emit_zz_eighth_turns(k, a, b, emit),
        None => emit(Gate::Rzz(theta), &[a, b]),
    }
}

/// `P(k · pi/4)` as named gates, `k` modulo 8.
fn emit_eighth_turns(k: u8, qubit: usize, emit: &mut impl FnMut(Gate, &[usize])) {
    let q = &[qubit];
    match k {
        0 => {}
        1 => emit(Gate::T, q),
        2 => emit(Gate::S, q),
        3 => {
            emit(Gate::S, q);
            emit(Gate::T, q);
        }
        4 => emit(Gate::Z, q),
        5 => {
            emit(Gate::Z, q);
            emit(Gate::T, q);
        }
        6 => emit(Gate::Sdg, q),
        _ => emit(Gate::Tdg, q),
    }
}

/// `Rzz(k · pi/4)` up to phase, `k` modulo 8: `Z ⊗ Z` at a half turn,
/// `(S ⊗ S) · Cz` and its adjoint at the quarter turns, and `Cx · P_b · Cx`
/// at the odd multiples.
fn emit_zz_eighth_turns(k: u8, a: usize, b: usize, emit: &mut impl FnMut(Gate, &[usize])) {
    match k {
        0 => {}
        2 | 6 => {
            let s = if k == 2 { Gate::S } else { Gate::Sdg };
            emit(s.clone(), &[a]);
            emit(s, &[b]);
            emit(Gate::Cz, &[a, b]);
        }
        4 => {
            emit(Gate::Z, &[a]);
            emit(Gate::Z, &[b]);
        }
        odd => {
            emit(Gate::Cx, &[a, b]);
            emit_eighth_turns(odd, b, emit);
            emit(Gate::Cx, &[a, b]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{cu_matrix_4x4, mat_mul_2x2};
    use std::f64::consts::{FRAC_PI_2, PI};

    fn u3(theta: f64, phi: f64, lam: f64) -> Gate {
        Gate::Fused(Box::new(crate::circuit::openqasm::Parser::u_matrix(
            theta, phi, lam,
        )))
    }

    fn product_of_lowering(gate: &Gate) -> [[Complex64; 2]; 2] {
        let mut acc = Gate::Id.matrix_2x2();
        lower_to_pauli_forms(gate, &[0], &mut |g, _| {
            acc = mat_mul_2x2(&g.matrix_2x2(), &acc)
        })
        .unwrap();
        acc
    }

    fn equal_up_to_phase<const N: usize>(a: &[[Complex64; N]; N], b: &[[Complex64; N]; N]) -> bool {
        let (r, c) = (0..N)
            .flat_map(|r| (0..N).map(move |c| (r, c)))
            .find(|&(r, c)| b[r][c].norm() > 0.5)
            .unwrap();
        let phase = a[r][c] / b[r][c];
        (0..N).all(|i| (0..N).all(|j| (a[i][j] - phase * b[i][j]).norm() < 1e-9))
    }

    #[test]
    fn eighth_turns_snaps_the_grid_and_rejects_the_rest() {
        assert_eq!(eighth_turns(0.0), Some(0));
        assert_eq!(eighth_turns(FRAC_PI_4), Some(1));
        assert_eq!(eighth_turns(-FRAC_PI_4), Some(7));
        assert_eq!(eighth_turns(3.0 * FRAC_PI_4), Some(3));
        assert_eq!(eighth_turns(2.0 * PI + FRAC_PI_2), Some(2));
        assert_eq!(eighth_turns(0.3), None);
    }

    #[test]
    fn general_u3_lowers_to_its_own_matrix_up_to_phase() {
        for (theta, phi, lam) in [
            (0.3, 0.7, -1.1),
            (0.9, -FRAC_PI_2, FRAC_PI_2),
            (0.5, 3.0, 3.0),
            (1.4, -3.0, 2.9),
            (PI, 0.4, 0.9),
            (1e-14, 0.5, 0.25),
            (2.2, 0.0, 0.0),
        ] {
            let gate = u3(theta, phi, lam);
            let rebuilt = product_of_lowering(&gate);
            assert!(
                equal_up_to_phase(&rebuilt, &gate.matrix_2x2()),
                "u3({theta}, {phi}, {lam})"
            );
        }
    }

    #[test]
    fn gate_forms_on_the_grid_count_as_clifford_plus_t() {
        assert_eq!(
            clifford_t_count(&u3(FRAC_PI_4, -FRAC_PI_2, FRAC_PI_2)),
            Some(1)
        );
        assert_eq!(clifford_t_count(&u3(FRAC_PI_2, 0.0, PI)), Some(0));
        assert_eq!(clifford_t_count(&u3(0.3, 0.7, -1.1)), None);
        assert_eq!(clifford_t_count(&Gate::Rz(FRAC_PI_4)), Some(1));
        assert_eq!(clifford_t_count(&Gate::Rz(3.0 * FRAC_PI_4)), Some(1));
        assert_eq!(clifford_t_count(&Gate::P(FRAC_PI_2)), Some(0));
        assert_eq!(clifford_t_count(&Gate::Rz(0.3)), None);
        assert_eq!(clifford_t_count(&Gate::Rzz(FRAC_PI_4)), Some(1));
        assert_eq!(clifford_t_count(&Gate::cu(Gate::X.matrix_2x2())), Some(0));
        assert_eq!(clifford_t_count(&Gate::cu(Gate::S.matrix_2x2())), Some(3));
        assert_eq!(clifford_t_count(&Gate::cu(Gate::H.matrix_2x2())), None);
        assert_eq!(clifford_t_count(&Gate::mcu(Gate::X.matrix_2x2(), 2)), None);
    }

    #[test]
    fn controlled_lowering_rebuilds_the_4x4() {
        let phase = Complex64::from_polar(1.0, 0.42);
        let scaled = |g: Gate| {
            let m = g.matrix_2x2();
            [
                [m[0][0] * phase, m[0][1] * phase],
                [m[1][0] * phase, m[1][1] * phase],
            ]
        };
        let zero = Complex64::new(0.0, 0.0);
        let diag = [
            [Complex64::from_polar(1.0, 0.3), zero],
            [zero, Complex64::from_polar(1.0, -1.7)],
        ];
        for target in [
            scaled(Gate::X),
            scaled(Gate::Y),
            scaled(Gate::Z),
            diag,
            Gate::T.matrix_2x2(),
        ] {
            let mut acc = identity_4x4();
            lower_to_pauli_forms(&Gate::cu(target), &[0, 1], &mut |g, tgts| {
                acc = mat_mul_4x4(&embed_4x4(&g, tgts), &acc);
            })
            .unwrap();
            assert!(
                equal_up_to_phase(&acc, &cu_matrix_4x4(&target)),
                "target {target:?}"
            );
        }
    }

    #[test]
    fn rejections_name_the_gate_form() {
        let err = lower_to_pauli_forms(&Gate::cu(Gate::H.matrix_2x2()), &[0, 1], &mut |_, _| {})
            .unwrap_err();
        assert!(err.contains("cu"), "{err}");
        let err = lower_to_clifford_t(&Gate::Rz(0.3), &[0], &mut |_, _| {}).unwrap_err();
        assert!(err.contains("rz") && err.contains("pi/4"), "{err}");
        let err = lower_to_clifford_t(&u3(0.3, 0.7, -1.1), &[0], &mut |_, _| {}).unwrap_err();
        assert!(err.contains("fused"), "{err}");
    }

    fn identity_4x4() -> [[Complex64; 4]; 4] {
        let mut m = [[Complex64::new(0.0, 0.0); 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = Complex64::new(1.0, 0.0);
        }
        m
    }

    fn mat_mul_4x4(a: &[[Complex64; 4]; 4], b: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
        let mut out = [[Complex64::new(0.0, 0.0); 4]; 4];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..4).map(|k| a[i][k] * b[k][j]).sum();
            }
        }
        out
    }

    // `matrix_4x4` puts targets[0] on the high bit of the basis index, so a
    // single-qubit gate on qubit `q` of the pair `[0, 1]` sits on bit `1 - q`.
    fn embed_4x4(gate: &Gate, targets: &[usize]) -> [[Complex64; 4]; 4] {
        if targets.len() == 2 {
            assert_eq!(targets, [0, 1]);
            return gate.matrix_4x4();
        }
        let m = gate.matrix_2x2();
        let bit = 1 - targets[0];
        let mut out = [[Complex64::new(0.0, 0.0); 4]; 4];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                if (i & !(1 << bit)) == (j & !(1 << bit)) {
                    *cell = m[(i >> bit) & 1][(j >> bit) & 1];
                }
            }
        }
        out
    }
}

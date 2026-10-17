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
    assert_eq!(eighth_turns(FRAC_PI_2 + 1e-10), None);
    assert_eq!(eighth_turns(FRAC_PI_2 + 1e-13), Some(2));
}

#[test]
fn an_angle_the_recognizer_refuses_is_not_snapped_to_the_grid() {
    let theta = FRAC_PI_2 + 1e-10;
    let phase = Gate::P(theta).matrix_2x2();
    assert!(Gate::recognize_matrix(&phase).is_none());
    let err = lower_to_clifford_t(&Gate::P(theta), &[0], &mut |_, _| {}).unwrap_err();
    assert!(err.contains("off the pi/4 grid"), "{err}");
    let mut lowered = Vec::new();
    lower_to_pauli_forms(&Gate::Fused(Box::new(phase)), &[0], &mut |g, _| {
        lowered.push(g)
    })
    .unwrap();
    assert!(
        lowered.iter().all(|g| !matches!(g, Gate::S | Gate::Sdg)),
        "{lowered:?}"
    );
    assert!(equal_up_to_phase(
        &product_of_lowering(&Gate::Fused(Box::new(phase))),
        &phase
    ));
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
    let err =
        lower_to_pauli_forms(&Gate::cu(Gate::H.matrix_2x2()), &[0, 1], &mut |_, _| {}).unwrap_err();
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

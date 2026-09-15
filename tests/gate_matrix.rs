//! Named gates against their matrices, taken from the definition rather
//! than from a lowering: a lowering that implements the wrong gate agrees
//! with itself.

mod common;

use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};

use num_complex::Complex64;
use prism_q::circuit::openqasm::{self, Dialect};
use prism_q::circuit::qasm_export;

use common::{SV_EPS, assert_unitary_close, circuit_unitary};

const THETA: f64 = 0.7;
const PHI: f64 = 0.3;
const LAMBDA: f64 = 0.4;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn zero() -> Complex64 {
    c(0.0, 0.0)
}

fn one() -> Complex64 {
    c(1.0, 0.0)
}

fn phase(angle: f64) -> Complex64 {
    Complex64::from_polar(1.0, angle)
}

fn mat(rows: &[&[Complex64]]) -> Vec<Vec<Complex64>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

/// Identity of dimension `dim` with the listed index pairs exchanged.
fn permutation(dim: usize, swaps: &[(usize, usize)]) -> Vec<Vec<Complex64>> {
    let mut m = vec![vec![zero(); dim]; dim];
    let mut image: Vec<usize> = (0..dim).collect();
    for &(a, b) in swaps {
        image.swap(a, b);
    }
    for (col, &row) in image.iter().enumerate() {
        m[row][col] = one();
    }
    m
}

/// Identity of dimension `dim` carrying `e^{i theta}` on one diagonal entry.
fn diagonal_phase(dim: usize, index: usize, theta: f64) -> Vec<Vec<Complex64>> {
    let mut m = vec![vec![zero(); dim]; dim];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = if i == index { phase(theta) } else { one() };
    }
    m
}

/// Block-diagonal `I (+) target`, the controlled form with `targets[0]` the
/// control.
fn controlled(target: [[Complex64; 2]; 2]) -> Vec<Vec<Complex64>> {
    mat(&[
        &[one(), zero(), zero(), zero()],
        &[zero(), one(), zero(), zero()],
        &[zero(), zero(), target[0][0], target[0][1]],
        &[zero(), zero(), target[1][0], target[1][1]],
    ])
}

fn v_matrix() -> [[Complex64; 2]; 2] {
    [[c(0.5, 0.5), c(0.5, -0.5)], [c(0.5, -0.5), c(0.5, 0.5)]]
}

fn vi_matrix() -> [[Complex64; 2]; 2] {
    [[c(0.5, -0.5), c(0.5, 0.5)], [c(0.5, 0.5), c(0.5, -0.5)]]
}

struct GateCase {
    /// Gate application exactly as a program writes it, targeting `q[0]`
    /// upward so the assembled unitary is the gate's own matrix.
    call: &'static str,
    num_qubits: usize,
    /// True when the source angles mean different things under each dialect.
    native_family: bool,
    expected: fn() -> Vec<Vec<Complex64>>,
}

const fn case(
    call: &'static str,
    num_qubits: usize,
    expected: fn() -> Vec<Vec<Complex64>>,
) -> GateCase {
    GateCase {
        call,
        num_qubits,
        native_family: false,
        expected,
    }
}

const fn native_case(
    call: &'static str,
    num_qubits: usize,
    expected: fn() -> Vec<Vec<Complex64>>,
) -> GateCase {
    GateCase {
        call,
        num_qubits,
        native_family: true,
        expected,
    }
}

/// Braket's gate set, spelled as Braket spells it. Matrices are transcribed
/// from the Braket definitions, which share PRISM-Q's convention that
/// `targets[0]` is the most significant bit of a multi-qubit block.
fn braket_cases() -> Vec<GateCase> {
    vec![
        case("i q[0];", 1, || mat(&[&[one(), zero()], &[zero(), one()]])),
        case("x q[0];", 1, || mat(&[&[zero(), one()], &[one(), zero()]])),
        case("y q[0];", 1, || {
            mat(&[&[zero(), c(0.0, -1.0)], &[c(0.0, 1.0), zero()]])
        }),
        case("z q[0];", 1, || {
            mat(&[&[one(), zero()], &[zero(), c(-1.0, 0.0)]])
        }),
        case("h q[0];", 1, || {
            mat(&[
                &[c(FRAC_1_SQRT_2, 0.0), c(FRAC_1_SQRT_2, 0.0)],
                &[c(FRAC_1_SQRT_2, 0.0), c(-FRAC_1_SQRT_2, 0.0)],
            ])
        }),
        case("s q[0];", 1, || {
            mat(&[&[one(), zero()], &[zero(), c(0.0, 1.0)]])
        }),
        case("si q[0];", 1, || {
            mat(&[&[one(), zero()], &[zero(), c(0.0, -1.0)]])
        }),
        case("t q[0];", 1, || {
            mat(&[&[one(), zero()], &[zero(), phase(PI / 4.0)]])
        }),
        case("ti q[0];", 1, || {
            mat(&[&[one(), zero()], &[zero(), phase(-PI / 4.0)]])
        }),
        case("v q[0];", 1, || {
            let v = v_matrix();
            mat(&[&[v[0][0], v[0][1]], &[v[1][0], v[1][1]]])
        }),
        case("vi q[0];", 1, || {
            let v = vi_matrix();
            mat(&[&[v[0][0], v[0][1]], &[v[1][0], v[1][1]]])
        }),
        case("phaseshift(0.7) q[0];", 1, || {
            mat(&[&[one(), zero()], &[zero(), phase(THETA)]])
        }),
        case("rx(0.7) q[0];", 1, || {
            let (co, si) = ((THETA / 2.0).cos(), (THETA / 2.0).sin());
            mat(&[&[c(co, 0.0), c(0.0, -si)], &[c(0.0, -si), c(co, 0.0)]])
        }),
        case("ry(0.7) q[0];", 1, || {
            let (co, si) = ((THETA / 2.0).cos(), (THETA / 2.0).sin());
            mat(&[&[c(co, 0.0), c(-si, 0.0)], &[c(si, 0.0), c(co, 0.0)]])
        }),
        case("rz(0.7) q[0];", 1, || {
            mat(&[
                &[phase(-THETA / 2.0), zero()],
                &[zero(), phase(THETA / 2.0)],
            ])
        }),
        case("prx(0.7, 0.3) q[0];", 1, || {
            let (co, si) = ((THETA / 2.0).cos(), (THETA / 2.0).sin());
            let off = c(0.0, -si);
            mat(&[
                &[c(co, 0.0), off * phase(-PHI)],
                &[off * phase(PHI), c(co, 0.0)],
            ])
        }),
        case("U(0.7, 0.3, 0.4) q[0];", 1, || {
            let (co, si) = ((THETA / 2.0).cos(), (THETA / 2.0).sin());
            mat(&[
                &[c(co, 0.0), -phase(LAMBDA) * si],
                &[phase(PHI) * si, phase(PHI + LAMBDA) * co],
            ])
        }),
        native_case("gpi(0.3) q[0];", 1, || {
            mat(&[&[zero(), phase(-PHI)], &[phase(PHI), zero()]])
        }),
        native_case("gpi2(0.3) q[0];", 1, || {
            let r = c(FRAC_1_SQRT_2, 0.0);
            let off = c(0.0, -FRAC_1_SQRT_2);
            mat(&[&[r, off * phase(-PHI)], &[off * phase(PHI), r]])
        }),
        case("cnot q[0], q[1];", 2, || permutation(4, &[(2, 3)])),
        case("cy q[0], q[1];", 2, || {
            controlled([[zero(), c(0.0, -1.0)], [c(0.0, 1.0), zero()]])
        }),
        case("cz q[0], q[1];", 2, || diagonal_phase(4, 3, PI)),
        case("cv q[0], q[1];", 2, || controlled(v_matrix())),
        case("swap q[0], q[1];", 2, || permutation(4, &[(1, 2)])),
        case("iswap q[0], q[1];", 2, || {
            mat(&[
                &[one(), zero(), zero(), zero()],
                &[zero(), zero(), c(0.0, 1.0), zero()],
                &[zero(), c(0.0, 1.0), zero(), zero()],
                &[zero(), zero(), zero(), one()],
            ])
        }),
        case("ecr q[0], q[1];", 2, || {
            let r = c(FRAC_1_SQRT_2, 0.0);
            let i = c(0.0, FRAC_1_SQRT_2);
            mat(&[
                &[zero(), zero(), r, i],
                &[zero(), zero(), i, r],
                &[r, -i, zero(), zero()],
                &[-i, r, zero(), zero()],
            ])
        }),
        case("pswap(0.7) q[0], q[1];", 2, || {
            mat(&[
                &[one(), zero(), zero(), zero()],
                &[zero(), zero(), phase(THETA), zero()],
                &[zero(), phase(THETA), zero(), zero()],
                &[zero(), zero(), zero(), one()],
            ])
        }),
        case("xy(0.7) q[0], q[1];", 2, || {
            let (co, si) = ((THETA / 2.0).cos(), (THETA / 2.0).sin());
            mat(&[
                &[one(), zero(), zero(), zero()],
                &[zero(), c(co, 0.0), c(0.0, si), zero()],
                &[zero(), c(0.0, si), c(co, 0.0), zero()],
                &[zero(), zero(), zero(), one()],
            ])
        }),
        case("xx(0.7) q[0], q[1];", 2, || {
            let (co, si) = ((THETA / 2.0).cos(), (THETA / 2.0).sin());
            let (d, o) = (c(co, 0.0), c(0.0, -si));
            mat(&[
                &[d, zero(), zero(), o],
                &[zero(), d, o, zero()],
                &[zero(), o, d, zero()],
                &[o, zero(), zero(), d],
            ])
        }),
        case("yy(0.7) q[0], q[1];", 2, || {
            let (co, si) = ((THETA / 2.0).cos(), (THETA / 2.0).sin());
            let (d, o) = (c(co, 0.0), c(0.0, -si));
            mat(&[
                &[d, zero(), zero(), -o],
                &[zero(), d, o, zero()],
                &[zero(), o, d, zero()],
                &[-o, zero(), zero(), d],
            ])
        }),
        case("zz(0.7) q[0], q[1];", 2, || {
            let (m, p) = (phase(-THETA / 2.0), phase(THETA / 2.0));
            mat(&[
                &[m, zero(), zero(), zero()],
                &[zero(), p, zero(), zero()],
                &[zero(), zero(), p, zero()],
                &[zero(), zero(), zero(), m],
            ])
        }),
        case("cphaseshift(0.7) q[0], q[1];", 2, || {
            diagonal_phase(4, 3, THETA)
        }),
        case("cphaseshift00(0.7) q[0], q[1];", 2, || {
            diagonal_phase(4, 0, THETA)
        }),
        case("cphaseshift01(0.7) q[0], q[1];", 2, || {
            diagonal_phase(4, 1, THETA)
        }),
        case("cphaseshift10(0.7) q[0], q[1];", 2, || {
            diagonal_phase(4, 2, THETA)
        }),
        native_case("ms(0.3, 0.4, 0.7) q[0], q[1];", 2, || {
            let (co, si) = ((THETA / 2.0).cos(), (THETA / 2.0).sin());
            let (d, o) = (c(co, 0.0), c(0.0, -si));
            let (sum, diff) = (PHI + LAMBDA, PHI - LAMBDA);
            mat(&[
                &[d, zero(), zero(), o * phase(-sum)],
                &[zero(), d, o * phase(-diff), zero()],
                &[zero(), o * phase(diff), d, zero()],
                &[o * phase(sum), zero(), zero(), d],
            ])
        }),
        case("ccnot q[0], q[1], q[2];", 3, || permutation(8, &[(6, 7)])),
        case("cswap q[0], q[1], q[2];", 3, || permutation(8, &[(5, 6)])),
    ]
}

fn program(call: &str, num_qubits: usize) -> String {
    format!("OPENQASM 3.0;\nqubit[{num_qubits}] q;\n{call}\n")
}

fn unitary_of(call: &str, num_qubits: usize, dialect: Dialect) -> Vec<Vec<Complex64>> {
    let source = program(call, num_qubits);
    let circuit = openqasm::parse_with(&source, dialect)
        .unwrap_or_else(|e| panic!("`{call}` did not parse: {e}"));
    circuit_unitary(&circuit)
}

#[test]
fn braket_gate_matrices() {
    for case in braket_cases() {
        let dialect = if case.native_family {
            Dialect::Braket
        } else {
            Dialect::Native
        };
        let actual = unitary_of(case.call, case.num_qubits, dialect);
        assert_unitary_close(&actual, &(case.expected)(), SV_EPS, case.call);
    }
}

// Only the hardware-native family reads its angles differently, so every other
// name must be untouched by the dialect it is parsed under.
#[test]
fn dialect_moves_only_the_native_family() {
    for case in braket_cases() {
        let native = unitary_of(case.call, case.num_qubits, Dialect::Native);
        let braket = unitary_of(case.call, case.num_qubits, Dialect::Braket);
        let differs = native
            .iter()
            .zip(&braket)
            .any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).norm() > SV_EPS));
        assert_eq!(
            differs, case.native_family,
            "`{}` dialect sensitivity",
            case.call
        );
    }
}

// A quarter turn is pi/2 radians, so a native call and the Braket call whose
// every angle is scaled by tau name the same unitary. `ms` carries three, and
// all of them scale.
#[test]
fn native_turns_agree_with_braket_radians() {
    let cases: [(&str, usize, &[f64]); 4] = [
        ("gpi", 1, &[0.25]),
        ("gpi2", 1, &[0.3]),
        ("ms", 2, &[0.1, 0.2]),
        ("ms", 2, &[0.1, 0.2, 0.125]),
    ];
    for (name, num_qubits, turns) in cases {
        let targets = match num_qubits {
            1 => "q[0]",
            _ => "q[0], q[1]",
        };
        let call = |angles: &[f64]| {
            let args: Vec<String> = angles.iter().map(f64::to_string).collect();
            format!("{name}({}) {targets};", args.join(", "))
        };
        let radians: Vec<f64> = turns.iter().map(|t| t * TAU).collect();
        let native = unitary_of(&call(turns), num_qubits, Dialect::Native);
        let braket = unitary_of(&call(&radians), num_qubits, Dialect::Braket);
        assert_unitary_close(&native, &braket, SV_EPS, &call(turns));
    }
}

// An alias must be the same gate, not merely a gate of the same shape.
#[test]
fn braket_aliases_match_their_targets() {
    let pairs = [
        ("i q[0];", "id q[0];", 1),
        ("si q[0];", "sdg q[0];", 1),
        ("ti q[0];", "tdg q[0];", 1),
        ("v q[0];", "sx q[0];", 1),
        ("vi q[0];", "sxdg q[0];", 1),
        ("phaseshift(0.7) q[0];", "p(0.7) q[0];", 1),
        ("prx(0.7, 0.3) q[0];", "r(0.7, 0.3) q[0];", 1),
        ("cphaseshift(0.7) q[0], q[1];", "cp(0.7) q[0], q[1];", 2),
        ("cv q[0], q[1];", "csx q[0], q[1];", 2),
        ("xx(0.7) q[0], q[1];", "rxx(0.7) q[0], q[1];", 2),
        ("yy(0.7) q[0], q[1];", "ryy(0.7) q[0], q[1];", 2),
        ("zz(0.7) q[0], q[1];", "rzz(0.7) q[0], q[1];", 2),
        ("ccnot q[0], q[1], q[2];", "ccx q[0], q[1], q[2];", 3),
    ];
    for (braket, native, num_qubits) in pairs {
        let lhs = unitary_of(braket, num_qubits, Dialect::Native);
        let rhs = unitary_of(native, num_qubits, Dialect::Native);
        assert_unitary_close(&lhs, &rhs, SV_EPS, &format!("{braket} vs {native}"));
    }
}

// `xy` is Braket's XY interaction. The bare two-letter rotation aliases must
// not pull it into the `r<letters>` rule and silently make it `rxy`.
#[test]
fn xy_is_the_interaction_not_a_pauli_rotation() {
    let interaction = unitary_of("xy(0.7) q[0], q[1];", 2, Dialect::Native);
    let rotation = unitary_of("rxy(0.7) q[0], q[1];", 2, Dialect::Native);
    let same = interaction
        .iter()
        .zip(&rotation)
        .all(|(a, b)| a.iter().zip(b).all(|(x, y)| (x - y).norm() < SV_EPS));
    assert!(!same, "`xy` resolved to the `rxy` Pauli rotation");
}

// Export names a gate however PRISM-Q spells it, which need not be the
// spelling the source used, but the unitary has to survive the trip. A gate
// the exporter cannot spell fails here rather than at a user's round trip.
#[test]
fn every_gate_round_trips_through_export() {
    for case in braket_cases() {
        let dialect = if case.native_family {
            Dialect::Braket
        } else {
            Dialect::Native
        };
        let source = program(case.call, case.num_qubits);
        let circuit = openqasm::parse_with(&source, dialect).unwrap();
        let exported = qasm_export::to_qasm3(&circuit)
            .unwrap_or_else(|e| panic!("`{}` did not export: {e}", case.call));
        // Export always writes PRISM-Q's own spelling, so the text reads back
        // under the native dialect whatever the source was written in.
        let reparsed = openqasm::parse(&exported)
            .unwrap_or_else(|e| panic!("`{}` exported to text that failed: {e}", case.call));
        assert_unitary_close(
            &circuit_unitary(&reparsed),
            &(case.expected)(),
            SV_EPS,
            &format!("{} round trip", case.call),
        );
    }
}

// Every case is a unitary; a transcription slip in the expected side would
// most likely break this first.
#[test]
fn expected_matrices_are_unitary() {
    for case in braket_cases() {
        let m = (case.expected)();
        let dim = m.len();
        for row in 0..dim {
            for col in 0..dim {
                let entry: Complex64 = (0..dim).map(|k| m[k][row].conj() * m[k][col]).sum();
                let want = if row == col { one() } else { zero() };
                assert!(
                    (entry - want).norm() < SV_EPS,
                    "`{}` expected matrix is not unitary at [{row}][{col}]",
                    case.call
                );
            }
        }
    }
}

//! Gate modifiers against the unitaries they name: `ctrl`, `negctrl`, `inv`
//! and `pow`, each anchored on the unmodified gate's own matrix.

mod common;

use num_complex::Complex64;
use prism_q::PrismError;
use prism_q::circuit::openqasm;

use common::{SV_EPS, circuit_unitary};

fn unitary(qubits: usize, body: &str) -> Vec<Vec<Complex64>> {
    let source = format!("OPENQASM 3.0;\nqubit[{qubits}] q;\n{body}\n");
    let circuit = openqasm::parse(&source).unwrap_or_else(|e| panic!("`{body}`: {e}"));
    circuit_unitary(&circuit)
}

fn parse_err(qubits: usize, body: &str) -> PrismError {
    let source = format!("OPENQASM 3.0;\nqubit[{qubits}] q;\n{body}\n");
    openqasm::parse(&source)
        .err()
        .unwrap_or_else(|| panic!("`{body}` should not parse"))
}

/// `u` with one control added at `q[0]`.
///
/// [`circuit_unitary`] indexes in gate-matrix order, where `q[0]` is the high
/// bit, so the control is the top half of the index rather than the bottom.
/// `fires_on` is the control value the gate acts under, so `false` is what
/// `negctrl` spells.
fn controlled(u: &[Vec<Complex64>], fires_on: bool) -> Vec<Vec<Complex64>> {
    let body = u.len();
    let dim = body * 2;
    (0..dim)
        .map(|row| {
            (0..dim)
                .map(|column| {
                    let (row_control, column_control) = (row >= body, column >= body);
                    if row_control != column_control {
                        Complex64::new(0.0, 0.0)
                    } else if row_control == fires_on {
                        u[row % body][column % body]
                    } else if row == column {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                })
                .collect()
        })
        .collect()
}

/// Gate calls on two qubits, spelled so `q[0], q[1]` can be shifted to
/// `q[1], q[2]` under a control.
const TWO_QUBIT_CALLS: [&str; 8] = [
    "cx", "cz", "swap", "iswap", "dcx", "rzz(0.7)", "crx(0.4)", "cp(1.1)",
];

// `ctrl @ U` is `U` with one more control, whatever lowering `U` needed. The
// expected side is built from the unmodified gate's own matrix, so a wrong
// lowering cannot agree with a wrong expectation.
#[test]
fn a_control_adds_one_control_to_the_gate_unitary() {
    for call in TWO_QUBIT_CALLS {
        let body = unitary(2, &format!("{call} q[0], q[1];"));
        let actual = unitary(3, &format!("ctrl @ {call} q[0], q[1], q[2];"));
        common::assert_unitary_close(
            &actual,
            &controlled(&body, true),
            SV_EPS,
            &format!("ctrl @ {call}"),
        );
    }
}

#[test]
fn a_control_reaches_single_qubit_and_lowered_gates() {
    for call in [
        "h",
        "t",
        "rx(0.3)",
        "u3(0.3, 0.4, 0.5)",
        "u2(0.2, 0.9)",
        "sx",
    ] {
        let body = unitary(1, &format!("{call} q[0];"));
        let actual = unitary(2, &format!("ctrl @ {call} q[0], q[1];"));
        common::assert_unitary_close(
            &actual,
            &controlled(&body, true),
            SV_EPS,
            &format!("ctrl @ {call}"),
        );
    }
}

// `negctrl` is `ctrl` with the polarity flipped: the gate fires on |0>.
#[test]
fn a_negative_control_fires_on_zero() {
    for call in ["x", "h", "rz(0.6)"] {
        let body = unitary(1, &format!("{call} q[0];"));
        let actual = unitary(2, &format!("negctrl @ {call} q[0], q[1];"));
        common::assert_unitary_close(
            &actual,
            &controlled(&body, false),
            SV_EPS,
            &format!("negctrl @ {call}"),
        );
    }
    let swap = unitary(2, "swap q[0], q[1];");
    common::assert_unitary_close(
        &unitary(3, "negctrl @ swap q[0], q[1], q[2];"),
        &controlled(&swap, false),
        SV_EPS,
        "negctrl @ swap",
    );
}

// A chain consumes one qubit per control, in the order the modifiers are
// written, so the polarity of each follows its own position.
#[test]
fn a_control_chain_consumes_qubits_in_order() {
    let pairs = [
        ("ctrl @ ctrl @ x q[0], q[1], q[2];", "ccx q[0], q[1], q[2];"),
        (
            "ctrl @ negctrl @ x q[0], q[1], q[2];",
            "x q[1];\nccx q[0], q[1], q[2];\nx q[1];",
        ),
        (
            "negctrl @ ctrl @ x q[0], q[1], q[2];",
            "x q[0];\nccx q[0], q[1], q[2];\nx q[0];",
        ),
        (
            "negctrl @ negctrl @ x q[0], q[1], q[2];",
            "x q[0];\nx q[1];\nccx q[0], q[1], q[2];\nx q[1];\nx q[0];",
        ),
    ];
    for (modified, explicit) in pairs {
        common::assert_unitary_close(
            &unitary(3, modified),
            &unitary(3, explicit),
            SV_EPS,
            modified,
        );
    }
}

#[test]
fn a_control_reaches_a_user_gate_body() {
    let definition = "gate g(t) a, b { h a; cx a, b; rz(t) b; }\n";
    let body = unitary(2, &format!("{definition}g(0.8) q[0], q[1];"));
    let actual = unitary(3, &format!("{definition}ctrl @ g(0.8) q[0], q[1], q[2];"));
    common::assert_unitary_close(&actual, &controlled(&body, true), SV_EPS, "ctrl @ g");
}

// `ctrl @ swap` is the Fredkin gate the `cswap` keyword already spells, so the
// two have to agree instruction for instruction in effect.
#[test]
fn a_controlled_swap_is_the_fredkin_gate() {
    common::assert_unitary_close(
        &unitary(3, "ctrl @ swap q[0], q[1], q[2];"),
        &unitary(3, "cswap q[0], q[1], q[2];"),
        SV_EPS,
        "ctrl @ swap",
    );
}

// The closed-form power is anchored where the half turn already has a name.
#[test]
fn fractional_powers_match_the_named_half_turns() {
    for (powered, named) in [
        ("pow(0.5) @ x q[0];", "sx q[0];"),
        ("pow(-0.5) @ x q[0];", "sxdg q[0];"),
        ("pow(0.5) @ z q[0];", "s q[0];"),
        ("pow(-0.5) @ z q[0];", "sdg q[0];"),
        ("pow(0.5) @ s q[0];", "t q[0];"),
        ("pow(0.5) @ sdg q[0];", "tdg q[0];"),
    ] {
        common::assert_unitary_close(&unitary(1, powered), &unitary(1, named), SV_EPS, powered);
    }
}

// Where no name exists, the power still has to compose back to the gate.
#[test]
fn a_fractional_power_composes_back_to_the_gate() {
    for call in [
        "x", "y", "z", "h", "t", "rx(0.9)", "ry(2.3)", "p(1.7)", "sx",
    ] {
        for parts in [2usize, 3, 5] {
            let repeated: String = (0..parts)
                .map(|_| format!("pow(1/{parts}) @ {call} q[0];\n"))
                .collect();
            common::assert_unitary_close(
                &unitary(1, &repeated),
                &unitary(1, &format!("{call} q[0];")),
                SV_EPS,
                &format!("{call} in {parts} parts"),
            );
        }
    }
}

// A whole-number power of a wider gate is repetition, which needs no matrix.
#[test]
fn an_integer_power_repeats_a_wider_gate() {
    for (powered, expected) in [
        ("pow(2) @ cx q[0], q[1];", ""),
        ("pow(3) @ swap q[0], q[1];", "swap q[0], q[1];"),
        ("pow(-1) @ cx q[0], q[1];", "cx q[0], q[1];"),
        ("pow(2) @ rzz(0.5) q[0], q[1];", "rzz(1.0) q[0], q[1];"),
        ("pow(0) @ iswap q[0], q[1];", ""),
    ] {
        common::assert_unitary_close(
            &unitary(2, powered),
            &unitary(
                2,
                if expected.is_empty() {
                    "id q[0];"
                } else {
                    expected
                },
            ),
            SV_EPS,
            powered,
        );
    }
}

// `inv` and `pow` commute with a control, so a chain mixing them lands on the
// same unitary whichever way it is read.
#[test]
fn a_control_commutes_with_inverse_and_power() {
    let body = unitary(1, "inv @ t q[0];");
    common::assert_unitary_close(
        &unitary(2, "ctrl @ inv @ t q[0], q[1];"),
        &controlled(&body, true),
        SV_EPS,
        "ctrl @ inv @ t",
    );
    let squared = unitary(1, "pow(2) @ t q[0];");
    common::assert_unitary_close(
        &unitary(2, "ctrl @ pow(2) @ t q[0], q[1];"),
        &controlled(&squared, true),
        SV_EPS,
        "ctrl @ pow(2) @ t",
    );
}

// The declines name the gate and say what is missing, rather than reading as a
// parse failure.
#[test]
fn unsupported_modifier_targets_are_named() {
    // A fraction needs the whole lowering as one matrix, so a span wider than
    // the composition cap declines rather than running out of memory.
    let err = parse_err(6, "pow(0.5) @ mcx q[0], q[1], q[2], q[3], q[4], q[5];");
    let text = format!("{err}");
    assert!(
        matches!(err, PrismError::UnsupportedConstruct { .. }) && text.contains("spans more than"),
        "got {text}"
    );

    // A control takes a qubit of its own, so the unmodified argument count is
    // one short rather than merely unsupported.
    assert!(matches!(
        parse_err(3, "ctrl @ x q[0];"),
        PrismError::GateArity { .. }
    ));
}

// A `def` is a subroutine rather than a gate, so the spec gives it no
// controlled form and neither does this.
#[test]
fn a_subroutine_call_has_no_controlled_form() {
    let source = "OPENQASM 3.0;\nqubit[2] q;\ndef g(qubit a) { h a; }\nctrl @ g(q[0]);\n";
    let err = openqasm::parse(source).unwrap_err();
    assert!(
        matches!(err, PrismError::UnsupportedConstruct { .. })
            && format!("{err}").contains("subroutine"),
        "got {err}"
    );
}

// A fractional power is the principal one, which is not what scaling the
// axis-angle pair gives once the rotation passes pi: there `cos(a)` turns
// negative and the half-angle picks the other square root. Composing the
// result back to the gate cannot catch that, since both roots compose.
#[test]
fn a_fractional_power_of_a_long_rotation_is_the_principal_one() {
    for (turn, half) in [
        ("rx(3*pi/2)", "rx(3*pi/4)"),
        ("rx(-3*pi/2)", "rx(-3*pi/4)"),
        ("ry(5*pi/3)", "ry(5*pi/6)"),
        ("rz(1.9*pi)", "rz(0.95*pi)"),
        // `p(1.5*pi)` is `p(-0.5*pi)` as a matrix, and the principal root
        // follows the eigenvalue rather than the written angle.
        ("p(1.5*pi)", "p(-0.25*pi)"),
    ] {
        common::assert_unitary_close(
            &unitary(1, &format!("pow(0.5) @ {turn} q[0];")),
            &unitary(1, &format!("{half} q[0];")),
            SV_EPS,
            turn,
        );
    }
}

// A half turn written either way round is the same gate, and its square root
// has to follow. The eigenvalue sits on the branch cut, where the two
// spellings land on opposite sides by one rounding step.
#[test]
fn the_two_spellings_of_a_half_turn_take_the_same_root() {
    for (positive, negative) in [("p(pi)", "p(-pi)"), ("rz(2*pi)", "rz(-2*pi)")] {
        common::assert_unitary_close(
            &unitary(1, &format!("pow(0.5) @ {positive} q[0];")),
            &unitary(1, &format!("pow(0.5) @ {negative} q[0];")),
            SV_EPS,
            positive,
        );
    }
    common::assert_unitary_close(
        &unitary(1, "pow(0.5) @ p(-pi) q[0];"),
        &unitary(1, "s q[0];"),
        SV_EPS,
        "sqrt of a negative half turn",
    );
}

// A literal exponent bounds what it can ask for, since the repetition and the
// matrix product both run it out rather than returning.
#[test]
fn an_oversized_power_is_rejected_rather_than_run() {
    for body in [
        "pow(1000000000000000000) @ x q[0];",
        "pow(10 ** 18) @ cx q[0], q[1];",
        "pow(-2000000) @ h q[0];",
    ] {
        let err = parse_err(2, body);
        assert!(
            format!("{err}").contains("repeats a gate more than"),
            "`{body}`: got {err}"
        );
    }
}

// The language puts no requirement on the space around `@`.
#[test]
fn a_modifier_chain_needs_no_space_around_its_separator() {
    for (tight, spaced) in [
        ("ctrl@x q[0], q[1];", "ctrl @ x q[0], q[1];"),
        ("inv@ t q[0];", "inv @ t q[0];"),
        ("pow(2)@t q[0];", "pow(2) @ t q[0];"),
        (
            "negctrl @ctrl@ x q[0], q[1], q[2];",
            "negctrl @ ctrl @ x q[0], q[1], q[2];",
        ),
    ] {
        common::assert_unitary_close(&unitary(3, tight), &unitary(3, spaced), SV_EPS, tight);
    }
}

/// Gate calls whose lowering is a matrix rather than a named controlled form,
/// so `ctrl @` reaches them only through a synthesized one.
const MATRIX_CALLS: [&str; 10] = [
    "ecr",
    "xy(0.3)",
    "pswap(0.4)",
    "cphaseshift00(0.5)",
    "cphaseshift01(0.6)",
    "cphaseshift10(0.7)",
    "syc",
    "sqrt_iswap",
    "xx_plus_yy(0.3, 0.4)",
    "rxx(0.9)",
];

// A two-qubit gate carried as a matrix has no controlled variant wide enough
// to hold it, so it is synthesized. The expected side is still the unmodified
// gate's own matrix with one control added.
#[test]
fn a_control_reaches_a_gate_carried_as_a_matrix() {
    for call in MATRIX_CALLS {
        let body = unitary(2, &format!("{call} q[0], q[1];"));
        let actual = unitary(3, &format!("ctrl @ {call} q[0], q[1], q[2];"));
        common::assert_unitary_close(
            &actual,
            &controlled(&body, true),
            SV_EPS,
            &format!("ctrl @ {call}"),
        );
    }
}

// The synthesis does not care how many controls it is given, so a chain has to
// agree with the same gate controlled twice.
#[test]
fn a_matrix_gate_takes_a_chain_of_controls() {
    for call in ["ecr", "xy(0.3)", "rxx(0.9)"] {
        let body = unitary(2, &format!("{call} q[0], q[1];"));
        let once = controlled(&body, true);
        let twice = controlled(&once, true);
        common::assert_unitary_close(
            &unitary(4, &format!("ctrl @ ctrl @ {call} q[0], q[1], q[2], q[3];")),
            &twice,
            SV_EPS,
            &format!("ctrl @ ctrl @ {call}"),
        );
        // A negative control differs only in the value it fires on.
        common::assert_unitary_close(
            &unitary(3, &format!("negctrl @ {call} q[0], q[1], q[2];")),
            &controlled(&body, false),
            SV_EPS,
            &format!("negctrl @ {call}"),
        );
    }
}

// A Pauli rotation wider than two qubits has no matrix accessor and is
// controlled through its CNOT ladder instead.
#[test]
fn a_control_reaches_a_wide_pauli_rotation() {
    for call in ["rxyz(0.7)", "rzzz(1.1)", "ryx z(0.3)"] {
        let call = call.replace(' ', "");
        let body = unitary(3, &format!("{call} q[0], q[1], q[2];"));
        let actual = unitary(4, &format!("ctrl @ {call} q[0], q[1], q[2], q[3];"));
        common::assert_unitary_close(
            &actual,
            &controlled(&body, true),
            SV_EPS,
            &format!("ctrl @ {call}"),
        );
    }
}

// A fraction of a wider gate is a matrix power, read off the spectrum and
// synthesized back into the variants the backends carry. Composing the parts
// has to return the gate.
#[test]
fn a_fractional_power_of_a_wider_gate_composes_back() {
    for (qubits, call) in [
        (2usize, "cx q[0], q[1];"),
        (2, "cz q[0], q[1];"),
        (2, "swap q[0], q[1];"),
        (2, "iswap q[0], q[1];"),
        (2, "ecr q[0], q[1];"),
        (2, "rxx(0.9) q[0], q[1];"),
        (2, "xy(0.4) q[0], q[1];"),
        (3, "ccx q[0], q[1], q[2];"),
        (3, "rxyz(0.7) q[0], q[1], q[2];"),
    ] {
        for parts in [2usize, 3] {
            let repeated: String = (0..parts)
                .map(|_| format!("pow(1/{parts}) @ {call}\n"))
                .collect();
            common::assert_unitary_close(
                &unitary(qubits, &repeated),
                &unitary(qubits, call),
                SV_EPS,
                &format!("{call} in {parts} parts"),
            );
        }
    }
}

// The square root of a swap is a named gate elsewhere, so the fraction is
// anchored rather than only self-consistent.
#[test]
fn a_half_power_of_swap_is_the_square_root_of_swap() {
    let half = unitary(2, "pow(0.5) @ swap q[0], q[1];");
    let expected = {
        let (a, b) = (Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5));
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        vec![
            vec![o, z, z, z],
            vec![z, a, b, z],
            vec![z, b, a, z],
            vec![z, z, z, o],
        ]
    };
    common::assert_unitary_close(&half, &expected, SV_EPS, "sqrt(swap)");
}

// `gphase` scales the whole state. The phase is unobservable on its own but a
// `state_vector` result reports it, and under a control it becomes an ordinary
// relative phase, so it is carried rather than dropped.
#[test]
fn a_global_phase_scales_the_state() {
    let phased = unitary(2, "gphase(0.7);\nh q[0];");
    let plain = unitary(2, "h q[0];");
    let scale = Complex64::from_polar(1.0, 0.7);
    for (row, entries) in phased.iter().enumerate() {
        for (column, entry) in entries.iter().enumerate() {
            assert!(
                (entry - plain[row][column] * scale).norm() < SV_EPS,
                "({row}, {column}) is {entry}"
            );
        }
    }
}

// `ctrl @ gphase(theta)` is `p(theta)` on the control, which is what makes a
// dropped global phase a wrong answer rather than a harmless one.
#[test]
fn a_controlled_global_phase_is_a_phase_gate() {
    common::assert_unitary_close(
        &unitary(1, "ctrl @ gphase(0.7) q[0];"),
        &unitary(1, "p(0.7) q[0];"),
        SV_EPS,
        "ctrl @ gphase",
    );
    common::assert_unitary_close(
        &unitary(2, "ctrl @ ctrl @ gphase(0.4) q[0], q[1];"),
        &unitary(2, "cp(0.4) q[0], q[1];"),
        SV_EPS,
        "ctrl @ ctrl @ gphase",
    );
    common::assert_unitary_close(
        &unitary(1, "negctrl @ gphase(0.7) q[0];"),
        &unitary(1, "x q[0];\np(0.7) q[0];\nx q[0];"),
        SV_EPS,
        "negctrl @ gphase",
    );
}

// `inv` and `pow` fold into the angle rather than needing an expansion.
#[test]
fn a_global_phase_folds_its_modifiers() {
    for (modified, plain) in [
        ("inv @ gphase(0.7);", "gphase(-0.7);"),
        ("pow(3) @ gphase(0.2);", "gphase(0.6);"),
        ("pow(0.5) @ gphase(0.8);", "gphase(0.4);"),
        ("inv @ ctrl @ gphase(0.7) q[0];", "p(-0.7) q[0];"),
    ] {
        common::assert_unitary_close(&unitary(1, modified), &unitary(1, plain), SV_EPS, modified);
    }
}

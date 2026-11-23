//! OpenQASM `input` declarations: affine angle links, supplied values, and the
//! gradient and replay paths a scaled link reaches.

use crate::common;

use common::{SEED, SV_EPS};
use prism_q::circuit::dynamic::ClassicalValue;
use prism_q::circuit::fusion::fuse_circuit;
use prism_q::circuit::openqasm;
use prism_q::sim::gradient::{run_expectation_gradient, run_expectation_gradient_shift};
use prism_q::sim::run_expectation_values;
use prism_q::{
    Circuit, Gate, Parameters, PauliObservable, PauliTerm, PreparedCircuit, PrismError,
    simulate_program,
};

fn assert_same_unitary(got: &Circuit, want: &Circuit, label: &str) {
    common::assert_unitary_close(
        &common::circuit_unitary(got),
        &common::circuit_unitary(want),
        SV_EPS,
        label,
    );
}

fn unsupported(err: PrismError, needle: &str, label: &str) {
    match err {
        PrismError::UnsupportedConstruct { construct, .. } => {
            assert!(construct.contains(needle), "{label}: {construct}");
        }
        other => panic!("{label}: expected UnsupportedConstruct, got {other:?}"),
    }
}

fn invalid(err: PrismError, needle: &str, label: &str) {
    match err {
        PrismError::InvalidParameter { message } => {
            assert!(message.contains(needle), "{label}: {message}");
        }
        other => panic!("{label}: expected InvalidParameter, got {other:?}"),
    }
}

const AFFINE: &str = "OPENQASM 3.0;
input float theta;
input angle phi;
qubit[3] q;
gate half(a) x { rz(a / 2) x; }
def turn(qubit x, float w) { ry(pi - w) x; }
h q;
rx(-theta / 2) q[0];
rx(2 * theta + pi) q[1];
rzz(theta - 0.5) q[0], q[1];
for int i in [0:2] { rz(phi * (i + 1)) q[i]; }
half(phi) q[2];
turn(q[0], 3 * theta);
rx(theta + theta) q[2];
";

fn affine_literal(theta: f64, phi: f64) -> Circuit {
    openqasm::parse(&format!(
        "OPENQASM 3.0;
qubit[3] q;
h q;
rx({}) q[0];
rx({}) q[1];
rzz({}) q[0], q[1];
rz({phi}) q[0];
rz({}) q[1];
rz({}) q[2];
rz({}) q[2];
ry({}) q[0];
rx({}) q[2];
",
        -theta / 2.0,
        2.0 * theta + std::f64::consts::PI,
        theta - 0.5,
        2.0 * phi,
        3.0 * phi,
        phi / 2.0,
        std::f64::consts::PI - 3.0 * theta,
        2.0 * theta,
    ))
    .expect("literal")
}

#[test]
fn an_affine_input_angle_binds_as_the_expression_written() {
    let (template, params) = openqasm::parse_parametric(AFFINE).expect("parse");
    assert_eq!(params.num_slots(), 2);
    assert_eq!(params.links().len(), 9);
    let first = params.links()[0];
    assert_eq!((first.slot, first.scale, first.offset), (0, -0.5, 0.0));
    let turn = params
        .links()
        .iter()
        .find(|link| link.scale == -3.0)
        .expect("the def body links with the chain of scales");
    assert_eq!(turn.offset, std::f64::consts::PI);

    for (theta, phi) in [(0.0, 0.0), (0.41, 1.27), (-2.3, 5.9)] {
        let bound = params.bind(&template, &[theta, phi]).expect("bind");
        assert_same_unitary(&bound, &affine_literal(theta, phi), "affine bind");
        let values = params.values(&bound).expect("values");
        assert!((values[0] - theta).abs() < 1e-12 && (values[1] - phi).abs() < 1e-12);
    }
}

#[test]
fn the_template_holds_each_angle_at_an_input_of_zero() {
    let (template, _) = openqasm::parse_parametric(AFFINE).expect("parse");
    assert_same_unitary(&template, &affine_literal(0.0, 0.0), "template");
}

#[test]
fn an_affine_input_carries_its_scale_into_every_gradient() {
    let source = "OPENQASM 3.0;
input float theta;
qubit[3] q;
h q;
rx(-3 * theta / 2 + 0.4) q[0];
rzz(theta * 2) q[0], q[1];
ry(theta) q[2];
cx q[1], q[2];
";
    let (template, params) = openqasm::parse_parametric(source).expect("parse");
    let obs = PauliObservable::from_terms([
        (1.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (0.5, vec![PauliTerm::x(2)]),
        (-0.7, vec![PauliTerm::y(0)]),
    ])
    .unwrap();
    let energy = |theta: f64| {
        let circuit = params.bind(&template, &[theta]).unwrap();
        let observables: Vec<Vec<PauliTerm>> = obs.terms().iter().map(|(_, p)| p.clone()).collect();
        let per_term = run_expectation_values(&circuit, &observables, SEED).unwrap();
        obs.terms()
            .iter()
            .zip(per_term)
            .map(|((c, _), v)| c * v)
            .sum::<f64>()
    };
    let theta = 0.37;
    let eps = 1e-5;
    let expected = (energy(theta + eps) - energy(theta - eps)) / (2.0 * eps);

    let bound = params.bind(&template, &[theta]).unwrap();
    let adjoint = run_expectation_gradient(&bound, &obs, &params, SEED).unwrap();
    let shift = run_expectation_gradient_shift(&bound, &obs, &params, SEED).unwrap();
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
    let prepared_adjoint = prepared.expectation_gradient(&[theta], &obs, SEED).unwrap();
    let prepared_shift = prepared
        .expectation_gradient_shift(&[theta], &obs, SEED)
        .unwrap();
    for (label, gradient) in [
        ("adjoint", adjoint.gradient),
        ("shift", shift.gradient),
        ("prepared adjoint", prepared_adjoint.gradient),
        ("prepared shift", prepared_shift.gradient),
    ] {
        assert!(
            (gradient[0] - expected).abs() < 1e-6,
            "{label}: {} vs finite difference {expected}",
            gradient[0]
        );
    }
}

// Sixteen qubits so the prepared circuit captures a fusion plan, which replays
// the angles `bind` wrote; a scaled link has to reach the fused stream intact.
#[test]
fn a_scaled_link_rebinds_through_the_fusion_plan() {
    let source = "OPENQASM 3.0;
input float theta;
input angle phi;
qubit[16] q;
for int i in [0:15] { ry(2 * theta + i * 0.125) q[i]; }
for int i in [0:14] { cx q[i], q[i + 1]; }
for int i in [0:15] { rz(-phi / 2 + pi) q[i]; }
";
    let (template, params) = openqasm::parse_parametric(source).expect("parse");
    assert!(params.links().iter().all(|link| link.scale != 1.0));
    let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
    assert!(prepared.reuses_fusion_plan());
    for (theta, phi) in [(0.3, 1.1), (-1.7, 0.25), (2.9, -4.0)] {
        let mut reference = Circuit::new(16, 0);
        for i in 0..16 {
            reference.add_gate(Gate::Ry(2.0 * theta + i as f64 * 0.125), &[i]);
        }
        for i in 0..15 {
            reference.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for i in 0..16 {
            reference.add_gate(Gate::Rz(-phi / 2.0 + std::f64::consts::PI), &[i]);
        }
        let expected = fuse_circuit(&reference, true).into_owned();
        let got = prepared.bind_fused(&[theta, phi]).unwrap();
        assert_eq!(got.instructions.len(), expected.instructions.len());
        for (at, (a, b)) in got
            .instructions
            .iter()
            .zip(&expected.instructions)
            .enumerate()
        {
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "instruction {at}");
        }
    }
}

#[test]
fn input_angles_the_binding_surface_cannot_carry_reject_by_name() {
    const PROLOGUE: &str =
        "OPENQASM 3.0;\ninput float t;\ninput float u;\nqubit[3] q;\nbit[3] c;\n";
    for (label, body, needle) in [
        ("function of an input", "rx(sin(t)) q[0];", "not affine"),
        ("product of inputs", "rx(t * t) q[0];", "not affine"),
        (
            "two inputs on a gate built as one matrix",
            "xx_plus_yy(t, u) q[0], q[1];",
            "two `input` parameters",
        ),
        ("inverted gate", "inv @ rx(t) q[0];", "modifier"),
        ("powered gate", "pow(2) @ U(t, u, 0) q[0];", "modifier"),
        (
            "two controls",
            "ctrl(2) @ rx(t) q[0], q[1], q[2];",
            "modifier",
        ),
        (
            "control on a gate with no controlled form",
            "ctrl @ rzz(t) q[0], q[1], q[2];",
            "modifier",
        ),
        (
            "global phase under two controls",
            "ctrl(2) @ gphase(t) q[0], q[1];",
            "more than one control",
        ),
        (
            "gate carrying no angle",
            "ms(t, 0, 0) q[0], q[1];",
            "`input` on",
        ),
        (
            "lowered gate",
            "xx_plus_yy(t, 0.2) q[0], q[1];",
            "`input` on",
        ),
        (
            "guarded region",
            "if (c[0]) { rx(t) q[0]; }",
            "guarded region",
        ),
        (
            "guarded statement",
            "if (c[0]) rx(t) q[0];",
            "guarded region",
        ),
        ("classical value", "float x = t;", "input `t`"),
        ("loop bound", "for int i in [0:t] { h q[0]; }", "input `t`"),
        (
            "modified def",
            "def d(qubit a, float w) { rx(w) a; }\ninv @ d(q[0], t);",
            "modifier",
        ),
        (
            "rounding def parameter",
            "def d(qubit a, float[32] w) { rx(w) a; }\nd(q[0], t);",
            "rounds it",
        ),
        (
            "gate parameter outside an angle",
            "gate g(a) x { pow(a) @ x x; }\ng(t) q[0];",
            "which an `input` reaches",
        ),
    ] {
        let err = openqasm::parse_parametric(&format!("{PROLOGUE}{body}"))
            .err()
            .unwrap_or_else(|| panic!("{label}: expected a rejection"));
        unsupported(err, needle, label);
    }
}

/// Parse `body` with each `{name}` an input, and again with each a literal.
fn parametric_and_literal(body: &str, values: &[f64; 3]) -> ((Circuit, Parameters), Circuit) {
    const NAMES: [&str; 3] = ["t", "u", "v"];
    let mut parametric = body.to_string();
    let mut literal = body.to_string();
    for (name, value) in NAMES.iter().zip(values) {
        parametric = parametric.replace(&format!("{{{name}}}"), name);
        literal = literal.replace(&format!("{{{name}}}"), &format!("({value:?})"));
    }
    let inputs = "input float[64] t;\ninput float[64] u;\ninput angle v;\n";
    let wrap = |inputs: &str, body: &str| format!("OPENQASM 3.0;\n{inputs}qubit[3] q;\n{body}\n");
    (
        openqasm::parse_parametric(&wrap(inputs, &parametric)).expect("parametric"),
        openqasm::parse(&wrap("", &literal)).expect("literal"),
    )
}

#[test]
fn inputs_bind_through_gates_that_lower_to_rotations() {
    for (label, body) in [
        ("U", "U({t}, {u}, {v}) q[0];"),
        (
            "u3, u2 and u1",
            "u3({t}, {u}, {v}) q[1]; u2({u}, {v}) q[0]; u1({t}) q[2];",
        ),
        ("U with a constant argument", "h q; U({t}, 0, 0.5) q[1];"),
        ("broadcast U", "U({t}, {u}, {v}) q;"),
        (
            "controlled phase",
            "h q; cp({t}) q[0], q[1]; cphase(-{u} / 2 + 0.3) q[2], q[0];",
        ),
        (
            "controlled rotations",
            "h q; crz({t}) q[0], q[1]; crx({u}) q[1], q[2]; cry({v}) q[2], q[0];",
        ),
        ("cu", "h q; cu({t}, {u}, {v}, 0.7) q[0], q[1];"),
        (
            "cu with a bound phase",
            "h q; cu(0.4, {u}, 0, {t}) q[2], q[0];",
        ),
        ("r", "r({t}, {u}) q[0]; prx(0.3, {v}) q[1];"),
        (
            "one control on a rotation",
            "h q; ctrl @ rx({t}) q[0], q[1]; negctrl @ rz({u}) q[1], q[2]; \
             ctrl @ ry({v}) q[2], q[1]; ctrl @ p({v}) q[0], q[2]; ctrl @ U({t}, {u}, {v}) q[2], q[0];",
        ),
        (
            "global phase",
            "h q; gphase({t}); ctrl @ gphase({u}) q[1]; inv @ gphase({v}); \
             pow(2) @ negctrl @ gphase({t}) q[0];",
        ),
        (
            "gate body",
            "gate g(a, b) x, y { U(a, b, a / 2) x; cp(b) x, y; gphase(a); }\n\
             h q; g({t}, {u}) q[0], q[1]; g(0.2, {v}) q[1], q[2];",
        ),
        (
            "def body",
            "def d(qubit a, qubit b, float w) { crz(w) a, b; U(w, 0.1, w) b; }\n\
             h q; d(q[0], q[1], 3 * {t});",
        ),
    ] {
        for values in [[0.0, 0.0, 0.0], [0.41, 1.27, -0.6], [-2.3, 5.9, 3.3]] {
            let ((template, params), literal) = parametric_and_literal(body, &values);
            assert_eq!(params.num_slots(), 3, "{label}");
            let bound = params.bind(&template, &values).expect("bind");
            assert_same_unitary(&bound, &literal, label);
        }
    }
}

#[test]
fn an_angle_over_several_inputs_binds_as_the_sum_written() {
    for (label, body) in [
        ("rotation", "rx({t} + {u}) q[0];"),
        (
            "pair rotation",
            "h q; rzz({t} - 2 * {u} + {v} / 3 + 0.3) q[0], q[1];",
        ),
        (
            "Pauli rotation",
            "h q; rxyz(({t} + {v}) / 2) q[0], q[1], q[2];",
        ),
        ("U", "U({t} + {u}, {u} - {v}, 2 * {v} + {t} - 0.1) q[0];"),
        (
            "controlled phase",
            "h q; cp({t} + {u}) q[0], q[1]; crz({u} - {v}) q[1], q[2];",
        ),
        ("cu", "h q; cu({t}, {u}, {v}, {t} + {u}) q[0], q[1];"),
        ("global phase", "h q; gphase({t} - {u});"),
        (
            "gate body",
            "gate g(a, b) x { rz(a + b) x; ry(a - b) x; }\nh q; g({t} + {v}, {u}) q[0];",
        ),
        ("terms that cancel", "rx({t} + {u} - {t}) q[0];"),
    ] {
        for values in [[0.0, 0.0, 0.0], [0.41, 1.27, -0.6], [-2.3, 5.9, 3.3]] {
            let ((template, params), literal) = parametric_and_literal(body, &values);
            let bound = params.bind(&template, &values).expect("bind");
            assert_same_unitary(&bound, &literal, label);
        }
    }
}

#[test]
fn an_angle_over_several_inputs_differentiates_in_each() {
    let source = "OPENQASM 3.0;
input float t;
input float u;
qubit[2] q;
h q;
U(t + 2 * u, u, t) q[0];
cp(t - u) q[0], q[1];
rx(u / 2 + t) q[1];
";
    let (template, params) = openqasm::parse_parametric(source).expect("parse");
    let obs = PauliObservable::from_terms([
        (1.0, vec![PauliTerm::z(0), PauliTerm::x(1)]),
        (-0.4, vec![PauliTerm::y(0)]),
    ])
    .unwrap();
    let energy = |values: [f64; 2]| {
        let circuit = params.bind(&template, &values).unwrap();
        let observables: Vec<Vec<PauliTerm>> = obs.terms().iter().map(|(_, p)| p.clone()).collect();
        let per_term = run_expectation_values(&circuit, &observables, SEED).unwrap();
        obs.terms()
            .iter()
            .zip(per_term)
            .map(|((c, _), v)| c * v)
            .sum::<f64>()
    };
    let at = [0.37, -1.1];
    let eps = 1e-5;
    let bound = params.bind(&template, &at).unwrap();
    let adjoint = run_expectation_gradient(&bound, &obs, &params, SEED).unwrap();
    let shift = run_expectation_gradient_shift(&bound, &obs, &params, SEED).unwrap();
    for slot in 0..2 {
        let (mut up, mut down) = (at, at);
        up[slot] += eps;
        down[slot] -= eps;
        let expected = (energy(up) - energy(down)) / (2.0 * eps);
        for (label, gradient) in [("adjoint", &adjoint.gradient), ("shift", &shift.gradient)] {
            assert!(
                (gradient[slot] - expected).abs() < 1e-6,
                "{label} slot {slot}: {} vs finite difference {expected}",
                gradient[slot]
            );
        }
    }
}

// Sixteen qubits so fusion runs every pass. The template holds each angle at an
// input of zero, where the lowered gates are identities and capture declines,
// so the plan is taken from a template bound at a generic point.
#[test]
fn lowered_inputs_rebind_through_the_fusion_plan() {
    let body = |a: &str, b: &str, c: &str| {
        format!(
            "qubit[16] q;
for int i in [0:15] {{ U({a} + i * 0.1, {b}, {c} - {a}) q[i]; }}
for int i in [0:14] {{ cp({b} * (i + 1) / 4) q[i], q[i + 1]; }}
for int i in [0:14] {{ crz({c} + {b}) q[i + 1], q[i]; }}
for int i in [0:15] {{ U({c}, {a}, {b}) q[i]; }}
"
        )
    };
    let source = format!(
        "OPENQASM 3.0;\ninput float[64] a;\ninput float[64] b;\ninput float[64] c;\n{}",
        body("a", "b", "c")
    );
    let (template, params) = openqasm::parse_parametric(&source).expect("parse");
    let seeded = params.bind(&template, &[0.9, -0.4, 1.6]).unwrap();
    let mut prepared = PreparedCircuit::new(seeded, params.clone()).unwrap();
    assert!(prepared.reuses_fusion_plan());
    let mut refusing = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
    for values in [
        [0.3, 1.1, -0.7],
        [-1.7, 0.25, 2.2],
        [2.9, -4.0, 0.0],
        [0.3, 1.1, -0.7],
    ] {
        let [a, b, c] = values.map(|value| format!("({value:?})"));
        let literal = openqasm::parse(&format!("OPENQASM 3.0;\n{}", body(&a, &b, &c))).unwrap();
        let expected = common::run_and_state(&literal);
        let rebound = common::run_and_state(prepared.bind_fused(&values).unwrap());
        common::assert_amps_close(&rebound, &expected, SV_EPS, "bind_fused");
        let fresh = common::run_and_state(refusing.bind_fused(&values).unwrap());
        common::assert_amps_close(&fresh, &expected, SV_EPS, "fusion per binding");
    }
}

#[test]
fn supplied_inputs_fold_where_a_constant_belongs() {
    let source = "OPENQASM 3.0;
input int[8] n;
input bool flip;
input float theta;
input angle phi;
input uint[2] pick;
input bit[2] mask;
qubit[4] q;
for int i in [0:n] { h q[i]; }
if (flip) x q[3];
rx(theta) q[pick];
rz(phi) q[0];
";
    let (template, params) = openqasm::parse_parametric_with_inputs(
        source,
        &[
            ("n", ClassicalValue::Int(2)),
            ("flip", ClassicalValue::Bool(true)),
            ("theta", ClassicalValue::Float(0.3)),
            ("pick", ClassicalValue::Int(1)),
            ("mask", ClassicalValue::Int(0)),
        ],
    )
    .expect("parse");
    assert_eq!(params.num_slots(), 1);
    assert_eq!(params.name_of(0), Some("phi"));
    let bound = params.bind(&template, &[0.8]).unwrap();
    let literal = openqasm::parse(
        "OPENQASM 3.0;\nqubit[4] q;\nh q[0];\nh q[1];\nh q[2];\nx q[3];\nrx(0.3) q[1];\nrz(0.8) q[0];\n",
    )
    .unwrap();
    assert_same_unitary(&bound, &literal, "supplied inputs");
}

#[test]
fn input_values_that_cannot_stand_are_rejected_by_name() {
    let source = "OPENQASM 3.0;\ninput int[8] n;\nqubit[1] q;\n";
    invalid(
        openqasm::parse_parametric(source).unwrap_err(),
        "input `n`",
        "unsupplied int",
    );
    invalid(
        openqasm::parse_parametric_with_inputs(
            source,
            &[("n", ClassicalValue::Int(1)), ("m", ClassicalValue::Int(1))],
        )
        .unwrap_err(),
        "`m`",
        "unknown name",
    );
    invalid(
        openqasm::parse_parametric_with_inputs(
            source,
            &[("n", ClassicalValue::Int(1)), ("n", ClassicalValue::Int(2))],
        )
        .unwrap_err(),
        "supplied twice",
        "duplicate name",
    );
    assert!(matches!(
        openqasm::parse_parametric_with_inputs(source, &[("n", ClassicalValue::Float(1.5))]),
        Err(PrismError::Parse { .. })
    ));
    unsupported(
        openqasm::parse_parametric_with_inputs(
            "OPENQASM 3.0;\ninput bit[2] mask;\nqubit[1] q;\n",
            &[("mask", ClassicalValue::Int(1))],
        )
        .unwrap_err(),
        "set bit",
        "set bit outside a dynamic program",
    );
}

#[test]
fn a_dynamic_program_reads_supplied_inputs_and_declines_unsupplied_ones() {
    let source = "OPENQASM 3.0;
input bit[2] mask;
input int[4] reps;
input float theta;
qubit[2] q;
bit[2] c;
if (mask[1]) x q[1];
for int i in [1:reps] { rx(theta) q[0]; }
c = measure q;
";
    let program = openqasm::parse_dynamic_with_inputs(
        source,
        &[
            ("mask", ClassicalValue::Int(2)),
            ("reps", ClassicalValue::Int(2)),
            ("theta", ClassicalValue::Float(std::f64::consts::FRAC_PI_2)),
        ],
    )
    .expect("parse");
    let shots = simulate_program(&program)
        .seed(SEED)
        .shots(64)
        .unwrap()
        .shots;
    // `mask` holds bits 0 and 1, so `c` reads bits 2 and 3.
    assert!(shots.iter().all(|shot| shot == &[false, true, true, true]));

    unsupported(
        openqasm::parse_dynamic_with_inputs(
            source,
            &[
                ("mask", ClassicalValue::Int(2)),
                ("reps", ClassicalValue::Int(2)),
            ],
        )
        .unwrap_err(),
        "input `theta`",
        "unsupplied angle",
    );
}

#[test]
fn a_link_with_a_zero_scale_fails_validation() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::Rx(0.0), &[0]);
    let mut params = Parameters::new(1);
    params.link_affine(0, 0, 0.0, 1.0);
    assert!(matches!(
        params.bind(&circuit, &[0.5]),
        Err(PrismError::InvalidParameter { .. })
    ));
}

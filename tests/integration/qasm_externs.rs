//! OpenQASM `extern` declarations bound to registered functions: parse-time
//! folds, runtime calls in a dynamic program, and the declines.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::common;

use common::{SEED, SV_EPS};
use prism_q::circuit::dynamic::{
    BinaryOp, ClassicalExpr, ClassicalType, ClassicalValue, ExternCall,
};
use prism_q::circuit::openqasm::{self, Externs};
use prism_q::{Circuit, DynamicProgramBuilder, Gate, PrismError, simulate_program};

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

fn counted(
    calls: &Arc<AtomicUsize>,
    function: impl Fn(&[ClassicalValue]) -> ClassicalValue + Send + Sync + 'static,
) -> impl Fn(&[ClassicalValue]) -> prism_q::Result<ClassicalValue> + Send + Sync + 'static {
    let calls = Arc::clone(calls);
    move |args| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(function(args))
    }
}

fn negate(args: &[ClassicalValue]) -> prism_q::Result<ClassicalValue> {
    match args {
        [ClassicalValue::Bool(bit)] => Ok(ClassicalValue::Bool(!bit)),
        other => panic!("`negate` takes one bool, got {other:?}"),
    }
}

#[test]
fn a_call_on_constants_folds_at_parse_time() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut externs = Externs::new();
    externs
        .register(
            "scale",
            counted(&calls, |args| {
                ClassicalValue::Float(args[0].to_float() * args[1].to_float())
            }),
        )
        .register(
            "wrap",
            counted(&calls, |args| ClassicalValue::Int(args[0].to_int())),
        );
    let source = "OPENQASM 3.0;
extern scale(int[8], float) -> float;
extern wrap(int) -> uint[2];
qubit[2] q;
def twice(int n) -> int { return 2 * wrap(n); }
rx(scale(257, pi / 4)) q[0];
int m = twice(5);
for int i in [1:m] { rz(0.25) q[1]; }
int[8] k;
k = wrap(7);
scale(k, 1.0);
";
    let circuit = openqasm::parse_with_externs(source, &externs).expect("parse");
    // `int[8]` stores 257 as 1, and `uint[2]` stores 5 as 1.
    let want =
        openqasm::parse("OPENQASM 3.0; qubit[2] q; rx(pi / 4) q[0]; rz(0.25) q[1]; rz(0.25) q[1];")
            .unwrap();
    assert_same_unitary(&circuit, &want, "folded");
    assert_eq!(calls.load(Ordering::Relaxed), 4);

    let (template, params) = openqasm::parse_parametric_with_externs(
        "OPENQASM 3.0; extern scale(int[8], float) -> float; input float t; input int n;
         qubit[1] q; rx(t + scale(n, 0.5)) q[0];",
        &[("n", ClassicalValue::Int(3))],
        &externs,
    )
    .expect("parametric parse");
    assert_eq!(params.num_slots(), 1);
    let bound = params.bind(&template, &[0.25]).unwrap();
    let want = openqasm::parse("OPENQASM 3.0; qubit[1] q; rx(1.75) q[0];").unwrap();
    assert_same_unitary(&bound, &want, "parametric");
}

#[test]
fn a_call_on_a_measured_bit_runs_per_shot_and_feeds_a_condition() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut externs = Externs::new();
    externs
        .register(
            "same",
            counted(&calls, |args| ClassicalValue::Bool(args[0].truthy())),
        )
        .register("negate", negate)
        .register("half", |args| {
            Ok(ClassicalValue::Float(args[0].to_float() / 2.0))
        });
    let source = "OPENQASM 3.0;
extern same(bit) -> bool;
extern negate(bool) -> bit;
extern half(float) -> float;
qubit[4] q;
bit[4] c;
h q[0];
c[0] = measure q[0];
if (same(c[0])) x q[1];
c[1] = measure q[1];
c[2] = negate(c[0]);
float theta = 2 * pi * c[0];
rx(half(theta)) q[3];
c[3] = measure q[3];
";
    let program = openqasm::parse_dynamic_with_externs(source, &[], &externs).expect("parse");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "nothing runs at parse time"
    );
    assert!(program.static_circuit().is_none());
    let shots = simulate_program(&program)
        .seed(SEED)
        .shots(200)
        .unwrap()
        .shots;
    assert!(calls.load(Ordering::Relaxed) > 0);
    for shot in &shots {
        assert_eq!(shot[1], shot[0], "the condition read the call");
        assert_eq!(shot[2], !shot[0], "the bit write read the call");
        assert_eq!(shot[3], shot[0], "the rotation angle read the call");
    }
    assert!(shots.iter().any(|shot| shot[0]) && shots.iter().any(|shot| !shot[0]));
}

#[test]
fn a_result_of_the_wrong_kind_is_invalid() {
    let mut externs = Externs::new();
    externs.register("f", |_| Ok(ClassicalValue::Float(1.5)));
    invalid(
        openqasm::parse_with_externs(
            "OPENQASM 3.0; extern f(int) -> int; qubit[1] q; rx(f(1)) q[0];",
            &externs,
        )
        .unwrap_err(),
        "declared `int[64]`",
        "parse-time call",
    );

    let program = openqasm::parse_dynamic_with_externs(
        "OPENQASM 3.0; extern f(bit) -> bool; qubit[1] q; bit[1] c;
         h q[0]; c[0] = measure q[0]; if (f(c[0])) x q[0];",
        &[],
        &externs,
    )
    .expect("the runtime call parses");
    invalid(
        simulate_program(&program).seed(SEED).shots(4).unwrap_err(),
        "declared `bool`",
        "runtime call",
    );
}

#[test]
fn an_error_from_the_function_reaches_the_caller() {
    let mut externs = Externs::new();
    externs.register("f", |_| {
        Err(PrismError::InvalidParameter {
            message: "lookup table has no entry".to_string(),
        })
    });
    invalid(
        openqasm::parse_with_externs(
            "OPENQASM 3.0; extern f(int) -> float; qubit[1] q; rx(f(1)) q[0];",
            &externs,
        )
        .unwrap_err(),
        "lookup table",
        "callback error",
    );
}

#[test]
fn entry_points_without_externs_decline_every_declaration() {
    let source = "OPENQASM 3.0; extern f(int) -> float; qubit[1] q;";
    let needle = "_with_externs";
    unsupported(openqasm::parse(source).unwrap_err(), needle, "parse");
    unsupported(
        openqasm::parse_parametric_with_inputs(source, &[]).unwrap_err(),
        needle,
        "parse_parametric_with_inputs",
    );
    unsupported(
        openqasm::parse_dynamic(source).unwrap_err(),
        needle,
        "parse_dynamic",
    );
    unsupported(
        openqasm::parse_dynamic_with_inputs(source, &[]).unwrap_err(),
        needle,
        "parse_dynamic_with_inputs",
    );
}

#[test]
fn an_integer_result_for_a_float_is_promoted() {
    let mut externs = Externs::new();
    externs
        .register("three", |_| Ok(ClassicalValue::Int(3)))
        .register("yes", |_| Ok(ClassicalValue::Bool(true)));
    let circuit = openqasm::parse_with_externs(
        "OPENQASM 3.0; extern three(int) -> float[32]; qubit[1] q; rx(three(0)) q[0];",
        &externs,
    )
    .expect("parse");
    let want = openqasm::parse("OPENQASM 3.0; qubit[1] q; rx(3.0) q[0];").unwrap();
    assert_same_unitary(&circuit, &want, "promoted");

    let mut angles = Externs::new();
    angles.register("turn", |_| Ok(ClassicalValue::Int(7)));
    let circuit = openqasm::parse_with_externs(
        "OPENQASM 3.0; extern turn(int) -> angle; qubit[1] q; rz(turn(0)) q[0];",
        &angles,
    )
    .expect("parse");
    let want = openqasm::parse(&format!(
        "OPENQASM 3.0; qubit[1] q; rz({}) q[0];",
        7.0f64.rem_euclid(std::f64::consts::TAU)
    ))
    .unwrap();
    assert_same_unitary(&circuit, &want, "promoted angle");

    invalid(
        openqasm::parse_with_externs(
            "OPENQASM 3.0; extern yes(int) -> float; qubit[1] q; rx(yes(0)) q[0];",
            &externs,
        )
        .unwrap_err(),
        "declared `float[64]`",
        "a bool stays strict",
    );
}

#[test]
fn a_declaration_with_no_registered_function_is_invalid() {
    let source = "OPENQASM 3.0; extern f(int) -> float; qubit[1] q;";
    let mut externs = Externs::new();
    externs.register("g", |_| Ok(ClassicalValue::Float(0.0)));
    invalid(
        openqasm::parse_with_externs(source, &externs).unwrap_err(),
        "extern `f`",
        "another name registered",
    );
    assert!(
        openqasm::parse_with_externs("OPENQASM 3.0; qubit[1] q;", &externs).is_ok(),
        "an unused registration is fine"
    );
}

#[test]
fn unsupported_extern_forms_decline_by_name() {
    let mut externs = Externs::new();
    for name in ["f", "g"] {
        externs.register(name, |_| Ok(ClassicalValue::Float(0.0)));
    }
    let parse = |body: &str| {
        openqasm::parse_with_externs(
            &format!("OPENQASM 3.0; qubit[1] q; bit[1] c; {body}"),
            &externs,
        )
        .unwrap_err()
    };
    unsupported(parse("extern f(int);"), "no result type", "no result");
    unsupported(
        parse("extern f(complex[float]) -> float;"),
        "complex",
        "complex",
    );
    unsupported(
        parse("extern f(bit[4]) -> float;"),
        "bit[4]",
        "bit register",
    );
    unsupported(
        parse("extern f(int) -> angle[8];"),
        "angle[8]",
        "fixed angle",
    );
    unsupported(
        parse("extern f(duration) -> float;"),
        "duration",
        "duration",
    );
    unsupported(
        parse("extern f(readonly array[int[8], 2]) -> float;"),
        "readonly",
        "array",
    );
    unsupported(
        parse("extern f(bit) -> bool; c[0] = measure q[0]; if (f(c[0])) x q[0];"),
        "",
        "measured argument under parse",
    );
    unsupported(
        openqasm::parse_parametric_with_externs(
            "OPENQASM 3.0; extern f(float) -> float; input float t; qubit[1] q; rx(f(t)) q[0];",
            &[],
            &externs,
        )
        .unwrap_err(),
        "",
        "unsupplied input argument",
    );

    assert!(matches!(
        parse("extern f(int) -> float; rx(f(1, 2)) q[0];"),
        PrismError::Parse { .. }
    ));
    assert!(matches!(
        parse("extern f(int) -> float; gate f a { x a; }"),
        PrismError::Parse { .. }
    ));
    assert!(matches!(
        parse("extern f(int) -> float; f(1) q[0];"),
        PrismError::Parse { .. }
    ));
}

#[test]
fn a_def_call_bounds_a_for_range() {
    let looped = |range: &str| {
        format!(
            "OPENQASM 3.0; def twice(int n) -> int {{ return 2 * n; }} qubit[1] q;
             for int i in {range} {{ rx(0.1) q[0]; }}"
        )
    };
    let want = openqasm::parse("OPENQASM 3.0; qubit[1] q; rx(1.0) q[0];").unwrap();
    for range in [
        "[1:twice(5)]",
        "[twice(0):twice(1):twice(9) + 1]",
        "{twice(1), 0, 1, 2, 3, 5, 6, 7, 8, 9}",
    ] {
        let source = looped(range);
        let circuit = openqasm::parse(&source).expect(range);
        assert_same_unitary(&circuit, &want, range);
        let program = openqasm::parse_dynamic(&source).expect(range);
        let circuit = program.static_circuit().expect("no runtime control flow");
        assert_same_unitary(circuit, &want, range);
    }

    let source = "OPENQASM 3.0; qubit[2] q; bit[1] c;
def prepare(qubit a) -> int { x a; return 3; }
for int i in [1:prepare(q[1])] { rx(0.25) q[0]; }";
    let want = openqasm::parse("OPENQASM 3.0; qubit[2] q; x q[1]; rx(0.75) q[0];").unwrap();
    assert_same_unitary(&openqasm::parse(source).unwrap(), &want, "def with gates");
}

#[test]
fn a_hand_built_extern_call_runs_per_shot() {
    let int8 = ClassicalType::Int { width: 8 };
    let mut builder = DynamicProgramBuilder::new(2, 2);
    builder.add_gate(Gate::H, &[0]).add_measure(0, 0);
    let call = ExternCall::new(
        "succ",
        vec![int8],
        int8,
        |args| Ok(ClassicalValue::Int(args[0].to_int() + 127)),
        vec![ClassicalExpr::Bit(0)],
    );
    assert_eq!(call.name(), "succ");
    assert_eq!(call.args(), &[ClassicalExpr::Bit(0)]);
    let call = ClassicalExpr::Extern(Box::new(call));
    assert_eq!(call, call.clone());
    let wraps = ClassicalExpr::binary(BinaryOp::Lt, call, ClassicalExpr::from(0i64));
    builder.begin_if(wraps);
    builder.add_gate(Gate::X, &[1]);
    builder.end().unwrap();
    builder.add_measure(1, 1);
    let program = builder.build().expect("build");
    let shots = simulate_program(&program)
        .seed(SEED)
        .shots(100)
        .unwrap()
        .shots;
    for shot in &shots {
        assert_eq!(shot[1], shot[0], "1 + 127 wraps to -128 in 8 bits");
    }
    assert!(shots.iter().any(|shot| shot[0]) && shots.iter().any(|shot| !shot[0]));
}

#[test]
fn a_hand_built_extern_call_is_checked_at_build() {
    let build = |params: Vec<ClassicalType>, args: Vec<ClassicalExpr>| {
        let mut builder = DynamicProgramBuilder::new(1, 1);
        let call = ExternCall::new("f", params, ClassicalType::Bool, |_| Ok(true.into()), args);
        builder.write_bits(&[0], ClassicalExpr::Extern(Box::new(call)));
        builder.build().unwrap_err()
    };
    invalid(
        build(vec![ClassicalType::Bool], vec![]),
        "takes 1 argument(s), got 0",
        "arity",
    );
    invalid(
        build(vec![ClassicalType::Int { width: 65 }], vec![0i64.into()]),
        "width 65",
        "width",
    );
    assert!(matches!(
        build(vec![ClassicalType::Bool], vec![ClassicalExpr::Bit(3)]),
        PrismError::InvalidClassicalBit { index: 3, .. }
    ));
}

//! Nesting, stack and expansion limits, most of them reached through `parse_dynamic`.

use super::*;

#[test]
fn deep_nesting_is_a_parse_error_not_a_stack_overflow() {
    assert_too_deep(parse(&nested_parens(3_000)));
    assert_too_deep(parse(&nested_ifs(3_000)));
    assert_too_deep(parse(&long_sum(10_000)));
    assert_too_deep(parse(&format!(
        "OPENQASM 3.0;\nqubit[1] q;\nrx({}1) q[0];",
        "-".repeat(3_000)
    )));
    assert_too_deep(parse(&format!(
        "OPENQASM 3.0;\nqubit[1] q;\nrx(1{}) q[0];",
        " ** 1".repeat(3_000)
    )));
    assert_too_deep(parse_dynamic(&nested_parens(3_000)));
}

// The deepest sources the limit admits parse and evaluate on a 1 MiB stack, the
// default for a Windows main thread.
#[test]
fn nesting_just_inside_the_limit_fits_a_small_stack() {
    let depth = MAX_NESTING - 8;
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            let c = parse(&nested_parens(depth)).unwrap();
            assert_eq!(c.gate_count(), 1);
            let source = nested_ifs(depth);
            let tokens = lexer::tokenize(&source).unwrap();
            crate::circuit::qasm::parser::parse_program(&tokens).unwrap();
            let c = parse(&long_sum(depth)).unwrap();
            let Instruction::Gate {
                gate: Gate::Rx(theta),
                ..
            } = &c.instructions[0]
            else {
                panic!("expected rx");
            };
            assert!((theta - 0.001 * (depth - 1) as f64).abs() < 1e-9);
            parse_dynamic(&nested_parens(depth)).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

// Each gate calls the one before it twice, so twenty definitions expand to
// 2^20 instructions though no single construct passes its own bound.
#[test]
fn doubling_gate_definitions_exhaust_the_expansion_budget() {
    let mut source = String::from("OPENQASM 3.0;\nqubit[1] q;\ngate g0 a { x a; }\n");
    for level in 1..=20 {
        let prev = level - 1;
        source.push_str(&format!("gate g{level} a {{ g{prev} a; g{prev} a; }}\n"));
    }
    source.push_str("g20 q[0];\n");
    assert_expands_past(parse_within(&source, 1 << 12));

    let small = source.replace("g20 q[0];", "g5 q[0];");
    assert_eq!(parse_within(&small, 1 << 12).unwrap().gate_count(), 32);
}

#[test]
fn nested_loops_exhaust_the_expansion_budget_even_when_empty() {
    let empty = "OPENQASM 3.0;\nqubit[1] q;\n\
                 for uint i in [0:999999] { for uint j in [0:999999] { } }";
    assert_expands_past(parse_within(empty, 1 << 12));
    let busy = "OPENQASM 3.0;\nqubit[1] q;\n\
                for uint i in [0:999] { for uint j in [0:999] { x q[0]; } }";
    assert_expands_past(parse_within(busy, 1 << 12));
    let pow = "OPENQASM 3.0;\nqubit[1] q;\ngate g a { x a; h a; }\npow(1000000) @ g q[0];";
    assert_expands_past(parse_within(pow, 1 << 12));
}

// Each `x` costs one unit and each call or loop pass one more for itself, and
// nothing is charged again by the level that collects it: four passes of
// 2 * (2 + 1) + 1 + 1 units, plus five top-level statements, make 37.
#[test]
fn the_expansion_budget_charges_work_once() {
    let source = "OPENQASM 3.0;
qubit[1] q;
gate g a { x a; x a; }
                  gate h2 a { g a; g a; }
for uint i in [0:3] { h2 q[0]; }";
    assert_eq!(parse_within(source, 37).unwrap().gate_count(), 16);
    assert_expands_past(parse_within(source, 36));
}

// A def body gets its own copy of the caller's values and constants, so each
// call pays a unit for every name it copies.
#[test]
fn a_def_call_charges_the_names_it_copies() {
    let calls = format!(
        "OPENQASM 3.0;
qubit[1] q;
def f(qubit a) {{ x a; }}
{}",
        "f(q[0]);
"
        .repeat(8)
    );
    assert_eq!(parse_within(&calls, 1 << 10).unwrap().gate_count(), 8);
    let constants: String = (0..100)
        .map(|i| {
            format!(
                "const int k{i} = {i};
"
            )
        })
        .collect();
    let crowded = calls.replace(
        "qubit[1] q;
",
        &format!(
            "qubit[1] q;
{constants}"
        ),
    );
    assert_expands_past(parse_within(&crowded, 1 << 10));
    assert_eq!(parse(&crowded).unwrap().gate_count(), 8);
}

#[test]
fn oversized_registers_are_parse_errors() {
    let wide = |decls: &str| {
        let err = parse(&format!("OPENQASM 3.0;\n{decls}")).unwrap_err();
        assert!(err.to_string().contains("declares more than"), "{err}");
    };
    wide("qubit[9223372036854775807] a;\nqubit[9223372036854775807] b;");
    wide("bit[9223372036854775807] a;\nbit[9223372036854775807] b;");
    wide("bit[16777217] c;");
    wide("qubit[16777216] a;\nqubit[1] b;");
    wide("x $18446744073709551615;");
    let err = parse_dynamic("OPENQASM 3.0;\nx $18446744073709551615;").unwrap_err();
    assert!(err.to_string().contains("declares more than"), "{err}");
}

// A step that would carry the next index past the integer range carries it
// past the end of the range too, so the walk stops there.
#[test]
fn a_range_step_past_the_integer_range_ends_the_range() {
    let c = parse("OPENQASM 3.0;\nqubit[4] q;\nx q[1:9223372036854775807:3];").unwrap();
    assert_eq!(c.gate_count(), 1);
    let c = parse(
        "OPENQASM 3.0;\nqubit[1] q;\n\
         for int i in [1:9223372036854775807:9223372036854775807] { x q[0]; }",
    )
    .unwrap();
    assert_eq!(c.gate_count(), 1);
}

#[test]
fn a_repeated_pow_is_bounded_before_it_allocates() {
    let err = parse("OPENQASM 3.0;\nqubit[2] q;\npow(1000000) @ pow(1000000) @ cx q[0], q[1];")
        .unwrap_err()
        .to_string();
    assert!(err.contains("expands past"), "{err}");
}

#[test]
fn a_runtime_rotation_is_charged_per_qubit() {
    let source = |passes: usize| {
        format!(
            "OPENQASM 3.0;\nqubit[100] q;\nbit[1] c;\nfloat t = 0;\n\
             c[0] = measure q[0];\nif (c[0]) t = pi;\n\
             for int i in [1:{passes}] {{ rx(t) q; }}"
        )
    };
    assert!(super::super::dynamic::parse_dynamic_within(&source(5), &[], None, 1000).is_ok());
    let err = super::super::dynamic::parse_dynamic_within(&source(20), &[], None, 1000)
        .unwrap_err()
        .to_string();
    assert!(err.contains("expands past"), "{err}");
}

#[test]
fn nested_fors_with_measured_exits_walk_in_linear_time() {
    let nested = |depth: usize| {
        let mut source = String::from(
            "OPENQASM 3.0;
qubit[2] q;
bit[2] c;
c[0] = measure q[0];
",
        );
        for level in 0..depth {
            source += &format!("for int i{level} in [0:0] {{ if (c[0]) {{ break; }} ");
        }
        source += "x q[1];";
        source += &" }".repeat(depth);
        source
    };
    let walked = |depth: usize| {
        super::super::runtime_analysis::WALKED.with(|walked| walked.set(0));
        parse_dynamic(&nested(depth)).unwrap();
        super::super::runtime_analysis::WALKED.with(Cell::get)
    };
    let (shallow, deep) = (walked(8), walked(16));
    assert!(
        deep <= shallow * 3,
        "{shallow} statements walked at depth 8, {deep} at 16"
    );
    let (shallow, deep) = (walked(8), walked(120));
    assert!(
        deep * 8 <= shallow * 120 * 2,
        "{shallow} walked at 8, {deep} at 120"
    );
}

#[test]
fn a_pass_of_blocks_costs_more_budget_than_a_pass_of_instructions() {
    let head = "OPENQASM 3.0;
qubit[2] q;
bit[2] c;
c[0] = measure q[0];
";
    let needs = |body: &str| {
        let source = format!("{head}for int j in [0:99] {{ if (c[0]) {{ {body} }} }}");
        let fits = |budget| {
            super::super::dynamic::parse_dynamic_within(&source, &[], None, budget).is_ok()
        };
        let (mut low, mut high) = (1, 1 << 16);
        while low < high {
            let mid = (low + high) / 2;
            if fits(mid) { high = mid } else { low = mid + 1 }
        }
        low
    };
    let (plain, exits) = (needs("x q[0];"), needs("break;"));
    assert!(
        exits > 2 * plain,
        "a pass of blocks needs {exits}, one of instructions {plain}"
    );
}

#[test]
fn nested_boxes_just_inside_the_limit_fit_a_small_stack() {
    let depth = MAX_NESTING - 8;
    let source = format!(
        "OPENQASM 3.0;\nqubit[1] q;\n{}x q[0];{}",
        "box { ".repeat(depth),
        " }".repeat(depth)
    );
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            assert_eq!(parse(&source).unwrap().gate_count(), 1);
            parse_dynamic(&source).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

// A 64 KiB stack holds a few levels at most, so every walk below has to move
// onto heap segments to finish.
#[test]
fn sources_inside_the_limit_outgrow_a_tiny_stack() {
    let depth = MAX_NESTING - 8;
    let header = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\n";
    let sources = [
        nested_parens(depth),
        long_sum(depth),
        format!(
            "{header}rx({}1{}) q[0];",
            "sin(".repeat(depth),
            ")".repeat(depth)
        ),
        format!("{header}rx({}1) q[0];", "-".repeat(depth)),
        format!(
            "{header}const array[int, 1] a = {{0}};\nx q[{}0{}];",
            "a[".repeat(depth),
            "]".repeat(depth)
        ),
        format!(
            "{header}{}x q[0];{}",
            "box { ".repeat(depth),
            " }".repeat(depth)
        ),
        format!(
            "{header}{}x q[0];{}",
            (0..depth)
                .map(|i| format!("for int i{i} in [0:0] {{ "))
                .collect::<String>(),
            " }".repeat(depth)
        ),
        format!(
            "{header}c[0] = measure q[0];\nint x = {}c[0];\nif (x == 1) x q[1];",
            "-".repeat(depth)
        ),
    ];
    std::thread::Builder::new()
        .stack_size(64 << 10)
        .spawn(move || {
            for source in &sources {
                parse_dynamic(source).unwrap();
                if !source.contains("measure") {
                    parse(source).unwrap();
                    parse_braket(source).unwrap();
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn boxes_past_the_nesting_limit_are_a_parse_error() {
    let source = format!(
        "OPENQASM 3.0;\nqubit[1] q;\n{}x q[0];{}",
        "box { ".repeat(3_000),
        " }".repeat(3_000)
    );
    assert_too_deep(parse(&source));
    assert_too_deep(parse_dynamic(&source));
}

#[test]
fn gate_calls_through_deep_boxes_fit_a_small_stack() {
    let depth = MAX_NESTING - 8;
    let mut source = String::from("OPENQASM 3.0;\nqubit[1] q;\ngate g0 a { x a; }\n");
    for level in 1..32 {
        source.push_str(&format!(
            "gate g{level} a {{ {}g{} a;{} }}\n",
            "box { ".repeat(depth),
            level - 1,
            " }".repeat(depth)
        ));
    }
    source.push_str("g31 q[0];\n");
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            let result = parse(&source);
            assert!(
                result.as_ref().is_ok_and(|c| c.gate_count() == 1)
                    || result
                        .as_ref()
                        .is_err_and(|e| e.to_string().contains("nests deeper")),
                "{result:?}"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

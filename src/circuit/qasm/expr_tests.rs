use std::collections::HashMap;

use super::super::lexer::tokenize;
use super::*;

fn value_with(source: &str, vars: Option<&HashMap<&str, f64>>) -> Result<f64> {
    let tokens = tokenize(source)?;
    let mut stream = Stream::new(&tokens);
    let expr = parse(&mut stream)?;
    if !stream.at_end() {
        return Err(stream.expected("the end of the expression"));
    }
    eval(&expr, 1, vars)
}

fn value(source: &str) -> f64 {
    value_with(source, None).unwrap_or_else(|e| panic!("`{source}`: {e}"))
}

fn error(source: &str) -> String {
    match value_with(source, None) {
        Err(PrismError::Parse { message, .. }) => message,
        other => panic!("`{source}` gave {other:?}"),
    }
}

#[test]
fn precedence_matches_the_language() {
    for (source, expected) in [
        ("1 + 2 * 3", 7.0),
        ("(1 + 2) * 3", 9.0),
        ("2 ** 3 ** 2", 512.0),
        // `**` binds tighter than unary minus, so this is the negation of a
        // power and not the square of a negative.
        ("-2 ** 2", -4.0),
        ("2 ** -1", 0.5),
        ("2 ** -1 % 3", 0.5),
        ("7 % 3", 1.0),
        ("-  -3", 3.0),
        ("+3", 3.0),
    ] {
        assert!(
            (value(source) - expected).abs() < 1e-12,
            "`{source}` gave {}",
            value(source)
        );
    }
}

#[test]
fn literals_read_in_every_radix() {
    for (source, expected) in [
        ("0xff", 255.0),
        ("0XFF", 255.0),
        ("0b1010", 10.0),
        ("0o17", 15.0),
        ("1_000", 1000.0),
        ("0x_f_f", 255.0),
        (".5", 0.5),
        ("1e3", 1000.0),
        ("1.5e-2", 0.015),
    ] {
        assert_eq!(value(source), expected, "`{source}`");
    }
}

#[test]
fn constants_and_builtins_resolve() {
    assert!((value("pi") - std::f64::consts::PI).abs() < 1e-12);
    assert!((value("\u{3c0}") - std::f64::consts::PI).abs() < 1e-12);
    assert!((value("tau") - std::f64::consts::TAU).abs() < 1e-12);
    assert!((value("euler") - std::f64::consts::E).abs() < 1e-12);
    assert_eq!(value("true"), 1.0);
    assert_eq!(value("false"), 0.0);
    assert_eq!(value("mod(7, 3)"), 1.0);
    assert_eq!(value("pow(2, 10)"), 1024.0);
    assert_eq!(value("ceiling(1.2)"), 2.0);
    assert_eq!(value("popcount(0b1011)"), 3.0);
    assert!((value("arcsin(1)") - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
}

#[test]
fn a_variable_table_is_read_when_one_is_given() {
    let mut vars = HashMap::new();
    vars.insert("theta", 0.25);
    assert_eq!(value_with("theta * 4", Some(&vars)).unwrap(), 1.0);
    assert!(value_with("theta", None).is_err());
}

// The same tree evaluates against different bindings, which is what lets a
// loop body be parsed once and run per pass.
#[test]
fn one_tree_evaluates_against_many_bindings() {
    let tokens = tokenize("k * 2 + 1").unwrap();
    let mut stream = Stream::new(&tokens);
    let expr = parse(&mut stream).unwrap();
    for k in 0..4i64 {
        let mut vars = HashMap::new();
        vars.insert("k", k as f64);
        assert_eq!(eval(&expr, 1, Some(&vars)).unwrap(), (k * 2 + 1) as f64);
    }
}

#[test]
fn failures_name_what_went_wrong() {
    assert!(error("1 / 0").contains("division by zero"));
    assert!(error("1 % 0").contains("modulo by zero"));
    assert!(error("nope").contains("unknown identifier `nope`"));
    assert!(error("nope(1)").contains("unknown function `nope`"));
    assert!(error("sin(1, 2)").contains("takes 1 argument(s), got 2"));
    assert!(error("popcount(-1)").contains("non-negative integer"));
    assert!(error("(1").contains("unmatched `(`"));
    assert!(error("sin(1").contains("unmatched `(` after function `sin`"));
    assert!(error("1 +").contains("unexpected end of expression"));
    assert!(error("log(0)").contains("non-finite"));
}

#[test]
fn an_expression_reports_the_names_it_reads() {
    let tokens = tokenize("2 * theta + sin(phi)").unwrap();
    let mut stream = Stream::new(&tokens);
    let expr = parse(&mut stream).unwrap();
    assert!(expr.mentions("theta"));
    assert!(expr.mentions("phi"));
    assert!(!expr.mentions("psi"));
    assert!(expr.as_ident().is_none());

    let tokens = tokenize("theta").unwrap();
    let mut stream = Stream::new(&tokens);
    assert_eq!(parse(&mut stream).unwrap().as_ident(), Some("theta"));
}

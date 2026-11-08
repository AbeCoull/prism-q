use super::*;

fn kinds(source: &str) -> Vec<Kind> {
    tokenize(source)
        .unwrap_or_else(|e| panic!("`{source}`: {e}"))
        .iter()
        .map(|token| token.kind)
        .filter(|kind| *kind != Kind::Eof)
        .collect()
}

fn texts(source: &str) -> Vec<String> {
    tokenize(source)
        .unwrap_or_else(|e| panic!("`{source}`: {e}"))
        .iter()
        .filter(|token| token.kind != Kind::Eof)
        .map(|token| token.text.to_string())
        .collect()
}

// An operator that is a prefix of a longer one must not win: `**` is a power
// and `++` joins aliases, and reading either as two tokens changes the program.
#[test]
fn operators_take_their_longest_spelling() {
    for (source, expected) in [
        ("**", vec![Kind::Pow]),
        ("* *", vec![Kind::Star, Kind::Star]),
        ("++", vec![Kind::Concat]),
        ("+ +", vec![Kind::Plus, Kind::Plus]),
        ("->", vec![Kind::Arrow]),
        ("- >", vec![Kind::Minus, Kind::Gt]),
        ("==", vec![Kind::EqEq]),
        ("!=", vec![Kind::NotEq]),
        ("+=", vec![Kind::AddAssign]),
        ("/=", vec![Kind::DivAssign]),
        ("=", vec![Kind::Assign]),
    ] {
        assert_eq!(kinds(source), expected, "`{source}`");
    }
}

#[test]
fn numbers_split_from_what_follows_them() {
    for (source, expected) in [
        ("0", vec![Kind::Int]),
        ("3.0", vec![Kind::Float]),
        (".5", vec![Kind::Float]),
        ("1e3", vec![Kind::Float]),
        ("1e-3", vec![Kind::Float]),
        ("0xff", vec![Kind::Int]),
        ("0b1010", vec![Kind::Int]),
        ("0o17", vec![Kind::Int]),
        ("1_000", vec![Kind::Int]),
        // `e` opens an exponent only when digits follow, so a number beside a
        // name stays two tokens.
        ("2 euler", vec![Kind::Int, Kind::Ident]),
        ("0:2", vec![Kind::Int, Kind::Colon, Kind::Int]),
        (
            "0:-1:3",
            vec![
                Kind::Int,
                Kind::Colon,
                Kind::Minus,
                Kind::Int,
                Kind::Colon,
                Kind::Int,
            ],
        ),
    ] {
        assert_eq!(kinds(source), expected, "`{source}`");
    }
}

// A unit belongs to the number only when nothing name-like follows it, so `2sx`
// stays the number and the name it lexed as before durations existed.
#[test]
fn a_time_unit_glues_to_its_number() {
    for source in [
        "10ns",
        "2.5us",
        "4dt",
        "3ms",
        "1s",
        "1e3ns",
        "1_000ns",
        "3\u{b5}s",
        "3\u{3bc}s",
    ] {
        assert_eq!(kinds(source), vec![Kind::Duration], "`{source}`");
        assert_eq!(texts(source), vec![source.to_string()], "`{source}`");
    }
    for (source, expected) in [
        ("2sx", vec![Kind::Int, Kind::Ident]),
        ("2 ns", vec![Kind::Int, Kind::Ident]),
        ("2nsec", vec![Kind::Int, Kind::Ident]),
        ("2e", vec![Kind::Int, Kind::Ident]),
        ("0x1f", vec![Kind::Int]),
    ] {
        assert_eq!(kinds(source), expected, "`{source}`");
    }
}

// A `/*` inside a line comment or a string opens nothing, and a `//` inside a
// string ends nothing.
#[test]
fn comments_and_strings_do_not_reach_into_each_other() {
    assert_eq!(kinds("a // /* b\nc"), vec![Kind::Ident, Kind::Ident]);
    assert_eq!(
        texts(r#"include "a//b.inc";"#),
        vec![
            "include".to_string(),
            "a//b.inc".to_string(),
            ";".to_string()
        ]
    );
    assert_eq!(texts(r#""a/*b""#), vec!["a/*b".to_string()]);
    assert_eq!(kinds("a /* b\nc */ d"), vec![Kind::Ident, Kind::Ident]);
}

#[test]
fn a_line_number_survives_every_kind_of_trivia() {
    let source = "a\n// comment\n/* two\nlines */\nb\n\"text\"\nc";
    let tokens = tokenize(source).unwrap();
    let lines: Vec<u32> = tokens
        .iter()
        .filter(|token| token.kind != Kind::Eof)
        .map(|token| token.line)
        .collect();
    assert_eq!(lines, vec![1, 5, 6, 7]);
}

#[test]
fn a_pragma_keeps_its_whole_line() {
    let tokens = tokenize("h q[0];\n#pragma braket result state_vector\nx q[0];").unwrap();
    let pragma = tokens
        .iter()
        .find(|token| token.kind == Kind::Pragma)
        .expect("pragma token");
    assert_eq!(pragma.text, "#pragma braket result state_vector");
    assert_eq!(pragma.line, 2);
}

#[test]
fn a_physical_qubit_carries_its_index() {
    let tokens = tokenize("cx $0, $12;").unwrap();
    let physical: Vec<&str> = tokens
        .iter()
        .filter(|token| token.kind == Kind::Physical)
        .map(|token| token.text)
        .collect();
    assert_eq!(physical, vec!["0", "12"]);
}

#[test]
fn a_column_is_reported_where_the_token_starts() {
    let tokens = tokenize("qubit[4] q;").unwrap();
    assert_eq!((tokens[0].line, tokens[0].column), (1, 1));
    assert_eq!((tokens[1].line, tokens[1].column), (1, 6));
    assert_eq!(tokens[3].text, "]");
    assert_eq!(tokens[3].column, 8);
}

#[test]
fn malformed_input_is_rejected_with_its_line() {
    for (source, line) in [
        ("a\nb\n/* open", 3usize),
        ("a\n\"open", 2),
        ("a\n$x", 2),
        ("a\n0x", 2),
        ("a\n`", 2),
    ] {
        match tokenize(source) {
            Err(PrismError::Parse { line: at, .. }) => assert_eq!(at, line, "`{source}`"),
            other => panic!("`{source}` gave {other:?}"),
        }
    }
}

#[test]
fn a_name_may_open_with_any_letter() {
    assert_eq!(texts("θ_1 γ πx a_é"), ["θ_1", "γ", "πx", "a_é"]);
    assert!(tokenize("x → y").is_err());
}

#[test]
fn both_pragma_spellings_take_their_line() {
    for source in ["#pragma a.b c\nx", "pragma a.b c\nx"] {
        assert_eq!(kinds(source), [Kind::Pragma, Kind::Ident], "`{source}`");
    }
}

// A modifier's `@` follows a name; only an `@` opening a statement starts an
// annotation.
#[test]
fn only_a_statement_opening_at_is_an_annotation() {
    assert_eq!(
        kinds("@ns.tag rest of line\nx;"),
        [Kind::Ident, Kind::Semicolon]
    );
    assert_eq!(
        kinds("x; @tag\ny;"),
        [Kind::Ident, Kind::Semicolon, Kind::Ident, Kind::Semicolon]
    );
    assert_eq!(kinds("inv @h"), [Kind::Ident, Kind::At, Kind::Ident]);
}

#[test]
fn a_calibration_body_is_one_token() {
    let source = "defcal x $0 { play(d, \"}\"); /* } */ // }\n { nested } }\nx;";
    assert_eq!(
        kinds(source),
        [
            Kind::Ident,
            Kind::Ident,
            Kind::Physical,
            Kind::Calibration,
            Kind::Ident,
            Kind::Semicolon
        ]
    );
    let tokens = tokenize(source).unwrap();
    assert_eq!(tokens[4].line, 3, "lines inside the body still count");
    assert_eq!(kinds("cal { ... }"), [Kind::Ident, Kind::Calibration]);
    // Not at a statement start, `cal` is an ordinary name and `{` an ordinary brace.
    assert_eq!(kinds("x cal {"), [Kind::Ident, Kind::Ident, Kind::LBrace]);
    assert!(tokenize("cal { open").is_err());
}

#[test]
fn a_leading_byte_order_mark_is_skipped() {
    assert_eq!(kinds("\u{feff}x;"), [Kind::Ident, Kind::Semicolon]);
    assert!(tokenize("x;\u{feff}").is_err());
}

#[test]
fn imaginary_literals_and_rank_markers_lex_as_their_own_tokens() {
    assert_eq!(kinds("1.5im"), vec![Kind::Imaginary]);
    assert_eq!(kinds("2 \tim"), vec![Kind::Imaginary]);
    assert_eq!(texts("3im + 1"), vec!["3im", "+", "1"]);
    assert_eq!(kinds("2 imag"), vec![Kind::Int, Kind::Ident]);
    assert_eq!(kinds("#dim = 2"), vec![Kind::Dim, Kind::Assign, Kind::Int]);
    assert_eq!(kinds("#dimension"), vec![Kind::Pragma]);
}

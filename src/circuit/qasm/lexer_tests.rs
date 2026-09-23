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
    let lines: Vec<usize> = tokens
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

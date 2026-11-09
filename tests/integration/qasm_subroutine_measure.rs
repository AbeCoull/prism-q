//! Subroutines that measure into bits of their own, measured bits read as
//! values, and register slices and index sets passed to expression calls.

use crate::common::SEED;

use prism_q::circuit::dynamic::ClassicalValue;
use prism_q::circuit::openqasm::{self, Externs};
use prism_q::{DynamicProgram, PrismError, simulate_program};

fn shots(program: &DynamicProgram, count: usize) -> Vec<Vec<bool>> {
    let result = simulate_program(program).seed(SEED).shots(count).unwrap();
    for shot in &result.shots {
        assert_eq!(
            shot.len(),
            program.num_classical_bits(),
            "a shot reports private bits"
        );
    }
    result.shots
}

fn dynamic(source: &str, bits: usize, private: usize) -> DynamicProgram {
    let program = openqasm::parse_dynamic(source).unwrap_or_else(|err| panic!("{err}"));
    assert_eq!(program.num_classical_bits(), bits, "public width");
    assert_eq!(program.num_private_bits(), private, "private width");
    program
}

fn unsupported(err: PrismError, needle: &str, label: &str) {
    match err {
        PrismError::UnsupportedConstruct { construct, .. } => {
            assert!(construct.contains(needle), "{label}: {construct}");
        }
        other => panic!("{label}: expected UnsupportedConstruct, got {other:?}"),
    }
}

#[test]
fn a_def_measures_into_a_bit_it_declares() {
    let source = "OPENQASM 3.0;
qubit[3] q;
bit[2] c;
def herald(qubit a, qubit flag) {
  bit r;
  h a;
  r = measure a;
  if (r) x flag;
}
herald(q[0], q[1]);
c[0] = measure q[0];
c[1] = measure q[1];
";
    let program = dynamic(source, 2, 1);
    let shots = shots(&program, 200);
    for shot in &shots {
        assert_eq!(shot[1], shot[0], "the flag follows the private outcome");
    }
    assert!(shots.iter().any(|shot| shot[0]) && shots.iter().any(|shot| !shot[0]));
    unsupported(openqasm::parse(source).unwrap_err(), "bit r", "parse");
}

#[test]
fn each_call_takes_its_own_private_bits() {
    let source = "OPENQASM 3.0;
qubit[3] q;
bit[3] c;
def copy(qubit a, qubit b) {
  bit[2] r;
  r[0] = measure a;
  r[1] = measure a;
  if (r == 3) x b;
}
x q[0];
copy(q[0], q[1]);
for int i in [0:1] { copy(q[0], q[2]); }
c = measure q;
";
    let program = dynamic(source, 3, 6);
    for shot in shots(&program, 20) {
        assert_eq!(shot, [true, true, false], "q[2] flips twice");
    }
}

#[test]
fn a_dropped_bit_result_lands_on_private_bits() {
    let source = "OPENQASM 3.0;
qubit[1] q;
bit[2] c;
def mx(qubit a) -> bit {
  h a;
  return measure a;
}
mx(q[0]);
c[0] = measure q[0];
h q[0];
c[1] = measure q[0];
";
    let program = dynamic(source, 2, 1);
    let shots = shots(&program, 200);
    assert!(shots.iter().any(|shot| shot[0]) && shots.iter().any(|shot| !shot[0]));
    unsupported(
        openqasm::parse(source).unwrap_err(),
        "drops its `bit` result",
        "parse",
    );
}

#[test]
fn a_runtime_call_initializes_its_private_bits_on_every_call() {
    let source = "OPENQASM 3.0;
qubit[2] q;
bit[2] c;
def kick(qubit a) {
  bit r = 1;
  if (r) x a;
  r = measure a;
}
kick(q[0]);
kick(q[1]);
kick(q[1]);
c = measure q;
";
    let program = dynamic(source, 2, 3);
    for shot in shots(&program, 20) {
        assert_eq!(shot, [true, false]);
    }
}

#[test]
fn private_storage_declines_where_a_pass_would_share_it() {
    let inlined = "OPENQASM 3.0;
qubit[2] q;
bit[1] c;
def f(qubit a) { bit r; r = measure a; if (r) x a; }
c[0] = measure q[0];
while (c[0]) { f(q[1]); c[0] = measure q[0]; }
";
    unsupported(
        openqasm::parse_dynamic(inlined).unwrap_err(),
        "inlined into a `while` loop",
        "while",
    );
    let parameter = "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
def f(qubit a, bit b) { b = measure a; }
f(q[0], c[0]);
";
    unsupported(
        openqasm::parse_dynamic(parameter).unwrap_err(),
        "bit parameter",
        "parameter",
    );
}

#[test]
fn measured_bits_read_as_values() {
    let source = "OPENQASM 3.0;
qubit[4] q;
bit[4] c;
x q[1];
c[0] = measure q[0];
c[1] = measure q[1];
int x = c[0] + 2 * c[1];
if (x == 2) x q[2];
if (c[1] && !c[0]) x q[3];
c[2] = measure q[2];
c[3] = measure q[3];
";
    let program = dynamic(source, 4, 0);
    for shot in shots(&program, 20) {
        assert_eq!(shot, [false, true, true, true]);
    }
    for (body, label) in [
        ("int x = c[0] + 2 * c[1];", "an integer"),
        ("rx(c[0] * pi) q[0];", "an angle"),
        ("bool b = c[0] == c[1];", "a comparison"),
    ] {
        let source = format!("OPENQASM 3.0; qubit[2] q; bit[2] c; c = measure q; {body}");
        unsupported(
            openqasm::parse(&source).unwrap_err(),
            "reads a classical bit as a value",
            label,
        );
    }
}

#[test]
fn slices_and_sets_read_as_values() {
    let source = "OPENQASM 3.0;
qubit[5] q;
bit[5] c;
x q[0];
x q[2];
c[0:2] = measure q[0:2];
int n = popcount(c[0:1]);
int m = popcount(c[{0, 2}]);
uint[3] v = c[{2, 0, 1}];
if (n == 1 && m == 2) x q[3];
if (v == 3) x q[4];
c[3] = measure q[3];
c[4] = measure q[4];
";
    let program = dynamic(source, 5, 0);
    for shot in shots(&program, 20) {
        assert_eq!(shot, [true, false, true, true, true]);
    }
    unsupported(
        openqasm::parse("OPENQASM 3.0; qubit[2] q; bit[2] c; int n = popcount(c[{0, 1}]);")
            .unwrap_err(),
        "reads a classical bit as a value",
        "parse",
    );
}

#[test]
fn an_extern_takes_a_slice_or_a_set_as_a_bit_register() {
    let mut externs = Externs::new();
    externs.register("decode", |args| {
        Ok(ClassicalValue::Bool(args[0] == ClassicalValue::Int(0b101)))
    });
    let source = "OPENQASM 3.0;
extern decode(bit[3]) -> bool;
qubit[5] q;
bit[5] c;
x q[0];
x q[2];
c[0:2] = measure q[0:2];
if (decode(c[0:2])) x q[3];
if (decode(c[{1, 0, 2}])) x q[4];
c[3] = measure q[3];
c[4] = measure q[4];
";
    let program = openqasm::parse_dynamic_with_externs(source, &[], &externs).unwrap();
    assert_eq!(program.num_classical_bits(), 5);
    for shot in shots(&program, 20) {
        assert_eq!(
            shot,
            [true, false, true, true, false],
            "{{1, 0, 2}} reads 0b110"
        );
    }
    let wide = "OPENQASM 3.0; extern decode(bit[65]) -> bool; qubit[1] q;";
    unsupported(
        openqasm::parse_with_externs(wide, &externs).unwrap_err(),
        "bit[65]",
        "wide",
    );
}

#[test]
fn popcount_runs_at_runtime_and_folds_at_parse_time() {
    let program = dynamic(
        "OPENQASM 3.0; qubit[3] q; bit[3] c; x q[0]; x q[1]; c[0:1] = measure q[0:1];
         if (popcount(c) == popcount(3)) x q[2]; c[2] = measure q[2];",
        3,
        0,
    );
    for shot in shots(&program, 10) {
        assert_eq!(shot, [true, true, true]);
    }
}

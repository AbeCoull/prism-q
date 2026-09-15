//! Shot-based evaluation of Braket result requests: the eigenvalue series a
//! `sample` reports, and the statistics `expectation`, `variance` and
//! `probability` derive from the same record.

mod common;

use prism_q::PrismError;
use prism_q::circuit::openqasm;
use prism_q::sim::ResultValue;
use prism_q::simulate;

use common::SEED;

/// Enough shots that a mean sits inside `STATISTICAL_EPS` of its exact value
/// for every fixture here, with the seed fixed so the margin is not a gamble.
const SHOTS: usize = 20_000;
const STATISTICAL_EPS: f64 = 0.05;

fn source(qubits: usize, body: &str) -> String {
    format!("OPENQASM 3.0;\nqubit[{qubits}] q;\n{body}\n")
}

fn sampled(qubits: usize, body: &str, shots: usize) -> Vec<ResultValue> {
    let text = source(qubits, body);
    let program = openqasm::parse_braket(&text).unwrap_or_else(|e| panic!("`{body}`: {e}"));
    simulate(&program.circuit)
        .seed(SEED)
        .braket_results_sampled(&program.results, shots)
        .unwrap_or_else(|e| panic!("`{body}`: {e}"))
}

fn sampled_err(qubits: usize, body: &str, shots: usize) -> PrismError {
    let text = source(qubits, body);
    let program = openqasm::parse_braket(&text).unwrap_or_else(|e| panic!("`{body}`: {e}"));
    simulate(&program.circuit)
        .seed(SEED)
        .braket_results_sampled(&program.results, shots)
        .err()
        .unwrap_or_else(|| panic!("`{body}` should not evaluate"))
}

fn exact(qubits: usize, body: &str) -> Vec<ResultValue> {
    let text = source(qubits, body);
    let program = openqasm::parse_braket(&text).unwrap_or_else(|e| panic!("`{body}`: {e}"));
    simulate(&program.circuit)
        .seed(SEED)
        .braket_results(&program.results)
        .unwrap_or_else(|e| panic!("`{body}`: {e}"))
}

fn series(value: &ResultValue) -> &[Vec<f64>] {
    match value {
        ResultValue::Sample(series) => series,
        other => panic!("expected a sample, got {other:?}"),
    }
}

fn numbers(value: &ResultValue) -> &[f64] {
    match value {
        ResultValue::Expectation(values)
        | ResultValue::Variance(values)
        | ResultValue::Probability(values) => values,
        other => panic!("expected a numeric result, got {other:?}"),
    }
}

fn assert_constant(single: &[f64], expected: f64, shots: usize, label: &str) {
    assert_eq!(single.len(), shots, "{label}: wrong series length");
    let stray = single.iter().find(|shot| (*shot - expected).abs() > 1e-9);
    assert!(stray.is_none(), "{label}: a shot gave {stray:?}");
}

fn assert_all_constant(value: &ResultValue, expected: f64, shots: usize, label: &str) {
    for (index, single) in series(value).iter().enumerate() {
        assert_constant(single, expected, shots, &format!("{label}: series {index}"));
    }
}

fn assert_close(actual: &[f64], expected: &[f64], eps: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: wrong length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= eps,
            "{label}: entry {index} is {got}, expected {want}"
        );
    }
}

// An eigenstate of the observable gives the same eigenvalue every shot, so a
// wrong rotation shows up as a series that is not constant rather than as a
// mean that is merely off.
#[test]
fn an_eigenstate_samples_its_own_eigenvalue_every_shot() {
    let cases = [
        ("x", "h q[0];", "x q[0]", 1.0),
        ("x, minus", "x q[0];\nh q[0];", "x q[0]", -1.0),
        ("y", "h q[0];\ns q[0];", "y q[0]", 1.0),
        ("y, minus", "h q[0];\nsdg q[0];", "y q[0]", -1.0),
        ("z", "", "z q[0]", 1.0),
        ("z, minus", "x q[0];", "z q[0]", -1.0),
        ("hadamard", "ry(pi/4) q[0];", "h q[0]", 1.0),
        ("hadamard, minus", "x q[0];\nry(pi/4) q[0];", "h q[0]", -1.0),
        ("identity", "rx(0.7) q[0];", "i q[0]", 1.0),
    ];
    for (label, body, observable, expected) in cases {
        let request = format!("{body}\n#pragma braket result sample {observable}");
        let computed = sampled(1, &request, SHOTS);
        assert_all_constant(&computed[0], expected, SHOTS, label);
    }
}

// A Hermitian matrix reaches its eigenvalues through a numerical
// diagonalization, so the cases that could break it are the ones a naive
// factorization gets wrong: two eigenvalues equal in magnitude and opposite in
// sign, and a spectrum that is entirely negative.
#[test]
fn a_hermitian_observable_samples_its_own_spectrum() {
    let cases = [
        ("pauli x as a matrix", "h q[0];", "[[0, 1], [1, 0]]", 1.0),
        (
            "pauli x as a matrix, minus",
            "x q[0];\nh q[0];",
            "[[0, 1], [1, 0]]",
            -1.0,
        ),
        ("all positive", "", "[[2, 0], [0, 3]]", 2.0),
        ("all positive, upper", "x q[0];", "[[2, 0], [0, 3]]", 3.0),
        ("all negative", "", "[[-2, 0], [0, -3]]", -2.0),
        ("all negative, upper", "x q[0];", "[[-2, 0], [0, -3]]", -3.0),
        ("degenerate", "rx(0.9) q[0];", "[[4, 0], [0, 4]]", 4.0),
        (
            "off diagonal and complex",
            "h q[0];\ns q[0];",
            "[[0, -1im], [1im, 0]]",
            1.0,
        ),
    ];
    for (label, body, matrix, expected) in cases {
        let request =
            format!("{body}\n#pragma braket result sample hermitian({matrix}) q[0]").to_string();
        let computed = sampled(1, &request, SHOTS);
        assert_all_constant(&computed[0], expected, SHOTS, label);
    }
}

// A two-qubit Hermitian observable also fixes a target order, so the same
// matrix on the reversed targets reads a different eigenvalue.
#[test]
fn a_two_qubit_hermitian_observable_respects_its_target_order() {
    // Diagonal, so the eigenvalue of |q0 q1> is the entry its index selects
    // with q[0] as the high bit.
    const MATRIX: &str = "[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]";
    for (body, forward, reversed) in [
        ("", 1.0, 1.0),
        ("x q[1];", 2.0, 3.0),
        ("x q[0];", 3.0, 2.0),
        ("x q[0];\nx q[1];", 4.0, 4.0),
    ] {
        for (order, expected) in [("q[0], q[1]", forward), ("q[1], q[0]", reversed)] {
            let request =
                format!("{body}\n#pragma braket result sample hermitian({MATRIX}) {order}");
            let computed = sampled(2, &request, SHOTS);
            assert_all_constant(&computed[0], expected, SHOTS, &format!("{body} {order}"));
        }
    }
}

// A single-qubit observable with no target list reports one series per qubit,
// the same shape the exact path gives.
#[test]
fn an_untargeted_observable_reports_one_series_per_qubit() {
    let computed = sampled(
        3,
        "x q[0];\nx q[2];\n#pragma braket result sample z all",
        SHOTS,
    );
    let all = series(&computed[0]);
    assert_eq!(all.len(), 3);
    for (qubit, expected) in [(0usize, -1.0), (1, 1.0), (2, -1.0)] {
        assert_constant(&all[qubit], expected, SHOTS, &format!("qubit {qubit}"));
    }
}

// The sampled mean and variance have to converge on the values the exact path
// computes in closed form, across observables that are not eigenbases of the
// state.
#[test]
fn sampled_statistics_converge_on_the_exact_values() {
    let bodies = [
        "h q[0];\ncnot q[0], q[1];",
        "rx(0.7) q[0];\nry(1.3) q[1];\ncz q[0], q[1];",
        "h q[0];\nt q[0];\nh q[0];\ncnot q[0], q[1];\nrz(0.4) q[1];",
    ];
    let observables = ["z q[0] @ z q[1]", "x q[0]", "y q[1]", "h q[0]", "z all"];
    for body in bodies {
        for observable in observables {
            let request = format!(
                "{body}\n#pragma braket result expectation {observable}\n\
                 #pragma braket result variance {observable}"
            );
            let label = format!("{body} / {observable}");
            let shot = sampled(2, &request, SHOTS);
            let closed = exact(2, &request);
            assert_close(
                numbers(&shot[0]),
                numbers(&closed[0]),
                STATISTICAL_EPS,
                &format!("{label}: expectation"),
            );
            assert_close(
                numbers(&shot[1]),
                numbers(&closed[1]),
                STATISTICAL_EPS,
                &format!("{label}: variance"),
            );
        }
    }
}

// A `probability` reads the computational basis. A rotation applied for some
// other observable in the same program must not reach it, which the rotated
// record would show as four equal outcomes instead of two.
#[test]
fn a_probability_stays_in_the_computational_basis() {
    let bell = "h q[0];\ncnot q[0], q[1];";
    let alone = sampled(
        2,
        &format!("{bell}\n#pragma braket result probability"),
        SHOTS,
    );
    assert_close(
        numbers(&alone[0]),
        &[0.5, 0.0, 0.0, 0.5],
        STATISTICAL_EPS,
        "probability alone",
    );

    let beside = sampled(
        2,
        &format!(
            "{bell}\n#pragma braket result expectation x q[0] @ x q[1]\n\
             #pragma braket result probability"
        ),
        SHOTS,
    );
    assert_close(
        numbers(&beside[1]),
        &[0.5, 0.0, 0.0, 0.5],
        STATISTICAL_EPS,
        "probability beside a rotated observable",
    );
    assert_close(
        numbers(&beside[0]),
        &[1.0],
        STATISTICAL_EPS,
        "expectation beside a probability",
    );
}

// A probability target list is read with the first target as the high bit, so
// reversing it reverses which outcome carries the weight.
#[test]
fn a_probability_target_list_is_ordered() {
    let computed = sampled(
        2,
        "x q[0];\n#pragma braket result probability q[0], q[1]\n\
         #pragma braket result probability q[1], q[0]",
        64,
    );
    assert_close(numbers(&computed[0]), &[0.0, 0.0, 1.0, 0.0], 0.0, "q0, q1");
    assert_close(numbers(&computed[1]), &[0.0, 1.0, 0.0, 0.0], 0.0, "q1, q0");
}

// Two observables reading one qubit in different bases cannot share a
// measurement, and answering from whichever was applied first would be wrong
// rather than approximate.
#[test]
fn observables_that_cannot_share_a_measurement_are_rejected() {
    let err = sampled_err(
        2,
        "h q[0];\n#pragma braket result expectation x q[0]\n\
         #pragma braket result expectation z q[0]",
        64,
    );
    let text = format!("{err}");
    assert!(
        matches!(err, PrismError::InvalidParameter { .. }) && text.contains("qubit 0"),
        "got {text}"
    );

    // The identity reads no basis, so it shares a qubit with anything.
    let shared = sampled(
        2,
        "h q[0];\n#pragma braket result expectation x q[0]\n\
         #pragma braket result expectation i q[0]",
        64,
    );
    assert_close(numbers(&shared[0]), &[1.0], 1e-9, "x on its eigenstate");
    assert_close(numbers(&shared[1]), &[1.0], 1e-9, "identity");

    // Different qubits never conflict.
    let apart = sampled(
        2,
        "h q[0];\n#pragma braket result expectation x q[0]\n\
         #pragma braket result expectation z q[1]",
        64,
    );
    assert_close(numbers(&apart[0]), &[1.0], 1e-9, "x on q0");
    assert_close(numbers(&apart[1]), &[1.0], 1e-9, "z on q1");
}

// The three requests that report the state itself have no shot-based reading,
// and zero shots has no statistics to report.
#[test]
fn requests_without_a_shot_reading_are_declined() {
    for request in [
        "#pragma braket result state_vector",
        "#pragma braket result amplitude \"00\"",
        "#pragma braket result density_matrix",
    ] {
        let err = sampled_err(2, &format!("h q[0];\n{request}"), 64);
        assert!(
            matches!(err, PrismError::BackendUnsupported { .. }),
            "`{request}`: got {err}"
        );
    }

    let err = sampled_err(2, "h q[0];\n#pragma braket result expectation z q[0]", 0);
    assert!(
        matches!(err, PrismError::InvalidParameter { .. }),
        "got {err}"
    );
}

// Sampling reads the same circuit the exact path does, so a program that
// measures on its own keeps those measurements and the appended readout takes
// its own classical bits.
#[test]
fn a_program_that_measures_keeps_its_own_measurements() {
    let computed = sampled(
        2,
        "bit[1] c;\nx q[0];\nc[0] = measure q[0];\n\
         #pragma braket result sample z q[0]",
        128,
    );
    assert_constant(&series(&computed[0])[0], -1.0, 128, "measured then sampled");
}

// A noisy program samples from the mixture rather than from a pure state, so
// the mean moves off 1 by twice the flip probability.
#[test]
fn a_noisy_program_samples_from_the_mixture() {
    let text = source(
        1,
        "id q[0];\n#pragma braket noise bit_flip(0.25) q[0]\n\
         #pragma braket result expectation z q[0]",
    );
    let program = openqasm::parse_braket(&text).unwrap();
    let noise = program.noise.expect("a noise model");
    let values = simulate(&program.circuit)
        .seed(SEED)
        .noise(&noise)
        .braket_results_sampled(&program.results, SHOTS)
        .unwrap();
    assert_close(numbers(&values[0]), &[0.5], STATISTICAL_EPS, "bit flip");
}

// A three-qubit Hermitian observable has no gate wide enough to carry its
// rotation, so the rotation is reduced to multi-controlled gates. A diagonal
// matrix makes the eigenvalue pairing visible: each basis state reads the
// entry its own index selects.
#[test]
fn a_three_qubit_hermitian_observable_samples_its_spectrum() {
    let diagonal: Vec<String> = (0i32..8)
        .map(|row| {
            let entries: Vec<String> = (0..8)
                .map(|column| {
                    if row == column {
                        format!("{}", row - 3)
                    } else {
                        "0".to_string()
                    }
                })
                .collect();
            format!("[{}]", entries.join(", "))
        })
        .collect();
    let matrix = format!("[{}]", diagonal.join(", "));
    for (body, expected) in [
        ("", -3.0),
        ("x q[2];", -2.0),
        ("x q[1];", -1.0),
        ("x q[0];", 1.0),
        ("x q[0];\nx q[1];\nx q[2];", 4.0),
    ] {
        let request =
            format!("{body}\n#pragma braket result sample hermitian({matrix}) q[0], q[1], q[2]");
        let computed = sampled(3, &request, 256);
        assert_constant(&series(&computed[0])[0], expected, 256, body);
    }
}

// A non-diagonal wide observable still has to agree with the exact path, which
// reaches the same value through the Pauli decomposition instead.
#[test]
fn a_wide_hermitian_observable_agrees_with_the_exact_value() {
    const MATRIX: &str = "[[1, 0, 0, 0.5, 0, 0, 0, 0], \
                           [0, -2, 0, 0, 0, 0, 0.25, 0], \
                           [0, 0, 3, 0, 0, 0, 0, 0], \
                           [0.5, 0, 0, 0, 0, 0, 0, 0], \
                           [0, 0, 0, 0, -1, 0, 0, 0], \
                           [0, 0, 0, 0, 0, 2, 0, 0], \
                           [0, 0.25, 0, 0, 0, 0, 0, 0], \
                           [0, 0, 0, 0, 0, 0, 0, 4]]";
    let body = "h q[0];\nry(0.7) q[1];\ncnot q[1], q[2];\nrz(0.3) q[2];";
    let request =
        format!("{body}\n#pragma braket result expectation hermitian({MATRIX}) q[0], q[1], q[2]");
    let shot = sampled(3, &request, SHOTS);
    let closed = exact(3, &request);
    assert_close(
        numbers(&shot[0]),
        numbers(&closed[0]),
        0.15,
        "wide hermitian expectation",
    );
}

// The same basis on the same qubits shares a measurement even when one request
// names its targets and the other says `all`.
#[test]
fn the_same_basis_written_two_ways_shares_one_measurement() {
    let computed = sampled(
        2,
        "h q[0];\nh q[1];\n#pragma braket result expectation x q[0]\n\
         #pragma braket result expectation x all",
        512,
    );
    assert_close(numbers(&computed[0]), &[1.0], 1e-9, "x on q0");
    assert_close(numbers(&computed[1]), &[1.0, 1.0], 1e-9, "x on all");
}

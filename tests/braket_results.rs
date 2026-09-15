//! Computing the result requests a Braket program declared, and the basis-state
//! order the values come back in.

mod common;

use num_complex::Complex64;
use prism_q::PrismError;
use prism_q::circuit::openqasm;
use prism_q::sim::ResultValue;
use prism_q::simulate;

use common::{SEED, SV_EPS};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// A state whose amplitudes are all distinct and complex, so a permuted or
/// conjugated answer cannot pass.
const FIXTURE: &str = "h q[0];\ncnot q[0], q[1];\nt q[1];\nrx(0.7) q[0];";

fn source(qubits: usize, body: &str) -> String {
    format!("OPENQASM 3.0;\nqubit[{qubits}] q;\n{body}\n")
}

fn evaluate(qubits: usize, body: &str) -> Vec<ResultValue> {
    let text = source(qubits, body);
    let program = openqasm::parse_braket(&text).unwrap_or_else(|e| panic!("`{body}`: {e}"));
    simulate(&program.circuit)
        .seed(SEED)
        .braket_results(&program.results)
        .unwrap_or_else(|e| panic!("`{body}`: {e}"))
}

fn evaluate_err(qubits: usize, body: &str) -> PrismError {
    let text = source(qubits, body);
    let program = openqasm::parse_braket(&text).unwrap_or_else(|e| panic!("`{body}`: {e}"));
    simulate(&program.circuit)
        .seed(SEED)
        .braket_results(&program.results)
        .err()
        .unwrap_or_else(|| panic!("`{body}` should not evaluate"))
}

fn state_of(qubits: usize, body: &str) -> Vec<Complex64> {
    let text = source(qubits, body);
    let program = openqasm::parse_braket(&text).unwrap();
    simulate(&program.circuit)
        .seed(SEED)
        .state_vector()
        .unwrap()
}

/// Lift a `2^k` block on `targets` to the whole register. `targets[0]` is the
/// high bit of the block index, as it is in a Braket observable and in a gate's
/// own 4x4; every other qubit carries the identity.
fn embed(qubits: usize, targets: &[usize], block: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let dim = 1usize << qubits;
    let width = targets.len();
    let named = targets.iter().fold(0usize, |mask, &q| mask | 1 << q);
    let local = |index: usize| {
        targets.iter().enumerate().fold(0usize, |acc, (bit, &q)| {
            acc | (index >> q & 1) << (width - 1 - bit)
        })
    };
    (0..dim)
        .map(|row| {
            (0..dim)
                .map(|column| {
                    if row & !named != column & !named {
                        ZERO
                    } else {
                        block[local(row)][local(column)]
                    }
                })
                .collect()
        })
        .collect()
}

fn multiply(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    a.iter()
        .map(|row| {
            (0..b.len())
                .map(|column| {
                    row.iter()
                        .enumerate()
                        .map(|(k, entry)| entry * b[k][column])
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn identity(dim: usize) -> Vec<Vec<Complex64>> {
    (0..dim)
        .map(|row| {
            (0..dim)
                .map(|c| if c == row { ONE } else { ZERO })
                .collect()
        })
        .collect()
}

/// `<psi|O|psi>`, which is real for a Hermitian `O`.
fn braket_form(state: &[Complex64], operator: &[Vec<Complex64>]) -> f64 {
    let applied: Vec<Complex64> = operator
        .iter()
        .map(|row| row.iter().zip(state).map(|(m, a)| m * a).sum())
        .collect();
    state
        .iter()
        .zip(&applied)
        .map(|(a, b)| (a.conj() * b).re)
        .sum()
}

fn row(entries: &[Complex64]) -> Vec<Complex64> {
    entries.to_vec()
}

fn pauli(axis: char) -> Vec<Vec<Complex64>> {
    let i = Complex64::new(0.0, 1.0);
    match axis {
        'x' => vec![row(&[ZERO, ONE]), row(&[ONE, ZERO])],
        'y' => vec![row(&[ZERO, -i]), row(&[i, ZERO])],
        'z' => vec![row(&[ONE, ZERO]), row(&[ZERO, -ONE])],
        'h' => {
            let s = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
            vec![row(&[s, s]), row(&[s, -s])]
        }
        _ => identity(2),
    }
}

/// A Hermitian 2x2 and 4x4 with no symmetry to hide an index swap.
fn hermitian_2x2() -> Vec<Vec<Complex64>> {
    vec![
        row(&[Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0)]),
        row(&[Complex64::new(2.0, 1.0), Complex64::new(-3.0, 0.0)]),
    ]
}

fn hermitian_4x4() -> Vec<Vec<Complex64>> {
    let c = |re: f64, im: f64| Complex64::new(re, im);
    vec![
        row(&[c(-6.0, 0.0), c(2.0, 1.0), c(-3.0, 0.0), c(-5.0, 2.0)]),
        row(&[c(2.0, -1.0), c(0.0, 0.0), c(2.0, -1.0), c(-5.0, 4.0)]),
        row(&[c(-3.0, 0.0), c(2.0, 1.0), c(0.0, 0.0), c(-4.0, 3.0)]),
        row(&[c(-5.0, -2.0), c(-5.0, -4.0), c(-4.0, -3.0), c(-6.0, 0.0)]),
    ]
}

/// One tensor factor as the qubits it acts on and the matrix it carries.
type Block = (Vec<usize>, Vec<Vec<Complex64>>);

/// Each case is a pragma observable beside the same operator as blocks, so the
/// expected side is hand-written matrices rather than a second lowering.
fn observable_cases() -> Vec<(&'static str, Vec<Block>)> {
    vec![
        ("z(q[0])", vec![(vec![0], pauli('z'))]),
        ("x(q[1])", vec![(vec![1], pauli('x'))]),
        ("y(q[0])", vec![(vec![0], pauli('y'))]),
        ("h(q[0])", vec![(vec![0], pauli('h'))]),
        ("i(q[1])", vec![(vec![1], identity(2))]),
        (
            "z(q[0]) @ x(q[1])",
            vec![(vec![0], pauli('z')), (vec![1], pauli('x'))],
        ),
        (
            "h(q[0]) @ h(q[1])",
            vec![(vec![0], pauli('h')), (vec![1], pauli('h'))],
        ),
        (
            "y(q[1]) @ i(q[0])",
            vec![(vec![1], pauli('y')), (vec![0], identity(2))],
        ),
        (
            "hermitian([[1, 2-1im], [2+1im, -3]]) q[0]",
            vec![(vec![0], hermitian_2x2())],
        ),
        (
            "hermitian([[1, 2-1im], [2+1im, -3]]) q[1] @ z(q[0])",
            vec![(vec![1], hermitian_2x2()), (vec![0], pauli('z'))],
        ),
        (
            "hermitian([[-6+0im, 2+1im, -3+0im, -5+2im], [2-1im, 0im, 2-1im, -5+4im], \
             [-3+0im, 2+1im, 0im, -4+3im], [-5-2im, -5-4im, -4-3im, -6+0im]]) q[0], q[1]",
            vec![(vec![0, 1], hermitian_4x4())],
        ),
        (
            "hermitian([[-6+0im, 2+1im, -3+0im, -5+2im], [2-1im, 0im, 2-1im, -5+4im], \
             [-3+0im, 2+1im, 0im, -4+3im], [-5-2im, -5-4im, -4-3im, -6+0im]]) q[1], q[0]",
            vec![(vec![1, 0], hermitian_4x4())],
        ),
    ]
}

fn operator_of(qubits: usize, blocks: &[Block]) -> Vec<Vec<Complex64>> {
    blocks
        .iter()
        .fold(identity(1 << qubits), |acc, (targets, block)| {
            multiply(&acc, &embed(qubits, targets, block))
        })
}

#[test]
fn expectations_match_the_observable_matrix() {
    let state = state_of(2, FIXTURE);
    for (pragma, blocks) in observable_cases() {
        let body = format!("{FIXTURE}\n#pragma braket result expectation {pragma}");
        let computed = evaluate(2, &body);
        let [ResultValue::Expectation(values)] = computed.as_slice() else {
            panic!("`{pragma}`: expected one expectation");
        };
        assert_eq!(values.len(), 1, "`{pragma}` names its targets");
        let expected = braket_form(&state, &operator_of(2, &blocks));
        assert!(
            (values[0] - expected).abs() < SV_EPS,
            "`{pragma}`: got {}, expected {expected}",
            values[0]
        );
    }
}

// `Var(O) = <O^2> - <O>^2`, the spread of the operator itself. A tensor product
// of Paulis squares to the identity and reads `1 - <O>^2`; a Hermitian matrix
// does not, which is what makes the square worth computing.
#[test]
fn variances_match_the_squared_observable_matrix() {
    let state = state_of(2, FIXTURE);
    for (pragma, blocks) in observable_cases() {
        let body = format!("{FIXTURE}\n#pragma braket result variance {pragma}");
        let computed = evaluate(2, &body);
        let [ResultValue::Variance(values)] = computed.as_slice() else {
            panic!("`{pragma}`: expected one variance");
        };
        let operator = operator_of(2, &blocks);
        let mean = braket_form(&state, &operator);
        let expected = braket_form(&state, &multiply(&operator, &operator)) - mean * mean;
        assert!(
            (values[0] - expected).abs() < SV_EPS,
            "`{pragma}`: got {}, expected {expected}",
            values[0]
        );
    }
}

// An observable with no target list is applied to every qubit in parallel and
// reports one value each, not one value for the tensor product of all of them.
#[test]
fn an_untargeted_observable_reports_one_value_per_qubit() {
    let body = "h q[0];\nx q[1];\n#pragma braket result expectation z all";
    let computed = evaluate(3, body);
    let [ResultValue::Expectation(values)] = computed.as_slice() else {
        panic!("expected one expectation");
    };
    assert_eq!(values.len(), 3);
    assert!(values[0].abs() < SV_EPS, "h leaves <Z> at zero");
    assert!((values[1] + 1.0).abs() < SV_EPS, "x flips <Z> to -1");
    assert!(
        (values[2] - 1.0).abs() < SV_EPS,
        "an untouched qubit stays +1"
    );
}

// The observable lowers where the source line is still known, so a shape the
// evaluator could not serve is reported against the pragma that wrote it.
#[test]
fn a_tensor_product_needs_targets_on_every_factor() {
    for body in [
        "h q[0];\n#pragma braket result expectation z all @ x(q[1])",
        "h q[0];\n#pragma braket result expectation z(q[0]) @ x all",
    ] {
        let text = source(2, body);
        let err = openqasm::parse_braket(&text).unwrap_err();
        assert!(
            matches!(err, PrismError::Parse { .. }) && format!("{err}").contains("targets"),
            "`{body}`: {err}"
        );
    }
}

// Braket writes qubit 0 as the most significant bit of a basis index, the
// opposite of the convention every native terminal uses, so `x q[0]` lands at
// index 2 of a two-qubit result and not at index 1.
#[test]
fn basis_states_come_back_in_braket_order() {
    let body = "x q[0];\n\
                #pragma braket result state_vector\n\
                #pragma braket result probability all\n\
                #pragma braket result amplitude \"10\", \"01\"\n\
                #pragma braket result density_matrix all";
    let values = evaluate(2, body);
    let [
        ResultValue::StateVector(state),
        ResultValue::Probability(probabilities),
        ResultValue::Amplitude(amplitudes),
        ResultValue::DensityMatrix(rho),
    ] = values.as_slice()
    else {
        panic!("expected four results, got {values:?}");
    };

    assert_eq!(simulate_index_of_one(state), 2, "q[0] is the high bit");
    assert!((probabilities[2] - 1.0).abs() < SV_EPS, "{probabilities:?}");
    assert_eq!(amplitudes[0].0, "10");
    assert!(
        (amplitudes[0].1.norm() - 1.0).abs() < SV_EPS,
        "{amplitudes:?}"
    );
    assert!(amplitudes[1].1.norm() < SV_EPS, "`01` is empty");
    assert!((rho[2][2] - ONE).norm() < SV_EPS, "{rho:?}");
}

fn simulate_index_of_one(state: &[Complex64]) -> usize {
    state
        .iter()
        .position(|a| (a.norm() - 1.0).abs() < SV_EPS)
        .expect("one populated amplitude")
}

// The order of a `probability` target list is the order of the reported bits,
// so naming the same two qubits the other way round moves the mass.
#[test]
fn a_probability_target_list_is_ordered() {
    let body = "x q[0];\n\
                #pragma braket result probability q[0], q[1]\n\
                #pragma braket result probability q[1], q[0]";
    let computed = evaluate(2, body);
    let [
        ResultValue::Probability(first),
        ResultValue::Probability(second),
    ] = computed.as_slice()
    else {
        panic!("expected two probabilities");
    };
    common::assert_probs_close(first, &[0.0, 0.0, 1.0, 0.0], SV_EPS, "q[0] high");
    common::assert_probs_close(second, &[0.0, 1.0, 0.0, 0.0], SV_EPS, "q[1] high");
}

// A per-qubit marginal cannot show correlation: both qubits of a Bell pair read
// an even split, and only the joint distribution says they agree.
#[test]
fn a_subset_probability_keeps_correlation() {
    let body = "h q[0];\ncnot q[0], q[1];\n\
                #pragma braket result probability q[0], q[1]\n\
                #pragma braket result probability q[0]";
    let computed = evaluate(2, body);
    let [
        ResultValue::Probability(joint),
        ResultValue::Probability(single),
    ] = computed.as_slice()
    else {
        panic!("expected two probabilities");
    };
    common::assert_probs_close(joint, &[0.5, 0.0, 0.0, 0.5], SV_EPS, "bell joint");
    common::assert_probs_close(single, &[0.5, 0.5], SV_EPS, "bell marginal");
}

#[test]
fn a_noise_model_reaches_the_results() {
    let body = "h q[0];\ncnot q[0], q[1];\n\
                #pragma braket noise bit_flip(0.25) q[1]\n\
                #pragma braket result probability all\n\
                #pragma braket result expectation z(q[0]) @ z(q[1])";
    let text = source(2, body);
    let program = openqasm::parse_braket(&text).unwrap();
    let noise = program.noise.expect("a noise model");
    let values = simulate(&program.circuit)
        .seed(SEED)
        .backend(prism_q::BackendKind::DensityMatrix)
        .noise(&noise)
        .braket_results(&program.results)
        .unwrap();
    let [
        ResultValue::Probability(probabilities),
        ResultValue::Expectation(zz),
    ] = values.as_slice()
    else {
        panic!("expected two results");
    };
    // A flip on one half of a Bell pair moves that quarter of the weight onto
    // the disagreeing strings, which drags <ZZ> from 1 to 1 - 2 * 0.25.
    common::assert_probs_close(
        probabilities,
        &[0.375, 0.125, 0.125, 0.375],
        SV_EPS,
        "flipped bell",
    );
    assert!((zz[0] - 0.5).abs() < SV_EPS, "{zz:?}");
}

// `sample` reports per-shot eigenvalues, which an exact evaluation has no
// measurement record to draw from. Declining names it rather than reporting a
// mean under the sample's name.
#[test]
fn a_sample_request_is_declined_by_name() {
    let err = evaluate_err(2, "h q[0];\n#pragma braket result sample z(q[0])");
    let text = format!("{err}");
    assert!(
        matches!(err, PrismError::BackendUnsupported { .. }) && text.contains("sample"),
        "got: {text}"
    );
}

#[test]
fn a_non_hermitian_observable_matrix_is_rejected() {
    let text = source(
        1,
        "h q[0];\n#pragma braket result expectation hermitian([[0, 1], [0, 0]]) q[0]",
    );
    let err = openqasm::parse_braket(&text).unwrap_err();
    assert!(
        format!("{err}").contains("hermitian"),
        "the error should name hermiticity, got: {err}"
    );
}

// A bitstring that is not bits fails at the pragma; one of the wrong length
// fails where the register width is known.
#[test]
fn an_amplitude_bitstring_must_cover_the_register() {
    let text = source(2, "h q[0];\n#pragma braket result amplitude \"02\"");
    assert!(matches!(
        openqasm::parse_braket(&text),
        Err(PrismError::Parse { .. })
    ));
    for label in ["0", "000"] {
        let body = format!("h q[0];\n#pragma braket result amplitude \"{label}\"");
        assert!(
            matches!(evaluate_err(2, &body), PrismError::InvalidParameter { .. }),
            "`{label}`"
        );
    }
}

// One traversal serves every observable request, so asking for the same
// observable twice and asking for its variance beside it agree to the bit.
#[test]
fn repeated_observables_agree_across_one_traversal() {
    let body = format!(
        "{FIXTURE}\n\
         #pragma braket result expectation z(q[0]) @ x(q[1])\n\
         #pragma braket result variance z(q[0]) @ x(q[1])\n\
         #pragma braket result expectation z(q[0]) @ x(q[1])"
    );
    let computed = evaluate(2, &body);
    let [
        ResultValue::Expectation(first),
        ResultValue::Variance(variance),
        ResultValue::Expectation(second),
    ] = computed.as_slice()
    else {
        panic!("expected three results");
    };
    assert_eq!(first, second);
    // A tensor product of Paulis squares to the identity.
    assert!((variance[0] - (1.0 - first[0] * first[0])).abs() < SV_EPS);
}

// Two qubits cannot tell a bit reversal from a bit swap, so the order claim is
// only pinned at three. Distinct amplitudes make a permutation visible.
#[test]
fn three_qubit_basis_order_is_a_reversal_and_not_a_swap() {
    let body = "ry(0.7) q[0];\nry(1.1) q[1];\nry(1.9) q[2];\n\
                #pragma braket result state_vector\n\
                #pragma braket result probability all";
    let computed = evaluate(3, body);
    let [
        ResultValue::StateVector(state),
        ResultValue::Probability(probabilities),
    ] = computed.as_slice()
    else {
        panic!("expected two results, got {computed:?}");
    };
    let native = state_of(3, body);
    for index in 0..8 {
        let reversed = (index & 1) << 2 | (index & 2) | (index >> 2 & 1);
        assert!(
            (state[index] - native[reversed]).norm() < SV_EPS,
            "amplitude {index} is {} against {}",
            state[index],
            native[reversed]
        );
        assert!(
            (probabilities[index] - native[reversed].norm_sqr()).abs() < SV_EPS,
            "probability {index}"
        );
    }
    // The state is a product of three unequal rotations, so no two amplitudes
    // agree and the mapping above has only one solution.
    for index in 0..8 {
        for other in index + 1..8 {
            assert!(
                (state[index].norm() - state[other].norm()).abs() > 1e-3,
                "amplitudes {index} and {other} are too close to discriminate"
            );
        }
    }
}

// A reduced density matrix reverses both indices, and an off-diagonal entry is
// what tells that apart from reversing only the row.
#[test]
fn a_density_matrix_reverses_both_indices() {
    let body = "h q[0];\nt q[0];\nx q[1];\n\
                #pragma braket result density_matrix q[0], q[1]\n\
                #pragma braket result density_matrix q[1], q[0]";
    let computed = evaluate(2, body);
    let [
        ResultValue::DensityMatrix(forward),
        ResultValue::DensityMatrix(reversed),
    ] = computed.as_slice()
    else {
        panic!("expected two results, got {computed:?}");
    };
    // `q[1]` is set, so with `q[0]` first the weight sits on `01` and `11`.
    for (row, column) in [(1usize, 1usize), (3, 3), (1, 3), (3, 1)] {
        assert!(
            forward[row][column].norm() > 0.2,
            "forward ({row}, {column}) is {}",
            forward[row][column]
        );
    }
    assert!(forward[0][0].norm() < SV_EPS && forward[2][2].norm() < SV_EPS);
    // Naming `q[1]` first moves the same weight onto `10` and `11`.
    for (row, column) in [(2usize, 2usize), (3, 3), (2, 3), (3, 2)] {
        assert!(
            reversed[row][column].norm() > 0.2,
            "reversed ({row}, {column}) is {}",
            reversed[row][column]
        );
    }
    // The off-diagonal carries a phase, so reversing only the row would leave
    // the conjugate where the entry belongs.
    assert!(
        (forward[1][3] - reversed[2][3]).norm() < SV_EPS,
        "{} against {}",
        forward[1][3],
        reversed[2][3]
    );
    assert!(forward[1][3].im.abs() > 0.1, "the phase has to be visible");
}

// A tensor product names each qubit once, and an identity factor is no
// exception: it reads the same qubit another factor already claimed.
#[test]
fn a_tensor_product_naming_one_qubit_twice_is_rejected() {
    for body in [
        "#pragma braket result expectation z(q[0]) @ x(q[0])",
        "#pragma braket result expectation i(q[0]) @ z(q[0])",
        "#pragma braket result variance z(q[0]) @ z(q[1]) @ y(q[1])",
    ] {
        let text = source(2, &format!("h q[0];\n{body}"));
        let err = openqasm::parse_braket(&text)
            .err()
            .unwrap_or_else(|| panic!("`{body}` should not parse"));
        assert!(format!("{err}").contains("twice"), "`{body}`: got {err}");
    }
}

// An exact result reads one output state, which a measurement, a reset or a
// conditional leaves the circuit without. Braket rejects the same programs at
// zero shots, and answering from one collapsed branch would be a wrong number
// rather than a missing one.
#[test]
fn an_exact_request_declines_a_circuit_that_is_not_unitary() {
    for body in [
        "bit[1] c;\nh q[0];\nc[0] = measure q[0];",
        "h q[0];\nreset q[0];",
    ] {
        let err = evaluate_err(2, &format!("{body}\n#pragma braket result probability all"));
        assert!(
            matches!(err, PrismError::IncompatibleBackend { .. }),
            "`{body}`: got {err}"
        );
    }
}

// A `state_vector` request beside a marginal makes the export the cheaper
// source for it, so the two routes have to agree entry for entry.
#[test]
fn a_marginal_agrees_whether_or_not_a_state_export_is_at_hand() {
    for targets in [
        "q[0]",
        "q[1]",
        "q[2]",
        "q[0], q[1]",
        "q[2], q[0]",
        "q[0], q[1], q[2]",
        "q[2], q[1], q[0]",
    ] {
        for kind in ["probability", "density_matrix"] {
            let request = format!("#pragma braket result {kind} {targets}");
            let alone = evaluate(
                3,
                &format!(
                    "{FIXTURE}
{request}"
                ),
            );
            let beside = evaluate(
                3,
                &format!(
                    "{FIXTURE}
{request}
#pragma braket result state_vector"
                ),
            );
            match (&alone[0], &beside[0]) {
                (ResultValue::Probability(a), ResultValue::Probability(b)) => {
                    common::assert_probs_close(a, b, SV_EPS, &format!("{kind} {targets}"));
                }
                (ResultValue::DensityMatrix(a), ResultValue::DensityMatrix(b)) => {
                    for (row, (left, right)) in a.iter().zip(b).enumerate() {
                        for (column, (left, right)) in left.iter().zip(right).enumerate() {
                            assert!(
                                (left - right).norm() < SV_EPS,
                                "{kind} {targets} ({row}, {column}): {left} against {right}"
                            );
                        }
                    }
                }
                other => panic!("{kind} {targets} gave {other:?}"),
            }
        }
    }
}

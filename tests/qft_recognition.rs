//! Expanded QFTs folded into `QftBlock`: the routed statevector run must match the
//! same gates applied one at a time, compared as complex amplitudes from random
//! input states, and sequences that are not an exact QFT must stay as gates.

use std::borrow::Cow;
use std::f64::consts::PI;

use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::fusion::fuse_circuit;
use prism_q::circuit::{
    Circuit, Instruction, expand_qft_blocks, openqasm, qasm_export, recognize_qft_blocks,
};
use prism_q::circuits::phase_estimation_circuit;
use prism_q::gates::Gate;
use prism_q::sim;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 42;
const EPS: f64 = 1e-10;

#[derive(Clone, Copy, Debug)]
struct Form {
    inverse: bool,
    swaps: bool,
    big_endian: bool,
}

const FORMS: [Form; 8] = {
    let mut forms = [Form {
        inverse: false,
        swaps: false,
        big_endian: false,
    }; 8];
    let mut i = 0;
    while i < 8 {
        forms[i] = Form {
            inverse: i & 1 != 0,
            swaps: i & 2 != 0,
            big_endian: i & 4 != 0,
        };
        i += 1;
    }
    forms
};

fn block(start: usize, num: usize, form: Form) -> Gate {
    Gate::QftBlock {
        start: start as u8,
        num: num as u8,
        inverse: form.inverse,
        swaps: form.swaps,
        big_endian: form.big_endian,
    }
}

fn with_block(n: usize, start: usize, num: usize, form: Form) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let targets: Vec<usize> = (start..start + num).collect();
    c.add_gate(block(start, num, form), &targets);
    c
}

fn expanded(n: usize, start: usize, num: usize, form: Form) -> Circuit {
    expand_qft_blocks(&with_block(n, start, num, form)).into_owned()
}

/// The gate order Qiskit's QFT decomposes to: `h` on the top qubit, then `cp` from
/// it to each lower qubit nearest first, then the swaps.
fn qiskit_order(n: usize, num: usize, inverse: bool, swaps: bool) -> Circuit {
    let mut gates: Vec<(Gate, Vec<usize>)> = Vec::new();
    for j in (0..num).rev() {
        gates.push((Gate::H, vec![j]));
        for k in (0..j).rev() {
            gates.push((Gate::cphase(PI / (1u64 << (j - k)) as f64), vec![j, k]));
        }
    }
    if swaps {
        for i in 0..num / 2 {
            gates.push((Gate::Swap, vec![i, num - 1 - i]));
        }
    }
    if inverse {
        gates.reverse();
        for (gate, _) in &mut gates {
            *gate = gate.inverse();
        }
    }
    let mut c = Circuit::new(n, 0);
    for (gate, targets) in gates {
        c.add_gate(gate, &targets);
    }
    c
}

/// The order textbooks write with `q[0]` the most significant bit: `h q[j]`, then
/// `cp(pi/2^(k-j)) q[k], q[j]` for each later `k`, then the swaps innermost first.
fn msb_first_order(n: usize, num: usize, swaps: bool) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in 0..num {
        c.add_gate(Gate::H, &[j]);
        for k in j + 1..num {
            c.add_gate(Gate::cphase(PI / (1u64 << (k - j)) as f64), &[k, j]);
        }
    }
    if swaps {
        for i in (0..num / 2).rev() {
            c.add_gate(Gate::Swap, &[num - 1 - i, i]);
        }
    }
    c
}

fn random_state(n: usize, seed: u64) -> Vec<Complex64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state: Vec<Complex64> = (0..1usize << n)
        .map(|_| {
            Complex64::new(
                rng.random::<f64>() * 2.0 - 1.0,
                rng.random::<f64>() * 2.0 - 1.0,
            )
        })
        .collect();
    let norm = state.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    for a in &mut state {
        *a /= norm;
    }
    state
}

/// Apply each instruction as written, outside the routed pipeline.
fn gate_by_gate(circuit: &Circuit, init: &[Complex64]) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(SEED);
    backend
        .init_from_state(init.to_vec(), circuit.num_classical_bits)
        .unwrap();
    for inst in &circuit.instructions {
        backend.apply(inst).unwrap();
    }
    backend.state_vector().to_vec()
}

/// The routed run: recognition, fusion, then the kernels.
fn routed(circuit: &Circuit, init: &[Complex64]) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on_state(&mut backend, circuit, init).unwrap();
    backend.state_vector().to_vec()
}

#[track_caller]
fn assert_close(actual: &[Complex64], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let worst = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).norm())
        .fold(0.0f64, f64::max);
    assert!(worst < EPS, "{label}: max amplitude error {worst:e}");
}

#[track_caller]
fn assert_routed_matches_gates(circuit: &Circuit, seeds: u64, label: &str) {
    for s in 0..seeds {
        let init = random_state(circuit.num_qubits, SEED + s);
        assert_close(
            &routed(circuit, &init),
            &gate_by_gate(circuit, &init),
            &format!("{label}, seed {}", SEED + s),
        );
    }
}

fn blocks(circuit: &Circuit) -> Vec<Gate> {
    circuit
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Gate {
                gate: gate @ Gate::QftBlock { .. },
                ..
            } => Some(gate.clone()),
            _ => None,
        })
        .collect()
}

#[track_caller]
fn assert_unrecognized(circuit: &Circuit, label: &str) {
    assert!(
        matches!(recognize_qft_blocks(circuit), Cow::Borrowed(_)),
        "{label}: recognized {:?}",
        blocks(&recognize_qft_blocks(circuit))
    );
}

#[test]
fn every_form_folds_and_matches_its_gates() {
    for n in 1..=12 {
        for form in FORMS {
            let circuit = expanded(n, 0, n, form);
            let expected = if n >= 2 {
                vec![block(0, n, form)]
            } else {
                vec![]
            };
            assert_eq!(
                blocks(&recognize_qft_blocks(&circuit)),
                expected,
                "n {n}, {form:?}"
            );
            assert_routed_matches_gates(&circuit, 3, &format!("n {n}, {form:?}"));
        }
    }
}

#[test]
fn qiskit_and_msb_first_orders_fold() {
    for n in 2..=12 {
        for (inverse, swaps) in [(false, true), (false, false), (true, true), (true, false)] {
            let circuit = qiskit_order(n, n, inverse, swaps);
            let form = Form {
                inverse,
                swaps,
                big_endian: false,
            };
            assert_eq!(
                blocks(&recognize_qft_blocks(&circuit)),
                vec![block(0, n, form)],
                "qiskit order, n {n}, {form:?}"
            );
            assert_routed_matches_gates(&circuit, 2, &format!("qiskit order, n {n}, {form:?}"));
        }
        for swaps in [true, false] {
            let circuit = msb_first_order(n, n, swaps);
            let form = Form {
                inverse: false,
                swaps,
                big_endian: true,
            };
            assert_eq!(
                blocks(&recognize_qft_blocks(&circuit)),
                vec![block(0, n, form)],
                "msb-first order, n {n}, swaps {swaps}"
            );
            assert_routed_matches_gates(&circuit, 2, &format!("msb-first, n {n}, swaps {swaps}"));
        }
    }
}

#[test]
fn sub_ranges_fold_only_from_qubit_zero_at_ten_or_more() {
    let n = 12;
    for form in FORMS {
        let low = expanded(n, 0, 10, form);
        assert_eq!(
            blocks(&recognize_qft_blocks(&low)),
            vec![block(0, 10, form)]
        );
        assert_routed_matches_gates(&low, 2, &format!("0..10 of 12, {form:?}"));

        for (start, num) in [(0, 9), (2, 10), (1, 11), (3, 5)] {
            let circuit = expanded(n, start, num, form);
            assert_unrecognized(&circuit, &format!("{start}..{} of 12", start + num));
            assert_routed_matches_gates(&circuit, 1, &format!("{start}..+{num}, {form:?}"));
        }
    }
}

#[test]
fn qft_embedded_among_other_gates_folds() {
    let n = 12;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut noise = |c: &mut Circuit| {
        for _ in 0..20 {
            let a = rng.random_range(0..n);
            let b = (a + 1 + rng.random_range(0..n - 1)) % n;
            match rng.random_range(0..4) {
                0 => c.add_gate(Gate::Ry(rng.random::<f64>()), &[a]),
                1 => c.add_gate(Gate::Cx, &[a, b]),
                2 => c.add_gate(Gate::cphase(rng.random::<f64>()), &[a, b]),
                _ => c.add_gate(Gate::H, &[a]),
            };
        }
    };
    for form in FORMS {
        let mut circuit = Circuit::new(n, 0);
        noise(&mut circuit);
        circuit
            .instructions
            .extend(expanded(n, 0, 11, form).instructions);
        noise(&mut circuit);
        circuit
            .instructions
            .extend(expanded(n, 0, n, form).instructions);
        noise(&mut circuit);
        assert_eq!(
            blocks(&recognize_qft_blocks(&circuit)),
            vec![block(0, 11, form), block(0, n, form)],
            "{form:?}"
        );
        assert_routed_matches_gates(&circuit, 2, &format!("embedded, {form:?}"));
    }
}

#[test]
fn phase_estimation_inverse_qft_folds() {
    for n in [11, 12, 13] {
        let circuit = phase_estimation_circuit(n);
        let form = Form {
            inverse: true,
            swaps: true,
            big_endian: false,
        };
        assert_eq!(
            blocks(&recognize_qft_blocks(&circuit)),
            vec![block(0, n - 1, form)]
        );
        assert_routed_matches_gates(&circuit, 2, &format!("qpe {n}"));
    }
}

// The kernel runs every block that starts at qubit 0, including widths recognition
// never emits; past 15 qubits it takes the parallel and high-stride paths.
#[test]
fn native_kernel_matches_expansion_on_every_chunk_width() {
    for (n, num) in [
        (5, 2),
        (8, 3),
        (12, 7),
        (16, 9),
        (16, 16),
        (17, 14),
        (17, 16),
    ] {
        for form in FORMS {
            let init = random_state(n, SEED + num as u64);
            assert_close(
                &gate_by_gate(&with_block(n, 0, num, form), &init),
                &gate_by_gate(&expanded(n, 0, num, form), &init),
                &format!("0..{num} of {n}, {form:?}"),
            );
        }
    }
}

#[test]
fn blocks_off_qubit_zero_match_their_expansion() {
    for form in FORMS {
        let init = random_state(9, SEED);
        assert_close(
            &gate_by_gate(&with_block(9, 3, 5, form), &init),
            &gate_by_gate(&expanded(9, 3, 5, form), &init),
            &format!("3..8 of 9, {form:?}"),
        );
    }
}

#[test]
fn block_inverse_undoes_the_block() {
    for form in FORMS {
        let mut circuit = with_block(10, 0, 10, form);
        let targets: Vec<usize> = (0..10).collect();
        circuit.add_gate(block(0, 10, form).inverse(), &targets);
        let init = random_state(10, SEED);
        assert_close(
            &gate_by_gate(&circuit, &init),
            &init,
            &format!("{form:?} then its inverse"),
        );
    }
}

#[test]
fn one_wrong_angle_is_not_recognized() {
    for form in FORMS {
        let mut circuit = expanded(8, 0, 8, form);
        // The narrowest column's phase sits in every QFT the stream could hold.
        let index = circuit
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, inst)| {
                matches!(inst, Instruction::Gate { gate: Gate::Cu(_), targets }
                    if targets.contains(&if form.big_endian { 7 } else { 0 })
                        && targets.contains(&if form.big_endian { 6 } else { 1 }))
            })
            .map(|(i, _)| i)
            .next()
            .unwrap();
        let sign = if form.inverse { -1.0 } else { 1.0 };
        let Instruction::Gate { gate, .. } = &mut circuit.instructions[index] else {
            unreachable!()
        };
        *gate = Gate::cphase(sign * (PI / 2.0 + 1e-9));
        assert_unrecognized(&circuit, &format!("{form:?}"));
        assert_routed_matches_gates(&circuit, 1, &format!("wrong angle, {form:?}"));
    }
}

#[test]
fn a_gate_inside_the_qft_is_not_recognized() {
    for form in FORMS {
        let mut circuit = expanded(8, 0, 8, form);
        let gates = circuit.instructions.len();
        let at = if form.inverse { gates - 3 } else { 3 };
        circuit.instructions.insert(
            at,
            Instruction::Gate {
                gate: Gate::T,
                targets: prism_q::circuit::smallvec![if form.big_endian { 0 } else { 7 }],
            },
        );
        assert_unrecognized(&circuit, &format!("{form:?}"));
        assert_routed_matches_gates(&circuit, 1, &format!("interleaved, {form:?}"));
    }
}

#[test]
fn a_truncated_qft_is_not_recognized() {
    for form in FORMS {
        let mut circuit = expanded(8, 0, 8, form);
        // The last H of a forward column walk, or the first of an inverse one,
        // after any leading swaps.
        let mut h = circuit
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, inst)| matches!(inst, Instruction::Gate { gate: Gate::H, .. }))
            .map(|(i, _)| i);
        let index = if form.inverse {
            h.next().unwrap()
        } else {
            h.next_back().unwrap()
        };
        circuit.instructions.remove(index);
        assert_unrecognized(&circuit, &format!("{form:?}"));
        assert_routed_matches_gates(&circuit, 1, &format!("truncated, {form:?}"));
    }
}

#[test]
fn non_contiguous_qubits_are_not_recognized() {
    let relabel = |circuit: &Circuit, n: usize, map: &dyn Fn(usize) -> usize| {
        let mut out = Circuit::new(n, 0);
        for inst in &circuit.instructions {
            let Instruction::Gate { gate, targets } = inst else {
                unreachable!()
            };
            let targets: Vec<usize> = targets.iter().map(|&q| map(q)).collect();
            out.add_gate(gate.clone(), &targets);
        }
        out
    };
    for form in FORMS {
        let qft = expanded(10, 0, 10, form);
        assert_unrecognized(
            &relabel(&qft, 20, &|q| 2 * q),
            &format!("even qubits, {form:?}"),
        );
        let shuffled = relabel(&qft, 10, &|q| (3 * q) % 10);
        assert_unrecognized(&shuffled, &format!("shuffled qubits, {form:?}"));
        assert_routed_matches_gates(&shuffled, 1, &format!("shuffled, {form:?}"));
    }
}

fn qiskit_qasm(n: usize, inverse: bool) -> String {
    let sign = if inverse { "-" } else { "" };
    let mut gates = Vec::new();
    for j in (0..n).rev() {
        gates.push(format!("h q[{j}];"));
        for k in (0..j).rev() {
            gates.push(format!("cp({sign}pi/{}) q[{j}],q[{k}];", 1u64 << (j - k)));
        }
    }
    for i in 0..n / 2 {
        gates.push(format!("swap q[{i}],q[{}];", n - 1 - i));
    }
    if inverse {
        gates.reverse();
    }
    format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\n{}\n",
        gates.join("\n")
    )
}

#[test]
fn qasm_qft_reaches_the_fused_stream_as_a_block_and_round_trips() {
    for inverse in [false, true] {
        let n = 12;
        let parsed = openqasm::parse(&qiskit_qasm(n, inverse)).unwrap();
        let form = Form {
            inverse,
            swaps: true,
            big_endian: false,
        };
        let recognized = recognize_qft_blocks(&parsed);
        let fused = fuse_circuit(&recognized, true);
        assert_eq!(blocks(&fused), vec![block(0, n, form)], "inverse {inverse}");
        assert_eq!(fused.instructions.len(), 1, "inverse {inverse}");

        let exported = qasm_export::to_qasm3(&recognized).unwrap();
        let reparsed = openqasm::parse(&exported).unwrap();
        assert_eq!(
            blocks(&recognize_qft_blocks(&reparsed)),
            vec![block(0, n, form)],
            "inverse {inverse}"
        );
        assert_eq!(
            qasm_export::to_qasm3(&reparsed).unwrap(),
            exported,
            "inverse {inverse}"
        );
        assert_routed_matches_gates(&parsed, 2, &format!("qasm, inverse {inverse}"));
    }
}

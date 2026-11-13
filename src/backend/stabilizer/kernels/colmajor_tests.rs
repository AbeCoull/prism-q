use super::*;
use crate::backend::Backend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::stabilizer::kernels::batch::apply_gates_word_batch_with;
use crate::circuit::{Circuit, ClassicalCondition, SmallVec};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[test]
fn transpose64_moves_bit_j_of_word_i_to_bit_i_of_word_j() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let input: [u64; 64] = std::array::from_fn(|_| rng.random());
    let mut out = input;
    transpose64(&mut out);
    for (i, &row) in input.iter().enumerate() {
        for (j, &col) in out.iter().enumerate() {
            assert_eq!((col >> i) & 1, (row >> j) & 1, "({i}, {j})");
        }
    }
}

const ONE_QUBIT: [Gate; 9] = [
    Gate::H,
    Gate::S,
    Gate::Sdg,
    Gate::X,
    Gate::Y,
    Gate::Z,
    Gate::SX,
    Gate::SXdg,
    Gate::Id,
];
const TWO_QUBIT: [Gate; 3] = [Gate::Cx, Gate::Cz, Gate::Swap];

fn conditional(condition: ClassicalCondition, gate: Gate, targets: &[usize]) -> Instruction {
    Instruction::Conditional {
        condition,
        gate,
        targets: SmallVec::from_slice(targets),
    }
}

fn random_clifford_with_measures(n: usize, layers: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, n);
    for layer in 0..layers {
        for _ in 0..2 * n {
            match rng.random_range(0..8) {
                0 | 1 => {
                    let q = rng.random_range(0..n);
                    c.add_gate(
                        ONE_QUBIT[rng.random_range(0..ONE_QUBIT.len())].clone(),
                        &[q],
                    );
                }
                2 => {
                    let q = rng.random_range(0..n);
                    let bit = rng.random_range(0..n);
                    c.instructions.push(conditional(
                        ClassicalCondition::BitIsOne(bit),
                        Gate::H,
                        &[q],
                    ));
                }
                _ => {
                    let a = rng.random_range(0..n);
                    let mut b = rng.random_range(0..n);
                    while b == a {
                        b = rng.random_range(0..n);
                    }
                    c.add_gate(
                        TWO_QUBIT[rng.random_range(0..TWO_QUBIT.len())].clone(),
                        &[a, b],
                    );
                }
            }
        }
        match layer % 3 {
            0 => {
                let q = rng.random_range(0..n);
                c.add_measure(q, q);
            }
            1 => {
                for q in (0..n).step_by(7) {
                    c.add_measure(q, q);
                }
            }
            _ => c.add_reset(rng.random_range(0..n)),
        }
    }
    c
}

fn fresh(n: usize, cbits: usize, lazy: bool) -> StabilizerBackend {
    let mut b = if lazy {
        StabilizerBackend::new_lazy(42)
    } else {
        StabilizerBackend::new(42)
    };
    b.init(n, cbits).unwrap();
    b
}

fn assert_same(a: &StabilizerBackend, b: &StabilizerBackend, what: &str) {
    assert_eq!(a.gate_row_start, b.gate_row_start, "{what}: row start");
    assert!(a.xz == b.xz, "{what}: tableau bits differ");
    assert!(a.phase == b.phase, "{what}: phases differ");
    assert_eq!(
        a.classical_bits, b.classical_bits,
        "{what}: outcomes differ"
    );
}

fn first_run_takes_columns(circuit: &Circuit) -> bool {
    let n = circuit.num_qubits;
    let (_, cross) = col_run(&circuit.instructions, true);
    columns_enabled(n) && run_takes_columns(cross, n.div_ceil(64))
}

fn columns_against_rows(circuit: &Circuit, lazy: bool, what: &str) {
    let (n, cbits) = (circuit.num_qubits, circuit.num_classical_bits);
    let mut rows = fresh(n, cbits, lazy);
    rows.apply_instructions_word_batch_with(&circuit.instructions, false)
        .unwrap();
    let mut cols = fresh(n, cbits, lazy);
    cols.apply_instructions_word_batch_with(&circuit.instructions, true)
        .unwrap();
    assert_same(&rows, &cols, what);
}

#[cfg(feature = "parallel")]
fn on_threads(threads: usize, f: impl FnOnce() + Send) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f);
}

#[cfg(not(feature = "parallel"))]
fn on_threads(_threads: usize, f: impl FnOnce() + Send) {
    f();
}

#[test]
fn columns_match_rows_on_the_random_pairs_bench_circuits() {
    for threads in [1, 4] {
        on_threads(threads, || {
            for (n, depth) in [(500, 10), (1000, 10), (500, 50), (1000, 50), (500, 100)] {
                let circuit = crate::circuits::clifford_random_pairs(n, depth, 0xDEAD_BEEF);
                assert!(first_run_takes_columns(&circuit), "n={n} d={depth}");
                for lazy in [true, false] {
                    let what = format!("random_pairs n={n} d={depth} lazy={lazy} t={threads}");
                    columns_against_rows(&circuit, lazy, &what);
                }
            }
        });
    }
}

#[test]
fn columns_match_rows_with_interleaved_measurements_resets_and_conditionals() {
    for threads in [1, 4] {
        on_threads(threads, || {
            for (n, seed) in [(256, 1u64), (300, 2), (511, 3), (640, 4)] {
                let circuit = random_clifford_with_measures(n, 6, seed);
                assert!(first_run_takes_columns(&circuit), "n={n}");
                for lazy in [true, false] {
                    let what = format!("n={n} seed={seed} lazy={lazy} t={threads}");
                    columns_against_rows(&circuit, lazy, &what);
                }
            }
        });
    }
}

#[test]
fn columns_match_rows_after_a_measured_prefix() {
    let n = 400;
    let mut circuit = random_clifford_with_measures(n, 3, 9);
    let tail = crate::circuits::clifford_random_pairs(n, 20, 7);
    circuit.instructions.extend(tail.instructions);
    for q in 0..n {
        circuit.add_measure(q, q);
    }
    for lazy in [true, false] {
        columns_against_rows(&circuit, lazy, &format!("prefix lazy={lazy}"));
    }
}

// One Clifford run with exactly `cross` cross-word gates among same-word gates,
// conditionals on a zero bit (one taken, one skipped) and a barrier, then a
// measurement so the run ends before the end of the call.
fn run_with_cross_word_gates(n: usize, cross: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 2);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for k in 0..cross {
        let w = k % (n / 64);
        let a = w * 64 + rng.random_range(0..64);
        let b = ((w + 1) * 64 + rng.random_range(0..64)) % n;
        for _ in 0..8 {
            let base = rng.random_range(0..n / 64) * 64;
            let (x, y) = (
                base + rng.random_range(0..32),
                base + 32 + rng.random_range(0..32),
            );
            c.add_gate(TWO_QUBIT[rng.random_range(0..3)].clone(), &[x, y]);
            c.add_gate(ONE_QUBIT[rng.random_range(0..9)].clone(), &[x]);
        }
        c.add_gate(TWO_QUBIT[k % 3].clone(), &[a, b]);
    }
    c.instructions
        .push(conditional(ClassicalCondition::BitIsZero(0), Gate::S, &[1]));
    c.add_barrier(&[0, 1]);
    c.instructions
        .push(conditional(ClassicalCondition::BitIsOne(0), Gate::H, &[2]));
    c.add_gate(Gate::Cz, &[3, n - 1]);
    c.add_measure(0, 1);
    c.add_gate(Gate::Cx, &[0, n - 1]);
    c
}

#[test]
fn switching_point_takes_columns_at_one_cross_word_gate_per_word() {
    assert!(!columns_enabled(MIN_QUBITS_FOR_COLS - 1));
    assert!(columns_enabled(MIN_QUBITS_FOR_COLS));
    for n in [256usize, 320, 383] {
        let nw = n.div_ceil(64);
        // The builder adds one trailing cross-word CZ to the run.
        for cross in [nw - 2, nw - 1] {
            let circuit = run_with_cross_word_gates(n, cross, n as u64 + cross as u64);
            let (len, counted) = col_run(&circuit.instructions, true);
            assert_eq!(counted, cross + 1, "n={n}");
            assert_eq!(len, circuit.instructions.len() - 2, "n={n}");
            assert_eq!(run_takes_columns(counted, nw), cross + 1 == nw, "n={n}");
            for lazy in [true, false] {
                let what = format!("n={n} cross={counted} lazy={lazy}");
                columns_against_rows(&circuit, lazy, &what);
                let mut one_by_one = fresh(n, 2, lazy);
                for instruction in &circuit.instructions {
                    one_by_one.apply(instruction).unwrap();
                }
                let mut cols = fresh(n, 2, lazy);
                cols.apply_instructions_word_batch_with(&circuit.instructions, true)
                    .unwrap();
                assert_same(&one_by_one, &cols, &format!("{what} against per gate"));
            }
        }
    }
}

#[test]
fn gates_only_columns_match_rows_and_per_gate_dispatch() {
    for (n, depth) in [(256, 8), (300, 20), (700, 12)] {
        let mut circuit = crate::circuits::clifford_random_pairs(n, depth, 0xDEAD_BEEF);
        circuit.instructions.insert(
            n,
            Instruction::Barrier {
                qubits: SmallVec::from_slice(&[0, 1]),
            },
        );
        let mut rows = fresh(n, 0, true);
        apply_gates_word_batch_with(&mut rows, &circuit.instructions, false).unwrap();
        let mut cols = fresh(n, 0, true);
        apply_gates_word_batch_with(&mut cols, &circuit.instructions, true).unwrap();
        let mut one_by_one = fresh(n, 0, true);
        for instruction in &circuit.instructions {
            one_by_one.apply(instruction).unwrap();
        }
        assert!(first_run_takes_columns(&circuit), "n={n}");
        assert_same(&rows, &cols, &format!("gates only n={n}"));
        assert_same(
            &one_by_one,
            &cols,
            &format!("gates only n={n} against per gate"),
        );
    }
}

#[test]
fn gates_only_run_ends_at_a_conditional_the_batch_skips() {
    let n = 320;
    let mut circuit = crate::circuits::clifford_random_pairs(n, 6, 5);
    let at = circuit.instructions.len() / 2;
    circuit.instructions.insert(
        at,
        conditional(ClassicalCondition::BitIsZero(0), Gate::H, &[7]),
    );
    assert_eq!(col_run(&circuit.instructions, false).0, at);
    let mut rows = fresh(n, 1, false);
    apply_gates_word_batch_with(&mut rows, &circuit.instructions, false).unwrap();
    let mut cols = fresh(n, 1, false);
    apply_gates_word_batch_with(&mut cols, &circuit.instructions, true).unwrap();
    assert_same(&rows, &cols, "gates only with a conditional");
}

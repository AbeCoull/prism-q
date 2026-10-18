use crate::backend::Backend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::circuit::Circuit;
use crate::gates::Gate;
use crate::sim;

fn run_and_count_zero(circuit: &Circuit) -> usize {
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, circuit).unwrap();
    b.classical_results().iter().filter(|x| !**x).count()
}

// The scalar CX row update, one gate at a time, that the word formula
// has to reproduce bit for bit including the phase.
fn cx_chain_reference(xw: &mut u64, zw: &mut u64, p: &mut bool, first: u8, last: u8) {
    for k in first..last {
        let (c, t) = (k, k + 1);
        let xa = (*xw >> c) & 1;
        let za = (*zw >> c) & 1;
        let xb = (*xw >> t) & 1;
        let zb = (*zw >> t) & 1;
        *p ^= (xa & zb & (xb ^ za ^ 1)) == 1;
        if xa == 1 {
            *xw ^= 1u64 << t;
        }
        if zb == 1 {
            *zw ^= 1u64 << c;
        }
    }
}

#[test]
fn cx_chain_word_formula_matches_the_scalar_chain() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..4000 {
        let a = (next() % 64) as u8;
        let b = (next() % 64) as u8;
        let (first, last) = if a <= b { (a, b) } else { (b, a) };
        let (x0, z0, p0) = (next(), next(), next() & 1 == 1);
        let (mut xw, mut zw, mut p) = (x0, z0, p0);
        let (mut xr, mut zr, mut pr) = (x0, z0, p0);
        super::apply_cx_chain(&mut xw, &mut zw, &mut p, first, last);
        cx_chain_reference(&mut xr, &mut zr, &mut pr, first, last);
        assert_eq!(
            (xw, zw, p),
            (xr, zr, pr),
            "range {first}..={last} on {x0:#x} {z0:#x}"
        );
    }
}

#[test]
fn cx_chain_run_matches_per_gate_rows() {
    let n = 500;
    let prefix = crate::circuits::clifford_random_pairs(n, 4, 0);
    let mut base = StabilizerBackend::new(0);
    sim::run_on(&mut base, &prefix).unwrap();
    for (start, end) in [
        (3usize, 498usize),
        (0, 64),
        (63, 65),
        (60, 200),
        (128, 192),
        (100, 127),
    ] {
        let mut a = base.clone();
        a.gate_rows_view().apply_cx_chain_run(start, end);
        let mut b = base.clone();
        let mut chain = Circuit::new(n, 0);
        for k in start..end {
            chain.add_gate(Gate::Cx, &[k, k + 1]);
        }
        for instr in &chain.instructions {
            b.apply(instr).unwrap();
        }
        let (xa, pa) = a.raw_tableau();
        let (xb, pb) = b.raw_tableau();
        let stride = 2 * n.div_ceil(64);
        let bad: Vec<usize> = (0..2 * n)
            .filter(|&r| xa[r * stride..(r + 1) * stride] != xb[r * stride..(r + 1) * stride])
            .collect();
        let badp: Vec<usize> = (0..2 * n).filter(|&r| pa[r] != pb[r]).collect();
        assert!(
            bad.is_empty() && badp.is_empty(),
            "chain {start}..{end}: {} rows differ {:?}, {} phases differ {:?}",
            bad.len(),
            &bad[..bad.len().min(8)],
            badp.len(),
            &badp[..badp.len().min(8)]
        );
    }
}

#[test]
fn cx_chain_len_stops_where_the_chain_does() {
    let mut c = Circuit::new(8, 0);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Cx, &[3, 4]);
    c.add_gate(Gate::Cx, &[4, 5]);
    c.add_gate(Gate::Cx, &[6, 7]);
    assert_eq!(super::cx_chain_len(&c.instructions), 3);
    assert_eq!(super::cx_chain_len(&c.instructions[3..]), 1);
    let mut d = Circuit::new(8, 0);
    d.add_gate(Gate::Cx, &[3, 2]);
    d.add_gate(Gate::H, &[0]);
    assert_eq!(super::cx_chain_len(&d.instructions), 0);
    assert_eq!(super::cx_chain_len(&d.instructions[1..]), 0);
}

fn assert_runs(circuit: &Circuit) {
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, circuit).unwrap();
}

#[test]
fn batch_path_193q_basic_clifford() {
    let n = 200;
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n {
        c.add_gate(Gate::S, &[q]);
        c.add_gate(Gate::Z, &[q]);
    }
    assert_runs(&c);
}

#[test]
fn batch_path_overlapping_1q_targets_segment_flush() {
    let n = 200;
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::S, &[q]);
        c.add_gate(Gate::Sdg, &[q]);
    }
    assert_runs(&c);
}

#[test]
fn batch_path_cross_word_two_qubit_gates() {
    let n = 200;
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 128]);
    c.add_gate(Gate::Cz, &[5, 192]);
    c.add_gate(Gate::Swap, &[60, 199]);
    assert_runs(&c);
}

#[test]
fn batch_path_all_1q_gate_types() {
    let n = 200;
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::S, &[q]);
        c.add_gate(Gate::Sdg, &[q]);
        c.add_gate(Gate::X, &[q]);
        c.add_gate(Gate::Y, &[q]);
        c.add_gate(Gate::Z, &[q]);
        c.add_gate(Gate::SX, &[q]);
        c.add_gate(Gate::SXdg, &[q]);
        c.add_gate(Gate::Id, &[q]);
    }
    assert_runs(&c);
}

#[test]
fn batch_path_zero_state_unchanged_after_identity_loop() {
    let n = 200;
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::Id, &[q]);
    }
    for q in 0..n {
        c.instructions.push(crate::circuit::Instruction::Measure {
            qubit: q,
            classical_bit: q,
        });
    }
    let zeros = run_and_count_zero(&c);
    assert_eq!(zeros, n);
}

fn add_measure_all(c: &mut Circuit) {
    let n = c.num_qubits;
    for q in 0..n {
        c.instructions.push(crate::circuit::Instruction::Measure {
            qubit: q,
            classical_bit: q,
        });
    }
}

fn sequential_reference_bits(circuit: &Circuit, seed: u64) -> Vec<bool> {
    let mut b = StabilizerBackend::new(seed);
    b.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    for instr in &circuit.instructions {
        b.apply(instr).unwrap();
    }
    b.classical_results().to_vec()
}

fn batched_bits(circuit: &Circuit, seed: u64) -> Vec<bool> {
    let mut b = StabilizerBackend::new(seed);
    b.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    b.apply_instructions(&circuit.instructions).unwrap();
    b.classical_results().to_vec()
}

#[test]
fn measure_batch_matches_sequential_ghz_997q() {
    let n = 997;
    let mut c = Circuit::new(n, n);
    c.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    add_measure_all(&mut c);
    assert_eq!(batched_bits(&c, 42), sequential_reference_bits(&c, 42));
}

#[test]
fn measure_batch_matches_sequential_random_clifford_300q() {
    let n = 300;
    let mut c = crate::circuits::clifford_heavy_circuit(n, 8, 42);
    c.num_classical_bits = n;
    add_measure_all(&mut c);
    assert_eq!(batched_bits(&c, 42), sequential_reference_bits(&c, 42));
}

#[test]
fn measure_batch_matches_sequential_interleaved_measures() {
    let n = 300;
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..64 {
        c.instructions.push(crate::circuit::Instruction::Measure {
            qubit: q,
            classical_bit: q,
        });
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    add_measure_all(&mut c);
    assert_eq!(batched_bits(&c, 42), sequential_reference_bits(&c, 42));
}

#[test]
fn measure_batch_splits_on_repeated_qubit() {
    let n = 300;
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    add_measure_all(&mut c);
    for q in 0..32 {
        c.instructions.push(crate::circuit::Instruction::Measure {
            qubit: q,
            classical_bit: q,
        });
    }
    assert_eq!(batched_bits(&c, 42), sequential_reference_bits(&c, 42));
}

#[test]
fn measure_batch_splits_on_shared_classical_bit() {
    let n = 300;
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..32 {
        c.instructions.push(crate::circuit::Instruction::Measure {
            qubit: q,
            classical_bit: 0,
        });
    }
    add_measure_all(&mut c);
    assert_eq!(batched_bits(&c, 42), sequential_reference_bits(&c, 42));
}

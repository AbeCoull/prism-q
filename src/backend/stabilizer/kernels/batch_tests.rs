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

fn make_cross_word_gate(kind: u8, q0: usize, q1: usize) -> super::CrossWordGate {
    let w0 = q0 / 64;
    let w1 = q1 / 64;
    assert_ne!(w0, w1, "cross-word gate requires distinct words");
    super::CrossWordGate {
        kind,
        w0: w0 as u16,
        w1: w1 as u16,
        b0: (q0 % 64) as u8,
        b1: (q1 % 64) as u8,
    }
}

fn apply_cross_word_per_gate(b: &mut StabilizerBackend, gates: &[super::CrossWordGate]) {
    for g in gates {
        let q0 = g.w0 as usize * 64 + g.b0 as usize;
        let q1 = g.w1 as usize * 64 + g.b1 as usize;
        let gate = match g.kind {
            super::BatchGate::CX => Gate::Cx,
            super::BatchGate::CZ => Gate::Cz,
            _ => Gate::Swap,
        };
        b.apply(&crate::circuit::Instruction::Gate {
            gate,
            targets: crate::circuit::SmallVec::from_slice(&[q0, q1]),
        })
        .unwrap();
    }
}

fn seed_dense_tableau_with_nonzero_tail(n: usize, lazy: bool, seed: u64) -> StabilizerBackend {
    let prefix = crate::circuits::clifford_random_pairs(n, 8, seed);
    let mut base = if lazy {
        StabilizerBackend::new_lazy(seed)
    } else {
        StabilizerBackend::new(seed)
    };
    sim::run_on(&mut base, &prefix).unwrap();

    // Populate non-zero bits and phase in the trailing rows (including the
    // scratch row at 2*n) so partial trailing <64-row blocks are exercised.
    let stride = 2 * n.div_ceil(64);
    let mut state = seed ^ 0xA5A5_5A5A_1234_5678u64;
    let mut next_u64 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state | 1
    };
    for r in [2 * n - 1, 2 * n] {
        for word in &mut base.xz[r * stride..(r + 1) * stride] {
            *word ^= next_u64();
        }
        base.phase[r] = true;
    }
    base
}

fn assert_pcc_matches_per_gate(
    base: &StabilizerBackend,
    gates: &[super::CrossWordGate],
    lazy: bool,
    what: &str,
) {
    let n = base.num_qubits();
    let stride = 2 * n.div_ceil(64);
    let mut a = base.clone();
    a.gate_rows_view().pcc_apply_cross_word(gates);

    let mut b = base.clone();
    apply_cross_word_per_gate(&mut b, gates);

    let (xa, pa) = a.raw_tableau();
    let (xb, pb) = b.raw_tableau();
    let (x_base, p_base) = base.raw_tableau();

    if lazy {
        assert_eq!(
            xa[..n * stride],
            x_base[..n * stride],
            "{what}: lazy mode modified destabilizer xz rows"
        );
        assert_eq!(
            pa[..n],
            p_base[..n],
            "{what}: lazy mode modified destabilizer phases"
        );
    }

    assert_eq!(xa, xb, "{what}: tableau xz mismatch");
    assert_eq!(pa, pb, "{what}: tableau phase mismatch");
}

#[test]
fn pcc_cross_word_matches_per_gate_across_par_threshold_and_threads() {
    let thread_counts: &[usize] = if cfg!(feature = "parallel") {
        &[1, 4]
    } else {
        &[1]
    };

    for &n in &[127usize, 128, 256, 320, 500] {
        for &lazy in &[false, true] {
            let base = seed_dense_tableau_with_nonzero_tail(n, lazy, n as u64 + 17);
            let last_w = (n - 1) / 64;
            let q_hi = n - 1;
            let gates = vec![
                make_cross_word_gate(super::BatchGate::CX, last_w * 64, 5),
                make_cross_word_gate(super::BatchGate::CZ, 17, 64 + 19),
                make_cross_word_gate(super::BatchGate::SWAP, q_hi, 2),
                make_cross_word_gate(super::BatchGate::CX, 64 + 1, 63),
                make_cross_word_gate(super::BatchGate::CZ, 2, last_w * 64),
                make_cross_word_gate(super::BatchGate::SWAP, 5, 64 + 1),
            ];

            let mut outputs = Vec::new();
            for &threads in thread_counts {
                let mut res = base.clone();
                on_threads(threads, || {
                    assert_pcc_matches_per_gate(
                        &base,
                        &gates,
                        lazy,
                        &format!("n={n} lazy={lazy} threads={threads}"),
                    );
                    res.gate_rows_view().pcc_apply_cross_word(&gates);
                });
                outputs.push(res);
            }
            for w in outputs.windows(2) {
                assert_eq!(w[0].raw_tableau(), w[1].raw_tableau());
            }
        }
    }
}

#[test]
fn pcc_cross_word_shared_word_multiple_qubits_preserves_untouched_bits() {
    let n = 320;
    // Word 0 has 5 cached qubits {0, 1, 7, 31, 63}; word 2 has 4 cached qubits
    // {128, 130, 159, 191}; word 3 is untouched; gates arrive unsorted with
    // mixed directions across CX, CZ, SWAP.
    let gates = vec![
        make_cross_word_gate(super::BatchGate::CX, 191, 63),
        make_cross_word_gate(super::BatchGate::CZ, 1, 130),
        make_cross_word_gate(super::BatchGate::SWAP, 256, 31),
        make_cross_word_gate(super::BatchGate::CX, 128, 7),
        make_cross_word_gate(super::BatchGate::CZ, 159, 0),
        make_cross_word_gate(super::BatchGate::SWAP, 64 + 9, 191),
        make_cross_word_gate(super::BatchGate::CX, 0, 64 + 40),
        make_cross_word_gate(super::BatchGate::CZ, 319, 159),
    ];

    for lazy in [false, true] {
        let base = seed_dense_tableau_with_nonzero_tail(n, lazy, 99);
        let thread_counts: &[usize] = if cfg!(feature = "parallel") {
            &[1, 4]
        } else {
            &[1]
        };
        for &threads in thread_counts {
            on_threads(threads, || {
                assert_pcc_matches_per_gate(
                    &base,
                    &gates,
                    lazy,
                    &format!("shared_word lazy={lazy} threads={threads}"),
                );
            });
        }
    }
}

#[test]
fn pcc_cross_word_partial_trailing_blocks() {
    // Cover active_rows % 64 in {0, 1, 2, 3, 63}.
    // With lazy=true (gs=n, active_rows=n+1): n=319 -> 0, n=256 -> 1,
    // n=257 -> 2, n=258 -> 3, n=318 -> 63.
    // With lazy=false (gs=0, active_rows=2*n+1): n=256 -> 1, n=257 -> 3,
    // n=287 -> 63.
    let cases = [
        (319usize, true, 0usize),
        (256, true, 1),
        (257, true, 2),
        (258, true, 3),
        (318, true, 63),
        (256, false, 1),
        (257, false, 3),
        (287, false, 63),
    ];

    for (n, lazy, expected_rem) in cases {
        let gs = if lazy { n } else { 0 };
        let active_rows = (2 * n + 1) - gs;
        assert_eq!(active_rows % 64, expected_rem);

        let base = seed_dense_tableau_with_nonzero_tail(n, lazy, 700 + n as u64);
        let gates = vec![
            make_cross_word_gate(super::BatchGate::CX, n - 1, 0),
            make_cross_word_gate(super::BatchGate::CZ, 64 + 5, 192 + 3),
            make_cross_word_gate(super::BatchGate::SWAP, 128 + 11, 2),
            make_cross_word_gate(super::BatchGate::CX, 3, 64 + 62),
            make_cross_word_gate(super::BatchGate::CZ, 192 + 3, 1),
        ];

        let thread_counts: &[usize] = if cfg!(feature = "parallel") {
            &[1, 4]
        } else {
            &[1]
        };
        for &threads in thread_counts {
            on_threads(threads, || {
                assert_pcc_matches_per_gate(
                    &base,
                    &gates,
                    lazy,
                    &format!("partial n={n} lazy={lazy} rem={expected_rem} threads={threads}"),
                );
            });
        }
    }
}

#[test]
fn pcc_cross_word_empty_is_noop() {
    for lazy in [false, true] {
        let base = seed_dense_tableau_with_nonzero_tail(256, lazy, 11);
        let mut a = base.clone();
        a.gate_rows_view().pcc_apply_cross_word(&[]);
        assert_eq!(a.raw_tableau(), base.raw_tableau());
    }
}

#[test]
fn pcc_cross_word_rng_and_measurement_parity() {
    let n = 320usize;
    let nw = n.div_ceil(64);
    assert_eq!(nw, 5);

    for seed in [42u64, 7, 1337] {
        let mut c = crate::circuits::clifford_random_pairs(n, 12, seed);
        c.num_classical_bits = n;
        c.add_measure(0, 0);

        // Exactly 4 cross-word gates (4 >= 4 and 4 < nw = 5) so colmajor
        // declines and pcc_apply_cross_word executes even with columns=true.
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::S, &[65]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::Cx, &[192, 3]);
        c.add_gate(Gate::Cz, &[64 + 7, 256 + 11]);
        c.add_gate(Gate::Swap, &[128 + 5, 4]);
        c.add_gate(Gate::Cx, &[256 + 20, 64 + 13]);
        c.add_gate(Gate::SX, &[4]);
        add_measure_all(&mut c);

        for lazy in [false, true] {
            let mut per_gate = if lazy {
                StabilizerBackend::new_lazy(seed)
            } else {
                StabilizerBackend::new(seed)
            };
            per_gate.init(n, n).unwrap();
            for instr in &c.instructions {
                per_gate.apply(instr).unwrap();
            }

            let thread_counts: &[usize] = if cfg!(feature = "parallel") {
                &[1, 4]
            } else {
                &[1]
            };
            for &threads in thread_counts {
                for columns in [false, true] {
                    on_threads(threads, || {
                        let mut batched = if lazy {
                            StabilizerBackend::new_lazy(seed)
                        } else {
                            StabilizerBackend::new(seed)
                        };
                        batched.init(n, n).unwrap();
                        batched
                            .apply_instructions_word_batch_with(&c.instructions, columns)
                            .unwrap();
                        assert_eq!(
                            batched.classical_results(),
                            per_gate.classical_results(),
                            "seed={seed} lazy={lazy} columns={columns} threads={threads}"
                        );
                        let stride = 2 * nw;
                        let (xa, pa) = batched.raw_tableau();
                        let (xb, pb) = per_gate.raw_tableau();
                        assert_eq!(
                            &xa[..2 * n * stride],
                            &xb[..2 * n * stride],
                            "seed={seed} lazy={lazy} columns={columns} threads={threads}"
                        );
                        assert_eq!(
                            &pa[..2 * n],
                            &pb[..2 * n],
                            "seed={seed} lazy={lazy} columns={columns} threads={threads}"
                        );
                    });
                }
            }
        }
    }
}

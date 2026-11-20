use super::*;
use crate::CircuitBuilder;
use crate::sim::compiled::PauliVec;

fn pauli(n: usize, x_bits: &[usize], z_bits: &[usize]) -> PauliVec {
    let mut p = PauliVec::new(n.div_ceil(64));
    for &q in x_bits {
        p.x[q / 64] |= 1u64 << (q % 64);
    }
    for &q in z_bits {
        p.z[q / 64] |= 1u64 << (q % 64);
    }
    p
}

#[test]
fn propagate_h_swaps_x_z() {
    let mut p = pauli(4, &[1], &[]);
    propagate_backward(&mut p, &Gate::H, &[1]);
    assert_eq!(p.x[0] & 0b10, 0);
    assert_eq!(p.z[0] & 0b10, 0b10);
}

#[test]
fn propagate_s_and_sdg_add_z_when_x_set() {
    let mut p = pauli(2, &[0], &[]);
    propagate_backward(&mut p, &Gate::S, &[0]);
    assert_eq!(p.z[0] & 1, 1);
    let mut q = pauli(2, &[0], &[]);
    propagate_backward(&mut q, &Gate::Sdg, &[0]);
    assert_eq!(q.z[0] & 1, 1);
}

#[test]
fn propagate_sx_and_sxdg_add_x_when_z_set() {
    let mut p = pauli(2, &[], &[0]);
    propagate_backward(&mut p, &Gate::SX, &[0]);
    assert_eq!(p.x[0] & 1, 1);
    let mut q = pauli(2, &[], &[0]);
    propagate_backward(&mut q, &Gate::SXdg, &[0]);
    assert_eq!(q.x[0] & 1, 1);
}

#[test]
fn propagate_pauli_gates_noop() {
    let p0 = pauli(2, &[0], &[1]);
    for gate in [Gate::X, Gate::Y, Gate::Z, Gate::Id] {
        let mut p = p0.clone();
        propagate_backward(&mut p, &gate, &[0]);
        assert_eq!(p.x, p0.x);
        assert_eq!(p.z, p0.z);
    }
}

#[test]
fn propagate_cx_flips_target_x_and_control_z() {
    let mut p = pauli(2, &[0], &[1]);
    propagate_backward(&mut p, &Gate::Cx, &[0, 1]);
    assert_eq!(p.x[0] & 0b11, 0b11);
    assert_eq!(p.z[0] & 0b11, 0b11);
}

#[test]
fn propagate_cz_adds_z_on_partner_when_x_set() {
    let mut p = pauli(2, &[0], &[]);
    propagate_backward(&mut p, &Gate::Cz, &[0, 1]);
    assert_eq!(p.z[0] & 0b10, 0b10);
    assert_eq!(p.x[0] & 0b01, 0b01);
}

#[test]
fn propagate_swap_exchanges_pauli_bits() {
    let mut p = pauli(2, &[0], &[1]);
    propagate_backward(&mut p, &Gate::Swap, &[0, 1]);
    assert_eq!(p.x[0] & 0b11, 0b10);
    assert_eq!(p.z[0] & 0b11, 0b01);
}

#[test]
fn propagate_unsupported_gate_is_noop() {
    let p0 = pauli(1, &[], &[0]);
    let mut p = p0.clone();
    propagate_backward(&mut p, &Gate::T, &[0]);
    assert_eq!(p.x, p0.x);
    assert_eq!(p.z, p0.z);
}

#[test]
fn build_measurement_rows_empty_when_no_measurements() {
    let circuit = CircuitBuilder::new(1).h(0).build();
    let rows = build_measurement_rows(&circuit);
    assert!(rows.is_empty());
}

#[test]
fn build_measurement_rows_z_for_unentangled_meas() {
    let circuit = CircuitBuilder::new_with_classical(2, 2)
        .measure(0, 0)
        .measure(1, 1)
        .build();
    let rows = build_measurement_rows(&circuit);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].1, 0);
    assert_eq!(rows[1].1, 1);
    assert_eq!(rows[0].0.x[0], 0);
    assert_eq!(rows[1].0.x[0], 0);
    assert_eq!(rows[0].0.z[0] & 1, 1);
    assert_eq!(rows[1].0.z[0] & 0b10, 0b10);
}

#[test]
fn build_measurement_rows_h_swaps_to_x() {
    let circuit = CircuitBuilder::new_with_classical(1, 1)
        .h(0)
        .measure(0, 0)
        .build();
    let rows = build_measurement_rows(&circuit);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0.x[0] & 1, 1);
    assert_eq!(rows[0].0.z[0] & 1, 0);
}

#[test]
fn propagate_cz_flips_q0_when_x1_set() {
    let mut p = pauli(2, &[1], &[]);
    propagate_backward(&mut p, &Gate::Cz, &[0, 1]);
    assert_eq!(p.z[0] & 0b01, 0b01);
    assert_eq!(p.x[0] & 0b10, 0b10);
}

#[test]
fn build_measurement_rows_includes_conditional_gates() {
    use crate::circuit::{ClassicalCondition, Instruction, SmallVec};
    use smallvec::smallvec;
    let mut c = crate::circuit::Circuit::new(1, 1);
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::H,
        targets: smallvec![0],
    });
    c.instructions.push(Instruction::Measure {
        qubit: 0,
        classical_bit: 0,
    });
    let _: SmallVec<[usize; 4]> = smallvec![];
    let rows = build_measurement_rows(&c);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0.x[0] & 1, 1);
}

fn build_avx_pauli(q: usize, m_words: usize) -> Vec<Vec<u64>> {
    let mut v = vec![vec![0u64; m_words]; q.max(1)];
    for word in v[0].iter_mut() {
        *word = 0xA5A5_A5A5_A5A5_A5A5;
    }
    v
}

#[test]
fn batch_h_avx2_path_193_meas() {
    let m_words = 4;
    let mut x = build_avx_pauli(2, m_words);
    let mut z = build_avx_pauli(2, m_words);
    for word in z[0].iter_mut() {
        *word = 0x5A5A_5A5A_5A5A_5A5A;
    }
    let mut sign = vec![0u64; m_words];
    let x_before = x[0].clone();
    let z_before = z[0].clone();
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::H, &[0], m_words);
    assert_eq!(x[0], z_before);
    assert_eq!(z[0], x_before);
}

#[test]
fn batch_s_and_sdg_avx2_path() {
    let m_words = 4;
    let mut x = build_avx_pauli(2, m_words);
    let mut z = build_avx_pauli(2, m_words);
    let mut sign = vec![0u64; m_words];
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::S, &[0], m_words);
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::Sdg, &[0], m_words);
    for word in z[0].iter().take(m_words) {
        assert_eq!(*word & 0xA5A5_A5A5_A5A5_A5A5, 0xA5A5_A5A5_A5A5_A5A5);
    }
}

#[test]
fn batch_x_y_z_avx2_paths() {
    let m_words = 4;
    let mut x = build_avx_pauli(2, m_words);
    let mut z = build_avx_pauli(2, m_words);
    let mut sign = vec![0u64; m_words];
    let z_initial = z[0].clone();
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::X, &[0], m_words);
    for w in 0..m_words {
        assert_eq!(sign[w], z_initial[w]);
    }
    let mut sign2 = vec![0u64; m_words];
    batch_propagate_backward(&mut x, &mut z, &mut sign2, &Gate::Z, &[0], m_words);
    let mut sign3 = vec![0u64; m_words];
    batch_propagate_backward(&mut x, &mut z, &mut sign3, &Gate::Y, &[0], m_words);
    for w in 0..m_words {
        assert_eq!(sign3[w], x[0][w] ^ z[0][w]);
    }
}

// Run `gates` over `m_words` measurement words at once, then over each word on its
// own (the scalar path), and require the two to agree word for word.
fn assert_batch_matches_single_words(gates: &[(Gate, &[usize])], m_words: usize) {
    let rows = |patterns: [u64; 2]| -> Vec<Vec<u64>> {
        patterns
            .iter()
            .map(|p| {
                (0..m_words)
                    .map(|w| p.rotate_left(7 * w as u32 + 1))
                    .collect()
            })
            .collect()
    };
    let x0 = rows([0xA5A5_A5A5_A5A5_A5A5, 0x3C3C_3C3C_3C3C_3C3C]);
    let z0 = rows([0x5A5A_5A5A_5A5A_5A5A, 0x0FF0_0FF0_0FF0_0FF0]);
    let (mut x, mut z, mut sign) = (x0.clone(), z0.clone(), vec![0u64; m_words]);
    for (gate, targets) in gates {
        batch_propagate_backward(&mut x, &mut z, &mut sign, gate, targets, m_words);
    }
    for w in 0..m_words {
        let mut xw: Vec<Vec<u64>> = x0.iter().map(|r| vec![r[w]]).collect();
        let mut zw: Vec<Vec<u64>> = z0.iter().map(|r| vec![r[w]]).collect();
        let mut sw = vec![0u64];
        for (gate, targets) in gates {
            batch_propagate_backward(&mut xw, &mut zw, &mut sw, gate, targets, 1);
        }
        for q in 0..2 {
            assert_eq!(x[q][w], xw[q][0], "x[{q}] word {w}");
            assert_eq!(z[q][w], zw[q][0], "z[{q}] word {w}");
        }
        assert_eq!(sign[w], sw[0], "sign word {w}");
    }
}

#[test]
fn batch_sx_avx2_paths() {
    assert_batch_matches_single_words(&[(Gate::SX, &[0])], 4);
    assert_batch_matches_single_words(&[(Gate::SXdg, &[0])], 4);
}

#[test]
fn batch_cx_cz_swap_avx2_paths() {
    for gate in [Gate::Cx, Gate::Cz, Gate::Swap] {
        assert_batch_matches_single_words(&[(gate, &[0, 1])], 4);
    }
}

#[test]
fn batch_path_with_tail_words() {
    assert_batch_matches_single_words(&[(Gate::H, &[0]), (Gate::S, &[1]), (Gate::Cx, &[0, 1])], 5);
}

#[test]
fn colmajor_forward_sim_matches_stabilizer_tableau() {
    use crate::backend::Backend;
    use crate::backend::stabilizer::StabilizerBackend;
    use rand::{RngExt, SeedableRng};

    let one_qubit = [
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::Id,
        Gate::SX,
        Gate::SXdg,
    ];
    let two_qubit = [Gate::Cx, Gate::Cz, Gate::Swap];

    // Row words are 2n/64: 97 and 130 reach the AVX2 threshold of four, 130 and 300
    // add tail words past the vector chunks.
    for n in [5, 64, 97, 130, 300] {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut c = Circuit::new(n, 0);
        for _ in 0..12 * n {
            if rng.random_range(0..3) == 0 {
                let a = rng.random_range(0..n);
                let b = (a + rng.random_range(1..n)) % n;
                c.add_gate(
                    two_qubit[rng.random_range(0..two_qubit.len())].clone(),
                    &[a, b],
                );
            } else {
                let q = rng.random_range(0..n);
                c.add_gate(
                    one_qubit[rng.random_range(0..one_qubit.len())].clone(),
                    &[q],
                );
            }
        }

        let (xz, phase, nw) = colmajor_forward_sim(n, &c.instructions).unwrap();

        let mut reference = StabilizerBackend::new(42);
        reference.init(n, 0).unwrap();
        reference.apply_gates_only(&c.instructions).unwrap();
        let (ref_xz, ref_phase) = reference.export_tableau().unwrap();

        assert_eq!(
            xz[..],
            ref_xz[..2 * n * 2 * nw],
            "tableau bits differ at n={n}"
        );
        assert_eq!(
            phase[..],
            ref_phase[..2 * n],
            "tableau signs differ at n={n}"
        );
    }
}

#[test]
fn colmajor_forward_sim_matches_stabilizer_tableau_with_late_fresh_qubits() {
    use crate::backend::Backend;
    use crate::backend::stabilizer::StabilizerBackend;
    use rand::{RngExt, SeedableRng};

    let one_qubit = [Gate::H, Gate::S, Gate::Sdg, Gate::Y, Gate::SX, Gate::SXdg];
    let two_qubit = [Gate::Cx, Gate::Cz, Gate::Swap];

    // Qubits enter in shuffled order, so the touched bound jumps ahead of qubits still
    // fresh below it. At 300 and 600 the destabilizer and stabilizer word ranges stay
    // apart for most of the circuit and the stabilizer range starts off a vector boundary.
    for n in [5, 97, 130, 300, 600] {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.random_range(0..=i));
        }
        let mut c = Circuit::new(n, 0);
        c.add_gate(Gate::H, &[order[0]]);
        for k in 1..n {
            let fresh = order[k];
            let old = order[rng.random_range(0..k)];
            c.add_gate(
                two_qubit[rng.random_range(0..two_qubit.len())].clone(),
                &[old, fresh],
            );
            for _ in 0..3 {
                let a = order[rng.random_range(0..=k)];
                if rng.random_range(0..2) == 0 {
                    let b = order[rng.random_range(0..=k)];
                    if a != b {
                        c.add_gate(
                            two_qubit[rng.random_range(0..two_qubit.len())].clone(),
                            &[a, b],
                        );
                    }
                } else {
                    c.add_gate(
                        one_qubit[rng.random_range(0..one_qubit.len())].clone(),
                        &[a],
                    );
                }
            }
        }

        let (xz, phase, nw) = colmajor_forward_sim(n, &c.instructions).unwrap();

        let mut reference = StabilizerBackend::new(42);
        reference.init(n, 0).unwrap();
        reference.apply_gates_only(&c.instructions).unwrap();
        let (ref_xz, ref_phase) = reference.export_tableau().unwrap();

        assert_eq!(
            xz[..],
            ref_xz[..2 * n * 2 * nw],
            "tableau bits differ at n={n}"
        );
        assert_eq!(
            phase[..],
            ref_phase[..2 * n],
            "tableau signs differ at n={n}"
        );
    }
}

// Aaronson-Gottesman `g`: the phase exponent, in units of `i`, that the bit-`b` letters
// contribute to the product of the source and destination rows.
fn ag_phase(x1: u64, z1: u64, x2: u64, z2: u64, b: u32) -> i64 {
    let bit = |w: u64| ((w >> b) & 1) as i64;
    let (x2, z2) = (bit(x2), bit(z2));
    match (bit(x1), bit(z1)) {
        (0, 0) => 0,
        (1, 1) => z2 - x2,
        (1, 0) => z2 * (2 * x2 - 1),
        _ => x2 * (1 - 2 * z2),
    }
}

// Row lengths straddle each kernel tier threshold (NEON at 2 words, AVX2 at 4, AVX-512
// at 8) and leave odd tails past each vector width.
#[test]
fn rowmul_phase_matches_ag_phase_rule() {
    use rand::{RngExt, SeedableRng};

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for nw in [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 35] {
        for _ in 0..64 {
            let mut src: Vec<u64> = (0..2 * nw).map(|_| rng.random()).collect();
            let mut dst: Vec<u64> = (0..2 * nw).map(|_| rng.random()).collect();
            if rng.random_bool(0.25) {
                dst.clone_from(&src);
            }
            if rng.random_bool(0.25) {
                dst[..nw].fill(0);
            }
            if nw > 0 && rng.random_bool(0.25) {
                let start = rng.random_range(0..nw);
                for half in [0, nw] {
                    let words = half + start..half + (start + 8).min(nw);
                    src[words.clone()].fill(0);
                    dst[words].fill(0);
                }
            }
            let (src_phase, dst_phase) = (rng.random_bool(0.5), rng.random_bool(0.5));
            let mut exponent = 2 * i64::from(src_phase) + 2 * i64::from(dst_phase);
            for w in 0..nw {
                for b in 0..64 {
                    exponent += ag_phase(src[w], src[nw + w], dst[w], dst[nw + w], b);
                }
            }
            let product: Vec<u64> = src.iter().zip(&dst).map(|(s, d)| s ^ d).collect();
            let phase = rowmul_phase(&src, &mut dst, nw, src_phase, dst_phase);
            assert_eq!(phase, exponent.rem_euclid(4) >= 2, "{nw} words");
            assert_eq!(dst, product, "{nw} words");
        }
    }
}

// Each AVX-512 propagation kernel against its AVX2 counterpart on random rows, at
// lengths that leave every masked tail width from one to seven words.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_propagation_kernels_match_avx2() {
    use rand::{RngExt, SeedableRng};

    // SAFETY: the caller establishes AVX-512F, which implies AVX2; every row holds
    // `m` words.
    unsafe fn apply(rows: &mut [Vec<u64>; 5], kernel: usize, wide: bool, m: usize) {
        let [x0, z0, x1, z1, s] = rows;
        macro_rules! pick {
            ($narrow:ident, $wide:ident($($arg:expr),*)) => {
                // SAFETY: same contract as the enclosing unsafe fn.
                unsafe {
                    if wide {
                        $wide($($arg),*)
                    } else {
                        $narrow($($arg),*)
                    }
                }
            };
        }
        match kernel {
            0 => pick!(
                batch_propagate_h_avx2,
                batch_propagate_h_avx512(x0, z0, s, m)
            ),
            1 => pick!(
                batch_propagate_s_avx2,
                batch_propagate_s_avx512(x0, z0, s, m, true)
            ),
            2 => pick!(
                batch_propagate_s_avx2,
                batch_propagate_s_avx512(x0, z0, s, m, false)
            ),
            3 => pick!(
                batch_propagate_sign_xor2_avx2,
                batch_propagate_sign_xor2_avx512(s, x0, z0, m)
            ),
            4 => pick!(
                batch_propagate_sx_avx2,
                batch_propagate_sx_avx512(x0, z0, s, m, false)
            ),
            5 => pick!(
                batch_propagate_sx_avx2,
                batch_propagate_sx_avx512(x0, z0, s, m, true)
            ),
            6 => pick!(
                batch_propagate_cx_avx2,
                batch_propagate_cx_avx512(x0, z0, x1, z1, s, m)
            ),
            7 => pick!(
                batch_propagate_cz_avx2,
                batch_propagate_cz_avx512(x0, z0, x1, z1, s, m)
            ),
            _ => pick!(
                batch_propagate_swap_avx2,
                batch_propagate_swap_avx512(x0, z0, x1, z1, m)
            ),
        }
    }

    if !crate::backend::word_ops::avx512_test_host() {
        return;
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for m in [8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 37] {
        let rows: [Vec<u64>; 5] = std::array::from_fn(|_| (0..m).map(|_| rng.random()).collect());
        for kernel in 0..9 {
            let mut narrow = rows.clone();
            let mut wide = rows.clone();
            // SAFETY: AVX-512F detected; every row holds `m` words.
            unsafe {
                apply(&mut narrow, kernel, false, m);
                apply(&mut wide, kernel, true, m);
            }
            assert_eq!(wide, narrow, "kernel {kernel}, {m} words");
        }
    }
}

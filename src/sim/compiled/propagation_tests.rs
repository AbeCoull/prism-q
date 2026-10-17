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

#[test]
fn batch_sx_avx2_paths() {
    let m_words = 4;
    let mut x = build_avx_pauli(2, m_words);
    let mut z = build_avx_pauli(2, m_words);
    let mut sign = vec![0u64; m_words];
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::SX, &[0], m_words);
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::SXdg, &[0], m_words);
}

#[test]
fn batch_cx_cz_swap_avx2_paths() {
    let m_words = 4;
    let mut x = build_avx_pauli(2, m_words);
    let mut z = build_avx_pauli(2, m_words);
    let mut sign = vec![0u64; m_words];
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::Cx, &[0, 1], m_words);
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::Cz, &[0, 1], m_words);
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::Swap, &[0, 1], m_words);
    assert_eq!(x.len(), 2);
}

#[test]
fn batch_path_with_tail_words() {
    let m_words = 5;
    let mut x = vec![vec![0xFFu64; m_words]; 2];
    let mut z = vec![vec![0xAAu64; m_words]; 2];
    let mut sign = vec![0u64; m_words];
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::H, &[0], m_words);
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::S, &[1], m_words);
    batch_propagate_backward(&mut x, &mut z, &mut sign, &Gate::Cx, &[0, 1], m_words);
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

#[cfg(target_arch = "x86_64")]
#[test]
fn rowmul_phase_masks_match_the_ag_phase_rule() {
    for letters in 0..16u64 {
        let bit = |k: u32| (letters >> k) & 1;
        let (x1, z1, x2, z2) = (bit(0), bit(1), bit(2), bit(3));
        let (plus, nonzero) = super::rowmul_spec::phase_masks(x1, z1, x2, z2);
        let phase = 2 * (plus & 1) as i64 - (nonzero & 1) as i64;
        assert_eq!(phase, super::rowmul_spec::ag_phase(x1, z1, x2, z2, 0));
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn rowmul_avx2_matches_bitwise_phase_rule() {
    use rand::{RngExt, SeedableRng};

    fn run<const N: usize>(rng: &mut rand_chacha::ChaCha8Rng) {
        for _ in 0..64 {
            let src: [u64; N] = rng.random();
            let mut dst: [u64; N] = rng.random();
            if rng.random_bool(0.25) {
                dst = src;
            }
            if rng.random_bool(0.25) {
                dst[..N / 2].fill(0);
            }
            super::rowmul_spec::check(src, dst, rng.random());
        }
    }

    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    run::<0>(&mut rng);
    run::<2>(&mut rng);
    run::<6>(&mut rng);
    run::<8>(&mut rng);
    run::<10>(&mut rng);
    run::<14>(&mut rng);
    run::<16>(&mut rng);
    run::<18>(&mut rng);
}

use super::kernels::{rowmul_words, xor_words};
use super::*;
use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction};
use crate::error::PrismError;
use crate::gates::Gate;
use crate::sim;
use num_complex::Complex64;

#[test]
fn test_init_tableau() {
    let mut b = StabilizerBackend::new(42);
    b.init(3, 0).unwrap();
    assert_eq!(b.n, 3);
    let stride = b.stride();
    let nw = b.num_words;
    // Destabilizer 0 = X_0
    assert_eq!(b.xz[0] & 1, 1);
    assert_eq!((b.xz[0] >> 1) & 1, 0);
    assert_eq!(b.xz[nw] & 1, 0);
    // Stabilizer 0 (row 3) = Z_0
    assert_eq!(b.xz[3 * stride] & 1, 0);
    assert_eq!(b.xz[3 * stride + nw] & 1, 1);
}

#[test]
fn test_x_flips() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[true]);
}

#[test]
fn test_z_on_zero() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::Z, &[0]);
    c.add_measure(0, 0);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[false]);
}

#[test]
fn test_y_flips() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::Y, &[0]);
    c.add_measure(0, 0);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[true]);
}

#[test]
fn test_hzh_equals_x() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Z, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[true]);
}

#[test]
fn test_s_squared_is_z() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[true]);
}

#[test]
fn test_s_sdg_cancel() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::Sdg, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[false]);
}

#[test]
fn test_h_superposition_deterministic() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);

    let mut b1 = StabilizerBackend::new(42);
    sim::run_on(&mut b1, &c).unwrap();
    let r1 = b1.classical_results()[0];

    let mut b2 = StabilizerBackend::new(42);
    sim::run_on(&mut b2, &c).unwrap();
    let r2 = b2.classical_results()[0];

    assert_eq!(r1, r2);
}

#[test]
fn test_bell_correlated() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);

    for seed in [42, 100, 999, 12345] {
        let mut b = StabilizerBackend::new(seed);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(
            b.classical_results()[0],
            b.classical_results()[1],
            "Bell state measurements must be equal (seed {seed})"
        );
    }
}

#[test]
fn test_ghz_4_correlated() {
    let mut c = Circuit::new(4, 4);
    c.add_gate(Gate::H, &[0]);
    for i in 0..3 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    for i in 0..4 {
        c.add_measure(i, i);
    }

    for seed in [42, 100, 999] {
        let mut b = StabilizerBackend::new(seed);
        sim::run_on(&mut b, &c).unwrap();
        let results = b.classical_results();
        assert!(
            results.iter().all(|&x| x == results[0]),
            "GHZ-4 measurements must be equal (seed {seed})"
        );
    }
}

#[test]
fn test_swap() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[true, false]);
}

#[test]
fn test_cz_on_11() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[true, true]);
}

#[test]
fn test_rejects_t_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::T, &[0]);
    let mut b = StabilizerBackend::new(42);
    b.init(1, 0).unwrap();
    let err = b.apply(&c.instructions[0]).unwrap_err();
    assert!(matches!(err, PrismError::BackendUnsupported { .. }));
}

#[test]
fn test_rejects_rx_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(0.5), &[0]);
    let mut b = StabilizerBackend::new(42);
    b.init(1, 0).unwrap();
    let err = b.apply(&c.instructions[0]).unwrap_err();
    assert!(matches!(err, PrismError::BackendUnsupported { .. }));
}

#[test]
fn test_probs_zero_state() {
    let mut b = StabilizerBackend::new(42);
    b.init(1, 0).unwrap();
    let probs = b.probabilities().unwrap();
    assert_eq!(probs, vec![1.0, 0.0]);
}

#[test]
fn test_probs_one_state() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::X, &[0]);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert_eq!(probs, vec![0.0, 1.0]);
}

#[test]
fn test_probs_plus_state() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0] - 0.5).abs() < 1e-12);
    assert!((probs[1] - 0.5).abs() < 1e-12);
}

#[test]
fn test_probs_bell_state() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0] - 0.5).abs() < 1e-12);
    assert!(probs[1].abs() < 1e-12);
    assert!(probs[2].abs() < 1e-12);
    assert!((probs[3] - 0.5).abs() < 1e-12);
}

#[test]
fn test_probs_ghz_4() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    for i in 0..3 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert_eq!(probs.len(), 16);
    assert!((probs[0] - 0.5).abs() < 1e-12);
    assert!((probs[15] - 0.5).abs() < 1e-12);
    let rest_sum: f64 = probs[1..15].iter().sum();
    assert!(rest_sum.abs() < 1e-12);
}

#[test]
fn test_1000_qubit_ghz() {
    let n = 1000;
    let mut c = Circuit::new(n, n);
    c.add_gate(Gate::H, &[0]);
    for i in 0..n - 1 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    for i in 0..n {
        c.add_measure(i, i);
    }
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let results = b.classical_results();
    assert!(results.iter().all(|&x| x == results[0]));
}

#[test]
fn test_id_no_change() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::Id, &[0]);
    c.add_measure(0, 0);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[false]);
}

#[test]
fn test_double_x_is_identity() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[false]);
}

#[test]
fn test_supports_fused_gates() {
    let b = StabilizerBackend::new(42);
    assert!(!b.supports_fused_gates());
}

#[test]
fn test_rejects_cu_gate() {
    let h_mat = Gate::H.matrix_2x2();
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cu(Box::new(h_mat)), &[0, 1]);
    let mut b = StabilizerBackend::new(42);
    b.init(2, 0).unwrap();
    let err = b.apply(&c.instructions[0]).unwrap_err();
    assert!(matches!(err, PrismError::BackendUnsupported { .. }));
}

#[test]
fn test_rejects_mcu_gate() {
    use crate::gates::McuData;
    let x_mat = Gate::X.matrix_2x2();
    let mcu = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 2,
    }));
    let mut c = Circuit::new(3, 0);
    c.add_gate(mcu, &[0, 1, 2]);
    let mut b = StabilizerBackend::new(42);
    b.init(3, 0).unwrap();
    let err = b.apply(&c.instructions[0]).unwrap_err();
    assert!(matches!(err, PrismError::BackendUnsupported { .. }));
}

#[test]
fn test_export_statevector_zero_state() {
    let mut b = StabilizerBackend::new(42);
    b.init(1, 0).unwrap();
    let sv = b.export_statevector().unwrap();
    assert!((sv[0].re - 1.0).abs() < 1e-10);
    assert!(sv[0].im.abs() < 1e-10);
    assert!(sv[1].norm_sqr() < 1e-10);
}

#[test]
fn test_export_statevector_one_state() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::X, &[0]);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let sv = b.export_statevector().unwrap();
    assert!(sv[0].norm_sqr() < 1e-10);
    assert!((sv[1].norm_sqr() - 1.0).abs() < 1e-10);
}

#[test]
fn test_export_statevector_plus_state() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let sv = b.export_statevector().unwrap();
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((sv[0].re - expected).abs() < 1e-10);
    assert!((sv[1].re - expected).abs() < 1e-10);
}

#[test]
fn test_export_statevector_minus_state() {
    // H X |0⟩ = H|1⟩ = (|0⟩ - |1⟩)/√2
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::H, &[0]);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let sv = b.export_statevector().unwrap();
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!((sv[0].re - expected).abs() < 1e-10);
    assert!((sv[1].re + expected).abs() < 1e-10);
}

#[test]
fn test_export_statevector_bell_state_matches_sv() {
    use crate::backend::statevector::StatevectorBackend;

    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);

    let mut stab = StabilizerBackend::new(42);
    sim::run_on(&mut stab, &c).unwrap();
    let stab_sv = stab.export_statevector().unwrap();

    let mut sv = StatevectorBackend::new(42);
    sim::run_on(&mut sv, &c).unwrap();
    let sv_ref = sv.state_vector();

    // Probabilities must match exactly
    for (s, r) in stab_sv.iter().zip(sv_ref.iter()) {
        assert!(
            (s.norm_sqr() - r.norm_sqr()).abs() < 1e-10,
            "prob mismatch: stab={}, sv={}",
            s.norm_sqr(),
            r.norm_sqr()
        );
    }

    // Global phase: find ratio between first non-zero pair
    let global_phase = find_global_phase(&stab_sv, sv_ref);

    // After removing global phase, amplitudes must match
    for (s, r) in stab_sv.iter().zip(sv_ref.iter()) {
        let adjusted = s * global_phase;
        assert!(
            (adjusted - r).norm() < 1e-10,
            "amplitude mismatch after phase: stab*phase={adjusted:?}, sv={r:?}"
        );
    }
}

#[test]
fn test_export_statevector_ghz3_matches_sv() {
    use crate::backend::statevector::StatevectorBackend;

    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);

    let mut stab = StabilizerBackend::new(42);
    sim::run_on(&mut stab, &c).unwrap();
    let stab_sv = stab.export_statevector().unwrap();

    let mut sv = StatevectorBackend::new(42);
    sim::run_on(&mut sv, &c).unwrap();
    let sv_ref = sv.state_vector();

    let global_phase = find_global_phase(&stab_sv, sv_ref);
    for (s, r) in stab_sv.iter().zip(sv_ref.iter()) {
        let adjusted = s * global_phase;
        assert!(
            (adjusted - r).norm() < 1e-10,
            "GHZ3 mismatch: stab*phase={adjusted:?}, sv={r:?}"
        );
    }
}

#[test]
fn test_export_statevector_complex_clifford_matches_sv() {
    use crate::backend::statevector::StatevectorBackend;

    // Deeper Clifford circuit with S, Sdg, CZ, SX to exercise i-phases
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::SX, &[3]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Sdg, &[1]);
    c.add_gate(Gate::H, &[3]);
    c.add_gate(Gate::S, &[2]);
    c.add_gate(Gate::Swap, &[0, 3]);

    let mut stab = StabilizerBackend::new(42);
    sim::run_on(&mut stab, &c).unwrap();
    let stab_sv = stab.export_statevector().unwrap();

    let mut sv = StatevectorBackend::new(42);
    sim::run_on(&mut sv, &c).unwrap();
    let sv_ref = sv.state_vector();

    let global_phase = find_global_phase(&stab_sv, sv_ref);
    for (i, (s, r)) in stab_sv.iter().zip(sv_ref.iter()).enumerate() {
        let adjusted = s * global_phase;
        assert!(
            (adjusted - r).norm() < 1e-10,
            "4q Clifford mismatch at index {i}: stab*phase={adjusted:?}, sv={r:?}"
        );
    }
}

#[test]
fn test_export_statevector_all_paulis_match_sv() {
    use crate::backend::statevector::StatevectorBackend;

    // Y gate introduces i-phase: Y|0⟩ = i|1⟩
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Y, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::S, &[1]);

    let mut stab = StabilizerBackend::new(42);
    sim::run_on(&mut stab, &c).unwrap();
    let stab_sv = stab.export_statevector().unwrap();

    let mut sv = StatevectorBackend::new(42);
    sim::run_on(&mut sv, &c).unwrap();
    let sv_ref = sv.state_vector();

    let global_phase = find_global_phase(&stab_sv, sv_ref);
    for (i, (s, r)) in stab_sv.iter().zip(sv_ref.iter()).enumerate() {
        let adjusted = s * global_phase;
        assert!(
            (adjusted - r).norm() < 1e-10,
            "Pauli test mismatch at {i}: stab*phase={adjusted:?}, sv={r:?}"
        );
    }
}

/// Find the global phase ratio between two state vectors.
/// Returns the complex scalar c such that stab * c ≈ reference.
fn find_global_phase(stab: &[Complex64], reference: &[Complex64]) -> Complex64 {
    for (s, r) in stab.iter().zip(reference.iter()) {
        if s.norm_sqr() > 1e-10 && r.norm_sqr() > 1e-10 {
            return r / s;
        }
    }
    Complex64::new(1.0, 0.0)
}

#[test]
fn test_rejects_cphase_gate() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_4), &[0, 1]);
    let mut b = StabilizerBackend::new(42);
    b.init(2, 0).unwrap();
    let err = b.apply(&c.instructions[0]).unwrap_err();
    assert!(matches!(err, PrismError::BackendUnsupported { .. }));
}

#[test]
fn test_xor_words_various_lengths() {
    for len in 0..=17 {
        let src: Vec<u64> = (0..len)
            .map(|i| 0xAAAA_BBBB_CCCC_0000u64 | i as u64)
            .collect();
        let original: Vec<u64> = (0..len)
            .map(|i| 0x1111_2222_3333_0000u64 | (i as u64 * 7))
            .collect();

        let mut expected = original.clone();
        for i in 0..len {
            expected[i] ^= src[i];
        }

        let mut actual = original.clone();
        if len > 0 {
            unsafe { xor_words(actual.as_mut_ptr(), src.as_ptr(), len) };
        }

        assert_eq!(actual, expected, "xor_words mismatch at len={len}");
    }
}

#[test]
fn test_xor_words_all_ones_and_zeros() {
    let len = 8;
    let src = vec![u64::MAX; len];
    let mut dst = vec![0u64; len];
    unsafe { xor_words(dst.as_mut_ptr(), src.as_ptr(), len) };
    assert!(dst.iter().all(|&v| v == u64::MAX));

    unsafe { xor_words(dst.as_mut_ptr(), src.as_ptr(), len) };
    assert!(dst.iter().all(|&v| v == 0));
}

#[test]
fn test_rowmul_words_zero_src() {
    let nw = 4;
    let mut dst_x = vec![0xFFu64; nw];
    let mut dst_z = vec![0xAAu64; nw];
    let src_x = vec![0u64; nw];
    let src_z = vec![0u64; nw];
    let sum = rowmul_words(&mut dst_x, &mut dst_z, &src_x, &src_z, 0);
    assert_eq!(dst_x, vec![0xFFu64; nw]);
    assert_eq!(dst_z, vec![0xAAu64; nw]);
    assert_eq!(sum & 3, 0);
}

#[test]
fn test_rowmul_words_matches_manual() {
    let nw = 3;
    let src_x = vec![0b1010u64, 0b1100u64, 0b0011u64];
    let src_z = vec![0b0110u64, 0b1010u64, 0b1001u64];
    let orig_dst_x = vec![0b1100u64, 0b0110u64, 0b1010u64];
    let orig_dst_z = vec![0b0011u64, 0b1001u64, 0b0110u64];

    let mut manual_x = orig_dst_x.clone();
    let mut manual_z = orig_dst_z.clone();
    let mut manual_sum = 4u64;
    for w in 0..nw {
        let x1 = src_x[w];
        let z1 = src_z[w];
        let x2 = manual_x[w];
        let z2 = manual_z[w];
        let new_x = x1 ^ x2;
        let new_z = z1 ^ z2;
        manual_x[w] = new_x;
        manual_z[w] = new_z;
        if (x1 | z1 | x2 | z2) != 0 {
            let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
            let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
            manual_sum = manual_sum.wrapping_add(2 * pos.count_ones() as u64);
            manual_sum = manual_sum.wrapping_sub(nonzero.count_ones() as u64);
        }
    }

    let mut fn_x = orig_dst_x.clone();
    let mut fn_z = orig_dst_z.clone();
    let fn_sum = rowmul_words(&mut fn_x, &mut fn_z, &src_x, &src_z, 4);

    assert_eq!(fn_x, manual_x);
    assert_eq!(fn_z, manual_z);
    assert_eq!(fn_sum & 3, manual_sum & 3);
}

#[test]
fn test_rowmul_words_phase_y_times_x() {
    let src_x = vec![1u64];
    let src_z = vec![1u64];
    let mut dst_x = vec![1u64];
    let mut dst_z = vec![0u64];
    let sum = rowmul_words(&mut dst_x, &mut dst_z, &src_x, &src_z, 0);
    assert_eq!(dst_x[0], 0);
    assert_eq!(dst_z[0], 1);
    assert!(
        (sum & 3) >= 2,
        "Y*X should give phase -1 (sum&3={})",
        sum & 3
    );
}

#[test]
fn test_rowmul_words_simd_large() {
    for nw in [4, 5, 8, 9, 16, 17] {
        let src_x: Vec<u64> = (0..nw)
            .map(|i| 0xAAAA_BBBB_CCCC_0000u64 | i as u64)
            .collect();
        let src_z: Vec<u64> = (0..nw)
            .map(|i| 0x1111_2222_3333_0000u64 | (i * 7) as u64)
            .collect();
        let orig_x: Vec<u64> = (0..nw)
            .map(|i| 0x5555_6666_7777_0000u64 | (i * 3) as u64)
            .collect();
        let orig_z: Vec<u64> = (0..nw)
            .map(|i| 0x9999_AAAA_BBBB_0000u64 | (i * 5) as u64)
            .collect();

        let mut ref_x = orig_x.clone();
        let mut ref_z = orig_z.clone();
        let mut ref_sum = 2u64;
        for w in 0..nw {
            let x1 = src_x[w];
            let z1 = src_z[w];
            let x2 = ref_x[w];
            let z2 = ref_z[w];
            let new_x = x1 ^ x2;
            let new_z = z1 ^ z2;
            ref_x[w] = new_x;
            ref_z[w] = new_z;
            if (x1 | z1 | x2 | z2) != 0 {
                let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
                let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
                ref_sum = ref_sum.wrapping_add(2 * pos.count_ones() as u64);
                ref_sum = ref_sum.wrapping_sub(nonzero.count_ones() as u64);
            }
        }

        let mut fn_x = orig_x.clone();
        let mut fn_z = orig_z.clone();
        let fn_sum = rowmul_words(&mut fn_x, &mut fn_z, &src_x, &src_z, 2);

        assert_eq!(fn_x, ref_x, "XOR mismatch at nw={nw}");
        assert_eq!(fn_z, ref_z, "XOR mismatch at nw={nw}");
        assert_eq!(
            fn_sum & 3,
            ref_sum & 3,
            "Phase mismatch at nw={nw}: fn_sum={fn_sum}, ref_sum={ref_sum}"
        );
    }
}

#[test]
fn test_rowmul_refactor_preserves_ghz_correctness() {
    let n = 10;
    let mut c = Circuit::new(n, n);
    c.add_gate(Gate::H, &[0]);
    for i in 0..n - 1 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    for i in 0..n {
        c.add_measure(i, i);
    }

    for seed in 0..20u64 {
        let mut b = StabilizerBackend::new(seed);
        sim::run_on(&mut b, &c).unwrap();
        let results = b.classical_results();
        let first = results[0];
        assert!(
            results.iter().all(|&r| r == first),
            "GHZ violation at seed {seed}: {results:?}"
        );
    }
}

#[test]
fn test_rowmul_refactor_preserves_probabilities() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0] - 0.5).abs() < 1e-10);
    assert!(probs[1].abs() < 1e-10);
    assert!(probs[2].abs() < 1e-10);
    assert!((probs[3] - 0.5).abs() < 1e-10);
    assert!(probs[4].abs() < 1e-10);
    assert!(probs[5].abs() < 1e-10);
    assert!((probs[6]).abs() < 1e-10);
    assert!(probs[7].abs() < 1e-10);
}

#[test]
fn test_rowmul_multi_word_correctness() {
    let n = 65;
    let mut c = Circuit::new(n, n);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 64]);
    c.add_measure(0, 0);
    c.add_measure(64, 1);

    for seed in 0..10u64 {
        let mut b = StabilizerBackend::new(seed);
        sim::run_on(&mut b, &c).unwrap();
        let results = b.classical_results();
        assert_eq!(
            results[0], results[1],
            "Bell pair violation at seed {seed}: q0={}, q64={}",
            results[0], results[1]
        );
    }
}

#[test]
fn test_sgi_500q_clifford_d10_matches_gate_by_gate() {
    use crate::circuits;
    let n = 500;
    let mut circuit = circuits::clifford_heavy_circuit(n, 10, 42);
    circuit.num_classical_bits = n;
    for i in 0..n {
        circuit.add_measure(i, i);
    }

    let mut b1 = StabilizerBackend::new(42);
    b1.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    for instr in &circuit.instructions {
        b1.apply(instr).unwrap();
    }
    let r1 = b1.classical_results().to_vec();

    let mut b2 = StabilizerBackend::new(42);
    sim::run_on(&mut b2, &circuit).unwrap();
    let r2 = b2.classical_results().to_vec();

    assert_eq!(
        r1, r2,
        "SGI 500q Clifford d10: gate-by-gate vs apply_instructions mismatch"
    );
}

#[test]
fn test_sgi_300q_ghz_all_agree() {
    let n = 300;
    let mut c = Circuit::new(n, n);
    c.add_gate(Gate::H, &[0]);
    for i in 0..n - 1 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    for i in 0..n {
        c.add_measure(i, i);
    }
    let mut b = StabilizerBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let results = b.classical_results();
    assert!(
        results.iter().all(|&x| x == results[0]),
        "GHZ 300q: not all qubits agree"
    );
}

#[test]
fn test_sgi_index_consistency() {
    let n = 300;
    let mut circuit = crate::circuits::clifford_heavy_circuit(n, 5, 42);
    circuit.num_classical_bits = 0;

    let mut b = StabilizerBackend::new(42);
    b.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    b.apply_instructions(&circuit.instructions).unwrap();

    let stride = b.stride();
    let nw = b.num_words;
    for q in 0..n {
        for &g in &b.qubit_active[q] {
            let g = g as usize;
            let row = &b.xz[g * stride..(g + 1) * stride];
            let word = q / 64;
            let bit_mask = 1u64 << (q % 64);
            let x = row[word] & bit_mask;
            let z = row[nw + word] & bit_mask;
            assert!(
                x != 0 || z != 0,
                "qubit_active[{q}] contains generator {g} which has I on qubit {q}"
            );
        }
    }

    for g in 0..2 * n {
        let row = &b.xz[g * stride..(g + 1) * stride];
        for q in 0..n {
            let word = q / 64;
            let bit_mask = 1u64 << (q % 64);
            let x = row[word] & bit_mask;
            let z = row[nw + word] & bit_mask;
            let active = x != 0 || z != 0;
            let in_index = b.qubit_active[q].contains(&(g as u32));
            assert_eq!(
                active, in_index,
                "generator {g} qubit {q}: active={active} but in_index={in_index}"
            );
        }
    }
}

#[test]
fn test_pcc_random_pairs_matches_gate_by_gate() {
    use crate::circuits;
    let n = 500;
    let mut circuit = circuits::clifford_random_pairs(n, 10, 42);
    circuit.num_classical_bits = n;
    for i in 0..n {
        circuit.add_measure(i, i);
    }

    let mut b1 = StabilizerBackend::new(42);
    b1.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    for instr in &circuit.instructions {
        b1.apply(instr).unwrap();
    }
    let r1 = b1.classical_results().to_vec();

    let mut b2 = StabilizerBackend::new(42);
    sim::run_on(&mut b2, &circuit).unwrap();
    let r2 = b2.classical_results().to_vec();

    assert_eq!(
        r1, r2,
        "PCC 500q random-pairs d10: gate-by-gate vs apply_instructions mismatch"
    );
}

#[test]
fn filtered_bell_pairs_correct() {
    use crate::circuits;

    // Test pre-measurement probabilities (no RNG dependence)
    let n_pairs = 5;
    let n = n_pairs * 2;
    let circuit = circuits::independent_bell_pairs(n_pairs);

    let mut filt = FilteredStabilizerBackend::new(42);
    filt.init(n, 0).unwrap();
    for inst in &circuit.instructions {
        filt.apply(inst).unwrap();
    }

    let filt_probs = filt.probabilities().unwrap();
    let mono_probs = {
        let mut mono = StabilizerBackend::new(42);
        mono.init(n, 0).unwrap();
        mono.apply_instructions(&circuit.instructions).unwrap();
        mono.compute_probabilities()
    };

    assert_eq!(filt_probs.len(), mono_probs.len());
    for (i, (&f, &m)) in filt_probs.iter().zip(mono_probs.iter()).enumerate() {
        assert!(
            (f - m).abs() < 1e-10,
            "prob mismatch at index {i}: filtered={f}, monolithic={m}"
        );
    }
}

#[test]
fn filtered_bell_pairs_measurement() {
    use crate::circuits;
    let n_pairs = 10;
    let n = n_pairs * 2;
    let mut circuit = circuits::independent_bell_pairs(n_pairs);
    circuit.num_classical_bits = n;
    for i in 0..n {
        circuit.add_measure(i, i);
    }

    let mut filt = FilteredStabilizerBackend::new(42);
    filt.init(n, n).unwrap();
    for inst in &circuit.instructions {
        filt.apply(inst).unwrap();
    }
    let bits = filt.classical_results();

    for i in 0..n_pairs {
        assert_eq!(
            bits[2 * i],
            bits[2 * i + 1],
            "filtered: bell pair {i} qubits disagree"
        );
    }
}

#[test]
fn filtered_with_blocks_matches_monolithic() {
    use crate::circuits;
    let n_pairs = 5;
    let n = n_pairs * 2;
    let circuit = circuits::independent_bell_pairs(n_pairs);

    let blocks = circuit.independent_subsystems();
    assert_eq!(blocks.len(), n_pairs);

    let mut filt = FilteredStabilizerBackend::new(42);
    filt.init_with_blocks(n, 0, &blocks).unwrap();
    for inst in &circuit.instructions {
        filt.apply(inst).unwrap();
    }
    let filt_probs = filt.probabilities().unwrap();

    let mut mono = StabilizerBackend::new(42);
    mono.init(n, 0).unwrap();
    mono.apply_instructions(&circuit.instructions).unwrap();
    let mono_probs = mono.compute_probabilities();

    for (i, (&f, &m)) in filt_probs.iter().zip(mono_probs.iter()).enumerate() {
        assert!(
            (f - m).abs() < 1e-10,
            "prob mismatch at index {i}: filtered={f}, monolithic={m}"
        );
    }
}

#[test]
fn lazy_destab_matches_eager() {
    for n in [3, 5, 10] {
        let circuit = crate::circuits::clifford_heavy_circuit(n, 10, 42);

        let mut eager = StabilizerBackend::new(42);
        eager.init(n, 0).unwrap();
        eager.apply_gates_only(&circuit.instructions).unwrap();
        let eager_probs = eager.compute_probabilities();

        let mut lazy = StabilizerBackend::new(42);
        lazy.init(n, 0).unwrap();
        lazy.enable_lazy_destab();
        lazy.apply_gates_only(&circuit.instructions).unwrap();
        let lazy_probs = lazy.compute_probabilities();

        for (i, (&e, &l)) in eager_probs.iter().zip(lazy_probs.iter()).enumerate() {
            assert!(
                (e - l).abs() < 1e-10,
                "n={n} prob mismatch at {i}: eager={e}, lazy={l}"
            );
        }
    }
}

#[test]
fn lazy_destab_measure_matches_eager() {
    fn run_measurement_shots(
        cliff: &Circuit,
        n: usize,
        num_shots: usize,
        seed: u64,
        lazy_destab: bool,
    ) -> Vec<Vec<bool>> {
        let mut shots = Vec::with_capacity(num_shots);
        for i in 0..num_shots {
            let mut backend = StabilizerBackend::new(seed.wrapping_add(i as u64));
            backend.init(n, n).unwrap();
            if lazy_destab {
                backend.enable_lazy_destab();
            }
            for inst in &cliff.instructions {
                backend.apply(inst).unwrap();
            }
            for q in 0..n {
                backend
                    .apply(&Instruction::Measure {
                        qubit: q,
                        classical_bit: q,
                    })
                    .unwrap();
            }
            shots.push(backend.classical_results().to_vec());
        }
        shots
    }

    for n in [3, 5, 8] {
        let cliff = crate::circuits::clifford_heavy_circuit(n, 10, 42);
        let num_shots = 2_000;
        let eager_shots = run_measurement_shots(&cliff, n, num_shots, 42, false);
        let lazy_shots = run_measurement_shots(&cliff, n, num_shots, 42, true);

        for q in 0..n {
            let eager_ones = eager_shots.iter().filter(|shot| shot[q]).count();
            let lazy_ones = lazy_shots.iter().filter(|shot| shot[q]).count();
            let delta =
                (eager_ones as isize - lazy_ones as isize).unsigned_abs() as f64 / num_shots as f64;
            assert!(
                delta < 0.08,
                "n={n} q={q}: lazy/eager marginal mismatch too large ({delta:.3})"
            );
        }
    }
}

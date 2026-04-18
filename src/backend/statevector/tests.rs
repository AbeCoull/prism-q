use super::*;
use crate::circuit::Circuit;
use crate::sim;

fn assert_probs_close(actual: &[f64], expected: &[f64], eps: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "probability vector length mismatch"
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < eps,
            "prob[{i}]: expected {e}, got {a} (diff {})",
            (a - e).abs()
        );
    }
}

#[test]
fn test_x_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::X, &[0]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[0.0, 1.0], 1e-12);
}

#[test]
fn test_h_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[0.5, 0.5], 1e-12);
}

#[test]
fn test_bell_state() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[0.5, 0.0, 0.0, 0.5], 1e-12);
}

#[test]
fn test_swap() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[0.0, 1.0, 0.0, 0.0], 1e-12);
}

#[test]
fn test_cz() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert_probs_close(&probs, &[0.25, 0.25, 0.25, 0.25], 1e-12);
    let sv = b.state_vector();
    let eps = 1e-12;
    assert!((sv[3].re - (-0.5)).abs() < eps);
}

#[test]
fn test_measurement_deterministic() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_eq!(b.classical_results(), &[true]);
}

#[test]
fn test_rz_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rz(std::f64::consts::PI), &[0]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[1.0, 0.0], 1e-12);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_bell_state_16q() {
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0] - 0.5).abs() < 1e-12);
    assert!((probs[3] - 0.5).abs() < 1e-12);
    let rest_sum: f64 = probs[1..3].iter().chain(probs[4..].iter()).sum();
    assert!(rest_sum.abs() < 1e-12);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_swap_16q() {
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::Swap, &[0, 15]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[1 << 15] - 1.0).abs() < 1e-12);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_cz_16q() {
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let sv = b.state_vector();
    assert!((sv[3].re - (-0.5)).abs() < 1e-12);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_high_target_qubit_16q() {
    let n = 16;
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[n - 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0] - 0.5).abs() < 1e-12);
    assert!((probs[1 << (n - 1)] - 0.5).abs() < 1e-12);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_high_target_tiled_20q() {
    let n = 20;
    let target = 15;
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[target]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0] - 0.5).abs() < 1e-12);
    assert!((probs[1 << target] - 0.5).abs() < 1e-12);
    let rest_sum: f64 = probs
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != 0 && i != (1 << target))
        .map(|(_, p)| p)
        .sum();
    assert!(rest_sum.abs() < 1e-12);
}

#[test]
fn test_cu_h_controlled_hadamard() {
    let h_mat = Gate::H.matrix_2x2();
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Cu(Box::new(h_mat)), &[1, 0]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[0.0, 0.0, 0.5, 0.5], 1e-12);
}

#[test]
fn test_cu_no_action_when_control_zero() {
    let h_mat = Gate::H.matrix_2x2();
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cu(Box::new(h_mat)), &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[1.0, 0.0, 0.0, 0.0], 1e-12);
}

#[test]
fn test_cu_matches_cx() {
    let x_mat = Gate::X.matrix_2x2();
    let mut c1 = Circuit::new(2, 0);
    c1.add_gate(Gate::H, &[0]);
    c1.add_gate(Gate::Cu(Box::new(x_mat)), &[0, 1]);

    let mut c2 = Circuit::new(2, 0);
    c2.add_gate(Gate::H, &[0]);
    c2.add_gate(Gate::Cx, &[0, 1]);

    let mut b1 = StatevectorBackend::new(42);
    sim::run_on(&mut b1, &c1).unwrap();
    let mut b2 = StatevectorBackend::new(42);
    sim::run_on(&mut b2, &c2).unwrap();

    let sv1 = b1.state_vector();
    let sv2 = b2.state_vector();
    for (a, b) in sv1.iter().zip(sv2) {
        assert!((a - b).norm() < 1e-12);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_cu_16q() {
    let h_mat = Gate::H.matrix_2x2();
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::Cu(Box::new(h_mat)), &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[1] - 0.5).abs() < 1e-12);
    assert!((probs[3] - 0.5).abs() < 1e-12);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_measure_deterministic_16q() {
    let mut c = Circuit::new(16, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);

    let mut b1 = StatevectorBackend::new(42);
    sim::run_on(&mut b1, &c).unwrap();
    let r1 = b1.classical_results()[0];

    let mut b2 = StatevectorBackend::new(42);
    sim::run_on(&mut b2, &c).unwrap();
    let r2 = b2.classical_results()[0];

    assert_eq!(r1, r2);
}

#[test]
fn test_insert_zero_bit_basic() {
    assert_eq!(insert_zero_bit(0b11, 0), 0b110);
    assert_eq!(insert_zero_bit(0b11, 1), 0b101);
    assert_eq!(insert_zero_bit(0b11, 2), 0b011);
    assert_eq!(insert_zero_bit(0, 0), 0);
    assert_eq!(insert_zero_bit(1, 0), 0b10);
}

#[test]
fn test_mask_indices_bijection_1ctrl() {
    for num_qubits in 2..=6 {
        for control in 0..num_qubits {
            for target in 0..num_qubits {
                if control == target {
                    continue;
                }
                let ctrl_mask = 1usize << control;
                let tgt_mask = 1usize << target;
                let (lo, hi) = if control < target {
                    (control, target)
                } else {
                    (target, control)
                };
                let num_iters = 1usize << (num_qubits - 2);
                let mut seen = std::collections::HashSet::new();

                for i in 0..num_iters {
                    let base = insert_zero_bit(insert_zero_bit(i, lo), hi);
                    let idx0 = base | ctrl_mask;
                    let idx1 = idx0 | tgt_mask;

                    assert_ne!(idx0 & ctrl_mask, 0, "ctrl bit not set");
                    assert_ne!(idx1 & ctrl_mask, 0, "ctrl bit not set");
                    assert_eq!(idx0 & tgt_mask, 0, "tgt bit set in idx0");
                    assert_ne!(idx1 & tgt_mask, 0, "tgt bit not set in idx1");
                    assert_eq!(idx1, idx0 | tgt_mask);
                    assert!(idx0 < (1 << num_qubits));
                    assert!(idx1 < (1 << num_qubits));
                    assert!(seen.insert(idx0), "duplicate idx0={idx0}");
                    assert!(seen.insert(idx1), "duplicate idx1={idx1}");
                }

                assert_eq!(seen.len(), 1 << (num_qubits - 1));
            }
        }
    }
}

#[test]
fn test_cu_control_less_than_target() {
    let rz_mat = Gate::Rz(0.7).matrix_2x2();

    let mut c1 = Circuit::new(4, 0);
    c1.add_gate(Gate::X, &[0]);
    c1.add_gate(Gate::H, &[2]);
    c1.add_gate(Gate::Cu(Box::new(rz_mat)), &[0, 2]);

    let mut c2 = Circuit::new(4, 0);
    c2.add_gate(Gate::X, &[2]);
    c2.add_gate(Gate::H, &[0]);
    c2.add_gate(Gate::Cu(Box::new(rz_mat)), &[2, 0]);

    let mut b1 = StatevectorBackend::new(42);
    sim::run_on(&mut b1, &c1).unwrap();
    let mut b2 = StatevectorBackend::new(42);
    sim::run_on(&mut b2, &c2).unwrap();

    let p1: f64 = b1.probabilities().unwrap().iter().sum();
    let p2: f64 = b2.probabilities().unwrap().iter().sum();
    assert!((p1 - 1.0).abs() < 1e-12);
    assert!((p2 - 1.0).abs() < 1e-12);
}

#[test]
fn test_mcu_toffoli() {
    use crate::gates::McuData;
    let x_mat = Gate::X.matrix_2x2();
    let toffoli = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 2,
    }));

    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(toffoli.clone(), &[1, 2, 0]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0b111] - 1.0).abs() < 1e-12);

    let mut c2 = Circuit::new(3, 0);
    c2.add_gate(Gate::X, &[2]);
    c2.add_gate(toffoli, &[1, 2, 0]);
    let mut b2 = StatevectorBackend::new(42);
    sim::run_on(&mut b2, &c2).unwrap();
    let probs2 = b2.probabilities().unwrap();
    assert!((probs2[0b100] - 1.0).abs() < 1e-12);
}

#[test]
fn test_mcu_ccz() {
    use crate::gates::McuData;
    let z_mat = Gate::Z.matrix_2x2();
    let ccz = Gate::Mcu(Box::new(McuData {
        mat: z_mat,
        num_controls: 2,
    }));

    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(ccz, &[0, 1, 2]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let sv = b.state_vector();
    let expected_amp = 1.0 / 8.0_f64.sqrt();
    assert!((sv[7].re - (-expected_amp)).abs() < 1e-12);
    for (i, amp) in sv.iter().enumerate().take(7) {
        assert!((amp.re - expected_amp).abs() < 1e-12, "sv[{i}] wrong");
    }
}

#[test]
fn test_mcu_3_controls() {
    use crate::gates::McuData;
    let x_mat = Gate::X.matrix_2x2();
    let cccx = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 3,
    }));

    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(cccx.clone(), &[0, 1, 2, 3]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0b1111] - 1.0).abs() < 1e-12);

    let mut c2 = Circuit::new(4, 0);
    c2.add_gate(Gate::X, &[1]);
    c2.add_gate(Gate::X, &[2]);
    c2.add_gate(cccx, &[0, 1, 2, 3]);
    let mut b2 = StatevectorBackend::new(42);
    sim::run_on(&mut b2, &c2).unwrap();
    let probs2 = b2.probabilities().unwrap();
    assert!((probs2[0b0110] - 1.0).abs() < 1e-12);
}

#[test]
fn test_mcu_matches_cu_for_single_control() {
    use crate::gates::McuData;
    let h_mat = Gate::H.matrix_2x2();

    let mut c_cu = Circuit::new(3, 0);
    c_cu.add_gate(Gate::X, &[0]);
    c_cu.add_gate(Gate::X, &[1]);
    c_cu.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat: h_mat,
            num_controls: 2,
        })),
        &[0, 1, 2],
    );

    let mut b_cu = StatevectorBackend::new(42);
    sim::run_on(&mut b_cu, &c_cu).unwrap();
    let probs = b_cu.probabilities().unwrap();
    assert!((probs[0b011] - 0.5).abs() < 1e-12);
    assert!((probs[0b111] - 0.5).abs() < 1e-12);
}

#[test]
fn test_mask_indices_bijection_2ctrl() {
    for num_qubits in 3..=5 {
        let controls = vec![0usize, 2];
        let target = 1usize;
        let ctrl_mask: usize = controls.iter().map(|&q| 1usize << q).fold(0, |a, b| a | b);
        let tgt_mask = 1usize << target;
        let num_special = controls.len() + 1;
        let mut sorted = controls.clone();
        sorted.push(target);
        sorted.sort_unstable();

        if num_special > num_qubits {
            continue;
        }

        let num_iters = 1usize << (num_qubits - num_special);
        let mut seen = std::collections::HashSet::new();

        for i in 0..num_iters {
            let mut base = i;
            for &q in &sorted {
                base = insert_zero_bit(base, q);
            }
            let idx0 = base | ctrl_mask;
            let idx1 = idx0 | tgt_mask;

            for &ctrl in &controls {
                assert_ne!(idx0 & (1 << ctrl), 0, "ctrl bit {ctrl} not set");
            }
            assert_eq!(idx0 & tgt_mask, 0, "tgt bit set in idx0");
            assert_ne!(idx1 & tgt_mask, 0, "tgt bit not set in idx1");
            assert!(idx0 < (1 << num_qubits));
            assert!(idx1 < (1 << num_qubits));
            assert!(seen.insert(idx0), "duplicate idx0={idx0}");
            assert!(seen.insert(idx1), "duplicate idx1={idx1}");
        }

        let expected = 1usize << (num_qubits - num_special + 1);
        assert_eq!(seen.len(), expected);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_mcu_toffoli_16q() {
    use crate::gates::McuData;
    let x_mat = Gate::X.matrix_2x2();
    let toffoli = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 2,
    }));

    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(toffoli, &[0, 1, 2]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0b111] - 1.0).abs() < 1e-12);
}

#[test]
fn test_cu_phase_cz_equivalent() {
    let mut c1 = Circuit::new(2, 0);
    c1.add_gate(Gate::H, &[0]);
    c1.add_gate(Gate::H, &[1]);
    c1.add_gate(Gate::cphase(std::f64::consts::PI), &[0, 1]);

    let mut c2 = Circuit::new(2, 0);
    c2.add_gate(Gate::H, &[0]);
    c2.add_gate(Gate::H, &[1]);
    c2.add_gate(Gate::Cz, &[0, 1]);

    let mut b1 = StatevectorBackend::new(42);
    sim::run_on(&mut b1, &c1).unwrap();
    let mut b2 = StatevectorBackend::new(42);
    sim::run_on(&mut b2, &c2).unwrap();

    let sv1 = b1.state_vector();
    let sv2 = b2.state_vector();
    for (a, b) in sv1.iter().zip(sv2) {
        assert!((a - b).norm() < 1e-12);
    }
}

#[test]
fn test_cu_phase_no_action_control_zero() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::cphase(1.0), &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let sv = b.state_vector();
    let h = 1.0 / 2.0_f64.sqrt();
    assert!((sv[0].re - h).abs() < 1e-12);
    assert!(sv[1].norm() < 1e-12);
    assert!((sv[2].re - h).abs() < 1e-12);
    assert!(sv[3].norm() < 1e-12);
}

#[test]
fn test_cu_phase_applies_phase() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_4), &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let sv = b.state_vector();
    let expected = num_complex::Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    assert!((sv[3] - expected).norm() < 1e-12);
}

#[test]
fn test_cu_phase_both_qubit_orderings() {
    let theta = 0.7;
    let mut c1 = Circuit::new(3, 0);
    c1.add_gate(Gate::X, &[0]);
    c1.add_gate(Gate::H, &[2]);
    c1.add_gate(Gate::cphase(theta), &[0, 2]);

    let mut c2 = Circuit::new(3, 0);
    c2.add_gate(Gate::X, &[0]);
    c2.add_gate(Gate::H, &[2]);
    c2.add_gate(Gate::cphase(theta), &[2, 0]);

    let mut b1 = StatevectorBackend::new(42);
    sim::run_on(&mut b1, &c1).unwrap();
    let mut b2 = StatevectorBackend::new(42);
    sim::run_on(&mut b2, &c2).unwrap();

    let p1 = b1.probabilities().unwrap();
    let p2 = b2.probabilities().unwrap();
    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn test_cu_phase_qft_4q() {
    let n = 4;
    let mut c = Circuit::new(n, 0);
    for i in 0..n {
        c.add_gate(Gate::H, &[i]);
        for j in (i + 1)..n {
            let theta = std::f64::consts::TAU / (1u64 << (j - i)) as f64;
            c.add_gate(Gate::cphase(theta), &[i, j]);
        }
    }
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    for p in &probs {
        assert!((p - 1.0 / 16.0).abs() < 1e-10);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_cu_phase_16q() {
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_3), &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let sv = b.state_vector();
    let expected = num_complex::Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_3);
    assert!((sv[3] - expected).norm() < 1e-12);
}

#[test]
fn test_batch_phase_lut_single_group() {
    use num_complex::Complex64;
    let n = 6;
    let control = 0;
    let phases: Vec<(usize, Complex64)> = (1..5)
        .map(|q| {
            let theta = std::f64::consts::TAU / (1u64 << q) as f64;
            (q, Complex64::from_polar(1.0, theta))
        })
        .collect();

    let mut b_lut = StatevectorBackend::new(42);
    b_lut.init(n, 0).unwrap();
    for i in 0..(1usize << n) {
        b_lut.state[i] = Complex64::new(i as f64, (i as f64) * 0.1);
    }
    let original: Vec<Complex64> = b_lut.state.clone();

    b_lut.apply_batch_phase(control, &phases);

    let ctrl_mask = 1usize << control;
    for (i, (actual, &orig)) in b_lut.state.iter().zip(original.iter()).enumerate() {
        if (i & ctrl_mask) == 0 {
            assert!(
                (actual - orig).norm() < 1e-12,
                "control-0 element {i} should be unchanged"
            );
        } else {
            let mut expected_phase = Complex64::new(1.0, 0.0);
            for &(q, p) in &phases {
                if (i & (1 << q)) != 0 {
                    expected_phase *= p;
                }
            }
            let expected = orig * expected_phase;
            assert!(
                (actual - expected).norm() < 1e-10,
                "element {i}: expected {expected}, got {actual}",
            );
        }
    }
}

#[test]
fn test_batch_phase_lut_multi_group() {
    use num_complex::Complex64;
    let n = 12;
    let control = 0;
    let phases: Vec<(usize, Complex64)> = (1..11)
        .map(|q| {
            let theta = std::f64::consts::TAU / (1u64 << (q % 6 + 1)) as f64;
            (q, Complex64::from_polar(1.0, theta))
        })
        .collect();
    assert_eq!(phases.len(), 10);

    let mut b = StatevectorBackend::new(42);
    b.init(n, 0).unwrap();
    for i in 0..(1usize << n) {
        b.state[i] = Complex64::new(1.0 / (1usize << n) as f64, 0.0);
    }
    let original: Vec<Complex64> = b.state.clone();

    b.apply_batch_phase(control, &phases);

    let ctrl_mask = 1usize << control;
    for (i, (actual, &orig)) in b.state.iter().zip(original.iter()).enumerate() {
        if (i & ctrl_mask) == 0 {
            assert!((actual - orig).norm() < 1e-14);
        } else {
            let mut expected_phase = Complex64::new(1.0, 0.0);
            for &(q, p) in &phases {
                if (i & (1 << q)) != 0 {
                    expected_phase *= p;
                }
            }
            let expected = orig * expected_phase;
            assert!(
                (actual - expected).norm() < 1e-10,
                "element {i}: diff {}",
                (actual - expected).norm()
            );
        }
    }
}

#[test]
fn test_batch_phase_lut_large_k() {
    use num_complex::Complex64;
    let n = 20;
    let control = 0;
    let phases: Vec<(usize, Complex64)> = (1..n)
        .map(|q| {
            let theta = std::f64::consts::TAU / (1u64 << q) as f64;
            (q, Complex64::from_polar(1.0, theta))
        })
        .collect();
    assert_eq!(phases.len(), 19);

    let mut b = StatevectorBackend::new(42);
    b.init(n, 0).unwrap();
    b.state[0] = Complex64::new(0.0, 0.0);
    let all_ones = (1usize << n) - 1;
    b.state[all_ones] = Complex64::new(1.0, 0.0);

    b.apply_batch_phase(control, &phases);

    assert!(b.state[0].norm() < 1e-14, "state[0] should remain zero");

    let mut expected_phase = Complex64::new(1.0, 0.0);
    for &(_, p) in &phases {
        expected_phase *= p;
    }
    let expected = Complex64::new(1.0, 0.0) * expected_phase;
    assert!(
        (b.state[all_ones] - expected).norm() < 1e-10,
        "all-ones state: expected {expected}, got {}, diff {}",
        b.state[all_ones],
        (b.state[all_ones] - expected).norm()
    );
}

#[test]
fn test_multi_fused_matches_individual() {
    use crate::gates::MultiFusedData;

    let n = 4;
    let gates_data: Vec<(usize, [[Complex64; 2]; 2])> = vec![
        (0, Gate::H.matrix_2x2()),
        (1, Gate::T.matrix_2x2()),
        (2, Gate::Rx(1.23).matrix_2x2()),
        (3, Gate::Ry(0.45).matrix_2x2()),
    ];

    let mut b1 = StatevectorBackend::new(42);
    b1.init(n, 0).unwrap();
    for &(target, mat) in &gates_data {
        b1.apply_single_gate(target, mat);
    }
    let probs1 = b1.probabilities().unwrap();

    let mut b2 = StatevectorBackend::new(42);
    b2.init(n, 0).unwrap();
    b2.apply(&Instruction::Gate {
        gate: Gate::MultiFused(Box::new(MultiFusedData {
            gates: gates_data.clone(),
            all_diagonal: false,
        })),
        targets: gates_data.iter().map(|&(t, _)| t).collect(),
    })
    .unwrap();
    let probs2 = b2.probabilities().unwrap();

    assert_probs_close(&probs1, &probs2, 1e-12);
}

#[test]
fn test_multi_fused_via_sim_run() {
    let n = 16;
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Ry(0.5 + q as f64 * 0.1), &[q]);
        c.add_gate(Gate::Rz(1.0 + q as f64 * 0.2), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 1]);
    for q in 0..n {
        c.add_gate(Gate::Rx(0.3 + q as f64 * 0.15), &[q]);
    }

    let mut b1 = StatevectorBackend::new(42);
    b1.init(n, 0).unwrap();
    for inst in &c.instructions {
        b1.apply(inst).unwrap();
    }
    let probs1 = b1.probabilities().unwrap();

    let mut b2 = StatevectorBackend::new(42);
    sim::run_on(&mut b2, &c).unwrap();
    let probs2 = b2.probabilities().unwrap();

    assert_probs_close(&probs1, &probs2, 1e-10);
}

#[test]
fn test_multi_fused_mixed_targets() {
    use crate::gates::MultiFusedData;

    let n = 8;
    let gates_data: Vec<(usize, [[Complex64; 2]; 2])> = vec![
        (0, Gate::H.matrix_2x2()),
        (3, Gate::S.matrix_2x2()),
        (7, Gate::T.matrix_2x2()),
        (1, Gate::Rx(0.5).matrix_2x2()),
        (5, Gate::Ry(1.0).matrix_2x2()),
    ];

    let mut b1 = StatevectorBackend::new(42);
    b1.init(n, 0).unwrap();
    for &(target, mat) in &gates_data {
        b1.apply_single_gate(target, mat);
    }
    let probs1 = b1.probabilities().unwrap();

    let mut b2 = StatevectorBackend::new(42);
    b2.init(n, 0).unwrap();
    b2.apply(&Instruction::Gate {
        gate: Gate::MultiFused(Box::new(MultiFusedData {
            gates: gates_data.clone(),
            all_diagonal: false,
        })),
        targets: gates_data.iter().map(|&(t, _)| t).collect(),
    })
    .unwrap();
    let probs2 = b2.probabilities().unwrap();

    assert_probs_close(&probs1, &probs2, 1e-12);
}

#[test]
fn test_multi_fused_fusion_pass() {
    use crate::circuit::fusion::{fuse_multi_1q_gates, fuse_single_qubit_gates};

    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::S, &[2]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Ry(1.0), &[2]);
    c.add_gate(Gate::Rz(2.0), &[3]);

    let pass1 = fuse_single_qubit_gates(&c);
    let pass2 = fuse_multi_1q_gates(pass1);

    let multi_count = pass2
        .instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate {
                    gate: Gate::MultiFused(_),
                    ..
                }
            )
        })
        .count();
    assert!(multi_count >= 1, "should have at least one MultiFused gate");

    let mut b1 = StatevectorBackend::new(42);
    b1.init(c.num_qubits, 0).unwrap();
    for inst in &c.instructions {
        b1.apply(inst).unwrap();
    }
    let probs1 = b1.probabilities().unwrap();

    let mut b2 = StatevectorBackend::new(42);
    b2.init(pass2.num_qubits, 0).unwrap();
    for inst in &pass2.instructions {
        b2.apply(inst).unwrap();
    }
    let probs2 = b2.probabilities().unwrap();

    assert_probs_close(&probs1, &probs2, 1e-12);
}

#[test]
fn test_multi_fused_three_tier_targets() {
    use crate::gates::MultiFusedData;

    let n = 18;
    let gates_data: Vec<(usize, [[Complex64; 2]; 2])> = vec![
        (2, Gate::H.matrix_2x2()),
        (8, Gate::Rx(0.7).matrix_2x2()),
        (13, Gate::S.matrix_2x2()),
        (15, Gate::Ry(1.3).matrix_2x2()),
        (17, Gate::Rz(2.1).matrix_2x2()),
    ];

    let mut b1 = StatevectorBackend::new(42);
    b1.init(n, 0).unwrap();
    for &(target, mat) in &gates_data {
        b1.apply_single_gate(target, mat);
    }
    let probs1 = b1.probabilities().unwrap();

    let mut b2 = StatevectorBackend::new(42);
    b2.init(n, 0).unwrap();
    b2.apply(&Instruction::Gate {
        gate: Gate::MultiFused(Box::new(MultiFusedData {
            gates: gates_data.clone(),
            all_diagonal: false,
        })),
        targets: gates_data.iter().map(|&(t, _)| t).collect(),
    })
    .unwrap();
    let probs2 = b2.probabilities().unwrap();

    assert_probs_close(&probs1, &probs2, 1e-12);
}

// ---- High-qubit parallel correctness tests ----
//
// These tests target specific parallel code paths at and above the
// PARALLEL_THRESHOLD_QUBITS (14) boundary to validate that Rayon-parallelized
// kernels produce identical results to sequential execution.

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_threshold_boundary_14q() {
    let mut c = Circuit::new(14, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0] - 0.5).abs() < 1e-12);
    assert!((probs[3] - 0.5).abs() < 1e-12);
    let rest_sum: f64 = probs
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != 0 && i != 3)
        .map(|(_, p)| p)
        .sum();
    assert!(rest_sum.abs() < 1e-12);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_batch_phase_16q() {
    use std::f64::consts::PI;
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::H, &[0]);
    for target in 1..8 {
        let theta = PI / (1u64 << target) as f64;
        c.add_gate(Gate::cphase(theta), &[0, target]);
    }
    let mut b_par = StatevectorBackend::new(42);
    sim::run_on(&mut b_par, &c).unwrap();
    let probs_par = b_par.probabilities().unwrap();

    let mut b_seq = StatevectorBackend::new(42);
    b_seq.init(16, 0).unwrap();
    for instr in &c.instructions {
        b_seq.apply(instr).unwrap();
    }
    let probs_seq = b_seq.probabilities().unwrap();

    assert_probs_close(&probs_par, &probs_seq, 1e-10);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_diagonal_gates_16q() {
    let mut c = Circuit::new(16, 0);
    for q in 0..16 {
        c.add_gate(Gate::H, &[q]);
    }
    c.add_gate(Gate::Z, &[0]);
    c.add_gate(Gate::S, &[3]);
    c.add_gate(Gate::T, &[7]);
    c.add_gate(Gate::P(std::f64::consts::FRAC_PI_3), &[11]);
    c.add_gate(Gate::Rz(1.5), &[15]);

    let mut b_par = StatevectorBackend::new(42);
    sim::run_on(&mut b_par, &c).unwrap();
    let probs_par = b_par.probabilities().unwrap();

    let mut b_seq = StatevectorBackend::new(42);
    b_seq.init(16, 0).unwrap();
    for instr in &c.instructions {
        b_seq.apply(instr).unwrap();
    }
    let probs_seq = b_seq.probabilities().unwrap();

    assert_probs_close(&probs_par, &probs_seq, 1e-10);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_multi_fused_three_tier_20q() {
    let mut c = Circuit::new(20, 0);
    for q in 0..20 {
        c.add_gate(Gate::H, &[q]);
    }
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    let expected = 1.0 / (1u64 << 20) as f64;
    for (i, &p) in probs.iter().enumerate() {
        assert!(
            (p - expected).abs() < 1e-10,
            "prob[{i}] = {p}, expected {expected}"
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_cphase_distant_16q() {
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[15]);
    c.add_gate(Gate::cphase(std::f64::consts::PI), &[15, 0]);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let sv = b.state_vector();
    let idx_00 = 1 << 15;
    let idx_11 = (1 << 15) | 1;
    let h = std::f64::consts::FRAC_1_SQRT_2;
    assert!((sv[idx_00].re - h).abs() < 1e-12);
    assert!((sv[idx_11].re - (-h)).abs() < 1e-12);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_multi_2q_20q() {
    let mut c = Circuit::new(20, 0);
    for q in 0..20 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in (0..19).step_by(2) {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let mut b_par = StatevectorBackend::new(42);
    sim::run_on(&mut b_par, &c).unwrap();
    let probs_par = b_par.probabilities().unwrap();

    let mut b_seq = StatevectorBackend::new(42);
    b_seq.init(20, 0).unwrap();
    for instr in &c.instructions {
        b_seq.apply(instr).unwrap();
    }
    let probs_seq = b_seq.probabilities().unwrap();

    assert_probs_close(&probs_par, &probs_seq, 1e-10);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_measure_deterministic_20q() {
    let mut c = Circuit::new(20, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let bits = b.classical_results();
    assert!(bits[0], "qubit 0 in |1> must measure 1");
    let probs = b.probabilities().unwrap();
    assert!((probs[1] - 1.0).abs() < 1e-12);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_all_identity_20q() {
    let mut c = Circuit::new(20, 0);
    for q in 0..20 {
        c.add_gate(Gate::Id, &[q]);
    }
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!(
        (probs[0] - 1.0).abs() < 1e-12,
        "all-identity must stay |0...0>"
    );
    for (i, &p) in probs.iter().enumerate().skip(1) {
        assert!(p.abs() < 1e-14, "prob[{i}] should be 0, got {p}");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_self_inverse_cancel_20q() {
    let mut c = Circuit::new(20, 0);
    for q in 0..20 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..19 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in (0..19).rev() {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..20 {
        c.add_gate(Gate::H, &[q]);
    }
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!(
        (probs[0] - 1.0).abs() < 1e-10,
        "H-CX-CXinv-H must cancel to |0...0>, got prob[0]={}",
        probs[0]
    );
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_conditional_gate_16q() {
    use crate::circuit::{smallvec, ClassicalCondition};
    let mut c = Circuit::new(16, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: smallvec![15],
    });
    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let bits = b.classical_results();
    assert!(bits[0], "q[0] was |1>, must measure 1");
    let probs = b.probabilities().unwrap();
    let idx_both = 1 | (1 << 15);
    assert!(
        (probs[idx_both] - 1.0).abs() < 1e-12,
        "conditional X on q[15] should fire; expected |1...01>, got prob[{}]={}",
        idx_both,
        probs[idx_both]
    );
}

// ── Deferred measurement normalization tests ──

#[test]
fn test_deferred_norm_single_measure() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);

    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();

    assert!(
        b.pending_norm != 1.0,
        "pending_norm should be deferred after superposition measurement"
    );

    let probs = b.probabilities().unwrap();
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-12,
        "probabilities must sum to 1.0, got {}",
        sum
    );
    let outcome = b.classical_results()[0];
    let expected_idx = if outcome { 1 } else { 0 };
    assert!(
        (probs[expected_idx] - 1.0).abs() < 1e-12,
        "after measure, collapsed state prob[{}]={}, expected 1.0",
        expected_idx,
        probs[expected_idx]
    );
}

#[test]
fn test_deferred_norm_multi_measure() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);

    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();

    let bits = b.classical_results();
    assert!(bits[0], "q[0] was |1>");
    assert!(bits[1], "q[1] was |1>");

    let probs = b.probabilities().unwrap();
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-12,
        "probabilities must sum to 1.0 after 2 measurements, got {}",
        sum
    );
    assert!(
        (probs[3] - 1.0).abs() < 1e-12,
        "state should be |11>; prob[3]={}",
        probs[3]
    );
}

#[test]
fn test_deferred_norm_gate_after_measure() {
    let mut c = Circuit::new(3, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.add_gate(Gate::H, &[1]);

    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();

    assert!(b.classical_results()[0], "q[0] was |1>");

    let probs = b.probabilities().unwrap();
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-12,
        "probabilities must sum to 1.0, got {}",
        sum
    );
    // State: q0=|1>, q1=H|0>=(|0>+|1>)/sqrt2, q2=|0>
    // -> |001> and |011> each with prob 0.5
    assert!(
        (probs[0b001] - 0.5).abs() < 1e-12,
        "expected prob[001]=0.5, got {}",
        probs[0b001]
    );
    assert!(
        (probs[0b011] - 0.5).abs() < 1e-12,
        "expected prob[011]=0.5, got {}",
        probs[0b011]
    );
}

#[test]
fn test_deferred_norm_export_statevector() {
    use num_complex::Complex64;

    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);

    let mut b = StatevectorBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();

    let sv = b.export_statevector().unwrap();
    let norm: f64 = sv.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        (norm - 1.0).abs() < 1e-12,
        "exported statevector must be normalized, got norm={}",
        norm
    );
    assert!(
        (sv[1] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
        "state should be |1>, got sv[1]={:?}",
        sv[1]
    );
}

#[cfg(feature = "gpu")]
mod gpu_scaffold {
    use super::*;
    use crate::error::PrismError;
    use crate::gpu::GpuContext;

    #[test]
    fn with_gpu_stores_context() {
        let ctx = GpuContext::stub_for_tests();
        let backend = StatevectorBackend::new(42).with_gpu(ctx);
        assert_eq!(backend.name(), "statevector");
    }

    #[test]
    fn init_with_gpu_returns_unsupported_until_kernels_land() {
        let ctx = GpuContext::stub_for_tests();
        let mut backend = StatevectorBackend::new(42).with_gpu(ctx);
        let err = backend.init(4, 0).unwrap_err();
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));
    }

    #[test]
    fn without_gpu_init_still_works() {
        let mut backend = StatevectorBackend::new(42);
        assert!(backend.init(4, 0).is_ok());
        assert_eq!(backend.num_qubits(), 4);
    }

    /// Real-device smoke test: |0000⟩ state round-trips correctly through the GPU path.
    ///
    /// When CUDA is not available or the driver rejects the PTX, prints a SKIP message and
    /// returns without failing. This avoids fighting CI on machines without a usable GPU.
    #[test]
    fn gpu_init_and_readback_zero_state() {
        let ctx = match GpuContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("SKIP: no usable GPU ({e})");
                return;
            }
        };
        let mut backend = StatevectorBackend::new(42).with_gpu(ctx);
        backend.init(4, 0).expect("GPU init failed");

        let probs = backend.probabilities().expect("GPU probabilities failed");
        assert_eq!(probs.len(), 16);
        assert!((probs[0] - 1.0).abs() < 1e-12, "p[0] = {}", probs[0]);
        for (i, &p) in probs.iter().enumerate().skip(1) {
            assert!(p.abs() < 1e-12, "p[{}] = {} should be 0", i, p);
        }

        let sv = backend.export_statevector().expect("GPU export failed");
        assert_eq!(sv.len(), 16);
        assert!((sv[0].re - 1.0).abs() < 1e-12 && sv[0].im.abs() < 1e-12);
        for amp in &sv[1..] {
            assert!(amp.norm() < 1e-12);
        }
    }

    /// Bell state end-to-end on GPU, comparing against CPU. Exercises H (1q) + CX (2q)
    /// kernels, host↔device transfer, and the full dispatcher.
    #[test]
    fn gpu_bell_state_matches_cpu() {
        let ctx = match GpuContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        use crate::circuit::{Circuit, Instruction};
        use crate::gates::Gate;

        let mut circuit = Circuit::new(2, 0);
        circuit.instructions.push(Instruction::Gate {
            gate: Gate::H,
            targets: crate::circuit::smallvec![0],
        });
        circuit.instructions.push(Instruction::Gate {
            gate: Gate::Cx,
            targets: crate::circuit::smallvec![0, 1],
        });

        let mut cpu = StatevectorBackend::new(42);
        cpu.init(2, 0).unwrap();
        for inst in &circuit.instructions {
            cpu.apply(inst).unwrap();
        }
        let cpu_probs = cpu.probabilities().unwrap();

        let mut gpu = StatevectorBackend::new(42).with_gpu(ctx);
        gpu.init(2, 0).unwrap();
        for inst in &circuit.instructions {
            gpu.apply(inst).unwrap();
        }
        let gpu_probs = gpu.probabilities().unwrap();

        assert_eq!(cpu_probs.len(), gpu_probs.len());
        for (i, (c, g)) in cpu_probs.iter().zip(gpu_probs.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-12,
                "p[{}]: cpu={}, gpu={}, diff={}",
                i,
                c,
                g,
                (c - g).abs()
            );
        }
        // Bell state: p[00] ≈ p[11] ≈ 0.5; p[01] = p[10] = 0.
        assert!((gpu_probs[0] - 0.5).abs() < 1e-12);
        assert!((gpu_probs[3] - 0.5).abs() < 1e-12);
        assert!(gpu_probs[1].abs() < 1e-12);
        assert!(gpu_probs[2].abs() < 1e-12);
    }
}

use super::*;
use crate::circuit::smallvec;
use std::f64::consts::PI;

const EPS: f64 = 1e-12;

fn init_backend(n: usize) -> ProductStateBackend {
    let mut b = ProductStateBackend::new(42);
    b.init(n, 0).unwrap();
    b
}

#[test]
fn test_init_all_zero() {
    let b = init_backend(4);
    let probs = b.probabilities().unwrap();
    assert!((probs[0] - 1.0).abs() < EPS);
    for p in &probs[1..] {
        assert!(p.abs() < EPS);
    }
}

#[test]
fn test_h_creates_superposition() {
    let mut b = init_backend(1);
    b.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec![0],
    })
    .unwrap();
    let probs = b.probabilities().unwrap();
    assert!((probs[0] - 0.5).abs() < EPS);
    assert!((probs[1] - 0.5).abs() < EPS);
}

#[test]
fn test_x_flips() {
    let mut b = init_backend(1);
    b.apply(&Instruction::Gate {
        gate: Gate::X,
        targets: smallvec![0],
    })
    .unwrap();
    let probs = b.probabilities().unwrap();
    assert!(probs[0].abs() < EPS);
    assert!((probs[1] - 1.0).abs() < EPS);
}

#[test]
fn test_parametric_gates() {
    for gate in [Gate::Rx(PI / 3.0), Gate::Ry(PI / 4.0), Gate::Rz(PI / 6.0)] {
        let mut b = init_backend(1);
        b.apply(&Instruction::Gate {
            gate,
            targets: smallvec![0],
        })
        .unwrap();
        let probs = b.probabilities().unwrap();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "probabilities must sum to 1");
    }
}

#[test]
fn test_multi_qubit_independent() {
    let mut b = init_backend(3);
    b.apply(&Instruction::Gate {
        gate: Gate::X,
        targets: smallvec![0],
    })
    .unwrap();
    b.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec![1],
    })
    .unwrap();

    let probs = b.probabilities().unwrap();
    // q0=1, q1=+, q2=0 → indices with bit0=1, bit2=0
    // |001⟩ (idx 1) = 0.5, |011⟩ (idx 3) = 0.5
    assert!((probs[1] - 0.5).abs() < EPS);
    assert!((probs[3] - 0.5).abs() < EPS);
    for (i, &p) in probs.iter().enumerate() {
        if i != 1 && i != 3 {
            assert!(p.abs() < EPS, "prob[{i}] should be 0, got {p}");
        }
    }
}

#[test]
fn test_rejects_cx() {
    let mut b = init_backend(2);
    let result = b.apply(&Instruction::Gate {
        gate: Gate::Cx,
        targets: smallvec![0, 1],
    });
    assert!(matches!(result, Err(PrismError::BackendUnsupported { .. })));
}

#[test]
fn test_rejects_cz() {
    let mut b = init_backend(2);
    let result = b.apply(&Instruction::Gate {
        gate: Gate::Cz,
        targets: smallvec![0, 1],
    });
    assert!(matches!(result, Err(PrismError::BackendUnsupported { .. })));
}

#[test]
fn test_rejects_swap() {
    let mut b = init_backend(2);
    let result = b.apply(&Instruction::Gate {
        gate: Gate::Swap,
        targets: smallvec![0, 1],
    });
    assert!(matches!(result, Err(PrismError::BackendUnsupported { .. })));
}

#[test]
fn test_rejects_cu() {
    let mut b = init_backend(2);
    let result = b.apply(&Instruction::Gate {
        gate: Gate::Cu(Box::new(Gate::H.matrix_2x2())),
        targets: smallvec![0, 1],
    });
    assert!(matches!(result, Err(PrismError::BackendUnsupported { .. })));
}

#[test]
fn test_rejects_mcu() {
    use crate::gates::McuData;
    let mut b = init_backend(3);
    let result = b.apply(&Instruction::Gate {
        gate: Gate::Mcu(Box::new(McuData {
            mat: Gate::X.matrix_2x2(),
            num_controls: 2,
        })),
        targets: smallvec![0, 1, 2],
    });
    assert!(matches!(result, Err(PrismError::BackendUnsupported { .. })));
}

#[test]
fn test_measurement_collapses() {
    let mut b = ProductStateBackend::new(42);
    b.init(1, 1).unwrap();
    b.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec![0],
    })
    .unwrap();
    b.apply(&Instruction::Measure {
        qubit: 0,
        classical_bit: 0,
    })
    .unwrap();
    let outcome = b.classical_results()[0];

    let probs = b.probabilities().unwrap();
    if outcome {
        assert!(probs[0].abs() < EPS);
        assert!((probs[1] - 1.0).abs() < EPS);
    } else {
        assert!((probs[0] - 1.0).abs() < EPS);
        assert!(probs[1].abs() < EPS);
    }
}

#[test]
fn test_probabilities_tensor_product() {
    let mut b = init_backend(3);
    // q0: H → equal superposition
    b.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec![0],
    })
    .unwrap();
    // q1: X → |1⟩
    b.apply(&Instruction::Gate {
        gate: Gate::X,
        targets: smallvec![1],
    })
    .unwrap();
    // q2: stays |0⟩

    let probs = b.probabilities().unwrap();
    assert_eq!(probs.len(), 8);

    // q0=0/1 (50/50), q1=1 (certain), q2=0 (certain)
    // |010⟩ (idx 2) = 0.5, |011⟩ (idx 3) = 0.5
    assert!((probs[2] - 0.5).abs() < EPS);
    assert!((probs[3] - 0.5).abs() < EPS);
    for (i, &p) in probs.iter().enumerate() {
        if i != 2 && i != 3 {
            assert!(p.abs() < EPS, "prob[{i}] should be 0, got {p}");
        }
    }
}

#[test]
fn test_sample_basis_states_matches_the_dense_distribution() {
    let mut b = init_backend(6);
    for q in 0..6 {
        b.apply(&Instruction::Gate {
            gate: Gate::Ry(0.3 + 0.2 * q as f64),
            targets: smallvec![q],
        })
        .unwrap();
    }

    let probs = b.probabilities().unwrap();
    let shots = 40_000;
    let samples = b.sample_basis_states(shots, 42).unwrap();

    let mut counts = vec![0usize; probs.len()];
    for shot in 0..shots {
        let index = (0..6)
            .filter(|&q| samples.bit(shot, q))
            .fold(0, |acc, q| acc | 1 << q);
        counts[index] += 1;
    }
    for (index, &p) in probs.iter().enumerate() {
        let frequency = counts[index] as f64 / shots as f64;
        let band = 4.0 * (p * (1.0 - p) / shots as f64).sqrt() + 0.002;
        assert!(
            (frequency - p).abs() < band,
            "basis state {index}: sampled {frequency:.6} against {p:.6}"
        );
    }
}

#[test]
fn test_sample_basis_states_repeats_from_the_seed() {
    let mut b = init_backend(3);
    b.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec![1],
    })
    .unwrap();
    let first = b.sample_basis_states(64, 42).unwrap();
    let second = b.sample_basis_states(64, 42).unwrap();
    let other_seed = b.sample_basis_states(64, 43).unwrap();

    let bits = |s: &BasisSamples| -> Vec<bool> {
        (0..64)
            .flat_map(|shot| (0..3).map(move |q| (shot, q)))
            .map(|(shot, q)| s.bit(shot, q))
            .collect()
    };
    assert_eq!(bits(&first), bits(&second));
    assert_ne!(bits(&first), bits(&other_seed));
}

#[test]
fn test_pauli_expectations_are_normalization_independent() {
    let mut b = init_backend(2);
    b.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec![0],
    })
    .unwrap();
    let scaled = b.qubits[1].map(|amp| amp * 3.0);
    b.qubits[1] = scaled;

    let values = b
        .pauli_expectations(&[
            vec![PauliTerm::x(0)],
            vec![PauliTerm::z(1)],
            vec![PauliTerm::x(0), PauliTerm::z(1)],
            vec![],
        ])
        .unwrap();
    for (got, want) in values.iter().zip(&[1.0, 1.0, 1.0, 1.0]) {
        assert!((got - want).abs() < EPS, "got {got}, want {want}");
    }
}

#[test]
fn test_pauli_expectations_reject_bad_observables() {
    let b = init_backend(2);
    assert!(matches!(
        b.pauli_expectations(&[vec![PauliTerm::z(2)]]),
        Err(PrismError::InvalidQubit { .. })
    ));
    assert!(matches!(
        b.pauli_expectations(&[vec![PauliTerm::z(0), PauliTerm::x(0)]]),
        Err(PrismError::InvalidParameter { .. })
    ));
}

#[test]
fn test_fused_gate() {
    use crate::gates::mat_mul_2x2;

    let fused_mat = mat_mul_2x2(&Gate::T.matrix_2x2(), &Gate::H.matrix_2x2());
    let mut b = init_backend(1);
    b.apply(&Instruction::Gate {
        gate: Gate::Fused(Box::new(fused_mat)),
        targets: smallvec![0],
    })
    .unwrap();

    let mut b2 = init_backend(1);
    b2.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec![0],
    })
    .unwrap();
    b2.apply(&Instruction::Gate {
        gate: Gate::T,
        targets: smallvec![0],
    })
    .unwrap();

    let p1 = b.probabilities().unwrap();
    let p2 = b2.probabilities().unwrap();
    for (a, b) in p1.iter().zip(&p2) {
        assert!((a - b).abs() < EPS);
    }
}

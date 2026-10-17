use super::*;

fn factored_2x3() -> Probabilities {
    Probabilities::Factored {
        blocks: vec![
            FactoredBlock {
                probs: vec![0.25, 0.75],
                mask: 0b001,
            },
            FactoredBlock {
                probs: vec![0.1, 0.2, 0.3, 0.4],
                mask: 0b110,
            },
        ],
        total_qubits: 3,
    }
}

#[test]
fn dense_basic_accessors() {
    let p = Probabilities::Dense(vec![0.1, 0.2, 0.3, 0.4]);
    assert_eq!(p.len(), 4);
    assert!(!p.is_empty());
    assert_eq!(p.get(2), 0.3);
    assert_eq!(p.get(3), 0.4);
    assert_eq!(p.to_vec(), vec![0.1, 0.2, 0.3, 0.4]);
    let collected: Vec<f64> = p.iter().collect();
    assert_eq!(collected, vec![0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn factored_get_matches_to_vec() {
    let p = factored_2x3();
    assert_eq!(p.len(), 8);
    let dense = p.to_vec();
    for (i, d) in dense.iter().enumerate() {
        assert!((p.get(i) - d).abs() < 1e-12);
    }
    let sum: f64 = dense.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn factored_iter_matches_to_vec() {
    let p = factored_2x3();
    let dense = p.to_vec();
    let iter_vec: Vec<f64> = p.iter().collect();
    assert_eq!(iter_vec.len(), dense.len());
    for (a, b) in iter_vec.iter().zip(dense.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn factored_iter_size_hint_exact() {
    let p = factored_2x3();
    let mut it = p.iter();
    assert_eq!(it.size_hint(), (8, Some(8)));
    assert_eq!(it.len(), 8);
    it.next();
    assert_eq!(it.size_hint(), (7, Some(7)));
    for _ in 0..7 {
        it.next();
    }
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert!(it.next().is_none());
}

#[test]
fn dense_iter_size_hint_exact() {
    let p = Probabilities::Dense(vec![0.5, 0.5]);
    let it = p.iter();
    assert_eq!(it.size_hint(), (2, Some(2)));
}

#[test]
fn extract_block_bits_scalar_via_get() {
    let blocks = vec![FactoredBlock {
        probs: vec![0.0, 0.0, 0.0, 1.0],
        mask: 0b1010,
    }];
    let p = Probabilities::Factored {
        blocks,
        total_qubits: 4,
    };
    assert!((p.get(0b1010) - 1.0).abs() < 1e-12);
    assert!(p.get(0b0010).abs() < 1e-12);
    assert!(p.get(0b1000).abs() < 1e-12);
}

#[cfg(feature = "parallel")]
#[test]
fn factored_to_vec_parallel_path() {
    let mut blocks = Vec::new();
    for i in 0..5 {
        blocks.push(FactoredBlock {
            probs: vec![0.125; 8],
            mask: 0b111u64 << (3 * i),
        });
    }
    let p = Probabilities::Factored {
        blocks,
        total_qubits: 15,
    };
    let v = p.to_vec();
    assert_eq!(v.len(), 1 << 15);
    let sum: f64 = v.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
    for (i, val) in v.iter().enumerate() {
        assert!((p.get(i) - val).abs() < 1e-12);
    }
}

//! With `PRISM_MULTI_2Q_TILE_BITS=18` the in-place `Multi2q` tile reaches qubit 17,
//! one past the L3 tier. Own file: the knob is read once per process.

mod common;

use common::caps;
use common::gate_fixtures::{g, sample_4x4};
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::gates::{Gate, Multi2qData};

#[test]
fn multi_2q_at_the_widest_tile_applies_each_gate_once() {
    caps::set_once(&[("PRISM_MULTI_2Q_TILE_BITS", "18")]);
    let n = 22;
    let mat = sample_4x4();
    let gates = vec![(0, 17, mat), (1, 2, mat), (3, 16, mat)];

    let mut batched = StatevectorBackend::new(42);
    batched.init(n, 0).unwrap();
    let mut reference = StatevectorBackend::new(42);
    reference.init(n, 0).unwrap();
    for q in 0..n {
        batched
            .apply(&g(Gate::Ry(0.3 + 0.05 * q as f64), &[q]))
            .unwrap();
        reference
            .apply(&g(Gate::Ry(0.3 + 0.05 * q as f64), &[q]))
            .unwrap();
    }

    for &(q0, q1, mat) in &gates {
        reference
            .apply(&g(Gate::Fused2q(Box::new(mat)), &[q0, q1]))
            .unwrap();
    }
    batched
        .apply(&g(
            Gate::Multi2q(Box::new(Multi2qData::new(gates))),
            &[0, 1, 2, 3, 16, 17],
        ))
        .unwrap();

    for (i, (a, b)) in batched
        .state_vector()
        .iter()
        .zip(reference.state_vector())
        .enumerate()
    {
        assert!(
            (a - b).norm() < 1e-10 * b.norm().max(1.0),
            "amplitude {i}: batched {a}, reference {b}"
        );
    }
}

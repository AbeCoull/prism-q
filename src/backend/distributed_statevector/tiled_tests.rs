use super::*;
use crate::backend::distributed_statevector::DistributedStatevectorBackend;
use crate::circuit::builder::CircuitBuilder;
use crate::circuit::{Circuit, ClassicalCondition, smallvec};
use crate::distributed::DistributedContext;

const SEED: u64 = 42;
const TOL: f64 = 1e-10;

fn tiled_run(circuit: &Circuit, window_qubits: usize) -> TiledStatevector {
    let mut config = TileConfig::new(window_qubits);
    config.max_pass_rate = 1.0;
    let mut state =
        TiledStatevector::new(circuit.num_qubits, circuit.num_classical_bits, SEED, config)
            .expect("tiled state");
    state.run(&circuit.instructions).expect("tiled run");
    state
}

fn dense_reference(circuit: &Circuit) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(SEED);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .expect("dense init");
    for instruction in &circuit.instructions {
        backend.apply(instruction).expect("dense apply");
    }
    backend.export_statevector().expect("dense export")
}

fn serial_distributed_reference(circuit: &Circuit) -> (Vec<Complex64>, Vec<bool>) {
    let mut backend = DistributedStatevectorBackend::new(DistributedContext::serial(), SEED);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .expect("serial init");
    for instruction in &circuit.instructions {
        backend.apply(instruction).expect("serial apply");
    }
    let state = backend.export_statevector().expect("serial export");
    (state, backend.classical_results().to_vec())
}

fn assert_states_match(expected: &[Complex64], actual: &[Complex64], label: &str) {
    assert_eq!(expected.len(), actual.len(), "{label}: length mismatch");
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (e - a).norm() < TOL,
            "{label}: amplitude {i} expected {e}, got {a}"
        );
    }
}

#[test]
fn fresh_state_is_the_zero_ket() {
    let mut state = TiledStatevector::new(5, 0, SEED, TileConfig::new(2)).expect("state");
    let amps = state.export_statevector().expect("export");
    assert_eq!(amps.len(), 32);
    assert!((amps[0] - Complex64::new(1.0, 0.0)).norm() < TOL);
    assert!(amps[1..].iter().all(|a| a.norm() < TOL));
}

#[test]
fn x_on_qubit_zero_sets_index_one() {
    let mut b = CircuitBuilder::new(4);
    b.x(0);
    let mut state = tiled_run(&b.build(), 2);
    let amps = state.export_statevector().expect("export");
    assert!((amps[1] - Complex64::new(1.0, 0.0)).norm() < TOL);
}

fn mixed_unitary_circuit(n: usize) -> Circuit {
    let x = Gate::X.matrix_2x2();
    let mut b = CircuitBuilder::new(n);
    for q in 0..n {
        b.h(q);
    }
    for q in 0..n - 1 {
        b.cx(q, q + 1);
    }
    b.t(0)
        .s(n - 1)
        .rz(0.37, n / 2)
        .rx(0.9, 1)
        .ry(-0.4, n - 2)
        .cz(0, n - 1)
        .rzz(0.7, 1, n - 1)
        .swap(0, n - 1)
        .swap(n - 2, n - 1)
        .cx(n - 1, 0)
        .cphase(0.5, n - 2, 1)
        .mcu(x, &[0, n - 1], n - 2);
    for q in 0..n {
        b.h(q);
    }
    b.build()
}

#[test]
fn unitary_circuit_matches_dense_at_every_window_split() {
    let n = 6;
    let circuit = mixed_unitary_circuit(n);
    let expected = dense_reference(&circuit);
    for w in 1..=n {
        let mut state = tiled_run(&circuit, w);
        let actual = state.export_statevector().expect("export");
        assert_states_match(&expected, &actual, &format!("window {w}"));
    }
}

#[test]
fn diagonal_wall_joins_the_open_pass() {
    let n = 6;
    let mut b = CircuitBuilder::new(n);
    b.h(0)
        .h(1)
        .t(0)
        .t(4)
        .t(5)
        .cz(0, 5)
        .cz(4, 5)
        .rzz(0.3, 2, 5)
        .rz(1.1, 4)
        .cphase(0.25, 3, 4);
    let circuit = b.build();
    let mut state = tiled_run(&circuit, 2);
    assert_eq!(state.passes(), 1, "one streaming pass for the whole wall");
    assert_eq!(state.tiles_streamed(), 16, "each tile read once");
    let actual = state.export_statevector().expect("export");
    assert_states_match(&dense_reference(&circuit), &actual, "diagonal wall");
}

#[test]
fn tile_bit_controls_are_free() {
    let mut b = CircuitBuilder::new(6);
    b.h(5).cx(5, 0).cz(5, 1).rzz(0.4, 5, 0);
    let circuit = b.build();
    let mut state = tiled_run(&circuit, 2);
    assert_eq!(state.passes(), 2, "one pair pass for H, one window pass");
    let actual = state.export_statevector().expect("export");
    assert_states_match(&dense_reference(&circuit), &actual, "tile-bit controls");
}

#[test]
fn measurement_reset_and_conditional_match_the_serial_distributed_backend() {
    let n = 5;
    let mut b = CircuitBuilder::new_with_classical(n, 2);
    b.h(0)
        .cx(0, 4)
        .rx(0.3, 2)
        .measure(4, 0)
        .conditional(ClassicalCondition::BitIsOne(0), Gate::X, &[1])
        .reset(0)
        .ry(0.8, 3)
        .measure(1, 1);
    let circuit = b.build();
    let (expected_state, expected_bits) = serial_distributed_reference(&circuit);
    for w in 1..=n {
        let mut state = tiled_run(&circuit, w);
        assert_eq!(
            state.classical_results(),
            &expected_bits[..],
            "window {w}: classical bits"
        );
        let actual = state.export_statevector().expect("export");
        assert_states_match(&expected_state, &actual, &format!("window {w}"));
    }
}

#[test]
fn tile_thrashing_schedule_is_rejected_before_any_streaming() {
    let mut b = CircuitBuilder::new(5);
    b.h(3).h(4).h(3).h(4);
    let circuit = b.build();
    let mut state = TiledStatevector::new(5, 0, SEED, TileConfig::new(2)).expect("state");
    let err = state.run(&circuit.instructions).unwrap_err();
    match err {
        PrismError::BackendUnsupported { operation, .. } => {
            assert!(
                operation.contains("pass rate"),
                "unexpected message: {operation}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(state.tiles_streamed(), 0, "rejection precedes storage I/O");

    let mut config = TileConfig::new(2);
    config.max_pass_rate = 1.0;
    let mut state = TiledStatevector::new(5, 0, SEED, config).expect("state");
    state.run(&circuit.instructions).expect("override run");
    let actual = state.export_statevector().expect("export");
    assert_states_match(&dense_reference(&circuit), &actual, "override");
}

#[test]
fn short_schedules_run_under_the_default_threshold() {
    let mut b = CircuitBuilder::new(4);
    b.h(3);
    let mut state = TiledStatevector::new(4, 0, SEED, TileConfig::new(2)).expect("state");
    state.run(&b.build().instructions).expect("run");
}

#[test]
fn invalid_window_geometry_is_rejected() {
    assert!(TiledStatevector::new(4, 0, SEED, TileConfig::new(0)).is_err());
    assert!(TiledStatevector::new(4, 0, SEED, TileConfig::new(5)).is_err());
    assert!(TiledStatevector::new(0, 0, SEED, TileConfig::new(1)).is_err());
}

#[test]
fn fused_gate_spanning_a_tile_bit_is_rejected() {
    let z = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let identity = [
        [one, z, z, z],
        [z, one, z, z],
        [z, z, one, z],
        [z, z, z, one],
    ];
    let spanning = Instruction::Gate {
        gate: Gate::Fused2q(Box::new(identity)),
        targets: smallvec![0, 3],
    };
    let mut state = TiledStatevector::new(4, 0, SEED, TileConfig::new(2)).expect("state");
    let err = state.run(std::slice::from_ref(&spanning)).unwrap_err();
    assert!(matches!(err, PrismError::BackendUnsupported { .. }));

    let local = Instruction::Gate {
        gate: Gate::Fused2q(Box::new(identity)),
        targets: smallvec![0, 1],
    };
    state
        .run(std::slice::from_ref(&local))
        .expect("window-local fused gate");
}

#[test]
fn storage_is_removed_on_drop() {
    let mut b = CircuitBuilder::new(4);
    b.h(0);
    let circuit = b.build();
    let mut state = TiledStatevector::new(4, 0, SEED, TileConfig::new(2)).expect("state");
    state.run(&circuit.instructions).expect("run");
    let dir = state.storage_dir().to_path_buf();
    assert!(dir.exists());
    drop(state);
    assert!(!dir.exists());
}

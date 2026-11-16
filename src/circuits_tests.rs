use super::*;

const SEED: u64 = 42;

#[test]
#[should_panic(expected = "measure_split_circuit needs at least 8 qubits")]
fn measure_split_circuit_rejects_fewer_than_eight_qubits() {
    measure_split_circuit(7, 1, SEED);
}

#[test]
#[should_panic(expected = "ghz_circuit needs at least 1 qubit")]
fn ghz_circuit_rejects_zero_qubits() {
    ghz_circuit(0);
}

#[test]
#[should_panic(expected = "assertion failed: n >= 2")]
fn controlled_rotation_circuit_rejects_one_qubit() {
    controlled_rotation_circuit(1, 1, SEED);
}

#[test]
#[should_panic(expected = "needs at least two blocks of at least two qubits")]
fn disjoint_block_layers_circuit_rejects_single_qubit_blocks() {
    disjoint_block_layers_circuit(4, 1, 1, SEED);
}

#[test]
#[should_panic(expected = "needs at least two blocks of at least two qubits")]
fn disjoint_block_layers_circuit_rejects_a_block_size_that_does_not_divide() {
    disjoint_block_layers_circuit(7, 2, 1, SEED);
}

#[test]
#[should_panic(expected = "needs at least two blocks of at least two qubits")]
fn disjoint_block_layers_circuit_rejects_a_single_block() {
    disjoint_block_layers_circuit(4, 4, 1, SEED);
}

#[test]
#[should_panic(expected = "diagonal_mixed_circuit needs at least 2 qubits")]
fn diagonal_mixed_circuit_rejects_one_qubit() {
    diagonal_mixed_circuit(1, 1, SEED);
}

#[test]
#[should_panic(expected = "diagonal_mixed_circuit needs at least 2 qubits")]
fn diagonal_mixed_circuit_rejects_one_qubit_without_layers() {
    diagonal_mixed_circuit(1, 0, SEED);
}

#[test]
#[should_panic(expected = "needs at least 2 qubits per block")]
fn clifford_prefix_measured_blocks_rejects_one_qubit_blocks() {
    clifford_prefix_measured_blocks(2, 1, 1, SEED);
}

#[test]
#[should_panic(expected = "sparse_walk_circuit needs k <= n and n >= 2")]
fn sparse_walk_circuit_rejects_more_walkers_than_qubits() {
    sparse_walk_circuit(4, 5, 1, SEED);
}

#[test]
#[should_panic(expected = "sparse_walk_circuit needs k <= n and n >= 2")]
fn sparse_walk_circuit_rejects_one_qubit() {
    sparse_walk_circuit(1, 1, 1, SEED);
}

#[test]
#[should_panic(expected = "phase_estimation_circuit needs at least 2 qubits")]
fn phase_estimation_circuit_rejects_zero_qubits() {
    phase_estimation_circuit(0);
}

#[test]
#[should_panic(expected = "phase_estimation_circuit needs at least 2 qubits")]
fn phase_estimation_circuit_rejects_one_qubit() {
    phase_estimation_circuit(1);
}

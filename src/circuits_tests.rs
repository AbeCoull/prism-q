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

#[test]
#[should_panic(expected = "qft_circuit needs at least 1 qubit")]
fn qft_circuit_rejects_zero_qubits() {
    qft_circuit(0);
}

#[test]
#[should_panic(expected = "QFT block size must fit in u8")]
fn qft_circuit_rejects_more_than_255_qubits() {
    qft_circuit(256);
}

#[test]
#[should_panic(expected = "w_state_circuit needs at least 1 qubit")]
fn w_state_circuit_rejects_zero_qubits() {
    w_state_circuit(0);
}

#[test]
fn phase_estimation_circuit_65_qubits_succeeds_without_shift_overflow() {
    let c = phase_estimation_circuit(65);
    assert_eq!(c.num_qubits, 65);
    for inst in &c.instructions {
        if let Instruction::Gate {
            gate: Gate::Fused2q(mat),
            ..
        } = inst
        {
            assert!(
                mat.iter()
                    .flatten()
                    .all(|z| z.re.is_finite() && z.im.is_finite())
            );
        }
    }
}

#[test]
fn jordan_wigner_hamiltonian_handles_zero_one_and_zero_terms() {
    assert!(jordan_wigner_hamiltonian(0, 10, SEED).is_empty());
    assert!(jordan_wigner_hamiltonian(2, 0, SEED).is_empty());
    let one_mode = jordan_wigner_hamiltonian(1, 10, SEED);
    assert!(!one_mode.is_empty());
}

#[test]
fn zero_qubit_builders_do_not_underflow() {
    assert_eq!(random_circuit(0, 2, SEED).num_qubits, 0);
    assert_eq!(hardware_efficient_ansatz(0, 2, SEED).num_qubits, 0);
    assert_eq!(clifford_heavy_circuit(0, 2, SEED).num_qubits, 0);
    assert_eq!(qaoa_circuit(0, 2, SEED).num_qubits, 0);
}

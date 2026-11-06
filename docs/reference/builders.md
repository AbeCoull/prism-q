# Circuit Builders

Prebuilt circuits for benchmarks and tests, in `prism_q::circuits` (`src/circuits.rs`).
Every function in the table returns a `Circuit` ready for `simulate(&circuit)`.

| Function | Description |
|----------|-------------|
| `qft_circuit(n)` | Quantum Fourier Transform |
| `random_circuit(n, depth, seed)` | Random gates at given depth |
| `hardware_efficient_ansatz(n, layers, seed)` | HEA with Ry/Rz + CX |
| `clifford_heavy_circuit(n, depth, seed)` | Random Clifford (adjacent CX) |
| `clifford_random_pairs(n, depth, seed)` | Random Clifford (random pair CX) |
| `ghz_circuit(n)` | GHZ state (H + CX chain) |
| `qaoa_circuit(n, layers, seed)` | QAOA MaxCut |
| `single_qubit_rotation_circuit(n, depth, seed)` | Single-qubit rotations only |
| `clifford_t_circuit(n, depth, t_fraction, seed)` | Clifford+T with tunable T ratio |
| `w_state_circuit(n)` | W state preparation |
| `quantum_volume_circuit(n, depth, seed)` | Quantum volume (random SU(4)) |
| `cz_chain_circuit(n, depth, seed)` | CZ chains |
| `phase_estimation_circuit(n)` | Quantum phase estimation |
| `independent_bell_pairs(n_pairs)` | Independent Bell pairs |
| `independent_random_blocks(blocks, size, depth, seed)` | Independent random blocks |
| `partially_independent_circuit(n, depth, seed)` | Mostly independent halves with a few crossing gates |
| `measure_split_circuit(n, layers, seed)` | A chain that mid-circuit measurements and resets halve, then rotation layers on the survivors; `n >= 8` |
| `controlled_rotation_circuit(n, layers, seed)` | Local rotations alternating with controlled Rx and Ry on offset ring pairings |
| `disjoint_block_layers_circuit(n, k, layers, seed)` | Layers of disjoint `k`-qubit blocks |
| `diagonal_mixed_circuit(n, layers, seed)` | Diagonal families mixed with single-qubit rotations |
| `local_clifford_blocks(num_blocks, block_size, depth, seed)` | Clifford blocks with no coupling between them |
| `clifford_prefix_measured_blocks(num_blocks, block_size, depth, seed)` | Clifford blocks, then per block rotations, a mid-circuit measurement with feed-forward, and a final measurement |
| `brickwork_circuit(n, depth, seed)` | Alternating even and odd two-qubit layers |
| `matched_brickwork_circuit(n, depth, seed)` | Brickwork whose layers pair up for cancellation |
| `sparse_walk_circuit(n, k, depth, seed)` | Walk that keeps roughly `k` amplitudes live |

`jordan_wigner_hamiltonian(n, max_terms, seed)` returns a Hamiltonian instead, a
`Vec<(f64, Vec<PauliTerm>)>` in the form `expectation_gradient` takes: the Jordan-Wigner
transform of a seeded random fermionic operator on `n` modes, cut to the `max_terms`
largest coefficients.

## Example

```rust
use prism_q::circuits::qft_circuit;
use prism_q::simulate;

let circuit = qft_circuit(10);
let result = simulate(&circuit).seed(42).run().unwrap();
```

Hand-built circuits go through [`CircuitBuilder`](../getting-started/first-circuit.md).
Signatures are on [docs.rs](https://docs.rs/prism-q/latest/prism_q/circuits/).

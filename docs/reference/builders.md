# Circuit Builders

Pre-built circuits for benchmarking and testing, in `src/circuits.rs`. Each returns a
`Circuit` you can pass straight to `simulate(&circuit)` or any backend.

| Function | Description |
|----------|-------------|
| `qft_circuit(n)` | Quantum Fourier Transform |
| `random_circuit(n, depth, seed)` | Random gates at given depth |
| `hardware_efficient_ansatz(n, layers, seed)` | HEA with Ry/Rz + CX |
| `clifford_heavy_circuit(n, depth, seed)` | Random Clifford (adjacent CX) |
| `clifford_random_pairs(n, depth, seed)` | Random Clifford (random pair CX) |
| `ghz_circuit(n)` | GHZ state (H + CX chain) |
| `qaoa_circuit(n, layers, seed)` | QAOA MaxCut |
| `single_qubit_rotation_circuit(n, depth, seed)` | 1q rotations only |
| `clifford_t_circuit(n, depth, t_fraction, seed)` | Clifford+T with tunable T ratio |
| `w_state_circuit(n)` | W state preparation |
| `quantum_volume_circuit(n, depth, seed)` | Quantum volume (random SU(4)) |
| `cz_chain_circuit(n, depth, seed)` | CZ chains |
| `phase_estimation_circuit(n)` | Quantum phase estimation |
| `independent_bell_pairs(n_pairs)` | Independent Bell pairs |
| `independent_random_blocks(blocks, size, depth, seed)` | Independent random blocks |

## Example

```rust
use prism_q::circuits::qft_circuit;
use prism_q::simulate;

let circuit = qft_circuit(10);
let result = simulate(&circuit).seed(42).run().unwrap();
```

For hand-built circuits, use the [`CircuitBuilder`](../getting-started/first-circuit.md)
fluent API instead.

The complete generated API documentation lives on [docs.rs](https://docs.rs/prism-q).

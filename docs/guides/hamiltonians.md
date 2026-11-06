# Hamiltonian Evolution

A Hamiltonian is a `PauliObservable`, `H = sum c_j P_j`, the same type the expectation
terminals read. Evolving under it appends Pauli rotations to a circuit, so the result runs
on any backend and fuses like any other gate.

## Single rotations

`Circuit::add_pauli_rotation(theta, factors)` and `CircuitBuilder::pauli_rotation` append
`exp(-i theta P / 2)` for one Pauli string `P`. A weight-1 string lands as `rx`, `ry` or
`rz`, `ZZ` as `rzz`, and any other string as the native `PauliRot`, which the statevector
applies in one pass and fusion batches at 16 qubits and above.

## Trotter product formulas

`Circuit::add_trotter` and `CircuitBuilder::trotter` append a product-formula
approximation of `exp(-i H t)`: `steps` steps of length `dt = t / steps`, each term as
the rotation `exp(-i c_j P_j dt)`. `TrotterOrder::First` is the Lie-Trotter formula,
`Second` the symmetric one, and `Suzuki(k)` the recursive formula of order `2k` at
`5^(k - 1)` second-order substeps per step.

Identity terms are dropped, since they contribute only a global phase, and so are zero
coefficients. Consecutive rotations of one string merge, so a second-order step over `m`
terms emits `2m - 1` rotations and adjacent steps share their boundary rotation.

```rust
# use prism_q::{CircuitBuilder, PauliObservable, PauliTerm, TrotterOrder, simulate};
let ising = PauliObservable::from_terms([
    (1.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
    (0.5, vec![PauliTerm::x(0)]),
    (0.5, vec![PauliTerm::x(1)]),
])?;
let evolved = CircuitBuilder::new(2)
    .h(0)
    .h(1)
    .trotter(&ising, 1.0, 8, TrotterOrder::Second)
    .build();
let energy = simulate(&evolved).seed(42).observable_expectation(&ising)?.mean;
assert!((energy - 1.0).abs() < 1e-2); // <H> is conserved up to the Trotter error
# Ok::<(), prism_q::PrismError>(())
```

Wider Pauli strings export to OpenQASM as the `r` spelling PRISM-Q reads back (see
[Supported gates](./openqasm.md#supported-gates)); run `circuit::expand_pauli_rotations`
first for output another toolchain reads.

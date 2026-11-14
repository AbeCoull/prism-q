# Large and Structured Circuits

Goal: run three circuits too wide for a dense statevector, each on the engine its
structure allows, and read from the result metadata whether the answer is exact.

A dense statevector holds `2**n` complex amplitudes at 16 bytes each, so 30 qubits take
16 GiB. Past that, circuit structure decides what can run.

## Clifford circuits: thousands of qubits

A circuit of only Clifford gates (H, S, CX, CZ and the Paulis) runs on a stabilizer
tableau in polynomial time, and automatic dispatch picks it.

```python
from prism_q import CircuitBuilder, simulate

n = 1000
builder = CircuitBuilder(n, n).h(0)
for q in range(n - 1):
    builder.cx(q, q + 1)
ghz = builder.measure_all().build()

counts = simulate(ghz).seed(42).sample_counts(1000)
print(counts.metadata.backend)          # CompiledStabilizer
print(len(counts.counts()))             # 2
```

The two outcomes are all zeros and all ones. `CompiledStabilizer` is the sampler that
compiles the measurement record once and draws every shot from it.

## Clifford plus a few T gates

One T gate takes a circuit out of the stabilizer formalism, but a few can still be handled
exactly. This chain applies `h t h` to every sixth qubit, ten T gates in all, then
entangles the register with a CX ladder:

```python
def kicked_chain(n, measure):
    builder = CircuitBuilder(n, n if measure else 0)
    for q in range(0, n, 6):
        builder.h(q).t(q).h(q)
    for q in range(n - 1):
        builder.cx(q, q + 1)
    if measure:
        builder.measure_all()
    return builder.build()


chain = kicked_chain(60, measure=False)
print(chain.t_count(), chain.is_clifford_only())   # 10 False
```

Each `h t h` leaves its qubit with `<Z> = cos(pi/4)`, and the ladder makes `Z` on the last
qubit the parity of all of them, so `<Z59> = cos(pi/4)**10 = 1/32`. Deterministic Pauli
propagation computes that exactly by pushing the observable backwards through the
circuit:

```python
from prism_q import BackendKind

values = (
    simulate(chain)
    .backend(BackendKind.deterministic_pauli())
    .seed(42)
    .expectation([[(59, "Z")]])
)
print(round(values.values[0], 6), values.metadata.is_exact)   # 0.03125 True
```

Shots on the measured version go to the stabilizer-rank engine unprompted. It tracks a
sum of stabilizer states that grows with the T count, not the width:

```python
shots = simulate(kicked_chain(60, measure=True)).seed(42).shots(2000)
print(shots.metadata.backend, shots.metadata.is_exact)   # StabilizerRank True
print(round(shots.shots[:, 59].mean(), 4))              # 0.4855, expected 31/64 = 0.4844
```

The [Clifford+T guide](../guides/clifford-t.md) covers the four Clifford+T engines and
their limits.

## Low entanglement: matrix product states

A two-layer hardware-efficient ansatz on 60 qubits is neither Clifford nor small, but its
entanglement stays low. Asked for `Z` on every qubit, automatic dispatch sends it to a
matrix product state with a bond cap of 256. Asked for one local observable, it contracts
that observable's light cone on the tensor network instead, which is exact and cheaper.

```python
from prism_q import circuits

wide = circuits.hardware_efficient_ansatz(60, 2)
result = simulate(wide).seed(42).expectation([[(q, "Z")] for q in range(60)])
meta = result.metadata
print(meta.backend, meta.is_exact, meta.fidelity_lower_bound)   # Mps False 1.0
print(meta.bond.peak, meta.bond.cap, meta.bond.saturated)       # 4 256 False
```

`is_exact` is `False` because the MPS engine can truncate. The bond report says whether
it did: the widest bond needed was 4, under the cap of 256, so nothing was cut and
`fidelity_lower_bound` is 1.0. With the cap forced down to 2, the bond saturates:

```python
tight = simulate(wide).backend(BackendKind.mps(2)).seed(42).expectation(
    [[(0, "Z")]]
)
print(round(result.values[0], 4), round(tight.values[0], 4))   # 0.3301 0.4464
print(tight.metadata.bond.saturated)                           # True
```

A saturated bond means truncation may have moved the answer. Here it moved it by 0.12,
and the fidelity bound drops to 0.0, the worst case it can certify.

## Refusing approximate answers

`require_exact()` turns a route that could truncate into an error naming it:

```python
from prism_q import PrismError

try:
    simulate(wide).seed(42).require_exact().expectation([[(q, "Z")] for q in range(60)])
except PrismError as exc:
    print(exc.kind)                 # incompatible_backend
```

The check uses the circuit and the requested backend, before anything runs. The
1000-qubit Clifford sample passes under automatic dispatch. The Clifford+T shots pass
only with `.backend(BackendKind.stabilizer_rank())`, because automatic dispatch cannot
promise the exact route in advance. Deterministic Pauli propagation is always rejected,
since it truncates once the term count passes its budget.

## In Rust

```rust
use prism_q::{BackendKind, CircuitBuilder, Exactness, simulate};

let n = 1000;
let mut builder = CircuitBuilder::new_with_classical(n, n);
builder.h(0);
for q in 0..n - 1 {
    builder.cx(q, q + 1);
}
let ghz = builder.measure_all().build();
let counts = simulate(&ghz).seed(42).require_exact().sample_counts(1000)?;
assert_eq!(counts.counts.len(), 2);
assert_eq!(counts.metadata.exactness, Exactness::Exact);

let wide = prism_q::circuits::hardware_efficient_ansatz(60, 2, 42);
let observables = vec![vec![prism_q::PauliTerm::z(0)]];
let result = simulate(&wide)
    .backend(BackendKind::Mps { max_bond_dim: 256 })
    .seed(42)
    .expectation_values_reported(&observables)?;
let bond = result.metadata.bond.expect("an MPS run reports its bond");
assert!(!bond.saturated());
# Ok::<(), prism_q::PrismError>(())
```

Next: draw any of these circuits with the [Drawing Circuits](../guides/drawing.md) guide,
or read how dispatch decides in
[Simulation Engine and Dispatch](../architecture/engine.md).

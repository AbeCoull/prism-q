# Clifford+T Simulation

Circuits that mix Clifford gates with a modest number of `T` gates sit between the
polynomial stabilizer regime and the exponential statevector regime. Three strategies
cover them, chosen by T-count, qubit count, and whether Monte Carlo error is acceptable.

```admonish tip title="Which strategy?"
- **Few T gates, exact result needed**: stabilizer rank (`run_stabilizer_rank`).
- **Many T gates, marginals only**: stochastic Pauli propagation (`run_spp`).
- **Moderate T-count, exact or bounded-error expectation values**: deterministic sparse
  Pauli dynamics (`run_spd`).
```

When the T-count permits, these run before the standard
[dispatch tree](../architecture/engine.md).

## Accepted gate forms

All three strategies take Clifford gates, `T`, `Tdg`, `Rz`, `P` and `Rzz` as they are,
and lower `Rx`, `Ry` and `PauliRot` to Clifford conjugation around one `Rz`. A `Fused`
matrix (the parser's `u`, `u3`, `u2`, `r`, `gpi` and `gpi2`) lowers to the named gate or
`Rz` it equals up to a global phase, or else to its Rz-Ry-Rz Euler triple. A `cu` lowers
when its target is diagonal or a Pauli up to phase (`cy`, `cp`, `crz`, `cs`) and is
rejected otherwise, as are `ccx` and the other multi-controlled forms. The Pauli engines
take any rotation angle. Stabilizer rank needs every lowered Z rotation on the pi/4 grid;
`Rz(pi/4)`, `P(pi/4)` and their odd multiples count as one T each, so
`is_clifford_plus_t` and the automatic route treat them like `T`.

## Stabilizer rank (`src/sim/stabilizer_rank.rs`)

Exact probability output is capped because it returns a dense vector of 2^n entries.
Shot sampling has no hard qubit cap and takes one of two forms.

For terminal measurements of a unitary circuit, the state is a Clifford tableau `C`
applied to `|0>` on most qubits and a dense register `φ` on the rest. A T gate on qubit
`q` rotates about `P = C† Z_q C`. When `P` flips a qubit still in `|0>`, CX gates
controlled on that qubit clear `P` from the other `|0>` qubits, fold into `C`, and move
the qubit into the register, so the register grows by at most one qubit per T gate and
holds at most `2^t` amplitudes. Sampling reduces each measured `C† Z_q C` once: an outcome
that flips a `|0>` qubit is a fair coin, and the rest are read from one table of register
marginals. Random Clifford depth 10 with 10 T gates takes about 0.5 ms for 1024 shots from
16 to 32 qubits and 77 ms at 1000 qubits. The register is capped at 20 qubits.

Mid-circuit measurements, or a register past that cap, carry coherent weighted MPS
branches instead. Clifford gates mutate each branch, `T` and `Tdg` split branches, and a
measurement computes outcome probabilities from the weighted ensemble before projecting
every branch onto the sampled outcome. Branch count, MPS bond growth and measurement count
set how far this form scales.

The dense probability path keeps a weighted sum of stabilizer states. Each T gate doubles
the term count through `T = alpha*I + beta*Z`; Clifford gates cost O(n²) per term, and
weighted amplitudes accumulate into exact probabilities.

| Function | Use |
|----------|-----|
| `run_stabilizer_rank` | Exact probabilities (t ≤ 20, n ≤ 25) |
| `run_stabilizer_rank_approx` | Approximate under a term budget (higher t counts) |
| `run_stabilizer_rank_shots` | Shot-based sampling with no fixed qubit cap |
| `stabilizer_overlap_sq` | Inner product between stabilizer states |

`run_stabilizer_rank_approx` truncates deterministically: after each `T` it drops the
smallest-magnitude terms beyond the budget, renormalizes the probabilities it returns, and
reports the summed 1-norm of what it dropped as `discarded_weight`. Writing that as `d`,
the returned distribution differs from the exact one by at most `4 * d`, and
`StabRankResult::fidelity_bound` converts `d` to a fidelity floor. The bound assumes
worst-case interference between the non-orthogonal branch states, so an aggressive budget
drives it to zero. `Auto` never selects this function, nor the exact expansion for
probabilities: at every width the expansion can hold, a statevector also fits and is
faster, so the automatic probability route goes to the dispatch tree. Only a direct call
reaches the pruned expansion.

## Stochastic Pauli Propagation (`src/sim/unified_pauli.rs`)

Backward-propagates measurement observables as Pauli strings. Clifford gates conjugate in
O(1), and T gates branch stochastically into two weighted Pauli paths. A path costs
O(d×n/64) whatever the T-count. Marginal probabilities come from Monte Carlo estimation.

```rust
# use prism_q::{circuits::clifford_t_circuit, run_spp};
# let circuit = clifford_t_circuit(8, 10, 0.1, 42);
# let (num_samples, seed) = (1000, 42);
let result = run_spp(&circuit, num_samples, seed)?; // SppResult
# Ok::<(), prism_q::PrismError>(())
```

## Deterministic Sparse Pauli Dynamics (`src/sim/unified_pauli.rs`)

Backward-propagates a weighted sum of Pauli strings stored in a HashMap. T gates branch
X and Y terms deterministically, and identical strings merge. The result is exact for
small T-counts and approximate with bounded error for larger ones.

```rust
# use prism_q::{SpdTruncation, circuits::clifford_t_circuit, run_spd, run_spd_with};
# let circuit = clifford_t_circuit(8, 10, 0.1, 42);
# let (epsilon, max_terms) = (1e-8, 1 << 16);
# let truncation = SpdTruncation::Budget { max_terms };
let result = run_spd(&circuit, epsilon, max_terms)?; // SpdResult
let result = run_spd_with(&circuit, &truncation)?;   // SpdResult, any policy
# Ok::<(), prism_q::PrismError>(())
```

Both truncation policies are available from the builder as well as the free functions:

```rust
# use prism_q::{BackendKind, PauliTerm, SpdTruncation, circuits::clifford_t_circuit, simulate};
# let circuit = clifford_t_circuit(8, 10, 0.1, 42);
# let observables = vec![vec![PauliTerm::z(0)], vec![PauliTerm::z(0), PauliTerm::z(1)]];
let values = simulate(&circuit)
    .seed(42)
    .backend(BackendKind::DeterministicPauli {
        truncation: SpdTruncation::Budget { max_terms: 1 << 16 },
    })
    .expectation_values(&observables)?;
# Ok::<(), prism_q::PrismError>(())
```

`SpdTruncation::Threshold { epsilon, max_terms }` drops terms below `epsilon` once the sum
passes `max_terms`, and `max_terms = 0` disables pruning for an exact run. With an
`epsilon` too small to hold the growth, the sum climbs to the engine's internal ceiling
and the run errors there rather than returning an over-truncated value.

`SpdTruncation::Budget { max_terms }` instead drops exactly enough of the
smallest-magnitude terms to return to the budget. There is no threshold to guess, and
growth the budget caps cannot reach the ceiling, so it suits runs where the term count is
the binding constraint. A budget of zero is rejected.

Both report the discarded coefficient magnitude on the engine result, and it bounds the
error: `|error| <= sum |discarded|`. It is not a fidelity bound.

`observable_expectation` propagates a weighted `PauliObservable` as one sum, with each
term's weight on its coefficient. `max_terms` caps that whole sum rather than each term,
`epsilon` compares weighted coefficients (so a term weighted below `epsilon` can drop
whole), and the bound applies to the weighted value. The same holds on `PauliPath`. A
merged sum past the internal ceiling falls back to one propagation per term.

## Pauli path propagation under noise (`src/sim/unified_pauli.rs`)

The same weighted Pauli sum, carried through a noise model. Select
`BackendKind::PauliPath { epsilon, max_terms }` and attach a noise model; the engine
answers `expectation_values` and `observable_expectation` and nothing else.

```rust
# use prism_q::{BackendKind, NoiseModel, PauliTerm, circuits::clifford_t_circuit, simulate};
# let circuit = clifford_t_circuit(8, 10, 0.1, 42);
# let observables = vec![vec![PauliTerm::z(0)], vec![PauliTerm::z(0), PauliTerm::z(1)]];
let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
let values = simulate(&circuit)
    .seed(42)
    .backend(BackendKind::PauliPath { epsilon: 1e-8, max_terms: 1 << 16 })
    .noise(&noise)
    .expectation_values(&observables)?;
# Ok::<(), prism_q::PrismError>(())
```

The error has two independent parts, and only the truncation is approximate.

The channels are exact. Each is applied as its adjoint on the Pauli basis, not as a twirl,
so depolarizing, dephasing, thermal relaxation and amplitude damping reproduce the density
matrix to machine precision at `max_terms = 0`. Amplitude damping is not unital, and the
identity term its adjoint produces from `Z` is carried. A channel with no Pauli-basis form
(custom Kraus, two-qubit Kraus, readout error) is rejected rather than approximated.

The truncation reports its own bound. With `max_terms = 0` nothing is dropped and the
value is exact. With a budget set, terms whose magnitude falls below `epsilon` are dropped
once the sum exceeds the budget, and the total dropped magnitude bounds the error in the
returned value. The bound is a worst case, not an estimate: it holds because every
remaining operation is a contraction in the Pauli 1-norm.

The term count, not the qubit count, decides whether the engine is usable. Every
non-Clifford rotation in the observable's backward light cone can double the sum, and
every channel shrinks it. Where noise wins, a circuit stays cheap at widths no dense
representation reaches. Where it does not, the run hits the budget and reports a large
discarded mass: use the density matrix or trajectory averaging instead.

The observable's weight moves that count more than width or depth, since the backward
light cone opens from every letter it starts with. On a two-layer hardware-efficient
ansatz under 1% depolarizing, `Z` on one qubit stays at 11 terms at 20 qubits or 100; `Z`
on two adjacent qubits fills a 16384-term budget by 30 qubits; a `Z` on every qubit fills
it at 20 and returns a discarded mass larger than the observable's own norm, which flags
the answer as unusable instead of returning a wrong one quietly. Check the reported
discarded mass against the precision needed before trusting a truncated run.

[`clifford_t_circuit`](../reference/builders.md) builds Clifford+T test circuits directly.

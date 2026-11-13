# Clifford+T Simulation

Circuits that mix Clifford gates with a modest number of `T` gates sit between the
polynomial stabilizer regime and the exponential statevector regime. Four engines cover
them.

```admonish tip title="Which engine?"
- An exact result with T gates that reach at most 20 qubits: stabilizer rank
  (`run_stabilizer_rank`).
- Many T gates, marginals only: stochastic Pauli propagation, SPP (`run_spp`).
- Moderate T count, exact or bounded-error expectation values: deterministic Pauli
  propagation, SPD (`run_spd`).
- Expectation values under a noise model: Pauli path (`BackendKind::PauliPath`).
```

`Auto` reaches three of them ahead of the
[dispatch tree](../architecture/engine.md#auto-dispatch-decision-tree):

- Terminal shots go to stabilizer rank on a circuit of 16 qubits or more when the
  Clifford register (the form below) fits 20 qubits and costs less than the dense run.
  The T count sets no limit of its own: 24 T gates on a 14-qubit register at 26 qubits
  take about 1 ms for 1024 shots against 1.5 s on the statevector. Past the statevector
  cap, a lowered one included, a register that fits takes the shots ahead of the MPS; a
  circuit that would grow it past 20 goes to the dispatch tree. Shots with a mid-circuit
  measurement, a classical guard or a reset move their measurements to the end when they
  can and take the register under the same tests, the dense side priced as one run per
  outcome history the shots can reach. Past the statevector cap, a circuit whose
  measurements stay in place takes the weighted branches below up to 6 T gates; the rest
  go to the dispatch tree.
- Marginals on a unitary Clifford+T circuit of 12 qubits or more go to SPD.
- Expectation values on a Clifford circuit go to SPD.

SPP and Pauli path run only when selected.
[Routes the tree does not show](../architecture/engine.md#routes-the-tree-does-not-show)
has the exact conditions.

## Accepted gate forms

All four engines take Clifford gates, `T`, `Tdg`, `Rz`, `P` and `Rzz` as they are. SPD
and Pauli path branch on `Rx` and `Ry` too; SPP and stabilizer rank lower them, and every
engine lowers `PauliRot`, to Clifford conjugation around one `Rz`. A `Fused`
matrix (the parser's `u`, `u3`, `u2`, `r`, `gpi` and `gpi2`) lowers to the named gate or
`Rz` it equals up to a global phase, or else to its Rz-Ry-Rz Euler triple. A `cu` lowers
when its target is diagonal or a Pauli up to phase (`cy`, `cp`, `crz`, `cs`) and is
rejected otherwise, as are `ccx` and the other multi-controlled forms. The Pauli engines
take any rotation angle. Stabilizer rank needs every lowered Z rotation on the pi/4 grid;
`Rz(pi/4)`, `P(pi/4)` and their odd multiples count as one T each, so
`is_clifford_plus_t` and the automatic route treat them like `T`.

## Stabilizer rank (`src/sim/stabilizer_rank.rs`)

Exact probability output is capped because it returns a dense vector of 2^n entries.
Shot sampling is not bound by the statevector cap and takes one of two forms.

For terminal measurements of a unitary circuit, the state is a Clifford tableau `C`
applied to `|0>` on most qubits and a dense register `φ` on the rest. A T gate on qubit
`q` rotates about `P = C† Z_q C`. When `P` flips a qubit still in `|0>`, CX gates
controlled on that qubit clear `P` from the other `|0>` qubits, fold into `C`, and move
the qubit into the register, so the register grows by at most one qubit per T gate and
holds at most `2^t` amplitudes. Sampling reduces each measured `C† Z_q C` once: an outcome
that flips a `|0>` qubit is a fair coin, and the rest are read from one table of register
marginals. Random Clifford depth 10 with 10 T gates takes about 0.5 ms for 1024 shots from
16 to 32 qubits and 77 ms at 1000 qubits. The register is capped at 20 qubits: a circuit
that would grow it past 20 returns `BackendUnsupported`, and `Auto` checks the width
from the tableau alone and sends such a circuit to the dispatch tree instead.

A circuit with mid-circuit measurements, guards or resets first moves its measurements to
the end: a reset hands the qubit's later gates to a fresh qubit in `|0>`, and a guarded
Pauli becomes the same Pauli controlled by the measured qubit. The rewritten circuit is
unitary with terminal measurements and the same T count, so the register above samples
it. A guard on a gate other than a Pauli, a gate on a measured qubit with no reset
between, a guard comparing two measured bits, or a guarded body holding a measurement
keeps the measurements in place.

Those circuits carry coherent weighted MPS branches instead. Clifford gates mutate each
branch, `T` and `Tdg` split branches, and a measurement computes outcome probabilities
from the weighted ensemble before projecting every branch onto the sampled outcome.
Branch count, MPS bond growth and measurement count set how far this form scales; past 20
T gates, guarded ones included, the branch count would pass its cap and the run returns
`BackendUnsupported` before the first shot.

The dense probability path reads its probabilities off the same Clifford register the
terminal shots use. Branches of `T = alpha*I + beta*Z` that land on one basis state of the
register merge into one coefficient, so a circuit holds `2^k` terms, where `k` is the
number of qubits the T gates reach and `k <= min(n, t)`. The T count sets no limit. A
register wider than 20 qubits falls back to one branch per T choice, capped at `2^20`.

| Function | Use |
|----------|-----|
| `run_stabilizer_rank` | Exact probabilities (register width ≤ 20, n ≤ 25) |
| `run_stabilizer_rank_approx` | Approximate under a coefficient budget (register width ≤ 20, n ≤ 25) |
| `run_stabilizer_rank_shots` | Shot sampling past the statevector cap (register of at most 20 qubits) |
| `stabilizer_overlap_sq` | Inner product between stabilizer states |

`run_stabilizer_rank_approx` runs the same register and, after each `T`, zeroes all but
the budget's largest coefficients, ties kept in index order. The coefficients are
orthonormal, so each cut drops a known norm `d_j`. The result reports:

- `discarded_weight`: `D = sum d_j`. The exact state lies within `D` of the kept one.
- `kept_norm_sqr`: `N^2 = 1 - sum d_j^2`, the kept norm before the probabilities are
  renormalized.
- `fidelity_bound()`: `F >= ((1 + N^2 - D^2) / (2N))^2`, never below `1 - D^2`. The
  returned distribution differs from the exact one by at most `2 sqrt(1 - F)` in summed
  absolute difference. A single cut gives `F = N^2` exactly.

On the 10-qubit benchmark circuits at budget 128 the floor is 0.880 at 16 T and 0.744 at
20 T. A register wider than 20 qubits is refused. `Auto` never selects this function, nor
the exact expansion for probabilities: while the register fits, the exact route answers
without error, and past it a statevector also fits.

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

The builder takes both truncation policies too:

```rust
# use prism_q::{BackendKind, PauliTerm, SpdTruncation, simulate};
# use prism_q::circuits::clifford_t_circuit;
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

`SpdTruncation::Budget { max_terms }` instead drops exactly enough of the smallest terms
to return to the budget. It has no threshold to guess and never reaches the ceiling, so
it suits runs where the term count is the binding constraint. A budget of zero is
rejected.

Both report the discarded coefficient magnitude on the engine result, which bounds the
error as `|error| <= sum |discarded|`. It is not a fidelity bound.

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
# use prism_q::{BackendKind, NoiseModel, PauliTerm, simulate};
# use prism_q::circuits::clifford_t_circuit;
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

Only the truncation is approximate. Each channel applies as its exact adjoint on the
Pauli basis, not as a twirl, so depolarizing, dephasing, thermal relaxation and amplitude
damping reproduce the density matrix to machine precision at `max_terms = 0`. Amplitude
damping is not unital, and the identity term its adjoint produces from `Z` is carried. A
channel with no Pauli-basis form (custom Kraus, two-qubit Kraus, readout error) is
rejected rather than approximated.

With a budget set, terms whose magnitude falls below `epsilon` are dropped once the sum
exceeds the budget, and the total dropped magnitude bounds the error in the returned
value. The bound is a worst case, and it holds because every remaining operation is a
contraction in the Pauli 1-norm.

The term count, not the qubit count, decides whether the engine is usable. Every
non-Clifford rotation in the observable's backward light cone can double the sum, and
every channel shrinks it. Where noise wins, a circuit stays cheap at widths no dense
representation reaches. Where it does not, the run hits the budget and reports a large
discarded mass: use the density matrix or trajectory averaging instead.

The observable's weight moves that count more than width or depth, since the light cone
opens from every letter it starts with. On a two-layer hardware-efficient
ansatz under 1% depolarizing, `Z` on one qubit stays at 11 terms at 20 qubits or 100; `Z`
on two adjacent qubits fills a 16384-term budget by 30 qubits; a `Z` on every qubit fills
it at 20 and returns a discarded mass larger than the observable's own norm, which flags
the answer as unusable instead of returning a wrong one quietly. Check the reported
discarded mass against the precision needed before trusting a truncated run.

[`clifford_t_circuit`](../reference/builders.md) builds random Clifford+T test circuits.

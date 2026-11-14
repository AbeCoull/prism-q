# Backends

PRISM-Q ships nine CPU backends: statevector, stabilizer, factored stabilizer, sparse,
MPS, product state, tensor network, factored and density matrix. The statevector,
stabilizer and density matrix also run on CUDA, and the `distributed` feature adds a
statevector sharded across MPI ranks. Four Clifford+T engines (stabilizer rank, SPP, SPD
and the Pauli path) hold no state and sit beside the backends. The
[simulation engine](./engine.md) picks a backend unless the caller names one, and never
picks the density matrix or the distributed backend.
[Choosing a Backend](../getting-started/choosing-a-backend.md) covers the same ground by
task.

Scroll wide circuit figures horizontally, or focus them and use the arrow keys.
Open the full SVG for gate tooltips; rotation angles are in radians.

<figure class="circuit-figure">
<div class="circuit-scroll" role="region" aria-label="GHZ circuit" tabindex="0">
<img src="../diagrams/ghz_5.svg" alt="A Hadamard followed by a chain of four controlled X gates across five qubits.">
</div>
<figcaption>GHZ preparation uses Clifford gates and routes to a stabilizer backend. <a href="../diagrams/ghz_5.svg">Open full SVG</a>.</figcaption>
</figure>

## Reset semantics

`reset` is the channel `rho -> |0⟩⟨0| ⊗ tr_q rho` on every backend: the qubit is traced
out and replaced by `|0⟩`, leaving the rest of the register in the mixture the trace
produces. Projecting onto `|0⟩` and renormalizing agrees with it only when the qubit is
unentangled; otherwise projection also collapses its partners into the branch correlated
with `|0⟩`. Resetting qubit 1 of a Bell pair leaves `⟨Z0⟩ = 0` under the channel and
`⟨Z0⟩ = 1` under projection.

A pure-state backend runs one trajectory instead: sample the outcome, collapse onto it,
and apply X when it is 1, consuming one draw from its RNG stream. Averaged over shots that
is the channel. The density matrix applies the channel directly, with no draw.
`tests/integration/reset_channel.rs` pins the contract across backends.

## Memory budget

A backend whose state would not fit returns `PrismError::ResourceLimit` instead of
handing work to another. The error names the backend, the requirement, the cap, the layer
that set the cap, and the `Limits` field and environment variable that override it.
`BackendKind::Auto` chooses from circuit structure before any backend is constructed. The
check lives in `Backend::init`, so `run_on` gets the same guard as `simulate`.

The statevector cap comes from detected physical memory: half of it is the budget, and
the cap is the widest register whose `2^n` `Complex64` amplitudes fit, so 29 qubits on a
16 GiB host and 30 on 32 GiB. `PRISM_MAX_SV_QUBITS` overrides it with a qubit count, and
an unparseable value warns and leaves detection in charge rather than disabling the cap.
When detection itself fails every cap is disabled, with a warning on stderr, since a
guessed budget is worse than an unknown one: a register too wide for the machine then
aborts on allocation rather than routing to another backend. `src/backend/memory.rs`
holds the derivation and the other dense caps that share it.

| Cap | `Limits` field and variable | Default |
|-----|----------|---------|
| Statevector state | `max_statevector_qubits`, `PRISM_MAX_SV_QUBITS` | Largest `2^n` `Complex64` state fitting half of detected physical memory |
| Density-matrix state | `max_density_matrix_qubits`, `PRISM_MAX_DM_QUBITS`, bounded by `PRISM_MAX_SV_QUBITS` | `floor(cap_sv / 2)`, since a density matrix of `n` qubits is a `2n`-qubit statevector |
| Dense probability output | `max_probability_qubits`, `PRISM_MAX_PROB_QUBITS` | Same budget over `f64` |
| Dense statevector export | `max_export_qubits`, `PRISM_MAX_EXPORT_QUBITS` | Same budget over `Complex64` |
| Dense outcome sampling | environment only, `PRISM_MAX_DENSE_OUTCOME_BITS` | Same budget over two `f64` per outcome |
| Sparse entry count | `max_sparse_qubits`, `PRISM_MAX_SPARSE_QUBITS` (the map holds at most `2^q` entries) | Same budget at 64 bytes per entry across the double-buffered maps |
| Factored merged-block width | `max_factored_merge_qubits`, `PRISM_MAX_FACTORED_MERGE_QUBITS` | Same budget over `Complex64` |
| MPS gate workspace | `max_mps_workspace_qubits`, `PRISM_MAX_MPS_WORKSPACE_QUBITS` (at most `2^q` amplitudes of live contraction buffers) | Same budget over `Complex64` |
| Tensor-network peak intermediate | `max_tn_peak_qubits`, `PRISM_MAX_TN_PEAK_QUBITS` (at most `2^q` elements in the largest planned intermediate, summed over the slices running at once) | Same budget over `Complex64` |
| Factored stabilizer merged-cluster width | `max_stabilizer_cluster_qubits`, `PRISM_MAX_STABILIZER_CLUSTER_QUBITS` | Widest joint tableau fitting the same budget, counted as `2n + 1` rows of `2 * ceil(n / 64)` words and halved to cover the peak while both source tableaux are still live |

Each cap resolves through four layers, highest first: the request's `Limits`
(`Simulate::limits`), the process default (`set_default_limits`), the environment, and
detection. A layer that sets `memory_budget_bytes` re-derives every cap from that budget
and discards the per-cap values below it, so a wide `PRISM_MAX_SV_QUBITS` cannot lift a
cap past a tenant budget. Caps resolve once per request and travel with the backend. No
gate kernel reads them.

The five growth caps (sparse, factored, MPS, tensor network, factored stabilizer) are
independent of `PRISM_MAX_SV_QUBITS`. Those backends exist to run above the statevector
cap, so lowering it to steer routing must not shrink what they hold. Their defaults come
from the same detected budget. The factored stabilizer cap is the only one that is not a
`2^n` amplitude count: a cluster costs `O(n^2 / 64)` words, and a dense cap would hold it
to a qubit ceiling Clifford circuits pass routinely.

The density-matrix cap is the tighter of its own override and half the statevector cap,
computed in one place so dispatch validation and the `init` guard agree. Raising
`PRISM_MAX_DM_QUBITS` past that bound needs `PRISM_MAX_SV_QUBITS` raised too, and the
rejection says so, naming the density matrix rather than the statevector it allocates
internally.

Parallel noisy trajectories are the one path holding several states: each Rayon thread
runs its own backend, so peak memory is `threads * state(n)`. The path is limited to
circuits below 14 qubits; above that, trajectories run serially under the ordinary cap.
Pauli-noise runs grouped by error pattern follow the same rule and add a pattern table
plus 16 bytes a shot, with no `2^n` outcome table. Damping runs sharing a no-jump prefix
hold a walked state beside each shot state, so above the split width they run only where
two states fit. The distributed statevector runs trajectories serially at every width,
one slice per rank re-initialized per shot, and groups no error patterns.

Three backends grow after `init`, and each checks before allocating:

- a factored merge checks the merged block width against the merge cap;
- the sparse map checks a branching gate's worst fan-out against the entry cap, so a
  rejection can fire one gate early on a state that would have deduplicated below it;
- MPS gate application checks its live contraction buffers against the workspace cap.
  That bounds the workspace, not `max_bond_dim`: a large bond cap is fine while bonds
  stay small.

## Statevector

Full-state simulation in a flat `Vec<Complex64>` of 2^n amplitudes. The primary backend
up to the memory cap, 29 qubits on a 16 GiB host.

Gate kernels use enum dispatch with specialized routines for CX, CZ, SWAP, Cu, MCU, Rzz,
BatchRzz, BatchPhase, DiagonalBatch, MultiFused and PauliRot. PauliRot is one pass over
`(j, j ^ xmask)` pairs for `exp(-i θ P / 2)`, or a parity-phase sweep for a Z-only
string; backends without the kernel receive the CNOT-ladder lowering from
`expand_pauli_rotations`. Single-qubit gates go through `PreparedGate1q` with FMA SIMD.

MultiFused runs a tiled kernel whose tile sizes come from the cache topology the OS
reports (see the [performance guide](../guides/performance.md)), with a cheaper path for
an all-diagonal batch.

Rayon parallelism starts at 15 qubits with `par_chunks_mut` and `MIN_PAR_ELEMS = 4096`
per task. BMI2 `_pext_u64` speeds BatchPhase, BatchRzz and DiagonalBatch LUT indexing.

Normalization is deferred: `pending_norm` accumulates the factors a measurement collapse
or a no-jump damping branch would rescale by, instead of sweeping the state each time.

## Stabilizer

Aaronson-Gottesman bit-packed tableau for Clifford circuits, O(n²) time and space,
scaling to thousands of qubits. Kernels use wordwise bit operations and `popcount` for
phases. Supports H, S, Sdg, SX, SXdg, X, Y, Z, Id, CX, CZ, SWAP, measurement, reset and
classical conditionals. Destabilizer rows materialize lazily, when probabilities are
requested.

On 256 qubits and up, a run of Clifford gates with at least one cross-word two-qubit
gate per tableau word moves to a column-major copy of the gate rows: one 64x64 bit-block
transpose each way, and each gate updates its qubits' columns, O(n/64) words, instead of
a pass over every row. The run ends at the first measurement, reset, region or
non-Clifford instruction.

Probability extraction is coset enumeration with GF(2) Gaussian elimination, O(2^k) in
the number k of non-diagonal generators rather than O(2^n).

`export_tableau` returns the rows as `(words, phases)` and `init_from_tableau` accepts the
same pair, for a warm start or a checkpoint. Packing: `2n + 1` rows of `2 * ceil(n / 64)`
`u64` words, destabilizer rows `0..n`, stabilizer rows `n..2n`, then a scratch row the
importer ignores. Each row is its X words then its Z words, with qubit `q` at bit `q % 64`
of word `q / 64`, and `phases[r]` is true when row `r` carries sign -1. The import checks
lengths and that destabilizer `i` anticommutes with stabilizer `i`, nothing more: rows
that break the rest of the commutation structure give wrong outcomes silently. The random
stream restarts from the importing backend's seed. The factored stabilizer imports the
pair as one cluster, held to the merged-cluster cap, and exports every cluster into the
joint layout.

## Factored Stabilizer

`FactoredStabilizerBackend` keeps one tableau per cluster, starting at one qubit per
cluster. A 2q gate across clusters merges their tableaux; measurement and reset can split
independent tableaux apart again, so product structure never pays for the joint tableau.
`Auto` takes it for Clifford circuits of at least 16 qubits that split into blocks.

## Sparse

`FxHashMap<usize, Complex64>` (the crate's multiply-xor hasher) for states with few
non-zero amplitudes, O(k) memory. After gates that can shrink or cancel amplitudes,
entries with |a|² at or below a pruning threshold (1e-28, roundoff level) are removed and
the rest rescaled to keep the norm. A run with the threshold raised above the default
reports `Approximate` with a fidelity bound from the dropped weight.

An entry costs about 16 times a dense amplitude, so a state holding more than roughly a
sixteenth of its basis runs slower than a dense vector of the same width. There is no
mid-run handoff to the statevector: an explicitly selected sparse run on a densifying
circuit degrades in place. Automatic dispatch picks this backend only above the
statevector cap, and a run that branches past the entry cap rejects the gate. Keys are
`usize` basis indices, so a circuit wider than `usize::BITS` qubits is rejected at `init`
and automatic dispatch sends it to MPS.

## MPS (Matrix Product State)

Chain of rank-3 tensors with adaptive bond dimension (default max 256), O(n·χ²) memory.
Single-qubit gates absorb with FMA SIMD over bond-dimension slices. Two-qubit gates
contract adjacent sites, apply, and SVD-truncate back; non-adjacent gates route through
SWAP chains. The SVD is faer's for `m×n ≥ 256` and a Jacobi sweep below.

Once any bond reaches 128, a truncating cut first moves the orthogonality center to the
cut, so the discarded singular weight is the state's 2-norm error. Below that, cuts run
ungauged and treat drops under four machine epsilons of the largest value as rounding.

## Product State

Per-qubit `[Complex64; 2]` storage, O(n) memory, O(1) per single-qubit gate. Rejects
entangling gates and is selected automatically for circuits with no 2q gates.

Shots and Pauli expectations read the per-qubit states instead of a `2^n` probability
vector, so both stay O(n) at widths no dense route reaches
([Sampling Architecture](./samplers.md)).

## Tensor Network

Deferred contraction planned on metadata: a greedy min-size pass, with seeded restarts
when its peak intermediate grows large, picks the pair order. Gates append tensors, and
contraction happens at probability extraction, where `PRISM_MAX_PROB_QUBITS` guards the
dense readout and an explicit run past it errors naming the cap rather than reporting
`probabilities: None`.

Every contraction checks its planned peak intermediate against `PRISM_MAX_TN_PEAK_QUBITS`
before allocating. A plan over the cap is sliced and stays exact, holding no more memory
than an unsliced run. Only a plan still over the cap at `PRISM_MAX_TN_SLICES`
assignments is rejected, naming the cap. `BackendKind::TensorNetworkBounded { tolerance }`
truncates before it slices, with the same first-order bound as the MPS; the
[Capability and Support Matrix](../guides/capabilities.md#what-kind-of-answer-a-result-is)
gives both kinds' contract.

Measurement and reset draw from the single-qubit reduced density matrix and absorb the
renormalizing projector into the tensor holding the measured qubit's output leg, so
mid-circuit measurement has no width ceiling and the tensor count does not grow across
measurements.

A one-qubit reduced density matrix and a Pauli expectation contract the network against
its conjugate, so they cost what the doubled network costs. An identity factor is a
closed leg: a weight-`k` observable adds `k` tensors, not `n`.

The trajectory engine needs only that one-qubit reduction for general noise, so
amplitude damping, phase damping, thermal relaxation and one-qubit custom Kraus
channels run here under explicit dispatch. Two-qubit Kraus channels do not, because the
backend declines the two-qubit reduction.

## Factored

Starts with n independent 1-qubit states and merges by tensor product only when a 2q gate
bridges groups. Sub-states of 15 qubits or more use the statevector's parallel kernels.
Selected when subsystem decomposition detects partial independence.

## Density Matrix

Exact mixed-state evolution. The density operator `rho` for `n` qubits is a `4^n`
`Complex64` buffer, row-major: index `(r << n) | c` holds `⟨r|rho|c⟩`. That is a
`2n`-qubit statevector whose high `n` qubits index the ket (row) and low `n` the bra
(column), so `U rho U^dagger` is two statevector passes: `U` on the ket register, and the
gate's conjugate form on the bra register, or a conjugation of the buffer around the
pass for a variant with no conjugate form.

`Rzz` and the batched diagonal families (`BatchPhase`, `BatchRzz`, `DiagonalBatch`) take
one pass per sandwich over a combined table, and a `MultiFused` or `Multi2q` block runs
its ket constituents and their bra conjugates through the statevector tiled kernels
together.

Memory is `16 * 4^n` bytes: about 14 qubits on a 16 GiB host and 15 on 32 GiB
(`PRISM_MAX_DM_QUBITS` moves it within the statevector budget). With a device attached
(`BackendKind::DensityMatrixGpu`, or `with_gpu` on the backend) the buffer lives in VRAM,
budgeted against free memory at `init`. An 11 GiB card holds 14 qubits, so the device is
for throughput, not width. `Auto` and `AutoGpu` never select either kind, and the device
kind has no host fallback.

With a noise model attached it is the exact route for every `Simulate` terminal except
the adjoint gradient, which needs a pure state to walk backward. `run` and `marginals`
reject a model carrying readout error and point at `sample_counts`.
[Noise across the terminals](./engine.md#noise-across-the-terminals) has the full rules.

## Stabilizer Rank

`src/sim/stabilizer_rank.rs` expands each T gate as `T = αI + βZ`, so a circuit with `t`
T gates is a weighted sum of up to `2^t` stabilizer branches. One shared
`StabilizerBackend` runs every Clifford once. Each branch is a weight and a signed Pauli
string, the Zs earlier T positions chose, conjugated through the Cliffords that followed.
Clifford gates cost O(n²) for all branches together and each T gate doubles the count.

- Exact probabilities (`run_stabilizer_rank`) come from the Clifford register below, which
  sums branches that differ by a stabilizer of the base state, so they stop at 25 qubits
  and a register of 20 qubits at any T count. A wider register falls back to the branch
  sum, capped at `2^20` terms.
- `run_stabilizer_rank_approx` cuts the same register back to its budget after each T
  gate. The coefficients are orthonormal, so each cut drops an exact norm `d_j`; the
  exact state lies within `sum d_j` of the kept one, and `fidelity_bound()` turns that and
  the kept norm into a fidelity floor. A register past 20 qubits is refused.
- Terminal shots run on a Clifford tableau over a dense register holding only the qubits
  the T rotations reach (`src/sim/clifford_register.rs`, at most 20 such qubits). Shots
  with mid-circuit measurement keep coherent branches as MPS, project each outcome, and
  contract pairwise branch overlaps with no dense state.
- Expectation values and marginals read the same tableau in the product gauge
  (`src/sim/clifford_parts.rs`): a T gate that reaches a qubit in `|0>` adds it as a
  one-qubit state, every other T gate is an excess rotation, and qubits an excess rotation
  links form a part. The state is a product over parts, so a part past the width cap
  answers through its excess rotations and does not block the others.

`BackendKind::StabilizerRank` rejects a circuit with no T gate, pointing at the
stabilizer. `Auto` takes it for shots, expectation values and marginals; the thresholds
are in
[routes the tree does not show](./engine.md#routes-the-tree-does-not-show).

## SPP and SPD

Both propagate an observable backward through a unitary circuit of Clifford gates and
Pauli rotations (`src/sim/unified_pauli.rs`). Neither holds a state or is a `Backend`, and
the dispatcher reaches them directly. A rotation at a non-Clifford angle splits each
anticommuting string into two.

SPP (`StochasticPauli { num_samples }`, `run_spp`) samples one branch per rotation and
averages `num_samples` backward walks, so its cost follows samples and gate count, not
`2^n`. The result carries a standard error.

SPD (`DeterministicPauli { truncation }`, `run_spd`) carries the whole weighted sum.
`SpdTruncation::Threshold { epsilon, max_terms }` drops terms below `epsilon` once the sum
passes `max_terms`, and `Budget { max_terms }` drops the smallest surplus to hold the sum
at the budget. With nothing to drop the run is exact and errors at a fixed term ceiling
rather than truncating. Either way the discarded coefficient mass bounds the error,
`|error| <= sum |discarded|`. `Auto` sends marginals on Clifford+T circuits from 12 qubits
here when the register parts decline, and expectation values on Clifford circuits.

## Pauli Path

Heisenberg propagation of an observable through a noisy circuit as a weighted sum of
Pauli strings, reached directly like SPP and SPD. It serves `expectation_values` and
`observable_expectation` (and the parameter-shift gradient built on them); `run`,
`shots`, `marginals` and `probabilities` are rejected naming those two terminals.

`Rz` and `Rzz` split each anticommuting term into a `cos` and a `-i sin` branch, which
grows the sum, and a noise channel scales each term by its action on the letters it
touches, which shrinks it. The engine is polynomial where shrinking wins: a
circuit whose noise rate outpaces its density of non-Clifford rotations. Observable
weight drives the term count more than width or depth: under light depolarizing noise a
weight-one `Z` holds the sum flat as the register widens, while a full-width `Z` chain
fills any budget.

Channels enter through their adjoint on the Pauli basis, not a twirl, so nothing is
approximated at the channel. `Pauli`, `Depolarizing`, `PhaseDamping` and
`TwoQubitDepolarizing` scale each letter by one eigenvalue. `AmplitudeDamping` and
`ThermalRelaxation` are not unital: their adjoint sends `Z` to `(1 - gamma) Z + gamma I`,
and a thermal channel with nonzero `excited_population` scales the identity branch by the
steady-state polarization `1 - 2 p`. `Custom` Kraus, `Kraus2q` and readout error have no
Pauli-basis form and are rejected naming the density matrix.

With `max_terms = 0` the run is exact and errors at the shared term ceiling instead of
truncating. With a budget set, terms below `epsilon` are dropped once the sum exceeds it.
Every channel and Clifford conjugation is a contraction in the Pauli 1-norm, so the
discarded coefficient mass bounds the error. A run that truncated nothing reports itself
exact whatever its budget.

## State diagnostics

`Simulate::reduced_density_matrix`, `Simulate::entanglement_entropy` and
`Simulate::overlap` read the output state, so all three require a unitary circuit: a
measurement, reset or conditional leaves one seeded branch. Each asks a single backend
for the answer in its own representation. An explicitly selected
backend with no kernel reports `BackendUnsupported` naming itself and the diagnostic,
with no dense export fallback. Under `BackendKind::Auto` a backend that cannot answer is
replaced by the statevector while the circuit fits its cap; a partially independent
circuit keeps the factored backend and a Clifford circuit its tableau.

| Backend | Reduced density matrix | Entanglement entropy | State overlap |
| --- | --- | --- | --- |
| Statevector (host or device) | Partial trace over the complement | One thin SVD of the reshaped amplitudes, with the Schmidt spectrum | Dense dot product |
| Sparse | Grouped over the traced index | Declines | Lookup join over the nonzeros against another sparse map, at any width |
| Factored | Kronecker of the per-block traces | One thin SVD per block the cut splits, summed; the Schmidt spectrum is their sorted Kronecker product | Dense export |
| Product state | Kronecker of the per-qubit factors | `0`, with the single Schmidt value `1` | Product of the per-qubit inner products against another product state, at any width |
| Density matrix | Partial trace of the mixture | Declines: a mixture has no Schmidt decomposition | Declines: the fidelity of two mixtures is not an inner product |
| MPS | One environment sweep over the span of the named sites, tracing the rest | One SVD at the cut, or the eigenvalues of the reduced density matrix when the subsystem is not contiguous in chain order | Chain contraction against another chain in the same site order, at any width |
| Tensor network | Declines | Declines | Dense export |
| Stabilizer, factored-stabilizer | Projector onto the generators supported inside the subsystem | Rank of the generators restricted to the cut, less the subsystem size, in units of `ln 2`, with a flat spectrum | Rank of the two tableaux merged, at any width, on the host stabilizer; a device tableau and the factored form take the dense export |
| Distributed statevector | Rank-local partial trace after relabeling the kept qubits local, summed across ranks; a subsystem wider than the slice reads the gathered state | Declines | Dense export |

Entropy is von Neumann in nats (a Bell pair reads `ln 2`). Schmidt values come back
descending, squares summing to one. A stabilizer cut of rank `r` has `2^r` equal weights.
Past the dense export cap that list no longer fits, so the entropy comes back alone with
`EntropyResult::schmidt_values` at `None`, as it does on the factored backend when the
product of its blocks' spectra no longer fits. The reduced density matrix is row major
with side `2^k` and trace one, priced as a `2k`-qubit statevector against the export cap.
A noise model sends the marginal and the entropy to the density matrix, which answers the
marginal of the exact mixture and declines the entropy.

`Simulate::overlap` takes a second seeded builder, so each side has its own backend, seed
and start state; both circuits must declare the same width. The result is the squared
modulus of the inner product of the two normalized states, not the amplitude, since a
tableau keeps no global phase and every MPS truncation moves one. Every route divides by
both norms, so an unnormalized chain answers like a unit-norm state. Two states in the
same representation take the native route at any width; any other pair uses a dense
export of both, up to the export cap. A noise model on either side is rejected.

## What a backend reports about its own result

Four `Backend` methods fill the result's
[metadata](../guides/capabilities.md#what-kind-of-answer-a-result-is): `resolved`,
`exactness`, `placement` and `bond_report`. All four have defaults, and a backend keeping
the `resolved` default is named by `Backend::name`. They are reports, not predictions:
`exactness` is read after the run, from the singular values actually discarded, and
`placement` after any device fallback.

The MPS sums discarded weight per SVD and returns `1 - total` as the fidelity lower bound.
That is a first-order estimate, not a certificate: errors compound across SVDs, and the
strict infidelity bound is the square of the summed square roots, which matches the sum
only when one SVD truncates.

The decomposed route runs one backend per independent block and merges: exactness is the
weakest of the parts, the fidelity bound is the product, the bond report is the peak over
the MPS blocks against their shared cap, and placement is `Device` only when every block
was. Per-shot routes evolve one state per shot and keep the weakest claim, with the bound
a minimum rather than a product.

The [GPU backend](../guides/gpu.md) has its own guide, and the distributed statevector is
covered in the [Capability and Support Matrix](../guides/capabilities.md).

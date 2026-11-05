# Backends

PRISM-Q ships nine CPU backends, an optional CUDA path on the statevector, stabilizer and
density-matrix backends, and a feature-gated distributed statevector that shards the
dense state across MPI ranks. The [simulation engine](./engine.md) picks a backend unless
the caller names one. The density matrix and the distributed backend are explicit only.
The tensor network is explicit except on an expectation or marginals terminal, where
`Auto` reaches it for a wide shallow unitary circuit. The task-oriented version of this
material is the [Backends Deep Dive guide](../guides/backends.md).

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

A pure-state backend cannot hold the mixture, so it runs one trajectory: sample the
outcome, collapse onto it, and apply X when it is 1. Averaged over shots that is the
channel, and each reset consumes one draw from the backend's RNG stream. The density
matrix applies the channel directly, with no draw. `tests/integration/reset_channel.rs`
pins the contract across backends against the density-matrix oracle.

## Memory budget

A circuit that does not fit in memory is an error. No backend hands work to another when
its state would not fit: it returns `PrismError::ResourceLimit` naming itself, the
requirement, the cap, the layer that set the cap, and the `Limits` field and environment
variable that override it. Choosing another backend is the caller's decision, and
`BackendKind::Auto` makes it from circuit structure before any backend is constructed.

The check lives in `Backend::init`, which every execution path passes before reserving
state, so `run_on` gets the same guard as `simulate`.

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
cap past a tenant budget. Caps resolve once per request and travel with the backend; no
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
internally. When physical memory cannot be detected the caps are disabled with a warning,
since a guessed budget is worse than an unknown one.

Parallel noisy trajectories are the one path holding several states: each Rayon thread
runs its own backend, so peak memory is `threads * state(n)`. The path is limited to
circuits below 14 qubits, where a statevector replica is 256 KiB and a full pool stays in
the tens of megabytes; above that, trajectories run serially under the ordinary cap.
Pauli-noise runs grouped by error pattern follow the same rule and add a pattern table
plus 16 bytes a shot; the outcome draw walks the amplitudes once and builds no `2^n`
table. Damping runs sharing a no-jump prefix hold a walked state beside each shot state,
so above the split width they run only where two states fit. The distributed statevector
runs trajectories serially at every width, one slice per rank re-initialized per shot,
and groups no error patterns.

Three backends grow after `init`, and each checks at the growth event before allocating:

- a factored merge checks the merged block width against the merge cap;
- the sparse map checks a branching gate's worst-case fan-out against the entry cap, so a
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

MultiFused uses a three-tier tiled kernel: an L2 tile sized by the cache-topology budget
(2^14 amplitudes at 256 KB), 2^17 L3 tiles, and shared passes of up to three targets. The
budget is read once per process from the OS; see the
[performance guide](../guides/performance.md). The L3 tier runs only when every worker
gets its own tile and the concurrent tiles fit half the reported last-level cache;
otherwise its targets join the shared passes. An all-diagonal MultiFused batch takes a
dedicated path, one complex multiply per element against 4+2 for a full 2×2.

Rayon parallelism starts at 15 qubits with `par_chunks_mut` and `MIN_PAR_ELEMS = 4096`
per task. BMI2 `_pext_u64` speeds BatchPhase, BatchRzz and DiagonalBatch LUT indexing.

Normalization is deferred: `pending_norm` accumulates the factors instead of rescaling
the state after each measurement collapse or no-jump damping branch, which then costs one
pass over the `|1>` half. A circuit with neither never touches it.

The QFT is a representative statevector workload, dense with controlled-phase gates that
the fusion pipeline batches:

<figure class="circuit-figure">
<div class="circuit-scroll" role="region" aria-label="Quantum Fourier transform circuit" tabindex="0">
<img src="../diagrams/qft_4.svg" alt="Four-qubit quantum Fourier transform expanded into Hadamards, controlled phase rotations, and swaps.">
</div>
<figcaption>Controlled phases span the register; the final swaps reverse qubit order. <a href="../diagrams/qft_4.svg">Open full SVG</a>.</figcaption>
</figure>

## Stabilizer

Aaronson-Gottesman bit-packed tableau for Clifford circuits, O(n²) time and space,
scaling to thousands of qubits. Kernels use wordwise bit operations and `popcount` for
phases. Supports H, S, Sdg, SX, SXdg, X, Y, Z, Id, CX, CZ, SWAP, measurement, reset and
classical conditionals.

- Word-group batching fuses several 1q gate flushes into one tableau pass.
- Type-grouped masks apply all gates of one Pauli type with a single wordwise op.
- Sparse Generator Indexing (SGI) keeps per-qubit active generator lists, so row
  operations target those rows instead of scanning the tableau.
- Destabilizer rows materialize lazily, when probabilities are requested.

Probability extraction is coset enumeration with GF(2) Gaussian elimination, O(2^k) in
the number k of non-diagonal generators rather than O(2^n).

`export_tableau` returns the rows as `(words, phases)` and `init_from_tableau` accepts the
same pair, so one run's Clifford state can start another (a warm start or a checkpoint).
Packing: `2n + 1` rows of `2 * ceil(n / 64)` `u64` words, destabilizer rows `0..n`,
stabilizer rows `n..2n`, then a scratch row the importer ignores. Each row is its X words
then its Z words, with qubit `q` at bit `q % 64` of word `q / 64` in each half, and
`phases[r]` is true when row `r` carries sign -1. The import checks lengths and that
destabilizer `i` anticommutes with stabilizer `i`, nothing more: rows that break the rest
of the commutation structure are accepted and give wrong outcomes silently. The random
stream restarts from the importing backend's seed. The factored backend imports the pair
as one cluster over every qubit, held to the merged-cluster cap, and its `export_tableau`
scatters every cluster into the joint layout.

**Factored Stabilizer** (`FactoredStabilizerBackend`): per-cluster tableaux, starting at
one qubit per cluster. A 2q gate across clusters merges their tableaux; measurement and
reset can split independent sub-tableaux again, so product structure avoids full-tableau
work.

## Sparse

`FxHashMap<usize, Complex64>` (the crate's multiply-xor hasher) for states with few
non-zero amplitudes, O(k) memory. After gates that can shrink or cancel amplitudes,
entries with |a|² at or below a pruning threshold (1e-28, roundoff level) are removed and
the rest rescaled to keep the norm. A run with the threshold raised above the default
reports `Approximate` with a fidelity bound from the dropped weight. Suited to large
registers whose support stays concentrated on few basis states.

The per-entry gate cost is about 16x the statevector's per-amplitude cost on a mixed
diagonal and permutation workload (the `sparse/densify` bench rows), so a state past
roughly 1/16 load factor runs slower than a dense vector of the same width. There is no
mid-run handoff to the statevector. Automatic dispatch picks this backend only above the
statevector cap, and a run that branches past the entry cap rejects the gate. Keys are
`usize` basis indices, so a circuit wider than `usize::BITS` qubits is rejected at `init`
and automatic dispatch sends it to MPS. An explicitly selected sparse run on a densifying
circuit degrades in place, measured at up to 24x the dense cost when fully dense at 20
qubits.

## MPS (Matrix Product State)

Chain of rank-3 tensors with adaptive bond dimension (default max 256), O(n·χ²) memory.
Single-qubit gates absorb with FMA SIMD over bond-dimension slices. Two-qubit gates
contract adjacent sites, apply, and SVD-truncate back; non-adjacent gates route through
SWAP chains.

SVD dispatch is hybrid: faer (bidiagonalization plus divide and conquer) for m×n ≥ 256,
hand-rolled Jacobi below.

Once any bond reaches 128, a truncating cut first moves the orthogonality center to the
cut, so the discarded singular weight is the state's 2-norm error rather than a value the
ungauged environment amplifies. Below that, cuts run ungauged and treat drops under four
machine epsilons of the largest value as rounding.

## Product State

Per-qubit `[Complex64; 2]` storage, O(n) memory, O(1) per single-qubit gate. Rejects
entangling gates; selected automatically for circuits with no 2q gates.

Shots and Pauli expectations read the per-qubit states, not a `2^n` probability vector,
so both stay O(n) at widths no dense route reaches. See
[Sampling Architecture](./samplers.md).

## Tensor Network

Deferred contraction planned on metadata. A greedy min-size pass picks the pair order,
seeded noisy restarts rerun it when the greedy tree's peak intermediate grows large, and
the kernel replays the winner. Gates append tensors; contraction happens at probability
extraction, where `PRISM_MAX_PROB_QUBITS` guards the dense readout and an explicit run
past it errors naming the cap rather than reporting `probabilities: None`.

Every contraction, dense or doubled, checks its planned peak intermediate against
`PRISM_MAX_TN_PEAK_QUBITS` before allocating. A plan over the cap is sliced: legs are
fixed one at a time, greedily by the peak each buys, and the network is contracted once
per assignment and summed, trading a time factor for a divided peak. Only a plan still
over the cap at `PRISM_MAX_TN_SLICES` assignments is rejected, naming the cap. Slices run
through Rayon, only as many at once as fit under the cap together, so a sliced run holds
no more than an unsliced one. The result is exact.

`BackendKind::TensorNetworkBounded { tolerance }` adds truncation;
`BackendKind::TensorNetwork` never truncates. Under the bounded kind a contraction over
the cap first factors an intermediate across the cut separating its partner-facing legs
from the rest, keeping the new bond only as far as discarding that fraction of the cut's
squared weight allows, and replans; slicing covers whatever is still over. The route
reports `Approximate` with 1 minus the summed per-cut discarded weights, the same
first-order estimate as the MPS, and `require_exact()` rejects it.

Measurement and reset do not contract to the dense state. The outcome draws from the
single-qubit reduced density matrix and the renormalizing projector is absorbed into the
tensor holding the measured qubit's output leg, so mid-circuit measurement has no width
ceiling and the tensor count does not grow across measurements.

Two more queries avoid the dense route by contracting the network against its conjugate.
The bra copy's legs are shifted clear of the ket index space, and each qubit's boundary
is either closed against its twin (a trace) or joined through an operator. A one-qubit
reduced density matrix leaves that qubit's ket and bra open and returns a `2x2`; a Pauli
expectation joins each non-identity factor through its operator and contracts to a
scalar. Both cost what the doubled network costs, not what the qubit count suggests. An
identity factor is a closed leg, so a weight-`k` observable adds `k` tensors, not `n`.
The planner needs nothing extra for open indices: an index held by exactly one tensor is
open, and the greedy ordering carries it to the result.

The one-qubit reduced density matrix is what the trajectory engine needs for general
noise, so amplitude damping, phase damping, thermal relaxation and one-qubit custom Kraus
channels run here under explicit dispatch. Two-qubit Kraus channels do not, because the
backend declines the two-qubit reduction.

## Factored

Dynamic split-state simulation. Starts with n independent 1-qubit states and merges by
tensor product only when a 2q gate bridges groups. Sub-states of 15 qubits or more use
the statevector's parallel kernels. Selected when subsystem decomposition detects partial
independence.

## Density Matrix

Exact mixed-state evolution. The density operator `rho` for `n` qubits is a `4^n`
`Complex64` buffer, row-major: index `(r << n) | c` holds `⟨r|rho|c⟩`. That is a
`2n`-qubit statevector whose high `n` qubits index the ket (row) and low `n` the bra
(column), so gates reuse the statevector kernels. `U` on the ket register gives `U rho`;
`rho U^dagger` takes the gate's conjugate form on the bra register where one exists and
otherwise conjugates the buffer around the pass. `U rho U^dagger` is therefore two
statevector passes, plus two conjugations for variants with no conjugate form.

Diagonal gates do better. `Rzz` and the batched families (`BatchPhase`, `BatchRzz`,
`DiagonalBatch`) are diagonal on both sides, so ket and bra phases cancel wherever the
registers agree on the targets' parity, and each sandwich is one pass over a combined
table. A `MultiFused` or `Multi2q` block runs its ket constituents and their bra
conjugates through the statevector tiled kernels together, with `Multi2q` cut into the
runs those kernels apply in list order. A block whose batch would take as many passes as
its constituents one at a time keeps the per-constituent route.

Memory is `16 * 4^n` bytes: about 14 qubits on a 16 GiB host and 15 on 32 GiB
(`PRISM_MAX_DM_QUBITS` moves it within the statevector budget). With a device attached
(`BackendKind::DensityMatrixGpu`, or `with_gpu` on the backend) the buffer lives in VRAM,
budgeted against free memory at `init`. An 11 GiB card holds 13 qubits, so the device
adds about one qubit and is for throughput, not width. On the device the unitary half
runs the dense statevector kernels over the embedded buffer; channels, projection, reset
and the diagonal and Pauli readouts have their own kernels. Both kinds are explicit only:
`Auto` and `AutoGpu` never select them, and the device kind has no host fallback.

With a noise model attached it is the exact route for every `Simulate` terminal except
the adjoint gradient. The mixture is evolved once and observables, marginals,
probabilities and shots read that evolution; the parameter-shift gradient evaluates it
once per shifted angle. No evolution holds readout error, so `run` and `marginals` reject
a model carrying it and point at `sample_counts`. The adjoint backpropagates against a
pure state, and a channel has no reverse evolution to walk. See
[Noise across the terminals](./engine.md#noise-across-the-terminals) for what the route
accepts and what stays on trajectory averaging.

## Pauli Path

Heisenberg propagation of an observable through a noisy circuit as a weighted sum of
Pauli strings. It holds no state, so it is not a `Backend`: like SPP and SPD the
dispatcher reaches it directly. It serves `expectation_values` and
`observable_expectation` (and the parameter-shift gradient built on them); `run`,
`shots`, `marginals` and `probabilities` are rejected naming those two terminals.

The observable starts as one term and propagates backward. Clifford gates conjugate it
term by term. `Rz` and `Rzz` split each anticommuting term into a `cos` and a `-i sin`
branch, which grows the sum. A noise channel scales each term by its action on the
letters it touches, which shrinks it. The engine is polynomial where shrinking wins: a
circuit whose noise rate outpaces its density of non-Clifford rotations. Watch the term
count at a given width, not the width.

Observable weight drives the term count more than width or depth. On a two-layer
hardware-efficient ansatz under 1% depolarizing, `Z` on one qubit holds the sum at 11
terms from 20 qubits to 100, `Z` on two adjacent qubits reaches a 16384-term budget by
30 qubits, and a full-width `Z` chain reaches it at every width.
`term_count_is_width_independent_at_unit_weight` pins both halves.

Channels enter through their adjoint on the Pauli basis, not a twirl, so nothing is
approximated at the channel. A unital Pauli channel (`Pauli`, `Depolarizing`,
`PhaseDamping`, `TwoQubitDepolarizing`) scales each letter by one eigenvalue.
`AmplitudeDamping` and `ThermalRelaxation` are not unital: their adjoint sends `Z` to
`(1 - gamma) Z + gamma I`, and the sum carries the identity branch as a second term. A
thermal channel with nonzero `excited_population` scales that term by the steady-state
polarization `1 - 2 p`, so it vanishes at a maximally mixed steady state. `Custom` Kraus,
`Kraus2q` and readout error have no Pauli-basis form and are rejected naming the density
matrix.

With `max_terms = 0` the run is exact and errors at the shared term ceiling instead of
truncating. With a budget set, terms below `epsilon` are dropped once the sum exceeds it,
and the discarded coefficient mass bounds the error: every channel and Clifford
conjugation is a contraction in the Pauli 1-norm, so a dropped term moves the result by
at most its own magnitude. A run that truncated nothing reports itself exact whatever its
budget.

## State diagnostics

`Simulate::reduced_density_matrix`, `Simulate::entanglement_entropy` and
`Simulate::overlap` read the output state, so all three require a unitary circuit: a
measurement, reset or conditional leaves one seeded branch, not the state the diagnostic
is defined on. Each resolves to a single backend, as the native expectation path does,
and asks it for the answer in its own representation. An explicitly selected backend
with no kernel reports `BackendUnsupported` naming itself and the diagnostic, with no
dense export fallback.

Under `BackendKind::Auto` a resolved backend that cannot answer is replaced by the
statevector while the circuit fits its cap. A partially independent circuit keeps the
factored backend (entropy and marginal block by block) and a Clifford circuit its
tableau (both without expansion).

| Backend | Reduced density matrix | Entanglement entropy | State overlap |
| --- | --- | --- | --- |
| Statevector (host or device) | Partial trace over the complement | One thin SVD of the reshaped amplitudes, with the Schmidt spectrum | Dense dot product |
| Sparse | Grouped over the traced index | Declines | Lookup join over the nonzeros against another sparse map, at any width |
| Factored | Kronecker of the per-block traces | One thin SVD per block the cut splits, summed; the Schmidt spectrum is their sorted Kronecker product | Dense export |
| Product state | Kronecker of the per-qubit factors | `0`, with the single Schmidt value `1` | Product of the per-qubit inner products against another product state, at any width |
| Density matrix | Partial trace of the mixture | Declines: a mixture has no Schmidt decomposition | Declines: the fidelity of two mixtures is not an inner product |
| MPS | One environment sweep over the span the named sites occupy, keeping those open and tracing the rest | One SVD at the cut, or the eigenvalues of the reduced density matrix when the subsystem is not contiguous in chain order | Chain contraction against another chain in the same site order, at any width |
| Tensor network | Declines | Declines | Dense export |
| Stabilizer, factored-stabilizer | Projector onto the generators supported inside the subsystem | Rank of the generators restricted to the cut, less the subsystem size, in units of `ln 2`, with the flat spectrum that rank stands for | Rank of the two tableaux merged, at any width while both hold their rows on the host, on the stabilizer; a device-resident tableau and the factored form take the dense export |
| Distributed statevector | Rank-local partial trace once the kept qubits are relabeled into local positions, summed across ranks; a subsystem wider than the local slice reads the gathered state | Declines | Dense export |

Entropy is von Neumann in nats (a Bell pair reads `ln 2`). Schmidt values come back
descending, squares summing to one whatever norm the representation carried. A
stabilizer cut of rank `r` has `2^r` equal weights, read off one elimination. Past the
dense export cap that list no longer fits, so the entropy comes back alone with
`EntropyResult::schmidt_values` at `None`; the factored backend does the same when the
product of its blocks' spectra no longer fits. The reduced density matrix is row major
with side `2^k` and trace one, and its `4^k` entries are priced as a `2k`-qubit
statevector against the dense export cap. A noise model sends the marginal and the
entropy to the density matrix, which answers the marginal of the exact mixture and
declines the entropy.

`Simulate::overlap` takes a second seeded builder, so each side has its own backend, seed
and start state; both circuits must declare the same width. The result is the squared
modulus of the inner product of the two normalized states. The amplitude is not
reported, since a tableau keeps no global phase and every MPS truncation moves one. Every
route divides by both norms, so an unnormalized chain answers like a unit-norm state. Two
states in the same representation take the native route at any width; any other pair
uses a dense export of both, which reaches as far as the export cap. A noise model on
either side is rejected, since the fidelity of two mixtures is a different computation.

## Auto dispatch policy

Routing constants are static values in `src/sim/dispatch.rs`, except the dense memory
caps, which `src/backend/memory.rs` derives from detected physical memory.

Capability limits are detected: a wrong one means a failed allocation or a swapping
host, and physical memory is cheap to read and hard to misread. `PRISM_MAX_SV_QUBITS`
and its siblings override the derivation.

Performance thresholds stay static, with an environment override where the value is
worth sweeping. The device crossover is the worked case: a default of 14 qubits with
`PRISM_GPU_MIN_QUBITS` beside it, not a figure read off the installed card. Detecting
cache geometry or device throughput wrongly is a silent slowdown no test catches, while
a wrong static default is the same slowdown on every host and shows in one measurement.
Runtime detection waits for a host that can validate it.

Changing a threshold takes a measurement. The `auto/crossover` group in
`benches/circuits.rs` holds both arms either side of the factored-stabilizer floor, the
exact stabilizer-rank budget and the Pauli marginals floor. The fusion floors and the
parallel threshold have no such pair, since nothing selects between them at run time:
moving one means building both values and comparing the binaries.

## What a backend reports about its own result

Four `Backend` methods put provenance on every result. `resolved` names the engine;
`exactness` says whether the representation can discard state weight and how much this
run discarded; `placement` says whether the state lived on the device; `bond_report`
gives the peak bond dimension against the cap for a representation bounded by one. All
four have defaults, and a backend keeping the `resolved` default is named by
`Backend::name`.

These are reports, not predictions: `exactness` is read after the run, from the
singular values actually discarded, and `placement` after any device fallback. Only the MPS answers
`bond_report`, and a peak at the cap is the hard signal that the cap bound the run. The
MPS sums discarded weight per SVD and returns `1 - total` as the fidelity lower bound.
That is a first-order estimate, not a certificate: errors compound across SVDs, and the
strict infidelity bound is the square of the summed square roots, which matches the sum
only when one SVD truncates.

The decomposed route runs one backend per independent block and merges: exactness is the
weakest of the parts, the fidelity bound is the product, the bond report is the peak over
the MPS blocks against their shared cap, and placement is `Device` only when every block
was. Per-shot routes evolve one state per shot and keep the weakest claim, with the bound
a minimum rather than a product.

The [GPU backend](../guides/gpu.md) is documented as a user guide. The distributed
statevector backend is covered in the
[Capability and Support Matrix](../guides/capabilities.md).

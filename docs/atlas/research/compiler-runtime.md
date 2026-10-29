# Compiler and runtime source audit

Source snapshot: `7adde35d0bacaff71711148223ac8df06a008d01`. The companion
`src/data/runtime-evidence.json` contains machine-readable claim records. All findings
below are source observations or explicitly marked mathematical models. No benchmark
was run for this audit, and no timing claim is established by reading implementation.

## Circuit representation

`src/circuit/mod.rs:44` defines `Circuit` as register widths and an instruction vector.
`Instruction` at line 1659 carries gates, measurements, resets, barriers, conditionals,
guarded regions, and save points. Gate targets use `SmallVec<[usize; 4]>`. A circuit
visualization must retain measurement and classical dependencies, rather than treating
every instruction as a unitary matrix.

`src/gates/mod.rs:24` defines the gate enum. Larger matrices and batch data are boxed;
`src/gates/tests.rs:430` checks the 16-byte enum size. `Fused` holds a 2 by 2 matrix,
`Fused2q` a 4 by 4 matrix, and `MultiFused` or `Multi2q` holds a tiled batch. A batch is
not necessarily one large dense unitary. `Gate::Unitary` documents its own packing:
`targets[0]` is the most significant matrix-index bit, even though global qubit zero
is the least significant statevector bit. These are distinct conventions.

`PreparedCircuit` in `src/circuit/prepared.rs:27` reuses parameter-binding structure,
fusion provenance, and dispatch work. Its matrix replay falls back to the full pipeline
when recognition or diagonality changes. The internal provenance tracer in
`src/circuit/plan.rs` supports parameter replay; it is not a public optimizer-step
animation API. A chapter claiming actual intermediate passes needs instrumentation
or generated fixtures, not a JavaScript reimplementation labeled native execution.

## Fusion order and legal movement

The public entry is `fuse_circuit_for_profile`, `src/circuit/fusion.rs:1704`.
`supports_fused = false` returns the circuit borrowed. Otherwise the host pipeline in
`fuse_at_width`, line 1717, runs in this order:

1. Recursively fuse guarded region bodies.
2. Cancel self-inverse two-qubit pairs.
3. Recognize Rzz patterns.
4. Batch Rzz operations when the buffer width is at least 14.
5. Return early below the general fusion floor of 10.
6. Fuse single-qubit matrices, reorder single-qubit gates, cancel newly adjacent
   self-inverse pairs, and fuse single-qubit matrices again.
7. Absorb eligible single-qubit operations into two-qubit gates at width 12.
8. Fuse runs on the same unordered qubit pair at width 12.
9. Choose the stage planner when `fusion_stage::stages_at` holds; otherwise run
   `batch_passes`.

The ordinary batching branch, `fusion.rs:1776`, applies multi-1q batching at 14,
tile reordering and multi-2q batching at 12, Pauli-rotation batching at 16,
controlled-phase and diagonal batching at 14, then post-phase 1q batching. The last
floor is 14 for a buffer whose width equals the circuit width and 18 for a wider
buffer. Density matrices therefore gate on their buffer width, typically twice the
logical width. Native floors differ from reduced test configurations.

The stage planner is a material part of current behavior. Its floor is derived from
detected last-level cache capacity in `fusion_stage.rs:31`: the first complex128
statevector width that exceeds that cache, with a minimum floor of 16 and a fallback
floor of 20. It applies only when buffer and circuit widths agree and the width fits
in a 64-bit mask. A fixed claim that all circuits above 20 qubits use this planner
would be incorrect across hosts. `plan_with`, line 143, schedules operations whose
dependencies are satisfied into tile, Pauli, and diagonal stages. Measurements,
resets, conditionals, regions, barriers, saves, and unsupported batch operations end
a segment. Its heuristic trades high-qubit tile capacity against ready operations;
it does not solve an optimal scheduling problem.

The device profile, `fusion.rs:1757`, retains cancellation, Rzz recognition,
single-qubit fusion and reordering, and same-pair two-qubit runs. It avoids absorbing
single-qubit gates around an isolated CX into a dense 4 by 4 gate. Device lowering
then has the original permutation and local gates available to schedule.

Composition multiplies on the left: chronological `G1, G2, G3` becomes
`G3 * G2 * G1`. `cancel_self_inverse_pairs`, line 154, cancels CX, CZ and SWAP pairs
only when no intervening instruction touches their qubits. Reversed CX targets do
not cancel; CZ and SWAP permit reversed target order. `reorder_1q_gates`, line 362,
moves gates across disjoint qubits and applies explicit diagonal rules: a diagonal
1q operation can cross CX on its control, CZ or Rzz on either qubit, and supported
diagonal batches. It cannot cross CX on its target by that rule. This is a concrete
set of dependency rules, not unrestricted symbolic commutation.

## Memory and roofline models

For a full sweep of a complex128 statevector, one read and one write of all
`2^n` amplitudes transfers `2 * 16 * 2^n = 32 * 2^n` payload bytes. This is a
mathematical traffic model. Cache residence, specialized permutation or diagonal
kernels, write behavior, and repeated tile reuse can change actual DRAM traffic.

For `g` independent full sweeps, the model is `g * 32 * 2^n` bytes. A hypothetical
uniform fusion group of size `f` gives `ceil(g/f)` sweeps, but the real planner's
pass count must come from actual output batches. `time >= bytes / bandwidth` is a
bandwidth lower bound under the assumed bandwidth, not predicted elapsed time.
The roofline bound is `min(peak_flops, bandwidth * arithmetic_intensity)`; comparisons
require compatible FLOP and byte conventions. Dense two-qubit fusion can increase
arithmetic while reducing sweeps, so fewer gates alone does not establish speedup.

## Dispatch depends on the requested answer

The base family tree in `src/sim/dispatch.rs:500` selects product state for no
entangling gates, stabilizer for Clifford circuits, then sparse or MPS above the
statevector cap, factored for partial independence, and dense statevector otherwise.
Sparse routing checks the platform index width and either structural sparsity or a
probe that stays at no more than 128 map entries through every gate. The probe
declines reset, save, and nonterminal measurement cases. The memory cap is resolved
through `src/limits.rs`, and the default automatic MPS bond cap is 256, with a
request-level override available.

That base tree is not the entire dispatch story. `src/sim/route.rs:18` can decompose
independent components, select factored stabilizers, or exploit a temporal Clifford
prefix. `decide_shots`, line 223, considers compiled Clifford sampling, deferred
measurement, Clifford registers, terminal statevector sampling, stabilizer rank,
native sampling and per-shot execution. `decide_expectations`, line 274, can use
deterministic Pauli propagation or tensor contraction. Non-Clifford content does
not force dense simulation. Noise and arbitrary initial states further constrain
the available paths.

Use `simulate(&circuit).plan(Query::...)`, `src/sim/plan.rs:153`, to generate actual
dispatch fixtures. `Plan` exposes backend, rule, placement, exactness, alternatives,
fallbacks, memory estimate, and predictable failures. A memory estimate excludes
circuit and fusion storage, allocator overhead, and process memory; `Unknown` must
remain unknown in a visualization. Recorded plans need the terminal, feature set,
limits, hardware context and source revision. A backend explorer based only on the
base tree must be labeled a simplified source model.

## Sparse and factored representations

`src/backend/sparse.rs:91` stores `FxHashMap<usize, Complex64>` plus a retained
swap buffer. Memory depends on support, with map overhead beyond the amplitude
payload. Hadamard layers can expand support exponentially. The default pruning
threshold is `1e-28` in squared magnitude; affected operations prune and rescale the
kept state. Its index representation cannot exceed `usize::BITS` qubits. Dense
multi-qubit `Gate::Unitary` is unsupported. A diagram must not imply arbitrary
width or universal gate compatibility merely because support remains small.

`src/backend/factored/mod.rs:108` stores dense states per entangled group, a sorted
global qubit list for each group, and a global-to-group map. Crossing groups merges
them by tensor product, `merge_substates`, line 196. Measurements and resets can
split a qubit out once a group reaches the seven-qubit split floor. Worst-case
entanglement grows one group to a full dense statevector. Sparse connectivity in a
gate diagram does not itself guarantee small groups throughout execution.

## MPS and tensor networks

`src/backend/mps/mod.rs` stores site tensors with shape `(chi_left, 2, chi_right)`.
The flat index is `alpha * (2 * chi_right) + physical * chi_right + beta`. Local gates
act on one site; adjacent pair gates use SVD and nonadjacent pairs route through
SWAP chains. `truncated_svd_rank`, `src/backend/mps/svd.rs:233`, retains singular
values strictly above `epsilon * largest`, keeps at least one, and applies the bond
cap. The default relative epsilon is `1e-12`.

`record_truncation`, `mps/mod.rs:476`, adds the discarded squared singular values
divided by the total squared singular values for each cut. This total can exceed
one. Cuts that can lose meaningful weight move the orthogonality center as needed;
the implementation must not be visualized as truncating an arbitrary ungauged
tensor and interpreting its raw singular values as physical Schmidt values.
Truncation does not immediately renormalize the chain; result queries normalize.

`RunMetadata::fidelity_lower_bound`, `src/sim/metadata.rs:214`, explicitly documents
that the MPS value `max(0, 1 - sum(local discarded weights))` is a first-order
estimate, not a certificate. Compounded error needs the square of the summed square
roots in a conservative infidelity bound. Display the reported field with that
qualification. `BondReport` distinguishes peak retained dimension from the cap;
reaching the cap alone does not measure how much weight was lost. `S <= log2(chi)`
is a mathematical relation for a normalized Schmidt spectrum, not a runtime claim.

`src/backend/tensornetwork.rs` is a separate deferred-contraction backend, not an
alias for MPS. Gates append tensors; queries trigger contraction. The metadata
planner uses greedy pair choice and seeded restarts for large intermediates,
`plan_with_restarts`, line 856. Exact index slicing trades time for lower peak
memory. A positive bounded-contraction tolerance permits rank reduction across
selected cuts before replanning and slicing. The exact mode, bounded mode, MPS
bond cap, and local SVD epsilon must remain separate controls and concepts.

## Existing validation and benchmark infrastructure

`tests/integration/fusion_correctness.rs` compares fused and unfused execution,
including complex amplitudes with `1e-10` tolerance. The unfused reference calls the
backend directly; the fused path uses `sim::run_on`. `src/circuit/fusion_tests.rs`
includes matrix-order and dependency counterexamples. `fusion_stage_tests.rs:89`
compares staged streams against unfused states, while later tests check tile fit
and deterministic scheduling. MPS, sparse and tensor-network integration suites
compare against statevector results. Test existence is evidence of intended
coverage; only a recorded test run establishes that a revision passes.

`benches/common/mod.rs:14` fixes circuit seed `0xDEAD_BEEF`. Criterion uses 30 samples
by default, adjustable by environment, with a ten-sample fast tier. The fast tier
is triage. `scripts/bench_ab.sh` compares separate binaries with controls and
discarded warmups; `scripts/bench_interleave.sh` interleaves variants in one binary
for changes expressible through a coarse switch. Both default to a 5 percent
regression threshold and distinguish PASS, FAIL, busy host, and RERUN states.
RERUN is not evidence of no regression. Existing script comments describe the
statistical procedure; they do not provide a new measured result for this atlas.

`.github/workflows/ci.yml` checks formatting, clippy, nextest and doctests, with
additional release-profile and reduced-threshold lanes. GPU and distributed paths
have their own configurations. Benchmark records in the atlas must preserve raw
provenance, CPU, OS, compiler and flags, features, thread count, precision, terminal,
seed, timing methodology, and exactness. No competitor ranking or native browser
performance equivalence follows from this audit.

# Compiler and runtime source audit

Source snapshot: `7adde35d0bacaff71711148223ac8df06a008d01`. The companion
`src/data/runtime-evidence.json` holds the claim records. Everything below is read
from the source or marked as a mathematical model. No benchmark was run for this audit.

## Circuit representation

`src/circuit/mod.rs:44` defines `Circuit` as register widths and an instruction vector.
`Instruction` at line 1659 carries gates, measurements, resets, barriers, conditionals,
guarded regions, and save points. Gate targets use `SmallVec<[usize; 4]>`. Because
measurement and classical dependencies are part of the IR, a circuit view cannot
treat every instruction as a unitary matrix.

`src/gates/mod.rs:24` defines the gate enum. Larger matrices and batch data are boxed;
`src/gates/tests.rs:430` checks the 16-byte enum size. `Fused` holds a 2 by 2 matrix,
`Fused2q` a 4 by 4 matrix, and `MultiFused` or `Multi2q` holds a tiled batch, which
need not be one dense unitary. `Gate::Unitary` documents its own packing: `targets[0]`
is the most significant matrix-index bit, while global qubit zero is the least
significant statevector bit.

`PreparedCircuit` in `src/circuit/prepared.rs:27` reuses parameter-binding structure,
fusion provenance, and dispatch work. Its matrix replay falls back to the full pipeline
when recognition or diagonality changes. The internal provenance tracer in
`src/circuit/plan.rs` exists for parameter replay and has no public API. Showing real
intermediate passes therefore needs instrumentation or recorded fixtures; a
JavaScript reimplementation would not be native output.

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
buffer. Density matrices gate on their buffer width, usually twice the logical
width. Tests run with lower floors than native builds.

The stage planner changes behavior on common widths. Its floor comes from the
detected last-level cache in `fusion_stage.rs:31`: the first complex128 statevector
width that exceeds that cache, at least 16, with a fallback of 20. It applies only
when buffer and circuit widths agree and the width fits in a 64-bit mask, so the
width at which it takes over differs between hosts. `plan_with`, line 143, schedules
operations whose dependencies are met into tile, Pauli and diagonal stages.
Measurements, resets, conditionals, regions, barriers, saves and unsupported batch
operations end a segment. The heuristic trades high-qubit tile capacity against ready
operations and makes no optimality claim.

The device profile, `fusion.rs:1757`, retains cancellation, Rzz recognition,
single-qubit fusion and reordering, and same-pair two-qubit runs. It leaves the
single-qubit gates around an isolated CX unabsorbed, so device lowering can schedule
the permutation and the local gates itself.

Composition multiplies on the left: chronological `G1, G2, G3` becomes
`G3 * G2 * G1`. `cancel_self_inverse_pairs`, line 154, cancels CX, CZ and SWAP pairs
only when no intervening instruction touches their qubits. Reversed CX targets do
not cancel; CZ and SWAP permit reversed target order. `reorder_1q_gates`, line 362,
moves gates across disjoint qubits and applies explicit diagonal rules: a diagonal
1q operation can cross CX on its control, CZ or Rzz on either qubit, and supported
diagonal batches. It cannot cross CX on its target. Only these listed rules apply;
there is no general symbolic commutation.

## Memory and roofline models

For a full sweep of a complex128 statevector, one read and one write of all
`2^n` amplitudes moves `2 * 16 * 2^n = 32 * 2^n` payload bytes. Cache residence,
specialized permutation or diagonal kernels, write behavior and tile reuse all change
the real DRAM traffic.

For `g` independent full sweeps, the model is `g * 32 * 2^n` bytes. A hypothetical
uniform fusion group of size `f` gives `ceil(g/f)` sweeps; the planner's real pass
count has to be read from its output batches. `time >= bytes / bandwidth` is a lower
bound at the assumed bandwidth, not a predicted time. The roofline bound is
`min(peak_flops, bandwidth * arithmetic_intensity)`, and comparing two of them needs
the same FLOP and byte conventions. Dense two-qubit fusion can add arithmetic while
removing sweeps, so a shorter gate list is not by itself faster.

## Dispatch depends on the requested answer

The base family tree in `src/sim/dispatch.rs:500` selects product state for no
entangling gates, stabilizer for Clifford circuits, then sparse or MPS above the
statevector cap, factored for partial independence, and dense statevector otherwise.
Sparse routing checks the platform index width and either structural sparsity or a
probe that stays at no more than 128 map entries through every gate. The probe
declines reset, save, and nonterminal measurement cases. The memory cap is resolved
through `src/limits.rs`, and the default automatic MPS bond cap is 256, with a
request-level override available.

Several routes sit in front of that tree. `src/sim/route.rs:18` can decompose
independent components, select factored stabilizers, or exploit a temporal Clifford
prefix. `decide_shots`, line 223, considers compiled Clifford sampling, deferred
measurement, Clifford registers, terminal statevector sampling, stabilizer rank,
native sampling and per-shot execution. `decide_expectations`, line 274, can use
deterministic Pauli propagation or tensor contraction. Non-Clifford content does
not force dense simulation. Noise and arbitrary initial states narrow the available
paths.

`simulate(&circuit).plan(Query::...)`, `src/sim/plan.rs:153`, produces real dispatch
fixtures. `Plan` exposes backend, rule, placement, exactness, alternatives, fallbacks,
memory estimate and predictable failures. The memory estimate excludes circuit and
fusion storage, allocator overhead and process memory, and an `Unknown` estimate has
no byte count to show. A recorded plan needs the terminal, feature set, limits,
hardware context and source revision. A backend explorer built from the base tree
alone would be a simplified model and should say so.

## Sparse and factored representations

`src/backend/sparse.rs:91` stores `FxHashMap<usize, Complex64>` plus a retained
swap buffer. Memory depends on support, with map overhead beyond the amplitude
payload. Hadamard layers can expand support exponentially. The default pruning
threshold is `1e-28` in squared magnitude; affected operations prune and rescale the
kept state. The index type limits width to `usize::BITS` qubits, and dense
multi-qubit `Gate::Unitary` is unsupported, however small the support stays.

`src/backend/factored/mod.rs:108` stores dense states per entangled group, a sorted
global qubit list for each group, and a global-to-group map. Crossing groups merges
them by tensor product, `merge_substates`, line 196. Measurements and resets can
split a qubit out once a group reaches the seven-qubit split floor. Worst-case
entanglement grows one group to a full dense statevector. Sparse connectivity in
the circuit does not guarantee the groups stay small.

## MPS and tensor networks

`src/backend/mps/mod.rs` stores site tensors with shape `(chi_left, 2, chi_right)`.
The flat index is `alpha * (2 * chi_right) + physical * chi_right + beta`. Local gates
act on one site; adjacent pair gates use SVD and nonadjacent pairs route through
SWAP chains. `truncated_svd_rank`, `src/backend/mps/svd.rs:233`, retains singular
values strictly above `epsilon * largest`, keeps at least one, and applies the bond
cap. The default relative epsilon is `1e-12`.

`record_truncation`, `mps/mod.rs:476`, adds the discarded squared singular values
divided by the total squared singular values for each cut. This total can exceed
one. Before a cut that can lose meaningful weight, the implementation moves the
orthogonality center there, so the singular values it truncates are Schmidt values
of the state, not of an arbitrary ungauged tensor. Truncation does not renormalize
the chain; result queries do.

`RunMetadata::fidelity_lower_bound`, `src/sim/metadata.rs:214`, documents the MPS
value `max(0, 1 - sum(local discarded weights))` as a first-order estimate. A
conservative infidelity bound uses the square of the summed square roots instead.
`BondReport` separates the peak retained dimension from the cap; hitting the cap says
nothing about how much weight was lost. `S <= log2(chi)` holds for a normalized
Schmidt spectrum and is not something the runtime reports.

`src/backend/tensornetwork.rs` is a separate backend that defers contraction.
Gates append tensors; queries trigger contraction. The metadata
planner uses greedy pair choice and seeded restarts for large intermediates,
`plan_with_restarts`, line 856. Exact index slicing trades time for lower peak
memory. A positive bounded-contraction tolerance permits rank reduction across
selected cuts before replanning and slicing. Exact mode, bounded mode, the MPS
bond cap and the local SVD epsilon are four separate settings.

## Existing validation and benchmark infrastructure

`tests/integration/fusion_correctness.rs` compares fused and unfused execution,
including complex amplitudes with `1e-10` tolerance. The unfused reference calls the
backend directly; the fused path uses `sim::run_on`. `src/circuit/fusion_tests.rs`
includes matrix-order and dependency counterexamples. `fusion_stage_tests.rs:89`
compares staged streams against unfused states, while later tests check tile fit
and deterministic scheduling. MPS, sparse and tensor-network integration suites
compare against statevector results. These show intended coverage; only a test
run shows that a revision passes.

`benches/common/mod.rs:14` fixes circuit seed `0xDEAD_BEEF`. Criterion uses 30 samples
by default, adjustable by environment, with a ten-sample fast tier. The fast tier
is triage. `scripts/bench_ab.sh` compares separate binaries with controls and
discarded warmups; `scripts/bench_interleave.sh` interleaves variants in one binary
for changes expressible through a coarse switch. Both default to a 5 percent
regression threshold and distinguish PASS, FAIL, busy host, and RERUN states.
RERUN means no verdict. The atlas contains no results from either script.

`.github/workflows/ci.yml` checks formatting, clippy, nextest and doctests, with
additional release-profile and reduced-threshold lanes. GPU and distributed paths
have their own configurations. Any benchmark record added to the atlas needs raw
provenance, CPU, OS, compiler and flags, features, thread count, precision, terminal,
seed, timing method and exactness.

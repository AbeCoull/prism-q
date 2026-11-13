# QEC Decoding

How a native QEC program becomes a detector error model, how that model is written and
read as text, and the three decoders that consume it. Program execution is on
[QEC programs](./qec-programs.md), and workflow examples in the
[QEC guide](../guides/qec.md).

## Detector error model

`QecProgram::detector_error_model` derives a `DetectorErrorModel`: the independent
error mechanisms implied by the program's noise annotations, detectors and
observables. Matching and belief propagation decoders consume this model rather than
raw detector samples.

The derivation reuses the [noisy data flow](./qec-programs.md#noisy-data-flow). Every
annotation expands into its fault branches, three per target for one-qubit channels and
fifteen per pair for two-qubit ones, and zero-probability branches do not fire. Each
branch's record mask projects through the detector and observable rows to its symptom,
the detectors and observables it flips.

Branches merge into mechanisms at fault-site granularity (one target of a one-qubit
annotation, or one target pair of a two-qubit one):

- Branches at one site are mutually exclusive, so same-symptom branches sum. A
  `DEPOLARIZE1` site whose X and Y branches flip the same records yields one mechanism
  at exactly `2p/3`.
- Distinct sites are independent, distinct targets of one annotation included, so
  same-symptom mechanisms compose as `p = p1(1-p2) + p2(1-p1)`.

Faults that flip no detector and no observable are omitted. Mechanisms keep program
order (the position of the annotation that first produced each symptom). They are
independent in the model even where the branches were exclusive, so model statistics
agree with the sampler to second order in the branch probabilities, not exactly.

Sampler semantics carry over. A measurement error argument (`M(p)`) is a
pre-measurement Pauli fault on that record's detectors, noise on an already-measured
qubit contributes nothing, and a two-qubit channel with one measured target enters as
its exact one-qubit marginal. Non-Clifford gates and reuse of a measured qubit without
reset are rejected, as on the noisy compiled sampling path. So are leakage annotations, whose
leak flag no independent mechanism expresses.

Hypergraph mechanisms (more than two detectors, as `DEPOLARIZE2` produces) stay intact
in the derived model, and check-matrix decoders consume them directly. A matching
decoder needs the graphlike form, which is opt-in. `decompose_graphlike` returns a new
model in which each hypergraph mechanism is replaced by two or more existing graphlike
mechanisms. Their non-empty detector sets partition its detectors, their observable XOR
matches it, and the mechanism's probability composes into every component. A
`DEPOLARIZE2` `Y`-`Y` branch, for example, splits into the same channel's single-qubit
branches. Cross-component correlations are lost, but single-detector marginals are
unchanged. A mechanism with no such cover is an error naming its symptom. Decomposition
is never applied silently, and the writer serializes whichever model it is given.

## Text format

`DetectorErrorModel::to_text` renders the model in the common detector error model text
format that external matching and belief propagation decoders read. The emitted
subset:

```text
error(<p>) D<i> ... L<j> ...
error(<p>) D<i> D<k> ^ D<m> L<j> ...
detector D<i>
detector(<c0>, <c1>, ...) D<i>
logical_observable L<j>
```

- One `error` line per mechanism in mechanism order, with its probability and its
  `D`-prefixed detector and `L`-prefixed observable indices, both ascending.
- A mechanism on more than two detectors is written as `^`-separated components: the
  suggestion it was imported with, otherwise the cover `decompose_graphlike` would
  pick, otherwise flat when no cover exists.
- One `detector` line per detector in index order, with the program's detector
  coordinates when present, and one `logical_observable` line per observable slot.

Indices are zero-based and dense, and probabilities print with enough digits to
round-trip exactly. There are no `repeat` blocks and no `shift_detectors`, because
derivation merges mechanisms across the whole program by symptom and the per-round
structure is not in the model. A distance-13 surface memory over 1000 rounds at uniform
circuit noise writes 3.4 million mechanisms, about 200 MB of text.

`DetectorErrorModel::from_text` reads the format back: `error(p)` with `D`, `L` and `^`
targets, `detector` with optional coordinates, `logical_observable`,
`shift_detectors(<coordinate shift>) <detector shift>`, and nested `repeat N { ... }`
blocks, expanded as read. Shifts accumulate in program order through the expansion.
Instruction names are case-insensitive, `error[tag](p)` tags are ignored, and `#` starts
a comment. Each `error` line becomes one mechanism in line order, not merged with others
of the same symptom, and a target named twice in one component cancels. Its `^`
components are kept as `ErrorMechanism::suggested_decomposition`. `decompose_graphlike`
splits along them when every component flips one or two detectors, adding any component
no existing mechanism matches. Detector and observable counts are one past the highest
index named. A `repeat` whose expansion would exceed 2^25 instructions is rejected
before it runs. Export then import reproduces the mechanisms, the detector coordinates,
and the text itself.

Example, one syndrome round of the three-qubit repetition memory under `X_ERROR(0.05)`
on the data qubits:

```text
error(0.05) D0
error(0.05) D0 D1
error(0.05) D1
detector D0
detector D1
```

In Python, `QecProgram.detector_error_model()` returns the model with
`probabilities()` (float64), `detector_matrix()` and `observable_matrix()` (bool,
detectors or observables by mechanisms), `detector_coords()`,
`suggested_decompositions()`, and `to_text()`. `DetectorErrorModel.from_text` reads
one. The matrix triple feeds check-matrix decoder constructors directly.

## Decoders

Three decoders in the crate take a run from samples to a logical error rate with no
external tool:

| Decoder | Model | Correction |
| --- | --- | --- |
| `UnionFindDecoder` | graphlike | union-find with peeling, almost-linear cost |
| `MatchingDecoder` | graphlike | exact minimum-weight perfect matching |
| `BpOsdDecoder` | any, hypergraph included | belief propagation, then ordered statistics |

All three compile from a `DetectorErrorModel` and share one batch driver.
`decode_packed` takes a `PackedShots` of detector samples in either layout and returns
shot-major predicted observable flips. `logical_error_rate` decodes and counts shots
whose prediction differs from the sampled observables in any observable. Scratch is
reused across shots, and under the `parallel` feature large batches split into Rayon
chunks with one scratch per chunk. A shot the model cannot explain rejects the batch
with an error naming the first such shot.

Mechanisms that flip no detector cannot enter any decoder's graph or check matrix.
Their probability mass is a floor under the logical error rate over the model.
Mechanisms sharing one detector set collapse to the most probable, in all three
decoders.

### Union-find

Weighted cluster growth with peeling (Delfosse and Nickerson, arXiv:1709.06218).
`UnionFindDecoder::from_model` compiles a graphlike model: detectors become vertices, a
two-detector mechanism an internal edge, a one-detector mechanism a boundary edge, each
weighted `ln((1-p)/p)` clamped at zero. A mechanism with more than two detectors is
rejected with a pointer to `decompose_graphlike`.

Decoding is deterministic: no randomness, ties break by ascending edge index in
mechanism order, and serial and Rayon results are identical. Under the `parallel`
feature, batches of at least 1024 shots decode in parallel over 256-shot chunks with no
per-shot allocation. A model with no observables decodes serially at any batch size.

On the enumerable distance-3 models the exact union-find failure rate sits between the
maximum-likelihood rate and `P(two or more faults)` plus the measured single-fault
misses.

### Minimum-weight perfect matching

`MatchingDecoder::from_model` builds the union-find graph (same edge dedup, same
weights) and quantizes every weight to an even integer, `2 * round(w / w_max * 2^23)`.
Matching is exact for the quantized weights. Each shot's correction is a minimum-weight
edge set whose boundary is the shot's defects, with the open boundary absorbing any
number of chain ends. Construction runs one multi-source Dijkstra from every boundary
edge, giving each detector its exact boundary distance.

Per shot the decoder runs Edmonds' primal-dual blossom algorithm in the cut
formulation, organized like sparse blossom (Higgott and Gidney, arXiv:2303.15933).
Each defect grows its own Dijkstra ball instead of sparse blossom's vertex-owning
regions, so a shrinking T node needs no undo on the graph. Every tight pair carries its
exact distance, and the result is optimal by LP duality. Arithmetic is exact `i64`:
with even weights every event falls on an integer time.

Per shot with `k` defects, Dijkstra work is `O(B log B)` over the `B` vertex entries the
balls settle. The matcher performs at most `k` augmentations, `O(k^2 P log P)` worst
case over `P` discovered pairs. At low defect density balls stop at the first collision
or the boundary, so work stays close to linear in `k`. Decoding allocates nothing per
shot after warm-up.

Brute-force checks on random graphs of up to 48 detectors confirm optimality, and on
the distance-3 models `rate_ML <= rate_MWPM <= rate_UF`.

### BP+OSD

`BpOsdDecoder` takes any model. Each mechanism flipping at least one detector is a
column of the check matrix (detectors are rows) with prior log-likelihood ratio
`ln((1-p)/p)`. Belief propagation runs a flooding schedule on the Tanner graph with the
product-sum rule or scaled min-sum (default scaling 0.625). It runs at most 30
iterations by default and stops once its hard decision reproduces the syndrome.

A converged decision can still carry a zero-syndrome cycle on top of a lighter
solution. On the distance-3 color code, the first min-sum iteration satisfies the
syndrome of a single center-qubit fault with that fault plus a weight-3 logical. The
decoder therefore reruns the OSD search restricted to the decision's support and the
checks it touches, and keeps the lighter of the two, at `O(k * r)` for a decision of
`k` columns touching `r` checks.

When belief propagation does not converge, ordered-statistics decoding (Panteleev and
Kalachev, arXiv:1904.02703; Roffe et al., arXiv:2005.07016) sorts columns by posterior
log-likelihood and reduces the check matrix in that order over GF(2). A nonzero
right-hand side below the rank rejects the batch. Each candidate fixes the free
(non-pivot) bits and reads the pivot bits from the reduced rows, and the candidate of
least prior weight wins:

| `OsdMethod` | Candidates |
| --- | --- |
| `Zero` | the all-zero free pattern |
| `CombinationSweep { order }` | plus every single free bit and every pair among the first `order` free bits (default order 7) |
| `Exhaustive { order }` | all `2^order` patterns over the first `order` free bits in Gray-code order; exact maximum-likelihood error decoding once `order` covers every free column |

Belief propagation costs `O(iterations * nnz)` per shot. Elimination costs
`O(rank * rows * columns / 64)` word operations, and each candidate one packed XOR over
the pivot rows plus a weighted walk of its set bits. Batches of 64 shots or more decode
in parallel over 16-shot chunks.

On the [15, 7, 5] BCH code every error of weight at most two is recovered under every
rule and OSD method. `decompose_graphlike` drops the correlations of the hyperedges it
splits, and that is where BP+OSD gains: on the distance-3 surface memory at p = 0.02
under two-qubit depolarizing noise (20k shots, seed 42), union-find fails 603 times and
exact matching 654 times over the decomposed model, BP+OSD 436 times over the full one.

Python exposes `UnionFindDecoder(model)`, `MatchingDecoder(model)` and
`BpOsdDecoder(model, ...)`, each with `decode(detectors) -> (shots, num_observables)`
over numpy bool arrays, `decode_packed(detectors)` over the packed `uint8` rows of
`packed_detectors()`, and `logical_error_rate(detectors, observables)` over either
form.

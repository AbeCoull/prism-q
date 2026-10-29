# Technical limitations and provenance

The atlas audits PRISM-Q revision `7adde35d0bacaff71711148223ac8df06a008d01`.
Source links and a SHA-256 manifest preserve that snapshot. This is not a claim
that future PRISM-Q revisions have identical branches, thresholds, or semantics.

## Evidence levels

| Level | Supplied evidence | What it establishes |
| --- | --- | --- |
| Native execution snapshots | `fixtures/native-gates.json` | Actual small-state amplitudes before and after `StatevectorBackend::apply` |
| Native compiler snapshots | `fixtures/native-fusion.json` | Actual pass outputs, matrix payloads, kernel traversal counters, and full-state comparisons for the recorded input |
| Source audit | Three research reports and 65 evidence records | Data layouts, branch predicates, algorithms, and explicit limitations at the pinned revision |
| Mathematical reference | Browser models and independent dense-operator tests | Unitary evolution, normalization, bit ordering, address arithmetic, and modeled units |
| Analytical hardware model | Cache rows, logical traffic, roofline, routing probe | Predictions conditional on stated inputs, never measured runtimes |
| Empirical performance | No dataset imported | No throughput, crossover, scaling, or competitor claim |

The native gate fixture was generated with Rust 1.99.0 on Windows 10.0.19045, an
Intel Core i7-6700K, the `parallel` feature, the dev profile, and empty `RUSTFLAGS`.
The fixture itself records these values plus the exporter source. The source
tree includes disclosed test-only export changes. The exporter is preserved as
part of the fixture provenance; it was not part of the audited base revision.
The input formula is deterministic and the backend seed is 42.

Native snapshots contain amplitudes, not an instrumented instruction, cache,
thread, or ISA trace. Source analysis supplies the hardware explanation. Selecting
NEON does not assert that the native snapshots were generated on an ARM host.
There is no WebAssembly implementation or native performance result in the page.

The compiler fixture adds 31 circuit records and 432 snapshots. Both fixture families
retain the actual recording checkout revision and dirty-tree flag separately from
the audited revision. `dirtyTree: true` discloses the added test-only exporter wiring
and counter visibility. There are 223 canonical producer hashes and embedded
exporter text. Its eight-thread CPU execution observes the
host's reported 8 MiB last-level cache and stage-planner floor of 20. Device-profile
IR is also verified on the CPU. No GPU runtime or launch count was recorded.
The [compiler recording contract](research/fusion-traces.md) specifies the pass,
counter, numerical, and target-order conventions.

Generation checks the audited source hashes before execution. The comparison omits
only the exact test-module registration added to `fusion.rs`; full producer hashes
still include that registration. Recording fingerprints cover every `src` file,
the audited file set, Cargo manifests, build configuration, and exporter scripts.
Effective environment metadata includes tile overrides, reordering configuration,
available parallelism, `RUSTFLAGS`, and `CARGO_ENCODED_RUSTFLAGS`. Evidence verification
checks those recording fingerprints, so new commits can retain the audit revision
without falsely identifying their checkout as the original audit commit.

## Numerical boundaries

The CPU dense backend uses `Complex64` from `num-complex`, meaning complex128
storage, two f64 components. The browser uses JavaScript numbers and Float64Array.
Native FMA can round differently from separate browser operations. Golden tests
require amplitude error below 1e-12, not bitwise identity or a universal accuracy
guarantee for arbitrarily deep circuits. Norm preservation alone does not prove
that the correct unitary was applied.

The exporter executes the gate matrix through `Gate::Fused`. PRISM-Q tests its
diagonality with a 1e-12 off-diagonal magnitude threshold, so Z, Rz, and sufficiently
small off-diagonal rotations may use independent scaling. The route display
uses that same classification. The browser's educational matrix model retains
the full mathematical matrix, including sub-threshold terms; near the native
classification threshold its values can differ by the skipped terms. Matching
native fixtures use the fixed nonzero angle pi/3.

The five stages are instructional subdivisions of gate application. No partial
native instruction trace is implied. The numerical panels show the selected
gate's input and completed output. Circuit depth means repeated application of
the same unitary to the same target, with input at depth minus one.

The execution microscope evaluates source-derived SIMD instruction groups in
JavaScript. It is not disassembly or an instrumented instruction trace. The
compiler's full-state error checks compare each recorded IR against unfused
execution on one deterministic normalized complex input. They do not prove
operator equivalence for arbitrary inputs. Independent browser tests also embed
small native matrices and check chronological composition.

The dependency DAG conservatively orders operations sharing a qubit. It is not
the native planner's internal graph. Transformed instruction ancestry was not
exported and is not inferred. The operator landscape encodes magnitude as height
and phase as a spoke in a complex plane; it does not represent physical storage.
Zero coefficients have zero height and undefined phase. Very small phase spokes
are omitted at the disclosed visual threshold.

MPS research describes the actual relative cutoff, bond cap, and local discarded
weights. Their cumulative sum may exceed one and does not certify final global
infidelity. Stabilizer, noise, QEC, CUDA, MPI, and dispatch details are audited
dossiers, not currently implemented interactive chapters. Dispatch is not
recreated from simplified width or gate-count controls.

## Architectural boundaries

The isometric scene is a numerical storage and register schematic, not a physical
CPU or DRAM layout. Uniform slot heights encode uniform f64 storage width. The
pair guide is an index relation, not a wire. Cache lines assume a 64-byte line
and a line-aligned base; the underlying Vec does not promise that base alignment.
The scene shows at most 64 amplitudes, using explicitly marked address windows
for larger states. Relative byte offsets remain exact.

AVX2 capability and actual register width are separate. General one-qubit and
diagonal gates have different predicates. Scalar fields are not SIMD lanes.
The displayed register banks show inputs and ascending memory lane order, not
the complete live register allocation produced by a compiler. No disassembly
was captured; intrinsic explanations link to the actual source implementation.

Pair distance does not imply random stream traversal, cache misses, TLB misses,
NUMA placement, or bandwidth saturation. Threaded route labels establish source
partitioning, not task count, physical core count, affinity, or scaling efficiency.
The roofline uses decimal GB/s and GFLOP/s and binary memory-size units. It assumes
full read/write sweeps and ideal grouped reuse. Diagonal skips, register pressure,
launch cost, allocation, cache residency, and dense fusion arithmetic can violate
those assumptions. A streaming lower bound is conditional on traffic actually
reaching the specified memory level.

## Review and visual validation

Three genuine parallel research agents covered mathematics, runtime, and hardware.
The runtime researcher subsequently implemented the renderers. The mathematics
researcher independently checked the application model and built a dense tensor
reference, but was also the author of the mathematical prose. The hardware
researcher cross-reviewed mathematical/runtime claims and the application; its
hardware prose received self-review. These are bounded cross-reviews, not six
independently staffed teams. Findings and fixes are recorded in the adversarial
review report.

The expansion used a compiler recorder, a compiler-contract reviewer, and an
execution-microscope implementer with browser checks. The compiler reviewer also
wrote the independent matrix replay tests, so that portion combines review and
implementation. Matrix packing, true zero columns, and the rewrite counterexample
received explicit cross-review.

Browser checks cover controls, share-link restoration, dialog focus, manual stages,
mobile overflow, reduced-motion preference, and SVG fallback. Screenshot comparisons
use reviewed platform-specific baselines; they do not prove mathematical correctness
or absence of all possible label collisions. Font assets are bundled locally.
The WebM advances the actual five page stages at an arbitrary presentation pace;
that pace is unrelated to simulator execution time.

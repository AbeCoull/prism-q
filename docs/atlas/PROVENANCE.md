# Technical limitations and provenance

The atlas audits PRISM-Q revision `7adde35d0bacaff71711148223ac8df06a008d01`.
Source links and a SHA-256 manifest pin that snapshot. Later revisions can change
branches, thresholds or semantics.

## Evidence levels

| Level                      | Supplied evidence                                    | What it establishes                                                                                         |
| -------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Native execution snapshots | `fixtures/native-gates.json`                         | Small-state amplitudes before and after `StatevectorBackend::apply`                                         |
| Native compiler snapshots  | `fixtures/native-fusion.json`                        | Pass outputs, matrix payloads, kernel traversal counters, and full-state comparisons for the recorded input |
| Source audit               | Three research reports and 65 evidence records       | Data layouts, branch predicates, algorithms, and limitations at the pinned revision                         |
| Mathematical reference     | Browser models and independent dense-operator tests  | Unitary evolution, normalization, bit ordering, address arithmetic, and modeled units                       |
| Analytical hardware model  | Cache rows, logical traffic, roofline, routing probe | Predictions from stated inputs, never measured runtimes                                                     |
| Empirical performance      | No dataset imported                                  | No throughput, crossover, scaling, or competitor claim                                                      |

The native gate fixture was recorded with Rust 1.99.0 on Windows 10.0.19045, an
Intel Core i7-6700K, the `parallel` feature, the dev profile, and empty `RUSTFLAGS`.
The fixture stores these values and the exporter source. The exporter was added
after the audited revision, so it is kept with the fixture. The input formula is
deterministic and the backend seed is 42.

Native snapshots hold amplitudes. They are not an instruction, cache, thread or
ISA trace; the hardware explanation comes from the source audit. Selecting NEON
does not mean the snapshots came from an ARM host. The page has no WebAssembly
build and shows no native performance result.

The compiler fixture adds 31 circuit records and 432 snapshots. Both fixture families
record the checkout revision and dirty-tree flag separately from the audited
revision. The committed fixtures were recorded from a clean tree (`dirtyTree: false`)
and carry 221 producer hashes plus the embedded exporter text. The recording ran on
eight CPU threads, saw an 8 MiB last-level cache, and used a stage planner floor of 20. Device-profile IR was also executed on the CPU; no GPU runtime or launch count
was recorded. The [compiler recording contract](research/fusion-traces.md) gives
the pass, counter, numerical and target-order conventions.

Generation checks the audited source hashes before it runs. That comparison skips
only the test-module registration added to `fusion.rs`; the full producer hashes
include it. Recording fingerprints cover every `src` file, the audited file set,
the Cargo manifests, build configuration and exporter scripts. Environment metadata
records tile overrides, reordering configuration, available parallelism, `RUSTFLAGS`
and `CARGO_ENCODED_RUSTFLAGS`. Verification checks those fingerprints, which lets a
later commit keep the audit revision without claiming to be the audit commit.

## Numerical boundaries

The CPU dense backend stores `Complex64` from `num-complex`: complex128, two f64
components. The browser uses JavaScript numbers and `Float64Array`. Native FMA can
round differently from separate browser operations. Golden tests require amplitude
error below 1e-12; they do not require bitwise identity or bound error for
arbitrarily deep circuits. Norm preservation alone does not show that the right
unitary was applied.

The exporter applies each gate matrix through `Gate::Fused`. PRISM-Q treats a fused
matrix as diagonal when its off-diagonal magnitudes are below 1e-12, so Z, Rz and
rotations with tiny off-diagonal terms can take the independent scaling path. The
route display uses the same test. The browser model keeps the full matrix, including
terms below the threshold, so near the threshold its values can differ by those
terms. The matching native fixtures use the fixed nonzero angle pi/3.

The five stages split one gate application for teaching. They do not correspond to
a partial native instruction trace. The numerical panels show the selected gate's
input and finished output. Circuit depth means applying the same unitary to the
same target repeatedly; the input at depth d is the output at depth d minus one.

The execution microscope evaluates the source SIMD instruction groups in
JavaScript; nothing is disassembled or traced. The compiler's full-state error
checks compare each recorded IR with unfused execution on one deterministic
normalized complex input. That does not prove operator equivalence for every input.
Separate browser tests embed small native matrices and check chronological
composition.

The dependency graph orders any two operations that share a qubit. It is not the
native planner's internal graph. Instruction ancestry through transforms was not
exported, and the page does not infer it. The operator landscape draws magnitude
as height and phase as a spoke in the complex plane; it has nothing to do with
physical storage. Zero coefficients have zero height and no phase. Phase spokes
below the stated visual threshold are omitted.

The MPS research covers the relative cutoff, bond cap and local discarded weights.
The cumulative sum can exceed one and does not bound final global infidelity.
Stabilizer, noise, QEC, CUDA, MPI and dispatch are documented as audited dossiers
without interactive chapters. Dispatch is not simulated from width or gate-count
controls.

## Architectural boundaries

The isometric scene is a schematic of numerical storage and registers, not a
physical CPU or DRAM layout. Every slot has the same height because every f64 has
the same width. The pair guide marks an index relation, not a wire. Cache lines
assume 64 bytes and a line-aligned base, which `Vec` does not guarantee. The scene
draws at most 64 amplitudes and uses marked address windows for larger states.
Relative byte offsets stay exact.

AVX2 capability and the register width actually used are separate questions, and
general one-qubit gates and diagonal gates use different predicates. Scalar fields
are not SIMD lanes. The register banks show inputs in ascending memory lane order,
not the compiler's full register allocation. No disassembly was captured; intrinsic
explanations link to the source.

A large pair distance does not imply random access, cache or TLB misses, NUMA
placement or saturated bandwidth. Threaded route labels describe how the source
partitions work, not task count, core count, affinity or scaling. The roofline uses
decimal GB/s and GFLOP/s and binary memory sizes. It assumes full read/write sweeps
and ideal reuse within a group. Diagonal skips, register pressure, launch cost,
allocation, cache residency and dense fusion arithmetic can all break those
assumptions. The streaming lower bound holds only if the traffic actually reaches
the stated memory level.

## Browser validation

Browser checks cover controls, share-link restoration, dialog focus, manual stages,
mobile overflow, the reduced-motion preference and the SVG fallback. Screenshot
comparisons use reviewed baselines for each platform. They catch layout changes;
they say nothing about mathematical correctness and cannot rule out every label
collision. Fonts are bundled. The WebM steps through the five page stages at a
presentation pace unrelated to simulator execution time.

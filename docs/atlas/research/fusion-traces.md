# Native compiler traces

The compiler laboratory reads `fixtures/native-fusion.json`. Generate it with
`node docs/atlas/scripts/generate-fusion-traces.mjs`. The script runs one ignored Rust
unit test with `parallel` enabled and an eight-thread Rayon pool. Export requires an
explicit `PRISM_ATLAS_FUSION_OUTPUT` destination. Ordinary tests never write fixtures.

## Captured evidence

`src/circuit/fusion_atlas_tests.rs` calls the real internal pass functions in the order
used by `fuse_at_width`, `device_passes`, and `batch_passes`. It records the original IR
and the output of each pass. Disabled width floors retain the same IR with
`enabled: false`; these are explicit skipped passes, not executions. The recursive
region pass is included, although these fixtures contain gate-only circuits.

Every final stream is compared with `fuse_circuit_for_profile` gate by gate, including
targets and the complete native payload. No browser optimizer reconstructs these
intermediates. The exporter uses `Tracer::off()`: the recipe representation and tracer
constructor are private to the plan module. Stage-local instruction indices are not
source provenance, and a fused instruction does not claim a list of original gates.

Three input families run at widths 8, 10, 12, 14, and 16 under both Host and Device
profiles. An additional Host record runs `diagonal-pair-runs` at width 20:

- `legal-disjoint` places CX gates around a rotation on a disjoint qubit, places 1q
  inverse pairs around independent operations, and spreads rotations across the register.
- `blocked-dependency` places CX between two X operations on its control and includes
  noncommuting rotations along a chain. Sharing a control prevents treating those X
  gates as a disjoint cancellation.
- `diagonal-pair-runs` includes CX-Rz-CX recognition, diagonal runs, and repeated 2q
  blocks with 1q operations on their pair.

The stage planner follows the host's actual `stages_at` decision. That decision depends
on the last-level cache and the equality of logical width and state-buffer width. A
fixture without a `stage-planner` snapshot does not claim planner coverage. The Device
profile describes its compiler IR; all verification here executes on the CPU and says
nothing about GPU launch counts or GPU runtime.

## Numerical contract

Each circuit starts from the same normalized dense complex vector, using the documented
integer formula with seed offset 42. This exercises both components throughout the
entire state, including amplitudes that a zero initial state might never reach. Each
intermediate IR executes directly through `StatevectorBackend::apply`, without another
fusion pass. All amplitudes are compared with unfused execution. The exporter requires
maximum complex absolute error below `1e-12` and norm-squared error below `1e-12`.
Six displayed samples accompany those full-state checks; the samples alone do not
establish equivalence. This is one deterministic input per circuit, not a proof of
operator equivalence for arbitrary inputs.

Matrix payloads are row-major nested arrays of `[real, imaginary]` pairs. A 2q payload
uses the gate matrix convention in which `targets[0]` is the more significant local
index bit. Global state indexing still uses `q[0]` as the least significant bit.
Batch entries retain their native list order. `targets` is the instruction's raw IR
target list; `support` includes every qubit carried inside its payload, sorted and
deduplicated. This distinction matters for a shared-control `BatchPhase`.

## Counter contract

`instruction.kernelSweeps` reads the existing test-only `sweeps::tick` counter after
each native instruction. The counter resets after initialization, so zero fill is
excluded. Each stage's `measuredKernelSweeps` is the sum of those native counts and
must equal `sweeps::sweep_profile(...).passes()` for the same raw IR and pool width.
`modeledFullStatePasses` and `tileApplications` come from that static sweep profile.
The tick sites count kernel route entries on the calling thread; they do not count
worker tasks, memory transactions, cache misses, bytes transferred, or elapsed time.
The only existing-module change is test-only visibility for `take_ticks`.

The generator records CPU, OS, Rust version, compiler flags, thread count, reported
last-level cache capacity, the actual stage-planner floor, source
revision, dirty-tree state, and SHA-256 hashes of the relevant producer sources. No
benchmark runs or throughput claims are part of this export. Source hashes normalize
CRLF to LF. The complete exporter source is embedded because newly added export tooling
does not exist at the recorded base revision. When direct compiler execution is blocked,
the compiler version comes from the completed Cargo build's compiler metadata, with that
origin recorded explicitly in the original fixture. Current generation requires
`rustc --version` and records the actual checkout revision separately from its
audited implementation revision. Complete producer hashes cover all `src` files,
audited files, Cargo manifests, build configuration, and exporter scripts. Tile
and reordering environment overrides, available parallelism, and encoded compiler
flags are retained. Test-only registration is excluded only from the audited
source comparison, never from the full recording fingerprint.

# Native compiler traces

The compiler laboratory reads `fixtures/native-fusion.json`. Generate it with
`node docs/atlas/scripts/generate-fusion-traces.mjs`. The script runs one ignored Rust
unit test with `parallel` enabled and an eight-thread Rayon pool. The test writes
only when `PRISM_ATLAS_FUSION_OUTPUT` names a destination, so ordinary test runs
never touch the fixture.

## Captured evidence

`src/circuit/fusion_atlas_tests.rs` calls the real internal pass functions in the order
used by `fuse_at_width`, `device_passes`, and `batch_passes`. It records the original IR
and the output of each pass. A pass below its width floor is recorded with the
unchanged IR and `enabled: false`. The recursive region pass is listed even though
these circuits contain only gates.

Every final stream is compared with `fuse_circuit_for_profile` gate by gate, including
targets and the full native payload. The browser never recomputes these
intermediates. The exporter uses `Tracer::off()` because the recipe representation
and tracer constructor are private to the plan module. As a result, stage-local
indices do not say which original gates a fused instruction came from.

Three input families run at widths 8, 10, 12, 14, and 16 under both Host and Device
profiles. An additional Host record runs `diagonal-pair-runs` at width 20:

- `legal-disjoint` places CX gates around a rotation on a disjoint qubit, places 1q
  inverse pairs around independent operations, and spreads rotations across the register.
- `blocked-dependency` places CX between two X operations on its control and adds
  noncommuting rotations along a chain. Because the X gates share the CX control,
  they cannot cancel as a disjoint pair.
- `diagonal-pair-runs` includes CX-Rz-CX recognition, diagonal runs, and repeated 2q
  blocks with 1q operations on their pair.

The stage planner follows the host's own `stages_at` decision, which depends on the
last-level cache and on logical width equalling state-buffer width. A record without
a `stage-planner` snapshot did not reach the planner. Device-profile records describe
the compiler IR only. Everything here executes on the CPU, so it says nothing about
GPU launch counts or GPU runtime.

## Numerical contract

Each circuit starts from the same normalized dense complex vector, using the documented
integer formula with seed offset 42, so both components are populated across the
whole state, including amplitudes a zero initial state might never reach. Each
intermediate IR runs directly through `StatevectorBackend::apply`, without another
fusion pass, and every amplitude is compared with unfused execution. The exporter
requires maximum complex absolute error and norm-squared error both below `1e-12`.
The six displayed samples are for reading; the full-state check is what establishes
agreement. One deterministic input per circuit does not prove operator equivalence
for all inputs.

Matrix payloads are row-major nested arrays of `[real, imaginary]` pairs. A 2q payload
uses the gate matrix convention in which `targets[0]` is the more significant local
index bit. Global state indexing still uses `q[0]` as the least significant bit.
Batch entries retain their native list order. `targets` is the instruction's raw IR
target list; `support` includes every qubit carried inside its payload, sorted and
deduplicated. The two differ for a shared-control `BatchPhase`.

## Counter contract

`instruction.kernelSweeps` reads the existing test-only `sweeps::tick` counter after
each native instruction. The counter resets after initialization, so zero fill is
excluded. Each stage's `measuredKernelSweeps` is the sum of those native counts and
must equal `sweeps::sweep_profile(...).passes()` for the same raw IR and pool width.
`modeledFullStatePasses` and `tileApplications` come from that static sweep profile.
The tick sites count kernel route entries on the calling thread, not worker tasks,
memory transactions, cache misses, bytes or elapsed time. The only change to an
existing module is test-only visibility for `take_ticks`.

The generator records CPU, OS, Rust version, compiler flags, thread count, reported
last-level cache size, the stage planner floor, source revision, dirty-tree state,
and SHA-256 hashes of the producer sources. The export runs no benchmark. Source
hashes normalize CRLF to LF. The exporter source is embedded because it does not
exist at the audited base revision. Generation requires `rustc --version` and records
the checkout revision separately from the audited revision. Producer hashes cover
all `src` files, the audited files, Cargo manifests, build configuration and exporter
scripts. Tile and reordering overrides, available parallelism and encoded compiler
flags are recorded too. The test-only registration is left out of the audited source
comparison and kept in the full recording fingerprint.

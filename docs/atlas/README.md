# Inside PRISM-Q: From Quantum Algebra to Silicon

An interactive research atlas of quantum simulation, isolated from the simulator.
Two interactive chapters connect source code to numerical experiments: statevector
gate execution and the fusion compiler. The remaining nine chapters are source
dossiers, not completed interactive laboratories.

![Statevector gate laboratory](public/preview/statevector-desktop.png)

[Interactive atlas](https://abecoull.github.io/prism-q/atlas/) ·
[Five-stage video preview](public/preview/statevector.webm) ·
[Source-evidence matrix](research/source-evidence.md) ·
[Technical limitations](PROVENANCE.md)

![Native compiler laboratory](public/preview/compiler-laboratory.png)

[SIMD microscope](public/preview/execution-microscope.png) ·
[3D native matrix inspector](public/preview/operator-landscape.png) ·
[Rewrite counterexample on mobile](public/preview/rewrite-challenge-mobile.png)

## Run and validate

Node 22.12 or newer is required. Dependencies and local font assets are locked.

```sh
cd docs/atlas
npm ci
npm run dev
```

Open the printed local address. The experiment supports 2 through 16 qubits,
all target bits, seven unitary gates, rotation angle, four repeated steps,
CPU capability assumptions, pair selection, and five manual execution stages.
The displayed pair drives mathematics, indices, memory, hardware, and validation.
The SVG view supports keyboard selection and works without WebGL. The share
button serializes experiment parameters into a reproducible URL.

The execution microscope evaluates individual load, lane-swap, complex-product,
FMA, and store groups, with selectable matrix coefficients and an Argand view.
These are source-derived educational steps, not captured machine instructions.

The compiler laboratory replays 31 native circuit records with 432 pass snapshots.
Select the circuit family, recorded width, Host or Device profile, and pass. Inspect
the actual IR, a conservative dependency graph, native complex matrices, or a 3D
coefficient landscape. Kernel traversal counts, numerical checks, and conditional
traffic models remain separate. On the recording host (8 MiB last-level cache,
planner floor 20 qubits), the 20-qubit Host record reaches the cache stage planner:
the full pipeline takes native kernel traversals from 150 to 8, of which the planner
contributes 41 to 8. A host with a larger cache records a higher floor and keeps the
batching passes at this width. These counts are not timing results. An adversarial rewrite sandbox demonstrates why deleting
two X gates across a CX can preserve normalization while changing the answer.

```sh
npm run evidence
npm test
npm run build
npx playwright install chromium
npm run test:production
npm run test:visual
```

On Windows, browser tests use an installed Microsoft Edge if available, without
requiring a browser download. `npm run evidence` runs from the full simulator
checkout and verifies 65 evidence records, 40 source hashes, 140 native gate cases,
12 sampled 16-qubit native cases, and the compiler fixture's 31 records and 432
snapshots. Source hashes normalize CRLF to LF. Each fixture is checked against the
sources of the commit that last wrote it (the working tree while it is uncommitted),
so it needs full Git history. Later source changes do not fail verification; the
command lists them as files the fixtures predate.

Visual baselines are platform and browser dependent. The reviewed Windows Edge
baselines are committed. Enable comparisons with `ATLAS_VISUAL=1`; generate a
new platform's baselines with `npm run test:visual:update`, inspect every image,
then rerun comparisons. CI runs the functional browser tests on Linux Chromium and
no screenshot comparisons, because only Windows Edge baselines exist.

## Evidence and replay

The browser does not execute native Rust, CUDA, MPI, or SIMD. The default small
example replays actual native PRISM-Q amplitudes and compares them with the browser
model. Native cases cover every target at widths 2 through 6, the seven gates,
and four repeated steps at theta = pi/3. Sampled 16-qubit H, Z, and Rz cases cover
the threaded tile, paired-slice, and diagonal scaling routes; tests compare their
samples with the browser model, and the exporter checks every amplitude against a
scalar reference. Other parameter combinations are labeled educational models. Selecting a CPU architecture changes the instructional route
model, not the ISA of the recorded fixture.

Regenerate native fixtures from the repository root:

```sh
node docs/atlas/scripts/generate-traces.mjs
node docs/atlas/scripts/generate-fusion-traces.mjs
cd docs/atlas
npm run evidence
```

This runs `cargo run --quiet --features parallel --example atlas_trace` with seed
42, records environment metadata, and saves amplitude snapshots. It does not run
a benchmark. The gate exporter uses the existing backend API. The compiler exporter
is an ignored unit test that invokes real internal passes and reads existing
test-only kernel counters. Production kernels are unchanged. Compiler records
execute on the CPU, including records using the Device compiler profile.
See [trace schemas](TRACE-SCHEMA.md) and the [compiler recording contract](research/fusion-traces.md).

Both generators check the audited implementation before running. Recordings retain
the actual checkout revision and dirty-tree flag separately from the audited
revision. Canonical source fingerprints include every file under `src` that Git
tracks or would track (ignored files are excluded), audited files, Cargo manifests,
build configuration, and exporter scripts. For a clean recording, commit the
producers first, regenerate, and commit the fixtures on their own. The audit
comparison excludes only the disclosed test-module registration in `fusion.rs`;
the complete recording fingerprints retain it. A change to audited implementation
requires a new source review before recording. A new commit containing the same
audited implementation does not require pretending that its HEAD is the old audit.

Claims are explicitly tagged as implementation, mathematical theory, analytical
model, empirical measurement, or proposal. This release contains no empirical
timing dataset or proposed optimization shown as implemented behavior.
The numerical reference uses a separately constructed full Kronecker operator,
rather than duplicating the pair iterator.

## Structure and extension

| Directory | Responsibility |
| --- | --- |
| `src/models/` | Pure quantum algebra, address arithmetic, and hardware cost models |
| `src/data/` | Typed evidence, immutable source references, Anatomy geometry |
| `src/renderers/` | React Three Fiber and interactive SVG, independent of arithmetic |
| `src/components/` | Reusable mathematical typesetting |
| `src/chapters/` | Native compiler replay and operator inspection |
| `src/store.ts` | Synchronized experiment state and share URL validation |
| `fixtures/` | Versioned native amplitude snapshots and provenance |
| `tests/` | Independent numerical references, hardware boundaries, browser checks |
| `research/` | Audits, source matrix, design mapping, recording contract |
| `scripts/` | Trace export, source verification, geometry generation, capture |

For a new chapter, audit its source first and add evidence with an immutable
revision, exact source range, claim type, and limitations. Keep numerical models
pure and test against an independently constructed reference or actual native
output. Pass model state into renderers. Add controls that change an algorithmic
quantity, not just a camera. Preserve a keyboard and non-WebGL path.

After a source audit, update its records and revision, then run
`node scripts/update-evidence.mjs` to regenerate the Markdown matrix and source
hash manifest. This command acknowledges the current source as audited; it is
not a substitute for checking the implementation. Native fixtures and evidence
must retain their audited base revision. Both recording families preserve the
actual checkout revision, current producer hashes, and disclosed export changes.
The two fixture families retain separate trace schemas.

SVG paths are generated with the real [Anatomy](https://github.com/wheresryan22/anatomy)
kit at a pinned revision. [Visual design](research/visual-design.md) explains the
geometric mapping, license, and regeneration steps. Manim and Blender assets were
not needed for this chapter's live, numerical mechanisms.

## Deployment and previews

`npm run build` emits a static `dist/` with relative asset paths, suitable for a
subdirectory. The existing documentation workflow builds and validates the atlas,
then copies it into `docs/book/atlas/` before the single GitHub Pages deployment.
This preserves the documentation site at its existing root. Publication on Pages
occurs after these changes are pushed and the workflow succeeds.

With a preview server running, capture screenshots and the manual five-stage
video with `node scripts/capture-preview.mjs http://127.0.0.1:4173`.
Capture the execution microscope, compiler, and operator landscape with
`node scripts/capture-expansion.mjs http://127.0.0.1:4173`.
Playwright's matching ffmpeg helper is needed for video recording. Captures are
visual explanations, not execution-time measurements.

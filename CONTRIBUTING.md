# Contributing to PRISM-Q

## Build

```bash
cargo build                           # default: Rayon parallelism plus faer SVD
cargo build --no-default-features     # single-threaded core
cargo build --features "parallel gpu" # add the optional CUDA backends
cargo build --all-features            # everything; needs MPI and libclang, see below
```

The `gpu` feature builds without a CUDA toolkit. Running it needs a CUDA device, an
NVIDIA driver for CUDA 12.0 or newer, and NVRTC from a CUDA 12 toolkit, all loaded at run
time. NVRTC compiles the kernels to SASS for the device's compute capability, or to PTX
when the NVRTC predates the device.

`rust-toolchain.toml` pins the development toolchain to an exact stable release, which
rustup installs on the first `cargo` call in the checkout. CI uses the same release
except in the MSRV job and the nightly jobs (Miri, ThreadSanitizer, fuzzing, public
API), so lints and codegen move only with the pin. Bump it in its own change: update the
channel and the `dtolnay/rust-toolchain@<version>` steps in `.github/workflows/`, then
fix the new lints. The supported floor is a separate number; see
[Compatibility](docs/guides/compatibility.md).

### From source

```bash
git clone https://github.com/AbeCoull/prism-q.git
cd prism-q
cargo build --release --features parallel
```

To pin a downstream crate to a specific revision:

```bash
cargo add prism-q --git https://github.com/AbeCoull/prism-q --features parallel
```

## Test and lint

```bash
cargo nextest run --features "parallel gpu distributed"
cargo test --doc --features "parallel gpu distributed"
cargo fmt --check
cargo clippy --all-targets --features "parallel gpu distributed" -- -D warnings -D clippy::undocumented_unsafe_blocks
cargo doc --no-deps --features "parallel gpu distributed"
```

These run on any host: the `gpu` and `distributed` features build without a CUDA toolkit
or MPI.

Before opening a pull request, run `scripts/pre_pr.sh`. It runs the gate at `parallel`,
then the CI checks the branch's diff against `origin/main` reaches: the doc build, the
public API snapshot, semver, the Python parity manifest, `--no-default-features` and
cross-target clippy. `--plan` lists what it would run and why.

`--all-features` adds `distributed-mpi`, which needs a system MPI installation and
libclang for its bindgen step. Run it when a change touches the MPI surface or the
`Cargo.toml` feature wiring. On Windows, dot-source `scripts/mpi-env.ps1` first
(`. scripts\mpi-env.ps1`) to load MSVC, MS-MPI, and libclang into the PowerShell session:

```bash
cargo nextest run --all-features
cargo clippy --all-targets --all-features -- -D warnings -D clippy::undocumented_unsafe_blocks
```

Use `cargo test` with the same features when `cargo-nextest` is not installed. Keep
doctests on `cargo test --doc` until nextest doctest support is no longer experimental.

Integration tests are modules of one binary, `tests/integration/main.rs`: add a file
there and declare it with `mod`, then select it with a filter such as
`cargo nextest run --features parallel -E 'test(/^backend_matrix::/)'`. A file that sets
an environment variable, builds the global Rayon pool, or calls `set_default_limits`
stays its own `tests/<name>.rs` binary, since `cargo test` runs a binary's tests in one
process.

The GPU golden tests (`--test golden_gpu --test golden_gpu_density_matrix` under
`--features "parallel gpu"`) skip without a CUDA device, so a green run on a host
without a card means untested, and no CI job opens a device. Before merging a change to
`src/gpu/`, the gate set, the fusion pipeline, or a kernel table shape, run
`scripts/test-gpu.ps1` on a host with a card; it sets `PRISM_REQUIRE_GPU=1` so a missing
device fails the run.

## Coverage

```bash
# requires: rustup component add llvm-tools-preview && cargo install cargo-llvm-cov
cargo llvm-cov nextest --features parallel                 # terminal summary, as CI measures it
cargo llvm-cov nextest --features parallel --html --open   # browseable HTML report
```

CI measures coverage at `--features parallel`. `--all-features` adds the MPI transport,
with the MPI and libclang requirements above.

## Fuzzing

`fuzz/` holds `cargo fuzz` targets for the OpenQASM front end (`qasm_parse` on raw
bytes, `qasm_tokens` on programs assembled from QASM fragments), which must never panic,
and for fusion (`fusion`), which checks a fused statevector run against a plain apply
loop to 1e-10 per amplitude. The crate is its own workspace and needs nightly:

```bash
# requires: cargo install cargo-fuzz
cd fuzz
mkdir -p corpus/qasm_parse
cargo +nightly fuzz run qasm_parse corpus/qasm_parse seeds/qasm_parse -- -dict=qasm.dict
```

On Windows MSVC the AddressSanitizer runtime (`clang_rt.asan_dynamic-x86_64.dll`, in the
MSVC `bin\Hostx64\x64` directory) must be on `PATH`.
`tests/integration/fuzz_replay.rs` replays `fuzz/seeds/` and a seeded sweep through the
same checks on stable, and the `Fuzz` workflow runs each target weekly and uploads any
crash input.

## Documentation site

`docs/` publishes as an mdBook site to GitHub Pages through `.github/workflows/docs.yml`.
Preview locally:

```bash
cargo install mdbook   # once
mdbook serve docs      # serves at http://localhost:3000
```

`docs/SUMMARY.md` lists the pages, and rendered output lands in `docs/book/`
(gitignored). The workflow generates `sitemap.xml` from the built HTML. Publishing needs
the repository's Pages source set to "GitHub Actions" once, under Settings > Pages.

## Benchmarks

```bash
cargo bench --bench circuits --features parallel        # circuit macrobenchmarks
cargo bench --bench bench_driver --features parallel    # gate microbenchmarks
```

Always use `--features parallel`, since baselines were taken with Rayon enabled. Run one
`cargo bench` process at a time; competing Rayon pools add noise.

### Profiling

```bash
./scripts/flamegraph.sh "qft_textbook/16"     # unix, needs `cargo install flamegraph`
.\scripts\flamegraph.ps1 "qft_textbook/16"    # windows
```

The script header lists the platform profiler it needs and where the SVG lands.

### Regression checks

Numbers held to the 5% gate come from the adjacent-binary A/B. It builds both bench
binaries, checks the working tree did not change between the builds, runs them back to
back, and reports each row against its own same-code control:

```bash
./scripts/bench_ab.sh --filter '^factored/noise_kraus/' --ref main
```

Two `cargo bench` runs minutes apart are not an A/B on a development host: the rebuild
lands between them, and a byte-identical control group has read as much as +98%. Report
every change with its control column, and say when a delta sits inside the noise floor.
`benches/README.md` has the method and the rows this cannot resolve.

Stored baselines still serve for tracking a number over weeks and for the CI gate,
where both sides run in one job:

```bash
cargo bench --features parallel
./scripts/bench_check.sh save --name "before"

cargo bench --features parallel
./scripts/bench_check.sh compare --baseline "before"
./scripts/bench_check.sh table --baseline "before"
```

`compare` exits nonzero on regression and `table` emits a markdown summary for the PR
description. Both need `jq` and `bc`; `bench_ab.sh` needs neither.

## PR guidelines

- Include before/after benchmark numbers for performance-sensitive changes.
- Tests, clippy, fmt and the doc build all pass.
- Fixed seeds: `42` for tests, `0xDEAD_BEEF` for benchmark circuits.
- The pull request template at `.github/PULL_REQUEST_TEMPLATE.md` captures the required
  checklist.

## Releases

`.github/workflows/release.yml` runs on manual dispatch. It reads the conventional
commits since the last tag, resolves a bump level, and hands that level to
`cargo release --no-confirm --execute`, which publishes to crates.io.

The level comes from `scripts/release_bump_level.sh` in two stages:

| Commits since the last tag | Conventional level | Level used while the crate is 0.x |
| --- | --- | --- |
| `type!:` subject, or a `BREAKING CHANGE:` body | major | **minor** |
| `feat:` | minor | minor |
| `fix:` or `perf:` | patch | patch |
| anything else | none | none |

The second column is the specification; the third is what ships below 1.0. Cargo reads
`0.27.0` as `^0.27.0`, so under 0.x the minor position is already the compatibility
boundary: `0.27` to `0.28` signals a break to every downstream caret requirement.
`cargo release major` on 0.x would resolve to 1.0.0, which crates.io allows yanking but
never unpublishing, and which claims an API stability the crate has not reached. The
clamp reads the major version from `Cargo.toml`, so a deliberate 1.0.0 restores the
major path with no edit to the script.

`scripts/release_bump_level_test.sh` checks the mapping against synthetic commit lists,
asserting the level and the resulting version (a `feat!:` commit on 0.27.0 resolves to
`minor` and publishes 0.28.0). The `release-bump-level` CI job runs it on every pull
request, and the release workflow runs it again before reading the commit range.

The workflow never reaches 1.0.0 on its own: set that version in `Cargo.toml` by hand
and release from there.

The wheel follows in the same run: the workflow writes the new version into
`bindings/python/Cargo.toml` and tags it `py-v<version>`, so the two never drift. What a
version promises, which feature flags are part of the surface, and the MSRV pin are in
[Compatibility](docs/guides/compatibility.md).

## CI

PRs run formatting, clippy, nextest, doctests, doc build, coverage, the release
bump-level check, aarch64 cross-compile, macOS ARM64 tests, and `cargo-deny`
(security advisories and license audit). The GPU job compiles the `gpu` feature without
a CUDA toolkit and runs the kernel name registry test, which needs no device; no CI job
opens a device, so `scripts/test-gpu.ps1` covers the device suites on a host with a card.

## Hot-path rules

- No heap allocation in gate-application inner loops.
- Enum dispatch only in gate kernels. No trait objects.
- `// SAFETY:` comment on all `unsafe` blocks (see Comments and documentation).

## Comments and documentation

Whether to write a comment and how to write it are separate questions. The rules below
settle the first. The file being edited settles the second: match its comment density
and voice in whatever does get written.

### Inline comments

- Avoid inline comments. Code should carry its meaning through naming and structure.
  Two exceptions: `// SAFETY:` comments (required, see below) and non-obvious
  algorithmic reasoning that naming cannot express. When in doubt, leave the comment
  out.
- No comment that restates the line it sits above.
- No TODO, FIXME, HACK, or XXX markers, and no tracker IDs. A finished change contains
  no stubs. Record deferred work in an issue or in the PR description, not in the code.

### Docstrings

- A docstring must add a fact the name, signature, and types do not already carry: a
  unit, an ordering or packing convention, a default, an invariant, a
  cross-reference, a contract. If no such fact exists, write no docstring. Restating
  the signature is noise.
- One line is the norm. More than three lines is reserved for real contract or
  algorithm prose (a trait's call-order contract, a backend's applicability). Never
  pad to look thorough, and never document to hit coverage.
- Trivial self-describing items (`into_x`, `len`, `is_empty`, obvious constructors,
  fields whose name says everything) get nothing.
- A `pub` item that is not meant as public API gets demoted to `pub(crate)` instead of
  documented. Reduce visibility before writing docs nobody should read.
- Document private items only when they are complex.
- No `///` on `#[test]` functions. The test name or a plain comment carries any
  explanation.

### Docstring format

- First line: one sentence ending in a period. Noun phrase for types, traits, and
  modules ("Top-level error type for PRISM-Q operations."). Imperative for functions
  ("Append a gate operation.").
- Wrap prose at roughly 90 columns, matching the surrounding file.
- Use intra-doc links for crate items and drop generic arguments from the link target:
  link `DistributedContext` behind its `Arc`, not `Arc<DistributedContext>` as a whole.
  Items from dependency crates stay plain code spans.
- `# Panics`: when a family of methods shares one panic condition (builder methods
  asserting index bounds), state it once at the type level. A per-method `# Panics`
  is for a condition that is unique to the method and not obvious from it.
- `# Errors`: only when the failure conditions are not evident from the error
  variants returned.
- `# Safety` on every fully `pub` unsafe fn (clippy `missing_safety_doc` under
  `-D warnings` enforces this), and on any unsafe fn called from outside its defining
  module whose preconditions the caller must uphold: pointer validity, aliasing, index
  disjointness. A kernel whose only precondition is a CPU feature check needs no
  `# Safety` section; the call-site `// SAFETY:` comment carries it.
- `# Examples` with a runnable doctest on top-level entry points only (`run_qasm`,
  `simulate`, `CircuitBuilder`, `run_qec_program` and peers). Keep runnable doctests
  out of feature-gated modules: doctests link, so a doctest behind `gpu` needs a CUDA
  host, and CI runs doctests with `parallel` and with `--no-default-features` only.
  Use `no_run` or plain text blocks there.
- Panic policy: invalid user input (QASM text, incompatible backend) returns
  `PrismError`. API misuse (out-of-range indices, wrong-variant accessors) panics and
  is documented under `# Panics`. Do not wrap infallible paths in `Result`.

### Module docs

- Every file whose items are publicly reachable opens with a `//!` block of one to
  three lines saying what the module holds. No implementation narration in module
  headers.
- Backend modules follow the standard template after the summary paragraph:
  `# Memory layout`, `# Gate support`, `# When to prefer this backend`,
  `# When NOT to use this backend`. Each section can be a sentence or a short list.

### SAFETY comments

- Every `unsafe` block carries `// SAFETY:` stating the invariant that makes it sound.
  clippy `undocumented_unsafe_blocks` under `-D warnings` enforces presence; review
  enforces substance.
- New unsafe code outside the established kernel patterns (SIMD dispatch, `SendPtr`
  parallelism, GPU launches) also states why the safe alternative is insufficient,
  with a measured number when the justification is performance.
- Inside an `unsafe fn`, a block that merely discharges the function's own contract
  uses exactly: `// SAFETY: same contract as the enclosing unsafe fn.` Do not restate
  the contract and do not coin variants.
- Recurring facts use the canonical stem verbatim, extended with the site-specific
  clause where applicable: `// SAFETY: NEON is baseline on aarch64`;
  `// SAFETY: AVX2 detected` when the runtime dispatch happened upstream;
  `// SAFETY: AVX2 checked above` (naming exactly what the guard checks) when the
  feature check is visible in the same function.
- Do not normalize existing phrasings in unrelated diffs. Align a line only when the
  change already touches it.

## Adding a backend

1. Create `src/backend/<name>.rs` (or a directory module) and implement the `Backend`
   trait, plus the `backend::sealed::Sealed` marker that keeps it crate-only.
2. Add `pub mod <name>;` to `src/backend/mod.rs`.
3. Write unit tests (single-qubit, two-qubit, measurement at minimum) and golden tests
   against the statevector backend.
4. Add benchmark entries in `benches/circuits.rs`.
5. Update `docs/architecture/engine.md` with the backend's position in the dispatch tree,
   and `docs/architecture/backends.md` with its description.

## Questions

Open an issue or start a discussion on the repo.

# Contributing to PRISM-Q

## Build

```bash
cargo build                           # core (no parallelism)
cargo build --features parallel       # Rayon parallelism plus faer SVD
cargo build --features "parallel gpu" # add the optional CUDA statevector backend
cargo build --all-features            # everything
```

The `gpu` feature requires the CUDA toolkit (12.x or newer) and a CUDA capable device.
PTX is compiled at runtime via NVRTC against the device's compute capability.

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
cargo nextest run --all-features
cargo test --doc --all-features
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings -D clippy::undocumented_unsafe_blocks
cargo doc --no-deps --features "parallel gpu distributed"
```

The doc build covers the `gpu` and `distributed` surfaces on any host: rustdoc compiles
but never links, so no CUDA toolkit is needed. An `--all-features` doc build additionally
requires an MPI installation (the `mpi` crate probes for one in its build script).

Use `cargo test --all-features` when `cargo-nextest` is not installed. Keep doctests on
`cargo test --doc` until nextest doctest support is no longer experimental.

GPU golden tests run under `cargo test --features "parallel gpu" --test golden_gpu` and
skip automatically when no CUDA device is present.

## Coverage

```bash
# requires: rustup component add llvm-tools-preview && cargo install cargo-llvm-cov
cargo llvm-cov --all-features                     # terminal summary
cargo llvm-cov --all-features --html --open       # browseable HTML report
```

## Documentation site

The architecture guide and glossary in `docs/` publish as an mdBook site to GitHub Pages
via `.github/workflows/docs.yml`. Preview locally:

```bash
cargo install mdbook   # once
mdbook serve docs      # serves at http://localhost:3000
```

The book is rooted at `docs/` (`docs/book.toml`); `docs/SUMMARY.md` lists the pages and
rendered output lands in `docs/book/` (gitignored). Publishing requires the repository
Pages source set to "GitHub Actions" once under Settings > Pages.

The workflow generates `sitemap.xml` from the built HTML, so pages in `SUMMARY.md` are
indexed automatically.

## Benchmarks

```bash
cargo bench --bench circuits --features parallel        # circuit macrobenchmarks
cargo bench --bench bench_driver --features parallel    # gate microbenchmarks
```

Always use `--features parallel`. Baselines were taken with Rayon enabled. Do not run
multiple `cargo bench` processes at once. Rayon contention causes noisy results.

### Regression checks

Before/after numbers that have to hold to the 5% gate go through the
adjacent-binary A/B. It builds both bench binaries first, verifies the working tree
did not move between the two builds, then runs them adjacent with no rebuild in
between, and reports every row against its own same-code control:

```bash
./scripts/bench_ab.sh --filter '^factored/noise_kraus/' --ref main
```

Separate `cargo bench` invocations minutes apart are not a valid A/B on a
development host: the rebuild lands between them and a byte-identical control group
has read as much as +98%. Do not report a change without its control column, and
say so plainly when a delta lands inside the noise floor. `benches/README.md` has
the method and the list of rows this cannot resolve.

Stored baselines remain useful for tracking a number over weeks and for the CI
gate, where both sides run in one job:

```bash
cargo bench --features parallel
./scripts/bench_check.sh save --name "before"

cargo bench --features parallel
./scripts/bench_check.sh compare --baseline "before"
./scripts/bench_check.sh table --baseline "before"
```

`compare` exits non zero on regression. `table` emits a markdown summary for the PR
description. Both need `jq` and `bc`; `bench_ab.sh` needs neither.

## PR guidelines

- Include before/after benchmark numbers for performance-sensitive changes.
- All tests pass, clippy clean, fmt clean, doc build clean.
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

The second column is the specification. The third is what actually ships below
1.0, and the difference is deliberate. Cargo reads `0.27.0` as `^0.27.0`, so under
0.x the minor position is the compatibility boundary: `0.27` to `0.28` already
signals a break to every downstream caret requirement. `cargo release major` on a
0.x version resolves to 1.0.0 instead, which crates.io permits yanking but never
unpublishing, and which claims an API stability the crate has not reached.

The clamp is gated on the major version read from `Cargo.toml`, not hardcoded, so
a deliberate 1.0.0 restores the major path with no edit to the script.

`scripts/release_bump_level_test.sh` drives the mapping against synthetic commit
lists, asserting both the level and the version it produces (a `feat!:` commit on
0.27.0 resolves to `minor` and publishes 0.28.0). The `release-bump-level` CI job
runs it on every pull request, and the release workflow runs it again before
reading the commit range.

To cut 1.0.0, set the version in `Cargo.toml` by hand and release from there. The
workflow will not reach it on its own.

## CI

PRs run formatting, clippy, nextest, doctests, doc build, coverage, the release
bump-level check, aarch64 cross-compile, macOS ARM64 tests, and `cargo-deny`
(security advisories plus license audit).

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
  Two exceptions: `// SAFETY:` comments (required, see below) and genuinely non-obvious
  algorithmic reasoning that naming cannot express. When in doubt, leave the comment
  out.
- No comment that restates the line it sits above.
- No TODO, FIXME, HACK, or XXX markers, and no tracker IDs. A finished change contains
  no stubs. Record deferred work in an issue or in the PR description, not in the code.

### Docstrings

- A docstring must add a fact the name, signature, and types do not already carry: a
  unit, an ordering or packing convention, a default, an invariant, a
  cross-reference, a contract. If no such fact exists, write no docstring. Restating
  the signature is noise, the same defect as a comment restating its line.
- One line is the norm. More than three lines is reserved for real contract or
  algorithm prose (a trait's call-order contract, a backend's applicability). Never
  pad to look thorough, and never document to hit coverage.
- Trivial self-describing items (`into_x`, `len`, `is_empty`, obvious constructors,
  fields whose name says everything) get nothing.
- A `pub` item that is not meant as public API gets demoted to `pub(crate)` instead of
  documented. Reduce visibility before writing docs nobody should read.
- Document private items only when they are genuinely complex.
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
   trait.
2. Add `pub mod <name>;` to `src/backend/mod.rs`.
3. Write unit tests (single-qubit, two-qubit, measurement at minimum) and golden tests
   against the statevector backend.
4. Add benchmark entries in `benches/circuits.rs`.
5. Update `docs/architecture/engine.md` with the backend's position in the dispatch tree,
   and `docs/architecture/backends.md` with its description.

## Questions

Open an issue or start a discussion on the repo.

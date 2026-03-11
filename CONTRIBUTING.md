# Contributing to PRISM-Q

## Build

```bash
cargo build                       # core (no parallelism)
cargo build --features parallel   # Rayon parallelism + faer SVD
cargo build --all-features        # everything
```

## Test & lint

```bash
cargo test --all-features
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features
```

## Coverage

```bash
# requires: rustup component add llvm-tools-preview && cargo install cargo-llvm-cov
cargo llvm-cov --all-features                     # terminal summary
cargo llvm-cov --all-features --html --open       # browseable HTML report
```

## Benchmarks

```bash
cargo bench --bench circuits --features parallel        # circuit macrobenchmarks
cargo bench --bench bench_driver --features parallel    # gate microbenchmarks
```

Always use `--features parallel` — baselines were taken with Rayon enabled. Don't run multiple bench processes at once (Rayon contention causes noisy results).

## PR guidelines

- Include before/after benchmark numbers for performance-sensitive changes.
- All tests pass, clippy clean, fmt clean, doc build clean.
- Fixed seeds: `42` for tests, `0xDEAD_BEEF` for benchmark circuits.

## CI

PRs run: formatting, clippy, tests, doc build, coverage, aarch64 cross-compile, macOS ARM64 tests, and `cargo-deny` (security advisories + license audit).

## Hot-path rules

- No heap allocation in gate-application inner loops.
- Enum dispatch only in gate kernels (no trait objects).
- `// SAFETY:` comment on all `unsafe` blocks.

## Adding a backend

1. Create `src/backend/<name>.rs` and implement the `Backend` trait.
2. Add `pub mod <name>;` to `src/backend/mod.rs`.
3. Write unit tests (single-qubit, two-qubit, measurement at minimum) and golden tests against the statevector backend.
4. Add benchmark entries in `benches/circuits.rs`.

## Questions?

Open an issue or start a discussion on the repo.

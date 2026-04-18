# Contributing to PRISM-Q

## Build

```bash
cargo build                           # core (no parallelism)
cargo build --features parallel       # Rayon parallelism plus faer SVD
cargo build --features "parallel gpu" # add the optional CUDA statevector backend
cargo build --all-features            # everything
```

The `gpu` feature requires the CUDA toolkit (12.x or newer) and a CUDA-capable device.
PTX is compiled at runtime via NVRTC against the device's compute capability.

## Test and lint

```bash
cargo test --all-features
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features
```

GPU golden tests run under `cargo test --features "parallel gpu" --test golden_gpu` and
skip automatically when no CUDA device is present.

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

Always use `--features parallel`. Baselines were taken with Rayon enabled. Do not run
multiple `cargo bench` processes at once. Rayon contention causes noisy results.

## PR guidelines

- Include before/after benchmark numbers for performance-sensitive changes.
- All tests pass, clippy clean, fmt clean, doc build clean.
- Fixed seeds: `42` for tests, `0xDEAD_BEEF` for benchmark circuits.
- The pull request template at `.github/PULL_REQUEST_TEMPLATE.md` captures the required
  checklist.

## CI

PRs run formatting, clippy, tests, doc build, coverage, aarch64 cross-compile,
macOS ARM64 tests, and `cargo-deny` (security advisories plus license audit).

## Hot-path rules

- No heap allocation in gate-application inner loops.
- Enum dispatch only in gate kernels. No trait objects.
- `// SAFETY:` comment on all `unsafe` blocks.

## Adding a backend

1. Create `src/backend/<name>.rs` (or a directory module) and implement the `Backend`
   trait.
2. Add `pub mod <name>;` to `src/backend/mod.rs`.
3. Write unit tests (single-qubit, two-qubit, measurement at minimum) and golden tests
   against the statevector backend.
4. Add benchmark entries in `benches/circuits.rs`.
5. Update `docs/architecture.md` with the backend's position in the dispatch tree.

## Questions

Open an issue or start a discussion on the repo.

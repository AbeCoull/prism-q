# Installation

PRISM-Q is a Rust library. Add it to a project with Cargo:

```bash
cargo add prism-q                          # Rayon parallelism + faer SVD (default)
cargo add prism-q --no-default-features    # single-threaded, minimal dependencies
```

Or add it to `Cargo.toml` directly:

```toml
[dependencies]
prism-q = "0.16"
```

## Feature flags

| Feature | Default | Enables |
|---------|---------|---------|
| `parallel` | yes | Rayon parallel kernels (≥14 qubits) and the faer SVD path for MPS |
| `serialization` | no | `serde` derives for circuits and results |
| `gpu` | no | Optional CUDA backend (see the [GPU guide](../guides/gpu.md)) |

```admonish tip title="Keep parallel on for performance"
The published benchmarks were taken with `parallel` enabled. Without it, 16+ qubit runs
fall back to single-threaded kernels and are not comparable to the baselines. Disable it
only when you need a minimal-dependency, single-threaded build.
```

## Building from source

```bash
git clone https://github.com/AbeCoull/prism-q
cd prism-q
cargo build --release
```

## Running the test suite

```bash
cargo nextest run --all-features                          # unit + integration tests
cargo test --doc --all-features                           # doctests
cargo clippy --all-targets --all-features -- -D warnings  # lint
```

Use `cargo test --all-features` if `cargo-nextest` is unavailable.

Next: build [Your First Circuit](./first-circuit.md).

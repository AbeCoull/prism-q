# Installation

```bash
cargo add prism-q                          # Rayon parallelism + faer SVD (default)
cargo add prism-q --no-default-features    # single-threaded, minimal dependencies
pip install prism-q                        # Python bindings
```

`cargo add` writes the current release into `Cargo.toml`; to pin a version by hand, take
it from [crates.io](https://crates.io/crates/prism-q). The Python package is covered in
[Python Bindings](../guides/python.md).

## Feature flags

| Feature | Default | Enables |
|---------|---------|---------|
| `parallel` | yes | Rayon parallel kernels (≥15 qubits) and the faer SVD path for MPS |
| `gpu` | no | Optional CUDA backend (see the [GPU guide](../guides/gpu.md)) |
| `distributed` | no | Statevector partitioning across ranks (see [Capabilities](../guides/capabilities.md)) |
| `distributed-mpi` | no | `distributed` plus the MPI transport |

```admonish tip title="Keep parallel on for performance"
The published benchmarks were taken with `parallel` enabled. Without it, 16+ qubit runs
use single-threaded kernels and are not comparable to the baselines.
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

Use `cargo test --all-features` if `cargo-nextest` is unavailable. `--all-features`
includes `distributed-mpi`, which needs a system MPI installation and libclang for its
bindgen step; without those, name the features instead, for example
`--features "parallel gpu"`.

Next: build [Your First Circuit](./first-circuit.md).

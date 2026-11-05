# Installation

```bash
cargo add prism-q                          # Rayon parallelism + faer SVD (default)
cargo add prism-q --no-default-features    # single-threaded, minimal dependencies
pip install prism-q                        # Python bindings
```

`cargo add` writes the current release into `Cargo.toml`; to pin a version by hand, take
it from [crates.io](https://crates.io/crates/prism-q).

## Python

PyPI carries `abi3` wheels for Python 3.11 and newer on Linux x86_64 and aarch64
(manylinux), macOS arm64 and Windows x64. Other platforms build from the source
distribution, which needs a Rust toolchain. The Linux and Windows wheels carry the CUDA
paths, which load the NVIDIA driver and NVRTC at run time; the `cuda12` extra installs
NVRTC from PyPI:

```bash
pip install "prism-q[cuda12]"
```

Without a driver or NVRTC the package runs on the CPU and only `GpuContext(...)` raises.
The package is covered in [Python Bindings](../guides/python.md).

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
cargo nextest run --features parallel                          # unit + integration tests
cargo test --doc --features parallel                           # doctests
cargo clippy --all-targets --features parallel -- -D warnings  # lint
```

Use `cargo test --features parallel` if `cargo-nextest` is unavailable. These need only
a Rust toolchain. The GPU, distributed and MPI arms are covered in the
[contributing guide](https://github.com/AbeCoull/prism-q/blob/main/CONTRIBUTING.md#test-and-lint);
its `--all-features` runs add `distributed-mpi`, which needs a system MPI installation
and libclang.

Next: build [Your First Circuit](./first-circuit.md).

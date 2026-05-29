# PRISM-Q

PRISM-Q is a Rust quantum circuit simulator built for speed. It dispatches across
multiple specialized backends, runs a multi pass fusion pipeline, and uses AVX2, FMA,
and BMI2 SIMD kernels in the inner loop. CPU kernels are the default path, with
optional CUDA support for statevector and experimental stabilizer workloads. Input is
OpenQASM 3.0 with backward compatible 2.0 syntax.

## Where to go next

- [Architecture](./architecture.md): backends, dispatch tree, fusion pipeline, and SIMD strategy.
- [Glossary](./glossary.md): terminology used across the documentation.
- [API reference](https://docs.rs/prism-q): generated Rust API documentation.
- [Source and issues](https://github.com/AbeCoull/prism-q): repository on GitHub.

## Install

```bash
cargo add prism-q                         # Rayon parallelism + faer SVD (default)
cargo add prism-q --no-default-features    # single-threaded, minimal dependencies
```

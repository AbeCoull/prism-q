# Compatibility and Versioning

## Rust version

The minimum supported Rust version is 1.89.0, pinned in three places updated together:
`rust-version` in `Cargo.toml`, `msrv` in `clippy.toml`, and the `CI_MSRV` variable the
`msrv` CI job builds against. That job checks the library with `parallel`, `gpu` and
`distributed`. `distributed-mpi` needs a system MPI to build and is not checked at the
floor. Raising the MSRV is a minor bump while the crate is below 1.0. The development
toolchain is separate: `rust-toolchain.toml` pins an exact stable release (see
[Contributing](https://github.com/AbeCoull/prism-q/blob/main/CONTRIBUTING.md#build)).

## Feature flags

Four feature flags are part of the surface: `parallel`, on by default, and `gpu`,
`distributed` and `distributed-mpi`, each off. `bench-fast` and `bench-internal` are not.
They shape benchmark runs, gate items no caller should reach for, and may change or
disappear in any release. `unstable-pulse` is not part of it either: the `pulse` module
it enables may change or disappear in any release.

## Versions

The crate and the Python wheel carry one version. A release bumps the crate, writes the
same version into the bindings manifest and tags the wheel `py-v<version>`, so a wheel and
a crate with the same number were built from one commit. Below 1.0 the minor position is
the compatibility boundary. Cargo reads `0.32.0` as `^0.32.0`, so `0.32` to `0.33` already
signals a break to every caret requirement, and the release tooling resolves a breaking
change to a minor bump rather than 1.0.0.

A version promises the surface tabulated in [API surface](../architecture/api-surface.md)
and the Python package. Which backend `Auto` picks for a circuit, where the fusion
thresholds sit, how fast a kernel runs and what the benchmark rows are named are
implementation, and they move without a bump. Public enums are `#[non_exhaustive]` apart
from a few that mirror a closed set, which that page names, so a new variant is additive
and a `match` outside the crate keeps a wildcard arm. The `Backend` trait is sealed, so
only the crate implements it and a new method is additive. Items hidden from the rendered
API docs, such as `prism_q::circuit::fusion`, are reachable for tests and tooling but are
not part of the surface.

## Seeded output

A seed reproduces a result within a version, under the thread-count rules each path
states in [Determinism](../architecture/threading-simd.md#determinism). Across versions it
does not: a minor release may change what a seed draws, and its release notes say so. Such
a change moves which bits a seed lands on, not the distribution they come from; the
distribution changes only to fix a bug.

Seeded output has changed in these releases:

- 0.33.0 seeds shot `i` with a SplitMix64 hash of the run seed and `i` instead of
  `seed + i`. That covers per-shot replay of circuits with mid-circuit measurements, the
  distributed per-shot loop, noise trajectories (serial and parallel, noisy stabilizer
  trajectories included), the block seeds of a decomposed run, and the per-qubit and
  per-observable seeds of Stochastic Pauli Propagation. Fixed-seed results on those routes
  changed, and on a replay route `shots(1)` with seed `x` no longer reproduces `run()`
  with seed `x`.
- 0.35.0 changes three sampling routes and fixes a fourth. Product-state and factored
  terminal shots and counts draw in blocks of 256 shots, one stream per block, so they
  fill in parallel; the dense sampler over factored block probabilities, which also serves
  decomposed terminal shots, draws the same blocks so the two still agree shot for shot.
  Noiseless circuits with mid-circuit measurements on the host statevector evolve once per
  outcome history and split shots binomially, so a shot no longer corresponds to a run on
  its own seed; once only measurements remain, each branch draws its readout from its own
  state in one pass. Dynamic programs on the host statevector branch the same way, so
  their shots no longer follow `mix_seed(seed, i)`. Compiled Clifford counts at high shot
  counts draw from a corrected binomial sampler, which also fixes their spread.

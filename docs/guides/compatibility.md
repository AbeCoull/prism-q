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

## Changed results

A breaking release can keep a public field and change what it reports. These fields
changed meaning:

- 0.39.0: `run_stabilizer_rank_approx` runs on the Clifford register and changes three
  fields of its `StabRankResult`. `discarded_weight` is the sum over cuts of the exact
  2-norm each cut dropped, where it was the summed 1-norm of the dropped branch weights; a
  caller's `4 * discarded_weight` remains a valid bound on the summed absolute difference
  from the exact distribution. `num_terms` and `pruned_count` count register
  coefficients, not branches. The new field `kept_norm_sqr` carries the kept norm, and
  `fidelity_bound()` uses both. A register wider than 20 qubits now returns
  `BackendUnsupported`, and `seed` is ignored.
- 0.39.0: `run_stabilizer_rank`, and `run()` on `BackendKind::StabilizerRank`, build
  the Clifford register up to 24 qubits wide, where the cap was 20. A circuit whose
  register is 21 to 24 qubits wide and carries more than 20 T gates now returns its
  probabilities, where it returned `BackendUnsupported` for too many terms.
  `run_stabilizer_rank_approx` keeps the 20 qubit cap.
- 0.39.0: `RunMetadata::exactness` on an MPS run reports the run instead of the route.
  A run whose cuts discarded a summed squared weight of at most 1e-24 is `Exact` at any
  bond cap, where any chain wider than bond 1 was `Approximate` with a bound of 1.0.
  Python's `is_exact` follows it. `require_exact` still declines the MPS route before the
  run.

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
- 0.39.0: these routes draw differently:
  - Shots from the Clifford register draw differently on every circuit that route
    serves: `Auto` and explicit `StabilizerRank` terminal shots of a Clifford+T circuit,
    and circuits whose mid-circuit measurements move to the end. The register is split
    into independent parts, and shots are drawn in blocks of 256 with one ChaCha8 stream
    per block. More circuits reach the register: one wider than 20 qubits qualifies when
    each part fits 24 qubits or divides into two sides of at most 20, up to 64 register
    qubits in all, and those circuits move from the statevector, or past the statevector
    cap from the MPS. Fixed-seed shots on all of these circuits changed. Probabilities
    from `run_stabilizer_rank` are unchanged.
  - The Clifford register reads the outcome table of a part it holds whole as the Walsh
    transform of the expectation values of the part's measured rows, wherever that costs
    less than diagonalizing the part's amplitudes. The two tables agree to about 1e-15, so
    a fixed-seed shot changes only where its draw lands that close to the boundary between
    two outcomes. The cheaper readout also moves `Auto` terminal shots of some Clifford+T
    circuits from the statevector to the register, those whose register is nearly as wide
    as the circuit, and fixed-seed shots on those circuits changed.
  - The Clifford register holds a part of 25 or 26 qubits whole, where the widest whole
    part was 24, when the part has no cut into two sides of at most 20 across at most 12
    row pairs. Those circuits move to the register: `Auto` terminal shots from the
    statevector, or past the statevector cap from the MPS, and explicit `StabilizerRank`
    shots return where they returned `BackendUnsupported`. Fixed-seed shots on those
    circuits changed. A part of 25 or 26 qubits with such a cut keeps it and its shots.
    The amplitudes summed over the parts are held to the process statevector cap, so
    under a cap below 25 a register past it declines where it was taken.
  - `Auto` terminal shots of a Clifford+T circuit are priced between the Clifford
    register and the dense run instead of capping the T count. Circuits with more than 20
    T gates, or at least as many T gates as qubits, whose register fits 20 qubits move to
    the register from the statevector, or past the statevector cap from the MPS. A
    register nearly as wide as the circuit, or one wider than the independent blocks the
    circuit splits into, moves the other way. Fixed-seed shots on those circuits changed.
  - `Auto` shots of a Clifford+T circuit with a mid-circuit measurement, a guard or a
    reset sample from the Clifford register once its measurements move to the end, in
    place of the weighted MPS branches, and explicit `StabilizerRank` does the same.
    Within the statevector cap, a circuit the register does not take runs on the
    statevector walk; past the cap, one whose measurements stay in place keeps the
    branches up to 6 T gates and goes to the dispatch tree beyond. Fixed-seed shots on
    those circuits changed.
  - `Auto` shots of a circuit that reaches the mid-circuit walk drop parity ancillas. An
    ancilla that only collects CX gates from data qubits and X flips, is measured, and is
    then reset or left alone becomes a parity measurement on those data qubits, so the
    walk runs on a smaller register. Fixed-seed shots on those circuits changed;
    explicit `Statevector` keeps the ancillas and its shots.
  - The host statevector walk skips the probability pass of a reset, or of a repeated
    measurement, when the qubit was last measured as 1 and nothing has touched it since,
    taking the probability as exactly 1. The pass returned 1 to within about 1e-14, so
    later probabilities on that branch move in the last bits, and a fixed-seed outcome
    changes only where a draw lands that close to a boundary.
  - More noisy and noiseless Clifford circuits with mid-circuit measurements move from
    per-shot tableau replay to the compiled sampler, which draws a different stream:
    circuits that use a measured qubit again without a reset, active reset by a guarded
    `X` among them, guarded Paulis under a register comparison reading several measured
    bits (the form `switch` and lookup decoding take), and asymmetric readout error on a
    bit a guard reads. Circuits that compiled before keep their shots.
  - QEC measurement records that a noisy or feed-forward program reads are sampled from
    one noiseless reference run spread by random Pauli frames, in place of a compiled
    sampler over the deferred circuit. Kept records, and detectors or observables that
    are random without noise, change for a seed; detectors, observables and postselection
    counts that are fixed without noise do not.
  - Compiled Clifford sampling with a GPU context draws its random bits on the device,
    from one seed per call and one stream per 64 shots and 32 parity columns, in place of
    bits drawn on the host. Fixed-seed shots, counts and marginals from the device
    sampler changed, and they no longer depend on the Rayon thread count.
  - Terminal shots and counts from a GPU statevector are drawn on the device against a
    scan of the state in blocks of 4096 amplitudes, with each uniform scaled by the
    total weight, in place of a host walk over the probability vector, when the walk
    moves fewer bytes than reading the `2^n` probabilities back: fewer shots than about
    two thirds of the amplitudes. The uniforms are unchanged, so a fixed-seed shot over
    the full register moves only where its draw lands within rounding of the boundary
    between two outcomes, and then to the neighbouring outcome. Counts drawn on the
    device whose map is anything but qubit `i` into bit `i` across the whole register, a
    partial or permuted map, walk the full state where they sampled a table keyed by the
    measured bits, so fixed-seed counts on those circuits changed. A run with more shots
    keeps the host sampler and its output, including the table for a partial map.

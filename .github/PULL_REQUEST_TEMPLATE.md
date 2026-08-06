<!--
Review this template before deleting sections that do not apply. Keep the headings that
match the PR so reviewers know what was considered and skipped.
-->

## Summary

Describe what changed and why in two or three sentences. Focus on the motivation, not the
diff. Link any related issue or discussion.

## Scope

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Refactor or cleanup
- [ ] Documentation
- [ ] Build, CI, or tooling

## Benchmarks (required for any change that touches a hot path)

Run `./scripts/bench_ab.sh --filter <rows> --ref <base>` and paste its table below. It
builds both bench binaries first and runs them adjacent, which is the only method that
reads correctly on a development host: separate `cargo bench` invocations minutes apart
have moved a byte-identical control group by as much as +98%. See `benches/README.md`.

Include at least one control row the diff cannot reach. A control that moves is the
signal that the change altered a shared callee's inlining or the binary layout, which no
target row will show on its own.

If this PR only changes documentation or non-performance code, write "N/A, no hot-path
changes" and skip the table.

| Benchmark | Before | After | Change | Control (same code) | Within 5% threshold? |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

Keep the control column. A change no larger than its row's same-code control spread is
noise, not a result; say so rather than rounding it into a win.

Name the tier if the numbers are not gate-precise. A `bench-fast` or reduced-sample run
is triage and should say so rather than be quoted as a measurement.

Regression verdict: PASS / FAIL / WAIVER

If WAIVER, explain why the regression is acceptable and what future work will recover it.

If the CI gate fails, check the failing row's control column before treating it as a
regression: a row the diff cannot reach, or one whose control moved as far as its mean,
is the runner and not the change. Say which, and paste the local A/B on the same two
commits. Do not merge through a red gate without that.

## Correctness

- [ ] `cargo nextest run --features "parallel gpu distributed"` passes locally
- [ ] `cargo test --doc --features "parallel gpu distributed"` passes
- [ ] `cargo clippy --all-targets --features "parallel gpu distributed" -- -D warnings -D clippy::undocumented_unsafe_blocks` passes
- [ ] `cargo fmt --check` passes
- [ ] `cargo doc --no-deps --features "parallel gpu distributed"` passes (CI denies
      rustdoc warnings)
- [ ] Docstrings on new and changed `pub` items add information the name and
      signature do not carry, and stay concise; none restate the signature
- [ ] New gate, backend, or fusion pass has golden tests against the statevector backend
- [ ] GPU-affecting change runs `cargo test --features "parallel gpu" --test golden_gpu`

`--all-features` pulls `distributed-mpi`, whose `mpi-sys` build script needs an MPI
toolkit, which is why CI enumerates features instead. Use it only when the diff touches
the MPI surface or the `Cargo.toml` feature wiring, and run it from an environment where
`bindgen` finds libclang and the MSVC headers (`scripts/mpi-env.ps1`).

## Hotspot notes

If the change touches a hot path, list the functions affected and attach profiler or
flamegraph output showing where time went. If no hot paths were touched, write "N/A".

## Architecture or design changes

- [ ] `docs/architecture/` updated if the change is structural
- [ ] Design or research notes added where required for new subsystems

## Breaking changes

List any public API breakage, config format changes, or behavioral differences a user
might hit after upgrading. Write "None" if there are none.

## Risks and rollback

Describe the blast radius if this lands and turns out wrong. Note any feature flags or
dispatch guards that let the change be rolled back without reverting the commit.

## Pre-merge checklist

- [ ] Commit messages follow the style rules
- [ ] No TODO or FIXME markers; deferred work is filed as an issue or noted above
- [ ] No secrets, credentials, or local config added
- [ ] No new dependencies without a rationale
- [ ] CI is green

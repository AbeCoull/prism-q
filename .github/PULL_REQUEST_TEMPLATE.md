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

Run from a quiet machine with `--features parallel` and paste the numbers below. If this
PR only changes documentation or non-performance code, write "N/A, no hot-path changes"
and skip the table.

| Benchmark | Before | After | Change | Within 5% threshold? |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

Regression verdict: PASS / FAIL / WAIVER

If WAIVER, explain why the regression is acceptable and what future work will recover it.

## Correctness

- [ ] `cargo test --all-features` passes locally
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] `cargo fmt --check` passes
- [ ] `cargo doc --no-deps --all-features` passes
- [ ] New public API has docstrings
- [ ] New gate, backend, or fusion pass has golden tests against the statevector backend
- [ ] GPU-affecting change runs `cargo test --features "parallel gpu" --test golden_gpu`

## Hotspot notes

If the change touches a hot path, list the functions affected and attach profiler or
flamegraph output showing where time went. If no hot paths were touched, write "N/A".

## Architecture or design changes

- [ ] `docs/architecture.md` updated if the change is structural
- [ ] Design or research notes added where required for new subsystems
- [ ] `task_tracker.md` updated to mark items done and record measured deltas

## Breaking changes

List any public API breakage, config format changes, or behavioral differences a user
might hit after upgrading. Write "None" if there are none.

## Risks and rollback

Describe the blast radius if this lands and turns out wrong. Note any feature flags or
dispatch guards that let the change be rolled back without reverting the commit.

## Pre-merge checklist

- [ ] Commit messages follow the style rules
- [ ] No secrets, credentials, or local config added
- [ ] No new dependencies without a rationale
- [ ] CI is green

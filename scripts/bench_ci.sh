#!/usr/bin/env bash
# Run the benchmark subset used by the CI regression gate.
#
# Delegates to the adjacent-binary A/B. Measuring the base branch, building the
# head branch, then measuring head is not a valid comparison: the build sits
# between the two measurements and drifts them by more than the gate on its own
# (see scripts/bench_ab.sh). The A/B builds both binaries first, then runs them
# adjacent in ref, new, new, ref order, so linear drift cancels and every row
# carries a same-code control column.
#
# The filter covers representative hot paths rather than the full Criterion
# suite. A row must exist on the reference commit as well as on the working
# tree, or it drops out of the comparison and the row-count guard fails the run.
#
# The `bench-fast` tier holds the wall time down: it pins samples to Criterion's
# floor of 10 and shortens the warmup and measurement windows. These rows are
# large on purpose, and Criterion only divides `measurement_time` across the
# samples while one iteration still fits the budget, so at 100 samples
# `statevector/qpe_t_gate/22q` alone reports 293.8s per pass against six passes.
# The A/B's four passes give two independent measurements per binary and a
# control column per row, which is what the gate reads; a tighter single-pass
# mean on top of that is not worth the wall time.
#
# Environment:
#   PRISM_BENCH_REF       git ref for the reference build (required)
#   PRISM_BENCH_REF_DIR   reference worktree path, cached between CI runs
#   CI_BENCH_FEATURES     cargo feature list (default: parallel,bench-fast)
#   REGRESSION_THRESHOLD  regression gate in percent (default: 5.0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FEATURES="${CI_BENCH_FEATURES:-parallel,bench-fast}"

if [[ -z "${PRISM_BENCH_REF:-}" ]]; then
    echo "Error: PRISM_BENCH_REF must name the reference commit." >&2
    exit 1
fi

# The `auto/` twins of the two 22q rows are deliberately absent. They run the
# same circuits through dispatch, which resolves to the statevector backend
# already covered above, so they doubled the most expensive pair in the set
# without covering a distinct kernel.
CIRCUITS_FILTER="^(statevector/(scalability_d5/22|qft_textbook/22|qpe_t_gate/22q|qaoa_l3/20)"
CIRCUITS_FILTER+="|stabilizer/(scaling/1000|measurement/ghz_measure_all/1000)"
CIRCUITS_FILTER+="|compiled_sampler/(noiseless/noiseless_1000q_10k|noisy/noisy_1000q_10k))$"

CIRCUITS_ROWS=8

args=(
    --bench circuits
    --filter "$CIRCUITS_FILTER"
    --min-rows "$CIRCUITS_ROWS"
    --ref "$PRISM_BENCH_REF"
    --features "$FEATURES"
    --out "$PROJECT_DIR/bench_results/ci-ab.md"
)

if [[ -n "${PRISM_BENCH_REF_DIR:-}" ]]; then
    args+=(--ref-dir "$PRISM_BENCH_REF_DIR")
fi

exec bash "$SCRIPT_DIR/bench_ab.sh" "${args[@]}"

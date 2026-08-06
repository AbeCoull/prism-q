#!/usr/bin/env bash
# Run the benchmark subset used by the CI regression gate.
#
# Delegates to the adjacent-binary A/B in scripts/bench_ab.sh, which explains why
# measuring the base, building head, then measuring head is not a comparison.
#
# The filter covers representative hot paths rather than the full Criterion
# suite, and `--min-rows` fails the run if one stops matching. `bench-fast` keeps
# the four passes affordable: the 22q rows run seconds per iteration, where the
# sample count sets the cost outright.
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

# The `auto/` twins of the 22q rows are absent on purpose: dispatch resolves them
# to the statevector rows above, doubling the most expensive pair for nothing.
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

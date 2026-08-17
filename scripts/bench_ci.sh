#!/usr/bin/env bash
# Run the benchmark subset used by the CI regression gate.
#
# Delegates to the adjacent-binary A/B in scripts/bench_ab.sh, which explains why
# measuring the base, building head, then measuring head is not a comparison.
#
# The filter covers representative hot paths rather than the full Criterion
# suite, and `--min-rows` fails the run if one stops matching. `bench-fast` keeps
# the four passes affordable: the widest rows run milliseconds per iteration, so
# the sample count sets the cost outright.
#
# Environment:
#   PRISM_BENCH_REF       git ref for the reference build (required)
#   PRISM_BENCH_REF_EXE   prebuilt reference executable. When it names an
#                         existing file the reference build is skipped and the
#                         worktree is never created, which halves the build cost.
#                         Ignored when the file is absent, so a cache miss falls
#                         back to building the reference.
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

# Sizes are the smallest that still exercise the paths the gate exists to watch,
# not the smallest available. `MIN_QUBITS_FOR_DIAG_BATCH = 16` and
# `MIN_QUBITS_FOR_POST_PHASE_BATCH = 18` in `circuit/fusion.rs` are the floors:
# below 16 the diagonal-batch families never form, and below 18 the post-phase
# rebatch never runs, so a smaller row would gate a different pipeline than the
# one that ships. `qft_textbook` has no 18 in its size list and stays at 20.
# Six passes over these eight rows measured at 73-75s in total, against about six
# minutes per pass for the 22q and 1000q set this replaces, which is the
# difference between a gate that fits its timeout and one that does not.
#
# The `auto/` twins are absent on purpose: dispatch resolves them to the
# statevector rows above, doubling the most expensive pair for nothing.
CIRCUITS_FILTER="^(statevector/(scalability_d5/18|qft_textbook/20|qpe_t_gate/16q|qaoa_l3/16)"
CIRCUITS_FILTER+="|stabilizer/(scaling/500|measurement/ghz_measure_all/500)"
CIRCUITS_FILTER+="|compiled_sampler/(noiseless/noiseless_500q_10k|noisy/noisy_500q_10k))$"

CIRCUITS_ROWS=8

args=(
    --bench circuits
    --filter "$CIRCUITS_FILTER"
    --min-rows "$CIRCUITS_ROWS"
    --ref "$PRISM_BENCH_REF"
    --features "$FEATURES"
    --out "$PROJECT_DIR/bench_results/ci-ab.md"
)

if [[ -n "${PRISM_BENCH_REF_EXE:-}" && -f "${PRISM_BENCH_REF_EXE}" ]]; then
    echo "Reference executable restored from cache; skipping the reference build."
    args+=(--ref-exe "$PRISM_BENCH_REF_EXE")
elif [[ -n "${PRISM_BENCH_REF_DIR:-}" ]]; then
    echo "No cached reference executable; building the reference from a worktree."
    args+=(--ref-dir "$PRISM_BENCH_REF_DIR")
fi

exec bash "$SCRIPT_DIR/bench_ab.sh" "${args[@]}"

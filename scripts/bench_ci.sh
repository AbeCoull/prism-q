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
# A FAIL is confirmed before it blocks. The set runs a second time and only a row
# that regressed in both runs counts, because at ten samples per lane a single
# run of this tier is not a result: four runs of identical code against identical
# code produced three PASSes and one FAIL, the FAIL on `statevector/scalability_d5/18`
# at +5.3% against a 4.2% control spread. Confirmation costs nothing on a green
# run and one extra set on a red one.
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
# difference between a gate that fits its timeout and one that does not. Under
# `bench-fast` a row costs its 1s measurement window rather than its iteration
# time, so a millisecond row and a microsecond row price the same here.
#
# The `auto/` twins are absent on purpose: dispatch resolves them to the
# statevector rows above, doubling the most expensive pair for nothing.
#
# The stabilizer measurement row is `wall_measure_all` rather than
# `ghz_measure_all` for two reasons. A GHZ chain collapses once and reads the
# rest of the register off deterministically, so it prices almost none of the
# measurement work the row exists to watch, where a CNOT wall collapses every
# qubit at random. And at 500 qubits the GHZ row runs in the microsecond band,
# where this fixture is a known cache cliff: one PR saw it read +5.7% and fail,
# then -1.4% and pass on a rerun of the same commit, with tight same-code
# controls both times, because the reference lane itself moved 113 to 165
# microseconds between runs on a byte-identical cached executable. A gate whose
# verdict blocks a merge cannot rest on a row that moves 46% for reasons a
# within-run control cannot see.
CIRCUITS_FILTER="^(statevector/(scalability_d5/18|qft_textbook/20|qpe_t_gate/16q|qaoa_l3/16)"
CIRCUITS_FILTER+="|stabilizer/(scaling/500|measurement/wall_measure_all/1000)"
CIRCUITS_FILTER+="|compiled_sampler/(noiseless/noiseless_500q_10k|noisy/noisy_500q_10k))$"

CIRCUITS_ROWS=8

args=(
    --bench circuits
    --filter "$CIRCUITS_FILTER"
    --min-rows "$CIRCUITS_ROWS"
    --ref "$PRISM_BENCH_REF"
    --features "$FEATURES"
)

if [[ -n "${PRISM_BENCH_REF_EXE:-}" && -f "${PRISM_BENCH_REF_EXE}" ]]; then
    echo "Reference executable restored from cache; skipping the reference build."
    args+=(--ref-exe "$PRISM_BENCH_REF_EXE")
elif [[ -n "${PRISM_BENCH_REF_DIR:-}" ]]; then
    echo "No cached reference executable; building the reference from a worktree."
    args+=(--ref-dir "$PRISM_BENCH_REF_DIR")
fi

FIRST="$PROJECT_DIR/bench_results/ci-ab.md"
CONFIRM="$PROJECT_DIR/bench_results/ci-ab-confirm.md"
rm -f "$CONFIRM"

# The regressed rows are the bullet list that follows the FAIL verdict, which
# ends at the next line that is neither blank nor a bullet.
regressed_rows() {
    awk '
        /^\*\*Regression verdict\*\*: FAIL/ { inside = 1; next }
        !inside { next }
        /^- `/ { gsub(/^- `|`$/, ""); print; next }
        /^$/ { next }
        { inside = 0 }
    ' "$1"
}

if bash "$SCRIPT_DIR/bench_ab.sh" "${args[@]}" --out "$FIRST"; then
    exit 0
fi

if ! grep -q '^\*\*Regression verdict\*\*: FAIL' "$FIRST"; then
    exit 1
fi

FIRST_ROWS="$(regressed_rows "$FIRST")"
if [[ -z "$FIRST_ROWS" ]]; then
    echo "Report says FAIL but names no row. Blocking rather than confirming a list that could not be read." >&2
    exit 1
fi

echo
echo "Regression reported. Running the set again; only a row that regresses twice blocks."
echo

bash "$SCRIPT_DIR/bench_ab.sh" "${args[@]}" --out "$CONFIRM" || true

if ! grep -q '^\*\*Regression verdict\*\*: FAIL' "$CONFIRM"; then
    echo "Second run reports no regression. Treating the first as unconfirmed."
    exit 0
fi

CONFIRMED="$(comm -12 <(sort <<<"$FIRST_ROWS") <(regressed_rows "$CONFIRM" | sort))"

if [[ -z "$CONFIRMED" ]]; then
    echo "Both runs reported a regression, on disjoint rows. Neither is confirmed."
    exit 0
fi

echo "Regressed in both runs:"
printf '%s\n' "$CONFIRMED" | sed 's/^/  /'
exit 1

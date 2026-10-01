#!/usr/bin/env bash
# Verify the FAIL confirmation in scripts/bench_ci.sh against a stub bench_ab.sh.
#
# Each case copies bench_ci.sh into a scratch tree beside a stub that writes a
# canned report and exit status for the first and the confirming run, then
# checks the gate's exit status. A first FAIL may clear only on a valid second
# result: a PASS, or a FAIL on disjoint rows. RERUN, NO DATA, host-busy and a
# missing report leave it standing.
#
# Usage: scripts/bench_ci_test.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

PASS=0
FAIL=0

report() {
    case "$1" in
        pass)    printf '**Regression verdict**: PASS at 5.0%%.\n' ;;
        fail:*)  printf '**Regression verdict**: FAIL. 1 row(s) regressed beyond 5.0%% and beyond their own control spread.\n\n- `%s`\n\nTrailer.\n' "${1#fail:}" ;;
        failnorow) printf '**Regression verdict**: FAIL. 1 row(s) regressed beyond 5.0%% and beyond their own control spread.\n\nTrailer.\n' ;;
        rerun)   printf '**Regression verdict**: RERUN. The working-tree binary moved as a lane.\n' ;;
        nodata)  printf '**Regression verdict**: NO DATA. No row appeared in all 6 passes.\n' ;;
        none)    ;;
    esac
}

# Cases name the first and second run as `exit/report`; report `none` writes no file.
check() {
    local label="$1" first="$2" second="$3" want="$4" want_runs="$5"
    local tree="$SCRATCH/$label"
    mkdir -p "$tree/scripts" "$tree/bench_results"
    cp "$SCRIPT_DIR/bench_ci.sh" "$tree/scripts/bench_ci.sh"
    cat > "$tree/scripts/bench_ab.sh" <<'STUB'
#!/usr/bin/env bash
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runs=$(( $(cat "$dir/runs" 2>/dev/null || echo 0) + 1 ))
echo "$runs" > "$dir/runs"
out=""
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--out" ]]; then out="$2"; shift 2; else shift; fi
done
spec="$(sed -n "${runs}p" "$dir/spec")"
if [[ "${spec#*/}" != none ]]; then
    cp "$dir/report$runs" "$out"
fi
exit "${spec%%/*}"
STUB
    printf '%s\n%s\n' "$first" "$second" > "$tree/scripts/spec"
    report "${first#*/}" > "$tree/scripts/report1"
    report "${second#*/}" > "$tree/scripts/report2"

    local got runs
    PRISM_BENCH_REF=stub bash "$tree/scripts/bench_ci.sh" > "$tree/log" 2>&1
    got=$?
    runs="$(cat "$tree/scripts/runs" 2>/dev/null || echo 0)"

    if [[ "$got" == "$want" && "$runs" == "$want_runs" ]]; then
        PASS=$((PASS + 1))
        printf 'ok    %-44s exit %s, %s run(s)\n' "$label" "$got" "$runs"
    else
        FAIL=$((FAIL + 1))
        printf 'FAIL  %-44s want exit %s/%s run(s), got %s/%s\n' "$label" "$want" "$want_runs" "$got" "$runs"
        sed 's/^/      /' "$tree/log"
    fi
}

echo "=== bench_ci FAIL confirmation ==="

check first-pass                     0/pass            0/none            0 1
check first-rerun-blocks             3/rerun           0/none            1 1
check first-host-busy-blocks         2/none            0/none            1 1
check confirm-pass-clears            1/fail:a/1        0/pass            0 2
check confirm-same-row-blocks        1/fail:a/1        1/fail:a/1        1 2
check confirm-disjoint-rows-clear    1/fail:a/1        1/fail:b/2        0 2
check confirm-rerun-blocks           1/fail:a/1        3/rerun           1 2
check confirm-nodata-blocks          1/fail:a/1        1/nodata          1 2
check confirm-host-busy-blocks       1/fail:a/1        2/none            1 2
check confirm-missing-report-blocks  1/fail:a/1        1/none            1 2
check confirm-pass-without-report    1/fail:a/1        0/none            1 2
check confirm-fail-without-rows      1/fail:a/1        1/failnorow       1 2

echo
echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]

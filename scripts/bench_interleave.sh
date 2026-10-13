#!/usr/bin/env bash
# Same-binary interleaved A/B benchmark comparison.
#
# For a change that can be switched at a coarse point (a fusion pass, a routing
# decision, a kernel choice), build one binary holding both implementations behind
# `prism_q::bench_variant_b()` and measure the two variants from that one binary.
# There is one build instead of two and one copy of the executable, so a copy that
# lands in a slow layout slows both variants alike instead of one lane.
#
#   1. Build the working tree once (or take --exe).
#   2. Run every row once per variant in Criterion's test mode, and stop before
#      any measurement if either fails.
#   3. Run one discarded invocation per variant, then ROUNDS rounds of four short
#      Criterion invocations over the filter, ordered A B B A in odd rounds and
#      B A A B in even ones. Each invocation is its own process with
#      PRISM_BENCH_VARIANT set to a or b.
#   4. Per row and round, take the paired change (B1 + B2) / (A1 + A2) - 1 of
#      each invocation's estimate, by default its fastest sample per iteration.
#      Linear drift across the four invocations cancels exactly in that ratio,
#      and alternating the order cancels any cost tied to a position in the round.
#   5. Drop a round as disturbed when, on the median row, the two invocations of
#      one variant disagree by more than the threshold. Load from other work
#      shows up there, and the test never compares A with B, so dropping a round
#      cannot bias the change. Replacement rounds run until ROUNDS undisturbed
#      ones are in, up to half as many again.
#   6. Report each row's median paired change across the undisturbed rounds with
#      a distribution-free 95% confidence interval for that median (order
#      statistics of the rounds), for the filter rows and the control rows.
#
# Verdicts, per row:
#   target rows   FAIL when the interval lies wholly above the threshold,
#                 undecided when it straddles the threshold, otherwise within
#                 the gate (named faster, slower, or flat).
#   control rows  rows the switch does not reach. FAIL when the interval lies
#                 wholly above the threshold (the change reached further than
#                 intended); off zero when it lies wholly outside half the
#                 threshold either way; wide when it reaches past the threshold
#                 either way.
# The run is RERUN when a control is off zero or wide, when a target row is
# undecided and nothing failed, or when fewer than six rounds were undisturbed.
#
# Usage:
#   scripts/bench_interleave.sh -f '^statevector/qft_textbook/(16|20)$' -c '^mps/random_d10/16$'
#   scripts/bench_interleave.sh -f '^sparse/' --exe /tmp/circuits-switch --rounds 12
#   scripts/bench_interleave.sh -f '^sparse/' --light               # triage, about a fifth of the time
#   scripts/bench_interleave.sh --build-only /tmp/circuits-switch   # for --exe or a queue
#
# Options:
#   --filter,   -f  Criterion filter regex for the rows the switch changes (required)
#   --control,  -c  Criterion filter regex for rows the switch does not reach. Run
#                   in the same rounds, in their own invocation per slot.
#   --bench,    -b  bench target (default: circuits)
#   --features      cargo feature list (default: "parallel bench-internal"; the
#                   switch only exists under bench-internal)
#   --exe           prebuilt bench executable; skips the build. The caller owns the
#                   claim that it was built from the tree under test.
#   --build-only    build the working tree's bench binary, copy it to this path,
#                   and exit. Delegates to scripts/bench_ab.sh, so a binary cached by
#                   either script is what the other expects.
#   --build-dir     with --build-only, build this checkout instead
#   --project-dir   checkout the run reports on and writes bench_results/ into
#   --rounds        A B B A rounds (default 16, light 6, floor 6). Six is the
#                   fewest whose extreme order statistics still give 95%. Sixteen
#                   held a same-code run of six circuits rows to within about 2%
#                   (3.7% on a 190 us row) in 17 minutes on the reference host.
#   --samples       Criterion samples per row (default 10, the floor)
#   --window        Criterion measurement window in seconds (default 1, light 0.5),
#                   after a warm-up of half the window. Groups that pin their own
#                   keep it.
#   --light         triage tier: 6 rounds at a 0.5s window. The verdict line names
#                   the tier.
#   --estimate      per-invocation estimate: min (default), median, or mean.
#                   min is the fastest of the Criterion samples per iteration:
#                   load from other work only ever adds time, and on the reference
#                   host it read a third to half the round-to-round spread of the
#                   mean. Use mean for a change that alters the spread of
#                   iteration times rather than their floor.
#   --min-rows      fail when fewer than this many filter rows appear in every
#                   invocation (default 1)
#   --out           markdown output path (default bench_results/ix-<stamp>.md).
#                   The raw per-invocation table is written beside it as .tsv.
#   --from-tsv      skip building and measuring and report from the raw table a
#                   previous run wrote, busy rounds included. Settings recorded
#                   only at run time read "from tsv".
#   --threshold, -t regression gate in percent (default: REGRESSION_THRESHOLD or 5.0)
#
# Host guards, as in scripts/bench_ab.sh:
#   --max-host-load percent of all CPUs other work may hold (default 10, 0
#                   disables). Read before the first invocation, where a busy host
#                   stops the run, and before each round, where it is recorded
#                   only: the disturbed-round test decides which rounds count.
#   --wait-idle     seconds to wait for an idle host before giving up (default 0)
#
# Exit status: 0 PASS, 1 FAIL or NO DATA, 2 host too busy to start, 3 RERUN,
# 4 a variant failed the smoke test.
#
# Environment:
#   REGRESSION_THRESHOLD   regression gate in percent (default: 5.0)
#   PRISM_BENCH_HOST_LOAD  fixed host load reading in percent, for the script's tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

FILTER=""
CONTROL_FILTER=""
BENCH="circuits"
FEATURES="parallel bench-internal"
EXE=""
BUILD_ONLY=""
BUILD_DIR=""
ROUNDS=""
SAMPLES=10
WINDOW=""
LIGHT=""
MIN_ROWS=1
ESTIMATE="min"
OUT=""
FROM_TSV=""
THRESHOLD="${REGRESSION_THRESHOLD:-5.0}"
MAX_HOST_LOAD=10
WAIT_IDLE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter|-f)    FILTER="$2"; shift 2 ;;
        --control|-c)   CONTROL_FILTER="$2"; shift 2 ;;
        --bench|-b)     BENCH="$2"; shift 2 ;;
        --features)     FEATURES="$2"; shift 2 ;;
        --exe)          EXE="$2"; shift 2 ;;
        --build-only)   BUILD_ONLY="$2"; shift 2 ;;
        --build-dir)    BUILD_DIR="$2"; shift 2 ;;
        --project-dir)  PROJECT_DIR="$(cd "$2" && pwd)"; shift 2 ;;
        --rounds)       ROUNDS="$2"; shift 2 ;;
        --samples)      SAMPLES="$2"; shift 2 ;;
        --window)       WINDOW="$2"; shift 2 ;;
        --light)        LIGHT=1; shift ;;
        --min-rows)     MIN_ROWS="$2"; shift 2 ;;
        --estimate)     ESTIMATE="$2"; shift 2 ;;
        --out)          OUT="$2"; shift 2 ;;
        --from-tsv)     FROM_TSV="$2"; shift 2 ;;
        --threshold|-t) THRESHOLD="$2"; shift 2 ;;
        --max-host-load) MAX_HOST_LOAD="$2"; shift 2 ;;
        --wait-idle)    WAIT_IDLE="$2"; shift 2 ;;
        -h|--help)      awk 'NR > 1 && !/^#/ { exit } NR > 1' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$BUILD_ONLY" ]]; then
    exec bash "$SCRIPT_DIR/bench_ab.sh" --build-only "$BUILD_ONLY" \
        --build-dir "${BUILD_DIR:-$PROJECT_DIR}" --bench "$BENCH" --features "$FEATURES"
fi
if [[ -z "$FILTER" && -z "$FROM_TSV" ]]; then
    echo "Error: --filter is required. Name the rows the switch changes." >&2
    exit 1
fi

if [[ -n "$LIGHT" ]]; then
    TIER="light (triage)"
    ROUNDS="${ROUNDS:-6}"
    WINDOW="${WINDOW:-0.5}"
else
    TIER="full"
    ROUNDS="${ROUNDS:-16}"
    WINDOW="${WINDOW:-1}"
fi
WARM_UP="$(awk -v w="$WINDOW" 'BEGIN { printf "%g", w / 2 }')"
case "$ESTIMATE" in
    mean) EST_COL=6 ;; median) EST_COL=7 ;; min) EST_COL=8 ;;
    *) echo "Error: --estimate is mean, median, or min." >&2; exit 1 ;;
esac
if (( ROUNDS < 6 )); then
    echo "Error: --rounds is $ROUNDS; fewer than 6 rounds cannot give a 95% interval." >&2
    exit 1
fi
if (( SAMPLES < 10 )); then
    echo "Error: --samples is $SAMPLES; Criterion's floor is 10." >&2
    exit 1
fi

cd "$PROJECT_DIR"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

native_path() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1"
    else
        printf '%s' "$1"
    fi
}

host_cpu() {
    if [[ -r /proc/cpuinfo ]]; then
        awk -F': ' '/model name/ { print $2; exit }' /proc/cpuinfo
    elif command -v sysctl >/dev/null 2>&1 && sysctl -n machdep.cpu.brand_string >/dev/null 2>&1; then
        sysctl -n machdep.cpu.brand_string
    else
        printf '%s' "${PROCESSOR_IDENTIFIER:-unknown}"
    fi
}

# Busy share of every CPU over about two seconds, as a whole percent, or empty
# where the host offers no reading.
host_load() {
    if [[ -n "${PRISM_BENCH_HOST_LOAD:-}" ]]; then
        printf '%s' "$PRISM_BENCH_HOST_LOAD"
    elif [[ -r /proc/stat ]]; then
        local first second
        first="$(awk '/^cpu / { b = $2 + $3 + $4 + $7 + $8 + $9; print b, b + $5 + $6; exit }' /proc/stat)"
        sleep 2
        second="$(awk '/^cpu / { b = $2 + $3 + $4 + $7 + $8 + $9; print b, b + $5 + $6; exit }' /proc/stat)"
        awk -v a="$first" -v b="$second" 'BEGIN {
            split(a, x, " "); split(b, y, " ")
            if (y[2] > x[2]) { printf "%d", (y[1] - x[1]) * 100 / (y[2] - x[2]) }
        }'
    elif command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu >/dev/null 2>&1; then
        ps -A -o %cpu= | awk -v n="$(sysctl -n hw.ncpu)" '{ s += $1 } END { printf "%d", s / n }'
    fi
}

busy_processes() {
    if command -v wmic >/dev/null 2>&1; then
        wmic path Win32_PerfFormattedData_PerfProc_Process get Name,PercentProcessorTime 2>/dev/null |
            tr -d '\r' |
            awk 'NR > 1 && $1 != "_Total" && $1 != "Idle" && $2 + 0 > 0 { print $2 "% " $1 }' |
            sort -rn | head -5
    else
        ps -eo pcpu,comm --sort=-pcpu 2>/dev/null | sed -n '2,6p'
    fi
}

# Stop, or wait up to --wait-idle, while other work holds more of the host than
# --max-host-load.
preflight() {
    (( MAX_HOST_LOAD > 0 )) || return 0
    local load waited=0
    while :; do
        load="$(host_load)"
        if [[ -z "$load" ]]; then
            echo ">>> no host load reading here, so the load guard is off for this run"
            MAX_HOST_LOAD=0
            return 0
        fi
        if (( load <= MAX_HOST_LOAD )); then
            echo ">>> host ${load}% busy before measuring"
            return 0
        fi
        if (( waited >= WAIT_IDLE )); then
            echo "Error: the host is ${load}% busy before measuring, over --max-host-load ${MAX_HOST_LOAD}%." >&2
            busy_processes | sed 's/^/    /' >&2
            echo "  Stop that work, pass --wait-idle SECONDS to wait for it, or raise" >&2
            echo "  --max-host-load if the load is part of what is being measured." >&2
            exit 2
        fi
        echo ">>> host ${load}% busy, waiting for it to fall under ${MAX_HOST_LOAD}% (${waited}s of ${WAIT_IDLE}s)"
        sleep 28
        waited=$(( waited + 30 ))
    done
}

# `full_id<TAB>mean_ns<TAB>median_ns<TAB>min_ns` for every row an invocation
# measured: Criterion's mean and median point estimates, and the fastest sample
# per iteration from sample.json.
snapshot() {
    find "$1" -path "*/new/estimates.json" -print 2>/dev/null | while IFS= read -r est; do
        local dir="${est%estimates.json}"
        [[ -f "$dir/benchmark.json" ]] || continue
        local id stats files=("$est")
        [[ -f "$dir/sample.json" ]] && files+=("$dir/sample.json")
        id="$(awk 'match($0, /"full_id":"[^"]*"/) {
                print substr($0, RSTART + 11, RLENGTH - 12); exit
            }' "$dir/benchmark.json")"
        stats="$(awk '
            function point(obj,    seg) {
                seg = substr(est, index(est, "\"" obj "\":{"))
                if (match(seg, /"point_estimate":[-0-9.eE+]+/)) {
                    return substr(seg, RSTART + 17, RLENGTH - 17)
                }
                return ""
            }
            FILENAME ~ /estimates\.json$/ { est = est $0; next }
            {
                if (match($0, /"iters":\[[^]]*\]/)) { ni = split(substr($0, RSTART + 9, RLENGTH - 10), it, ",") }
                if (match($0, /"times":\[[^]]*\]/)) { nt = split(substr($0, RSTART + 9, RLENGTH - 10), tm, ",") }
                for (i = 1; i <= ni && i <= nt; i++) {
                    if (it[i] > 0 && (low == "" || tm[i] / it[i] < low)) { low = tm[i] / it[i] }
                }
            }
            END { printf "%s\t%s\t%s", point("mean"), point("median"), low }' "${files[@]}")"
        [[ -n "$id" && -n "$stats" && "$stats" != $'\t'* ]] && printf '%s\t%s\n' "$id" "$stats"
    done
}

# Run every row of the filters once per variant in Criterion's test mode; the
# last `Testing` line in a failing log names the row.
smoke_test() {
    local variant="$1" filter log status row rows
    for filter in "$FILTER" ${CONTROL_FILTER:+"$CONTROL_FILTER"}; do
        log="$WORKDIR/smoke-$variant.log"
        status=0
        PRISM_BENCH_VARIANT="$variant" CRITERION_HOME="$(native_path "$WORKDIR/smoke-home")" \
            "$RUN_EXE" --test "$filter" > "$log" 2>&1 || status=$?
        rows="$(grep -c '^Testing ' "$log" || true)"
        if (( status != 0 )); then
            row="$(awk '/^Testing / { id = substr($0, 9) } END { print id }' "$log")"
            echo "" >&2
            echo "Error: smoke test failed on variant $variant: it exited $status on row '${row:-unknown}'." >&2
            echo "  Filter: $filter" >&2
            tail -n 15 "$log" | sed 's/^/    /' >&2
            echo "  No measurement ran. Fix the row or leave it out of the filter." >&2
            exit 4
        fi
        echo "    variant $variant: $rows row(s) ran once under '$filter'"
    done
}

# One Criterion invocation per segment of a slot, appended to the raw table as
# `round slot variant segment id mean_ns`.
RAW="$WORKDIR/raw.tsv"
: > "$RAW"
run_slot() {
    local round="$1" slot="$2" variant="$3"
    local seg filter home log rows
    for seg in target control; do
        filter="$FILTER"
        if [[ "$seg" == control ]]; then
            [[ -n "$CONTROL_FILTER" ]] || continue
            filter="$CONTROL_FILTER"
        fi
        home="$WORKDIR/crit-$round-$slot-$seg"
        log="$WORKDIR/run-$round-$slot-$seg.log"
        mkdir -p "$home"
        if ! PRISM_BENCH_VARIANT="$variant" CRITERION_HOME="$(native_path "$home")" \
            PRISM_BENCH_SAMPLES="$SAMPLES" "$RUN_EXE" --bench \
            --warm-up-time "$WARM_UP" --measurement-time "$WINDOW" "$filter" > "$log" 2>&1; then
            tail -n 15 "$log" >&2
            echo "Error: the bench binary failed in round $round, variant $variant, '$filter'." >&2
            exit 1
        fi
        rows="$(snapshot "$home" | awk -v r="$round" -v s="$slot" -v v="$variant" -v g="$seg" '
            BEGIN { OFS = "\t" } { print r, s, v, g, $1, $2, $3, $4 }' | tee -a "$RAW" | wc -l | tr -d '[:space:]')"
        if (( rows == 0 )); then
            echo "Error: no Criterion estimates for '$filter' in round $round, variant $variant." >&2
            exit 1
        fi
        rm -rf "$home"
    done
}

# Rounds whose same-variant invocations disagree past the threshold on the
# median row, as `round spread_percent`, from the raw table so far.
disturbed_rounds() {
    awk -F'\t' -v col="$EST_COL" -v threshold="$THRESHOLD" '
        function abs(x) { return x < 0 ? -x : x }
        NF >= 6 && $1 > 0 {
            k = $1 SUBSEP $5 SUBSEP $3
            if (k in first) { second[k] = $col + 0 } else { first[k] = $col + 0 }
            ids[$5] = 1
            if ($1 + 0 > rounds) { rounds = $1 + 0 }
        }
        END {
            for (r = 1; r <= rounds; r++) {
                n = 0
                for (id in ids) {
                    ka = r SUBSEP id SUBSEP "a"; kb = r SUBSEP id SUBSEP "b"
                    if (!(ka in second) || !(kb in second) || first[ka] <= 0 || first[kb] <= 0) { continue }
                    a = abs(second[ka] / first[ka] - 1); b = abs(second[kb] / first[kb] - 1)
                    x[++n] = 100 * (a > b ? a : b)
                }
                for (i = 2; i <= n; i++) {
                    t = x[i]; j = i - 1
                    while (j >= 1 && x[j] > t) { x[j + 1] = x[j]; j-- }
                    x[j + 1] = t
                }
                if (n > 0) {
                    med = (n % 2) ? x[(n + 1) / 2] : (x[n / 2] + x[n / 2 + 1]) / 2
                    if (med > threshold) { printf "%d %.1f\n", r, med }
                }
            }
        }' "$RAW"
}

run_round() {
    local round="$1" load slot order
    if (( MAX_HOST_LOAD > 0 )); then
        load="$(host_load)"
        if [[ -n "$load" ]] && (( load > MAX_HOST_LOAD )); then
            echo ">>> host ${load}% busy before round $round, over ${MAX_HOST_LOAD}%"
            LOADED_ROUNDS="${LOADED_ROUNDS}${LOADED_ROUNDS:+, }round $round at ${load}%"
        fi
    fi
    if (( round % 2 == 1 )); then order=(a b b a); else order=(b a a b); fi
    echo ">>> round $round (${order[*]})"
    for slot in 1 2 3 4; do
        run_slot "$round" "$slot" "${order[slot - 1]}"
    done
}

# --- Measure ---

LOADED_ROUNDS=""
STARTED="$(date +%s)"
if [[ -z "$FROM_TSV" ]]; then
    echo "=== PRISM-Q interleaved same-binary A/B ==="
    echo "  bench:     $BENCH"
    echo "  filter:    $FILTER"
    [[ -n "$CONTROL_FILTER" ]] && echo "  control:   $CONTROL_FILTER"
    echo "  features:  $FEATURES"
    echo "  tier:      $TIER, $ROUNDS rounds, $SAMPLES samples, ${WARM_UP}s + ${WINDOW}s windows"
    echo "  threshold: ${THRESHOLD}%"
    echo ""

    if [[ -n "$EXE" ]]; then
        [[ -f "$EXE" ]] || { echo "Error: --exe '$EXE' does not exist." >&2; exit 1; }
        RUN_EXE="$EXE"
        PROVENANCE="supplied via \`--exe\`, not built here"
    else
        RUN_EXE="$WORKDIR/exe"
        bash "$SCRIPT_DIR/bench_ab.sh" --build-only "$RUN_EXE" --bench "$BENCH" --features "$FEATURES"
        PROVENANCE="built from the working tree"
    fi
    COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] && COMMIT="$COMMIT with uncommitted changes"

    echo ">>> smoke test: every row once per variant in Criterion's test mode"
    smoke_test a
    smoke_test b
    echo ""

    preflight
    echo ">>> warmup: one discarded invocation per variant"
    warm_start="$(date +%s)"
    run_slot 0 1 a
    run_slot 0 2 b
    : > "$RAW"
    per_slot=$(( ($(date +%s) - warm_start + 1) / 2 ))
    printf '>>> projected: about %d min for %d rounds\n\n' $(( (per_slot * 4 + 2) * ROUNDS / 60 + 1 )) "$ROUNDS"

    for (( round = 1; round <= ROUNDS; round++ )); do
        run_round "$round"
    done
    while (( round <= ROUNDS + ROUNDS / 2 )); do
        disturbed="$(disturbed_rounds | wc -l | tr -d '[:space:]')"
        (( round - 1 - disturbed < ROUNDS )) || break
        echo ">>> $disturbed round(s) disturbed so far; running a replacement"
        run_round "$round"
        round=$(( round + 1 ))
    done
    ROUNDS_RUN=$(( round - 1 ))
    WALL="$(( ($(date +%s) - STARTED) / 60 )) min $(( ($(date +%s) - STARTED) % 60 )) s"
else
    [[ -f "$FROM_TSV" ]] || { echo "Error: --from-tsv '$FROM_TSV' does not exist." >&2; exit 1; }
    awk -F'\t' 'NF >= 6' "$FROM_TSV" > "$RAW"
    LOADED_ROUNDS="$(awk -F'\t' '$1 == "#loaded" { print $2 }' "$FROM_TSV")"
    PROVENANCE="from tsv"; COMMIT="from tsv"; WALL="from tsv"
    ROUNDS="$(awk -F'\t' '$1 > m { m = $1 } END { print m + 0 }' "$RAW")"
    ROUNDS_RUN="$ROUNDS"
fi
DISTURBED="$(disturbed_rounds | awk '{ printf "%s%s:%s", (NR > 1 ? " " : ""), $1, $2 }')"

# --- Report ---

if [[ -z "$OUT" ]]; then
    OUT="$PROJECT_DIR/bench_results/ix-$(date +%Y-%m-%d_%H%M%S).md"
fi
mkdir -p "$(dirname "$OUT")"
if [[ -z "$FROM_TSV" ]]; then
    cp "$RAW" "${OUT%.md}.tsv"
    [[ -z "$LOADED_ROUNDS" ]] || printf '#loaded\t%s\n' "$LOADED_ROUNDS" >> "${OUT%.md}.tsv"
fi

set +e
{
    echo "## Interleaved same-binary A/B: \`$BENCH\`"
    echo ""
    echo "| Setting | Value |"
    echo "| --- | --- |"
    echo "| Filter | \`${FILTER:-from tsv}\` |"
    [[ -n "$CONTROL_FILTER" ]] && echo "| Control filter | \`$CONTROL_FILTER\` |"
    echo "| Binary | $PROVENANCE, commit $COMMIT |"
    echo "| Variants | a and b of \`PRISM_BENCH_VARIANT\`, one process per invocation |"
    echo "| Features | \`$FEATURES\` |"
    echo "| Tier | $TIER |"
    echo "| Rounds | $ROUNDS_RUN run for $ROUNDS undisturbed, A B B A then B A A B, after one discarded invocation per variant |"
    echo "| Criterion | $SAMPLES samples, ${WARM_UP}s warm-up, ${WINDOW}s measurement (groups that pin their own keep it) |"
    echo "| Estimate | \`$ESTIMATE\` of each invocation; median paired change across rounds, 95% interval from order statistics |"
    if (( MAX_HOST_LOAD > 0 )); then
        echo "| Host load guard | ${MAX_HOST_LOAD}% of all CPUs, read before each round |"
    else
        echo "| Host load guard | off |"
    fi
    echo "| CPU | $(host_cpu) |"
    echo "| OS | $(uname -srm) |"
    echo "| Toolchain | $(rustc --version 2>/dev/null || echo unknown) |"
    echo "| Wall time | $WALL |"
    echo "| Threshold | ${THRESHOLD}% |"
    echo ""
    awk -v threshold="$THRESHOLD" -v min_rows="$MIN_ROWS" -v loaded="$LOADED_ROUNDS" \
        -v light="${LIGHT:-0}" -v col="$EST_COL" -v disturbed="$DISTURBED" '
BEGIN {
    FS = "\t"; SEP = "\x1f"
    nd = split(disturbed, dl, " ")
    for (i = 1; i <= nd; i++) {
        split(dl[i], dp, ":"); drop[dp[1] + 0] = dp[2]
        dropped_text = dropped_text (i > 1 ? ", " : "") "round " dp[1] " at " dp[2] "%"
    }
}

{
    round = $1; variant = $3; seg = $4; id = $5; ns = $col + 0
    if (!(id in kind)) { kind[id] = seg; order[++n] = id }
    sum[id SEP round SEP variant] += ns
    cnt[id SEP round SEP variant]++
    j = ++vals[id SEP variant]
    all[id SEP variant SEP j] = ns
    if (round > rounds) { rounds = round }
}

function fmt(ns) {
    if (ns >= 1000000000) { return sprintf("%.3f s", ns / 1000000000) }
    if (ns >= 1000000)    { return sprintf("%.2f ms", ns / 1000000) }
    if (ns >= 1000)       { return sprintf("%.2f us", ns / 1000) }
    return sprintf("%.1f ns", ns)
}
function pct(v) { return sprintf("%+.1f%%", v) }

function sort_n(a, m,    i, j, t) {
    for (i = 2; i <= m; i++) {
        t = a[i]; j = i - 1
        while (j >= 1 && a[j] > t) { a[j + 1] = a[j]; j-- }
        a[j + 1] = t
    }
}
function median_of(a, m) { return (m % 2) ? a[(m + 1) / 2] : (a[m / 2] + a[m / 2 + 1]) / 2 }

# Largest k with P(Binomial(m, 1/2) <= k - 1) <= 0.025: the interval
# [x(k), x(m + 1 - k)] then covers the median with at least 95%.
function ci_rank(m,    k, c, tail, total) {
    total = 2 ^ m; c = 1; tail = 0; k = 0
    while (k < m / 2) {
        tail += c
        if (tail / total > 0.025) { break }
        k++
        c = c * (m - k + 1) / k
    }
    return k
}
function coverage(m, k,    i, c, tail) {
    c = 1; tail = 0
    for (i = 0; i < k; i++) { tail += c; c = c * (m - i) / (i + 1) }
    return 100 * (1 - 2 * tail / 2 ^ m)
}

END {
    print "| Benchmark | A | B | Change | 95% interval | Rounds | Verdict |"
    print "| --- | ---: | ---: | ---: | :---: | ---: | --- |"
    compared = 0; ctl_rows = 0
    for (i = 1; i <= n; i++) {
        id = order[i]
        m = 0
        for (r = 1; r <= rounds; r++) {
            if (r in drop) { continue }
            ka = id SEP r SEP "a"; kb = id SEP r SEP "b"
            if (cnt[ka] != 2 || cnt[kb] != 2 || sum[ka] <= 0 || sum[kb] <= 0) { continue }
            x[++m] = (sum[kb] / sum[ka] - 1) * 100
        }
        if (m < 6) { short_rows = short_rows "\n- `" id "` (" m " complete rounds)"; continue }
        sort_n(x, m)
        med = median_of(x, m)
        k = ci_rank(m)
        if (k < 1) { k = 1 }
        lo = x[k]; hi = x[m + 1 - k]
        cov = coverage(m, k)
        if (cov < min_cov || min_cov == 0) { min_cov = cov }

        na = vals[id SEP "a"]; nb = vals[id SEP "b"]
        for (j = 1; j <= na; j++) { va[j] = all[id SEP "a" SEP j] }
        for (j = 1; j <= nb; j++) { vb[j] = all[id SEP "b" SEP j] }
        sort_n(va, na); sort_n(vb, nb)
        abs_a = median_of(va, na); abs_b = median_of(vb, nb)

        if (kind[id] == "control") {
            ctl_rows++
            if (lo > threshold) {
                verdict = "control moved"; regressed[++regressions] = id
            } else if (lo > threshold / 2 || hi < -threshold / 2) {
                verdict = "control off zero"; off[++offs] = id
            } else if (hi > threshold || lo < -threshold) {
                verdict = "control wide"; off[++offs] = id
            } else {
                verdict = "control"
            }
        } else {
            compared++
            if (lo > threshold) {
                verdict = "regressed"; regressed[++regressions] = id
            } else if (hi > threshold) {
                verdict = "undecided"; undecided[++undecideds] = id
            } else if (lo > 0) {
                verdict = "slower"
            } else if (hi < 0) {
                verdict = "faster"
            } else {
                verdict = "flat"
            }
            if (hi - lo > widest) { widest = hi - lo }
        }
        printf "| `%s` | %s | %s | %s | %s to %s | %d | %s |\n",
            id, fmt(abs_a), fmt(abs_b), pct(med), pct(lo), pct(hi), m, verdict
    }

    print ""
    printf "A and B are the median of each variant over its invocations. Change is the median over\n"
    printf "rounds of (B1 + B2) / (A1 + A2) - 1; the interval covers it with at least %.1f%%.\n", min_cov
    printf "%d filter row(s) and %d control row(s); widest filter interval %.1f points.\n",
        compared, ctl_rows, widest
    if (nd > 0) {
        printf "Dropped as disturbed (same-variant spread on the median row): %s.\n", dropped_text
    }
    if (loaded != "") {
        printf "The host read busy before %s.\n", loaded
    }
    if (short_rows != "") {
        printf "\nRows left out for missing invocations:%s\n", short_rows
    }
    print ""

    tier_note = light ? " Light tier, triage only: run the full tier before claiming a number." : ""
    if (rounds >= 6 && rounds - nd < 6) {
        printf "**Regression verdict**: RERUN. Only %d of %d rounds were undisturbed, too few for a 95%% interval. The host was too busy to measure.%s\n",
            rounds - nd, rounds, tier_note
        exit 3
    }
    if (compared == 0 || compared < min_rows) {
        printf "**Regression verdict**: NO DATA. %d filter row(s) completed at least 6 rounds, expected at least %d.\n",
            compared, (min_rows > 1 ? min_rows : 1)
        exit 1
    }
    if (offs > 0) {
        printf "**Regression verdict**: RERUN. %d control row(s) did not read flat, so the two variants differ on rows the switch should not reach, or the rounds were too noisy to tell.%s\n\n",
            offs, tier_note
        for (i = 1; i <= offs; i++) { printf "- `%s`\n", off[i] }
        if (regressions > 0) {
            print "\nRows past the gate in this run, not a result:\n"
            for (i = 1; i <= regressions; i++) { printf "- `%s`\n", regressed[i] }
        }
        exit 3
    }
    if (regressions > 0) {
        printf "**Regression verdict**: FAIL. %d row(s) regressed with the whole interval past %s%%.%s\n\n",
            regressions, threshold, tier_note
        for (i = 1; i <= regressions; i++) { printf "- `%s`\n", regressed[i] }
        exit 1
    }
    if (undecideds > 0) {
        printf "**Regression verdict**: RERUN. %d row(s) have an interval that straddles %s%%, too wide to decide. Add rounds or narrow the filter.%s\n\n",
            undecideds, threshold, tier_note
        for (i = 1; i <= undecideds; i++) { printf "- `%s`\n", undecided[i] }
        exit 3
    }
    printf "**Regression verdict**: PASS at %s%%.%s\n", threshold, tier_note
    exit 0
}
' "$RAW"
} > "$OUT"
STATUS=$?
set -e

echo "=== report: $OUT ==="
cat "$OUT"
exit "$STATUS"

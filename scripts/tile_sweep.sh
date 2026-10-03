#!/usr/bin/env bash
# Sweep one environment knob over a set of values with a single bench binary.
#
# The tool for a cache-geometry question on a host other than the one the kernels
# were tuned on: build the bench target once, then run the same executable at each
# value of the knob, forward through the list and then reversed, so drift over the
# run cancels between the two orderings. Every value is measured once per run, and
# a row's number is the minimum over runs: a slow lane or a busy minute raises a
# measurement and never lowers it, so the minimum is the estimate least touched by
# the host.
#
# The report compares every value with the first one in --values, which should be
# the host's current default, and carries each value's own spread over the runs.
# A change smaller than that spread is noise. These are triage numbers: the
# sample counts are low and a shared runner is never quiet. A move past 20% is a
# finding, a move under 10% is not, and a default changes only after an A/B on
# the full tier of scripts/bench_ab.sh confirms the chosen value.
#
# Usage:
#   scripts/tile_sweep.sh -f '^statevector/(qv|qft_textbook)/2[02]$'
#   scripts/tile_sweep.sh -f '^statevector/qv/24$' --values '256 512 1024' --runs 4
#   scripts/tile_sweep.sh -f '^statevector/qv/20$' --var PRISM_MULTI_2Q_TILE_BITS --values '13 14 15 16'
#
# Options:
#   --filter,  -f   Criterion filter regex (required)
#   --var           environment variable to sweep (default PRISM_TILE_KB)
#   --values        space-separated values, the first is the baseline
#                   (default "256 128 512 1024 2048")
#   --runs          passes over the values, alternating forward and reversed
#                   (default 3)
#   --samples       Criterion samples per row (default 10, the floor)
#   --window        Criterion measurement window in seconds (default 1.5, with a
#                   0.5s warm-up; groups that pin their own keep it)
#   --bench,   -b   bench target (default circuits)
#   --features      cargo feature list (default parallel)
#   --exe           prebuilt bench executable; skips the build
#   --out           markdown output path (default bench_results/tile-sweep-<stamp>.md)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

FILTER=""
VAR="PRISM_TILE_KB"
VALUES="256 128 512 1024 2048"
RUNS=3
SAMPLES=10
WINDOW=1.5
BENCH="circuits"
FEATURES="parallel"
EXE=""
OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter|-f) FILTER="$2"; shift 2 ;;
        --var)       VAR="$2"; shift 2 ;;
        --values)    VALUES="$2"; shift 2 ;;
        --runs)      RUNS="$2"; shift 2 ;;
        --samples)   SAMPLES="$2"; shift 2 ;;
        --window)    WINDOW="$2"; shift 2 ;;
        --bench|-b)  BENCH="$2"; shift 2 ;;
        --features)  FEATURES="$2"; shift 2 ;;
        --exe)       EXE="$2"; shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        -h|--help)   awk 'NR > 1 && !/^#/ { exit } NR > 1' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$FILTER" ]]; then
    echo "Error: --filter is required." >&2
    exit 1
fi
read -r -a VALUE_LIST <<< "$VALUES"
if (( ${#VALUE_LIST[@]} < 2 )); then
    echo "Error: --values needs a baseline and at least one other value." >&2
    exit 1
fi
if (( RUNS < 1 )); then
    echo "Error: --runs must be at least 1." >&2
    exit 1
fi

for cmd in awk cargo; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd is required." >&2; exit 1; }
done

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
    elif command -v wmic >/dev/null 2>&1; then
        wmic cpu get name 2>/dev/null | awk 'NR == 2 { sub(/[ \t\r]+$/, ""); print }'
    else
        echo "unknown"
    fi
}

# The cache geometry the run saw, for the record beside the numbers.
host_caches() {
    if [[ -d /sys/devices/system/cpu/cpu0/cache ]]; then
        for dir in /sys/devices/system/cpu/cpu0/cache/index*; do
            printf 'L%s %s %s shared by %s; ' "$(cat "$dir/level")" "$(cat "$dir/type")" \
                "$(cat "$dir/size")" "$(cat "$dir/shared_cpu_list")"
        done
        echo
    elif command -v sysctl >/dev/null 2>&1 && sysctl -n hw.l2cachesize >/dev/null 2>&1; then
        sysctl hw.l1dcachesize hw.l2cachesize hw.l3cachesize hw.perflevel0.l2cachesize \
            hw.perflevel0.cpusperl2 hw.physicalcpu hw.logicalcpu 2>/dev/null | tr '\n' ';'
        echo
    elif command -v wmic >/dev/null 2>&1; then
        wmic cpu get L2CacheSize,L3CacheSize,NumberOfCores,NumberOfLogicalProcessors 2>/dev/null |
            awk 'NR == 2 { gsub(/[ \t\r]+/, " "); print "L2 " $1 " KB, L3 " $2 " KB, " $3 " cores, " $4 " threads" }'
    else
        echo "unknown"
    fi
}

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
STAMP="$(date +%Y%m%d-%H%M%S)"
if [[ -z "$OUT" ]]; then
    mkdir -p "$PROJECT_DIR/bench_results"
    OUT="$PROJECT_DIR/bench_results/tile-sweep-$STAMP.md"
fi

echo "=== PRISM-Q knob sweep ==="
echo "  bench:    $BENCH"
echo "  filter:   $FILTER"
echo "  knob:     $VAR in ${VALUE_LIST[*]} (baseline ${VALUE_LIST[0]})"
echo "  runs:     $RUNS, $SAMPLES samples, ${WINDOW}s window"
echo ""

if [[ -n "$EXE" ]]; then
    [[ -f "$EXE" ]] || { echo "Error: --exe '$EXE' does not exist." >&2; exit 1; }
    cp "$EXE" "$WORKDIR/exe"
else
    bash "$SCRIPT_DIR/bench_ab.sh" --build-only "$WORKDIR/exe" --bench "$BENCH" --features "$FEATURES"
fi
chmod +x "$WORKDIR/exe"
echo ""

# `full_id<TAB>mean_ns` for every row a pass measured; the mean is the first point
# estimate in the JSON object, which serde writes before the median.
snapshot() {
    local home="$1"
    find "$home" -path "*/new/estimates.json" -print 2>/dev/null | while IFS= read -r est; do
        local bm="${est%estimates.json}benchmark.json"
        [[ -f "$bm" ]] || continue
        local id mean
        id="$(awk 'match($0, /"full_id":"[^"]*"/) {
                print substr($0, RSTART + 11, RLENGTH - 12); exit
            }' "$bm")"
        mean="$(awk '{
                seg = $0
                cut = index(seg, "\"median\"")
                if (cut > 0) seg = substr(seg, 1, cut - 1)
                if (match(seg, /"point_estimate":[-0-9.eE+]+/)) {
                    print substr(seg, RSTART + 17, RLENGTH - 17)
                }
                exit
            }' "$est")"
        [[ -n "$id" && -n "$mean" ]] && printf '%s\t%s\n' "$id" "$mean"
    done
}

MERGED="$WORKDIR/merged.tsv"
: > "$MERGED"
measure_value() {
    local run="$1" value="$2"
    local home="$WORKDIR/crit-$run-$value"
    local log="$WORKDIR/pass-$run-$value.log"
    mkdir -p "$home"
    echo ">>> run $run, $VAR=$value"
    if ! env "$VAR=$value" CRITERION_HOME="$(native_path "$home")" PRISM_BENCH_SAMPLES="$SAMPLES" \
        "$WORKDIR/exe" --bench --warm-up-time 0.5 --measurement-time "$WINDOW" "$FILTER" > "$log" 2>&1; then
        cat "$log" >&2
        echo "Error: the bench binary failed at $VAR=$value." >&2
        exit 1
    fi
    local rows
    rows="$(snapshot "$home" | awk -v r="$run" -v v="$value" 'BEGIN { OFS = "\t" } { print r, v, $1, $2 }' | tee -a "$MERGED" | wc -l | tr -d '[:space:]')"
    if (( rows == 0 )); then
        echo "Error: no Criterion estimates for '$FILTER' at $VAR=$value." >&2
        exit 1
    fi
    echo "    $rows rows"
}

# A discarded pass warms the binary and the page cache before anything counts.
echo ">>> warmup (discarded)"
env "$VAR=${VALUE_LIST[0]}" CRITERION_HOME="$(native_path "$WORKDIR/warm")" PRISM_BENCH_SAMPLES="$SAMPLES" \
    "$WORKDIR/exe" --bench --warm-up-time 0.5 --measurement-time "$WINDOW" "$FILTER" > "$WORKDIR/warm.log" 2>&1 ||
    { cat "$WORKDIR/warm.log" >&2; echo "Error: the bench binary failed during warmup." >&2; exit 1; }
echo ""

for (( run = 1; run <= RUNS; run++ )); do
    if (( run % 2 == 1 )); then
        for value in "${VALUE_LIST[@]}"; do measure_value "$run" "$value"; done
    else
        for (( i = ${#VALUE_LIST[@]} - 1; i >= 0; i-- )); do measure_value "$run" "${VALUE_LIST[$i]}"; done
    fi
    echo ""
done

{
    echo "## Knob sweep: \`$VAR\` over \`$BENCH\`"
    echo ""
    echo "| Setting | Value |"
    echo "| --- | --- |"
    echo "| Filter | \`$FILTER\` |"
    echo "| Values | ${VALUE_LIST[*]} (baseline ${VALUE_LIST[0]}) |"
    echo "| Runs | $RUNS, forward then reversed, one measurement per value per run |"
    echo "| Samples | $SAMPLES per row, ${WINDOW}s window (triage tier) |"
    echo "| Features | \`$FEATURES\` |"
    echo "| Commit | $(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown) |"
    echo "| CPU | $(host_cpu) |"
    echo "| Caches | $(host_caches) |"
    echo "| OS | $(uname -sr) $(uname -m) |"
    echo "| Toolchain | $(rustc --version) |"
    echo ""
    echo "Each cell is the minimum over runs, with the value's own spread (max over min)"
    echo "in parentheses. Change is against the baseline column."
    echo ""
    awk -F'\t' -v values="${VALUE_LIST[*]}" '
        {
            run = $1; value = $2; id = $3; mean = $4
            key = id SUBSEP value
            if (!(key in lo) || mean < lo[key]) lo[key] = mean
            if (!(key in hi) || mean > hi[key]) hi[key] = mean
            if (!(id in seen)) { seen[id] = 1; order[++n] = id }
        }
        function fmt(ns) {
            if (ns >= 1e9) return sprintf("%.3f s", ns / 1e9)
            if (ns >= 1e6) return sprintf("%.2f ms", ns / 1e6)
            if (ns >= 1e3) return sprintf("%.2f us", ns / 1e3)
            return sprintf("%.0f ns", ns)
        }
        END {
            m = split(values, v, " ")
            printf "| Benchmark |"
            for (j = 1; j <= m; j++) printf " %s%s |", v[j], (j == 1 ? " (baseline)" : "")
            printf "\n| --- |"
            for (j = 1; j <= m; j++) printf " ---: |"
            printf "\n"
            for (i = 1; i <= n; i++) {
                id = order[i]
                base = lo[id SUBSEP v[1]]
                printf "| `%s` |", id
                for (j = 1; j <= m; j++) {
                    key = id SUBSEP v[j]
                    if (!(key in lo)) { printf " n/a |"; continue }
                    spread = (hi[key] / lo[key] - 1) * 100
                    if (j == 1) printf " %s (%.1f%%) |", fmt(lo[key]), spread
                    else printf " %s %+.1f%% (%.1f%%) |", fmt(lo[key]), (lo[key] / base - 1) * 100, spread
                }
                printf "\n"
            }
        }' "$MERGED"
} > "$OUT"

echo "=== report: $OUT ==="
cat "$OUT"

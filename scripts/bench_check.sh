#!/usr/bin/env bash
# Benchmark analysis tool for PRISM-Q.
#
# Reads Criterion JSON data from target/criterion/, saves baselines,
# compares runs, and produces PR-ready tables with regression detection.
#
# Usage:
#   ./scripts/bench_check.sh save                            # save baseline (auto-dated)
#   ./scripts/bench_check.sh save --name "pre-fusion"        # save with custom name
#   ./scripts/bench_check.sh compare                         # compare against latest baseline
#   ./scripts/bench_check.sh compare --baseline "pre-fusion"
#   ./scripts/bench_check.sh compare --filter "qft"          # only QFT benchmarks
#   ./scripts/bench_check.sh table --baseline "pre-fusion"   # PR-ready markdown
#   ./scripts/bench_check.sh list                            # list saved baselines
#
# Environment variables:
#   REGRESSION_THRESHOLD  Override default 5% threshold
#
# Prerequisites:
#   - Run `cargo bench` first — this tool reads stored results.
#   - Requires: jq, bc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CRITERION_DIR="$PROJECT_DIR/target/criterion"
BASELINE_DIR="$PROJECT_DIR/bench_results/baselines"
THRESHOLD="${REGRESSION_THRESHOLD:-5.0}"
SOURCE="new"

# --- Argument parsing ---

COMMAND="${1:-compare}"
shift 2>/dev/null || true

NAME=""
BASELINE=""
FILTER=""
MARKDOWN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name|-n)      NAME="$2"; shift 2 ;;
        --baseline|-b)  BASELINE="$2"; shift 2 ;;
        --filter|-f)    FILTER="$2"; shift 2 ;;
        --threshold|-t) THRESHOLD="$2"; shift 2 ;;
        --source|-s)    SOURCE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# --- Helpers ---

check_deps() {
    for cmd in jq bc; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "Error: '$cmd' is required but not installed." >&2
            exit 1
        fi
    done
}

format_duration() {
    local ns="$1"
    if (( $(echo "$ns >= 1000000000" | bc -l) )); then
        printf "%.3f s" "$(echo "$ns / 1000000000" | bc -l)"
    elif (( $(echo "$ns >= 1000000" | bc -l) )); then
        printf "%.2f ms" "$(echo "$ns / 1000000" | bc -l)"
    elif (( $(echo "$ns >= 1000" | bc -l) )); then
        printf "%.2f us" "$(echo "$ns / 1000" | bc -l)"
    else
        printf "%.1f ns" "$ns"
    fi
}

get_criterion_estimates() {
    local subdir="${1:-new}"

    if [[ ! -d "$CRITERION_DIR" ]]; then
        echo "Error: No Criterion data at $CRITERION_DIR — run 'cargo bench' first." >&2
        exit 1
    fi

    # Find all estimates.json in the target subdirectory, emit JSON lines
    find "$CRITERION_DIR" -path "*/$subdir/estimates.json" -print0 | while IFS= read -r -d '' est_file; do
        local bench_dir
        bench_dir="$(dirname "$est_file")"
        local bm_file="$bench_dir/benchmark.json"
        [[ -f "$bm_file" ]] || continue

        local full_id mean_ns ci_lo ci_hi stddev
        full_id="$(jq -r '.full_id' "$bm_file")"
        mean_ns="$(jq -r '.mean.point_estimate' "$est_file")"
        ci_lo="$(jq -r '.mean.confidence_interval.lower_bound' "$est_file")"
        ci_hi="$(jq -r '.mean.confidence_interval.upper_bound' "$est_file")"
        stddev="$(jq -r '.std_dev.point_estimate' "$est_file")"

        echo "$full_id|$mean_ns|$ci_lo|$ci_hi|$stddev"
    done | sort
}

# --- Commands ---

cmd_save() {
    check_deps
    local name="${NAME:-$(date +%Y-%m-%d_%H%M%S)}"
    local out_file="$BASELINE_DIR/$name.json"

    mkdir -p "$BASELINE_DIR"

    local rust_ver cpu_name
    rust_ver="$(rustc --version 2>/dev/null || echo 'unknown')"
    cpu_name="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed 's/.*: //' || echo 'unknown')"

    local count=0
    local benchmarks="{"

    while IFS='|' read -r id mean ci_lo ci_hi stddev; do
        [[ -z "$id" ]] && continue
        if (( count > 0 )); then benchmarks+=","; fi
        benchmarks+="$(printf '"%s":{"mean_ns":%s,"ci_lo_ns":%s,"ci_hi_ns":%s,"stddev_ns":%s}' \
            "$id" "$mean" "$ci_lo" "$ci_hi" "$stddev")"
        count=$((count + 1))
    done < <(get_criterion_estimates "$SOURCE")

    benchmarks+="}"

    if (( count == 0 )); then
        echo "Error: No benchmark estimates found in '$SOURCE'. Run 'cargo bench' first." >&2
        exit 1
    fi

    local date_str
    date_str="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    jq -n \
        --arg name "$name" \
        --arg date "$date_str" \
        --arg rust "$rust_ver" \
        --arg cpu "$cpu_name" \
        --arg src "$SOURCE" \
        --argjson count "$count" \
        --argjson benchmarks "$benchmarks" \
        '{name:$name,date:$date,rust_version:$rust,cpu:$cpu,source:$src,count:$count,benchmarks:$benchmarks}' \
        > "$out_file"

    echo "=== Baseline saved ==="
    echo "  Name:       $name"
    echo "  Benchmarks: $count"
    echo "  File:       $out_file"
}

cmd_compare() {
    check_deps
    local baseline_name="${BASELINE:-$NAME}"

    # Resolve baseline file
    if [[ -z "$baseline_name" ]]; then
        if [[ ! -d "$BASELINE_DIR" ]]; then
            echo "Error: No baselines directory. Run './scripts/bench_check.sh save' first." >&2
            exit 1
        fi
        local latest
        latest="$(ls -t "$BASELINE_DIR"/*.json 2>/dev/null | head -1)"
        if [[ -z "$latest" ]]; then
            echo "Error: No baselines found." >&2
            exit 1
        fi
        baseline_name="$(basename "$latest" .json)"
    fi

    local base_file="$BASELINE_DIR/$baseline_name.json"
    if [[ ! -f "$base_file" ]]; then
        echo "Error: Baseline '$baseline_name' not found at $base_file" >&2
        exit 1
    fi

    local base_date
    base_date="$(jq -r '.date' "$base_file")"

    # Collect current estimates into an associative array
    declare -A curr_mean
    while IFS='|' read -r id mean ci_lo ci_hi stddev; do
        [[ -z "$id" ]] && continue
        curr_mean["$id"]="$mean"
    done < <(get_criterion_estimates "$SOURCE")

    if (( ${#curr_mean[@]} == 0 )); then
        echo "Error: No current benchmark data in '$SOURCE'. Run 'cargo bench' first." >&2
        exit 1
    fi

    # Build comparison
    local reg_count=0 imp_count=0 total=0
    local rows=()

    while IFS= read -r id; do
        [[ -z "$id" ]] && continue
        if [[ -n "$FILTER" ]] && ! echo "$id" | grep -qE "$FILTER"; then
            continue
        fi

        local base_mean
        base_mean="$(jq -r ".benchmarks.\"$id\".mean_ns // empty" "$base_file")"
        [[ -z "$base_mean" ]] && continue

        local cur="${curr_mean[$id]:-}"
        [[ -z "$cur" ]] && continue

        local pct
        pct="$(echo "scale=1; ($cur - $base_mean) * 100 / $base_mean" | bc -l)"

        local sign=""
        if (( $(echo "$pct >= 0" | bc -l) )); then sign="+"; fi

        rows+=("$id|$base_mean|$cur|$sign$pct%")
        total=$((total + 1))

        if (( $(echo "$pct > $THRESHOLD" | bc -l) )); then
            reg_count=$((reg_count + 1))
        elif (( $(echo "$pct < -$THRESHOLD" | bc -l) )); then
            imp_count=$((imp_count + 1))
        fi
    done < <(jq -r '.benchmarks | keys[]' "$base_file" | sort)

    if (( total == 0 )); then
        echo "No matching benchmarks found in both baseline and current data."
        exit 0
    fi

    if [[ "$MARKDOWN" == "true" ]]; then
        echo ""
        echo "| Benchmark | Before | After | Change |"
        echo "|-----------|--------|-------|--------|"
        for row in "${rows[@]}"; do
            IFS='|' read -r id before after change <<< "$row"
            printf "| \`%s\` | %s | %s | %s |\n" "$id" "$(format_duration "$before")" "$(format_duration "$after")" "$change"
        done
        echo ""
        local verdict="PASS"
        if (( reg_count > 0 )); then verdict="FAIL"; fi
        echo "**Regression verdict**: $verdict (threshold: ${THRESHOLD}%)"
        return
    fi

    # Console output
    echo "=== PRISM-Q Benchmark Comparison ==="
    echo "  Baseline:  $baseline_name ($base_date)"
    echo "  Threshold: ${THRESHOLD}%"
    echo ""

    local max_name=9
    for row in "${rows[@]}"; do
        local id="${row%%|*}"
        local len=${#id}
        if (( len > max_name )); then max_name=$len; fi
    done
    if (( max_name > 50 )); then max_name=50; fi

    printf "%-${max_name}s  %12s  %12s  %9s\n" "Benchmark" "Before" "After" "Change"
    printf '%*s\n' $((max_name + 40)) '' | tr ' ' '-'

    for row in "${rows[@]}"; do
        IFS='|' read -r id before after change <<< "$row"
        printf "%-${max_name}s  %12s  %12s  %9s\n" \
            "$id" "$(format_duration "$before")" "$(format_duration "$after")" "$change"
    done

    echo ""
    echo "  $total benchmarks | $reg_count regressions | $imp_count improvements | $((total - reg_count - imp_count)) unchanged"

    if (( reg_count > 0 )); then
        echo ""
        echo "VERDICT: FAIL"
        echo "  $reg_count benchmark(s) regressed beyond ${THRESHOLD}%"
        exit 1
    else
        echo ""
        echo "VERDICT: PASS"
        exit 0
    fi
}

cmd_list() {
    if [[ ! -d "$BASELINE_DIR" ]]; then
        echo "No baselines saved yet. Run './scripts/bench_check.sh save' first."
        exit 0
    fi

    local files
    files="$(ls -t "$BASELINE_DIR"/*.json 2>/dev/null || true)"
    if [[ -z "$files" ]]; then
        echo "No baselines saved yet."
        exit 0
    fi

    echo "=== Saved Baselines ==="
    echo ""
    while IFS= read -r f; do
        local name date count
        name="$(basename "$f" .json)"
        date="$(jq -r '.date' "$f")"
        count="$(jq '.benchmarks | length' "$f")"
        printf "  %-30s  %s  (%d benchmarks)\n" "$name" "$date" "$count"
    done <<< "$files"
}

# --- Dispatch ---

case "$COMMAND" in
    save)    cmd_save ;;
    compare) cmd_compare ;;
    table)   MARKDOWN=true; cmd_compare ;;
    list)    cmd_list ;;
    *)       echo "Unknown command: $COMMAND. Use: save, compare, table, list" >&2; exit 1 ;;
esac

#!/usr/bin/env bash
# Run the benchmark subset used by the CI regression gate.
#
# This intentionally covers representative hot paths without running the full
# Criterion suite. Use `CI_BENCH_FEATURES` to override the default feature set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PRISM_BENCH_PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
FEATURES="${CI_BENCH_FEATURES:-parallel,bench-fast}"

cd "$PROJECT_DIR"

echo "=== PRISM-Q CI benchmark subset ==="
echo "Features: $FEATURES"
echo ""

rm -rf "$PROJECT_DIR/target/criterion"

echo ">>> cargo bench --bench circuits --features $FEATURES --no-run"
cargo bench --bench circuits --features "$FEATURES" --no-run
echo ""

count_estimates() {
    if [[ ! -d "$PROJECT_DIR/target/criterion" ]]; then
        echo 0
        return
    fi

    find "$PROJECT_DIR/target/criterion" -path "*/new/estimates.json" | wc -l | tr -d ' '
}

bench_executable() {
    local bench="$1"
    local deps_dir="$PROJECT_DIR/target/release/deps"
    local candidate
    local newest=""
    local newest_mtime=0
    local mtime

    shopt -s nullglob
    for candidate in "$deps_dir"/"$bench"-* "$deps_dir"/"$bench"-*.exe; do
        [[ -f "$candidate" && -x "$candidate" ]] || continue
        mtime="$(stat -c "%Y" "$candidate" 2>/dev/null || stat -f "%m" "$candidate")"
        if (( mtime > newest_mtime )); then
            newest="$candidate"
            newest_mtime="$mtime"
        fi
    done
    shopt -u nullglob

    if [[ -z "$newest" ]]; then
        echo "Error: benchmark executable not found for $bench in $deps_dir" >&2
        exit 1
    fi

    echo "$newest"
}

run_bench() {
    local bench="$1"
    local filter="$2"
    local expected_count="$3"
    local exe
    local before_count
    local after_count
    local added_count

    exe="$(bench_executable "$bench")"
    before_count="$(count_estimates)"

    echo ">>> $exe --bench $filter"
    "$exe" --bench "$filter"
    after_count="$(count_estimates)"
    added_count=$((after_count - before_count))
    if (( added_count < expected_count )); then
        echo "Error: benchmark filter produced $added_count Criterion estimates, expected at least $expected_count:" >&2
        echo "  $bench $filter" >&2
        exit 1
    fi
    echo ""
}

# Keep CI filters on benchmark IDs that exist on the base branch. All selected
# circuits are at least 22 qubits.
CIRCUITS_FILTER="^(statevector/(scalability_d5/22|qft_textbook/22|qpe_t_gate/22q)"
CIRCUITS_FILTER+="|stabilizer/scaling/1000"
CIRCUITS_FILTER+="|auto/qft_textbook/22"
CIRCUITS_FILTER+="|compiled_sampler/noiseless/noiseless_1000q_10k)$"

run_bench "circuits" "$CIRCUITS_FILTER" 6

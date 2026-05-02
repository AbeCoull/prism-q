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

count_estimates() {
    if [[ ! -d "$PROJECT_DIR/target/criterion" ]]; then
        echo 0
        return
    fi

    find "$PROJECT_DIR/target/criterion" -path "*/new/estimates.json" | wc -l | tr -d ' '
}

run_bench() {
    local bench="$1"
    local filter="$2"
    local before_count
    local after_count

    before_count="$(count_estimates)"

    echo ">>> cargo bench --bench $bench --features $FEATURES -- $filter"
    cargo bench --bench "$bench" --features "$FEATURES" -- "$filter"
    after_count="$(count_estimates)"
    if (( after_count <= before_count )); then
        echo "Error: benchmark filter produced no Criterion estimates: $bench $filter" >&2
        exit 1
    fi
    echo ""
}

BENCH_DRIVER_FILTERS=(
    "single_qubit_gates/h_gate/20"
    "two_qubit_gates/cx_gate/20"
    "measurement/measure_superposition/20"
    "e2e_qasm/ghz_5"
)

CIRCUITS_FILTERS=(
    "statevector/random_d10/20"
    "statevector/qft_textbook/20"
    "statevector/qaoa_l3/20"
    "stabilizer/scaling/1000"
    "auto/clifford_d10/20"
    "compiled_sampler/noiseless/noiseless_1000q_10k"
)

for filter in "${BENCH_DRIVER_FILTERS[@]}"; do
    run_bench "bench_driver" "$filter"
done

for filter in "${CIRCUITS_FILTERS[@]}"; do
    run_bench "circuits" "$filter"
done

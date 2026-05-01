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

run_bench() {
    local bench="$1"
    local filter="$2"

    echo ">>> cargo bench --bench $bench --features $FEATURES -- $filter"
    cargo bench --bench "$bench" --features "$FEATURES" -- "$filter"
    echo ""
}

BENCH_DRIVER_FILTERS=(
    "single_qubit_gates/h_gate"
    "two_qubit_gates/cx_gate"
    "measurement/measure_superposition"
    "e2e_qasm"
)

CIRCUITS_FILTERS=(
    "statevector/random_d10/16"
    "statevector/qft_textbook/16"
    "statevector/qaoa_l3/16"
    "stabilizer/scaling/1000"
    "auto/clifford_d10/20"
    "compiled_sampler/noiseless"
)

for filter in "${BENCH_DRIVER_FILTERS[@]}"; do
    run_bench "bench_driver" "$filter"
done

for filter in "${CIRCUITS_FILTERS[@]}"; do
    run_bench "circuits" "$filter"
done

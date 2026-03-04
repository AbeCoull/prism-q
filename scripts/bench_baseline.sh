#!/usr/bin/env bash
# Save a benchmark baseline for future regression comparison.
#
# Usage:
#   ./scripts/bench_baseline.sh              # full suite
#   ./scripts/bench_baseline.sh --quick      # smoke subset only
#
# Output: baseline stored in target/criterion/ (Criterion's default location).
# Copy target/criterion/ to an artifact store for CI use.

set -euo pipefail

BASELINE_NAME="${BASELINE_NAME:-prism_baseline}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "=== PRISM-Q Benchmark Baseline ==="
echo "Baseline name: $BASELINE_NAME"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

if [[ "${1:-}" == "--quick" ]]; then
    echo "Running smoke subset..."
    cargo bench --bench bench_driver -- \
        --save-baseline "$BASELINE_NAME" \
        --warm-up-time 2 \
        --measurement-time 5 \
        "single_qubit_gates/h_gate" \
        "two_qubit_gates/cx_gate" \
        "e2e_qasm"
else
    echo "Running full benchmark suite..."
    cargo bench -- --save-baseline "$BASELINE_NAME"
fi

echo ""
echo "=== Baseline saved as '$BASELINE_NAME' ==="
echo "Location: target/criterion/"
echo ""
echo "To compare against this baseline later:"
echo "  ./scripts/bench_compare.sh"

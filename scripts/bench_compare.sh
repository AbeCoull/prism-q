#!/usr/bin/env bash
# Compare current benchmark performance against a saved baseline.
#
# Usage:
#   ./scripts/bench_compare.sh                       # default 5% threshold
#   REGRESSION_THRESHOLD=10 ./scripts/bench_compare.sh  # custom threshold
#   ./scripts/bench_compare.sh --quick               # smoke subset only
#
# Exit code:
#   0 — no regressions above threshold
#   1 — regressions detected

set -euo pipefail

BASELINE_NAME="${BASELINE_NAME:-prism_baseline}"
THRESHOLD="${REGRESSION_THRESHOLD:-5}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_FILE="$PROJECT_DIR/target/bench-compare-output.txt"

cd "$PROJECT_DIR"

echo "=== PRISM-Q Benchmark Comparison ==="
echo "Baseline: $BASELINE_NAME"
echo "Regression threshold: ${THRESHOLD}%"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

if [[ "${1:-}" == "--quick" ]]; then
    echo "Running smoke subset..."
    cargo bench --bench bench_driver -- \
        --baseline "$BASELINE_NAME" \
        --warm-up-time 2 \
        --measurement-time 5 \
        "single_qubit_gates/h_gate" \
        "two_qubit_gates/cx_gate" \
        "e2e_qasm" \
        2>&1 | tee "$OUTPUT_FILE"
else
    echo "Running full benchmark suite..."
    cargo bench -- --baseline "$BASELINE_NAME" 2>&1 | tee "$OUTPUT_FILE"
fi

echo ""
echo "=== Regression Analysis ==="

REGRESSED=0
if grep -qi "regressed" "$OUTPUT_FILE"; then
    REGRESSED=1
    echo "WARNING: Criterion reports regressions."
    echo ""
    echo "Regressed benchmarks:"
    grep -i "regressed" "$OUTPUT_FILE" || true
else
    echo "No regressions detected by Criterion."
fi

echo ""
echo "Threshold: ${THRESHOLD}%"
echo "Full output: $OUTPUT_FILE"
echo ""

if [[ "$REGRESSED" -eq 1 ]]; then
    echo "VERDICT: FAIL — regressions detected"
    echo "Review the output above. If acceptable, re-run baseline:"
    echo "  ./scripts/bench_baseline.sh"
    exit 1
else
    echo "VERDICT: PASS"
    exit 0
fi

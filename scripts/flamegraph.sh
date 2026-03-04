#!/usr/bin/env bash
# Generate a flamegraph for a specific Criterion benchmark.
#
# Prerequisites:
#   cargo install flamegraph
#   Linux: perf (apt install linux-tools-common linux-tools-$(uname -r))
#   macOS: dtrace (built-in; may need SIP partial disable for kernel stacks)
#
# Usage:
#   ./scripts/flamegraph.sh "qft_textbook/16"
#   ./scripts/flamegraph.sh "hea_l5/20" --bench-file circuits
#   PROFILE_TIME=30 ./scripts/flamegraph.sh "qft_textbook/22"
#
# Output: bench_results/flamegraph-<sanitized-filter>.svg
#
# The bench profile (Cargo.toml) already has debug=true and strip="none",
# so symbol names are preserved in the flamegraph.

set -euo pipefail

FILTER="${1:?Usage: $0 <benchmark-filter> [--bench-file <name>]}"
BENCH_FILE="circuits"
PROFILE_TIME="${PROFILE_TIME:-10}"

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench-file)
            BENCH_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_DIR/bench_results"

mkdir -p "$OUTPUT_DIR"

# Sanitize filter for filename (replace / and spaces with -)
SAFE_NAME="$(echo "$FILTER" | tr '/ ' '--')"
OUTPUT_FILE="$OUTPUT_DIR/flamegraph-${SAFE_NAME}.svg"

echo "=== PRISM-Q Flamegraph ==="
echo "Benchmark: $BENCH_FILE"
echo "Filter:    $FILTER"
echo "Duration:  ${PROFILE_TIME}s"
echo "Output:    $OUTPUT_FILE"
echo ""

cd "$PROJECT_DIR"

cargo flamegraph \
    --bench "$BENCH_FILE" \
    --features parallel \
    -o "$OUTPUT_FILE" \
    -- --bench --profile-time "$PROFILE_TIME" "$FILTER"

echo ""
echo "=== Flamegraph saved ==="
echo "Open in a browser: $OUTPUT_FILE"

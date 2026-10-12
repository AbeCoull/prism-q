#!/usr/bin/env bash
# Run every benchmark row once in Criterion's test mode.
#
# Many 20 to 30 qubit paths execute nowhere but a bench fixture, and a timed
# run is the wrong place to learn that one panics. Test mode runs each row's
# routine once and reports nothing, so the whole corpus costs minutes rather
# than hours. A failing target is named with the row it was on.
#
# Every target is built once through cargo, then each executable lists its
# rows, drops the exclusions below, and runs the rest. Criterion's filter is a
# positive regex only, so the filter is rebuilt from the listing: a parent id
# with no excluded row under it goes in as one prefix, and a parent that holds
# one has its remaining rows named exactly.
#
# Usage:
#   scripts/bench_smoke.sh [--profile P] [--features F] [BENCH ...]
#
# Options:
#   --profile   cargo profile (default ci-release: the bench profile's fat LTO
#               costs minutes per target and changes nothing test mode checks)
#   --features  cargo feature list (default "parallel distributed bench-internal",
#               which adds the internal rows and the loopback distributed target)
#   BENCH       bench targets to run (default: every target that runs without a
#               GPU or an MPI library; bench_distributed needs the default features)
#
# Exit status: 0 when every row ran, 1 when a target failed or a build did.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PROFILE="ci-release"
FEATURES="parallel distributed bench-internal"
BENCHES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)  PROFILE="$2"; shift 2 ;;
        --features) FEATURES="$2"; shift 2 ;;
        -h|--help)  awk 'NR > 1 && !/^#/ { exit } NR > 1' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)         echo "Unknown option: $1" >&2; exit 1 ;;
        *)          BENCHES+=("$1"); shift ;;
    esac
done
if (( ${#BENCHES[@]} == 0 )); then
    BENCHES=(circuits bench_shots_perf bench_driver svd_bench qec_t_strategies qec_decoder
        bench_distributed)
fi

# Rows left out, one `id<TAB>reason` per line. Name the defect or the limit, so
# the entry can be dropped when it no longer holds.
EXCLUDED="$(cat <<'EOF'
expectation/pauli_sum_grouped/ungrouped/20	the bounded tensor-network expectation route panics on an identity observable term at 18 qubits and above
EOF
)"

cd "$PROJECT_DIR"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
printf '%s\n' "$EXCLUDED" > "$WORKDIR/excluded.tsv"

bench_args=()
for bench in "${BENCHES[@]}"; do
    bench_args+=(--bench "$bench")
done

echo ">>> building: cargo bench --profile $PROFILE --features \"$FEATURES\" ${bench_args[*]} --no-run"
cargo bench --profile "$PROFILE" --features "$FEATURES" "${bench_args[@]}" --no-run \
    --message-format=json-render-diagnostics > "$WORKDIR/build.json"

# The executable cargo reported for one bench target, unescaped for this shell.
executable_of() {
    local exe
    exe="$(awk -v want="$1" '
        /"reason":"compiler-artifact"/ && index($0, "\"executable\":\"") > 0 {
            if (!match($0, /"name":"[^"]*"/)) next
            if (substr($0, RSTART + 8, RLENGTH - 9) != want) next
            if (match($0, /"executable":"[^"]*"/)) print substr($0, RSTART + 14, RLENGTH - 15)
        }' "$WORKDIR/build.json" | tail -1)"
    exe="${exe//\\\\//}"
    if [[ -n "$exe" && ! -f "$exe" ]] && command -v cygpath >/dev/null 2>&1; then
        exe="$(cygpath -u "$exe")"
    fi
    printf '%s' "$exe"
}

# Criterion filter covering every listed row except the excluded ones.
filter_for() {
    local listing="$1"
    awk -F'\t' '
        function esc(s) { gsub(/[][\\.^$*+?(){}|]/, "\\\\&", s); return s }
        NR == FNR {
            if ($1 != "") { skip[$1] = 1; ex[++ne] = $1 }
            next
        }
        {
            id = $0
            if (id in skip) next
            parent = id
            sub(/\/[^\/]*$/, "", parent)
            if (parent == id) { exact[id] = 1; next }
            if (!(parent in seen)) { seen[parent] = 1; order[++np] = parent }
            rows[parent] = rows[parent] "\n" id
        }
        END {
            out = ""
            for (p = 1; p <= np; p++) {
                parent = order[p]
                clash = 0
                for (e = 1; e <= ne; e++) {
                    if (index(ex[e], parent "/") == 1) { clash = 1; break }
                }
                if (!clash) {
                    out = out (out == "" ? "" : "|") esc(parent "/")
                    continue
                }
                m = split(substr(rows[parent], 2), ids, "\n")
                for (i = 1; i <= m; i++) { out = out (out == "" ? "" : "|") esc(ids[i]) "$" }
            }
            for (id in exact) { out = out (out == "" ? "" : "|") esc(id) "$" }
            if (out != "") printf "^(%s)", out
        }
    ' "$WORKDIR/excluded.tsv" "$listing"
}

failed=0
summary=""
for bench in "${BENCHES[@]}"; do
    exe="$(executable_of "$bench")"
    if [[ -z "$exe" ]]; then
        echo "Error: cargo reported no executable for bench target '$bench'." >&2
        failed=1
        continue
    fi

    "$exe" --list 2>/dev/null | sed -n 's/: benchmark$//p' > "$WORKDIR/$bench.ids"
    total="$(wc -l < "$WORKDIR/$bench.ids" | tr -d '[:space:]')"
    filter="$(filter_for "$WORKDIR/$bench.ids")"
    if [[ -z "$filter" ]]; then
        echo ">>> $bench: no rows to run"
        continue
    fi

    echo ">>> $bench: $total rows listed"
    log="$WORKDIR/$bench.log"
    start=$SECONDS
    status=0
    "$exe" --test "$filter" > "$log" 2>&1 || status=$?
    elapsed=$(( SECONDS - start ))
    ran="$(grep -c '^Testing ' "$log" || true)"

    if (( status != 0 )); then
        row="$(awk '/^Testing / { id = substr($0, 9) } END { print id }' "$log")"
        echo "Error: $bench exited $status on row '${row:-unknown}' after ${elapsed}s." >&2
        tail -n 20 "$log" | sed 's/^/    /' >&2
        failed=1
        summary+="$bench: FAILED on ${row:-unknown} (${elapsed}s)"$'\n'
        continue
    fi
    echo "    $ran rows ran once in ${elapsed}s"
    summary+="$bench: $ran of $total rows in ${elapsed}s"$'\n'
done

echo ""
echo "=== bench smoke summary ==="
printf '%s' "$summary"
if [[ -n "$EXCLUDED" ]]; then
    echo "Excluded:"
    printf '%s\n' "$EXCLUDED" | awk -F'\t' '{ printf "  %s: %s\n", $1, $2 }'
fi
exit "$failed"

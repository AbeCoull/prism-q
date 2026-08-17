#!/usr/bin/env bash
# Adjacent-binary A/B benchmark comparison.
#
# The gating method for any before/after claim on a development host. Separate
# `cargo bench` invocations minutes apart are not a valid A/B: the rebuild and
# whatever else the machine does between them drift the measurement by more than
# the 5% regression gate, and a byte-identical control group has read as much as
# +98% under that pattern. This script removes the build and the source edit from
# between the two measurements.
#
#   1. Build the bench binary from the working tree and copy it aside.
#   2. Build the same bench binary from a reference git ref in a separate
#      worktree and copy it aside.
#   3. Verify the working tree did not change between the two builds.
#   4. Run the two binaries adjacent, no build in between: one discarded warmup
#      pass each, then four measured passes.
#   5. Emit a markdown table with a same-code control column per row.
#
# The measured pass order is ref, new, new, ref. Both means are centred on the
# same point in time, so linear drift cancels, and each binary is measured twice
# so every row carries its own noise floor. A change smaller than that floor is
# reported as noise, not as a win.
#
# Usage:
#   scripts/bench_ab.sh --filter '^factored/noise_kraus/'
#   scripts/bench_ab.sh -f '^density_matrix/' -r main -b circuits
#   scripts/bench_ab.sh -f '^sparse/' --ref-dir /tmp/prism-q-ref   # reuse the build
#
# Options:
#   --filter,   -f  Criterion filter regex, applied to every pass (required)
#   --ref,      -r  git ref for the reference build (default: HEAD)
#   --bench,    -b  bench target (default: circuits)
#   --features      cargo feature list (default: parallel)
#   --ref-dir       reference worktree path. Persists between runs, so the
#                   reference build is cached. Default: a temporary directory
#                   removed on exit.
#   --ref-exe       prebuilt reference executable. Skips the reference build and
#                   the worktree entirely, so only the working tree is compiled.
#                   Takes precedence over --ref-dir. The caller owns the claim
#                   that the executable was built from --ref: nothing here can
#                   check it, so key whatever cache supplies it on the ref sha,
#                   the toolchain, and the feature list. The sha already pins the
#                   lockfile it was built from.
#   --build-only    build the working-tree bench binary, copy it to this path,
#                   and exit without measuring. The producer side of --ref-exe:
#                   the build runs through the same code path as the A/B's own
#                   build, so a binary cached by one is what the other expects.
#                   --filter is not required in this mode.
#   --min-rows      fail when fewer than this many rows appear in all four
#                   passes, catching a filter that stopped matching a renamed
#                   benchmark id. Default: 1.
#   --out           markdown output path (default: bench_results/ab-<stamp>.md)
#
# Environment:
#   REGRESSION_THRESHOLD   regression gate in percent (default: 5.0)
#   PRISM_BENCH_SAMPLES    samples per row, recorded in the report
#   PRISM_BENCH_PLOTS      set to render Criterion HTML (about 58s per row)
#   PRISM_BENCH_HIGH_QUBITS  set to admit the 28q/30q statevector rows
#
# Requires git, awk, and cargo. Deliberately not jq or bc: neither is present on
# the reference host, which is why the stored-baseline tooling never ran there.
#
# Batch every row you need into one invocation. Touching `src/lib.rs` rebuilds
# the `circuits` target in about 202s because `[profile.bench]` inherits
# `lto = "fat"` with `codegen-units = 1`, and filtering does not avoid the relink.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

FILTER=""
REF="HEAD"
BENCH="circuits"
FEATURES="parallel"
REF_DIR=""
REF_EXE=""
BUILD_ONLY=""
OUT=""
MIN_ROWS=1
THRESHOLD="${REGRESSION_THRESHOLD:-5.0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter|-f)   FILTER="$2"; shift 2 ;;
        --ref|-r)      REF="$2"; shift 2 ;;
        --bench|-b)    BENCH="$2"; shift 2 ;;
        --features)    FEATURES="$2"; shift 2 ;;
        --ref-dir)     REF_DIR="$2"; shift 2 ;;
        --ref-exe)     REF_EXE="$2"; shift 2 ;;
        --build-only)  BUILD_ONLY="$2"; shift 2 ;;
        --min-rows)    MIN_ROWS="$2"; shift 2 ;;
        --out)         OUT="$2"; shift 2 ;;
        --threshold|-t) THRESHOLD="$2"; shift 2 ;;
        -h|--help)     awk 'NR > 1 && !/^#/ { exit } NR > 1' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$FILTER" && -z "$BUILD_ONLY" ]]; then
    echo "Error: --filter is required. Name the rows to compare." >&2
    exit 1
fi

cd "$PROJECT_DIR"

for cmd in git awk cargo; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: '$cmd' not found." >&2; exit 1; }
done

WORKDIR="$(mktemp -d)"
KEEP_REF_DIR=true
if [[ -z "$REF_DIR" ]]; then
    REF_DIR="$WORKDIR/ref"
    KEEP_REF_DIR=false
fi

cleanup() {
    if [[ "$KEEP_REF_DIR" == "false" && -d "$REF_DIR" ]]; then
        git worktree remove --force "$REF_DIR" 2>/dev/null || true
    fi
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

# Windows-native path for variables read by cargo and by the bench binary. MSYS
# converts arguments but not arbitrary environment variables, so a POSIX path in
# CARGO_TARGET_DIR or CRITERION_HOME reaches the native binary unusable.
native_path() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1"
    else
        printf '%s' "$1"
    fi
}

# Content hash of the working tree: HEAD, every tracked modification against it,
# and every untracked file that is not ignored. Recomputed after the reference
# build so a save from the editor between the two builds is caught rather than
# silently pairing two binaries that no longer differ by the change under test.
tree_fingerprint() {
    {
        git rev-parse HEAD
        git diff HEAD --binary
        git ls-files -o --exclude-standard -z | while IFS= read -r -d '' f; do
            printf '%s ' "$f"
            git hash-object "$f"
        done
    } | git hash-object --stdin
}

# Build the bench target in `dir` and copy the resulting executable to `out`.
# The path comes from cargo's own artifact message, not from an mtime scan of
# the deps directory, so a build that relinks nothing still resolves correctly.
#
# Each worktree builds into its own target directory. Sharing one directory
# looks like a free dependency cache and is not: cargo derives the same unit
# metadata for both worktrees, so the second build overwrites the first's
# artifact and fingerprint, and every later build reports "Finished" in under a
# second while handing back whichever binary was linked last. The report then
# reads as a pure noise floor, because both binaries really are the same one.
# The reference worktree keeps its cache between runs whenever --ref-dir names
# a path that persists.
build_bench_exe() {
    local dir="$1" out="$2" label="$3"
    local log="$WORKDIR/build-$label.json"

    echo ">>> building $label: cargo bench --bench $BENCH --features \"$FEATURES\" --no-run"
    (
        cd "$dir"
        CARGO_TARGET_DIR="$(native_path "$dir/target")" \
            cargo bench --bench "$BENCH" --features "$FEATURES" --no-run --message-format=json
    ) > "$log"

    local exe
    exe="$(awk -v want="$BENCH" '
        /"reason":"compiler-artifact"/ && index($0, "\"executable\":\"") > 0 {
            name = ""
            if (match($0, /"name":"[^"]*"/)) {
                name = substr($0, RSTART + 8, RLENGTH - 9)
            }
            if (name != want) next
            if (match($0, /"executable":"[^"]*"/)) {
                print substr($0, RSTART + 14, RLENGTH - 15)
            }
        }' "$log" | tail -1)"

    if [[ -z "$exe" ]]; then
        echo "Error: cargo reported no executable for bench target '$BENCH'." >&2
        echo "  artifact log: $log" >&2
        exit 1
    fi

    # Cargo emits an escaped Windows path in JSON; unescape before touching it.
    exe="${exe//\\\\//}"
    if [[ ! -f "$exe" ]] && command -v cygpath >/dev/null 2>&1; then
        exe="$(cygpath -u "$exe")"
    fi
    if [[ ! -f "$exe" ]]; then
        echo "Error: bench executable '$exe' does not exist." >&2
        exit 1
    fi

    cp "$exe" "$out"
    echo "    $out"
}

# Extract `full_id<TAB>mean_ns` for every row a pass measured. The mean is the
# first point estimate in the JSON object, which serde writes before the median.
snapshot() {
    local home="$1" out="$2"

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
    done | sort > "$out"
}

run_pass() {
    local idx="$1" label="$2" exe="$3"
    local home="$WORKDIR/crit-$idx"
    mkdir -p "$home"

    echo ">>> pass $idx ($label)"
    CRITERION_HOME="$(native_path "$home")" "$exe" --bench "$FILTER"

    snapshot "$home" "$WORKDIR/pass-$idx.tsv"
    local rows
    rows="$(wc -l < "$WORKDIR/pass-$idx.tsv" | tr -d '[:space:]')"
    if (( rows == 0 )); then
        echo "Error: pass $idx produced no Criterion estimates under $home." >&2
        echo "  Either --filter '$FILTER' matched nothing, or CRITERION_HOME was ignored." >&2
        exit 1
    fi
    echo "    pass $idx measured $rows rows"
    echo ""
}

# One discarded pass per binary before anything is recorded.
#
# Without them the first measured pass absorbs the whole cold start (page faults
# on the freshly linked executable, cache and turbo state after two builds) and
# whichever binary owns it looks slow. On the reference host the first ordering
# tried put the reference binary in pass 1 and its same-code control read -17.1%,
# -15.2%, and -18.4% on three rows while the second binary's control read within
# 1%: a systematic penalty against the earlier binary, not host noise. Both
# binaries now enter the measured passes with the same warm history.
warm_up() {
    local idx="$1" exe="$2"
    local home="$WORKDIR/warm-$idx"
    mkdir -p "$home"
    echo ">>> warmup $idx (discarded)"
    CRITERION_HOME="$(native_path "$home")" "$exe" --bench "$FILTER" > /dev/null
    echo ""
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

# --- Build both binaries ---

if [[ -n "$BUILD_ONLY" ]]; then
    mkdir -p "$(dirname "$BUILD_ONLY")"
    build_bench_exe "$PROJECT_DIR" "$BUILD_ONLY" "build-only"
    exit 0
fi

REF_SHA="$(git rev-parse --short "$REF")"
FINGERPRINT_BEFORE="$(tree_fingerprint)"

echo "=== PRISM-Q adjacent-binary A/B ==="
echo "  bench:     $BENCH"
echo "  filter:    $FILTER"
echo "  features:  $FEATURES"
echo "  reference: $REF ($REF_SHA)"
echo "  threshold: ${THRESHOLD}%"
echo ""

build_bench_exe "$PROJECT_DIR" "$WORKDIR/exe-new" "new"

if [[ -n "$REF_EXE" ]]; then
    if [[ ! -f "$REF_EXE" ]]; then
        echo "Error: --ref-exe '$REF_EXE' does not exist." >&2
        exit 1
    fi
    echo ">>> using the supplied reference executable $REF_EXE"
    cp "$REF_EXE" "$WORKDIR/exe-ref"
    chmod +x "$WORKDIR/exe-ref"
    REF_PROVENANCE="supplied via \`--ref-exe\`, not built here"
else
    if [[ -d "$REF_DIR/.git" || -f "$REF_DIR/.git" ]]; then
        echo ">>> reusing reference worktree $REF_DIR"
        git -C "$REF_DIR" checkout --detach --force "$REF_SHA" >/dev/null 2>&1
        git -C "$REF_DIR" clean -fdq
    else
        echo ">>> adding reference worktree $REF_DIR at $REF_SHA"
        git worktree add --detach --force "$REF_DIR" "$REF_SHA" >/dev/null
    fi

    build_bench_exe "$REF_DIR" "$WORKDIR/exe-ref" "ref"
    REF_PROVENANCE="built from a worktree at \`$REF_DIR\`"

    # Only meaningful when a second build follows the first: it catches an editor
    # save landing between them, which would leave two binaries that no longer
    # differ by the change under test.
    FINGERPRINT_AFTER="$(tree_fingerprint)"
    if [[ "$FINGERPRINT_BEFORE" != "$FINGERPRINT_AFTER" ]]; then
        echo "Error: the working tree changed between the two builds." >&2
        echo "  The two binaries no longer differ by the change under test, so the" >&2
        echo "  comparison would be meaningless. Re-run with the tree settled." >&2
        exit 1
    fi
fi

# The two binaries are never byte identical even from identical sources: the link
# timestamp and the PDB GUID differ on every link. Same code is decided from git
# instead. Those bytes are all that differs, measured 2026-08-17: the same commit
# built in two directories has byte-identical `.text`, so the build directory is
# not a reason to distrust a row on this target.
if [[ -z "$(git status --porcelain)" && "$(git rev-parse HEAD)" == "$(git rev-parse "$REF_SHA")" ]]; then
    echo "Note: the working tree matches $REF exactly, so both binaries carry the"
    echo "      same code and every row is a control row. This is the same-code"
    echo "      noise floor of the host, with no change under test."
    echo ""
fi

# --- Measure ---

echo "=== two discarded warmup passes, then four adjacent passes ==="
echo ""
warm_up 1 "$WORKDIR/exe-ref"
warm_up 2 "$WORKDIR/exe-new"
run_pass 1 "ref" "$WORKDIR/exe-ref"
run_pass 2 "new" "$WORKDIR/exe-new"
run_pass 3 "new" "$WORKDIR/exe-new"
run_pass 4 "ref" "$WORKDIR/exe-ref"

# --- Report ---

MERGED="$WORKDIR/merged.tsv"
: > "$MERGED"
for idx in 1 2 3 4; do
    awk -v p="$idx" 'BEGIN { OFS = "\t" } { print p, $1, $2 }' "$WORKDIR/pass-$idx.tsv" >> "$MERGED"
done

if [[ -z "$OUT" ]]; then
    OUT="$PROJECT_DIR/bench_results/ab-$(date +%Y-%m-%d_%H%M%S).md"
fi
mkdir -p "$(dirname "$OUT")"

HIGH_QUBITS_STATE="off"
if [[ -n "${PRISM_BENCH_HIGH_QUBITS:-}" ]]; then
    HIGH_QUBITS_STATE="enabled"
fi

set +e
{
    echo "## Adjacent-binary A/B: \`$BENCH\`"
    echo ""
    echo "| Setting | Value |"
    echo "| --- | --- |"
    echo "| Filter | \`$FILTER\` |"
    echo "| Reference | \`$REF\` ($REF_SHA) |"
    echo "| Reference binary | $REF_PROVENANCE |"
    echo "| Features | \`$FEATURES\` |"
    echo "| Pass order | ref, new discarded, then ref, new, new, ref (adjacent, no rebuild) |"
    echo "| Samples | ${PRISM_BENCH_SAMPLES:-30} |"
    echo "| High-qubit rows | $HIGH_QUBITS_STATE |"
    echo "| CPU | $(host_cpu) |"
    echo "| OS | $(uname -srm) |"
    echo "| Toolchain | $(rustc --version) |"
    echo "| Bench profile | \`lto = \"fat\"\`, \`codegen-units = 1\`, \`debug = \"line-tables-only\"\` |"
    echo "| Threshold | ${THRESHOLD}% |"
    echo ""

    awk -v threshold="$THRESHOLD" -v min_rows="$MIN_ROWS" '
        BEGIN { FS = "\t"; SEP = "\x1f" }

        {
            mean[$1 SEP $2] = $3
            if ($1 == 1) { order[++n] = $2 }
        }

        function fmt(ns) {
            if (ns >= 1000000000) { return sprintf("%.3f s", ns / 1000000000) }
            if (ns >= 1000000)    { return sprintf("%.2f ms", ns / 1000000) }
            if (ns >= 1000)       { return sprintf("%.2f us", ns / 1000) }
            return sprintf("%.1f ns", ns)
        }

        function pct(v) { return sprintf("%+.1f%%", v) }
        function abs(v) { return v < 0 ? -v : v }

        END {
            print "| Benchmark | Ref | New | Change | Control (ref) | Control (new) | Verdict |"
            print "| --- | ---: | ---: | ---: | ---: | ---: | --- |"

            compared = 0
            regressions = 0
            unresolvable = 0
            worst_floor = 0

            for (i = 1; i <= n; i++) {
                id = order[i]
                a1 = mean[1 SEP id]; b1 = mean[2 SEP id]
                b2 = mean[3 SEP id]; a2 = mean[4 SEP id]
                if (a1 == "" || b1 == "" || b2 == "" || a2 == "") { continue }

                ref = (a1 + a2) / 2
                new = (b1 + b2) / 2
                change = (new - ref) * 100 / ref
                ctl_ref = (a2 - a1) * 100 / a1
                ctl_new = (b2 - b1) * 100 / b1
                floor = abs(ctl_ref) > abs(ctl_new) ? abs(ctl_ref) : abs(ctl_new)
                if (floor > worst_floor) { worst_floor = floor }

                if (abs(change) <= floor) {
                    verdict = "noise"
                } else if (change < 0) {
                    verdict = "faster"
                } else {
                    verdict = "slower"
                }

                if (change > threshold && change > floor) { regressions++ }
                if (floor > threshold) { unresolvable++; unresolved[unresolvable] = id }

                printf "| `%s` | %s | %s | %s | %s | %s | %s |\n",
                    id, fmt(ref), fmt(new), pct(change), pct(ctl_ref), pct(ctl_new), verdict
                compared++
            }

            print ""
            printf "%d rows compared. Worst same-code control spread: %.1f%%.\n", compared, worst_floor
            print ""

            if (compared == 0) {
                print "**Regression verdict**: NO DATA. No row appeared in all four passes."
                exit 1
            }

            if (compared < min_rows) {
                printf "**Regression verdict**: NO DATA. %d row(s) appeared in all four passes, expected at least %d. The filter no longer matches every benchmark id it names.\n",
                    compared, min_rows
                exit 1
            }

            status = 0
            if (regressions > 0) {
                status = 1
                printf "**Regression verdict**: FAIL. %d row(s) regressed beyond %s%% and beyond their own control spread.\n",
                    regressions, threshold
            } else {
                printf "**Regression verdict**: PASS at %s%%.\n", threshold
            }

            if (unresolvable > 0) {
                print ""
                printf "%d row(s) have a same-code control spread above the %s%% gate, so this host cannot resolve the gate on them:\n",
                    unresolvable, threshold
                print ""
                for (i = 1; i <= unresolvable; i++) { printf "- `%s`\n", unresolved[i] }
                print ""
                print "Treat their Change column as unmeasured, not as a result."
            }

            exit status
        }
    ' "$MERGED"
} > "$OUT"
REPORT_STATUS=$?
set -e

cat "$OUT"
echo ""
echo "Markdown written to $OUT"
exit "$REPORT_STATUS"

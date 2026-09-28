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
#      pass each at 10 samples, then four measured passes.
#   5. Emit a markdown table with a same-code control column per row.
#
# The measured pass order is ref, new, new, ref. Both means are centred on the
# same point in time, so linear drift cancels, and each binary is measured twice
# so every row carries its own noise floor. A change smaller than that floor is
# reported as noise, not as a win.
#
# The full tier runs Criterion's per-row windows at 1s warm-up and 3s
# measurement rather than the 3s and 5s defaults: a microsecond row collects
# thousands of iterations per sample either way, and a row whose one iteration
# outlasts the window is priced by its sample count, not the window. Rows the
# warmup pass projects past --slow-row-seconds in one pass at the full count
# run at --slow-samples instead, with a verdict of their own; the report names
# them and their count.
#
# --light trades precision for wall clock: three measured passes (ref, new,
# ref), windows of 0.5s and 1.5s, and 10 samples. A row costs about a third of
# the full tier. The report names the tier and the verdict is triage, not a
# gate result: the new binary is measured once, so only the reference side
# carries a control.
#
# Usage:
#   scripts/bench_ab.sh --filter '^factored/noise_kraus/'
#   scripts/bench_ab.sh -f '^density_matrix/' -r main -b circuits
#   scripts/bench_ab.sh -f '^sparse/' --ref-dir /tmp/prism-q-ref   # reuse the build
#   scripts/bench_ab.sh -f '^x/affected/' -c '^x/(control_a|control_b)/'
#   scripts/bench_ab.sh -f '^statevector/' --light                  # triage in a third of the time
#
# Options:
#   --filter,   -f  Criterion filter regex, applied to every pass (required)
#   --control,  -c  second regex whose rows are measured at --control-samples
#                   rather than at the full count. A control has to read flat,
#                   not precisely, and on this corpus the controls are routinely
#                   the expensive rows: an A/B on 2026-09-04 spent 61% of its
#                   wall clock on four control rows that all reported noise.
#                   Must not overlap --filter. A row matched by both is an error
#                   rather than a silent downgrade to the lower count.
#   --control-samples  sample count for --control rows (default 10, Criterion's
#                   floor). Standard error is 1.7x the 30-sample interval, which
#                   is ample to show a control sitting flat.
#   --light         triage tier: passes ref, new, ref with Criterion windows of
#                   0.5s warm-up and 1.5s measurement and 10 samples (the sample
#                   count still yields to PRISM_BENCH_SAMPLES). Cuts a row to
#                   about a fifth of the full tier. Groups that pin their own
#                   measurement_time keep it, so those rows shrink less. The
#                   verdict line names the tier; carry a gate claim on the full
#                   tier only.
#   --slow-row-seconds  a row the warmup pass projects past this many seconds
#                   in one pass at the full sample count runs at --slow-samples
#                   instead (default 30, 0 disables). On this corpus three rows
#                   over a second each were 45% of a 25-row run at 30 samples.
#   --slow-samples  sample count for those rows (default 15, floor 10). Their
#                   standard error is 1.4x the full count's, and each sample
#                   already averages a whole circuit run.
#   --warm-samples  sample count for the two discarded warmup passes (default
#                   10, floor 10). They warm the binary and price every row;
#                   the count they run at changes neither.
#   --max-row-seconds  abort when Criterion projects one row past this many
#                   seconds in a single pass (default 240, 0 disables). Six
#                   passes run, so an unnoticed row costs six times the
#                   projection. The check costs one iteration of that row rather
#                   than the row, because Criterion prints its projection before
#                   collecting: the corpus row that provoked this projected
#                   4200s per pass, seven hours across the six, and was caught
#                   after one 140s iteration.
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
#   --build-dir     with --build-only, build this checkout instead of the one the
#                   script sits in, so one copy of the script can fill a cache
#                   from several worktrees.
#   --new-exe       prebuilt working-tree executable. Skips the working-tree
#                   build and its fingerprint check; the caller owns the claim,
#                   as with --ref-exe. The two together run no build at all.
#   --project-dir   checkout the run reports on and writes bench_results/ into,
#                   instead of the one the script sits in.
#   --min-rows      fail when fewer than this many rows appear in every measured
#                   pass, catching a filter that stopped matching a renamed
#                   benchmark id. Default: 1.
#   --out           markdown output path (default: bench_results/ab-<stamp>.md)
#
# Host guards (on by default):
#   --max-host-load percent of all CPUs other work may hold while the host sits
#                   between passes (default 10, 0 disables). Checked before the
#                   first pass, where a busy host stops the run, and before each
#                   measured pass, where it is recorded in the report. One busy
#                   thread on an 8-thread host is 12.5%.
#   --wait-idle     seconds to wait for the host to fall under --max-host-load
#                   before giving up (default 0). A queued run passes a long wait
#                   so it starts when the host settles instead of failing.
#
# Verdict guards:
#   A run whose same-binary controls lean one way across most rows reads RERUN
#   (exit 3), not PASS or FAIL: the median control, signed, past half the
#   threshold means one binary's passes moved as a lane, the pattern a run
#   straight after a cold build or under background load shows, and the change
#   column carries that shift on every row.
#   --no-confirm    by default a full-tier FAIL re-runs the regressed rows once
#                   from fresh copies of both binaries. A row that does not
#                   regress again leaves the run at RERUN rather than FAIL: a
#                   copied binary can land in a slow layout for one run, and the
#                   controls cannot see it because they compare a binary with
#                   itself.
#   --escalate      with --light, re-run every row that moved past half the
#                   threshold on the full tier against the same two binaries,
#                   and take the verdict from that run.
#
# Exit status: 0 PASS, 1 FAIL, 2 host too busy to start, 3 RERUN.
#
# Environment:
#   REGRESSION_THRESHOLD   regression gate in percent (default: 5.0)
#   PRISM_BENCH_SAMPLES    samples per row, recorded in the report. Set it to 10
#                          for a triage sweep that finds which rows moved, then
#                          re-run at the default on those rows alone.
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
CONTROL_FILTER=""
CONTROL_SAMPLES=10
SLOW_ROW_SECONDS=30
SLOW_SAMPLES=15
WARM_SAMPLES=10
MAX_ROW_SECONDS=240
SAMPLES="${PRISM_BENCH_SAMPLES:-30}"
LIGHT=""
REF="HEAD"
BENCH="circuits"
FEATURES="parallel"
REF_DIR=""
REF_EXE=""
BUILD_ONLY=""
BUILD_DIR=""
NEW_EXE=""
OUT=""
MIN_ROWS=1
THRESHOLD="${REGRESSION_THRESHOLD:-5.0}"
MAX_HOST_LOAD=10
WAIT_IDLE=0
CONFIRM=1
ESCALATE=""
# Set on the runs this script starts itself: "confirm" for the re-run of a FAIL,
# "escalate" for the full-tier run behind --escalate.
NESTED="${PRISM_BENCH_AB_NESTED:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter|-f)   FILTER="$2"; shift 2 ;;
        --control|-c)  CONTROL_FILTER="$2"; shift 2 ;;
        --control-samples) CONTROL_SAMPLES="$2"; shift 2 ;;
        --max-row-seconds) MAX_ROW_SECONDS="$2"; shift 2 ;;
        --slow-row-seconds) SLOW_ROW_SECONDS="$2"; shift 2 ;;
        --slow-samples) SLOW_SAMPLES="$2"; shift 2 ;;
        --warm-samples) WARM_SAMPLES="$2"; shift 2 ;;
        --light)       LIGHT=1; shift ;;
        --ref|-r)      REF="$2"; shift 2 ;;
        --bench|-b)    BENCH="$2"; shift 2 ;;
        --features)    FEATURES="$2"; shift 2 ;;
        --ref-dir)     REF_DIR="$2"; shift 2 ;;
        --ref-exe)     REF_EXE="$2"; shift 2 ;;
        --build-only)  BUILD_ONLY="$2"; shift 2 ;;
        --build-dir)   BUILD_DIR="$2"; shift 2 ;;
        --new-exe)     NEW_EXE="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$(cd "$2" && pwd)"; shift 2 ;;
        --max-host-load) MAX_HOST_LOAD="$2"; shift 2 ;;
        --wait-idle)   WAIT_IDLE="$2"; shift 2 ;;
        --no-confirm)  CONFIRM=""; shift ;;
        --escalate)    ESCALATE=1; shift ;;
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

# Everything the tier changes, in one place. Both tiers set Criterion's windows
# and take the pass sequence the header describes.
if [[ -n "$LIGHT" ]]; then
    TIER="light (triage)"
    SAMPLES="${PRISM_BENCH_SAMPLES:-10}"
    CRITERION_ARGS=(--warm-up-time 0.5 --measurement-time 1.5)
    WINDOWS="0.5s warm-up, 1.5s measurement"
    PASS_LABELS=(ref new ref)
else
    TIER="full"
    CRITERION_ARGS=(--warm-up-time 1 --measurement-time 3)
    WINDOWS="1s warm-up, 3s measurement"
    PASS_LABELS=(ref new new ref)
fi
PASS_COUNT="${#PASS_LABELS[@]}"

if [[ -n "$ESCALATE" && -z "$LIGHT" ]]; then
    echo "Error: --escalate re-runs light-tier rows on the full tier, so it needs --light." >&2
    exit 1
fi
# The light verdict is triage, so there is nothing to confirm; a nested run
# never starts another of its own kind.
if [[ -n "$LIGHT" || "$NESTED" == "confirm" ]]; then
    CONFIRM=""
fi
if [[ -n "$NESTED" ]]; then
    ESCALATE=""
fi
PASS_ORDER="$(IFS=,; echo "${PASS_LABELS[*]}")"
PASS_ORDER="${PASS_ORDER//,/, }"

# Criterion clamps below 10 without saying so, which would put a count in the
# report that no row was measured at.
if (( CONTROL_SAMPLES < 10 )); then
    echo "Error: --control-samples is $CONTROL_SAMPLES; Criterion's floor is 10." >&2
    exit 1
fi
if (( SLOW_SAMPLES < 10 )); then
    echo "Error: --slow-samples is $SLOW_SAMPLES; Criterion's floor is 10." >&2
    exit 1
fi
if (( WARM_SAMPLES < 10 )); then
    echo "Error: --warm-samples is $WARM_SAMPLES; Criterion's floor is 10." >&2
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

# Extract `full_id<TAB>mean_ns<TAB>samples` for every row a pass measured. The
# mean is the first point estimate in the JSON object, which serde writes before
# the median. The sample count travels with the row so a report can say which
# rows were measured at the control count.
snapshot() {
    local home="$1" out="$2" samples="$3"

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

        [[ -n "$id" && -n "$mean" ]] && printf '%s\t%s\t%s\n' "$id" "$mean" "$samples"
    done | sort > "$out"
}

# The (segment, filter, samples) triples one pass runs, in order. A warmup pass
# runs the whole filter at the warmup count. A measured pass runs the rows the
# warmup priced under --slow-row-seconds at the full count and the rest at
# --slow-samples, then the controls, each segment into its own CRITERION_HOME
# so the sample counts stay separable.
FAST_FILTER=""
SLOW_FILTER=""
SLOW_IDS=""
pass_segments() {
    if [[ -n "${WARMING:-}" ]]; then
        printf 'main\t%s\t%s\n' "$FILTER" "$WARM_SAMPLES"
    elif [[ -n "$SLOW_FILTER" ]]; then
        printf 'main\t%s\t%s\n' "$FAST_FILTER" "$SAMPLES"
        printf 'slow\t%s\t%s\n' "$SLOW_FILTER" "$SLOW_SAMPLES"
    else
        printf 'main\t%s\t%s\n' "$FILTER" "$SAMPLES"
    fi
    if [[ -n "$CONTROL_FILTER" ]]; then
        printf 'ctl\t%s\t%s\n' "$CONTROL_FILTER" "$CONTROL_SAMPLES"
    fi
}

# Anchor a list of benchmark ids as one Criterion filter, one id per line in.
ids_to_filter() {
    awk '
        { gsub(/[][\\.^$*+?(){}|]/, "\\\\&"); ids = ids (ids == "" ? "" : "|") $0 }
        END { if (ids != "") printf "^(%s)$", ids }
    '
}

# Split the rows the first warmup pass ran into fast and slow by what each
# would cost per pass at the full count: Criterion prints its projection and
# the iteration count before collecting, so one iteration prices the row. The
# main segment then carries the fast rows and a slow segment the rest.
split_slow_rows() {
    local log="$1"
    local fast slow
    awk -v full="$SAMPLES" -v budget="$SLOW_ROW_SECONDS" '
        /Collecting [0-9]+ samples in estimated/ {
            id = $2
            sub(/:$/, "", id)
            if (match($0, /estimated [0-9.]+ s/)) {
                secs = substr($0, RSTART + 10, RLENGTH - 12) + 0
            } else { next }
            iters = 0
            if (match($0, /\([0-9]+ iterations\)/)) {
                iters = substr($0, RSTART + 1, RLENGTH - 13) + 0
            }
            if (iters > 0 && secs / iters * full > budget) {
                print "slow\t" id
            } else {
                print "fast\t" id
            }
        }
    ' "$log" > "$WORKDIR/rows.tsv"
    fast="$(awk -F'\t' '$1 == "fast" { print $2 }' "$WORKDIR/rows.tsv" | ids_to_filter)"
    slow="$(awk -F'\t' '$1 == "slow" { print $2 }' "$WORKDIR/rows.tsv" | ids_to_filter)"
    if [[ -n "$slow" && -n "$fast" ]]; then
        FAST_FILTER="$fast"
        SLOW_FILTER="$slow"
        SLOW_IDS="$(awk -F'\t' '$1 == "slow" { print $2 }' "$WORKDIR/rows.tsv")"
    elif [[ -n "$slow" ]]; then
        # Every row is slow: one segment at the reduced count, still named.
        SLOW_FILTER=""
        SLOW_IDS="$(awk -F'\t' '$1 == "slow" { print $2 }' "$WORKDIR/rows.tsv")"
        SAMPLES="$SLOW_SAMPLES"
    fi
}

# The first per-pass projection Criterion has printed that is over budget.
#
# Criterion writes the projection before it starts collecting, so this sees a
# row's cost after one iteration rather than after the row.
over_budget_row() {
    awk -v budget="$MAX_ROW_SECONDS" -v scale="${2:-1}" '
        /Collecting [0-9]+ samples in estimated/ {
            if (match($0, /estimated [0-9.]+ s/)) {
                secs = substr($0, RSTART + 10, RLENGTH - 12) + 0
                if (scale > 1 && match($0, /\([0-9]+ iterations\)/)) {
                    iters = substr($0, RSTART + 1, RLENGTH - 13) + 0
                    if (iters > 0 && secs / iters * scale > secs) { secs = secs / iters * scale }
                }
                if (secs > budget) {
                    id = $2
                    sub(/:$/, "", id)
                    printf "%s projects %.0fs per pass", id, secs
                    exit
                }
            }
        }
    ' "$1"
}

# Total seconds one measured pass will take, from the warmup projections: a row
# whose iterations fit the window costs the window, and a row that outlasts it
# costs one iteration times the count it will run at.
projected_pass_seconds() {
    cat "$@" 2>/dev/null | awk -v full="$SAMPLES" -v budget="$SLOW_ROW_SECONDS" \
        -v slow="$SLOW_SAMPLES" '
        /Collecting [0-9]+ samples in estimated/ {
            if (!match($0, /estimated [0-9.]+ s/)) { next }
            secs = substr($0, RSTART + 10, RLENGTH - 12) + 0
            iters = 0
            if (match($0, /\([0-9]+ iterations\)/)) {
                iters = substr($0, RSTART + 1, RLENGTH - 13) + 0
            }
            if (iters > 0) {
                at_full = secs / iters * full
                if (budget > 0 && at_full > budget) { at_full = secs / iters * slow }
                if (at_full > secs) { secs = at_full }
            }
            total += secs
        }
        END { printf "%.0f", total }
    '
}

# Run one segment in the background so the projection guard can end it early.
#
# `set -e` does not reach a backgrounded child, so the status comes from `wait`
# and is checked here. Without that a panicking bench binary would read as a
# pass that measured nothing.
run_segment() {
    local exe="$1" home="$2" filter="$3" samples="$4" log="$5" quiet="$6"
    local pid status over shown total scale
    # A warmup pass runs fewer samples than the measured passes will, so its
    # projection is scaled up to the full count before it meets the budget.
    scale=1
    if [[ -n "${WARMING:-}" ]]; then
        scale=$(( SAMPLES > samples ? SAMPLES / samples : 1 ))
    fi

    mkdir -p "$home"
    : > "$log"
    CRITERION_HOME="$(native_path "$home")" PRISM_BENCH_SAMPLES="$samples" \
        "$exe" --bench ${CRITERION_ARGS[@]+"${CRITERION_ARGS[@]}"} "$filter" > "$log" 2>&1 &
    pid=$!

    over=""
    shown=0
    while kill -0 "$pid" 2>/dev/null; do
        if [[ "$quiet" != "quiet" ]]; then
            total="$(wc -l < "$log" | tr -d '[:space:]')"
            if (( total > shown )); then
                sed -n "$((shown + 1)),${total}p" "$log"
                shown=$total
            fi
        fi
        if (( MAX_ROW_SECONDS > 0 )); then
            over="$(over_budget_row "$log" "$scale")"
            if [[ -n "$over" ]]; then
                kill "$pid" 2>/dev/null || true
                break
            fi
        fi
        sleep 2
    done

    status=0
    wait "$pid" 2>/dev/null || status=$?

    if [[ -n "$over" ]]; then
        echo "" >&2
        echo "Error: $over, past the ${MAX_ROW_SECONDS}s --max-row-seconds budget." >&2
        echo "  $(( PASS_COUNT + 2 )) passes run, so that row alone costs about that many times as much." >&2
        echo "  Narrow --filter, move the row behind --control, pass --light, lower" >&2
        echo "  PRISM_BENCH_SAMPLES, or pass --max-row-seconds 0 to measure it anyway." >&2
        exit 1
    fi
    if (( status != 0 )); then
        [[ "$quiet" == "quiet" ]] && cat "$log" >&2
        echo "Error: the bench binary exited $status during '$filter'." >&2
        exit 1
    fi
    if [[ "$quiet" != "quiet" ]]; then
        total="$(wc -l < "$log" | tr -d '[:space:]')"
        if (( total > shown )); then
            sed -n "$((shown + 1)),${total}p" "$log"
        fi
    fi
}

# Run every segment of one pass and collect them into one sorted table.
measure() {
    local exe="$1" home="$2" out="$3" logbase="$4" quiet="$5"
    local seg filter samples

    : > "$out"
    while IFS=$'\t' read -r seg filter samples; do
        run_segment "$exe" "$home/$seg" "$filter" "$samples" "$logbase-$seg.log" "$quiet"
        snapshot "$home/$seg" "$WORKDIR/segment.tsv" "$samples"
        cat "$WORKDIR/segment.tsv" >> "$out"
    done < <(pass_segments)

    sort -o "$out" "$out"
}

run_pass() {
    local idx="$1" label="$2" exe="$3"
    local home="$WORKDIR/crit-$idx"
    local out="$WORKDIR/pass-$idx.tsv"
    mkdir -p "$home"

    echo ">>> pass $idx ($label)"
    measure "$exe" "$home" "$out" "$WORKDIR/pass-$idx" "loud"

    local rows dupes
    rows="$(wc -l < "$out" | tr -d '[:space:]')"
    if (( rows == 0 )); then
        echo "Error: pass $idx produced no Criterion estimates under $home." >&2
        echo "  Either --filter '$FILTER' matched nothing, or CRITERION_HOME was ignored." >&2
        exit 1
    fi
    dupes="$(cut -f1 "$out" | uniq -d)"
    if [[ -n "$dupes" ]]; then
        echo "Error: two segments both measured these rows:" >&2
        printf '  %s\n' $dupes >&2
        echo "  A row measured at two sample counts has no single control column." >&2
        echo "  Make the two regexes disjoint." >&2
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
    echo ">>> warmup $idx (discarded, $WARM_SAMPLES samples)"
    WARMING=1 measure "$exe" "$home" "$WORKDIR/warm-$idx.tsv" "$WORKDIR/warm-$idx" "quiet"
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

# Busy share of every CPU over about two seconds, as a whole percent, or empty
# where the host offers no reading. MSYS and Cygwin expose the Windows counters
# through /proc/stat, so one reader covers Windows, Linux, and WSL.
host_load() {
    if [[ -r /proc/stat ]]; then
        local first second
        first="$(awk '/^cpu / { b = $2 + $3 + $4 + $7 + $8 + $9; print b, b + $5 + $6; exit }' /proc/stat)"
        sleep 2
        second="$(awk '/^cpu / { b = $2 + $3 + $4 + $7 + $8 + $9; print b, b + $5 + $6; exit }' /proc/stat)"
        awk -v a="$first" -v b="$second" 'BEGIN {
            split(a, x, " "); split(b, y, " ")
            if (y[2] > x[2]) { printf "%d", (y[1] - x[1]) * 100 / (y[2] - x[2]) }
        }'
    elif command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu >/dev/null 2>&1; then
        ps -A -o %cpu= | awk -v n="$(sysctl -n hw.ncpu)" '{ s += $1 } END { printf "%d", s / n }'
    fi
}

# The processes holding the most CPU, for the message that stops a run.
busy_processes() {
    if command -v wmic >/dev/null 2>&1; then
        wmic path Win32_PerfFormattedData_PerfProc_Process get Name,PercentProcessorTime 2>/dev/null |
            tr -d '\r' |
            awk 'NR > 1 && $1 != "_Total" && $1 != "Idle" && $2 + 0 > 0 { print $2 "% " $1 }' |
            sort -rn | head -5
    else
        ps -eo pcpu,comm --sort=-pcpu 2>/dev/null | sed -n '2,6p'
    fi
}

# Stop, or wait up to --wait-idle, while other work holds more of the host than
# --max-host-load. Every row of the run would carry that load, and the controls
# only see the part of it that changes between passes.
preflight() {
    (( MAX_HOST_LOAD > 0 )) || return 0
    local load waited=0
    while :; do
        load="$(host_load)"
        if [[ -z "$load" ]]; then
            echo ">>> no host load reading here, so the load guard is off for this run"
            MAX_HOST_LOAD=0
            return 0
        fi
        if (( load <= MAX_HOST_LOAD )); then
            echo ">>> host ${load}% busy before measuring"
            echo ""
            return 0
        fi
        if (( waited >= WAIT_IDLE )); then
            echo "Error: the host is ${load}% busy before measuring, over --max-host-load ${MAX_HOST_LOAD}%." >&2
            busy_processes | sed 's/^/    /' >&2
            echo "  Stop that work, pass --wait-idle SECONDS to wait for it, or raise" >&2
            echo "  --max-host-load if the load is part of what is being measured." >&2
            exit 2
        fi
        echo ">>> host ${load}% busy, waiting for it to fall under ${MAX_HOST_LOAD}% (${waited}s of ${WAIT_IDLE}s)"
        if (( waited % 600 == 0 )); then
            busy_processes | sed 's/^/    /'
        fi
        sleep 28
        waited=$(( waited + 30 ))
    done
}

# Read the host between passes, when none of this script's work is running, and
# record a pass that starts on a busy host.
LOADED_PASSES=""
check_load() {
    (( MAX_HOST_LOAD > 0 )) || return 0
    local load
    load="$(host_load)"
    if [[ -n "$load" ]] && (( load > MAX_HOST_LOAD )); then
        echo ">>> host ${load}% busy before pass $1, over ${MAX_HOST_LOAD}%"
        LOADED_PASSES="${LOADED_PASSES}${LOADED_PASSES:+, }pass $1 at ${load}%"
    fi
}

# --- Build both binaries ---

if [[ -n "$BUILD_ONLY" ]]; then
    mkdir -p "$(dirname "$BUILD_ONLY")"
    build_bench_exe "$(cd "${BUILD_DIR:-$PROJECT_DIR}" && pwd)" "$BUILD_ONLY" "build-only"
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

if [[ -n "$NEW_EXE" ]]; then
    if [[ ! -f "$NEW_EXE" ]]; then
        echo "Error: --new-exe '$NEW_EXE' does not exist." >&2
        exit 1
    fi
    echo ">>> using the supplied working-tree executable $NEW_EXE"
    cp "$NEW_EXE" "$WORKDIR/exe-new"
    chmod +x "$WORKDIR/exe-new"
    NEW_PROVENANCE="supplied via \`--new-exe\`, not built here"
else
    build_bench_exe "$PROJECT_DIR" "$WORKDIR/exe-new" "new"
    NEW_PROVENANCE="built from the working tree"
fi

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
    if [[ -z "$NEW_EXE" && "$FINGERPRINT_BEFORE" != "$FINGERPRINT_AFTER" ]]; then
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

echo "=== $TIER tier: two discarded warmup passes, then $PASS_COUNT adjacent passes ($PASS_ORDER) ==="
echo ""
preflight
warm_up 1 "$WORKDIR/exe-ref"

# Criterion projected every row during the warmup, so the slow rows can be
# named and the remaining passes priced before they are spent. The price is
# printed rather than enforced: the guard is per row, and a filter can be slow
# by holding many cheap rows instead.
if (( SLOW_ROW_SECONDS > 0 )); then
    split_slow_rows "$WORKDIR/warm-1-main.log"
    if [[ -n "$SLOW_IDS" ]]; then
        echo ">>> rows projected past ${SLOW_ROW_SECONDS}s per pass at $SAMPLES samples, run at $SLOW_SAMPLES:"
        printf '    %s\n' $SLOW_IDS
        echo ""
    fi
fi
PASS_SECONDS="$(projected_pass_seconds "$WORKDIR"/warm-1-*.log)"
if [[ -n "$PASS_SECONDS" && "$PASS_SECONDS" != "0" ]]; then
    printf ">>> projected: about %d min for the %d measured passes\n" \
        $(( PASS_SECONDS * PASS_COUNT / 60 )) "$PASS_COUNT"
    echo ""
fi

warm_up 2 "$WORKDIR/exe-new"
for idx in $(seq 1 "$PASS_COUNT"); do
    label="${PASS_LABELS[idx - 1]}"
    check_load "$idx"
    run_pass "$idx" "$label" "$WORKDIR/exe-$label"
done

# --- Report ---

MERGED="$WORKDIR/merged.tsv"
: > "$MERGED"
for idx in $(seq 1 "$PASS_COUNT"); do
    awk -v p="$idx" 'BEGIN { OFS = "\t" } { print p, $1, $2, $3 }' "$WORKDIR/pass-$idx.tsv" >> "$MERGED"
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
    if [[ -n "$CONTROL_FILTER" ]]; then
        echo "| Control filter | \`$CONTROL_FILTER\`, measured at $CONTROL_SAMPLES samples |"
    fi
    echo "| Reference | \`$REF\` ($REF_SHA) |"
    echo "| Reference binary | $REF_PROVENANCE |"
    echo "| Working-tree binary | $NEW_PROVENANCE |"
    echo "| Features | \`$FEATURES\` |"
    echo "| Tier | $TIER |"
    echo "| Pass order | ref, new discarded at $WARM_SAMPLES samples, then $PASS_ORDER (adjacent, no rebuild) |"
    echo "| Criterion windows | $WINDOWS (groups that pin their own keep it) |"
    echo "| Samples | $SAMPLES |"
    if (( SLOW_ROW_SECONDS > 0 )); then
        echo "| Slow rows | over ${SLOW_ROW_SECONDS}s per pass at $SAMPLES samples run at $SLOW_SAMPLES |"
    fi
    echo "| Row budget | ${MAX_ROW_SECONDS}s per pass |"
    if (( MAX_HOST_LOAD > 0 )); then
        echo "| Host load guard | ${MAX_HOST_LOAD}% of all CPUs, read before each measured pass |"
    else
        echo "| Host load guard | off |"
    fi
    echo "| High-qubit rows | $HIGH_QUBITS_STATE |"
    echo "| CPU | $(host_cpu) |"
    echo "| OS | $(uname -srm) |"
    echo "| Toolchain | $(rustc --version) |"
    echo "| Bench profile | \`lto = \"fat\"\`, \`codegen-units = 1\`, \`debug = \"line-tables-only\"\` |"
    echo "| Threshold | ${THRESHOLD}% |"
    echo ""

    awk -v threshold="$THRESHOLD" -v min_rows="$MIN_ROWS" -v full="$SAMPLES" \
        -v light="${LIGHT:-0}" -v passes="$PASS_COUNT" -v slow_ids="$SLOW_IDS" \
        -v slow_at="$SLOW_SAMPLES" -v loaded="$LOADED_PASSES" \
        -v moved_file="$WORKDIR/moved.txt" -v regressed_file="$WORKDIR/regressed.txt" '
        BEGIN {
            FS = "\t"; SEP = "\x1f"
            slow_n = split(slow_ids, slow_list, "\n")
            for (k = 1; k <= slow_n; k++) { if (slow_list[k] != "") { is_slow[slow_list[k]] = 1 } }
        }

        {
            mean[$1 SEP $2] = $3
            if ($1 == 1) { order[++n] = $2; samples[$2] = $4 }
        }

        function fmt(ns) {
            if (ns >= 1000000000) { return sprintf("%.3f s", ns / 1000000000) }
            if (ns >= 1000000)    { return sprintf("%.2f ms", ns / 1000000) }
            if (ns >= 1000)       { return sprintf("%.2f us", ns / 1000) }
            return sprintf("%.1f ns", ns)
        }

        function pct(v) { return sprintf("%+.1f%%", v) }
        function abs(v) { return v < 0 ? -v : v }

        function median(src, m,    i, j, t, a) {
            for (i = 1; i <= m; i++) { a[i] = src[i] }
            for (i = 2; i <= m; i++) {
                t = a[i]; j = i - 1
                while (j >= 1 && a[j] > t) { a[j + 1] = a[j]; j-- }
                a[j + 1] = t
            }
            return (m % 2) ? a[(m + 1) / 2] : (a[m / 2] + a[m / 2 + 1]) / 2
        }

        END {
            print "| Benchmark | Ref | New | Change | Paired | Control (ref) | Control (new) | Samples | Verdict |"
            print "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"

            compared = 0
            regressions = 0
            unresolvable = 0
            worst_floor = 0
            reduced = 0
            reduced_floor = 0
            reduced_at = 0
            moved_controls = 0
            slowed = 0

            for (i = 1; i <= n; i++) {
                id = order[i]
                # Light tier passes are ref, new, ref: the new binary is measured once
                # and has no control of its own, so the row floor is the ref spread.
                if (light) {
                    a1 = mean[1 SEP id]; b1 = mean[2 SEP id]; a2 = mean[3 SEP id]
                    if (a1 == "" || b1 == "" || a2 == "") { continue }
                    new = b1
                    ctl_new = 0
                    ctl_new_text = "n/a"
                } else {
                    a1 = mean[1 SEP id]; b1 = mean[2 SEP id]
                    b2 = mean[3 SEP id]; a2 = mean[4 SEP id]
                    if (a1 == "" || b1 == "" || b2 == "" || a2 == "") { continue }
                    new = (b1 + b2) / 2
                    ctl_new = (b2 - b1) * 100 / b1
                    ctl_new_text = pct(ctl_new)
                }

                ref = (a1 + a2) / 2
                change = (new - ref) * 100 / ref
                ctl_ref = (a2 - a1) * 100 / a1
                floor = abs(ctl_ref) > abs(ctl_new) ? abs(ctl_ref) : abs(ctl_new)

                # Reported, not gated. The lane means pool passes 1 and 4 against
                # 2 and 3, so a load transient inside one pass biases that lane
                # whole. Passes 1 and 2 are adjacent in wall-clock order, as are
                # 3 and 4, so a transient spanning either boundary moves both
                # sides of that pair together. Two independent readings that
                # disagree in sign are the single-run form of two runs that
                # disagree, which this project already treats as no result.
                if (light) {
                    paired_text = "n/a"
                } else {
                    pr1 = (b1 - a1) * 100 / a1
                    pr2 = (b2 - a2) * 100 / a2
                    if (pr1 * pr2 < 0) {
                        paired = 0
                    } else {
                        paired = abs(pr1) < abs(pr2) ? abs(pr1) : abs(pr2)
                        if (pr1 < 0) { paired = -paired }
                    }
                    paired_text = pct(paired)
                    if (abs(paired) > threshold && abs(change) <= threshold) { paired_only++ }
                    if (abs(change) > threshold && abs(paired) <= threshold) { lane_only++ }
                }

                # A row measured at the reduced control count has a wider own-spread by
                # construction, so it neither sets the host noise floor nor joins the
                # unresolvable list. It can still fail the gate: a control moving past
                # its own spread and past the threshold is collateral damage either way.
                # A slow row runs at its own reduced count but is a row under test:
                # it keeps its verdict, sets the floor, and is counted on its own line.
                at = samples[id] + 0
                control_row = (at > 0 && at != full + 0 && !(id in is_slow))
                if (id in is_slow) { slowed++ }
                if (control_row) {
                    reduced++
                    reduced_at = at
                    if (floor > reduced_floor) { reduced_floor = floor }
                } else if (floor > worst_floor) {
                    worst_floor = floor
                }

                # A reduced-count row is named as the control it is rather than given a
                # direction. Ten samples resolve "flat" but not a percentage, so calling
                # one faster or slower would dress noise as a result.
                if (control_row) {
                    if (abs(change) > threshold) {
                        verdict = "control moved"
                        moved_controls++
                    } else {
                        verdict = "control"
                    }
                } else if (abs(change) <= floor) {
                    verdict = "noise"
                } else if (change < 0) {
                    verdict = "faster"
                } else {
                    verdict = "slower"
                }

                if (change > threshold && change > floor) {
                    regressions++; regressed[regressions] = id
                    print id > regressed_file
                }
                if (!control_row) {
                    nc++
                    cr[nc] = ctl_ref
                    cn[nc] = ctl_new
                    if (abs(change) > threshold / 2) { print id > moved_file }
                }
                if (floor > threshold && !control_row) {
                    unresolvable++; unresolved[unresolvable] = id
                }

                printf "| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |\n",
                    id, fmt(ref), fmt(new), pct(change), paired_text, pct(ctl_ref),
                    ctl_new_text, samples[id], verdict
                compared++
            }

            print ""
            if (!light && (paired_only > 0 || lane_only > 0)) {
                printf "Paired disagrees with Change past the %.1f%% threshold on %d row(s): %d only paired, %d only lane-mean. Reported for comparison; the verdict is still the Change column.\n\n",
                    threshold, paired_only + lane_only, paired_only, lane_only
            }
            printf "%d rows compared. Worst same-code control spread: %.1f%%",
                compared, worst_floor
            if (slowed > 0) {
                printf ", with %d slow row(s) at %d samples", slowed, slow_at
            }
            if (reduced > 0) {
                printf " across the %d row(s) at %d samples.\n", compared - reduced, full
                printf "%d control row(s) ran at %d samples, widest own-spread %.1f%%. A reduced count\n",
                    reduced, reduced_at, reduced_floor
                print "widens that spread by construction, so it is not the host noise floor."
            } else {
                print "."
            }
            print ""

            if (compared == 0) {
                printf "**Regression verdict**: NO DATA. No row appeared in all %d passes.\n", passes
                exit 1
            }

            if (compared < min_rows) {
                printf "**Regression verdict**: NO DATA. %d row(s) appeared in all %d passes, expected at least %d. The filter no longer matches every benchmark id it names.\n",
                    compared, passes, min_rows
                exit 1
            }

            # The light tier names itself on the verdict line so the table cannot be
            # pasted into a PR as a gate result.
            tier_note = ""
            if (light) {
                tier_note = " Light tier, triage only: re-run the rows that moved on the full tier before claiming a number."
            }
            # A lane that moved as a whole shows up as a median control well off
            # zero. Row noise scatters controls both ways and leaves the median
            # near it; a cold start or a background job shifts one binary every
            # row at once, and each change then carries the shift.
            shift_text = ""
            if (nc >= 4) {
                mref = median(cr, nc)
                if (abs(mref) > threshold / 2) {
                    shift_text = sprintf("The reference binary read a median of %s between its own two passes across %d rows", pct(mref), nc)
                }
                if (!light) {
                    mnew = median(cn, nc)
                    if (abs(mnew) > threshold / 2 && abs(mnew) >= abs(mref)) {
                        shift_text = sprintf("The working-tree binary read a median of %s between its own two passes across %d rows", pct(mnew), nc)
                    }
                }
            }

            status = 0
            if (shift_text != "") {
                status = 3
                printf "**Regression verdict**: RERUN. %s, so that binary moved as a lane and every row'"'"'s change carries the shift. A run straight after a cold build or under background load reads this way.%s\n",
                    shift_text, tier_note
                if (regressions > 0) {
                    print ""
                    print "Rows past the gate in this run, not a result:"
                    print ""
                    for (i = 1; i <= regressions; i++) { printf "- `%s`\n", regressed[i] }
                }
            } else if (regressions > 0 && loaded != "") {
                status = 3
                printf "**Regression verdict**: RERUN. %d row(s) read past %s%%, but the host was busy before %s.%s\n",
                    regressions, threshold, loaded, tier_note
                print ""
                for (i = 1; i <= regressions; i++) { printf "- `%s`\n", regressed[i] }
            } else if (regressions > 0) {
                status = 1
                printf "**Regression verdict**: FAIL. %d row(s) regressed beyond %s%% and beyond their own control spread.%s\n",
                    regressions, threshold, tier_note
                print ""
                for (i = 1; i <= regressions; i++) { printf "- `%s`\n", regressed[i] }
            } else {
                printf "**Regression verdict**: PASS at %s%%.%s\n", threshold, tier_note
                if (loaded != "") {
                    printf "\nThe host was busy before %s; read the control columns before trusting a flat row.\n", loaded
                }
            }

            if (moved_controls > 0) {
                print ""
                printf "%d control row(s) moved beyond %s%%. A control is meant to sit flat, so\n",
                    moved_controls, threshold
                print "either the change reached further than intended or the lane drifted. Re-measure"
                print "the row at the full sample count before reading anything else in this table."
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

# Run this script again on a subset of rows against the two binaries already
# built, and append that run's report under a heading. Each run copies both
# executables into its own directory, so a follow-up measures fresh copies.
follow_up() {
    local kind="$1" filter="$2" title="$3"
    shift 3
    local sub_out="${OUT%.md}-$kind.md" status=0
    PRISM_BENCH_AB_NESTED="$kind" bash "${BASH_SOURCE[0]}" \
        --project-dir "$PROJECT_DIR" --bench "$BENCH" --features "$FEATURES" --ref "$REF" \
        --ref-exe "$WORKDIR/exe-ref" --new-exe "$WORKDIR/exe-new" \
        --filter "$filter" --out "$sub_out" --threshold "$THRESHOLD" \
        --max-host-load "$MAX_HOST_LOAD" --wait-idle "$WAIT_IDLE" \
        --slow-row-seconds "$SLOW_ROW_SECONDS" --slow-samples "$SLOW_SAMPLES" \
        --warm-samples "$WARM_SAMPLES" --max-row-seconds "$MAX_ROW_SECONDS" "$@" || status=$?
    if [[ -f "$sub_out" ]]; then
        { echo ""; echo "## $title"; echo ""; sed '1{/^## /d}' "$sub_out"; } >> "$OUT"
    else
        printf '\n## %s\n\nThe run stopped before writing a report (exit %d).\n' "$title" "$status" >> "$OUT"
    fi
    return "$status"
}

if [[ -n "$CONFIRM" && "$REPORT_STATUS" == 1 && -s "$WORKDIR/regressed.txt" ]]; then
    echo ">>> re-running the regressed rows from fresh copies of both binaries"
    CONFIRM_STATUS=0
    follow_up confirm "$(ids_to_filter < "$WORKDIR/regressed.txt")" \
        "Confirmation: the regressed rows from fresh copies" || CONFIRM_STATUS=$?
    if (( CONFIRM_STATUS == 1 )); then
        printf '\n**Final verdict**: FAIL, confirmed from fresh copies of both binaries.\n' >> "$OUT"
    else
        REPORT_STATUS=3
        printf '\n**Final verdict**: RERUN. Fresh copies did not reproduce the regression, which points at the layout one copy landed in rather than the change. Measure again before claiming either way.\n' >> "$OUT"
    fi
fi

if [[ -n "$ESCALATE" ]]; then
    if [[ -s "$WORKDIR/moved.txt" ]]; then
        echo ">>> re-running the rows that moved past half the threshold on the full tier"
        ESCALATE_STATUS=0
        follow_up escalate "$(ids_to_filter < "$WORKDIR/moved.txt")" \
            "Full-tier run of the rows that moved" || ESCALATE_STATUS=$?
        REPORT_STATUS=$ESCALATE_STATUS
        printf '\n**Final verdict**: the full-tier run above decides (exit %d).\n' "$ESCALATE_STATUS" >> "$OUT"
    else
        printf '\nNo row moved past half the threshold, so nothing went to the full tier.\n' >> "$OUT"
    fi
fi

cat "$OUT"
echo ""
echo "Markdown written to $OUT"
exit "$REPORT_STATUS"

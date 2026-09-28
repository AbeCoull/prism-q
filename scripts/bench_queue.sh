#!/usr/bin/env bash
# Run a list of benchmark steps unattended: build every binary first, then
# measure one step at a time on a quiet host.
#
# A queue file is bash, sourced, and describes steps with two commands:
#
#   ab   NAME DIR [bench_ab.sh options]   an A/B of the checkout at DIR
#   run  NAME DIR -- COMMAND [ARGS]       any other measurement, run in DIR
#
#   ab   fstab   /work/pq-dispatch -r 315b43e -f '^auto/crossover/auto/fstab_' --light
#   ab   spd     /work/pq-spd      -r dea250c -f '^auto/crossover/(auto|statevector)/spd_wide_'
#   run  sweep   /work/pq-pairs    -- python sweep.py
#
# The queue runs in three phases.
#
#   1. Plan. Source the file and list the steps.
#   2. Build. Every A/B binary is built before anything is measured, through
#      bench_ab.sh --build-only, into a cache keyed on the commit, the features,
#      the bench target, and the toolchain. Steps that share a reference share
#      one build of it, and a later queue reuses whatever is cached. A step whose
#      checkout has uncommitted changes is keyed on those changes as well.
#   3. Measure. Each step runs in turn with no build in between. An A/B runs
#      against its cached binaries with --wait-idle, so it starts once the host
#      settles rather than failing on a busy one.
#
# Progress goes to a status file, one START and one END line per step with its
# exit code, then ALLDONE. Each step logs in full to its own file; nothing is
# piped through tail, since a buffered pipe makes a live step and a dead one
# look the same.
#
# Usage:
#   scripts/bench_queue.sh [options] QUEUE_FILE
#
# Options:
#   --schedule      run the queue detached from this shell. On Windows this
#                   registers a scheduled task and starts it, because a child of
#                   a terminal session dies with that session's job object even
#                   under nohup; elsewhere it uses setsid and nohup.
#   --dry-run       print the plan and which builds are cached, then stop
#   --out DIR       reports, logs, and the status file (default bench_results/queue-<stamp>)
#   --cache DIR     binary cache (default bench_results/bin)
#   --ref-root DIR  where reference worktrees live (default: next to the repository)
#   --wait-idle S   seconds each A/B waits for an idle host (default 3600)
#   --task-name N   scheduled task name for --schedule on Windows (default
#                   PrismBenchQueue); a second queue needs its own while one runs
#
# Exit status: 0 when every step exited 0, otherwise 1. A step's own exit code
# is in the status file; bench_ab.sh uses 1 for FAIL, 2 for a host too busy to
# start, and 3 for RERUN.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCH_AB="$SCRIPT_DIR/bench_ab.sh"

SCHEDULE=""
DRY_RUN=""
OUT_DIR=""
CACHE_DIR=""
REF_ROOT=""
WAIT_IDLE=3600
TASK_NAME="PrismBenchQueue"
QUEUE_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --schedule)  SCHEDULE=1; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --out)       OUT_DIR="$2"; shift 2 ;;
        --cache)     CACHE_DIR="$2"; shift 2 ;;
        --ref-root)  REF_ROOT="$2"; shift 2 ;;
        --wait-idle) WAIT_IDLE="$2"; shift 2 ;;
        --task-name) TASK_NAME="$2"; shift 2 ;;
        -h|--help)   awk 'NR > 1 && !/^#/ { exit } NR > 1' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)          echo "Unknown option: $1" >&2; exit 1 ;;
        *)           QUEUE_FILE="$1"; shift ;;
    esac
done

if [[ -z "$QUEUE_FILE" || ! -f "$QUEUE_FILE" ]]; then
    echo "Error: name a queue file. See --help for its format." >&2
    exit 1
fi
QUEUE_FILE="$(cd "$(dirname "$QUEUE_FILE")" && pwd)/$(basename "$QUEUE_FILE")"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/bench_results/queue-$(date +%Y-%m-%d_%H%M%S)}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_DIR/bench_results/bin}"
REF_ROOT="${REF_ROOT:-$(dirname "$PROJECT_DIR")}"
mkdir -p "$OUT_DIR" "$CACHE_DIR" "$REF_ROOT"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
CACHE_DIR="$(cd "$CACHE_DIR" && pwd)"
REF_ROOT="$(cd "$REF_ROOT" && pwd)"
STATUS="$OUT_DIR/status.txt"

# A scheduled task starts from a bare environment, so put cargo on PATH here
# rather than trusting the caller's profile.
export PATH="$HOME/.cargo/bin:$PATH"

# --- Detach ---

if [[ -n "$SCHEDULE" ]]; then
    args=(--out "$OUT_DIR" --cache "$CACHE_DIR" --ref-root "$REF_ROOT" --wait-idle "$WAIT_IDLE" "$QUEUE_FILE")
    if command -v schtasks >/dev/null 2>&1 && command -v cygpath >/dev/null 2>&1; then
        task="$TASK_NAME"
        runner="$OUT_DIR/run.sh"
        {
            echo "#!/usr/bin/env bash"
            printf 'exec bash %q' "$SELF"
            printf ' %q' "${args[@]}"
            printf ' > %q 2>&1\n' "$OUT_DIR/queue.log"
        } > "$runner"
        bash_exe="$(cygpath -w "$(command -v bash)")"
        # The task runs once, now. Its trigger is set far ahead so it cannot
        # fire a second copy; /Run starts this one.
        MSYS_NO_PATHCONV=1 schtasks /Create /TN "$task" \
            /TR "\"$bash_exe\" -l \"$(cygpath -w "$runner")\"" \
            /SC ONCE /SD 12/31/2030 /ST 23:59 /F >/dev/null || exit 1
        MSYS_NO_PATHCONV=1 schtasks /Run /TN "$task" >/dev/null || exit 1
        echo "Scheduled as task $task; it survives this shell."
    else
        setsid nohup bash "$SELF" "${args[@]}" > "$OUT_DIR/queue.log" 2>&1 < /dev/null &
        echo "Detached with setsid, pid $!."
    fi
    echo "Status: $STATUS"
    exit 0
fi

# --- Plan ---

STEP_KIND=()
STEP_NAME=()
STEP_DIR=()
STEP_ARGS=()

# Steps keep their arguments joined on the unit separator, so a regex keeps its
# quoting between the plan and the run.
join_args() {
    local joined="" arg
    for arg in "$@"; do joined+="$arg"$'\x1f'; done
    printf '%s' "$joined"
}

split_args() {
    local IFS=$'\x1f'
    read -r -a SPLIT <<< "$1"
}

ab() {
    local name="$1" dir="$2"
    shift 2
    STEP_KIND+=(ab); STEP_NAME+=("$name"); STEP_DIR+=("$(cd "$dir" && pwd)")
    STEP_ARGS+=("$(join_args "$@")")
}

run() {
    local name="$1" dir="$2"
    shift 2
    [[ "${1:-}" == "--" ]] && shift
    STEP_KIND+=(run); STEP_NAME+=("$name"); STEP_DIR+=("$(cd "$dir" && pwd)")
    STEP_ARGS+=("$(join_args "$@")")
}

# shellcheck source=/dev/null
source "$QUEUE_FILE"

if (( ${#STEP_NAME[@]} == 0 )); then
    echo "Error: $QUEUE_FILE defines no steps." >&2
    exit 1
fi

# The options of an A/B step that decide which binary it needs.
step_build_options() {
    split_args "$1"
    REF_ARG="HEAD"; FEATURES_ARG="parallel"; BENCH_ARG="circuits"
    local i
    for (( i = 0; i < ${#SPLIT[@]}; i++ )); do
        case "${SPLIT[i]}" in
            --ref|-r)   REF_ARG="${SPLIT[i + 1]}" ;;
            --features) FEATURES_ARG="${SPLIT[i + 1]}" ;;
            --bench|-b) BENCH_ARG="${SPLIT[i + 1]}" ;;
        esac
    done
}

TOOLCHAIN_KEY="$(rustc --version 2>/dev/null | git hash-object --stdin | cut -c1-8)"

# The cache file for the checkout at $1 at commit $2, holding any uncommitted
# change in the key.
cache_path() {
    local dir="$1" sha="$2" dirty=""
    if [[ "$3" == "worktree" && -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]]; then
        dirty="-dirty$(git -C "$dir" diff HEAD --binary | git hash-object --stdin | cut -c1-8)"
    fi
    printf '%s/%s-%s%s-%s-%s' "$CACHE_DIR" "$BENCH_ARG" "$sha" "$dirty" \
        "$(printf '%s' "$FEATURES_ARG" | tr -c 'a-zA-Z0-9\n' '_')" "$TOOLCHAIN_KEY"
}

NEW_EXE_OF=()
REF_EXE_OF=()
REF_SHA_OF=()
for (( s = 0; s < ${#STEP_NAME[@]}; s++ )); do
    NEW_EXE_OF+=(""); REF_EXE_OF+=(""); REF_SHA_OF+=("")
    [[ "${STEP_KIND[s]}" == ab ]] || continue
    step_build_options "${STEP_ARGS[s]}"
    dir="${STEP_DIR[s]}"
    new_sha="$(git -C "$dir" rev-parse --short=10 HEAD)"
    ref_sha="$(git -C "$dir" rev-parse --short=10 "$REF_ARG")" || { echo "Error: step ${STEP_NAME[s]}: unknown ref $REF_ARG" >&2; exit 1; }
    NEW_EXE_OF[s]="$(cache_path "$dir" "$new_sha" worktree)"
    REF_EXE_OF[s]="$(cache_path "$dir" "$ref_sha" ref)"
    REF_SHA_OF[s]="$ref_sha"
done

echo "=== Queue: $QUEUE_FILE ==="
for (( s = 0; s < ${#STEP_NAME[@]}; s++ )); do
    if [[ "${STEP_KIND[s]}" == ab ]]; then
        hit_new="build"; [[ -f "${NEW_EXE_OF[s]}" ]] && hit_new="cached"
        hit_ref="build"; [[ -f "${REF_EXE_OF[s]}" ]] && hit_ref="cached"
        printf '  %-3d ab   %-20s %s (new %s, ref %s %s)\n' "$((s + 1))" "${STEP_NAME[s]}" \
            "${STEP_DIR[s]}" "$hit_new" "${REF_SHA_OF[s]}" "$hit_ref"
    else
        split_args "${STEP_ARGS[s]}"
        printf '  %-3d run  %-20s %s: %s\n' "$((s + 1))" "${STEP_NAME[s]}" "${STEP_DIR[s]}" "${SPLIT[*]}"
    fi
done
echo "  out:   $OUT_DIR"
echo "  cache: $CACHE_DIR"
[[ -n "$DRY_RUN" ]] && exit 0

: > "$STATUS"
mark() { echo "$1 $(date '+%Y-%m-%d %H:%M:%S')" >> "$STATUS"; }

# --- Build ---

# Build the checkout at $1 into $2 unless it is cached. A reference commit gets
# its own worktree under --ref-root, reused across queues.
build_into() {
    local dir="$1" exe="$2" features="$3" bench="$4" log="$5"
    [[ -f "$exe" ]] && return 0
    bash "$BENCH_AB" --build-only "$exe.tmp" --build-dir "$dir" --features "$features" \
        --bench "$bench" > "$log" 2>&1 || return 1
    mv "$exe.tmp" "$exe"
}

mark "START build"
build_failed=0
for (( s = 0; s < ${#STEP_NAME[@]}; s++ )); do
    [[ "${STEP_KIND[s]}" == ab ]] || continue
    step_build_options "${STEP_ARGS[s]}"
    name="${STEP_NAME[s]}"
    if ! build_into "${STEP_DIR[s]}" "${NEW_EXE_OF[s]}" "$FEATURES_ARG" "$BENCH_ARG" "$OUT_DIR/$name-build-new.log"; then
        mark "BUILD-FAILED $name new"; build_failed=1; continue
    fi
    ref_dir="$REF_ROOT/prism-q-ref-${REF_SHA_OF[s]}"
    if [[ ! -f "${REF_EXE_OF[s]}" ]]; then
        if [[ ! -e "$ref_dir/.git" ]]; then
            git -C "${STEP_DIR[s]}" worktree add --detach --force "$ref_dir" "${REF_SHA_OF[s]}" > /dev/null 2>&1 || {
                mark "BUILD-FAILED $name ref worktree"; build_failed=1; continue
            }
        fi
        if ! build_into "$ref_dir" "${REF_EXE_OF[s]}" "$FEATURES_ARG" "$BENCH_ARG" "$OUT_DIR/$name-build-ref.log"; then
            mark "BUILD-FAILED $name ref"; build_failed=1; continue
        fi
    fi
done
mark "END build failed=$build_failed"

# --- Measure ---

any_failed=0
for (( s = 0; s < ${#STEP_NAME[@]}; s++ )); do
    name="${STEP_NAME[s]}"
    split_args "${STEP_ARGS[s]}"
    mark "START $name"
    if [[ "${STEP_KIND[s]}" == ab ]]; then
        if [[ ! -f "${NEW_EXE_OF[s]}" || ! -f "${REF_EXE_OF[s]}" ]]; then
            mark "END $name exit=build-failed"; any_failed=1; continue
        fi
        bash "$BENCH_AB" --project-dir "${STEP_DIR[s]}" "${SPLIT[@]}" \
            --new-exe "${NEW_EXE_OF[s]}" --ref-exe "${REF_EXE_OF[s]}" \
            --wait-idle "$WAIT_IDLE" --out "$OUT_DIR/$name.md" > "$OUT_DIR/$name.log" 2>&1
        status=$?
    else
        ( cd "${STEP_DIR[s]}" && "${SPLIT[@]}" ) > "$OUT_DIR/$name.log" 2>&1
        status=$?
    fi
    (( status == 0 )) || any_failed=1
    mark "END $name exit=$status"
done
mark "ALLDONE"

# One line per step for whoever reads the status file first.
{
    echo ""
    for (( s = 0; s < ${#STEP_NAME[@]}; s++ )); do
        name="${STEP_NAME[s]}"
        if [[ -f "$OUT_DIR/$name.md" ]]; then
            verdict="$(grep -m1 -E '^\*\*Regression verdict' "$OUT_DIR/$name.md")"
            final="$(grep -E '^\*\*Final verdict' "$OUT_DIR/$name.md" | tail -1)"
            printf '%s: %s\n' "$name" "${final:-$verdict}"
        fi
    done
} >> "$STATUS"
exit "$any_failed"

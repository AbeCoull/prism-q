#!/usr/bin/env bash
# Measure several branches in one A/B: merge them onto one base in a throwaway
# checkout, then compare its tip with the base on the union of their rows.
#
# One build of the combined tree and one measurement replace one A/B per branch.
# A PASS clears every branch in the batch. A FAIL names rows, not branches: run
# the batch again with --only on each half to find the branch, about log2(N)
# more A/Bs. Batch only branches whose rows do not share code with another
# branch's change, since a speedup in one can hide a regression in the other on
# a shared row; a branch whose own counts moved on a shared row gets its own A/B.
#
# A batch file holds one branch per line with the bench target and the rows that
# cover it, and optionally rows it must not move. Blank lines and # comments are
# skipped:
#
#   perf/fused2q-pairs  circuits          ^statevector/(qft_textbook|qv)/20$
#   perf/mps-center     circuits          ^mps/random_d10/16$  ^product/scaling_d10/64$
#   perf/qec-walk       bench_shots_perf  ^qec_detector_sampling/surf_p001_drop/5$
#
# The script merges the branches with --no-ff in file order, stops on the first
# conflict and names it, then writes a queue file with one `ab` step per bench
# target, its filter and control the union of that target's lines. Without
# --schedule it prints the queue's plan; with it the queue runs detached.
#
# Usage:
#   scripts/bench_batch.sh [options] BATCH_FILE [-- STEP OPTIONS]
#
# Options:
#   --base REF      commit every branch merges onto and the A/B compares with
#                   (default origin/main)
#   --only LIST     merge only these lines of the batch, 1-based and comma
#                   separated, to bisect a FAIL
#   --dir DIR       combined checkout, reused between batches so dependencies
#                   stay built (default prism-q-batch beside the repository)
#   --schedule      run the queue through bench_queue.sh --schedule
#   --task-name N   scheduled task name (default PrismBenchBatch)
#
# Options after `--` go to every `ab` step, for example `-- --light --escalate`.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$(dirname "$(git -C "$SCRIPT_DIR" rev-parse --path-format=absolute --git-common-dir)")"

BASE="origin/main"
ONLY=""
DIR=""
SCHEDULE=""
TASK_NAME="PrismBenchBatch"
BATCH_FILE=""
STEP_OPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)      BASE="$2"; shift 2 ;;
        --only)      ONLY="$2"; shift 2 ;;
        --dir)       DIR="$2"; shift 2 ;;
        --schedule)  SCHEDULE=1; shift ;;
        --task-name) TASK_NAME="$2"; shift 2 ;;
        --)          shift; STEP_OPTS=("$@"); break ;;
        -h|--help)   awk 'NR > 1 && !/^#/ { exit } NR > 1' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)          echo "Unknown option: $1" >&2; exit 1 ;;
        *)           BATCH_FILE="$1"; shift ;;
    esac
done

if [[ -z "$BATCH_FILE" || ! -f "$BATCH_FILE" ]]; then
    echo "Error: name a batch file. See --help for its format." >&2
    exit 1
fi
DIR="${DIR:-$(dirname "$MAIN_DIR")/prism-q-batch}"
base_sha="$(git -C "$MAIN_DIR" rev-parse --short=10 "$BASE")" || exit 1

BRANCHES=()
TARGETS=()
FILTERS=()
CONTROLS=()
line_no=0
while read -r branch target filter control _; do
    [[ -z "$branch" || "$branch" == \#* ]] && continue
    line_no=$((line_no + 1))
    if [[ -n "$ONLY" && ",$ONLY," != *",$line_no,"* ]]; then
        continue
    fi
    if [[ -z "$filter" ]]; then
        echo "Error: line $line_no of $BATCH_FILE names no rows for $branch." >&2
        exit 1
    fi
    git -C "$MAIN_DIR" rev-parse --verify -q "$branch^{commit}" > /dev/null || {
        echo "Error: line $line_no names $branch, which is not a commit here." >&2
        exit 1
    }
    BRANCHES+=("$branch"); TARGETS+=("$target"); FILTERS+=("$filter"); CONTROLS+=("${control:-}")
done < "$BATCH_FILE"

if (( ${#BRANCHES[@]} == 0 )); then
    echo "Error: no branches selected from $BATCH_FILE." >&2
    exit 1
fi

# --- Combined tree ---

if [[ ! -e "$DIR/.git" ]]; then
    git -C "$MAIN_DIR" worktree add -q --detach "$DIR" "$base_sha" || exit 1
fi
git -C "$DIR" merge --abort > /dev/null 2>&1
git -C "$DIR" checkout -q -f --detach "$base_sha" || exit 1

echo "=== Batch onto $base_sha in $DIR ==="
merged=" "
for (( i = 0; i < ${#BRANCHES[@]}; i++ )); do
    b="${BRANCHES[i]}"
    [[ "$merged" == *" $b "* ]] && continue
    merged+="$b "
    if ! git -C "$DIR" merge -q --no-ff --no-verify --no-edit -m "batch: $b" "$b" > /dev/null 2>&1; then
        echo "Error: $b conflicts with the branches merged before it:" >&2
        git -C "$DIR" diff --name-only --diff-filter=U | sed 's/^/    /' >&2
        git -C "$DIR" merge --abort
        echo "  Rebase it, or measure it in a batch of its own." >&2
        exit 1
    fi
    printf '  merged %-32s %s\n' "$b" "$(git -C "$MAIN_DIR" rev-parse --short=10 "$b")"
done

# --- Queue ---

# The union of the filter, or with $2 = control the control, regexes for target $1.
union() {
    local target="$1" i re out=""
    for (( i = 0; i < ${#BRANCHES[@]}; i++ )); do
        [[ "${TARGETS[i]}" == "$target" ]] || continue
        re="${FILTERS[i]}"
        [[ "${2:-}" == control ]] && re="${CONTROLS[i]}"
        [[ -n "$re" ]] && out+="${out:+|}($re)"
    done
    printf '%s' "$out"
}

queue="$DIR.queue.sh"
{
    printf '# Batch onto %s:' "$base_sha"
    printf ' %s' "${BRANCHES[@]}"
    printf '\n'
    for target in $(printf '%s\n' "${TARGETS[@]}" | sort -u); do
        filter="$(union "$target")"
        control="$(union "$target" control)"
        printf 'ab %q %q -r %q -b %q -f %q' "$target" "$DIR" "$base_sha" "$target" "$filter"
        [[ -n "$control" ]] && printf ' -c %q' "$control"
        (( ${#STEP_OPTS[@]} )) && printf ' %q' "${STEP_OPTS[@]}"
        printf '\n'
    done
} > "$queue"
echo "  queue: $queue"

if [[ -n "$SCHEDULE" ]]; then
    exec bash "$SCRIPT_DIR/bench_queue.sh" --schedule --task-name "$TASK_NAME" "$queue"
fi
exec bash "$SCRIPT_DIR/bench_queue.sh" --dry-run "$queue"

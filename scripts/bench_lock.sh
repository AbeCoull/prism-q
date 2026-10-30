#!/usr/bin/env bash
# The host-wide marker for a measurement in progress, shared by bench_queue.sh
# and bench_ab.sh. Sourced, not run.
#
# The marker is bench_results/MEASURING in the main checkout, whichever worktree
# runs the measurement, so every session of this repository on the host sees
# the same file; PRISM_BENCH_LOCK names another path. It exists exactly while
# measured passes run, and its lines say who holds it:
#
#   pid: 1234
#   by: bench_queue.sh
#   task: PrismBenchQueue
#   out: /c/work/prism-q/bench_results/queue-2026-10-17_101010
#   started: 2026-10-17 10:10:10
#
# A session that sees the file must not run cargo, or anything else that loads
# the host, until it is gone. A marker whose pid is no longer running is stale:
# it is overwritten and the log says so. PRISM_BENCH_LOCK_HELD=1 in the
# environment means a parent process holds the marker already, so a child skips
# it; the queue exports it for every step and bench_ab.sh for its nested runs.

BENCH_LOCK=""
BENCH_LOCK_HELD=""

bench_lock_path() {
    if [[ -n "${PRISM_BENCH_LOCK:-}" ]]; then
        printf '%s' "$PRISM_BENCH_LOCK"
        return 0
    fi
    local common
    common="$(git -C "$1" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" \
        || common="$1/.git"
    printf '%s/bench_results/MEASURING' "$(dirname "$common")"
}

# Write the marker for this process. $1 is a checkout on the host, $2 the
# script name, $3 the task, $4 the out path, and a non-empty $5 takes the
# marker from a live holder. Returns 1 when a live process holds it.
bench_lock_acquire() {
    local dir="$1" by="$2" task="$3" out="$4" force="${5:-}"
    [[ -n "${PRISM_BENCH_LOCK_HELD:-}" ]] && return 0
    BENCH_LOCK="$(bench_lock_path "$dir")"
    if [[ -f "$BENCH_LOCK" ]]; then
        local pid
        pid="$(awk '/^pid: / { print $2; exit }' "$BENCH_LOCK")"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            if [[ -z "$force" ]]; then
                echo "Error: a measurement is running on this host. $BENCH_LOCK says:" >&2
                sed 's/^/    /' "$BENCH_LOCK" >&2
                echo "  Wait for it to finish, or pass --force-lock to measure anyway." >&2
                return 1
            fi
            echo ">>> taking $BENCH_LOCK from live pid $pid under --force-lock"
        else
            echo ">>> stale $BENCH_LOCK from pid ${pid:-unknown}, which is not running; overwriting it"
        fi
    fi
    mkdir -p "$(dirname "$BENCH_LOCK")"
    printf 'pid: %s\nby: %s\ntask: %s\nout: %s\nstarted: %s\n' \
        "$$" "$by" "$task" "$out" "$(date '+%Y-%m-%d %H:%M:%S')" > "$BENCH_LOCK"
    BENCH_LOCK_HELD=1
    export PRISM_BENCH_LOCK_HELD=1
    echo ">>> measuring: wrote $BENCH_LOCK (pid $$); no cargo on this host until it is gone"
    return 0
}

# Wait while a live process other than a parent holds the marker, polling every
# 30 seconds for at most $2 seconds. $1 is a checkout on the host. Returns 1 when
# the marker is still held at the limit. A build waits here, since cargo on the
# host is exactly what a measurement in progress must not see.
bench_lock_wait() {
    local dir="$1" limit="$2" waited=0 lock pid
    [[ -n "${PRISM_BENCH_LOCK_HELD:-}" ]] && return 0
    lock="$(bench_lock_path "$dir")"
    while [[ -f "$lock" ]]; do
        pid="$(awk '/^pid: / { print $2; exit }' "$lock")"
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        (( waited >= limit )) && return 1
        (( waited == 0 )) && echo ">>> waiting for the measurement holding $lock (pid $pid) to finish"
        sleep 30
        waited=$((waited + 30))
    done
    return 0
}

# Remove the marker this process wrote. One that another process has since
# taken under --force-lock is left to that process.
bench_lock_release() {
    [[ -n "$BENCH_LOCK_HELD" ]] || return 0
    BENCH_LOCK_HELD=""
    if [[ -f "$BENCH_LOCK" ]] && ! grep -q "^pid: $$\$" "$BENCH_LOCK"; then
        echo ">>> measuring done: $BENCH_LOCK is held by another process now, left in place"
        return 0
    fi
    rm -f "$BENCH_LOCK"
    echo ">>> measuring done: removed $BENCH_LOCK"
    return 0
}

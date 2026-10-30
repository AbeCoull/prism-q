#!/usr/bin/env bash
# Verify RERUN retries, --build-only, and the build-phase lock wait in
# scripts/bench_queue.sh against a stub bench_ab.sh.
#
# Each case copies bench_queue.sh and bench_lock.sh into a scratch tree beside a
# stub that fills the cache on --build-only and otherwise writes a canned report
# and exit status per measured run. The queue measures a scratch repository with
# two commits, so the reference worktree and cache keys are real.
#
# Usage: scripts/bench_queue_test.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$(mktemp -d)"
SLEEPER=""
trap '[[ -n "$SLEEPER" ]] && kill "$SLEEPER" 2>/dev/null; rm -rf "$SCRATCH"' EXIT

PASS=0
FAIL=0

REPO="$SCRATCH/repo"
git init -q "$REPO"
git -C "$REPO" -c user.name=t -c user.email=t@t commit -q --allow-empty -m one
git -C "$REPO" -c user.name=t -c user.email=t@t commit -q --allow-empty -m two

# Cases name the exit status of each measured run, space separated.
setup() {
    local tree="$SCRATCH/$1"
    mkdir -p "$tree/scripts"
    cp "$SCRIPT_DIR/bench_queue.sh" "$SCRIPT_DIR/bench_lock.sh" "$tree/scripts/"
    cat > "$tree/scripts/bench_ab.sh" <<'STUB'
#!/usr/bin/env bash
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-only) : > "$2"; exit 0 ;;
        --out) out="$2"; shift 2 ;;
        *) shift ;;
    esac
done
runs=$(( $(cat "$dir/runs" 2>/dev/null || echo 0) + 1 ))
echo "$runs" > "$dir/runs"
status="$(tr ' ' '\n' < "$dir/spec" | sed -n "${runs}p")"
printf '**Regression verdict**: run %s exit %s\n' "$runs" "$status" > "$out"
exit "$status"
STUB
    printf '%s' "$2" > "$tree/scripts/spec"
    printf 'ab step %q -r HEAD~1 -f x\n' "$REPO" > "$tree/queue.sh"
}

# Run the queue in case $1 with extra options, then compare its exit status,
# measured runs, and status-file lines against the expectations.
check() {
    local label="$1" want="$2" want_runs="$3" want_status="$4"
    shift 4
    local tree="$SCRATCH/$label" got runs
    PRISM_BENCH_LOCK="$tree/MEASURING" bash "$tree/scripts/bench_queue.sh" \
        --out "$tree/out" --cache "$tree/cache" --ref-root "$tree/refs" --wait-idle 0 \
        "$@" "$tree/queue.sh" > "$tree/log" 2>&1
    got=$?
    runs="$(cat "$tree/scripts/runs" 2>/dev/null || echo 0)"
    if [[ "$got" == "$want" && "$runs" == "$want_runs" ]] &&
       grep -qE "$want_status" "$tree/out/status.txt"; then
        PASS=$((PASS + 1))
        printf 'ok    %-32s exit %s, %s run(s)\n' "$label" "$got" "$runs"
    else
        FAIL=$((FAIL + 1))
        printf 'FAIL  %-32s want exit %s/%s run(s)/%s, got %s/%s\n' \
            "$label" "$want" "$want_runs" "$want_status" "$got" "$runs"
        sed 's/^/      /' "$tree/log" "$tree/out/status.txt" 2>/dev/null
    fi
}

echo "=== bench_queue retries, build-only, lock wait ==="

setup pass "0"
check pass 0 1 '^END step exit=0'

setup rerun-then-pass "3 0"
check rerun-then-pass 0 2 '^RETRY step attempt 2 after exit=3'
if [[ -f "$SCRATCH/rerun-then-pass/out/step-try1/step.md" ]] &&
   grep -q 'run 2 exit 0' "$SCRATCH/rerun-then-pass/out/step.md" &&
   grep -q '^step: .*1 earlier RERUN attempts' "$SCRATCH/rerun-then-pass/out/status.txt"; then
    PASS=$((PASS + 1)); echo "ok    rerun-then-pass keeps attempt 1 aside"
else
    FAIL=$((FAIL + 1)); echo "FAIL  rerun-then-pass did not keep attempt 1 in step-try1/"
fi

setup rerun-exhausted "3 3 3 0"
check rerun-exhausted 1 3 '^END step exit=3'

setup retries-off "3 0"
check retries-off 1 1 '^END step exit=3' --retries 0

setup fail-not-retried "1 0"
check fail-not-retried 1 1 '^END step exit=1'

setup host-busy-not-retried "2 0"
check host-busy-not-retried 1 1 '^END step exit=2'

setup build-only "0"
check build-only 0 0 '^ALLDONE' --build-only
if (( $(find "$SCRATCH/build-only/cache" -type f | wc -l) == 2 )); then
    PASS=$((PASS + 1)); echo "ok    build-only fills both cache entries"
else
    FAIL=$((FAIL + 1)); echo "FAIL  build-only did not fill both cache entries"
fi

setup lock-live "0"
sleep 300 &
SLEEPER=$!
printf 'pid: %s\nby: test\n' "$SLEEPER" > "$SCRATCH/lock-live/MEASURING"
check lock-live 1 0 '^LOCKED'
if [[ ! -d "$SCRATCH/lock-live/cache" ]] || (( $(find "$SCRATCH/lock-live/cache" -type f | wc -l) == 0 )); then
    PASS=$((PASS + 1)); echo "ok    lock-live built nothing"
else
    FAIL=$((FAIL + 1)); echo "FAIL  lock-live built while the marker was held"
fi

setup lock-stale "0"
printf 'pid: 999999\nby: test\n' > "$SCRATCH/lock-stale/MEASURING"
check lock-stale 0 1 '^END step exit=0'

echo
echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]

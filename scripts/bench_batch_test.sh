#!/usr/bin/env bash
# Verify the combined tree and queue file scripts/bench_batch.sh writes, against
# a stub bench_queue.sh.
#
# A scratch repository holds a base commit and three branches, two that merge
# cleanly and one that conflicts with the first. The stub records its arguments,
# so each case checks what was merged and what the queue file asks to measure.
#
# Usage: scripts/bench_batch_test.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

PASS=0
FAIL=0

REPO="$SCRATCH/repo"
git init -q -b main "$REPO"
g() { git -C "$REPO" -c user.name=t -c user.email=t@t "$@"; }
echo base > "$REPO/a.txt"; g add a.txt; g commit -q -m base
g checkout -q -b one;      echo one > "$REPO/one.txt";   g add one.txt;   g commit -q -m one
g checkout -q -b two main; echo two > "$REPO/two.txt";   g add two.txt;   g commit -q -m two
g checkout -q -b clash main; echo clash > "$REPO/one.txt"; g add one.txt; g commit -q -m clash
g checkout -q main
mkdir -p "$REPO/scripts"
cp "$SCRIPT_DIR/bench_batch.sh" "$REPO/scripts/"
cat > "$REPO/scripts/bench_queue.sh" <<'STUB'
#!/usr/bin/env bash
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$*" > "$dir/queue-args"
exit 0
STUB

cat > "$SCRATCH/batch.txt" <<'EOF'
# branch  target            rows                     control
one       circuits          ^sv/one/20$              ^mps/c/16$
two       circuits          ^sv/two/(16|20)$

two       bench_shots_perf  ^qec/two/5$
EOF

batch() {
    bash "$REPO/scripts/bench_batch.sh" --base main --dir "$SCRATCH/combined" "$@" > "$SCRATCH/log" 2>&1
}

ok() { PASS=$((PASS + 1)); printf 'ok    %s\n' "$1"; }
bad() {
    FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$1"
    sed 's/^/      /' "$SCRATCH/log" "$SCRATCH/combined.queue.sh" 2>/dev/null
}

# Decode the queue file the way bench_queue.sh does, one line per step:
# NAME|FILTER|CONTROL|EXTRA.
steps() {
    (
        ab() {
            local name="$1" f="" c="" extra=""
            shift 2
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    -f) f="$2"; shift 2 ;;
                    -c) c="$2"; shift 2 ;;
                    -r|-b) shift 2 ;;
                    *) extra+="$1 "; shift ;;
                esac
            done
            echo "$name|$f|$c|${extra% }"
        }
        # shellcheck source=/dev/null
        source "$SCRATCH/combined.queue.sh"
    )
}

echo "=== bench_batch combined tree and queue ==="

batch "$SCRATCH/batch.txt" -- --light --escalate
status=$?
want=$'bench_shots_perf|(^qec/two/5$)||--light --escalate\ncircuits|(^sv/one/20$)|(^sv/two/(16|20)$)|(^mps/c/16$)|--light --escalate'
if [[ "$status" -eq 0 && "$(steps)" == "$want" ]] && grep -q -- '^--dry-run ' "$REPO/scripts/queue-args" \
    && [[ "$(grep -c '^  merged two ' "$SCRATCH/log")" == 1 ]] \
    && git -C "$SCRATCH/combined" merge-base --is-ancestor one HEAD \
    && git -C "$SCRATCH/combined" merge-base --is-ancestor two HEAD; then
    ok "union per target, both branches merged, plan only"
else
    bad "union per target"; steps
fi

batch --only 1 "$SCRATCH/batch.txt"
if [[ "$(steps)" == 'circuits|(^sv/one/20$)|(^mps/c/16$)|' ]] \
    && ! git -C "$SCRATCH/combined" merge-base --is-ancestor two HEAD; then
    ok "--only keeps one line and resets the tree"
else
    bad "--only"; steps
fi

printf 'one circuits ^x$\nclash circuits ^y$\n' > "$SCRATCH/clash.txt"
batch "$SCRATCH/clash.txt"
if [[ $? -eq 1 ]] && grep -q 'clash conflicts' "$SCRATCH/log" && grep -q 'one.txt' "$SCRATCH/log" \
    && [[ -z "$(git -C "$SCRATCH/combined" status --porcelain --untracked-files=no)" ]]; then
    ok "conflict names the branch and aborts the merge"
else
    bad "conflict"
fi

printf 'nosuch circuits ^x$\n' > "$SCRATCH/missing.txt"
batch "$SCRATCH/missing.txt"
if [[ $? -eq 1 ]] && grep -q 'nosuch, which is not a commit' "$SCRATCH/log"; then
    ok "unknown branch is an error"
else
    bad "unknown branch"
fi

printf '#!/bin/sh\nexit 1\n' > "$REPO/.git/hooks/commit-msg"
chmod +x "$REPO/.git/hooks/commit-msg"
batch "$SCRATCH/batch.txt"
status=$?
rm "$REPO/.git/hooks/commit-msg"
if [[ "$status" -eq 0 ]] && git -C "$SCRATCH/combined" merge-base --is-ancestor two HEAD; then
    ok "a commit-msg hook does not block the batch merges"
else
    bad "commit-msg hook"
fi

batch --schedule --task-name T "$SCRATCH/batch.txt"
if grep -q -- '^--schedule --task-name T ' "$REPO/scripts/queue-args"; then
    ok "--schedule hands the queue to bench_queue.sh"
else
    bad "--schedule"
fi

echo
echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]

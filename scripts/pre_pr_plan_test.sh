#!/usr/bin/env bash
# Verify which checks scripts/pre_pr.sh plans for a set of changed paths.
#
# Each case hands `pre_pr.sh --plan` a list of paths through PRE_PR_CHANGED_FILES,
# so nothing is built and no diff is read, then checks the action the plan lists
# for each named check: run, skip, owed, or `-` for a check the plan leaves out.
# Paths that need no content may be invented; a case that relies on a file's
# `feature = "..."` text or on a manifest target names a real file.
#
# Usage: scripts/pre_pr_plan_test.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

PASS=0
FAIL=0

# check LABEL "PATH [PATH ...]" "CHECK=ACTION [CHECK=ACTION ...]"
check() {
    local label="$1" paths="$2" wants="$3"
    local list="$SCRATCH/$label.paths" log="$SCRATCH/$label.log"
    tr ' ' '\n' <<< "$paths" > "$list"
    PRE_PR_CHANGED_FILES="$list" bash "$SCRIPT_DIR/pre_pr.sh" --plan --base HEAD > "$log" 2>&1
    local status=$? want name action got bad=""
    [[ "$status" -eq 0 ]] || bad=" exit $status"
    for want in $wants; do
        name="${want%%=*}"
        action="${want#*=}"
        got="$(awk -v n="$name" '$1 == n { print $2; found = 1; exit } END { if (!found) print "-" }' "$log")"
        [[ "$got" == "$action" ]] || bad+=" $name=$got (want $action)"
    done
    if [[ -z "$bad" ]]; then
        PASS=$((PASS + 1))
        printf 'ok    %s\n' "$label"
    else
        FAIL=$((FAIL + 1))
        printf 'FAIL  %s:%s\n' "$label" "$bad"
        sed 's/^/      /' "$log"
    fi
}

echo "=== pre_pr.sh --plan triggers ==="

check gpu-kernel src/gpu/kernels/x.rs \
    "fmt=run clippy=run check-gpu-dist=run owed-gpu=owed nextest-dist=skip check-bench-internal=skip nextest-pulse=skip"
check gpu-module-outside-src-gpu src/backend/stabilizer/gpu.rs \
    "clippy=run check-gpu-dist=run owed-gpu=owed check-bench-internal=skip"
check gpu-noise-cache src/sim/noise/gpu_cache.rs \
    "check-gpu-dist=run owed-gpu=owed"
check gpu-golden tests/golden_gpu.rs \
    "nextest=run check-gpu-dist=run owed-gpu=owed doctest=skip"
check dispatch src/sim/dispatch.rs \
    "fmt=run check-gpu-dist=run owed-gpu=- nextest-dist=skip"
check distributed-comm src/distributed/comm.rs \
    "check-gpu-dist=run nextest-dist=run owed-mpi=owed owed-gpu=-"
check pulse src/pulse/x.rs \
    "clippy-pulse=run nextest-pulse=run doc-pulse=run check-gpu-dist=skip owed-gpu=-"
check pulse-test tests/pulse_compile.rs \
    "nextest-pulse=run doctest=skip"
check docs-page docs/x.md \
    "fmt=skip clippy=skip nextest=skip check-gpu-dist=skip check-bench-internal=skip sh-syntax=skip owed-gpu=-"
check script scripts/x.sh \
    "sh-syntax=run clippy=skip check-gpu-dist=skip test:x=-"
check bench-internal-source src/sim/clifford_register.rs \
    "check-bench-internal=run check-gpu-dist=skip owed-gpu=-"
check bench-internal-target tools/sweep_census.rs \
    "check-bench-internal=run check-gpu-dist=skip"
check bench-distributed benches/bench_distributed.rs \
    "check-bench-internal=run check-gpu-dist=run nextest-dist=skip"
check manifest Cargo.toml \
    "fmt=skip clippy=run nextest=run doctest=run check-gpu-dist=run check-bench-internal=run owed-gpu=-"
check gate-script scripts/pre_pr.sh \
    "test:pre_pr_plan=run sh-syntax=run clippy=skip"
check bench-ci-script scripts/bench_ci.sh \
    "test:bench_ci=run test:pre_pr_plan=-"

echo
echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]

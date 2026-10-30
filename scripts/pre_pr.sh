#!/usr/bin/env bash
# Run the local gate plus the extra checks a branch's diff owes before it becomes a
# pull request.
#
# The standard gate builds `parallel` with the local toolchain on one platform. Each
# extra check below mirrors a CI job that has gone red after that gate passed, and
# runs only when the diff can reach it. The diff is read against the merge base of
# HEAD with origin/main and covers committed, staged, unstaged and untracked files.
#
# Checks and what triggers them:
#
#   audit-*            every run. Added lines with an em dash; a bench_variant switch
#                      left in src/ or benches/; an edit to CHANGELOG.md, which
#                      git-cliff writes at release; and, when the checkout carries the
#                      local commit audit (.claude/skills/commit-and-pr/scripts/audit.sh),
#                      added lines naming a local-only path, a tracker ID, or an
#                      external simulator.
#   fmt                a .rs file
#   clippy, nextest    a Rust input: .rs, Cargo manifests, src/, tests/, benches/,
#                      examples/, bindings/, lint or toolchain config
#   doctest            src/, the manifests, or a page src/lib.rs includes for doctests
#   doc-build          a `///` or `//!` line under src/
#   public-api, semver, parity, clippy-py
#                      a `pub` item line under src/, or a line inside a `pub` enum,
#                      struct, trait or union; parity also on public-api.txt or
#                      parity.toml, clippy-py also on bindings/
#   clippy-no-default  a .rs file
#   clippy-<target>    a target_os, target_arch, target_feature or similar cfg line;
#                      the Linux and macOS targets the host has installed
#   check-gpu-dist     src/sim/dispatch.rs or a cfg(feature) line
#   nextest-dist       src/distributed/ or src/backend/distributed_statevector/
#   test:<script>      scripts/X.sh or scripts/X_test.sh, when scripts/X_test.sh exists
#   py-tools           a Python file under scripts/
#   sh-syntax          a .sh file
#   owed-gpu, owed-mpi src/gpu/, or the MPI surface: printed, never run, since they
#                      need the card or the MSVC and MS-MPI environment
#
# The public-api and semver checks skip, saying why, when the pinned nightly or the
# tool is missing. Their pins and feature lists are read from .github/workflows/ci.yml.
#
# Usage:
#   scripts/pre_pr.sh [options]
#
# Options:
#   --base REF     diff against REF instead of the merge base with origin/main
#   --plan         list every check, whether it runs, and what triggered it, then stop
#   --skip-gate    run the audits and the extras, not fmt, clippy, nextest or doctest
#   --fail-fast    stop at the first failing check
#   --title TEXT   the pull request title, counted with the commit subjects when
#                  semver-checks resolves the release type, as CI does
#
# Exit status: 0 when every check passed or was skipped, 1 when any failed, and 2
# when a benchmark measurement holds the host (see bench_lock.sh).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=scripts/bench_lock.sh
source "$SCRIPT_DIR/bench_lock.sh"

BASE=""
PLAN=""
SKIP_GATE=""
FAIL_FAST=""
TITLE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)      BASE="$2"; shift 2 ;;
        --plan)      PLAN=1; shift ;;
        --skip-gate) SKIP_GATE=1; shift ;;
        --fail-fast) FAIL_FAST=1; shift ;;
        --title)     TITLE="$2"; shift 2 ;;
        -h|--help)   awk 'NR > 1 && !/^#/ { exit } NR > 1' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$PROJECT_DIR" || exit 1

LOCK="$(bench_lock_path "$PROJECT_DIR")"
if [[ -f "$LOCK" ]]; then
    lock_pid="$(awk '/^pid: / { print $2; exit }' "$LOCK")"
    if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
        echo "Error: a measurement is running on this host. $LOCK says:" >&2
        sed 's/^/    /' "$LOCK" >&2
        echo "  Run this once it is gone." >&2
        exit 2
    fi
fi

if [[ -n "$BASE" ]]; then
    BASE_SHA="$(git rev-parse --verify --quiet "$BASE^{commit}")" || {
        echo "Error: unknown ref $BASE" >&2; exit 1; }
else
    BASE_SHA="$(git merge-base HEAD origin/main)" || {
        echo "Error: HEAD has no merge base with origin/main; pass --base" >&2; exit 1; }
fi

MAIN_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

# --- Read the diff ---

mapfile -t UNTRACKED < <(git ls-files --others --exclude-standard)
mapfile -t CHANGED < <({ git diff --name-only "$BASE_SHA"; printf '%s\n' "${UNTRACKED[@]}"; } \
    | sed '/^$/d' | sort -u)

# One record per added or removed line, fields joined on the unit separator: + or -,
# the file, its line number on that side, the hunk's context line, and the text.
RECORDS="$SCRATCH/records"
{
    git diff -U0 --no-color --no-ext-diff --src-prefix=a/ --dst-prefix=b/ "$BASE_SHA"
    for f in "${UNTRACKED[@]}"; do
        git diff --no-index -U0 --no-color --src-prefix=a/ --dst-prefix=b/ -- /dev/null "$f"
    done
} | LC_ALL=C awk -v US=$'\x1f' '
    /^diff / { header = 1; next }
    header && /^--- / { old = substr($0, 5); sub(/^a\//, "", old); next }
    header && /^\+\+\+ / {
        new = substr($0, 5); sub(/^b\//, "", new)
        file = (new == "/dev/null") ? old : new
        next
    }
    /^@@ / {
        header = 0
        split($0, h, " ")
        o = h[2]; sub(/^-/, "", o); sub(/,.*/, "", o); oline = o + 0
        n = h[3]; sub(/^\+/, "", n); sub(/,.*/, "", n); nline = n + 0
        ctx = $0; sub(/^@@ [^@]* @@ ?/, "", ctx)
        next
    }
    header { next }
    /^\+/ { print "+" US file US nline US ctx US substr($0, 2); nline++; next }
    /^-/  { print "-" US file US oline US ctx US substr($0, 2); oline++; next }
' > "$RECORDS"

# The first changed file matching the regex.
first_file() {
    printf '%s\n' "${CHANGED[@]}" | grep -m1 -E -- "$1"
}

# The first line matching kinds $1, file $2 and text $3 (regexes), as file:line: text.
first_hit() {
    K="$1" F="$2" T="$3" LC_ALL=C awk -F $'\x1f' '
        $1 ~ ENVIRON["K"] && $2 ~ ENVIRON["F"] && $5 ~ ENVIRON["T"] {
            t = $5; sub(/^[[:space:]]+/, "", t)
            printf "%s:%s: %s\n", $2, $3, substr(t, 1, 70); exit
        }' "$RECORDS"
}

# Like first_hit, matching the hunk's context line instead of a file, and only lines
# that carry code.
first_hit_in_context() {
    F="$1" C="$2" LC_ALL=C awk -F $'\x1f' '
        $2 ~ ENVIRON["F"] && $4 ~ ENVIRON["C"] && $5 !~ /^[[:space:]]*(\/\/.*)?$/ {
            t = $5; sub(/^[[:space:]]+/, "", t)
            printf "%s:%s: %s (in %s)\n", $2, $3, substr(t, 1, 50), substr($4, 1, 40); exit
        }' "$RECORDS"
}

ci_env() {
    sed -n "s/^  $1: //p" .github/workflows/ci.yml | head -1
}

find_python() {
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import pytest' >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

LOCAL_AUDIT=""
for dir in "$PROJECT_DIR" "$MAIN_DIR"; do
    if [[ -f "$dir/.claude/skills/commit-and-pr/scripts/audit.sh" ]]; then
        LOCAL_AUDIT="$dir/.claude/skills/commit-and-pr/scripts/audit.sh"
        break
    fi
done

EM_DASH=$'\xe2\x80\x94'
LINT=(-- -D warnings -D clippy::undocumented_unsafe_blocks)
DOC_FEATURES="$(ci_env CI_DOC_FEATURES)"
DOC_FEATURES="${DOC_FEATURES:-parallel,gpu,distributed}"
API_NIGHTLY="$(ci_env API_NIGHTLY)"
API_TOOL_VERSION="$(ci_env API_TOOL_VERSION)"
HOST_TARGET="$(rustc -vV 2>/dev/null | sed -n 's/^host: //p')"

# --- Checks ---

audit_em_dash() {
    local hits
    hits="$(D="$EM_DASH" LC_ALL=C awk -F $'\x1f' '$1 == "+" && index($5, ENVIRON["D"]) {
        printf "  %s:%s: %s\n", $2, $3, $5 }' "$RECORDS")"
    [[ -z "$hits" ]] && return 0
    echo "Added lines with an em dash; use a comma, colon, period or parentheses:"
    printf '%s\n' "$hits"
    return 1
}

# Lines go to the audit prefixed, so none starts with the `#` it reads as a comment
# and the line numbers it reports map back through added.map. The bench output
# directory is named legitimately by the scripts, workflows and benches/README.md
# that write or document it, so it is stripped there before the audit reads them.
audit_local_refs() {
    local lines="$SCRATCH/added.txt" map="$SCRATCH/added.map" out hits
    : > "$lines"; : > "$map"
    L="$lines" M="$map" LC_ALL=C awk -F $'\x1f' '
        $1 != "+" || $2 == ".gitignore" { next }
        {
            t = $5
            if ($2 ~ /^(scripts|\.github)\// || $2 == "benches/README.md") gsub(/bench_results\//, "", t)
            print "+ " t > ENVIRON["L"]
            print $2 ":" $3 > ENVIRON["M"]
        }' "$RECORDS"
    out="$(cd "$MAIN_DIR" && bash "$LOCAL_AUDIT" prose "$lines")"
    hits="$(printf '%s\n' "$out" \
        | grep -E 'internal tracker ID|path outside the repository|external simulator' \
        | while read -r _ n reason; do
            printf '  %s: %s\n' "$(sed -n "${n%:}p" "$map")" "$reason"
        done)"
    [[ -z "$hits" ]] && return 0
    echo "Added lines that name something local or banned from public text:"
    printf '%s\n' "$hits"
    return 1
}

audit_bench_switch() {
    local hits
    hits="$(git grep -n --untracked -e bench_variant -- src benches \
        | grep -v -E '^(src/env_knobs\.rs|benches/README\.md):' \
        | grep -v -E '^src/lib\.rs:[0-9]+:[[:space:]]*pub use ')"
    [[ -z "$hits" ]] && return 0
    echo "A bench_variant switch is still in the tree. A default build runs variant A,"
    echo "so the measured change never ships until the switch is gone:"
    printf '  %s\n' "$hits"
    return 1
}

audit_changelog() {
    echo "CHANGELOG.md is written by git-cliff at release from the commit subjects."
    echo "Revert the edit; a user-visible change goes in the PR title, with ! when breaking."
    return 1
}

check_fmt()          { cargo fmt --all -- --check; }
check_clippy()       { cargo clippy --all-targets --features parallel "${LINT[@]}"; }
check_nextest()      { cargo nextest run --features parallel; }
check_doctest()      { cargo test --doc --features parallel; }
check_no_default()   { cargo clippy --all-targets --no-default-features "${LINT[@]}"; }
check_gpu_dist()     { cargo check --all-targets --features "parallel gpu distributed"; }
check_clippy_py()    { cargo clippy -p prism-q-py --all-targets "${LINT[@]}"; }
check_nextest_dist() { cargo nextest run --features "parallel distributed" -E 'test(distributed)'; }

check_doc_build() {
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features "${DOC_FEATURES//,/ }"
}

check_cross() {
    cargo clippy --target "$1" --lib --features parallel "${LINT[@]}"
}

check_public_api() {
    local now="$SCRATCH/public-api.txt"
    cargo +"$API_NIGHTLY" public-api --features "$DOC_FEATURES" -sss > "$now" || return 1
    diff -u --strip-trailing-cr docs/architecture/public-api.txt "$now" && return 0
    echo "The public API changed. If intended, regenerate the snapshot with"
    echo "  cargo +$API_NIGHTLY public-api --features \"$DOC_FEATURES\" -sss > docs/architecture/public-api.txt"
    return 1
}

check_semver() {
    local tag subjects bodies level
    tag="$(git describe --tags --abbrev=0 --match 'v[0-9]*')" || return 1
    subjects="$(git log --format=%s "$tag..HEAD")"$'\n'"$TITLE"
    bodies="$(git log --format=%b "$tag..HEAD")"
    level="$(bash -c 'source scripts/release_bump_level.sh && conventional_bump_level "$1" "$2"' \
        _ "$subjects" "$bodies")" || return 1
    level="${level/none/patch}"
    echo "semver-checks against $tag as a $level release"
    cargo semver-checks --package prism-q --baseline-version "${tag#v}" --release-type "$level" \
        --only-explicit-features --features "$DOC_FEATURES"
}

check_parity() {
    "$PYTHON" -m pytest -q bindings/python/tests/test_parity.py -k "not resolves"
}

check_py_tools() {
    "$PYTHON" -m unittest discover -s scripts -p 'test_correctness_tools.py'
}

check_script_test() { bash "$1"; }

check_sh_syntax() {
    local f status=0
    for f in "${SH_FILES[@]}"; do
        [[ -f "$f" ]] || continue
        bash -n "$f" || status=1
    done
    return "$status"
}

# --- Plan ---

NAMES=()
ACTIONS=()
REASONS=()
COMMANDS=()

add() {
    NAMES+=("$1"); ACTIONS+=("$2"); REASONS+=("$3"); COMMANDS+=("${*:4}")
}

run_if() {
    local name="$1" trigger="$2" quiet="$3"
    shift 3
    if [[ -n "$trigger" ]]; then
        add "$name" run "$trigger" "$*"
    else
        add "$name" skip "$quiet"
    fi
}

add audit-em-dash run "every run" audit_em_dash
if [[ -n "$LOCAL_AUDIT" ]]; then
    add audit-local-refs run "every run" audit_local_refs
else
    add audit-local-refs skip "no local commit audit in this checkout"
fi
add audit-bench-switch run "every run" audit_bench_switch
run_if audit-changelog "$(first_file '^CHANGELOG\.md$')" "CHANGELOG.md untouched" audit_changelog

RS_FILE="$(first_file '\.rs$')"
RUST_INPUT="$(printf '%s\n' "${CHANGED[@]}" | grep -v -E '\.md$' \
    | grep -m1 -E '\.rs$|(^|/)Cargo\.(toml|lock)$|^(src|tests|benches|examples|bindings)/|^(clippy|rustfmt|rust-toolchain)\.toml$|^\.cargo/|^\.config/')"
mapfile -t DOC_PAGES < <(sed -n 's/.*include_str!("\.\.\/\([^"]*\.md\)").*/\1/p' src/lib.rs)
DOCTEST_INPUT="$(first_file '^src/|(^|/)Cargo\.(toml|lock)$')"
if [[ -z "$DOCTEST_INPUT" ]]; then
    for page in "${DOC_PAGES[@]}"; do
        DOCTEST_INPUT="$(printf '%s\n' "${CHANGED[@]}" | grep -m1 -x -F -- "$page")" && break
    done
fi

if [[ -n "$SKIP_GATE" ]]; then
    for name in fmt clippy nextest doctest; do add "$name" skip "--skip-gate"; done
else
    run_if fmt "$RS_FILE" "no .rs file changed" check_fmt
    run_if clippy "$RUST_INPUT" "no Rust input changed" check_clippy
    run_if nextest "$RUST_INPUT" "no Rust input changed" check_nextest
    run_if doctest "$DOCTEST_INPUT" "no src/, manifest or doctested page changed" check_doctest
fi

run_if clippy-no-default "$RS_FILE" "no .rs file changed" check_no_default

CROSS_HIT="$(first_hit '^[+-]$' '\.rs$' 'target_(os|arch|feature|family|env|vendor|pointer_width)|cfg!?\((unix|windows)')"
INSTALLED_TARGETS="$(rustup target list --installed 2>/dev/null)"
for target in x86_64-unknown-linux-gnu aarch64-apple-darwin; do
    name="clippy-${target%%-*}-${target#*-*-}"
    if [[ -z "$CROSS_HIT" ]]; then
        add "$name" skip "no platform cfg line changed"
    elif [[ "$target" == "$HOST_TARGET" ]]; then
        add "$name" skip "host target, covered by clippy"
    elif ! grep -qx -- "$target" <<< "$INSTALLED_TARGETS"; then
        add "$name" skip "target missing: rustup target add $target"
    else
        add "$name" run "$CROSS_HIT" check_cross "$target"
    fi
done
CROSS_UNBUILT="$(first_hit '^[+-]$' '^(examples|tests|benches|bindings)/' 'target_(os|arch)')"

GPU_DIST_HIT="$(first_file '^src/sim/dispatch\.rs$')"
GPU_DIST_HIT="${GPU_DIST_HIT:-$(first_hit '^[+-]$' '\.rs$' 'cfg(_attr)?!?\(.*feature')}"
run_if check-gpu-dist "$GPU_DIST_HIT" "no dispatch or cfg(feature) change" check_gpu_dist

PUB_HIT="$(first_hit '^[+-]$' '^src/.*\.rs$' '^[[:space:]]*pub[[:space:]]+((unsafe|async|const|extern)[[:space:]]+)*(fn|struct|enum|trait|type|const|static|mod|use|union)[[:space:]]|^[[:space:]]*pub[[:space:]]+[a-z_][a-z0-9_]*[[:space:]]*:')"
PUB_HIT="${PUB_HIT:-$(first_hit_in_context '^src/.*\.rs$' '^pub (enum|struct|trait|union) ')}"

if [[ -z "$PUB_HIT" ]]; then
    add public-api skip "no public item changed"
elif [[ -z "$API_NIGHTLY" ]] || ! rustup toolchain list 2>/dev/null | grep -q "^$API_NIGHTLY"; then
    add public-api skip "toolchain missing: rustup toolchain install $API_NIGHTLY"
elif ! installed="$(cargo public-api --version 2>/dev/null)"; then
    add public-api skip "tool missing: cargo install cargo-public-api --locked --version $API_TOOL_VERSION"
elif [[ "$installed" != *" $API_TOOL_VERSION"* ]]; then
    add public-api skip "cargo-public-api is ${installed##* }, CI pins $API_TOOL_VERSION"
else
    add public-api run "$PUB_HIT" check_public_api
fi

if [[ -z "$PUB_HIT" ]]; then
    add semver skip "no public item changed"
elif ! cargo semver-checks --version >/dev/null 2>&1; then
    add semver skip "tool missing: cargo install cargo-semver-checks --locked"
else
    add semver run "$PUB_HIT" check_semver
fi

PYTHON="$(find_python)"
PARITY_HIT="${PUB_HIT:-$(first_file '^(docs/architecture/public-api\.txt|bindings/python/parity\.toml)$')}"
if [[ -z "$PARITY_HIT" ]]; then
    add parity skip "no public item changed"
elif [[ -z "$PYTHON" ]]; then
    add parity skip "no Python with pytest on PATH"
else
    add parity run "$PARITY_HIT" check_parity
fi

PY_BINDINGS_HIT="${PUB_HIT:-$(first_file '^bindings/')}"
run_if clippy-py "$PY_BINDINGS_HIT" "no public item or bindings/ change" check_clippy_py

run_if nextest-dist "$(first_file '^src/(distributed|backend/distributed_statevector)/')" \
    "no distributed change" check_nextest_dist

run_if doc-build "$(first_hit '^[+-]$' '^src/.*\.rs$' '^[[:space:]]*//[/!]')" \
    "no docstring changed under src/" check_doc_build

SCRIPT_TESTS=()
for f in "${CHANGED[@]}"; do
    [[ "$f" =~ ^scripts/(.+)\.sh$ ]] || continue
    stem="${BASH_REMATCH[1]%_test}"
    test_script="scripts/${stem}_test.sh"
    [[ -f "$test_script" ]] || continue
    [[ " ${SCRIPT_TESTS[*]} " == *" $test_script "* ]] && continue
    SCRIPT_TESTS+=("$test_script")
    add "test:${stem}" run "$f" check_script_test "$test_script"
done

PY_TOOLS_HIT="$(first_file '^scripts/[^/]*\.py$')"
if [[ -n "$PY_TOOLS_HIT" && -z "$PYTHON" ]]; then
    add py-tools skip "no Python with pytest on PATH"
else
    run_if py-tools "$PY_TOOLS_HIT" "no Python tooling changed" check_py_tools
fi

mapfile -t SH_FILES < <(printf '%s\n' "${CHANGED[@]}" | grep -E '\.sh$')
run_if sh-syntax "${SH_FILES[0]:-}" "no shell script changed" check_sh_syntax

GPU_HIT="$(first_file '^src/gpu/')"
[[ -n "$GPU_HIT" ]] && add owed-gpu owed "$GPU_HIT" \
    "scripts/test-gpu.ps1 on a host with a card; a run without one skips the GPU goldens"
MPI_HIT="$(first_file '^src/.*mpi[^/]*$|^examples/dist_mpi_check\.rs$')"
MPI_HIT="${MPI_HIT:-$(first_hit '^[+-]$' '\.(rs|toml)$' 'MpiComm|distributed-mpi')}"
[[ -n "$MPI_HIT" ]] && add owed-mpi owed "$MPI_HIT" \
    "--all-features clippy and nextest after . scripts/mpi-env.ps1, then scripts/test-mpi.ps1"

echo "=== pre-PR checks: $(git rev-parse --short HEAD) against $(git rev-parse --short "$BASE_SHA") ==="
echo "  rustc: $(rustc --version 2>/dev/null). CI takes the newest stable."
echo "  files: ${#CHANGED[@]} changed, ${#UNTRACKED[@]} untracked"
for f in "${UNTRACKED[@]}"; do echo "    untracked: $f"; done
echo ""
for (( i = 0; i < ${#NAMES[@]}; i++ )); do
    printf '  %-24s %-5s %s\n' "${NAMES[i]}" "${ACTIONS[i]}" "${REASONS[i]}"
done
if [[ -n "$CROSS_UNBUILT" ]]; then
    echo ""
    echo "  A platform gate outside src/ is changed, which --lib does not build. Read its arms:"
    echo "    $CROSS_UNBUILT"
fi
[[ -n "$PLAN" ]] && exit 0

# --- Run ---

fmt_time() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }

STATUS=()
TIMES=()
any_failed=0
stopped=""
for (( i = 0; i < ${#NAMES[@]}; i++ )); do
    if [[ -n "$stopped" || "${ACTIONS[i]}" != run ]]; then
        STATUS+=("$([[ -n "$stopped" && "${ACTIONS[i]}" == run ]] && echo "-" || echo "${ACTIONS[i]^^}")")
        TIMES+=("")
        continue
    fi
    echo ""
    echo "=== ${NAMES[i]} (${REASONS[i]})"
    read -r -a command <<< "${COMMANDS[i]}"
    start=$SECONDS
    if "${command[@]}"; then
        STATUS+=(PASS)
    else
        STATUS+=(FAIL)
        any_failed=1
        [[ -n "$FAIL_FAST" ]] && stopped=1
    fi
    TIMES+=("$(fmt_time $(( SECONDS - start )))")
done

echo ""
echo "=== Summary ==="
for (( i = 0; i < ${#NAMES[@]}; i++ )); do
    note=""
    [[ "${ACTIONS[i]}" != run ]] && note="${REASONS[i]}"
    [[ "${ACTIONS[i]}" == owed ]] && note="${COMMANDS[i]}"
    printf '  %-24s %-5s %7s  %s\n' "${NAMES[i]}" "${STATUS[i]}" "${TIMES[i]}" "$note"
done
[[ -n "$stopped" ]] && echo "  Stopped at the first failure (--fail-fast)."
exit "$any_failed"

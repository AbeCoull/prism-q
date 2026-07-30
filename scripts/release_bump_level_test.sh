#!/usr/bin/env bash
# Verify scripts/release_bump_level.sh against synthetic commit lists.
#
# The release workflow has no test harness of its own, so the bump logic lives
# in sourceable functions and this script drives them directly. Each case states
# the commit subjects, the commit bodies, the manifest version, and the version
# the resolved level produces, so the pre-1.0 clamp is asserted rather than
# eyeballed.
#
# Usage: scripts/release_bump_level_test.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release_bump_level.sh
source "$SCRIPT_DIR/release_bump_level.sh"

PASS=0
FAIL=0

# Version cargo-release produces by applying `level` to `version`. Mirrors
# cargo-release's own arithmetic so a case can state the published number.
next_version() {
    local level="$1" version="$2"
    local major="${version%%.*}"
    local rest="${version#*.}"
    local minor="${rest%%.*}"
    local patch="${rest#*.}"

    case "$level" in
        major) echo "$((major + 1)).0.0" ;;
        minor) echo "${major}.$((minor + 1)).0" ;;
        patch) echo "${major}.${minor}.$((patch + 1))" ;;
        none)  echo "$version" ;;
        *)     echo "unknown-level" ;;
    esac
}

check() {
    local label="$1" subjects="$2" bodies="$3" version="$4"
    local want_level="$5" want_version="$6"

    local raw got_level got_version
    raw="$(conventional_bump_level "$subjects" "$bodies")"
    got_level="$(clamp_bump_level "$raw" "${version%%.*}")"
    got_version="$(next_version "$got_level" "$version")"

    if [[ "$got_level" == "$want_level" && "$got_version" == "$want_version" ]]; then
        PASS=$((PASS + 1))
        printf 'ok    %-46s %s -> %s (%s)\n' "$label" "$version" "$got_version" "$got_level"
    else
        FAIL=$((FAIL + 1))
        printf 'FAIL  %-46s want %s/%s, got %s/%s (raw %s)\n' \
            "$label" "$want_level" "$want_version" "$got_level" "$got_version" "$raw"
    fi
}

echo "=== release bump level ==="

# The bug this harness exists for: a breaking commit on 0.x must not cut a 1.0.
check "breaking subject on 0.x" \
    "feat!: drop the inert serialization feature" "" "0.27.0" minor "0.28.0"
check "scoped breaking subject on 0.x" \
    "feat(backend)!: change the trait signature" "" "0.27.0" minor "0.28.0"
check "breaking body on 0.x" \
    "refactor: rework the noise API" $'BREAKING CHANGE: NoiseModel is now per qubit.' \
    "0.27.0" minor "0.28.0"
check "breaking subject wins over feat on 0.x" \
    $'chore: bump deps\nfeat!: remove the old builder' "" "0.27.0" minor "0.28.0"

# The clamp is version gated, so a deliberate 1.0 keeps the major path working.
check "breaking subject on 1.x" \
    "feat!: drop the old API" "" "1.4.2" major "2.0.0"
check "breaking body on 1.x" \
    "refactor: rework the noise API" $'BREAKING CHANGE: NoiseModel is now per qubit.' \
    "1.4.2" major "2.0.0"

# Non-breaking mappings are unchanged by the clamp.
check "feat maps to minor" "feat: add a factored trajectory row" "" "0.27.0" minor "0.28.0"
check "scoped feat maps to minor" "feat(bench): add a noise row" "" "0.27.0" minor "0.28.0"
check "fix maps to patch" "fix: correct the reset channel" "" "0.27.0" patch "0.27.1"
check "perf maps to patch" "perf(backend): drop a buffer pass" "" "0.27.0" patch "0.27.1"
check "feat outranks fix" $'fix: correct the reset channel\nfeat: add a row' "" \
    "0.27.0" minor "0.28.0"
check "docs and perf map to patch" $'docs: describe the A/B method\nperf: drop a pass' "" \
    "0.27.0" patch "0.27.1"

# Nothing release worthy.
check "chore only maps to none" "chore: tidy the tracker" "" "0.27.0" none "0.27.0"
check "empty range maps to none" "" "" "0.27.0" none "0.27.0"
check "whitespace range maps to none" $'\n  \n' "" "0.27.0" none "0.27.0"

# Guards against widening the breaking match.
check "mid line breaking text is not breaking" \
    "docs: link the changelog" "See BREAKING CHANGE: notes in the guide." \
    "0.27.0" none "0.27.0"
check "bang without colon is not breaking" \
    "feat! add a row" "" "0.27.0" none "0.27.0"

echo ""
echo "=== manifest version ==="

MANIFEST_FIXTURE="$(mktemp)"
trap 'rm -f "$MANIFEST_FIXTURE"' EXIT
cat > "$MANIFEST_FIXTURE" <<'EOF'
[package]
name = "prism-q"
version = "0.27.0"
edition = "2024"

[workspace]
members = [".", "bindings/python"]

[dependencies]
num-complex = { version = "9.9.9" }
EOF

fixture_version="$(crate_version "$MANIFEST_FIXTURE")"
if [[ "$fixture_version" == "0.27.0" ]]; then
    PASS=$((PASS + 1))
    echo "ok    package version read past workspace and dependency tables"
else
    FAIL=$((FAIL + 1))
    echo "FAIL  package version: want 0.27.0, got '$fixture_version'"
fi

repo_version="$(crate_version "$SCRIPT_DIR/../Cargo.toml")"
if [[ "$repo_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    PASS=$((PASS + 1))
    echo "ok    repository manifest version parses ($repo_version)"
else
    FAIL=$((FAIL + 1))
    echo "FAIL  repository manifest version did not parse: '$repo_version'"
fi

echo ""
echo "$PASS passed, $FAIL failed"
if (( FAIL > 0 )); then
    exit 1
fi

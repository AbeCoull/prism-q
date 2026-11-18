#!/usr/bin/env bash
# Verify scripts/compat_breaking_title.sh against synthetic edits of a
# compatibility page.
#
# The check runs only on pull requests, so its section tracking and title
# parsing live in sourceable functions and this script drives them directly.
# Each case states the edited page, the pull request title, and whether the
# check passes, with the diff produced by git from a fixture page.
#
# Usage: scripts/compat_breaking_title_test.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/compat_breaking_title.sh
source "$SCRIPT_DIR/compat_breaking_title.sh"
set +e

PASS=0
FAIL=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

BASE_PAGE="$(cat <<'EOF'
# Compatibility and Versioning

## Versions

A minor release may break the API below 1.0.

## Changed results

A breaking release can keep a public field and change what it reports.

- 0.30.0: an older entry.

## Seeded output

Seeded output has changed in these releases:

- 0.33.0 seeds shot `i` with a hash.

### Routes

Per route notes.

## Support

Ask on the tracker.
EOF
)"

# Pass or fail verdict for `title` against the page edited to `new_page`.
verdict() {
    local new_page="$1" title="$2" crlf="${3:-}"
    printf '%s\n' "$BASE_PAGE" > "$WORK/a.md"
    printf '%s\n' "$new_page" > "$WORK/b.md"
    if [[ -n "$crlf" ]]; then
        sed -i 's/$/\r/' "$WORK/a.md" "$WORK/b.md"
    fi

    local diff sections
    diff="$(git -c core.autocrlf=false diff --no-index -U1000000 "$WORK/a.md" "$WORK/b.md")"
    sections="$(compat_sections_added "$diff")"
    if [[ -z "$sections" ]] || title_is_breaking "$title"; then
        echo pass
    else
        echo fail
    fi
}

check() {
    local label="$1" new_page="$2" title="$3" want="$4" crlf="${5:-}"
    local got
    got="$(verdict "$new_page" "$title" "$crlf")"
    if [[ "$got" == "$want" ]]; then
        PASS=$((PASS + 1))
        printf 'ok    %-52s %s\n' "$label" "$got"
    else
        FAIL=$((FAIL + 1))
        printf 'FAIL  %-52s want %s, got %s\n' "$label" "$want" "$got"
    fi
}

# Page with `line` inserted after the first line matching `anchor`.
insert_after() {
    local anchor="$1" line="$2"
    printf '%s\n' "$BASE_PAGE" | awk -v anchor="$anchor" -v line="$line" '
        { print } !done && index($0, anchor) == 1 { print line; done = 1 }'
}

# Page with the first line matching `anchor` removed.
remove_line() {
    local anchor="$1"
    printf '%s\n' "$BASE_PAGE" | awk -v anchor="$anchor" '
        !done && index($0, anchor) == 1 { done = 1; next } { print }'
}

CHANGED_ENTRY="$(insert_after "- 0.30.0" "- Unreleased: a field reports the run.")"
SEEDED_ENTRY="$(insert_after "- 0.33.0" "- Unreleased: product shots draw in blocks.")"
SUBSECTION_ENTRY="$(insert_after "Per route notes." "A new route note.")"

echo "=== compatibility entries need a breaking title ==="

check "changed results entry without bang fails" "$CHANGED_ENTRY" \
    "perf(sim): report exactness per run" fail
check "seeded output entry without bang fails" "$SEEDED_ENTRY" \
    "perf(sim): draw product shots in blocks" fail
check "entry in a subsection without bang fails" "$SUBSECTION_ENTRY" \
    "docs: note a route" fail
check "unscoped title without bang fails" "$SEEDED_ENTRY" \
    "perf: draw product shots in blocks" fail
check "entry in a CRLF page without bang fails" "$SEEDED_ENTRY" \
    "perf(sim): draw product shots in blocks" fail crlf

check "changed results entry with bang passes" "$CHANGED_ENTRY" \
    "perf(sim)!: report exactness per run" pass
check "seeded output entry with bang passes" "$SEEDED_ENTRY" \
    "perf(sim)!: draw product shots in blocks" pass
check "unscoped title with bang passes" "$SEEDED_ENTRY" \
    "perf!: draw product shots in blocks" pass

echo ""
echo "=== edits outside the watched sections ==="

check "entry under another heading passes" \
    "$(insert_after "A minor release" "A patch release never breaks.")" \
    "docs: describe patch releases" pass
check "entry after the watched sections passes" \
    "$(insert_after "Ask on the tracker." "Or open a discussion.")" \
    "docs: add a support route" pass
check "removed changed results entry passes" "$(remove_line "- 0.30.0")" \
    "docs: drop an old entry" pass
check "removed seeded output entry passes" "$(remove_line "- 0.33.0")" \
    "docs: drop an old entry" pass
check "blank line in a watched section passes" "$(insert_after "- 0.30.0" "")" \
    "docs: space the list" pass
check "page unchanged passes" "$BASE_PAGE" "docs: no change" pass

echo ""
echo "=== bang placement ==="

check "bang in the summary does not count" "$SEEDED_ENTRY" \
    "perf(sim): draw product shots in blocks!" fail
check "bang after the colon does not count" "$SEEDED_ENTRY" \
    "perf(sim):! draw product shots in blocks" fail
check "bang without a colon does not count" "$SEEDED_ENTRY" \
    "perf(sim)! draw product shots in blocks" fail
check "bang before the scope does not count" "$SEEDED_ENTRY" \
    "perf!(sim): draw product shots in blocks" fail
check "empty title fails" "$SEEDED_ENTRY" "" fail

echo ""
echo "=== repository page ==="

repo_page="$SCRIPT_DIR/../$COMPAT_DOC"
for section in "${COMPAT_SECTIONS[@]}"; do
    if grep -qx "## $section" "$repo_page"; then
        PASS=$((PASS + 1))
        echo "ok    $COMPAT_DOC has the heading \"## $section\""
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  $COMPAT_DOC lost the heading \"## $section\"; update COMPAT_SECTIONS"
    fi
done

echo ""
echo "$PASS passed, $FAIL failed"
if (( FAIL > 0 )); then
    exit 1
fi

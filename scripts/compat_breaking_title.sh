#!/usr/bin/env bash
# Require a breaking pull request title when the diff adds an entry to the
# "Changed results" or "Seeded output" sections of docs/guides/compatibility.md.
#
# Those sections list changes a caller can observe across versions. The squash
# merge takes its commit subject from the pull request title, and the release
# automation reads the `!` in that subject to choose the bump and file the
# changelog entry, so an entry in either section without a `!` in the title
# ships a moved result as a change that breaks nothing.
#
# A section runs from its heading to the next heading of the same or a higher
# level, so `###` subsections stay inside it. Lines that are only whitespace,
# and removed lines, do not count.
#
# Usage:
#   PR_TITLE="perf(sim)!: ..." scripts/compat_breaking_title.sh [BASE]
#
# BASE defaults to origin/main; the diff is BASE...HEAD. Exits 1 when the diff
# adds a line to a watched section and the title is not breaking.
#
# scripts/compat_breaking_title_test.sh sources this file and drives the pure
# functions below against synthetic diffs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/release_bump_level.sh
source "$SCRIPT_DIR/release_bump_level.sh"

COMPAT_DOC="docs/guides/compatibility.md"
COMPAT_SECTIONS=("Changed results" "Seeded output")

# Succeed when a conventional-commit title carries the breaking `!`. Reuses the
# release bump mapping so the two cannot disagree about what a breaking title is.
title_is_breaking() {
    local title="$1"
    [[ "$(conventional_bump_level "$title" "")" == "major" ]]
}

# Print each watched section that a unified diff adds a line to, once per
# section, in order of first appearance. The diff must carry enough context to
# show the governing heading, so the caller passes a context wider than the file.
compat_sections_added() {
    local diff="$1"
    local watched
    watched="$(IFS='|'; echo "${COMPAT_SECTIONS[*]}")"

    printf '%s\n' "$diff" | awk -v watched="$watched" '
        BEGIN {
            n = split(watched, names, "|")
            for (i = 1; i <= n; i++) want[names[i]] = 1
            section = ""; level = 0
        }
        /^@@ / { in_hunk = 1; next }
        !in_hunk || /^-/ { next }
        {
            mark = substr($0, 1, 1)
            if (mark != "+" && mark != " ") next
            line = substr($0, 2)
            if (match(line, /^#+[ \t]/)) {
                depth = RLENGTH - 1
                if (section == "" || depth <= level) {
                    title = substr(line, RLENGTH + 1)
                    sub(/[ \t\r#]+$/, "", title)
                    if (title in want) { section = title; level = depth }
                    else { section = ""; level = 0 }
                }
            }
            if (mark == "+" && section != "" && line ~ /[^ \t\r]/ && !(section in seen)) {
                seen[section] = 1
                print section
            }
        }
    '
}

main() {
    local base="${1:-origin/main}"
    local title="${PR_TITLE:-}"

    local diff sections
    diff="$(git diff -U1000000 "$base...HEAD" -- "$COMPAT_DOC")"
    sections="$(compat_sections_added "$diff")"

    if [[ -z "$sections" ]]; then
        echo "No entry added under the watched sections of $COMPAT_DOC."
        return 0
    fi

    if title_is_breaking "$title"; then
        echo "Entries added under: $(paste -sd, <<< "$sections"); the title is breaking."
        return 0
    fi

    local section
    while IFS= read -r section; do
        echo "::error file=$COMPAT_DOC::An entry was added under \"$section\", which records a change a caller can observe across versions."
    done <<< "$sections"
    echo "Mark the title breaking with \`!\` before the colon, for example \`perf(sim)!: ...\`."
    echo "The squash merge takes its subject from the title, and the release automation"
    echo "reads that \`!\` to label the change. Title: $title"
    return 1
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

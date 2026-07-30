#!/usr/bin/env bash
# Resolve the cargo-release bump level for a commit range.
#
# Two stages. The first maps conventional commits to a level exactly as the
# specification describes. The second clamps that level against the crate's
# current major version, because below 1.0 a "major" bump is not available:
# Cargo reads `0.27.0` as `^0.27.0`, so under 0.x the minor is the compatibility
# boundary and a breaking change belongs there. `cargo release major` on a 0.x
# version would publish 1.0.0, which crates.io lets you yank but never
# unpublish, and which declares an API stability the crate has not reached.
#
# Once the crate ships a deliberate 1.0.0 the clamp stops firing and breaking
# commits resolve to major again, with no change to this script.
#
# Usage:
#   scripts/release_bump_level.sh [RANGE]
#
# Emits `key=value` lines on stdout, suitable for appending straight to
# `$GITHUB_OUTPUT`:
#
#   level=minor        the level to hand cargo-release
#   raw_level=major    the level before the pre-1.0 clamp
#   version=0.27.0     the version read from the manifest
#
# RANGE defaults to `<last tag>..HEAD`, or `HEAD` when the repository has no
# tags. Override the manifest with the MANIFEST environment variable.
#
# scripts/release_bump_level_test.sh sources this file and drives the pure
# functions below against synthetic commit lists.

set -euo pipefail

# Bump level implied by conventional-commit subjects and bodies, ignoring the
# current version. Echoes one of: major, minor, patch, none.
conventional_bump_level() {
    local subjects="$1"
    local bodies="$2"

    if [[ -z "${subjects//[[:space:]]/}" ]]; then
        echo "none"
        return
    fi

    if printf '%s\n' "$subjects" | grep -qiE '^[a-z]+(\(.+\))?!:'; then
        echo "major"
    elif printf '%s\n' "$bodies" | grep -q '^BREAKING CHANGE:'; then
        echo "major"
    elif printf '%s\n' "$subjects" | grep -qE '^feat(\(.+\))?:'; then
        echo "minor"
    elif printf '%s\n' "$subjects" | grep -qE '^(fix|perf)(\(.+\))?:'; then
        echo "patch"
    else
        echo "none"
    fi
}

# Clamp a conventional-commit level against the crate's current major version.
# Below 1.0 a major bump becomes a minor bump; every other level passes through.
clamp_bump_level() {
    local level="$1"
    local major="$2"

    if [[ "$level" == "major" && "$major" == "0" ]]; then
        echo "minor"
    else
        echo "$level"
    fi
}

# Version string from the `[package]` table of a Cargo manifest. Stops at the
# first version key inside that table, so `[workspace]` and dependency tables
# below it cannot shadow it.
crate_version() {
    local manifest="${1:-Cargo.toml}"

    awk -F'"' '
        /^\[/ { in_package = ($0 == "[package]"); next }
        in_package && /^version[[:space:]]*=/ { print $2; exit }
    ' "$manifest"
}

main() {
    local range="${1:-}"
    if [[ -z "$range" ]]; then
        local last_tag
        last_tag="$(git describe --tags --abbrev=0 2>/dev/null || echo "")"
        if [[ -z "$last_tag" ]]; then
            range="HEAD"
        else
            range="${last_tag}..HEAD"
        fi
    fi

    local subjects bodies version raw_level level
    subjects="$(git log --format="%s" "$range" 2>/dev/null || true)"
    bodies="$(git log --format="%b" "$range" 2>/dev/null || true)"
    version="$(crate_version "${MANIFEST:-Cargo.toml}")"

    if [[ -z "$version" ]]; then
        echo "Error: no [package] version in ${MANIFEST:-Cargo.toml}" >&2
        exit 1
    fi

    raw_level="$(conventional_bump_level "$subjects" "$bodies")"
    level="$(clamp_bump_level "$raw_level" "${version%%.*}")"

    echo "level=$level"
    echo "raw_level=$raw_level"
    echo "version=$version"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

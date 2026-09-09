#!/usr/bin/env bash
# Measure whether the build directory moves code layout on an ELF target.
#
# scripts/bench_ab.sh builds its reference binary in a separate worktree, so an
# A/B compares two binaries produced at two different paths. On the project's
# Windows host that was measured not to matter: the same commit built in two
# directories produced byte-identical `.text`, and `--remap-path-prefix` was
# reverted there as a no-op that only invalidates cargo fingerprints. ELF is the
# case that could still differ, because line tables live in the binary rather
# than in a side file, and the regression gate runs on Linux.
#
# Builds HEAD twice through the same `bench_ab.sh --build-only` route the gate
# uses, once at a short path and once at a much longer one, then compares the
# `.text` sections. Both trees come from `git worktree add` off HEAD, so
# uncommitted edits are excluded deliberately: the path has to be the only
# variable between the two builds.
#
# Usage: scripts/elf_layout_probe.sh [--base DIR] [--bench NAME]
#                                    [--features LIST] [--out FILE]
#
# This reports, it does not gate. A hash mismatch is a finding, not a defect, so
# the exit status is 0 either way and nothing here belongs in a required check.
#
# Requires git, cargo, sha256sum, and binutils (readelf, objcopy). Two full
# `lto = "fat"` bench builds, so budget upwards of twenty minutes cold.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

BASE=""
BENCH="circuits"
FEATURES="${CI_BENCH_FEATURES:-parallel,bench-fast}"
OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)        BASE="$2"; shift 2 ;;
        --bench|-b)    BENCH="$2"; shift 2 ;;
        --features)    FEATURES="$2"; shift 2 ;;
        --out)         OUT="$2"; shift 2 ;;
        -h|--help)     awk 'NR > 1 && !/^#/ { exit } NR > 1' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$PROJECT_DIR"

for cmd in git cargo sha256sum readelf objcopy; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: '$cmd' not found." >&2; exit 1; }
done

KEEP_BASE=false
if [[ -z "$BASE" ]]; then
    BASE="$(mktemp -d)"
else
    KEEP_BASE=true
    mkdir -p "$BASE"
fi
[[ -n "$OUT" ]] || OUT="$PROJECT_DIR/bench_results/elf-layout-probe.md"
mkdir -p "$(dirname "$OUT")"

# Path length is what shifts layout when anything does, so the two trees differ
# by nesting depth rather than by name.
SHORT_TREE="$BASE/a"
LONG_TREE="$BASE/elf-layout-probe-path-length-segment-one"
LONG_TREE="$LONG_TREE/elf-layout-probe-path-length-segment-two"
LONG_TREE="$LONG_TREE/elf-layout-probe-path-length-segment-three/b"

COMMIT="$(git rev-parse HEAD)"

cleanup() {
    git worktree remove --force "$SHORT_TREE" >/dev/null 2>&1 || true
    git worktree remove --force "$LONG_TREE" >/dev/null 2>&1 || true
    if [[ "$KEEP_BASE" == "false" ]]; then
        rm -rf "$BASE"
    fi
}
trap cleanup EXIT

echo "=== ELF layout probe ==="
echo "  commit:    $COMMIT"
echo "  bench:     $BENCH"
echo "  features:  $FEATURES"
echo "  short:     $SHORT_TREE (${#SHORT_TREE} chars)"
echo "  long:      $LONG_TREE (${#LONG_TREE} chars)"
echo ""

build_at() {
    local tree="$1" label="$2"
    mkdir -p "$(dirname "$tree")"
    git worktree add --detach --force "$tree" "$COMMIT" >/dev/null
    echo ">>> building $label at $tree"
    bash "$tree/scripts/bench_ab.sh" \
        --bench "$BENCH" \
        --features "$FEATURES" \
        --build-only "$BASE/$label-exe"
    echo ""
}

build_at "$SHORT_TREE" short
build_at "$LONG_TREE" long

# objcopy always writes a copy of the whole binary as well; it goes to one
# scratch path and is discarded, leaving the two inputs untouched for `stat`.
dump_text() {
    objcopy --dump-section ".text=$2" "$1" "$BASE/objcopy-discard"
}

dump_text "$BASE/short-exe" "$BASE/short.text"
dump_text "$BASE/long-exe" "$BASE/long.text"
rm -f "$BASE/objcopy-discard"

sha_of()   { sha256sum "$1" | awk '{ print $1 }'; }
bytes_of() { stat -c %s "$1"; }
text_hdr() { readelf --wide --section-headers "$1" | grep -E '[[:space:]]\.text[[:space:]]'; }

SHORT_SHA="$(sha_of "$BASE/short.text")"
LONG_SHA="$(sha_of "$BASE/long.text")"

VERDICT="MATCH"
if [[ "$SHORT_SHA" != "$LONG_SHA" ]]; then
    VERDICT="MISMATCH"
    KEEP_BASE=true
fi

{
    echo "## ELF layout probe: \`$BENCH\`"
    echo ""
    echo "One commit built twice, differing only in the directory it was built in. A"
    echo "matching \`.text\` means the build path does not move code on this target, which"
    echo "is what the adjacent-binary A/B in \`scripts/bench_ab.sh\` assumes when it builds"
    echo "its reference in a worktree."
    echo ""
    echo "| Setting | Value |"
    echo "| --- | --- |"
    echo "| Commit | \`$COMMIT\` |"
    echo "| Bench target | \`$BENCH\` |"
    echo "| Features | \`$FEATURES\` |"
    echo "| OS | $(uname -srm) |"
    echo "| Toolchain | $(rustc --version) |"
    echo "| Bench profile | \`lto = \"fat\"\`, \`codegen-units = 1\`, \`debug = \"line-tables-only\"\` |"
    echo "| Path length delta | $(( ${#LONG_TREE} - ${#SHORT_TREE} )) characters |"
    echo ""
    echo "| Build | Path chars | Binary bytes | \`.text\` bytes | \`.text\` sha256 |"
    echo "| --- | ---: | ---: | ---: | --- |"
    printf '| short | %s | %s | %s | `%s` |\n' \
        "${#SHORT_TREE}" "$(bytes_of "$BASE/short-exe")" "$(bytes_of "$BASE/short.text")" "$SHORT_SHA"
    printf '| long | %s | %s | %s | `%s` |\n' \
        "${#LONG_TREE}" "$(bytes_of "$BASE/long-exe")" "$(bytes_of "$BASE/long.text")" "$LONG_SHA"
    echo ""
    echo "Built at:"
    echo ""
    echo "- short: \`$SHORT_TREE\`"
    echo "- long: \`$LONG_TREE\`"
    echo ""
    echo "\`readelf --wide --section-headers\`, the \`.text\` row of each:"
    echo ""
    echo '```'
    echo "short: $(text_hdr "$BASE/short-exe")"
    echo "long:  $(text_hdr "$BASE/long-exe")"
    echo '```'
    echo ""
    if [[ "$VERDICT" == "MATCH" ]]; then
        echo "**Layout verdict**: MATCH. The build directory does not move \`.text\` on this"
        echo "ELF target. The reference worktree in \`scripts/bench_ab.sh\` hands the gate a"
        echo "binary that differs from the head build in code alone, and \`--remap-path-prefix\`"
        echo "would change nothing there while invalidating every cargo fingerprint. No action"
        echo "follows from this result."
    else
        echo "**Layout verdict**: MISMATCH. The build directory moves \`.text\` on this ELF"
        echo "target, so an A/B across two build paths measures layout alongside code. Equal"
        echo "\`.text\` sizes above mean the same code placed differently; different sizes mean"
        echo "the two builds do not hold the same code, which is a larger problem than layout."
        echo "Next step: pass \`--remap-path-prefix\` for the package root to both builds in"
        echo "\`scripts/bench_ab.sh\`, then re-run this probe and confirm it reports MATCH. The"
        echo "two binaries and their extracted sections are kept under \`$BASE\` for"
        echo "\`objdump -d\` on the first differing block."
    fi
    echo ""
    echo "This settles code layout, not timing. Two binaries with identical \`.text\` can"
    echo "still time differently on a shared runner, and separating that from a real"
    echo "regression needs a quiet dedicated host rather than a second hash."
    echo ""
    echo "Reporting only. A mismatch is a finding, not a defect, so this never fails a run."
} > "$OUT"

cat "$OUT"
echo ""
echo "Markdown written to $OUT"

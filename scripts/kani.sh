#!/usr/bin/env bash
# Usage: bash scripts/kani.sh [--list | LEG...]
#
# Run the Kani legs of .github/workflows/ci.yml with the command line CI uses, on Linux
# (WSL2 or a container) with kani-verifier installed and `cargo kani setup` done. The
# legs come from the `kani` job's matrix on x86_64 and the `kani-neon` job's on aarch64.
# With no LEG every leg for this machine runs; `--list` prints their names.
set -euo pipefail
cd "$(dirname "$0")/.."

case "$(uname -m)" in
    x86_64) job=kani ;;
    aarch64 | arm64) job=kani-neon ;;
    *) echo "kani.sh: no Kani job for $(uname -m)" >&2; exit 2 ;;
esac

# One line per leg, "name|harness filters|jobs", read from the job's matrix. Carriage
# returns are stripped so a checkout with CRLF line endings parses the same.
all_legs="$(
    tr -d '\r' < .github/workflows/ci.yml |
        awk -v job="  $job:" '
            $0 == job { inside = 1; next }
            inside && /^  [a-z]/ { exit }
            inside && /- \{ name:/ { print }
        ' |
        sed -E "s/.*name: ([^,]+), harness(es)?: '([^']*)', jobs: ([0-9]+).*/\1|\3|\4/"
)"
if [[ -z "$all_legs" ]]; then
    echo "kani.sh: parsed no legs for the $job job from .github/workflows/ci.yml" >&2
    exit 2
fi

if [[ "${1:-}" == --list ]]; then
    cut -d'|' -f1 <<< "$all_legs"
    exit 0
fi

for leg in "$@"; do
    cut -d'|' -f1 <<< "$all_legs" | grep -qx -- "$leg" ||
        { echo "kani.sh: no leg $leg in $job" >&2; exit 2; }
done

failed=()
while IFS='|' read -r name filters jobs; do
    [[ $# -eq 0 || " $* " == *" $name "* ]] || continue
    args=()
    for h in $filters; do args+=(--harness "$h"); done
    echo "== $job ($name)"
    cargo kani --no-default-features -Z stubbing -Z unstable-options \
        "${args[@]}" --harness-timeout 8m \
        --jobs "$jobs" --output-format terse < /dev/null || failed+=("$name")
done <<< "$all_legs"

if (( ${#failed[@]} )); then
    echo "kani.sh: failed legs: ${failed[*]}" >&2
    exit 1
fi

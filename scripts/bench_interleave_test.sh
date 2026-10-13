#!/usr/bin/env bash
# Verify the statistics, verdicts, and exit codes of scripts/bench_interleave.sh.
#
# Most cases feed a synthetic raw table through --from-tsv. Each row carries a
# base time, a linear drift per invocation, deterministic noise, and an effect on
# variant b, so a case states what the measurement would have read and the
# expected verdict. The last cases run the whole script against a stub bench
# binary to cover the A B B A order, the smoke test, and the host-load guard.
#
# Usage: scripts/bench_interleave_test.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/bench_interleave.sh"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

PASS=0
FAIL=0

# Write a raw table for ROUNDS rounds. Each row spec is
# `segment:id:base_ns:effect_b:noise:drift`, with effect_b a factor on variant b,
# noise a relative amplitude, and drift a relative step per invocation. A spec
# `outlier:R:F` multiplies every variant b reading in round R by F, and `burst:R:F`
# multiplies the first invocation of round R by F, as load from other work would.
table() {
    local rounds="$1"
    shift
    awk -v rounds="$rounds" -v specs="$*" 'BEGIN {
        OFS = "\t"
        nspec = split(specs, spec, " ")
        t = 0
        for (r = 1; r <= rounds; r++) {
            split((r % 2) ? "a b b a" : "b a a b", order, " ")
            for (s = 1; s <= 4; s++) {
                v = order[s]; t++
                for (i = 1; i <= nspec; i++) {
                    split(spec[i], f, ":")
                    if (f[1] == "outlier" || f[1] == "burst") { continue }
                    ns = f[3] * (1 + f[6] * t)
                    if (v == "b") { ns *= f[4] }
                    # A fixed pseudo-random sequence in [-1, 1].
                    h = sin(t * 12.9898 + i * 78.233) * 43758.5453
                    if (h < 0) { h = -h }
                    u = 2 * (h - int(h)) - 1
                    ns *= 1 + f[5] * u
                    for (k = 1; k <= nspec; k++) {
                        split(spec[k], g, ":")
                        if (g[1] == "outlier" && g[2] == r && v == "b") { ns *= g[3] }
                        if (g[1] == "burst" && g[2] == r && s == 1) { ns *= g[3] }
                    }
                    print r, s, v, f[1], f[2], ns, ns, ns
                }
            }
        }
    }'
}

# check LABEL WANT_EXIT WANT_VERDICT ROUNDS SPEC... [-- RUNNER OPTIONS]
check() {
    local label="$1" want="$2" want_verdict="$3" rounds="$4"
    shift 4
    local specs=() extra=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do specs+=("$1"); shift; done
    [[ "${1:-}" == "--" ]] && { shift; extra=("$@"); }
    local dir="$SCRATCH/$label"
    mkdir -p "$dir"
    table "$rounds" "${specs[@]}" > "$dir/raw.tsv"
    if [[ "$label" == *loaded* || "$label" == *busy-reading* ]]; then
        printf '#loaded\tround 3 at 40%%\n' >> "$dir/raw.tsv"
    fi
    local got verdict
    bash "$RUNNER" --from-tsv "$dir/raw.tsv" --out "$dir/report.md" --max-host-load 0 \
        ${extra[@]+"${extra[@]}"} > "$dir/log" 2>&1
    got=$?
    verdict="$(grep -m1 -oE '^\*\*Regression verdict\*\*: [A-Z ]+' "$dir/report.md" 2>/dev/null | sed 's/.*: //; s/ *$//')"
    if [[ "$got" == "$want" && "$verdict" == "$want_verdict" ]]; then
        PASS=$((PASS + 1))
        printf 'ok    %-34s exit %s, %s\n' "$label" "$got" "$verdict"
    else
        FAIL=$((FAIL + 1))
        printf 'FAIL  %-34s want exit %s %s, got %s %s\n' "$label" "$want" "$want_verdict" "$got" "${verdict:-no verdict}"
        sed 's/^/      /' "$dir/log"
    fi
}

# The interval a report gives a row, as `lo hi` in percent.
interval_of() {
    awk -F'|' -v id="\`$2\`" '$2 ~ id {
        gsub(/[ %+]/, "", $6); split($6, p, "to"); print p[1], p[2]; exit
    }' "$1"
}

echo "=== bench_interleave statistics and verdicts ==="

check flat-under-drift          0 PASS    10 target:x/a:1000:1:0.01:0.005 target:x/b:5000000:1:0.01:-0.01 control:c/a:800:1:0.01:0.005
check regression-fails          1 FAIL    10 target:x/a:1000:1.10:0.01:0.005 target:x/b:5000:1:0.01:0 control:c/a:800:1:0.01:0
check faster-passes             0 PASS    10 target:x/a:1000:0.90:0.01:0.005 control:c/a:800:1:0.01:0
check under-gate-slower-passes  0 PASS    10 target:x/a:1000:1.02:0.005:0.01 control:c/a:800:1:0.005:0
check straddle-reruns           3 RERUN   10 target:x/a:1000:1.05:0.04:0 control:c/a:800:1:0.005:0
check control-off-zero-reruns   3 RERUN   10 target:x/a:1000:1:0.005:0 control:c/a:800:1.04:0.005:0
check control-moved-fails       1 FAIL    10 target:x/a:1000:1:0.005:0 control:c/a:800:1.08:0.005:0
check control-wide-reruns       3 RERUN   10 target:x/a:1000:1:0.005:0 control:c/a:800:1:0.10:0
check off-zero-outranks-fail    3 RERUN   10 target:x/a:1000:1.10:0.005:0 control:c/a:800:1.04:0.005:0
check busy-reading-fail-stands  1 FAIL    10 target:x/a:1000:1.10:0.005:0 control:c/a:800:1:0.005:0
check pass-on-loaded-passes     0 PASS    10 target:x/a:1000:1:0.005:0 control:c/a:800:1:0.005:0
check one-outlier-round-passes  0 PASS    10 target:x/a:1000:1:0.005:0 control:c/a:800:1:0.005:0 outlier:4:1.30
check burst-rounds-dropped      1 FAIL    16 target:x/a:1000:1.10:0.005:0 control:c/a:800:1:0.005:0 burst:1:1.5 burst:2:1.5 burst:3:1.5 burst:5:1.5 burst:7:1.5
check too-busy-reruns           3 RERUN   10 target:x/a:1000:1:0.005:0 burst:1:1.5 burst:2:1.5 burst:3:1.5 burst:4:1.5 burst:5:1.5
check threshold-flag-honored    0 PASS    10 target:x/a:1000:1.10:0.005:0 control:c/a:800:1:0.005:0 -- --threshold 15
check five-rounds-no-data       1 "NO DATA" 5 target:x/a:1000:1:0.005:0
check min-rows-no-data          1 "NO DATA" 10 target:x/a:1000:1:0.005:0 -- --min-rows 2

REGRESSION_THRESHOLD=15 check regression-threshold-env 0 PASS 10 target:x/a:1000:1.10:0.005:0 control:c/a:800:1:0.005:0

# The interval is the order statistics the binomial gives: rounds 1 to 10 with
# paired changes 1..10 percent put a 95% interval at the 2nd and 9th values.
dir="$SCRATCH/order-statistics"
mkdir -p "$dir"
awk 'BEGIN { OFS = "\t"; for (r = 1; r <= 10; r++) {
    b = 1000 * (1 + r / 100)
    print r, 1, "a", "target", "x/a", 1000, 1000, 1000
    print r, 2, "b", "target", "x/a", b, b, b
    print r, 3, "b", "target", "x/a", b, b, b
    print r, 4, "a", "target", "x/a", 1000, 1000, 1000 } }' > "$dir/raw.tsv"
bash "$RUNNER" --from-tsv "$dir/raw.tsv" --out "$dir/report.md" --max-host-load 0 -t 50 > /dev/null 2>&1
read -r lo hi <<< "$(interval_of "$dir/report.md" x/a)"
if [[ "$lo" == "2.0" && "$hi" == "9.0" ]] && grep -q '| +5.5% |' "$dir/report.md"; then
    PASS=$((PASS + 1)); printf 'ok    %-34s median +5.5%%, interval %s to %s\n' order-statistics "$lo" "$hi"
else
    FAIL=$((FAIL + 1)); printf 'FAIL  %-34s want +5.5%% in 2.0 to 9.0, got %s to %s\n' order-statistics "$lo" "$hi"
    sed 's/^/      /' "$dir/report.md"
fi

echo
echo "=== bench_interleave end to end against a stub binary ==="

# A stand-in for a Criterion bench binary: rows x/a, x/b, c/a; variant b is
# slower on x/b by STUB_EFFECT; every invocation drifts the times up by 0.3%;
# STUB_BURST lists invocations that run three times slow, as under load; STUB_PANIC
# names a variant that panics in test mode.
STUB="$SCRATCH/stub-exe"
cat > "$STUB" <<'STUB'
#!/usr/bin/env bash
mode=""; filter=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) mode=test; shift ;;
        --bench) mode=bench; shift ;;
        --warm-up-time|--measurement-time) shift 2 ;;
        *) filter="$1"; shift ;;
    esac
done
home="$CRITERION_HOME"
command -v cygpath >/dev/null 2>&1 && home="$(cygpath -u "$home")"
calls=$(( $(cat "$STUB_STATE/calls" 2>/dev/null || echo 0) + 1 ))
echo "$calls" > "$STUB_STATE/calls"
[[ "$mode" == bench ]] && echo "$PRISM_BENCH_VARIANT" >> "$STUB_STATE/order"
for row in x/a x/b c/a; do
    [[ "$row" =~ $filter ]] || continue
    if [[ "$mode" == test ]]; then
        echo "Testing $row"
        if [[ "${STUB_PANIC:-}" == "$PRISM_BENCH_VARIANT" && "$row" == x/b ]]; then
            echo "thread 'main' panicked"; exit 101
        fi
        echo "Success"
        continue
    fi
    ns="$(awk -v c="$calls" -v v="$PRISM_BENCH_VARIANT" -v row="$row" -v e="${STUB_EFFECT:-1}" \
        -v burst=" ${STUB_BURST:-} " 'BEGIN {
            ns = 1000 * (1 + 0.003 * c); if (v == "b" && row == "x/b") ns *= e
            if (index(burst, " " c " ") > 0) ns *= 3
            printf "%.3f", ns }')"
    mkdir -p "$home/$row/new"
    printf '{"group_id":"%s","full_id":"%s"}\n' "${row%%/*}" "$row" > "$home/$row/new/benchmark.json"
    printf '{"mean":{"point_estimate":%s,"standard_error":1.0},"median":{"point_estimate":1.0}}\n' "$ns" \
        > "$home/$row/new/estimates.json"
    printf '{"sampling_mode":"Linear","iters":[1.0],"times":[%s]}\n' "$ns" > "$home/$row/new/sample.json"
done
STUB
chmod +x "$STUB"

# e2e LABEL WANT_EXIT [ENV=VALUE...] -- [RUNNER OPTIONS]
e2e() {
    local label="$1" want="$2"
    shift 2
    local envs=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
    shift
    local dir="$SCRATCH/e2e-$label"
    mkdir -p "$dir/state"
    env STUB_STATE="$dir/state" ${envs[@]+"${envs[@]}"} bash "$RUNNER" --exe "$STUB" \
        -f '^x/' -c '^c/' --rounds 6 --out "$dir/report.md" --project-dir "$SCRIPT_DIR/.." "$@" \
        > "$dir/log" 2>&1
    E2E_STATUS=$?
    E2E_DIR="$dir"
    if [[ "$E2E_STATUS" == "$want" ]]; then
        return 0
    fi
    FAIL=$((FAIL + 1))
    printf 'FAIL  %-34s want exit %s, got %s\n' "$label" "$want" "$E2E_STATUS"
    sed 's/^/      /' "$dir/log"
    return 1
}

if e2e regression 1 STUB_EFFECT=1.10 -- --max-host-load 0; then
    # Warmup a then b, then rounds alternating a b b a and b a a b, each slot
    # one target and one control invocation.
    want_order="a a b b"
    for (( r = 1; r <= 6; r++ )); do
        if (( r % 2 )); then want_order+=" a a b b b b a a"; else want_order+=" b b a a a a b b"; fi
    done
    got_order="$(tr '\n' ' ' < "$E2E_DIR/state/order" | sed 's/ $//')"
    if [[ "$got_order" == "$want_order" ]] && grep -q '`x/b`.*regressed' "$E2E_DIR/report.md" \
        && grep -q '`x/a`.*flat' "$E2E_DIR/report.md" && [[ -f "$E2E_DIR/report.tsv" ]]; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 1, A B B A order, x/b regressed\n' e2e-regression
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s order %s\n' e2e-regression "$got_order"
        sed 's/^/      /' "$E2E_DIR/report.md"
    fi
fi

if e2e same-code 0 -- --max-host-load 0; then
    PASS=$((PASS + 1)); printf 'ok    %-34s exit 0, drift cancels\n' e2e-same-code
fi

# Invocations 1 to 4 are the smoke test and 5 to 8 the warmup, so 9 and 10 are the
# first slot of round 1: that round is disturbed and a seventh runs in its place.
if e2e replacement-round 1 STUB_EFFECT=1.10 STUB_BURST="9 10" -- --max-host-load 0; then
    if [[ "$(wc -l < "$E2E_DIR/state/order" | tr -d '[:space:]')" == 60 ]] \
        && grep -q 'round 1 at' "$E2E_DIR/report.md" && grep -q '`x/b`.*regressed' "$E2E_DIR/report.md"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 1, round 1 dropped and replaced\n' e2e-replacement-round
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s no replacement round\n' e2e-replacement-round
        sed 's/^/      /' "$E2E_DIR/report.md"
    fi
fi

if e2e smoke-failure 4 STUB_PANIC=b -- --max-host-load 0; then
    if [[ ! -s "$E2E_DIR/state/order" ]] && grep -q "variant b.*'x/b'" "$E2E_DIR/log"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 4, nothing measured\n' e2e-smoke-failure
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s measured after a smoke failure\n' e2e-smoke-failure
    fi
fi

if e2e host-busy 2 PRISM_BENCH_HOST_LOAD=50 -- --max-host-load 10; then
    if [[ ! -s "$E2E_DIR/state/order" ]]; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 2, nothing measured\n' e2e-host-busy
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s measured on a busy host\n' e2e-host-busy
    fi
fi

echo
echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]

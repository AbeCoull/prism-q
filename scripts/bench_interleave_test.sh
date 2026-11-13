#!/usr/bin/env bash
# Verify the statistics, verdicts, and exit codes of scripts/bench_interleave.sh.
#
# Most cases feed a synthetic raw table through --from-tsv. Each row carries a
# base time, a linear drift per invocation, deterministic noise, and an effect on
# variant b, so a case states what the measurement would have read and the
# expected verdict. The last cases run the whole script against a stub bench
# binary to cover the A B B A order and its several-variant form, the smoke test,
# the MEASURING marker, a TERM while a bench binary runs, and the host-load
# guard. Further cases cover two-binary mode (--base-exe) and the bench_ab.sh
# side of it: --interleave dispatch, the merge-base reference default, and the
# clean-tree same-code error, in a scratch repository with both executables
# supplied, so nothing builds.
#
# Usage: scripts/bench_interleave_test.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/bench_interleave.sh"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
# The end-to-end cases measure, so the runner writes a MEASURING marker; keep it
# out of the checkout, where a real measurement would see it.
export PRISM_BENCH_LOCK="$SCRATCH/MEASURING"

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
check five-rounds-no-data       5 "NO DATA" 5 target:x/a:1000:1:0.005:0
check min-rows-no-data          5 "NO DATA" 10 target:x/a:1000:1:0.005:0 -- --min-rows 2

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

# Two variants in one table, in the order the runner uses: b regresses 10% on x/a,
# c reads flat. Each gets its own verdict and the run takes the worse.
dir="$SCRATCH/two-variants"
mkdir -p "$dir"
awk 'BEGIN { OFS = "\t"; t = 0
    for (r = 1; r <= 10; r++) {
        split((r % 2) ? "a b c c b a" : "b c a a c b", order, " ")
        for (s = 1; s <= 6; s++) {
            v = order[s]; t++
            ns = 1000 * (1 + 0.002 * t); if (v == "b") ns *= 1.10
            print r, s, v, "target", "x/a", ns, ns, ns
            print r, s, v, "control", "c/a", 800 * (1 + 0.002 * t), 0, 800 * (1 + 0.002 * t)
        } } }' > "$dir/raw.tsv"
bash "$RUNNER" --from-tsv "$dir/raw.tsv" --out "$dir/report.md" --max-host-load 0 > /dev/null 2>&1
got=$?
if [[ "$got" == 1 ]] && grep -q '^\*\*Regression verdict\*\*: FAIL. Each variant against a: b FAIL, c PASS.$' "$dir/report.md" \
    && grep -q '^\*\*Variant b verdict\*\*: FAIL' "$dir/report.md" && grep -q '^\*\*Variant c verdict\*\*: PASS' "$dir/report.md" \
    && [[ "$(grep -c '^\*\*Regression verdict' "$dir/report.md")" == 1 ]]; then
    PASS=$((PASS + 1)); printf 'ok    %-34s exit 1, b FAIL, c PASS\n' two-variants
else
    FAIL=$((FAIL + 1)); printf 'FAIL  %-34s exit %s\n' two-variants "$got"
    sed 's/^/      /' "$dir/report.md"
fi

echo
echo "=== bench_interleave end to end against a stub binary ==="

# A stand-in for a Criterion bench binary: rows x/a, x/b, c/a; variant b is
# slower on x/b by STUB_EFFECT; every invocation drifts the times up by 0.3%;
# STUB_BURST lists invocations that run three times slow, as under load; STUB_PANIC
# names a variant that panics in test mode. STUB_SLOW_AS names an executable, by
# file name, whose x/b runs slower by STUB_EFFECT whatever the variant, and every
# invocation appends `mode variant name` to the state's `who` file.
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
if [[ -n "${STUB_SLEEP:-}" && "$mode" == bench ]]; then
    echo "$$" > "$STUB_STATE/pid"
    exec sleep "$STUB_SLEEP"
fi
calls=$(( $(cat "$STUB_STATE/calls" 2>/dev/null || echo 0) + 1 ))
echo "$calls" > "$STUB_STATE/calls"
[[ "$mode" == bench ]] && echo "$PRISM_BENCH_VARIANT" >> "$STUB_STATE/order"
me="$(basename "$0")"
echo "$mode ${PRISM_BENCH_VARIANT:-none} $me" >> "$STUB_STATE/who"
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
        -v burst=" ${STUB_BURST:-} " -v slow="${STUB_SLOW_AS:-}" -v me="$me" 'BEGIN {
            ns = 1000 * (1 + 0.003 * c); if ((v == "b" || me == slow) && row == "x/b") ns *= e
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

if e2e two-variants 1 STUB_EFFECT=1.10 -- --max-host-load 0 --variants b,c; then
    # Warmup a b c, then a b c c b a and b c a a c b, each slot one target and
    # one control invocation.
    want_order="a a b b c c"
    for (( r = 1; r <= 6; r++ )); do
        if (( r % 2 )); then want_order+=" a a b b c c c c b b a a"; else want_order+=" b b c c a a a a c c b b"; fi
    done
    got_order="$(tr '\n' ' ' < "$E2E_DIR/state/order" | sed 's/ $//')"
    if [[ "$got_order" == "$want_order" ]] \
        && grep -q 'Each variant against a: b FAIL, c PASS' "$E2E_DIR/report.md"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 1, mirrored order, b FAIL, c PASS\n' e2e-two-variants
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s order %s\n' e2e-two-variants "$got_order"
        sed 's/^/      /' "$E2E_DIR/report.md"
    fi
fi

if e2e bad-variant 5 -- --max-host-load 0 --variants b,a; then
    PASS=$((PASS + 1)); printf 'ok    %-34s exit 5, a is the base\n' e2e-bad-variant
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
echo "=== bench_interleave two-binary mode ==="

# The order the runner uses for one variant against a over six rounds, after the
# warmup, each slot one target and one control invocation.
ABBA_ORDER="a a b b"
for (( r = 1; r <= 6; r++ )); do
    if (( r % 2 )); then ABBA_ORDER+=" a a b b b b a a"; else ABBA_ORDER+=" b b a a a a b b"; fi
done

# Variant a runs base-exe and b the --exe stub, whose x/b is 10% slower; both see
# PRISM_BENCH_VARIANT=a, so only the executable tells them apart.
cp "$STUB" "$SCRATCH/base-exe"
if e2e two-binary 1 STUB_EFFECT=1.10 STUB_SLOW_AS=stub-exe -- --max-host-load 0 --base-exe "$SCRATCH/base-exe"; then
    who="$E2E_DIR/state/who"
    got_order="$(awk '$1 == "bench" { print ($3 == "base-exe" ? "a" : "b") }' "$who" | tr '\n' ' ' | sed 's/ $//')"
    if [[ "$got_order" == "$ABBA_ORDER" ]] && ! awk '$2 != "a"' "$who" | grep -q . \
        && grep -q '^test a base-exe$' "$who" && grep -q '^test a stub-exe$' "$who" \
        && grep -q '^## Interleaved two-binary A/B' "$E2E_DIR/report.md" \
        && grep -q '^| Mode | two binaries' "$E2E_DIR/report.md" \
        && grep -q '^| Layout check | .*control rows' "$E2E_DIR/report.md" \
        && grep -q '`x/b`.*regressed' "$E2E_DIR/report.md" && grep -q $'^#two-binary\t' "$E2E_DIR/report.tsv"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 1, a on --base-exe, b on --exe\n' e2e-two-binary
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s order %s\n' e2e-two-binary "$got_order"
        sed 's/^/      /' "$E2E_DIR/report.md"
    fi
    bash "$RUNNER" --from-tsv "$E2E_DIR/report.tsv" --out "$E2E_DIR/again.md" --max-host-load 0 > /dev/null 2>&1
    got=$?
    if [[ "$got" == 1 ]] && grep -q '^## Interleaved two-binary A/B' "$E2E_DIR/again.md" \
        && grep -q '^| Binaries | from tsv |' "$E2E_DIR/again.md"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 1, mode read back from the tsv\n' two-binary-from-tsv
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s exit %s\n' two-binary-from-tsv "$got"
        sed 's/^/      /' "$E2E_DIR/again.md"
    fi
fi

# Two-binary options the runner refuses before running anything: LABEL MESSAGE OPTIONS.
refuse() {
    local label="$1" want="$2" dir="$SCRATCH/refuse-$1" got
    shift 2
    mkdir -p "$dir/state"
    STUB_STATE="$dir/state" bash "$RUNNER" --project-dir "$SCRIPT_DIR/.." --max-host-load 0 \
        --out "$dir/report.md" "$@" > "$dir/log" 2>&1
    got=$?
    if [[ "$got" == 5 ]] && grep -q -- "$want" "$dir/log" && [[ ! -e "$dir/state/who" ]]; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 5, nothing ran\n' "$label"
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s want exit 5 and %s, got %s\n' "$label" "$want" "$got"
        sed 's/^/      /' "$dir/log"
    fi
}
refuse two-binary-variants 'must stay b' -f '^x/' -c '^c/' --exe "$STUB" --base-exe "$SCRATCH/base-exe" --variants b,c
refuse two-binary-no-control 'needs --control' -f '^x/' --exe "$STUB" --base-exe "$SCRATCH/base-exe"
refuse two-binary-no-exe 'needs --exe' -f '^x/' -c '^c/' --base-exe "$SCRATCH/base-exe"
refuse two-binary-missing-base 'does not exist' -f '^x/' -c '^c/' --exe "$STUB" --base-exe "$SCRATCH/nothing"

echo
echo "=== bench_ab.sh reference default and --interleave ==="

# main holds c1 and c2, feat branches from c2 with c3. Without a remote the
# default reference is the merge-base of HEAD and main.
AB_REPO="$SCRATCH/ab-repo"
git init -q -b main "$AB_REPO"
g() { git -C "$AB_REPO" -c user.name=t -c user.email=t@t "$@"; }
echo one > "$AB_REPO/f.txt"; g add f.txt; g commit -q -m c1
echo two > "$AB_REPO/f.txt"; g commit -q -am c2
g checkout -q -b feat
echo three > "$AB_REPO/f.txt"; g commit -q -am c3
C1="$(g rev-parse --short HEAD~2)"
C2="$(g rev-parse --short HEAD~1)"

# ab_run LABEL WANT_EXIT [ENV=VALUE...] -- [bench_ab.sh OPTIONS], with both
# executables supplied, so nothing builds.
ab_run() {
    local label="$1" want="$2"
    shift 2
    local envs=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
    shift
    AB_DIR="$SCRATCH/ab-$label"
    mkdir -p "$AB_DIR/state"
    env STUB_STATE="$AB_DIR/state" ${envs[@]+"${envs[@]}"} bash "$SCRIPT_DIR/bench_ab.sh" \
        --project-dir "$AB_REPO" --ref-exe "$STUB" --new-exe "$STUB" -f '^x/' -c '^c/' \
        --light --max-host-load 0 --out "$AB_DIR/report.md" "$@" > "$AB_DIR/log" 2>&1
    AB_STATUS=$?
    if [[ "$AB_STATUS" == "$want" ]]; then
        return 0
    fi
    FAIL=$((FAIL + 1))
    printf 'FAIL  %-34s want exit %s, got %s\n' "$label" "$want" "$AB_STATUS"
    sed 's/^/      /' "$AB_DIR/log"
    return 1
}

if ab_run interleave-merge-base 1 STUB_EFFECT=1.10 STUB_SLOW_AS=exe-new -- --interleave; then
    if grep -q "^| Reference | merge-base of HEAD and \`main\` ($C2) |" "$AB_DIR/report.md" \
        && grep -q '^## Interleaved two-binary A/B' "$AB_DIR/report.md" \
        && grep -q '^| Method | `bench_ab.sh --interleave` |' "$AB_DIR/report.md" \
        && grep -q '`x/b`.*regressed' "$AB_DIR/report.md" \
        && [[ "$(awk '$1 == "bench" { print $3; exit }' "$AB_DIR/state/who")" == exe-ref ]] \
        && grep -q '^test a exe-ref$' "$AB_DIR/state/who" && grep -q '^test a exe-new$' "$AB_DIR/state/who"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 1, ref is a, reference %s\n' ab-interleave-merge-base "$C2"
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s report or order wrong\n' ab-interleave-merge-base
        sed 's/^/      /' "$AB_DIR/report.md"
    fi
fi

g update-ref refs/remotes/origin/main HEAD~2
if ab_run origin-main-first 0 -- --interleave; then
    if grep -q "^| Reference | merge-base of HEAD and \`origin/main\` ($C1) |" "$AB_DIR/report.md" \
        && grep -q '^\*\*Regression verdict\*\*: PASS' "$AB_DIR/report.md"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 0, reference %s\n' ab-origin-main-first "$C1"
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s reference not from origin/main\n' ab-origin-main-first
        sed 's/^/      /' "$AB_DIR/report.md"
    fi
fi
g update-ref -d refs/remotes/origin/main

g checkout -q main
if ab_run clean-same-code 5 -- --interleave; then
    if grep -q 'same code' "$AB_DIR/log" && grep -q -- '--ref' "$AB_DIR/log" && [[ ! -e "$AB_DIR/state/who" ]]; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 5, names --ref, nothing ran\n' ab-clean-same-code
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s message or a run\n' ab-clean-same-code
        sed 's/^/      /' "$AB_DIR/log"
    fi
fi

if ab_run explicit-head 0 -- --interleave --ref HEAD; then
    if grep -q '^\*\*Regression verdict\*\*: CONTROL (same code)' "$AB_DIR/report.md" \
        && grep -q '^| Same code |' "$AB_DIR/report.md"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 0, CONTROL (same code)\n' ab-explicit-head
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s no CONTROL verdict\n' ab-explicit-head
        sed 's/^/      /' "$AB_DIR/report.md"
    fi
fi

echo edit > "$AB_REPO/f.txt"
if ab_run dirty-at-merge-base 0 -- --interleave; then
    PASS=$((PASS + 1)); printf 'ok    %-34s exit 0, an uncommitted change runs\n' ab-dirty-at-merge-base
fi
g checkout -q -- f.txt

if ab_run escalate-refused 5 -- --interleave --escalate; then
    if grep -q -- '--escalate' "$AB_DIR/log" && [[ ! -e "$AB_DIR/state/who" ]]; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 5, nothing ran\n' ab-escalate-refused
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s message or a run\n' ab-escalate-refused
    fi
fi

if ab_run interleave-no-control 5 -- --interleave -c ''; then
    if grep -q 'needs --control' "$AB_DIR/log"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 5\n' ab-interleave-no-control
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s no message\n' ab-interleave-no-control
    fi
fi

if ab_run smoke-failure 4 STUB_PANIC=a -- --interleave --ref HEAD; then
    if [[ ! -s "$AB_DIR/state/order" ]]; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 4, nothing measured\n' ab-interleave-smoke-failure
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s measured after a smoke failure\n' ab-interleave-smoke-failure
    fi
fi

git init -q -b other "$SCRATCH/ab-nomain"
git -C "$SCRATCH/ab-nomain" -c user.name=t -c user.email=t@t commit -q --allow-empty -m one
if ab_run no-base 5 -- --interleave --project-dir "$SCRATCH/ab-nomain"; then
    if grep -q 'neither origin/main nor main' "$AB_DIR/log"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 5, names both bases\n' ab-no-base
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s no message\n' ab-no-base
    fi
fi

if [[ ! -e "$PRISM_BENCH_LOCK" ]]; then
    PASS=$((PASS + 1)); printf 'ok    %-34s every run removed its marker\n' e2e-marker-released
else
    FAIL=$((FAIL + 1)); printf 'FAIL  %-34s a run left %s behind\n' e2e-marker-released "$PRISM_BENCH_LOCK"
    rm -f "$PRISM_BENCH_LOCK"
fi

sleep 300 &
HOLDER=$!
printf 'pid: %s\nby: test\n' "$HOLDER" > "$PRISM_BENCH_LOCK"
if e2e lock-held 2 -- --max-host-load 0; then
    if [[ ! -s "$E2E_DIR/state/order" ]] && grep -q "^pid: $HOLDER\$" "$PRISM_BENCH_LOCK"; then
        PASS=$((PASS + 1)); printf 'ok    %-34s exit 2, nothing measured, marker kept\n' e2e-lock-held
    else
        FAIL=$((FAIL + 1)); printf 'FAIL  %-34s measured under a live marker\n' e2e-lock-held
    fi
fi
kill "$HOLDER" 2>/dev/null
wait "$HOLDER" 2>/dev/null
rm -f "$PRISM_BENCH_LOCK"

# A TERM to the runner ends the bench binary it is waiting on and removes the
# marker, rather than leaving the binary measuring after the runner is gone.
dir="$SCRATCH/e2e-term"
mkdir -p "$dir/state"
STUB_STATE="$dir/state" STUB_SLEEP=300 bash "$RUNNER" --exe "$STUB" -f '^x/' --rounds 6 \
    --out "$dir/report.md" --project-dir "$SCRIPT_DIR/.." --max-host-load 0 > "$dir/log" 2>&1 &
runner=$!
for (( i = 0; i < 100; i++ )); do [[ -s "$dir/state/pid" ]] && break; sleep 0.2; done
kill -TERM "$runner" 2>/dev/null
wait "$runner"
got=$?
child="$(cat "$dir/state/pid" 2>/dev/null)"
if [[ "$got" == 143 && -n "$child" ]] && ! kill -0 "$child" 2>/dev/null && [[ ! -e "$PRISM_BENCH_LOCK" ]]; then
    PASS=$((PASS + 1)); printf 'ok    %-34s exit 143, binary ended, marker removed\n' e2e-term
else
    FAIL=$((FAIL + 1)); printf 'FAIL  %-34s exit %s, child %s\n' e2e-term "$got" "${child:-none}"
    [[ -n "$child" ]] && kill "$child" 2>/dev/null
    sed 's/^/      /' "$dir/log"
fi

echo
echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]

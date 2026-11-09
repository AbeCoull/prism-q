#!/usr/bin/env bash
# Run the benchmark subset used by the CI regression gate.
#
# Delegates to the adjacent-binary A/B in scripts/bench_ab.sh, which explains why
# measuring the base, building head, then measuring head is not a comparison.
#
# The filter covers representative hot paths rather than the full Criterion
# suite, and `--min-rows` fails the run if one stops matching. `bench-fast` keeps
# the four passes affordable: the widest rows run milliseconds per iteration, so
# the sample count sets the cost outright.
#
# A FAIL is confirmed before it blocks. The set runs a second time and only a row
# that regressed in both runs counts, because at ten samples per lane a single
# run of this tier is not a result: four runs of identical code against identical
# code produced three PASSes and one FAIL, the FAIL on `statevector/scalability_d5/18`
# at +5.3% against a 4.2% control spread. Confirmation costs nothing on a green
# run and one extra set on a red one. Only a valid second result clears the
# first: a PASS, or a FAIL on disjoint rows. A RERUN, NO DATA, a busy host or a
# missing report leaves the FAIL standing. `scripts/bench_ci_test.sh` covers each.
#
# Environment:
#   PRISM_BENCH_REF       git ref for the reference build (required)
#   PRISM_BENCH_REF_EXE   prebuilt reference executable. When it names an
#                         existing file the reference build is skipped and the
#                         worktree is never created, which halves the build cost.
#                         Ignored when the file is absent, so a cache miss falls
#                         back to building the reference.
#   PRISM_BENCH_REF_DIR   reference worktree path, cached between CI runs
#   CI_BENCH_FEATURES     cargo feature list (default: parallel,bench-fast)
#   REGRESSION_THRESHOLD  regression gate in percent (default: 5.0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FEATURES="${CI_BENCH_FEATURES:-parallel,bench-fast}"

if [[ -z "${PRISM_BENCH_REF:-}" ]]; then
    echo "Error: PRISM_BENCH_REF must name the reference commit." >&2
    exit 1
fi

# One or two rows per simulation path, not per bench group. The gate exists to
# notice that a path regressed, and a second row of the same shape doubles the
# wall clock for a signal the first already carries. Every backend the crate
# ships has a row here except the GPU and distributed ones, which have bench
# targets of their own.
#
# Sizes are the smallest that still exercise the paths the gate exists to watch,
# not the smallest available. `MIN_QUBITS_FOR_DIAG_BATCH = 14` and
# `MIN_QUBITS_FOR_POST_PHASE_BATCH = 18` in `circuit/fusion.rs` are the floors:
# below 14 the diagonal-batch families never form, and below 18 the post-phase
# rebatch never runs, so a smaller row would gate a different pipeline than the
# one that ships. `qft_textbook` has no 18 in its size list and stays at 20.
#
# Under `bench-fast` a row whose iteration fits the 1s measurement window costs
# the window, so a 0.2 ms row and a 60 ms row price the same, and rows above
# about 100 ms per iteration are chosen against: their smaller siblings gate the
# same code for a few seconds each. Rows are also chosen against noise, measured
# as the drift between two runs of one binary. `statevector/qv/16` is absent for
# that reason (same-code controls of 6.3% to 8.0%), and the guarded region runs
# at 20 qubits, where the 16-qubit row drifted.
#
# The `auto/` sweep twins are absent on purpose: dispatch resolves them to the
# statevector rows below, doubling the most expensive pair for nothing.
# `auto/expectation` is the exception, because the scalar tensor route it takes
# is reached from no other row here.
#
# The stabilizer measurement row is `wall_measure_all` rather than
# `ghz_measure_all` for two reasons. A GHZ chain collapses once and reads the
# rest of the register off deterministically, so it prices almost none of the
# measurement work the row exists to watch, where a CNOT wall collapses every
# qubit at random. And at 500 qubits the GHZ row runs in the microsecond band,
# where this fixture is a known cache cliff: one PR saw it read +5.7% and fail,
# then -1.4% and pass on a rerun of the same commit, with tight same-code
# controls both times, because the reference lane itself moved 113 to 165
# microseconds between runs on a byte-identical cached executable. A gate whose
# verdict blocks a merge cannot rest on a row that moves 46% for reasons a
# within-run control cannot see.
CIRCUITS_ROW_LIST=(
    # Dense statevector: scaling, the structured phase and swap workload, a
    # non-Clifford shape, ZZ rotations with mixer layers, and the diagonal-batch
    # families.
    statevector/scalability_d5/18
    statevector/qft_textbook/20
    statevector/qpe_t_gate/16q
    statevector/qaoa_l3/16
    statevector/diag_mixed_l6/20

    # Clifford tableau, terminal measurement, and the block-split variant.
    stabilizer/scaling/500
    stabilizer/measurement/wall_measure_all/1000
    factored_stabilizer/local_blocks/10x5/50

    # Compiled shot sampling, noiseless and under a Pauli model.
    compiled_sampler/noiseless/noiseless_500q_10k
    compiled_sampler/noisy/noisy_500q_10k

    # Matrix product state: gate application with truncation, then sampling.
    mps/random_d10/16
    mps/sampling/32

    # Sparse map state: gate application, then sampling.
    sparse/random_d10/20
    sparse/sampling/32

    # Density matrix: the fused unitary sandwich and a Kraus channel.
    density_matrix/rzz_layers_fused/10
    density_matrix/noisy_channels/kraus_amp_damp/10

    # Split-state and tensor network, the latter for contraction and for the
    # mid-circuit measurement path that forces a dense collapse.
    factored/partial_independence/factored/20
    tn/linear_chain/12
    tn/midmeasure_chain/20

    # Product state, no entanglement.
    product/scaling_d10/64

    # Clifford+T: the deterministic and stochastic Pauli engines, and the
    # approximate stabilizer-rank route with its pruning.
    clifford_t/spd/10q_12t
    spp/spp_10k/20q_8t
    stabilizer_rank/approx_1024/10q_20t

    # Analytic terminals: adjoint gradients, Pauli-sum expectations, and the
    # scalar tensor route Auto takes for an expectation terminal.
    gradient/qaoa_l1/adjoint/18
    expectation/pauli_sum/18
    auto/expectation/brickwork_d4/20

    # Noise: trajectory sampling on a non-Clifford circuit, the compiled Pauli
    # sampler, and the Pauli path engine.
    noisy_sampling/trajectory_pauli/non_clifford_12q_512
    noisy_sampling/compiled_pauli/clifford_100q_10k
    pauli_path/noisy_expectation/path/50

    # Circuit-level paths that no backend row reaches: a guarded region and
    # parameter binding without a rebuild.
    dynamic/guarded_region/20
    vqe/bind/16
)

CIRCUITS_FILTER="^($(IFS='|'; echo "${CIRCUITS_ROW_LIST[*]}"))$"
CIRCUITS_ROWS=${#CIRCUITS_ROW_LIST[@]}

# The job builds head in place straight before measuring. A same-code run
# measured off such a build read 9.1% slower at the median on the development
# host, though one load reading after the build already looked idle, so the
# first pass waits for the runner to read quiet for a minute without a break.
# Busy time counts against --wait-idle, and a runner that stays busy is refused.
args=(
    --bench circuits
    --filter "$CIRCUITS_FILTER"
    --min-rows "$CIRCUITS_ROWS"
    --ref "$PRISM_BENCH_REF"
    --features "$FEATURES"
    --wait-idle 120
    --settle 60
)

if [[ -n "${PRISM_BENCH_REF_EXE:-}" && -f "${PRISM_BENCH_REF_EXE}" ]]; then
    echo "Reference executable restored from cache; skipping the reference build."
    args+=(--ref-exe "$PRISM_BENCH_REF_EXE")
elif [[ -n "${PRISM_BENCH_REF_DIR:-}" ]]; then
    echo "No cached reference executable; building the reference from a worktree."
    args+=(--ref-dir "$PRISM_BENCH_REF_DIR")
fi

FIRST="$PROJECT_DIR/bench_results/ci-ab.md"
CONFIRM="$PROJECT_DIR/bench_results/ci-ab-confirm.md"
rm -f "$CONFIRM"

# The regressed rows are the bullet list that follows the FAIL verdict, which
# ends at the next line that is neither blank nor a bullet.
regressed_rows() {
    awk '
        /^\*\*Regression verdict\*\*: FAIL/ { inside = 1; next }
        !inside { next }
        /^- `/ { gsub(/^- `|`$/, ""); print; next }
        /^$/ { next }
        { inside = 0 }
    ' "$1"
}

if bash "$SCRIPT_DIR/bench_ab.sh" "${args[@]}" --out "$FIRST"; then
    exit 0
fi

if ! grep -q '^\*\*Regression verdict\*\*: FAIL' "$FIRST"; then
    exit 1
fi

FIRST_ROWS="$(regressed_rows "$FIRST")"
if [[ -z "$FIRST_ROWS" ]]; then
    echo "Report says FAIL but names no row. Blocking rather than confirming a list that could not be read." >&2
    exit 1
fi

echo
echo "Regression reported. Running the set again; only a row that regresses twice blocks."
echo

CONFIRM_STATUS=0
bash "$SCRIPT_DIR/bench_ab.sh" "${args[@]}" --out "$CONFIRM" || CONFIRM_STATUS=$?

CONFIRM_VERDICT=""
if [[ -f "$CONFIRM" ]]; then
    CONFIRM_VERDICT="$(sed -n 's/^\*\*Regression verdict\*\*: \([A-Z][A-Z ]*[A-Z]\).*/\1/p' "$CONFIRM")"
fi

case "$CONFIRM_STATUS/$CONFIRM_VERDICT" in
    0/PASS)
        echo "Second run passes. Treating the first as unconfirmed."
        exit 0
        ;;
    1/FAIL)
        ;;
    *)
        echo "Second run gave no valid result (exit $CONFIRM_STATUS, verdict ${CONFIRM_VERDICT:-none}). The first FAIL stands." >&2
        exit 1
        ;;
esac

CONFIRM_ROWS="$(regressed_rows "$CONFIRM")"
if [[ -z "$CONFIRM_ROWS" ]]; then
    echo "Second report says FAIL but names no row. Blocking rather than clearing on a list that could not be read." >&2
    exit 1
fi

CONFIRMED="$(comm -12 <(sort <<<"$FIRST_ROWS") <(sort <<<"$CONFIRM_ROWS"))"

if [[ -z "$CONFIRMED" ]]; then
    echo "Both runs reported a regression, on disjoint rows. Neither is confirmed."
    exit 0
fi

echo "Regressed in both runs:"
printf '%s\n' "$CONFIRMED" | sed 's/^/  /'
exit 1

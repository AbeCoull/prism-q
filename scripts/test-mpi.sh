#!/usr/bin/env bash
# Build and run the distributed backend rank correctness check under a Linux
# MPI launcher. Mirrors scripts/test-mpi.ps1.
#
# Usage: scripts/test-mpi.sh [--ranks "1 2 4"] [--hostfile FILE] [--release]
#                            [--skip-lib-tests] [--require]
#        scripts/test-mpi.sh --timed [--reps 5] [--chunk 65536] [--hostfile FILE]
#
# Builds the check binary with Rayon enabled (the shipped combination, which
# needs MPI_THREAD_FUNNELED at init), then launches it across N ranks under
# three configurations (default, tiled exchange, relabeling off). Rank 0
# asserts the gathered result matches the one process statevector reference
# and reports the world size it saw, which must equal N. A final three rank run
# asserts the power of two requirement is rejected.
#
# --require fails the run where it would otherwise skip the Python checks
# (mpi4py absent), and sets PRISM_REQUIRE_MPI_RANKS so test_distributed.py
# fails rather than skips without MPI support and asserts the world size.
#
# --timed runs only the timed arm at 2 and 4 ranks and prints one TIMED: line
# per rank, circuit, and width. With --hostfile the ranks spread one per host
# first, so 2 ranks is the cross-host case. --chunk 0 leaves the exchange
# untiled.
#
# MPIEXEC selects the launcher (default mpirun). Open MPI flags are assumed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ranks="1 2 4"; hostfile=""; timed=0; reps=5; chunk=65536; profile=debug; lib_tests=1; require=0
while [ $# -gt 0 ]; do
    case "$1" in
        --ranks) ranks="$2"; shift ;;
        --hostfile) hostfile="$2"; shift ;;
        --release) profile=release ;;
        --skip-lib-tests) lib_tests=0 ;;
        --require) require=1 ;;
        --timed) timed=1; profile=release ;;
        --reps) reps="$2"; shift ;;
        --chunk) chunk="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

mpiexec="${MPIEXEC:-mpirun}"
command -v "$mpiexec" >/dev/null || { echo "MPI launcher '$mpiexec' not found" >&2; exit 1; }
launch=("$mpiexec")
[ -z "$hostfile" ] || launch+=(--hostfile "$hostfile" --map-by node --bind-to none)
[ -n "$hostfile" ] || launch+=(--oversubscribe --bind-to none)

# Launch `cmd` on N ranks with the named variables set for this launch only and
# forwarded to every rank. A local launch hands ranks the launcher's whole
# environment, so exporting them instead would leak one configuration into the
# next.
run() {
    local n="$1"; shift
    local flags=()
    for kv in "$@"; do flags+=(-x "${kv%%=*}"); done
    env "$@" "${launch[@]}" -n "$n" "${flags[@]}" "${cmd[@]}"
}

features="parallel distributed-mpi"
if [ "$timed" = 1 ]; then
    echo; echo "== Building check binary (release) =="
    cargo build --release --example dist_mpi_check --features "$features"
    cmd=(target/release/examples/dist_mpi_check)
    for n in 2 4; do
        echo; echo "== $mpiexec -n $n dist_mpi_check [timed, chunk $chunk, reps $reps] =="
        run "$n" "PRISM_DIST_TIMED_REPS=$reps" "PRISM_DIST_EXCHANGE_CHUNK=$chunk" | grep '^TIMED:' | sort
    done
    exit 0
fi

if [ "$lib_tests" = 1 ]; then
    echo; echo "== Running SerialComm lib tests =="
    cargo test --features "$features" --lib distributed
fi

echo; echo "== Building check binary ($profile) =="
if [ "$profile" = release ]; then
    cargo build --release --example dist_mpi_check --features "$features"
    cmd=(target/release/examples/dist_mpi_check)
else
    cargo build --example dist_mpi_check --features "$features"
    cmd=(target/debug/examples/dist_mpi_check)
fi

declare -A meas shots
for n in $ranks; do
    for cfg in default tiled-exchange no-relabel; do
        case "$cfg" in
            default) extra=() ;;
            tiled-exchange) extra=("PRISM_DIST_EXCHANGE_CHUNK=4096") ;;
            no-relabel) extra=("PRISM_DIST_RELABEL=0") ;;
        esac
        echo; echo "== $mpiexec -n $n dist_mpi_check [$cfg] =="
        out=$(run "$n" "PRISM_DIST_MIN_LOCAL_QUBITS=1" "${extra[@]}")
        echo "$out"
        grep -q "^OK: $n ranks," <<< "$out" || { echo "rank 0 did not report a world of $n ranks at $n/$cfg" >&2; exit 1; }
        meas["$n/$cfg"]=$(echo "$out" | grep -o 'outcome_sig=[^ ]*' | head -1)
        shots["$n/$cfg"]=$(echo "$out" | grep -o 'shots_sig=[^ ]*' | head -1)
        [ -n "${meas["$n/$cfg"]}" ] || { echo "measurement signature missing at $n/$cfg" >&2; exit 1; }
        [ -n "${shots["$n/$cfg"]}" ] || { echo "shots signature missing at $n/$cfg" >&2; exit 1; }
    done
done
if [ "$(printf '%s\n' "${meas[@]}" | sort -u | wc -l)" != 1 ]; then
    echo "measurement outcomes differ across configurations:" >&2
    for k in "${!meas[@]}"; do echo "  $k ${meas[$k]}" >&2; done
    exit 1
fi
echo; echo "Measurement determinism: ${meas["${ranks%% *}/default"]} identical across ranks and configurations."
if [ "$(printf '%s\n' "${shots[@]}" | sort -u | wc -l)" != 1 ]; then
    echo "shot outcomes differ across configurations:" >&2
    for k in "${!shots[@]}"; do echo "  $k ${shots[$k]}" >&2; done
    exit 1
fi
echo "Shot sampling determinism: ${shots["${ranks%% *}/default"]} identical across ranks and configurations."

echo; echo "== $mpiexec -n 3 dist_mpi_check (expected to fail) =="
if run 3 "PRISM_DIST_MIN_LOCAL_QUBITS=1"; then
    echo "a rank count that is not a power of two must fail" >&2; exit 1
fi
echo "Rejected as expected."

echo; echo "== Python distributed checks =="
if python3 -c "import mpi4py" 2>/dev/null; then
    python3 -m maturin develop --manifest-path bindings/python/Cargo.toml --features distributed-mpi
    cmd=(python3 -m pytest bindings/python/tests/test_distributed.py -q -rs -p no:cacheprovider)
    required() { [ "$require" = 0 ] || echo "PRISM_REQUIRE_MPI_RANKS=$1"; }
    env $(required 1) "${cmd[@]}"
    for n in $ranks; do
        echo; echo "== $mpiexec -n $n pytest test_distributed.py =="
        run "$n" PRISM_DIST_MIN_LOCAL_QUBITS=1 $(required "$n")
    done
elif [ "$require" = 1 ]; then
    echo "mpi4py is not installed, and --require forbids skipping the Python checks." >&2
    exit 1
else
    echo "mpi4py is not installed; skipping the Python checks."
fi
echo; echo "All distributed MPI checks passed."

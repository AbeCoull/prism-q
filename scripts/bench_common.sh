#!/usr/bin/env bash
# Helpers shared by bench_ab.sh, bench_interleave.sh, and bench_queue.sh. Sourced,
# not run.
#
# preflight reads MAX_HOST_LOAD, WAIT_IDLE, and SETTLE from the caller and lowers
# MAX_HOST_LOAD to 0 where the host offers no load reading.

# Windows-native path for variables read by cargo and by the bench binary. MSYS
# converts arguments but not arbitrary environment variables, so a POSIX path in
# CARGO_TARGET_DIR or CRITERION_HOME reaches the native binary unusable.
native_path() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1"
    else
        printf '%s' "$1"
    fi
}

# Content hash of the checkout at $1 (default: the current directory): HEAD,
# every tracked modification against it, and every untracked file that is not
# ignored. Two equal hashes mean a build of either checkout compiles the same
# sources.
tree_fingerprint() {
    local dir="${1:-.}"
    {
        git -C "$dir" rev-parse HEAD
        git -C "$dir" diff HEAD --binary
        git -C "$dir" ls-files -o --exclude-standard -z | while IFS= read -r -d '' f; do
            printf '%s ' "$f"
            git -C "$dir" hash-object "$f"
        done
    } | git hash-object --stdin
}

# The verdict word of the first `**Regression verdict**:` line in report $1, such
# as PASS, FAIL, RERUN, NO DATA, or CONTROL, or nothing when the report is absent
# or carries none.
report_verdict() {
    [[ -f "$1" ]] || return 0
    sed -n 's/^\*\*Regression verdict\*\*: \([A-Z][A-Z ]*[A-Z]\).*/\1/p' "$1" | head -n 1
}

host_cpu() {
    if [[ -r /proc/cpuinfo ]]; then
        awk -F': ' '/model name/ { print $2; exit }' /proc/cpuinfo
    elif command -v sysctl >/dev/null 2>&1 && sysctl -n machdep.cpu.brand_string >/dev/null 2>&1; then
        sysctl -n machdep.cpu.brand_string
    else
        printf '%s' "${PROCESSOR_IDENTIFIER:-unknown}"
    fi
}

# Busy share of every CPU over about two seconds, as a whole percent, or empty
# where the host offers no reading. MSYS and Cygwin expose the Windows counters
# through /proc/stat, so one reader covers Windows, Linux, and WSL.
# PRISM_BENCH_HOST_LOAD fixes the reading, for the scripts' tests.
host_load() {
    if [[ -n "${PRISM_BENCH_HOST_LOAD:-}" ]]; then
        printf '%s' "$PRISM_BENCH_HOST_LOAD"
    elif [[ -r /proc/stat ]]; then
        local first second
        first="$(awk '/^cpu / { b = $2 + $3 + $4 + $7 + $8 + $9; print b, b + $5 + $6; exit }' /proc/stat)"
        sleep 2
        second="$(awk '/^cpu / { b = $2 + $3 + $4 + $7 + $8 + $9; print b, b + $5 + $6; exit }' /proc/stat)"
        awk -v a="$first" -v b="$second" 'BEGIN {
            split(a, x, " "); split(b, y, " ")
            if (y[2] > x[2]) { printf "%d", (y[1] - x[1]) * 100 / (y[2] - x[2]) }
        }'
    elif command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu >/dev/null 2>&1; then
        ps -A -o %cpu= | awk -v n="$(sysctl -n hw.ncpu)" '{ s += $1 } END { printf "%d", s / n }'
    fi
}

# The processes holding the most CPU, for the message that stops a run. MSYS ps
# sees only MSYS processes, so Windows asks the performance counters through CIM
# (wmic, the older route to the same class, is gone from current Windows).
busy_processes() {
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -NonInteractive -Command \
            "Get-CimInstance Win32_PerfFormattedData_PerfProc_Process |
             Where-Object { \$_.Name -ne '_Total' -and \$_.Name -ne 'Idle' -and \$_.PercentProcessorTime -gt 0 } |
             Sort-Object PercentProcessorTime -Descending | Select-Object -First 5 |
             ForEach-Object { '{0}% {1}' -f \$_.PercentProcessorTime, \$_.Name }" 2>/dev/null |
            tr -d '\r'
    else
        ps -A -o pcpu= -o comm= 2>/dev/null | sort -rn | head -5
    fi
}

# Stop with exit 2, or wait up to WAIT_IDLE seconds, while other work holds more of
# the host than MAX_HOST_LOAD percent. Under SETTLE the host must then read quiet
# for that many seconds in a row before the caller measures; a busy reading
# restarts the window and the busy time counts against WAIT_IDLE.
preflight() {
    local load="" waited=0 quiet_since="" settle="${SETTLE:-0}"
    if (( MAX_HOST_LOAD > 0 )); then
        load="$(host_load)"
        if [[ -z "$load" ]]; then
            echo ">>> no host load reading here, so the load guard is off for this run"
            MAX_HOST_LOAD=0
        fi
    fi
    if (( MAX_HOST_LOAD <= 0 )); then
        if (( settle > 0 )); then
            echo ">>> settling for ${settle}s before measuring"
            sleep "$settle"
        fi
        return 0
    fi
    while :; do
        if (( load <= MAX_HOST_LOAD )); then
            [[ -n "$quiet_since" ]] || quiet_since=$SECONDS
            if (( SECONDS - quiet_since >= settle )); then
                if (( settle > 0 )); then
                    echo ">>> host ${load}% busy before measuring, quiet for $(( SECONDS - quiet_since ))s"
                else
                    echo ">>> host ${load}% busy before measuring"
                fi
                echo ""
                return 0
            fi
            sleep 3
        else
            quiet_since=""
            if (( waited >= WAIT_IDLE )); then
                echo "Error: the host is ${load}% busy before measuring, over --max-host-load ${MAX_HOST_LOAD}%." >&2
                busy_processes | sed 's/^/    /' >&2
                echo "  Stop that work, pass --wait-idle SECONDS to wait for it, or raise" >&2
                echo "  --max-host-load if the load is part of what is being measured." >&2
                exit 2
            fi
            echo ">>> host ${load}% busy, waiting for it to fall under ${MAX_HOST_LOAD}% (${waited}s of ${WAIT_IDLE}s)"
            if (( waited % 600 == 0 )); then
                busy_processes | sed 's/^/    /'
            fi
            sleep 28
            waited=$(( waited + 30 ))
        fi
        load="$(host_load)"
        load="${load:-0}"
    done
}

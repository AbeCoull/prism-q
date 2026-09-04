# Build and run the distributed backend rank correctness check.
#
# Usage:   powershell -ExecutionPolicy Bypass -File scripts\test-mpi.ps1 [-Ranks 4]
#
# Sets up the MPI build environment, builds the lib tests and the mpiexec check
# binary, then launches it across N ranks under three configurations (default,
# tiled exchange, relabeling off). Rank 0 asserts the gathered result matches
# the one process statevector reference and exits nonzero on mismatch. A final
# three rank run asserts the power of two requirement is rejected.

param(
    [int[]] $RankCounts = @(1, 2, 4)
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

. (Join-Path $scriptDir 'mpi-env.ps1')

Write-Host "`n== Building lib tests (distributed-mpi) =="
cargo test --features distributed-mpi --lib distributed --no-run
if ($LASTEXITCODE -ne 0) { throw "lib test build failed" }

Write-Host "`n== Running SerialComm lib tests =="
cargo test --features distributed-mpi --lib distributed
if ($LASTEXITCODE -ne 0) { throw "lib tests failed" }

Write-Host "`n== Building mpiexec check binary =="
cargo build --example dist_mpi_check --features distributed-mpi
if ($LASTEXITCODE -ne 0) { throw "example build failed" }

$exe = Join-Path $scriptDir '..\target\debug\examples\dist_mpi_check.exe'
$mpiexec = "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
if (-not (Test-Path $mpiexec)) { $mpiexec = "mpiexec" }

# Each configuration is run at every rank count. Results must not depend on
# either axis: tiling only splits a transfer into more messages, and relabeling
# only changes which amplitudes move. The check circuit is small, so the local
# qubit floor is relaxed everywhere to let it distribute on a single host.
$configs = @(
    @{ Name = 'default'; Env = @{} },
    @{ Name = 'tiled-exchange'; Env = @{ 'PRISM_DIST_EXCHANGE_CHUNK' = '4096' } },
    @{ Name = 'no-relabel'; Env = @{ 'PRISM_DIST_RELABEL' = '0' } }
)

$measSigs = @{}
$shotSigs = @{}
foreach ($n in $RankCounts) {
    foreach ($cfg in $configs) {
        $label = "$n/$($cfg.Name)"
        $envArgs = @('-env', 'PRISM_DIST_MIN_LOCAL_QUBITS', '1')
        foreach ($key in $cfg.Env.Keys) {
            $envArgs += @('-env', $key, $cfg.Env[$key])
        }
        Write-Host "`n== mpiexec -n $n dist_mpi_check [$($cfg.Name)] =="
        $out = & $mpiexec -n $n @envArgs $exe
        if ($LASTEXITCODE -ne 0) { throw "rank check failed at $label" }
        $out | ForEach-Object { Write-Host $_ }
        $match = $out | Select-String -Pattern 'outcome_sig=(\S+)' | Select-Object -First 1
        if (-not $match) {
            throw "measurement signature missing at $label"
        }
        $measSigs[$label] = $match.Matches[0].Groups[1].Value
        $match = $out | Select-String -Pattern 'shots_sig=(\S+)' | Select-Object -First 1
        if (-not $match) {
            throw "shots signature missing at $label"
        }
        $shotSigs[$label] = $match.Matches[0].Groups[1].Value
    }
}

$unique = $measSigs.Values | Select-Object -Unique
if ($unique.Count -gt 1) {
    throw "measurement outcomes differ across configurations: $($measSigs | Out-String)"
}
Write-Host "`nMeasurement determinism: signature $($unique) identical across ranks and configurations."

$uniqueShots = $shotSigs.Values | Select-Object -Unique
if ($uniqueShots.Count -gt 1) {
    throw "shot outcomes differ across configurations: $($shotSigs | Out-String)"
}
Write-Host "Shot sampling determinism: signature $($uniqueShots) identical across ranks and configurations."

# The backend requires a power of two rank count. Three ranks must fail loudly
# under a real launcher, not hang or produce a partial result.
Write-Host "`n== mpiexec -n 3 dist_mpi_check (expected to fail) =="
& $mpiexec -n 3 -env PRISM_DIST_MIN_LOCAL_QUBITS 1 $exe | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -eq 0) {
    throw "a rank count that is not a power of two must fail"
}
Write-Host "Rejected as expected (exit $LASTEXITCODE)."

# The Python surface is gated with the Rust one so the two cannot drift. The
# bindings are a separate crate and a separate build, and the check skips rather
# than fails when mpi4py is absent, which is what an interpreter without it
# reports.
Write-Host "`n== Python distributed checks =="
$py = & python -c "import mpi4py; print(mpi4py.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "mpi4py is not installed; skipping the Python checks."
} else {
    Write-Host "mpi4py $py"
    & python -m maturin develop --manifest-path bindings/python/Cargo.toml --features distributed-mpi
    if ($LASTEXITCODE -ne 0) { throw "python extension build failed" }
    & python -m pytest bindings/python/tests/test_distributed.py -q
    if ($LASTEXITCODE -ne 0) { throw "python distributed tests failed at one rank" }
    foreach ($n in $RankCounts) {
        Write-Host "`n== mpiexec -n $n pytest test_distributed.py =="
        & $mpiexec -n $n -env PRISM_DIST_MIN_LOCAL_QUBITS 1 python -m pytest bindings/python/tests/test_distributed.py -q
        if ($LASTEXITCODE -ne 0) { throw "python distributed tests failed at $n ranks" }
    }
}

Write-Host "`nAll distributed MPI checks passed."

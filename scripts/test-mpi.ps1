# Build and run the distributed backend's multi-rank correctness check.
#
# Usage:   powershell -ExecutionPolicy Bypass -File scripts\test-mpi.ps1 [-Ranks 4]
#
# Sets up the MPI build environment, builds the lib tests and the mpiexec check
# binary, then launches it across N ranks. Rank 0 asserts the gathered result
# matches the single-node statevector reference and exits non-zero on mismatch.

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

# The check circuit is small, so relax the local-qubit floor to let it
# distribute across ranks on a single host.
foreach ($n in $RankCounts) {
    Write-Host "`n== mpiexec -n $n dist_mpi_check =="
    & $mpiexec -n $n -env PRISM_DIST_MIN_LOCAL_QUBITS 1 $exe
    if ($LASTEXITCODE -ne 0) { throw "multi-rank check failed at -n $n" }
}

Write-Host "`nAll distributed MPI checks passed."

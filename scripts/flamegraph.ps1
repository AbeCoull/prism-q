# Generate a flamegraph for a specific Criterion benchmark.
#
# Prerequisites:
#   cargo install flamegraph
#   Windows: Requires Administrator (ETW tracing needs elevated privileges)
#
# Usage:
#   .\scripts\flamegraph.ps1 "qft_textbook/16"
#   .\scripts\flamegraph.ps1 "hea_l5/20" -BenchFile circuits
#   $env:PROFILE_TIME=30; .\scripts\flamegraph.ps1 "qft_textbook/22"
#
# Output: bench_results\flamegraph-<sanitized-filter>.svg
#
# The bench profile (Cargo.toml) already has debug=true and strip="none",
# so symbol names are preserved in the flamegraph.

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Filter,

    [string]$BenchFile = "circuits"
)

$ErrorActionPreference = "Stop"

# --- Admin check: ETW tracing requires elevation on Windows ---
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Flamegraph on Windows requires Administrator (ETW tracing)." -ForegroundColor Yellow
    Write-Host "Re-launching elevated..." -ForegroundColor Yellow
    Write-Host ""

    $argList = "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`" `"$Filter`""
    if ($BenchFile -ne "circuits") {
        $argList += " -BenchFile `"$BenchFile`""
    }

    Start-Process powershell -Verb RunAs -ArgumentList $argList
    exit 0
}

$ProfileTime = if ($env:PROFILE_TIME) { $env:PROFILE_TIME } else { "10" }
$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$OutputDir = Join-Path $ProjectDir "bench_results"

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# Sanitize filter for filename
$SafeName = $Filter -replace '[/ ]', '-'
$OutputFile = Join-Path $OutputDir "flamegraph-${SafeName}.svg"

Write-Host "=== PRISM-Q Flamegraph ===" -ForegroundColor Cyan
Write-Host "Benchmark: $BenchFile"
Write-Host "Filter:    $Filter"
Write-Host "Duration:  ${ProfileTime}s"
Write-Host "Output:    $OutputFile"
Write-Host ""

Push-Location $ProjectDir
try {
    cargo flamegraph `
        --bench $BenchFile `
        --features parallel `
        -o $OutputFile `
        -- --bench --profile-time $ProfileTime $Filter

    if ($LASTEXITCODE -ne 0) {
        throw "cargo flamegraph failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "=== Flamegraph saved ===" -ForegroundColor Green
    Write-Host "Open in a browser: $OutputFile"
} finally {
    Pop-Location
}

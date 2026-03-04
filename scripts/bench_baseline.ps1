# Save a benchmark baseline for future regression comparison.
#
# Usage:
#   .\scripts\bench_baseline.ps1              # full suite
#   .\scripts\bench_baseline.ps1 -Quick       # smoke subset only
#
# Output: baseline stored in target\criterion\

param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$BaselineName = if ($env:BASELINE_NAME) { $env:BASELINE_NAME } else { "prism_baseline" }
$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)

Push-Location $ProjectDir
try {
    Write-Host "=== PRISM-Q Benchmark Baseline ===" -ForegroundColor Cyan
    Write-Host "Baseline name: $BaselineName"
    Write-Host "Date: $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    Write-Host ""

    if ($Quick) {
        Write-Host "Running smoke subset..."
        cargo bench --bench bench_driver -- `
            --save-baseline $BaselineName `
            --warm-up-time 2 `
            --measurement-time 5 `
            "single_qubit_gates/h_gate" `
            "two_qubit_gates/cx_gate" `
            "e2e_qasm"
    } else {
        Write-Host "Running full benchmark suite..."
        cargo bench -- --save-baseline $BaselineName
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark run failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "=== Baseline saved as '$BaselineName' ===" -ForegroundColor Green
    Write-Host "Location: target\criterion\"
    Write-Host ""
    Write-Host "To compare against this baseline later:"
    Write-Host "  .\scripts\bench_compare.ps1"
} finally {
    Pop-Location
}

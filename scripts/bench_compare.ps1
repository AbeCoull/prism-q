# Compare current benchmark performance against a saved baseline.
#
# Usage:
#   .\scripts\bench_compare.ps1                                   # default 5%
#   $env:REGRESSION_THRESHOLD=10; .\scripts\bench_compare.ps1     # custom
#   .\scripts\bench_compare.ps1 -Quick                            # smoke subset
#
# Exit code:
#   0 — no regressions above threshold
#   1 — regressions detected

param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$BaselineName = if ($env:BASELINE_NAME) { $env:BASELINE_NAME } else { "prism_baseline" }
$Threshold = if ($env:REGRESSION_THRESHOLD) { $env:REGRESSION_THRESHOLD } else { "5" }
$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$OutputFile = Join-Path $ProjectDir "target\bench-compare-output.txt"

Push-Location $ProjectDir
try {
    Write-Host "=== PRISM-Q Benchmark Comparison ===" -ForegroundColor Cyan
    Write-Host "Baseline: $BaselineName"
    Write-Host "Regression threshold: ${Threshold}%"
    Write-Host "Date: $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))"
    Write-Host ""

    # Ensure target directory exists.
    New-Item -ItemType Directory -Path (Split-Path $OutputFile) -Force | Out-Null

    if ($Quick) {
        Write-Host "Running smoke subset..."
        cargo bench --bench bench_driver -- `
            --baseline $BaselineName `
            --warm-up-time 2 `
            --measurement-time 5 `
            "single_qubit_gates/h_gate" `
            "two_qubit_gates/cx_gate" `
            "e2e_qasm" `
            2>&1 | Tee-Object -FilePath $OutputFile
    } else {
        Write-Host "Running full benchmark suite..."
        cargo bench -- --baseline $BaselineName 2>&1 | Tee-Object -FilePath $OutputFile
    }

    Write-Host ""
    Write-Host "=== Regression Analysis ===" -ForegroundColor Cyan

    $Content = Get-Content $OutputFile -Raw
    $Regressed = $Content -match "(?i)regressed"

    if ($Regressed) {
        Write-Host "WARNING: Criterion reports regressions." -ForegroundColor Yellow
        Write-Host ""
        Select-String -Path $OutputFile -Pattern "(?i)regressed" | ForEach-Object { Write-Host $_.Line }
    } else {
        Write-Host "No regressions detected by Criterion." -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "Threshold: ${Threshold}%"
    Write-Host "Full output: $OutputFile"
    Write-Host ""

    if ($Regressed) {
        Write-Host "VERDICT: FAIL — regressions detected" -ForegroundColor Red
        Write-Host "Review the output above. If acceptable, re-run baseline:"
        Write-Host "  .\scripts\bench_baseline.ps1"
        exit 1
    } else {
        Write-Host "VERDICT: PASS" -ForegroundColor Green
        exit 0
    }
} finally {
    Pop-Location
}

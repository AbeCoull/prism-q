# Benchmark analysis tool for PRISM-Q.
#
# Reads Criterion JSON data from target/criterion/, saves baselines,
# compares runs, and produces PR-ready tables with regression detection.
#
# Usage:
#   .\scripts\bench_check.ps1 save                          # save baseline (auto-dated)
#   .\scripts\bench_check.ps1 save -Name "pre-fusion"       # save with custom name
#   .\scripts\bench_check.ps1 compare                       # compare against latest baseline
#   .\scripts\bench_check.ps1 compare -Baseline "pre-fusion"
#   .\scripts\bench_check.ps1 compare -Filter "qft"         # only QFT benchmarks
#   .\scripts\bench_check.ps1 table -Baseline "pre-fusion"  # PR-ready markdown
#   .\scripts\bench_check.ps1 list                          # list saved baselines
#
# Environment variables:
#   REGRESSION_THRESHOLD  Override default 5% threshold
#
# Prerequisites:
#   Run `cargo bench` first -- this tool reads stored results, it doesn't run benchmarks.

param(
    [Parameter(Position = 0)]
    [ValidateSet("save", "compare", "list", "table")]
    [string]$Command = "compare",

    [string]$Name,
    [string]$Baseline,

    [Nullable[double]]$Threshold,

    [string]$Filter,

    [ValidateSet("new", "base")]
    [string]$Source = "new"
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$CriterionDir = Join-Path $ProjectDir "target\criterion"
$BaselineDir = Join-Path $ProjectDir "bench_results\baselines"

if ($null -eq $Threshold) {
    $Threshold = if ($env:REGRESSION_THRESHOLD) { [double]$env:REGRESSION_THRESHOLD } else { 5.0 }
}

function Format-Duration([double]$ns) {
    if ($ns -ge 1e9) { return "{0:F3} s" -f ($ns / 1e9) }
    if ($ns -ge 1e6) { return "{0:F2} ms" -f ($ns / 1e6) }
    if ($ns -ge 1e3) { return "{0:F2} us" -f ($ns / 1e3) }
    return "{0:F1} ns" -f $ns
}

function Get-CriterionEstimates {
    param([string]$SubDir = "new")

    if (-not (Test-Path $CriterionDir)) {
        throw "No Criterion data at $CriterionDir -- run 'cargo bench' first."
    }

    $results = @{}

    $estimateFiles = Get-ChildItem -Path $CriterionDir -Recurse -Filter "estimates.json" |
        Where-Object { $_.Directory.Name -eq $SubDir }

    foreach ($file in $estimateFiles) {
        $benchDir = $file.Directory.FullName
        $benchmarkJson = Join-Path $benchDir "benchmark.json"

        if (-not (Test-Path $benchmarkJson)) { continue }

        $meta = Get-Content $benchmarkJson -Raw | ConvertFrom-Json
        $est = Get-Content $file.FullName -Raw | ConvertFrom-Json

        $results[$meta.full_id] = @{
            mean_ns   = $est.mean.point_estimate
            ci_lo_ns  = $est.mean.confidence_interval.lower_bound
            ci_hi_ns  = $est.mean.confidence_interval.upper_bound
            stddev_ns = $est.std_dev.point_estimate
        }
    }

    return $results
}

function Save-BenchBaseline {
    param([string]$BaselineName)

    if (-not $BaselineName) {
        $BaselineName = (Get-Date).ToString("yyyy-MM-dd_HHmmss")
    }

    $estimates = Get-CriterionEstimates -SubDir $Source
    if ($estimates.Count -eq 0) {
        throw "No benchmark estimates found in '$Source'. Run 'cargo bench' first."
    }

    New-Item -ItemType Directory -Path $BaselineDir -Force | Out-Null

    $rustVer = ""
    try { $rustVer = (& rustc --version 2>$null).Trim() } catch {}

    $cpuName = ""
    try { $cpuName = ((Get-CimInstance Win32_Processor).Name -replace '\s+', ' ').Trim() } catch {}

    $payload = [ordered]@{
        name         = $BaselineName
        date         = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        rust_version = $rustVer
        cpu          = $cpuName
        source       = $Source
        count        = $estimates.Count
        benchmarks   = [ordered]@{}
    }

    foreach ($key in ($estimates.Keys | Sort-Object)) {
        $payload.benchmarks[$key] = $estimates[$key]
    }

    $outFile = Join-Path $BaselineDir "$BaselineName.json"
    $payload | ConvertTo-Json -Depth 6 | Set-Content $outFile -Encoding UTF8

    Write-Host "=== Baseline saved ===" -ForegroundColor Green
    Write-Host "  Name:       $BaselineName"
    Write-Host "  Benchmarks: $($estimates.Count)"
    Write-Host "  File:       $outFile"
}

function Compare-BenchBaseline {
    param([string]$BaselineName, [bool]$Markdown)

    # Resolve baseline file
    if (-not $BaselineName) {
        if (-not (Test-Path $BaselineDir)) {
            throw "No baselines directory. Run '.\scripts\bench_check.ps1 save' first."
        }
        $latest = Get-ChildItem -Path $BaselineDir -Filter "*.json" |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $latest) {
            throw "No baselines found. Run '.\scripts\bench_check.ps1 save' first."
        }
        $BaselineName = $latest.BaseName
    }

    $baseFile = Join-Path $BaselineDir "$BaselineName.json"
    if (-not (Test-Path $baseFile)) {
        throw "Baseline '$BaselineName' not found at $baseFile"
    }

    $baseData = Get-Content $baseFile -Raw | ConvertFrom-Json
    $current = Get-CriterionEstimates -SubDir $Source

    if ($current.Count -eq 0) {
        throw "No current benchmark data in '$Source'. Run 'cargo bench' first."
    }

    # Collect all benchmark names present in both
    $baseNames = @($baseData.benchmarks.PSObject.Properties.Name)
    $allNames = ($baseNames + @($current.Keys)) | Sort-Object -Unique
    if ($Filter) {
        $allNames = $allNames | Where-Object { $_ -match $Filter }
    }

    # Build comparison rows
    $rows = @()
    foreach ($name in $allNames) {
        $baseBench = $baseData.benchmarks.$name
        $currBench = $current[$name]

        if (-not $baseBench -or -not $currBench) { continue }

        $bMean = $baseBench.mean_ns
        $cMean = $currBench.mean_ns
        $pctChange = (($cMean - $bMean) / $bMean) * 100.0

        $rows += [PSCustomObject]@{
            Name       = $name
            BeforeNs   = $bMean
            AfterNs    = $cMean
            PctChange  = $pctChange
        }
    }

    if ($rows.Count -eq 0) {
        Write-Host "No matching benchmarks found in both baseline and current data."
        return
    }

    $regressions = @($rows | Where-Object { $_.PctChange -gt $Threshold })
    $improvements = @($rows | Where-Object { $_.PctChange -lt (-$Threshold) })

    if ($Markdown) {
        Write-Output ''
        Write-Output '| Benchmark | Before | After | Change |'
        Write-Output '|-----------|--------|-------|--------|'
        foreach ($r in $rows) {
            $flag = ''
            if ($r.PctChange -gt $Threshold) { $flag = ' :x:' }
            elseif ($r.PctChange -lt (-$Threshold)) { $flag = ' :white_check_mark:' }
            $sign = if ($r.PctChange -ge 0) { '+' } else { '' }
            $before = Format-Duration $r.BeforeNs
            $after = Format-Duration $r.AfterNs
            $pct = '{0}{1:F1}%' -f $sign, $r.PctChange
            Write-Output ('| `{0}` | {1} | {2} | {3}{4} |' -f $r.Name, $before, $after, $pct, $flag)
        }
        Write-Output ''
        $verdict = if ($regressions.Count -gt 0) { 'FAIL' } else { 'PASS' }
        Write-Output ('**Regression verdict**: {0} (threshold: {1}%)' -f $verdict, $Threshold)
        return
    }

    # Console output
    Write-Host "=== PRISM-Q Benchmark Comparison ===" -ForegroundColor Cyan
    Write-Host "  Baseline:  $BaselineName ($($baseData.date))"
    if ($baseData.cpu) { Write-Host "  CPU:       $($baseData.cpu)" }
    Write-Host "  Threshold: ${Threshold}%"
    Write-Host ""

    $nameWidth = ($rows | ForEach-Object { $_.Name.Length } | Measure-Object -Maximum).Maximum
    $nameWidth = [Math]::Max($nameWidth, 9)
    $nameWidth = [Math]::Min($nameWidth, 50)

    $fmt = "{0,-$nameWidth}  {1,12}  {2,12}  {3,9}"
    Write-Host ($fmt -f "Benchmark", "Before", "After", "Change")
    Write-Host ("-" * ($nameWidth + 40))

    foreach ($r in $rows) {
        $sign = if ($r.PctChange -ge 0) { "+" } else { "" }
        $changeStr = "{0}{1:F1}%" -f $sign, $r.PctChange

        $color = "Gray"
        if ($r.PctChange -gt $Threshold) { $color = "Red" }
        elseif ($r.PctChange -lt (-$Threshold)) { $color = "Green" }

        $truncName = $r.Name
        if ($truncName.Length -gt $nameWidth) {
            $truncName = "..." + $truncName.Substring($truncName.Length - $nameWidth + 3)
        }

        $line = $fmt -f $truncName, (Format-Duration $r.BeforeNs), (Format-Duration $r.AfterNs), $changeStr
        Write-Host $line -ForegroundColor $color
    }

    Write-Host ""
    Write-Host ("  {0} benchmarks | {1} regressions | {2} improvements | {3} unchanged" -f `
        $rows.Count, $regressions.Count, $improvements.Count, `
        ($rows.Count - $regressions.Count - $improvements.Count))

    if ($regressions.Count -gt 0) {
        Write-Host ""
        Write-Host "VERDICT: FAIL" -ForegroundColor Red
        Write-Host "  $($regressions.Count) benchmark(s) regressed beyond ${Threshold}%:"
        foreach ($r in $regressions) {
            $sign = if ($r.PctChange -ge 0) { "+" } else { "" }
            Write-Host ("    {0} ({1}{2:F1}%)" -f $r.Name, $sign, $r.PctChange) -ForegroundColor Red
        }
        exit 1
    }
    else {
        Write-Host ""
        Write-Host "VERDICT: PASS" -ForegroundColor Green
        exit 0
    }
}

# --- Main dispatch ---

switch ($Command) {
    "save" {
        Save-BenchBaseline -BaselineName $Name
    }
    "compare" {
        $bl = if ($Baseline) { $Baseline } elseif ($Name) { $Name } else { $null }
        Compare-BenchBaseline -BaselineName $bl -Markdown $false
    }
    "table" {
        $bl = if ($Baseline) { $Baseline } elseif ($Name) { $Name } else { $null }
        Compare-BenchBaseline -BaselineName $bl -Markdown $true
    }
    "list" {
        if (-not (Test-Path $BaselineDir)) {
            Write-Host "No baselines saved yet. Run '.\scripts\bench_check.ps1 save' first."
            exit 0
        }
        $files = Get-ChildItem -Path $BaselineDir -Filter "*.json" |
            Sort-Object LastWriteTime -Descending
        if ($files.Count -eq 0) {
            Write-Host "No baselines saved yet."
            exit 0
        }
        Write-Host "=== Saved Baselines ===" -ForegroundColor Cyan
        Write-Host ""
        foreach ($f in $files) {
            $data = Get-Content $f.FullName -Raw | ConvertFrom-Json
            $count = ($data.benchmarks.PSObject.Properties | Measure-Object).Count
            Write-Host ("  {0,-30}  {1}  ({2} benchmarks)" -f $f.BaseName, $data.date, $count)
        }
    }
}

<#
.SYNOPSIS
    Extracts a text-based hotspot report from a flamegraph SVG.

.DESCRIPTION
    Parses inferno-generated flamegraph SVGs to produce a sorted list of
    functions by sample count. Shows prism_q kernel functions by default,
    filtering out thread scaffolding and Rayon plumbing.

    When -Duration is provided (or defaults to 10s matching flamegraph.ps1),
    estimated wall-clock times are computed from sample proportions.

    When -PerIter is provided (ms per circuit iteration, from Criterion or
    profile_circuit), a per-call column shows each function's share of a
    single circuit execution.

.EXAMPLE
    .\scripts\flamegraph_report.ps1
    .\scripts\flamegraph_report.ps1 -Path bench_results\flamegraph-qft_textbook-16.svg
    .\scripts\flamegraph_report.ps1 -Top 10
    .\scripts\flamegraph_report.ps1 -Duration 30          # profile ran for 30 seconds
    .\scripts\flamegraph_report.ps1 -PerIter 4.54         # ms per iteration (from Criterion/profile_circuit)
    .\scripts\flamegraph_report.ps1 -All                   # include rayon/std scaffolding
    .\scripts\flamegraph_report.ps1 -Raw                   # show full unsimplified names
#>

param(
    [string]$Path,
    [int]$Top = 30,
    [double]$Duration = 10.0,
    [double]$PerIter = 0,
    [switch]$All,
    [switch]$Raw
)

$ErrorActionPreference = "Stop"
$BenchResultsDir = Join-Path $PSScriptRoot "..\bench_results"

if (-not $Path) {
    $svgs = Get-ChildItem -Path $BenchResultsDir -Filter "flamegraph-*.svg" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    if (-not $svgs) {
        Write-Error "No flamegraph SVGs found in $BenchResultsDir. Run flamegraph.ps1 first."
        exit 1
    }
    $Path = $svgs[0].FullName
    Write-Host "  Using: $($svgs[0].Name)" -ForegroundColor DarkGray
}

if (-not (Test-Path $Path)) {
    Write-Error "File not found: $Path"
    exit 1
}

$content = Get-Content $Path -Raw

# Parse <g> blocks: extract title text and rect y-coordinate.
# Format: <g><title>name (N samples, X%)</title><rect ... y="Y" .../>
# The y-coordinate identifies stack depth (lower y = deeper = closer to leaf).
$gPattern = '<g><title>([^<]+?) \((\d[\d,]*) samples?, ([\d.]+)%\)</title><rect[^>]*?\by="(\d+)"'
$gMatches = [regex]::Matches($content, $gPattern)

if ($gMatches.Count -eq 0) {
    Write-Error "No sample data found in SVG. Is this an inferno-generated flamegraph?"
    exit 1
}

function Format-EstTime {
    param([double]$Seconds)
    if ($Seconds -ge 1.0) { return "{0:F2} s" -f $Seconds }
    $ms = $Seconds * 1000.0
    if ($ms -ge 1.0) { return "{0:F1} ms" -f $ms }
    $us = $ms * 1000.0
    if ($us -ge 1.0) { return "{0:F1} us" -f $us }
    return "{0:F0} ns" -f ($us * 1000.0)
}

function Simplify-Name {
    param([string]$FullName)

    if ($FullName -match '`(.+)') {
        $name = $Matches[1]
    } else {
        $name = $FullName
    }

    $name = $name -replace '&lt;', '<' -replace '&gt;', '>' -replace '&amp;', '&'

    if ($name -match '^0x[0-9A-Fa-f]+$') {
        return $null
    }

    if ($name -match '^([^(]+?)\(') {
        $name = $Matches[1].TrimEnd()
    }

    $prev = ''
    while ($prev -ne $name) {
        $prev = $name
        $name = $name -replace '<[^<>]*>', ''
    }

    $name = $name -replace 'impl\$\d+::', ''
    $name = $name -replace 'enum2\$\{([^}]+)\}', '$1'
    $name = $name -replace 'enum2\$', ''

    $name = $name -replace '^prism_q::backend::statevector::kernels::', 'kernels::'
    $name = $name -replace '^prism_q::backend::statevector::StatevectorBackend::', 'sv::'
    $name = $name -replace '^prism_q::backend::statevector::', 'sv::'
    $name = $name -replace '^prism_q::backend::simd::', 'simd::'
    $name = $name -replace '^prism_q::circuit::fusion::', 'fusion::'
    $name = $name -replace '^prism_q::circuit::', 'circuit::'
    $name = $name -replace '^prism_q::sim::', 'sim::'
    $name = $name -replace '^prism_q::gates::', 'gates::'
    $name = $name -replace '^prism_q::', 'prism_q::'

    $name = $name -replace '^rayon_core::', 'rayon::'
    $name = $name -replace '^rayon::iter::plumbing::', 'rayon::plumbing::'
    $name = $name -replace '^rayon::iter::for_each::', 'rayon::for_each::'
    $name = $name -replace '^rayon::iter::ParallelIterator::', 'rayon::par_iter::'
    $name = $name -replace '^rayon::registry::Registry::', 'rayon::Registry::'
    $name = $name -replace '^rayon::registry::', 'rayon::registry::'

    $name = $name -replace '^core::ops::function::impls::', 'core::call::'
    $name = $name -replace '^core::iter::traits::iterator::Iterator::', 'core::iter::'
    $name = $name -replace '^core::slice::', 'slice::'
    $name = $name -replace '^std::thread::local::', 'tls::'
    $name = $name -replace '^alloc::', 'alloc::'
    $name = $name -replace '^num_complex::', 'num_complex::'
    $name = $name -replace '^num_traits::', 'num_traits::'

    $name = $name.TrimEnd(':', ' ')
    if ($name.Length -gt 80) {
        $name = $name.Substring(0, 77) + '...'
    }
    return $name
}

$prismPrefixes = @(
    'kernels::', 'sv::', 'simd::', 'fusion::', 'circuit::',
    'sim::', 'gates::', 'prism_q::', 'num_complex::', 'num_traits::'
)

function Is-PrismFunction {
    param([string]$Name)
    foreach ($pfx in $prismPrefixes) {
        if ($Name.StartsWith($pfx)) { return $true }
    }
    return $false
}

# Build entries: for each unique function, track max samples (cumulative)
# and also track the frame at the lowest y-value (closest to leaf)
$entries = @{}
$totalSamples = 0

foreach ($m in $gMatches) {
    $rawName = $m.Groups[1].Value
    $samples = [int]($m.Groups[2].Value -replace ',', '')
    $pct = [double]$m.Groups[3].Value
    $yVal = [double]$m.Groups[4].Value

    if ($rawName -eq 'all') {
        $totalSamples = $samples
        continue
    }

    if ($Raw) {
        if ($rawName -match '`0x[0-9A-Fa-f]+$') { continue }
        $name = $rawName
    } else {
        $name = Simplify-Name $rawName
        if (-not $name) { continue }
    }

    if (-not $entries.ContainsKey($name)) {
        $entries[$name] = @{
            MaxSamples = $samples
            MaxPct = $pct
            MinY = $yVal
            MinYSamples = $samples
            MinYPct = $pct
        }
    } else {
        $e = $entries[$name]
        if ($samples -gt $e.MaxSamples) {
            $e.MaxSamples = $samples
            $e.MaxPct = $pct
        }
        if ($yVal -lt $e.MinY) {
            $e.MinY = $yVal
            $e.MinYSamples = $samples
            $e.MinYPct = $pct
        }
    }
}

if ($totalSamples -eq 0) {
    $totalSamples = ($entries.Values | Measure-Object -Property MaxSamples -Maximum).Maximum
}

# Apply filters unless -All
if (-not $All -and -not $Raw) {
    $filtered = @{}
    foreach ($kv in $entries.GetEnumerator()) {
        if (Is-PrismFunction $kv.Key) {
            $filtered[$kv.Key] = $kv.Value
        }
    }
    $entries = $filtered
}

# Sort by the leaf-level samples (MinYSamples) -- this approximates self-time
$sorted = $entries.GetEnumerator() |
    Sort-Object { $_.Value.MinYSamples } -Descending |
    Select-Object -First $Top

$showPerCall = $PerIter -gt 0

# Print report
Write-Host ""
Write-Host "=== Flamegraph Hotspot Report ===" -ForegroundColor Cyan
Write-Host "  Source:    $(Split-Path $Path -Leaf)"
Write-Host "  Frames:   $($entries.Count) unique functions (after filtering)"
Write-Host "  Samples:  $($totalSamples.ToString('N0')) total"
Write-Host "  Duration: ${Duration}s (estimated times = sample% * duration)"
if ($showPerCall) {
    Write-Host "  Per-iter: ${PerIter} ms (per-call = function share of one iteration)"
}
if (-not $All -and -not $Raw) {
    Write-Host '  Filter:   prism_q functions only (use -All to show everything)' -ForegroundColor DarkGray
}
Write-Host ""

if ($showPerCall) {
    Write-Host ("  {0,6}  {1,10}  {2,10}  {3,8}  {4}" -f 'Self%', 'Est. time', 'Per-call', 'Samples', 'Function')
    Write-Host ("  {0}  {1}  {2}  {3}  {4}" -f ('-' * 6), ('-' * 10), ('-' * 10), ('-' * 8), ('-' * 44))
} else {
    Write-Host ("  {0,6}  {1,10}  {2,8}  {3}" -f 'Self%', 'Est. time', 'Samples', 'Function')
    Write-Host ("  {0}  {1}  {2}  {3}" -f ('-' * 6), ('-' * 10), ('-' * 8), ('-' * 50))
}

foreach ($entry in $sorted) {
    $pct = $entry.Value.MinYPct
    $samples = $entry.Value.MinYSamples
    $name = $entry.Key
    $estTime = Format-EstTime ($pct / 100.0 * $Duration)

    $color = 'White'
    if ($pct -gt 20) { $color = 'Red' }
    elseif ($pct -gt 10) { $color = 'Yellow' }
    elseif ($pct -gt 5) { $color = 'DarkYellow' }

    if ($showPerCall) {
        $perCall = Format-EstTime ($pct / 100.0 * $PerIter / 1000.0)
        Write-Host ("  {0,5:F1}%  {1,10}  {2,10}  {3,8:N0}  {4}" -f $pct, $estTime, $perCall, $samples, $name) -ForegroundColor $color
    } else {
        Write-Host ("  {0,5:F1}%  {1,10}  {2,8:N0}  {3}" -f $pct, $estTime, $samples, $name) -ForegroundColor $color
    }
}

if (-not $showPerCall) {
    Write-Host ""
    Write-Host '  Tip: add -PerIter <ms> for per-call times (get value from profile_circuit TOTAL or Criterion)' -ForegroundColor DarkGray
}

Write-Host ""

# Run the GPU golden suites against a real device.
#
# Usage:   powershell -ExecutionPolicy Bypass -File scripts\test-gpu.ps1
#
# tests/golden_gpu.rs and tests/golden_gpu_density_matrix.rs compare device against
# host for every gate, channel, and fusion shape, but they skip silently when no
# CUDA device opens, so a green run on a host without a card means "not tested".
# This script sets PRISM_REQUIRE_GPU so a missing or unusable device fails the run
# instead. CI never opens a device (the gpu-check job runs the device-free kernel
# name registry test only), so run this on a host with a card before merging a
# change to src/gpu/, the gate set, the fusion pipeline, or a kernel table shape.
# Any of those can break the device path while every CPU gate stays green.

$ErrorActionPreference = 'Stop'
$clock = [System.Diagnostics.Stopwatch]::StartNew()

$env:PRISM_REQUIRE_GPU = '1'
$suites = @('--test', 'golden_gpu', '--test', 'golden_gpu_density_matrix')

Write-Host "`n== Building GPU golden suites (parallel gpu) =="
cargo nextest run --features "parallel gpu" @suites --no-run
if ($LASTEXITCODE -ne 0) { throw "GPU golden suite build failed" }

Write-Host "`n== Running GPU golden suites (PRISM_REQUIRE_GPU=1) =="
cargo nextest run --features "parallel gpu" @suites
if ($LASTEXITCODE -ne 0) { throw "GPU golden suites failed" }

$clock.Stop()
Write-Host ("`nAll GPU golden checks passed in {0:N0} s (build plus run)." -f $clock.Elapsed.TotalSeconds)

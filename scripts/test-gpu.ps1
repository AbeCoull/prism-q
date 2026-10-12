# Run the GPU golden suites against a real device.
#
# Usage:   powershell -ExecutionPolicy Bypass -File scripts\test-gpu.ps1
#
# tests/golden_gpu.rs and tests/golden_gpu_density_matrix.rs compare device against
# host for every gate, channel, and fusion shape, and the lib's gpu_scaffold tests
# round-trip a state through the device. tests/gpu_pass_census.rs checks the launch,
# pass and drain counters behind bench-internal. All of them skip silently when no CUDA
# device opens, so a green run on a host without a card means "not tested".
# This script sets PRISM_REQUIRE_GPU so a missing or unusable device fails the run
# instead. CI never opens a device (the gpu-check job runs the device-free kernel
# name registry test only), so run this on a host with a card before merging a
# change to src/gpu/, the gate set, the fusion pipeline, or a kernel table shape.
# Any of those can break the device path while every CPU gate stays green.

$ErrorActionPreference = 'Stop'
$clock = [System.Diagnostics.Stopwatch]::StartNew()

$env:PRISM_REQUIRE_GPU = '1'
$features = 'parallel gpu bench-internal'
$suites = @('--lib', '--test', 'golden_gpu', '--test', 'golden_gpu_density_matrix', '--test', 'gpu_pass_census')
$filter = @('-E', 'not kind(lib) or test(/gpu_scaffold::/)')

Write-Host "`n== Building GPU golden suites ($features) =="
cargo nextest run --features $features @suites --no-run
if ($LASTEXITCODE -ne 0) { throw "GPU golden suite build failed" }

Write-Host "`n== Running GPU golden suites (PRISM_REQUIRE_GPU=1) =="
cargo nextest run --features $features @suites @filter
if ($LASTEXITCODE -ne 0) { throw "GPU golden suites failed" }

$clock.Stop()
Write-Host ("`nAll GPU golden checks passed in {0:N0} s (build plus run)." -f $clock.Elapsed.TotalSeconds)

# Run the GPU golden suites against a real device.
#
# Usage:   powershell -ExecutionPolicy Bypass -File scripts\test-gpu.ps1
#
# tests/golden_gpu.rs and tests/golden_gpu_density_matrix.rs compare device against
# host for every gate, channel, and fusion shape. The lib's gpu_scaffold tests
# round-trip a state through the device, and the lib tests under src/gpu/ (gpu::)
# cover the device sampler walk, the kernels and the NVRTC image cache.
# tests/gpu_pass_census.rs checks the launch, pass and drain counters behind
# bench-internal, and the integration binary's device_plan_ test checks a device plan
# against its execution. The three golden and census binaries fail when no CUDA device
# opens unless PRISM_ALLOW_NO_GPU is set, which this script clears; the lib and
# integration tests skip unless PRISM_REQUIRE_GPU is set, which this script sets.
# CI never opens a device (the gpu-check job runs the device-free kernel
# name registry test only), so run this on a host with a card before merging a
# change to src/gpu/, the gate set, the fusion pipeline, or a kernel table shape.
# Any of those can break the device path while every CPU gate stays green.

$ErrorActionPreference = 'Stop'
$clock = [System.Diagnostics.Stopwatch]::StartNew()

$env:PRISM_REQUIRE_GPU = '1'
Remove-Item Env:PRISM_ALLOW_NO_GPU -ErrorAction SilentlyContinue
$features = 'parallel gpu bench-internal'
$suites = @('--lib', '--test', 'golden_gpu', '--test', 'golden_gpu_density_matrix', '--test', 'gpu_pass_census', '--test', 'integration')
$filter = @('-E', '(kind(test) & not binary(integration)) | (kind(lib) & (test(/gpu_scaffold::/) | test(/^gpu::/))) | (binary(integration) & test(/::device_plan_/))')

Write-Host "`n== Building GPU golden suites ($features) =="
cargo nextest run --features $features @suites --no-run
if ($LASTEXITCODE -ne 0) { throw "GPU golden suite build failed" }

Write-Host "`n== Running GPU golden suites (PRISM_REQUIRE_GPU=1) =="
cargo nextest run --features $features @suites @filter
if ($LASTEXITCODE -ne 0) { throw "GPU golden suites failed" }

$clock.Stop()
Write-Host ("`nAll GPU golden checks passed in {0:N0} s (build plus run)." -f $clock.Elapsed.TotalSeconds)

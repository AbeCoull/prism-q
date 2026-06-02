# Set up the environment for building and running the `distributed-mpi` feature.
#
# rsmpi (the `mpi` crate) needs three things that are not on PATH by default:
#   1. The MSVC developer environment (INCLUDE/LIB/Windows SDK) for bindgen and linking.
#   2. The MS-MPI SDK (MSMPI_INC / MSMPI_LIB64), installed via `winget install Microsoft.msmpisdk`.
#   3. libclang, for bindgen. VS 2022 bundles one under VC\Tools\Llvm.
#
# Dot-source this script before cargo:  . scripts\mpi-env.ps1
# Then:  cargo test --features distributed-mpi
# And to launch ranks:  mpiexec -n 4 target\debug\examples\dist_mpi_check.exe

$ErrorActionPreference = 'Stop'

function Find-VsInstall {
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $p = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath -nologo 2>$null
        if ($p) { return $p.Trim() }
    }
    foreach ($c in @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
    )) { if (Test-Path $c) { return $c } }
    throw "No Visual Studio with C++ tools found."
}

$vsRoot = Find-VsInstall
$vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under $vsRoot" }

# Import the MSVC developer environment (INCLUDE, LIB, WindowsSdkDir, PATH additions).
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
}

# MS-MPI SDK (machine-scope vars set by the installer).
$env:MSMPI_INC = [Environment]::GetEnvironmentVariable('MSMPI_INC', 'Machine')
$env:MSMPI_LIB64 = [Environment]::GetEnvironmentVariable('MSMPI_LIB64', 'Machine')
if (-not $env:MSMPI_INC -or -not (Test-Path (Join-Path $env:MSMPI_INC 'mpi.h'))) {
    throw "MS-MPI SDK not found. Install with: winget install --id Microsoft.msmpisdk"
}

# libclang for bindgen, bundled with VS.
$llvm = Join-Path $vsRoot "VC\Tools\Llvm\x64\bin"
if (Test-Path (Join-Path $llvm 'libclang.dll')) {
    $env:LIBCLANG_PATH = $llvm
}

Write-Host "MPI build environment ready:"
Write-Host "  VS:           $vsRoot"
Write-Host "  MSMPI_INC:    $env:MSMPI_INC"
Write-Host "  MSMPI_LIB64:  $env:MSMPI_LIB64"
Write-Host "  LIBCLANG_PATH:$env:LIBCLANG_PATH"

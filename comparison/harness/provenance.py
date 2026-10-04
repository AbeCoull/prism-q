"""Host, toolchain, and simulator version capture.

Every published number carries this block. A reader comparing their own run
against a committed reference needs it to tell a hardware difference apart from
a discrepancy in the claim.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRACKED_PACKAGES = ("qiskit", "qiskit-aer", "qsimcirq", "cirq-core", "numpy", "psutil")


def _run(cmd: list[str], cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _cpu_model() -> str:
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if model:
            return model
    elif sys.platform == "win32":
        model = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if model:
            return model
        model = os.environ.get("PROCESSOR_IDENTIFIER")
        if model:
            return model
    return platform.processor() or "unknown"


def _host() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu_model": _cpu_model(),
        "arch": platform.machine(),
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "ram_gb": None,
        "os": f"{platform.system()} {platform.release()}",
    }
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    return info


def _rustc_version(toolchain: str | None = None) -> str | None:
    cmd = ["rustc", "-vV"] if toolchain is None else ["rustc", f"+{toolchain}", "-vV"]
    out = _run(cmd)
    if not out:
        return None
    match = re.search(r"^release:\s*(.+)$", out, re.MULTILINE)
    return match.group(1).strip() if match else out.splitlines()[0]


def _cxx_version() -> str | None:
    """Compiler id and version from the CMake build tree of the QuEST comparator."""
    import glob

    pattern = os.path.join(REPO_ROOT, "comparison", "quest", "build", "CMakeFiles", "*", "CMakeCXXCompiler.cmake")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        compiler = re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^"]+)"\)', text)
        version = re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]+)"\)', text)
        if version:
            return f"{compiler.group(1) if compiler else 'unknown'} {version.group(1)}"
    return None


def _toolchain(features: str) -> dict[str, Any]:
    dirty = _run(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    return {
        "rustc": _rustc_version(),
        "rustc_peers": _rustc_version("nightly"),
        "cxx_compiler": _cxx_version(),
        "cargo_profile": "release",
        "cargo_features": features,
        "rustflags": os.environ.get("RUSTFLAGS", ""),
        "git_sha": _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "git_dirty": bool(dirty) if dirty is not None else None,
    }


def _packages() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect(features: str, threads: int, simulators: dict[str, str | None]) -> dict[str, Any]:
    return {
        "host": _host(),
        "toolchain": _toolchain(features),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "packages": _packages(),
        },
        "simulators": simulators,
        "threads": {
            "requested": threads,
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def quiet_machine_warnings() -> list[str]:
    """Load checks that catch the most common source of bad numbers."""
    warnings: list[str] = []
    try:
        import psutil

        load = psutil.cpu_percent(interval=1.0)
        if load > 25.0:
            warnings.append(f"CPU was {load:.0f}% busy before the run started; timings will be noisy")
        for proc in psutil.process_iter(["name"]):
            name = (proc.info.get("name") or "").lower()
            if name.startswith("cargo") or "criterion" in name:
                warnings.append(f"another cargo process is running ({name})")
                break
    except ImportError:
        warnings.append("psutil not installed; skipped quiet-machine checks")
    return warnings

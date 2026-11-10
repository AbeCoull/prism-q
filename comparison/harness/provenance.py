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

TRACKED_PACKAGES = (
    "qiskit",
    "qiskit-aer",
    "qsimcirq",
    "cirq-core",
    "numpy",
    "psutil",
    "scipy",
    "stim",
    "quimb",
    "cotengra",
    "cotengrust",
    "autoray",
    "numba",
    "llvmlite",
    "cytoolz",
    "kahypar",
)


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


def contraction_optimizer() -> dict[str, Any] | None:
    """The cotengra `auto-hq` settings quimb's tensor network runs with on this host, kahypar included."""
    import importlib.util
    import warnings

    if importlib.util.find_spec("cotengra") is None:
        return None
    from cotengra.hyperoptimizers.hyper import get_default_hq_methods, get_default_optlib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        methods = list(get_default_hq_methods())
    return {
        "preset": "auto-hq, a fresh optimizer per observable with cache=False",
        "hyper_methods": methods,
        "optlib": get_default_optlib(),
        "kahypar_available": importlib.util.find_spec("kahypar") is not None,
        "cotengrust_available": importlib.util.find_spec("cotengrust") is not None,
    }


def stim_build() -> dict[str, Any] | None:
    """Which of Stim's extension modules this interpreter loads; `_stim_avx2` uses 256-bit SIMD words."""
    import importlib.util

    if importlib.util.find_spec("stim") is None:
        return None
    import stim  # noqa: F401

    loaded = sorted(name for name in sys.modules if name.startswith("stim._stim_"))
    return {"version": metadata.version("stim"), "module": loaded[0] if loaded else None}


def collect(
    features: str, threads: int, simulators: dict[str, str | None], child_env: dict[str, str]
) -> dict[str, Any]:
    """Provenance block; `child_env` is the thread environment a child at the headline thread count gets."""
    return {
        "host": _host(),
        "toolchain": _toolchain(features),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "packages": _packages(),
            "contraction_optimizer": contraction_optimizer(),
            "stim_build": stim_build(),
        },
        "simulators": simulators,
        "threads": {
            "requested": threads,
            "rayon_num_threads": child_env.get("RAYON_NUM_THREADS"),
            "omp_num_threads": child_env.get("OMP_NUM_THREADS"),
        },
    }


ENV_KEYS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "RUSTFLAGS",
    "PRISM_MAX_SV_QUBITS",
    "PRISM_AVX512",
    "PRISM_NO_QFT_BLOCK",
    "OMP_PROC_BIND",
    "KMP_AFFINITY",
)


def host_details() -> dict[str, Any]:
    """CPU features, affinity, clock, power plan, GPU, NumPy BLAS and the relevant environment."""
    info: dict[str, Any] = {}
    try:
        from numpy._core._multiarray_umath import __cpu_features__

        wanted = ("SSE42", "AVX", "F16C", "FMA3", "AVX2", "AVX512F", "AVX512BW", "AVX512VL", "ASIMD", "SVE")
        info["cpu_flags"] = sorted(k for k in wanted if __cpu_features__.get(k))
    except ImportError:
        info["cpu_flags"] = None
    try:
        import psutil

        proc = psutil.Process()
        info["cpu_affinity"] = proc.cpu_affinity() if hasattr(proc, "cpu_affinity") else None
        freq = psutil.cpu_freq()
        info["cpu_freq_mhz"] = {"current": freq.current, "max": freq.max} if freq else None
    except ImportError:
        pass
    if sys.platform == "win32":
        info["power_plan"] = _run(["powercfg", "/getactivescheme"])
    gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    info["gpu"] = gpu.splitlines() if gpu else None
    try:
        import numpy as np

        config = np.show_config(mode="dicts")
        blas = config.get("Build Dependencies", {}).get("blas", {})
        info["numpy_blas"] = f"{blas.get('name')} {blas.get('version')}"
    except Exception:  # noqa: BLE001
        info["numpy_blas"] = None
    info["environment"] = {k: os.environ.get(k) for k in ENV_KEYS}
    info["timer"] = (
        "time.perf_counter in Python, std::time::Instant in Rust; QuEST, Spinoza, qip use their own monotonic clocks"
    )
    return info


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

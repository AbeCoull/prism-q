"""Simulator registry and process execution for the benchmark matrix.

Every measurement is one child process speaking the `exec` protocol (see
`tools/compare_runner.rs` and `worker.py`), so each observation carries its
own peak memory, startup cost and failure mode. The three native peers that
predate `exec` (QuEST, Spinoza, qip) are driven through their `probabilities`
and `time` commands, which gives the same timed region without the phase split.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

import psutil

from .workloads import Workload, gate_names

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXE = ".exe" if sys.platform == "win32" else ""
RUNNER = os.environ.get(
    "PRISMQ_CMP_RUNNER", os.path.join(REPO_ROOT, "target", "release", "examples", "compare_runner" + EXE)
)
PEERS = os.environ.get(
    "PRISMQ_CMP_PEERS", os.path.join(REPO_ROOT, "comparison", "peers", "target", "release", "peers" + EXE)
)
QUEST = os.environ.get(
    "PRISMQ_CMP_QUEST",
    os.path.join(
        REPO_ROOT, "comparison", "quest", "build", "Release" if sys.platform == "win32" else "", "quest_runner" + EXE
    ),
)
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")

UNITARY = {"h", "x", "z", "s", "sdg", "ry", "rz", "cx", "cz", "swap", "cp"}
CLIFFORD = {"h", "x", "z", "s", "sdg", "cx", "cz", "swap", "measure", "reset", "if"}
LEGACY_GATES = {"h", "cx", "swap", "ry", "rz", "cp"}
PROBS_BYTES_PER_AMP = 8
THREAD_ENV_KEYS = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS")
DENSE_MEMORY_FRACTION = 0.8
# Warm calls of these OpenMP peers ran faster with idle worker threads sleeping than spinning
# between calls, on both a 20- and a 24-qubit row; PRISM-Q's environment is left unchanged.
OMP_PASSIVE = (("OMP_WAIT_POLICY", "PASSIVE"),)
SPINOZA_H_FAULT = (
    "Spinoza's multithreaded H (2 or more threads, 15 or more qubits) returns a state 0.05 to 0.3 "
    "from the reference in total variation distance at git f900971"
)


@dataclass(frozen=True)
class Simulator:
    name: str
    label: str
    kind: str
    precision: str
    representation: str
    modes: frozenset[str]
    gates: frozenset[str] = frozenset(UNITARY | {"measure", "reset", "if"})
    dense: bool = False
    state_bytes_per_amp: int = 16
    backend: str | None = None
    single_threaded: bool = False
    env: tuple[tuple[str, str], ...] = ()
    gate_faults: tuple[tuple[str, str], ...] = ()


def _registry() -> dict[str, Simulator]:
    sims = [
        Simulator(
            "prismq",
            "PRISM-Q (auto dispatch)",
            "exec",
            "double",
            "dispatcher picks per circuit",
            frozenset({"probs", "counts", "expval", "stats"}),
            backend="auto",
        ),
        Simulator(
            "aer-statevector",
            "Qiskit Aer (statevector)",
            "python",
            "double",
            "dense statevector",
            frozenset({"probs", "counts", "expval"}),
            dense=True,
            env=OMP_PASSIVE,
        ),
        Simulator(
            "aer-statevector-f32",
            "Qiskit Aer (statevector, single)",
            "python",
            "single",
            "dense statevector",
            frozenset({"probs", "counts", "expval"}),
            dense=True,
            state_bytes_per_amp=8,
            env=OMP_PASSIVE,
        ),
        Simulator(
            "aer-automatic",
            "Qiskit Aer (automatic)",
            "python",
            "double",
            "Aer picks the method per circuit",
            frozenset({"probs", "counts"}),
            env=OMP_PASSIVE,
        ),
        Simulator(
            "aer-stabilizer",
            "Qiskit Aer (stabilizer)",
            "python",
            "exact (tableau)",
            "stabilizer tableau",
            frozenset({"counts"}),
            gates=frozenset(CLIFFORD),
            env=OMP_PASSIVE,
        ),
        Simulator(
            "aer-mps",
            "Qiskit Aer (matrix_product_state)",
            "python",
            "double",
            "matrix product state",
            frozenset({"probs", "counts", "expval"}),
            env=OMP_PASSIVE,
        ),
        Simulator(
            "stim",
            "Stim",
            "python",
            "exact (tableau)",
            "stabilizer tableau, compiled sampler",
            frozenset({"counts", "stats"}),
            gates=frozenset(CLIFFORD),
            single_threaded=True,
        ),
        Simulator(
            "quimb-tn",
            "quimb (tensor network, cotengra paths)",
            "python",
            "double",
            "tensor network contracted per observable over its reverse light cone",
            frozenset({"expval"}),
            gates=frozenset(UNITARY),
            env=OMP_PASSIVE,
        ),
        Simulator(
            "quimb-mps",
            "quimb (CircuitMPS)",
            "python",
            "double",
            "matrix product state",
            frozenset({"counts", "expval"}),
            gates=frozenset(UNITARY | {"measure"}),
            env=OMP_PASSIVE,
        ),
        Simulator(
            "qsim",
            "qsim (qsimcirq)",
            "python",
            "single",
            "dense statevector",
            frozenset({"probs", "counts"}),
            gates=frozenset(UNITARY | {"measure"}),
            dense=True,
            state_bytes_per_amp=8,
            env=OMP_PASSIVE,
        ),
        Simulator(
            "quest",
            "QuEST (OpenMP)",
            "legacy",
            "double",
            "dense statevector",
            frozenset({"probs"}),
            gates=frozenset(LEGACY_GATES),
            dense=True,
            env=OMP_PASSIVE,
        ),
        Simulator(
            "spinoza",
            "Spinoza",
            "legacy",
            "double",
            "dense statevector",
            frozenset({"probs"}),
            gates=frozenset(LEGACY_GATES),
            dense=True,
            gate_faults=(("h", SPINOZA_H_FAULT),),
        ),
        Simulator(
            "qip",
            "RustQIP (qip)",
            "legacy",
            "double",
            "dense statevector",
            frozenset({"probs"}),
            gates=frozenset(LEGACY_GATES),
            dense=True,
        ),
    ]
    return {s.name: s for s in sims}


REGISTRY = _registry()

PRISMQ_BACKENDS = {
    "statevector": ("dense statevector", True),
    "stabilizer": ("stabilizer tableau", False),
    "factored-stabilizer": ("independent tableaux", False),
    "sparse": ("sparse statevector", False),
    "product": ("product state", False),
    "factored": ("split dense sub-states", False),
    "tensor-network": ("tensor network", False),
}


def resolve(name: str) -> Simulator:
    """Registry entry for a configured name: `prismq:<backend>` and `aer-mps:<bond>` included."""
    base, _, option = name.partition(":")
    sim = REGISTRY.get(base)
    if sim is None:
        raise KeyError(f"unknown simulator `{name}`; known: {', '.join(REGISTRY)}")
    if not option:
        return sim
    if name == "prismq:no-qft-block":
        return Simulator(
            name,
            "PRISM-Q (auto dispatch, QFT recognition off)",
            "exec",
            "double",
            sim.representation,
            sim.modes,
            backend="auto",
            env=(("PRISM_NO_QFT_BLOCK", "1"),),
        )
    if base == "prismq":
        if option.startswith("mps"):
            representation, dense = f"matrix product state, bond cap {option.split('-')[-1]}", False
            backend = f"mps:{option.split('-')[-1]}"
        else:
            representation, dense = PRISMQ_BACKENDS[option]
            backend = option
        gates = CLIFFORD if backend in ("stabilizer", "factored-stabilizer") else sim.gates
        return Simulator(
            name,
            f"PRISM-Q ({backend})",
            "exec",
            "double",
            representation,
            sim.modes - {"stats"},
            gates=frozenset(gates),
            dense=dense,
            backend=backend,
        )
    if base in ("aer-mps", "quimb-mps"):
        return Simulator(
            name,
            sim.label.partition(" (")[0] + f" (MPS, bond cap {option})",
            "python",
            "double",
            f"matrix product state, bond cap {option}",
            sim.modes,
            gates=sim.gates,
            env=sim.env,
        )
    raise KeyError(f"`{base}` takes no option")


PRISMQ_MPS_SVD_EPSILON = 1e-12
TIMED_OUTPUT = {
    "probs": {
        "prismq": "probability vector",
        "aer": "Result holding the probability vector",
        "qsim": "probability vector",
        "quest": "probability vector",
        "spinoza": "probability vector",
        "qip": "probability vector",
    },
    "counts": {
        "prismq": "histogram",
        "aer": "Result holding a histogram",
        "qsim": "per-shot measurement array, which qsimcirq's `run` builds by expanding each shot into Python bools",
        "stim": "bit-packed per-shot array",
        "quimb": "per-shot sample list",
    },
    "expval": {
        "prismq": "expectation values",
        "aer": "Result holding expectation values",
        "quimb": "expectation values",
    },
    "stats": {"prismq": "bit-packed shot buffers", "stim": "bit-packed shot buffers"},
}


def timed_output(name: str, mode: str) -> str | None:
    """What a simulator's timed call returns in `mode`, by base family (`aer`, `quimb`, `prismq`, ...)."""
    return TIMED_OUTPUT.get(mode, {}).get(name.partition(":")[0].partition("-")[0])


MODE_WORDS = {
    "probs": "dense probability output",
    "counts": "shot sampling",
    "expval": "expectation values",
    "stats": "statistics-only sampling",
}


def applicable(sim: Simulator, workload: Workload) -> str | None:
    """None when the simulator can run the workload, otherwise the reason it cannot."""
    if workload.mode not in sim.modes:
        source = "adapter" if sim.kind == "legacy" else "simulator configuration"
        return f"{source} offers no {MODE_WORDS[workload.mode]}"
    used = gate_names(workload.text)
    faults = [f"{gate}: {why}" for gate, why in sim.gate_faults if gate in used]
    if faults:
        return "; ".join(faults)
    missing = used - sim.gates
    if missing:
        if sim.name in ("aer-stabilizer", "stim") or sim.backend in ("stabilizer", "factored-stabilizer"):
            return "not a Clifford circuit"
        return f"adapter carries no {', '.join(sorted(missing))}"
    if sim.dense:
        per_amp = sim.state_bytes_per_amp + (PROBS_BYTES_PER_AMP if workload.mode == "probs" else 0)
        need = per_amp * 2**workload.num_qubits
        have = psutil.virtual_memory().total * DENSE_MEMORY_FRACTION
        if need > have:
            return f"dense state of 2^{workload.num_qubits} amplitudes needs {need / 1024**3:.3g} GiB"
    executable = _executable(sim)
    if executable is not None and not os.path.exists(executable):
        return f"executable not built: {executable}"
    return None


def _executable(sim: Simulator) -> str | None:
    if sim.kind == "exec":
        return RUNNER
    if sim.kind == "legacy":
        return QUEST if sim.name == "quest" else PEERS
    return None


@dataclass
class ProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    wall_ms: float
    peak_rss_mb: float | None
    timed_out: bool


def _peak_bytes(proc: psutil.Process) -> float | None:
    try:
        info = proc.memory_info()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    peak = getattr(info, "peak_wset", None)
    return float(peak if peak is not None else info.rss)


MEMORY_METHOD = (
    "largest PeakWorkingSetSize over the child and its descendants, sampled every 20 ms and once "
    "more after exit; the counter only rises, so the last sample bounds the peak from below"
    if sys.platform == "win32"
    else "largest resident set size over the child and its descendants, sampled every 20 ms (a lower bound on the peak)"
)


def run_process(argv: list[str], stdin_text: str, env: dict[str, str], timeout_s: float) -> ProcessResult:
    """Run one child to completion, tracking the peak memory of its whole process tree.

    The tree matters on Windows, where a virtual environment's `python.exe` is a
    launcher that runs the interpreter as its own child.
    """
    start = time.perf_counter()
    child = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    tree: dict[int, psutil.Process] = {}
    try:
        tree[child.pid] = psutil.Process(child.pid)
    except psutil.NoSuchProcess:
        pass
    peaks: dict[int, float] = {}
    done = threading.Event()

    def sample() -> None:
        for proc in list(tree.values()):
            value = _peak_bytes(proc)
            if value:
                peaks[proc.pid] = max(peaks.get(proc.pid, 0.0), value)
            try:
                for grandchild in proc.children(recursive=True):
                    tree.setdefault(grandchild.pid, grandchild)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def poll() -> None:
        while not done.is_set():
            sample()
            done.wait(0.02)

    def kill_tree() -> None:
        for proc in list(tree.values()):
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        child.kill()

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    timed_out = finished = False
    try:
        try:
            stdout, stderr = child.communicate(stdin_text, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_tree()
            stdout, stderr = child.communicate()
        wall_ms = (time.perf_counter() - start) * 1000.0
        finished = True
    finally:
        if not finished:
            kill_tree()
        done.set()
        poller.join()
    sample()
    peak = max(peaks.values(), default=0.0)
    return ProcessResult(child.returncode, stdout, stderr, wall_ms, peak / 1024**2 if peak else None, timed_out)


_OOM_MARKERS = ("out of memory", "bad_alloc", "memory allocation", "MemoryError", "insufficient memory")
_CRASH_CODES = {-11, -6, 3221225477, 3221226505, 3221225725, 3221226356}


def classify(proc: ProcessResult) -> tuple[str, str | None]:
    if proc.timed_out:
        return "timeout", None
    if proc.returncode == 0:
        return "ok", None
    tail = (proc.stderr.strip().splitlines() or [""])[-1][:300]
    low = proc.stderr.lower()
    if proc.returncode in (3221225495,) or any(m.lower() in low for m in _OOM_MARKERS):
        return "oom", tail or f"exit {proc.returncode}"
    if proc.returncode in _CRASH_CODES or (proc.returncode or 0) < 0:
        return "crash", tail or f"exit {proc.returncode}"
    return "error", tail or f"exit {proc.returncode}"


def child_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    for key in THREAD_ENV_KEYS:
        env[key] = str(threads)
    env["PYTHONHASHSEED"] = "0"
    return env


def _last_json(stdout: str) -> dict[str, Any]:
    lines = stdout.strip().splitlines()
    if not lines:
        raise ValueError("no output")
    return json.loads(lines[-1])


def python_executable() -> str:
    return os.environ.get("PRISMQ_CMP_PYTHON", sys.executable)


def version(sim: Simulator) -> str | None:
    if sim.kind == "exec":
        argv = [RUNNER, "version"]
    elif sim.kind == "python":
        argv = [python_executable(), WORKER, sim.name.partition(":")[0], "version"]
    elif sim.name == "quest":
        argv = [QUEST, "version"]
    else:
        argv = [PEERS, sim.name, "version"]
    try:
        out = subprocess.run(argv, input="", capture_output=True, text=True, timeout=120, check=False)
        return _last_json(out.stdout).get("version")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def native_builds() -> dict[str, str | None]:
    """Compiler and flags each native peer binary reports it was built with."""
    builds: dict[str, str | None] = {}
    for key, argv in (("quest_runner", [QUEST, "version"]), ("peers", [PEERS, "qip", "version"])):
        try:
            out = subprocess.run(argv, input="", capture_output=True, text=True, timeout=120, check=False)
            builds[key] = _last_json(out.stdout).get("build")
        except (OSError, ValueError, subprocess.SubprocessError):
            builds[key] = None
    return builds


def execute(
    sim: Simulator,
    workload: Workload,
    threads: int,
    iterations: int,
    min_iterations: int,
    budget_s: float,
    timeout_s: float,
    out_path: str,
    keys_path: str | None = None,
) -> dict[str, Any]:
    """Run one measurement and return the observation fields it determines.

    `keys_path` names the observables or statistics file an `expval` or `stats` workload needs.
    """
    env = {**child_env(threads), **dict(sim.env)}
    budget = ["--budget-ms", str(budget_s * 1000.0), "--min-iterations", str(min_iterations)]
    if sim.kind in ("exec", "python"):
        common = ["--mode", workload.mode, "--shots", str(workload.shots), "--iterations", str(iterations)]
        common += ["--out", out_path, *budget]
        if keys_path:
            common += ["--keys", keys_path]
        if sim.kind == "exec":
            argv = [RUNNER, "exec", *common, "--backend", sim.backend or "auto"]
        else:
            argv = [python_executable(), WORKER, sim.name, "exec", *common, "--threads", str(threads)]
        proc = run_process(argv, workload.text, env, timeout_s)
        status, reason = classify(proc)
        obs: dict[str, Any] = {"status": status, "reason": reason, "peak_rss_mb": proc.peak_rss_mb}
        obs["phases"] = {"process_wall_ms": proc.wall_ms}
        if status != "ok":
            return obs
        try:
            data = _last_json(proc.stdout)
        except ValueError as exc:
            return {**obs, "status": "error", "reason": f"unparseable runner output: {exc}"}
        for key in ("startup_ms", "import_ms", "build_ms", "first_ms", "plan_median_us"):
            if key in data:
                obs["phases"][key] = data[key]
        obs["samples_ms"] = data["times_ms"]
        obs["internal_ms"] = data.get("internal_ms") or []
        if data.get("first_internal_ms") is not None:
            obs["first_internal_ms"] = data["first_internal_ms"]
        obs["version"] = data.get("version")
        for key in ("resolved_backend", "route", "engine", "exactness", "bond", "plan", "fusion_applied", "details"):
            if data.get(key) is not None:
                obs[key] = data[key]
        hashes = data.get("output_hashes") or []
        reseeds = data.get("reseeds_per_call", True)
        obs["reproducible"] = len(set(hashes)) == 1 if len(hashes) > 1 and reseeds else None
        return obs

    base = [QUEST] if sim.name == "quest" else [PEERS, sim.name]
    first = run_process([*base, "probabilities", out_path], workload.text, env, timeout_s)
    status, reason = classify(first)
    obs = {"status": status, "reason": reason, "peak_rss_mb": first.peak_rss_mb}
    obs["phases"] = {"process_wall_ms": first.wall_ms}
    if status != "ok":
        return obs
    # The cold process time bounds one call from above; the budget rule then
    # matches the one the exec runners apply internally.
    affordable = int(budget_s * 1000.0 // max(first.wall_ms, 1e-3))
    count = max(min_iterations, min(iterations, affordable))
    timed = run_process([*base, "time", str(count)], workload.text, env, timeout_s)
    status, reason = classify(timed)
    if status != "ok":
        return {**obs, "status": status, "reason": reason}
    try:
        data = _last_json(timed.stdout)
    except ValueError as exc:
        return {**obs, "status": "error", "reason": f"unparseable runner output: {exc}"}
    obs["samples_ms"] = data["times_ms"]
    obs["phases"]["timed_process_wall_ms"] = timed.wall_ms
    obs["peak_rss_mb"] = max(filter(None, [first.peak_rss_mb, timed.peak_rss_mb]), default=None)
    return obs


def temp_output(suffix: str) -> str:
    handle, path = tempfile.mkstemp(suffix=suffix, prefix="prismq-cmp-")
    os.close(handle)
    return path

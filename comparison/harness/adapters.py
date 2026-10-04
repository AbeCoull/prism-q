"""Simulator adapters.

Every adapter is timed on the same region: execute the circuit and materialize
the full 2^n probability vector. Building the circuit from the gate list sits
outside that region on every side, and each simulator's own optimization (gate
fusion in PRISM-Q, Aer and qsim) sits inside it, since that is what a user of
that simulator gets.

The Python-driven comparators (Aer, qsim) build their circuit objects from the
gate list through the simulator's API. The native comparators (QuEST, Spinoza,
RustQIP) are separate executables that read the same gate list on stdin and
answer with the same JSON, so the harness treats them alike.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np

from .corpus import Program

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXE = ".exe" if sys.platform == "win32" else ""
RUNNER = os.path.join(REPO_ROOT, "target", "release", "examples", "compare_runner" + EXE)
PEERS = os.path.join(REPO_ROOT, "comparison", "peers", "target", "release", "peers" + EXE)
QUEST = os.path.join(
    REPO_ROOT, "comparison", "quest", "build", "Release" if sys.platform == "win32" else "", "quest_runner" + EXE
)


class AdapterError(RuntimeError):
    pass


def parse_gate_list(text: str) -> tuple[int, list[tuple[str, list[int], float | None]]]:
    lines = [line.split() for line in text.split("\n") if line.strip()]
    if not lines or lines[0][0] != "qubits":
        raise AdapterError("gate list has no `qubits N` header")
    num_qubits = int(lines[0][1])
    gates: list[tuple[str, list[int], float | None]] = []
    for parts in lines[1:]:
        name = parts[0]
        if name in ("h",):
            gates.append((name, [int(parts[1])], None))
        elif name in ("cx", "swap"):
            gates.append((name, [int(parts[1]), int(parts[2])], None))
        elif name in ("ry", "rz"):
            gates.append((name, [int(parts[1])], float(parts[2])))
        elif name == "cp":
            gates.append((name, [int(parts[1]), int(parts[2])], float(parts[3])))
        else:
            raise AdapterError(f"unsupported gate `{' '.join(parts)}`")
    return num_qubits, gates


class Native:
    """A comparator driven as a subprocess over the shared gate-list protocol."""

    def __init__(self, name: str, label: str, argv: list[str], env: dict[str, str], timeout_s: float) -> None:
        self.name = name
        self.label = label
        self.argv = argv
        self.env = dict(os.environ, **env)
        self.timeout_s = timeout_s
        if not os.path.exists(argv[0]):
            raise AdapterError(f"{name}: executable not built: {argv[0]}")

    def _invoke(self, args: list[str], text: str) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                [*self.argv, *args],
                input=text,
                capture_output=True,
                text=True,
                env=self.env,
                check=False,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdapterError(f"timed out after {self.timeout_s:.0f} s") from exc
        if proc.returncode != 0:
            raise AdapterError(proc.stderr.strip()[:200] or f"exit code {proc.returncode}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def version(self) -> str | None:
        try:
            return self._invoke(["version"], "").get("version")
        except (AdapterError, ValueError):
            return None

    def time(self, program: Program, iterations: int) -> list[float]:
        return list(self._invoke(["time", str(iterations)], program.text)["times_ms"])

    def probabilities(self, program: Program) -> np.ndarray:
        handle, path = tempfile.mkstemp(suffix=".f64")
        os.close(handle)
        try:
            self._invoke(["probabilities", path], program.text)
            return np.fromfile(path, dtype="<f8")
        finally:
            os.unlink(path)


def prismq(threads: int, timeout_s: float) -> Native:
    return Native(
        "prismq",
        "PRISM-Q (auto dispatch)",
        [RUNNER],
        {"RAYON_NUM_THREADS": str(threads)},
        timeout_s,
    )


def spinoza(threads: int, timeout_s: float) -> Native:
    return Native(
        "spinoza",
        "Spinoza",
        [PEERS, "spinoza"],
        {"RAYON_NUM_THREADS": str(threads)},
        timeout_s,
    )


def qip(threads: int, timeout_s: float) -> Native:
    return Native(
        "qip",
        "RustQIP (qip)",
        [PEERS, "qip"],
        {"RAYON_NUM_THREADS": str(threads)},
        timeout_s,
    )


def quest(threads: int, timeout_s: float) -> Native:
    return Native(
        "quest",
        "QuEST (OpenMP)",
        [QUEST],
        {"OMP_NUM_THREADS": str(threads)},
        timeout_s,
    )


class Python:
    """A comparator driven in-process; subclasses build and execute the circuit."""

    name: str
    label: str

    def version(self) -> str | None:
        raise NotImplementedError

    def _prepare(self, program: Program) -> Any:
        raise NotImplementedError

    def _execute(self, prepared: Any) -> np.ndarray:
        raise NotImplementedError

    def time(self, program: Program, iterations: int) -> list[float]:
        prepared = self._prepare(program)
        try:
            self._execute(prepared)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"{type(exc).__name__}: {exc}") from exc
        times_ms: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            probs = self._execute(prepared)
            times_ms.append((time.perf_counter() - start) * 1000.0)
            del probs
        return times_ms

    def probabilities(self, program: Program) -> np.ndarray:
        return self._execute(self._prepare(program))

    def fixed_overhead_ms(self, iterations: int = 20) -> float:
        """Per-call cost of the Python round trip on a one-gate circuit.

        At small qubit counts this dominates the measurement, so it is recorded
        alongside the results rather than silently folded into them.
        """
        trivial = Program("overhead", 1, "qubits 1\nh 0\n", "", 1)
        prepared = self._prepare(trivial)
        self._execute(prepared)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            self._execute(prepared)
            samples.append((time.perf_counter() - start) * 1000.0)
        return float(np.median(samples))


class Aer(Python):
    def __init__(self, method: str, threads: int) -> None:
        from qiskit_aer import AerSimulator

        self.name = f"aer-{method}"
        self.label = f"Qiskit Aer ({method})"
        self.method = method
        self.backend = AerSimulator(method=method, max_parallel_threads=threads)

    def version(self) -> str | None:
        from importlib import metadata

        return metadata.version("qiskit-aer")

    def _prepare(self, program: Program) -> Any:
        from qiskit import QuantumCircuit

        num_qubits, gates = parse_gate_list(program.text)
        circuit = QuantumCircuit(num_qubits)
        for name, qubits, angle in gates:
            if name == "h":
                circuit.h(qubits[0])
            elif name == "cx":
                circuit.cx(qubits[0], qubits[1])
            elif name == "swap":
                circuit.swap(qubits[0], qubits[1])
            elif name == "ry":
                circuit.ry(angle, qubits[0])
            elif name == "rz":
                circuit.rz(angle, qubits[0])
            elif name == "cp":
                circuit.cp(angle, qubits[0], qubits[1])
        circuit.save_probabilities()
        return circuit

    def _execute(self, prepared: Any) -> np.ndarray:
        result = self.backend.run(prepared, shots=1).result()
        if not result.success:
            raise AdapterError(str(result.status))
        return np.asarray(result.data()["probabilities"], dtype=np.float64)


class Qsim(Python):
    def __init__(self, threads: int) -> None:
        import qsimcirq

        self.name = "qsim"
        self.label = "qsim (qsimcirq)"
        self.simulator = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=threads))

    def version(self) -> str | None:
        from importlib import metadata

        return metadata.version("qsimcirq")

    def _prepare(self, program: Program) -> Any:
        import cirq

        num_qubits, gates = parse_gate_list(program.text)
        qubits = cirq.LineQubit.range(num_qubits)
        ops = []
        for name, targets, angle in gates:
            q = [qubits[i] for i in targets]
            if name == "h":
                ops.append(cirq.H(q[0]))
            elif name == "cx":
                ops.append(cirq.CNOT(q[0], q[1]))
            elif name == "swap":
                ops.append(cirq.SWAP(q[0], q[1]))
            elif name == "ry":
                ops.append(cirq.ry(angle)(q[0]))
            elif name == "rz":
                ops.append(cirq.rz(angle)(q[0]))
            elif name == "cp":
                ops.append(cirq.CZPowGate(exponent=angle / np.pi)(q[0], q[1]))
        # Cirq orders the state index with the first qubit as the most
        # significant bit; reversing the order puts qubit 0 in the least
        # significant bit, the convention every other adapter uses.
        return cirq.Circuit(ops), list(reversed(qubits))

    def _execute(self, prepared: Any) -> np.ndarray:
        circuit, order = prepared
        state = self.simulator.simulate(circuit, qubit_order=order).final_state_vector
        return np.abs(state.astype(np.complex128)) ** 2


FACTORIES: dict[str, Callable[[int, float], Any]] = {
    "prismq": prismq,
    "aer-statevector": lambda threads, timeout: Aer("statevector", threads),
    "aer-automatic": lambda threads, timeout: Aer("automatic", threads),
    "qsim": lambda threads, timeout: Qsim(threads),
    "quest": quest,
    "spinoza": spinoza,
    "qip": qip,
}

DEFAULT_SIMULATORS = ("prismq", "aer-statevector", "aer-automatic", "qsim", "quest", "spinoza", "qip")


def build_all(names: list[str], threads: int, timeout_s: float) -> list[Any]:
    sims = []
    for name in names:
        factory = FACTORIES.get(name)
        if factory is None:
            raise AdapterError(f"unknown simulator `{name}`; known: {', '.join(FACTORIES)}")
        sims.append(factory(threads, timeout_s))
    return sims

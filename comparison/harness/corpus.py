"""Circuit corpus: the benchmark suite's own families, exported once as gate lists.

The circuits are the ones `docs/benchmarks.md` is measured on, built by the
`prism_q::circuits` generators and written out by `compare_runner export` as a
plain gate list (`qubits N`, then one gate per line). Every simulator replays
that list through its native API, so no simulator gets a private transpilation
pass and none parses OpenQASM. The list is hashed and the hash travels with the
results; `--check` refuses to compare runs whose hashes differ.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from typing import Any

FAMILIES: dict[str, str] = {
    "ghz": "H on qubit 0, then a CX chain",
    "qft": "textbook quantum Fourier transform: H, controlled phases, final swaps",
    "hea": "hardware-efficient ansatz: 5 layers of Ry and Rz on every qubit, then a CX chain",
    "qv": "quantum volume: n layers of random pairings, each pair a random SU(4) as CX and rotations",
}

GATE_SET = ("h", "cx", "swap", "ry", "rz", "cp")
CIRCUIT_SEED = "0xDEAD_BEEF"


@dataclass(frozen=True)
class Program:
    benchmark: str
    num_qubits: int
    text: str
    sha256: str
    num_operations: int

    def summary(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "num_qubits": self.num_qubits,
            "sha256": self.sha256,
            "num_operations": self.num_operations,
        }


@dataclass(frozen=True)
class Skip:
    benchmark: str
    num_qubits: int
    reason: str

    def summary(self) -> dict[str, Any]:
        return {"benchmark": self.benchmark, "num_qubits": self.num_qubits, "reason": self.reason}


def available_benchmarks() -> list[str]:
    return list(FAMILIES)


def build(runner: str, name: str, num_qubits: int) -> Program | Skip:
    if name not in FAMILIES:
        return Skip(name, num_qubits, f"unknown family; known: {', '.join(FAMILIES)}")
    proc = subprocess.run(
        [runner, "export", name, str(num_qubits)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return Skip(name, num_qubits, proc.stderr.strip()[:200] or f"export exited {proc.returncode}")
    text = proc.stdout.replace("\r\n", "\n")
    lines = [line for line in text.split("\n") if line.strip()]
    return Program(
        benchmark=name,
        num_qubits=num_qubits,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        num_operations=len(lines) - 1,
    )


def descriptor() -> dict[str, Any]:
    return {
        "name": "prism-q benchmark suite families",
        "families": dict(FAMILIES),
        "gate_set": list(GATE_SET),
        "circuit_seed": CIRCUIT_SEED,
        "source": "examples/compare_runner.rs export, from the prism_q::circuits generators",
    }

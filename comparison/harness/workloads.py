"""Canonical workload definitions for the benchmark matrix.

A workload is a gate list (the shared text format read by every runner), an
output mode, and the parameters that produced it. The list is generated once,
hashed, and handed unchanged to every simulator, so no simulator gets a
private transpilation pass. Random families draw from NumPy's PCG64 seeded per
workload; the structured families (QFT, QV) come from the `prism_q::circuits`
generators through `compare_runner export`, the same circuits as the published
benchmark suite.

Gate list format (see `tools/compare_runner.rs`): `qubits N`, an optional
`clbits M`, then one instruction per line. `h x z s sdg ry rz` act on one qubit,
`cx cz swap cp` on two, `measure q c` writes classical bit `c`, `reset q`
returns a qubit to |0>, and `if c v <gate>` applies a gate when bit `c` reads `v`.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CIRCUIT_SEED = 0xDEAD_BEEF
GATE_SET = ("h", "x", "z", "s", "sdg", "ry", "rz", "cx", "cz", "swap", "cp", "measure", "reset", "if")
CLIFFORD_1Q = ("h", "s", "sdg", "x", "z", "hs")

FAMILIES: dict[str, str] = {
    "rcs": (
        "dense random circuit: per layer an Rz Ry Rz rotation with uniform random angles on every "
        "qubit, then CX on alternating nearest-neighbour pairs (brickwork on a line)"
    ),
    "qft": (
        "textbook quantum Fourier transform from prism_q::circuits::qft_circuit, expanded to H, CP, "
        "SWAP, applied to a seeded random product state (Ry then Rz on every qubit); on |0...0> every "
        "CP has its control in |0>, so the output would be uniform whatever the CP angles"
    ),
    "qv": (
        "quantum volume from prism_q::circuits::quantum_volume_circuit: n layers, each a random "
        "pairing of the qubits with a random SU(4) per pair decomposed into CX, Ry and Rz"
    ),
    "clifford": (
        "random Clifford brickwork: per layer one of H, S, Sdg, X, Z, HS uniformly on every qubit, "
        "then CX on alternating nearest-neighbour pairs; every qubit measured at the end"
    ),
    "lowent": (
        "shallow dense brickwork on a wide register: the rcs layer at depth 4 or 8, so the bond "
        "dimension across any cut stays at most 2^ceil(d/2); every qubit measured at the end"
    ),
    "disjoint": (
        "independent 10-qubit components, each an rcs brickwork of depth 10 inside the component; "
        "`contiguous` puts component k on qubits 10k..10k+9, `strided` on k, k+K, k+2K, ...; "
        "every qubit measured at the end"
    ),
    "sampling": (
        "rcs brickwork of depth 10, then a terminal measurement of all qubits or of the lower "
        "half, sampled at several shot counts"
    ),
    "midcircuit": (
        "repeated parity checks: per round random rotations and one CX brickwork layer on the "
        "data qubits, each ancilla collects the parity of a data pair, is measured, feeds a "
        "conditional X back onto the pair's first qubit, and is reset; data measured at the end"
    ),
}


@dataclass(frozen=True)
class Workload:
    family: str
    num_qubits: int
    mode: str
    text: str
    params: dict[str, Any] = field(default_factory=dict)
    shots: int = 0

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def num_operations(self) -> int:
        return sum(1 for line in self.text.split("\n") if line.strip()) - 1 - (self.num_clbits > 0)

    @property
    def num_clbits(self) -> int:
        for line in self.text.split("\n")[:2]:
            if line.startswith("clbits "):
                return int(line.split()[1])
        return 0

    @property
    def workload_id(self) -> str:
        parts = [self.family, f"n{self.num_qubits}"]
        for key in sorted(self.params):
            value = self.params[key]
            if key == "seed":
                value = f"{value:x}"
            parts.append(f"{key}={value}")
        parts.append(self.mode if self.mode == "probs" else f"counts{self.shots}")
        return "/".join(parts)

    def summary(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "family": self.family,
            "num_qubits": self.num_qubits,
            "mode": self.mode,
            "shots": self.shots,
            "params": dict(self.params),
            "num_operations": self.num_operations,
            "num_clbits": self.num_clbits,
            "sha256": self.sha256,
        }


def _angle(rng: np.random.Generator) -> str:
    return repr(float(rng.uniform(0.0, 2.0 * math.pi)))


def _brickwork(lines: list[str], qubits: list[int], depth: int, rng: np.random.Generator) -> None:
    for layer in range(depth):
        for q in qubits:
            lines.append(f"rz {q} {_angle(rng)}")
            lines.append(f"ry {q} {_angle(rng)}")
            lines.append(f"rz {q} {_angle(rng)}")
        for i in range(layer % 2, len(qubits) - 1, 2):
            lines.append(f"cx {qubits[i]} {qubits[i + 1]}")


def _measure(lines: list[str], qubits: list[int]) -> None:
    for bit, q in enumerate(qubits):
        lines.append(f"measure {q} {bit}")


def _text(num_qubits: int, body: list[str], clbits: int = 0) -> str:
    head = [f"qubits {num_qubits}"]
    if clbits:
        head.append(f"clbits {clbits}")
    return "\n".join(head + body) + "\n"


def rcs(n: int, depth: int, seed: int = CIRCUIT_SEED) -> Workload:
    rng = np.random.Generator(np.random.PCG64(seed))
    body: list[str] = []
    _brickwork(body, list(range(n)), depth, rng)
    return Workload("rcs", n, "probs", _text(n, body), {"depth": depth, "seed": seed})


def sampling(n: int, shots: int, measured: str = "all", depth: int = 10, seed: int = CIRCUIT_SEED) -> Workload:
    rng = np.random.Generator(np.random.PCG64(seed))
    body: list[str] = []
    _brickwork(body, list(range(n)), depth, rng)
    targets = list(range(n)) if measured == "all" else list(range(n // 2))
    _measure(body, targets)
    return Workload(
        "sampling",
        n,
        "counts",
        _text(n, body, len(targets)),
        {"depth": depth, "measured": measured, "seed": seed},
        shots,
    )


def clifford(n: int, shots: int, depth: int = 8, seed: int = CIRCUIT_SEED) -> Workload:
    rng = np.random.Generator(np.random.PCG64(seed))
    body: list[str] = []
    for layer in range(depth):
        choices = rng.integers(0, len(CLIFFORD_1Q), size=n)
        for q, c in enumerate(choices):
            name = CLIFFORD_1Q[int(c)]
            if name == "hs":
                body.append(f"h {q}")
                body.append(f"s {q}")
            else:
                body.append(f"{name} {q}")
        for i in range(layer % 2, n - 1, 2):
            body.append(f"cx {i} {i + 1}")
    _measure(body, list(range(n)))
    return Workload("clifford", n, "counts", _text(n, body, n), {"depth": depth, "seed": seed}, shots)


def lowent(n: int, depth: int, shots: int, seed: int = CIRCUIT_SEED) -> Workload:
    rng = np.random.Generator(np.random.PCG64(seed))
    body: list[str] = []
    _brickwork(body, list(range(n)), depth, rng)
    _measure(body, list(range(n)))
    return Workload("lowent", n, "counts", _text(n, body, n), {"depth": depth, "seed": seed}, shots)


def disjoint(
    n: int, shots: int, layout: str = "contiguous", block: int = 10, depth: int = 10, seed: int = CIRCUIT_SEED
) -> Workload:
    if n % block:
        raise ValueError(f"{n} qubits do not split into {block}-qubit components")
    rng = np.random.Generator(np.random.PCG64(seed))
    count = n // block
    body: list[str] = []
    for k in range(count):
        if layout == "contiguous":
            members = list(range(k * block, (k + 1) * block))
        else:
            members = [k + count * j for j in range(block)]
        _brickwork(body, members, depth, rng)
    _measure(body, list(range(n)))
    return Workload(
        "disjoint",
        n,
        "counts",
        _text(n, body, n),
        {"block": block, "depth": depth, "layout": layout, "seed": seed},
        shots,
    )


def midcircuit(n: int, shots: int, rounds: int = 3, seed: int = CIRCUIT_SEED) -> Workload:
    if n % 3:
        raise ValueError("midcircuit needs n = 3k: 2k data qubits and k ancillas")
    anc = n // 3
    data = 2 * anc
    rng = np.random.Generator(np.random.PCG64(seed))
    body: list[str] = []
    for r in range(rounds):
        for q in range(data):
            body.append(f"ry {q} {_angle(rng)}")
            body.append(f"rz {q} {_angle(rng)}")
        for i in range(r % 2, data - 1, 2):
            body.append(f"cx {i} {i + 1}")
        for j in range(anc):
            a = data + j
            bit = r * anc + j
            body.append(f"cx {2 * j} {a}")
            body.append(f"cx {2 * j + 1} {a}")
            body.append(f"measure {a} {bit}")
            body.append(f"if {bit} 1 x {2 * j}")
            body.append(f"reset {a}")
    for i in range(data):
        body.append(f"measure {i} {rounds * anc + i}")
    clbits = rounds * anc + data
    return Workload("midcircuit", n, "counts", _text(n, body, clbits), {"rounds": rounds, "seed": seed}, shots)


def exported(runner: str, family: str, n: int, seed: int = CIRCUIT_SEED) -> Workload:
    proc = subprocess.run([runner, "export", family, str(n), f"{seed:x}"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:200] or f"export exited {proc.returncode}")
    params: dict[str, Any] = {"seed": seed} if family in ("qv", "hea") else {}
    return Workload(family, n, "probs", proc.stdout.replace("\r\n", "\n"), params)


def qft(runner: str, n: int, seed: int = CIRCUIT_SEED) -> Workload:
    rng = np.random.Generator(np.random.PCG64(seed))
    layer: list[str] = []
    for q in range(n):
        layer.append(f"ry {q} {_angle(rng)}")
        layer.append(f"rz {q} {_angle(rng)}")
    head, _, body = exported(runner, "qft", n).text.partition("\n")
    return Workload("qft", n, "probs", "\n".join([head, *layer, body]), {"input": "product", "seed": seed})


def parse(text: str) -> tuple[int, int, list[tuple]]:
    """Parse a gate list into (qubits, clbits, instructions).

    Instructions are `("gate", name, qubits, angle)`, `("measure", q, c)`,
    `("reset", q)` and `("if", c, v, name, qubits, angle)`.
    """
    lines = [line.split() for line in text.split("\n") if line.strip()]
    num_qubits = int(lines[0][1])
    clbits = 0
    start = 1
    if len(lines) > 1 and lines[1][0] == "clbits":
        clbits = int(lines[1][1])
        start = 2
    out: list[tuple] = []
    for parts in lines[start:]:
        head = parts[0]
        if head == "measure":
            out.append(("measure", int(parts[1]), int(parts[2])))
        elif head == "reset":
            out.append(("reset", int(parts[1])))
        elif head == "if":
            name, qubits, angle = _gate(parts[3:])
            out.append(("if", int(parts[1]), int(parts[2]), name, qubits, angle))
        else:
            name, qubits, angle = _gate(parts)
            out.append(("gate", name, qubits, angle))
    return num_qubits, clbits, out


def _gate(parts: list[str]) -> tuple[str, list[int], float | None]:
    name = parts[0]
    if name in ("h", "x", "z", "s", "sdg"):
        return name, [int(parts[1])], None
    if name in ("ry", "rz"):
        return name, [int(parts[1])], float(parts[2])
    if name in ("cx", "cz", "swap"):
        return name, [int(parts[1]), int(parts[2])], None
    if name == "cp":
        return name, [int(parts[1]), int(parts[2])], float(parts[3])
    raise ValueError(f"unsupported gate `{' '.join(parts)}`")


def gate_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.split("\n")[1:]:
        parts = line.split()
        if not parts or parts[0] == "clbits":
            continue
        names.add(parts[0])
        if parts[0] == "if":
            names.add(parts[3])
    return names

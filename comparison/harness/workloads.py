"""Canonical workload definitions for the benchmark matrix.

A workload is a gate list (the shared text format read by every runner), an
output mode, and the parameters that produced it. The modes are `probs` (the full
probability vector), `counts` (a shot histogram), `expval` (Z and adjacent ZZ
expectation values on an unmeasured circuit) and `stats` (only the shot statistics
the validator checks, for shot counts whose histogram would not fit in memory). The list is generated once,
hashed, and handed unchanged to every simulator, so no simulator gets a
private transpilation pass. Random families draw from NumPy's PCG64 seeded per
workload; the structured families (QFT, QV, HEA) come from the `prism_q::circuits`
generators through `compare_runner export`, the same circuits as the published
benchmark suite.

Gate list format (see `tools/compare_runner.rs`): `qubits N`, an optional
`clbits M`, then one instruction per line. `h x z s sdg rx ry rz` act on one qubit,
`cx cz swap cp rzz` on two, `measure q c` writes classical bit `c`, `reset q`
returns a qubit to |0>, and `if c v <gate>` applies a gate when bit `c` reads `v`.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
import subprocess
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

CIRCUIT_SEED = 0xDEAD_BEEF
GATE_SET = ("h", "x", "z", "s", "sdg", "rx", "ry", "rz", "cx", "cz", "swap", "cp", "rzz", "measure", "reset", "if")
CLIFFORD_1Q = ("h", "s", "sdg", "x", "z", "hs")

FAMILIES: dict[str, str] = {
    "rcs": (
        "dense random circuit: per layer an Rz Ry Rz rotation with uniform random angles on every "
        "qubit, then CX on alternating nearest-neighbour pairs (brickwork on a line); in `expval` "
        "mode the same circuit without measurements, read out as Z and adjacent ZZ expectation values"
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
        "dense brickwork on a wide register: the rcs layer at depth 4 to 16, so the bond "
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
    "ghz": "GHZ state: H on qubit 0, then a CX chain along the line; every qubit measured at the end",
    "bv": (
        "Bernstein-Vazirani: n-1 data qubits and one ancilla in |->, an oracle of one CX per set bit "
        "of a seeded random secret, H on the data, and the data measured; every shot reads the secret"
    ),
    "qaoa": (
        "QAOA MaxCut ansatz on a seeded random 3-regular graph: H on every qubit, then per layer RZZ "
        "on every edge with one random angle and an RX mixer with another; the edges ignore the "
        "line, so the circuit has no nearest-neighbour structure"
    ),
    "hea": (
        "hardware-efficient ansatz from prism_q::circuits::hardware_efficient_ansatz: five layers, each "
        "random Ry and Rz on every qubit followed by a CX chain along the line"
    ),
    "tfim": (
        "Trotterized transverse-field Ising chain from |0...0>, J = h = 1: per step RZZ(2 J dt) on "
        "even then odd bonds and RX(2 h dt) on every qubit; read out as the full probability vector "
        "or as Z and adjacent ZZ expectation values"
    ),
    "wstate": (
        "W state: the excitation on qubit 0 handed down the line by an Ry and CX ladder, leaving "
        "amplitude 1/sqrt(n) on each one-hot basis state; every qubit measured at the end"
    ),
    "adder": (
        "Cuccaro ripple-carry adder of two k-bit registers on 2k + 2 qubits, Toffolis expanded into "
        "H, CX and Rz(pi/4); register a in uniform superposition, register b a seeded random integer "
        "set with Ry(pi)"
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
    recipe: dict[str, Any] | None = field(default=None, compare=False)

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
        parts.append(self.mode if self.mode in ("probs", "expval") else f"{self.mode}{self.shots}")
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


RECIPES: dict[str, Callable[..., Workload]] = {}


def _recipe(fn: Callable[..., Workload]) -> Callable[..., Workload]:
    """Register a generator that needs no runner, so a stored workload can be rebuilt from its arguments."""
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def build(*args, **kwargs) -> Workload:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return replace(fn(*args, **kwargs), recipe={"fn": fn.__name__, "kwargs": dict(bound.arguments)})

    RECIPES[fn.__name__] = build
    return build


def from_recipe(recipe: dict[str, Any]) -> Workload:
    return RECIPES[recipe["fn"]](**recipe["kwargs"])


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


@_recipe
def rcs(n: int, depth: int, seed: int = CIRCUIT_SEED) -> Workload:
    rng = np.random.Generator(np.random.PCG64(seed))
    body: list[str] = []
    _brickwork(body, list(range(n)), depth, rng)
    return Workload("rcs", n, "probs", _text(n, body), {"depth": depth, "seed": seed})


@_recipe
def expval(n: int, depth: int, observables: int | None = None, seed: int = CIRCUIT_SEED) -> Workload:
    """The rcs circuit read out as expectation values.

    `observables` Z terms and as many adjacent ZZ terms, drawn as `validate.select_statistics`
    draws bits; None takes every qubit and every adjacent pair up to its cap.
    """
    base = rcs(n, depth, seed)
    params = dict(base.params)
    if observables is not None:
        params["observables"] = observables
    return Workload("rcs", n, "expval", base.text, params)


@_recipe
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


@_recipe
def clifford(n: int, shots: int, depth: int = 8, seed: int = CIRCUIT_SEED, mode: str = "counts") -> Workload:
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
    return Workload("clifford", n, mode, _text(n, body, n), {"depth": depth, "seed": seed}, shots)


@_recipe
def lowent(n: int, depth: int, shots: int, seed: int = CIRCUIT_SEED) -> Workload:
    rng = np.random.Generator(np.random.PCG64(seed))
    body: list[str] = []
    _brickwork(body, list(range(n)), depth, rng)
    _measure(body, list(range(n)))
    return Workload("lowent", n, "counts", _text(n, body, n), {"depth": depth, "seed": seed}, shots)


@_recipe
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


@_recipe
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


@_recipe
def ghz(n: int, shots: int, mode: str = "counts") -> Workload:
    body = ["h 0"] + [f"cx {q} {q + 1}" for q in range(n - 1)]
    _measure(body, list(range(n)))
    return Workload("ghz", n, mode, _text(n, body, n), {}, shots)


@_recipe
def bv(n: int, shots: int, seed: int = CIRCUIT_SEED) -> Workload:
    data = anc = n - 1
    rng = np.random.Generator(np.random.PCG64(seed))
    secret = rng.integers(0, 2, size=data)
    body = [f"x {anc}", f"h {anc}"] + [f"h {q}" for q in range(data)]
    body += [f"cx {q} {anc}" for q in range(data) if secret[q]]
    body += [f"h {q}" for q in range(data)]
    _measure(body, list(range(data)))
    return Workload("bv", n, "counts", _text(n, body, data), {"seed": seed}, shots)


def _regular_graph(n: int, degree: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """A random simple `degree`-regular graph: configuration model pairings, rejected until simple."""
    if n * degree % 2 or degree >= n:
        raise ValueError(f"no simple {degree}-regular graph on {n} vertices")
    stubs = np.repeat(np.arange(n), degree)
    for _ in range(10_000):
        rng.shuffle(stubs)
        pairs = stubs.reshape(-1, 2)
        edges = {(int(min(a, b)), int(max(a, b))) for a, b in pairs}
        if len(edges) == len(pairs) and all(a != b for a, b in edges):
            return sorted(edges)
    raise RuntimeError("the configuration model found no simple graph")


@_recipe
def qaoa(n: int, layers: int = 1, seed: int = CIRCUIT_SEED, degree: int = 3) -> Workload:
    rng = np.random.Generator(np.random.PCG64(seed))
    edges = _regular_graph(n, degree, rng)
    body = [f"h {q}" for q in range(n)]
    for _ in range(layers):
        gamma, beta = _angle(rng), _angle(rng)
        body += [f"rzz {a} {b} {gamma}" for a, b in edges]
        body += [f"rx {q} {beta}" for q in range(n)]
    return Workload("qaoa", n, "probs", _text(n, body), {"degree": degree, "layers": layers, "seed": seed})


def hea(runner: str, n: int, seed: int = CIRCUIT_SEED) -> Workload:
    return exported(runner, "hea", n, seed)


@_recipe
def tfim(n: int, steps: int, mode: str = "probs", dt: float = 0.1, observables: int | None = None) -> Workload:
    angle = repr(2.0 * dt)
    body: list[str] = []
    for _ in range(steps):
        for parity in (0, 1):
            body += [f"rzz {i} {i + 1} {angle}" for i in range(parity, n - 1, 2)]
        body += [f"rx {q} {angle}" for q in range(n)]
    params: dict[str, Any] = {"dt": dt, "steps": steps}
    if observables is not None:
        params["observables"] = observables
    return Workload("tfim", n, mode, _text(n, body), params)


@_recipe
def wstate(n: int, shots: int, mode: str = "counts") -> Workload:
    body = ["x 0"]
    for i in range(n - 1):
        half = math.acos(math.sqrt(1.0 / (n - i)))
        body += [f"ry {i + 1} {half!r}", f"cx {i} {i + 1}", f"ry {i + 1} {-half!r}", f"cx {i} {i + 1}"]
        body.append(f"cx {i + 1} {i}")
    if mode == "probs":
        return Workload("wstate", n, mode, _text(n, body), {})
    _measure(body, list(range(n)))
    return Workload("wstate", n, mode, _text(n, body, n), {}, shots)


def _toffoli(body: list[str], c1: int, c2: int, t: int) -> None:
    """CCX as H, CX and Rz(+-pi/4); each Rz differs from T or its inverse by a global phase only."""
    tp, tm = repr(math.pi / 4), repr(-math.pi / 4)
    body += [f"h {t}", f"cx {c2} {t}", f"rz {t} {tm}", f"cx {c1} {t}", f"rz {t} {tp}", f"cx {c2} {t}"]
    body += [f"rz {t} {tm}", f"cx {c1} {t}", f"rz {c2} {tp}", f"rz {t} {tp}", f"h {t}", f"cx {c1} {c2}"]
    body += [f"rz {c1} {tp}", f"rz {c2} {tm}", f"cx {c1} {c2}"]


@_recipe
def adder(bits: int, shots: int = 0, seed: int = CIRCUIT_SEED) -> Workload:
    """Cuccaro adder: qubit 0 the carry in, b_i on 2i + 1, a_i on 2i + 2, the carry out last; b becomes a + b."""
    n = 2 * bits + 2
    rng = np.random.Generator(np.random.PCG64(seed))
    b_value = int(rng.integers(0, 2**bits))
    a_q = [2 * i + 2 for i in range(bits)]
    b_q = [2 * i + 1 for i in range(bits)]
    body = [f"h {q}" for q in a_q] + [f"ry {b_q[i]} {math.pi!r}" for i in range(bits) if b_value >> i & 1]

    def maj(c: int, b: int, a: int) -> None:
        body.extend([f"cx {a} {b}", f"cx {a} {c}"])
        _toffoli(body, c, b, a)

    def uma(c: int, b: int, a: int) -> None:
        _toffoli(body, c, b, a)
        body.extend([f"cx {a} {c}", f"cx {c} {b}"])

    carries = [0] + a_q[:-1]
    for i in range(bits):
        maj(carries[i], b_q[i], a_q[i])
    body.append(f"cx {a_q[-1]} {n - 1}")
    for i in reversed(range(bits)):
        uma(carries[i], b_q[i], a_q[i])
    params = {"b": b_value, "bits": bits, "seed": seed}
    if not shots:
        return Workload("adder", n, "probs", _text(n, body), params)
    _measure(body, list(range(n)))
    return Workload("adder", n, "counts", _text(n, body, n), params, shots)


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
    if name in ("rx", "ry", "rz"):
        return name, [int(parts[1])], float(parts[2])
    if name in ("cx", "cz", "swap"):
        return name, [int(parts[1]), int(parts[2])], None
    if name in ("cp", "rzz"):
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

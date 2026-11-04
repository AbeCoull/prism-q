"""Independent exact reference for measurement statistics.

Shares no code with any simulator under test. The distribution of a few
classical bits depends only on the instructions in their causal past: gates
that touch a qubit already in the cone, measurements and resets of such qubits,
and the measurements whose bits feed a conditional inside the cone. That past
cone is extracted by a backward pass and evaluated exactly with a small NumPy
statevector that branches on every measurement in it. On local circuits the
cone stays narrow at any register width, which gives exact marginals for
circuits far beyond a dense statevector (10,000-qubit Clifford brickwork, a
200-qubit shallow circuit, disconnected components).
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

MAX_CONE_QUBITS = 20
MAX_CONE_MEASUREMENTS = 14
_PRUNE = 1e-14

_S2 = 1.0 / math.sqrt(2.0)
_FIXED: dict[str, np.ndarray] = {
    "h": np.array([[_S2, _S2], [_S2, -_S2]], dtype=np.complex128),
    "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "s": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=np.complex128),
}


class ConeTooWide(Exception):
    pass


def _matrix(name: str, angle: float | None) -> np.ndarray:
    if name in _FIXED:
        return _FIXED[name]
    if name == "ry":
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if name == "rz":
        return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
    u = np.eye(4, dtype=np.complex128)
    # Basis |a b> with a the first listed qubit (the control).
    if name == "cx":
        u = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
    elif name == "cz":
        u[3, 3] = -1
    elif name == "swap":
        u = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
    elif name == "cp":
        u[3, 3] = np.exp(1j * angle)
    else:
        raise ValueError(f"no matrix for {name}")
    return u.reshape(2, 2, 2, 2)


def _apply(psi: np.ndarray, name: str, axes: list[int], angle: float | None) -> np.ndarray:
    u = _matrix(name, angle)
    if len(axes) == 1:
        return np.moveaxis(np.tensordot(u, psi, axes=([1], [axes[0]])), 0, axes[0])
    out = np.tensordot(u, psi, axes=([2, 3], axes))
    return np.moveaxis(out, [0, 1], axes)


def cone(instructions: list[tuple], bits: Iterable[int]) -> list[tuple]:
    """Instructions in the causal past of `bits`, in forward order."""
    want_bits = set(bits)
    live_qubits: set[int] = set()
    kept: list[tuple] = []
    for inst in reversed(instructions):
        kind = inst[0]
        if kind == "measure":
            _, q, c = inst
            if c in want_bits:
                want_bits.discard(c)
                live_qubits.add(q)
                kept.append(inst)
            elif q in live_qubits:
                kept.append(inst)
        elif kind == "reset":
            if inst[1] in live_qubits:
                kept.append(inst)
        elif kind == "gate":
            if live_qubits.intersection(inst[2]):
                live_qubits.update(inst[2])
                kept.append(inst)
        elif kind == "if":
            if live_qubits.intersection(inst[4]):
                live_qubits.update(inst[4])
                want_bits.add(inst[1])
                kept.append(inst)
    kept.reverse()
    return kept


def bit_distribution(instructions: list[tuple], bits: list[int]) -> dict[tuple[int, ...], float]:
    """Exact joint distribution of `bits`, keyed by their values in order."""
    sub = cone(instructions, bits)
    qubits = sorted({q for inst in sub for q in _qubits(inst)})
    measurements = sum(1 for inst in sub if inst[0] in ("measure", "reset"))
    if len(qubits) > MAX_CONE_QUBITS or measurements > MAX_CONE_MEASUREMENTS:
        raise ConeTooWide(f"cone holds {len(qubits)} qubits and {measurements} measurements")
    axis = {q: i for i, q in enumerate(qubits)}
    psi = np.zeros((2,) * max(1, len(qubits)), dtype=np.complex128)
    psi[(0,) * psi.ndim] = 1.0
    result: dict[tuple[int, ...], float] = {}
    _branch(sub, 0, psi, {}, 1.0, axis, bits, result)
    return result


def _qubits(inst: tuple) -> list[int]:
    kind = inst[0]
    if kind == "measure" or kind == "reset":
        return [inst[1]]
    if kind == "gate":
        return list(inst[2])
    return list(inst[4])


def _branch(sub, start, psi, clbits, weight, axis, bits, result) -> None:
    for index in range(start, len(sub)):
        inst = sub[index]
        kind = inst[0]
        if kind == "gate":
            psi = _apply(psi, inst[1], [axis[q] for q in inst[2]], inst[3])
        elif kind == "if":
            if clbits.get(inst[1], 0) == inst[2]:
                psi = _apply(psi, inst[3], [axis[q] for q in inst[4]], inst[5])
        elif kind == "reset":
            a = axis[inst[1]]
            zero = np.take(psi, 0, axis=a)
            one = np.take(psi, 1, axis=a)
            p0 = float(np.vdot(zero, zero).real)
            p1 = float(np.vdot(one, one).real)
            # Reset is measure-then-flip; the two outcomes recombine as a
            # mixture, so each carries its own weight forward.
            for p, part in ((p0, zero), (p1, one)):
                if p * weight <= _PRUNE:
                    continue
                fresh = np.zeros_like(psi)
                index_tuple = [slice(None)] * psi.ndim
                index_tuple[a] = 0
                fresh[tuple(index_tuple)] = part / math.sqrt(p)
                _branch(sub, index + 1, fresh, clbits, weight * p, axis, bits, result)
            return
        else:
            _, q, c = inst
            a = axis[q]
            for outcome in (0, 1):
                part = np.take(psi, outcome, axis=a)
                p = float(np.vdot(part, part).real)
                if p * weight <= _PRUNE:
                    continue
                fresh = np.zeros_like(psi)
                index_tuple = [slice(None)] * psi.ndim
                index_tuple[a] = outcome
                fresh[tuple(index_tuple)] = part / math.sqrt(p)
                _branch(sub, index + 1, fresh, {**clbits, c: outcome}, weight * p, axis, bits, result)
            return
    key = tuple(clbits.get(b, 0) for b in bits)
    result[key] = result.get(key, 0.0) + weight


def statistics(instructions: list[tuple], singles: list[int], pairs: list[tuple[int, int]]) -> dict[str, dict]:
    """Exact P(bit = 1) for each single and P(a xor b = 1) for each pair."""
    out: dict[str, dict] = {}
    for b in singles:
        try:
            dist = bit_distribution(instructions, [b])
            out[f"b{b}"] = {"p": dist.get((1,), 0.0)}
        except ConeTooWide as exc:
            out[f"b{b}"] = {"p": None, "reason": str(exc)}
    for a, b in pairs:
        try:
            dist = bit_distribution(instructions, [a, b])
            out[f"b{a}^b{b}"] = {"p": dist.get((0, 1), 0.0) + dist.get((1, 0), 0.0)}
        except ConeTooWide as exc:
            out[f"b{a}^b{b}"] = {"p": None, "reason": str(exc)}
    return out

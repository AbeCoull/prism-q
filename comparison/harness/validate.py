"""Correctness checks for every measured output.

Dense outputs are compared by total variation distance against the reference
simulator's vector. Sampled outputs are reduced to a fixed set of statistics,
single-bit marginals P(b = 1) and adjacent-pair parities P(a xor b = 1), and each
is tested against its exact value from `reference.py` with a two-sided exact
binomial test. The threshold is Bonferroni-corrected so the whole set has a 1e-3
chance of a false alarm on a correct simulator. A statistic whose exact value
the cone reference cannot reach falls back to a two-sample z-test against
another simulator's counts. A bit the reference proves deterministic must match
exactly: one stray outcome is a failure. Below 100 shots only the deterministic
bits are checked and the output counts as not validated.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import binomtest, norm

from . import reference

FAMILY_ALPHA = 1e-3
MAX_STATISTICS = 128
MIN_SHOTS = 100
_DETERMINISTIC = 1e-12


def tvd(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape:
        return None
    return float(0.5 * np.abs(left - right).sum())


def classical_fidelity(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape:
        return None
    return float(np.sum(np.sqrt(np.clip(left, 0, None) * np.clip(right, 0, None))) ** 2)


def read_counts(path: str, num_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Distinct outcomes as a (K, num_bits) 0/1 matrix and their counts."""
    words_per_key = max(1, (num_bits + 63) // 64)
    keys: list[list[int]] = []
    counts: list[int] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, count = line.split(" ")
            words = [int(w, 16) for w in key.split(",")]
            words += [0] * (words_per_key - len(words))
            keys.append(words[:words_per_key])
            counts.append(int(count))
    if not keys:
        return np.zeros((0, num_bits), dtype=np.uint8), np.zeros(0, dtype=np.int64)
    packed = np.asarray(keys, dtype=np.uint64).view(np.uint8)
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :num_bits]
    return bits, np.asarray(counts, dtype=np.int64)


def select_statistics(num_bits: int, seed: int = 42) -> tuple[list[int], list[tuple[int, int]]]:
    if num_bits <= MAX_STATISTICS:
        singles = list(range(num_bits))
        pairs = [(b, b + 1) for b in range(num_bits - 1)]
        return singles, pairs
    rng = np.random.Generator(np.random.PCG64(seed))
    singles = sorted(int(x) for x in rng.choice(num_bits, size=MAX_STATISTICS, replace=False))
    starts = sorted(int(x) for x in rng.choice(num_bits - 1, size=MAX_STATISTICS, replace=False))
    return singles, [(b, b + 1) for b in starts]


def observed(bits: np.ndarray, counts: np.ndarray, singles, pairs) -> tuple[dict[str, float], int]:
    shots = int(counts.sum())
    out: dict[str, float] = {}
    for b in singles:
        out[f"b{b}"] = float(counts @ bits[:, b]) / shots
    for a, b in pairs:
        out[f"b{a}^b{b}"] = float(counts @ (bits[:, a] ^ bits[:, b])) / shots
    return out, shots


def exact_statistics(instructions: list[tuple], singles, pairs) -> dict[str, dict]:
    return reference.statistics(instructions, singles, pairs)


def check_counts(
    obs: dict[str, float],
    shots: int,
    exact: dict[str, dict],
    fallback: tuple[dict[str, float], int] | None,
) -> dict[str, Any]:
    m = len(obs)
    p_star = FAMILY_ALPHA / m if m else 0.0
    smallest_p = 1.0
    failures: list[str] = []
    exact_used = fallback_used = deterministic = 0
    for key, value in obs.items():
        p = exact.get(key, {}).get("p")
        if p is not None:
            exact_used += 1
            if p < _DETERMINISTIC or p > 1 - _DETERMINISTIC:
                deterministic += 1
                if abs(value - round(p)) > 0:
                    failures.append(f"{key}: deterministic {round(p)} but observed {value:.4f}")
                    smallest_p = 0.0
                continue
            if shots < MIN_SHOTS:
                continue
            p_value = binomtest(round(value * shots), shots, p).pvalue
        elif shots >= MIN_SHOTS and fallback is not None and key in fallback[0]:
            fallback_used += 1
            other, other_shots = fallback
            pooled = (value * shots + other[key] * other_shots) / (shots + other_shots)
            var = pooled * (1 - pooled) * (1 / shots + 1 / other_shots)
            if var <= 0:
                p_value = 1.0 if value == other[key] else 0.0
            else:
                p_value = float(2 * norm.sf(abs(value - other[key]) / math.sqrt(var)))
        else:
            continue
        smallest_p = min(smallest_p, p_value)
        if p_value < p_star:
            failures.append(f"{key}: p {p_value:.2e} below {p_star:.2e}")
    tested = exact_used + fallback_used if shots >= MIN_SHOTS else deterministic
    result = {
        "method": (
            "exact binomial test per statistic against the cone reference, two-sample z-test against "
            "another simulator where the cone is too wide; Bonferroni family-wise alpha 1e-3"
        ),
        "statistics": m,
        "tested": tested,
        "against_exact": exact_used,
        "against_simulator": fallback_used,
        "deterministic": deterministic,
        "p_threshold": p_star,
        "min_p_value": smallest_p,
        "pass": bool(tested) and not failures,
        "failures": failures[:10],
    }
    if shots < MIN_SHOTS and not failures:
        result.update(
            {
                "pass": None,
                "role": "too-few-shots",
                "reason": f"fewer than {MIN_SHOTS} shots: deterministic bits checked, statistics not tested",
            }
        )
    return result

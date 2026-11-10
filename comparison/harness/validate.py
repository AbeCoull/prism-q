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
bits are checked and the output counts as not validated. A shot total that differs from
the workload's, or an outcome key with bits past the classical register, fails.
Each check records `min_detectable`, the shift in a probability of 0.5 its test would flag;
a passing output above `WEAK_DETECTABLE` gets role `weakly-tested`.

A `stats` output carries those statistics as counts with its shot total and is tested
the same way. An `expval` output is checked value by value: against the exact value
from the cone reference within the precision's tolerance, against a simulator of
another family within the same tolerance where the cone is too wide, and left
untested otherwise.
"""

from __future__ import annotations

import gzip
import math
from typing import Any

import numpy as np
from scipy.stats import binomtest, norm

from . import reference

FAMILY_ALPHA = 1e-3
MAX_STATISTICS = 128
MIN_SHOTS = 100
WEAK_DETECTABLE = 0.05
_DETERMINISTIC = 1e-12
COUNTS_METHOD = (
    "exact binomial test per statistic against the cone reference, two-sample z-test against "
    "another simulator where the cone is too wide; Bonferroni family-wise alpha 1e-3"
)
STATS_METHOD = COUNTS_METHOD + ", on the statistics and shot total the simulator returns"
EXPVAL_METHOD = (
    "absolute difference per value against the cone reference (⟨Z⟩ = 1 - 2p, ⟨ZZ⟩ = 1 - 2 P(odd)), "
    "against a simulator of another family where the cone is too wide"
)


def tvd(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape:
        return None
    return float(0.5 * np.abs(left - right).sum())


def classical_fidelity(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape:
        return None
    return float(np.sum(np.sqrt(np.clip(left, 0, None) * np.clip(right, 0, None))) ** 2)


def read_counts(path: str, num_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Distinct outcomes as a (K, num_bits) 0/1 matrix and their counts; `path` may be gzipped.

    Raises `ValueError` on a malformed line or a key with a bit set at or past `num_bits`.
    """
    words_per_key = max(1, (num_bits + 63) // 64)
    keys: list[list[int]] = []
    counts: list[int] = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                key, count = line.split(" ")
                words = [int(w, 16) for w in key.split(",")]
                counts.append(int(count))
            except ValueError:
                raise ValueError(f"line {number} of the counts file is malformed: {line[:80]!r}") from None
            if any(words[words_per_key:]):
                raise ValueError(f"line {number} has an outcome wider than {num_bits} bits")
            words += [0] * (words_per_key - len(words))
            keys.append(words[:words_per_key])
    if not keys:
        return np.zeros((0, num_bits), dtype=np.uint8), np.zeros(0, dtype=np.int64)
    packed = np.asarray(keys, dtype=np.uint64).view(np.uint8)
    bits = np.unpackbits(packed, axis=1, bitorder="little")
    if bits[:, num_bits:].any():
        raise ValueError(f"an outcome has a bit set past the {num_bits}-bit classical register")
    return bits[:, :num_bits], np.asarray(counts, dtype=np.int64)


def select_statistics(
    num_bits: int, seed: int = 42, limit: int = MAX_STATISTICS
) -> tuple[list[int], list[tuple[int, int]]]:
    """Every bit and adjacent pair when there are at most `limit`, otherwise `limit` of each by seed."""
    if num_bits <= limit:
        singles = list(range(num_bits))
        pairs = [(b, b + 1) for b in range(num_bits - 1)]
        return singles, pairs
    rng = np.random.Generator(np.random.PCG64(seed))
    singles = sorted(int(x) for x in rng.choice(num_bits, size=limit, replace=False))
    starts = sorted(int(x) for x in rng.choice(num_bits - 1, size=limit, replace=False))
    return singles, [(b, b + 1) for b in starts]


def keys_for(workload) -> list[str]:
    """The observables an `expval` workload reads, or the statistics a `stats` workload returns.

    `Z5` and `Z5Z6` name expectation values; `b5` and `b5^b6` name a bit's count of ones and a
    pair's count of odd parities. Runners receive them one per line.
    """
    if workload.mode == "expval":
        limit = workload.params.get("observables") or MAX_STATISTICS
        singles, pairs = select_statistics(workload.num_qubits, limit=limit)
        return [f"Z{q}" for q in singles] + [f"Z{a}Z{b}" for a, b in pairs]
    singles, pairs = select_statistics(workload.num_clbits)
    return [f"b{b}" for b in singles] + [f"b{a}^b{b}" for a, b in pairs]


def exact_expectations(instructions: list[tuple], keys: list[str]) -> dict[str, dict]:
    """Exact value of each `Z` key from the cone reference, or None with the reason it is out of reach."""
    terms = {key: [int(x) for x in key.split("Z")[1:]] for key in keys}
    singles = [qs[0] for qs in terms.values() if len(qs) == 1]
    pairs = [(qs[0], qs[1]) for qs in terms.values() if len(qs) == 2]
    qubits = sorted({q for qs in terms.values() for q in qs})
    probs = reference.statistics(instructions + [("measure", q, q) for q in qubits], singles, pairs)
    out: dict[str, dict] = {}
    for key, qs in terms.items():
        entry = probs["^".join(f"b{q}" for q in qs)]
        p = entry["p"]
        out[key] = {"value": None, "reason": entry["reason"]} if p is None else {"value": 1.0 - 2.0 * p}
    return out


def read_values(path: str) -> dict[str, float]:
    """`key value` lines; `path` may be gzipped. Raises `ValueError` on a malformed line."""
    opener = gzip.open if path.endswith(".gz") else open
    out: dict[str, float] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                key, value = line.split()
                out[key] = float(value)
            except ValueError:
                raise ValueError(f"line {number} of the output is malformed: {line.strip()[:80]!r}") from None
    return out


def read_stats(path: str) -> tuple[dict[str, float], int]:
    """A `stats` output as observed frequencies and the shot total it reports."""
    raw = read_values(path)
    if "shots" not in raw:
        raise ValueError("the output carries no `shots` line")
    shots = int(raw.pop("shots"))
    return ({k: v / shots for k, v in raw.items()} if shots else {}), shots


def check_expectations(
    values: dict[str, float],
    keys: list[str],
    exact: dict[str, dict],
    other: tuple[str, dict[str, float]] | None,
    tolerance: float,
) -> dict[str, Any]:
    """Compare each value with its exact value, or with `other`'s where the exact value is out of reach."""
    failures: list[str] = []
    errors: list[float] = []
    differences: list[float] = []
    for key in keys:
        value = values.get(key)
        if value is None or not math.isfinite(value):
            failures.append(f"{key}: missing or not finite")
            continue
        target = exact.get(key, {}).get("value")
        if target is not None:
            errors.append(abs(value - target))
            if errors[-1] > tolerance:
                failures.append(f"{key}: {value:.10g} against exact {target:.10g}")
        elif other is not None and other[1].get(key) is not None:
            differences.append(abs(value - other[1][key]))
            if differences[-1] > tolerance:
                failures.append(f"{key}: {value:.10g} against {other[0]} {other[1][key]:.10g}")
    tested = len(errors) + len(differences)
    result: dict[str, Any] = {
        "method": EXPVAL_METHOD,
        "statistics": len(keys),
        "tested": tested,
        "against_exact": len(errors),
        "against_simulator": len(differences),
        "agreement_simulator": other[0] if differences else None,
        "tolerance": tolerance,
        "max_abs_error": max(errors) if errors else None,
        "max_abs_difference": max(differences) if differences else None,
        "pass": bool(tested) and not failures,
        "failures": failures[:10],
    }
    if not tested and not failures:
        result.update(
            {
                "pass": None,
                "role": "untested",
                "reason": "no value had an exact reference or a second simulator's value to compare with",
            }
        )
    return result


def observed(bits: np.ndarray, counts: np.ndarray, singles, pairs) -> tuple[dict[str, float], int]:
    shots = int(counts.sum())
    out: dict[str, float] = {}
    if shots == 0:
        return out, 0
    for b in singles:
        out[f"b{b}"] = float(counts @ bits[:, b]) / shots
    for a, b in pairs:
        out[f"b{a}^b{b}"] = float(counts @ (bits[:, a] ^ bits[:, b])) / shots
    return out, shots


def min_detectable(statistics: int, shots: int, fallback_shots: int | None = None) -> float | None:
    """Smallest shift in a probability of 0.5 that crosses the per-statistic threshold.

    The exact test's shift with `fallback_shots` None, the two-sample z-test's otherwise; None
    below `MIN_SHOTS`, where nothing statistical is tested.
    """
    if statistics <= 0 or shots < MIN_SHOTS:
        return None
    z = float(norm.isf(FAMILY_ALPHA / statistics / 2))
    spread = 1 / shots + (1 / fallback_shots if fallback_shots else 0.0)
    return z * 0.5 * math.sqrt(spread)


def exact_statistics(instructions: list[tuple], singles, pairs) -> dict[str, dict]:
    return reference.statistics(instructions, singles, pairs)


def check_counts(
    obs: dict[str, float],
    shots: int,
    exact: dict[str, dict],
    fallback: tuple[dict[str, float], int] | None,
    expected_shots: int | None = None,
) -> dict[str, Any]:
    m = len(obs)
    p_star = FAMILY_ALPHA / m if m else 0.0
    smallest_p = 1.0
    failures: list[str] = []
    if expected_shots is not None and shots != expected_shots:
        failures.append(f"observed {shots} shots, the workload asks for {expected_shots}")
        smallest_p = 0.0
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
    statistical = exact_used - deterministic + fallback_used
    detectable = None
    if shots >= MIN_SHOTS and statistical:
        detectable = min_detectable(m, shots, fallback[1] if fallback_used else None)
    result = {
        "method": COUNTS_METHOD,
        "statistics": m,
        "tested": tested,
        "against_exact": exact_used,
        "against_simulator": fallback_used,
        "deterministic": deterministic,
        "p_threshold": p_star,
        "min_p_value": smallest_p,
        "min_detectable": detectable,
        "pass": bool(tested) and not failures,
        "failures": failures[:10],
    }
    if result["pass"] and detectable is not None and detectable > WEAK_DETECTABLE:
        result["role"] = "weakly-tested"
        result["reason"] = f"smallest detectable shift in a marginal is {detectable:.2g}, above {WEAK_DETECTABLE:g}"
    if shots < MIN_SHOTS and not failures:
        result.update(
            {
                "pass": None,
                "role": "too-few-shots",
                "reason": f"fewer than {MIN_SHOTS} shots: deterministic bits checked, statistics not tested",
            }
        )
    elif not tested and not failures:
        result.update(
            {
                "pass": None,
                "role": "untested",
                "reason": "no statistic had an exact value or a second simulator's counts to test against",
            }
        )
    return result

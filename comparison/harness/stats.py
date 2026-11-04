"""Summary statistics for timing samples.

Spread is the interquartile range of the warm samples. The 95% interval on a
median, and on a ratio of medians, is a percentile bootstrap with a fixed seed,
so re-aggregating the same raw file gives the same numbers. With five samples
the bootstrap interval is coarse (it cannot leave the sample range); it is
reported so a reader sees that, not as a precise bound.
"""

from __future__ import annotations

from typing import Any

import numpy as np

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42


def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"n": 0}
    a = np.asarray(samples, dtype=np.float64)
    out: dict[str, Any] = {
        "n": int(a.size),
        "median_ms": float(np.median(a)),
        "p25_ms": float(np.percentile(a, 25)),
        "p75_ms": float(np.percentile(a, 75)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "iqr_rel": float((np.percentile(a, 75) - np.percentile(a, 25)) / np.median(a)) if np.median(a) else None,
    }
    if a.size >= 3:
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        boots = np.median(rng.choice(a, size=(BOOTSTRAP_RESAMPLES, a.size), replace=True), axis=1)
        out["ci95_low_ms"] = float(np.percentile(boots, 2.5))
        out["ci95_high_ms"] = float(np.percentile(boots, 97.5))
    return out


def ratio(numerator: list[float], denominator: list[float]) -> dict[str, Any] | None:
    """Median of `numerator` over median of `denominator`, with a bootstrap 95% interval."""
    if not numerator or not denominator:
        return None
    a = np.asarray(numerator, dtype=np.float64)
    b = np.asarray(denominator, dtype=np.float64)
    point = float(np.median(a) / np.median(b))
    out: dict[str, Any] = {"ratio": point}
    if a.size >= 3 and b.size >= 3:
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        ma = np.median(rng.choice(a, size=(BOOTSTRAP_RESAMPLES, a.size), replace=True), axis=1)
        mb = np.median(rng.choice(b, size=(BOOTSTRAP_RESAMPLES, b.size), replace=True), axis=1)
        r = ma / mb
        out["ci95_low"] = float(np.percentile(r, 2.5))
        out["ci95_high"] = float(np.percentile(r, 97.5))
    return out

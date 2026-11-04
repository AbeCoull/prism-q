"""Reduce `observations.jsonl` to tables. Reads only the raw file; writes CSV and JSON.

Outputs, all in the results directory:

- `samples.csv`: one row per timed sample, the long form every figure can be redrawn from.
- `summary.csv`: one row per observation, failures included, with its status and reason.
- `derived.json`: ratios against PRISM-Q, dispatch regret, thread scaling, sampling
  costs and seed spread, each computed only from observations that ran and failed no
  check. Ratios carry `validated`, true when both outputs passed a test or one is the
  row's reference; thread counts below a row's top count share its circuit and are not
  validated separately.

Records written before the current validation rules are read under them: a dense
record that is its row's reference gets role `reference`, and a sampled record under
`validate.MIN_SHOTS` shots that passed gets role `too-few-shots`.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

from . import stats, validate

COMPARE_GROUPS = ("headline", "sampling", "seeds", "full")

SHORT = {
    "prismq": "PRISM-Q",
    "aer-statevector": "Aer statevector",
    "aer-statevector-f32": "Aer statevector (fp32)",
    "aer-automatic": "Aer automatic",
    "aer-stabilizer": "Aer stabilizer",
    "aer-mps": "Aer MPS",
    "qsim": "qsim (fp32)",
    "quest": "QuEST",
    "spinoza": "Spinoza",
    "qip": "qip",
    "stim": "Stim",
}


def label(name: str) -> str:
    base, _, option = name.partition(":")
    return SHORT.get(base, base) + (f" ({option})" if option else "")


def shots_label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M shots"
    if n >= 1000:
        return f"{n // 1000}k shots"
    return f"{n} shot" + ("s" if n != 1 else "")


def row_label(wl: dict) -> str:
    p = wl["params"]
    bits = [wl["family"], f"{wl['num_qubits']}q"]
    if "depth" in p and wl["family"] in ("rcs", "lowent"):
        bits.append(f"d{p['depth']}")
    if wl["family"] == "disjoint":
        bits.append(p.get("layout", ""))
    if p.get("measured") == "half":
        bits.append("half measured")
    if wl["mode"] == "counts":
        bits.append(shots_label(wl["shots"]))
    return " ".join(bits)


def family_order(wl: dict) -> tuple:
    order = ["rcs", "qft", "qv", "clifford", "lowent", "disjoint", "midcircuit", "sampling"]
    fam = wl["family"]
    return (
        order.index(fam) if fam in order else 99,
        wl["params"].get("depth", 0),
        wl["num_qubits"],
        wl["shots"],
        str(wl["params"].get("layout")),
    )


def load(results_dir: str) -> tuple[dict, list[dict]]:
    with open(os.path.join(results_dir, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = []
    with open(os.path.join(results_dir, "observations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(_normalize(json.loads(line)))
    return manifest, records


def _normalize(record: dict) -> dict:
    v = record.get("validation")
    if v and v.get("pass") is True and v.get("reference") == record["simulator"]:
        record["validation"] = {"pass": None, "role": "reference", "reason": "reference vector for this row"}
    elif v and v.get("pass") is True and 0 < record["workload"]["shots"] < validate.MIN_SHOTS:
        record["validation"] = {
            **v,
            "pass": None,
            "role": "too-few-shots",
            "reason": f"fewer than {validate.MIN_SHOTS} shots: statistics not tested",
        }
    if isinstance(record.get("internal_ms"), (int, float)):
        record["first_internal_ms"] = record.pop("internal_ms")
        record["internal_ms"] = []
    return record


def validated(record: dict) -> bool:
    v = record.get("validation") or {}
    return v.get("pass") is True or v.get("role") == "reference"


def valid(record: dict) -> bool:
    """The observation ran and its output failed no check: validated, the reference, or too few shots to test."""
    if record.get("status") != "ok" or not record.get("samples_ms"):
        return False
    v = record.get("validation") or {}
    return validated(record) or v.get("role") == "too-few-shots"


def _write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _flat(record: dict) -> dict[str, Any]:
    wl = record["workload"]
    st = record.get("stats") or {}
    v = record.get("validation") or {}
    ph = record.get("phases") or {}
    return {
        "run_id": record["run_id"],
        "group": record["group"],
        "workload_id": wl["workload_id"],
        "family": wl["family"],
        "num_qubits": wl["num_qubits"],
        "mode": wl["mode"],
        "shots": wl["shots"],
        "sha256": wl["sha256"],
        "simulator": record["simulator"],
        "precision": record["precision"],
        "threads": record["threads"],
        "version": record.get("version"),
        "status": record["status"],
        "reason": record.get("reason"),
        "n_samples": st.get("n"),
        "median_ms": st.get("median_ms"),
        "p25_ms": st.get("p25_ms"),
        "p75_ms": st.get("p75_ms"),
        "ci95_low_ms": st.get("ci95_low_ms"),
        "ci95_high_ms": st.get("ci95_high_ms"),
        "first_ms": ph.get("first_ms"),
        "build_ms": ph.get("build_ms"),
        "startup_ms": (ph.get("startup_ms") or 0) + (ph.get("import_ms") or 0) if "import_ms" in ph else None,
        "process_wall_ms": ph.get("process_wall_ms"),
        "plan_median_us": ph.get("plan_median_us"),
        "peak_rss_mb": record.get("peak_rss_mb"),
        "resolved_backend": record.get("resolved_backend"),
        "route": record.get("route"),
        "reproducible": record.get("reproducible"),
        "validation_pass": v.get("pass"),
        "tvd": v.get("tvd"),
        "max_abs_z": v.get("max_abs_z"),
        "min_p_value": v.get("min_p_value"),
        "validation_role": v.get("role"),
    }


def ratios(records: list[dict]) -> list[dict]:
    """Comparator median over PRISM-Q median per group, workload, thread count and run.

    Above 1 means PRISM-Q took less time.
    """
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in records:
        if r["group"] in COMPARE_GROUPS and valid(r):
            by_key[(r["group"], r["workload"]["workload_id"], r["threads"], r["run_id"])][r["simulator"]] = r
    out = []
    for (group, wid, threads, run_id), sims in sorted(by_key.items()):
        base = sims.get("prismq")
        if base is None:
            continue
        for name, r in sorted(sims.items()):
            if name == "prismq":
                continue
            ratio = stats.ratio(r["samples_ms"], base["samples_ms"])
            out.append(
                {
                    "workload_id": wid,
                    "family": r["workload"]["family"],
                    "num_qubits": r["workload"]["num_qubits"],
                    "group": group,
                    "run_id": run_id,
                    "threads": threads,
                    "comparator": name,
                    "validated": validated(r) and validated(base),
                    "precision": r["precision"],
                    "prismq_median_ms": base["stats"]["median_ms"],
                    "comparator_median_ms": r["stats"]["median_ms"],
                    **(ratio or {}),
                }
            )
    return out


def dispatch(records: list[dict]) -> list[dict]:
    rows = defaultdict(list)
    for r in records:
        if r["group"] == "dispatch":
            rows[r["workload"]["workload_id"]].append(r)
    out = []
    for wid, recs in sorted(rows.items()):
        auto = next((r for r in recs if r["simulator"] == "prismq"), None)
        forced = [r for r in recs if r["simulator"].startswith("prismq:")]
        ok = [r for r in forced if valid(r)]
        best = min(ok, key=lambda r: r["stats"]["median_ms"], default=None)
        entry: dict[str, Any] = {
            "workload_id": wid,
            "family": recs[0]["workload"]["family"],
            "num_qubits": recs[0]["workload"]["num_qubits"],
            "auto_backend": auto.get("resolved_backend") if auto else None,
            "auto_route": auto.get("route") if auto else None,
            "auto_median_ms": auto["stats"].get("median_ms") if auto and valid(auto) else None,
            "plan_median_us": (auto.get("phases") or {}).get("plan_median_us") if auto else None,
            "best_forced": best["simulator"] if best else None,
            "best_forced_median_ms": best["stats"]["median_ms"] if best else None,
            "forced": {
                r["simulator"]: (
                    {"median_ms": r["stats"]["median_ms"], "backend": r.get("resolved_backend")}
                    if valid(r)
                    else {"status": r["status"] if r["status"] != "ok" else "invalid", "reason": r.get("reason")}
                )
                for r in forced
            },
        }
        if entry["auto_median_ms"] and entry["best_forced_median_ms"]:
            regret = stats.ratio(auto["samples_ms"], best["samples_ms"])
            entry["auto_over_best"] = regret
        out.append(entry)
    return out


def scaling(records: list[dict]) -> list[dict]:
    rows = defaultdict(dict)
    for r in records:
        if r["group"] == "threads" and valid_or_unvalidated(r):
            rows[(r["workload"]["workload_id"], r["simulator"])][r["threads"]] = r["stats"]["median_ms"]
    headline = {}
    for r in records:
        if r["group"] == "headline" and valid(r):
            headline[(r["workload"]["workload_id"], r["simulator"], r["threads"])] = r["stats"]["median_ms"]
    out = []
    for (wid, sim), by_threads in sorted(rows.items()):
        for (hw, hs, ht), median in headline.items():
            if hw == wid and hs == sim and ht not in by_threads:
                by_threads[ht] = median
        t1 = by_threads.get(1)
        for t, median in sorted(by_threads.items()):
            out.append(
                {
                    "workload_id": wid,
                    "simulator": sim,
                    "threads": t,
                    "median_ms": median,
                    "speedup_vs_1": t1 / median if t1 else None,
                    "efficiency": t1 / median / t if t1 else None,
                }
            )
    return out


def valid_or_unvalidated(r: dict) -> bool:
    if r.get("status") != "ok" or not r.get("samples_ms"):
        return False
    return r.get("validation") is None or valid(r)


def sampling_costs(records: list[dict]) -> list[dict]:
    """Fixed and per-shot cost from the 1-shot and largest-shot rows of each sampling circuit."""
    groups = defaultdict(dict)
    for r in records:
        if r["group"] == "sampling" and valid(r):
            wl = r["workload"]
            key = (wl["num_qubits"], wl["params"].get("measured"), r["simulator"])
            groups[key][wl["shots"]] = r["stats"]["median_ms"]
    out = []
    for (n, measured, sim), by_shots in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        shots = sorted(by_shots)
        entry = {"num_qubits": n, "measured": measured, "simulator": sim, "by_shots": by_shots}
        if len(shots) >= 2 and shots[0] == 1:
            top = shots[-1]
            entry["one_shot_ms"] = by_shots[1]
            entry["per_million_shots_ms"] = (by_shots[top] - by_shots[1]) / (top - 1) * 1e6
        out.append(entry)
    return out


def seed_spread(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        if r["group"] in ("seeds", "headline") and valid(r):
            wl = r["workload"]
            if "seed" not in wl["params"] or wl["num_qubits"] != 20:
                continue
            if wl["family"] == "rcs" and wl["params"].get("depth") != 10:
                continue
            groups[(wl["family"], r["simulator"])].append(r)
    out = []
    for (family, sim), recs in sorted(groups.items()):
        if len(recs) < 2:
            continue
        medians = [x["stats"]["median_ms"] for x in recs]
        within = [x["stats"].get("iqr_rel") for x in recs if x["stats"].get("iqr_rel") is not None]
        out.append(
            {
                "family": family,
                "simulator": sim,
                "seeds": len(recs),
                "medians_ms": medians,
                "across_seed_rel_range": (max(medians) - min(medians)) / float(np.median(medians)),
                "within_seed_rel_iqr_median": float(np.median(within)) if within else None,
            }
        )
    return out


def write(results_dir: str) -> dict[str, Any]:
    manifest, records = load(results_dir)
    samples = []
    for r in records:
        base = _flat(r)
        for i, value in enumerate(r.get("samples_ms") or []):
            samples.append({**base, "phase": "warm", "iteration": i, "value_ms": value})
        first = (r.get("phases") or {}).get("first_ms")
        if first is not None:
            samples.append({**base, "phase": "cold", "iteration": 0, "value_ms": first})
    sample_fields = [
        "run_id",
        "group",
        "workload_id",
        "family",
        "num_qubits",
        "mode",
        "shots",
        "simulator",
        "precision",
        "threads",
        "phase",
        "iteration",
        "value_ms",
    ]
    _write_csv(os.path.join(results_dir, "samples.csv"), samples, sample_fields)
    flat = [_flat(r) for r in records]
    _write_csv(os.path.join(results_dir, "summary.csv"), flat, list(flat[0].keys()) if flat else [])

    status_counts: dict[str, int] = defaultdict(int)
    for r in records:
        status_counts[r["status"]] += 1
    validation_failures = [
        {"workload_id": r["workload"]["workload_id"], "simulator": r["simulator"], "validation": r["validation"]}
        for r in records
        if (r.get("validation") or {}).get("pass") is False and (r.get("validation") or {}).get("pass_required", True)
    ]
    derived = {
        "run_id": manifest["run_id"],
        "profile": manifest["profile"],
        "observations": len(records),
        "status_counts": dict(status_counts),
        "validation_failures": validation_failures,
        "ratios": ratios(records),
        "dispatch": dispatch(records),
        "scaling": scaling(records),
        "sampling": sampling_costs(records),
        "seeds": seed_spread(records),
    }
    with open(os.path.join(results_dir, "derived.json"), "w", encoding="utf-8") as handle:
        json.dump(derived, handle, indent=2)
    print(f"aggregated {len(records)} observations into {results_dir}: {dict(status_counts)}")
    return derived

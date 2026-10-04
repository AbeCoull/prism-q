"""Markdown rendering of a results file.

Generated, never hand edited. Rows where PRISM-Q is slower render the same as
rows where it is faster; a table that shows one direction only is not evidence.
"""

from __future__ import annotations

from typing import Any

FRAMING = (
    "Comparative performance measurements against commonly used quantum simulators. "
    "These results are intended to make performance characteristics reproducible and "
    "transparent across representative workloads, not to rank projects: each simulator "
    "makes different trade-offs, and a ratio here describes one workload on one host "
    "under the controls listed below."
)


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 1.0:
        return f"{value * 1000:.0f} us"
    if value >= 1000.0:
        return f"{value / 1000:.2f} s"
    return f"{value:.1f} ms"


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def _timing(row: dict[str, Any], name: str) -> float | None:
    entry = row["timings"].get(name)
    if not entry or "median_ms" not in entry:
        return None
    return entry["median_ms"]


def render(results: dict[str, Any]) -> str:
    run = results["run"]
    prov = results["provenance"]
    host = prov["host"]
    tool = prov["toolchain"]
    adapters = results["adapters"]
    comparators = [n for n in adapters if n != "prismq"]

    lines: list[str] = []
    lines.append("# Comparative measurements")
    lines.append("")
    lines.append(FRAMING)
    lines.append("")
    lines.append(
        "Every simulator replays the same gate list for each circuit, built from the "
        "`prism_q::circuits` generators and hashed so a rerun can prove it did the same work. "
        f"The timed region is: {run['timed_region']}. "
        f"Ratios are the comparator's median over PRISM-Q's median, so a ratio above 1.00x "
        f"means PRISM-Q finished sooner and below 1.00x means the comparator did; a ratio "
        f"within {run['tie_band_pct']:.0f}% of 1.00x is reported as within band, because "
        "cross-process timing noise on one host is of that order."
    )
    lines.append("")

    lines.append("## Host and versions")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| CPU | {host['cpu_model']} |")
    lines.append(f"| Cores | {host['physical_cores']} physical, {host['logical_cores']} logical |")
    lines.append(f"| RAM | {host['ram_gb']} GB |")
    lines.append(f"| OS | {host['os']} |")
    lines.append(f"| rustc (PRISM-Q) | {tool['rustc']}, features `{tool['cargo_features']}`, profile release |")
    lines.append(f"| rustc (Spinoza, qip) | {tool['rustc_peers']}, profile release |")
    lines.append(f"| C++ compiler (QuEST) | {tool['cxx_compiler']} |")
    lines.append(f"| Python | {prov['python']['version']} |")
    lines.append(f"| Commit | {tool['git_sha']}{' (dirty tree)' if tool['git_dirty'] else ''} |")
    lines.append(f"| Threads | {run['threads']} on every simulator |")
    lines.append(f"| Iterations | {run['iterations']} timed per circuit after one warmup |")
    lines.append("")
    lines.append("| Simulator | Version | Settings |")
    lines.append("| --- | --- | --- |")
    for name, entry in adapters.items():
        lines.append(f"| {entry['label']} | {entry.get('version') or 'n/a'} | {entry.get('settings', '')} |")
    lines.append("")
    overheads = results.get("fixed_overhead_ms", {})
    if overheads:
        lines.append(
            "Per-call overhead of driving a comparator from Python on a one-gate circuit, "
            "recorded so small rows can be read correctly: "
            + ", ".join(f"{name} {_fmt_ms(value)}" for name, value in overheads.items())
            + "."
        )
        lines.append("")

    lines.append("## Results")
    lines.append("")
    header = ["Circuit", "Qubits", "Gates", "PRISM-Q"]
    for name in comparators:
        header += [name, "ratio"]
    header.append("max TVD")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    has_excluded = False
    for row in results["results"]:
        counted = row["num_qubits"] >= run["headline_min_qubits"]
        has_excluded = has_excluded or not counted
        marker = "" if counted else " \\*"
        cells = [
            row["benchmark"],
            str(row["num_qubits"]),
            str(row["num_operations"]),
            _fmt_ms(_timing(row, "prismq")),
        ]
        for name in comparators:
            timing = row["timings"].get(name, {})
            if "error" in timing:
                cells.append("not run")
                cells.append("n/a")
                continue
            cells.append(_fmt_ms(_timing(row, name)))
            cells.append(_fmt_ratio(row["ratio_vs_prismq"].get(name)) + marker)
        tvds = [e["tvd"] for e in row["equivalence"].values() if e.get("tvd") is not None]
        cells.append(f"{max(tvds):.1e}" if tvds else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if has_excluded:
        lines.append(
            f"\\* Below {run['headline_min_qubits']} qubits a run is comparable to the per-call "
            "overhead of a comparator, so these ratios describe framework cost rather than "
            "simulation and are left out of the summary."
        )
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"Counted over circuits with at least {run['headline_min_qubits']} qubits. "
        "A row where the comparator did not run is not counted."
    )
    lines.append("")
    lines.append("| Comparator | PRISM-Q sooner | Within band | Comparator sooner | Median ratio | Range |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, stats in results["summary"]["per_comparator"].items():
        rng = (
            f"{_fmt_ratio(stats['min_ratio'])} to {_fmt_ratio(stats['max_ratio'])}"
            if stats["min_ratio"] is not None
            else "n/a"
        )
        lines.append(
            f"| {name} | {stats['faster']} | {stats['within_band']} | {stats['slower']} | "
            f"{_fmt_ratio(stats['median_ratio'])} | {rng} |"
        )
    lines.append("")

    failures = results["summary"]["equivalence_failures"]
    lines.append("## Equivalence")
    lines.append("")
    if failures:
        lines.append(
            f"{len(failures)} circuit runs produced a probability vector differing from "
            f"{run['reference_simulator']} by more than "
            f"{run['equivalence_tolerance_tvd']:.0e} total variation distance; their timings "
            "are shown but not counted:"
        )
        lines.append("")
        for item in failures:
            lines.append(
                f"- {item['benchmark']} at {item['num_qubits']} qubits, {item['simulator']}, "
                f"TVD {item['tvd']:.2e}"
            )
    else:
        lines.append(
            f"Every simulator reproduced the reference probability vector "
            f"({run['reference_simulator']}) to within "
            f"{run['equivalence_tolerance_tvd']:.0e} total variation distance on every circuit. "
            "The tolerance separates a wrong answer, which lands near 1e-1, from rounding; "
            "the max TVD column shows the measured distance, and a value near 1e-6 is the "
            "single-precision comparator."
        )
    lines.append("")

    errors = results["summary"]["errors"]
    if errors:
        lines.append("## Rows not run")
        lines.append("")
        lines.append("Listed so the selection can be audited; a row is never dropped for how it performs.")
        lines.append("")
        for item in errors:
            lines.append(
                f"- {item['benchmark']} at {item['num_qubits']} qubits, {item['simulator']}: {item['error']}"
            )
        lines.append("")

    skipped = results["corpus"]["skipped"]
    if skipped:
        lines.append("## Circuits not generated")
        lines.append("")
        lines.append("| Circuit | Qubits | Reason |")
        lines.append("| --- | --- | --- |")
        for item in skipped:
            lines.append(f"| {item['benchmark']} | {item['num_qubits']} | {item['reason']} |")
        lines.append("")

    return "\n".join(lines)

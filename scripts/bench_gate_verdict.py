"""Combine the wall-clock gate with the callgrind twins into the benchmark verdict.

Usage: python scripts/bench_gate_verdict.py --rows scripts/bench_gate_rows.txt
           --results bench_results --iai-home iai-home [--corroborate PCT]

`--results` holds what scripts/bench_ci.sh wrote: `ci-outcome.txt` (pass, fail or
invalid) and `ci-regressed.txt`, the rows behind a fail. A regressed row blocks when its
policy in the rows file is `wall`, or when it is `corroborate` and its twin's instruction
count rose by at least `--corroborate` percent or has no base to compare against. A
`corroborate` row whose twin held flat is a warning: code layout and host noise move wall
clock without changing the instructions executed. `report` rows never block, and a row the
rows file does not list counts as `wall`. Exit 1 when anything blocks or the A/B gave no
valid result; a missing outcome means nothing was measured.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iai_summary import read_row

POLICIES = ("wall", "corroborate", "report", "ir")
TWIN_TARGET = "iai_twins"


def read_rows(path):
    """Map each Criterion row to its (policy, twin) pair from the tab-separated rows file."""
    rows = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 3 or fields[1] not in POLICIES:
            raise SystemExit(f"{path}:{number}: expected `row policy twin`, got `{line}`")
        row, policy, twin = fields
        rows[row] = (policy, None if twin == "-" else twin)
    return rows


def twin_changes(home):
    """Percent change in Ir for every twin with a base, None for a twin counted without one."""
    changes = {}
    if home is None or not home.is_dir():
        return changes
    for path in sorted(home.rglob("summary.json")):
        if TWIN_TARGET not in path.parts:
            continue
        name, old, new, _ = read_row(path)
        changes[name] = None if old in (None, 0) else (new - old) / old * 100.0
    return changes


def decide(regressed, rows, changes, corroborate):
    """One (row, policy, twin, Ir change, verdict) tuple per regressed row."""
    out = []
    for row in regressed:
        policy, twin = rows.get(row, ("wall", None))
        change = changes.get(twin) if twin else None
        if policy == "report":
            verdict = "report only"
        elif policy != "corroborate":
            verdict = "blocks"
        elif twin not in changes or change is None:
            verdict = "blocks (twin has no base to compare)"
        elif change >= corroborate:
            verdict = "blocks"
        else:
            verdict = "warning: instruction count flat, likely layout or host noise"
        out.append((row, policy, twin, change, verdict))
    return out


def render(outcome, decisions, corroborate):
    if outcome is None:
        return ["Nothing measured: the wall-clock gate wrote no outcome."]
    if outcome == "pass":
        return ["**Benchmark verdict**: PASS. No row regressed on wall clock in both runs."]
    if outcome == "invalid":
        return ["**Benchmark verdict**: FAIL. The wall-clock A/B gave no valid result; see its report."]
    lines = [
        "| Row | Policy | Twin | Twin Ir change | Verdict |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row, policy, twin, change, verdict in decisions:
        shown = "n/a" if change is None else f"{change:+.3f}%"
        lines.append(f"| `{row}` | {policy} | {f'`{twin}`' if twin else '-'} | {shown} | {verdict} |")
    blocked = [d for d in decisions if d[4].startswith("blocks")]
    lines.append("")
    if blocked:
        lines.append(f"**Benchmark verdict**: FAIL. {len(blocked)} row(s) block.")
    else:
        lines.append(
            "**Benchmark verdict**: PASS with warnings. Every wall-clock regression is on a row "
            f"whose twin's instruction count moved less than +{corroborate}%, or on a report-only row."
        )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--iai-home", type=Path)
    parser.add_argument("--corroborate", type=float, default=0.5)
    args = parser.parse_args()

    rows = read_rows(args.rows)
    outcome_file = args.results / "ci-outcome.txt"
    outcome = outcome_file.read_text(encoding="utf-8").strip() if outcome_file.exists() else None
    regressed_file = args.results / "ci-regressed.txt"
    regressed = regressed_file.read_text(encoding="utf-8").split() if regressed_file.exists() else []
    decisions = decide(regressed, rows, twin_changes(args.iai_home), args.corroborate) if outcome == "fail" else []

    print("\n".join(render(outcome, decisions, args.corroborate)))
    failed = outcome == "invalid" or any(d[4].startswith("blocks") for d in decisions)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Render the iai-callgrind summary.json files under one home directory as a Markdown table.

Usage: python scripts/iai_summary.py <iai-home> [--threshold PCT]

One row per benchmark with the callgrind instruction count (Ir) of the head run, the
baseline it was compared against when one was present, and the change. The verdict
comes from the runner's exit code; this script only reports.
"""

import argparse
import json
from pathlib import Path


def metric(value):
    return value.get("Int", value.get("Float"))


def read_row(path):
    summary = json.loads(path.read_text(encoding="utf-8"))
    name = f"{summary['function_name']}/{summary.get('id') or ''}".rstrip("/")
    for profile in summary["profiles"]:
        if profile["tool"] != "Callgrind":
            continue
        total = profile["summaries"]["total"]
        ir = total["summary"]["Callgrind"]["Ir"]["metrics"]
        if "Both" in ir:
            new, old = (metric(v) for v in ir["Both"])
        else:
            new, old = metric(ir.get("Left", ir.get("Right"))), None
        return name, old, new, bool(total["regressions"])
    raise ValueError(f"{path}: no callgrind profile")


def render(rows, threshold):
    compared = any(old is not None for _, old, _, _ in rows)
    lines = []
    if compared:
        lines.append("| Row | Base Ir | Head Ir | Change | |")
        lines.append("|-----|--------:|--------:|-------:|-|")
    else:
        lines.append("| Row | Ir |")
        lines.append("|-----|---:|")
    for name, old, new, regressed in sorted(rows):
        if not compared:
            lines.append(f"| `{name}` | {new:,} |")
            continue
        if old in (None, 0):
            lines.append(f"| `{name}` | n/a | {new:,} | new | |")
            continue
        change = (new - old) / old * 100.0
        flag = "FAIL" if regressed else ""
        lines.append(f"| `{name}` | {old:,} | {new:,} | {change:+.3f}% | {flag} |")
    if compared:
        lines.append("")
        lines.append(f"Threshold: +{threshold}% Ir on any row.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("home", type=Path)
    parser.add_argument("--threshold", default="5.0")
    args = parser.parse_args()
    paths = sorted(args.home.rglob("summary.json"))
    if not paths:
        raise SystemExit(f"no summary.json under {args.home}")
    print(render([read_row(p) for p in paths], args.threshold))


if __name__ == "__main__":
    main()

import argparse
from collections import defaultdict
from pathlib import Path
import re
import subprocess


def read_lcov(path, root):
    files = defaultdict(dict)
    current = None
    for record in path.read_text(encoding="utf-8").splitlines():
        if record.startswith("SF:"):
            source = Path(record[3:].replace("\\", "/"))
            source = (root / source).resolve()
            current = source.relative_to(root).as_posix() if source.is_relative_to(root) else None
            if current and not current.startswith("src/"):
                current = None
        elif record.startswith("DA:") and current:
            line, hits, *_ = record[3:].split(",")
            number = int(line)
            files[current][number] = files[current].get(number, 0) + int(hits)
        elif record == "end_of_record":
            current = None
    if not any(files.values()):
        raise ValueError("LCOV contains no src/ line records for this checkout")
    return dict(files)


def changed_lines(diff):
    changes = defaultdict(set)
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/src/") else None
        elif current and line.startswith("@@ "):
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                first = int(match[1])
                count = int(match[2]) if match[2] is not None else 1
                changes[current].update(range(first, first + count))
    return dict(changes)


def line_ranges(lines):
    groups = []
    for line in sorted(lines):
        if groups and groups[-1][1] + 1 == line:
            groups[-1][1] = line
        else:
            groups.append([line, line])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in groups)


def render_report(files, changes=None):
    subsystems = defaultdict(lambda: [0, 0])
    for name, lines in files.items():
        parts = name.split("/")
        subsystem = "/".join(parts[:3] if parts[1] == "backend" else parts[:2])
        subsystems[subsystem][0] += sum(hits > 0 for hits in lines.values())
        subsystems[subsystem][1] += len(lines)
    report = [
        "## Source line coverage",
        "",
        "Instrumented src/ lines, including inline tests. Feature-excluded files are not measured.",
        "Line hits do not establish branch coverage or assertion strength.",
        "",
        "| Subsystem | Hit / measured | Coverage |",
        "| --- | ---: | ---: |",
    ]
    for name, (hit, total) in sorted(subsystems.items(), key=lambda item: item[1][0] - item[1][1]):
        report.append(f"| {name} | {hit} / {total} | {100 * hit / total:.1f}% |")
    report.extend([
        "", "### Files with most uncovered lines", "",
        "| File | Uncovered | First 30 uncovered lines |", "| --- | ---: | --- |",
    ])
    gaps = {name: sorted(line for line, hits in lines.items() if hits == 0) for name, lines in files.items()}
    for name in sorted(gaps, key=lambda name: (-len(gaps[name]), name))[:20]:
        if gaps[name]:
            report.append(f"| {name} | {len(gaps[name])} | {line_ranges(gaps[name][:30])} |")
    if changes is not None:
        report.extend([
            "", "### Changed lines", "",
            "Unmapped lines may be comments, non-executable declarations, or excluded code.",
            "",
            "| File | Hit / measured changes | Uncovered changes | Unmapped changes |",
            "| --- | ---: | --- | ---: |",
        ])
        for name, lines in sorted(changes.items()):
            if not lines:
                continue
            measured = lines & files.get(name, {}).keys()
            uncovered = {line for line in measured if files[name][line] == 0}
            report.append(
                f"| {name} | {len(measured) - len(uncovered)} / {len(measured)} | "
                f"{line_ranges(uncovered) or '-'} | {len(lines - measured)} |"
            )
        if not any(changes.values()):
            report.append("No changed src/ lines.")
    return "\n".join(report) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize LCOV gaps by subsystem and changed line.")
    parser.add_argument("lcov", type=Path)
    parser.add_argument("--base", help="Git revision to compare against the current checkout.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    changes = None
    if args.base:
        diff = subprocess.check_output(
            ["git", "-c", "core.quotePath=false", "diff", "--no-ext-diff", "--no-renames",
             "--unified=0", args.base, "--", "src"],
            cwd=root, text=True, encoding="utf-8",
        )
        changes = changed_lines(diff)
    print(render_report(read_lcov(args.lcov, root), changes), end="")


if __name__ == "__main__":
    main()

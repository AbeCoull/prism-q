"""Tests for scripts/bench_gate_verdict.py and the rows file it reads."""

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import bench_gate_verdict as verdict

ROWS = """\
w/1 wall -
c/1 corroborate twin_c/q1
c/2 corroborate twin_d/q2
c/3 corroborate twin_missing/q3
r/1 report -
"""


def write_twin(home, name, old, new):
    function, bench_id = name.split("/")
    directory = home / "prism-q" / verdict.TWIN_TARGET / "twins" / f"{function}.{bench_id}"
    directory.mkdir(parents=True)
    metrics = {"Both": [{"Int": new}, {"Int": old}]} if old is not None else {"Left": {"Int": new}}
    summary = {
        "function_name": function,
        "id": bench_id,
        "profiles": [
            {
                "tool": "Callgrind",
                "summaries": {"total": {"summary": {"Callgrind": {"Ir": {"metrics": metrics}}}, "regressions": []}},
            }
        ],
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


class VerdictTests(unittest.TestCase):
    def run_verdict(self, outcome, regressed=(), twins=()):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = tmp / "rows.txt"
            rows.write_text(ROWS, encoding="utf-8")
            results = tmp / "results"
            results.mkdir()
            if outcome is not None:
                (results / "ci-outcome.txt").write_text(outcome + "\n", encoding="utf-8")
            if regressed:
                (results / "ci-regressed.txt").write_text("\n".join(regressed) + "\n", encoding="utf-8")
            home = tmp / "iai"
            for name, old, new in twins:
                write_twin(home, name, old, new)
            proc = subprocess.run(
                [sys.executable, str(SCRIPTS / "bench_gate_verdict.py"), "--rows", str(rows)]
                + ["--results", str(results), "--iai-home", str(home)],
                capture_output=True,
                text=True,
                check=False,
            )
            return proc.returncode, proc.stdout

    def test_nothing_measured_passes(self):
        self.assertEqual(self.run_verdict(None)[0], 0)

    def test_pass_passes(self):
        code, out = self.run_verdict("pass")
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)

    def test_invalid_fails(self):
        self.assertEqual(self.run_verdict("invalid")[0], 1)

    def test_wall_row_blocks(self):
        code, out = self.run_verdict("fail", ["w/1"])
        self.assertEqual(code, 1)
        self.assertIn("| `w/1` | wall | - | n/a | blocks |", out)

    def test_unlisted_row_blocks(self):
        self.assertEqual(self.run_verdict("fail", ["x/9"])[0], 1)

    def test_corroborated_row_blocks(self):
        code, out = self.run_verdict("fail", ["c/1"], [("twin_c/q1", 1000, 1020)])
        self.assertEqual(code, 1)
        self.assertIn("+2.000%", out)

    def test_flat_twin_warns(self):
        code, out = self.run_verdict("fail", ["c/1"], [("twin_c/q1", 1000, 1001)])
        self.assertEqual(code, 0)
        self.assertIn("warning", out)
        self.assertIn("PASS with warnings", out)

    def test_twin_without_base_blocks(self):
        code, out = self.run_verdict("fail", ["c/2"], [("twin_d/q2", None, 1000)])
        self.assertEqual(code, 1)
        self.assertIn("no base to compare", out)

    def test_missing_twin_blocks(self):
        self.assertEqual(self.run_verdict("fail", ["c/3"])[0], 1)

    def test_report_row_never_blocks(self):
        code, out = self.run_verdict("fail", ["r/1"])
        self.assertEqual(code, 0)
        self.assertIn("report only", out)

    def test_one_blocking_row_fails_the_set(self):
        code, _ = self.run_verdict("fail", ["c/1", "w/1"], [("twin_c/q1", 1000, 1000)])
        self.assertEqual(code, 1)


class RowsFileTests(unittest.TestCase):
    def test_every_twin_exists_in_its_bench_target(self):
        rows = verdict.read_rows(SCRIPTS / "bench_gate_rows.txt")
        targets = {
            "corroborate": (REPO / "benches" / "iai_twins.rs").read_text(encoding="utf-8"),
            "ir": (REPO / "benches" / "iai.rs").read_text(encoding="utf-8"),
        }
        for row, (policy, twin) in rows.items():
            if policy not in targets:
                self.assertIsNone(twin, row)
                continue
            self.assertIsNotNone(twin, row)
            function, bench_id = twin.split("/")
            pattern = rf"#\[bench::{re.escape(bench_id)}\(.*\n(#\[bench::.*\n)*fn {re.escape(function)}\("
            self.assertRegex(targets[policy], pattern, f"{row}: twin {twin}")

    def test_rows_are_unique_circuits_rows(self):
        text = (SCRIPTS / "bench_gate_rows.txt").read_text(encoding="utf-8")
        names = [line.split()[0] for line in text.splitlines() if line.strip() and not line.startswith("#")]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_mutations
import coverage_report


class CoverageReportTests(unittest.TestCase):
    def test_merge_records_and_ignore_files_outside_src(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report = root / "lcov.info"
            report.write_text(
                "SF:src/a.rs\nDA:2,0\nDA:3,0\nend_of_record\n"
                f"SF:{root / 'src/a.rs'}\nDA:2,3\nDA:4,1\nend_of_record\n"
                "SF:src\\b.rs\nDA:8,0\nend_of_record\n"
                "SF:tests/a.rs\nDA:1,0\nend_of_record\n"
                "SF:../outside.rs\nDA:1,0\nend_of_record\n", encoding="utf-8",
            )
            self.assertEqual(
                coverage_report.read_lcov(report, root),
                {"src/a.rs": {2: 3, 3: 0, 4: 1}, "src/b.rs": {8: 0}},
            )

    def test_empty_or_wrong_checkout_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report = root / "lcov.info"
            report.write_text("SF:tests/only.rs\nDA:1,1\nend_of_record\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no src/"):
                coverage_report.read_lcov(report, root)

    def test_changed_lines_include_additions_and_exclude_deletions(self):
        diff = (
            "+++ b/src/a.rs\n@@ -2,2 +2,3 @@\n"
            "@@ -10 +11 @@\n@@ -20,3 +20,0 @@\n"
            "+++ /dev/null\n@@ -1 +0,0 @@\n"
            "+++ b/tests/a.rs\n@@ -0,0 +1,8 @@\n"
        )
        self.assertEqual(coverage_report.changed_lines(diff), {"src/a.rs": {2, 3, 4, 11}})

    def test_unmapped_changes_are_not_reported_as_covered(self):
        report = coverage_report.render_report(
            {"src/a.rs": {1: 2, 2: 0}}, {"src/a.rs": {1, 2, 3}, "src/new.rs": {1, 2}}
        )
        self.assertIn("| src/a.rs | 1 / 2 | 2 | 1 |", report)
        self.assertIn("| src/new.rs | 0 / 0 | - | 2 |", report)


class MutationTests(unittest.TestCase):
    def test_baseline_receives_fresh_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.rs"
            source.write_text("original", encoding="utf-8")
            os.utime(source, (1, 1))
            output = root / "output"
            output.mkdir()

            def baseline(command, workspace, env, log, timeout):
                copied = workspace / source.name
                self.assertEqual(copied.read_text(encoding="utf-8"), "original")
                self.assertGreater(copied.stat().st_mtime_ns, source.stat().st_mtime_ns)
                return 100

            with patch.object(check_mutations.subprocess, "check_output", return_value=b"sample.rs\0"), \
                 patch.object(check_mutations, "run_command", side_effect=baseline):
                report = check_mutations.run_mutations(root, output, 1)
            self.assertEqual(report["status"], "baseline_failed")

    def test_only_test_failure_counts_as_caught(self):
        for code, expected in [(100, "caught"), (0, "survived"), (101, "error"),
                               (4, "error"), (-9, "error"), (None, "timeout")]:
            with self.subTest(code=code):
                self.assertEqual(check_mutations.mutation_status(code), expected)

    def test_restore_source_after_every_outcome(self):
        case = check_mutations.Mutation("sample", "sample.rs", "before", "after", "suite", "test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / case.path
            original = b"before\r\n"
            for code in [100, 0, 101, None]:
                with self.subTest(code=code):
                    source.write_bytes(original)
                    def run(*args):
                        self.assertEqual(source.read_bytes(), b"after\n")
                        return code
                    with patch.object(check_mutations, "run_command", side_effect=run):
                        check_mutations.check_mutation(case, root, {}, root, 1)
                    self.assertEqual(source.read_bytes(), original)
            with patch.object(check_mutations, "run_command", side_effect=OSError("launch failed")):
                with self.assertRaises(OSError):
                    check_mutations.check_mutation(case, root, {}, root, 1)
            self.assertEqual(source.read_bytes(), original)

    def test_source_drift_fails_without_running_cargo(self):
        case = check_mutations.Mutation("sample", "sample.rs", "before", "after", "suite", "test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in ["missing", "before before"]:
                with self.subTest(source=source):
                    (root / case.path).write_text(source, encoding="utf-8")
                    with patch.object(check_mutations, "run_command") as run:
                        result = check_mutations.check_mutation(case, root, {}, root, 1)
                    self.assertEqual(result["status"], "source_mismatch")
                    run.assert_not_called()

    def test_failed_baseline_prevents_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(check_mutations.subprocess, "check_output", return_value=b""), \
                 patch.object(check_mutations, "run_command", return_value=100), \
                 patch.object(check_mutations, "check_mutation") as mutate:
                report = check_mutations.run_mutations(root, root, 1)
            self.assertEqual(report["status"], "baseline_failed")
            mutate.assert_not_called()


if __name__ == "__main__":
    unittest.main()

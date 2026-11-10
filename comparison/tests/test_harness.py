"""Unit tests for the matrix harness: reference model, validation, statistics, workloads.

comparison/.venv/Scripts/python -m unittest discover -s comparison/tests -v
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matrix  # noqa: E402
from harness import aggregate, pages, profiles, reference, runners, stats, validate, workloads  # noqa: E402
from harness.worker import counts_text_from_ints  # noqa: E402


def _dense_probabilities(text: str) -> np.ndarray:
    """Full statevector of a unitary gate list, little-endian index, for cross-checks."""
    n, _, instructions = workloads.parse(text)
    psi = np.zeros((2,) * n, dtype=np.complex128)
    psi[(0,) * n] = 1.0
    for inst in instructions:
        if inst[0] == "gate":
            psi = reference._apply(psi, inst[1], list(inst[2]), inst[3])
    # Axis q is qubit q; flatten so qubit 0 is the least significant bit.
    return np.abs(np.transpose(psi, list(range(n))[::-1]).reshape(-1)) ** 2


class ReferenceTests(unittest.TestCase):
    def test_bell_parity_is_deterministic(self) -> None:
        text = "qubits 2\nclbits 2\nh 0\ncx 0 1\nmeasure 0 0\nmeasure 1 1\n"
        _, _, inst = workloads.parse(text)
        stats_ = reference.statistics(inst, [0, 1], [(0, 1)])
        self.assertAlmostEqual(stats_["b0"]["p"], 0.5)
        self.assertAlmostEqual(stats_["b0^b1"]["p"], 0.0)

    def test_cone_marginals_match_dense_statevector(self) -> None:
        wl = workloads.rcs(8, 4, seed=7)
        probs = _dense_probabilities(wl.text)
        measured = wl.text + "".join(f"measure {q} {q}\n" for q in range(8))
        measured = measured.replace("qubits 8\n", "qubits 8\nclbits 8\n", 1)
        _, _, inst = workloads.parse(measured)
        got = reference.statistics(inst, list(range(8)), [(q, q + 1) for q in range(7)])
        index = np.arange(probs.size)
        for q in range(8):
            expected = float(probs[(index >> q) & 1 == 1].sum())
            self.assertAlmostEqual(got[f"b{q}"]["p"], expected, places=12)
        for q in range(7):
            parity = ((index >> q) ^ (index >> (q + 1))) & 1
            self.assertAlmostEqual(got[f"b{q}^b{q + 1}"]["p"], float(probs[parity == 1].sum()), places=12)

    def test_feed_forward_cone_matches_whole_circuit(self) -> None:
        wl = workloads.midcircuit(6, 100, rounds=2, seed=11)
        _, clbits, inst = workloads.parse(wl.text)
        # Asking for every bit at once keeps the whole circuit in the cone.
        joint = reference.bit_distribution(inst, list(range(clbits)))
        for b in range(clbits):
            marginal = sum(p for key, p in joint.items() if key[b] == 1)
            self.assertAlmostEqual(reference.bit_distribution(inst, [b]).get((1,), 0.0), marginal, places=12)

    def test_reset_returns_qubit_to_zero(self) -> None:
        text = "qubits 1\nclbits 1\nh 0\nreset 0\nmeasure 0 0\n"
        _, _, inst = workloads.parse(text)
        self.assertAlmostEqual(reference.bit_distribution(inst, [0]).get((1,), 0.0), 0.0)

    def test_conditional_flip(self) -> None:
        text = "qubits 2\nclbits 2\nh 0\nmeasure 0 0\nif 0 1 x 1\nmeasure 1 1\n"
        _, _, inst = workloads.parse(text)
        self.assertAlmostEqual(reference.statistics(inst, [], [(0, 1)])["b0^b1"]["p"], 0.0)

    def test_wide_cone_is_refused(self) -> None:
        wl = workloads.rcs(24, 30, seed=3)
        text = wl.text.replace("qubits 24\n", "qubits 24\nclbits 1\n", 1) + "measure 12 0\n"
        _, _, inst = workloads.parse(text)
        with self.assertRaises(reference.ConeTooWide):
            reference.bit_distribution(inst, [0])


class ValidationTests(unittest.TestCase):
    def _counts_file(self, values: np.ndarray, num_bits: int) -> str:
        keys, counts = np.unique(values, return_counts=True)
        text = counts_text_from_ints([(int(k), int(c)) for k, c in zip(keys, counts)], num_bits)
        handle, path = tempfile.mkstemp(suffix=".counts")
        os.close(handle)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(os.unlink, path)
        return path

    def test_counts_round_trip_beyond_64_bits(self) -> None:
        value = (1 << 99) | (1 << 64) | 5
        path = self._counts_file(np.array([value, value, 3], dtype=object), 100)
        bits, counts = validate.read_counts(path, 100)
        row = bits[list(counts).index(2)]
        self.assertEqual([int(i) for i in np.flatnonzero(row)], [0, 2, 64, 99])

    def test_exact_samples_pass_and_biased_samples_fail(self) -> None:
        rng = np.random.default_rng(42)
        wl = workloads.sampling(8, 4000, seed=5)
        _, _, inst = workloads.parse(wl.text)
        probs = _dense_probabilities(wl.text.split("measure")[0])
        draws = rng.choice(probs.size, size=4000, p=probs / probs.sum())
        singles, pairs = validate.select_statistics(8)
        exact = validate.exact_statistics(inst, singles, pairs)
        bits, counts = validate.read_counts(self._counts_file(draws, 8), 8)
        good = validate.check_counts(*validate.observed(bits, counts, singles, pairs), exact, None)
        self.assertTrue(good["pass"], good)
        biased = draws ^ (rng.random(4000) < 0.2).astype(np.int64)
        bits, counts = validate.read_counts(self._counts_file(biased, 8), 8)
        bad = validate.check_counts(*validate.observed(bits, counts, singles, pairs), exact, None)
        self.assertFalse(bad["pass"])

    def test_deterministic_bit_rejects_one_stray_shot(self) -> None:
        exact = {"b0": {"p": 0.0}}
        result = validate.check_counts({"b0": 1 / 1000}, 1000, exact, None)
        self.assertFalse(result["pass"])

    def test_rare_outcome_is_not_a_false_alarm(self) -> None:
        result = validate.check_counts({"b0": 2 / 1000}, 1000, {"b0": {"p": 1e-4}}, None)
        self.assertTrue(result["pass"], result)

    def test_few_shots_are_not_validated(self) -> None:
        untested = validate.check_counts({"b0": 1.0}, 1, {"b0": {"p": 0.5}}, None)
        self.assertIsNone(untested["pass"])
        self.assertEqual(untested["role"], "too-few-shots")
        stray = validate.check_counts({"b0": 1.0}, 1, {"b0": {"p": 0.0}}, None)
        self.assertFalse(stray["pass"])

    def test_tvd(self) -> None:
        self.assertAlmostEqual(validate.tvd(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 1.0)


class StatsTests(unittest.TestCase):
    def test_bootstrap_is_deterministic(self) -> None:
        samples = [10.0, 11.0, 9.5, 10.2, 10.8]
        self.assertEqual(stats.summarize(samples), stats.summarize(samples))
        s = stats.summarize(samples)
        self.assertEqual(s["median_ms"], 10.2)
        self.assertLessEqual(s["ci95_low_ms"], s["median_ms"])
        self.assertGreaterEqual(s["ci95_high_ms"], s["median_ms"])

    def test_ratio_interval_brackets_point(self) -> None:
        r = stats.ratio([20.0, 21.0, 19.0], [10.0, 10.5, 9.5])
        self.assertAlmostEqual(r["ratio"], 2.0)
        self.assertLessEqual(r["ci95_low"], r["ratio"])
        self.assertGreaterEqual(r["ci95_high"], r["ratio"])


class WorkloadTests(unittest.TestCase):
    def test_generators_are_deterministic(self) -> None:
        self.assertEqual(workloads.rcs(10, 5).sha256, workloads.rcs(10, 5).sha256)
        self.assertNotEqual(workloads.rcs(10, 5).sha256, workloads.rcs(10, 5, seed=1).sha256)

    def test_counts_and_ids(self) -> None:
        wl = workloads.disjoint(40, 1000, layout="strided")
        self.assertEqual(wl.num_clbits, 40)
        self.assertIn("layout=strided", wl.workload_id)
        n, clbits, inst = workloads.parse(wl.text)
        self.assertEqual((n, clbits), (40, 40))
        for kind, *rest in inst:
            if kind == "gate" and len(rest[1]) == 2:
                a, b = rest[1]
                self.assertEqual(a % 4, b % 4, "a strided component's gates stay inside it")

    @unittest.skipUnless(os.path.exists(runners.RUNNER), "compare_runner is not built")
    def test_qft_input_makes_phases_observable(self) -> None:
        wl = workloads.qft(runners.RUNNER, 4)
        no_phase = "\n".join(line for line in wl.text.split("\n") if not line.startswith("cp "))
        self.assertGreater(validate.tvd(_dense_probabilities(wl.text), _dense_probabilities(no_phase)), 1e-3)

    def test_midcircuit_layout(self) -> None:
        wl = workloads.midcircuit(12, 10)
        self.assertEqual(wl.num_clbits, 3 * 4 + 8)
        self.assertEqual(workloads.gate_names(wl.text) >= {"measure", "reset", "if"}, True)


def _record(run_id: str, simulator: str, wid: str, median: float, **validation) -> dict:
    return {
        "run_id": run_id,
        "group": "headline",
        "workload": {"workload_id": wid, "family": "rcs", "num_qubits": 4, "mode": "probs", "shots": 0, "params": {}},
        "simulator": simulator,
        "precision": "double",
        "threads": 8,
        "status": "ok",
        "samples_ms": [median] * 3,
        "stats": {"median_ms": median},
        "validation": validation or {"pass": True},
    }


class RunFileTests(unittest.TestCase):
    def test_ratios_pair_within_one_run(self) -> None:
        records = [
            _record("a", "prismq", "w", 1.0),
            _record("a", "aer-statevector", "w", 2.0, role="reference", **{"pass": None}),
            _record("b", "prismq", "w", 4.0),
        ]
        rows = aggregate.ratios(records)
        self.assertEqual([(r["run_id"], r["ratio"], r["validated"]) for r in rows], [("a", 2.0, True)])

    def test_circuit_store_round_trips_and_deduplicates(self) -> None:
        out = tempfile.mkdtemp()
        known: set[str] = set()
        first, second = workloads.rcs(4, 1), workloads.rcs(4, 2)
        for wl in (first, second, first):
            matrix._save_circuit(out, wl, known)
        stored = matrix.load_circuits(out)
        self.assertEqual(stored, {first.sha256: first.text, second.sha256: second.text})

    def test_replaced_records_move_to_superseded(self) -> None:
        out = tempfile.mkdtemp()
        cell = profiles.Cell("headline", workloads.rcs(4, 1), ["prismq"])
        profile = profiles.Profile("t", "", [cell])
        old = [_record("a", "prismq", cell.workload.workload_id, 1.0), _record("a", "prismq", "qft/n4/probs", 1.0)]
        with open(os.path.join(out, "observations.jsonl"), "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in old)
        moved = matrix._supersede(out, profile, False, ["qft/"], "b")
        self.assertEqual(moved, 2)
        with open(os.path.join(out, "superseded.jsonl"), encoding="utf-8") as handle:
            kept = [json.loads(line) for line in handle]
        self.assertEqual({r["superseded_by"] for r in kept}, {"b"})
        with open(os.path.join(out, "observations.jsonl"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "")


class PageTests(unittest.TestCase):
    REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "matrix-reference-i7-6700k")

    def _render(self, tmp: str) -> tuple[str, dict]:
        results = os.path.join(tmp, "results")
        shutil.copytree(self.REFERENCE, results)
        docs = os.path.join(tmp, "docs")
        pages.write(results, docs)
        with open(os.path.join(docs, "index.md"), encoding="utf-8") as handle:
            page = handle.read()
        with open(os.path.join(docs, "data.js"), encoding="utf-8") as handle:
            script = handle.read()
        payload = script[script.index("=") + 1 :].strip().rstrip(";")
        return page, json.loads(payload)

    def test_reference_page_titles_and_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page, data = self._render(tmp)
            again, _ = self._render(os.path.join(tmp, "again"))
        self.assertEqual(page, again)
        self.assertIn("Aer takes 5.85x to 19.8x the PRISM-Q time", page)
        self.assertIn("Stim takes 0.66x to 1.47x the PRISM-Q time", page)
        self.assertIn("fastest forced backend on 7 rows", page)
        self.assertNotIn(chr(0x2014), page)
        for chart in ("summary", "sv64", "explore", "dispatch"):
            self.assertIn(f'data-chart="{chart}"', page)
        threads = [r for r in data["observations"] if r["group"] == "threads" and r["threads"] < 8]
        self.assertTrue(threads)
        self.assertTrue(all(r["status"] == "ok" and not r["checked"] for r in threads))


if __name__ == "__main__":
    unittest.main()

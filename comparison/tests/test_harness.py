"""Unit tests for the matrix harness: reference model, validation, statistics, workloads.

comparison/.venv/Scripts/python -m unittest discover -s comparison/tests -v
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matrix  # noqa: E402
from harness import aggregate, pages, profiles, provenance, reference, runners, stats, validate, workloads  # noqa: E402
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

    def test_nothing_testable_is_untested_not_failed(self) -> None:
        result = validate.check_counts({"b0": 0.5}, 1000, {"b0": {"p": None}}, None)
        self.assertIsNone(result["pass"])
        self.assertEqual(result["role"], "untested")

    def test_shot_total_must_match_the_workload(self) -> None:
        result = validate.check_counts({"b0": 0.5}, 999, {"b0": {"p": 0.5}}, None, expected_shots=1000)
        self.assertFalse(result["pass"])
        self.assertIn("999 shots", result["failures"][0])

    def test_key_wider_than_the_register_is_rejected(self) -> None:
        path = self._counts_file(np.array([1 << 5, 1]), 8)
        with self.assertRaises(ValueError):
            validate.read_counts(path, 4)

    def test_detectable_shift_follows_shots_and_the_test_applied(self) -> None:
        self.assertAlmostEqual(validate.min_detectable(256, 100), 0.231, places=3)
        self.assertAlmostEqual(validate.min_detectable(256, 100, 100), 0.326, places=3)
        self.assertIsNone(validate.min_detectable(256, 99))
        p = 0.3
        exact = {"b0": {"p": p}, "b1": {"p": p}}
        check = validate.check_counts({"b0": p, "b1": p}, 100, exact, None, 100)
        self.assertTrue(check["pass"])
        self.assertEqual(check["role"], "weakly-tested")
        self.assertAlmostEqual(check["min_detectable"], validate.min_detectable(2, 100))
        strong = validate.check_counts({"b0": p, "b1": p}, 10**5, exact, None, 10**5)
        self.assertNotIn("role", strong)
        self.assertTrue(aggregate.validated({"validation": check}))
        deterministic = validate.check_counts({"b0": 1.0}, 100, {"b0": {"p": 1.0}}, None, 100)
        self.assertIsNone(deterministic["min_detectable"])

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


ZOO = [
    lambda seed: workloads.ghz(40, 100),
    lambda seed: workloads.bv(40, 100, seed=seed),
    lambda seed: workloads.qaoa(12, layers=2, seed=seed),
    lambda seed: workloads.tfim(12, 3),
    lambda seed: workloads.tfim(64, 4, "expval", observables=4),
    lambda seed: workloads.wstate(30, 100),
    lambda seed: workloads.adder(4, seed=seed),
    lambda seed: workloads.adder(4, 100, seed=seed),
]


class ZooWorkloadTests(unittest.TestCase):
    def test_zoo_text_is_deterministic_and_parses(self) -> None:
        for make in ZOO:
            wl = make(workloads.CIRCUIT_SEED)
            self.assertEqual(wl.text, make(workloads.CIRCUIT_SEED).text, wl.workload_id)
            n, clbits, inst = workloads.parse(wl.text)
            self.assertEqual((n, clbits), (wl.num_qubits, wl.num_clbits))
            self.assertTrue(inst)

    def test_zoo_uses_only_gates_the_runners_carry(self) -> None:
        accepted = runners.UNITARY | {"measure"}
        for make in ZOO:
            wl = make(workloads.CIRCUIT_SEED)
            self.assertLessEqual(workloads.gate_names(wl.text), accepted, wl.workload_id)
            self.assertLessEqual(workloads.gate_names(wl.text), set(workloads.GATE_SET))

    def test_seeded_families_follow_their_seed(self) -> None:
        for make in (ZOO[1], ZOO[2], ZOO[6]):
            self.assertNotEqual(make(workloads.CIRCUIT_SEED).text, make(1).text)

    def test_legacy_peers_take_the_adder_but_not_rzz(self) -> None:
        self.assertLessEqual(workloads.gate_names(workloads.adder(4).text), runners.LEGACY_GATES)
        self.assertIn("rzz", runners.applicable(runners.resolve("quest"), workloads.qaoa(12)))

    def test_ghz_and_bv_are_clifford(self) -> None:
        for wl in (workloads.ghz(10, 10), workloads.bv(10, 10)):
            self.assertLessEqual(workloads.gate_names(wl.text), runners.CLIFFORD)
            self.assertIsNone(runners.applicable(runners.resolve("stim"), wl))

    def test_bv_reads_its_secret_on_every_shot(self) -> None:
        wl = workloads.bv(9, 100, seed=5)
        _, clbits, inst = workloads.parse(wl.text)
        secret = {int(line.split()[1]) for line in wl.text.split("\n") if line.startswith("cx ")}
        exact = reference.statistics(inst, list(range(clbits)), [])
        for bit in range(clbits):
            self.assertAlmostEqual(exact[f"b{bit}"]["p"], 1.0 if bit in secret else 0.0, places=12)

    def test_w_state_spreads_one_excitation(self) -> None:
        probs = _dense_probabilities(workloads.wstate(6, 0, mode="probs").text)
        one_hot = [1 << q for q in range(6)]
        np.testing.assert_allclose(probs[one_hot], 1 / 6, atol=1e-12)
        self.assertAlmostEqual(float(probs.sum() - probs[one_hot].sum()), 0.0, places=12)

    def test_adder_writes_the_sum_into_b(self) -> None:
        bits = 3
        wl = workloads.adder(bits, seed=11)
        probs = _dense_probabilities(wl.text)
        b0 = wl.params["b"]
        support = np.nonzero(probs > 1e-12)[0]
        self.assertEqual(len(support), 2**bits)
        for index in support:
            a = sum(((index >> (2 * i + 2)) & 1) << i for i in range(bits))
            b = sum(((index >> (2 * i + 1)) & 1) << i for i in range(bits))
            carry = (index >> (2 * bits + 1)) & 1
            self.assertEqual((index & 1, b + (carry << bits)), (0, a + b0))

    def test_qaoa_graph_is_three_regular(self) -> None:
        wl = workloads.qaoa(16)
        edges = [tuple(map(int, line.split()[1:3])) for line in wl.text.split("\n") if line.startswith("rzz ")]
        degree = np.bincount(np.array(edges).ravel(), minlength=16)
        self.assertTrue((degree == 3).all())
        self.assertEqual(len(set(edges)), len(edges))
        with self.assertRaises(ValueError):
            workloads.qaoa(15)

    def test_tfim_matches_its_trotter_layers(self) -> None:
        wl = workloads.tfim(5, 2, dt=0.25)
        lines = wl.text.strip().split("\n")[1:]
        self.assertEqual(sum(line.startswith("rzz ") for line in lines), 2 * 4)
        self.assertEqual(sum(line.startswith("rx ") for line in lines), 2 * 5)
        self.assertTrue(all(line.endswith(" 0.5") for line in lines))

    def test_new_gates_match_their_matrices(self) -> None:
        theta = 0.73
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        np.testing.assert_allclose(reference._matrix("rx", theta), [[c, -1j * s], [-1j * s, c]], atol=1e-12)
        zz = np.array([1.0, -1.0, -1.0, 1.0])
        rzz = reference._matrix("rzz", theta).reshape(4, 4)
        np.testing.assert_allclose(rzz, np.diag(np.exp(-0.5j * theta * zz)), atol=1e-12)

    def test_ladder_cells_climb_in_width_order(self) -> None:
        cells = profiles.ladder("mps/wstate", [workloads.wstate(n, 10) for n in (300, 100, 200)], ["prismq"], [8])
        self.assertEqual([c.workload.num_qubits for c in cells], [100, 200, 300])
        self.assertTrue(all(c.group == "scaling" and c.ladder == "mps/wstate" for c in cells))
        self.assertEqual({c.ceiling_s for c in cells}, {profiles.LADDER_POLICY["ceiling_s"]})

    def test_gate_smoke_covers_the_new_gates(self) -> None:
        for gate in ("rx", "rzz"):
            text = matrix.gate_smoke_text(gate)
            self.assertIn(f"\n{gate} 3 ", text)
            workloads.parse(text)


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


class RunnerTests(unittest.TestCase):
    def test_stabilizer_backends_refuse_non_clifford_circuits(self) -> None:
        for name in ("prismq:stabilizer", "prismq:factored-stabilizer"):
            sim = runners.resolve(name)
            self.assertEqual(runners.applicable(sim, workloads.rcs(4, 2)), "not a Clifford circuit", name)
            self.assertNotEqual(runners.applicable(sim, workloads.clifford(4, 100)), "not a Clifford circuit", name)

    def test_dense_memory_need_counts_each_simulators_bytes(self) -> None:
        wl = workloads.rcs(34, 1)
        self.assertIn("384 GiB", runners.applicable(runners.resolve("aer-statevector"), wl))
        self.assertIn("384 GiB", runners.applicable(runners.resolve("quest"), wl))
        self.assertIn("256 GiB", runners.applicable(runners.resolve("qsim"), wl))


class RegistryTests(unittest.TestCase):
    def test_qft_control_runs_auto_dispatch_with_recognition_off(self) -> None:
        sim = runners.resolve("prismq:no-qft-block")
        self.assertEqual(sim.backend, "auto")
        self.assertIn(("PRISM_NO_QFT_BLOCK", "1"), sim.env)
        self.assertEqual(sim.modes, runners.REGISTRY["prismq"].modes)

    def test_a_declared_gate_fault_makes_the_row_not_applicable(self) -> None:
        spinoza = runners.resolve("spinoza")
        with mock.patch.object(runners, "_executable", lambda sim: None):
            reason = runners.applicable(spinoza, workloads.Workload("qft", 3, "probs", "qubits 3\nh 0\n"))
            self.assertIn("multithreaded H", reason)
            self.assertIsNone(runners.applicable(spinoza, workloads.rcs(4, 2)))

    def test_single_threaded_simulators_run_at_one_thread(self) -> None:
        cell = profiles.Cell(
            "headline", workloads.clifford(4, 100), ["prismq", "stim"], [8], sim_threads=profiles._with_one_thread(8)
        )
        self.assertEqual(cell.threads_for("stim"), [1])
        self.assertEqual(cell.threads_for("prismq"), [1, 8])
        old = aggregate._normalize(_bare("r", "stim", workloads.clifford(4, 100), "headline", threads=8))
        self.assertEqual((old["threads"], old["threads_requested"]), (1, 8))

    def test_timed_output_by_family(self) -> None:
        self.assertEqual(runners.timed_output("aer-mps:256", "counts"), "a Result holding a histogram")
        self.assertEqual(runners.timed_output("stim", "counts"), "a bit-packed per-shot array")
        self.assertEqual(runners.timed_output("prismq:mps-256", "counts"), "a histogram")
        self.assertIsNone(runners.timed_output("stim", "probs"))


class StimWorkerTests(unittest.TestCase):
    def test_bit_packed_counts_land_in_classical_bit_order(self) -> None:
        try:
            from harness.worker import Stim
        except ImportError:
            self.skipTest("worker imports failed")
        try:
            sim = Stim("stim", 1)
        except ImportError:
            self.skipTest("stim not installed")
        text = "qubits 9\nclbits 9\nx 0\nx 8\n" + "".join(f"measure {q} {8 - q}\n" for q in range(9))
        circuit = sim.build(text, "counts")
        result = sim.execute(circuit, "counts", 10)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(sim.output(result, "counts").decode(), f"{(1 << 8) | 1:x} 10\n")


def _probs_record(run_id: str, simulator: str, precision: str = "double") -> dict:
    record = _record(run_id, simulator, "w", 1.0)
    record.pop("validation")
    record["precision"] = precision
    return record


class ValidateRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.cell = profiles.Cell("headline", workloads.rcs(2, 1), ["prismq", "aer-statevector", "quest"])
        self.vector = np.array([0.5, 0.25, 0.125, 0.125])

    def _output(self, name: str, vec: np.ndarray) -> str:
        path = os.path.join(self.dir, name + ".f64")
        vec.astype("<f8").tofile(path)
        return path

    def test_prismq_statevectors_check_each_other_without_a_reference(self) -> None:
        names = ("prismq:statevector", "prismq:gpu", "qsim")
        records = [_probs_record("a", s, "single" if s == "qsim" else "double") for s in names]
        outputs = {s: self._output(s, self.vector) for s in names}
        matrix._validate_row(self.cell, records, outputs)
        by = {r["simulator"]: r["validation"] for r in records}
        for name in names[:2]:
            self.assertEqual((by[name]["pass"], by[name]["role"]), (None, "untested"))
            self.assertEqual(by[name]["cross_check"]["tvd"], 0.0)
            self.assertIn("agrees with", by[name]["reason"])
        self.assertEqual(by["qsim"], {"pass": None, "role": "untested", "reason": matrix.NO_REFERENCE})
        self.assertTrue(aggregate.valid({**records[0], "status": "ok", "samples_ms": [1.0]}))
        outputs["prismq:gpu"] = self._output("gpu", self.vector[::-1].copy())
        records = [_probs_record("a", s) for s in names[:2]]
        matrix._validate_row(self.cell, records, outputs)
        for r in records:
            self.assertIs(r["validation"]["pass"], False)
            self.assertNotIn("role", r["validation"])
            self.assertIn("disagrees", r["validation"]["failures"][0])

    def test_aer_is_the_reference_only_when_quest_agrees(self) -> None:
        records = [_probs_record("a", s) for s in ("prismq", "aer-statevector", "quest")]
        outputs = {s: self._output(s, self.vector) for s in ("prismq", "aer-statevector", "quest")}
        matrix._validate_row(self.cell, records, outputs)
        by_sim = {r["simulator"]: r["validation"] for r in records}
        self.assertEqual(by_sim["aer-statevector"]["role"], "reference")
        self.assertEqual(by_sim["aer-statevector"]["cross_check"]["simulator"], "quest")
        self.assertTrue(by_sim["quest"]["pass"])
        self.assertTrue(by_sim["prismq"]["pass"])

    def test_aer_and_quest_disagreeing_leaves_the_row_unvalidated(self) -> None:
        records = [_probs_record("a", s) for s in ("prismq", "aer-statevector", "quest")]
        outputs = {s: self._output(s, self.vector) for s in ("prismq", "aer-statevector")}
        outputs["quest"] = self._output("quest", self.vector[::-1].copy())
        matrix._validate_row(self.cell, records, outputs)
        for r in records:
            self.assertIsNone(r["validation"]["pass"], r["simulator"])
            self.assertEqual(r["validation"]["role"], "untested")
            self.assertIn("disagree", r["validation"]["reason"])

    def test_rerun_without_the_reference_uses_the_stored_one(self) -> None:
        stored = _probs_record("a", "aer-statevector")
        stored["validation"] = {"pass": None, "role": "reference"}
        stored["output_file"] = matrix._store_output(self.dir, stored, self._output("aer", self.vector))
        rerun = _probs_record("b", "prismq")
        outputs = {"prismq": self._output("prismq", self.vector + np.array([1e-3, -1e-3, 0, 0]))}
        matrix._validate_row(self.cell, [rerun], outputs, [stored], self.dir)
        v = rerun["validation"]
        self.assertEqual((v["reference"], v["reference_run_id"]), ("aer-statevector", "a"))
        self.assertFalse(v["pass"])
        self.assertAlmostEqual(v["tvd"], 1e-3)

    def test_counts_with_the_wrong_shot_total_fail(self) -> None:
        wl = workloads.sampling(4, 200, seed=5)
        cell = profiles.Cell("sampling", wl, ["prismq"])
        record = _record("a", "prismq", wl.workload_id, 1.0)
        record.pop("validation")
        path = os.path.join(self.dir, "short.counts")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(counts_text_from_ints([(0, 150)], wl.num_clbits))
        matrix._validate_row(cell, [record], {"prismq": path})
        self.assertFalse(record["validation"]["pass"])
        self.assertIn("150 shots", record["validation"]["failures"][0])

    def test_empty_counts_file_fails_without_raising(self) -> None:
        wl = workloads.sampling(4, 200, seed=5)
        record = _record("a", "prismq", wl.workload_id, 1.0)
        record.pop("validation")
        path = os.path.join(self.dir, "empty.counts")
        open(path, "w").close()
        matrix._validate_row(profiles.Cell("sampling", wl, ["prismq"]), [record], {"prismq": path})
        self.assertFalse(record["validation"]["pass"])


class RunFileTests(unittest.TestCase):
    def test_ratios_pair_the_newest_record_of_each_simulator(self) -> None:
        records = [
            _record("a", "prismq", "w", 1.0),
            _record("a", "aer-statevector", "w", 2.0, role="reference", **{"pass": None}),
            _record("b", "prismq", "w", 4.0),
        ]
        rows = aggregate.ratios(records)
        self.assertEqual(
            [(r["run_id"], r["prismq_run_id"], r["ratio"], r["validated"]) for r in rows], [("a", "b", 0.5, True)]
        )

    def test_newest_drops_older_duplicates(self) -> None:
        records = [_record("b", "prismq", "w", 1.0), _record("a", "prismq", "w", 2.0), _record("a", "qsim", "w", 3.0)]
        self.assertEqual(
            [(r["run_id"], r["simulator"]) for r in aggregate.newest(records)], [("b", "prismq"), ("a", "qsim")]
        )

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
        wid = cell.workload.workload_id
        old = [_record("a", "prismq", wid, 1.0), _record("a", "prismq", "qft/n4/probs", 1.0)]
        new = _record("b", "prismq", wid, 1.0)
        with open(os.path.join(out, "observations.jsonl"), "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in old + [new])
        moved = matrix._supersede(out, profile.cells, False, ["qft/"], "b")
        self.assertEqual(moved, 2)
        with open(os.path.join(out, "superseded.jsonl"), encoding="utf-8") as handle:
            kept = [json.loads(line) for line in handle]
        self.assertEqual({r["superseded_by"] for r in kept}, {"b"})
        with open(os.path.join(out, "observations.jsonl"), encoding="utf-8") as handle:
            self.assertEqual([json.loads(line)["run_id"] for line in handle], ["b"])

    def test_drop_other_simulators_keeps_ones_filtered_out_of_the_run(self) -> None:
        out = tempfile.mkdtemp()
        cell = profiles.Cell("headline", workloads.rcs(4, 1), ["prismq", "aer-statevector", "qsim"])
        cell.simulators = ["qsim"]
        wid = cell.workload.workload_id
        old = [_record("a", s, wid, 1.0) for s in ("prismq", "aer-statevector", "qsim", "spinoza")]
        with open(os.path.join(out, "observations.jsonl"), "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in old + [_record("b", "qsim", wid, 1.0)])
        self.assertEqual(matrix._supersede(out, [cell], True, [], "b"), 2)
        with open(os.path.join(out, "observations.jsonl"), encoding="utf-8") as handle:
            left = sorted((r["run_id"], r["simulator"]) for r in map(json.loads, handle))
        self.assertEqual(left, [("a", "aer-statevector"), ("a", "prismq"), ("b", "qsim")])

    def test_superseding_drops_the_kept_output(self) -> None:
        out = tempfile.mkdtemp()
        cell = profiles.Cell("headline", workloads.rcs(4, 1), ["aer-statevector"])
        old = _record("a", "aer-statevector", cell.workload.workload_id, 1.0)
        source = os.path.join(out, "v.f64")
        np.zeros(16).tofile(source)
        old["output_file"] = matrix._store_output(out, old, source)
        with gzip.open(os.path.join(out, old["output_file"]), "rb") as handle:
            self.assertEqual(len(handle.read()), 128)
        with open(os.path.join(out, "observations.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(old) + "\n")
        matrix._supersede(out, [cell], False, [], "b")
        self.assertFalse(os.path.exists(os.path.join(out, old["output_file"])))


def _write(dir_: str, name: str, text: str) -> str:
    path = os.path.join(dir_, name)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def _bare(run_id: str, simulator: str, wl: workloads.Workload, group: str, threads: int = 8) -> dict:
    return {
        "run_id": run_id,
        "group": group,
        "workload": wl.summary(),
        "simulator": simulator,
        "precision": "double",
        "threads": threads,
        "status": "ok",
        "samples_ms": [1.0] * 3,
        "stats": {"median_ms": 1.0},
    }


class ExpvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.wl = workloads.expval(6, 3, 2)
        _, _, self.inst = workloads.parse(self.wl.text)
        self.keys = validate.keys_for(self.wl)
        self.exact = validate.exact_expectations(self.inst, self.keys)

    def _values(self, name: str, shift: float = 0.0, drop: str | None = None) -> str:
        lines = [f"{k} {self.exact[k]['value'] + shift!r}" for k in self.keys if k != drop]
        return _write(self.dir, name, "\n".join(lines) + "\n")

    def test_keys_name_z_and_adjacent_zz_terms(self) -> None:
        self.assertEqual(len(self.keys), 4)
        self.assertTrue(all(k.count("Z") == 1 for k in self.keys[:2]))
        for key in self.keys[2:]:
            a, b = (int(x) for x in key.split("Z")[1:])
            self.assertEqual(b, a + 1)
        self.assertIn("observables=2", self.wl.workload_id)
        self.assertTrue(self.wl.workload_id.endswith("/expval"))

    def test_exact_values_match_a_dense_statevector(self) -> None:
        probs = _dense_probabilities(self.wl.text)
        index = np.arange(probs.size)
        for key in self.keys:
            parity = np.zeros(probs.size, dtype=np.int64)
            for q in (int(x) for x in key.split("Z")[1:]):
                parity ^= (index >> q) & 1
            self.assertAlmostEqual(self.exact[key]["value"], float(probs @ (1 - 2 * parity)), places=12)

    def test_exact_path_passes_within_tolerance_and_fails_beyond(self) -> None:
        def check(path: str) -> dict:
            return validate.check_expectations(validate.read_values(path), self.keys, self.exact, None, 1e-8)

        good = check(self._values("a", 1e-10))
        self.assertTrue(good["pass"], good)
        self.assertEqual(good["against_exact"], 4)
        self.assertFalse(check(self._values("b", 1e-6))["pass"])
        self.assertFalse(check(self._values("c", drop=self.keys[0]))["pass"])

    def test_agreement_path_where_the_cone_is_too_wide(self) -> None:
        wide = {k: {"value": None, "reason": "too wide"} for k in self.keys}
        values = validate.read_values(self._values("a"))
        other = ("quimb-tn", validate.read_values(self._values("b", 5e-9)))
        agree = validate.check_expectations(values, self.keys, wide, other, 1e-8)
        self.assertTrue(agree["pass"], agree)
        self.assertEqual((agree["against_simulator"], agree["agreement_simulator"]), (4, "quimb-tn"))
        far = ("quimb-tn", validate.read_values(self._values("c", 1e-3)))
        self.assertFalse(validate.check_expectations(values, self.keys, wide, far, 1e-8)["pass"])
        alone = validate.check_expectations(values, self.keys, wide, None, 1e-8)
        self.assertIsNone(alone["pass"])
        self.assertEqual(alone["role"], "untested")

    def test_row_validation_and_the_sweep_is_not_pass_fail(self) -> None:
        cell = profiles.Cell("tn", self.wl, ["prismq", "quimb-tn"])
        records = [_bare("a", "prismq", self.wl, "tn"), _bare("a", "quimb-tn", self.wl, "tn")]
        outputs = {"prismq": self._values("p"), "quimb-tn": self._values("q", 1e-3)}
        matrix._validate_row(cell, records, outputs)
        self.assertTrue(records[0]["validation"]["pass"])
        self.assertFalse(records[1]["validation"]["pass"])
        sweep = profiles.Cell("bondsweep", self.wl, ["prismq:mps-2"])
        record = _bare("a", "prismq:mps-2", self.wl, "bondsweep")
        matrix._validate_row(sweep, [record], {"prismq:mps-2": self._values("m", 0.01)})
        v = record["validation"]
        self.assertFalse(v["pass"])
        self.assertFalse(v["pass_required"])
        self.assertAlmostEqual(v["max_abs_error"], 0.01)


class StatsModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        text = "qubits 3\nclbits 3\nh 0\ncx 0 1\nx 2\nmeasure 0 0\nmeasure 1 1\nmeasure 2 2\n"
        self.wl = workloads.Workload("clifford", 3, "stats", text, {"depth": 1}, 1000)
        self.cell = profiles.Cell("highshot", self.wl, ["prismq", "stim"])

    def _stats(self, name: str, shots: int, ones0: int, keys: list[str] | None = None) -> str:
        counts = {"b0": ones0, "b1": ones0, "b2": shots, "b0^b1": 0, "b1^b2": shots - ones0}
        keys = keys or validate.keys_for(self.wl)
        return _write(self.dir, name, f"shots {shots}\n" + "".join(f"{k} {counts[k]}\n" for k in keys))

    def test_keys_and_reader(self) -> None:
        self.assertEqual(validate.keys_for(self.wl), ["b0", "b1", "b2", "b0^b1", "b1^b2"])
        obs, shots = validate.read_stats(self._stats("a", 1000, 480))
        self.assertEqual(shots, 1000)
        self.assertAlmostEqual(obs["b0"], 0.48)
        with self.assertRaises(ValueError):
            validate.read_stats(_write(self.dir, "bad", "b0 3\n"))

    def test_statistics_pass_and_the_shot_total_is_checked(self) -> None:
        record = _bare("a", "prismq", self.wl, "highshot")
        matrix._validate_row(self.cell, [record], {"prismq": self._stats("a", 1000, 495)})
        v = record["validation"]
        self.assertTrue(v["pass"], v)
        self.assertEqual(v["method"], validate.STATS_METHOD)
        short = _bare("a", "prismq", self.wl, "highshot")
        matrix._validate_row(self.cell, [short], {"prismq": self._stats("b", 999, 495)})
        self.assertFalse(short["validation"]["pass"])
        self.assertIn("999 shots", " ".join(short["validation"]["failures"]))

    def test_a_deterministic_statistic_and_a_missing_one_fail(self) -> None:
        stray = _bare("a", "stim", self.wl, "highshot", threads=1)
        path = _write(self.dir, "s", "shots 1000\nb0 500\nb1 500\nb2 999\nb0^b1 0\nb1^b2 500\n")
        matrix._validate_row(self.cell, [stray], {"stim": path})
        self.assertFalse(stray["validation"]["pass"])
        missing = _bare("a", "stim", self.wl, "highshot", threads=1)
        matrix._validate_row(self.cell, [missing], {"stim": self._stats("m", 1000, 500, ["b0", "b1"])})
        self.assertFalse(missing["validation"]["pass"])


class ProfileCellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cells = profiles._wide_cells(8)

    def _group(self, name: str) -> list[profiles.Cell]:
        return [c for c in self.cells if c.group == name]

    def test_tn_rows_stay_inside_the_cone_reference(self) -> None:
        tn = self._group("tn")
        self.assertEqual([c.workload.num_qubits for c in tn], [32, 64, 128, 256])
        for cell in tn:
            self.assertEqual(cell.workload.mode, "expval")
            self.assertIn("quimb-tn", cell.simulators)
            self.assertIn("prismq:tensor-network", cell.simulators)
        _, _, inst = workloads.parse(tn[0].workload.text)
        pair = [k for k in validate.keys_for(tn[0].workload) if k.count("Z") == 2][:1]
        self.assertIsNotNone(validate.exact_expectations(inst, pair)[pair[0]]["value"])

    def test_deep_mps_rows_share_a_cap_and_declare_quimb_limits(self) -> None:
        deep = self._group("mpsdeep")
        base = [c for c in deep if c.workload.shots == profiles.DEEP_MPS_SHOTS]
        power = [c for c in deep if c.workload.shots == profiles.DEEP_MPS_POWER_SHOTS]
        self.assertEqual((len(base), len(power)), (10, 5))
        for cell in base:
            self.assertEqual(cell.simulators, ["prismq:mps-256", "aer-mps:256", "quimb-mps:256"])
            depth, n = cell.workload.params["depth"], cell.workload.num_qubits
            limit = profiles.QUIMB_MPS_COST[depth][1]
            reason = matrix.not_applicable(cell, "quimb-mps:256", runners.resolve("quimb-mps:256"))
            self.assertEqual(reason is not None, n > limit)
        for cell in power:
            self.assertEqual(cell.workload.params["depth"], 12)
            self.assertEqual(cell.simulators, ["prismq:mps-256", "aer-mps:256"])
        self.assertLess(validate.min_detectable(256, profiles.DEEP_MPS_POWER_SHOTS), 0.08)

    def test_sweep_spans_truncated_to_exact(self) -> None:
        sweep = self._group("bondsweep")
        self.assertEqual([c.workload.num_qubits for c in sweep], [32, 48])
        for cell in sweep:
            self.assertEqual(len(cell.simulators), 15)
            self.assertLessEqual(2 ** -(-cell.workload.params["depth"] // 2), max(profiles.SWEEP_CAPS))
            self.assertGreater(2 ** -(-cell.workload.params["depth"] // 2), min(profiles.SWEEP_CAPS))

    def test_high_shot_rows_and_thread_counts(self) -> None:
        rows = self._group("highshot")
        self.assertEqual(
            [(c.workload.num_qubits, c.workload.shots) for c in rows],
            [(100, 10**6), (1000, 10**6), (10000, 10**6), (100, 10**7), (1000, 10**7)],
        )
        for cell in rows:
            self.assertEqual(cell.workload.mode, "stats")
            self.assertEqual(cell.threads_for("prismq"), [1, 8])
            self.assertEqual(cell.threads_for("stim"), [1])
            self.assertIsNone(runners.applicable(runners.resolve("stim"), cell.workload))
        self.assertIn("statistics-only", runners.applicable(runners.resolve("aer-statevector"), rows[0].workload))


@unittest.skipUnless(os.path.exists(runners.RUNNER), "compare_runner is not built")
class PublishedProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = profiles.reference(runners.RUNNER, 24)
        cls.smoke = profiles.smoke(runners.RUNNER, 24)

    def test_every_zoo_family_runs_in_both_profiles(self) -> None:
        zoo = {"ghz", "bv", "qaoa", "hea", "tfim", "wstate", "adder"}
        for profile in (self.reference, self.smoke):
            self.assertLessEqual(zoo, {c.workload.family for c in profile.cells if c.group == "zoo"})

    def test_ladders_are_ordered_and_share_one_policy(self) -> None:
        ladders: dict[str, list[profiles.Cell]] = {}
        for cell in self.reference.cells:
            if cell.ladder:
                self.assertEqual(cell.group, "scaling")
                ladders.setdefault(cell.ladder, []).append(cell)
        self.assertGreaterEqual(len(ladders), 10)
        for key, cells in ladders.items():
            widths = [c.workload.num_qubits for c in cells]
            self.assertEqual(widths, sorted(set(widths)), key)
            self.assertEqual(len({(c.timeout_s, c.ceiling_s, tuple(c.simulators)) for c in cells}), 1, key)

    def test_ladders_reach_past_the_peers(self) -> None:
        top = {}
        for cell in self.reference.cells:
            if cell.ladder:
                top[cell.ladder] = max(top.get(cell.ladder, 0), cell.workload.num_qubits)
        self.assertGreaterEqual(top["dense/rcs"], 30)
        self.assertGreaterEqual(top["stabilizer/ghz"], 100_000)
        self.assertGreaterEqual(top["mps/lowent"], 1000)
        self.assertGreaterEqual(top["tn/tfim"], 1024)
        self.assertGreaterEqual(top["factored/disjoint"], 1000)

    def test_dense_ladders_force_an_exact_statevector_and_offer_the_gpu(self) -> None:
        dense = [c for c in self.reference.cells if (c.ladder or "").startswith("dense/")]
        self.assertTrue(dense)
        for cell in dense:
            self.assertIn("prismq:statevector", cell.simulators)
            self.assertIn("prismq:gpu", cell.simulators)
            self.assertNotIn("prismq", cell.simulators)

    def test_thread_scaling_reaches_every_core(self) -> None:
        rows = [c for c in self.reference.cells if c.group == "threads"]
        self.assertTrue(rows)
        for cell in rows:
            self.assertEqual(cell.threads[0], 1)
            self.assertEqual(cell.threads[-1], 24)

    def test_gpu_rungs_clear_the_device_crossover(self) -> None:
        for profile in (self.reference, self.smoke):
            for cell in profile.cells:
                if "prismq:gpu" in cell.simulators:
                    self.assertGreaterEqual(cell.workload.num_qubits, 14, cell.workload.workload_id)

    def test_stabilizer_ladders_pair_prismq_with_single_threaded_stim(self) -> None:
        cells = [c for c in self.reference.cells if (c.ladder or "").startswith("stabilizer/")]
        self.assertTrue(cells)
        for cell in cells:
            self.assertEqual(cell.threads_for("prismq"), [1, 24])
            self.assertEqual(cell.threads_for("stim"), [1])

    def test_gpu_runner_has_its_own_executable(self) -> None:
        sim = runners.resolve("prismq:gpu")
        self.assertEqual(sim.backend, "gpu")
        self.assertEqual(runners._executable(sim), runners.RUNNER_GPU)
        self.assertNotEqual(runners.RUNNER_GPU, runners.RUNNER)


class StoreTests(unittest.TestCase):
    def test_generated_workloads_store_a_recipe_that_rebuilds_them(self) -> None:
        made = [workloads.rcs(6, 3), workloads.disjoint(20, 100, layout="strided"), workloads.adder(3, 50)]
        made += [workloads.tfim(8, 2, "expval", observables=2), workloads.qaoa(8, seed=3)]
        with tempfile.TemporaryDirectory() as tmp:
            known: set[str] = set()
            for wl in made:
                matrix._save_circuit(tmp, wl, known)
            matrix._save_circuit(tmp, made[0], known)
            entries = matrix._store_entries(tmp)
            self.assertEqual(len(entries), len(made))
            self.assertTrue(all("recipe" in e and "text" not in e for e in entries))
            texts = matrix.load_circuits(tmp)
        self.assertEqual(texts, {wl.sha256: wl.text for wl in made})

    def test_exported_lists_keep_their_text_and_a_stale_recipe_is_refused(self) -> None:
        exported = workloads.Workload("qv", 2, "probs", "qubits 2\nh 0\ncx 0 1\n", {"seed": 1})
        with tempfile.TemporaryDirectory() as tmp:
            matrix._save_circuit(tmp, exported, set())
            self.assertEqual(matrix._store_entries(tmp)[0]["text"], exported.text)
            stale = {"sha256": "0" * 64, "recipe": workloads.rcs(4, 2).recipe}
            with gzip.open(os.path.join(tmp, matrix.CIRCUITS), "at", encoding="utf-8") as handle:
                handle.write(json.dumps(stale) + "\n")
            with self.assertRaises(ValueError):
                matrix.load_circuits(tmp)


class NativeTimingTests(unittest.TestCase):
    def test_native_peers_report_the_cold_call_from_inside_the_runner(self) -> None:
        calls = []

        def run_process(argv, stdin_text, env, timeout_s):
            calls.append(argv)
            if "probabilities" in argv:
                out = json.dumps({"length": 4, "first_ms": 12.5})
                return runners.ProcessResult(0, out, "", 900.0, 10.0, False)
            return runners.ProcessResult(0, json.dumps({"times_ms": [11.0, 12.0]}), "", 50.0, 10.0, False)

        with mock.patch.object(runners, "run_process", run_process):
            obs = runners.execute(runners.resolve("quest"), workloads.rcs(2, 1), 8, 3, 1, 1.0, 60.0, "out.f64")
        self.assertEqual(obs["phases"]["first_ms"], 12.5)
        self.assertEqual(obs["phases"]["process_wall_ms"], 900.0)
        cell = profiles.ladder("dense/rcs", [workloads.rcs(2, 1)], ["quest"], [8], ceiling_s=0.01)[0]
        self.assertEqual(matrix.ceiling_cause({"status": "ok", "phases": obs["phases"]}, cell), "budget")
        cell.ceiling_s = 0.5
        self.assertIsNone(matrix.ceiling_cause({"status": "ok", "phases": obs["phases"]}, cell))

    def test_prismq_statevector_faces_the_dense_peer_memory_limit(self) -> None:
        memory = mock.Mock(total=int(31.5 * 1024**3))
        with mock.patch.object(runners.psutil, "virtual_memory", lambda: memory):
            self.assertEqual(runners.dense_qubit_limit(), 30)
            sim = runners.resolve("prismq:statevector")
            wl = workloads.Workload("rcs", 30, "probs", "qubits 30\nh 0\n", {})
            with mock.patch.object(runners, "_executable", lambda sim: None):
                self.assertIsNone(runners.applicable(runners.resolve("aer-statevector"), wl))
        self.assertEqual(dict(sim.env), {"PRISM_MAX_SV_QUBITS": "30"})
        self.assertEqual(runners.resolve("prismq:gpu").env, ())


class CeilingTests(unittest.TestCase):
    def _cell(self) -> profiles.Cell:
        return profiles.ladder("dense/rcs", [workloads.rcs(4, 1)], ["prismq:statevector"], [8], ceiling_s=10.0)[0]

    def test_timeouts_memory_and_slow_calls_end_a_climb(self) -> None:
        cell = self._cell()
        self.assertEqual(matrix.ceiling_cause({"status": "timeout"}, cell), "timeout")
        self.assertEqual(matrix.ceiling_cause({"status": "oom"}, cell), "memory")
        refused = {"status": "error", "reason": "backend `statevector`: needs 31 qubits, exceeding the cap of 29"}
        self.assertEqual(matrix.ceiling_cause(refused, cell), "memory")
        self.assertEqual(matrix.ceiling_cause({"status": "ok", "phases": {"first_ms": 11_000.0}}, cell), "budget")
        self.assertIsNone(matrix.ceiling_cause({"status": "ok", "phases": {"first_ms": 9_000.0}}, cell))
        legacy = {"status": "ok", "phases": {"process_wall_ms": 12_000.0}}
        self.assertEqual(matrix.ceiling_cause(legacy, cell), "budget")
        self.assertIsNone(matrix.ceiling_cause({"status": "crash", "reason": "exit 3"}, cell))

    def test_run_skips_the_rungs_past_a_ceiling(self) -> None:
        rungs = [workloads.ghz(n, 10) for n in (8, 16, 32)]
        cells = profiles.ladder("stabilizer/ghz", rungs, ["prismq", "stim"], [8], ceiling_s=1.0)
        profile = profiles.Profile("t", "", cells)

        def execute(sim, wl, threads, *rest):
            if sim.name == "stim" and wl.num_qubits >= 16:
                return {"status": "timeout", "reason": None, "phases": {"process_wall_ms": 1.0}}
            return {"status": "ok", "reason": None, "samples_ms": [1.0], "phases": {"first_ms": 1.0}}

        args = mock.Mock(
            no_build=True,
            append=False,
            replace=False,
            retire=[],
            drop_other_simulators=False,
            skip_peer=[],
            cooldown_s=0,
            threads=8,
        )
        with tempfile.TemporaryDirectory() as tmp:
            args.out = tmp
            with (
                mock.patch.object(matrix, "load_profile", lambda a: profile),
                mock.patch.object(matrix, "smoke_check", lambda p: {}),
                mock.patch.object(matrix, "_validate_row", lambda *a, **k: None),
                mock.patch.object(runners, "execute", execute),
                mock.patch.object(runners, "version", lambda sim: "1"),
                mock.patch.object(runners, "_executable", lambda sim: None),
                mock.patch.object(provenance, "host_details", lambda: {}),
                mock.patch.object(provenance, "quiet_machine_warnings", lambda: []),
                mock.patch.object(runners, "native_builds", lambda: {}),
            ):
                self.assertEqual(matrix.cmd_run(args), 0)
            with open(os.path.join(tmp, "observations.jsonl"), encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
        stim = {r["workload"]["num_qubits"]: r for r in records if r["simulator"] == "stim"}
        self.assertEqual(stim[8]["status"], "ok")
        self.assertEqual((stim[16]["status"], stim[16]["ceiling"]), ("timeout", "timeout"))
        self.assertEqual(stim[32]["status"], "skipped")
        self.assertTrue(stim[32]["beyond_ceiling"])
        self.assertIn("timeout at 16 qubits", stim[32]["reason"])
        prismq = [r for r in records if r["simulator"] == "prismq"]
        self.assertTrue(all(r["status"] == "ok" and r["ladder"] == "stabilizer/ghz" for r in prismq))
        self.assertTrue(all(r["policy"]["ceiling_s"] == 1.0 for r in records))

    def test_a_harness_memory_refusal_is_a_ceiling(self) -> None:
        rungs = [workloads.rcs(n, 2) for n in (8, 16, 24)]
        cells = profiles.ladder("dense/rcs", rungs, ["prismq:statevector", "aer-statevector"], [8], ceiling_s=60.0)

        def applicable(sim, wl):
            if sim.name == "aer-statevector" and wl.num_qubits >= 16:
                return f"{runners.MEMORY_REFUSAL} 2^{wl.num_qubits} amplitudes needs 1 GiB"
            return None

        records = self._run(cells, applicable)
        aer = {r["workload"]["num_qubits"]: r for r in records if r["simulator"] == "aer-statevector"}
        self.assertEqual((aer[16]["status"], aer[16]["ceiling"]), ("not-applicable", "memory"))
        self.assertTrue(aer[24]["beyond_ceiling"])
        self.assertNotIn("ceiling", aer[8])
        statevector = [r for r in records if r["simulator"] == "prismq:statevector"]
        self.assertTrue(all(r["status"] == "ok" and "PRISM_MAX_SV_QUBITS" in r["env"] for r in statevector))

    def test_appended_runs_keep_the_recorded_ceilings(self) -> None:
        wl = workloads.ghz(16, 10)
        stored = {
            ("scaling", wl.workload_id): [
                {
                    **_ladder_rec("scaling", wl, "stim", 0, "stabilizer/ghz", threads=1, status="timeout"),
                    "ceiling": "timeout",
                },
                _ladder_rec("scaling", wl, "prismq", 1.0, "stabilizer/ghz"),
            ]
        }
        self.assertEqual(matrix._stored_ceilings(stored), {("stabilizer/ghz", "stim", 1): (16, "timeout at 16 qubits")})

    def _run(self, cells: list[profiles.Cell], applicable) -> list[dict]:
        def execute(sim, wl, threads, *rest):
            return {"status": "ok", "reason": None, "samples_ms": [1.0], "phases": {"first_ms": 1.0}}

        args = mock.Mock(
            no_build=True,
            append=False,
            replace=False,
            retire=[],
            drop_other_simulators=False,
            skip_peer=[],
            cooldown_s=0,
            threads=8,
        )
        with tempfile.TemporaryDirectory() as tmp:
            args.out = tmp
            with (
                mock.patch.object(matrix, "load_profile", lambda a: profiles.Profile("t", "", cells)),
                mock.patch.object(matrix, "smoke_check", lambda p: {}),
                mock.patch.object(matrix, "_validate_row", lambda *a, **k: None),
                mock.patch.object(runners, "execute", execute),
                mock.patch.object(runners, "applicable", applicable),
                mock.patch.object(runners, "version", lambda sim: "1"),
                mock.patch.object(provenance, "host_details", lambda: {}),
                mock.patch.object(provenance, "quiet_machine_warnings", lambda: []),
                mock.patch.object(runners, "native_builds", lambda: {}),
            ):
                self.assertEqual(matrix.cmd_run(args), 0)
            with open(os.path.join(tmp, "observations.jsonl"), encoding="utf-8") as handle:
                return [json.loads(line) for line in handle]


class SmokeTests(unittest.TestCase):
    def _profile(self, sims: list[str], wl: workloads.Workload | None = None) -> profiles.Profile:
        return profiles.Profile("t", "", [profiles.Cell("headline", wl or workloads.clifford(4, 100), list(sims))])

    def _fake(self, outputs: dict[str, str], status: dict[str, str] | None = None):
        def execute(sim, wl, threads, iterations, min_iterations, budget_s, timeout_s, out_path, keys_path=None):
            if (status or {}).get(sim.name):
                return {"status": status[sim.name], "reason": "boom"}
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(outputs.get(sim.name, outputs["prismq"]))
            return {"status": "ok"}

        return execute

    def _run(self, profile, outputs, status=None, version="1.0"):
        with (
            mock.patch.object(runners, "execute", self._fake(outputs, status)),
            mock.patch.object(runners, "version", lambda sim: version),
            mock.patch.object(runners, "_executable", lambda sim: None),
        ):
            return matrix.smoke_check(profile, quiet=True)

    def test_agreeing_peer_passes(self) -> None:
        result = self._run(self._profile(["prismq", "stim"]), {"prismq": "5 100\n"})
        self.assertTrue(result["stim"]["ok"], result)
        self.assertEqual(result["stim"]["checks"]["counts"]["result"], "pass")

    def test_disagreeing_or_failing_peer_fails(self) -> None:
        result = self._run(self._profile(["prismq", "stim"]), {"prismq": "5 100\n", "stim": "4 100\n"})
        self.assertFalse(result["stim"]["ok"])
        self.assertTrue(result["prismq"]["ok"])
        crashed = self._run(self._profile(["prismq", "stim"]), {"prismq": "5 100\n"}, {"stim": "crash"})
        self.assertFalse(crashed["stim"]["ok"])

    def test_missing_package_fails_and_unserved_mode_is_skipped(self) -> None:
        missing = self._run(self._profile(["stim"]), {"prismq": "5 100\n"}, version=None)
        self.assertFalse(missing["stim"]["ok"])
        expval = self._profile(["prismq:stabilizer", "quimb-tn"], workloads.expval(4, 2))
        result = self._run(expval, {"prismq": "Z0 0.5\n"})
        self.assertEqual(result["prismq:stabilizer"]["checks"]["expval"]["result"], "skipped")

    def test_dense_peers_smoke_each_declared_gate_at_their_top_thread_count(self) -> None:
        seen: dict[str, set[int]] = {}

        def execute(sim, wl, threads, iterations, min_iterations, budget_s, timeout_s, out_path, keys_path=None):
            seen.setdefault(sim.name, set()).add(threads)
            np.full(2**wl.num_qubits, 2.0**-wl.num_qubits).tofile(out_path)
            return {"status": "ok"}

        cell = profiles.Cell("headline", workloads.rcs(4, 2), ["prismq", "spinoza", "qip"], [8])
        with (
            mock.patch.object(runners, "execute", execute),
            mock.patch.object(runners, "version", lambda sim: "1.0"),
            mock.patch.object(runners, "_executable", lambda sim: None),
        ):
            result = matrix.smoke_check(profiles.Profile("t", "", [cell]), quiet=True)
        self.assertTrue(all(entry["ok"] for entry in result.values()), result)
        qip = result["qip"]["checks"]
        self.assertEqual(
            {k for k in qip if k.startswith("gate ")},
            {f"gate {g}" for g in ["preparation", *sorted(runners.LEGACY_GATES)]},
        )
        self.assertEqual(result["spinoza"]["checks"]["gate h"]["result"], "skipped")
        self.assertEqual(seen["spinoza"], {8})
        self.assertEqual(seen["qip"], {8})
        texts = {matrix.gate_smoke_text(g) for g in runners.UNITARY}
        self.assertEqual(len(texts), len(runners.UNITARY))

    def test_run_refuses_a_failing_peer_unless_skipped(self) -> None:
        args = mock.Mock(
            no_build=True,
            profile="smoke",
            threads=8,
            groups="headline",
            match="clifford",
            simulators=None,
            skip_peer=[],
        )
        with mock.patch.object(matrix, "smoke_check", lambda profile: {"stim": {"ok": False}}):
            self.assertEqual(matrix.cmd_run(args), 1)
        args.skip_peer = ["stim"]
        profile = matrix.load_profile(args)
        self.assertTrue(profile.cells)
        self.assertTrue(all("stim" not in c.simulators for c in profile.cells))
        args.skip_peer = ["aer-mps"]
        args.groups, args.match = "accuracy", None
        names = [s for c in matrix.load_profile(args).cells for s in c.simulators]
        self.assertTrue(names)
        self.assertFalse(any(s.startswith("aer-mps") for s in names))


class PageTests(unittest.TestCase):
    # A frozen copy of a complete reference run, so the page tests do not follow the published results.
    FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "page-reference")

    def _render(self, tmp: str, keep=None, extra=None) -> tuple[str, dict]:
        results = os.path.join(tmp, "results")
        os.makedirs(results)
        for name in ("manifest.json", "observations.jsonl"):
            with gzip.open(os.path.join(self.FIXTURE, name + ".gz"), "rb") as src:
                with open(os.path.join(results, name), "wb") as dst:
                    shutil.copyfileobj(src, dst)
        if keep is not None or extra:
            path = os.path.join(results, "observations.jsonl")
            with open(path, encoding="utf-8") as handle:
                records = [r for r in map(json.loads, handle) if keep is None or keep(r)]
            with open(path, "w", encoding="utf-8") as handle:
                handle.writelines(json.dumps(r) + "\n" for r in records + (extra or []))
        docs = os.path.join(tmp, "docs")
        pages.write(results, docs)
        with open(os.path.join(docs, "index.md"), encoding="utf-8") as handle:
            page = handle.read()
        with open(os.path.join(docs, "data.js"), encoding="utf-8") as handle:
            script = handle.read()
        payload = script[script.index("=") + 1 :].strip().rstrip(";")
        return " ".join(page.split()), json.loads(payload)

    def test_reference_page_titles_and_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page, data = self._render(tmp)
            again, _ = self._render(os.path.join(tmp, "again"))
        self.assertEqual(page, again)
        self.assertIn("Aer takes 5.11x to 10.9x the PRISM-Q time", page)
        self.assertNotIn("method-matched", page)
        self.assertIn("Peer median time over PRISM-Q median time, per section and peer", page)
        self.assertIn("Thread counts, PRISM-Q matched to the peer: ", page)
        self.assertIn("<td>fp32 peer, PRISM-Q fp64</td>", page)
        self.assertIn("rows take their reference vector from one simulator with no cross-check", page)
        self.assertIn("internal ", page)
        self.assertIn("QFT rows take their gate list from PRISM-Q's `qft_circuit` generator", page)
        self.assertIn("fastest forced backend on 7 rows", page)
        stim = [r for r in data["observations"] if r["simulator"] == "stim"]
        self.assertTrue(stim and all(r["threads"] == 1 for r in stim))
        aer = next(r for r in data["observations"] if r["simulator"] == "aer-statevector" and r["status"] == "ok")
        self.assertIn("first_ms", aer)
        self.assertTrue(any("internal_ms" in r for r in data["observations"] if r["simulator"].startswith("aer")))
        self.assertTrue(all("validated" in r for r in data["observations"]))
        self.assertNotIn(chr(0x2014), page)
        for chart in ("summary", "sv64", "explore", "dispatch"):
            self.assertIn(f'data-chart="{chart}"', page)
        threads = [r for r in data["observations"] if r["group"] == "threads" and r["threads"] < 8]
        self.assertTrue(threads)
        self.assertTrue(all(r["status"] == "ok" and not r["checked"] for r in threads))
        self.assertIn("4 physical cores, 8 hardware threads", page)
        self.assertIn("Aer automatic refuses 6 of 6 rows before running", page)
        self.assertIn("refused, statevector would not fit", page)
        self.assertIn('class="pq-kpis"', page)
        tiles = re.findall(
            r'<li class="pq-kpi"><a href="#([^"]+)"><span class="pq-kpi-l">([^<]+)</span>'
            r'<span class="pq-kpi-v">(.+?)</span><span class="pq-kpi-b"[^>]*></span>'
            r'<span class="pq-kpi-s">([^<]+)</span></a></li>',
            page,
        )
        self.assertEqual(page.count('class="pq-kpi"'), 9)
        self.assertIn(
            '<span class="pq-kpi-v"><span>5.11</span><i>x</i> <em>to</em> <span>69.2</span><i>x</i></span>', page
        )
        tiles = [(h, t, re.sub(r"<[^>]+>", "", v), n) for h, t, v, n in tiles]
        self.assertEqual(
            tiles,
            [
                ("statevector-fp64", "Statevector, fp64", "5.11x to 69.2x",
                 "Aer, QuEST, Spinoza and qip over PRISM-Q · 12 rows"),
                ("statevector-against-fp32-peers", "Statevector against fp32 peers", "1.42x to 14.8x",
                 "Aer fp32 and qsim fp32 over PRISM-Q · 12 rows · fp32 peers, PRISM-Q fp64"),
                ("stabilizer", "Stabilizer", "0.54x to 1.12x", "Stim over PRISM-Q · 3 rows · 1 thread"),
                ("high-shot-stabilizer", "High-shot stabilizer", "44.0x to 71.7x", "Stim over PRISM-Q · 5 rows · 1 thread"),
                ("matrix-product-state", "Matrix product state", "13.5x to 62.0x", "Aer MPS over PRISM-Q · 6 rows"),
                ("deep-mps", "Deep MPS", "2.09x to 18.1x",
                 "Aer MPS cap 256 and quimb MPS cap 256 over PRISM-Q MPS cap 256 · 15 rows"),
                ("tensor-network-expectation-values", "Tensor network expectation values", "2.71x to 303x",
                 "quimb TN and Aer MPS over PRISM-Q · 4 rows"),
                ("shot-sampling", "Shot sampling", "1.57x to 69.7x",
                 "Aer, Aer fp32 and qsim fp32 over PRISM-Q · 6 rows · PRISM-Q fp64"),
                ("disjoint-components", "Disjoint components", "2.83x to 142x", "Aer MPS over PRISM-Q · 4 rows"),
            ],
        )
        for href, heading, _, _ in tiles:
            self.assertIn(f"## {heading} {{#{href}}}", page)
        labels = re.findall(r'- <span class="pq-spec-l">([^<]+)</span> ', page)
        self.assertEqual(labels, ["Host", "Run", "Protocol", "Ratio", "Raw data"])
        self.assertIn("31.9 GB RAM, CPU only. Every number on this page comes from this host.", page)
        self.assertIn("profile `reference`, started 2026-11-03", page)
        self.assertLess(page.index('class="pq-issue"'), page.index("Median wall time per call"))
        self.assertIn(f'href="{pages.ISSUE_URL}"', page)
        toc = re.search(r'<nav class="pq-toc"[^>]*>(.+?)</nav>', page).group(1)
        self.assertIn('<a href="#statevector-fp64">Statevector, fp64</a>', toc)
        self.assertIn('<a href="#thread-scaling">Thread scaling</a>', toc)
        self.assertNotIn("reference host", page)
        self.assertNotIn('data-chart="scaling"', page)
        self.assertLess(page.index('pq-spec-l">Raw data<'), page.index('class="pq-kpis"'))
        self.assertLess(page.index('class="pq-kpis"'), page.index('data-chart="summary"'))
        self.assertIn('distance <span class="pq-nw">8.3e-15</span> in <span class="pq-nw">53.1 ms</span>', page)
        self.assertIn('<p class="pq-kicker">All sections · peer time over PRISM-Q time</p>', page)
        sv = next(
            r
            for r in data["observations"]
            if r["simulator"] == "aer-statevector" and r["group"] == "headline" and r["workload"].startswith("rcs")
            and r["threads"] == 8 and r["status"] == "ok"
        )
        pq = next(
            r
            for r in data["observations"]
            if r["simulator"] == "prismq" and r["group"] == "headline" and r["workload"] == sv["workload"] and r["threads"] == 8
        )
        self.assertIn("ratio", sv)
        self.assertNotIn("ratio", pq)
        self.assertAlmostEqual(sv["ratio"], sv["median_ms"] / pq["median_ms"], delta=0.01 * sv["ratio"])

    def test_partial_results_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page, _ = self._render(tmp, lambda r: r["group"] not in ("sampling", "accuracy", "threads"))
        self.assertIn("Aer takes 5.11x to 10.9x the PRISM-Q time", page)
        for chart in ("sampling", "bondcap", "threads"):
            self.assertNotIn(f'data-chart="{chart}"', page)

    def test_single_family_renders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page, _ = self._render(tmp, lambda r: r["workload"]["family"] == "clifford" and r["group"] == "headline")
        self.assertIn("Stim takes 0.54x to 1.12x the 1-thread PRISM-Q time, 100 to 10,000 qubits", page)
        self.assertIn("ratios pair Stim with the 1-thread PRISM-Q record", page)
        self.assertIn("PRISM-Q, a histogram; Stim, a bit-packed per-shot array", page)
        self.assertEqual(page.count('data-chart="'), 3)
        self.assertEqual(re.findall(r'class="pq-kpi"><a href="#([^"]+)"', page), ["stabilizer"])
        self.assertIn('<span class="pq-kpi-s">Stim over PRISM-Q · 3 rows · 1 thread</span>', page)
        self.assertIn("## Stabilizer {#stabilizer}", page)
        for chart in ("sv64", "dispatch", "memory", "threads"):
            self.assertNotIn(f'data-chart="{chart}"', page)

    def test_new_sections_render_from_their_records(self) -> None:
        extra = _wide_records()
        groups = {r["group"] for r in extra}

        def keep(r: dict) -> bool:
            return r["group"] not in groups

        with tempfile.TemporaryDirectory() as tmp:
            page, data = self._render(tmp, keep, extra)
            again, _ = self._render(os.path.join(tmp, "again"), keep, extra)
        self.assertEqual(page, again)
        for chart in ("highshot", "mpsdeep", "sweep", "tn"):
            self.assertIn(f'data-chart="{chart}"', page)
        self.assertIn("Stim takes 4.00x the 1-thread PRISM-Q time, 1,000 qubits", page)
        self.assertIn("<th>PRISM-Q 8 threads</th><th>PRISM-Q 1 thread</th><th>Stim 1 thread</th>", page)
        self.assertIn("Aer MPS cap 256 takes 2.00x the PRISM-Q MPS cap 256 time, 50 to 1,000 qubits", page)
        self.assertIn("quimb MPS cap 256 runs up to 50 qubits at depth 16", page)
        self.assertIn("PRISM-Q drops singular values at or below 1e-12 of the largest", page)
        self.assertNotIn("runs exactly under", page)
        self.assertIn("quimb TN takes 10.0x the PRISM-Q time, 64 qubits", page)
        self.assertIn("kahypar was unavailable, so cotengra searched paths with greedy and labels", page)
        self.assertIn('within <span class="pq-nw">1e-8</span> from bond cap 16 at 48 qubits', page)
        self.assertNotIn(chr(0x2014), page)
        sweep = [o for o in data["observations"] if o["group"] == "bondsweep"]
        self.assertEqual({o["bond_cap"] for o in sweep}, {2, 16})
        self.assertTrue(all("max_abs_error" in o for o in sweep))

    def test_scaling_section_shows_each_ceiling_and_the_widest_run(self) -> None:
        extra = _ladder_records()
        with tempfile.TemporaryDirectory() as tmp:
            page, data = self._render(tmp, None, extra)
        self.assertIn("## Scaling to the ceiling {#scaling}", page)
        self.assertIn('data-chart="scaling"', page)
        self.assertIn('<a href="#scaling">Scaling to the ceiling</a>', page)
        self.assertIn("Widest PRISM-Q run on the CPU per method: 29 statevector and 100,000 stabilizer", page)
        self.assertNotIn("None at", page)
        rows = dict(
            (cells[1], cells[2:])
            for cells in (
                re.findall(r"<td>([^<]*)</td>", row)
                for row in re.findall(r"<tr>(<td>(?:Random brickwork|GHZ)</td>.*?)</tr>", page)
            )
        )
        self.assertEqual(rows["PRISM-Q statevector"], ["double", "29§", "8.00 s", "memory at 30 qubits"])
        self.assertEqual(rows["PRISM-Q GPU"], ["double", "29§", "900 ms", "memory at 30 qubits"])
        self.assertEqual(rows["Aer"], ["double", "20‡", "5.00 ms", "memory at 29 qubits"])
        self.assertEqual(rows["QuEST"], ["double", "20", "6.00 ms", "wider rungs not run"])
        self.assertEqual(rows["qsim fp32"], ["single", "29*", "30.00 s", "one call over the rung budget at 30 qubits"])
        self.assertEqual(rows["Stim"], ["exact (tableau)", "1,000", "1.00 ms", "timeout at 10,000 qubits"])
        self.assertEqual(rows["PRISM-Q 1 thread"], ["double", "100,000", "100 ms", "top of the ladder"])
        tiles = re.findall(r'<li class="pq-reach-t" data-tab="([a-z]+)">(.*?)</li>', page)
        self.assertEqual([t[0] for t in tiles], ["dense", "stabilizer"])
        dense = re.findall(r'<span class="pq-reach-s">([^<]+)</span>', tiles[0][1])
        self.assertIn("<em>Statevector</em>", tiles[0][1])
        self.assertEqual(dense[0], "PRISM-Q statevector§ · 8.00 s · memory at 30 qubits")
        self.assertRegex(dense[1], r"^widest fp64 peer: (Aer|QuEST) at 20$")
        self.assertEqual(dense[2:], ["fp32 peer: qsim fp32 at 29", "GPU: PRISM-Q GPU at 29, no GPU peer"])
        stabilizer = re.findall(r'<span class="pq-reach-s">([^<]+)</span>', tiles[1][1])
        self.assertIn("1 thread: PRISM-Q 100,000, Stim 1,000", stabilizer)
        self.assertIn("times out (300 s on the Stabilizer ladders; 600 s on the Statevector ladders)", page)
        self.assertIn("`PRISM_MAX_SV_QUBITS=30`", page)
        self.assertIn("so the stabilizer ladders also run PRISM-Q at 1 thread", page)
        self.assertIn(pages.MARK_WORDS[pages.PAIRED], page)
        failures = page[page.index("## Correctness") :]
        self.assertIn("| qsim (fp32) | invalid | rcs 30q d10 |", failures)
        self.assertNotIn("| Stim | timeout |", failures)
        self.assertNotIn("PRISM-Q (statevector) | refused", failures)
        ghz = [o for o in data["observations"] if o.get("ladder") == "stabilizer/ghz" and o["simulator"] == "stim"]
        self.assertEqual({o["qubits"]: o.get("ceiling") for o in ghz}, {1000: None, 10000: "timeout", 100000: None})
        self.assertTrue(next(o for o in ghz if o["qubits"] == 100000)["beyond_ceiling"])
        qsim = {
            o["qubits"]: o for o in data["observations"] if o.get("ladder") == "dense/rcs" and o["simulator"] == "qsim"
        }
        self.assertEqual((qsim[29]["status"], qsim[29]["valid"]), ("ok", True))
        self.assertEqual((qsim[30]["status"], qsim[30]["ceiling"]), ("invalid", "budget"))
        self.assertEqual(data["ladders"], {"dense/rcs": "Random brickwork", "stabilizer/ghz": "GHZ"})

    def test_reach_handles_rungs_left_out_and_unchecked(self) -> None:
        wl = [workloads.rcs(n, 10) for n in (20, 24)]
        left = [_ladder_rec("scaling", w, "quest", 0, "dense/rcs", status="skipped", reason="left out") for w in wl]
        self.assertEqual(pages._reach(left), {"n": None, "record": None, "stop": "wider rungs not run"})
        cannot = [_ladder_rec("scaling", w, "quest", 0, "dense/rcs", status="not-applicable") for w in wl]
        self.assertEqual(pages._reach(cannot)["stop"], "cannot run at 20 qubits")
        unchecked = [
            aggregate._normalize(_ladder_rec("scaling", w, "qsim", 1.0, "dense/rcs", validation={"pass": None}))
            for w in wl
        ]
        reach = pages._reach(unchecked)
        self.assertEqual((reach["n"], reach["stop"], pages._mark(reach["record"])), (24, "top of the ladder", "*"))
        crashed = unchecked[:1] + [_ladder_rec("scaling", wl[1], "qsim", 0, "dense/rcs", status="crash")]
        self.assertEqual(pages._reach(crashed)["stop"], "crash at 24 qubits")

    def test_zoo_sections_render_beside_the_families(self) -> None:
        extra = []
        for wl in (workloads.qaoa(20), workloads.adder(9)):
            extra += [
                _ladder_rec("zoo", wl, "prismq", 2.0, ladder=None, validation={"pass": True, "tvd": 0.0}),
                _ladder_rec("zoo", wl, "aer-statevector", 8.0, ladder=None, validation={"pass": True, "tvd": 0.0}),
            ]
        wl = workloads.ghz(1000, 1000)
        passed = {"ladder": None, "validation": {"pass": True}}
        extra += [
            _ladder_rec("zoo", wl, "prismq", 1.0, **passed),
            _ladder_rec("zoo", wl, "prismq", 2.0, threads=1, **passed),
            _ladder_rec("zoo", wl, "stim", 3.0, threads=1, **passed),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            page, _ = self._render(tmp, None, extra)
        self.assertIn("## Circuit zoo, statevector {#circuit-zoo-statevector}", page)
        self.assertIn("Aer takes 4.00x the PRISM-Q time, 20 qubits", page)
        self.assertIn("## Circuit zoo, stabilizer {#circuit-zoo-stabilizer}", page)
        self.assertIn("Stim takes 1.50x the 1-thread PRISM-Q time, 1,000 qubits", page)
        self.assertIn("<td>qaoa 20q</td>", page)

    def test_status_words_tell_skipped_from_unrunnable_and_mark_checks(self) -> None:
        wl = workloads.clifford(4, 100)
        self.assertEqual(pages._status(None), "not run")
        self.assertEqual(pages._status({"status": "skipped"}), "not run")
        self.assertEqual(pages._status({"status": "not-applicable"}), "cannot run")
        refused = {"status": "oom", "reason": "Insufficient memory to run circuit using the statevector simulator"}
        self.assertEqual(pages._status(refused), "refused, statevector would not fit")
        self.assertEqual(pages._status({"status": "oom", "reason": "bad_alloc"}), "out of memory")
        record = _bare("r", "stim", wl, "headline", threads=1)
        record["validation"] = {"pass": None, "role": "untested"}
        self.assertEqual(pages._mark(record), pages.UNCHECKED)
        record["validation"] = {"pass": True, "role": "weakly-tested"}
        self.assertEqual(pages._mark(record), pages.WEAK)
        record["validation"] = {"pass": None, "role": "reference"}
        self.assertEqual(pages._mark(record), pages.UNCROSSED)
        record["validation"]["cross_check"] = {"simulator": "quest"}
        self.assertEqual(pages._mark(record), "")

    def test_unchecked_records_carry_no_ratio_in_the_chart_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, data = self._render(tmp)
        one_shot = [o for o in data["observations"] if o["group"] == "sampling" and o["shots"] == 1]
        self.assertTrue(one_shot)
        self.assertTrue(all(not o["validated"] and o["valid"] for o in one_shot if o["status"] == "ok"))
        self.assertTrue(all("ratio" not in o for o in one_shot))

    def test_suite_backend_claim_follows_the_record(self) -> None:
        rows = [{"family": "ghz", "qubits": 1024, "ms": 1.0}, {"family": "qft", "qubits": 20, "ms": 2.0}]
        suite = {"date": "d", "version": "v", "rows": rows}
        self.assertNotIn("Backend each family", "\n".join(pages._suite(suite)))
        rows[0]["backend"], rows[1]["backend"] = "Stabilizer", "Statevector"
        text = "\n".join(pages._suite(suite))
        self.assertIn("GHZ on `Stabilizer`; QFT on `Statevector`", text)

    def test_disclaimer_follows_the_intro(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page, _ = self._render(tmp)
        intro = page.index("tables mark")
        issue = page.index("[open a benchmark issue]")
        self.assertLess(intro, issue)
        self.assertLess(issue, page.index('- <span class="pq-spec-l">Host</span> Intel'))
        self.assertIn("https://github.com/AbeCoull/prism-q/issues/new?template=BENCHMARK.yml", page)

    def test_setup_names_a_dirty_commit_and_an_unrecorded_compiler(self) -> None:
        manifest = _provenance_manifest()
        manifest.pop("supplementary_runs")
        text = "\n".join(pages._setup(manifest, [_bare(BASE_RUN, "quest", workloads.rcs(4, 2), "headline")]))
        self.assertIn("`fc7e0c9699` with uncommitted changes", text)
        self.assertIn("compiler not recorded", text)
        self.assertNotIn("older run", text)

    def test_single_threaded_peer_keeps_the_requested_thread_count(self) -> None:
        record = aggregate._normalize(_bare(BASE_RUN, "stim", workloads.clifford(4, 100), "headline", threads=8))
        self.assertEqual((record["threads"], record["threads_requested"]), (1, 8))
        row = pages.data(_provenance_manifest(), [record], None)["observations"][0]
        self.assertEqual((row["threads"], row["threads_requested"]), (1, 8))

    def test_stabilizer_note_without_a_one_thread_prismq_record(self) -> None:
        wl = workloads.clifford(4, 100)
        recs = [_bare("r", "prismq", wl, "headline"), _bare("r", "stim", wl, "headline", threads=1)]
        notes = pages._notes("stabilizer", {"row": recs}, ["prismq", "stim"])
        self.assertIn("no 1-thread PRISM-Q record ran, so no ratio is shown", " ".join(notes))

    def test_warm_median_above_the_cold_call_is_named(self) -> None:
        record = _bare("r", "aer-statevector", workloads.rcs(16, 8), "headline")
        record.update(phases={"first_ms": 48.7}, stats={"median_ms": 89.9})
        notes = pages._shared_notes([record], ["aer-statevector"])
        self.assertIn("Aer rcs 16q d8 (48.7 ms cold, 89.9 ms warm)", " ".join(notes))

    def test_setup_uses_the_newest_run_provenance_and_names_older_groups(self) -> None:
        wl = workloads.rcs(4, 2)
        records = [
            _bare(NEW_RUN, "prismq", wl, "headline"),
            _bare(NEW_RUN, "quest", wl, "headline"),
            _bare(NEW_RUN, "stim", wl, "headline", threads=1),
            _bare(BASE_RUN, "prismq", wl, "dispatch"),
            _bare("20261103T202523-cccccc", "prismq:statevector", wl, "dispatch"),
        ]
        text = " ".join(" ".join(pages._setup(_provenance_manifest(), records)).split())
        self.assertIn("| PRISM-Q | 0.38.0, rustc 1.99.0, `parallel` |", text)
        self.assertIn("| Commit | `b012445717` |", text)
        self.assertIn("the rows below describe `20261110T124244-bbbbbb`", text)
        self.assertIn("built with MSVC 19.38, /O2", text)
        self.assertIn("package 1.16.0+avx2 loading `stim._stim_avx2`", text)
        self.assertIn("`OMP_WAIT_POLICY=PASSIVE`", text)
        self.assertNotIn("compiler not recorded", text)
        self.assertIn(
            "All 2 shown `dispatch` records come from the older runs `20261103T090018-aaaaaa` (2026-11-03, "
            "commit `fc7e0c9699` with uncommitted changes) and `20261103T202523-cccccc` (2026-11-03, commit "
            "not recorded).",
            text,
        )
        self.assertNotIn("`headline` records", text)

    def test_supplementary_run_records_its_own_provenance(self) -> None:
        prov = _provenance_manifest()["supplementary_runs"][-1]["provenance"]
        args = mock.Mock(
            no_build=True,
            profile="smoke",
            threads=8,
            groups="none",
            match=None,
            simulators=None,
            skip_peer=[],
            append=True,
            replace=False,
            retire=[],
            drop_other_simulators=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            args.out = tmp
            base = _provenance_manifest()
            base.pop("supplementary_runs")
            _write(tmp, "manifest.json", json.dumps(base))
            with (
                mock.patch.object(matrix, "smoke_check", lambda profile: {}),
                mock.patch.object(provenance, "collect", lambda *a: json.loads(json.dumps(prov))),
                mock.patch.object(provenance, "host_details", lambda: {}),
                mock.patch.object(provenance, "quiet_machine_warnings", lambda: []),
                mock.patch.object(runners, "native_builds", lambda: prov["toolchain"]["native_builds"]),
            ):
                self.assertEqual(matrix.cmd_run(args), 0)
            with open(os.path.join(tmp, "manifest.json"), encoding="utf-8") as handle:
                written = json.load(handle)
        run = written["supplementary_runs"][-1]
        for key in matrix.RUN_PROVENANCE_KEYS:
            self.assertIn(key, run)
        self.assertEqual(run["provenance"]["toolchain"]["git_sha"], prov["toolchain"]["git_sha"])
        self.assertEqual(run["simulators"]["quest"]["env"], dict(runners.REGISTRY["quest"].env))
        self.assertEqual(written["provenance"], base["provenance"])


BASE_RUN = "20261103T090018-aaaaaa"
NEW_RUN = "20261110T124244-bbbbbb"


def _provenance_manifest() -> dict:
    def run(sha: str, dirty: bool, prismq: str, native: dict | None, stim: dict | None, env: dict) -> dict:
        toolchain = {"rustc": "1.99.0", "rustc_peers": "1.99.0-nightly", "git_sha": sha, "git_dirty": dirty}
        if native:
            toolchain["native_builds"] = native
        host = {"cpu_model": "Test CPU", "physical_cores": 4, "logical_cores": 8, "os": "Test OS", "ram_gb": 32.0}
        versions = {"prismq": prismq, "quest": "v4.3.0", "stim": "1.16.0"}
        return {
            "provenance": {
                "host": host,
                "toolchain": toolchain,
                "python": {"version": "3.14.0", "packages": {}, "stim_build": stim},
                "simulators": versions,
            },
            "simulators": {
                name: {"label": name, "precision": "double", "env": env, "version": versions[name]} for name in versions
            },
        }

    base = run("fc7e0c96996590ccc69c1cd0fea7322365b9b9b7", True, "0.37.0", None, None, {})
    new = run(
        "b012445717f411e9144babfb40d1942a5fc788bf",
        False,
        "0.38.0",
        {"quest_runner": "MSVC 19.38, /O2", "peers": None},
        {"version": "1.16.0+avx2", "module": "stim._stim_avx2"},
        {"OMP_WAIT_POLICY": "PASSIVE"},
    )
    old_supplement = {"run_id": "20261103T202523-cccccc", "command": "matrix.py run --append"}
    return {
        "run_id": BASE_RUN,
        "profile": "test",
        **base,
        "supplementary_runs": [old_supplement, {"run_id": NEW_RUN, "command": "matrix.py run --append", **new}],
    }


def _ladder_rec(group, wl, sim, median, ladder="", threads=8, status="ok", **extra) -> dict:
    ran = status == "ok"
    record = {
        "schema": matrix.SCHEMA,
        "run_id": "z",
        "group": group,
        "workload": wl.summary(),
        "simulator": sim,
        "precision": runners.resolve(sim).precision,
        "threads": threads,
        "status": status,
        "samples_ms": [median] * 3 if ran else [],
        "stats": stats.summarize([median] * 3 if ran else []),
        "phases": {"first_ms": median} if ran else {},
        "policy": {"iterations": 3, "min_iterations": 1, "budget_s": 15.0, "timeout_s": 600.0},
        **extra,
    }
    if ladder:
        record["ladder"] = ladder
        record["policy"]["ceiling_s"] = 60.0
    return record


def _ladder_records() -> list[dict]:
    """A dense and a stabilizer ladder covering every way a climb ends.

    No fp64 reference past 20 qubits, a cross-checked PRISM-Q pair, a harness memory refusal, a
    runner memory refusal, a budget ceiling whose output failed its check, rungs the profile left
    out, a timeout, and PRISM-Q at 1 thread beside single-threaded Stim.
    """
    cap = {"env": {"PRISM_MAX_SV_QUBITS": "30"}, "resolved_backend": "Statevector"}
    paired = {
        "pass": None,
        "role": "untested",
        "reason": "no fp64 reference ran on this row; agrees",
        "cross_check": {"tvd": 0.0},
    }
    out = []
    for n in (20, 29, 30):
        wl = workloads.rcs(n, 10)
        dense = {"pass": True, "tvd": 0.0} if n == 20 else paired

        def add(sim, median, **extra):
            out.append(_ladder_rec("scaling", wl, sim, median, "dense/rcs", **extra))

        if n < 30:
            add("prismq:statevector", 0.5 if n == 20 else 8000.0, validation=dense, **cap)
            add("prismq:gpu", 0.4 if n == 20 else 900.0, validation=dense)
        else:
            reason = "backend `statevector`: the state needs 31 qubits, exceeding the cap of 30"
            add("prismq:statevector", 0, status="error", reason=reason, ceiling="memory", **cap)
            add("prismq:gpu", 0, status="oom", reason="device allocation failed", ceiling="memory")
        if n == 20:
            add("aer-statevector", 5.0, validation={"pass": None, "role": "reference"})
            add("quest", 6.0, validation={"pass": True, "tvd": 0.0})
            add("qsim", 2.0, precision="single", validation={"pass": True, "tvd": 0.0})
        elif n == 29:
            add(
                "aer-statevector", 0, status="not-applicable", reason="dense state of 2^29 amplitudes", ceiling="memory"
            )
            add("quest", 0, status="skipped", reason="left out")
            add("qsim", 30_000.0, precision="single", validation={"pass": None, "reason": "no fp64 reference ran"})
        else:
            past = {"reason": "past this ladder's ceiling (memory at 29 qubits)", "beyond_ceiling": True}
            add("aer-statevector", 0, status="skipped", **past)
            add("quest", 0, status="skipped", reason="left out")
            failed = {"pass": False, "failures": ["b3: p < 1e-6"], "tvd": 0.2}
            add("qsim", 70_000.0, precision="single", validation=failed, ceiling="budget")
    stab = {"timeout_s": 300.0}
    for n in (1000, 10_000, 100_000):
        wl = workloads.ghz(n, 1000)
        passed = {"validation": {"pass": True}}
        out.append(_ladder_rec("scaling", wl, "prismq", n / 2500, "stabilizer/ghz", **passed))
        out.append(_ladder_rec("scaling", wl, "prismq", n / 1000, "stabilizer/ghz", threads=1, **passed))
        if n == 1000:
            out.append(
                _ladder_rec(
                    "scaling", wl, "stim", 1.0, "stabilizer/ghz", threads=1, precision="exact (tableau)", **passed
                )
            )
        elif n == 10_000:
            out.append(
                _ladder_rec("scaling", wl, "stim", 0, "stabilizer/ghz", threads=1, status="timeout", ceiling="timeout")
            )
        else:
            past = {"reason": "past this ladder's ceiling (timeout at 10,000 qubits)", "beyond_ceiling": True}
            out.append(_ladder_rec("scaling", wl, "stim", 0, "stabilizer/ghz", threads=1, status="skipped", **past))
    for r in out:
        if r["ladder"].startswith("stabilizer"):
            r["policy"].update(stab)
    return out


def _wide_records() -> list[dict]:
    def rec(group, wl, sim, median, threads=8, status="ok", **extra):
        return {
            "schema": matrix.SCHEMA,
            "run_id": "z",
            "group": group,
            "workload": wl.summary(),
            "simulator": sim,
            "precision": "double",
            "threads": threads,
            "status": status,
            "samples_ms": [median] * 3 if status == "ok" else [],
            "stats": stats.summarize([median] * 3 if status == "ok" else []),
            "policy": {"iterations": 3, "min_iterations": 2, "budget_s": 30.0, "timeout_s": 600.0},
            **extra,
        }

    ok = {"validation": {"pass": True, "method": validate.EXPVAL_METHOD}}
    out = []
    hs = workloads.clifford(1000, 10**6, mode="stats")
    out += [
        rec("highshot", hs, "prismq", 5.0, **ok),
        rec("highshot", hs, "prismq", 10.0, threads=1, **ok),
        rec("highshot", hs, "stim", 40.0, threads=1, **ok),
    ]
    for n, depth in ((50, 16), (1000, 16)):
        wl = workloads.lowent(n, depth, 100)
        out += [rec("mpsdeep", wl, "prismq:mps-256", 10.0, **ok), rec("mpsdeep", wl, "aer-mps:256", 20.0, **ok)]
        if n == 50:
            out.append(rec("mpsdeep", wl, "quimb-mps:256", 30.0, **ok))
        else:
            out.append(rec("mpsdeep", wl, "quimb-mps:256", 0.0, status="not-applicable", reason="over budget"))
    tn = workloads.expval(64, 8, 4)
    opt = {"optimizer": {"hyper_methods": ["greedy", "labels"], "optlib": "sbplx", "kahypar_available": False}}
    out += [
        rec("tn", tn, "prismq", 2.0, **ok),
        rec("tn", tn, "prismq:tensor-network", 3.0, **ok),
        rec("tn", tn, "quimb-tn", 20.0, details=opt, **ok),
    ]
    for n in (32, 48):
        wl = workloads.expval(n, 8)
        for cap, err in ((2, 0.1), (16, 1e-12)):
            for prefix in ("prismq:mps-", "aer-mps:", "quimb-mps:"):
                v = {"pass": err < 1e-8, "pass_required": False, "max_abs_error": err, "method": validate.EXPVAL_METHOD}
                out.append(rec("bondsweep", wl, f"{prefix}{cap}", 1.0 + cap, validation=v))
    return out


if __name__ == "__main__":
    unittest.main()

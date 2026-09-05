import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import compare_brainbench
from compare_brainbench import compare_reports


def report(rows):
    return {"scenarios": [dict(id=key, dimension=dim, score=value,
                               skipped=False, detail="ok") for key, dim, value in rows]}


class PairedComparisonTests(unittest.TestCase):
    def test_repeats_are_not_independent_scenarios(self):
        base = report([("a", "retrieval", 0), ("b", "retrieval", 1)])
        candidate = report([("b", "retrieval", 1), ("a", "retrieval", 1)])
        result = compare_reports([base, base, base], [candidate, candidate, candidate])
        self.assertEqual(result["by_dimension"]["retrieval"]["n_scenarios"], 2)
        self.assertEqual(result["by_dimension"]["retrieval"]["mean_delta"], .5)

    def test_changed_scenario_set_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_reports([report([("a", "retrieval", 0)])],
                            [report([("b", "retrieval", 1)])])

    def test_failed_observation_is_unpaired_and_cannot_create_a_gain(self):
        base = report([("a", "retrieval", 0), ("b", "retrieval", 0)])
        base["scenarios"][0]["detail"] = "error: process exited"
        candidate = report([("a", "retrieval", 1), ("b", "retrieval", 1)])
        candidate["scenarios"][1]["skipped"] = True
        result = compare_reports([base], [candidate])
        self.assertEqual(result["baseline_errors"], 1)
        self.assertEqual(result["unpaired_scenarios"], ["retrieval/a", "retrieval/b"])
        self.assertEqual(result["scenarios"], [])
        self.assertEqual(result["by_dimension"], {})

    def test_duplicate_identity_is_rejected(self):
        repeated = report([("a", "retrieval", 0), ("a", "retrieval", 1)])
        with self.assertRaises(ValueError):
            compare_reports([repeated], [repeated])


class RunnerFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ["kbench.exe", "baseline.exe", "candidate.exe"]:
            (self.root / name).write_bytes(name.encode())
        (self.root / "dataset.json").write_text('{"scenarios": []}', encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def argv(self):
        return ["compare_brainbench.py",
                "--kbench", str(self.root / "kbench.exe"),
                "--baseline", str(self.root / "baseline.exe"),
                "--candidate", str(self.root / "candidate.exe"),
                "--dataset", str(self.root / "dataset.json"),
                "--out", str(self.root / "out"), "--repeats", "1"]

    def assert_incomplete_failure(self, side_effect, kind):
        run_patch = (mock.patch.object(subprocess, "run", side_effect=side_effect)
                     if isinstance(side_effect, BaseException)
                     else mock.patch.object(subprocess, "run", return_value=side_effect))
        with mock.patch.object(sys, "argv", self.argv()), run_patch:
            rc = compare_brainbench.main()
        artifact = json.loads((self.root / "out" / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(rc, 1)
        self.assertEqual(artifact["status"], "incomplete")
        self.assertNotIn("comparison", artifact)
        self.assertEqual(artifact["runs"][-1]["failure"]["kind"], kind)

    def test_nonzero_exit_is_persisted_before_runner_fails(self):
        completed = subprocess.CompletedProcess([], 7, stdout="partial", stderr="boom")
        self.assert_incomplete_failure(completed, "nonzero_exit")

    def test_timeout_is_persisted_before_runner_fails(self):
        timeout = subprocess.TimeoutExpired(["kbench"], 1800, output="partial", stderr="slow")
        self.assert_incomplete_failure(timeout, "timeout")

    def test_invalid_json_is_persisted_before_runner_fails(self):
        completed = subprocess.CompletedProcess([], 0, stdout="not json", stderr="warning")
        self.assert_incomplete_failure(completed, "invalid_json")

    def test_valid_json_with_wrong_report_shape_is_persisted_as_failure(self):
        for payload in ["null", "{}", '{"scenarios": [{}]}']:
            with self.subTest(payload=payload):
                completed = subprocess.CompletedProcess([], 0, stdout=payload, stderr="")
                self.assert_incomplete_failure(completed, "invalid_report")

    def test_changed_scenario_identity_finalizes_as_incomplete(self):
        baseline = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(report([("a", "retrieval", 1)])), stderr="")
        candidate = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(report([("b", "retrieval", 1)])), stderr="")
        with mock.patch.object(sys, "argv", self.argv()), \
             mock.patch.object(subprocess, "run", side_effect=[baseline, candidate]):
            rc = compare_brainbench.main()
        artifact = json.loads((self.root / "out" / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(rc, 1)
        self.assertEqual(artifact["status"], "incomplete")
        self.assertEqual(artifact["failure"]["kind"], "comparison_validation")
        self.assertNotIn("comparison", artifact)

    def test_duplicate_scenario_identity_finalizes_as_incomplete(self):
        duplicate = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(report([
                ("a", "retrieval", 0), ("a", "retrieval", 1)
            ])), stderr="")
        with mock.patch.object(sys, "argv", self.argv()), \
             mock.patch.object(subprocess, "run", return_value=duplicate):
            rc = compare_brainbench.main()
        artifact = json.loads((self.root / "out" / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(rc, 1)
        self.assertEqual(artifact["status"], "incomplete")
        self.assertEqual(artifact["failure"]["kind"], "comparison_validation")


if __name__ == "__main__":
    unittest.main()

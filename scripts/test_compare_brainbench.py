import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import compare_brainbench
from compare_brainbench import compare_reports


class ProcessOwnershipTests(unittest.TestCase):
    def test_timeout_terminates_descendants_before_parent_exit(self):
        import os
        import time
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "orphan-ran"
            child = "import time,pathlib; time.sleep(2); pathlib.Path(%r).write_text('orphan')" % str(marker)
            parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); print('ready',flush=True); time.sleep(30)" % child
            start = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                compare_brainbench.run_owned_tree([sys.executable, "-c", parent], env=dict(os.environ), timeout=.5)
            elapsed = time.monotonic() - start
            time.sleep(max(0, 2.4 - elapsed))
            self.assertFalse(marker.exists(), "timed-out inference descendant survived")


def report(rows):
    return {"session_configuration":{"warm_start":False,"include_ambient":False}, "scenarios": [dict(id=key, dimension=dim, score=value,
                               skipped=False, detail="ok") for key, dim, value in rows]}


class PairedComparisonTests(unittest.TestCase):
    def test_guard_override_is_explicit_and_isolated(self):
        base = {"KBENCH_EXPLICIT_FACT_GUARD": "true"}
        self.assertEqual(compare_brainbench.environment_for_side(base, None, explicit_fact_guard="false")["KBENCH_EXPLICIT_FACT_GUARD"], "false")
        self.assertEqual(base["KBENCH_EXPLICIT_FACT_GUARD"], "true")
        with self.assertRaises(ValueError):
            compare_brainbench.environment_for_side(base, None, explicit_fact_guard="maybe")

    def test_pairing_rejects_missing_or_false_isolation_provenance(self):
        for configuration in [None, {}, {"warm_start":True,"include_ambient":False},
                              {"warm_start":0,"include_ambient":False}]:
            sample = report([("a", "retrieval", 1)])
            sample["session_configuration"] = configuration
            with self.assertRaisesRegex(ValueError, "session_configuration"):
                compare_reports([sample], [report([("a", "retrieval", 1)])])

    def test_memory_summary_reports_unavailable_and_observed_peak_separately(self):
        base = report([("a", "retrieval", 1)])
        base["scenarios"][0]["observations"] = [
            dict(query="a", positive_recall_at_4=1, positive_hit_at_4=True,
                 positive_mrr=1, negative_injection=None, stale_injection=None,
                 latency_ms=1, first_query=True, model_text_bytes=80, mcp_result_bytes=100,
                 working_set_bytes=1000, peak_working_set_bytes=2000),
        ]
        summary = compare_reports([base], [base])["measurement_summary"]["baseline"]
        self.assertEqual(summary["max_mcp_peak_working_set_bytes"], 2000)
        self.assertEqual(summary["memory_observations"], 1)
        base["scenarios"][0]["observations"][0]["working_set_bytes"] = None
        base["scenarios"][0]["observations"][0]["peak_working_set_bytes"] = None
        summary = compare_reports([base], [base])["measurement_summary"]["baseline"]
        self.assertIsNone(summary["max_mcp_peak_working_set_bytes"])
        self.assertEqual(summary["memory_observations"], 0)

    def test_side_thread_overrides_preserve_base_and_allow_unset(self):
        base = {"KIMETSU_USER_BRAIN":"0", "KIMETSU_INTRA_THREADS":"8"}
        self.assertNotIn("KIMETSU_INTRA_THREADS", compare_brainbench.environment_for_side(base, 0))
        self.assertEqual(compare_brainbench.environment_for_side(base, 4)["KIMETSU_INTRA_THREADS"], "4")
        self.assertEqual(compare_brainbench.environment_for_side(base, None)["KIMETSU_INTRA_THREADS"], "8")
        self.assertEqual(base["KIMETSU_INTRA_THREADS"], "8")

    def test_side_model_overrides_allow_a_paired_reranker_comparison(self):
        base = {"KBENCH_RERANKER":"ms-marco-tinybert-l-2-v2"}
        self.assertEqual(compare_brainbench.environment_for_side(base, None, "off")["KBENCH_RERANKER"], "off")
        self.assertEqual(compare_brainbench.environment_for_side(base, 4, "ms-marco-minilm-l-4-v2")["KBENCH_RERANKER"], "ms-marco-minilm-l-4-v2")
        self.assertEqual(compare_brainbench.environment_for_side(base, None)["KBENCH_RERANKER"], base["KBENCH_RERANKER"])

    def test_side_cutoff_isolated_and_validated(self):
        base = {"KBENCH_RERANK_FLOOR": "0.3"}
        self.assertEqual(compare_brainbench.environment_for_side(base, None, None, .9)["KBENCH_RERANK_FLOOR"], "0.9")
        self.assertEqual(base["KBENCH_RERANK_FLOOR"], "0.3")
        for invalid in [-.1, 1.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                compare_brainbench.environment_for_side(base, None, None, invalid)

    def test_query_measurements_separate_recall_hit_and_first_query_latency(self):
        base = report([("a", "retrieval", .5)])
        base["scenarios"][0]["observations"] = [
            dict(query="a", positive_recall_at_4=.5, positive_hit_at_4=True,
                 positive_mrr=1, negative_injection=None, stale_injection=None,
                 latency_ms=100, first_query=True, model_text_bytes=80, mcp_result_bytes=100),
            dict(query="b", positive_recall_at_4=1, positive_hit_at_4=True,
                 positive_mrr=1, negative_injection=None, stale_injection=None,
                 latency_ms=20, first_query=False, model_text_bytes=180, mcp_result_bytes=200),
            dict(query="negative", positive_recall_at_4=None, positive_hit_at_4=None,
                 positive_mrr=None, negative_injection=False, stale_injection=None,
                 latency_ms=40, first_query=False, model_text_bytes=280, mcp_result_bytes=300),
        ]
        summary = compare_reports([base, base], [base, base])["measurement_summary"]["baseline"]
        self.assertEqual(summary["unique_queries"], 3)
        self.assertEqual(summary["positive_queries"], 2)
        self.assertEqual(summary["query_observations"], 6)
        self.assertEqual(summary["positive_recall_at_4"], .75)
        self.assertEqual(summary["positive_hit_at_4"], 1)
        self.assertEqual(summary["negative_injection_rate"], 0)
        self.assertEqual(summary["subsequent_query_p95_ms"], 40)
        self.assertEqual(summary["mean_mcp_result_bytes"], 200)

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
        run_patch = (mock.patch.object(compare_brainbench, "run_owned_tree", side_effect=side_effect)
                     if isinstance(side_effect, BaseException)
                     else mock.patch.object(compare_brainbench, "run_owned_tree", return_value=side_effect))
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
             mock.patch.object(compare_brainbench, "run_owned_tree", side_effect=[baseline, candidate]):
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
             mock.patch.object(compare_brainbench, "run_owned_tree", return_value=duplicate):
            rc = compare_brainbench.main()
        artifact = json.loads((self.root / "out" / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(rc, 1)
        self.assertEqual(artifact["status"], "incomplete")
        self.assertEqual(artifact["failure"]["kind"], "comparison_validation")


if __name__ == "__main__":
    unittest.main()

import unittest
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

    def test_skipped_and_failed_outcomes_remain_visible(self):
        base = report([("a", "retrieval", 0), ("b", "retrieval", 0)])
        base["scenarios"][0]["detail"] = "error: process exited"
        candidate = report([("a", "retrieval", 1), ("b", "retrieval", 1)])
        candidate["scenarios"][1]["skipped"] = True
        result = compare_reports([base], [candidate])
        self.assertEqual(result["baseline_errors"], 1)
        self.assertEqual(result["unpaired_scenarios"], ["retrieval/b"])
        self.assertEqual(result["by_dimension"]["retrieval"]["mean_delta"], 1)
        self.assertIsNone(result["by_dimension"]["retrieval"]["ci95"])

    def test_duplicate_identity_is_rejected(self):
        repeated = report([("a", "retrieval", 0), ("a", "retrieval", 1)])
        with self.assertRaises(ValueError):
            compare_reports([repeated], [repeated])


if __name__ == "__main__":
    unittest.main()

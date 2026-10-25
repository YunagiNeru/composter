import unittest

from som_probe.cli import _parser
from som_probe.tryout_comparison import (
    FIXED_CASES,
    SIMPLE_QUESTION_VARIANTS,
    _summary,
    choice_from_components,
    score_from_components,
)


class TryoutQuestionComparisonTests(unittest.TestCase):
    def test_fixed_cases_cover_choice_score_and_notification_without_dev_label_edits(self):
        counts = {kind: sum(row["dimension"] == kind for row in FIXED_CASES)
                  for kind in ("choice", "score", "noul")}
        self.assertEqual(counts, {"choice": 8, "score": 8, "noul": 4})
        self.assertEqual(len({row["case_id"] for row in FIXED_CASES}), len(FIXED_CASES))
        self.assertEqual({row["expected"] for row in FIXED_CASES if row["dimension"] == "choice"},
                         {"task", "none", "need_info"})
        self.assertTrue(all(row["rationale"] for row in FIXED_CASES))
        self.assertEqual(set(SIMPLE_QUESTION_VARIANTS), {
            "work_pending", "work_commitment", "date_info", "clock_time", "notification",
        })

    def test_choice_mapping_follows_approved_no_task_commitment_and_unknown_rules(self):
        self.assertEqual(choice_from_components(False, False), "none")
        self.assertEqual(choice_from_components(False, True), "none")
        self.assertEqual(choice_from_components(True, True), "task")
        self.assertEqual(choice_from_components(True, False), "need_info")
        with self.assertRaises(ValueError):
            choice_from_components(1, False)

    def test_score_mapping_keeps_date_only_and_time_only_at_stage_one(self):
        self.assertEqual(score_from_components(False, False), 0)
        self.assertEqual(score_from_components(True, False), 1)
        self.assertEqual(score_from_components(False, True), 1)
        self.assertEqual(score_from_components(True, True), 2)
        with self.assertRaises(ValueError):
            score_from_components(False, 0)

    def test_summary_separates_current_errors_from_base_and_paraphrase_results(self):
        rows = [{
            "dimension": "choice",
            "current": {"prediction": "none", "mismatch": True},
            "decomposed": {
                "base": {"prediction": "task", "mismatch": False},
                "paraphrase": {"prediction": "none", "mismatch": True},
            },
        }]
        summary = _summary(rows)["choice"]
        self.assertEqual(summary["current"], {"denominator": 1, "mismatches": 1, "matches": 0})
        self.assertEqual(summary["paired_improvements"], {"base": 1, "paraphrase": 0})
        self.assertEqual(summary["paired_regressions"], {"base": 0, "paraphrase": 0})
        self.assertEqual(summary["current_decomposed_decision_agreements"],
                         {"base": 0, "paraphrase": 1})

    def test_cli_exposes_comparison_and_model_specific_memory_commands(self):
        parser = _parser()
        comparison = parser.parse_args(["compare-tryout-questions"])
        laya_memory = parser.parse_args([
            "tryout-memory", "--model", "laya", "--source-run", "20261025-041751-58713e9e",
        ])
        sokudan_memory = parser.parse_args(["tryout-memory", "--model", "sokudan"])
        self.assertEqual(comparison.command, "compare-tryout-questions")
        self.assertEqual(laya_memory.model, "laya")
        self.assertEqual(sokudan_memory.model, "sokudan")


if __name__ == "__main__":
    unittest.main()

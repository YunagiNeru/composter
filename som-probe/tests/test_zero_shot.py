import unittest
from pathlib import Path

from som_probe.zero_shot import (
    ZeroShotError,
    build_hypotheses,
    common_prediction,
    load_config,
    official_scores_from_logits,
    reduce_score,
)
from som_probe.zero_shot_eval import _metric_summary


class ZeroShotAdapterTests(unittest.TestCase):
    def test_fixed_english_and_japanese_conditions_keep_their_own_templates(self):
        root = Path(__file__).resolve().parents[1]
        lock, evaluation = load_config(root)
        self.assertEqual(lock["candidate_b"]["status"], "BLOCKED")
        self.assertFalse(lock["candidate_b"]["artifacts_acquired"])
        english = evaluation["conditions"]["english"]
        japanese = evaluation["conditions"]["japanese"]
        self.assertIn("The text indicates", english["template"])
        self.assertIn("本文", japanese["template"])
        self.assertEqual(english["dimensions"]["choice"][0]["id"], "task")
        self.assertEqual(japanese["dimensions"]["choice"][0]["id"], "task")

    def test_hypothesis_uses_fixed_class_label_without_changing_premise(self):
        labels = [{"id": "task", "label": "an unfinished action"},
                  {"id": "none", "label": "no unfinished action"}]
        self.assertEqual(build_hypotheses("入力本文", labels, "The text indicates that {}."),
                         ["The text indicates that an unfinished action.",
                          "The text indicates that no unfinished action."])
        with self.assertRaises(ZeroShotError):
            build_hypotheses("", labels, "The text indicates that {}.")

    def test_official_single_label_scores_are_softmax_of_entailment_logits(self):
        result = official_scores_from_logits([[3.0, 2.0], [0.0, 1.0]])
        self.assertEqual(result["entailment_logits"], [2.0, 1.0])
        self.assertEqual(result["official_top_index"], 0)
        self.assertAlmostEqual(sum(result["official_scores"]), 1.0)
        self.assertAlmostEqual(result["pair_probabilities"][0][1], 1.0 / (1.0 + 2.718281828459045))

    def test_common_adapter_holds_tie_and_unsupported_top_instead_of_replacing_it(self):
        labels = [{"id": "task", "label": "task"}, {"id": "none", "label": "none"}]
        tied_logits = [[0.0, 1.0], [0.0, 1.0]]
        tied = official_scores_from_logits(tied_logits)
        self.assertEqual(common_prediction(labels, tied, tied_logits),
                         {"decision": None, "status": "HOLD", "reason": "official_score_tie"})
        unsupported_logits = [[2.0, 1.0], [1.0, 0.0]]
        unsupported = official_scores_from_logits(unsupported_logits)
        self.assertEqual(unsupported["official_top_index"], 0)
        self.assertEqual(common_prediction(labels, unsupported, unsupported_logits)["reason"],
                         "top_candidate_not_entailed")
        supported_logits = [[0.0, 2.0], [1.0, 0.0]]
        supported = official_scores_from_logits(supported_logits)
        self.assertEqual(common_prediction(labels, supported, supported_logits)["decision"], "task")

    def test_score_separates_stage_from_expected_value(self):
        date = {"status": "PASS", "candidate_ids": ["yes", "no"],
                "official_scores_not_calibrated": [0.8, 0.2],
                "common_prediction": {"decision": "yes"}}
        clock = {"status": "PASS", "candidate_ids": ["yes", "no"],
                 "official_scores_not_calibrated": [0.3, 0.7],
                 "common_prediction": {"decision": "no"}}
        result = reduce_score(date, clock)
        self.assertEqual(result["stage_prediction"], 1)
        self.assertAlmostEqual(result["score_expected_value_from_official_scores"], 1.1)
        self.assertNotIn("returned_score", result)

    def test_metrics_count_holds_as_wrong_and_keep_failure_denominator(self):
        cases = [
            {"dimension": "choice", "status": "PASS", "allowed_answers": ["task"], "common_prediction": "task"},
            {"dimension": "choice", "status": "HOLD", "allowed_answers": ["none"], "common_prediction": None},
            {"dimension": "choice", "status": "LENGTH_EXCEEDED", "allowed_answers": ["need_info"], "common_prediction": None},
        ]
        result = _metric_summary(cases, "choice")
        self.assertEqual(result["total_cases_denominator"], 3)
        self.assertEqual(result["status_counts"]["HOLD"], 1)
        self.assertEqual(result["status_counts"]["LENGTH_EXCEEDED"], 1)
        self.assertEqual(result["confusion_matrix"]["none"]["HOLD"], 1)
        self.assertEqual(result["confusion_matrix"]["need_info"]["ERROR"], 1)

    def test_noul_metrics_separate_false_positive_and_missed_candidate(self):
        cases = [
            {"dimension": "noul_deadline", "status": "PASS", "allowed_answers": [True], "common_prediction": True},
            {"dimension": "noul_deadline", "status": "PASS", "allowed_answers": [False], "common_prediction": True},
            {"dimension": "noul_deadline", "status": "HOLD", "allowed_answers": [True], "common_prediction": None},
        ]
        result = _metric_summary(cases, "noul_deadline")
        self.assertEqual(result["confusion_matrix"], {"TP": 1, "TN": 0, "FP": 1, "FN": 0})
        self.assertEqual(result["notification_candidates_false_positive"], 1)
        self.assertEqual(result["total_cases_denominator"], 3)


if __name__ == "__main__":
    unittest.main()

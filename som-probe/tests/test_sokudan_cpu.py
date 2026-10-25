import unittest

from som_probe.sokudan_cpu import (
    SokudanComparisonError,
    _case_diagnostics,
    adapt_official_answer,
)


class SokudanAdapterTests(unittest.TestCase):
    def test_choice_preserves_official_decision_and_display_probabilities(self):
        answer = {
            "type": "choice",
            "choice": "none",
            "probabilities": {"task": 0.1235, "none": 0.8765},
        }

        result = adapt_official_answer("choice", answer, ["task", "none"])

        self.assertEqual(result["decision"], "none")
        self.assertEqual(result["decision_slot"], 1)
        self.assertEqual(result["api_display_probabilities"], [0.1235, 0.8765])
        self.assertEqual(result["probability_origin"], "official_api_display_rounded_4")
        self.assertEqual(answer["choice"], "none")

    def test_choice_rejects_unknown_candidate_without_selecting_a_replacement(self):
        answer = {"type": "choice", "choice": "other", "probabilities": {"a": 0.5, "b": 0.5}}

        with self.assertRaises(SokudanComparisonError) as raised:
            adapt_official_answer("choice", answer, ["a", "b"])

        self.assertEqual(raised.exception.code, "UNKNOWN_CANDIDATE")

    def test_score_keeps_continuous_api_value_and_separates_stage_argmax(self):
        answer = {
            "type": "score",
            "score": 0.73593,
            "probabilities": {"0": 0.4000, "1": 0.3500, "2": 0.2500},
        }

        result = adapt_official_answer("score", answer, [0, 1, 2])

        self.assertEqual(result["stage_prediction"], 0)
        self.assertEqual(result["api_returned_value"], 0.73593)
        self.assertEqual(result["api_display_probabilities"], [0.4, 0.35, 0.25])
        self.assertIsNone(result.get("unrounded_probabilities"))

    def test_noul_evaluator_boundary_uses_returned_p_true(self):
        for value, expected in ((0.4999, False), (0.5, True), (0.5001, True)):
            with self.subTest(value=value):
                result = adapt_official_answer(
                    "noul", {"type": "bool", "noul": value}, [False, True]
                )

                self.assertEqual(result["api_returned_value"], value)
                self.assertIs(result["evaluator_boolean"], expected)
                self.assertEqual(result["evaluator_boolean_source"],
                                 "returned_P_true_greater_than_or_equal_to_0.5")

    def test_case_diagnostic_separates_draft_false_positive_from_laya_prediction(self):
        cases = [{
            "case_id": "notify-001",
            "questions": {"notify": {"type": "bool"}},
            "expected": {"notify": {"allowed_answers": [False]}},
        }]
        case_records = [{
            "case_id": "notify-001",
            "questions": [{
                "question_id": "notify",
                "question_type": "noul",
                "status": "PASS",
                "adapter": {
                    "question_type": "noul",
                    "candidate_ids": [False, True],
                    "evaluator_boolean": True,
                    "stage_prediction": None,
                    "api_returned_value": 0.8,
                    "api_display_probabilities": None,
                },
            }],
        }]
        laya_rows = [{
            "case_id": "notify-001",
            "question_id": "notify",
            "question_type": "noul",
            "candidate_ids": [False, True],
            "probabilities": [0.6, 0.4],
            "returned_noul": 0.4,
            "boolean_result": False,
        }]

        result = _case_diagnostics(cases, case_records, laya_rows)[0]

        self.assertEqual(result["sokudan"]["returned_P_true"], 0.8)
        self.assertIs(result["sokudan_match_to_unverified_draft"], False)
        self.assertIs(result["notification_out_of_scope_false_positive_against_draft"], True)
        self.assertIs(result["saved_laya"]["prediction"], False)
        self.assertIs(result["sokudan_matches_saved_laya"], False)


if __name__ == "__main__":
    unittest.main()

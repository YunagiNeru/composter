import unittest

from som_probe.sokudan_cpu import SokudanComparisonError, adapt_official_answer


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


if __name__ == "__main__":
    unittest.main()

import unittest

from som_probe.zero_shot_analysis import official_pipeline_case


class ZeroShotAnalysisTests(unittest.TestCase):
    def test_official_choice_output_is_separate_from_abstaining_adapter(self):
        row = {
            "case_id": "choice-001",
            "dimension": "choice",
            "status": "HOLD",
            "common_prediction": None,
            "model_specific_output": {
                "classification": {
                    "status": "PASS",
                    "official_prediction_id": "task",
                    "official_scores_not_calibrated": [0.61, 0.25, 0.14],
                }
            },
        }
        result = official_pipeline_case(row)
        self.assertEqual(result["common_prediction"], "task")
        self.assertIsNone(result["policy_prediction"])
        self.assertEqual(result["policy_status"], "HOLD")
        self.assertEqual(row["common_prediction"], None)

    def test_official_score_stage_is_derived_from_date_and_clock_top_labels(self):
        row = {
            "case_id": "score-001",
            "dimension": "score",
            "status": "HOLD",
            "common_prediction": None,
            "model_specific_output": {
                "date_info": {"status": "PASS", "official_prediction_id": "yes"},
                "clock_time": {"status": "PASS", "official_prediction_id": "no"},
                "score_expected_value_from_official_scores": 0.9,
            },
        }
        result = official_pipeline_case(row)
        self.assertEqual(result["common_prediction"], 1)
        self.assertEqual(result["score_expected_value_not_api_returned_score"], 0.9)
        self.assertEqual(result["policy_status"], "HOLD")

    def test_official_noul_boolean_uses_top_candidate_and_keeps_policy_hold(self):
        row = {
            "case_id": "noul-001",
            "dimension": "noul_deadline",
            "status": "HOLD",
            "common_prediction": None,
            "model_specific_output": {
                "classification": {
                    "status": "PASS",
                    "official_prediction_id": "no",
                    "official_scores_not_calibrated": [0.1, 0.9],
                }
            },
        }
        result = official_pipeline_case(row)
        self.assertIs(result["common_prediction"], False)
        self.assertEqual(result["policy_status"], "HOLD")


if __name__ == "__main__":
    unittest.main()

import unittest

from som_probe.runtime import (
    RuntimeErrorRecord, _comparison, _execution_device_names, _npu_error_code, _row_output,
    _score_value, _type_comparison,
)


class RuntimeFailureTests(unittest.TestCase):
    def test_score_value_matches_zero_based_expected_value_and_four_place_rounding(self):
        raw, returned = _score_value([0.2, 0.6, 0.2])

        self.assertEqual(raw, 1.0)
        self.assertEqual(returned, 1.0)

    def test_score_uses_one_temperature_and_excludes_invalid_candidate_slots(self):
        import numpy as np
        from som_probe.contracts import softmax

        row = {
            "case_id": "score-1", "question_id": "rating", "input_sha256": "fixed",
            "question_type": "score", "candidate_ids": [0, 1, 2], "candidate_count": 3,
            "temperature": 2.0,
        }
        logits = np.asarray([0.0, 2.0, -1.0, 100.0, 100.0], dtype=np.float32)
        actions = np.asarray([0.0, 1.0], dtype=np.float32)

        output = _row_output(row, logits, actions)
        expected_probabilities = softmax([0.0, 2.0, -1.0], 2.0)
        expected_raw = sum(index * probability for index, probability in enumerate(expected_probabilities))

        self.assertEqual(output["temperature"], 2.0)
        self.assertEqual(output["candidate_ids"], [0, 1, 2])
        self.assertEqual(output["score_scale"], [0, 1, 2])
        self.assertEqual(output["decision_slot"], 1)
        self.assertEqual(output["score_expected_unrounded"], expected_raw)
        self.assertEqual(output["returned_score"], round(expected_raw, 4))
        self.assertEqual(output["returned_score_rounding"], "python_round_4")
        self.assertAlmostEqual(sum(output["probabilities"]), 1.0)
        self.assertEqual(output["probability_sum"], sum(output["probabilities"]))
        self.assertNotAlmostEqual(output["returned_score"], 3.0)
        self.assertNotEqual(expected_probabilities, softmax([0.0, 2.0, -1.0], 4.0))

    def test_score_value_rejects_non_normalized_probabilities(self):
        with self.assertRaisesRegex(RuntimeErrorRecord, "確率の合計"):
            _score_value([0.2, 0.2])

    def test_keeps_ambiguous_operator_error_ambiguous(self):
        message = "Node 0 doesn't respect Custom Operator format or is an unrecognized operator"
        self.assertEqual(
            _npu_error_code("compile", message),
            "NPU_COMPILER_OPERATOR_FORMAT_OR_RECOGNITION_ERROR",
        )

    def test_classifies_other_compile_and_inference_failures(self):
        self.assertEqual(_npu_error_code("compile", "device unavailable"), "NPU_COMPILATION_FAILED")
        self.assertEqual(_npu_error_code("inference", "execution failed"), "NPU_INFERENCE_FAILED")
        self.assertEqual(_npu_error_code("cache_setup", "cache path already exists"), "NPU_CACHE_SETUP_FAILED")

    def test_execution_devices_keeps_npu_string_as_one_device(self):
        self.assertEqual(_execution_device_names("NPU"), ["NPU"])
        self.assertEqual(_execution_device_names(["NPU.0"]), ["NPU.0"])
        self.assertEqual(_execution_device_names("NPU,CPU"), ["NPU", "CPU"])

    def test_important_new_errors_counts_reference_decision_changes(self):
        reference = [{"case_id": "case-1", "input_sha256": "same",
                      "probabilities": [0.1, 0.9], "action_probabilities": [0.0, 1.0],
                      "decision_slot": 1}]
        candidate = [{"case_id": "case-1", "input_sha256": "same",
                      "probabilities": [0.7, 0.3], "action_probabilities": [0.0, 1.0],
                      "decision_slot": 0}]

        result = _comparison(reference, candidate)

        self.assertEqual(result["important_new_errors"], 1)
        self.assertEqual(result["decision_changes"], ["case-1"])
        self.assertIn("人の正解に対する誤り件数ではない", result["important_new_errors_definition"])

    def test_type_comparison_uses_candidate_ids_and_reports_valid_value_denominators(self):
        common = {"case_id": "case-1", "question_id": "q1", "input_sha256": "same",
                  "action_probabilities": [0.2, 0.8]}
        reference = [
            {**common, "question_type": "choice", "candidate_ids": ["a", "b"],
             "probabilities": [0.25, 0.75], "decision_slot": 1, "decision": "b"},
            {**common, "question_type": "score", "candidate_ids": [0, 1, 2],
             "probabilities": [0.2, 0.6, 0.2], "decision_slot": 1, "decision": 1,
             "temperature": 1.0, "probability_sum": 1.0, "score_scale": [0, 1, 2],
             "score_expected_unrounded": 1.0, "returned_score": 1.0,
             "returned_score_rounding": "python_round_4"},
            {**common, "question_type": "noul", "candidate_ids": [False, True],
             "probabilities": [0.3, 0.7], "decision_slot": 1, "decision": True,
             "boolean_result": True, "true_probability": 0.7},
        ]
        candidate = [
            {**common, "question_type": "choice", "candidate_ids": ["b", "a"],
             "probabilities": [0.75, 0.25], "decision_slot": 0, "decision": "b"},
            {**common, "question_type": "score", "candidate_ids": [0, 1, 2],
             "probabilities": [0.2, 0.3, 0.5], "decision_slot": 2, "decision": 2,
             "temperature": 1.0, "probability_sum": 1.0, "score_scale": [0, 1, 2],
             "score_expected_unrounded": 1.3, "returned_score": 1.3,
             "returned_score_rounding": "python_round_4"},
            {**common, "question_type": "noul", "candidate_ids": [False, True],
             "probabilities": [0.6, 0.4], "decision_slot": 0, "decision": False,
             "boolean_result": False, "true_probability": 0.4},
        ]

        result = _type_comparison(reference, candidate)
        by_type = result["by_question_type"]

        self.assertEqual(result["question_type_counts"], {"choice": 1, "score": 1, "noul": 1})
        self.assertEqual(by_type["choice"]["decision_candidate_id_match_ratio"], 1.0)
        self.assertEqual(by_type["choice"]["decision_slot_changes"], 1)
        self.assertEqual(by_type["choice"]["candidate_probability_mean_denominator"], 2)
        self.assertEqual(by_type["score"]["returned_score_mean_denominator"], 1)
        self.assertAlmostEqual(by_type["score"]["returned_score_mean_abs_difference"], 0.3)
        self.assertEqual(by_type["score"]["score_expected_unrounded_mean_denominator"], 1)
        self.assertAlmostEqual(by_type["score"]["score_expected_unrounded_mean_abs_difference"], 0.3)
        self.assertEqual(by_type["noul"]["boolean_result_changes"], 1)
        self.assertEqual(by_type["noul"]["true_probability_mean_denominator"], 1)


if __name__ == "__main__":
    unittest.main()

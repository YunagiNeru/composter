import unittest

from som_probe.runtime import (
    RuntimeErrorRecord, _compare_profile_inputs, _comparison, _execution_device_names,
    _npu_error_code, _row_output,
    _noul_return_value, _score_value, _type_comparison,
)


class RuntimeFailureTests(unittest.TestCase):
    def test_profile_comparison_preserves_active_tokens_candidates_and_specials(self):
        special = {"cls": 101, "sep": 102, "mask": 103, "pad": 0}

        def row(sequence_length, case_id):
            active = [101, 103, 11, 103, 12, 102]
            inputs = {
                "input_ids": [active + [0] * (sequence_length - len(active))],
                "attention_mask": [[1] * len(active) + [0] * (sequence_length - len(active))],
                "marker_pos": [[1, 3, 0, 0, 0, 0, 0, 0]],
                "marker_mask": [[True, True, False, False, False, False, False, False]],
                "qtype": [0],
            }
            return {"case_id": case_id, "question_id": "intent", "question_type": "choice",
                    "candidate_ids": ["a", "b"], "candidate_count": 2, "qtype": 0,
                    "temperature": 1.0, "input_sequence_length": sequence_length,
                    "special_token_ids": special, "inputs": inputs}

        baseline_rows = [row(256, f"case-{index:02}") for index in range(60)]
        candidate_rows = [row(128, f"case-{index:02}") for index in range(60)]
        result = _compare_profile_inputs(baseline_rows, candidate_rows,
                                        {"name": "s256-k8"}, {"name": "s128-k8"})
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["same_active_input_ids"], 60)

        changed = row(128, "case-00")
        changed["inputs"]["input_ids"][0][2] = 99
        with self.assertRaisesRegex(RuntimeErrorRecord, "有効input_ids"):
            _compare_profile_inputs(baseline_rows, [changed, *candidate_rows[1:]],
                                   {"name": "s256-k8"}, {"name": "s128-k8"})

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

    def test_noul_evaluator_threshold_uses_rounded_return_value(self):
        for raw, returned, judged in (
            (0.4999, 0.4999, False),
            (0.5, 0.5, True),
            (0.5001, 0.5001, True),
            (0.49996, 0.5, True),
        ):
            with self.subTest(raw=raw):
                actual_returned, actual_judged = _noul_return_value(raw)
                self.assertEqual(actual_returned, returned)
                self.assertIs(actual_judged, judged)

    def test_noul_output_keeps_raw_rounded_evaluator_and_argmax_values_separate(self):
        import math
        import numpy as np

        row = {"case_id": "noul-1", "question_id": "enabled", "input_sha256": "fixed",
               "question_type": "noul", "candidate_ids": [False, True], "candidate_count": 2,
               "temperature": 1.0}
        target_probability = 0.49996
        logits = np.asarray([0.0, math.log(target_probability / (1.0 - target_probability))], dtype=np.float64)
        output = _row_output(row, logits, np.asarray([0.0, 1.0], dtype=np.float64))

        self.assertAlmostEqual(output["true_probability"], target_probability, places=12)
        self.assertEqual(output["returned_noul"], 0.5)
        self.assertEqual(output["returned_noul_rounding"], "python_round_4")
        self.assertIs(output["boolean_result"], True)
        self.assertIs(output["argmax_boolean_result"], False)
        self.assertIs(output["decision"], False)

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
             "boolean_result": True, "argmax_boolean_result": True, "true_probability": 0.7,
             "returned_noul": 0.7, "returned_noul_rounding": "python_round_4"},
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
             "boolean_result": False, "argmax_boolean_result": False, "true_probability": 0.4,
             "returned_noul": 0.4, "returned_noul_rounding": "python_round_4"},
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
        self.assertEqual(by_type["noul"]["returned_noul_mean_denominator"], 1)


if __name__ == "__main__":
    unittest.main()

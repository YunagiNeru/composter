import unittest

from som_probe.runtime import _comparison, _execution_device_names, _npu_error_code


class RuntimeFailureTests(unittest.TestCase):
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
        self.assertIn("not human-label correctness", result["important_new_errors_definition"])


if __name__ == "__main__":
    unittest.main()

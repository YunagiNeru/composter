import io
import unittest
import sys
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from som_probe.diagnostics import (
    CONTRAST_FIXTURES,
    DIAGNOSTIC_CASES,
    DiagnosticError,
    SOURCE_RUN_ID,
    _compare_laya_input,
    _compare_laya_outputs,
    _compare_laya_custom_outputs,
    _normalise_adapter_answer,
    _normalise_laya_answer,
    _normalise_sokudan_official_answer,
    _target_mismatches,
    _run_child,
    _sokudan_outputs_match,
    main as diagnostic_main,
)


class FixedTryoutDiagnosticTests(unittest.TestCase):
    def test_contrast_questions_and_one_paraphrase_are_fixed(self):
        self.assertEqual([fixture["type"] for fixture in CONTRAST_FIXTURES], ["A", "B", "C"])
        self.assertEqual(len({fixture["question"] for fixture in CONTRAST_FIXTURES}), 3)
        self.assertTrue(all(fixture["paraphrase"] != fixture["question"]
                            for fixture in CONTRAST_FIXTURES))
        self.assertEqual(SOURCE_RUN_ID, "20261025-041751-58713e9e")

    def test_fixed_targets_read_score_stage_from_score_result_without_using_human_labels(self):
        records = {}
        for fixture in DIAGNOSTIC_CASES:
            predictions = {
                "choice": {"normalized": {"decision": fixture["provisional_targets"]["choice"]}},
                "score": {"normalized": {"stage_prediction": fixture["provisional_targets"]["score_stage"]}},
            }
            records[fixture["id"]] = {"answers": predictions, "official_api": predictions}
        self.assertEqual(_target_mismatches({"laya": {"records": records},
                                             "sokudan": {"records": records}}), [])

    def test_laya_direct_trace_accepts_only_the_expected_fixed_shape_padding(self):
        direct = {
            "question_id": "intent", "question_text": "本文の作業を判定",
            "api_question": {"type": "noul", "instructions": "本文の作業を判定",
                             "criteria": {"false": "対象外", "true": "作業"}},
            "candidate_ids": ["task", "none"], "rendered_options": ["作業", "対象外"],
            "state_token_ids": [17], "qtype": 0, "state_truncated": False,
            "state_tokens_dropped": 0, "special_token_ids": {"mask": 99},
            "language_parameter": None, "temperature_bucket": "choice:2",
            "effective_temperature": 1.25,
            "input_tensors": {"input_ids": [1, 99, 2], "attention_mask": [1, 1, 1],
                               "marker_pos": [1, 0], "marker_mask": [True, False]},
        }
        candidate = {
            "question_id": "intent", "question_text": "本文の作業を判定",
            "question_contract": {"type": "bool", "instructions": "本文の作業を判定",
                                  "criteria": {"false": "対象外", "true": "作業"}},
            "candidate_ids": ["task", "none"], "candidate_descriptions": ["作業", "対象外"],
            "state_token_ids": [17], "qtype": 0, "state_truncated": False,
            "state_tokens_dropped": 0, "required_total_tokens": 3,
            "language_parameter": None, "temperature_bucket": "choice:2", "temperature": 1.25,
            "special_token_ids": {"mask": 99, "pad": 0}, "input_sha256": "digest",
            "input_tensors": {"input_ids": [[1, 99, 2, 0, 0]], "attention_mask": [[1, 1, 1, 0, 0]],
                               "marker_pos": [[1, 0, 0]], "marker_mask": [[True, False, False]],
                               "qtype": [0]},
        }

        comparison = _compare_laya_input(direct, candidate)

        self.assertTrue(comparison["active_token_ids_match"])
        self.assertTrue(comparison["padding_only_shape_difference"])
        self.assertTrue(comparison["language_parameter_match"])
        self.assertTrue(comparison["temperature_bucket_match"])
        self.assertTrue(comparison["effective_temperature_match"])
        candidate["state_token_ids"] = [18]
        self.assertFalse(_compare_laya_input(direct, candidate)["state_token_ids_match"])
        candidate["temperature"] = 2.0
        self.assertFalse(_compare_laya_input(direct, candidate)["effective_temperature_match"])

    def test_laya_api_display_values_are_not_misreported_as_unrounded(self):
        noul = _normalise_laya_answer("noul", {"noul": 0.5001}, [False, True])
        self.assertIsNone(noul["probabilities"])
        self.assertIsNone(noul.get("true_probability"))
        self.assertEqual(noul["api_returned_value"], 0.5001)
        self.assertTrue(noul["decision"])

        score = _normalise_laya_answer(
            "score", {"score": 0.6, "probabilities": {"0": 0.45, "1": 0.50, "2": 0.05}},
            [0, 1, 2],
        )
        self.assertEqual(score["stage_prediction"], 1)
        self.assertIsNone(score["score_expected_unrounded"])
        self.assertEqual(score["returned_score"], 0.6)

    def test_noul_api_threshold_and_model_argmax_are_compared_separately(self):
        api = {"normalized": {
            "decision": True, "boolean_result": True, "evaluator_boolean": True,
            "api_returned_value": 0.5, "probabilities": None,
        }}
        cpu = {
            "decision": False, "boolean_result": True, "argmax_boolean_result": False,
            "candidate_ids": [False, True], "candidate_probabilities": [
                {"candidate_id": False, "probability": 0.50004},
                {"candidate_id": True, "probability": 0.49996},
            ], "returned_noul": 0.5,
        }

        api_to_cpu = _compare_laya_outputs(api, cpu, "noul")
        cpu_to_npu = _compare_laya_custom_outputs(cpu, dict(cpu), "noul")

        self.assertTrue(api_to_cpu["decision_match"])
        self.assertTrue(api_to_cpu["evaluator_boolean_match"])
        self.assertIsNone(api_to_cpu["argmax_boolean_match"])
        self.assertTrue(cpu_to_npu["decision_match"])
        self.assertTrue(cpu_to_npu["argmax_boolean_match"])

    def test_sokudan_comparison_compares_semantics_without_inventing_values(self):
        official = {"decision": True, "candidate_ids": [False, True],
                    "probabilities": None, "probability_origin": None,
                    "returned_noul": 0.5001, "boolean_result": True}
        adapter = dict(official)
        self.assertTrue(_sokudan_outputs_match("noul", official, adapter))
        adapter["returned_noul"] = 0.5
        self.assertFalse(_sokudan_outputs_match("noul", official, adapter))

    def test_sokudan_noul_adapter_keeps_official_rounding_origin(self):
        answer = {"api_returned_value": 0.5001, "evaluator_boolean": True,
                  "api_returned_value_rounding": "official_api_python_round_4"}
        normalized = _normalise_adapter_answer("noul", answer)
        self.assertEqual(normalized["returned_noul"], 0.5001)
        self.assertIsNone(normalized["true_probability"])
        self.assertTrue(normalized["boolean_result"])
        self.assertEqual(normalized["probability_origin"], "official_api_python_round_4")

    def test_sokudan_direct_normalizer_reads_official_values_without_the_adapter(self):
        choice = _normalise_sokudan_official_answer(
            "choice", {"type": "choice", "choice": "none",
                       "probabilities": {"task": 0.25, "none": 0.75}},
            ["task", "none"],
        )
        self.assertEqual(choice["decision"], "none")
        self.assertEqual(choice["probabilities"], [0.25, 0.75])

        score = _normalise_sokudan_official_answer(
            "score", {"type": "score", "score": 1.25,
                      "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}},
            [0, 1, 2],
        )
        self.assertEqual(score["stage_prediction"], 1)
        self.assertEqual(score["returned_score"], 1.25)

        noul = _normalise_sokudan_official_answer(
            "noul", {"type": "bool", "noul": 0.5}, [False, True],
        )
        self.assertEqual(noul["returned_noul"], 0.5)
        self.assertIs(noul["boolean_result"], True)

    def test_failed_child_process_saves_exit_code_and_diagnostics_in_the_run(self):
        process = Mock(returncode=17, stdout="", stderr="fixed diagnostic failure")
        run = Mock()

        with patch("som_probe.diagnostics.subprocess.run", return_value=process), \
             patch("som_probe.diagnostics.open_run", return_value=run):
            with self.assertRaisesRegex(DiagnosticError, "SOKUDAN_CHILD_EXIT_17"):
                _run_child(Path.cwd(), "20261025-000000-00000000", "sokudan", Path(sys.executable))

        run.write_json.assert_called_once_with("sokudan-process.json", {
            "phase": "sokudan",
            "command": [sys.executable, "-m", "som_probe.diagnostics", "--stage", "sokudan",
                        "--run-id", "20261025-000000-00000000"],
            "exit_code": 17, "stdout": "", "stderr": "fixed diagnostic failure",
        })

    def test_failed_stage_saves_exception_trace_to_its_local_run(self):
        run = Mock()
        with patch("som_probe.diagnostics.open_run", return_value=run), \
             patch("som_probe.diagnostics._run_stage", side_effect=ValueError("fixed stage failure")), \
             redirect_stdout(io.StringIO()):
            result = diagnostic_main(["--stage", "sokudan", "--run-id", "20261025-000000-00000000"])

        self.assertEqual(result, 2)
        args = run.write_json.call_args.args
        self.assertEqual(args[0], "sokudan-failure.json")
        self.assertIn("ValueError: fixed stage failure", args[1]["traceback"])


if __name__ == "__main__":
    unittest.main()

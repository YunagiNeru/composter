import unittest
import json
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

from som_probe.cli import _parser
from som_probe.storage import _SUCCESS_TERMINAL_STAGES
from som_probe.typed_model_eval import (
    _candidate_labels,
    _case_row,
    _gliner_infer,
    _gliner_selected_result,
    _gliner_task,
    _load_gliner,
    _load_cases,
    _model_parameters,
    _read_config,
    _ordered_laya_probabilities,
    _timed_workload,
)


ROOT = Path(__file__).resolve().parents[1]


class TypedModelEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock, cls.evaluation, cls.controls = _read_config(ROOT)
        cls.cases, _ = _load_cases(ROOT, cls.controls)

    def test_fixed_inputs_are_dev_v3_and_separate_controls(self):
        self.assertEqual(sum(case["dataset"] == "dev-v3" for case in self.cases), 60)
        self.assertEqual(sum(case["dataset"] == "typed-model-controls-v1" for case in self.cases), 22)
        self.assertTrue(all(case["verification_status"] == "ai_draft_unverified"
                            for case in self.cases))
        self.assertEqual(len({case["case_id"] for case in self.cases}), 82)

    def test_choice_and_score_labels_preserve_the_dev_v3_order_and_meaning(self):
        choice = next(case for case in self.cases
                      if case["dataset"] == "dev-v3" and case["question"]["type"] == "choice")
        score = next(case for case in self.cases
                     if case["dataset"] == "dev-v3" and case["question"]["type"] == "score")
        self.assertEqual(list(_candidate_labels(choice, self.evaluation)), ["task", "none", "need_info"])
        self.assertEqual(list(_candidate_labels(score, self.evaluation)), ["0", "1", "2"])
        self.assertEqual(_gliner_task(choice, self.evaluation)["decision"]["prompt"],
                         choice["question"]["instructions"])
        self.assertEqual(list(_gliner_task(choice, self.evaluation)["decision"]["labels"]),
                         ["task", "none", "need_info"])

    def test_noul_labels_are_fixed_in_false_then_true_order(self):
        noul = next(case for case in self.cases
                    if case["dataset"] == "dev-v3" and case["question"]["type"] == "bool")
        self.assertEqual(list(_candidate_labels(noul, self.evaluation)), ["false", "true"])
        self.assertEqual(list(_gliner_task(noul, self.evaluation)["decision"]["labels"]),
                         ["false", "true"])

    def test_gliner_public_confidence_is_only_a_selected_label_value(self):
        self.assertEqual(_gliner_selected_result({"label": "task", "confidence": 0.7}), ("task", 0.7))
        self.assertEqual(_gliner_selected_result({"label": "task", "confidence": "0.7"}), ("task", None))
        self.assertEqual(_gliner_selected_result({"labels": ["task", "none"]}), (None, None))

    def test_parameter_count_uses_laya_agent_model_and_gliner_model(self):
        class Inner:
            def parameters(self):
                return iter([type("Parameter", (), {"numel": lambda self: 3})()])

        class LayaAgent:
            model = Inner()

        class GLiNERModel:
            def parameters(self):
                return iter([type("Parameter", (), {"numel": lambda self: 7})()])

        self.assertEqual(sum(value.numel() for value in _model_parameters("laya-typed", LayaAgent())), 3)
        self.assertEqual(sum(value.numel() for value in _model_parameters("gliner25-multi-decide", GLiNERModel())), 7)

    def test_gliner_loader_uses_only_supported_offline_arguments(self):
        class Parameter:
            device = "cpu"
            dtype = "torch.float32"

            def numel(self):
                return 1

            def is_floating_point(self):
                return True

        model = Mock()
        model.architecture = "boundary"
        model.parameters.return_value = [Parameter()]
        auto = SimpleNamespace(from_pretrained=Mock(return_value=model))
        fake_torch = SimpleNamespace(__version__="2.9.1+cpu", float32="torch.float32",
                                     cuda=SimpleNamespace(is_available=lambda: False))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            (model_dir / "encoder_config").mkdir(parents=True)
            (model_dir / "encoder_config" / "config.json").write_text(
                json.dumps({"max_position_embeddings": 512}), encoding="utf-8")
            with patch.dict(sys.modules, {"gliner2": SimpleNamespace(AutoExtractor=auto)}):
                loaded, info = _load_gliner(root, model_dir, {}, fake_torch)

        self.assertIs(loaded, model)
        auto.from_pretrained.assert_called_once_with(
            str(model_dir), map_location="cpu", local_files_only=True)
        self.assertFalse(info["remote_code"])

    def test_laya_score_stage_uses_candidate_order_not_rounded_expected_value(self):
        labels = {"0": "なし", "1": "日付か時刻", "2": "日付と時刻"}
        self.assertEqual(_ordered_laya_probabilities(
            "score", {"probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}, labels),
            [0.2, 0.4, 0.4])

    def test_overlong_gliner_input_is_rejected_without_calling_model(self):
        case = next(case for case in self.cases if case["question"]["type"] == "choice")

        class UnusedModel:
            def classify_text(self, *args, **kwargs):
                raise AssertionError("overlong input must not reach inference")

        trace = {"status": "LENGTH_EXCEEDED", "encoded_input_id_count_including_schema": 513,
                 "truncation_performed": False}
        with patch("som_probe.typed_model_eval._gliner_preflight", return_value=trace):
            result = _gliner_infer(UnusedModel(), case, self.evaluation, 512)
        self.assertEqual(result["status"], "LENGTH_EXCEEDED")
        self.assertIsNone(result["prediction"])
        self.assertIsNone(result["model_specific_output"])

    def test_performance_failures_keep_counts_and_safe_error_types(self):
        def fail():
            raise ValueError("synthetic failure")

        result = _timed_workload(fail, repeats=2, warmups=1)
        self.assertEqual(result["warmup_failures"], 1)
        self.assertEqual(result["warmup_failure_codes"], {"ValueError": 1})
        self.assertEqual(result["failures"], 2)
        self.assertEqual(result["failure_codes"], {"ValueError": 2})
        self.assertEqual(result["successes"], 0)

    def test_case_diagnostic_keeps_model_decision_and_unverified_label_separate(self):
        case = next(case for case in self.cases if case["question"]["type"] == "choice")
        case["allowed_answers"] = ["task"]
        row = _case_row(case, {"status": "PASS", "prediction": "none",
                               "expected_value": None, "input": {"truncation_performed": False},
                               "model_specific_output": {"api": "raw"},
                               "adapter_output": {"decision": "none"}}, "fixture-model")
        self.assertEqual(row["common_prediction"], "none")
        self.assertFalse(row["draft_label_match"])
        self.assertEqual(row["human_quality_status"], "UNDETERMINED")
        self.assertEqual(row["observable_wrong_outputs_derived_from_allowed_answers"], ["none", "need_info"])

    def test_cpu_command_has_only_fixed_three_run_trial_count(self):
        args = _parser().parse_args(["typed-model-cpu", "--candidate", "both", "--trials", "3"])
        self.assertEqual(args.candidate, "both")
        self.assertEqual(args.trials, 3)

    def test_cpu_run_is_a_terminal_stage(self):
        self.assertIn("TYPED_MODEL_CPU", _SUCCESS_TERMINAL_STAGES)
        self.assertIn("TYPED_MODEL_ANALYSIS", _SUCCESS_TERMINAL_STAGES)


if __name__ == "__main__":
    unittest.main()

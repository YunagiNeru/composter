import os
import tempfile
import unittest
import copy
from pathlib import Path
from unittest.mock import patch

from som_probe.tryout import (
    TryoutError,
    _ExclusiveModelLock,
    _interactive_loop,
    _question_templates,
    _questions_for_templates,
    _tryout_error_text,
    _load_sokudan_session_without_progress,
    _LayaSession,
    format_results,
    run_tryout,
)

ROOT = Path(__file__).resolve().parents[1]


def sample_results():
    return {
        "choice": {
            "question_type": "choice",
            "decision": "task",
            "candidate_probabilities": [
                {"candidate_id": "task", "probability": 0.7},
                {"candidate_id": "none", "probability": 0.2},
                {"candidate_id": "need_info", "probability": 0.1},
            ],
        },
        "score": {
            "question_type": "score",
            "decision_slot": 2,
            "candidate_probabilities": [
                {"candidate_id": 0, "probability": 0.1},
                {"candidate_id": 1, "probability": 0.2},
                {"candidate_id": 2, "probability": 0.7},
            ],
            "score_expected_unrounded": 1.6,
            "returned_score": 1.6,
        },
        "noul": {
            "question_type": "noul",
            "boolean_result": True,
            "true_probability": 0.8,
            "returned_noul": 0.8,
        },
    }


class TryoutFormattingTests(unittest.TestCase):
    def test_dev_v3_uses_common_choice_rule_without_case_specific_hints(self):
        templates, canaries = _question_templates(ROOT)

        self.assertEqual(canaries["choice"]["case_id"], "task-001")
        self.assertIn("本文に明示された本人の行動", templates["choice"]["instructions"])
        self.assertNotIn("明日15時", templates["choice"]["instructions"])

    def test_all_three_sections_preserve_the_input_values(self):
        results = sample_results()
        before = copy.deepcopy(results)

        rendered = format_results(results)

        self.assertIn("【モデルの判断】", rendered)
        self.assertIn("作業の状態：未完了", rendered)
        self.assertIn("日時の具体性：2 — 日付（相対日付を含む）と時刻の両方", rendered)
        self.assertIn("通知候補：はい", rendered)
        self.assertIn("【判断の内訳】", rendered)
        self.assertIn("【数値照合用】", rendered)
        self.assertIn("段階の平均値（確率期待値", rendered)
        self.assertIn("Score選択段階（argmax）：2", rendered)
        self.assertIn("Score段階期待値（丸め前）：1.6", rendered)
        self.assertEqual(results, before)
        self.assertFalse(any("理由" in line for line in rendered.splitlines()))

    def test_score_argmax_is_not_replaced_by_probability_expectation(self):
        results = sample_results()
        results["score"].update({
            "decision_slot": 1,
            "candidate_probabilities": [
                {"candidate_id": 0, "probability": 0.45},
                {"candidate_id": 1, "probability": 0.50},
                {"candidate_id": 2, "probability": 0.05},
            ],
            "score_expected_unrounded": 0.60,
            "returned_score": 0.6,
        })

        rendered = format_results(results)

        self.assertIn("日時の具体性：1 — 日付・相対日付・曜日・期間だけ、または時刻だけ", rendered)
        self.assertIn("選択段階：argmaxで選択した段階 1", rendered)
        self.assertIn("段階の平均値（確率期待値、表示は小数4桁）：0.6000／2", rendered)
        self.assertIn("Score選択段階（argmax）：1", rendered)
        self.assertIn("Score段階期待値（丸め前）：0.6", rendered)

    def test_near_tie_keeps_the_returned_choice_and_marks_exact_ties(self):
        results = sample_results()
        results["choice"].update({
            "decision": "none",
            "candidate_probabilities": [
                {"candidate_id": "task", "probability": 0.49996},
                {"candidate_id": "none", "probability": 0.50000},
                {"candidate_id": "need_info", "probability": 0.00004},
            ],
        })
        rendered = format_results(results)
        none_line = next(line for line in rendered.splitlines() if '"none"：対象外' in line)
        self.assertIn("← 選択", none_line)
        self.assertNotIn("確実", rendered)

        results["choice"]["candidate_probabilities"][0]["probability"] = 0.5
        results["choice"]["candidate_probabilities"][1]["probability"] = 0.5
        rendered = format_results(results)
        self.assertIn("同率候補あり（選択はAPI／モデルが返した候補を表示）", rendered)
        self.assertIn('"none"：対象外　50.00% ← 選択', rendered)

    def test_sokudan_adapter_fields_render_without_rewriting_api_values(self):
        results = {
            "choice": {
                "question_type": "choice", "decision": "none",
                "candidate_ids": ["task", "none"],
                "api_display_probabilities": [0.2, 0.8],
            },
            "score": {
                "question_type": "score", "decision_slot": 1,
                "candidate_ids": [0, 1],
                "api_returned_value": 0.75,
                "api_display_probabilities": [0.4, 0.6],
            },
            "noul": {
                "question_type": "noul", "evaluator_boolean": True,
                "api_returned_value": 0.5001,
            },
        }

        rendered = format_results(results)

        self.assertIn("作業の状態：対象外", rendered)
        self.assertIn("通知候補：はい", rendered)
        self.assertIn("Score API返却値（公式API返却値）：0.75", rendered)
        self.assertIn("P(true)の表示：50.01%（公式API返却値・小数4桁丸め済み）", rendered)
        self.assertIn("Noul P(true)（丸め前モデル値）：未取得", rendered)
        self.assertIn("Noul 公式API返却値：0.5001", rendered)
        self.assertNotIn("0.50010000", rendered)

    def test_noul_threshold_uses_api_return_value_and_explains_half_boundary(self):
        results = sample_results()
        results["noul"] = {
            "question_type": "noul", "true_probability": 0.49999,
            "returned_noul": 0.5, "boolean_result": True,
        }

        rendered = format_results(results)

        self.assertIn("Noul P(true)（丸め前モデル値）：0.49999", rendered)
        self.assertIn("Noul モデル返却値：0.5", rendered)
        self.assertIn("今回値は境界の0.5", rendered)
        self.assertIn("通知候補：はい", rendered)

    def test_missing_raw_values_are_not_fabricated(self):
        results = sample_results()
        results["score"].pop("score_expected_unrounded")
        results["noul"] = {"question_type": "noul", "api_returned_value": 0.5,
                            "evaluator_boolean": True}

        rendered = format_results(results)

        self.assertIn("Score段階期待値（丸め前）：未取得", rendered)
        self.assertIn("Noul P(true)（丸め前モデル値）：未取得", rendered)
        self.assertIn("Noul 公式API返却値：0.5", rendered)
        self.assertNotIn("0.5000（丸め前", rendered)

    def test_official_noul_without_candidate_probabilities_displays_as_unavailable(self):
        results = sample_results()
        results["noul"] = {
            "question_type": "noul", "candidate_ids": [False, True],
            "api_returned_value": 0.5001, "evaluator_boolean": True,
        }
        context = {
            "choice": {"candidate_ids": ["task", "none", "need_info"]},
            "score": {"candidate_ids": [0, 1, 2]},
            "noul": {"candidate_ids": [False, True]},
        }

        rendered = format_results(results, context)

        self.assertIn("通知候補：はい", rendered)
        self.assertIn("候補確率：未取得", rendered)
        self.assertIn("P(true)の表示：50.01%", rendered)
        self.assertIn("Noul 公式API返却値：0.5001", rendered)

    def test_present_but_incomplete_probability_array_is_a_contract_error(self):
        results = sample_results()
        results["choice"].pop("candidate_probabilities")
        results["choice"].update({
            "candidate_ids": ["task", "none", "need_info"],
            "api_display_probabilities": [],
            "decision": "task",
        })

        with self.assertRaisesRegex(TryoutError, "DISPLAY_PROBABILITY_CONTRACT_INVALID"):
            format_results(results, {"choice": {"candidate_ids": ["task", "none", "need_info"]}})

    def test_boolean_and_returned_probability_mismatch_is_shown_without_replacement(self):
        results = sample_results()
        results["noul"] = {"question_type": "noul", "true_probability": 0.2,
                            "returned_noul": 0.2, "boolean_result": True}

        rendered = format_results(results)

        self.assertIn("通知候補：はい（返却値との真偽契約に不一致）", rendered)
        self.assertIn("契約エラー：返却値から計算する真偽", rendered)
        self.assertIn("Noul モデル返却値：0.2", rendered)

    def test_candidate_order_is_checked_and_contract_failure_has_a_plain_explanation(self):
        results = sample_results()
        results["choice"]["candidate_ids"] = ["none", "task", "need_info"]
        with self.assertRaisesRegex(TryoutError, "DISPLAY_CANDIDATE_ORDER_MISMATCH"):
            format_results(results, {"choice": {"candidate_ids": ["task", "none", "need_info"]}})
        self.assertIn("候補順とモデル出力", _tryout_error_text(TryoutError("DISPLAY_CANDIDATE_ORDER_MISMATCH")))
        self.assertIn("切り捨てていません", _tryout_error_text(TryoutError("INPUT_EXCEEDS_LAYA_SHAPE")))

    def test_sokudan_question_ids_remain_the_canonical_ids(self):
        templates = {kind: {"type": kind} for kind in ("choice", "score", "noul")}
        canaries = {
            "choice": {"question_id": "intent"},
            "score": {"question_id": "specificity"},
            "noul": {"question_id": "notify"},
        }

        questions = _questions_for_templates(templates, canaries)

        self.assertEqual(list(questions), ["intent", "specificity", "notify"])
        self.assertEqual(questions["intent"], templates["choice"])

    def test_duplicate_question_ids_are_rejected(self):
        templates = {kind: {} for kind in ("choice", "score", "noul")}
        canaries = {kind: {"question_id": "same"} for kind in templates}

        with self.assertRaisesRegex(TryoutError, "TRYOUT_TEMPLATE_ID_DUPLICATE"):
            _questions_for_templates(templates, canaries)


class TryoutLifecycleTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows named mutex is used by the tryout CLI")
    def test_named_lock_rejects_a_second_model_session(self):
        with _ExclusiveModelLock():
            with self.assertRaisesRegex(TryoutError, "ANOTHER_MODEL_SESSION_IS_RUNNING"):
                with _ExclusiveModelLock():
                    self.fail("second session acquired the model mutex")

    @unittest.skipUnless(os.name == "nt", "Windows named mutex is used by the tryout CLI")
    def test_quit_and_ctrl_c_close_session_and_release_lock_without_saving_text(self):
        marker = "この入力はどのファイルにも保存されない"

        class FakeSession:
            startup = f"fake model / CPU / PID：{os.getpid()}"

            def __init__(self):
                self.inputs = []
                self.closed = False

            def predict(self, text):
                self.inputs.append(text)
                return sample_results()

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = FakeSession()
            output = []
            answers = iter((marker, ":quit"))
            run_tryout(root, "sokudan", None, lambda _prompt: next(answers), output.append,
                       lambda: first)

            self.assertEqual(first.inputs, [marker])
            self.assertTrue(first.closed)
            self.assertTrue(any(f"PID：{os.getpid()}" in line for line in output))
            self.assertIn(f"入力：{marker}", "\n".join(output))
            self.assertEqual(list(root.iterdir()), [])

            second = FakeSession()
            messages = iter((KeyboardInterrupt(),))

            def interrupt(_prompt):
                raise next(messages)

            run_tryout(root, "sokudan", None, interrupt, lambda _line: None, lambda: second)

            self.assertTrue(second.closed)
            self.assertEqual(list(root.iterdir()), [])

    def test_sokudan_loading_progress_is_disabled_only_while_loading(self):
        from transformers.utils import logging as transformers_logging

        previous = transformers_logging.is_progress_bar_enabled()
        transformers_logging.enable_progress_bar()
        observed = []
        try:
            result = _load_sokudan_session_without_progress(
                lambda: observed.append(transformers_logging.is_progress_bar_enabled()) or "ready"
            )
        finally:
            if previous:
                transformers_logging.enable_progress_bar()
            else:
                transformers_logging.disable_progress_bar()
        self.assertEqual(result, "ready")
        self.assertEqual(observed, [False])
        self.assertEqual(transformers_logging.is_progress_bar_enabled(), previous)


class LayaStructuredChoiceTests(unittest.TestCase):
    def test_choice_entry_returns_structured_output_without_formatting_or_saving_state(self):
        state = "cpu_usage: 90\nram_usage: 15\nuser_event: NONE"
        question = {"type": "choice", "instructions": "Select one fixed action.",
                    "criteria": {"DROWSY_SHUFFLE": "slow, sleepy gait",
                                 "BRISK_WALK": "energetic gait"}}
        expected = {"question_type": "choice", "decision": "DROWSY_SHUFFLE",
                    "candidate_ids": ["DROWSY_SHUFFLE", "BRISK_WALK"],
                    "probabilities": [0.8, 0.2]}
        row = {
            "question_id": "pet_gait", "question_type": "choice", "question_text": "Select one fixed action.",
            "question_contract": question, "candidate_ids": expected["candidate_ids"],
            "candidate_descriptions": ["slow, sleepy gait", "energetic gait"],
            "required_total_tokens": 25, "input_sha256": "f" * 64, "state_truncated": False,
            "temperature": 1.0, "candidate_count": 2,
            "arrays": {"input_ids": [1], "attention_mask": [1], "marker_pos": [1],
                        "marker_mask": [True], "qtype": [0]},
        }

        class Port:
            def get_any_name(self):
                return "output"

        session = object.__new__(_LayaSession)
        session._tokenizer = object()
        session._config = {}
        session._compiled = lambda inputs: {Port(): inputs["input_ids"]}
        with patch("som_probe.tryout._laya_input", return_value=row) as build_input, \
                patch("som_probe.tryout._select_outputs", return_value=([3.0, 1.0], [1.0, 0.0])), \
                patch("som_probe.tryout.decode_laya_output", return_value=expected) as decode:
            output, trace = session.predict_choice("pet_gait", question, state)

        self.assertIs(output, expected)
        self.assertEqual(decode.call_args.args[:3], ("choice", expected["candidate_ids"], 1.0))
        self.assertEqual(build_input.call_args.args[-1], state)
        self.assertEqual(trace["input_sha256"], "f" * 64)
        self.assertNotIn(state, repr(trace))
        self.assertNotIn("input_tensors", trace)

    def test_choice_entry_rejects_non_choice_contract_before_inference(self):
        session = object.__new__(_LayaSession)
        with self.assertRaisesRegex(TryoutError, "PET_CHOICE_REQUEST_CONTRACT_INVALID"):
            session.predict_choice("pet_gait", {"type": "score"}, "cpu_usage: 90")


class TryoutInputWaitTests(unittest.TestCase):
    def test_waiting_and_blank_input_do_not_call_inference(self):
        class FakeSession:
            calls = 0

            def predict(self, _text):
                self.calls += 1
                return sample_results()

        session = FakeSession()
        output = []
        answers = iter(("", ":quit"))

        _interactive_loop(session, lambda _prompt: next(answers), output.append)

        self.assertEqual(session.calls, 0)
        self.assertTrue(any("空の文章" in line for line in output))


if __name__ == "__main__":
    unittest.main()

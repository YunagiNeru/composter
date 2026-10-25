import os
import tempfile
import unittest
from pathlib import Path

from som_probe.tryout import (
    TryoutError,
    _ExclusiveModelLock,
    _interactive_loop,
    _question_templates,
    _questions_for_templates,
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

    def test_japanese_decisions_come_before_numeric_details(self):
        lines = format_results(sample_results()).splitlines()

        self.assertEqual(lines[:3], [
            "作業の状態：未完了",
            "日時の具体性：2 — 日付または相対日付と時計時刻の両方が明記",
            "通知候補：はい",
        ])
        self.assertEqual(lines[3], "数値（参考）:")
        self.assertTrue(any("丸め前期待値=1.6" in line for line in lines))
        self.assertFalse(any("理由" in line or "文章" in line for line in lines))

    def test_sokudan_adapter_fields_render_without_rewriting_api_values(self):
        results = {
            "choice": {
                "question_type": "choice", "decision": "none",
                "candidate_ids": ["task", "none"],
                "api_display_probabilities": [0.2, 0.8],
            },
            "score": {
                "question_type": "score", "decision_slot": 1,
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
        self.assertIn("返却Score=0.75", rendered)
        self.assertIn("P(true)=0.50010000", rendered)
        self.assertIn("API返却P(true)=0.5001", rendered)

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
            startup = "fake model / CPU"

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
            self.assertTrue(any(line.startswith("推論プロセス PID：") for line in output))
            self.assertNotIn(marker, "\n".join(output))
            self.assertEqual(list(root.iterdir()), [])

            second = FakeSession()
            messages = iter((KeyboardInterrupt(),))

            def interrupt(_prompt):
                raise next(messages)

            run_tryout(root, "sokudan", None, interrupt, lambda _line: None, lambda: second)

            self.assertTrue(second.closed)
            self.assertEqual(list(root.iterdir()), [])


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

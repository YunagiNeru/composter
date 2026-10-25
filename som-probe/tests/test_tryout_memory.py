import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from som_probe import tryout_memory


class FakeRun:
    run_id = "20261025-130000-aaaaaaaa"

    def __init__(self):
        self.records = {}
        self.stages = []
        self.failures = []

    def read_manifest(self):
        return {
            "git_revision": "fixed-commit", "source_tree_sha256": "a" * 64,
            "worktree_status": "clean",
            "input_sha256": {"datasets/dev-v3.jsonl": "b" * 64},
        }

    def write_json(self, name, value):
        self.records[name] = value

    def stage(self, name, status, **fields):
        self.stages.append((name, status, fields))

    def failure(self, stage, code, error, **fields):
        self.failures.append((stage, code, fields))

    def blocked(self, stage, code, message, **fields):
        self.stages.append((stage, "BLOCKED", {"code": code, **fields}))


class TryoutMemoryTests(unittest.TestCase):
    def test_startup_pid_is_not_saved(self):
        text = tryout_memory._private_startup_text("Sokudan｜CPU｜PID：12345")
        self.assertEqual(text, "Sokudan｜CPU｜PID：[記録対象外]")

    def test_memory_samples_after_load_inference_and_idle_without_saving_text(self):
        run = FakeRun()
        samples = iter((
            {"status": "available", "working_set_bytes": 10, "private_bytes": 20},
            {"status": "available", "working_set_bytes": 11, "private_bytes": 21},
            {"status": "available", "working_set_bytes": 12, "private_bytes": 22},
        ))
        observed = {}

        def fake_tryout(root, model, source_run_id, input_fn, output_fn, **kwargs):
            observed["model"] = model
            observed["source_run_id"] = source_run_id
            observed["diagnostic_run_id"] = kwargs["diagnostic_run_id"]
            output_fn("ロード完了：Sokudan｜CPU｜PID：98765")
            observed["input"] = input_fn("文章> ")
            output_fn("【モデルの判断】")
            observed["quit"] = input_fn("文章> ")

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(tryout_memory.Run, "create", return_value=run), \
                patch.object(tryout_memory, "_expected_python", return_value=Path(sys.executable)), \
                patch.object(tryout_memory, "_memory_reader", return_value=lambda: next(samples)), \
                patch.object(tryout_memory.time, "sleep") as sleep, \
                patch("som_probe.tryout.run_tryout", side_effect=fake_tryout):
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "requirements").mkdir()
            (root / "configs" / "sokudan-cpu.lock.json").write_text("{}", encoding="utf-8")
            (root / "requirements" / "sokudan-cpu-win-x64.lock").write_text("fixed", encoding="utf-8")
            result = tryout_memory.run_tryout_memory(root, "sokudan")

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(set(result["phases"]), {
            "after_model_load", "after_one_inference", "after_idle_wait",
        })
        self.assertEqual(result["idle_wait_seconds"], 60)
        self.assertFalse(result["fixed_input_text_saved"])
        self.assertFalse(result["inference_output_saved"])
        self.assertNotIn(tryout_memory.FIXED_INPUT, str(run.records))
        self.assertNotIn("98765", str(run.records))
        self.assertEqual(observed["input"], tryout_memory.FIXED_INPUT)
        self.assertEqual(observed["quit"], ":quit")
        self.assertIsNone(observed["diagnostic_run_id"])
        sleep.assert_called_once_with(60)
        self.assertEqual(run.stages[-1][1], "PASS")


if __name__ == "__main__":
    unittest.main()

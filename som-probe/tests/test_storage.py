import tempfile
import unittest
from pathlib import Path

from som_probe.storage import Run, safe_message


class StorageTests(unittest.TestCase):
    def test_safe_message_redacts_paths_and_secrets(self):
        fake_path = "Q:" + "\\" + "fixture\\private\\data.json"
        fake_secret = "hf" + "_" + ("a" * 24)
        result = safe_message(f"failed {fake_path} {fake_secret}")
        self.assertNotIn(fake_path, result)
        self.assertNotIn(fake_secret, result)

    def test_run_does_not_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = Run.create(root, "G0")
            run.write_jsonl("cases.jsonl", [{"case_id": "case-001"}])
            with self.assertRaises(FileExistsError):
                run.write_jsonl("cases.jsonl", [{"case_id": "case-002"}])
            run.write_json("metadata.json", {"status": "BLOCKED"})
            with self.assertRaises(FileExistsError):
                run.write_json("metadata.json", {"status": "PASS"})

    def test_blocked_stage_keeps_the_failure_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Run.create(Path(directory), "G3")
            run.blocked("G3", "ARCHITECTURE_MISMATCH", "expected 3720, actual other")
            self.assertEqual(run.read_manifest()["stages"]["G3"]["status"], "BLOCKED")
            self.assertIn("ARCHITECTURE_MISMATCH", (run.path / "errors.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

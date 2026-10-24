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

    def test_run_pins_profile_and_input_comparison_source(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Run.create(Path(directory), "G3", {
                "name": "s128-k8", "batch": 1, "sequence_length": 128, "candidate_slots": 8,
            }, "20261024-000000-00000000")
            manifest = run.read_manifest()

            self.assertEqual(manifest["profile"], "s128-k8")
            self.assertEqual(manifest["profile_config"]["sequence_length"], 128)
            self.assertEqual(manifest["compare_input_run_id"], "20261024-000000-00000000")
            self.assertEqual(run.profile_config["name"], "s128-k8")

    def test_blocked_stage_keeps_the_failure_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Run.create(Path(directory), "G3")
            run.blocked("G3", "ARCHITECTURE_MISMATCH", "expected 3720, actual other")
            self.assertEqual(run.read_manifest()["stages"]["G3"]["status"], "BLOCKED")
            self.assertIn("ARCHITECTURE_MISMATCH", (run.path / "errors.jsonl").read_text(encoding="utf-8"))

    def test_run_records_revision_worktree_and_configuration_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "configs/probe.json", "configs/gates.json", "configs/models.lock.json",
                "datasets/manifest.json", "datasets/dev.jsonl",
                "requirements/run-win-x64.lock", "requirements/eval-win-x64.lock",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            run = Run.create(root, "G3")

            manifest = run.read_manifest()

            self.assertIn("git_revision", manifest)
            self.assertIn("source_tree_sha256", manifest)
            self.assertIsNone(manifest["worktree_dirty"])
            self.assertEqual(manifest["worktree_status"], "unavailable")
            self.assertEqual(set(manifest["input_sha256"]), {
                "configs/probe.json", "configs/gates.json", "configs/models.lock.json",
                "datasets/manifest.json", "datasets/dev.jsonl",
                "requirements/run-win-x64.lock", "requirements/eval-win-x64.lock",
            })
            self.assertTrue(all(len(value) == 64 for value in manifest["input_sha256"].values()))


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import unittest
from pathlib import Path

from som_probe.dataset import _strict_json, load_dataset


ROOT = Path(__file__).resolve().parents[1]


class DatasetVersionTests(unittest.TestCase):
    def test_original_and_revision_are_complete_separate_strict_datasets(self):
        original = load_dataset(ROOT, "dev-v1")
        revised = load_dataset(ROOT, "dev-v2")

        self.assertEqual(len(original["cases"]), 60)
        self.assertEqual(len(revised["cases"]), 60)
        self.assertEqual(original["identity"]["dataset_sha256"],
                         "dc63d8abf55a74acaf330db5f0dbd1be507a532d32a5def4d22ec8e35f26bb1c")
        self.assertEqual(revised["identity"]["dataset_sha256"],
                         "ab98482280a290fbfeda75c4ce755d072cae84fc2cd2552dbb24fc2a97102380")
        self.assertNotEqual(original["identity"]["dataset_sha256"],
                            revised["identity"]["dataset_sha256"])
        self.assertEqual(revised["identity"]["revision_ledger_sha256"],
                         hashlib.sha256((ROOT / "datasets/dev-v2-revisions.json").read_bytes()).hexdigest())
        self.assertEqual(revised["manifest"]["human_review_status"], "pending_for_all_60_cases")
        self.assertTrue(all(label["verification_status"] == "ai_draft_unverified"
                            for case in revised["cases"] for label in case["expected"].values()))
        self.assertTrue(all("target_errors" not in label
                            for case in revised["cases"] for label in case["expected"].values()))

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            _strict_json('{"case_id":"first","case_id":"second"}')

    def test_approved_choice_and_score_boundary_proposals_are_recorded(self):
        revised = load_dataset(ROOT, "dev-v2")
        cases = {case["case_id"]: case for case in revised["cases"]}
        self.assertEqual(cases["task-014"]["expected"]["intent"]["allowed_answers"], ["task"])
        self.assertEqual(cases["task-002"]["expected"]["intent"]["allowed_answers"], ["none"])
        self.assertEqual(cases["task-017"]["expected"]["intent"]["allowed_answers"], ["task"])
        for case_id in ("time-011", "time-017"):
            self.assertEqual(cases[case_id]["expected"]["specificity"]["allowed_answers"], [1])
        contract = revised["contract"]
        boundary = next(item for item in contract["score"]["boundary_examples"]
                        if item["id"] == "clock_time_only")
        self.assertEqual(boundary["expected_stage"], 1)
        self.assertFalse(boundary["in_dev_dataset"])

    def test_revision_ledger_covers_every_case_without_human_confirmation(self):
        revised = load_dataset(ROOT, "dev-v2")
        ledger = json.loads((ROOT / "datasets/dev-v2-revisions.json").read_text(encoding="utf-8"))
        self.assertEqual(set(ledger["cases"]), {case["case_id"] for case in revised["cases"]})
        self.assertTrue(all(entry["verification_status"] == "ai_draft_unverified"
                            and entry["human_review"] == "未実施"
                            and entry["reason"]
                            and isinstance(entry["changed_fields"], list)
                            for entry in ledger["cases"].values()))


if __name__ == "__main__":
    unittest.main()

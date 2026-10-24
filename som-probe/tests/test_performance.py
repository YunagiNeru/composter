import unittest
import os

from som_probe.performance import _four_request_bundles, _memory
from som_probe.cli import _parser
from som_probe.runtime import process_age_ms


class PerformancePlanTests(unittest.TestCase):
    def test_four_request_bundles_cover_each_fixed_case_once(self):
        rows = [
            {"case_id": f"{kind}-{index}", "question_type": kind}
            for kind in ("choice", "score", "noul")
            for index in range(20)
        ]

        bundles = _four_request_bundles(rows)
        flattened = [row["case_id"] for bundle in bundles for row in bundle]

        self.assertEqual(len(bundles), 15)
        self.assertTrue(all(len(bundle) == 4 for bundle in bundles))
        self.assertEqual(len(flattened), 60)
        self.assertEqual(len(set(flattened)), 60)
        self.assertEqual(set(flattened), {row["case_id"] for row in rows})

    def test_cli_exposes_separate_cache_and_idle_measurement_settings(self):
        args = _parser().parse_args([
            "perf-run", "--source-run", "20261024-000000-00000000",
            "--backend", "npu", "--trial", "2",
            "--cache-from-run", "20261024-000000-00000001", "--idle-seconds", "300",
        ])

        self.assertEqual(args.backend, "npu")
        self.assertEqual(args.cache_from_run, "20261024-000000-00000001")
        self.assertEqual(args.idle_seconds, 300)

    @unittest.skipUnless(os.name == "nt", "Windows APIを使うメモリ測定の確認")
    def test_windows_process_memory_and_startup_time_are_available(self):
        memory = _memory()
        age = process_age_ms()

        self.assertEqual(memory["status"], "available")
        self.assertGreater(memory["working_set_bytes"], 0)
        self.assertGreater(memory["private_bytes"], 0)
        self.assertIn("peak_pagefile_usage_bytes", memory)
        self.assertIsNotNone(age)
        self.assertGreater(age, 0)


if __name__ == "__main__":
    unittest.main()

import unittest
import os
from types import SimpleNamespace

from som_probe.performance import _four_request_bundles, _host_cpu_observation, _memory, _preprocess_case
from som_probe.cli import _parser
from som_probe.runtime import RuntimeErrorRecord, process_age_ms


class PerformancePlanTests(unittest.TestCase):
    def test_128_token_input_is_kept_whole_or_explicitly_rejected(self):
        class FixedTokenizer:
            def __init__(self, state_tokens):
                self.state_tokens = state_tokens

            def encode(self, text, add_special_tokens=False):
                ids = [7] * self.state_tokens if text == "状態" else [8]
                return SimpleNamespace(ids=ids)

        case = {
            "case_id": "length-128", "state": "状態",
            "questions": {"intent": {"type": "choice", "instructions": "選ぶ",
                                       "criteria": {"a": "候補A", "b": "候補B"}}},
        }
        saved = {"case_id": "length-128", "candidate_count": 2, "candidate_ids": ["a", "b"]}
        special = {"cls": 101, "sep": 102, "mask": 103, "pad": 0}
        mask_token = "[MASK]"

        accepted, token_count = _preprocess_case(case, saved, FixedTokenizer(119), special, mask_token, 128)
        self.assertEqual(token_count, 128)
        self.assertEqual(sum(accepted["inputs"]["attention_mask"][0]), 128)
        self.assertEqual(len(accepted["inputs"]["input_ids"][0]), 128)

        with self.assertRaisesRegex(RuntimeErrorRecord, "完全な入力が128トークン"):
            _preprocess_case(case, saved, FixedTokenizer(120), special, mask_token, 128)

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
            "--load-method", "path-compile",
        ])

        self.assertEqual(args.backend, "npu")
        self.assertEqual(args.cache_from_run, "20261024-000000-00000001")
        self.assertEqual(args.idle_seconds, 300)
        self.assertEqual(args.load_method, "path-compile")

        memory_args = _parser().parse_args([
            "memory-run", "--source-run", "20261024-000000-00000000",
            "--backend", "npu", "--load-method", "read-compile",
        ])
        self.assertEqual(memory_args.backend, "npu")
        self.assertEqual(memory_args.load_method, "read-compile")

    def test_host_cpu_observation_keeps_system_and_process_estimates_separate(self):
        observation = _host_cpu_observation((100, 1000), (110, 1100), 1.0, 0.1)

        self.assertEqual(observation["status"], "measured")
        self.assertEqual(observation["system_cpu_busy_percent"], 90.0)
        self.assertGreaterEqual(observation["estimated_other_host_busy_percent"], 0.0)
        self.assertEqual(observation["process_cpu_seconds"], 0.1)

    def test_cli_selects_fixed_profile_and_input_comparison_source(self):
        args = _parser().parse_args([
            "run-init", "--purpose", "G3", "--profile", "s128-k8",
            "--compare-input-run", "20261024-000000-00000000",
        ])
        self.assertEqual(args.profile, "s128-k8")
        self.assertEqual(args.compare_input_run, "20261024-000000-00000000")

    @unittest.skipUnless(os.name == "nt", "Windows APIを使うメモリ測定の確認")
    def test_windows_process_memory_and_startup_time_are_available(self):
        memory = _memory()
        age = process_age_ms()

        self.assertEqual(memory["status"], "available")
        self.assertGreater(memory["working_set_bytes"], 0)
        self.assertGreater(memory["private_bytes"], 0)
        self.assertIn("peak_pagefile_usage_bytes", memory)
        self.assertEqual(memory["api_fields_raw"]["PrivateUsage"], memory["private_bytes"])
        self.assertEqual(memory["api_fields_raw"]["PagefileUsage"], memory["pagefile_usage_bytes"])
        self.assertIn("lifetime peak Commit Charge", memory["semantics"]["PeakPagefileUsage"])
        self.assertIsNotNone(age)
        self.assertGreater(age, 0)


if __name__ == "__main__":
    unittest.main()

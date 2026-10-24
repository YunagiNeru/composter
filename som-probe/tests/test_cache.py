import json
import tempfile
import unittest
from pathlib import Path

from som_probe.runtime import _cache_inventory, _npu_cache_for_run
from som_probe.storage import Run


class CacheReuseTests(unittest.TestCase):
    def test_cache_reuse_copies_files_into_the_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Run.create(root, "G3")
            destination = Run.create(root, "G3_CACHE_REUSE")
            conversion = {"fp16_xml_sha256": "xml", "fp16_bin_sha256": "bin"}
            for run in (source, destination):
                run.write_json("conversion.json", conversion)
            source.stage("G2-cpu-check", "PASS")
            source.stage("G3", "PASS")
            source.write_json("npu-result.json", {
                "status": "PASS", "device_architecture": "3720",
                "compiled_model_properties": {
                    "EXECUTION_DEVICES": ["NPU"], "NPU_COMPILER_TYPE": "PLUGIN",
                },
            })
            original_cache = source.path / "npu-cache"
            (original_cache / "nested").mkdir(parents=True)
            (original_cache / "nested" / "compiled.bin").write_bytes(b"compiled data")
            original_hashes = _cache_inventory(original_cache)

            copied_cache, copied_hashes, mode = _npu_cache_for_run(
                root, destination, conversion, source.run_id
            )

            self.assertEqual(mode, "reuse")
            self.assertEqual(copied_hashes, original_hashes)
            self.assertEqual(_cache_inventory(copied_cache), original_hashes)
            self.assertNotEqual(copied_cache, original_cache)
            self.assertEqual(_cache_inventory(original_cache), original_hashes)
            self.assertEqual((original_cache / "nested" / "compiled.bin").read_bytes(), b"compiled data")


if __name__ == "__main__":
    unittest.main()

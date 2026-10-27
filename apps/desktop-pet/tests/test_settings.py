import json
import tempfile
import unittest
from pathlib import Path

from desktop_pet.settings import SettingsStore


class SettingsTests(unittest.TestCase):
    def test_save_load_is_versioned_and_clamps_size(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            settings, recovered = store.load()
            self.assertFalse(recovered)
            settings["window"]["max_area_percent"] = 99
            store.save(settings)
            loaded, recovered = store.load()
            self.assertFalse(recovered)
            self.assertEqual(loaded["schema_version"], 1)
            self.assertEqual(loaded["window"]["max_area_percent"], 10)

    def test_corrupt_settings_are_preserved_and_defaults_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("{invalid", encoding="utf-8")
            settings, recovered = SettingsStore(path).load()
            self.assertTrue(recovered)
            self.assertEqual(settings["schema_version"], 1)
            self.assertTrue(path.with_name("settings.json.corrupt").exists())


if __name__ == "__main__":
    unittest.main()

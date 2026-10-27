from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "window": {"x": None, "y": None, "monitor": None, "always_on_top": True,
               "max_area_percent": 3.0, "move_mode": False},
    "sensors": {"network_adapter": "auto", "disk_path": None, "network_sensitivity": 1.0},
    "sound": {"volume": 0.25, "muted": False},
    "ai_enabled": True,
    "animations_enabled": True,
    "pet": {"health": 100, "satisfaction": 55, "mood": "CALM", "bond": 20,
            "personality": "AFFECTIONATE"},
}


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        self.path = path or base / "Composter" / "DesktopPet" / "settings-v1.json"

    def load(self) -> tuple[dict[str, Any], bool]:
        if not self.path.exists():
            return deepcopy(DEFAULTS), False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema_version") != 1:
                raise ValueError("unsupported settings schema")
            return self._merge(data), False
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            backup = self.path.with_name(self.path.name + ".corrupt")
            try:
                os.replace(self.path, backup)
            except OSError:
                pass
            return deepcopy(DEFAULTS), True

    def save(self, settings: dict[str, Any]) -> None:
        data = self._merge(settings)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        payload = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        temporary.write_text(payload + "\n", encoding="utf-8")
        if self.path.exists():
            backup = self.path.with_suffix(".bak")
            try:
                backup.write_bytes(self.path.read_bytes())
            except OSError:
                pass
        os.replace(temporary, self.path)

    @staticmethod
    def _merge(value: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(DEFAULTS)
        for key in ("window", "sensors", "sound", "pet"):
            if isinstance(value.get(key), dict):
                result[key].update(value[key])
        for key in ("ai_enabled", "animations_enabled"):
            if isinstance(value.get(key), bool):
                result[key] = value[key]
        result["schema_version"] = 1
        result["window"]["max_area_percent"] = _number(
            result["window"].get("max_area_percent"), 3.0, 0.5, 10.0
        )
        result["sensors"]["network_sensitivity"] = _number(
            result["sensors"].get("network_sensitivity"), 1.0, 0.1, 8.0
        )
        result["sound"]["volume"] = _number(result["sound"].get("volume"), 0.25, 0.0, 1.0)
        result["sound"]["muted"] = bool(result["sound"].get("muted", False))
        result["pet"]["health"] = int(_number(result["pet"].get("health"), 100, 0, 100))
        result["pet"]["satisfaction"] = int(_number(result["pet"].get("satisfaction"), 55, 0, 100))
        result["pet"]["bond"] = int(_number(result["pet"].get("bond"), 20, 0, 100))
        if result["pet"].get("personality") not in {"AFFECTIONATE", "CAUTIOUS", "PLAYFUL"}:
            result["pet"]["personality"] = "AFFECTIONATE"
        if result["pet"].get("mood") not in {"CALM", "HAPPY", "SLEEPY", "PLAYFUL"}:
            result["pet"]["mood"] = "CALM"
        return result


def _number(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not 0 <= abs(number) < float("inf"):
        return default
    return min(high, max(low, number))

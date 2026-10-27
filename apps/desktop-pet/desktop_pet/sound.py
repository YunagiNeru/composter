from __future__ import annotations

import math
import os
import struct
import time
import wave
from pathlib import Path


class MeowPlayer:
    """Creates short local tones and plays them asynchronously through Windows."""

    def __init__(self) -> None:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        self.directory = base / "Composter" / "DesktopPet" / "generated-audio"
        self._last_played = 0.0

    def play(self, style: str, volume: float, muted: bool) -> bool:
        if muted or volume <= 0 or time.monotonic() - self._last_played < 15.0:
            return False
        if os.name != "nt":
            return False
        try:
            import winsound
            path = self._make_wave(style, volume)
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            self._last_played = time.monotonic()
            return True
        except (OSError, RuntimeError):
            return False

    def _make_wave(self, style: str, volume: float) -> Path:
        params = {
            "SLEEPY_MEOW": (430.0, 290.0, 0.52),
            "PLAYFUL_MEOW": (440.0, 780.0, 0.40),
            "SHORT_MEOW": (720.0, 510.0, 0.27),
        }
        start, end, duration = params.get(style, params["SHORT_MEOW"])
        gain = min(1.0, max(0.0, float(volume))) * 0.22
        rate = 22050
        samples = bytearray()
        for index in range(int(rate * duration)):
            t = index / rate
            phase = 2 * math.pi * (start * t + (end - start) * t * t / (2 * duration))
            envelope = min(1.0, t * 30.0) * max(0.0, min(1.0, (duration - t) * 9.0))
            sample = gain * envelope * (math.sin(phase) * 0.68 + math.sin(phase * 2.03) * 0.20
                                        + math.sin(phase * 3.01) * 0.12)
            samples.extend(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767)))
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{style.lower()}-{round(volume * 100):03d}.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(samples)
        return path

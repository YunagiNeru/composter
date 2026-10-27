from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import psutil
from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from .behavior import AppliedBehavior, PetMetrics, QUESTIONS, compose_behavior
from .settings import SettingsStore
from .settings_dialog import SettingsDialog
from .sensors import MetricSampler
from .sound import MeowPlayer
from .sprite_art import SpritePainter
from .theme import application_stylesheet, FONT_FAMILY, FONT_SIZE
from .window import PetWindow
from .worker_client import NpuWorkerClient


class DesktopPet:
    def __init__(self, args: argparse.Namespace) -> None:
        self.app_root = Path(__file__).resolve().parents[1]
        self.repository_root = self.app_root.parents[1]
        self.som_root = self.repository_root / "som-probe"
        self.ui_python = Path(sys.executable).resolve()
        self.runtime_python = self.som_root / ".local" / "venv-run" / "Scripts" / "python.exe"
        self.args = args
        self.application = QApplication(sys.argv[:1])
        self.application.setApplicationName("Composter Desktop Pet")
        self.application.setOrganizationName("Composter")
        self.application.setQuitOnLastWindowClosed(False)
        self.application.setStyleSheet(application_stylesheet())
        self.application.setFont(self._font())
        self.store = SettingsStore()
        self.settings, self.recovered_settings = self.store.load()
        self._forced_no_ai = args.no_ai
        if args.no_ai:
            self.settings["ai_enabled"] = False
        self.sampler = MetricSampler()
        self.meow_player = MeowPlayer()
        self.metrics: PetMetrics | None = None
        self.last_actions: dict[str, str] = {}
        self.last_outputs: dict[str, Any] = {}
        self.last_behavior: AppliedBehavior | None = None
        self.ai_status = "無効" if not self.settings["ai_enabled"] else "NPU読み込み待ち"
        self.ai_metadata: dict[str, Any] = {}
        self.worker: NpuWorkerClient | None = None
        self._last_state_hash = ""
        self._last_inference_at = 0.0
        self._ai_faulted = False
        self._current_event = "NONE"
        self._event_serial = 0
        self._expected_state_hash = ""
        self._demo_started = time.monotonic()

        self.window = PetWindow(self.settings)
        self.window.interaction.connect(self._on_interaction)
        self.window.quit_requested.connect(self.quit)
        self.window.settings_requested.connect(self._open_settings)
        self.window.pc_status_requested.connect(self.window.show_pc_status)
        self.window.ai_status_requested.connect(self.window.show_ai_status)
        self.window.moved.connect(self._on_moved)

        self.tray = self._create_tray()
        self.application.screenRemoved.connect(lambda _screen: self.window.rehome())
        self.application.screenAdded.connect(self._screen_added)
        for screen in self.application.screens():
            self._track_screen(screen)
        self.sensor_timer = QTimer(self.application)
        self.sensor_timer.setInterval(1000)
        self.sensor_timer.timeout.connect(self._sample_sensors)
        self.inference_timer = QTimer(self.application)
        self.inference_timer.setInterval(5000)
        self.inference_timer.timeout.connect(self._request_current_state)
        self.sensor_timer.start()
        self.inference_timer.start()

    @staticmethod
    def _font():
        from PySide6.QtGui import QFont
        return QFont(FONT_FAMILY, FONT_SIZE)

    def _create_tray(self) -> QSystemTrayIcon:
        image = SpritePainter.render("normal_idle", 0).scaled(
            32, 32, Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        tray = QSystemTrayIcon(QIcon(QPixmap.fromImage(image)), self.application)
        tray.setToolTip("Desktop Pet")
        menu = QMenu()
        menu.addAction("表示", self._restore_window)
        menu.addAction("AIの状態を見る", self.window.show_ai_status)
        menu.addSeparator()
        menu.addAction("終了", self.quit)
        tray.setContextMenu(menu)
        tray.activated.connect(self._tray_activated)
        if QSystemTrayIcon.isSystemTrayAvailable():
            tray.show()
        return tray

    def _screen_added(self, screen) -> None:
        self._track_screen(screen)
        self.window.rehome()

    def _track_screen(self, screen) -> None:
        screen.geometryChanged.connect(lambda _geometry: self.window.rehome())
        screen.logicalDotsPerInchChanged.connect(lambda _dpi: self.window.rehome())

    def run(self) -> int:
        self._sample_sensors()
        self.window.show()
        if self.recovered_settings and not self.args.demo_metrics:
            QTimer.singleShot(800, lambda: QMessageBox.warning(self.window, "設定", "設定を初期値へ戻しました"))
        if self.settings["ai_enabled"]:
            QTimer.singleShot(180, self._start_worker)
        return self.application.exec()

    def _start_worker(self) -> None:
        if self.worker is not None and self.worker.process.state() != QProcess.ProcessState.NotRunning:
            return
        self._ai_faulted = False
        self.worker = NpuWorkerClient(self.app_root, self.som_root, self.runtime_python, self.application)
        self.worker.status_changed.connect(self._worker_status)
        self.worker.ready.connect(self._worker_ready)
        self.worker.inference.connect(self._worker_inference)
        self.worker.failed.connect(self._worker_failed)
        self.worker.process_finished.connect(self._worker_exited)
        self.worker.start()
        self.window.set_ai_status("NPU読み込み中", pid=self.worker.pid)
        self.tray.setToolTip(f"Desktop Pet · Laya NPU · PID {self.worker.pid or '起動中'}")

    def _worker_status(self, _status: str, description: str) -> None:
        pid = self.worker.pid if self.worker else None
        self.ai_status = description
        self.window.set_ai_status(description, pid=pid)
        self.tray.setToolTip(f"Desktop Pet · {description} · PID {pid or '起動中'}")

    def _worker_ready(self, metadata: dict[str, Any]) -> None:
        self.ai_metadata = metadata
        devices = metadata.get("execution_devices", [])
        matches = (metadata.get("requested_device") == "NPU" and devices == ["NPU"]
                   and str(metadata.get("device_architecture")) == "3720"
                   and str(metadata.get("compiler_type", "")).upper().endswith("PLUGIN"))
        if not matches:
            self.ai_status = "AI利用不可（実行条件不一致）"
            self.window.set_ai_status(self.ai_status, pid=self.worker.pid if self.worker else None,
                                      metadata=metadata)
            if self.worker:
                self.worker.shutdown()
            self.last_actions = {}
            self._apply_behavior()
            return
        self.ai_status = "利用中（NPU／PLUGIN／3720）"
        self.window.set_ai_status(self.ai_status, pid=self.worker.pid if self.worker else None,
                                  metadata=metadata)
        self.tray.setToolTip(f"Desktop Pet · {self.ai_status} · PID {self.worker.pid if self.worker else '未取得'}")
        self._request_current_state(force=True)

    def _worker_inference(self, result: dict[str, Any]) -> None:
        current_state = self.metrics.as_model_state(self.settings["pet"], self._current_event) if self.metrics else ""
        current_hash = hashlib.sha256(current_state.encode("utf-8")).hexdigest()
        if result.get("state_sha256") != current_hash or current_hash != self._expected_state_hash:
            return
        actions = result.get("actions")
        outputs = result.get("model_outputs")
        if (not isinstance(actions, dict) or set(actions) != {"gait_style", "expression", "meow_style", "social_response"}
                or not isinstance(outputs, dict) or set(outputs) != set(actions)
                or any(not isinstance(value, str) or value not in QUESTIONS[key]["criteria"]
                       for key, value in actions.items())
                or any(not isinstance(outputs[key], dict)
                       or outputs[key].get("decision") != actions[key]
                       or outputs[key].get("candidate_ids") != list(QUESTIONS[key]["criteria"])
                       for key in actions)):
            self._worker_failed("WORKER_ACTION_CONTRACT_INVALID")
            return
        self.last_actions = dict(actions)
        self.last_outputs = outputs
        self.window.set_model_result(result, actions)
        self.ai_status = "利用中（NPU／PLUGIN／3720）"
        self.window.set_ai_status(self.ai_status, pid=self.worker.pid if self.worker else None,
                                  metadata=self.ai_metadata)
        self._last_inference_at = time.monotonic()
        self._apply_behavior()

    def _worker_failed(self, error: str) -> None:
        self._ai_faulted = True
        self.ai_status = f"AI利用不可（{error}）"
        self.last_actions = {}
        self.last_outputs = {}
        if self.worker is not None and self.worker.is_ready:
            self.worker.shutdown()
        self.window.set_ai_status(self.ai_status, pid=self.worker.pid if self.worker else None,
                                  metadata=self.ai_metadata)
        self.tray.setToolTip(f"Desktop Pet · {self.ai_status}")
        self._apply_behavior()

    def _worker_exited(self, exit_code: int) -> None:
        if self.worker is not None and not self.worker._stopping:
            self._worker_failed(f"NPUワーカー終了（{exit_code}）")

    def _sample_sensors(self) -> None:
        if self.args.demo_metrics:
            metrics = self._demo_sample()
        else:
            try:
                metrics = self.sampler.sample(self.settings)
            except Exception:
                metrics = PetMetrics(time.strftime("%Y-%m-%dT%H:%M:%S%z"), None, None, None,
                                     None, None, None, None, None, None,
                                     ("SENSOR_READ_FAILED",))
        self.metrics = metrics
        self._apply_behavior()
        self._try_meow()
        if self.worker is not None and self.worker.is_ready:
            state = metrics.as_model_state(self.settings["pet"], self._current_event)
            if hashlib.sha256(state.encode("utf-8")).hexdigest() != self._last_state_hash:
                self._request_current_state()

    def _demo_sample(self) -> PetMetrics:
        phases = (
            (90, 15, 40, "LOW", "MEDIUM", 65, None),
            (5, 85, 40, "LOW", "LOW", 65, None),
            (20, 15, 40, "HIGH", "HIGH", 65, None),
            (20, 85, 100, "HIGH", "HIGH", 92, 70),
        )
        phase = int((time.monotonic() - self._demo_started) // 6) % len(phases)
        cpu, ram, gpu, rx, tx, disk, busy = phases[phase]
        return PetMetrics(f"DEMO-{phase}", cpu, ram, gpu, rx, tx, 12_500_000,
                          25_000_000, disk, busy, ("DEMO_METRICS",))

    def _apply_behavior(self) -> None:
        if self.metrics is None:
            return
        ai_ready = self.worker is not None and self.worker.is_ready and bool(self.last_actions)
        behavior = compose_behavior(self.metrics, self.settings["pet"], self.settings,
                                    self.last_actions if ai_ready else None, rule_only=not ai_ready)
        self.last_behavior = behavior
        self.window.set_sensor_snapshot(self.metrics, behavior, self.settings)

    def _request_current_state(self, force: bool = False) -> None:
        if (not self.settings["ai_enabled"] or self._ai_faulted
                or self.metrics is None or self.worker is None):
            return
        if not self.worker.is_ready:
            return
        if not force and time.monotonic() - self._last_inference_at < 4.8:
            return
        state = self.metrics.as_model_state(self.settings["pet"], self._current_event)
        state_hash = hashlib.sha256(state.encode("utf-8")).hexdigest()
        self._last_state_hash = state_hash
        self._expected_state_hash = state_hash
        self.worker.infer(state)

    def _on_interaction(self, interaction: str) -> None:
        pet = self.settings["pet"]
        if interaction == "pet":
            cautious = pet.get("personality") == "CAUTIOUS" and int(pet.get("bond", 0)) < 45
            event = "PET"
            if cautious:
                self.window.interaction_animation("pet_dislike")
            else:
                self.window.interaction_animation("pet_happy")
                pet["bond"] = min(100, int(pet.get("bond", 0)) + 2)
                pet["satisfaction"] = min(100, int(pet.get("satisfaction", 0)) + 3)
                pet["mood"] = "HAPPY"
        elif interaction == "feed":
            event = "FEED"
            pet["satisfaction"] = min(100, int(pet.get("satisfaction", 0)) + 15)
            pet["health"] = min(100, int(pet.get("health", 0)) + 2)
            pet["mood"] = "HAPPY"
            self.window.interaction_animation("feed")
        else:
            event = "PLAY"
            pet["satisfaction"] = min(100, int(pet.get("satisfaction", 0)) + 8)
            pet["bond"] = min(100, int(pet.get("bond", 0)) + 3)
            pet["mood"] = "PLAYFUL"
            self.window.interaction_animation("play")
        self._current_event = event
        self._save_settings()
        self._apply_behavior()
        self._request_current_state(force=True)
        self._event_serial += 1
        serial = self._event_serial
        QTimer.singleShot(5000, lambda: self._clear_event(serial))

    def _clear_event(self, serial: int) -> None:
        if serial == self._event_serial:
            self._current_event = "NONE"
            self._request_current_state(force=True)

    def _try_meow(self) -> None:
        if (self.metrics is None or self.metrics.network_tx_level != "HIGH"
                or self.last_behavior is None or not self.settings.get("animations_enabled", True)):
            return
        sound = self.settings["sound"]
        self.meow_player.play(self.last_behavior.meow_style,
                              float(sound.get("volume", 0.25)), bool(sound.get("muted", False)))

    def _open_settings(self) -> None:
        monitor_names = [screen.name() for screen in self.application.screens()]
        disks = sorted({partition.mountpoint for partition in psutil.disk_partitions(all=False)})
        dialog = SettingsDialog(self.settings, monitor_names,
                                list(self.sampler.adapters), disks, self.window)
        if dialog.exec():
            self.settings = dialog.result_settings()
            self.window.apply_settings(self.settings)
            self._save_settings()
            if self.settings["ai_enabled"]:
                if self.worker is None or not self.worker.is_ready:
                    self._start_worker()
            else:
                if self.worker is not None:
                    self.worker.shutdown()
                self.ai_status = "無効"
                self.last_actions = {}
                self._apply_behavior()
            self._request_current_state(force=True)

    def _save_settings(self) -> None:
        if not self.args.demo_metrics and not self._forced_no_ai:
            self.store.save(self.settings)

    def _on_moved(self, x: int, y: int, monitor: str) -> None:
        self.settings["window"].update({"x": x, "y": y, "monitor": monitor or None})
        self._save_settings()

    def _restore_window(self) -> None:
        self.window.show()
        self.window.raise_()

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in {QSystemTrayIcon.ActivationReason.DoubleClick,
                      QSystemTrayIcon.ActivationReason.Trigger}:
            self._restore_window()

    def quit(self) -> None:
        self.sensor_timer.stop()
        self.inference_timer.stop()
        self.window.persist_position()
        self._save_settings()
        if self.worker is not None:
            self.worker.shutdown()
        self.sampler.close()
        self.tray.hide()
        self.application.quit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="desktop-pet")
    parser.add_argument("--no-ai", action="store_true", help="NPUワーカーを起動しない")
    parser.add_argument("--demo-metrics", action="store_true", help="架空のPC状態を注入する")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if os.name != "nt":
        raise SystemExit("このMVPはWindows専用です")
    args = parse_args(argv)
    return DesktopPet(args).run()


if __name__ == "__main__":
    raise SystemExit(main())

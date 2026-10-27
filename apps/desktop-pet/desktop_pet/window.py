from __future__ import annotations

import math
import time
from typing import Any

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QBitmap, QContextMenuEvent, QIcon, QImage, QMouseEvent, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QWidget

from .behavior import AppliedBehavior, PetMetrics
from .settings_dialog import SettingsDialog
from .sprite_art import ANIMATION_FRAMES, SpritePainter


class PetWindow(QWidget):
    interaction = Signal(str)
    quit_requested = Signal()
    settings_requested = Signal()
    pc_status_requested = Signal()
    ai_status_requested = Signal()
    moved = Signal(int, int, str)
    move_mode_changed = Signal(bool)
    ai_reload_requested = Signal()

    def __init__(self, settings: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.metrics: PetMetrics | None = None
        self.behavior: AppliedBehavior | None = None
        self.ai_status = "準備中"
        self.ai_metadata: dict[str, Any] = {}
        self.last_model_outputs: dict[str, Any] = {}
        self.last_input_sha256 = "未取得"
        self.worker_pid: int | None = None
        self.move_mode = bool(settings["window"].get("move_mode", False))
        self._drag_origin: QPoint | None = None
        self._window_origin: QPoint | None = None
        self._mouse_moved = False
        self._interaction_animation: str | None = None
        self._interaction_until = 0.0
        self._last_frame = -1
        self._last_animation = "normal_idle"
        self._last_ear_frame = -1
        self._last_tick_at = time.monotonic()
        self._walk_distance = 0.0
        self._wander_direction = -1
        self._pointer_inside = False
        self._menu_open = False
        self._smoothed_area_percent = float(settings["window"].get("max_area_percent", 3.0)) * 0.5
        self._target_area_percent = self._smoothed_area_percent
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(33)
        self._frame_timer.timeout.connect(self._tick)
        self._frame_timer.start()
        self.setWindowTitle("Composter Desktop Pet")
        self.setWindowFlags(self._window_flags())
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._restore_position()
        self._resize_for_area()
        self._refresh_mask()
        self.setCursor(Qt.CursorShape.OpenHandCursor if self.move_mode else Qt.CursorShape.ArrowCursor)

    def _window_flags(self):
        flags = Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
        if self.settings["window"].get("always_on_top", True):
            flags |= Qt.WindowType.WindowStaysOnTopHint
        return flags

    def set_sensor_snapshot(self, metrics: PetMetrics, behavior: AppliedBehavior,
                            settings: dict[str, Any]) -> None:
        self.metrics = metrics
        self.behavior = behavior
        self.settings = settings
        self._target_area_percent = behavior.display_area_percent
        self._tick()

    def set_ai_status(self, status: str, *, pid: int | None = None,
                      metadata: dict[str, Any] | None = None) -> None:
        self.ai_status = status
        self.worker_pid = pid
        if metadata is not None:
            self.ai_metadata = metadata
        self.update()

    def set_model_result(self, result: dict[str, Any], actions: dict[str, str]) -> None:
        self.last_model_outputs = result.get("model_outputs", {})
        self.last_input_sha256 = str(result.get("state_sha256", "未取得"))
        self.update()

    def set_move_mode(self, enabled: bool) -> None:
        self.move_mode = enabled
        self.settings["window"]["move_mode"] = enabled
        self.setCursor(Qt.CursorShape.OpenHandCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.move_mode_changed.emit(enabled)

    def apply_settings(self, settings: dict[str, Any]) -> None:
        old_monitor = self.settings["window"].get("monitor")
        self.settings = settings
        self.move_mode = bool(settings["window"].get("move_mode", False))
        self.setWindowFlags(self._window_flags())
        self._resize_for_area(rehome=old_monitor != settings["window"].get("monitor"))
        self.show()
        self._refresh_mask()

    def rehome(self) -> None:
        self._resize_for_area(rehome=True)

    def show_pc_status(self) -> None:
        if self.metrics is None:
            text = "センサーを読み込み中"
        else:
            m = self.metrics
            text = "\n".join((
                f"CPU　{_percent(m.cpu_percent)}",
                f"RAM　{_percent(m.ram_percent)}",
                f"GPU　{_percent(m.gpu_percent)}",
                f"受信　{_rate(m.network_rx_rate)}　{m.network_rx_level or '未取得'}",
                f"送信　{_rate(m.network_tx_rate)}　{m.network_tx_level or '未取得'}",
                f"ディスク容量　{_percent(m.disk_used_percent)}",
                f"ディスク稼働（全体）　{_percent(m.disk_busy_percent)}",
                f"取得時刻　{m.captured_at}",
            ))
        QMessageBox.information(self, "PCの状態", text)

    def show_ai_status(self) -> None:
        lines = [f"状態　{self.ai_status}", f"実行先　NPU", f"PID　{self.worker_pid or '未取得'}"]
        if self.ai_metadata:
            lines.extend((
                f"モデル　{self.ai_metadata.get('model_repository', '未取得')}@"
                f"{self.ai_metadata.get('model_revision', '未取得')}",
                f"精度　{self.ai_metadata.get('precision', '未取得')}",
                f"実デバイス　{', '.join(self.ai_metadata.get('execution_devices', [])) or '未取得'}",
                f"NPU　{self.ai_metadata.get('device_architecture', '未取得')}",
                f"コンパイラー　{self.ai_metadata.get('compiler_type', '未取得')}　"
                f"{self.ai_metadata.get('compiler_version', '未取得')}",
                f"IR SHA-256　{self.ai_metadata.get('ir_xml_sha256', '未取得')}",
                f"キャッシュ再利用　{_cache_readout(self.ai_metadata.get('loaded_from_cache'))}",
            ))
        if self.last_model_outputs:
            lines.append("")
            for question_id, output in self.last_model_outputs.items():
                decision = output.get("decision", "未取得")
                lines.append(f"{question_id}　{decision}")
                for row in output.get("candidate_probabilities", []):
                    lines.append(f"  {row.get('candidate_id')}　{float(row.get('probability', 0)):.1%}")
            lines.append(f"入力SHA-256　{self.last_input_sha256}")
        if self.behavior is not None:
            lines.extend(("", f"適用元　{self.behavior.source}",
                          f"適用した行動　{self.behavior.applied_actions}"))
            if self.behavior.warnings:
                lines.append(f"制約　{', '.join(self.behavior.warnings)}")
        QMessageBox.information(self, "AIの状態", "\n".join(lines))

    def _restore_position(self) -> None:
        x, y = self.settings["window"].get("x"), self.settings["window"].get("y")
        if isinstance(x, int) and isinstance(y, int):
            self.move(x, y)

    def _selected_screen(self):
        screens = QApplication.screens()
        requested = self.settings["window"].get("monitor")
        if requested:
            for screen in screens:
                if screen.name() == requested:
                    return screen
        return QApplication.primaryScreen() or (screens[0] if screens else None)

    def _resize_for_area(self, rehome: bool = False) -> None:
        screen = self._selected_screen()
        if screen is None:
            side = 128
            area = QRect(0, 0, 1920, 1080)
            display_area = area
        else:
            area = screen.availableGeometry()
            display_area = screen.geometry()
            percent = self._smoothed_area_percent
            side = display_side_for_area(display_area.width(), display_area.height(), percent)
        previous = self.size()
        if previous.isEmpty() or rehome:
            self.resize(side, side)
        elif abs(self.width() - side) >= 4:
            self.resize(side, side)
        if rehome or previous.isEmpty() or self._position_is_unset():
            x = area.x() + area.width() - side - 24
            y = area.y() + area.height() - side - 24
        else:
            x, y = self.x(), self.y()
        x = min(max(area.x(), x), area.x() + area.width() - side)
        y = min(max(area.y(), y), area.y() + area.height() - side)
        self.move(x, y)
        self._refresh_mask()

    def _position_is_unset(self) -> bool:
        return self.settings["window"].get("x") is None or self.settings["window"].get("y") is None

    def _frame(self, now: float) -> tuple[str, int]:
        if self._interaction_animation is not None:
            if now >= self._interaction_until:
                self._interaction_animation = None
            else:
                name = self._interaction_animation
                frames = ANIMATION_FRAMES[name]
                return name, int(now * 5) % frames
        if self.behavior is None:
            return "normal_idle", int(now / 5) % 2
        name = self.behavior.animation
        frames = ANIMATION_FRAMES[name]
        if not self.settings.get("animations_enabled", True):
            return name, 0
        movement = self.behavior.movement_rate
        if name in {"normal_idle", "sleepy_idle"}:
            return name, int(now / 4) % frames
        return name, int(now * max(0.15, movement) * frames) % frames

    def _tick(self) -> None:
        now = time.monotonic()
        elapsed = min(0.1, max(0.0, now - self._last_tick_at))
        self._last_tick_at = now
        if (self.behavior is not None and self.settings.get("animations_enabled", True)
                and not self._pointer_inside and not self._menu_open and not self.move_mode):
            self._walk_distance += self.behavior.movement_rate * 20.0 * elapsed
            travel = int(self._walk_distance)
            self._walk_distance -= travel
            if travel:
                screen = self._selected_screen()
                if screen is not None:
                    bounds = screen.availableGeometry()
                    left, right = bounds.left(), bounds.right() - self.width() + 1
                    next_x = self.x() + self._wander_direction * travel
                    if next_x < left or next_x > right:
                        self._wander_direction *= -1
                        next_x = min(right, max(left, next_x))
                    self.move(next_x, self.y())
        if self.behavior is not None and not self._pointer_inside and not self._menu_open:
            self._smoothed_area_percent += (self._target_area_percent - self._smoothed_area_percent) * 0.035
            self._resize_for_area()
        name, frame = self._frame(now)
        ear_rate = self.behavior.ear_rate if self.behavior is not None else 0.2
        ear_frame = int(now * ear_rate)
        ear_changed = ear_frame != self._last_ear_frame
        if self._last_frame != frame or self._last_animation != name or ear_changed:
            self._last_frame, self._last_animation = frame, name
            self._last_ear_frame = ear_frame
            self._refresh_mask(name, frame, ear_frame)
            self.update()

    def _refresh_mask(self, animation: str | None = None, frame: int | None = None,
                      ear_frame: int | None = None) -> None:
        name = animation or getattr(self, "_last_animation", "normal_idle")
        index = frame if frame is not None else getattr(self, "_last_frame", 0)
        fullness = self.behavior.body_fullness if self.behavior is not None else 0.5
        expression = self.behavior.expression if self.behavior is not None else None
        social = self.behavior.social_response if self.behavior is not None else None
        if ear_frame is None:
            ear_rate = self.behavior.ear_rate if self.behavior else 0.2
            ear_frame = int(time.monotonic() * ear_rate)
        image = SpritePainter.render(name, index, fullness, expression=expression,
                                     social_response=social, ear_frame=ear_frame)
        scaled = image.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                              Qt.TransformationMode.FastTransformation)
        self.setMask(QBitmap.fromImage(scaled.createAlphaMask()))

    def paintEvent(self, _event) -> None:
        if self.behavior is None:
            name, frame = self._frame(time.monotonic())
            fullness = 0.5
            expression = None
            social = None
            ear_rate = 0.2
        else:
            name, frame = self._frame(time.monotonic())
            fullness = self.behavior.body_fullness
            expression = self.behavior.expression
            social = self.behavior.social_response
            ear_rate = self.behavior.ear_rate
        ear_frame = int(time.monotonic() * ear_rate)
        source = SpritePainter.render(name, frame, fullness, expression=expression,
                                      social_response=social, ear_frame=ear_frame)
        scaled = source.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                               Qt.TransformationMode.FastTransformation)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(self.rect(), scaled)
        painter.end()

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        if self.move_mode:
            event.accept()
            return
        menu = QMenu(self)
        move_action = menu.addAction("移動モード")
        move_action.setCheckable(True)
        move_action.setChecked(self.move_mode)
        move_action.triggered.connect(lambda checked: self.set_move_mode(checked))
        touch_menu = menu.addMenu("ふれあい")
        touch_menu.addAction("撫でる", lambda: self.interaction.emit("pet"))
        touch_menu.addAction("餌をあげる", lambda: self.interaction.emit("feed"))
        touch_menu.addAction("遊ぶ", lambda: self.interaction.emit("play"))
        menu.addAction("PCの状態を見る", self.pc_status_requested.emit)
        menu.addAction("AIの状態を見る", self.ai_status_requested.emit)
        pet_menu = menu.addMenu("ペットを選択")
        current_pet = pet_menu.addAction("猫")
        current_pet.setEnabled(False)
        menu.addAction("設定", self.settings_requested.emit)
        menu.addSeparator()
        menu.addAction("非表示", self.hide)
        menu.addAction("終了", self.quit_requested.emit)
        self._menu_open = True
        menu.exec(event.globalPos())
        self._menu_open = False

    def enterEvent(self, event) -> None:
        self._pointer_inside = True
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._pointer_inside = False
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.move_mode and event.button() in {Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton}:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._drag_origin = event.globalPosition().toPoint()
            self._window_origin = self.pos()
            self._mouse_moved = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._drag_origin = event.globalPosition().toPoint()
            self._window_origin = self.pos()
            self._mouse_moved = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is None or self._window_origin is None:
            super().mouseMoveEvent(event)
            return
        delta = event.globalPosition().toPoint() - self._drag_origin
        if delta.manhattanLength() > 5:
            self._mouse_moved = True
        if self.move_mode:
            self.move(self._window_origin + delta)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        was_moving = self.move_mode
        moved = self._mouse_moved
        self._drag_origin = self._window_origin = None
        if was_moving:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if moved:
                self._save_position()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and not moved:
            self.interaction.emit("pet")
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self.move_mode:
            self.set_move_mode(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def _save_position(self) -> None:
        screen = self._selected_screen()
        self.settings["window"]["x"] = self.x()
        self.settings["window"]["y"] = self.y()
        self.settings["window"]["monitor"] = screen.name() if screen else None
        self.moved.emit(self.x(), self.y(), screen.name() if screen else "")

    def persist_position(self) -> None:
        self._save_position()

    def interaction_animation(self, interaction: str) -> None:
        animation_by_event = {"pet_happy": "pet_happy", "pet_dislike": "pet_dislike",
                              "feed": "eat", "play": "play"}
        if interaction in animation_by_event:
            self._interaction_animation = animation_by_event[interaction]
            self._interaction_until = time.monotonic() + 2.4
            self._tick()

    def _toggle_move_mode(self) -> None:
        self.set_move_mode(not self.move_mode)


def _percent(value: float | None) -> str:
    return "未取得" if value is None else f"{value:.1f}%"


def _rate(value: float | None) -> str:
    if value is None:
        return "未取得"
    return f"{value / (1024 * 1024):.2f} MiB/s"


def _cache_readout(value: Any) -> str:
    return "未取得" if value is None else "再利用" if value else "新規コンパイル"


def display_side_for_area(width: int, height: int, area_percent: float) -> int:
    """Return the square sprite size that respects the selected display-area limit."""
    if width <= 0 or height <= 0 or not math.isfinite(area_percent) or area_percent <= 0:
        raise ValueError("表示領域と占有率は正の有限値である必要があります")
    return max(1, int(math.sqrt(width * height * area_percent / 100.0)))

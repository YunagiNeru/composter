from __future__ import annotations

from copy import deepcopy
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    def __init__(self, settings: dict[str, Any], monitor_names: list[str],
                 adapter_names: list[str], disk_paths: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("設定")
        self.setModal(True)
        self.setMinimumWidth(300)
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._settings = deepcopy(settings)
        form = QFormLayout()
        self.area = QDoubleSpinBox()
        self.area.setRange(0.5, 10.0)
        self.area.setSingleStep(0.5)
        self.area.setSuffix(" %")
        self.area.setValue(float(settings["window"]["max_area_percent"]))
        form.addRow("最大表示サイズ", self.area)

        self.monitor = QComboBox()
        self.monitor.addItem("自動", None)
        for name in monitor_names:
            self.monitor.addItem(name, name)
        current_monitor = settings["window"].get("monitor")
        index = self.monitor.findData(current_monitor)
        self.monitor.setCurrentIndex(max(0, index))
        form.addRow("表示モニター", self.monitor)

        self.network = QComboBox()
        self.network.addItem("自動（物理NIC）", "auto")
        for name in adapter_names:
            self.network.addItem(name, name)
        index = self.network.findData(settings["sensors"].get("network_adapter", "auto"))
        self.network.setCurrentIndex(max(0, index))
        form.addRow("ネットワーク", self.network)

        self.disk = QComboBox()
        self.disk.addItem("システムドライブ", None)
        for path in disk_paths:
            self.disk.addItem(path, path)
        index = self.disk.findData(settings["sensors"].get("disk_path"))
        self.disk.setCurrentIndex(max(0, index))
        form.addRow("容量対象ディスク", self.disk)

        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0.1, 8.0)
        self.sensitivity.setSingleStep(0.1)
        self.sensitivity.setValue(float(settings["sensors"]["network_sensitivity"]))
        form.addRow("通信感度", self.sensitivity)

        self.volume = QDoubleSpinBox()
        self.volume.setRange(0, 1)
        self.volume.setSingleStep(0.05)
        self.volume.setValue(float(settings["sound"]["volume"]))
        form.addRow("音量", self.volume)

        self.muted = QCheckBox("ミュート")
        self.muted.setChecked(bool(settings["sound"]["muted"]))
        form.addRow("音", self.muted)
        self.topmost = QCheckBox("最前面に表示")
        self.topmost.setChecked(bool(settings["window"]["always_on_top"]))
        form.addRow("ウィンドウ", self.topmost)
        self.ai = QCheckBox("SOMを有効にする")
        self.ai.setChecked(bool(settings["ai_enabled"]))
        form.addRow("SOM", self.ai)
        self.animations = QCheckBox("アニメーション")
        self.animations.setChecked(bool(settings["animations_enabled"]))
        form.addRow("表示", self.animations)

        self.personality = QComboBox()
        for label, value in (("甘えん坊", "AFFECTIONATE"), ("慎重", "CAUTIOUS"),
                             ("遊び好き", "PLAYFUL")):
            self.personality.addItem(label, value)
        index = self.personality.findData(settings["pet"].get("personality", "AFFECTIONATE"))
        self.personality.setCurrentIndex(max(0, index))
        form.addRow("性格", self.personality)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def result_settings(self) -> dict[str, Any]:
        result = self._settings
        result["window"]["max_area_percent"] = self.area.value()
        result["window"]["monitor"] = self.monitor.currentData()
        result["window"]["always_on_top"] = self.topmost.isChecked()
        result["sensors"]["network_adapter"] = self.network.currentData()
        result["sensors"]["disk_path"] = self.disk.currentData()
        result["sensors"]["network_sensitivity"] = self.sensitivity.value()
        result["sound"]["volume"] = self.volume.value()
        result["sound"]["muted"] = self.muted.isChecked()
        result["ai_enabled"] = self.ai.isChecked()
        result["animations_enabled"] = self.animations.isChecked()
        result["pet"]["personality"] = self.personality.currentData()
        return result

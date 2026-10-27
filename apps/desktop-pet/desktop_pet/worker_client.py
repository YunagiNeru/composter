from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal


PROTOCOL_VERSION = 1


class NpuWorkerClient(QObject):
    status_changed = Signal(str, str)
    ready = Signal(dict)
    inference = Signal(dict)
    failed = Signal(str)
    process_finished = Signal(int)

    def __init__(self, app_root: Path, som_root: Path, runtime_python: Path, parent=None) -> None:
        super().__init__(parent)
        self.app_root = app_root
        self.som_root = som_root
        self.runtime_python = runtime_python
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self.process.readAllStandardError)
        self.process.errorOccurred.connect(self._process_error)
        self.process.finished.connect(self._process_finished)
        self._buffer = bytearray()
        self._ready = False
        self._worker_pid: int | None = None
        self._request_id = 0
        self._in_flight = False
        self._pending_state: str | None = None
        self._stopping = False

    @property
    def pid(self) -> int | None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return None
        return self._worker_pid

    @property
    def is_ready(self) -> bool:
        return self._ready and self.process.state() == QProcess.ProcessState.Running

    def start(self) -> None:
        if not self.runtime_python.is_file():
            self.failed.emit("NPU用Pythonが見つかりません")
            return
        environment = QProcessEnvironment.systemEnvironment()
        path = str(self.som_root) + os.pathsep + str(self.app_root)
        existing = environment.value("PYTHONPATH")
        if existing:
            path += os.pathsep + existing
        environment.insert("PYTHONPATH", path)
        environment.insert("HF_HUB_OFFLINE", "1")
        environment.insert("TRANSFORMERS_OFFLINE", "1")
        environment.insert("HF_HUB_DISABLE_TELEMETRY", "1")
        environment.insert("TOKENIZERS_PARALLELISM", "false")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(self.som_root))
        script = self.app_root / "desktop_pet" / "npu_worker.py"
        self._stopping = False
        self.status_changed.emit("loading", "Laya／NPU／FP16を読み込み中")
        self.process.start(str(self.runtime_python), [str(script), "--root", str(self.som_root)])

    def infer(self, state: str) -> None:
        if not self.is_ready:
            return
        if self._in_flight:
            self._pending_state = state
            return
        self._request_id += 1
        self._in_flight = True
        payload = {"protocol_version": PROTOCOL_VERSION, "type": "infer",
                   "request_id": self._request_id, "state": state}
        self.process.write((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                           .encode("utf-8"))

    def shutdown(self, wait_ms: int = 1500) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self._stopping = True
        if self.is_ready:
            payload = {"protocol_version": PROTOCOL_VERSION, "type": "shutdown"}
            self.process.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))
        self.process.closeWriteChannel()
        if not self.process.waitForFinished(wait_ms):
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()
                self.process.waitForFinished(1500)
        self._ready = False

    def _read_stdout(self) -> None:
        self._buffer.extend(bytes(self.process.readAllStandardOutput()))
        while b"\n" in self._buffer:
            line, _, remainder = self._buffer.partition(b"\n")
            self._buffer = bytearray(remainder)
            try:
                message = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.failed.emit("NPUワーカー応答を読み取れません")
                continue
            if not isinstance(message, dict) or message.get("protocol_version") != PROTOCOL_VERSION:
                self.failed.emit("NPUワーカーの通信契約が一致しません")
                continue
            kind = message.get("type")
            if kind == "status":
                worker_pid = message.get("pid")
                self._worker_pid = (worker_pid if isinstance(worker_pid, int)
                                    and worker_pid > 0 else None)
                self.status_changed.emit(str(message.get("status", "loading")),
                                         f"Laya／{message.get('device', 'NPU')}／{message.get('precision', 'FP16')}を読み込み中")
            elif kind == "ready":
                metadata = message.get("metadata", {})
                worker_pid = metadata.get("worker_pid") if isinstance(metadata, dict) else None
                if isinstance(worker_pid, int) and worker_pid > 0:
                    self._worker_pid = worker_pid
                self._ready = True
                self.ready.emit(metadata)
            elif kind == "inference":
                self._in_flight = False
                self.inference.emit(message)
                self._send_pending()
            elif kind in {"failure", "error"}:
                code = str(message.get("code", "NPU_WORKER_FAILED"))
                self.failed.emit(code)
                if message.get("request_id") == self._request_id:
                    self._in_flight = False
                    self._send_pending()
            elif kind == "shutdown-complete":
                self._ready = False

    def _send_pending(self) -> None:
        state, self._pending_state = self._pending_state, None
        if state is not None and not self._stopping:
            self.infer(state)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if not self._stopping:
            self._ready = False
            self.failed.emit(f"NPUワーカー起動・実行失敗（{error.name}）")

    def _process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._ready = False
        self._worker_pid = None
        self._in_flight = False
        if not self._stopping:
            self.failed.emit(f"NPUワーカー終了（終了コード {exit_code}）")
        self.process_finished.emit(exit_code)

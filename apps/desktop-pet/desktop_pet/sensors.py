from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Any

import psutil

from .behavior import PetMetrics


class _PdhValueUnion(ctypes.Union):
    _fields_ = [("long_value", ctypes.c_long), ("double_value", ctypes.c_double),
                ("large_value", ctypes.c_longlong), ("pointer_value", ctypes.c_void_p)]


class _PdhFormattedValue(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("status", ctypes.c_ulong), ("value", _PdhValueUnion)]


class DiskBusyCounter:
    """Reads the Windows PhysicalDisk % Disk Time counter; failures remain unavailable."""

    def __init__(self) -> None:
        self._query = None
        self._counter = None
        self._pdh = None
        if os.name != "nt":
            return
        try:
            self._pdh = ctypes.WinDLL("pdh.dll")
            self._pdh.PdhOpenQueryW.argtypes = (ctypes.c_wchar_p, ctypes.c_size_t,
                                                ctypes.POINTER(ctypes.c_void_p))
            self._pdh.PdhAddEnglishCounterW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p,
                                                        ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p))
            self._pdh.PdhCollectQueryData.argtypes = (ctypes.c_void_p,)
            self._pdh.PdhGetFormattedCounterValue.argtypes = (
                ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                ctypes.POINTER(_PdhFormattedValue),
            )
            self._pdh.PdhCloseQuery.argtypes = (ctypes.c_void_p,)
            self._pdh.PdhCloseQuery.restype = ctypes.c_ulong
            query, counter = ctypes.c_void_p(), ctypes.c_void_p()
            if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                return
            if self._pdh.PdhAddEnglishCounterW(
                    query, r"\PhysicalDisk(_Total)\% Disk Time", 0, ctypes.byref(counter)) != 0:
                self._pdh.PdhCloseQuery(query)
                return
            self._query, self._counter = query, counter
            self._pdh.PdhCollectQueryData(self._query)
        except (AttributeError, OSError, TypeError):
            self.close()

    def sample(self) -> float | None:
        if self._query is None or self._counter is None:
            return None
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return None
        value = _PdhFormattedValue()
        value_type = ctypes.c_ulong()
        result = self._pdh.PdhGetFormattedCounterValue(
            self._counter, 0x00000200, ctypes.byref(value_type), ctypes.byref(value)
        )
        if result != 0 or value.status != 0 or not 0 <= value.double_value < float("inf"):
            return None
        return min(100.0, max(0.0, value.double_value))

    def close(self) -> None:
        if self._pdh is not None and self._query is not None:
            try:
                self._pdh.PdhCloseQuery(self._query)
            except (AttributeError, OSError):
                pass
        self._query = self._counter = None


def physical_network_adapters() -> list[str]:
    if os.name != "nt":
        return []
    script = "(Get-NetAdapter -Physical | Where-Object Status -eq 'Up' | " \
             "Select-Object -ExpandProperty Name)"
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=4, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            return []
        return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    except (OSError, subprocess.TimeoutExpired):
        return []


class MetricSampler:
    def __init__(self) -> None:
        psutil.cpu_percent(interval=None)
        self.adapters: list[str] = []
        self.adapters_ready = False
        self._adapter_thread = threading.Thread(target=self._load_adapters,
                                                name="pet-network-discovery", daemon=True)
        self._adapter_thread.start()
        self._last_net = psutil.net_io_counters(pernic=True)
        self._last_at = time.monotonic()
        self._nvml: Any = None
        self._gpu_handle: Any = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            if pynvml.nvmlDeviceGetCount() > 0:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
            self._gpu_handle = None
        self._disk = DiskBusyCounter()

    def sample(self, settings: dict[str, Any]) -> PetMetrics:
        now = time.monotonic()
        interval = max(0.001, now - self._last_at)
        unavailable: list[str] = []
        cpu = _safe(psutil.cpu_percent(interval=None))
        ram = _safe(psutil.virtual_memory().percent)
        gpu = self._gpu_percent()
        if cpu is None:
            unavailable.append("CPU")
        if ram is None:
            unavailable.append("RAM")
        if gpu is None:
            unavailable.append("GPU")

        disk_path = settings.get("sensors", {}).get("disk_path") or os.environ.get("SystemDrive", "C:") + "\\"
        try:
            disk_used = _safe(psutil.disk_usage(disk_path).percent)
        except (OSError, ValueError, PermissionError):
            disk_used = None
        disk_busy = self._disk.sample()
        if disk_used is None:
            unavailable.append("DISK_CAPACITY")
        if disk_busy is None:
            unavailable.append("DISK_ACTIVITY")

        try:
            counters = psutil.net_io_counters(pernic=True)
        except (OSError, psutil.Error):
            counters = {}
        selected = settings.get("sensors", {}).get("network_adapter", "auto")
        names = self.adapters if selected == "auto" else [selected] if selected in self.adapters else []
        rx_rate, tx_rate = self._network_rates(counters, names, interval)
        if rx_rate is None or tx_rate is None:
            unavailable.append("NETWORK")
        sensitivity = _safe(settings.get("sensors", {}).get("network_sensitivity", 1.0)) or 1.0
        rx_level = _traffic_level(rx_rate, sensitivity) if rx_rate is not None else None
        tx_level = _traffic_level(tx_rate, sensitivity) if tx_rate is not None else None
        self._last_net, self._last_at = counters, now
        return PetMetrics(
            captured_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            cpu_percent=cpu, ram_percent=ram, gpu_percent=gpu,
            network_rx_level=rx_level, network_tx_level=tx_level,
            network_rx_rate=rx_rate, network_tx_rate=tx_rate,
            disk_used_percent=disk_used, disk_busy_percent=disk_busy,
            unavailable=tuple(unavailable),
        )

    def close(self) -> None:
        self._disk.close()
        self._adapter_thread.join(timeout=4.5)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _load_adapters(self) -> None:
        self.adapters = physical_network_adapters()
        self.adapters_ready = True

    def _gpu_percent(self) -> float | None:
        if self._nvml is None or self._gpu_handle is None:
            return None
        try:
            reading = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            return _safe(reading.gpu)
        except Exception:
            return None

    def _network_rates(self, counters: dict[str, Any], names: list[str],
                       interval: float) -> tuple[float | None, float | None]:
        available_names = [name for name in names if name in counters and name in self._last_net]
        if not available_names:
            return None, None
        try:
            rx = sum(max(0, counters[name].bytes_recv - self._last_net[name].bytes_recv)
                     for name in available_names)
            tx = sum(max(0, counters[name].bytes_sent - self._last_net[name].bytes_sent)
                     for name in available_names)
            return rx / interval, tx / interval
        except (KeyError, TypeError, AttributeError):
            return None, None


def _traffic_level(bytes_per_second: float, sensitivity: float) -> str:
    megabits = bytes_per_second * 8 / 1_000_000 * sensitivity
    if megabits < 20:
        return "LOW"
    if megabits < 200:
        return "MEDIUM"
    return "HIGH"


def _safe(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number < float("inf") else None

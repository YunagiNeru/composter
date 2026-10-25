"""秘匿情報を抑えたrun記録。"""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import Profile, profile_from_mapping, profile_mapping
from .dataset import DATASET_FILES, dataset_identity


_PATH = re.compile(r"(?i)(?:[a-z]:\\|\\\\)[^\s\"']+")
_SECRET = re.compile(r"(?i)(?:hf_[a-z0-9]{20,}|sk-[a-z0-9]{20,}|gh[pousr]_[a-z0-9]{20,})")
_PROVENANCE_FILES = (
    "configs/probe.json", "configs/gates.json", "configs/models.lock.json",
    "datasets/manifest.json", "datasets/dev.jsonl",
    "requirements/run-win-x64.lock", "requirements/eval-win-x64.lock",
)
_SUCCESS_TERMINAL_STAGES = frozenset({
    "G3", "NPU_DIAGNOSTIC", "PERFORMANCE", "MEMORY", "SOKUDAN_CPU",
})


def safe_message(message: str) -> str:
    value = _PATH.sub("[PATH]", message)
    value = _SECRET.sub("[REDACTED]", value)
    value = "".join(char for char in value if char >= " " or char in "\t")
    return value[:512]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    excluded = {".git", ".local", "__pycache__", ".pytest_cache"}
    for current, directories, files in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in excluded)
        for name in sorted(files):
            path = Path(current) / name
            if path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _provenance_files(root: Path, dataset_version: str = "dev-v1") -> dict[str, str]:
    result: dict[str, str] = {}
    for name in _PROVENANCE_FILES:
        path = root / name
        if not path.is_file():
            result[name] = "MISSING"
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[name] = digest
    selected = DATASET_FILES[dataset_version]
    for name in (selected["manifest"], selected["data"], selected["contract"], selected["revisions"]):
        if name is not None and name not in result:
            path = root / name
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
    return result


class Run:
    def __init__(self, root: Path, run_id: str):
        if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", run_id):
            raise ValueError("run_idの形式が不正です。")
        self.root = root.resolve()
        self.path = (self.root / ".local" / "runs" / run_id).resolve()
        if self.root not in self.path.parents:
            raise ValueError("run pathが作業領域外です。")
        self.run_id = run_id

    @classmethod
    def create(cls, root: Path, purpose: str, profile: dict[str, Any] | None = None,
               compare_input_run_id: str | None = None,
               dataset_version: str = "dev-v1", *,
               profile_applicable: bool = True) -> "Run":
        if profile_applicable:
            profile = profile_mapping(Profile()) if profile is None else profile_mapping(profile_from_mapping(profile))
            profile_name = profile["name"]
        else:
            if profile is not None or compare_input_run_id is not None:
                raise ValueError("固定Shapeを使わないRunへprofileやShape比較元は設定できません。")
            profile = None
            profile_name = "not_applicable"
        identity = dataset_identity(root, dataset_version)
        if compare_input_run_id is not None and not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", compare_input_run_id):
            raise ValueError("入力比較元Run IDの形式が不正です。")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        while True:
            run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
            run = cls(root, run_id)
            try:
                run.path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = "unavailable"
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True,
                text=True, timeout=5
            ).stdout
            dirty = bool(status.strip())
        except (OSError, subprocess.SubprocessError):
            status = None
            dirty = None
        _write_json(run.path / "manifest.json", {
            "run_id": run_id,
            "purpose": purpose,
            "status": "RUNNING",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": revision,
            "source_tree_sha256": _source_tree_digest(root),
            "worktree_dirty": dirty,
            "worktree_status": "clean" if dirty is False else ("dirty" if dirty else "unavailable"),
            "worktree_status_porcelain": status,
            "input_sha256": _provenance_files(root, dataset_version),
            "dataset": identity,
            "verified_model_artifact_sha256": {},
            "converted_ir_sha256": {},
            "profile": profile_name,
            "profile_config": profile,
            "compare_input_run_id": compare_input_run_id,
            "stages": {},
        })
        return run

    def read_manifest(self) -> dict[str, Any]:
        path = self.path / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError("run manifestがありません。")
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def profile_config(self) -> dict[str, Any]:
        manifest = self.read_manifest()
        value = manifest.get("profile_config")
        if manifest.get("profile") == "not_applicable" and value is None:
            raise ValueError("このRunには固定Shapeプロファイルがありません。")
        if value is None and manifest.get("profile") == "s256-k8":
            value = profile_mapping(Profile())
        profile = profile_mapping(profile_from_mapping(value))
        if manifest.get("profile") != profile["name"]:
            raise ValueError("Run manifestのプロファイル名と設定が一致しません。")
        return profile

    @property
    def dataset_version(self) -> str:
        return self.read_manifest().get("dataset", {}).get("dataset_version", "dev-v1")

    @property
    def dataset_identity(self) -> dict[str, Any]:
        manifest = self.read_manifest()
        selected = manifest.get("dataset")
        if selected is None:
            return dataset_identity(self.root, "dev-v1")
        return selected

    def stage(self, name: str, status: str, **fields: Any) -> None:
        manifest = self.read_manifest()
        if name == "G0" and isinstance(fields.get("verified_artifact_sha256"), dict):
            manifest["verified_model_artifact_sha256"] = fields["verified_artifact_sha256"]
        conversion_path = self.path / "conversion.json"
        if conversion_path.is_file():
            conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
            manifest["converted_ir_sha256"] = {
                key: conversion[key]
                for key in ("source_model_sha256", "fp32_xml_sha256", "fp32_bin_sha256",
                            "fp16_xml_sha256", "fp16_bin_sha256")
                if key in conversion
            }
        manifest["stages"][name] = {
            "status": status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        if status == "FAIL":
            manifest["status"] = "FAILED"
        elif status == "BLOCKED":
            manifest["status"] = "BLOCKED"
        elif status == "PASS" and name in _SUCCESS_TERMINAL_STAGES:
            manifest["status"] = "COMPLETE"
        elif manifest.get("status") in {"COMPLETE", "FAILED", "BLOCKED"}:
            pass
        else:
            manifest["status"] = "RUNNING"
        _write_json(self.path / "manifest.json", manifest)
        with (self.path / "stages.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": name, "status": status, **fields}, ensure_ascii=False, allow_nan=False) + "\n")

    def blocked(self, stage: str, code: str, message: str, **fields: Any) -> None:
        record = {
            "stage": stage,
            "status": "BLOCKED",
            "code": code,
            "error_type": "BlockedStage",
            "message": safe_message(message),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with (self.path / "errors.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.stage(stage, "BLOCKED", code=code, error_type="BlockedStage", message=record["message"], **fields)

    def failure(self, stage: str, code: str, error: Exception, **fields: Any) -> None:
        record = {
            "stage": stage,
            "status": "FAIL",
            "code": code,
            "error_type": type(error).__name__,
            "message": safe_message(str(error)),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with (self.path / "errors.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.stage(stage, "FAIL", code=code, error_type=type(error).__name__)

    def write_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        if Path(name).name != name:
            raise ValueError("出力名はrun直下のファイル名にしてください。")
        target = self.path / name
        if target.exists():
            raise FileExistsError(f"{name} は既に存在します。既存結果を上書きしません。")
        with target.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def write_json(self, name: str, value: dict[str, Any]) -> None:
        if Path(name).name != name:
            raise ValueError("出力名はrun直下のファイル名にしてください。")
        target = self.path / name
        with target.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def open_run(root: Path, run_id: str) -> Run:
    run = Run(root, run_id)
    if not run.path.is_dir():
        raise FileNotFoundError("指定runがありません。run-initで新規作成してください。")
    run.read_manifest()
    return run

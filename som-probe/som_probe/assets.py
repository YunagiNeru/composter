"""固定版の取得物を検証し、ローカル専用領域へ準備する。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


class AssetError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if ("\\" in value or ":" in value or path.is_absolute() or not path.parts
            or any(part in {".", ".."} for part in path.parts)):
        raise AssetError("INVALID_LOCK_PATH", "固定情報に作業領域外のパスがあります。")
    return Path(*path.parts)


def load_lock(root: Path) -> dict[str, Any]:
    try:
        lock = json.loads((root / "configs" / "models.lock.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetError("MODEL_LOCK_UNREADABLE", "モデル固定情報を読み込めません。") from exc
    if lock.get("schema_version") != 1 or not isinstance(lock.get("artifacts"), dict):
        raise AssetError("MODEL_LOCK_INVALID", "モデル固定情報の形式が不正です。")
    return lock


def asset_path(root: Path, name: str) -> Path:
    if name == "laya-source-archive.tar.gz":
        source = load_lock(root)["sources"]["laya_code"]
        filename = f"laya-{source['commit']}.tar.gz"
        return root / ".local" / "sources" / filename
    category, relative = name.split("/", 1)
    if category not in {"reference", "onnx"}:
        raise AssetError("INVALID_LOCK_PATH", "固定情報に未対応の取得先があります。")
    return root / ".local" / "models" / category / _relative_path(relative)


def verify_assets(root: Path) -> dict[str, str]:
    lock = load_lock(root)
    archive_hash = lock["sources"]["laya_code"].get("archive_sha256")
    if archive_hash != lock["artifacts"].get("laya-source-archive.tar.gz"):
        raise AssetError("MODEL_LOCK_INCONSISTENT", "Layaソースアーカイブのハッシュが固定情報内で一致しません。")
    download_targets: set[str] = set()
    for spec in _download_specs(lock):
        if not isinstance(spec, dict) or set(spec) != {"url", "target", "sha256", "size"}:
            raise AssetError("DOWNLOAD_LOCK_INVALID", "固定取得項目の形式が不正です。")
        target = _relative_path(spec["target"]).as_posix()
        if (set(spec) != {"url", "target", "sha256", "size"} or not isinstance(spec["size"], int)
                or spec["size"] <= 0 or target in download_targets
                or lock["artifacts"].get(target) != spec["sha256"]):
            raise AssetError("MODEL_LOCK_INCONSISTENT", "取得一覧と成果物ハッシュの固定情報が一致しません。")
        download_targets.add(target)
    if download_targets != set(lock["artifacts"]) - {"laya-source-archive.tar.gz"}:
        raise AssetError("MODEL_LOCK_INCONSISTENT", "固定成果物と取得一覧の構成が一致しません。")
    verified: dict[str, str] = {}
    for name, expected in lock["artifacts"].items():
        path = asset_path(root, name)
        if not path.is_file():
            raise AssetError("ASSET_MISSING", f"固定取得物がありません: {name}")
        expected_size = (lock["sources"]["laya_code"].get("archive_size") if name == "laya-source-archive.tar.gz"
                         else next(spec["size"] for spec in lock["downloads"] if spec["target"] == name))
        if path.stat().st_size != expected_size:
            raise AssetError("ASSET_SIZE_MISMATCH", f"固定取得物のサイズが一致しません: {name}")
        actual = sha256_file(path)
        if actual.lower() != str(expected).lower():
            raise AssetError("ASSET_HASH_MISMATCH", f"固定取得物のハッシュが一致しません: {name}")
        verified[name] = actual
    return verified


def _download(url: str, destination: Path, expected_sha256: str, expected_size: int) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"huggingface.co", "codeload.github.com"} or parsed.username or parsed.password:
        raise AssetError("DOWNLOAD_SOURCE_NOT_ALLOWED", "固定取得URLのHTTPSホストが許可リストにありません。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise AssetError("ASSET_EXISTS", f"取得先が既にあります。上書きしません: {destination.name}")
    temporary = destination.with_name(destination.name + ".part-" + uuid.uuid4().hex)
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as output:
            start = 0
            chunk_size = 32 * 1024 * 1024
            while start < expected_size:
                end = min(expected_size - 1, start + chunk_size - 1)
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "som-probe/0.1", "Range": f"bytes={start}-{end}"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise AssetError("RANGED_DOWNLOAD_UNSUPPORTED", f"固定取得物の範囲取得に失敗しました: {destination.name}")
                    content_range = response.headers.get("Content-Range", "")
                    if content_range != f"bytes {start}-{end}/{expected_size}":
                        raise AssetError("RANGED_DOWNLOAD_MISMATCH", f"固定取得物の範囲応答が一致しません: {destination.name}")
                    remaining = end - start + 1
                    while remaining:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            raise AssetError("RANGED_DOWNLOAD_SHORT", f"固定取得物の応答が途中で終了しました: {destination.name}")
                        digest.update(block)
                        output.write(block)
                        remaining -= len(block)
                start = end + 1
        if temporary.stat().st_size != expected_size:
            raise AssetError("ASSET_SIZE_MISMATCH", f"取得物のサイズが一致しません: {destination.name}")
        if digest.hexdigest().lower() != expected_sha256.lower():
            raise AssetError("ASSET_HASH_MISMATCH", f"取得物のハッシュが一致しません: {destination.name}")
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _source_archive_spec(lock: dict[str, Any]) -> tuple[str, str, int]:
    source = lock["sources"]["laya_code"]
    archive = source.get("archive")
    digest = source.get("archive_sha256")
    size = source.get("archive_size")
    if not isinstance(archive, str) or not isinstance(digest, str) or not isinstance(size, int):
        raise AssetError("SOURCE_LOCK_INVALID", "Layaソースの固定URLまたはハッシュがありません。")
    return archive, digest, size


def _download_specs(lock: dict[str, Any]) -> list[dict[str, str]]:
    specs = lock.get("downloads")
    if not isinstance(specs, list) or not specs:
        raise AssetError("DOWNLOAD_LOCK_INVALID", "固定取得一覧がありません。")
    return specs


def _extract_laya_source(root: Path, archive: Path, commit: str) -> Path:
    destination = root / ".local" / "sources" / f"laya-pinned-{commit}"
    package = destination / "laya"
    if package.is_dir():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    prefix = f"laya-{commit}/"
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                name = PurePosixPath(member.name)
                if ("\\" in member.name or ":" in member.name or name.is_absolute()
                        or any(part in {".", ".."} for part in name.parts)):
                    raise AssetError("SOURCE_ARCHIVE_PATH_INVALID", "Layaアーカイブ内に不正なパスがあります。")
                relative = member.name.removeprefix(prefix)
                if not relative or relative == member.name:
                    continue
                rel = PurePosixPath(relative)
                keep = relative == "LICENSE" or (rel.parts[0] == "laya" and member.isfile() and rel.suffix == ".py")
                if not keep:
                    continue
                target = destination.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise AssetError("SOURCE_ARCHIVE_INVALID", "Layaソースを読み出せません。")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except (OSError, tarfile.TarError) as exc:
        raise AssetError("SOURCE_ARCHIVE_INVALID", "固定Layaソースアーカイブを展開できません。") from exc
    if not package.is_dir():
        raise AssetError("SOURCE_PACKAGE_MISSING", "固定LayaアーカイブにPythonパッケージがありません。")
    return destination


def prepare_assets(root: Path, allow_download: bool) -> dict[str, Any]:
    lock = load_lock(root)
    archive_url, archive_hash, archive_size = _source_archive_spec(lock)
    downloads = _download_specs(lock)
    missing: list[tuple[Path, str, str, int]] = []
    archive = asset_path(root, "laya-source-archive.tar.gz")
    if not archive.is_file():
        missing.append((archive, archive_url, archive_hash, archive_size))
    for spec in downloads:
        required = {"url", "target", "sha256", "size"}
        if not isinstance(spec, dict) or set(spec) != required or not isinstance(spec["size"], int) or spec["size"] <= 0:
            raise AssetError("DOWNLOAD_LOCK_INVALID", "固定取得項目の形式が不正です。")
        target = _relative_path(spec["target"])
        if target.parts[0] not in {"reference", "onnx"}:
            raise AssetError("DOWNLOAD_LOCK_INVALID", "固定取得先が未対応です。")
        path = root / ".local" / "models" / target
        if not path.is_file():
            missing.append((path, spec["url"], spec["sha256"], spec["size"]))
    if missing and not allow_download:
        names = [path.name for path, _, _, _ in missing]
        raise AssetError("DOWNLOAD_NOT_AUTHORIZED", "取得物が不足しています。prepare --allow-download を指定してください。")
    for path, url, digest, size in missing:
        _download(url, path, digest, size)
    verified = verify_assets(root)
    source = lock["sources"]["laya_code"]
    source_dir = _extract_laya_source(root, archive, source["commit"])
    return {"status": "PASS", "verified_artifacts": len(verified), "laya_source_commit": source["commit"],
            "source_directory": source_dir.relative_to(root).as_posix()}


def make_execution_checkpoint(root: Path, run_id: str, *, contrast: bool = False) -> Path:
    """作業用コピーを作る。元の固定ファイルには書き込まない。"""
    source = root / ".local" / "models" / "reference"
    target_name = "reference-model-contrast" if contrast else "reference-model"
    target = root / ".local" / "runs" / run_id / target_name
    target.mkdir(parents=True, exist_ok=False)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.name == "tokenizer_config.json":
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, destination)
            except OSError:
                shutil.copy2(path, destination)
    return target

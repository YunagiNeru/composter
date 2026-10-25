"""Prepare the pinned Sokudan CPU comparison artifacts; never run from evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}".encode("ascii") + bytes([0]))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_root(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"ロック内の出力先が作業領域外です: {relative}")
    return path


def _verify_file(root: Path, relative: str, expected: dict[str, Any]) -> Path:
    path = _inside_root(root, relative)
    if not path.is_file():
        raise FileNotFoundError(f"固定取得物がありません: {relative}")
    if path.stat().st_size != expected["size"]:
        raise ValueError(f"固定取得物のサイズが不一致です: {relative}")
    if _sha256(path) != expected["sha256"]:
        raise ValueError(f"固定取得物のSHA-256が不一致です: {relative}")
    if "git_blob_sha1" in expected and _git_blob_sha1(path) != expected["git_blob_sha1"]:
        raise ValueError(f"固定取得物のGit blob SHA-1が不一致です: {relative}")
    return path


def _download(root: Path, relative: str, url: str, expected: dict[str, Any]) -> Path:
    path = _inside_root(root, relative)
    if path.exists():
        return _verify_file(root, relative, expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"未完了の取得ファイルが残っています: {partial.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "som-probe-pinned-preparation"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if partial.stat().st_size != expected["size"]:
            raise ValueError(f"取得物のサイズが不一致です: {relative}")
        if _sha256(partial) != expected["sha256"]:
            raise ValueError(f"取得物のSHA-256が不一致です: {relative}")
        if "git_blob_sha1" in expected and _git_blob_sha1(partial) != expected["git_blob_sha1"]:
            raise ValueError(f"取得物のGit blob SHA-1が不一致です: {relative}")
        os.replace(partial, path)
        return path
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def _safe_archive_path(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def _source_python_files(archive_path: Path, kind: str) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    if kind == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if any(not _safe_archive_path(name) for name in names):
                raise ValueError("固定ソースZIPに不正なパスがあります")
            root_prefix = names[0].split("/", 1)[0] + "/"
            for name in names:
                relative = name[len(root_prefix):] if name.startswith(root_prefix) else name
                if relative.startswith("sokudan/") and relative.endswith(".py"):
                    files[relative] = archive.read(name)
    else:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if any(not _safe_archive_path(member.name) for member in members):
                raise ValueError("固定sdistに不正なパスがあります")
            for member in members:
                prefix, separator, relative = member.name.partition("/")
                if (separator and member.isfile() and relative.startswith("sokudan/")
                        and relative.endswith(".py")):
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("固定sdistのPythonファイルを読み取れません")
                    files[relative] = stream.read()
    return files


def _audit_distributions(root: Path, lock: dict[str, Any]) -> dict[str, int]:
    package = lock["package"]
    source = _verify_file(root, package["source_archive"]["path"], package["source_archive"])
    wheel = _verify_file(root, package["wheel"]["path"], {
        "sha256": package["wheel"]["sha256"],
        "size": package["wheel"]["size"],
    })
    sdist = _verify_file(root, package["source_distribution"]["path"], {
        "sha256": package["source_distribution"]["sha256"],
        "size": package["source_distribution"]["size"],
    })
    source_files = _source_python_files(source, "zip")
    sdist_files = _source_python_files(sdist, "tar")
    if not source_files or source_files != sdist_files:
        raise ValueError("固定GitソースとPyPI sdistのPythonファイルが一致しません")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if any(not _safe_archive_path(name) for name in names):
            raise ValueError("PyPI wheelに不正なパスがあります")
        wheel_files = {name: archive.read(name) for name in names
                       if name.startswith("sokudan/") and name.endswith(".py")}
    if wheel_files != source_files:
        raise ValueError("固定GitソースとPyPI wheelのPythonファイルが一致しません")
    expected_count = package["source_archive"]["python_package_files_verified"]
    if len(source_files) != expected_count:
        raise ValueError("固定ソースのPythonファイル件数がロックと一致しません")
    return {"source_archive": len(source_files), "sdist": len(sdist_files), "wheel": len(wheel_files)}


def _artifact_url(repository: str, revision: str, filename: str) -> str:
    quoted = urllib.parse.quote(filename, safe="/")
    return f"https://huggingface.co/{repository}/resolve/{revision}/{quoted}"


def _prepare_runtime_model(root: Path, lock: dict[str, Any]) -> dict[str, str]:
    model = lock["model"]
    backbone = lock["backbone"]
    source_dir = _inside_root(root, model["directory"])
    backbone_dir = _inside_root(root, backbone["directory"])
    runtime_dir = _inside_root(
        root, f".local/models/sokudan-ja-310m-runtime-{model['revision']}"
    )
    original_config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    if original_config.get("backbone") != backbone["repository"]:
        raise ValueError("配布モデルのバックボーンIDがロックと一致しません")
    effective_config = dict(original_config)
    effective_config["backbone"] = str(backbone_dir.resolve())
    config_bytes = (json.dumps(effective_config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    if not runtime_dir.exists():
        runtime_dir.mkdir(parents=True, exist_ok=False)
        try:
            os.link(source_dir / "model.safetensors", runtime_dir / "model.safetensors")
            shutil.copyfile(source_dir / "calibration.json", runtime_dir / "calibration.json")
            (runtime_dir / "config.json").write_bytes(config_bytes)
        except Exception:
            # Only remove the new, incomplete overlay created by this operation.
            shutil.rmtree(runtime_dir)
            raise
    if not (runtime_dir / "model.safetensors").is_file():
        raise FileNotFoundError("ローダー用モデル重みがありません")
    if not os.path.samefile(source_dir / "model.safetensors", runtime_dir / "model.safetensors"):
        raise ValueError("ローダー用重みが固定スナップショットの同一ファイルではありません")
    if (runtime_dir / "config.json").read_bytes() != config_bytes:
        raise ValueError("ローダー用設定が配布設定からバックボーンID以外も変更されています")
    if _sha256(runtime_dir / "calibration.json") != model["files"]["calibration.json"]["sha256"]:
        raise ValueError("ローダー用校正ファイルのハッシュが一致しません")
    return {
        "directory": runtime_dir.relative_to(root).as_posix(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "weights_same_file_as_snapshot": "true",
        "calibration_sha256": _sha256(runtime_dir / "calibration.json"),
    }


def verify_artifacts(root: Path, *, prepare: bool) -> dict[str, Any]:
    lock_path = root / "configs" / "sokudan-cpu.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checked = 0
    package = lock["package"]
    downloadable = (
        (package["source_archive"]["path"], package["source_archive"]["url"], package["source_archive"]),
        (package["wheel"]["path"], package["wheel"]["url"],
         package["wheel"]),
        (package["source_distribution"]["path"], package["source_distribution"]["url"],
         package["source_distribution"]),
    )
    for relative, url, expected in downloadable:
        if prepare:
            _download(root, relative, url, expected)
        else:
            _verify_file(root, relative, expected)
        checked += 1
    for section in ("model", "backbone"):
        spec = lock[section]
        for name, expected in spec["files"].items():
            relative = f"{spec['directory']}/{name}"
            url = _artifact_url(spec["repository"], spec["revision"], name)
            if prepare:
                _download(root, relative, url, expected)
            else:
                _verify_file(root, relative, expected)
            checked += 1
    source_audit = _audit_distributions(root, lock)
    runtime = _prepare_runtime_model(root, lock) if prepare else None
    if runtime is None:
        runtime_dir = _inside_root(
            root, f".local/models/sokudan-ja-310m-runtime-{lock['model']['revision']}"
        )
        if not runtime_dir.is_dir():
            raise FileNotFoundError("ローダー用ローカル設定がありません。明示的な準備が必要です")
        expected = _prepare_runtime_model(root, lock)
        runtime = expected
    return {"status": "PASS", "verified_artifacts": checked, "source_audit": source_audit,
            "runtime_model": runtime}


def main() -> int:
    parser = argparse.ArgumentParser(description="固定Sokudan CPU比較取得物の準備・検査")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--allow-download", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        result = verify_artifacts(root, prepare=args.allow_download)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__,
                          "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

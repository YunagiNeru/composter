"""Acquire only the revision-pinned decision checkpoints and reviewed GLiNER2 wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_records(root: Path, required: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for name, expected in required.items():
        path = root / Path(*name.split("/"))
        if not path.is_file():
            raise FileNotFoundError(f"required snapshot file missing: {name}")
        digest = _sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"snapshot SHA-256 mismatch: {name}")
        result[name] = {"size": path.stat().st_size, "sha256": digest}
    return result


def _verify_revision(api: Any, repo_id: str, revision: str) -> Any:
    info = api.model_info(repo_id, revision=revision, token=False)
    if info.sha != revision:
        raise ValueError(f"Hub revision mismatch: {repo_id}")
    return info


def _download_model(root: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    repo_id = spec.get("resolved_repository", spec.get("repository"))
    revision = spec["revision"]
    destination = root / ".local" / "models" / f"{name}-{revision}"
    expected = spec["files"]
    if destination.exists():
        if any(value is None for value in expected.values()):
            raise FileExistsError(f"snapshot exists while its hash lock is incomplete: {name}")
        return {"status": "VERIFIED_EXISTING", "revision": revision,
                "files": _file_records(destination, expected)}

    api = HfApi(token=False)
    requested = spec.get("requested_repository", repo_id)
    requested_info = api.model_info(requested, token=False)
    if requested_info.id.lower() != repo_id.lower():
        raise ValueError(f"requested repository no longer resolves to pinned repository: {name}")
    info = _verify_revision(api, repo_id, revision)
    models_root = root / ".local" / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    staging = models_root / f".staging-{name}-{uuid.uuid4().hex}"
    cache = root / ".local" / "cache" / "typed-model-acquisition"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=staging,
            allow_patterns=list(expected),
            token=False,
        )
        files = _file_records(staging, expected)
        if info.sha != revision:
            raise ValueError(f"Hub revision changed during acquisition: {name}")
        os.replace(staging, destination)
        return {"status": "ACQUIRED", "requested_repository": requested,
                "resolved_repository": info.id, "revision": info.sha, "files": files}
    except BaseException:
        # Keep the staging directory as evidence for a failed/partial transfer.
        raise


def _download_wheel(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    destination = root / ".local" / "sources" / "gliner2-2.1.0-py3-none-any.whl"
    if destination.exists():
        actual = _sha256(destination)
        if destination.stat().st_size != spec["wheel_size"] or actual != spec["wheel_sha256"]:
            raise ValueError("existing GLiNER2 wheel does not match the reviewed source lock")
        return {"status": "VERIFIED_EXISTING", "size": destination.stat().st_size,
                "sha256": actual}

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = spec["wheel_url"]
    if not url.startswith("https://files.pythonhosted.org/"):
        raise ValueError("GLiNER2 wheel source is outside the approved PyPI file host")
    fd, temporary_name = tempfile.mkstemp(prefix="gliner2-", suffix=".part", dir=destination.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "som-probe/0.1"})
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise OSError("PyPI wheel download returned a non-200 response")
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
                digest.update(block)
                size += len(block)
        if size != spec["wheel_size"] or digest.hexdigest() != spec["wheel_sha256"]:
            raise ValueError("GLiNER2 wheel size or SHA-256 mismatch")
        os.link(temporary, destination)
        temporary.unlink()
        return {"status": "ACQUIRED", "size": size, "sha256": digest.hexdigest()}
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("network acquisition requires --allow-download")

    root = Path(__file__).resolve().parents[1]
    cache = root / ".local" / "cache" / "typed-model-acquisition"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    lock_path = root / "configs" / "typed-models.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    failures = 0
    for name, key in (("laya-typed", "candidate_laya_typed"),
                      ("gliner25-multi-decide", "candidate_gliner25_multi_decide")):
        try:
            results[key] = _download_model(root, name, lock[key])
        except Exception as exc:
            failures += 1
            results[key] = {"status": "FAILED", "error_type": type(exc).__name__,
                            "message": str(exc)[:512]}
    try:
        results["gliner2_source"] = _download_wheel(root, lock["gliner2_source"])
    except Exception as exc:
        failures += 1
        results["gliner2_source"] = {"status": "FAILED", "error_type": type(exc).__name__,
                                      "message": str(exc)[:512]}
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

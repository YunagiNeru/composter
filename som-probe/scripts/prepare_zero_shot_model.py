"""Explicitly acquire the allow-listed files of the pinned candidate-A snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPOSITORY = "Horizon-Labs/multilingual-zeroshot-small"
REVISION = "3fe1f0c48b589cf6220fd21a628a1dd56d3d41f8"
FILES = (
    "README.md",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / ".local" / "models" / f"multilingual-zeroshot-small-{REVISION}"

    if not args.allow_download:
        parser.error("モデル取得には --allow-download が必要です。")
    if destination.exists():
        raise SystemExit("取得先が既にあります。内容を上書きせず、別途確認してください。")

    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise SystemExit("公式取得元が返したcommit IDが指定revisionと一致しません。")
    snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=list(FILES),
    )
    file_records = {}
    for name in FILES:
        path = destination / name
        if not path.is_file():
            raise SystemExit(f"固定取得物がありません: {name}")
        file_records[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    print(json.dumps({
        "repository": REPOSITORY,
        "revision": info.sha,
        "files": file_records,
        "remote_code_executed": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

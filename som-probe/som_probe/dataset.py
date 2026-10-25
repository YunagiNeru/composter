"""Versioned evaluation datasets and their immutable run identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import ContractError, Profile, validate_dataset_case


DATASET_FILES = {
    "dev-v1": {
        "manifest": "datasets/manifest.json",
        "data": "datasets/dev.jsonl",
        "contract": None,
        "revisions": None,
        "contract_version": "evaluation-v1-legacy",
    },
    "dev-v2": {
        "manifest": "datasets/manifest-v2.json",
        "data": "datasets/dev-v2.jsonl",
        "contract": "configs/evaluation-contract-v2.json",
        "revisions": "datasets/dev-v2-revisions.json",
        "contract_version": "evaluation-v2",
    },
    "dev-v3": {
        "manifest": "datasets/manifest-v3.json",
        "data": "datasets/dev-v3.jsonl",
        "contract": "configs/evaluation-contract-v2.json",
        "revisions": "datasets/dev-v3-revisions.json",
        "contract_version": "evaluation-v2",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object key is duplicated: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _strict_json(raw: str) -> Any:
    return json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_nonfinite)


def dataset_identity(root: Path, version: str) -> dict[str, Any]:
    if version not in DATASET_FILES:
        raise ContractError("DATASET_VERSION_UNKNOWN", "未対応の評価データ版です。")
    files = DATASET_FILES[version]
    identity: dict[str, Any] = {
        "dataset_version": version,
        "dataset_manifest_path": files["manifest"],
        "dataset_manifest_sha256": _sha256(root / files["manifest"]),
        "dataset_path": files["data"],
        "dataset_sha256": _sha256(root / files["data"]),
        "scoring_contract_version": files["contract_version"],
        "scoring_contract_path": files["contract"],
        "scoring_contract_sha256": _sha256(root / files["contract"]) if files["contract"] else None,
        "revision_ledger_path": files["revisions"],
        "revision_ledger_sha256": _sha256(root / files["revisions"]) if files["revisions"] else None,
    }
    return identity


def load_dataset(
    root: Path,
    version: str,
    profile: Profile = Profile(),
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = dataset_identity(root, version)
    if any(identity[key] == "MISSING" for key in (
        "dataset_manifest_sha256", "dataset_sha256"
    )):
        raise ContractError("DATASET_FILES_MISSING", "評価データまたはmanifestがありません。")
    if expected_identity is not None and any(
        expected_identity.get(key) != value for key, value in identity.items()
    ):
        raise ContractError("DATASET_IDENTITY_MISMATCH", "Run作成後にデータ版・入力・採点契約の構成が変わりました。")

    manifest_path = root / identity["dataset_manifest_path"]
    data_path = root / identity["dataset_path"]
    try:
        manifest = _strict_json(manifest_path.read_text(encoding="utf-8"))
        data_spec = manifest["splits"]["dev"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ContractError("DATASET_MANIFEST_INVALID", "評価データmanifestを読み取れません。") from exc
    if (data_spec.get("path") != identity["dataset_path"]
            or data_spec.get("sha256") != identity["dataset_sha256"]):
        raise ContractError("DATASET_HASH_MISMATCH", "評価データmanifestと原本のハッシュが一致しません。")
    if version != "dev-v1" and (
        manifest.get("dataset_version") != version
        or manifest.get("scoring_contract", {}).get("version") != identity["scoring_contract_version"]
        or manifest.get("scoring_contract", {}).get("path") != identity["scoring_contract_path"]
        or manifest.get("scoring_contract", {}).get("sha256") != identity["scoring_contract_sha256"]
    ):
        raise ContractError("SCORING_CONTRACT_IDENTITY_MISMATCH", "データ版と採点契約の版・ハッシュが一致しません。")

    try:
        raw_lines = data_path.read_bytes().splitlines()
        if any(not line.strip() for line in raw_lines):
            raise ContractError("DATASET_BLANK_LINE", "JSONLに空行があります。")
        cases = [_strict_json(line.decode("utf-8")) for line in raw_lines]
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ContractError("DATASET_JSON_INVALID", "JSONL全行の厳格なJSON解析に失敗しました。") from exc
    if len(cases) != data_spec.get("cases") or not cases:
        raise ContractError("DATASET_COUNT_MISMATCH", "評価データ件数とmanifestが一致しません。")
    case_ids = [case.get("case_id") if isinstance(case, dict) else None for case in cases]
    if any(not isinstance(case_id, str) for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ContractError("DATASET_CASE_ID_INVALID", "評価データのcase_idが欠落または重複しています。")
    for case in cases:
        validate_dataset_case(case, profile, version)
    type_counts: dict[str, int] = {}
    for case in cases:
        question = next(iter(case["questions"].values()))
        kind = "bool" if question["type"] == "bool" else question["type"]
        type_counts[kind] = type_counts.get(kind, 0) + 1
    if type_counts != data_spec.get("question_types"):
        raise ContractError("DATASET_TYPE_COUNT_MISMATCH", "評価データの型別件数とmanifestが一致しません。")
    label_type_counts = {"choice": type_counts.get("choice", 0),
                         "score": type_counts.get("score", 0),
                         "noul": type_counts.get("bool", 0)}
    if (data_spec.get("label_types") is not None
            and label_type_counts != data_spec.get("label_types")):
        raise ContractError("DATASET_LABEL_TYPE_COUNT_MISMATCH", "許容ラベル型別件数とmanifestが一致しません。")
    labels = {label["verification_status"] for case in cases for label in case["expected"].values()}
    if labels != {"ai_draft_unverified"}:
        raise ContractError("LABEL_STATE_MISMATCH", "評価ラベルの人手確認状態が未確認ではありません。")

    contract = None
    if identity["scoring_contract_path"]:
        contract_path = root / identity["scoring_contract_path"]
        try:
            contract = _strict_json(contract_path.read_text(encoding="utf-8"))
            ledger = _strict_json((root / identity["revision_ledger_path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ContractError("SCORING_CONTRACT_INVALID", "採点契約または改訂台帳を読み取れません。") from exc
        revision_spec = manifest.get("revision_ledger", {})
        compatible_contract_datasets = {version, manifest.get("based_on_dataset_version")}
        if (contract.get("contract_version") != identity["scoring_contract_version"]
                or contract.get("dataset_version") not in compatible_contract_datasets
                or revision_spec.get("path") != identity["revision_ledger_path"]
                or revision_spec.get("sha256") != identity["revision_ledger_sha256"]
                or revision_spec.get("case_entries") != len(ledger.get("cases", {}))
                or identity["revision_ledger_sha256"] == "MISSING"
                or ledger.get("revision_id") != version
                or ledger.get("human_review_status") != "pending_for_all_60_cases"
                or set(ledger.get("cases", {})) != set(case_ids)):
            raise ContractError("REVISION_LEDGER_INVALID", "改訂台帳がデータ版・全case_idと一致しません。")
        if any(
            item.get("verification_status") != "ai_draft_unverified"
            or item.get("human_review") != "未実施"
            or not item.get("reason")
            or not isinstance(item.get("changed_fields"), list)
            for item in ledger["cases"].values()
        ):
            raise ContractError("REVISION_LEDGER_INVALID", "改訂台帳に人手確認状態または変更理由の欠落があります。")
        if (contract.get("label_verification_status") != "ai_draft_unverified"
                or manifest.get("label_verification_status") != "ai_draft_unverified"
                or manifest.get("human_review_status") != "pending_for_all_60_cases"
                or manifest.get("official_japanese_quality_status") != "UNDETERMINED"):
            raise ContractError("LABEL_STATE_MISMATCH", "品質確認状態が未確認・未判定のまま保持されていません。")
    return {"identity": identity, "manifest": manifest, "cases": cases, "contract": contract}

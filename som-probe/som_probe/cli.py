"""som-probeの小さなコマンドライン入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .assets import AssetError, prepare_assets, verify_assets
from .contracts import ContractError, Profile, validate_dataset_case, validate_request
from .runtime import convert_model, doctor, run_cpu_check, run_npu, run_npu_control, run_reference
from .storage import Run, open_run


EXIT_OK = 0
EXIT_INPUT = 2
EXIT_ENVIRONMENT = 3
EXIT_MODEL = 4
EXIT_QUALITY = 5
EXIT_RESOURCE = 6
EXIT_TIMEOUT = 7
EXIT_INTEGRITY = 8


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("INVALID_FILE", f"JSONを読み込めません: {path.name}") from exc


def validate_project(root: Path) -> dict[str, Any]:
    required = [
        "configs/probe.json", "configs/models.lock.json", "configs/gates.json",
        "datasets/manifest.json", "datasets/dev.jsonl",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ContractError("REQUIRED_FILES_MISSING", "必須ファイルがありません: " + ", ".join(missing))
    config = _json(root / "configs/probe.json")
    lock = _json(root / "configs/models.lock.json")
    gates = _json(root / "configs/gates.json")
    dataset_manifest = _json(root / "datasets/manifest.json")
    if config.get("profile") != {"name": "s256-k8", "batch": 1, "sequence_length": 256, "candidate_slots": 8}:
        raise ContractError("PROFILE_MISMATCH", "s256-k8の形状設定が計画値と一致しません。")
    if lock.get("schema_version") != 1 or not lock.get("sources") or not lock.get("downloads"):
        raise ContractError("MODEL_LOCK_INVALID", "モデル固定情報が不正です。")
    if gates.get("schema_version") != 1 or not {"G0", "G1", "G2", "G3", "G4", "G5"}.issubset(gates.get("stages", {})):
        raise ContractError("GATES_INVALID", "段階別ゲートが不足しています。")
    rows = []
    with (root / "datasets/dev.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
                validate_dataset_case(case)
            except (json.JSONDecodeError, ContractError) as exc:
                raise ContractError("DATASET_INVALID", f"dev.jsonl の {line_number} 行目が不正です。") from exc
            rows.append(case)
    dev = dataset_manifest.get("splits", {}).get("dev", {})
    if len(rows) != 60 or dev.get("cases") != 60:
        raise ContractError("DATASET_COUNT_MISMATCH", "G1用devデータは60件でなければなりません。")
    import hashlib
    dataset_hash = hashlib.sha256((root / "datasets/dev.jsonl").read_bytes()).hexdigest()
    if dataset_hash != dev.get("sha256"):
        raise ContractError("DATASET_HASH_MISMATCH", "G1用devデータのSHA-256が固定値と一致しません。")
    labels = {item["verification_status"] for case in rows for item in case["expected"].values()}
    if labels != {"ai_draft_unverified"}:
        raise ContractError("LABEL_STATE_MISMATCH", "未確認ラベルの状態が一致しません。")
    type_counts: dict[str, int] = {}
    for case in rows:
        for question in case["questions"].values():
            kind = question["type"]
            type_counts[kind] = type_counts.get(kind, 0) + 1
    if type_counts != dev.get("question_types"):
        raise ContractError("DATASET_TYPE_COUNT_MISMATCH", "devデータの型別件数が固定値と一致しません。")
    verified = verify_assets(root)
    return {"files_checked": len(required), "dev_cases": len(rows), "verified_artifacts": len(verified),
            "verified_artifact_sha256": verified,
            "label_state": "ai_draft_unverified", "official_quality_status": "BLOCKED_UNTIL_HUMAN_LABEL_REVIEW"}


def _request_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("INVALID_INPUT_FILE", "入力JSONを読み込めません。") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="som-probe")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("run-init", help="新しいローカルrunを作成")
    init.add_argument("--purpose", choices=("G0", "G1", "G2", "G3"), default="G0")
    validate = commands.add_parser("validate", help="設定、固定情報、60件の入力契約を検査")
    validate.add_argument("--run-id")
    request = commands.add_parser("validate-request", help="1件の入力JSONを検査")
    request.add_argument("--input", required=True)
    doctor = commands.add_parser("doctor", help="OpenVINOデバイスと実行時プロパティを記録")
    doctor.add_argument("--device", choices=("CPU", "NPU"), required=True)
    doctor.add_argument("--run-id")
    prepare = commands.add_parser("prepare", help="固定URLから取得物を取得し、ハッシュを検証")
    prepare.add_argument("--allow-download", action="store_true")
    prepare.add_argument("--run-id")
    reference = commands.add_parser("reference", help="元モデルのCPU参照実行と入力固定")
    reference.add_argument("--run-id", required=True)
    convert = commands.add_parser("convert", help="固定ShapeのFP32／FP16 OpenVINOモデルを作成")
    convert.add_argument("--run-id", required=True)
    cpu_check = commands.add_parser("cpu-check", help="元モデル・ONNX・OpenVINO CPUの差分を照合")
    cpu_check.add_argument("--run-id", required=True)
    npu = commands.add_parser("npu-run", help="G2合格後にNPUを明示指定して推論")
    npu.add_argument("--run-id", required=True)
    npu.add_argument("--cache-from-run", help="同条件で成功したG3 Runのキャッシュを再利用")
    probe = commands.add_parser("npu-probe", help="固定Shape対照モデルまたは保存済みLaya IRをNPU経路別に診断")
    probe.add_argument("--kind", choices=("control", "laya"), required=True)
    probe.add_argument("--compiler", choices=("default", "driver", "plugin"), required=True)
    probe.add_argument("--source-run", help="G2 PASS済みのLaya Run。kind=layaで必須")
    perf = commands.add_parser("perf-run", help="CPU 1/2スレッドまたはNPU PLUGINの固定入力予備測定")
    perf.add_argument("--source-run", required=True)
    perf.add_argument("--backend", choices=("cpu", "npu"), required=True)
    perf.add_argument("--threads", type=int, choices=(1, 2))
    perf.add_argument("--trial", type=int, choices=(1, 2, 3), required=True)
    perf.add_argument("--cache-from-run", help="NPUのキャッシュ再利用元Run。CPUでは使用不可")
    perf.add_argument("--idle-seconds", type=int, choices=(0, 300), default=0,
                      help="測定後の待機観測秒数（0または300）")
    review = commands.add_parser("review-html", help="60件の人手確認資料HTMLを生成")
    review.add_argument("--output", default="QUALITY-REVIEW-JA-60.html")
    return parser


def _run_for(root: Path, run_id: str | None, purpose: str) -> Run:
    return open_run(root, run_id) if run_id else Run.create(root, purpose)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = project_root()
    try:
        if args.command == "run-init":
            run = Run.create(root, args.purpose)
            print(json.dumps({"status": "RUNNING", "run_id": run.run_id}, ensure_ascii=False))
            return EXIT_OK
        if args.command == "validate-request":
            validate_request(_request_json(Path(args.input)))
            print(json.dumps({"status": "PASS", "code": "INPUT_VALID"}, ensure_ascii=False))
            return EXIT_OK
        if args.command == "validate":
            run = _run_for(root, args.run_id, "G0")
            try:
                result = validate_project(root)
                run.stage("G0", "PASS", **result)
                print(json.dumps({"status": "PASS", "run_id": run.run_id, **result}, ensure_ascii=False))
                return EXIT_OK
            except Exception as exc:
                code = exc.code if isinstance(exc, ContractError) else "VALIDATION_ERROR"
                run.failure("G0", code, exc)
                print(json.dumps({"status": "FAIL", "run_id": run.run_id, "code": code}, ensure_ascii=False))
                return EXIT_INPUT if isinstance(exc, ContractError) else EXIT_INTEGRITY
        if args.command == "doctor":
            run = _run_for(root, args.run_id, "G3")
            result = doctor(run, args.device, "3720")
            status = result["status"]
            print(json.dumps({"status": status, "run_id": run.run_id, "code": result.get("code")}, ensure_ascii=False))
            return EXIT_OK if status == "PASS" else EXIT_ENVIRONMENT
        if args.command == "prepare":
            run = _run_for(root, args.run_id, "G0")
            try:
                result = prepare_assets(root, args.allow_download)
                run.stage("G0-assets", "PASS", verified_artifacts=result["verified_artifacts"],
                          laya_source_commit=result["laya_source_commit"],
                          source_directory=result["source_directory"])
            except Exception as exc:
                code = getattr(exc, "code", "ASSET_PREPARATION_FAILED")
                run.failure("G0-assets", code, exc)
                print(json.dumps({"status": "FAIL", "run_id": run.run_id, "code": code}, ensure_ascii=False))
                return EXIT_INTEGRITY
            print(json.dumps({"status": "PASS", "run_id": run.run_id,
                              "verified_artifacts": result["verified_artifacts"],
                              "laya_source_commit": result["laya_source_commit"]}, ensure_ascii=False))
            return EXIT_OK
        if args.command in {"reference", "convert", "cpu-check", "npu-run"}:
            run = open_run(root, args.run_id)
            if args.command == "reference":
                result = run_reference(root, run)
            elif args.command == "convert":
                result = convert_model(root, run)
            elif args.command == "cpu-check":
                result = run_cpu_check(root, run)
            else:
                result = run_npu(root, run, "3720", cache_from_run=args.cache_from_run)
            print(json.dumps({"run_id": run.run_id, **result}, ensure_ascii=False))
            if args.command == "npu-run" and args.cache_from_run:
                if result.get("cache_test", {}).get("status") != "PASS":
                    return EXIT_INTEGRITY
            if result["status"] == "PASS" or result["status"] == "UNVERIFIED_LABEL_REFERENCE_ONLY":
                return EXIT_OK
            if result["status"] == "BLOCKED":
                return EXIT_ENVIRONMENT
            return EXIT_MODEL if result.get("code", "").startswith(("REFERENCE", "ONNX", "MODEL")) else EXIT_INTEGRITY
        if args.command == "npu-probe":
            if args.kind == "laya" and not args.source_run:
                raise ContractError("SOURCE_RUN_REQUIRED", "kind=layaでは--source-runが必要です。")
            if args.kind == "control" and args.source_run:
                raise ContractError("SOURCE_RUN_NOT_ALLOWED", "kind=controlでは--source-runを指定しません。")
            run = Run.create(root, "NPU_DIAGNOSTIC")
            if args.kind == "control":
                result = run_npu_control(root, run, args.compiler.upper(), "3720")
            else:
                source_run = open_run(root, args.source_run)
                result = run_npu(root, run, "3720", compiler_type=args.compiler.upper(), source_run=source_run)
            print(json.dumps({"run_id": run.run_id, **result}, ensure_ascii=False))
            return EXIT_OK if result["status"] == "PASS" else EXIT_ENVIRONMENT if result["status"] == "BLOCKED" else EXIT_MODEL
        if args.command == "perf-run":
            if args.backend == "cpu" and (args.threads not in (1, 2) or args.cache_from_run):
                raise ContractError("PERFORMANCE_OPTIONS_INVALID", "CPUは--threads 1/2が必要で、--cache-from-runは使えません。")
            if args.backend == "npu" and args.threads is not None:
                raise ContractError("PERFORMANCE_OPTIONS_INVALID", "NPUでは--threadsを指定しません。")
            from .performance import run_performance
            source_run = open_run(root, args.source_run)
            result = run_performance(root, source_run, args.backend, args.threads, args.trial,
                                     args.cache_from_run, args.idle_seconds)
            print(json.dumps(result, ensure_ascii=False))
            return EXIT_OK if result["status"] == "PASS" else EXIT_RESOURCE
        if args.command == "review-html":
            from .review import build_review_html
            destination = (root.parent / "docs" / args.output).resolve()
            docs_dir = (root.parent / "docs").resolve()
            if destination.parent != docs_dir or destination.suffix.lower() != ".html":
                raise ContractError("REVIEW_OUTPUT_INVALID", "出力先はdocs直下のHTMLに限定します。")
            output = build_review_html(root, destination)
            print(json.dumps({"status": "PASS", "output": str(output)}, ensure_ascii=False))
            return EXIT_OK
    except ContractError as exc:
        print(json.dumps({"status": "FAIL", "code": exc.code, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return EXIT_INPUT
    except AssetError as exc:
        print(json.dumps({"status": "FAIL", "code": exc.code, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return EXIT_INTEGRITY
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "code": "COMMAND_ERROR", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return EXIT_INTEGRITY
    parser = _parser()
    parser.error("未実装のコマンドです。")
    return EXIT_INPUT


if __name__ == "__main__":
    raise SystemExit(main())

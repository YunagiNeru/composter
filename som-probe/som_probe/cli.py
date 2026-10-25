"""som-probeの小さなコマンドライン入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .assets import AssetError, prepare_assets, verify_assets
from .contracts import ContractError, Profile, profile_from_mapping, profile_mapping, validate_request
from .dataset import DATASET_FILES, dataset_identity, load_dataset
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


def _profile_for(root: Path, selected: str | None = None) -> dict[str, Any]:
    config = _json(root / "configs/probe.json")
    profiles = config.get("profiles")
    if (config.get("default_profile") != "s256-k8" or not isinstance(profiles, dict)
            or set(profiles) != {"s256-k8", "s128-k8"}):
        raise ContractError("PROFILE_CONFIG_INVALID", "固定プロファイルが不足または変更されています。")
    for name, value in profiles.items():
        if profile_mapping(profile_from_mapping(value)) != value or value["name"] != name:
            raise ContractError("PROFILE_CONFIG_INVALID", "プロファイル名と固定形状が一致しません。")
    name = selected or config["default_profile"]
    if name not in profiles:
        raise ContractError("PROFILE_UNKNOWN", "指定された固定プロファイルがありません。")
    return profiles[name]


def validate_project(root: Path, selected_profile: str = "s256-k8",
                     dataset_version: str = "dev-v1") -> dict[str, Any]:
    required = [
        "configs/probe.json", "configs/models.lock.json", "configs/gates.json",
        "datasets/manifest.json", "datasets/dev.jsonl",
    ]
    if dataset_version != "dev-v1":
        files = DATASET_FILES[dataset_version]
        required.extend(path for path in (
            files["manifest"], files["data"], files["revisions"], files["contract"]
        ) if path is not None)
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ContractError("REQUIRED_FILES_MISSING", "必須ファイルがありません: " + ", ".join(missing))
    config = _json(root / "configs/probe.json")
    profile_spec = _profile_for(root, selected_profile)
    profile = profile_from_mapping(profile_spec)
    lock = _json(root / "configs/models.lock.json")
    gates = _json(root / "configs/gates.json")
    if lock.get("schema_version") != 1 or not lock.get("sources") or not lock.get("downloads"):
        raise ContractError("MODEL_LOCK_INVALID", "モデル固定情報が不正です。")
    if gates.get("schema_version") != 1 or not {"G0", "G1", "G2", "G3", "G4", "G5"}.issubset(gates.get("stages", {})):
        raise ContractError("GATES_INVALID", "段階別ゲートが不足しています。")
    snapshot = load_dataset(root, dataset_version, profile)
    rows = snapshot["cases"]
    verified = verify_assets(root)
    return {"files_checked": len(required), "profile": profile_spec,
            "dev_cases": len(rows), "verified_artifacts": len(verified),
            "verified_artifact_sha256": verified,
            "dataset_identity": snapshot["identity"],
            "question_type_counts": snapshot["manifest"]["splits"]["dev"]["question_types"],
            "label_state": "ai_draft_unverified", "official_quality_status": "UNDETERMINED"}


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
    init.add_argument("--profile", choices=("s256-k8", "s128-k8"), default=None)
    init.add_argument("--dataset-version", choices=("dev-v1", "dev-v2", "dev-v3"), default="dev-v1")
    init.add_argument("--compare-input-run", help="s128-k8入力を照合するs256-k8の元入力Run")
    validate = commands.add_parser("validate", help="設定、固定情報、60件の入力契約を検査")
    validate.add_argument("--run-id")
    validate.add_argument("--dataset-version", choices=("dev-v1", "dev-v2", "dev-v3"))
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
    perf.add_argument("--load-method", choices=("read-compile", "path-compile"), default="read-compile")
    memory = commands.add_parser("memory-run", help="通常時間測定と分離した段階別メモリ測定")
    memory.add_argument("--source-run", required=True)
    memory.add_argument("--backend", choices=("cpu", "npu"), required=True)
    memory.add_argument("--threads", type=int, choices=(1, 2))
    memory.add_argument("--load-method", choices=("read-compile", "path-compile"), required=True)
    memory.add_argument("--cache-from-run", help="NPUキャッシュ再利用元Run。CPUでは使用不可")
    review = commands.add_parser("review-html", help="60件の人手確認資料HTMLを生成")
    review.add_argument("--output", default="QUALITY-REVIEW-JA-60.html")
    review.add_argument("--dataset-version", choices=("dev-v1", "dev-v2", "dev-v3"), default="dev-v1")
    sokudan = commands.add_parser("sokudan-cpu", help="固定Sokudanモデルをdev-v3でCPU予備比較")
    sokudan.add_argument("--laya-reference-run", required=True,
                         help="同じdev-v3を使った保存済みLaya CPU参照Run")
    tryout = commands.add_parser("tryout", help="文章を対話入力してモデル判断を試用")
    tryout.add_argument("--model", choices=("laya", "sokudan"), required=True)
    tryout.add_argument("--source-run", help="Layaの固定s256-k8 IR取得元Run。Sokudanでは使用不可")
    commands.add_parser("diagnose-tryout", help="固定4入力で公式APIから対話表示までを照合")
    commands.add_parser("compare-tryout-questions", help="dev-v3現行質問と固定分解質問を両モデルCPUで比較")
    tryout_memory = commands.add_parser("tryout-memory", help="現行tryoutの3時点メモリを固定文で採取")
    tryout_memory.add_argument("--model", choices=("laya", "sokudan"), required=True)
    tryout_memory.add_argument("--source-run", help="Layaの固定s256-k8 IR取得元Run")
    zero_shot = commands.add_parser("zero-shot-cpu", help="固定候補AをCPUでdev-v3比較し、候補BはBLOCKEDとして記録")
    zero_shot.add_argument("--condition", choices=("english", "japanese", "all"), default="all")
    zero_shot.add_argument("--trials", type=int, choices=(3,), default=3)
    zero_shot.add_argument("--trial", type=int, choices=(1, 2, 3), help=argparse.SUPPRESS)
    zero_shot.add_argument("--run-id", help=argparse.SUPPRESS)
    zero_shot.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _run_for(root: Path, run_id: str | None, purpose: str,
             dataset_version: str = "dev-v1") -> Run:
    return open_run(root, run_id) if run_id else Run.create(root, purpose, dataset_version=dataset_version)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = project_root()
    try:
        if args.command == "run-init":
            profile = _profile_for(root, args.profile)
            if args.compare_input_run:
                if profile["name"] != "s128-k8":
                    raise ContractError("PROFILE_COMPARISON_INVALID", "--compare-input-runはs128-k8だけで使えます。")
                baseline = open_run(root, args.compare_input_run)
                if baseline.profile_config["name"] != "s256-k8":
                    raise ContractError("PROFILE_COMPARISON_INVALID", "比較元はs256-k8でなければなりません。")
                selected_identity = dataset_identity(root, args.dataset_version)
                if (baseline.dataset_version != args.dataset_version
                        or baseline.dataset_identity != selected_identity):
                    raise ContractError("PROFILE_DATASET_MISMATCH", "形状比較元とデータ版が一致しません。")
            run = Run.create(root, args.purpose, profile, args.compare_input_run, args.dataset_version)
            print(json.dumps({"status": "RUNNING", "run_id": run.run_id}, ensure_ascii=False))
            return EXIT_OK
        if args.command == "validate-request":
            validate_request(_request_json(Path(args.input)))
            print(json.dumps({"status": "PASS", "code": "INPUT_VALID"}, ensure_ascii=False))
            return EXIT_OK
        if args.command == "validate":
            run = _run_for(root, args.run_id, "G0", args.dataset_version or "dev-v1")
            try:
                if args.dataset_version and run.dataset_version != args.dataset_version:
                    raise ContractError("DATASET_VERSION_MISMATCH", "指定版とRun manifestのデータ版が一致しません。")
                result = validate_project(root, run.profile_config["name"], run.dataset_version)
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
            if args.kind == "control":
                run = Run.create(root, "NPU_DIAGNOSTIC")
                result = run_npu_control(root, run, args.compiler.upper(), "3720")
            else:
                source_run = open_run(root, args.source_run)
                run = Run.create(root, "NPU_DIAGNOSTIC", source_run.profile_config,
                                 dataset_version=source_run.dataset_version)
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
                                     args.cache_from_run, args.idle_seconds, args.load_method)
            print(json.dumps(result, ensure_ascii=False))
            return EXIT_OK if result["status"] == "PASS" else EXIT_RESOURCE
        if args.command == "memory-run":
            if args.backend == "cpu" and (args.threads not in (1, 2) or args.cache_from_run):
                raise ContractError("MEMORY_OPTIONS_INVALID", "CPUメモリ測定は1/2スレッドでキャッシュ再利用なしです。")
            if args.backend == "npu" and args.threads is not None:
                raise ContractError("MEMORY_OPTIONS_INVALID", "NPUメモリ測定では--threadsを指定しません。")
            from .performance import run_memory_diagnostic
            source_run = open_run(root, args.source_run)
            result = run_memory_diagnostic(root, source_run, args.backend, args.threads,
                                           args.load_method, args.cache_from_run)
            print(json.dumps(result, ensure_ascii=False))
            return EXIT_OK if result["status"] == "PASS" else EXIT_RESOURCE
        if args.command == "review-html":
            from .review import build_review_html
            destination = (root.parent / "docs" / args.output).resolve()
            docs_dir = (root.parent / "docs").resolve()
            if destination.parent != docs_dir or destination.suffix.lower() != ".html":
                raise ContractError("REVIEW_OUTPUT_INVALID", "出力先はdocs直下のHTMLに限定します。")
            output = build_review_html(root, destination, args.dataset_version)
            print(json.dumps({"status": "PASS", "output": str(output)}, ensure_ascii=False))
            return EXIT_OK
        if args.command == "sokudan-cpu":
            from .sokudan_cpu import run_sokudan_cpu
            result = run_sokudan_cpu(root, args.laya_reference_run)
            print(json.dumps(result, ensure_ascii=False))
            return EXIT_OK if result["status"] == "PASS" else EXIT_MODEL
        if args.command == "tryout":
            from .tryout import TryoutError, run_tryout
            try:
                run_tryout(root, args.model, args.source_run)
            except TryoutError as exc:
                print(f"対話試用を開始できません（{exc.code}）。", file=sys.stderr)
                return EXIT_MODEL
            except KeyboardInterrupt:
                print("起動を中断しました。", file=sys.stderr)
                return EXIT_OK
            return EXIT_OK
        if args.command == "diagnose-tryout":
            from .diagnostics import run_fixed_diagnostic
            result = run_fixed_diagnostic(root)
            print(json.dumps(result, ensure_ascii=False))
            return EXIT_OK if result["status"] == "COMPLETE" else EXIT_MODEL
        if args.command == "compare-tryout-questions":
            from .tryout_comparison import run_comparison
            result = run_comparison(root)
            print(json.dumps(result, ensure_ascii=False))
            return EXIT_OK if result["status"] == "COMPLETE" else EXIT_MODEL
        if args.command == "tryout-memory":
            from .tryout_memory import run_tryout_memory
            result = run_tryout_memory(root, args.model, args.source_run)
            print(json.dumps(result, ensure_ascii=False))
            return EXIT_OK if result.get("status") == "PASS" else EXIT_RESOURCE
        if args.command == "zero-shot-cpu":
            from .zero_shot import LANGUAGE_CONDITIONS
            from .zero_shot_eval import run_all_trials, run_child
            if args.child:
                if not args.run_id or args.trial is None or args.condition == "all":
                    raise ContractError("CHILD_TRIAL_ARGUMENTS_INVALID", "内部試験の引数が不足しています。")
                result = run_child(root, args.run_id, args.condition, args.trial)
                print(json.dumps(result, ensure_ascii=False))
                return EXIT_OK if result.get("status") == "COMPLETE" else EXIT_MODEL
            if args.run_id or args.trial is not None:
                raise ContractError("PUBLIC_TRIAL_ARGUMENTS_INVALID", "試験ごとのrun識別子は内部起動でのみ使います。")
            conditions = list(LANGUAGE_CONDITIONS) if args.condition == "all" else [args.condition]
            result = run_all_trials(root, conditions, args.trials)
            print(json.dumps(result, ensure_ascii=False))
            return EXIT_OK if result.get("status") == "COMPLETE" else EXIT_MODEL
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

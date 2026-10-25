"""Single-process memory snapshots for the existing interactive tryout path."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from .storage import Run


IDLE_SECONDS = 60
FIXED_CASE_ID = "work-task-v3"
FIXED_INPUT = "今週中に勤怠まとめて提出"
SOURCE_RUN_ID = "20261025-041751-58713e9e"


def _expected_python(root: Path, model: str) -> Path:
    environment = "venv-run" if model == "laya" else "venv-sokudan-cpu"
    return (root / ".local" / environment / "Scripts" / "python.exe").resolve()


def _private_startup_text(value: str) -> str:
    return re.sub(r"PID：\d+", "PID：[記録対象外]", value)


def _memory_reader():
    from .performance import _memory

    return _memory


def _lock_fingerprints(root: Path, model: str) -> dict[str, str]:
    from .assets import sha256_file

    if model == "laya":
        names = ("configs/models.lock.json", "requirements/run-win-x64.lock")
    else:
        names = ("configs/sokudan-cpu.lock.json", "requirements/sokudan-cpu-win-x64.lock")
    return {name: sha256_file(root / name) for name in names}


def run_tryout_memory(root: Path, model: str, source_run_id: str | None = None) -> dict[str, Any]:
    if model not in {"laya", "sokudan"}:
        return {"status": "FAILED", "code": "MODEL_CHOICE_INVALID"}
    if model == "laya" and source_run_id != SOURCE_RUN_ID:
        return {"status": "FAILED", "code": "LAYA_SOURCE_RUN_REQUIRED_OR_INVALID"}
    if model == "sokudan" and source_run_id is not None:
        return {"status": "FAILED", "code": "SOKUDAN_SOURCE_RUN_NOT_ALLOWED"}

    run = Run.create(root, f"current tryout memory phases {model}",
                     dataset_version="dev-v3", profile_applicable=False)
    expected_python = _expected_python(root, model)
    if os.path.normcase(str(Path(sys.executable).resolve())) != os.path.normcase(str(expected_python)):
        code = "TRYOUT_MEMORY_PYTHON_ENVIRONMENT_MISMATCH"
        run.blocked("TRYOUT_MEMORY", code, "モデル専用の固定仮想環境から実行してください。",
                    model=model)
        return {"status": "BLOCKED", "run_id": run.run_id, "code": code}

    memory_reader = _memory_reader()
    lock_fingerprints = _lock_fingerprints(root, model)
    started = time.perf_counter()
    phases: dict[str, Any] = {}
    startup = None
    input_calls = 0
    inference_succeeded = False
    inference_error_code = None

    def capture(name: str) -> None:
        phases[name] = {
            "elapsed_since_command_start_ms": (time.perf_counter() - started) * 1000,
            "counters": memory_reader(),
        }

    def output_fn(line: str) -> None:
        nonlocal startup, inference_succeeded, inference_error_code
        if line.startswith("ロード完了："):
            startup = _private_startup_text(line.removeprefix("ロード完了："))
            capture("after_model_load")
        elif line.startswith("【モデルの判断】"):
            inference_succeeded = True
        else:
            match = re.search(r"推論(?:できませんでした|に失敗しました)（([A-Z0-9_]+)）", line)
            if match:
                inference_error_code = match.group(1)

    def input_fn(_prompt: str) -> str:
        nonlocal input_calls
        input_calls += 1
        if input_calls == 1:
            return FIXED_INPUT
        if input_calls == 2:
            capture("after_one_inference")
            time.sleep(IDLE_SECONDS)
            capture("after_idle_wait")
            return ":quit"
        return ":quit"

    try:
        from .tryout import run_tryout

        run_tryout(root, model, source_run_id, input_fn, output_fn,
                   diagnostic_run_id=run.run_id if model == "laya" else None)
        phases_complete = set(phases) == {
            "after_model_load", "after_one_inference", "after_idle_wait",
        }
        metrics_available = phases_complete and all(
            phases[name]["counters"].get("status") == "available" for name in phases
        )
        result = {
            "status": "PASS" if phases_complete and metrics_available and inference_succeeded else "FAIL",
            "purpose": "CURRENT_TRYOUT_MEMORY_ONLY_NO_INTERACTIVE_TEXT_OR_OUTPUT_SAVED",
            "run_id": run.run_id,
            "git_revision": run.read_manifest()["git_revision"],
            "source_tree_sha256": run.read_manifest()["source_tree_sha256"],
            "worktree_status": run.read_manifest()["worktree_status"],
            "dataset_version": "dev-v3",
            "dataset_sha256": run.read_manifest()["input_sha256"].get("datasets/dev-v3.jsonl"),
            "model": model,
            "model_process_pid_recorded": False,
            "python_executable": str(Path(sys.executable).resolve()),
            "requested_route": ("OpenVINO NPU / PLUGIN / FP16" if model == "laya"
                                else "torch CPU / FP32"),
            "lock_sha256": lock_fingerprints,
            "startup_information_without_pid": startup,
            "source_run_id": source_run_id,
            "fixed_input_case_id": FIXED_CASE_ID,
            "fixed_input_sha256": hashlib.sha256(FIXED_INPUT.encode("utf-8")).hexdigest(),
            "fixed_input_text_saved": False,
            "inference_output_saved": False,
            "idle_wait_seconds": IDLE_SECONDS,
            "inference_succeeded": inference_succeeded,
            "inference_error_code": inference_error_code,
            "phases": phases,
            "memory_semantics": {
                "WorkingSetSize": "採取時点の現在Working Set。Commit Chargeとは合算しない。",
                "PrivateUsage_and_PagefileUsage": "採取時点の現在Commit Charge。ページファイルへの実書込量ではない。",
                "PeakWorkingSetSize": "プロセス起動以来の累積ピーク。段階別瞬間ピークではない。",
                "PeakPagefileUsage": "プロセス起動以来の累積Commit Chargeピーク。段階別瞬間ピークではない。",
            },
            "scope": "初期ロード、固定文1件後、無入力待機後のみ。繰り返し推論による増加は未測定。",
        }
        result_file = "tryout-memory-result.json" if result["status"] == "PASS" else "tryout-memory-diagnostic.json"
        run.write_json(result_file, result)
        if result["status"] == "PASS":
            run.stage("TRYOUT_MEMORY", "PASS", result_file=result_file,
                      model=model, phases=sorted(phases), inference_succeeded=inference_succeeded)
        else:
            run.failure("TRYOUT_MEMORY", inference_error_code or "MEMORY_PHASE_OR_COUNTER_FAILED",
                        RuntimeError("MEMORY_PHASE_OR_COUNTER_FAILED"),
                        result_file=result_file, model=model, phases=sorted(phases),
                        inference_succeeded=inference_succeeded)
            result["code"] = inference_error_code or "MEMORY_PHASE_OR_COUNTER_FAILED"
        return result
    except Exception as exc:
        code = getattr(exc, "code", "TRYOUT_MEMORY_FAILED")
        try:
            run.write_json("tryout-memory-diagnostic.json", {
                "status": "FAIL", "run_id": run.run_id, "model": model,
                "error_code": code, "error_type": type(exc).__name__,
                "phases": phases, "inference_succeeded": inference_succeeded,
                "scope": "入力文・推論結果は診断記録へ保存しない。",
            })
            run.failure("TRYOUT_MEMORY", code, RuntimeError(code), model=model,
                        phases=sorted(phases))
        except FileExistsError:
            run.failure("TRYOUT_MEMORY", code, RuntimeError(code), model=model,
                        phases=sorted(phases))
        return {"status": "FAILED", "run_id": run.run_id, "code": code}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("laya", "sokudan"), required=True)
    parser.add_argument("--source-run", help="Laya固定IR取得元Run")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    result = run_tryout_memory(root, args.model, args.source_run)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "PASS" else 6


if __name__ == "__main__":
    raise SystemExit(main())

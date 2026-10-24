"""Fixed-input CPU/NPU preliminary timing for the existing FP16 IR."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .assets import asset_path, load_lock, sha256_file, verify_assets
from .runtime import (
    RuntimeErrorRecord, _cache_inventory, _comparison, _draft_label_summary, _execution_device_names,
    _exception_details, _input_arrays, _input_digest, _json_safe, _load_run_conversion, _read_jsonl, _row_output,
    _select_outputs, _type_comparison, inspect_device, process_age_ms as _process_age_ms,
)
from .storage import Run, open_run


WARMUP_EVENTS = 20
MEASURED_EVENTS_PER_ROUND = 200
MEASUREMENT_ROUNDS = 3
INPUT_NAMES = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def _render_options(question: dict[str, Any]) -> list[str]:
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        return [str(key) if value in (None, "") else f"{key}: {value}" for key, value in criteria.items()]
    if kind == "score":
        return [f"level {index}: {value}" for index, value in enumerate(criteria)]
    return ["false: no, the statement does not hold", "true: yes, the statement holds"]


def _preprocess_case(case: dict[str, Any], saved: dict[str, Any], tokenizer, special: dict[str, int],
                     mask_token: str) -> tuple[dict[str, Any], int]:
    question_id, question = next(iter(case["questions"].items()))
    kind = "noul" if question["type"] == "bool" else question["type"]
    head = _encode(tokenizer, f"{kind} question: {question['instructions'].replace(mask_token, ' ')}")
    option_ids = [[special["mask"]] + _encode(tokenizer, " " + option.replace(mask_token, " "))[:48]
                  for option in _render_options(question)]
    budget = 192 - sum(len(option) for option in option_ids)
    if budget < 16:
        per_option = max(4, (192 - 16) // max(1, len(option_ids)))
        option_ids = [option[:per_option] for option in option_ids]
        budget = 192 - sum(len(option) for option in option_ids)
    ids = [special["cls"]] + head[:max(8, budget)] + [special["sep"]]
    markers: list[int] = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(special["sep"])
    state_ids = _encode(tokenizer, str(case["state"]).replace(mask_token, " "))
    state_ids = state_ids[:max(0, 256 - len(ids) - 1)]
    ids = (ids + state_ids + [special["sep"]])[:256]
    token_count = len(ids)
    inputs = {
        "input_ids": [ids + [special["pad"]] * (256 - len(ids))],
        "attention_mask": [[1] * len(ids) + [0] * (256 - len(ids))],
        "marker_pos": [markers + [0] * (8 - len(markers))],
        "marker_mask": [[True] * len(markers) + [False] * (8 - len(markers))],
        "qtype": [{"choice": 0, "score": 1, "noul": 2}[kind]],
    }
    row = {**saved, "question_id": question_id, "question_type": kind, "inputs": inputs,
           "input_sha256": _input_digest(inputs)}
    if row["candidate_count"] != len(markers):
        raise RuntimeErrorRecord("PERF_CANDIDATE_COUNT_MISMATCH", "予備測定の入力候補数が保存入力と一致しません。")
    return row, token_count


def _verify_preprocessor(root: Path, cases: list[dict[str, Any]],
                         input_rows: list[dict[str, Any]]) -> tuple[Any, dict[str, int], list[dict[str, Any]], dict[str, int], float]:
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise RuntimeErrorRecord("PERF_TOKENIZER_UNAVAILABLE", "実行用ロックのTokenizerを読み込めません。") from exc
    tokenizer_config = json.loads(asset_path(root, "reference/tokenizer/tokenizer_config.json").read_text(encoding="utf-8"))
    load_start = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(asset_path(root, "reference/tokenizer/tokenizer.json")))
    tokenizer_load_ms = (time.perf_counter() - load_start) * 1000
    special = {key: tokenizer.token_to_id(tokenizer_config[field])
               for key, field in (("cls", "cls_token"), ("sep", "sep_token"),
                                  ("mask", "mask_token"), ("pad", "pad_token"))}
    if any(value is None for value in special.values()):
        raise RuntimeErrorRecord("PERF_SPECIAL_TOKEN_MISSING", "固定Tokenizerの特殊トークンIDを確認できません。")
    if len(cases) != len(input_rows):
        raise RuntimeErrorRecord("PERF_INPUT_COUNT_MISMATCH", "固定データとG3入力件数が一致しません。")
    saved_by_id = {row["case_id"]: row for row in input_rows}
    prepared: list[dict[str, Any]] = []
    token_lengths: dict[str, int] = {}
    for case in cases:
        saved = saved_by_id.get(case["case_id"])
        if saved is None:
            raise RuntimeErrorRecord("PERF_INPUT_ID_MISMATCH", "固定データに対応するG3入力がありません。")
        row, token_count = _preprocess_case(case, saved, tokenizer, special, tokenizer_config["mask_token"])
        if row["input_sha256"] != saved["input_sha256"] or row["inputs"] != saved["inputs"]:
            raise RuntimeErrorRecord("PERF_TOKENIZER_PARITY_FAILED", f"{case['case_id']}の再生成入力がG3保存入力と一致しません。")
        token_lengths[case["case_id"]] = token_count
        prepared.append(row)
    return tokenizer, special, prepared, token_lengths, tokenizer_load_ms


def _four_request_bundles(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_type = {kind: [row for row in rows if row["question_type"] == kind]
               for kind in ("choice", "score", "noul")}
    if any(len(by_type[kind]) != 20 for kind in by_type):
        raise RuntimeErrorRecord("PERF_TYPE_DISTRIBUTION_INVALID", "4判断束を作る60件の型別件数が20件ずつではありません。")
    bundles = []
    for index in range(15):
        kind = ("choice", "score", "noul")[index % 3]
        extra = by_type[kind][15 + index // 3]
        bundles.append([by_type["choice"][index], by_type["score"][index], by_type["noul"][index], extra])
    return bundles


def _memory() -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "unavailable_non_windows"}
    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("Psapi.dll")
    kernel = ctypes.WinDLL("kernel32.dll")
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        return {"status": "unavailable", "error_code": ctypes.get_last_error()}
    return {"status": "available", "working_set_bytes": int(counters.WorkingSetSize),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "private_bytes": int(counters.PrivateUsage), "peak_private_bytes": int(counters.PeakPagefileUsage)}


def _available_ram() -> int | None:
    if os.name != "nt":
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.WinDLL("kernel32.dll").GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys)


def _system_cpu_ticks() -> tuple[int, int] | None:
    if os.name != "nt":
        return None
    class FileTime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]
    idle, kernel, user = FileTime(), FileTime(), FileTime()
    if not ctypes.WinDLL("kernel32.dll").GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    def value(item: Any) -> int:
        return (int(item.dwHighDateTime) << 32) | int(item.dwLowDateTime)
    return value(idle), value(kernel) + value(user)


def _tree_size(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"bytes": 0, "files": 0, "status": "missing"}
    total = count = 0
    for current, directories, files in os.walk(path):
        directories[:] = sorted(name for name in directories if name not in {"__pycache__", ".git"})
        for name in files:
            file_path = Path(current) / name
            try:
                total += file_path.stat().st_size
                count += 1
            except OSError:
                continue
    return {"bytes": total, "files": count, "status": "measured"}


def _percentiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    import numpy as np
    return {"samples": len(values), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": float(max(values))}


def _event(case_rows: list[tuple[dict[str, Any], dict[str, Any]]], tokenizer, special,
           mask_token: str, compiled) -> tuple[dict[str, float], list[dict[str, Any]]]:
    prep_ms = model_ms = post_ms = 0.0
    outputs = []
    for case, saved in case_rows:
        start = time.perf_counter()
        row, _ = _preprocess_case(case, saved, tokenizer, special, mask_token)
        arrays = _input_arrays(row)
        prep_ms += (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        result = compiled({name: arrays[name] for name in INPUT_NAMES})
        model_ms += (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        named = {port.get_any_name(): value for port, value in result.items()}
        logits, actions = _select_outputs(named)
        outputs.append(_row_output(row, logits, actions))
        post_ms += (time.perf_counter() - start) * 1000
    return {"preprocess_ms": prep_ms, "model_ms": model_ms, "postprocess_ms": post_ms}, outputs


def run_performance(root: Path, source_run: Run, backend: str, threads: int | None,
                    trial: int, cache_from_run: str | None = None,
                    idle_seconds: int = 0) -> dict[str, Any]:
    run = Run.create(root, f"PERF_{backend.upper()}_{threads or 'PLUGIN'}_TRIAL_{trial}")
    manifest = source_run.read_manifest()
    if manifest.get("stages", {}).get("G2-cpu-check", {}).get("status") != "PASS":
        run.blocked("PERFORMANCE", "CPU_PARITY_NOT_PASS", "参照RunのCPU照合がPASSしていません。")
        return {"status": "BLOCKED", "run_id": run.run_id, "code": "CPU_PARITY_NOT_PASS"}
    if manifest.get("stages", {}).get("G3", {}).get("status") != "PASS":
        run.blocked("PERFORMANCE", "G3_NOT_PASS", "最終G3がPASSしていないため性能測定を開始しません。")
        return {"status": "BLOCKED", "run_id": run.run_id, "code": "G3_NOT_PASS"}
    try:
        source_npu_result = json.loads((source_run.path / "npu-result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        run.blocked("PERFORMANCE", "G3_NPU_RESULT_MISSING", "最終G3のNPU結果を読み込めません。")
        return {"status": "BLOCKED", "run_id": run.run_id, "code": "G3_NPU_RESULT_MISSING"}
    source_properties = source_npu_result.get("compiled_model_properties", {})
    source_devices = _execution_device_names(source_properties.get("EXECUTION_DEVICES"))
    if (source_npu_result.get("status") != "PASS"
            or source_npu_result.get("device_architecture") != "3720"
            or not source_devices
            or any(not (name.upper() == "NPU" or name.upper().startswith("NPU.")) for name in source_devices)
            or not str(source_properties.get("NPU_COMPILER_TYPE", "")).upper().endswith("PLUGIN")):
        run.blocked("PERFORMANCE", "G3_NPU_EVIDENCE_INVALID", "性能測定の基準G3が3720上のNPU PLUGIN実推論を証明していません。")
        return {"status": "BLOCKED", "run_id": run.run_id, "code": "G3_NPU_EVIDENCE_INVALID"}
    current_manifest = run.read_manifest()
    if (manifest.get("git_revision") != current_manifest.get("git_revision")
            or manifest.get("source_tree_sha256") != current_manifest.get("source_tree_sha256")
            or manifest.get("input_sha256") != current_manifest.get("input_sha256")):
        run.blocked("PERFORMANCE", "SOURCE_FINGERPRINT_MISMATCH", "性能RunとG3入力Runのコミット・コード・設定・データ・依存ロックが一致しません。")
        return {"status": "BLOCKED", "run_id": run.run_id, "code": "SOURCE_FINGERPRINT_MISMATCH"}
    try:
        launch_age_ms = _process_age_ms()
        phase_memory: dict[str, Any] = {"process_start": _memory()}
        openvino_import_start = time.perf_counter()
        import openvino as ov
        openvino_import_ms = (time.perf_counter() - openvino_import_start) * 1000
        phase_memory["after_openvino_import"] = _memory()
        phase_memory["available_physical_ram_after_import"] = _available_ram()

        conversion = _load_run_conversion(root, source_run)
        actual_assets = verify_assets(root)
        if actual_assets != manifest.get("verified_model_artifact_sha256"):
            raise RuntimeErrorRecord("MODEL_ARTIFACT_FINGERPRINT_MISMATCH", "取得モデルの検証ハッシュがG3 Runと一致しません。")
        xml = root / conversion["fp16_xml"]
        if sha256_file(xml) != conversion["fp16_xml_sha256"] or sha256_file(xml.with_suffix(".bin")) != conversion["fp16_bin_sha256"]:
            raise RuntimeErrorRecord("CONVERTED_MODEL_HASH_MISMATCH", "性能測定のFP16 IRハッシュが一致しません。")
        input_rows = _read_jsonl(source_run.path / "inputs.jsonl")
        references = _read_jsonl(source_run.path / "reference.jsonl")
        cases = [json.loads(line) for line in (root / "datasets" / "dev.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        saved_by_id = {row["case_id"]: row for row in input_rows}
        special_config = json.loads(asset_path(root, "reference/tokenizer/tokenizer_config.json").read_text(encoding="utf-8"))
        tokenizer, special, preflight_rows, token_lengths, tokenizer_load_ms = _verify_preprocessor(
            root, cases, input_rows
        )
        if len(preflight_rows) != 60:
            raise RuntimeErrorRecord("PERF_INPUT_COUNT_MISMATCH", "性能測定の入力が60件ではありません。")
        phase_memory["after_tokenizer_and_fixed_input_check"] = _memory()
        core_start = time.perf_counter()
        core = ov.Core()
        core_ready_ms = (time.perf_counter() - core_start) * 1000
        phase_memory["after_openvino_core"] = _memory()
        model_read_start = time.perf_counter()
        model = core.read_model(str(xml))
        model_read_ms = (time.perf_counter() - model_read_start) * 1000
        phase_memory["after_model_read"] = _memory()
        compile_config: dict[str, Any] = {}
        cache_before: dict[str, str] = {}
        cache_dir: Path | None = None
        device_evidence: dict[str, Any] | None = None
        device_inspection_ms: float | None = None
        if backend == "cpu":
            if threads not in (1, 2) or cache_from_run:
                raise RuntimeErrorRecord("PERF_OPTIONS_INVALID", "CPUは1/2スレッドでキャッシュ再利用なしに限定します。")
            compile_config = {"INFERENCE_NUM_THREADS": threads}
            phase_memory["before_compile"] = _memory()
            compile_start = time.perf_counter()
            compiled = core.compile_model(model, "CPU", compile_config)
            compile_ms = (time.perf_counter() - compile_start) * 1000
            try:
                actual_threads = int(compiled.get_property("INFERENCE_NUM_THREADS"))
            except Exception as exc:
                raise RuntimeErrorRecord("CPU_THREAD_EVIDENCE_UNAVAILABLE", "CompiledModelから実スレッド数を取得できません。") from exc
            if actual_threads != threads:
                raise RuntimeErrorRecord("CPU_THREAD_MISMATCH", "実スレッド数が指定値と一致しません。")
            compiled_properties = {"INFERENCE_NUM_THREADS": actual_threads}
            execution_devices = ["CPU"]
            cache_mode = "not_applicable"
            cache_source = None
        else:
            if threads is not None:
                raise RuntimeErrorRecord("PERF_OPTIONS_INVALID", "NPUではCPUスレッド指定を使いません。")
            device_start = time.perf_counter()
            device = inspect_device("NPU", "3720")
            device_inspection_ms = (time.perf_counter() - device_start) * 1000
            if device["status"] != "PASS":
                raise RuntimeErrorRecord(device["code"], "NPUの実アーキテクチャを確認できません。")
            device_evidence = device
            cache_dir, cache_before, cache_mode = __import__("som_probe.runtime", fromlist=["_npu_cache_for_run"])._npu_cache_for_run(
                root, run, conversion, cache_from_run)
            cache_source = cache_from_run
            compile_config = {"NPU_COMPILER_TYPE": "PLUGIN", "CACHE_DIR": str(cache_dir)}
            phase_memory["before_compile"] = _memory()
            compile_start = time.perf_counter()
            compiled = core.compile_model(model, "NPU", compile_config)
            compile_ms = (time.perf_counter() - compile_start) * 1000
            compiled_properties: dict[str, Any] = {}
            for key in ("EXECUTION_DEVICES", "NPU_COMPILER_TYPE", "NPU_COMPILER_VERSION",
                        "RUNTIME_REQUIREMENTS", "LOADED_FROM_CACHE"):
                try:
                    compiled_properties[key] = _json_safe(compiled.get_property(key))
                except Exception as exc:
                    compiled_properties[key] = {"unavailable": True, "error_type": type(exc).__name__,
                                                "message": str(exc)}
            execution_devices = _execution_device_names(compiled_properties.get("EXECUTION_DEVICES"))
            if (not execution_devices
                    or any(not (name.upper() == "NPU" or name.upper().startswith("NPU."))
                           for name in execution_devices)
                    or not str(compiled_properties.get("NPU_COMPILER_TYPE", "")).upper().endswith("PLUGIN")):
                raise RuntimeErrorRecord("NPU_EXECUTION_EVIDENCE_MISMATCH", "NPU実行先またはPLUGIN経路を確認できません。")
            if cache_mode == "reuse" and compiled_properties.get("LOADED_FROM_CACHE") is not True:
                raise RuntimeErrorRecord("NPU_CACHE_REUSE_NOT_CONFIRMED", "OpenVINOがキャッシュ読込を確認しませんでした。")
            compiled_properties = {key: value if isinstance(value, (str, int, float, bool, list, dict)) else str(value)
                                  for key, value in compiled_properties.items()}
        phase_memory["after_compile"] = _memory()
        phase_memory["available_physical_ram_after_compile"] = _available_ram()
        first_start = time.perf_counter()
        first_result = compiled({name: _input_arrays(preflight_rows[0])[name] for name in INPUT_NAMES})
        first_inference_ms = (time.perf_counter() - first_start) * 1000
        phase_memory["after_first_inference"] = _memory()
        result_named = {port.get_any_name(): value for port, value in first_result.items()}
        first_logits, first_actions = _select_outputs(result_named)
        _row_output(preflight_rows[0], first_logits, first_actions)
        startup = {"process_age_at_entry_ms": launch_age_ms,
                   "process_start_to_first_inference_ms": _process_age_ms(),
                   "openvino_import_ms": openvino_import_ms, "tokenizer_load_ms": tokenizer_load_ms,
                   "npu_device_inspection_ms": device_inspection_ms,
                   "openvino_core_ms": core_ready_ms, "model_read_ms": model_read_ms,
                   "compile_ms": compile_ms, "first_inference_ms": first_inference_ms}
        references_by_id = {row["case_id"]: row for row in references}
        output_by_id: dict[str, dict[str, Any]] = {}
        for row in preflight_rows:
            result = compiled({name: _input_arrays(row)[name] for name in INPUT_NAMES})
            named = {port.get_any_name(): value for port, value in result.items()}
            logits, actions = _select_outputs(named)
            output_by_id[row["case_id"]] = _row_output(row, logits, actions)
        actual_outputs = [output_by_id[row["case_id"]] for row in input_rows]
        original = [references_by_id[row["case_id"]] for row in input_rows]
        output_comparison = _comparison(original, actual_outputs)
        type_comparison = _type_comparison(original, actual_outputs)
        bundles_1 = [[row] for row in preflight_rows]
        bundles_4 = _four_request_bundles(preflight_rows)
        case_pairs = {case["case_id"]: (case, saved_by_id[case["case_id"]]) for case in cases}
        raw_samples: list[dict[str, Any]] = []
        segment_samples: dict[int, dict[str, list[float]]] = {
            size: {key: [] for key in ("preprocess_ms", "model_ms", "postprocess_ms")} for size in (1, 4)
        }
        e2e_samples: dict[int, list[float]] = {1: [], 4: []}
        per_round: dict[int, dict[int, dict[str, list[float]]]] = {
            size: {round_number: {"e2e_ms": [], "preprocess_ms": [], "model_ms": [], "postprocess_ms": []}
                   for round_number in range(1, MEASUREMENT_ROUNDS + 1)}
            for size in (1, 4)
        }
        round_failures: dict[int, dict[int, int]] = {
            size: {round_number: 0 for round_number in range(1, MEASUREMENT_ROUNDS + 1)}
            for size in (1, 4)
        }
        measured_process_cpu_ms = 0.0
        failure_count = 0
        warmup_failure_count = 0
        measurement_failure_count = 0
        measurement_failures_by_size = {1: 0, 4: 0}
        failed_samples: list[dict[str, Any]] = []
        for event_size, bundles in ((1, bundles_1), (4, bundles_4)):
            for _ in range(WARMUP_EVENTS):
                bundle = bundles[_ % len(bundles)]
                try:
                    _event([case_pairs[row["case_id"]] for row in bundle], tokenizer, special,
                           special_config["mask_token"], compiled)
                except Exception as exc:
                    failure_count += 1
                    warmup_failure_count += 1
                    failed_samples.append({"phase": "warmup", "event_size": event_size,
                                           "error_type": type(exc).__name__, "message": str(exc)[:512]})
            if event_size == 4:
                phase_memory["steady_after_warmup"] = _memory()
                phase_memory["available_physical_ram_after_warmup"] = _available_ram()
            for round_index in range(MEASUREMENT_ROUNDS):
                for sample_index in range(MEASURED_EVENTS_PER_ROUND):
                    bundle_index = (round_index * MEASURED_EVENTS_PER_ROUND + sample_index) % len(bundles)
                    bundle = bundles[bundle_index]
                    wall_start = time.perf_counter()
                    cpu_start = time.process_time()
                    try:
                        segments, _ = _event([case_pairs[row["case_id"]] for row in bundle], tokenizer, special,
                                             special_config["mask_token"], compiled)
                        elapsed = (time.perf_counter() - wall_start) * 1000
                        cpu_elapsed = (time.process_time() - cpu_start) * 1000
                        measured_process_cpu_ms += cpu_elapsed
                        e2e_samples[event_size].append(elapsed)
                        per_round[event_size][round_index + 1]["e2e_ms"].append(elapsed)
                        for key in segment_samples[event_size]:
                            segment_samples[event_size][key].append(segments[key] / event_size)
                            per_round[event_size][round_index + 1][key].append(segments[key] / event_size)
                        raw_samples.append({"event_size": event_size, "round": round_index + 1,
                                            "sample": sample_index + 1, "case_ids": [row["case_id"] for row in bundle],
                                            "e2e_ms": elapsed, "process_cpu_ms": cpu_elapsed,
                                            "preprocess_total_ms": segments["preprocess_ms"],
                                            "model_total_ms": segments["model_ms"],
                                            "postprocess_total_ms": segments["postprocess_ms"]})
                    except Exception as exc:
                        failure_count += 1
                        measurement_failure_count += 1
                        measurement_failures_by_size[event_size] += 1
                        round_failures[event_size][round_index + 1] += 1
                        failed_samples.append({"phase": "measurement", "event_size": event_size,
                                               "round": round_index + 1, "sample": sample_index + 1,
                                               "error_type": type(exc).__name__, "message": str(exc)[:512]})
                        raw_samples.append({"event_size": event_size, "round": round_index + 1,
                                            "sample": sample_index + 1, "case_ids": [row["case_id"] for row in bundle],
                                            "status": "FAIL"})
        idle: dict[str, Any] = {"requested_seconds": idle_seconds, "status": "not_requested"}
        if idle_seconds:
            before_cpu = time.process_time()
            before_memory = _memory()
            before_available_ram = _available_ram()
            before_system_cpu = _system_cpu_ticks()
            wait_start = time.perf_counter()
            time.sleep(idle_seconds)
            after_system_cpu = _system_cpu_ticks()
            system_cpu_percent = None
            if before_system_cpu is not None and after_system_cpu is not None:
                idle_delta = after_system_cpu[0] - before_system_cpu[0]
                total_delta = after_system_cpu[1] - before_system_cpu[1]
                if total_delta > 0:
                    system_cpu_percent = max(0.0, min(100.0, 100.0 * (1 - idle_delta / total_delta)))
            idle = {"requested_seconds": idle_seconds,
                    "observed_seconds": time.perf_counter() - wait_start,
                    "process_cpu_seconds": time.process_time() - before_cpu,
                    "system_cpu_percent": system_cpu_percent,
                    "system_cpu_evidence": "GetSystemTimes delta across the idle interval" if system_cpu_percent is not None else "unavailable",
                    "available_physical_ram_before": before_available_ram,
                    "available_physical_ram_after": _available_ram(),
                    "memory_before": before_memory, "memory_after": _memory(), "status": "measured"}
        run.write_jsonl("performance-samples.jsonl", raw_samples)
        if failed_samples:
            run.write_json("performance-failures.json", {"count": len(failed_samples), "failures": failed_samples})
        lock = load_lock(root)
        model_disk = {name: {"bytes": asset_path(root, name).stat().st_size, "sha256": digest}
                      for name, digest in lock["artifacts"].items()}
        model_ir_disk = {"fp16": _tree_size((root / conversion["fp16_xml"]).parent),
                         "fp32": _tree_size((root / conversion["fp32_xml"]).parent)}
        run_env_path = root / ".local" / "venv-run"
        runtime_lock_sha = hashlib.sha256((root / "requirements" / "run-win-x64.lock").read_bytes()).hexdigest()
        result_record = {
            "status": "PASS" if failure_count == 0 else "FAIL",
            "purpose": "PRELIMINARY_RESOURCE_MEASUREMENT_NOT_QUALITY_OR_LOW_RESOURCE_ACCEPTANCE",
            "run_id": run.run_id, "source_run": source_run.run_id,
            "source_revision": manifest.get("git_revision"), "trial": trial,
            "environment": {"python_version": sys.version, "python_executable": sys.executable,
                            "openvino_version": ov.__version__, "run_lock_sha256": runtime_lock_sha},
            "backend": backend, "threads_requested": threads,
            "threads_actual": threads if backend == "cpu" else None,
            "compiled_model_properties": compiled_properties,
            "execution_devices": execution_devices,
            "device_evidence": device_evidence,
            "compiler_type": compiled_properties.get("NPU_COMPILER_TYPE") if backend == "npu" else "CPU",
            "model": {"precision": "FP16", "xml_sha256": conversion["fp16_xml_sha256"],
                      "bin_sha256": conversion["fp16_bin_sha256"], "source_model_sha256": conversion["source_model_sha256"]},
            "inputs": {"count": 60, "question_count": 60, "input_hash_match_count": 60,
                       "token_length_min": min(token_lengths.values()),
                       "token_length_mean": sum(token_lengths.values()) / len(token_lengths),
                       "token_length_max": max(token_lengths.values()),
                       "candidate_count_min": min(row["candidate_count"] for row in input_rows),
                       "candidate_count_mean": sum(row["candidate_count"] for row in input_rows) / len(input_rows),
                       "candidate_count_max": max(row["candidate_count"] for row in input_rows),
                       "question_type_counts": {kind: sum(row["question_type"] == kind for row in input_rows)
                                                for kind in ("choice", "score", "noul")}},
            "settings": {"profile": "s256-k8", "warmup_events_per_size": WARMUP_EVENTS,
                         "measured_events_per_round": MEASURED_EVENTS_PER_ROUND,
                         "rounds": MEASUREMENT_ROUNDS, "event_sizes": [1, 4],
                         "four_request_bundle": "15 synthetic groups; 60 fixed cases once each, balanced across Choice/Score/Noul",
                         "profiling_enabled": False, "cache_mode": cache_mode,
                         "cache_source_run": cache_source, "cache_before": cache_before,
                         "cache_after": _cache_inventory(cache_dir) if cache_dir else {},
                         "compile_configuration": compile_config},
            "startup": startup,
            "memory": {"samples": phase_memory,
                       "process_peak_since_start_after_run": _memory(),
                       "npu_memory": {"status": "not_exposed_by_measured_runtime"}},
            "latency": {str(size): {"event_e2e": _percentiles(e2e_samples[size]),
                                     "per_judgment": {key: _percentiles(values)
                                                      for key, values in segment_samples[size].items()},
                                     "rounds": {str(round_number): {
                                         "event_e2e": _percentiles(per_round[size][round_number]["e2e_ms"]),
                                         "per_judgment": {key: _percentiles(per_round[size][round_number][key])
                                                          for key in ("preprocess_ms", "model_ms", "postprocess_ms")},
                                         "failed_event_count": round_failures[size][round_number],
                                     } for round_number in range(1, MEASUREMENT_ROUNDS + 1)},
                                     "successful_event_denominator": len(e2e_samples[size]),
                                     "requested_event_denominator": MEASURED_EVENTS_PER_ROUND * MEASUREMENT_ROUNDS,
                                     "failed_event_count": measurement_failures_by_size[size]}
                        for size in (1, 4)},
            "process_cpu_ms_measured_successful_events": measured_process_cpu_ms,
            "process_cpu_ms_per_judgment": measured_process_cpu_ms / max(
                1, sum(size * len(e2e_samples[size]) for size in (1, 4))),
            "runtime_lock_sha256": runtime_lock_sha,
            "failure_count": failure_count,
            "failure_counts": {"warmup": warmup_failure_count,
                               "measurement": measurement_failure_count,
                               "measurement_by_event_size": {str(key): value for key, value in measurement_failures_by_size.items()}},
            "idle": idle,
            "disk_usage": {"locked_model_artifacts": model_disk, "converted_ir": model_ir_disk,
                           "venv_run": _tree_size(run_env_path),
                           "npu_cache": _tree_size(cache_dir) if cache_dir else {"status": "not_applicable", "bytes": 0}},
            "output_comparison_to_reference": output_comparison,
            "type_output_comparison_to_reference": type_comparison,
            "draft_label_reference": _draft_label_summary(root, references),
            "quality_status": "UNDETERMINED",
            "low_resource_status": "NOT_ASSESSED_ON_THIS_HOST",
        }
        run.write_json("performance-result.json", result_record)
        run.stage("PERFORMANCE", result_record["status"], result_file="performance-result.json",
                  source_run=source_run.run_id, backend=backend, threads=threads,
                  failure_count=failure_count, cache_mode=cache_mode)
        return result_record
    except Exception as exc:
        run.write_json("performance-diagnostic.json", {
            "backend": backend, "threads": threads, "trial": trial,
            "source_run": source_run.run_id, "exception": _exception_details(exc),
        })
        run.failure("PERFORMANCE", getattr(exc, "code", "PERFORMANCE_RUN_FAILED"), exc,
                    backend=backend, threads=threads, source_run=source_run.run_id)
        return {"status": "FAIL", "run_id": run.run_id,
                "code": getattr(exc, "code", "PERFORMANCE_RUN_FAILED"),
                "error_type": type(exc).__name__, "message": str(exc)[:512]}

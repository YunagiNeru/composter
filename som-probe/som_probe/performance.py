"""Fixed-input CPU/NPU preliminary timing for the existing FP16 IR."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

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
                     mask_token: str, sequence_length: int) -> tuple[dict[str, Any], int]:
    question_id, question = next(iter(case["questions"].items()))
    kind = "noul" if question["type"] == "bool" else question["type"]
    head = _encode(tokenizer, f"{kind} question: {question['instructions'].replace(mask_token, ' ')}")
    option_ids = [[special["mask"]] + _encode(tokenizer, " " + option.replace(mask_token, " "))
                  for option in _render_options(question)]
    if any(len(option) > 49 for option in option_ids):
        raise RuntimeErrorRecord("OPTION_TRUNCATION_REQUIRED", f"{case['case_id']}の候補が48トークン上限を超えています。")
    budget = 192 - sum(len(option) for option in option_ids)
    if budget < 16:
        per_option = max(4, (192 - 16) // max(1, len(option_ids)))
        if any(len(option) > per_option for option in option_ids):
            raise RuntimeErrorRecord("OPTION_TRUNCATION_REQUIRED", f"{case['case_id']}の候補群を固定ヘッダーに収められません。")
        budget = 192 - sum(len(option) for option in option_ids)
    if len(head) > max(8, budget):
        raise RuntimeErrorRecord("INSTRUCTION_TRUNCATION_REQUIRED", f"{case['case_id']}の質問文を固定ヘッダーに収められません。")
    ids = [special["cls"]] + head + [special["sep"]]
    markers: list[int] = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(special["sep"])
    state_ids = _encode(tokenizer, str(case["state"]).replace(mask_token, " "))
    ids.extend(state_ids)
    ids.append(special["sep"])
    if len(ids) > sequence_length:
        raise RuntimeErrorRecord("INPUT_SEQUENCE_LENGTH_EXCEEDED",
                                 f"{case['case_id']}の完全な入力が{sequence_length}トークンを超えています。")
    token_count = len(ids)
    inputs = {
        "input_ids": [ids + [special["pad"]] * (sequence_length - len(ids))],
        "attention_mask": [[1] * len(ids) + [0] * (sequence_length - len(ids))],
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
                         input_rows: list[dict[str, Any]], sequence_length: int,
                         memory_recorder: Callable[[str], None] | None = None
                         ) -> tuple[Any, dict[str, int], list[dict[str, Any]], dict[str, int], float]:
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise RuntimeErrorRecord("PERF_TOKENIZER_UNAVAILABLE", "実行用ロックのTokenizerを読み込めません。") from exc
    tokenizer_config = json.loads(asset_path(root, "reference/tokenizer/tokenizer_config.json").read_text(encoding="utf-8"))
    load_start = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(asset_path(root, "reference/tokenizer/tokenizer.json")))
    tokenizer_load_ms = (time.perf_counter() - load_start) * 1000
    if memory_recorder is not None:
        memory_recorder("after_tokenizer_load")
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
        if saved.get("input_sequence_length", 256) != sequence_length:
            raise RuntimeErrorRecord("PERF_PROFILE_MISMATCH", "保存入力と性能測定の系列長が一致しません。")
        row, token_count = _preprocess_case(case, saved, tokenizer, special,
                                            tokenizer_config["mask_token"], sequence_length)
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
    from ctypes import wintypes

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        return {"status": "unavailable", "error_code": ctypes.get_last_error()}
    raw_fields = {"WorkingSetSize": int(counters.WorkingSetSize),
                  "PeakWorkingSetSize": int(counters.PeakWorkingSetSize),
                  "PrivateUsage": int(counters.PrivateUsage),
                  "PagefileUsage": int(counters.PagefileUsage),
                  "PeakPagefileUsage": int(counters.PeakPagefileUsage)}
    return {"status": "available", "api_fields_raw": raw_fields,
            "working_set_bytes": raw_fields["WorkingSetSize"],
            "peak_working_set_bytes": raw_fields["PeakWorkingSetSize"],
            "private_bytes": raw_fields["PrivateUsage"],
            "pagefile_usage_bytes": raw_fields["PagefileUsage"],
            "peak_pagefile_usage_bytes": raw_fields["PeakPagefileUsage"],
            "semantics": {"WorkingSetSize": "current resident working set",
                          "PeakWorkingSetSize": "working-set peak since process start",
                          "PrivateUsage": "current process Commit Charge",
                          "PagefileUsage": "current process Commit Charge",
                          "PeakPagefileUsage": "process lifetime peak Commit Charge"}}


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


def _host_cpu_observation(before: tuple[int, int] | None, after: tuple[int, int] | None,
                          elapsed_seconds: float, process_cpu_seconds: float) -> dict[str, Any]:
    if before is None or after is None or elapsed_seconds <= 0 or not os.cpu_count():
        return {"status": "unavailable", "process_cpu_seconds": process_cpu_seconds}
    idle_ticks = after[0] - before[0]
    total_ticks = after[1] - before[1]
    if total_ticks <= 0:
        return {"status": "unavailable", "process_cpu_seconds": process_cpu_seconds}
    system_busy = max(0.0, min(100.0, 100.0 * (1 - idle_ticks / total_ticks)))
    process_capacity = 100.0 * process_cpu_seconds / (elapsed_seconds * os.cpu_count())
    return {"status": "measured", "system_cpu_busy_percent": system_busy,
            "process_cpu_percent_of_host_capacity": process_capacity,
            "estimated_other_host_busy_percent": max(0.0, system_busy - process_capacity),
            "logical_cpu_count": os.cpu_count(), "process_cpu_seconds": process_cpu_seconds,
            "elapsed_seconds": elapsed_seconds,
            "method": "GetSystemTimes aggregate busy minus this process CPU normalized by logical CPUs"}


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
        sequence_length = saved.get("input_sequence_length", 256)
        row, _ = _preprocess_case(case, saved, tokenizer, special, mask_token, sequence_length)
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


def run_memory_diagnostic(root: Path, source_run: Run, backend: str, threads: int | None,
                          load_method: str, cache_from_run: str | None = None) -> dict[str, Any]:
    run = Run.create(root, f"MEMORY_{backend.upper()}_{threads or 'PLUGIN'}_{load_method}",
                     source_run.profile_config)
    source_manifest = source_run.read_manifest()
    current_manifest = run.read_manifest()
    if (source_manifest.get("stages", {}).get("G2-cpu-check", {}).get("status") != "PASS"
            or source_manifest.get("stages", {}).get("G3", {}).get("status") != "PASS"):
        run.blocked("MEMORY", "G3_NOT_PASS", "CPU照合とG3実推論がPASSしたRunが必要です。")
        return {"status": "BLOCKED", "run_id": run.run_id, "code": "G3_NOT_PASS"}
    if any(source_manifest.get(key) != current_manifest.get(key)
           for key in ("git_revision", "source_tree_sha256", "input_sha256")):
        run.blocked("MEMORY", "SOURCE_FINGERPRINT_MISMATCH", "診断RunとG3 Runのコード・入力構成が一致しません。")
        return {"status": "BLOCKED", "run_id": run.run_id, "code": "SOURCE_FINGERPRINT_MISMATCH"}
    try:
        if backend not in {"cpu", "npu"} or load_method not in {"read-compile", "path-compile"}:
            raise RuntimeErrorRecord("MEMORY_OPTIONS_INVALID", "バックエンドまたはロード方式が不正です。")
        if (backend == "cpu" and (threads not in (1, 2) or cache_from_run)
                or backend == "npu" and threads is not None):
            raise RuntimeErrorRecord("MEMORY_OPTIONS_INVALID", "CPU/NPUのメモリ測定設定が不正です。")
        started = time.perf_counter()
        process_age_at_entry_ms = _process_age_ms()
        phases: dict[str, Any] = {}

        def capture(name: str) -> None:
            counters = _memory()
            if counters.get("status") != "available":
                raise RuntimeErrorRecord("MEMORY_COUNTERS_UNAVAILABLE", "Windowsプロセスメモリカウンターを取得できません。")
            phases[name] = {"elapsed_since_process_entry_ms": (time.perf_counter() - started) * 1000,
                            "counters": counters, "available_physical_ram_bytes": _available_ram()}

        capture("process_entry")
        import openvino as ov
        capture("after_openvino_library_import")
        conversion = _load_run_conversion(root, source_run)
        actual_assets = verify_assets(root)
        if actual_assets != source_manifest.get("verified_model_artifact_sha256"):
            raise RuntimeErrorRecord("MODEL_ARTIFACT_FINGERPRINT_MISMATCH", "モデル取得物のハッシュがG3 Runと一致しません。")
        xml = root / conversion["fp16_xml"]
        if (sha256_file(xml) != conversion["fp16_xml_sha256"]
                or sha256_file(xml.with_suffix(".bin")) != conversion["fp16_bin_sha256"]):
            raise RuntimeErrorRecord("CONVERTED_MODEL_HASH_MISMATCH", "固定IRのXML/BINハッシュがG3 Runと一致しません。")
        input_rows = _read_jsonl(source_run.path / "inputs.jsonl")
        references = _read_jsonl(source_run.path / "reference.jsonl")
        cases = [json.loads(line) for line in (root / "datasets" / "dev.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if len(input_rows) != 60 or len(references) != 60 or len(cases) != 60:
            raise RuntimeErrorRecord("MEMORY_INPUT_COUNT_INVALID", "固定入力、参照出力、ケースは各60件必要です。")
        special_config = json.loads(asset_path(root, "reference/tokenizer/tokenizer_config.json").read_text(encoding="utf-8"))
        tokenizer, special, prepared, lengths, tokenizer_ms = _verify_preprocessor(
            root, cases, input_rows, source_run.profile_config["sequence_length"],
            lambda name: capture(name))
        capture("after_tokenizer_and_fixed_input_validation")
        input_by_id = {row["case_id"]: row for row in input_rows}
        if any(row["input_sha256"] != input_by_id[row["case_id"]]["input_sha256"]
               for row in prepared):
            raise RuntimeErrorRecord("PERF_TOKENIZER_PARITY_FAILED", "測定入力の再生成ハッシュがG3保存入力と異なります。")
        core = ov.Core()
        capture("after_openvino_core_creation")
        model = None
        if load_method == "read-compile":
            model = core.read_model(str(xml))
            capture("after_ir_read_model")
        else:
            phases["ir_read_model"] = {"status": "not_separately_observable",
                                        "reason": "compile_modelへ固定IRパスを直接渡すため、read_model境界がない"}
        compile_config: dict[str, Any]
        cache_dir: Path | None = None
        cache_before: dict[str, str] = {}
        cache_mode = "not_applicable"
        device_evidence: dict[str, Any] | None = None
        if backend == "cpu":
            compile_config = {"INFERENCE_NUM_THREADS": threads}
            device_name = "CPU"
        else:
            device_evidence = inspect_device("NPU", "3720")
            if device_evidence.get("status") != "PASS":
                raise RuntimeErrorRecord(device_evidence.get("code", "NPU_DEVICE_BLOCKED"),
                                         "NPU実アーキテクチャ3720を確認できません。")
            cache_dir, cache_before, cache_mode = __import__(
                "som_probe.runtime", fromlist=["_npu_cache_for_run"]
            )._npu_cache_for_run(root, run, conversion, cache_from_run)
            compile_config = {"NPU_COMPILER_TYPE": "PLUGIN", "CACHE_DIR": str(cache_dir)}
            device_name = "NPU"
        capture("before_compile_or_cache_load")
        compile_input = model if model is not None else str(xml)
        compiled = core.compile_model(compile_input, device_name, compile_config)
        properties: dict[str, Any] = {}
        property_names = (("INFERENCE_NUM_THREADS",) if backend == "cpu" else
                          ("EXECUTION_DEVICES", "NPU_COMPILER_TYPE", "NPU_COMPILER_VERSION", "LOADED_FROM_CACHE"))
        for key in property_names:
            try:
                properties[key] = _json_safe(compiled.get_property(key))
            except Exception as exc:
                properties[key] = {"unavailable": True, "error_type": type(exc).__name__, "message": str(exc)}
        if backend == "cpu" and int(properties.get("INFERENCE_NUM_THREADS", -1)) != threads:
            raise RuntimeErrorRecord("CPU_THREAD_MISMATCH", "CompiledModelの実スレッド数が要求値と異なります。")
        execution_devices = _execution_device_names(properties.get("EXECUTION_DEVICES")) if backend == "npu" else ["CPU"]
        if backend == "npu":
            if (not execution_devices or any(not (name.upper() == "NPU" or name.upper().startswith("NPU."))
                                              for name in execution_devices)
                    or not str(properties.get("NPU_COMPILER_TYPE", "")).upper().endswith("PLUGIN")):
                raise RuntimeErrorRecord("NPU_EXECUTION_EVIDENCE_MISMATCH", "NPU PLUGIN実行先を確認できません。")
            if cache_mode == "reuse" and properties.get("LOADED_FROM_CACHE") is not True:
                raise RuntimeErrorRecord("NPU_CACHE_REUSE_NOT_CONFIRMED", "キャッシュ再利用を確認できません。")
        capture("after_compile_or_cache_load")
        first = prepared[0]
        first_arrays = {name: _input_arrays(first)[name] for name in INPUT_NAMES}
        capture("after_inference_input_arrays_created")
        first_result = compiled(first_arrays)
        capture("after_first_compiledmodel_call_request_creation_and_inference")
        result_named = {port.get_any_name(): value for port, value in first_result.items()}
        first_logits, first_actions = _select_outputs(result_named)
        _row_output(first, first_logits, first_actions)
        for _ in range(WARMUP_EVENTS):
            compiled(first_arrays)
        capture("after_20_warm_inferences")
        runtime_lock_sha = hashlib.sha256((root / "requirements" / "run-win-x64.lock").read_bytes()).hexdigest()
        record = {
            "status": "PASS", "purpose": "MEMORY_DIAGNOSTIC_SEPARATE_FROM_NORMAL_TIMING",
            "run_id": run.run_id, "source_run": source_run.run_id,
            "git_revision": source_manifest.get("git_revision"),
            "source_tree_sha256": source_manifest.get("source_tree_sha256"),
            "source_input_sha256": source_manifest.get("input_sha256"),
            "profile": source_run.profile_config,
            "backend": backend, "threads": threads, "load_method": load_method,
            "environment": {"python_version": sys.version, "python_executable": sys.executable,
                            "openvino_version": ov.__version__, "runtime_lock_sha256": runtime_lock_sha},
            "process_age_at_diagnostic_entry_ms": process_age_at_entry_ms,
            "model": {"precision": "FP16", "xml_sha256": conversion["fp16_xml_sha256"],
                      "bin_sha256": conversion["fp16_bin_sha256"],
                      "xml_bytes": xml.stat().st_size, "bin_bytes": xml.with_suffix(".bin").stat().st_size},
            "compile_configuration": compile_config, "compiled_model_properties": properties,
            "execution_devices": execution_devices, "device_evidence": device_evidence,
            "cache": {"mode": cache_mode, "source_run": cache_from_run,
                      "before": cache_before, "after": _cache_inventory(cache_dir) if cache_dir else {}},
            "stages": phases,
            "object_retention": {
                "openvino_core_python_objects": 1,
                "openvino_model_python_references": 1 if load_method == "read-compile" else 0,
                "compiled_model_python_references": 1,
                "compiled_model_call_count_after_warmup": WARMUP_EVENTS + 1,
                "same_compiled_model_reference_used_for_all_calls": True,
                "infer_request_explicitly_created_by_probe": 0,
                "infer_request_internal_reference": "CompiledModel.__call__は最初の呼出しで内部InferRequestを作り、後続呼出しで再利用する。OpenVINO APIは内部参照数を公開しない。",
                "loaded_case_rows": len(cases), "saved_input_rows": len(input_rows),
                "saved_reference_rows": len(references), "prepared_input_rows": len(prepared),
                "saved_input_tensor_values": sum(
                    len(value) * len(value[0]) if value and isinstance(value[0], list) else len(value)
                    for row in input_rows for value in row["inputs"].values()
                ),
                "prepared_input_tensor_values": sum(
                    len(value) * len(value[0]) if value and isinstance(value[0], list) else len(value)
                    for row in prepared for value in row["inputs"].values()
                ),
                "large_model_or_graph_copies_added_by_probe": 0,
                "lifetimes": {"read_model_object": "read-compile時はコンパイルからプロセス終了まで1参照。path-compileはPython参照なし。",
                              "compiled_model": "コンパイル完了からプロセス終了まで1参照。",
                              "infer_request": "初回CompiledModel呼出し時に内部生成され、CompiledModel内で再利用。",
                              "input_rows_and_prepared_rows": "診断Runのプロセス終了まで各60行を保持。",
                              "temporary_numpy_input_arrays": "初回要求と各ウォーム要求の呼出しスコープ内。",
                              "note": "保持数は診断関数が保持するPython側参照数。OpenVINO内部のグラフ複製数は公開APIから数えられない。"},
            },
            "input_summary": {"rows": len(prepared), "sequence_length": source_run.profile_config["sequence_length"],
                              "token_count_min": min(lengths.values()), "token_count_max": max(lengths.values()),
                              "special_tokenizer_read_count": 1},
            "memory_semantics": {
                "WorkingSetSize": "各段階の現在Working Set。Private Bytesと合算しない。",
                "PrivateUsage_and_PagefileUsage": "各段階の現在Commit Charge。ディスクへのページ書込量ではない。",
                "PeakPagefileUsage": "プロセス開始から各採取時点までの累積最大Commit Charge。段階別瞬間ピークではない。",
                "PeakWorkingSetSize": "プロセス開始から各採取時点までの累積最大Working Set。",
            },
            "inference_request_evidence": {
                "source": "OpenVINO 2026 Python CompiledModel API",
                "url": "https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.CompiledModel.html",
                "method": "CompiledModel.__call__のみを反復。create_infer_requestは追加で呼ばず、requestを重複保持しない。",
            },
        }
        run.write_json("memory-result.json", record)
        run.stage("MEMORY", "PASS", result_file="memory-result.json", source_run=source_run.run_id,
                  backend=backend, load_method=load_method, cache_mode=cache_mode)
        return record
    except Exception as exc:
        run.write_json("memory-diagnostic.json", {"source_run": source_run.run_id,
                                                   "exception": _exception_details(exc)})
        run.failure("MEMORY", getattr(exc, "code", "MEMORY_DIAGNOSTIC_FAILED"), exc,
                    source_run=source_run.run_id, backend=backend, load_method=load_method)
        return {"status": "FAIL", "run_id": run.run_id,
                "code": getattr(exc, "code", "MEMORY_DIAGNOSTIC_FAILED"),
                "error_type": type(exc).__name__, "message": str(exc)[:512]}


def run_performance(root: Path, source_run: Run, backend: str, threads: int | None,
                    trial: int, cache_from_run: str | None = None,
                    idle_seconds: int = 0, load_method: str = "read-compile") -> dict[str, Any]:
    run = Run.create(root, f"PERF_{backend.upper()}_{threads or 'PLUGIN'}_{load_method}_TRIAL_{trial}",
                     source_run.profile_config)
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
        if load_method not in {"read-compile", "path-compile"}:
            raise RuntimeErrorRecord("PERF_LOAD_METHOD_INVALID", "未対応のIRロード方式です。")
        launch_age_ms = _process_age_ms()
        openvino_import_start = time.perf_counter()
        import openvino as ov
        openvino_import_ms = (time.perf_counter() - openvino_import_start) * 1000

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
            root, cases, input_rows, source_run.profile_config["sequence_length"]
        )
        if len(preflight_rows) != 60:
            raise RuntimeErrorRecord("PERF_INPUT_COUNT_MISMATCH", "性能測定の入力が60件ではありません。")
        core_start = time.perf_counter()
        core = ov.Core()
        core_ready_ms = (time.perf_counter() - core_start) * 1000
        model: Any | None = None
        model_read_ms: float | None = None
        if load_method == "read-compile":
            model_read_start = time.perf_counter()
            model = core.read_model(str(xml))
            model_read_ms = (time.perf_counter() - model_read_start) * 1000
        compile_config: dict[str, Any] = {}
        cache_before: dict[str, str] = {}
        cache_dir: Path | None = None
        device_evidence: dict[str, Any] | None = None
        device_inspection_ms: float | None = None
        if backend == "cpu":
            if threads not in (1, 2) or cache_from_run:
                raise RuntimeErrorRecord("PERF_OPTIONS_INVALID", "CPUは1/2スレッドでキャッシュ再利用なしに限定します。")
            compile_config = {"INFERENCE_NUM_THREADS": threads}
            compile_start = time.perf_counter()
            compiled = core.compile_model(model if model is not None else str(xml), "CPU", compile_config)
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
            compile_start = time.perf_counter()
            compiled = core.compile_model(model if model is not None else str(xml), "NPU", compile_config)
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
        first_start = time.perf_counter()
        first_result = compiled({name: _input_arrays(preflight_rows[0])[name] for name in INPUT_NAMES})
        first_inference_ms = (time.perf_counter() - first_start) * 1000
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
        bundle_state_counts = [
            len({json.dumps(case_pairs[row["case_id"]][0]["state"], ensure_ascii=False, sort_keys=True)
                 for row in bundle})
            for bundle in bundles_4
        ]
        if len(bundle_state_counts) != 15 or any(count != 4 for count in bundle_state_counts):
            raise RuntimeErrorRecord("PERF_BUNDLE_STATE_REPEATED", "4判断束には異なる4状態が必要です。")
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
        host_load_by_round: dict[str, Any] = {}
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
            for round_index in range(MEASUREMENT_ROUNDS):
                host_start = _system_cpu_ticks()
                host_wall_start = time.perf_counter()
                host_process_start = time.process_time()
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
                host_load_by_round[f"event-{event_size}-round-{round_index + 1}"] = _host_cpu_observation(
                    host_start, _system_cpu_ticks(), time.perf_counter() - host_wall_start,
                    time.process_time() - host_process_start)
        idle: dict[str, Any] = {"requested_seconds": idle_seconds, "status": "not_requested"}
        if idle_seconds:
            before_cpu = time.process_time()
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
                    "status": "measured"}
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
            "settings": {"profile": source_run.profile_config["name"],
                         "sequence_length": source_run.profile_config["sequence_length"],
                         "warmup_events_per_size": WARMUP_EVENTS,
                         "measured_events_per_round": MEASURED_EVENTS_PER_ROUND,
                         "rounds": MEASUREMENT_ROUNDS, "event_sizes": [1, 4],
                         "four_request_bundle": {"groups": 15, "cases_per_group": 4,
                                                 "unique_states_per_group": bundle_state_counts,
                                                 "all_60_fixed_cases_once": True,
                                                 "balanced_question_types": {"choice": 20, "score": 20, "noul": 20}},
                         "profiling_enabled": False, "memory_diagnostics": "separate memory-run records",
                         "load_method": load_method, "cache_mode": cache_mode,
                         "cache_source_run": cache_source, "cache_before": cache_before,
                         "cache_after": _cache_inventory(cache_dir) if cache_dir else {},
                         "compile_configuration": compile_config},
            "startup": startup,
            "memory": {"status": "MEASURED_IN_SEPARATE_MEMORY_DIAGNOSTIC_RUN"},
            "host_load_by_round": host_load_by_round,
            "measurement_environment": {"other_applications_stopped": False,
                                         "scenario": "single-process basic performance sample",
                                         "host_busy_estimate_includes_system_activity": True,
                                         "concurrent_competition_trial": "NOT_RUN"},
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
            "draft_label_reference": _draft_label_summary(root, references, source_run),
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

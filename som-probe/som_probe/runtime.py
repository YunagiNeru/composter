"""CPU参照、固定形状変換、OpenVINO照合、明示NPU実行。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
import ctypes
from pathlib import Path
from typing import Any

from .assets import AssetError, asset_path, load_lock, make_execution_checkpoint, sha256_file, verify_assets
from .contracts import ContractError, Profile, profile_from_mapping, softmax
from .dataset import dataset_identity, load_dataset
from .storage import Run, open_run, safe_message


class RuntimeErrorRecord(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _score_value(probabilities: list[float]) -> tuple[float, float]:
    """Laya score: expected zero-based candidate index, returned rounded to 4 places."""
    if not probabilities or any(not math.isfinite(float(value)) or float(value) < 0 for value in probabilities):
        raise RuntimeErrorRecord("SCORE_PROBABILITIES_INVALID", "Score確率が空、非有限、または負数です。")
    total = sum(float(value) for value in probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeErrorRecord("SCORE_PROBABILITIES_UNNORMALIZED", "Score候補確率の合計が1ではありません。")
    expected = float(sum(index * float(value) for index, value in enumerate(probabilities)))
    return expected, round(expected, 4)


def _score_return_contract(root: Path) -> dict[str, Any]:
    lock = load_lock(root)
    source = lock["sources"]["laya_code"]
    return {
        "repository": source["repository"],
        "revision": source["commit"],
        "file": "laya/agent.py",
        "function": "Agent._decode_answers",
        "source_lines": "1378-1382",
        "formula": "round(sum(i * p[i] for i in range(k)), 4)",
        "scale": "zero-based candidate index 0..k-1",
        "valid_candidates": "logits[:candidate_count] only",
        "temperature": "divide each valid logit by the selected Laya temperature once before softmax",
        "normalization": "stable softmax over valid candidates; probability sum must be 1",
        "rounding": "Python round(expected_value, 4)",
    }


def _noul_return_value(true_probability: float) -> tuple[float, bool]:
    """Apply Laya's rounded Noul API field, then its evaluator threshold."""
    probability = float(true_probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeErrorRecord("NOUL_PROBABILITY_INVALID", "NoulのP(true)が確率範囲外です。")
    returned = round(probability, 4)
    return returned, returned >= 0.5


def _noul_return_contract(root: Path) -> dict[str, Any]:
    source = load_lock(root)["sources"]["laya_code"]
    return {
        "repository": source["repository"],
        "revision": source["commit"],
        "model_return_file": "laya/agent.py",
        "model_return_function": "Agent._decode_answers",
        "model_return_source_lines": "1398-1401",
        "returned_field": "noul",
        "returned_value": "round(float(P(true)), 4)",
        "candidate_order": [False, True],
        "raw_probability_field": "true_probability",
        "temperature_and_normalization": "softmax is applied to the valid False/True logits once using the configured temperature before return rounding",
        "evaluator_file": "laya/evals.py",
        "evaluator_source_lines": "421-424",
        "evaluator_boolean": "bool(returned_noul >= 0.5)",
        "rounding": "Python round(value, 4)",
        "argmax_is_recorded_separately": True,
    }


def process_age_ms() -> float | None:
    """現在プロセスの開始からの経過時間をWindowsで取得する。"""
    if os.name != "nt":
        return None
    from ctypes import wintypes

    class FileTime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]
    created, exited, kernel_time, user_time = FileTime(), FileTime(), FileTime(), FileTime()
    kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(FileTime), ctypes.POINTER(FileTime),
                                       ctypes.POINTER(FileTime), ctypes.POINTER(FileTime)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), ctypes.byref(created), ctypes.byref(exited),
                                  ctypes.byref(kernel_time), ctypes.byref(user_time)):
        return None
    ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
    created_unix = ticks / 10_000_000 - 11644473600
    return max(0.0, (time.time() - created_unix) * 1000)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def inspect_device(device: str, expected_arch: str = "3720") -> dict[str, Any]:
    try:
        import openvino as ov
        from openvino import properties as props
    except Exception as exc:
        return {"status": "BLOCKED", "code": "OPENVINO_UNAVAILABLE", "device": device,
                "error_type": type(exc).__name__, "message": safe_message(str(exc))}
    try:
        core = ov.Core()
        available = list(core.available_devices)
    except Exception as exc:
        return {"status": "BLOCKED", "code": "RUNTIME_INITIALIZATION_FAILED", "device": device,
                "openvino_version": ov.__version__, "error_type": type(exc).__name__,
                "message": safe_message(str(exc))}
    if device not in available:
        return {"status": "BLOCKED", "code": "DEVICE_NOT_RECOGNIZED", "device": device,
                "available_devices": available, "openvino_version": ov.__version__, "device_properties": {}}
    try:
        supported = list(core.get_property(device, props.supported_properties))
    except Exception as exc:
        return {"status": "BLOCKED", "code": "SUPPORTED_PROPERTIES_UNAVAILABLE", "device": device,
                "available_devices": available, "openvino_version": ov.__version__,
                "error_type": type(exc).__name__, "message": safe_message(str(exc)), "device_properties": {}}
    properties: dict[str, Any] = {}
    wanted = {
        "SUPPORTED_PROPERTIES", "DEVICE_ARCHITECTURE", "FULL_DEVICE_NAME", "OPTIMIZATION_CAPABILITIES",
        "DEVICE_TYPE", "NPU_DRIVER_VERSION", "NPU_COMPILER_VERSION", "DRIVER_VERSION",
        "RUNTIME_REQUIREMENTS", "DEVICE_UUID", "DEVICE_LUID", "DEVICE_PCI_INFO",
    }
    for key in supported:
        name = str(getattr(key, "name", key))
        if name.upper() not in wanted:
            continue
        try:
            value = core.get_property(device, key)
            if name.upper() in {"DEVICE_UUID", "DEVICE_LUID", "DEVICE_PCI_INFO"}:
                properties[name] = "[PRIVATE_DEVICE_IDENTIFIER_RECORDED_LOCALLY]"
            else:
                properties[name] = _json_safe(value)
        except Exception as exc:
            properties[name] = {"unavailable": True, "error_type": type(exc).__name__,
                                "message": safe_message(str(exc))}
    arch = properties.get("DEVICE_ARCHITECTURE")
    if device == "NPU":
        if arch is None or isinstance(arch, dict):
            status, code = "BLOCKED", "ARCHITECTURE_UNAVAILABLE"
        elif str(arch).strip() != expected_arch:
            status, code = "BLOCKED", "ARCHITECTURE_MISMATCH"
        else:
            status, code = "PASS", "NPU_ARCHITECTURE_MATCH"
    else:
        status, code = "PASS", "DEVICE_AVAILABLE"
    return {"status": status, "code": code, "device": device, "expected_architecture": expected_arch,
            "available_devices": available, "openvino_version": ov.__version__,
            "supported_properties": [str(getattr(key, "name", key)) for key in supported],
            "device_properties": properties}


def _unique_json(run: Run, stem: str, value: dict[str, Any]) -> str:
    index = 1
    while True:
        name = f"{stem}.json" if index == 1 else f"{stem}-{index}.json"
        try:
            run.write_json(name, value)
            return name
        except FileExistsError:
            index += 1


def doctor(run: Run, device: str, expected_arch: str) -> dict[str, Any]:
    result = inspect_device(device, expected_arch)
    saved_as = _unique_json(run, f"environment-{device}", result)
    status = result["status"]
    stage = "G3-doctor" if device == "NPU" else "G0-doctor"
    run.stage(stage, status, code=result["code"], environment_file=saved_as,
              openvino_version=result.get("openvino_version"), device=device,
              expected_architecture=expected_arch,
              actual_architecture=result.get("device_properties", {}).get("DEVICE_ARCHITECTURE"))
    return {"status": status, "code": result["code"], "environment_file": saved_as,
            "device": device, "openvino_version": result.get("openvino_version"),
            "actual_architecture": result.get("device_properties", {}).get("DEVICE_ARCHITECTURE")}


def _load_cases(root: Path, run: Run | None = None) -> list[dict[str, Any]]:
    version = run.dataset_version if run is not None else "dev-v1"
    profile = profile_from_mapping(run.profile_config) if run is not None else Profile()
    identity = run.dataset_identity if run is not None else None
    try:
        return load_dataset(root, version, profile, expected_identity=identity)["cases"]
    except ContractError as exc:
        raise RuntimeErrorRecord(exc.code, str(exc)) from exc


def _evaluation_key(case_id: str, question_id: str, input_sha256: str,
                    identity: dict[str, Any]) -> str:
    value = {
        "case_id": case_id,
        "question_id": question_id,
        "dataset_version": identity["dataset_version"],
        "dataset_sha256": identity["dataset_sha256"],
        "input_sha256": input_sha256,
        "scoring_contract_version": identity["scoring_contract_version"],
        "scoring_contract_sha256": identity["scoring_contract_sha256"],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _output_identity_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in (
        "dataset_version", "dataset_sha256", "scoring_contract_version",
        "scoring_contract_sha256", "evaluation_key",
    )}


def _assert_output_identity(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    keys = ("case_id", "question_id", "input_sha256", "dataset_version", "dataset_sha256",
            "scoring_contract_version", "scoring_contract_sha256", "evaluation_key")
    if any(expected.get(key) != actual.get(key) for key in keys):
        raise RuntimeErrorRecord("INPUT_IDENTITY_MISMATCH",
                                 "比較する出力のcase・データ版・入力・採点契約識別子が一致しません。")


def _checkpoint_hashes(lock: dict[str, Any]) -> dict[str, str]:
    return {
        "model.safetensors": lock["artifacts"]["reference/model.safetensors"],
        "rl_agent_config.json": lock["artifacts"]["reference/rl_agent_config.json"],
        "encoder/config.json": lock["artifacts"]["reference/encoder/config.json"],
        "tokenizer/tokenizer.json": lock["artifacts"]["reference/tokenizer/tokenizer.json"],
        "tokenizer/tokenizer_config.json": lock["artifacts"]["reference/tokenizer/tokenizer_config.json"],
    }


def _load_agent(root: Path, run: Run):
    verify_assets(root)
    lock = load_lock(root)
    source_dir = root / ".local" / "sources" / f"laya-pinned-{lock['sources']['laya_code']['commit']}"
    if not (source_dir / "laya" / "agent.py").is_file():
        raise RuntimeErrorRecord("PINNED_SOURCE_MISSING", "監査済みLayaソースがありません。prepareを実行してください。")
    source_parent = str(source_dir.resolve())
    if source_parent not in sys.path:
        sys.path.insert(0, source_parent)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        from laya.agent import Agent
    except Exception as exc:
        raise RuntimeErrorRecord("PINNED_SOURCE_IMPORT_FAILED", f"固定Layaソースを読み込めません: {type(exc).__name__}") from exc
    checkpoint = make_execution_checkpoint(root, run.run_id)
    try:
        agent = Agent(str(checkpoint), device="cpu", compile=False, fast=False,
                      expected_sha256=_checkpoint_hashes(lock))
    except Exception as exc:
        raise RuntimeErrorRecord("REFERENCE_MODEL_LOAD_FAILED", f"元モデルをCPUへ読み込めません: {type(exc).__name__}: {safe_message(str(exc))}") from exc
    if str(agent.device) != "cpu":
        raise RuntimeErrorRecord("REFERENCE_DEVICE_MISMATCH", "元モデルがCPU以外で読み込まれました。")
    if getattr(agent, "parallel_options", False):
        raise RuntimeErrorRecord("UNSUPPORTED_INPUT_LAYOUT", "固定ONNX経路にない並列候補配置がモデルで有効です。")
    return agent


def _prepare_question(question: dict[str, Any]) -> dict[str, Any]:
    if question["type"] == "bool":
        return {"type": "noul", "instructions": question["instructions"]}
    return question


def _ensure_no_question_truncation(tokenizer, question: dict[str, Any], internal: dict[str, Any],
                                   qtype: str, head_max_len: int = 192) -> None:
    from laya.common import encode_text, render_options

    mask = tokenizer.mask_token
    instructions = str(internal["ins"]).replace(mask, " ")
    head_tokens = encode_text(tokenizer, f"{qtype} question: {instructions}", add_special_tokens=False)["input_ids"]
    options = render_options(internal)
    option_tokens = []
    for option in options:
        tokens = encode_text(tokenizer, " " + option.replace(mask, " "), add_special_tokens=False)["input_ids"]
        if len(tokens) > 48:
            raise RuntimeErrorRecord("OPTION_TRUNCATION_REQUIRED", "候補文がモデルの48トークン上限を超えています。")
        option_tokens.append([tokenizer.mask_token_id] + tokens)
    option_budget = head_max_len - sum(len(item) for item in option_tokens)
    if option_budget < 16:
        per_option = max(4, (head_max_len - 16) // max(1, len(option_tokens)))
        if any(len(item) > per_option for item in option_tokens):
            raise RuntimeErrorRecord("OPTION_TRUNCATION_REQUIRED", "候補群が固定ヘッダー長へ収まりません。")
        option_budget = head_max_len - sum(len(item) for item in option_tokens)
    if len(head_tokens) > max(8, option_budget):
        raise RuntimeErrorRecord("INSTRUCTION_TRUNCATION_REQUIRED", "質問文が固定ヘッダー長へ収まりません。")


def _input_digest(inputs: dict[str, Any]) -> str:
    raw = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_input_rows(root: Path, run: Run, agent) -> list[dict[str, Any]]:
    import numpy as np
    from laya.common import QTYPES, encode_text, render_options, serialize_state

    profile = run.profile_config
    sequence_length = profile["sequence_length"]
    identity = run.dataset_identity
    special_token_ids = {
        "cls": int(agent.tok.cls_token_id), "sep": int(agent.tok.sep_token_id),
        "mask": int(agent.tok.mask_token_id), "pad": int(agent.tok.pad_token_id),
    }
    rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    input_contract_errors: list[dict[str, str]] = []
    for case in _load_cases(root, run):
        question_id, raw_question = next(iter(case["questions"].items()))
        question = _prepare_question(raw_question)
        Agent = type(agent)
        Agent._check_question(question_id, question)
        internal = {question_id: Agent._to_internal(question)}
        internal_question = internal[question_id]
        rendered_instruction = str(internal_question["ins"]).replace(agent.tok.mask_token, " ")
        instruction_tokens = encode_text(
            agent.tok, f"{internal_question['t']} question: {rendered_instruction}",
            add_special_tokens=False,
        )["input_ids"]
        rendered_candidates = render_options(internal_question)
        candidate_tokens = [
            encode_text(agent.tok, " " + option.replace(agent.tok.mask_token, " "),
                        add_special_tokens=False)["input_ids"]
            for option in rendered_candidates
        ]
        candidate_token_counts = [len(tokens) for tokens in candidate_tokens]
        serialized_state = serialize_state(case["state"]).replace(agent.tok.mask_token, " ")
        state_tokens = encode_text(agent.tok, serialized_state, add_special_tokens=False)["input_ids"]
        untruncated_ids = [int(agent.tok.cls_token_id), *instruction_tokens, int(agent.tok.sep_token_id)]
        for tokens in candidate_tokens:
            untruncated_ids.extend([int(agent.tok.mask_token_id), *tokens])
        untruncated_ids.extend([int(agent.tok.sep_token_id), *state_tokens, int(agent.tok.sep_token_id)])
        required_tokens = len(untruncated_ids)
        special_counts = {
            name: sum(token_id == special_id for token_id in untruncated_ids)
            for name, special_id in (("cls", int(agent.tok.cls_token_id)),
                                     ("sep", int(agent.tok.sep_token_id)),
                                     ("mask", int(agent.tok.mask_token_id)),
                                     ("pad", int(agent.tok.pad_token_id)))
        }
        length_row: dict[str, Any] = {
            "case_id": case["case_id"], "question_id": question_id,
            "dataset_version": identity["dataset_version"],
            "dataset_sha256": identity["dataset_sha256"],
            "scoring_contract_version": identity["scoring_contract_version"],
            "scoring_contract_sha256": identity["scoring_contract_sha256"],
            "profile": profile["name"], "profile_limit_tokens": sequence_length,
            "required_total_tokens": required_tokens,
            "instruction_tokens": len(instruction_tokens),
            "candidate_tokens_without_masks": candidate_token_counts,
            "candidate_masks": len(candidate_token_counts),
            "serialized_state_tokens": len(state_tokens),
            "special_token_counts_in_untruncated_input": special_counts,
            "fixed_profile_padding_tokens": (sequence_length - required_tokens
                                              if required_tokens <= sequence_length else None),
            "status": "ELIGIBLE" if required_tokens <= sequence_length else "EXCEEDS_PROFILE_LIMIT",
            "truncation_detected": False,
            "truncation_performed": False,
        }
        try:
            _ensure_no_question_truncation(agent.tok, question, internal_question, question["type"])
        except RuntimeErrorRecord as exc:
            length_row["status"] = "INPUT_CONTRACT_ERROR"
            length_row["error_code"] = exc.code
            input_contract_errors.append({"case_id": case["case_id"], "code": exc.code})
            length_rows.append(length_row)
            continue
        if required_tokens > sequence_length:
            length_rows.append(length_row)
            continue
        item = agent._encode_state(case["state"], [question_id], internal,
                                   max_len=sequence_length, head_max_len=192)[0]
        if item["state_stats"]["truncated"] or len(item["ids"]) != required_tokens:
            length_row["status"] = "INPUT_TRUNCATION_DETECTED"
            length_row["truncation_detected"] = True
            length_row["truncation_performed"] = bool(item["state_stats"]["truncated"])
            length_row["model_built_tokens"] = len(item["ids"])
            input_contract_errors.append({"case_id": case["case_id"], "code": "INPUT_TRUNCATION_DETECTED"})
            length_rows.append(length_row)
            continue
        if item["options"]["options_distinct"] != item["options"]["options"]:
            length_row["status"] = "CANDIDATES_COLLAPSED"
            input_contract_errors.append({"case_id": case["case_id"], "code": "CANDIDATES_COLLAPSED"})
            length_rows.append(length_row)
            continue
        if item["options"]["tokens_per_option"] is not None:
            length_row["status"] = "OPTION_TRUNCATION_REQUIRED"
            input_contract_errors.append({"case_id": case["case_id"], "code": "OPTION_TRUNCATION_REQUIRED"})
            length_rows.append(length_row)
            continue
        length_row["model_built_tokens"] = len(item["ids"])
        length_rows.append(length_row)
        expected_candidates = (len(raw_question["criteria"]) if raw_question["type"] in {"choice", "score"}
                               else 2)
        if len(item["markers"]) != expected_candidates:
            raise RuntimeErrorRecord("CANDIDATE_COUNT_MISMATCH", "候補数とマーカー数が一致しません。")
        if not 2 <= len(item["markers"]) <= 8 or len(item["ids"]) > sequence_length:
            raise RuntimeErrorRecord("FIXED_INPUT_SHAPE_INVALID", "トークン列または候補数が固定形状を超えています。")
        pad_id = special_token_ids["pad"]
        token_ids = np.full((1, sequence_length), pad_id, dtype=np.int64)
        attention = np.zeros((1, sequence_length), dtype=np.int64)
        token_ids[0, :len(item["ids"])] = np.asarray(item["ids"], dtype=np.int64)
        attention[0, :len(item["ids"])] = 1
        marker_pos = np.zeros((1, 8), dtype=np.int64)
        marker_mask = np.zeros((1, 8), dtype=np.bool_)
        marker_pos[0, :len(item["markers"])] = np.asarray(item["markers"], dtype=np.int64)
        marker_mask[0, :len(item["markers"])] = True
        qtype = int(QTYPES[question["type"]])
        inputs = {"input_ids": token_ids.tolist(), "attention_mask": attention.tolist(),
                  "marker_pos": marker_pos.tolist(), "marker_mask": marker_mask.tolist(),
                  "qtype": [qtype]}
        size = len(item["markers"])
        bucket = __import__("laya.common", fromlist=["temp_bucket"]).temp_bucket(qtype, size)
        temperature = float(agent.temperature_by_options.get(bucket, agent.temperature[qtype]))
        input_sha256 = _input_digest(inputs)
        rows.append({"case_id": case["case_id"], "question_id": question_id,
                     "dataset_version": identity["dataset_version"],
                     "dataset_sha256": identity["dataset_sha256"],
                     "scoring_contract_version": identity["scoring_contract_version"],
                     "scoring_contract_sha256": identity["scoring_contract_sha256"],
                     "evaluation_key": _evaluation_key(case["case_id"], question_id,
                                                       input_sha256, identity),
                     "source_question_type": raw_question["type"],
                     "question_type": "noul" if raw_question["type"] == "bool" else raw_question["type"],
                     "qtype": qtype,
                     "candidate_count": size,
                     "candidate_ids": list(raw_question["criteria"].keys()) if raw_question["type"] == "choice"
                     else (list(range(size)) if raw_question["type"] == "score" else [False, True]),
                     "temperature": temperature, "input_sequence_length": sequence_length,
                     "required_total_tokens": required_tokens,
                     "special_token_ids": special_token_ids,
                     "input_sha256": input_sha256, "inputs": inputs,
                     "label_status": "ai_draft_unverified"})
    run.write_jsonl("input-lengths.jsonl", length_rows)
    if input_contract_errors:
        raise RuntimeErrorRecord("INPUT_CONTRACT_INVALID",
                                 "入力前処理で候補重複または切捨て条件を検出しました。")
    if not rows:
        raise RuntimeErrorRecord("NO_PROFILE_ELIGIBLE_CASES", "プロファイル上限以内の入力ケースがありません。")
    return rows


def _compare_profile_inputs(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]],
                            baseline_profile: dict[str, Any], candidate_profile: dict[str, Any],
                            baseline_lengths: list[dict[str, Any]],
                            candidate_lengths: list[dict[str, Any]]) -> dict[str, Any]:
    if (baseline_profile.get("name") != "s256-k8" or candidate_profile.get("name") != "s128-k8"
            or len(baseline_lengths) != 60 or len(candidate_lengths) != 60):
        raise RuntimeErrorRecord("PROFILE_INPUT_COMPARISON_INVALID", "形状比較は同一60件のs256-k8とs128-k8の長さ台帳に限定します。")
    baseline_length_by_id = {row["case_id"]: row for row in baseline_lengths}
    candidate_length_by_id = {row["case_id"]: row for row in candidate_lengths}
    if len(baseline_length_by_id) != 60 or baseline_length_by_id.keys() != candidate_length_by_id.keys():
        raise RuntimeErrorRecord("PROFILE_INPUT_CASE_MISMATCH", "形状比較の60件の長さ台帳が一致しません。")
    for case_id, base in baseline_length_by_id.items():
        other = candidate_length_by_id[case_id]
        if (base.get("required_total_tokens") != other.get("required_total_tokens")
                or base.get("dataset_sha256") != other.get("dataset_sha256")
                or base.get("scoring_contract_sha256") != other.get("scoring_contract_sha256")):
            raise RuntimeErrorRecord("PROFILE_LENGTH_IDENTITY_MISMATCH", f"{case_id}の入力内容または採点契約が異なります。")
    expected_baseline_ids = {
        case_id for case_id, row in baseline_length_by_id.items()
        if row.get("required_total_tokens", 10**9) <= 256 and row.get("status") == "ELIGIBLE"
    }
    expected_candidate_ids = {
        case_id for case_id, row in candidate_length_by_id.items()
        if row.get("required_total_tokens", 10**9) <= 128 and row.get("status") == "ELIGIBLE"
    }
    baseline_by_id = {row["case_id"]: row for row in baseline_rows}
    candidate_by_id = {row["case_id"]: row for row in candidate_rows}
    if (baseline_by_id.keys() != expected_baseline_ids
            or candidate_by_id.keys() != expected_candidate_ids
            or not candidate_by_id.keys() <= baseline_by_id.keys()):
        raise RuntimeErrorRecord("PROFILE_EXECUTED_CASE_SET_MISMATCH",
                                 "形状別の実行件数が総入力長台帳の適格集合と一致しません。")
    active_lengths: list[int] = []
    for case_id in sorted(baseline_by_id):
        baseline = baseline_by_id[case_id]
        if case_id not in candidate_by_id:
            continue
        candidate = candidate_by_id[case_id]
        base_inputs, next_inputs = baseline["inputs"], candidate["inputs"]
        base_mask = base_inputs["attention_mask"][0]
        next_mask = next_inputs["attention_mask"][0]
        base_length, next_length = sum(base_mask), sum(next_mask)
        if (baseline.get("input_sequence_length") != 256 or candidate.get("input_sequence_length") != 128
                or len(base_inputs["input_ids"][0]) != 256 or len(next_inputs["input_ids"][0]) != 128
                or base_length != baseline_length_by_id[case_id].get("required_total_tokens")
                or next_length != candidate_length_by_id[case_id].get("required_total_tokens")
                or base_length != next_length or not 0 < base_length <= 128
                or base_mask != [1] * base_length + [0] * (256 - base_length)
                or next_mask != [1] * next_length + [0] * (128 - next_length)):
            raise RuntimeErrorRecord("PROFILE_INPUT_LENGTH_MISMATCH", f"{case_id}の有効長または固定Shapeが不正です。")
        if baseline.get("special_token_ids") != candidate.get("special_token_ids"):
            raise RuntimeErrorRecord("PROFILE_SPECIAL_TOKEN_MISMATCH", f"{case_id}の特殊トークンIDが異なります。")
        special = baseline["special_token_ids"]
        base_active = base_inputs["input_ids"][0][:base_length]
        next_active = next_inputs["input_ids"][0][:next_length]
        if base_active != next_active:
            raise RuntimeErrorRecord("PROFILE_ACTIVE_TOKEN_MISMATCH", f"{case_id}の有効input_idsが形状間で変わりました。")
        if (base_inputs["input_ids"][0][base_length:] != [special["pad"]] * (256 - base_length)
                or next_inputs["input_ids"][0][next_length:] != [special["pad"]] * (128 - next_length)
                or base_active[0] != special["cls"] or base_active[-1] != special["sep"]):
            raise RuntimeErrorRecord("PROFILE_PADDING_OR_SPECIAL_TOKEN_MISMATCH",
                                     f"{case_id}のpaddingまたは先頭・終端特殊トークンが不正です。")
        if (baseline["candidate_ids"] != candidate["candidate_ids"]
                or baseline["candidate_count"] != candidate["candidate_count"]
                or baseline["question_type"] != candidate["question_type"]
                or baseline["qtype"] != candidate["qtype"]
                or baseline["temperature"] != candidate["temperature"]
                or base_inputs["marker_pos"] != next_inputs["marker_pos"]
                or base_inputs["marker_mask"] != next_inputs["marker_mask"]
                or base_inputs["qtype"] != next_inputs["qtype"]):
            raise RuntimeErrorRecord("PROFILE_CANDIDATE_LAYOUT_MISMATCH",
                                     f"{case_id}の候補ID・順序・位置・型・温度が形状間で変わりました。")
        if any(base_active[position] != special["mask"]
               for position, enabled in zip(base_inputs["marker_pos"][0], base_inputs["marker_mask"][0])
               if enabled):
            raise RuntimeErrorRecord("PROFILE_MASK_MARKER_MISMATCH", f"{case_id}の候補マスク位置に[MASK]がありません。")
        active_lengths.append(base_length)
    return {
        "status": "PASS", "baseline_profile": "s256-k8", "candidate_profile": "s128-k8",
        "requested_cases": 60, "baseline_executed_cases": len(baseline_rows),
        "candidate_executed_cases": len(candidate_rows),
        "candidate_excluded_case_ids": sorted(set(baseline_length_by_id) - expected_candidate_ids),
        "candidate_over_128_case_ids": sorted(
            case_id for case_id, row in candidate_length_by_id.items()
            if row.get("required_total_tokens", 0) > 128
        ),
        "baseline_excluded_case_ids": sorted(set(baseline_length_by_id) - expected_baseline_ids),
        "cases": len(active_lengths), "same_active_input_ids": len(active_lengths),
        "same_candidate_ids_order_positions_masks_qtype_temperature": len(active_lengths),
        "same_special_token_ids_and_active_sequence_boundaries": len(active_lengths),
        "sequence_shape_before": [1, 256], "sequence_shape_after": [1, 128],
        "active_token_length_min": min(active_lengths), "active_token_length_max": max(active_lengths),
        "overlength_cases_rejected": 0,
    }


def _model_output(agent, input_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    input_names = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(input_rows), 4):
        batch = input_rows[start:start + 4]
        tensors = {name: torch.from_numpy(np.concatenate([np.asarray(row["inputs"][name]) for row in batch], axis=0))
                   for name in input_names}
        with torch.inference_mode():
            result = agent.model(tensors["input_ids"], tensors["attention_mask"], tensors["marker_pos"],
                                tensors["marker_mask"], tensors["qtype"])
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeErrorRecord("REFERENCE_OUTPUT_INVALID", "元モデルの出力構造が2系統ではありません。")
        logits, act_logits = result
        logits = logits.float().cpu().numpy()
        actions = torch.softmax(act_logits.float(), -1).cpu().numpy()
        for index, row in enumerate(batch):
            count = row["candidate_count"]
            probs = softmax(logits[index, :count].astype(float).tolist(), row["temperature"])
            decision_index = max(range(len(probs)), key=probs.__getitem__)
            decision = row["candidate_ids"][decision_index]
            score_expected, returned_score = _score_value(probs) if row["question_type"] == "score" else (None, None)
            noul_returned, noul_boolean = (
                _noul_return_value(probs[1]) if row["question_type"] == "noul" else (None, None)
            )
            output_rows.append({"case_id": row["case_id"], "question_id": row["question_id"],
                                "input_sha256": row["input_sha256"],
                                **_output_identity_fields(row),
                                "question_type": row["question_type"],
                                "candidate_ids": row["candidate_ids"], "probabilities": probs,
                                "candidate_probabilities": [
                                    {"candidate_id": candidate_id, "probability": probability}
                                    for candidate_id, probability in zip(row["candidate_ids"], probs)
                                ],
                                "decision_slot": decision_index, "decision": decision,
                                "temperature": float(row["temperature"]),
                                "probability_sum": float(sum(probs)),
                                "score_scale": list(range(count)) if row["question_type"] == "score" else None,
                                "score_expected_unrounded": score_expected,
                                "returned_score": returned_score,
                                "returned_score_rounding": "python_round_4" if row["question_type"] == "score" else None,
                                "boolean_result": noul_boolean,
                                "argmax_boolean_result": bool(decision_index) if row["question_type"] == "noul" else None,
                                "true_probability": probs[1] if row["question_type"] == "noul" else None,
                                "returned_noul": noul_returned,
                                "returned_noul_rounding": "python_round_4" if row["question_type"] == "noul" else None,
                                "action_probabilities": actions[index].astype(float).tolist()})
    return output_rows


def _input_coverage(run: Run, rows: list[dict[str, Any]]) -> dict[str, Any]:
    length_rows = _read_jsonl(run.path / "input-lengths.jsonl")
    limit = run.profile_config["sequence_length"]
    excluded = [row["case_id"] for row in length_rows if row.get("status") != "ELIGIBLE"]
    over_limit = [row["case_id"] for row in length_rows
                  if row.get("required_total_tokens", 0) > limit]
    lengths = [row["required_total_tokens"] for row in length_rows]
    return {
        "requested_cases": len(length_rows), "executed_cases": len(rows),
        "denominator": len(rows), "profile_limit_tokens": limit,
        "excluded_case_ids": excluded, "excluded_count": len(excluded),
        "over_limit_case_ids": over_limit, "over_limit_count": len(over_limit),
        "minimum_total_tokens": min(lengths, default=None),
        "maximum_total_tokens": max(lengths, default=None),
        "silent_truncation": False, "silent_exclusion": False,
        "length_measurement": "指示・全候補・マーカー/区切り特殊token・本文を含む、固定前の全系列長",
    }


def run_reference(root: Path, run: Run) -> dict[str, Any]:
    try:
        agent = _load_agent(root, run)
        rows = _build_input_rows(root, run, agent)
        profile_equivalence = None
        compare_input_run_id = run.read_manifest().get("compare_input_run_id")
        if compare_input_run_id:
            baseline_run = open_run(root, compare_input_run_id)
            baseline_manifest = baseline_run.read_manifest()
            current_manifest = run.read_manifest()
            if (baseline_run.profile_config["name"] != "s256-k8"
                    or run.profile_config["name"] != "s128-k8"
                    or baseline_manifest.get("stages", {}).get("G1", {}).get("status")
                    != "UNVERIFIED_LABEL_REFERENCE_ONLY"
                    or baseline_manifest.get("worktree_dirty") is not False
                    or current_manifest.get("worktree_dirty") is not False
                    or any(baseline_manifest.get(key) != current_manifest.get(key)
                           for key in ("git_revision", "source_tree_sha256", "input_sha256"))):
                raise RuntimeErrorRecord("PROFILE_INPUT_BASELINE_MISMATCH",
                                         "入力比較元とコード・設定・データ・依存ロックが一致しません。")
            baseline_rows = _read_jsonl(baseline_run.path / "inputs.jsonl")
            baseline_lengths = _read_jsonl(baseline_run.path / "input-lengths.jsonl")
            candidate_lengths = _read_jsonl(run.path / "input-lengths.jsonl")
            profile_equivalence = _compare_profile_inputs(
                baseline_rows, rows, baseline_run.profile_config, run.profile_config,
                baseline_lengths, candidate_lengths,
            )
        reference = _model_output(agent, rows)
        coverage = _input_coverage(run, rows)
        run.write_jsonl("inputs.jsonl", rows)
        run.write_jsonl("reference.jsonl", reference)
        result = {"status": "UNVERIFIED_LABEL_REFERENCE_ONLY", "cases": len(rows),
                  "input_coverage": coverage,
                  "label_state": "ai_draft_unverified", "model_device": str(agent.device),
                  "input_hashes_recorded": True,
                  "score_return_contract": _score_return_contract(root),
                  "noul_return_contract": _noul_return_contract(root),
                  "profile_input_equivalence": profile_equivalence,
                  "question_type_counts": {
                      kind: sum(row["question_type"] == kind for row in rows)
                      for kind in ("choice", "score", "noul")
                  }}
        result["draft_label_reference"] = _draft_label_summary(root, reference, run)
        run.write_json("draft-label-reference.json", result["draft_label_reference"])
        run.stage("G1", result["status"], **{k: v for k, v in result.items() if k != "status"})
        return result
    except Exception as exc:
        code = getattr(exc, "code", "REFERENCE_FAILED")
        run.failure("G1", code, exc)
        return {"status": "FAIL", "code": code}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeErrorRecord("REFERENCE_REQUIRED", f"runに必要な{path.name}がありません。")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _input_arrays(row: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    expected = {"input_ids": np.int64, "attention_mask": np.int64, "marker_pos": np.int64,
                "marker_mask": np.bool_, "qtype": np.int64}
    if set(row["inputs"]) != set(expected):
        raise RuntimeErrorRecord("INPUT_CONTRACT_MISMATCH", "保存済み入力の項目が固定契約と一致しません。")
    arrays = {name: np.asarray(row["inputs"][name], dtype=dtype) for name, dtype in expected.items()}
    sequence_length = row.get("input_sequence_length", 256)
    if sequence_length not in (128, 256):
        raise RuntimeErrorRecord("INPUT_PROFILE_INVALID", "入力の系列長が固定プロファイルではありません。")
    shapes = {"input_ids": (1, sequence_length), "attention_mask": (1, sequence_length), "marker_pos": (1, 8),
              "marker_mask": (1, 8), "qtype": (1,)}
    if any(arrays[name].shape != shape for name, shape in shapes.items()):
        raise RuntimeErrorRecord("INPUT_SHAPE_MISMATCH", "保存済み入力が固定Shapeと一致しません。")
    if _input_digest({key: value.tolist() for key, value in arrays.items()}) != row["input_sha256"]:
        raise RuntimeErrorRecord("INPUT_HASH_MISMATCH", "保存済み入力のハッシュが一致しません。")
    return arrays


def _select_outputs(named_arrays: dict[str, Any]) -> tuple[Any, Any]:
    import numpy as np

    logits = []
    actions = []
    for name, value in named_arrays.items():
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[0] == 1 and array.shape[1] == 8:
            logits.append((name, array))
        elif array.ndim == 2 and array.shape[0] == 1 and array.shape[1] != 8:
            actions.append((name, array))
    if len(logits) != 1 or len(actions) != 1:
        raise RuntimeErrorRecord("MODEL_OUTPUT_SHAPE_INVALID", "モデル出力から8候補logitsと判断logitsを一意に特定できません。")
    return logits[0][1][0], actions[0][1][0]


def _row_output(row: dict[str, Any], logits: Any, actions: Any) -> dict[str, Any]:
    values = [float(value) for value in logits[:row["candidate_count"]]]
    probs = softmax(values, float(row["temperature"]))
    decision_index = max(range(len(probs)), key=probs.__getitem__)
    decision = row["candidate_ids"][decision_index]
    score_expected, returned_score = _score_value(probs) if row["question_type"] == "score" else (None, None)
    noul_returned, noul_boolean = (
        _noul_return_value(probs[1]) if row["question_type"] == "noul" else (None, None)
    )
    return {"case_id": row["case_id"], "question_id": row["question_id"],
            "input_sha256": row["input_sha256"], **_output_identity_fields(row),
            "probabilities": probs,
            "question_type": row["question_type"], "candidate_ids": row["candidate_ids"],
            "candidate_probabilities": [
                {"candidate_id": candidate_id, "probability": probability}
                for candidate_id, probability in zip(row["candidate_ids"], probs)
            ],
            "decision_slot": decision_index, "decision": decision,
            "temperature": float(row["temperature"]),
            "probability_sum": float(sum(probs)),
            "score_scale": list(range(row["candidate_count"])) if row["question_type"] == "score" else None,
            "score_expected_unrounded": score_expected,
            "returned_score": returned_score,
            "returned_score_rounding": "python_round_4" if row["question_type"] == "score" else None,
            "boolean_result": noul_boolean,
            "argmax_boolean_result": bool(decision_index) if row["question_type"] == "noul" else None,
            "true_probability": probs[1] if row["question_type"] == "noul" else None,
            "returned_noul": noul_returned,
            "returned_noul_rounding": "python_round_4" if row["question_type"] == "noul" else None,
            "action_probabilities": softmax([float(value) for value in actions])}


def _identity_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _type_comparison(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if len(reference) != len(candidate):
        raise RuntimeErrorRecord("OUTPUT_COUNT_MISMATCH", "型別比較の出力件数が一致しません。")
    metrics: dict[str, dict[str, Any]] = {}
    for expected, actual in zip(reference, candidate):
        kind = expected.get("question_type")
        if kind not in {"choice", "score", "noul"} or actual.get("question_type") != kind:
            raise RuntimeErrorRecord("QUESTION_TYPE_MISMATCH", "比較する質問型が一致しません。")
        _assert_output_identity(expected, actual)
        expected_ids = expected.get("candidate_ids")
        actual_ids = actual.get("candidate_ids")
        if not isinstance(expected_ids, list) or not isinstance(actual_ids, list):
            raise RuntimeErrorRecord("CANDIDATE_ID_MISSING", "候補IDが出力にありません。")
        if len(expected_ids) != len(expected["probabilities"]) or len(actual_ids) != len(actual["probabilities"]):
            raise RuntimeErrorRecord("CANDIDATE_PROBABILITY_WIDTH_MISMATCH", "候補IDと候補確率の数が一致しません。")
        expected_map = {_identity_key(key): value for key, value in zip(expected_ids, expected["probabilities"])}
        actual_map = {_identity_key(key): value for key, value in zip(actual_ids, actual["probabilities"])}
        if len(expected_map) != len(expected_ids) or expected_map.keys() != actual_map.keys():
            raise RuntimeErrorRecord("CANDIDATE_ID_MISMATCH", "候補ID集合が比較する出力間で一致しません。")
        item = metrics.setdefault(kind, {
            "cases": 0, "candidate_probability_value_count": 0, "candidate_probability_abs_differences": [],
            "action_probability_value_count": 0, "action_probability_abs_differences": [],
            "decision_id_changes": [], "decision_slot_changes": 0,
            "decision_id_matches": 0, "score_abs_differences": [], "score_expected_abs_differences": [],
            "true_probability_abs_differences": [], "returned_noul_abs_differences": [],
        })
        item["cases"] += 1
        item["candidate_probability_value_count"] += len(expected_map)
        item["candidate_probability_abs_differences"].extend(
            abs(float(expected_map[key]) - float(actual_map[key])) for key in expected_map
        )
        expected_actions = expected["action_probabilities"]
        actual_actions = actual["action_probabilities"]
        if len(expected_actions) != len(actual_actions):
            raise RuntimeErrorRecord("ACTION_OUTPUT_WIDTH_MISMATCH", "判断確率の幅が比較する出力間で一致しません。")
        item["action_probability_value_count"] += len(expected_actions)
        item["action_probability_abs_differences"].extend(
            abs(float(a) - float(b)) for a, b in zip(expected_actions, actual_actions)
        )
        if expected.get("decision_slot") == actual.get("decision_slot"):
            item["decision_slot_matches"] = item.get("decision_slot_matches", 0) + 1
        else:
            item["decision_slot_changes"] += 1
        if _identity_key(expected.get("decision")) == _identity_key(actual.get("decision")):
            item["decision_id_matches"] += 1
        else:
            item["decision_id_changes"].append(expected["case_id"])
        if kind == "score":
            for output in (expected, actual):
                raw_expected, rounded_expected = _score_value(output["probabilities"])
                expected_scale = list(range(len(output["candidate_ids"])))
                if (output.get("score_scale") != expected_scale
                        or output["candidate_ids"] != expected_scale
                        or not math.isclose(float(output.get("score_expected_unrounded")), raw_expected,
                                            rel_tol=0.0, abs_tol=1e-12)
                        or float(output.get("returned_score")) != rounded_expected
                        or output.get("returned_score_rounding") != "python_round_4"
                        or not math.isclose(float(output.get("probability_sum")), 1.0,
                                            rel_tol=0.0, abs_tol=1e-6)):
                    raise RuntimeErrorRecord("SCORE_OUTPUT_CONTRACT_MISMATCH",
                                             "Scoreの尺度、期待値、丸め、または確率正規化がLaya契約と一致しません。")
            item["score_abs_differences"].append(
                abs(float(expected["returned_score"]) - float(actual["returned_score"]))
            )
            item["score_expected_abs_differences"].append(
                abs(float(expected["score_expected_unrounded"])
                    - float(actual["score_expected_unrounded"]))
            )
        elif kind == "noul":
            for output in (expected, actual):
                probabilities = output["probabilities"]
                if output["candidate_ids"] != [False, True] or len(probabilities) != 2:
                    raise RuntimeErrorRecord("NOUL_CANDIDATE_ORDER_MISMATCH",
                                             "Noul候補は[False, True]の順でなければなりません。")
                raw_probability = float(output.get("true_probability"))
                returned_probability, evaluator_boolean = _noul_return_value(raw_probability)
                if (not math.isclose(raw_probability, float(probabilities[1]), rel_tol=0.0, abs_tol=1e-12)
                        or float(output.get("returned_noul")) != returned_probability
                        or output.get("returned_noul_rounding") != "python_round_4"
                        or output.get("boolean_result") is not evaluator_boolean
                        or output.get("argmax_boolean_result") is not (output.get("decision_slot") == 1)
                        or output.get("decision") is not output.get("argmax_boolean_result")):
                    raise RuntimeErrorRecord("NOUL_OUTPUT_CONTRACT_MISMATCH",
                                             "Noulの生確率、返却丸め値、評価器判定、argmaxが一致しません。")
            item["true_probability_abs_differences"].append(
                abs(float(expected["true_probability"]) - float(actual["true_probability"]))
            )
            item["returned_noul_abs_differences"].append(
                abs(float(expected["returned_noul"]) - float(actual["returned_noul"]))
            )
            item["boolean_result_matches"] = item.get("boolean_result_matches", 0) + int(
                _identity_key(expected.get("boolean_result")) == _identity_key(actual.get("boolean_result"))
            )
            item["argmax_boolean_result_matches"] = item.get("argmax_boolean_result_matches", 0) + int(
                _identity_key(expected.get("argmax_boolean_result"))
                == _identity_key(actual.get("argmax_boolean_result"))
            )
    result: dict[str, Any] = {"question_type_counts": {}, "by_question_type": {}}
    for kind in ("choice", "score", "noul"):
        item = metrics.get(kind, {
            "cases": 0, "candidate_probability_value_count": 0, "candidate_probability_abs_differences": [],
            "action_probability_value_count": 0, "action_probability_abs_differences": [],
            "decision_id_changes": [], "decision_slot_changes": 0,
            "decision_id_matches": 0, "decision_slot_matches": 0,
            "score_abs_differences": [], "score_expected_abs_differences": [],
            "true_probability_abs_differences": [], "returned_noul_abs_differences": [],
        })
        probs = item.pop("candidate_probability_abs_differences")
        actions = item.pop("action_probability_abs_differences")
        scores = item.pop("score_abs_differences")
        expected_scores = item.pop("score_expected_abs_differences")
        true_probs = item.pop("true_probability_abs_differences")
        noul_returns = item.pop("returned_noul_abs_differences")
        count = item["cases"]
        result["question_type_counts"][kind] = count
        item.update({
            "decision_id_match_ratio": item["decision_id_matches"] / max(1, count),
            "decision_candidate_id_match_ratio": item["decision_id_matches"] / max(1, count),
            "decision_slot_match_ratio": item.get("decision_slot_matches", 0) / max(1, count),
            "candidate_probability_mean_abs_difference": float(np.mean(probs)) if probs else 0.0,
            "candidate_probability_max_abs_difference": float(max(probs, default=0.0)),
            "action_probability_mean_abs_difference": float(np.mean(actions)) if actions else 0.0,
            "action_probability_max_abs_difference": float(max(actions, default=0.0)),
            "candidate_probability_mean_denominator": len(probs),
            "action_probability_mean_denominator": len(actions),
        })
        if kind == "score":
            item["returned_score_value_count"] = len(scores)
            item["returned_score_mean_abs_difference"] = float(np.mean(scores)) if scores else 0.0
            item["returned_score_max_abs_difference"] = float(max(scores, default=0.0))
            item["returned_score_mean_denominator"] = len(scores)
            item["score_expected_unrounded_value_count"] = len(expected_scores)
            item["score_expected_unrounded_mean_abs_difference"] = (
                float(np.mean(expected_scores)) if expected_scores else 0.0
            )
            item["score_expected_unrounded_max_abs_difference"] = max(expected_scores, default=0.0)
            item["score_expected_unrounded_mean_denominator"] = len(expected_scores)
        if kind == "noul":
            item["true_probability_mean_abs_difference"] = float(np.mean(true_probs)) if true_probs else 0.0
            item["true_probability_mean_denominator"] = len(true_probs)
            item["returned_noul_value_count"] = len(noul_returns)
            item["returned_noul_mean_abs_difference"] = float(np.mean(noul_returns)) if noul_returns else 0.0
            item["returned_noul_max_abs_difference"] = float(max(noul_returns, default=0.0))
            item["returned_noul_mean_denominator"] = len(noul_returns)
            item["boolean_result_changes"] = count - item.get("boolean_result_matches", 0)
            item["boolean_result_match_ratio"] = item.get("boolean_result_matches", 0) / max(1, count)
            item["argmax_boolean_result_changes"] = count - item.get("argmax_boolean_result_matches", 0)
            item["argmax_boolean_result_match_ratio"] = item.get("argmax_boolean_result_matches", 0) / max(1, count)
        result["by_question_type"][kind] = item
    return result


def _output_contract_errors(output: dict[str, Any], case: dict[str, Any], question_id: str,
                            identity: dict[str, Any]) -> list[str]:
    question = case["questions"][question_id]
    kind = "noul" if question["type"] == "bool" else question["type"]
    errors: list[str] = []
    input_hash = output.get("input_sha256")
    if (not isinstance(input_hash, str) or len(input_hash) != 64
            or any(char not in "0123456789abcdef" for char in input_hash)):
        errors.append("INPUT_HASH_INVALID")
    else:
        if (output.get("dataset_version") != identity["dataset_version"]
                or output.get("dataset_sha256") != identity["dataset_sha256"]
                or output.get("scoring_contract_version") != identity["scoring_contract_version"]
                or output.get("scoring_contract_sha256") != identity["scoring_contract_sha256"]
                or output.get("evaluation_key") != _evaluation_key(
                    case["case_id"], question_id, input_hash, identity)):
            errors.append("EVALUATION_FINGERPRINT_MISMATCH")
    if output.get("case_id") != case["case_id"] or output.get("question_id") != question_id:
        errors.append("CASE_OR_QUESTION_ID_MISMATCH")
    if output.get("question_type") != kind:
        errors.append("QUESTION_TYPE_MISMATCH")

    if kind == "choice":
        candidate_ids = list(question["criteria"].keys())
    elif kind == "score":
        candidate_ids = list(range(len(question["criteria"])))
    else:
        candidate_ids = [False, True]
    actual_ids = output.get("candidate_ids")
    if (not isinstance(actual_ids, list)
            or _identity_key(actual_ids) != _identity_key(candidate_ids)):
        errors.append("UNKNOWN_OR_REORDERED_CANDIDATE_IDS")

    probabilities = output.get("probabilities")
    valid_probabilities = (
        isinstance(probabilities, list) and len(probabilities) == len(candidate_ids)
        and all(type(value) in (int, float) and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0 for value in probabilities)
    )
    if not valid_probabilities:
        errors.append("PROBABILITY_SHAPE_OR_VALUE_INVALID")
    else:
        probability_sum = sum(float(value) for value in probabilities)
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
            errors.append("PROBABILITY_SUM_INVALID")
        if (type(output.get("probability_sum")) not in (int, float)
                or not math.isfinite(float(output["probability_sum"]))
                or not math.isclose(float(output["probability_sum"]), probability_sum,
                                    rel_tol=0.0, abs_tol=1e-12)):
            errors.append("PROBABILITY_SUM_FIELD_MISMATCH")
        candidate_probabilities = output.get("candidate_probabilities")
        if (not isinstance(candidate_probabilities, list)
                or len(candidate_probabilities) != len(candidate_ids)
                or any(not isinstance(item, dict)
                       or _identity_key(item.get("candidate_id")) != _identity_key(candidate_id)
                       or type(item.get("probability")) not in (int, float)
                       or not math.isfinite(float(item["probability"]))
                       or float(item["probability"]) != float(probabilities[index])
                       for index, (item, candidate_id) in enumerate(zip(candidate_probabilities, candidate_ids)))):
            errors.append("CANDIDATE_PROBABILITY_OUTPUT_MISMATCH")
    slot = output.get("decision_slot")
    if type(slot) is not int or not 0 <= slot < len(candidate_ids):
        errors.append("DECISION_SLOT_INVALID")
    elif (not isinstance(actual_ids, list) or slot >= len(actual_ids)
          or _identity_key(output.get("decision")) != _identity_key(actual_ids[slot])):
        errors.append("DECISION_CANDIDATE_MISMATCH")

    action_probabilities = output.get("action_probabilities")
    if (not isinstance(action_probabilities, list) or not action_probabilities
            or any(type(value) not in (int, float) or not math.isfinite(float(value))
                   or float(value) < 0.0 for value in action_probabilities)
            or not math.isclose(sum(map(float, action_probabilities)), 1.0,
                                rel_tol=0.0, abs_tol=1e-6)):
        errors.append("ACTION_PROBABILITY_OUTPUT_INVALID")

    if kind == "score" and valid_probabilities:
        try:
            raw_score, returned_score = _score_value(list(map(float, probabilities)))
            if (output.get("score_scale") != candidate_ids
                    or type(output.get("score_expected_unrounded")) not in (int, float)
                    or not math.isclose(float(output["score_expected_unrounded"]), raw_score,
                                        rel_tol=0.0, abs_tol=1e-12)
                    or type(output.get("returned_score")) not in (int, float)
                    or float(output["returned_score"]) != returned_score
                    or output.get("returned_score_rounding") != "python_round_4"):
                errors.append("SCORE_API_CONTRACT_MISMATCH")
        except (RuntimeErrorRecord, TypeError, ValueError, OverflowError):
            errors.append("SCORE_API_CONTRACT_MISMATCH")
    elif kind == "noul" and valid_probabilities:
        try:
            true_probability = float(probabilities[1])
            returned, evaluator_result = _noul_return_value(true_probability)
            if (type(output.get("true_probability")) not in (int, float)
                    or float(output["true_probability"]) != true_probability
                    or type(output.get("returned_noul")) not in (int, float)
                    or float(output["returned_noul"]) != returned
                    or output.get("returned_noul_rounding") != "python_round_4"
                    or type(output.get("boolean_result")) is not bool
                    or output["boolean_result"] is not evaluator_result
                    or type(output.get("argmax_boolean_result")) is not bool
                    or output["argmax_boolean_result"] is not (slot == 1)
                    or output.get("decision") is not output.get("argmax_boolean_result")):
                errors.append("NOUL_API_OR_EVALUATOR_CONTRACT_MISMATCH")
        except (RuntimeErrorRecord, TypeError, ValueError, OverflowError):
            errors.append("NOUL_API_OR_EVALUATOR_CONTRACT_MISMATCH")
    return errors


def _draft_label_summary(root: Path, outputs: list[dict[str, Any]],
                         run: Run | None = None) -> dict[str, Any]:
    cases_list = _load_cases(root, run)
    cases = {case["case_id"]: case for case in cases_list}
    identity = run.dataset_identity if run is not None else dataset_identity(root, "dev-v1")
    question_types = ("choice", "score", "noul")
    labels_by_type: dict[str, list[tuple[dict[str, Any], str, dict[str, Any]]]] = {
        kind: [] for kind in question_types
    }
    for case in cases_list:
        question_id, question = next(iter(case["questions"].items()))
        kind = "noul" if question["type"] == "bool" else question["type"]
        labels_by_type[kind].append((case, question_id, case["expected"][question_id]))

    majority: dict[str, Any] = {}
    for kind, entries in labels_by_type.items():
        counts: dict[str, int] = {}
        values: dict[str, Any] = {}
        for _, _, label in entries:
            for answer in label["allowed_answers"]:
                key = _identity_key(answer)
                counts[key] = counts.get(key, 0) + 1
                values[key] = answer
        first_question = entries[0][0]["questions"][entries[0][1]]
        if kind == "choice":
            order = list(first_question["criteria"].keys())
        elif kind == "score":
            order = list(range(len(first_question["criteria"])))
        else:
            order = [False, True]
        tie_order = {_identity_key(value): index for index, value in enumerate(order)}
        selected_key = min(counts, key=lambda key: (-counts[key], tie_order.get(key, len(order))))
        majority[kind] = {"answer": values[selected_key], "label_count": counts[selected_key],
                           "tie_break": "候補順", "derived_from_all_dev_labels": len(entries)}

    metrics: dict[str, dict[str, Any]] = {}
    output_contract_errors: list[dict[str, Any]] = []
    case_metrics: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for output in outputs:
        case = cases.get(output.get("case_id"))
        if case is None or output.get("question_id") not in case["questions"]:
            errors = ["UNKNOWN_CASE_OR_QUESTION"]
            output_contract_errors.append({"case_id": output.get("case_id"), "errors": errors})
            case_metrics.append({"case_id": output.get("case_id"), "evaluation_key": output.get("evaluation_key"),
                                 "valid_output": False, "output_contract_errors": errors})
            continue
        question_id = output["question_id"]
        question = case["questions"][question_id]
        label = case["expected"][question_id]
        kind = "noul" if question["type"] == "bool" else question["type"]
        errors = _output_contract_errors(output, case, question_id, identity)
        if output.get("evaluation_key") in seen_keys:
            errors.append("DUPLICATE_EVALUATION_KEY")
        if isinstance(output.get("evaluation_key"), str):
            seen_keys.add(output["evaluation_key"])
        if errors:
            output_contract_errors.append({"case_id": case["case_id"], "errors": sorted(set(errors))})
            case_metrics.append({"case_id": case["case_id"], "evaluation_key": output.get("evaluation_key"),
                                 "valid_output": False, "output_contract_errors": sorted(set(errors))})
            continue

        judged = output["boolean_result"] if kind == "noul" else output["decision"]
        semantic_match = any(_identity_key(judged) == _identity_key(answer)
                             for answer in label["allowed_answers"])
        item = metrics.setdefault(kind, {
            "requested_cases": len(labels_by_type[kind]), "valid_executed_cases": 0,
            "draft_label_matches": 0, "majority_baseline_matches": 0,
        })
        item["valid_executed_cases"] += 1
        item["draft_label_matches"] += int(semantic_match)
        baseline_value = majority[kind]["answer"]
        item["majority_baseline_matches"] += int(any(
            _identity_key(baseline_value) == _identity_key(answer)
            for answer in label["allowed_answers"]
        ))
        case_metric: dict[str, Any] = {
            "case_id": case["case_id"], "evaluation_key": output["evaluation_key"],
            "valid_output": True, "draft_label_match": semantic_match,
            "majority_baseline_answer": baseline_value,
        }
        if kind == "choice":
            item["candidate_id_matches"] = item.get("candidate_id_matches", 0) + int(semantic_match)
            case_metric.update({"decision_slot": output["decision_slot"],
                                "candidate_id": output["decision"]})
        elif kind == "score":
            stage_match = any(output["decision_slot"] == answer for answer in label["allowed_answers"])
            distance = min(abs(float(output["returned_score"]) - float(answer))
                           for answer in label["allowed_answers"])
            item["stage_label_matches"] = item.get("stage_label_matches", 0) + int(stage_match)
            item.setdefault("continuous_abs_errors", []).append(distance)
            item.setdefault("majority_continuous_abs_errors", []).append(
                abs(float(output["returned_score"]) - float(baseline_value))
            )
            case_metric.update({"stage_label_match": stage_match,
                                "decision_slot": output["decision_slot"],
                                "returned_score": output["returned_score"],
                                "nearest_allowed_stage_abs_error": distance})
        else:
            returned_expected, expected_boolean = _noul_return_value(output["true_probability"])
            item["api_compatible_cases"] = item.get("api_compatible_cases", 0) + int(
                output["returned_noul"] == returned_expected
            )
            item["evaluator_boolean_compatible_cases"] = item.get(
                "evaluator_boolean_compatible_cases", 0
            ) + int(output["boolean_result"] is expected_boolean)
            item["argmax_boolean_result_matches"] = item.get("argmax_boolean_result_matches", 0) + int(
                output["argmax_boolean_result"] is (output["decision_slot"] == 1)
            )
            item["real_notification_operations_tested"] = False
            case_metric.update({"notification_candidate_boolean": output["boolean_result"],
                                "true_probability": output["true_probability"],
                                "returned_noul": output["returned_noul"],
                                "api_compatible": output["returned_noul"] == returned_expected,
                                "argmax_boolean_result": output["argmax_boolean_result"]})
        case_metrics.append(case_metric)

    for kind in question_types:
        item = metrics.setdefault(kind, {
            "requested_cases": len(labels_by_type[kind]), "valid_executed_cases": 0,
            "draft_label_matches": 0, "majority_baseline_matches": 0,
        })
        denominator = item["valid_executed_cases"]
        item["draft_label_match_ratio"] = item["draft_label_matches"] / max(1, denominator)
        item["majority_baseline_match_ratio"] = item["majority_baseline_matches"] / max(1, denominator)
        item["majority_baseline"] = majority[kind] | {
            "evaluated_on_same_dev_cases": True,
            "independent_predeclared_baseline": False,
        }
        if kind == "score":
            errors = item.pop("continuous_abs_errors", [])
            baseline_errors = item.pop("majority_continuous_abs_errors", [])
            item["stage_label_match_ratio"] = item.get("stage_label_matches", 0) / max(1, denominator)
            item["continuous_score_error"] = {
                "absolute_error_sum": sum(errors), "mean_absolute_error": sum(errors) / max(1, len(errors)),
                "denominator": len(errors), "error_definition": "abs(returned_score - nearest allowed stage value)",
            }
            item["majority_baseline_continuous_error"] = {
                "absolute_error_sum": sum(baseline_errors),
                "mean_absolute_error": sum(baseline_errors) / max(1, len(baseline_errors)),
                "denominator": len(baseline_errors),
                "error_definition": "abs(majority stage value - returned_score)",
            }
        elif kind == "noul":
            item["api_compatible_ratio"] = item.get("api_compatible_cases", 0) / max(1, denominator)
            item["evaluator_boolean_compatible_ratio"] = (
                item.get("evaluator_boolean_compatible_cases", 0) / max(1, denominator)
            )
            item["argmax_boolean_result_match_ratio"] = (
                item.get("argmax_boolean_result_matches", 0) / max(1, denominator)
            )
            item["notification_operation_status"] = "NOT_TESTED"

    valid_count = sum(item["valid_executed_cases"] for item in metrics.values())
    return {
        "status": "UNVERIFIED_DRAFT_LABELS_REFERENCE_ONLY",
        "label_verification_status": "ai_draft_unverified",
        "official_japanese_quality_status": "UNDETERMINED",
        "dataset_version": identity["dataset_version"],
        "dataset_sha256": identity["dataset_sha256"],
        "scoring_contract_version": identity["scoring_contract_version"],
        "scoring_contract_sha256": identity["scoring_contract_sha256"],
        "requested_cases": len(cases_list), "executed_output_cases": len(outputs),
        "valid_output_cases": valid_count, "invalid_output_cases": len(output_contract_errors),
        "output_contract_errors": output_contract_errors,
        "by_question_type": metrics,
        "case_metrics": case_metrics,
        "majority_baseline_method": "全60件の同一ドラフトラベルから型別の最多許容回答を算出。候補順で同数を解消し、同じ開発集合で比較するインサンプル参考値。",
        "note": "draft label一致はAI未確認ラベルとの参考一致。人の正解に対する誤り数ではなく、正式品質は未判定。Noulは通知候補の分類だけを評価し、通知操作は未実施。",
    }


def _onnx_cpu(root: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import onnxruntime as ort

    model_path = asset_path(root, "onnx/model.onnx")
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = min(8, os.cpu_count() or 1)
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeErrorRecord("ONNX_PROVIDER_MISMATCH", "元ONNX参照の実行先がCPU単独ではありません。")
    inputs = {item.name for item in session.get_inputs()}
    required = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}
    if inputs != required:
        raise RuntimeErrorRecord("ONNX_INPUT_CONTRACT_MISMATCH", "元ONNXの入力名が固定入力契約と異なります。")
    outputs: list[dict[str, Any]] = []
    for row in rows:
        arrays = _input_arrays(row)
        named = dict(zip((item.name for item in session.get_outputs()),
                          session.run(None, {key: arrays[key] for key in required})))
        logits, actions = _select_outputs(named)
        outputs.append(_row_output(row, logits, actions))
    return outputs, {"providers": providers, "onnxruntime_version": ort.__version__,
                     "model_sha256": sha256_file(model_path)}


def _static_shape(model: Any, sequence_length: int) -> dict[str, dict[str, Any]]:
    import openvino as ov

    if sequence_length not in (128, 256):
        raise RuntimeErrorRecord("PROFILE_INVALID", "固定Shapeの系列長が不正です。")
    shapes = {"input_ids": ov.PartialShape([1, sequence_length]),
              "attention_mask": ov.PartialShape([1, sequence_length]),
              "marker_pos": ov.PartialShape([1, 8]), "marker_mask": ov.PartialShape([1, 8]),
              "qtype": ov.PartialShape([1])}
    expected_types = {"input_ids": ov.Type.i64, "attention_mask": ov.Type.i64,
                      "marker_pos": ov.Type.i64, "marker_mask": ov.Type.boolean, "qtype": ov.Type.i64}
    available = {name for port in model.inputs for name in port.get_names()}
    if available != set(shapes):
        raise RuntimeErrorRecord("OV_INPUT_CONTRACT_MISMATCH", "OpenVINO変換入力が固定入力契約と異なります。")
    model.reshape(shapes)
    recorded: dict[str, dict[str, Any]] = {}
    for port in model.inputs:
        if not port.partial_shape.is_static:
            raise RuntimeErrorRecord("OV_FIXED_SHAPE_FAILED", "OpenVINOの入力Shapeが静的ではありません。")
        name = port.get_any_name()
        shape = tuple(port.partial_shape.to_shape())
        if shape != tuple(shapes[name].to_shape()):
            raise RuntimeErrorRecord("OV_FIXED_SHAPE_FAILED", "OpenVINO変換後に固定Shapeを確認できません。")
        if port.element_type != expected_types[name]:
            raise RuntimeErrorRecord("OV_INPUT_DTYPE_MISMATCH", f"OpenVINO入力{name}のdtypeが固定契約と一致しません。")
        recorded[name] = {"shape": list(shape), "dtype": str(port.element_type)}
    return recorded


def _verify_saved_ir(root: Path, xml: Path, sequence_length: int) -> dict[str, dict[str, Any]]:
    import openvino as ov

    model = ov.Core().read_model(str(xml))
    return _static_shape(model, sequence_length)


def convert_model(root: Path, run: Run) -> dict[str, Any]:
    try:
        import openvino as ov

        verify_assets(root)
        source = asset_path(root, "onnx/model.onnx")
        digest = sha256_file(source)
        profile = run.profile_config
        sequence_length = profile["sequence_length"]
        destination = run.path / "models" / "converted"
        fp32_xml = destination / "fp32" / "model.xml"
        fp16_xml = destination / "fp16" / "model.xml"
        model = ov.convert_model(str(source))
        input_contract = _static_shape(model, sequence_length)
        (destination / "fp32").mkdir(parents=True)
        ov.save_model(model, str(fp32_xml), compress_to_fp16=False)
        (destination / "fp16").mkdir(parents=True)
        ov.save_model(model, str(fp16_xml), compress_to_fp16=True)
        saved_contracts = {
            "fp32": _verify_saved_ir(root, fp32_xml, sequence_length),
            "fp16": _verify_saved_ir(root, fp16_xml, sequence_length),
        }
        if saved_contracts["fp32"] != input_contract or saved_contracts["fp16"] != input_contract:
            raise RuntimeErrorRecord("SAVED_IR_CONTRACT_MISMATCH", "保存後IRのshapeまたはdtypeが変換元と異なります。")
        record = {"source_model_sha256": digest, "profile": profile["name"],
                  "sequence_length": sequence_length, "input_contract": input_contract,
                  "saved_ir_input_contracts": saved_contracts,
                  "fp32_xml": fp32_xml.relative_to(root).as_posix(),
                  "fp32_xml_sha256": sha256_file(fp32_xml),
                  "fp32_bin_sha256": sha256_file(fp32_xml.with_suffix(".bin")),
                  "fp16_xml": fp16_xml.relative_to(root).as_posix(),
                  "fp16_xml_sha256": sha256_file(fp16_xml),
                  "fp16_bin_sha256": sha256_file(fp16_xml.with_suffix(".bin")),
                  "openvino_version": ov.__version__}
        run.write_json("conversion.json", record)
        run.stage("G2-convert", "PASS", **record)
        return {"status": "PASS", "profile": record["profile"], "openvino_version": ov.__version__}
    except Exception as exc:
        code = getattr(exc, "code", "CONVERSION_FAILED")
        run.failure("G2-convert", code, exc)
        return {"status": "FAIL", "code": code}


def _openvino_cpu(root: Path, rows: list[dict[str, Any]], conversion: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import openvino as ov

    xml = root / conversion["fp32_xml"]
    if sha256_file(xml) != conversion["fp32_xml_sha256"] or sha256_file(xml.with_suffix(".bin")) != conversion["fp32_bin_sha256"]:
        raise RuntimeErrorRecord("CONVERTED_MODEL_HASH_MISMATCH", "FP32変換物のハッシュが一致しません。")
    core = ov.Core()
    model = core.read_model(str(xml))
    compiled = core.compile_model(model, "CPU")
    names = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    outputs: list[dict[str, Any]] = []
    for row in rows:
        arrays = _input_arrays(row)
        result = compiled({name: arrays[name] for name in names})
        named = {port.get_any_name(): value for port, value in result.items()}
        logits, actions = _select_outputs(named)
        outputs.append(_row_output(row, logits, actions))
    return outputs, {"device": "CPU", "openvino_version": ov.__version__}


def _comparison(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if len(reference) != len(candidate):
        raise RuntimeErrorRecord("OUTPUT_COUNT_MISMATCH", "比較する出力件数が一致しません。")
    diffs: list[float] = []
    action_diffs: list[float] = []
    matches = 0
    changed: list[str] = []
    for expected, actual in zip(reference, candidate):
        _assert_output_identity(expected, actual)
        if len(expected["probabilities"]) != len(actual["probabilities"]):
            raise RuntimeErrorRecord("OUTPUT_WIDTH_MISMATCH", "比較する候補数が一致しません。")
        diffs.extend(np.abs(np.asarray(expected["probabilities"]) - np.asarray(actual["probabilities"])).tolist())
        action_diffs.extend(np.abs(np.asarray(expected["action_probabilities"]) - np.asarray(actual["action_probabilities"])).tolist())
        if expected["decision_slot"] == actual["decision_slot"]:
            matches += 1
        else:
            changed.append(expected["case_id"])
    return {"cases": len(reference), "input_match_ratio": 1.0,
            "candidate_probability_value_count": len(diffs),
            "mean_probability_abs_difference_denominator": len(diffs),
            "action_probability_value_count": len(action_diffs),
            "max_probability_abs_difference": float(max(diffs, default=0.0)),
            "mean_probability_abs_difference": float(np.mean(diffs)) if diffs else 0.0,
            "max_action_probability_abs_difference": float(max(action_diffs, default=0.0)),
            "decision_match_ratio": matches / max(1, len(reference)),
            "important_new_errors": len(changed), "decision_changes": changed,
            "important_new_errors_definition":
                "同一入力での参照出力と変換後出力のdecision_slot相違件数。未確認ラベルに対する数値であり、人の正解に対する誤り件数ではない。"}


def _execution_device_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _exception_details(error: BaseException) -> dict[str, Any]:
    chain: list[dict[str, str]] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "repr": repr(current), "message": str(current)})
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return {"type": type(error).__name__, "repr": repr(error), "message": str(error),
            "chain": chain,
            "traceback_chain": "".join(traceback.format_exception(
                type(error), error, error.__traceback__, chain=True))}


def run_cpu_check(root: Path, run: Run) -> dict[str, Any]:
    try:
        inputs = _read_jsonl(run.path / "inputs.jsonl")
        reference = _read_jsonl(run.path / "reference.jsonl")
        conversion = json.loads((run.path / "conversion.json").read_text(encoding="utf-8"))
        onnx, onnx_env = _onnx_cpu(root, inputs)
        openvino, ov_env = _openvino_cpu(root, inputs, conversion)
        export = _comparison(reference, onnx)
        conversion_parity = _comparison(onnx, openvino)
        overall = _comparison(reference, openvino)
        gates = json.loads((root / "configs" / "gates.json").read_text(encoding="utf-8"))["stages"]["G2"]
        checks = {
            "input_token_id_match_ratio": overall["input_match_ratio"] >= gates["input_token_id_match_ratio"],
            "fp32_max_probability_abs_difference": overall["max_probability_abs_difference"] <= gates["fp32_max_probability_abs_difference"],
            "transformed_decision_match_ratio": overall["decision_match_ratio"] >= gates["transformed_decision_match_ratio"],
            "important_new_errors": overall["important_new_errors"] == gates["important_new_errors"],
            "upstream_onnx_export_parity": export["max_probability_abs_difference"] <= gates["fp32_max_probability_abs_difference"]
            and export["decision_match_ratio"] >= gates["transformed_decision_match_ratio"],
            "openvino_conversion_parity": conversion_parity["max_probability_abs_difference"] <= gates["fp32_max_probability_abs_difference"]
            and conversion_parity["decision_match_ratio"] >= gates["transformed_decision_match_ratio"],
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        result = {"status": status, "checks": checks, "original_torch_vs_onnx": export,
                  "onnx_vs_openvino_cpu": conversion_parity, "original_torch_vs_openvino_cpu": overall,
                  "type_comparison": {
                      "original_torch_vs_onnx": _type_comparison(reference, onnx),
                      "onnx_vs_openvino_cpu": _type_comparison(onnx, openvino),
                      "original_torch_vs_openvino_cpu": _type_comparison(reference, openvino),
                  },
                  "score_return_contract": _score_return_contract(root),
                  "noul_return_contract": _noul_return_contract(root),
                  "draft_label_reference": _draft_label_summary(root, reference, run),
                  "onnx_runtime": onnx_env, "openvino_cpu": ov_env,
                  "label_status": "ai_draft_unverified", "japanese_quality_status": "UNDETERMINED"}
        run.write_jsonl("onnx-cpu.jsonl", onnx)
        run.write_jsonl("openvino-cpu.jsonl", openvino)
        run.write_json("cpu-parity.json", result)
        run.stage("G2-cpu-check", status, **{k: v for k, v in result.items() if k != "status"})
        return {"status": status, "checks": checks, "max_probability_abs_difference": overall["max_probability_abs_difference"],
                "decision_match_ratio": overall["decision_match_ratio"], "important_new_errors": overall["important_new_errors"]}
    except Exception as exc:
        code = getattr(exc, "code", "CPU_CHECK_FAILED")
        run.failure("G2-cpu-check", code, exc)
        return {"status": "FAIL", "code": code}


def _load_run_conversion(root: Path, run: Run) -> dict[str, Any]:
    try:
        record = json.loads((run.path / "conversion.json").read_text(encoding="utf-8"))
        if (record.get("profile") != run.profile_config["name"]
                or record.get("sequence_length") != run.profile_config["sequence_length"]):
            raise RuntimeErrorRecord("CONVERSION_PROFILE_MISMATCH", "変換IRとRunプロファイルが一致しません。")
        return record
    except RuntimeErrorRecord:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeErrorRecord("CONVERSION_REQUIRED", "同一runでG2固定Shape変換を完了してください。") from exc


def _npu_error_code(phase: str, message: str) -> str:
    normalized = message.lower()
    if phase == "cache_setup":
        return "NPU_CACHE_SETUP_FAILED"
    if phase == "compile" and ("unrecognized operator" in normalized or "custom operator" in normalized):
        return "NPU_COMPILER_OPERATOR_FORMAT_OR_RECOGNITION_ERROR"
    if phase == "compile":
        return "NPU_COMPILATION_FAILED"
    return "NPU_INFERENCE_FAILED"


def _cache_inventory(path: Path) -> dict[str, str]:
    return {item.relative_to(path).as_posix(): sha256_file(item)
            for item in sorted(path.rglob("*")) if item.is_file()}


def _npu_cache_for_run(root: Path, run: Run, conversion: dict[str, Any],
                       cache_from_run: str | None) -> tuple[Path, dict[str, str], str]:
    if cache_from_run is None:
        cache_dir = run.path / "npu-cache"
        cache_dir.mkdir(parents=True, exist_ok=False)
        return cache_dir, {}, "empty"
    if cache_from_run == run.run_id:
        raise RuntimeErrorRecord("CACHE_SOURCE_RUN_MISMATCH", "同一Runのキャッシュは再利用元にできません。")
    source_run = open_run(root, cache_from_run)
    source_manifest = source_run.read_manifest()
    current_manifest = run.read_manifest()
    if (source_run.profile_config != run.profile_config
            or source_manifest.get("git_revision") != current_manifest.get("git_revision")
            or source_manifest.get("source_tree_sha256") != current_manifest.get("source_tree_sha256")
            or source_manifest.get("input_sha256") != current_manifest.get("input_sha256")):
        raise RuntimeErrorRecord("CACHE_SOURCE_FINGERPRINT_MISMATCH", "キャッシュ元とコード・設定・データ・依存ロックが一致しません。")
    source_conversion = _load_run_conversion(root, source_run)
    for key in ("fp16_xml_sha256", "fp16_bin_sha256"):
        if source_conversion.get(key) != conversion.get(key):
            raise RuntimeErrorRecord("CACHE_SOURCE_IR_MISMATCH", "キャッシュ元とFP16 IRのハッシュが一致しません。")
    try:
        source_result = json.loads((source_run.path / "npu-result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeErrorRecord("CACHE_SOURCE_RESULT_MISSING", "キャッシュ元RunのNPU結果がありません。") from exc
    properties = source_result.get("compiled_model_properties", {})
    if (source_result.get("status") != "PASS" or source_result.get("device_architecture") != "3720"
            or _execution_device_names(properties.get("EXECUTION_DEVICES")) != ["NPU"]
            or not str(properties.get("NPU_COMPILER_TYPE", "")).upper().endswith("PLUGIN")):
        raise RuntimeErrorRecord("CACHE_SOURCE_NOT_COMPATIBLE", "キャッシュ元が3720上のPLUGIN NPU成功Runではありません。")
    source_cache_dir = source_run.path / "npu-cache"
    files = _cache_inventory(source_cache_dir) if source_cache_dir.is_dir() else {}
    if not files:
        raise RuntimeErrorRecord("CACHE_SOURCE_EMPTY", "キャッシュ元に再利用可能なコンパイル済みファイルがありません。")
    cache_dir = run.path / "npu-cache"
    shutil.copytree(source_cache_dir, cache_dir)
    if _cache_inventory(cache_dir) != files:
        raise RuntimeErrorRecord("CACHE_COPY_INTEGRITY_FAILED", "再利用キャッシュのコピー後ハッシュが元と一致しません。")
    return cache_dir, files, "reuse"


def run_npu(root: Path, run: Run, expected_arch: str = "3720",
            cache_from_run: str | None = None, compiler_type: str = "PLUGIN",
            source_run: Run | None = None) -> dict[str, Any]:
    phase = "precondition"
    startup: dict[str, Any] = {"process_age_at_entry_ms": process_age_ms()}
    execution_devices: list[str] = []
    data_run = source_run or run
    stage_name = "NPU_DIAGNOSTIC" if source_run else "G3"
    if data_run.dataset_identity != run.dataset_identity:
        run.blocked(stage_name, "NPU_DATASET_IDENTITY_MISMATCH",
                    "NPU実行Runと入力Runのデータ版・採点契約ハッシュが一致しません。")
        return {"status": "BLOCKED", "code": "NPU_DATASET_IDENTITY_MISMATCH"}
    settings: dict[str, Any] = {"explicit_device": "NPU", "expected_architecture": expected_arch,
                                "fallback_allowed": False, "requested_compiler_type": compiler_type,
                                "source_run": data_run.run_id}
    cache_dir: Path | None = None
    manifest = data_run.read_manifest()
    g2 = manifest.get("stages", {}).get("G2-cpu-check", {}).get("status")
    if g2 != "PASS":
        run.blocked(stage_name, "CPU_PARITY_NOT_PASS", "参照RunのG2 CPU照合がPASSしていないため、NPU出力の正しさを判定しません。")
        return {"status": "BLOCKED", "code": "CPU_PARITY_NOT_PASS"}
    try:
        inputs = _read_jsonl(data_run.path / "inputs.jsonl")
        reference = _read_jsonl(data_run.path / "reference.jsonl")
        if run.profile_config != data_run.profile_config:
            raise RuntimeErrorRecord("NPU_PROFILE_MISMATCH", "NPU Runと入力Runの固定プロファイルが一致しません。")
        conversion = _load_run_conversion(root, data_run)
        device_start = time.perf_counter()
        device = inspect_device("NPU", expected_arch)
        startup["device_inspection_ms"] = (time.perf_counter() - device_start) * 1000
        saved_as = _unique_json(run, "environment-NPU", device)
        run.stage(f"{stage_name}-device-check", device["status"], code=device["code"], environment_file=saved_as,
                  actual_architecture=device.get("device_properties", {}).get("DEVICE_ARCHITECTURE"),
                  expected_architecture=expected_arch)
        if device["status"] != "PASS":
            run.blocked(stage_name, device["code"], device["message"] if "message" in device else "NPU実値が期待アーキテクチャと一致しません。")
            return {"status": "BLOCKED", "code": device["code"], "environment_file": saved_as}
        import openvino as ov

        xml = root / conversion["fp16_xml"]
        phase = "compiled_model_integrity"
        if sha256_file(xml) != conversion["fp16_xml_sha256"] or sha256_file(xml.with_suffix(".bin")) != conversion["fp16_bin_sha256"]:
            raise RuntimeErrorRecord("CONVERTED_MODEL_HASH_MISMATCH", "FP16変換物のハッシュが一致しません。")
        core_start = time.perf_counter()
        core = ov.Core()
        startup["openvino_core_ms"] = (time.perf_counter() - core_start) * 1000
        model_read_start = time.perf_counter()
        model = core.read_model(str(xml))
        startup["model_read_ms"] = (time.perf_counter() - model_read_start) * 1000
        settings["fp16_xml_sha256"] = conversion["fp16_xml_sha256"]
        settings["fp16_bin_sha256"] = conversion["fp16_bin_sha256"]
        phase = "cache_setup"
        cache_dir, cache_before, cache_mode = _npu_cache_for_run(root, run, conversion, cache_from_run)
        compile_configuration = {"CACHE_DIR": str(cache_dir)}
        if compiler_type != "DEFAULT":
            compile_configuration["NPU_COMPILER_TYPE"] = compiler_type
        settings["cache_mode"] = cache_mode
        settings["cache_source_run"] = cache_from_run
        settings["requested_compile_configuration"] = compile_configuration
        settings["cache_before_compile"] = cache_before
        if cache_mode == "empty" and cache_before:
            raise RuntimeErrorRecord("EMPTY_CACHE_NOT_EMPTY", "空キャッシュ試験のキャッシュディレクトリが空ではありません。")
        settings["npu_properties_before_compile"] = {}
        for key in ("DEVICE_ARCHITECTURE", "NPU_COMPILER_TYPE", "NPU_COMPILER_VERSION",
                    "NPU_DRIVER_VERSION", "CACHE_DIR", "NPU_BYPASS_UMD_CACHING"):
            try:
                settings["npu_properties_before_compile"][key] = _json_safe(core.get_property("NPU", key))
            except Exception as property_error:
                settings["npu_properties_before_compile"][key] = {
                    "unavailable": True, "error_type": type(property_error).__name__,
                    "message": safe_message(str(property_error))}
        phase = "compile"
        compile_start = time.perf_counter()
        compiled = core.compile_model(model, "NPU", compile_configuration)
        startup["compile_ms"] = (time.perf_counter() - compile_start) * 1000
        compiled_properties: dict[str, Any] = {}
        for key in ("EXECUTION_DEVICES", "NPU_COMPILER_TYPE", "NPU_COMPILER_VERSION", "RUNTIME_REQUIREMENTS",
                    "LOADED_FROM_CACHE"):
            try:
                compiled_properties[key] = _json_safe(compiled.get_property(key))
            except Exception as property_error:
                compiled_properties[key] = {
                    "unavailable": True, "error_type": type(property_error).__name__,
                    "message": safe_message(str(property_error))}
        settings["compiled_model_properties"] = compiled_properties
        execution_devices = _execution_device_names(compiled_properties.get("EXECUTION_DEVICES"))
        cache_after = _cache_inventory(cache_dir)
        settings["cache_after_compile"] = cache_after
        try:
            if not execution_devices:
                raise RuntimeError("EXECUTION_DEVICESが文字列または配列で取得できません。")
        except Exception as exc:
            settings["compiled_model_created"] = True
            _unique_json(run, "npu-diagnostic", {
                "phase": "execution_device_evidence", "settings": settings,
                "compiled_model_created": True, "exception": _exception_details(exc)})
            run.blocked(stage_name, "EXECUTION_DEVICE_EVIDENCE_UNAVAILABLE",
                        f"実行先プロパティを取得できません: {type(exc).__name__}")
            return {"status": "BLOCKED", "code": "EXECUTION_DEVICE_EVIDENCE_UNAVAILABLE"}
        if any(not (value.upper() == "NPU" or value.upper().startswith("NPU."))
               for value in execution_devices):
            run.blocked(stage_name, "EXECUTION_DEVICE_MISMATCH", "コンパイル済みモデルの実行先がNPU単独ではありません。",
                        execution_devices=execution_devices)
            return {"status": "BLOCKED", "code": "EXECUTION_DEVICE_MISMATCH"}
        compiler_type = compiled_properties.get("NPU_COMPILER_TYPE")
        expected_compiler = settings["requested_compiler_type"]
        if expected_compiler != "DEFAULT" and (
                not isinstance(compiler_type, str) or not compiler_type.upper().endswith(expected_compiler)):
            settings["compiled_model_created"] = True
            _unique_json(run, "npu-diagnostic", {
                "phase": "compiler_path_evidence", "settings": settings,
                "compiled_model_created": True,
                "exception": {"type": "CompilerPathEvidenceUnavailable", "message":
                    "CompiledModelのNPU_COMPILER_TYPEが要求したPLUGIN経路と一致しません。"}})
            run.blocked(stage_name, "NPU_COMPILER_PATH_MISMATCH",
                        "CompiledModelの実コンパイラーが要求経路と一致しません。",
                        actual_compiler_type=compiler_type)
            return {"status": "BLOCKED", "code": "NPU_COMPILER_PATH_MISMATCH"}
        names = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
        outputs: list[dict[str, Any]] = []
        phase = "inference"
        inference_start = time.perf_counter()
        for index, row in enumerate(inputs):
            first_output_start = time.perf_counter()
            arrays = _input_arrays(row)
            result = compiled({name: arrays[name] for name in names})
            named = {port.get_any_name(): value for port, value in result.items()}
            logits, actions = _select_outputs(named)
            outputs.append(_row_output(row, logits, actions))
            if index == 0:
                startup["first_input_to_output_ms"] = (time.perf_counter() - first_output_start) * 1000
                startup["process_start_to_first_output_ms"] = process_age_ms()
        startup["all_executable_inference_and_output_ms"] = (time.perf_counter() - inference_start) * 1000
        phase = "comparison"
        comparison = _comparison(reference, outputs)
        type_comparison = _type_comparison(reference, outputs)
        gates = json.loads((root / "configs" / "gates.json").read_text(encoding="utf-8"))["stages"]["G3"]
        checks = {
            "fp16_max_probability_abs_difference": comparison["max_probability_abs_difference"] <= gates["fp16_max_probability_abs_difference"],
            "fp16_mean_probability_abs_difference": comparison["mean_probability_abs_difference"] <= gates["fp16_mean_probability_abs_difference"],
            "transformed_decision_match_ratio": comparison["decision_match_ratio"] >= gates["transformed_decision_match_ratio"],
            "important_new_errors": comparison["important_new_errors"] == gates["important_new_errors"],
            "execution_devices_npu_only": True,
            "architecture_matches": True,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        record = {"status": status, "explicit_device": "NPU", "execution_devices": execution_devices,
                  "requested_compile_configuration": settings["requested_compile_configuration"],
                  "compiled_model_properties": compiled_properties,
                  "cache_before_compile": settings["cache_before_compile"],
                  "cache_after_compile": settings["cache_after_compile"],
                  "cache_test": {
                      "mode": cache_mode,
                      "source_run": cache_from_run,
                      "loaded_from_cache": compiled_properties.get("LOADED_FROM_CACHE"),
                      "status": ("PASS" if cache_mode == "empty" else
                                ("PASS" if compiled_properties.get("LOADED_FROM_CACHE") is True else "FAIL")),
                  },
                  "device_architecture": device["device_properties"].get("DEVICE_ARCHITECTURE"),
                  "startup_timing": startup,
                  "checks": checks, "comparison_to_original_cpu": comparison,
                  "type_comparison_to_original_cpu": type_comparison,
                  "score_return_contract": _score_return_contract(root),
                  "noul_return_contract": _noul_return_contract(root),
                  "draft_label_reference": _draft_label_summary(root, reference, data_run),
                  "important_new_errors_definition": comparison["important_new_errors_definition"],
                  "label_status": "ai_draft_unverified", "human_label_errors": "UNDETERMINED",
                  "japanese_quality_status": "UNDETERMINED",
                  "openvino_version": ov.__version__}
        record["source_run"] = data_run.run_id
        run.write_jsonl("npu.jsonl", outputs)
        result_name = "npu-diagnostic-result.json" if source_run else "npu-result.json"
        run.write_json(result_name, record)
        run.stage(stage_name, status, result_file=result_name,
                  **{k: v for k, v in record.items() if k != "status"})
        if cache_mode == "reuse":
            run.stage("CACHE_REUSE", record["cache_test"]["status"],
                      source_run=cache_from_run,
                      loaded_from_cache=compiled_properties.get("LOADED_FROM_CACHE"),
                      cache_before_compile=cache_before, cache_after_compile=cache_after)
        return {"status": status, "execution_devices": execution_devices,
                "cache_test": record["cache_test"],
                "device_architecture": record["device_architecture"], "checks": checks,
                "max_probability_abs_difference": comparison["max_probability_abs_difference"],
                "mean_probability_abs_difference": comparison["mean_probability_abs_difference"],
                "decision_match_ratio": comparison["decision_match_ratio"]}
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code is None:
            code = _npu_error_code(phase, str(exc))
        ir_operator_counts: dict[str, int] = {}
        if phase == "compile" and "model" in locals():
            from collections import Counter
            try:
                ir_operator_counts = dict(sorted(Counter(
                    operation.get_type_name() for operation in model.get_ordered_ops()
                ).items()))
            except Exception:
                ir_operator_counts = {"inspection": -1}
        diagnostic = {"phase": phase, "settings": settings,
                      "compiled_model_created": "compiled" in locals(),
                      "execution_devices": execution_devices,
                      "exception": _exception_details(exc)}
        if cache_dir is not None:
            diagnostic["cache_after_failure"] = sorted(path.name for path in cache_dir.iterdir())
        diagnostic_name = _unique_json(run, "npu-diagnostic", diagnostic)
        run.failure(stage_name, code, exc, phase=phase, settings=settings,
                    execution_devices=execution_devices,
                    local_diagnostic_file=diagnostic_name,
                    triage_performed=["OpenVINO NPU architecture property", "explicit NPU compilation request",
                                      "converted FP16 IR operator inventory", "full exception chain saved locally"],
                    ir_operator_counts=ir_operator_counts,
                    unconfirmed=["operator identified by proprietary NPU compiler", "actual NPU inference result"])
        return {"status": "FAIL", "code": code}


def run_npu_control(root: Path, run: Run, compiler_type: str,
                    expected_arch: str = "3720") -> dict[str, Any]:
    phase = "environment"
    try:
        import numpy as np
        import openvino as ov

        device = inspect_device("NPU", expected_arch)
        environment_file = _unique_json(run, "environment-NPU", device)
        if device["status"] != "PASS":
            run.blocked("NPU_DIAGNOSTIC", device["code"], "NPU実値が期待値と一致しません。",
                        environment_file=environment_file)
            return {"status": "BLOCKED", "code": device["code"]}
        core = ov.Core()
        phase = "control_model_creation"
        left = ov.opset13.parameter([1, 8], ov.Type.f32, name="left")
        right = ov.opset13.parameter([1, 8], ov.Type.f32, name="right")
        model = ov.Model([ov.opset13.add(left, right, name="sum")], [left, right], "npu_add_control")
        model_dir = run.path / "control-model"
        model_dir.mkdir(exist_ok=False)
        xml = model_dir / "model.xml"
        ov.save_model(model, str(xml))
        left_value = np.asarray([[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]], dtype=np.float32)
        right_value = np.asarray([[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]], dtype=np.float32)
        cpu = core.compile_model(core.read_model(str(xml)), "CPU", {"INFERENCE_NUM_THREADS": 1})
        cpu_output = np.asarray(next(iter(cpu({"left": left_value, "right": right_value}).values())))
        query = core.query_model(core.read_model(str(xml)), "NPU")
        cache_dir = run.path / "npu-cache"
        cache_dir.mkdir(exist_ok=False)
        configuration = {"CACHE_DIR": str(cache_dir)}
        if compiler_type != "DEFAULT":
            configuration["NPU_COMPILER_TYPE"] = compiler_type
        phase = "compile"
        compiled = core.compile_model(core.read_model(str(xml)), "NPU", configuration)
        properties: dict[str, Any] = {}
        for key in ("EXECUTION_DEVICES", "NPU_COMPILER_TYPE", "NPU_COMPILER_VERSION",
                    "RUNTIME_REQUIREMENTS", "LOADED_FROM_CACHE"):
            try:
                properties[key] = _json_safe(compiled.get_property(key))
            except Exception as exc:
                properties[key] = {"unavailable": True, "error_type": type(exc).__name__,
                                    "message": safe_message(str(exc))}
        execution = _execution_device_names(properties.get("EXECUTION_DEVICES"))
        actual_compiler = properties.get("NPU_COMPILER_TYPE")
        if execution != ["NPU"]:
            run.blocked("NPU_DIAGNOSTIC", "EXECUTION_DEVICE_MISMATCH",
                        "対照モデルの実行先がNPU単独ではありません。", execution_devices=execution)
            return {"status": "BLOCKED", "code": "EXECUTION_DEVICE_MISMATCH"}
        if compiler_type != "DEFAULT" and (
                not isinstance(actual_compiler, str) or not actual_compiler.upper().endswith(compiler_type)):
            run.blocked("NPU_DIAGNOSTIC", "NPU_COMPILER_PATH_MISMATCH",
                        "対照モデルの実コンパイラーが要求経路と一致しません。",
                        requested_compiler=compiler_type, actual_compiler=actual_compiler)
            return {"status": "BLOCKED", "code": "NPU_COMPILER_PATH_MISMATCH"}
        phase = "inference"
        npu_output = np.asarray(next(iter(compiled({"left": left_value, "right": right_value}).values())))
        maximum_difference = float(np.max(np.abs(cpu_output - npu_output)))
        nodes = len(model.get_ordered_ops())
        record = {
            "status": "PASS" if maximum_difference == 0.0 else "FAIL",
            "kind": "nonconstant_fixed_shape_add",
            "shape": [1, 8], "dtype": "float32",
            "inputs_sha256": hashlib.sha256(left_value.tobytes() + right_value.tobytes()).hexdigest(),
            "requested_compiler": compiler_type, "compile_configuration": configuration,
            "compiled_model_properties": properties, "execution_devices": execution,
            "device_architecture": device["device_properties"].get("DEVICE_ARCHITECTURE"),
            "query_model": {"mapped_nodes": len(query), "graph_nodes": nodes,
                            "unmapped_nodes": max(0, nodes - len(query))},
            "cpu_vs_npu": {"maximum_absolute_difference": maximum_difference,
                           "cpu_output": cpu_output.tolist(), "npu_output": npu_output.tolist()},
            "cache_before_compile": {}, "cache_after_compile": _cache_inventory(cache_dir),
            "openvino_version": ov.__version__, "environment_file": environment_file,
        }
        run.write_json("npu-probe.json", record)
        run.stage("NPU_DIAGNOSTIC", record["status"], **{k: v for k, v in record.items() if k != "status"})
        return {"status": record["status"], "execution_devices": execution,
                "actual_compiler": actual_compiler, "max_cpu_difference": maximum_difference}
    except Exception as exc:
        diagnostic_name = _unique_json(run, "npu-diagnostic", {
            "phase": phase, "exception": _exception_details(exc), "requested_compiler": compiler_type,
        })
        run.failure("NPU_DIAGNOSTIC", _npu_error_code(phase, str(exc)), exc,
                    local_diagnostic_file=diagnostic_name, phase=phase)
        return {"status": "FAIL", "code": _npu_error_code(phase, str(exc))}

"""CPU参照、固定形状変換、OpenVINO照合、明示NPU実行。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .assets import AssetError, asset_path, load_lock, make_execution_checkpoint, sha256_file, verify_assets
from .contracts import ContractError, softmax, validate_dataset_case
from .storage import Run, safe_message


class RuntimeErrorRecord(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


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


def _load_cases(root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((root / "datasets" / "manifest.json").read_text(encoding="utf-8"))
    path = root / manifest["splits"]["dev"]["path"]
    expected = manifest["splits"]["dev"]["sha256"]
    if sha256_file(path) != expected:
        raise RuntimeErrorRecord("DATASET_HASH_MISMATCH", "devデータのハッシュが固定値と一致しません。")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != 60:
        raise RuntimeErrorRecord("DATASET_COUNT_MISMATCH", "devデータは60件でなければなりません。")
    for case in cases:
        validate_dataset_case(case)
    return cases


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
    from laya.common import QTYPES

    rows: list[dict[str, Any]] = []
    for case in _load_cases(root):
        question_id, raw_question = next(iter(case["questions"].items()))
        question = _prepare_question(raw_question)
        Agent = type(agent)
        Agent._check_question(question_id, question)
        internal = {question_id: Agent._to_internal(question)}
        _ensure_no_question_truncation(agent.tok, question, internal[question_id], question["type"])
        item = agent._encode_state(case["state"], [question_id], internal, max_len=256, head_max_len=192)[0]
        if item["state_stats"]["truncated"]:
            raise RuntimeErrorRecord("STATE_TRUNCATION_REQUIRED", "状態文が256トークン固定入力へ収まりません。")
        if item["options"]["options_distinct"] != item["options"]["options"]:
            raise RuntimeErrorRecord("CANDIDATES_COLLAPSED", "固定形状へ整形した候補がトークン上で重複しました。")
        if item["options"]["tokens_per_option"] is not None:
            raise RuntimeErrorRecord("OPTION_TRUNCATION_REQUIRED", "候補が固定ヘッダー内で切り詰められました。")
        expected_candidates = (len(raw_question["criteria"]) if raw_question["type"] in {"choice", "score"}
                               else 2)
        if len(item["markers"]) != expected_candidates:
            raise RuntimeErrorRecord("CANDIDATE_COUNT_MISMATCH", "候補数とマーカー数が一致しません。")
        if not 2 <= len(item["markers"]) <= 8 or len(item["ids"]) > 256:
            raise RuntimeErrorRecord("FIXED_INPUT_SHAPE_INVALID", "トークン列または候補数が固定形状を超えています。")
        pad_id = int(agent.tok.pad_token_id)
        token_ids = np.full((1, 256), pad_id, dtype=np.int64)
        attention = np.zeros((1, 256), dtype=np.int64)
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
        rows.append({"case_id": case["case_id"], "question_id": question_id,
                     "question_type": raw_question["type"], "qtype": qtype,
                     "candidate_count": size,
                     "candidate_ids": list(raw_question["criteria"].keys()) if raw_question["type"] == "choice"
                     else (list(range(size)) if raw_question["type"] == "score" else [False, True]),
                     "temperature": temperature, "input_sha256": _input_digest(inputs), "inputs": inputs,
                     "label_status": "ai_draft_unverified"})
    return rows


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
            output_rows.append({"case_id": row["case_id"], "question_id": row["question_id"],
                                "input_sha256": row["input_sha256"], "probabilities": probs,
                                "decision_slot": decision_index, "decision": decision,
                                "action_probabilities": actions[index].astype(float).tolist()})
    return output_rows


def run_reference(root: Path, run: Run) -> dict[str, Any]:
    try:
        agent = _load_agent(root, run)
        rows = _build_input_rows(root, run, agent)
        reference = _model_output(agent, rows)
        run.write_jsonl("inputs.jsonl", rows)
        run.write_jsonl("reference.jsonl", reference)
        result = {"status": "UNVERIFIED_LABEL_REFERENCE_ONLY", "cases": len(rows),
                  "label_state": "ai_draft_unverified", "model_device": str(agent.device),
                  "input_hashes_recorded": True}
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
    shapes = {"input_ids": (1, 256), "attention_mask": (1, 256), "marker_pos": (1, 8),
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
    return {"case_id": row["case_id"], "question_id": row["question_id"],
            "input_sha256": row["input_sha256"], "probabilities": probs,
            "decision_slot": decision_index, "decision": row["candidate_ids"][decision_index],
            "action_probabilities": softmax([float(value) for value in actions])}


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


def _static_shape(model: Any) -> None:
    import openvino as ov

    shapes = {"input_ids": ov.PartialShape([1, 256]), "attention_mask": ov.PartialShape([1, 256]),
              "marker_pos": ov.PartialShape([1, 8]), "marker_mask": ov.PartialShape([1, 8]),
              "qtype": ov.PartialShape([1])}
    available = {name for port in model.inputs for name in port.get_names()}
    if available != set(shapes):
        raise RuntimeErrorRecord("OV_INPUT_CONTRACT_MISMATCH", "OpenVINO変換入力が固定入力契約と異なります。")
    model.reshape(shapes)
    for port in model.inputs:
        if not port.partial_shape.is_static:
            raise RuntimeErrorRecord("OV_FIXED_SHAPE_FAILED", "OpenVINOの入力Shapeが静的ではありません。")
        shape = tuple(port.partial_shape.to_shape())
        if shape != tuple(shapes[port.get_any_name()].to_shape()):
            raise RuntimeErrorRecord("OV_FIXED_SHAPE_FAILED", "OpenVINO変換後に固定Shapeを確認できません。")


def convert_model(root: Path, run: Run) -> dict[str, Any]:
    try:
        import openvino as ov

        verify_assets(root)
        source = asset_path(root, "onnx/model.onnx")
        digest = sha256_file(source)
        destination = run.path / "models" / "converted"
        fp32_xml = destination / "fp32" / "model.xml"
        fp16_xml = destination / "fp16" / "model.xml"
        model = ov.convert_model(str(source))
        _static_shape(model)
        (destination / "fp32").mkdir(parents=True)
        ov.save_model(model, str(fp32_xml), compress_to_fp16=False)
        (destination / "fp16").mkdir(parents=True)
        ov.save_model(model, str(fp16_xml), compress_to_fp16=True)
        record = {"source_model_sha256": digest, "profile": "s256-k8",
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
        if expected["case_id"] != actual["case_id"] or expected["input_sha256"] != actual["input_sha256"]:
            raise RuntimeErrorRecord("INPUT_IDENTITY_MISMATCH", "比較する出力の入力IDまたはハッシュが異なります。")
        if len(expected["probabilities"]) != len(actual["probabilities"]):
            raise RuntimeErrorRecord("OUTPUT_WIDTH_MISMATCH", "比較する候補数が一致しません。")
        diffs.extend(np.abs(np.asarray(expected["probabilities"]) - np.asarray(actual["probabilities"])).tolist())
        action_diffs.extend(np.abs(np.asarray(expected["action_probabilities"]) - np.asarray(actual["action_probabilities"])).tolist())
        if expected["decision_slot"] == actual["decision_slot"]:
            matches += 1
        else:
            changed.append(expected["case_id"])
    return {"cases": len(reference), "input_match_ratio": 1.0,
            "max_probability_abs_difference": float(max(diffs, default=0.0)),
            "mean_probability_abs_difference": float(np.mean(diffs)) if diffs else 0.0,
            "max_action_probability_abs_difference": float(max(action_diffs, default=0.0)),
            "decision_match_ratio": matches / max(1, len(reference)),
            "important_new_errors": len(changed), "decision_changes": changed}


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
        return record
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeErrorRecord("CONVERSION_REQUIRED", "同一runでG2固定Shape変換を完了してください。") from exc


def _npu_error_code(phase: str, message: str) -> str:
    normalized = message.lower()
    if phase == "compile" and ("unrecognized operator" in normalized or "custom operator" in normalized):
        return "NPU_COMPILER_OPERATOR_FORMAT_OR_RECOGNITION_ERROR"
    if phase == "compile":
        return "NPU_COMPILATION_FAILED"
    return "NPU_INFERENCE_FAILED"


def run_npu(root: Path, run: Run, expected_arch: str = "3720") -> dict[str, Any]:
    phase = "precondition"
    execution_devices: list[str] = []
    settings: dict[str, Any] = {"explicit_device": "NPU", "expected_architecture": expected_arch,
                                "fallback_allowed": False}
    manifest = run.read_manifest()
    g2 = manifest.get("stages", {}).get("G2-cpu-check", {}).get("status")
    if g2 != "PASS":
        run.blocked("G3", "CPU_PARITY_NOT_PASS", "G2 CPU照合がPASSしていないため、NPU出力の正しさを判定しません。")
        return {"status": "BLOCKED", "code": "CPU_PARITY_NOT_PASS"}
    try:
        inputs = _read_jsonl(run.path / "inputs.jsonl")
        reference = _read_jsonl(run.path / "reference.jsonl")
        conversion = _load_run_conversion(root, run)
        device = inspect_device("NPU", expected_arch)
        saved_as = _unique_json(run, "environment-NPU", device)
        run.stage("G3-device-check", device["status"], code=device["code"], environment_file=saved_as,
                  actual_architecture=device.get("device_properties", {}).get("DEVICE_ARCHITECTURE"),
                  expected_architecture=expected_arch)
        if device["status"] != "PASS":
            run.blocked("G3", device["code"], device["message"] if "message" in device else "NPU実値が期待アーキテクチャと一致しません。")
            return {"status": "BLOCKED", "code": device["code"], "environment_file": saved_as}
        import openvino as ov

        xml = root / conversion["fp16_xml"]
        phase = "compiled_model_integrity"
        if sha256_file(xml) != conversion["fp16_xml_sha256"] or sha256_file(xml.with_suffix(".bin")) != conversion["fp16_bin_sha256"]:
            raise RuntimeErrorRecord("CONVERTED_MODEL_HASH_MISMATCH", "FP16変換物のハッシュが一致しません。")
        core = ov.Core()
        model = core.read_model(str(xml))
        settings["fp16_xml_sha256"] = conversion["fp16_xml_sha256"]
        settings["fp16_bin_sha256"] = conversion["fp16_bin_sha256"]
        phase = "compile"
        compiled = core.compile_model(model, "NPU")
        try:
            execution_devices = [str(value) for value in compiled.get_property("EXECUTION_DEVICES")]
        except Exception as exc:
            run.blocked("G3", "EXECUTION_DEVICE_EVIDENCE_UNAVAILABLE", f"実行先プロパティを取得できません: {type(exc).__name__}")
            return {"status": "BLOCKED", "code": "EXECUTION_DEVICE_EVIDENCE_UNAVAILABLE"}
        if not execution_devices or any("npu" not in value.lower() for value in execution_devices):
            run.blocked("G3", "EXECUTION_DEVICE_MISMATCH", "コンパイル済みモデルの実行先がNPU単独ではありません。",
                        execution_devices=execution_devices)
            return {"status": "BLOCKED", "code": "EXECUTION_DEVICE_MISMATCH"}
        names = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
        outputs: list[dict[str, Any]] = []
        phase = "inference"
        for row in inputs:
            arrays = _input_arrays(row)
            result = compiled({name: arrays[name] for name in names})
            named = {port.get_any_name(): value for port, value in result.items()}
            logits, actions = _select_outputs(named)
            outputs.append(_row_output(row, logits, actions))
        phase = "comparison"
        comparison = _comparison(reference, outputs)
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
                  "device_architecture": device["device_properties"].get("DEVICE_ARCHITECTURE"),
                  "checks": checks, "comparison_to_original_cpu": comparison,
                  "label_status": "ai_draft_unverified", "japanese_quality_status": "UNDETERMINED",
                  "openvino_version": ov.__version__}
        run.write_jsonl("npu.jsonl", outputs)
        run.write_json("npu-result.json", record)
        run.stage("G3", status, **{k: v for k, v in record.items() if k != "status"})
        return {"status": status, "execution_devices": execution_devices,
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
        run.failure("G3", code, exc, phase=phase, settings=settings,
                    execution_devices=execution_devices,
                    triage_performed=["OpenVINO NPU architecture property", "explicit NPU compilation request",
                                      "converted FP16 IR operator inventory"],
                    ir_operator_counts=ir_operator_counts,
                    unconfirmed=["operator identified by proprietary NPU compiler", "actual NPU inference result"])
        return {"status": "FAIL", "code": code}

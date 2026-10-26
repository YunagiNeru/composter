"""Fixed, local-only CPU evaluation for two pretrained typed-decision models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .storage import Run, open_run, safe_message
from .zero_shot_eval import (
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_DATASET_SHA256,
    _all_metrics,
    _baseline_metrics,
    _baseline_rows,
    _majority_baselines,
    _metric_summary,
    _read_dev_cases,
    _strict_json,
)


EVAL_CONFIG = "configs/typed-model-evaluation-v1.json"
MODEL_LOCK = "configs/typed-models.lock.json"
CONTROL_DATA = "datasets/typed-model-controls-v1.json"
RUN_STAGE = "TYPED_MODEL_CPU"
ANALYSIS_STAGE = "TYPED_MODEL_ANALYSIS"
HORIZON_ANALYSIS_RUN_ID = "20261026-000956-fdf7ed16"
TRIAL_COUNT = 3
MAX_GLINER_INPUT_IDS = 512


class TypedModelError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_config(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        lock = _strict_json((root / MODEL_LOCK).read_text(encoding="utf-8"))
        evaluation = _strict_json((root / EVAL_CONFIG).read_text(encoding="utf-8"))
        controls = _strict_json((root / CONTROL_DATA).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TypedModelError("TYPED_MODEL_CONFIGURATION_INVALID") from exc
    condition = evaluation.get("primary_cpu_condition", {})
    if (lock.get("schema_version") != 1
            or evaluation.get("evaluation_version") != "typed-model-evaluation-v1"
            or evaluation.get("dataset_version") != "dev-v3"
            or evaluation.get("scoring_contract_version") != "evaluation-v2"
            or evaluation.get("label_state") != "ai_draft_unverified"
            or condition.get("device") != "cpu"
            or condition.get("torch_dtype") != "float32"
            or condition.get("torch_threads") != 2
            or condition.get("independent_process_trials") != TRIAL_COUNT
            or condition.get("tokenizer_truncation") != "forbidden"
            or condition.get("model_hub_access_during_evaluation") != "offline"
            or controls.get("created_before_candidate_inference") is not True
            or controls.get("verification_status") != "ai_draft_unverified"):
        raise TypedModelError("TYPED_MODEL_CONFIGURATION_CONTRACT_MISMATCH")
    return lock, evaluation, controls


def _candidate_spec(lock: dict[str, Any], candidate: str) -> dict[str, Any]:
    key = {"laya-typed": "candidate_laya_typed",
           "gliner25-multi-decide": "candidate_gliner25_multi_decide"}.get(candidate)
    if key is None:
        raise TypedModelError("TYPED_MODEL_CANDIDATE_UNKNOWN")
    spec = lock.get(key, {})
    if spec.get("status") != "PINNED_FOR_CPU_EVALUATION" or not spec.get("revision"):
        raise TypedModelError("TYPED_MODEL_REVISION_NOT_PINNED")
    return spec


def _model_directory(root: Path, candidate: str, spec: dict[str, Any]) -> Path:
    folder = "laya-typed" if candidate == "laya-typed" else "gliner25-multi-decide"
    return root / ".local" / "models" / f"{folder}-{spec['revision']}"


def _verify_model_files(root: Path, candidate: str, spec: dict[str, Any]) -> tuple[Path, dict[str, dict[str, Any]]]:
    model_dir = _model_directory(root, candidate, spec)
    verified: dict[str, dict[str, Any]] = {}
    files = spec.get("files")
    if not isinstance(files, dict) or not files:
        raise TypedModelError("TYPED_MODEL_FILE_LOCK_EMPTY")
    for relative, expected_digest in files.items():
        path = model_dir / Path(*relative.split("/"))
        if not path.is_file():
            raise TypedModelError("TYPED_MODEL_FILE_MISSING")
        digest = _sha256_file(path)
        if digest != expected_digest:
            raise TypedModelError("TYPED_MODEL_FILE_HASH_MISMATCH")
        verified[relative] = {"size_bytes": path.stat().st_size, "sha256": digest}
    return model_dir, verified


def _verify_gliner_wheel(root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    spec = lock.get("gliner2_source", {})
    path = root / ".local" / "sources" / "gliner2-2.1.0-py3-none-any.whl"
    if (not path.is_file() or path.stat().st_size != spec.get("wheel_size")
            or _sha256_file(path) != spec.get("wheel_sha256")):
        raise TypedModelError("GLINER2_WHEEL_HASH_MISMATCH")
    return {"package": spec.get("package"), "version": spec.get("version"),
            "tag": spec.get("tag"), "commit": spec.get("commit"),
            "wheel_size": path.stat().st_size, "wheel_sha256": _sha256_file(path),
            "remote_code_execution": spec.get("remote_code_execution")}


def _expected_dependencies(root: Path) -> dict[str, str]:
    import importlib.metadata

    try:
        lock = _strict_json((root / MODEL_LOCK).read_text(encoding="utf-8"))
        requirements = lock["runtime_dependencies"]
        versions = {package: importlib.metadata.version(package)
                    for package in requirements}
    except Exception as exc:
        raise TypedModelError("TYPED_MODEL_DEPENDENCY_VERSION_UNAVAILABLE") from exc
    if versions != requirements:
        raise TypedModelError("TYPED_MODEL_DEPENDENCY_VERSION_MISMATCH")
    return versions


def _verify_laya_source(root: Path, source_lock: dict[str, Any]) -> dict[str, str]:
    import tarfile

    commit = source_lock.get("commit")
    archive = root / ".local" / "sources" / f"laya-{commit}.tar.gz"
    source_dir = root / ".local" / "sources" / f"laya-pinned-{commit}"
    if (not commit or not archive.is_file()
            or _sha256_file(archive) != source_lock.get("archive_sha256")):
        raise TypedModelError("PINNED_LAYA_SOURCE_ARCHIVE_MISMATCH")
    prefix = f"laya-{commit}/"
    count = 0
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".py"):
                    continue
                relative = member.name[len(prefix):]
                if not relative.startswith("laya/"):
                    continue
                extracted = bundle.extractfile(member)
                target = source_dir / Path(*relative.split("/"))
                if extracted is None or not target.is_file():
                    raise TypedModelError("PINNED_LAYA_SOURCE_FILE_MISSING")
                with extracted:
                    if extracted.read() != target.read_bytes():
                        raise TypedModelError("PINNED_LAYA_SOURCE_TREE_MISMATCH")
                count += 1
    except TypedModelError:
        raise
    except Exception as exc:
        raise TypedModelError("PINNED_LAYA_SOURCE_UNREADABLE") from exc
    if count == 0:
        raise TypedModelError("PINNED_LAYA_SOURCE_EMPTY")
    digest = hashlib.sha256()
    for path in sorted((source_dir / "laya").rglob("*.py")):
        if path.is_symlink():
            continue
        relative = path.relative_to(source_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"commit": commit, "archive_sha256": source_lock["archive_sha256"],
            "verified_python_files": count, "runtime_tree_sha256": digest.hexdigest()}


def _load_cases(root: Path, controls: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dev = _read_dev_cases(root)
    data_path = root / "datasets" / "dev-v3.jsonl"
    raw_by_id = {
        row["case_id"]: row
        for row in (_strict_json(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    templates: dict[str, dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for item in dev:
        raw = raw_by_id.get(item["case_id"])
        if raw is None or raw.get("state") != item["state"]:
            raise TypedModelError("DEV_V3_CASE_JOIN_MISMATCH")
        question_id, question = next(iter(raw["questions"].items()))
        qtype = question["type"]
        templates.setdefault(qtype, copy.deepcopy(question))
        output.append({**item, "question_id": question_id, "question": question,
                       "input_sha256": _json_sha256({"case_id": item["case_id"], "state": item["state"],
                                                     "question": question,
                                                     "allowed_answers": item["allowed_answers"],
                                                     "dataset_sha256": EXPECTED_DATASET_SHA256,
                                                     "contract_sha256": EXPECTED_CONTRACT_SHA256})})
    expected_types = {"choice", "score", "bool"}
    if set(templates) != expected_types:
        raise TypedModelError("DEV_V3_QUESTION_TEMPLATE_SET_INVALID")

    control_rows = controls.get("cases")
    if not isinstance(control_rows, list) or not control_rows:
        raise TypedModelError("TYPED_MODEL_CONTROLS_EMPTY")
    for row in control_rows:
        dimension = row.get("dimension")
        qtype = "bool" if dimension == "noul" else dimension
        question = templates.get(qtype)
        if (qtype not in expected_types or not isinstance(row.get("state"), str)
                or not isinstance(row.get("allowed_answers"), list)
                or not isinstance(question, dict)):
            raise TypedModelError("TYPED_MODEL_CONTROL_INVALID")
        output.append({"case_id": row["case_id"], "dataset": controls["fixture_version"],
                       "control_group": row.get("group"),
                       "dimension": "noul_deadline" if dimension == "noul" else dimension,
                       "state": row["state"], "allowed_answers": row["allowed_answers"],
                       "rationale": row.get("rationale"),
                       "verification_status": controls["verification_status"],
                       "target_errors": [], "expected_components": row.get("expected_components"),
                       "question_id": f"control_{row['case_id']}",
                       "question": copy.deepcopy(question),
                       "input_sha256": _json_sha256({"case_id": row["case_id"], "state": row["state"],
                                                     "question": question,
                                                     "allowed_answers": row["allowed_answers"],
                                                     "dataset_version": controls["fixture_version"],
                                                     "contract_sha256": EXPECTED_CONTRACT_SHA256})})
    ids = [row["case_id"] for row in output]
    if len(ids) != len(set(ids)):
        raise TypedModelError("TYPED_MODEL_CASE_ID_DUPLICATE")
    identity = {"version": "dev-v3", "sha256": EXPECTED_DATASET_SHA256,
                "scoring_contract_sha256": EXPECTED_CONTRACT_SHA256,
                "control_fixture_version": controls["fixture_version"],
                "control_fixture_sha256": _sha256_file(root / CONTROL_DATA),
                "dev_case_count": len(dev), "control_case_count": len(control_rows),
                "labels": "ai_draft_unverified"}
    return output, {"templates": templates, "identity": identity}


def _question_kind(question: dict[str, Any]) -> str:
    kind = question.get("type")
    if kind == "bool":
        return "noul_deadline"
    if kind in {"choice", "score"}:
        return kind
    raise TypedModelError("TYPED_MODEL_QUESTION_TYPE_UNKNOWN")


def _candidate_labels(case: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, str]:
    question = case["question"]
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice" and isinstance(criteria, dict):
        return {str(key): str(value) for key, value in criteria.items()}
    if kind == "score" and isinstance(criteria, list):
        return {str(index): str(value) for index, value in enumerate(criteria)}
    if kind == "bool":
        fixed = evaluation.get("gliner_noul_labels")
        if (not isinstance(fixed, dict) or list(fixed) != ["false", "true"]
                or any(not isinstance(value, str) for value in fixed.values())):
            raise TypedModelError("GLINER_NOUL_LABEL_MAPPING_INVALID")
        return dict(fixed)
    raise TypedModelError("TYPED_MODEL_CANDIDATE_LABELS_INVALID")


def _gliner_task(case: dict[str, Any], evaluation: dict[str, Any],
                 task_id: str = "decision") -> dict[str, Any]:
    labels = _candidate_labels(case, evaluation)
    return {task_id: {"labels": labels,
                      "prompt": case["question"].get("instructions", ""),
                      "multi_label": False}}


def _laya_question(case: dict[str, Any]) -> dict[str, Any]:
    question = copy.deepcopy(case["question"])
    if question.get("type") == "bool":
        question["type"] = "noul"
    return question


def _fixed_device_environment(root: Path) -> Path:
    expected = (root / ".local" / "venv-typed-model-cpu").resolve()
    if os.name != "nt" or Path(sys.prefix).resolve() != expected:
        raise TypedModelError("TYPED_MODEL_DEDICATED_VENV_REQUIRED")
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HOME": str(root / ".local" / "cache" / "typed-model-cpu"),
        "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false",
        "USE_TF": "0",
    })
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    return expected


def _set_torch_cpu(threads: int) -> Any:
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise TypedModelError("CUDA_VISIBLE_OR_DEVICE_NOT_CPU")
    return torch


def _model_parameters(candidate: str, model: Any) -> Any:
    wrapped_model = model.model if candidate == "laya-typed" else model
    return wrapped_model.parameters()


def _load_laya(root: Path, model_dir: Path, spec: dict[str, Any],
               source_lock: dict[str, Any], torch: Any) -> tuple[Any, dict[str, Any]]:
    source = _verify_laya_source(root, source_lock)
    source_dir = root / ".local" / "sources" / f"laya-pinned-{source['commit']}"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    try:
        from laya import load

        agent = load(str(model_dir), device="cpu", backend="eager", compile=False, fast=False,
                     expected_sha256=spec["files"])
    except Exception as exc:
        raise TypedModelError("LAYA_TYPED_OFFICIAL_API_LOAD_FAILED") from exc
    devices = sorted({str(parameter.device) for parameter in agent.model.parameters()})
    dtypes = sorted({str(parameter.dtype) for parameter in agent.model.parameters()
                     if parameter.is_floating_point()})
    if str(agent.device) != "cpu" or devices != ["cpu"] or dtypes != ["torch.float32"]:
        raise TypedModelError("LAYA_TYPED_ACTUAL_DEVICE_OR_DTYPE_MISMATCH")
    runtime = {"loader": "pinned laya.load", "api_version": getattr(__import__("laya"), "__version__", "unknown"),
               "requested_device": "cpu", "actual_agent_device": str(agent.device),
               "actual_parameter_devices": devices, "actual_parameter_dtypes": dtypes,
               "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
               "checkpoint_storage_dtype": spec.get("artifact_encoder_config", {}).get("tensor_storage"),
               "checkpoint_parameter_count": spec.get("artifact_encoder_config", {}).get("checkpoint_parameter_count"),
               "laya_source": source,
               "input_limits": {"max_len": 256, "head_max_len": 192}}
    return agent, runtime


def _load_gliner(root: Path, model_dir: Path, spec: dict[str, Any], torch: Any) -> tuple[Any, dict[str, Any]]:
    from gliner2 import AutoExtractor

    try:
        model = AutoExtractor.from_pretrained(
            str(model_dir), map_location="cpu", local_files_only=True,
        )
        model.to(device="cpu", dtype=torch.float32)
        model.eval()
    except Exception as exc:
        raise TypedModelError("GLINER_BOUNDARY_OFFICIAL_API_LOAD_FAILED") from exc
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()
                     if parameter.is_floating_point()})
    architecture = getattr(model, "architecture", None)
    if architecture != "boundary":
        raise TypedModelError("GLINER_BOUNDARY_ARCHITECTURE_MISMATCH")
    if devices != ["cpu"] or dtypes != ["torch.float32"]:
        raise TypedModelError("GLINER_ACTUAL_DEVICE_OR_DTYPE_MISMATCH")
    config_path = model_dir / "encoder_config" / "config.json"
    encoder_config = _strict_json(config_path.read_text(encoding="utf-8"))
    max_positions = encoder_config.get("max_position_embeddings")
    if type(max_positions) is not int or max_positions <= 0:
        raise TypedModelError("GLINER_ENCODER_POSITION_LIMIT_UNAVAILABLE")
    splitter = getattr(getattr(model, "processor", None), "word_splitter", None)
    splitter_name = getattr(splitter, "__name__", splitter.__class__.__name__ if splitter is not None else "unavailable")
    runtime = {"loader": "gliner2.AutoExtractor.from_pretrained", "package_version": "2.1.0",
               "checkpoint_architecture": architecture, "requested_device": "cpu",
               "actual_parameter_devices": devices, "actual_parameter_dtypes": dtypes,
               "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
               "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
               "encoder_max_position_embeddings": max_positions,
               "word_splitter": splitter_name, "max_len_argument": None,
               "remote_code": False}
    return model, runtime


def _laya_preflight(agent: Any, case: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    question = _laya_question(case)
    qid = str(case["question_id"])
    internal = {qid: agent._to_internal(question)}
    try:
        items = agent._encode_state(case["state"], [qid], internal,
                                    max_len=256, head_max_len=192)
        state_stats = items[0].get("state_stats", {})
        tensor_shapes = {}
        tensor_hashes = {}
        for name, value in items[0].items():
            if hasattr(value, "shape"):
                array = np.asarray(value)
                tensor_shapes[name] = list(array.shape)
                if array.dtype.kind in "biuf" and array.size:
                    tensor_hashes[name] = hashlib.sha256(array.tobytes()).hexdigest()
    except Exception as exc:
        raise TypedModelError("LAYA_INPUT_PREFLIGHT_FAILED") from exc
    truncated = bool(state_stats.get("truncated", False))
    return {"status": "LENGTH_EXCEEDED" if truncated else "PASS",
            "model_input_text_sha256": _json_sha256(case["state"]),
            "state_token_count": state_stats.get("state_tokens"),
            "state_tokens_dropped": state_stats.get("state_tokens_dropped", 0),
            "truncation_performed": truncated,
            "max_len": 256, "head_max_len": 192,
            "input_tensor_shapes": tensor_shapes, "input_tensor_sha256": tensor_hashes,
            "question_id": qid, "question_sha256": _json_sha256(question),
            "candidate_order": list(_candidate_labels(case, evaluation))}


def _ordered_laya_probabilities(kind: str, answer: dict[str, Any], labels: dict[str, str]) -> list[float] | None:
    raw = answer.get("probabilities")
    if not isinstance(raw, dict):
        return None
    ordered_ids = list(labels)
    try:
        return [float(raw[str(key)] if kind == "score" else raw[key]) for key in ordered_ids]
    except (KeyError, TypeError, ValueError):
        return None


def _validate_probability_vector(values: list[float] | None, count: int) -> bool:
    return (values is not None and len(values) == count
            and all(math.isfinite(value) and value >= 0 for value in values)
            and math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=0.003))


def _laya_infer(agent: Any, case: dict[str, Any], evaluation: dict[str, Any], *,
                check_length: bool = True) -> dict[str, Any]:
    import torch

    kind = _question_kind(case["question"])
    labels = _candidate_labels(case, evaluation)
    preflight = _laya_preflight(agent, case, evaluation)
    if check_length and preflight["status"] == "LENGTH_EXCEEDED":
        return {"status": "LENGTH_EXCEEDED", "prediction": None,
                "expected_value": None, "input": preflight,
                "model_specific_output": None, "adapter_output": None}
    api_question = _laya_question(case)
    qid = str(case["question_id"])
    try:
        with torch.inference_mode():
            output = agent.system_one(case["state"], {qid: api_question}, lang=None,
                                     max_len=256, head_max_len=192)
    except Exception as exc:
        return {"status": "FAIL", "prediction": None, "expected_value": None,
                "input": preflight, "error_type": type(exc).__name__,
                "error_message": safe_message(str(exc)), "model_specific_output": None,
                "adapter_output": None}
    if output.get("usage", {}).get("truncated") is not False:
        return {"status": "LENGTH_EXCEEDED", "prediction": None,
                "expected_value": None, "input": preflight,
                "model_specific_output": {"api_usage": output.get("usage")},
                "adapter_output": None}
    answers = output.get("answers")
    answer = answers.get(qid) if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        return {"status": "CONTRACT_ERROR", "prediction": None, "expected_value": None,
                "input": preflight, "model_specific_output": {"api_usage": output.get("usage")},
                "adapter_output": None}
    if kind == "choice":
        prediction = answer.get("choice")
        values = _ordered_laya_probabilities(kind, answer, labels)
        valid = (prediction in labels and _validate_probability_vector(values, len(labels)))
        adapter = {"decision": prediction, "candidate_ids": list(labels),
                   "api_probabilities_in_candidate_order": values,
                   "probability_origin": "official Laya API; rounded display values"}
        expected_value = None
    elif kind == "score":
        values = _ordered_laya_probabilities(kind, answer, labels)
        prediction = max(range(len(values)), key=values.__getitem__) if _validate_probability_vector(values, len(labels)) else None
        returned_score = answer.get("score")
        valid = (prediction is not None and isinstance(returned_score, (int, float))
                 and not isinstance(returned_score, bool) and math.isfinite(float(returned_score)))
        adapter = {"stage_prediction": prediction,
                   "stage_prediction_source": "argmax of official API's ordered rounded stage probabilities",
                   "argmax_tie": bool(values is not None and sum(value == max(values) for value in values) > 1),
                   "candidate_ids": list(labels), "api_stage_probabilities_in_candidate_order": values,
                   "returned_continuous_score": returned_score,
                   "continuous_score_source": "official Laya API"}
        expected_value = float(returned_score) if valid else None
    else:
        returned = answer.get("noul")
        valid = (isinstance(returned, (int, float)) and not isinstance(returned, bool)
                 and math.isfinite(float(returned)) and 0 <= float(returned) <= 1)
        prediction = float(returned) >= 0.5 if valid else None
        adapter = {"decision": prediction, "p_true_api_returned_rounded": returned,
                   "boolean_rule": "API return >= 0.5", "probability_origin": "official Laya API"}
        expected_value = None
    return {"status": "PASS" if valid else "CONTRACT_ERROR", "prediction": prediction,
            "expected_value": expected_value, "input": preflight,
            "model_specific_output": {"official_api_answer": answer,
                                      "api_usage": output.get("usage")},
            "adapter_output": adapter}


def _gliner_preflight(model: Any, case: dict[str, Any], evaluation: dict[str, Any],
                      max_positions: int) -> dict[str, Any]:
    import numpy as np

    kind = _question_kind(case["question"])
    task = _gliner_task(case, evaluation)
    try:
        schema = model._classification_schema(task)
        schema_dicts, _ = model._build_schema_dicts_and_metadata([schema])
        record = model.processor.transform_record(case["state"], schema_dicts[0], max_len=None)
        input_ids = [int(value) for value in record.input_ids]
        input_length = len(input_ids)
        state_words = len(list(model.processor.word_splitter(record.text, lower=True)))
        input_hash = hashlib.sha256(np.asarray(input_ids, dtype=np.int64).tobytes()).hexdigest()
        normalized_text = record.text
    except Exception as exc:
        raise TypedModelError("GLINER_INPUT_PREFLIGHT_FAILED") from exc
    too_long = input_length > max_positions
    return {"status": "LENGTH_EXCEEDED" if too_long else "PASS",
            "raw_state_sha256": _json_sha256(case["state"]),
            "state_passed_unchanged": case["state"],
            "processor_normalized_state": normalized_text,
            "state_word_count_by_official_splitter": state_words,
            "model_input_word_token_count_including_schema": len(record.text_tokens),
            "encoded_input_id_count_including_schema": input_length,
            "encoder_max_position_embeddings": max_positions,
            "truncation_performed": False,
            "processor_input_ids_sha256": input_hash,
            "question_prompt": case["question"].get("instructions", ""),
            "question_sha256": _json_sha256(case["question"]),
            "labels_in_order": list(_candidate_labels(case, evaluation)),
            "task_schema_sha256": _json_sha256(task), "max_len_argument": None,
            "word_splitter": getattr(getattr(model.processor, "word_splitter", None),
                                      "__name__", "checkpoint default")}


def _gliner_selected_result(raw: Any) -> tuple[str | None, float | None]:
    if isinstance(raw, dict):
        label = raw.get("label")
        confidence = raw.get("confidence")
    elif isinstance(raw, (tuple, list)) and len(raw) == 2:
        label, confidence = raw
    else:
        return None, None
    if not isinstance(label, str):
        return None, None
    if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1):
        return label, None
    return label, float(confidence)


def _gliner_infer(model: Any, case: dict[str, Any], evaluation: dict[str, Any],
                  max_positions: int, *, check_length: bool = True,
                  task_id: str = "decision") -> dict[str, Any]:
    kind = _question_kind(case["question"])
    labels = _candidate_labels(case, evaluation)
    task = _gliner_task(case, evaluation, task_id)
    try:
        preflight = _gliner_preflight(model, case, evaluation, max_positions)
    except TypedModelError as exc:
        return {"status": "CONTRACT_ERROR", "prediction": None,
                "expected_value": None, "input": None,
                "error_code": exc.code, "model_specific_output": None,
                "adapter_output": None}
    if check_length and preflight["status"] == "LENGTH_EXCEEDED":
        return {"status": "LENGTH_EXCEEDED", "prediction": None,
                "expected_value": None, "input": preflight,
                "model_specific_output": None, "adapter_output": None}
    try:
        raw = model.classify_text(case["state"], task, include_confidence=True,
                                  format_results=True, max_len=None)
    except Exception as exc:
        return {"status": "FAIL", "prediction": None, "expected_value": None,
                "input": preflight, "error_type": type(exc).__name__,
                "error_message": safe_message(str(exc)), "model_specific_output": None,
                "adapter_output": None}
    returned = raw.get(task_id) if isinstance(raw, dict) else None
    label, confidence = _gliner_selected_result(returned)
    valid = label in labels and confidence is not None
    prediction: Any = None
    expected_value = None
    if valid:
        if kind == "choice":
            prediction = label
        elif kind == "score":
            try:
                prediction = int(label)
                valid = str(prediction) == label and prediction in (0, 1, 2)
            except ValueError:
                valid = False
        else:
            prediction = label == "true" if label in {"true", "false"} else None
            valid = prediction is not None
    adapter = {"decision": prediction, "selected_candidate_id": label,
               "selected_label_confidence_not_calibrated": confidence,
               "full_candidate_probabilities": "unavailable from official public formatter",
               "p_true": "unavailable from official public formatter" if kind == "noul_deadline" else None,
               "continuous_score": "not returned by official API" if kind == "score" else None}
    return {"status": "PASS" if valid else "CONTRACT_ERROR", "prediction": prediction,
            "expected_value": expected_value, "input": preflight,
            "model_specific_output": {"official_api_result": raw,
                                      "selected_label_confidence_semantics": "official formatter value; not calibrated"},
            "adapter_output": adapter}


def _case_row(case: dict[str, Any], inference: dict[str, Any], candidate: str) -> dict[str, Any]:
    prediction = inference.get("prediction")
    adapter = inference.get("adapter_output") or {}
    score_value_source = adapter.get("continuous_score_source")
    allowed = case.get("allowed_answers", [])
    errors = []
    if inference.get("status") == "PASS" and prediction not in allowed:
        errors = [str(value) for value in _candidate_output_ids(case) if value not in allowed]
    return {**case, "model_id": candidate, "status": inference.get("status"),
            "common_prediction": prediction,
            "score_value_for_metric": inference.get("expected_value"),
            "score_value_source": score_value_source or "not returned by official API",
            "score_api_returned_continuous_value": (
                adapter.get("returned_continuous_score")
                if score_value_source == "official Laya API" else None
            ),
            "draft_label_match": prediction in allowed if inference.get("status") == "PASS" else None,
            "human_quality_status": "UNDETERMINED",
            "model_specific_output": inference.get("model_specific_output"),
            "adapter_output": inference.get("adapter_output"),
            "input_trace": inference.get("input"),
            "contract_or_execution_error": {
                key: inference[key] for key in ("error_type", "error_message", "error_code") if key in inference
            },
            "observable_wrong_outputs_derived_from_allowed_answers": errors,
            "e2e_time_ms": inference.get("e2e_time_ms"),
            "process_cpu_time_ms": inference.get("process_cpu_time_ms"),
            "case_input_sha256": case["input_sha256"]}


def _candidate_output_ids(case: dict[str, Any]) -> list[Any]:
    kind = _question_kind(case["question"])
    if kind == "choice":
        return list(case["question"].get("criteria", {}))
    if kind == "score":
        return list(range(len(case["question"].get("criteria", []))))
    return [False, True]


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {name: sum(row.get("status") == name for row in rows)
            for name in ("PASS", "HOLD", "LENGTH_EXCEEDED", "CONTRACT_ERROR", "FAIL")}


def _metrics_by_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        result[dataset] = _all_metrics(group)
    return result


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _timed_workload(call: Callable[[], Any], repeats: int, warmups: int) -> dict[str, Any]:
    warmup_failures = 0
    warmup_failure_codes: dict[str, int] = {}
    for _ in range(warmups):
        try:
            call()
        except Exception as exc:
            warmup_failures += 1
            code = exc.code if isinstance(exc, TypedModelError) else type(exc).__name__
            warmup_failure_codes[code] = warmup_failure_codes.get(code, 0) + 1
    times: list[float] = []
    failures = 0
    failure_codes: dict[str, int] = {}
    for _ in range(repeats):
        started = time.perf_counter()
        try:
            call()
        except Exception as exc:
            failures += 1
            code = exc.code if isinstance(exc, TypedModelError) else type(exc).__name__
            failure_codes[code] = failure_codes.get(code, 0) + 1
        else:
            times.append((time.perf_counter() - started) * 1000.0)
    return {"attempts": repeats, "warmup_attempts": warmups,
            "warmup_failures": warmup_failures, "warmup_failure_codes": warmup_failure_codes,
            "failures": failures, "failure_codes": failure_codes,
            "successes": len(times), "e2e_p50_ms": _percentile(times, 0.50),
            "e2e_p95_ms": _percentile(times, 0.95),
            "e2e_min_ms": min(times) if times else None,
            "e2e_max_ms": max(times) if times else None,
            "samples_ms": times}


def _performance_calls(candidate: str, model: Any, cases: list[dict[str, Any]],
                       evaluation: dict[str, Any], max_positions: int | None = None) -> dict[str, Callable[[], Any]]:
    first_by_kind = {}
    for case in cases:
        kind = _question_kind(case["question"])
        if kind not in first_by_kind and case["dataset"] == "dev-v3":
            first_by_kind[kind] = case
    if set(first_by_kind) != {"choice", "score", "noul_deadline"}:
        raise TypedModelError("PERFORMANCE_FIXED_INPUTS_MISSING")
    calls: dict[str, Callable[[], Any]] = {}
    if candidate == "laya-typed":
        for kind, case in first_by_kind.items():
            question = _laya_question(case)
            qid = str(case["question_id"])
            calls[kind] = lambda state=case["state"], qid=qid, question=question: model.system_one(
                state, {qid: question}, lang=None, max_len=256, head_max_len=192)
        bundle = [(_laya_question(first_by_kind[kind]), str(first_by_kind[kind]["question_id"]))
                  for kind in ("choice", "score", "noul_deadline")]
        calls["three_type_bundle"] = lambda: model.system_one(
            first_by_kind["choice"]["state"], {qid: question for question, qid in bundle},
            lang=None, max_len=256, head_max_len=192)
    else:
        for kind, case in first_by_kind.items():
            tasks = _gliner_task(case, evaluation)
            calls[kind] = lambda state=case["state"], tasks=tasks: model.classify_text(
                state, tasks, include_confidence=True, format_results=True, max_len=None)
        bundle_tasks = {}
        bundle_state = first_by_kind["choice"]["state"]
        for kind in ("choice", "score", "noul_deadline"):
            bundle_tasks.update(_gliner_task(first_by_kind[kind], evaluation, task_id=kind))
        calls["three_type_bundle"] = lambda: model.classify_text(
            bundle_state, bundle_tasks, include_confidence=True, format_results=True, max_len=None)
    return calls


def _load_horizon_baseline(root: Path, dev_cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run = open_run(root, HORIZON_ANALYSIS_RUN_ID)
    manifest = run.read_manifest()
    stage = manifest.get("stages", {}).get("ZERO_SHOT_ANALYSIS", {})
    result_path = run.path / "analysis-result.json"
    rows_path = run.path / "analysis-cases.jsonl"
    if (manifest.get("status") != "COMPLETE" or stage.get("status") != "PASS"
            or not result_path.is_file() or not rows_path.is_file()
            or _sha256_file(result_path) != stage.get("analysis_result_sha256")
            or _sha256_file(rows_path) != stage.get("analysis_cases_sha256")):
        raise TypedModelError("HORIZON_SAVED_ANALYSIS_IDENTITY_MISMATCH")
    result = _strict_json(result_path.read_text(encoding="utf-8"))
    if (result.get("analysis_kind") != "saved-output reanalysis only; no model loaded and no inference performed"
            or result.get("candidate_a_revision") != "3fe1f0c48b589cf6220fd21a628a1dd56d3d41f8"
            or result.get("dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256
            or result.get("dataset", {}).get("scoring_contract_sha256") != EXPECTED_CONTRACT_SHA256
            or "english" not in result.get("conditions", {})):
        raise TypedModelError("HORIZON_SAVED_ANALYSIS_FINGERPRINT_MISMATCH")
    raw_rows = [_strict_json(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in raw_rows if row.get("condition") == "english" and row.get("dataset") == "dev-v3"]
    expected = {case["case_id"]: case for case in dev_cases if case["dataset"] == "dev-v3"}
    if len(selected) != 60 or {row.get("case_id") for row in selected} != set(expected):
        raise TypedModelError("HORIZON_SAVED_CASE_SET_MISMATCH")
    rows = []
    for row in selected:
        case = expected[row["case_id"]]
        if row.get("state_sha256") != hashlib.sha256(case["state"].encode("utf-8")).hexdigest():
            raise TypedModelError("HORIZON_SAVED_INPUT_MISMATCH")
        prediction = row.get("official_pipeline_prediction")
        rows.append({"case_id": row["case_id"], "dataset": "dev-v3", "dimension": row["dimension"],
                     "state": row["state"], "allowed_answers": row["allowed_answers"],
                     "verification_status": row.get("verification_status"),
                     "status": "PASS" if prediction is not None else row.get("official_output_status", "HOLD"),
                     "common_prediction": prediction,
                     "score_expected_value_not_api_returned_score": row.get("score_expected_value_not_api_returned_score"),
                     "model_specific_output": row.get("model_specific_output")})
    identity = {"run_id": HORIZON_ANALYSIS_RUN_ID, "condition": "english (model-card recommended labels/template)",
                "analysis_result_sha256": _sha256_file(result_path),
                "analysis_cases_sha256": _sha256_file(rows_path),
                "candidate_revision": result["candidate_a_revision"]}
    return rows, identity


def _saved_baselines(root: Path, dev_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_rows, baseline_identity = _baseline_rows(root)
    baseline_metrics = {name: _remove_unavailable_score_subgroups(_baseline_metrics(dev_rows, rows))
                        for name, rows in baseline_rows.items()}
    horizon_rows, horizon_identity = _load_horizon_baseline(root, dev_rows)
    baseline_metrics["horizon_english_official"] = _remove_unavailable_score_subgroups(_all_metrics(horizon_rows))
    majority = _majority_baselines(dev_rows)
    return {"metrics": baseline_metrics, "majority": majority,
            "laya_sokudan_identity": baseline_identity,
            "horizon_identity": horizon_identity,
            "horizon_rows": horizon_rows,
            "laya_rows": baseline_rows["laya"], "sokudan_rows": baseline_rows["sokudan"]}, baseline_identity


def _remove_unavailable_score_subgroups(metrics: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(metrics)
    score = cleaned.get("score")
    if isinstance(score, dict):
        score.pop("subgroups_with_explicit_component_targets", None)
        score["date_clock_component_subgroups"] = "Not recoverable from the saved model outputs."
    return cleaned


def _stage_case_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {dataset: _all_metrics([row for row in rows if row.get("dataset") == dataset])
              for dataset in sorted({row.get("dataset") for row in rows})}
    groups = sorted({row["control_group"] for row in rows if row.get("control_group")})
    result["control_groups"] = {group: _all_metrics(
        [row for row in rows if row.get("control_group") == group]) for group in groups}
    for dataset in ("dev-v3", "typed-model-controls-v1"):
        score = result.get(dataset, {}).get("score")
        if score is not None:
            score.pop("subgroups_with_explicit_component_targets", None)
            score["date_clock_component_subgroups"] = (
                "dev-v3 has no per-case component labels."
                if dataset == "dev-v3" else
                "Fixed controls have component labels, but this model returns a joint stage only."
            )
            score_rows = [row for row in rows
                          if row.get("dataset") == dataset and row.get("dimension") == "score"
                          and row.get("status") == "PASS"]
            sources = sorted({
                row.get("score_value_source")
                or (row.get("adapter_output") or {}).get("continuous_score_source")
                or "not returned by official API"
                for row in score_rows
            })
            score["score_value_source"] = sources[0] if len(sources) == 1 else sources
            score["score_expectation_is_model_api_score"] = (
                bool(score_rows) and all(source == "official Laya API" for source in sources)
            )
            score["score_api_returned_continuous_value_count"] = sum(
                (row.get("score_api_returned_continuous_value") is not None
                 or (row.get("adapter_output") or {}).get("returned_continuous_score") is not None)
                for row in score_rows
            )
    return result


def run_child(root: Path, run_id: str, candidate: str, trial: int) -> dict[str, Any]:
    run = open_run(root, run_id)
    phase = "VALIDATE_CONFIGURATION"
    inference_started = False
    try:
        if trial not in (1, 2, 3):
            raise TypedModelError("TYPED_MODEL_TRIAL_INVALID")
        _fixed_device_environment(root)
        lock, evaluation, controls = _read_config(root)
        spec = _candidate_spec(lock, candidate)
        model_dir, verified_files = _verify_model_files(root, candidate, spec)
        gliner_source = _verify_gliner_wheel(root, lock) if candidate == "gliner25-multi-decide" else None
        dependencies = _expected_dependencies(root)
        cases, case_meta = _load_cases(root, controls)
        baseline_bundle, _ = _saved_baselines(root, [case for case in cases if case["dataset"] == "dev-v3"])
        import torch

        torch = _set_torch_cpu(evaluation["primary_cpu_condition"]["torch_threads"])
        memory_before = _memory_snapshot()
        load_started = time.perf_counter()
        phase = "LOAD_MODEL"
        if candidate == "laya-typed":
            model, execution = _load_laya(root, model_dir, spec, lock["laya_api_source"], torch)
            max_positions = None
        else:
            model, execution = _load_gliner(root, model_dir, spec, torch)
            max_positions = execution["encoder_max_position_embeddings"]
        model_load_seconds = time.perf_counter() - load_started
        memory_after_load = _memory_snapshot()
        if execution.get("actual_parameter_devices") != ["cpu"]:
            raise TypedModelError("TYPED_MODEL_NOT_CPU")
        if execution.get("actual_parameter_dtypes") != ["torch.float32"]:
            raise TypedModelError("TYPED_MODEL_NOT_FP32")

        def infer(case: dict[str, Any], *, preflight: bool = True) -> dict[str, Any]:
            nonlocal inference_started, phase
            output = (_laya_infer(model, case, evaluation, check_length=preflight)
                      if candidate == "laya-typed"
                      else _gliner_infer(model, case, evaluation, max_positions, check_length=preflight))
            if output.get("model_specific_output") is not None or output.get("status") == "FAIL":
                inference_started = True
                phase = "MODEL_INFERENCE"
            return output

        performance_calls = _performance_calls(candidate, model, cases, evaluation, max_positions)
        fixed_choice = next(case for case in cases if case["dataset"] == "dev-v3"
                            and _question_kind(case["question"]) == "choice")
        first_result = infer(fixed_choice)
        memory_after_first = _memory_snapshot()
        idle_seconds = evaluation["primary_cpu_condition"]["idle_wait_seconds"]
        idle_cpu_before = time.process_time()
        idle_wall_before = time.perf_counter()
        time.sleep(idle_seconds)
        idle_wall_seconds = time.perf_counter() - idle_wall_before
        idle_cpu_seconds = time.process_time() - idle_cpu_before
        memory_after_idle = _memory_snapshot()

        perf_spec = evaluation["primary_cpu_condition"]
        phase = "PERFORMANCE_MEASUREMENT"
        timings = {
            name: _timed_workload(call, perf_spec["performance_repeats_per_workload"],
                                  perf_spec["performance_warmups"])
            for name, call in performance_calls.items()
        }
        memory_after_repeat = _memory_snapshot()
        run_rows = []
        phase = "QUALITY_CASE_EVALUATION"
        for case in cases:
            started_wall = time.perf_counter()
            started_cpu = time.process_time()
            inference = infer(case)
            inference["e2e_time_ms"] = (time.perf_counter() - started_wall) * 1000.0
            inference["process_cpu_time_ms"] = (time.process_time() - started_cpu) * 1000.0
            run_rows.append(_case_row(case, inference, candidate))
        memory_before_exit = _memory_snapshot()
        metrics = _stage_case_metrics(run_rows)
        dev_metrics = metrics["dev-v3"]
        comparisons = {name: {dimension: values[dimension]
                              for dimension in ("choice", "score", "noul_deadline")}
                       for name, values in baseline_bundle["metrics"].items()}
        model_bytes = sum(record["size_bytes"] for record in verified_files.values())
        phase = "BUILD_RESULT"
        result = {
            "run_id": run.run_id,
            "candidate": candidate,
            "trial": trial,
            "run_status_semantics": "Run COMPLETE means collection finished; quality is not passed by this status.",
            "quality_status": "UNDETERMINED",
            "label_state": "ai_draft_unverified",
            "model": {"requested_repository": spec.get("requested_repository", spec.get("repository")),
                      "resolved_repository": spec.get("resolved_repository", spec.get("repository")),
                      "revision": spec["revision"], "card_license": spec.get("card_license"),
                      "files": verified_files, "total_locked_bytes": model_bytes,
                      "artifact_metadata_conflict": spec.get("artifact_encoder_config", {}).get("metadata_conflict"),
                      "known_limitations": spec.get("known_limitations", []),
                      "file_lock_sha256": _sha256_file(root / MODEL_LOCK)},
            "runtime_dependencies": dependencies,
            "dependency_lock_sha256": _sha256_file(root / "requirements" / "typed-model-cpu-win-x64.lock"),
            "dependency_input_sha256": _sha256_file(root / "requirements" / "typed-model-cpu-win-x64.in"),
            "inference_library_source": gliner_source,
            "execution": {**execution, "requested_device": "cpu", "dtype": "float32",
                          "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                          "torch_cuda_available": torch.cuda.is_available(),
                          "model_hub_access": "offline", "local_only_load": True,
                          "actual_parameter_count": int(sum(parameter.numel()
                                                            for parameter in _model_parameters(candidate, model)))},
            "dataset": case_meta["identity"],
            "baseline_identity": {"laya": baseline_bundle["laya_sokudan_identity"],
                                  "horizon": baseline_bundle["horizon_identity"]},
            "case_counts": {"total": len(run_rows), "dev_v3": 60,
                            "fixed_controls": len(run_rows) - 60},
            "status_counts": _status_counts(run_rows),
            "timing": {"model_load_seconds": model_load_seconds,
                       "quality_case_e2e_ms": {
                           "p50": _percentile([row["e2e_time_ms"] for row in run_rows if row.get("e2e_time_ms") is not None], 0.50),
                           "p95": _percentile([row["e2e_time_ms"] for row in run_rows if row.get("e2e_time_ms") is not None], 0.95),
                           "note": "Quality-set timings are descriptive only; performance p95 comes from repeated fixed workloads."},
                       "performance_workloads": timings,
                       "performance_input_case_ids": {name: fixed_choice["case_id"] if name == "three_type_bundle" else
                                                       next(case["case_id"] for case in cases if case["dataset"] == "dev-v3"
                                                            and _question_kind(case["question"]) == name)
                                                       for name in timings},
                       "warmup_count": perf_spec["performance_warmups"],
                       "repeats_per_workload": perf_spec["performance_repeats_per_workload"]},
            "memory": {"api": "Windows GetProcessMemoryInfo(PROCESS_MEMORY_COUNTERS_EX)",
                       "stages": {"before_model_load": memory_before,
                                  "after_model_load": memory_after_load,
                                  "after_one_fixed_inference": memory_after_first,
                                  "after_idle_wait": memory_after_idle,
                                  "after_repeated_inference": memory_after_repeat,
                                  "before_process_exit": memory_before_exit},
                       "idle_observation": {"requested_seconds": idle_seconds,
                                            "actual_wall_seconds": idle_wall_seconds,
                                            "process_cpu_time_delta_seconds": idle_cpu_seconds,
                                            "background_inference": False}},
            "input_policy": {"truncation_forbidden": True,
                             "length_exceeded_not_inferred": True,
                             "gliner_max_position_limit": max_positions,
                             "gliner_default_word_splitter": execution.get("word_splitter")},
            "metrics_by_dataset": metrics,
            "comparison_to_saved_models_dev_v3": comparisons,
            "same_dev_v3_draft_majority_baselines": baseline_bundle["majority"],
            "per_case_outputs": "typed-model-cases.jsonl",
            "inference_failures_are_case_level": True,
            "model_adoption": "NOT_DECIDED",
        }
        run.write_jsonl("typed-model-cases.jsonl", run_rows)
        phase = "SAVE_RESULTS"
        run.write_json("typed-model-result.json", result)
        run.stage(RUN_STAGE, "PASS", candidate=candidate, trial=trial,
                  model_revision=spec["revision"], model_files_sha256={name: value["sha256"] for name, value in verified_files.items()},
                  evaluation_config_sha256=_sha256_file(root / EVAL_CONFIG),
                  control_data_sha256=_sha256_file(root / CONTROL_DATA),
                  dataset_sha256=EXPECTED_DATASET_SHA256,
                  scoring_contract_sha256=EXPECTED_CONTRACT_SHA256,
                  case_count=len(run_rows), status_counts=result["status_counts"],
                  result_sha256=_sha256_file(run.path / "typed-model-result.json"),
                  cases_sha256=_sha256_file(run.path / "typed-model-cases.jsonl"),
                  official_japanese_quality_status="UNDETERMINED")
        return {"status": "COMPLETE", "run_id": run.run_id, "candidate": candidate,
                "trial": trial, "status_counts": result["status_counts"]}
    except Exception as exc:
        code = exc.code if isinstance(exc, TypedModelError) else "TYPED_MODEL_TRIAL_FAILED"
        cause = exc.__cause__
        details = {"underlying_error_type": type(cause).__name__ if cause is not None else None,
                   "underlying_error_message": safe_message(str(cause)) if cause is not None else None}
        try:
            run.write_json("typed-model-diagnostic.json", {
                "run_id": run.run_id, "candidate": candidate, "trial": trial,
                "status": "FAILED", "code": code, "error_type": type(exc).__name__,
                "message": safe_message(str(exc)), "failure_phase": phase,
                "model_inference_performed": inference_started,
                **details
            })
        except FileExistsError:
            pass
        run.failure(RUN_STAGE, code, RuntimeError(safe_message(str(exc))),
                    candidate=candidate, trial=trial, failure_phase=phase,
                    model_inference_performed=inference_started,
                    **details)
        return {"status": "FAILED", "run_id": run.run_id, "candidate": candidate,
                "trial": trial, "code": code}


def _memory_snapshot() -> dict[str, Any]:
    from .performance import _memory

    return _memory()


def _load_saved_trial(root: Path, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = open_run(root, run_id)
    manifest = run.read_manifest()
    result_path = run.path / "typed-model-result.json"
    cases_path = run.path / "typed-model-cases.jsonl"
    if (manifest.get("status") != "COMPLETE"
            or manifest.get("stages", {}).get(RUN_STAGE, {}).get("status") != "PASS"
            or not result_path.is_file() or not cases_path.is_file()):
        raise TypedModelError("TYPED_MODEL_TRIAL_RESULT_NOT_COMPLETE")
    result = _strict_json(result_path.read_text(encoding="utf-8"))
    rows = [_strict_json(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stage = manifest["stages"][RUN_STAGE]
    if (_sha256_file(result_path) != stage.get("result_sha256")
            or _sha256_file(cases_path) != stage.get("cases_sha256")
            or result.get("run_id") != run_id
            or result.get("model", {}).get("revision") != stage.get("model_revision")
            or len(rows) != 82):
        raise TypedModelError("TYPED_MODEL_TRIAL_RESULT_FINGERPRINT_MISMATCH")
    return result, rows


def analyze_saved_trials(root: Path, laya_run_ids: list[str], gliner_run_ids: list[str]) -> dict[str, Any]:
    if (len(laya_run_ids) != TRIAL_COUNT or len(gliner_run_ids) != TRIAL_COUNT
            or len(set(laya_run_ids + gliner_run_ids)) != TRIAL_COUNT * 2):
        raise TypedModelError("TYPED_MODEL_SAVED_ANALYSIS_RUN_SELECTION_INVALID")
    lock, _, controls = _read_config(root)
    cases, case_meta = _load_cases(root, controls)
    if len(cases) != 82:
        raise TypedModelError("TYPED_MODEL_CASE_COUNT_INVALID")
    expected_cases = {case["case_id"]: case for case in cases}
    dev_cases = [case for case in cases if case["dataset"] == "dev-v3"]
    baselines, _ = _saved_baselines(root, dev_cases)
    selected = {"laya-typed": laya_run_ids,
                "gliner25-multi-decide": gliner_run_ids}
    source_records: dict[str, list[dict[str, Any]]] = {}
    for candidate, run_ids in selected.items():
        spec = _candidate_spec(lock, candidate)
        expected_model_files = spec.get("files", {})
        records = []
        trial_rows = []
        for expected_trial, run_id in enumerate(run_ids, start=1):
            result, rows = _load_saved_trial(root, run_id)
            source_run = open_run(root, run_id)
            source_manifest = source_run.read_manifest()
            source_stage = source_manifest.get("stages", {}).get(RUN_STAGE, {})
            if (result.get("candidate") != candidate or result.get("trial") != expected_trial
                    or result.get("model", {}).get("revision") != spec["revision"]
                    or result.get("dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256
                    or result.get("dataset", {}).get("scoring_contract_sha256") != EXPECTED_CONTRACT_SHA256
                    or result.get("model", {}).get("file_lock_sha256") != _sha256_file(root / MODEL_LOCK)
                    or {name: record.get("sha256")
                        for name, record in result.get("model", {}).get("files", {}).items()}
                    != expected_model_files
                    or result.get("dependency_lock_sha256")
                    != _sha256_file(root / "requirements" / "typed-model-cpu-win-x64.lock")
                    or result.get("dependency_input_sha256")
                    != _sha256_file(root / "requirements" / "typed-model-cpu-win-x64.in")
                    or result.get("dataset", {}).get("control_fixture_sha256") != _sha256_file(root / CONTROL_DATA)
                    or source_stage.get("evaluation_config_sha256") != _sha256_file(root / EVAL_CONFIG)
                    or source_stage.get("control_data_sha256") != _sha256_file(root / CONTROL_DATA)
                    or source_stage.get("dataset_sha256") != EXPECTED_DATASET_SHA256
                    or source_stage.get("scoring_contract_sha256") != EXPECTED_CONTRACT_SHA256):
                raise TypedModelError("TYPED_MODEL_SAVED_ANALYSIS_FINGERPRINT_MISMATCH")
            if {row.get("case_id") for row in rows} != set(expected_cases):
                raise TypedModelError("TYPED_MODEL_SAVED_ANALYSIS_CASE_SET_MISMATCH")
            for row in rows:
                current = expected_cases[row["case_id"]]
                if (row.get("case_input_sha256") != current.get("input_sha256")
                        or row.get("dataset") != current.get("dataset")
                        or row.get("allowed_answers") != current.get("allowed_answers")):
                    raise TypedModelError("TYPED_MODEL_SAVED_ANALYSIS_INPUT_MISMATCH")
            records.append({"run_id": run_id, "trial": expected_trial, "status": "COMPLETE"})
            trial_rows.append(rows)
        signatures = [_prediction_signature(rows) for rows in trial_rows]
        if not all(signature == signatures[0] for signature in signatures[1:]):
            raise TypedModelError("TYPED_MODEL_SAVED_ANALYSIS_TRIAL_OUTPUTS_DIFFER")
        source_records[candidate] = records

    summary_run = Run.create(root, "saved typed-model CPU output analysis; no model loaded or inference run",
                             dataset_version="dev-v3", profile_applicable=False)
    try:
        candidate_results = {
            candidate: _summarize_candidate(root, candidate, records, baselines)
            for candidate, records in source_records.items()
        }
        candidate_summaries = {
            candidate: {key: value for key, value in result.items() if key != "cases_from_trial_1"}
            for candidate, result in candidate_results.items()
        }
        output_rows = [row for result in candidate_results.values()
                       for row in result["cases_from_trial_1"]]
        sources = {}
        for candidate, records in source_records.items():
            sources[candidate] = []
            for record in records:
                run = open_run(root, record["run_id"])
                manifest = run.read_manifest()
                stage = manifest["stages"][RUN_STAGE]
                sources[candidate].append({
                    "run_id": record["run_id"],
                    "git_revision": manifest.get("git_revision"),
                    "source_tree_sha256": manifest.get("source_tree_sha256"),
                    "result_sha256": stage.get("result_sha256"),
                    "cases_sha256": stage.get("cases_sha256"),
                })
        result = {
            "analysis_run_id": summary_run.run_id,
            "analysis_kind": "saved-output reanalysis only; no model loaded and no inference performed",
            "status": "COMPLETE",
            "quality_status": "UNDETERMINED",
            "label_state": "ai_draft_unverified",
            "model_adoption": "NOT_DECIDED",
            "dataset": case_meta["identity"],
            "model_lock_sha256": _sha256_file(root / MODEL_LOCK),
            "evaluation_config_sha256": _sha256_file(root / EVAL_CONFIG),
            "control_data_sha256": _sha256_file(root / CONTROL_DATA),
            "scoring_contract_sha256": EXPECTED_CONTRACT_SHA256,
            "source_runs": sources,
            "saved_baselines": {
                "metrics": baselines["metrics"],
                "same_dev_v3_draft_majority": baselines["majority"],
                "identity": {"laya_sokudan": baselines["laya_sokudan_identity"],
                             "horizon": baselines["horizon_identity"]},
            },
            "candidate_trials": candidate_summaries,
            "comparison_metrics_semantics": {
                "Score": "stage metrics use the fixed label contract; continuous-value MAE is computed only from the value actually returned by an official API.",
                "Laya Typed Decisions Score": "official API returned continuous Score; source is recorded per case and metric.",
                "GLiNER2.5 Score": "official API returns selected label and confidence only; no continuous Score is synthesized.",
                "confidence": "model-specific confidence is not cross-model calibrated",
                "quality": "preliminary only; all dev-v3 labels remain unverified human drafts",
            },
            "cases_file": "typed-model-saved-analysis-cases.jsonl",
        }
        summary_run.write_jsonl("typed-model-saved-analysis-cases.jsonl", output_rows)
        summary_run.write_json("typed-model-saved-analysis-result.json", result)
        summary_run.stage(
            ANALYSIS_STAGE, "PASS",
            source_run_ids={candidate: [record["run_id"] for record in records]
                            for candidate, records in source_records.items()},
            source_result_hashes={candidate: [entry["result_sha256"] for entry in entries]
                                  for candidate, entries in sources.items()},
            dataset_sha256=EXPECTED_DATASET_SHA256,
            scoring_contract_sha256=EXPECTED_CONTRACT_SHA256,
            result_sha256=_sha256_file(summary_run.path / "typed-model-saved-analysis-result.json"),
            cases_sha256=_sha256_file(summary_run.path / "typed-model-saved-analysis-cases.jsonl"),
            official_japanese_quality_status="UNDETERMINED")
        return {"status": "COMPLETE", "analysis_run_id": summary_run.run_id,
                "source_run_ids": {candidate: [record["run_id"] for record in records]
                                   for candidate, records in source_records.items()},
                "quality_status": "UNDETERMINED"}
    except Exception as exc:
        code = exc.code if isinstance(exc, TypedModelError) else "TYPED_MODEL_ANALYSIS_FAILED"
        summary_run.failure(ANALYSIS_STAGE, code, RuntimeError(safe_message(str(exc))))
        return {"status": "FAILED", "analysis_run_id": summary_run.run_id, "code": code}


def _prediction_signature(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {row["case_id"]: _json_sha256({"prediction": row.get("common_prediction"),
                                         "status": row.get("status"),
                                         "output": row.get("model_specific_output")})
            for row in rows}


def _summarize_candidate(root: Path, candidate: str, runs: list[dict[str, Any]],
                         baselines: dict[str, Any]) -> dict[str, Any]:
    complete = []
    for record in runs:
        if record.get("status") != "COMPLETE":
            continue
        result, rows = _load_saved_trial(root, record["run_id"])
        complete.append({"run_id": record["run_id"], "trial": result["trial"],
                         "result": result, "rows": rows})
    signatures = [_prediction_signature(entry["rows"]) for entry in complete]
    same = len(signatures) == TRIAL_COUNT and all(signature == signatures[0] for signature in signatures[1:])
    per_trial_metrics = {entry["run_id"]: _stage_case_metrics(entry["rows"])
                         for entry in complete}
    baseline_metrics = baselines["metrics"]
    comparisons = {}
    for entry in complete:
        candidate_metrics = entry["result"]["metrics_by_dataset"].get("dev-v3", {})
        comparisons[entry["run_id"]] = {
            dimension: {"candidate": candidate_metrics.get(dimension),
                        "saved_laya": baseline_metrics["laya"].get(dimension),
                        "saved_sokudan": baseline_metrics["sokudan"].get(dimension),
                        "saved_horizon_english_official": baseline_metrics["horizon_english_official"].get(dimension),
                        "same_set_majority": baselines["majority"].get(dimension)}
            for dimension in ("choice", "score", "noul_deadline")
        }
    case_rows = []
    if complete:
        reference = complete[0]["rows"]
        for row in reference:
            adapter = row.get("adapter_output") or {}
            score_value_source = (row.get("score_value_source")
                                  or adapter.get("continuous_score_source")
                                  or "not returned by official API")
            case_rows.append({"candidate": candidate, "reference_run_id": complete[0]["run_id"],
                              "case_id": row["case_id"], "dataset": row["dataset"],
                              "dimension": row["dimension"], "state": row["state"],
                              "allowed_answers": row["allowed_answers"],
                              "verification_status": row["verification_status"],
                              "prediction": row["common_prediction"], "status": row["status"],
                              "model_specific_output": row["model_specific_output"],
                              "adapter_output": adapter,
                              "score_value_for_metric": row.get("score_value_for_metric",
                                                                 row.get("score_expected_value_not_api_returned_score")),
                              "score_value_source": score_value_source,
                              "score_api_returned_continuous_value": (
                                  row.get("score_api_returned_continuous_value")
                                  if row.get("score_api_returned_continuous_value") is not None
                                  else adapter.get("returned_continuous_score")
                              ),
                              "input_trace": row["input_trace"],
                              "draft_label_match": row["draft_label_match"],
                              "human_quality_status": "UNDETERMINED"})
    return {"candidate": candidate, "trial_run_ids": [row["run_id"] for row in runs],
            "complete_trial_count": len(complete), "trial_statuses": runs,
            "prediction_and_output_identical_across_all_three_trials": same,
            "metrics_by_trial_run": per_trial_metrics,
            "comparison_to_saved_models_by_trial": comparisons,
            "cases_from_trial_1": case_rows}


def _run_child_process(root: Path, candidate: str, trial: int, run: Run) -> dict[str, Any]:
    command = [sys.executable, "-m", "som_probe.cli", "typed-model-cpu",
               "--candidate", candidate, "--child", "--run-id", run.run_id,
               "--trial", str(trial)]
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "CUDA_VISIBLE_DEVICES": "",
                "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0",
                "HF_HOME": str(root / ".local" / "cache" / "typed-model-cpu")})
    try:
        completed = subprocess.run(command, cwd=root, env=env, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, check=False)
    except OSError as exc:
        run.failure(RUN_STAGE, "TYPED_MODEL_CHILD_PROCESS_START_FAILED", exc,
                    candidate=candidate, trial=trial)
        return {"status": "FAILED", "run_id": run.run_id, "candidate": candidate,
                "trial": trial, "code": "TYPED_MODEL_CHILD_PROCESS_START_FAILED"}
    manifest = run.read_manifest()
    status = manifest.get("status")
    if status == "RUNNING":
        run.failure(RUN_STAGE, "TYPED_MODEL_CHILD_PROCESS_NO_TERMINAL_STATE",
                    RuntimeError("child did not write a terminal Run state"),
                    candidate=candidate, trial=trial, child_return_code=completed.returncode)
        status = run.read_manifest().get("status")
    if status == "COMPLETE":
        stage = manifest.get("stages", {}).get(RUN_STAGE, {})
        return {"status": "COMPLETE", "run_id": run.run_id, "candidate": candidate,
                "trial": trial, "status_counts": stage.get("status_counts", {})}
    diagnostic = run.path / "typed-model-diagnostic.json"
    code = _strict_json(diagnostic.read_text(encoding="utf-8")).get("code") if diagnostic.is_file() else "TYPED_MODEL_CHILD_FAILED"
    return {"status": status or "FAILED", "run_id": run.run_id, "candidate": candidate,
            "trial": trial, "code": code, "child_return_code": completed.returncode}


def run_all_trials(root: Path, selected: str, trials: int = TRIAL_COUNT) -> dict[str, Any]:
    if trials != TRIAL_COUNT or selected not in {"both", "laya-typed", "gliner25-multi-decide"}:
        raise TypedModelError("TYPED_MODEL_TRIAL_SELECTION_INVALID")
    _fixed_device_environment(root)
    lock, evaluation, controls = _read_config(root)
    # Verify the dataset and all frozen controls before starting any model process.
    cases, _ = _load_cases(root, controls)
    if len(cases) != 82:
        raise TypedModelError("TYPED_MODEL_CASE_COUNT_INVALID")
    candidate_ids = ["laya-typed", "gliner25-multi-decide"] if selected == "both" else [selected]
    summary_run = Run.create(root, "typed pretrained model comparison from independent CPU Runs",
                             dataset_version="dev-v3", profile_applicable=False)
    trial_results: dict[str, list[dict[str, Any]]] = {}
    try:
        for candidate in candidate_ids:
            runs = []
            for trial in range(1, TRIAL_COUNT + 1):
                run = Run.create(root, f"{candidate} CPU fixed evaluation trial {trial}",
                                 dataset_version="dev-v3", profile_applicable=False)
                runs.append(_run_child_process(root, candidate, trial, run))
            trial_results[candidate] = runs
        dev_rows = [case for case in cases if case["dataset"] == "dev-v3"]
        baseline_bundle, _ = _saved_baselines(root, dev_rows)
        candidates = {candidate: _summarize_candidate(root, candidate, rows, baseline_bundle)
                      for candidate, rows in trial_results.items()}
        case_rows = [case for entry in candidates.values() for case in entry["cases_from_trial_1"]]
        summaries = {candidate: {key: value for key, value in entry.items() if key != "cases_from_trial_1"}
                     for candidate, entry in candidates.items()}
        result = {"analysis_run_id": summary_run.run_id,
                  "purpose": "CPU comparison and saved baseline analysis; no old model was rerun",
                  "status": "COMPLETE", "quality_status": "UNDETERMINED",
                  "label_state": "ai_draft_unverified", "model_adoption": "NOT_DECIDED",
                  "dataset": {"version": "dev-v3", "sha256": EXPECTED_DATASET_SHA256,
                              "scoring_contract_sha256": EXPECTED_CONTRACT_SHA256,
                              "controls_fixture": CONTROL_DATA,
                              "controls_sha256": _sha256_file(root / CONTROL_DATA),
                              "dev_cases": 60, "fixed_controls": len(cases) - 60},
                  "evaluation_lock_sha256": _sha256_file(root / EVAL_CONFIG),
                  "model_lock_sha256": _sha256_file(root / MODEL_LOCK),
                  "saved_baselines": {"metrics": baseline_bundle["metrics"],
                                      "same_dev_v3_draft_majority": baseline_bundle["majority"],
                                      "identity": {"laya_sokudan": baseline_bundle["laya_sokudan_identity"],
                                                   "horizon": baseline_bundle["horizon_identity"]}},
                  "candidate_trials": summaries,
                  "candidate_trial_runs": trial_results,
                  "comparison_metrics_semantics": {
                      "Choice": "candidate IDs and saved decisions against the same dev-v3 draft labels",
                      "Score": "stage metrics are comparable; continuous value metrics are API-specific and GLiNER does not return a continuous Score",
                      "Noul": "deadline-notification-candidate decision; no user notification operation was tested",
                      "confidence": "model-specific confidence is not cross-model calibrated",
                      "controls": "AI-created fixed technical controls, separate from dev-v3",
                      "all_quality_results": "preliminary only; human verification remains incomplete"}}
        summary_run.write_jsonl("typed-model-comparison-cases.jsonl", case_rows)
        result["cases_file"] = "typed-model-comparison-cases.jsonl"
        summary_run.write_json("typed-model-comparison-result.json", result)
        summary_run.stage(ANALYSIS_STAGE, "PASS",
                          candidate_run_ids={candidate: [row["run_id"] for row in rows]
                                             for candidate, rows in trial_results.items()},
                          dataset_sha256=EXPECTED_DATASET_SHA256,
                          scoring_contract_sha256=EXPECTED_CONTRACT_SHA256,
                          result_sha256=_sha256_file(summary_run.path / "typed-model-comparison-result.json"),
                          cases_sha256=_sha256_file(summary_run.path / "typed-model-comparison-cases.jsonl"),
                          official_japanese_quality_status="UNDETERMINED")
        return {"status": "COMPLETE", "analysis_run_id": summary_run.run_id,
                "candidate_trial_runs": trial_results,
                "candidates_with_three_completed_runs": {
                    candidate: entry["complete_trial_count"] == TRIAL_COUNT
                    for candidate, entry in candidates.items()},
                "prediction_and_output_stability": {
                    candidate: entry["prediction_and_output_identical_across_all_three_trials"]
                    for candidate, entry in candidates.items()},
                "quality_status": "UNDETERMINED"}
    except Exception as exc:
        code = exc.code if isinstance(exc, TypedModelError) else "TYPED_MODEL_ANALYSIS_FAILED"
        summary_run.failure(ANALYSIS_STAGE, code, RuntimeError(safe_message(str(exc))))
        return {"status": "FAILED", "analysis_run_id": summary_run.run_id, "code": code,
                "candidate_trial_runs": trial_results}

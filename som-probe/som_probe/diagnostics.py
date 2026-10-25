"""Fixed-input diagnostics for the interactive model trial paths."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .storage import Run, open_run


SOURCE_RUN_ID = "20261025-041751-58713e9e"
DIAGNOSTIC_CASES = (
    {"id": "fixed-01", "state": "明日ヤマトから荷物届く",
     "provisional_targets": {"choice": "none", "score_stage": 1}},
    {"id": "fixed-02", "state": "今週中に勤怠まとめて提出",
     "provisional_targets": {"choice": "task", "score_stage": 1}},
    {"id": "fixed-03", "state": "へびさば！S3の検証する",
     "provisional_targets": {"choice": "task", "score_stage": 0}},
    {"id": "fixed-04", "state": "スコーン食べた",
     "provisional_targets": {"choice": "none", "score_stage": 0}},
)
CONTRAST_FIXTURES = (
    {
        "type": "A",
        "question": "本文の作業は、すでに完了していますか",
        "paraphrase": "本文に記載された作業は、もう終わっていますか",
        "positive": "資料を提出した。",
        "negative": "資料はまだ提出していない。",
    },
    {
        "type": "B",
        "question": "本文に時計の時刻が明記されていますか",
        "paraphrase": "本文には何時かを示す時刻が書かれていますか",
        "positive": "明日15時に荷物が届く。",
        "negative": "明日、荷物が届く。",
    },
    {
        "type": "C",
        "question": "本文では通知を希望していますか",
        "paraphrase": "本文の書き手は通知を求めていますか",
        "positive": "明日提出する。通知してほしい。",
        "negative": "明日提出する。通知はいらない。",
    },
)


class DiagnosticError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_dev_v3_questions(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "datasets" / "dev-v3.jsonl"
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            for question_id, question in case["questions"].items():
                kind = "noul" if question["type"] == "bool" else question["type"]
                if kind in {"choice", "score", "noul"} and kind not in selected:
                    selected[kind] = {"question_id": question_id, "case_id": case["case_id"],
                                      "state": case["state"], "question": copy.deepcopy(question)}
            if len(selected) == 3:
                break
    if set(selected) != {"choice", "score", "noul"}:
        raise DiagnosticError("DEV_V3_QUESTION_TEMPLATE_INVALID")
    return selected


def _candidate_ids(kind: str, question: dict[str, Any]) -> list[Any]:
    if kind == "choice":
        return list(question["criteria"])
    if kind == "score":
        return list(range(len(question["criteria"])))
    return [False, True]


def _laya_question(question: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(question)
    if result.get("type") == "bool":
        result["type"] = "noul"
    return result


def _official_probabilities(kind: str, answer: dict[str, Any], candidate_ids: list[Any]) -> list[float] | None:
    values = answer.get("probabilities")
    if not isinstance(values, dict):
        return None
    keys = [str(value) for value in candidate_ids] if kind == "score" else candidate_ids
    try:
        return [float(values[str(key)] if kind == "score" else values[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return None


def _normalise_laya_answer(kind: str, answer: dict[str, Any], candidate_ids: list[Any]) -> dict[str, Any]:
    probabilities = _official_probabilities(kind, answer, candidate_ids)
    if kind == "choice":
        decision = answer.get("choice")
        return {"decision": decision, "candidate_ids": candidate_ids, "probabilities": probabilities,
                "probability_origin": "official_api_display_rounded_4"}
    if kind == "score":
        stage = (max(range(len(probabilities)), key=probabilities.__getitem__)
                 if probabilities else None)
        return {"decision": stage, "decision_slot": stage, "stage_prediction": stage,
                "stage_prediction_source": "official_api_display_rounded_4_probabilities",
                "candidate_ids": candidate_ids, "probabilities": probabilities,
                "score_expected_unrounded": None, "returned_score": answer.get("score"),
                "probability_origin": "official_api_display_rounded_4"}
    returned = answer.get("noul")
    evaluator_boolean = (float(returned) >= 0.5) if returned is not None else None
    return {"candidate_ids": candidate_ids, "probabilities": None,
            "decision": evaluator_boolean, "boolean_result": evaluator_boolean,
            "argmax_boolean_result": None,
            "api_returned_value": returned,
            "evaluator_boolean": evaluator_boolean,
            "probability_origin": "official_api_python_round_4"}


def _normalise_adapter_answer(kind: str, answer: dict[str, Any]) -> dict[str, Any]:
    values = answer.get("candidate_probabilities")
    probabilities = ([float(item["probability"]) for item in values]
                     if isinstance(values, list) else answer.get("api_display_probabilities"))
    return {"decision": answer.get("decision"), "decision_slot": answer.get("decision_slot"),
            "stage_prediction": answer.get("stage_prediction"),
            "stage_prediction_source": answer.get("stage_prediction_source"),
            "candidate_ids": answer.get("candidate_ids"), "probabilities": probabilities,
            "score_expected_unrounded": answer.get("score_expected_unrounded"),
            "returned_score": answer.get("returned_score", answer.get("api_returned_value")) if kind == "score" else None,
            "true_probability": answer.get("true_probability"),
            "returned_noul": answer.get("returned_noul", answer.get("api_returned_value")) if kind == "noul" else None,
            "boolean_result": answer.get("boolean_result", answer.get("evaluator_boolean")),
            "argmax_boolean_result": answer.get("argmax_boolean_result"),
            "probability_origin": answer.get("probability_origin")}


def _official_laya_trace(agent: Any, question_id: str, question: dict[str, Any], state: str) -> dict[str, Any]:
    import numpy as np
    from laya.common import collate_items, encode_text, render_options, serialize_state, temp_bucket

    api_question = _laya_question(question)
    internal = {question_id: agent._to_internal(api_question)}
    items = agent._encode_state(state, [question_id], internal, max_len=256, head_max_len=192)
    batch = collate_items([items], agent.tok.pad_token_id)
    internal_question = internal[question_id]
    state_ids = encode_text(
        agent.tok, serialize_state(state).replace(agent.tok.mask_token, " "), add_special_tokens=False,
    )["input_ids"]
    qtype = int(items[0]["qtype"])
    candidate_count = len(items[0]["markers"])
    temperature_bucket = temp_bucket(qtype, candidate_count)
    temperature = float(agent.temperature_by_options.get(
        temperature_bucket, agent.temperature[qtype],
    ))
    return {
        "question_id": question_id,
        "api_question": api_question,
        "question_sha256": _sha256(api_question),
        "question_text": internal_question["ins"],
        "candidate_ids": _candidate_ids("noul" if question["type"] == "bool" else question["type"], question),
        "rendered_options": render_options(internal_question),
        "state_token_ids": [int(value) for value in state_ids],
        "input_tensors": {name: np.asarray(batch[name][0].cpu()).tolist()
                           for name in ("input_ids", "attention_mask", "marker_pos", "marker_mask")},
        "special_token_ids": {"cls": int(agent.tok.cls_token_id), "sep": int(agent.tok.sep_token_id),
                              "mask": int(agent.tok.mask_token_id), "pad": int(agent.tok.pad_token_id)},
        "qtype": qtype,
        "language_parameter": None,
        "temperature_bucket": temperature_bucket,
        "effective_temperature": temperature,
        "state_truncated": bool(items[0]["state_stats"]["truncated"]),
        "state_tokens_dropped": int(items[0]["state_stats"]["state_tokens_dropped"]),
    }


def _load_official_laya(root: Path, run: Run):
    from .assets import load_lock, make_execution_checkpoint, verify_assets
    from .runtime import _checkpoint_hashes

    verify_assets(root)
    lock = load_lock(root)
    source = lock["sources"]["laya_code"]
    source_dir = (root / ".local" / "sources" / f"laya-pinned-{source['commit']}").resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        from laya.agent import Agent
    except Exception:
        raise DiagnosticError("PINNED_LAYA_API_IMPORT_FAILED") from None
    checkpoint = make_execution_checkpoint(root, run.run_id)
    try:
        agent = Agent(str(checkpoint), device="cpu", compile=False, fast=False,
                      expected_sha256=_checkpoint_hashes(lock))
    except Exception:
        raise DiagnosticError("PINNED_LAYA_API_LOAD_FAILED") from None
    if str(agent.device) != "cpu":
        raise DiagnosticError("PINNED_LAYA_API_DEVICE_MISMATCH")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in agent.model.parameters()
                               if parameter.is_floating_point()})
    if parameter_dtypes != ["torch.float32"]:
        raise DiagnosticError("PINNED_LAYA_API_PRECISION_MISMATCH")
    return agent, lock, parameter_dtypes


def _run_laya_direct(root: Path, run: Run) -> dict[str, Any]:
    from .tryout import _ExclusiveModelLock

    with _ExclusiveModelLock():
        return _run_laya_direct_locked(root, run)


def _run_laya_direct_locked(root: Path, run: Run) -> dict[str, Any]:
    import torch

    agent, lock, parameter_dtypes = _load_official_laya(root, run)
    questions = _read_dev_v3_questions(root)
    records: dict[str, Any] = {}
    try:
        for fixture in DIAGNOSTIC_CASES:
            state = fixture["state"]
            records[fixture["id"]] = {"state": state, "provisional_targets": fixture["provisional_targets"],
                                       "target_status": "AI_PROVISIONAL_NOT_HUMAN_REVIEWED", "answers": {}}
            api_questions = {questions[kind]["question_id"]: _laya_question(questions[kind]["question"])
                             for kind in ("choice", "score", "noul")}
            with torch.inference_mode():
                output = agent.system_one(state, api_questions, lang=None,
                                          max_len=256, head_max_len=192)
            if output.get("usage", {}).get("truncated") is not False:
                raise DiagnosticError("LAYA_DIRECT_INPUT_TRUNCATED")
            displayed: dict[str, Any] = {}
            display_context: dict[str, Any] = {}
            for kind in ("choice", "score", "noul"):
                question_id = questions[kind]["question_id"]
                api_question = api_questions[question_id]
                trace = _official_laya_trace(agent, question_id, api_question, state)
                answer = output["answers"][question_id]
                candidate_ids = _candidate_ids(kind, questions[kind]["question"])
                normalized = _normalise_laya_answer(kind, answer, candidate_ids)
                records[fixture["id"]]["answers"][kind] = {
                    "question_id": question_id, "question": api_question,
                    "official_api_answer": answer, "normalized": normalized,
                    "usage": output.get("usage"), "input_trace": trace,
                }
                displayed[kind] = _format_direct_for_screen(kind, answer, candidate_ids)
                from .tryout import _candidate_descriptions
                display_context[kind] = {
                    "candidate_ids": candidate_ids,
                    "candidate_descriptions": _candidate_descriptions(api_question),
                }
            from .tryout import format_results
            records[fixture["id"]]["screen"] = "入力：" + state + "\n" + format_results(
                displayed, display_context)
    finally:
        agent.__exit__(None, None, None)
    return {
        "stage": "LAYA_DIRECT_CPU_FP32",
        "backend": "pinned official Laya Agent.system_one",
        "requested_device": "cpu", "actual_device": str(agent.device),
        "floating_parameter_dtypes": parameter_dtypes,
        "model_revision": lock["sources"]["reference_checkpoint"]["revision"],
        "source_revision": lock["sources"]["laya_code"]["commit"],
        "model_artifact_sha256": lock["artifacts"],
        "input_profile": "s256-k8 API overrides max_len=256, head_max_len=192",
        "question_dataset_sha256": hashlib.sha256((root / "datasets" / "dev-v3.jsonl").read_bytes()).hexdigest(),
        "records": records,
    }


def _format_direct_for_screen(kind: str, answer: dict[str, Any], candidate_ids: list[Any]) -> dict[str, Any]:
    if kind == "choice":
        probabilities = _official_probabilities(kind, answer, candidate_ids) or []
        decision = answer.get("choice")
        return {"question_type": kind, "decision": decision,
                "decision_slot": candidate_ids.index(decision) if decision in candidate_ids else None,
                "candidate_ids": candidate_ids,
                "api_display_probabilities": probabilities,
                "probability_origin": "official_api_display_rounded_4"}
    if kind == "score":
        probabilities = _official_probabilities(kind, answer, candidate_ids) or []
        stage = max(range(len(probabilities)), key=probabilities.__getitem__) if probabilities else None
        return {"question_type": kind, "decision_slot": stage, "stage_prediction": stage,
                "stage_prediction_source": "official_api_display_rounded_4_probabilities",
                "candidate_ids": candidate_ids, "api_display_probabilities": probabilities,
                "api_returned_value": answer.get("score"), "score_expected_unrounded": None}
    returned = answer.get("noul")
    return {"question_type": kind, "candidate_ids": candidate_ids,
            "api_returned_value": returned,
            "evaluator_boolean": (float(returned) >= 0.5) if returned is not None else None}


def _run_laya_paths(root: Path, run: Run) -> dict[str, Any]:
    import gc

    from .tryout import (_ExclusiveModelLock, _LayaSession, _question_templates, format_results)

    templates, canaries = _question_templates(root)
    sessions: dict[str, Any] = {}
    results: dict[str, Any] = {"records": {}, "execution": {}}
    with _ExclusiveModelLock():
        for device in ("CPU", "NPU"):
            session = _LayaSession(root, SOURCE_RUN_ID, templates, canaries,
                                   execution_device=device, diagnostic_run_id=run.run_id)
            sessions[device] = session
            device_records = {}
            for fixture in DIAGNOSTIC_CASES:
                output, trace = session.predict_with_trace(fixture["state"])
                device_records[fixture["id"]] = {
                    "state": fixture["state"], "results": output, "input_trace": trace,
                    "screen": "入力：" + fixture["state"] + "\n" +
                             format_results(output, session.display_context()),
                }
            results["records"][device] = device_records
            results["execution"][device] = {"startup": session.startup,
                                            "execution_devices": session._execution_devices,
                                            "execution_devices_property": session._properties.get("EXECUTION_DEVICES"),
                                            "compiler_type": str(session._properties.get("NPU_COMPILER_TYPE", "未取得")),
                                            "compiler_version": str(session._properties.get("NPU_COMPILER_VERSION", "未取得")),
                                            "loaded_from_cache": session._properties.get("LOADED_FROM_CACHE", "未取得"),
                                            "device_architecture": session._actual_architecture,
                                            "device_properties": session._device_properties,
                                            "cache_dir": str(session._cache_dir) if session._cache_dir else None}
            session.close()
            del sessions[device]
            del session
            gc.collect()
    return {"stage": "LAYA_OPENVINO_CPU_AND_NPU_PLUGIN", "requested_devices": ["CPU", "NPU"],
            "source_run_id": SOURCE_RUN_ID, "records": results["records"],
            "execution": results["execution"]}


def _sokudan_official_trace(agent: Any, state: str, questions: dict[str, Any]) -> dict[str, Any]:
    from sokudan.encoding.question import encode_joint
    from sokudan.schema.question import parse_questions

    parsed = parse_questions(questions)
    result: dict[str, Any] = {}
    state_ids = list(agent.tokenizer(state, add_special_tokens=False)["input_ids"])
    for question_id, question in parsed.items():
        encoded = encode_joint(question, state, agent.tokenizer, input_order=agent.backend.input_order)
        tensors = {"input_ids": list(encoded.input_ids), "attention_mask": list(encoded.attention_mask),
                   "marker_positions": list(encoded.marker_positions)}
        result[question_id] = {
            "input_tensors": tensors, "input_tensor_sha256": _sha256({**tensors,
                "ordered": bool(encoded.ordered), "input_order": agent.backend.input_order}),
            "candidate_ids": ([False, True] if question.type == "bool" else
                              list(range(len(question.labels))) if question.type == "score" else
                              list(question.labels)),
            "marker_texts": list(question.labels), "state_token_ids": state_ids,
            "required_total_tokens": len(encoded.input_ids), "truncated": bool(encoded.truncated),
            "ordered": bool(encoded.ordered), "input_order": agent.backend.input_order,
        }
    return result


def _run_sokudan(root: Path, run: Run) -> dict[str, Any]:
    import gc

    from .tryout import (_ExclusiveModelLock, _SokudanSession, _question_templates, format_results)
    from .sokudan_cpu import _validate_calibration

    questions_by_kind = _read_dev_v3_questions(root)
    templates, canaries = _question_templates(root)
    questions = {questions_by_kind[kind]["question_id"]: questions_by_kind[kind]["question"]
                 for kind in ("choice", "score", "noul")}
    session = None
    records: dict[str, Any] = {}
    lock = None
    verification = None
    with _ExclusiveModelLock():
        session = _SokudanSession(root, templates, canaries)
        lock = session._lock
        verification = session._verification
        try:
            for fixture in DIAGNOSTIC_CASES:
                state = fixture["state"]
                direct_input = _sokudan_official_trace(session._agent, state, questions)
                direct_before = _canonical(questions)
                official = session._agent.predict(state, copy.deepcopy(questions), order_marginalize=False)
                if _canonical(questions) != direct_before or official.get("usage", {}).get("state_truncated") is not False:
                    raise DiagnosticError("SOKUDAN_DIRECT_INPUT_CONTRACT_FAILED")
                question_kinds = {qid: "noul" if question["type"] == "bool" else question["type"]
                                  for qid, question in questions.items()}
                question_counts = {qid: len(_official_sokudan_candidate_ids(question))
                                   for qid, question in questions.items()}
                _validate_calibration(session._agent, official, list(questions), question_kinds,
                                      question_counts, session._expected_temperature)
                official_calibration = {
                    key: official.get(key) for key in ("calibrated", "calibrated_answers", "encoding")
                }
                adapted, adapter_trace = session.predict_with_trace(state)
                direct_answers: dict[str, Any] = {}
                screen_adapted = {}
                for kind in ("choice", "score", "noul"):
                    question_id = questions_by_kind[kind]["question_id"]
                    answer = official["answers"][question_id]
                    ids = _official_sokudan_candidate_ids(questions[question_id])
                    normalized = _normalise_sokudan_official_answer(kind, answer, ids)
                    screen_adapted[kind] = adapted[kind]
                    direct_answers[kind] = {
                        "question_id": question_id, "question": questions[question_id],
                        "official_api_answer": answer, "normalized": normalized,
                        "input_trace": direct_input[question_id],
                    }
                direct_display = {
                    kind: _format_sokudan_direct_for_screen(kind, direct_answers[kind]["normalized"])
                    for kind in ("choice", "score", "noul")
                }
                direct_context = session.display_context()
                direct_screen = "入力：" + state + "\n" + format_results(
                    direct_display, direct_context)
                records[fixture["id"]] = {
                    "state": state, "provisional_targets": fixture["provisional_targets"],
                    "target_status": "AI_PROVISIONAL_NOT_HUMAN_REVIEWED",
                    "official_api": direct_answers, "official_api_screen": direct_screen,
                    "official_api_calibration": official_calibration,
                    "adapter": {"results": adapted, "trace": adapter_trace,
                                "screen": "入力：" + state + "\n" + format_results(
                                    screen_adapted, session.display_context())},
                }
        finally:
            session.close()
            del session
            gc.collect()
    return {"stage": "SOKUDAN_OFFICIAL_API_AND_ADAPTER_CPU", "requested_backend": "torch",
            "requested_device": "cpu", "requested_dtype": "float32",
            "model_revision": lock["model"]["revision"] if lock else None,
            "backbone_revision": lock["backbone"]["revision"] if lock else None,
            "calibration_sha256": lock["execution"]["calibration_sha256"] if lock else None,
            "verification": ({"runtime_model": verification["runtime_model"],
                              "artifact_lock_sha256": verification["artifact_lock_sha256"]}
                             if verification else {}),
            "records": records}


def _official_sokudan_candidate_ids(question: dict[str, Any]) -> list[Any]:
    if question.get("type") == "choice":
        return list(question["criteria"])
    if question.get("type") == "score":
        return list(range(len(question["criteria"])))
    if question.get("type") == "bool":
        return [False, True]
    raise DiagnosticError("SOKUDAN_DIRECT_QUESTION_TYPE_INVALID")


def _normalise_sokudan_official_answer(
    kind: str, answer: dict[str, Any], candidate_ids: list[Any],
) -> dict[str, Any]:
    """Read official API fields directly, independently of the tryout adapter."""
    expected_type = "bool" if kind == "noul" else kind
    if not isinstance(answer, dict) or answer.get("type") != expected_type:
        raise DiagnosticError("SOKUDAN_DIRECT_ANSWER_TYPE_INVALID")
    result = {"decision": None, "decision_slot": None, "stage_prediction": None,
              "candidate_ids": list(candidate_ids), "probabilities": None,
              "returned_score": None, "returned_noul": None,
              "boolean_result": None, "probability_origin": None}
    if kind in {"choice", "score"}:
        returned = answer.get("choice") if kind == "choice" else answer.get("score")
        probabilities_by_id = answer.get("probabilities")
        key_order = [str(value) for value in candidate_ids] if kind == "score" else candidate_ids
        if not isinstance(probabilities_by_id, dict) or list(probabilities_by_id) != key_order:
            raise DiagnosticError("SOKUDAN_DIRECT_PROBABILITY_KEYS_INVALID")
        probabilities = [float(probabilities_by_id[key]) for key in key_order]
        if (any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
                or abs(sum(probabilities) - 1.0) > len(probabilities) * 0.0000501):
            raise DiagnosticError("SOKUDAN_DIRECT_PROBABILITY_RANGE_INVALID")
        stage = max(range(len(probabilities)), key=probabilities.__getitem__)
        result.update({"probabilities": probabilities,
                       "probability_origin": "official_api_display_rounded_4"})
        if kind == "choice":
            if returned not in candidate_ids:
                raise DiagnosticError("SOKUDAN_DIRECT_UNKNOWN_CANDIDATE")
            result.update({"decision": returned, "decision_slot": candidate_ids.index(returned)})
        else:
            returned_score = float(returned)
            if (not math.isfinite(returned_score)
                    or not 0.0 <= returned_score <= float(len(candidate_ids) - 1)):
                raise DiagnosticError("SOKUDAN_DIRECT_SCORE_RANGE_INVALID")
            result.update({"decision": stage, "decision_slot": stage, "stage_prediction": stage,
                           "stage_prediction_source": "argmax_of_official_api_display_rounded_4_probabilities",
                           "returned_score": returned_score})
        return result
    returned_noul = float(answer["noul"])
    if not 0.0 <= returned_noul <= 1.0:
        raise DiagnosticError("SOKUDAN_DIRECT_NOUL_RANGE_INVALID")
    result.update({"returned_noul": returned_noul,
                   "boolean_result": returned_noul >= 0.5,
                   "probability_origin": "official_api_python_round_4"})
    return result


def _format_sokudan_direct_for_screen(kind: str, normalized: dict[str, Any]) -> dict[str, Any]:
    return _normalise_for_screen(kind, normalized)


def _normalise_for_screen(kind: str, normalized: dict[str, Any]) -> dict[str, Any]:
    if kind == "choice":
        ids = normalized["candidate_ids"]
        probabilities = normalized.get("probabilities") or []
        return {"question_type": kind, "decision": normalized.get("decision"),
                "decision_slot": normalized.get("decision_slot"), "candidate_ids": ids,
                "api_display_probabilities": probabilities,
                "probability_origin": normalized.get("probability_origin")}
    if kind == "score":
        return {"question_type": kind, "decision_slot": normalized.get("stage_prediction"),
                "stage_prediction": normalized.get("stage_prediction"),
                "stage_prediction_source": normalized.get("stage_prediction_source"),
                "candidate_ids": normalized.get("candidate_ids"),
                "api_display_probabilities": normalized.get("probabilities"),
                "api_returned_value": normalized.get("returned_score"),
                "score_expected_unrounded": normalized.get("score_expected_unrounded")}
    return {"question_type": kind, "candidate_ids": normalized.get("candidate_ids"),
            "api_returned_value": normalized.get("returned_noul", normalized.get("api_returned_value")),
            "evaluator_boolean": normalized.get("boolean_result", normalized.get("evaluator_boolean"))}


def _run_laya_contrasts(root: Path, run: Run) -> list[dict[str, Any]]:
    import torch

    from .tryout import _ExclusiveModelLock

    results = []
    with _ExclusiveModelLock():
        agent, _, _ = _load_official_laya(root, run)
        try:
            for fixture in CONTRAST_FIXTURES:
                for variant, prompt in (("base", fixture["question"]), ("paraphrase", fixture["paraphrase"])):
                    question_id = f"contrast-{fixture['type']}-{variant}"
                    question = {"type": "noul", "instructions": prompt,
                                "criteria": {"false": "いいえ", "true": "はい"},
                                "labels": {"false": "いいえ", "true": "はい"}}
                    for polarity, state, expected in (("positive", fixture["positive"], True),
                                                       ("negative", fixture["negative"], False)):
                        trace = _official_laya_trace(agent, question_id, question, state)
                        with torch.inference_mode():
                            output = agent.system_one(state, {question_id: question},
                                                     max_len=256, head_max_len=192)
                        if output.get("usage", {}).get("truncated") is not False:
                            raise DiagnosticError("LAYA_CONTRAST_INPUT_TRUNCATED")
                        answer = output["answers"][question_id]
                        results.append({"type": fixture["type"], "variant": variant,
                                        "question": prompt, "polarity": polarity, "state": state,
                                        "expected_boolean": expected,
                                        "api_returned_noul": answer.get("noul"),
                                        "evaluator_boolean": (float(answer["noul"]) >= 0.5),
                                        "input_trace": trace})
        finally:
            agent.__exit__(None, None, None)
    return results


def _run_contrasts(root: Path, run: Run, model: str) -> dict[str, Any]:
    from .tryout import _ExclusiveModelLock, _SokudanSession, _question_templates
    from .sokudan_cpu import _validate_calibration

    if model == "laya":
        results = _run_laya_contrasts(root, run)
    else:
        results = []
        questions = _question_templates(root)
        templates, canaries = questions
        with _ExclusiveModelLock():
            session = _SokudanSession(root, templates, canaries)
            try:
                for fixture in CONTRAST_FIXTURES:
                    for variant, prompt in (("base", fixture["question"]), ("paraphrase", fixture["paraphrase"])):
                        question_id = f"contrast-{fixture['type']}-{variant}"
                        question = {"type": "bool", "instructions": prompt,
                                    "false_label": "いいえ", "true_label": "はい"}
                        questions_for_api = {question_id: question}
                        for polarity, state, expected in (("positive", fixture["positive"], True),
                                                           ("negative", fixture["negative"], False)):
                            trace = _sokudan_official_trace(session._agent, state, questions_for_api)[question_id]
                            output = session._agent.predict(state, copy.deepcopy(questions_for_api),
                                                            order_marginalize=False)
                            if output.get("usage", {}).get("state_truncated") is not False:
                                raise DiagnosticError("SOKUDAN_CONTRAST_INPUT_TRUNCATED")
                            _validate_calibration(session._agent, output, [question_id],
                                                  {question_id: "noul"}, {question_id: 2},
                                                  session._expected_temperature)
                            answer = output["answers"][question_id]
                            results.append({"type": fixture["type"], "variant": variant,
                                            "question": prompt, "polarity": polarity, "state": state,
                                            "expected_boolean": expected,
                                            "api_returned_noul": answer.get("noul"),
                                            "evaluator_boolean": float(answer["noul"]) >= 0.5,
                                            "input_trace": trace})
            finally:
                session.close()
    return {"stage": f"{model.upper()}_CONTRAST_CPU", "results": results,
            "label_status": "AI_FIXED_DIAGNOSTIC_EXPECTATIONS_NOT_HUMAN_QUALITY_LABELS"}


def _target_mismatches(results: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches = []
    for model in ("laya", "sokudan"):
        data = results[model]["records"]
        for fixture in DIAGNOSTIC_CASES:
            case = data[fixture["id"]]
            for kind, target in fixture["provisional_targets"].items():
                if model == "laya":
                    actual = case["answers"][kind]["normalized"]
                else:
                    actual = case["official_api"][kind]["normalized"]
                observed = actual.get("decision") if kind == "choice" else actual.get("stage_prediction")
                if observed != target:
                    mismatches.append({"model": model, "case_id": fixture["id"], "type": kind,
                                       "expected": target, "observed": observed,
                                       "target_status": "AI_PROVISIONAL_NOT_HUMAN_REVIEWED"})
    return mismatches


def _compare_laya_input(direct: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    direct_tensors = direct["input_tensors"]
    candidate_tensors = candidate["input_tensors"]
    direct_ids = direct_tensors["input_ids"]
    direct_mask = direct_tensors["attention_mask"]
    candidate_ids = candidate_tensors["input_ids"][0]
    candidate_mask = candidate_tensors["attention_mask"][0]
    active_direct = [token for token, mask in zip(direct_ids, direct_mask) if mask]
    active_candidate = [token for token, mask in zip(candidate_ids, candidate_mask) if mask]
    direct_positions = [value for value, mask in zip(direct_tensors["marker_pos"],
                                                       direct_tensors["marker_mask"]) if mask]
    candidate_positions = [value for value, mask in zip(candidate_tensors["marker_pos"][0],
                                                          candidate_tensors["marker_mask"][0]) if mask]
    direct_state_ids = direct["state_token_ids"]
    candidate_state_ids = candidate["state_token_ids"]
    candidate_question = copy.deepcopy(candidate["question_contract"])
    if candidate_question.get("type") == "bool":
        candidate_question["type"] = "noul"
    pad_id = candidate["special_token_ids"]["pad"]
    custom_padding_start = len(active_candidate)
    custom_padding_valid = (
        all(value == pad_id for value in candidate_ids[custom_padding_start:])
        and all(not value for value in candidate_mask[custom_padding_start:])
        and all(value == 0 for value in candidate_tensors["marker_pos"][0][len(candidate_positions):])
        and all(not value for value in candidate_tensors["marker_mask"][0][len(candidate_positions):])
    )
    effective_temperature = candidate.get("temperature")
    temperature_match = (
        effective_temperature is not None
        and float(direct["effective_temperature"]) == float(effective_temperature)
    )
    return {
        "active_token_ids_match": active_direct == active_candidate,
        "state_token_ids_match": direct_state_ids == candidate_state_ids,
        "candidate_ids_order_match": direct["candidate_ids"] == candidate["candidate_ids"],
        "question_id_match": direct["question_id"] == candidate["question_id"],
        "question_contract_match": _canonical(direct["api_question"]) == _canonical(candidate_question),
        "question_text_match": direct["question_text"] == candidate["question_text"],
        "candidate_text_order_match": direct["rendered_options"] == candidate["candidate_descriptions"],
        "marker_positions_match": direct_positions == candidate_positions,
        "qtype_match": direct["qtype"] == candidate_tensors["qtype"][0],
        "language_parameter_match": direct.get("language_parameter") is None
        and candidate.get("language_parameter") is None,
        "temperature_bucket_match": direct.get("temperature_bucket") == candidate.get("temperature_bucket"),
        "effective_temperature_match": temperature_match,
        "official_effective_temperature": direct.get("effective_temperature"),
        "converted_path_temperature": effective_temperature,
        "direct_markers_are_masks": all(
            direct_ids[position] == direct["special_token_ids"]["mask"]
            for position in direct_positions
        ),
        "custom_markers_are_masks": all(
            candidate_ids[position] == candidate["special_token_ids"]["mask"]
            for position in candidate_positions
        ),
        "padding_only_shape_difference": (
            active_direct == active_candidate and all(direct_mask)
            and len(candidate_ids) > len(direct_ids) == len(active_direct) and custom_padding_valid
        ),
        "direct_input_shape": [len(direct_ids)],
        "custom_input_shape": [len(candidate_ids)],
        "direct_active_tokens": len(active_direct),
        "custom_active_tokens": len(active_candidate),
        "direct_state_token_sha256": _sha256(direct_state_ids),
        "custom_state_token_sha256": _sha256(candidate_state_ids),
        "truncation": {"direct": direct["state_truncated"],
                       "direct_dropped_tokens": direct["state_tokens_dropped"],
                       "custom": candidate["state_truncated"],
                       "custom_dropped_tokens": candidate["state_tokens_dropped"],
                       "custom_required_tokens": candidate["required_total_tokens"]},
        "input_sha256": {"direct": _sha256(direct_tensors),
                         "custom": candidate["input_sha256"]},
    }


def _compare_laya_outputs(direct: dict[str, Any], candidate: dict[str, Any], kind: str,
                          *, noul_decision_mode: str = "api") -> dict[str, Any]:
    left = direct["normalized"]
    right = _normalise_adapter_answer(kind, candidate)
    left_decision = left.get("decision", left.get("stage_prediction"))
    right_decision = right.get("decision", right.get("stage_prediction"))
    if kind == "noul" and noul_decision_mode == "api":
        left_decision = left.get("boolean_result", left.get("evaluator_boolean", left_decision))
        right_decision = right.get("boolean_result", right.get("evaluator_boolean", right_decision))
    probability_differences = []
    if isinstance(left.get("probabilities"), list) and isinstance(right.get("probabilities"), list):
        if len(left["probabilities"]) == len(right["probabilities"]):
            probability_differences = [abs(float(a) - float(b))
                                       for a, b in zip(left["probabilities"], right["probabilities"])]
    returned_difference = None
    left_return = (left.get("returned_score") if kind == "score" else
                   left.get("api_returned_value") if kind == "noul" else None)
    right_return = (right.get("returned_score") if kind == "score" else
                    right.get("returned_noul") if kind == "noul" else None)
    if left_return is not None and right_return is not None:
        returned_difference = abs(float(left_return) - float(right_return))
    left_boolean = left.get("boolean_result", left.get("evaluator_boolean"))
    right_boolean = right.get("boolean_result", right.get("evaluator_boolean"))
    left_argmax = left.get("argmax_boolean_result")
    right_argmax = right.get("argmax_boolean_result")
    return {
        "decision_match": left_decision == right_decision,
        "evaluator_boolean_match": (left_boolean is right_boolean
                                     if left_boolean is not None and right_boolean is not None else None),
        "argmax_boolean_match": (left_argmax is right_argmax
                                  if left_argmax is not None and right_argmax is not None else None),
        "probability_abs_differences": probability_differences,
        "probability_max_abs_difference": max(probability_differences, default=None),
        "api_returned_value_abs_difference": returned_difference,
        "direct_api_output": left,
        "converted_path_output": right,
    }


def _compare_laya_custom_outputs(reference: dict[str, Any], candidate: dict[str, Any], kind: str) -> dict[str, Any]:
    return _compare_laya_outputs(
        {"normalized": _normalise_adapter_answer(kind, reference)}, candidate, kind,
        noul_decision_mode="model",
    )


def _compare_sokudan_input(direct: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    tensors = direct["input_tensors"]
    return {
        "input_ids_match": tensors["input_ids"] == prepared.get("input_ids"),
        "attention_mask_match": tensors["attention_mask"] == prepared.get("attention_mask"),
        "marker_positions_match": tensors["marker_positions"] == prepared.get("marker_positions"),
        "candidate_ids_order_match": direct["candidate_ids"] == prepared.get("candidate_ids"),
        "state_token_ids_match": direct["state_token_ids"] == prepared.get("state_token_ids"),
        "direct_input_sha256": direct["input_tensor_sha256"],
        "adapter_input_sha256": prepared.get("input_tensor_sha256"),
        "required_total_tokens": direct["required_total_tokens"],
        "adapter_truncated": prepared.get("truncated"),
        "direct_truncated": direct["truncated"],
    }


def _sokudan_outputs_match(kind: str, direct: dict[str, Any], adapter: dict[str, Any]) -> bool:
    if kind == "choice":
        keys = ("decision", "candidate_ids", "probabilities", "probability_origin")
    elif kind == "score":
        keys = ("stage_prediction", "candidate_ids", "probabilities", "returned_score",
                "stage_prediction_source", "probability_origin")
    else:
        keys = ("returned_noul", "boolean_result", "probability_origin")
    return _canonical({key: direct.get(key) for key in keys}) == _canonical(
        {key: adapter.get(key) for key in keys})


def _build_comparison(root: Path, direct_laya: dict[str, Any], laya_paths: dict[str, Any],
                      sokudan: dict[str, Any]) -> dict[str, Any]:
    gates = json.loads((root / "configs" / "gates.json").read_text(encoding="utf-8"))["stages"]
    comparison: dict[str, Any] = {"laya": {}, "sokudan": {}, "input_text_fingerprints": {}}
    for fixture in DIAGNOSTIC_CASES:
        case_id = fixture["id"]
        comparison["input_text_fingerprints"][case_id] = hashlib.sha256(
            fixture["state"].encode("utf-8")).hexdigest()
        comparison["laya"][case_id] = {}
        for kind in ("choice", "score", "noul"):
            direct = direct_laya["records"][case_id]["answers"][kind]
            direct_trace = direct["input_trace"]
            for device in ("CPU", "NPU"):
                record = laya_paths["records"][device][case_id]
                trace = record["input_trace"][kind]
                input_check = _compare_laya_input(direct_trace, trace)
                output_check = _compare_laya_outputs(direct, record["results"][kind], kind)
                comparison["laya"][case_id].setdefault(kind, {})[device] = {
                    "input": input_check, "output": output_check,
                    "screen": record["screen"],
                }
            comparison["laya"][case_id][kind]["NPU"]["cpu_to_npu_output"] = (
                _compare_laya_custom_outputs(
                    laya_paths["records"]["CPU"][case_id]["results"],
                    laya_paths["records"]["NPU"][case_id]["results"], kind,
                )
            )
        sokudan_case = sokudan["records"][case_id]
        comparison["sokudan"][case_id] = {}
        adapter_trace = sokudan_case["adapter"]["trace"]
        direct_questions = sokudan_case["official_api"]
        for kind in ("choice", "score", "noul"):
            question_id = direct_questions[kind]["question_id"]
            direct = direct_questions[kind]
            prepared = adapter_trace["prepared_inputs"][question_id]
            input_check = _compare_sokudan_input(direct["input_trace"], prepared)
            adapter_answer = sokudan_case["adapter"]["results"][kind]
            adapter_calibration = {
                key: adapter_trace["official_output"].get(key)
                for key in ("calibrated", "calibrated_answers", "encoding")
            }
            comparison["sokudan"][case_id][kind] = {
                "input": input_check,
                "official_answer_equals_adapter_raw_api": (
                    _canonical(sokudan_case["official_api"][kind]["official_api_answer"])
                    == _canonical(adapter_trace["official_output"]["answers"][question_id])
                ),
                "official_api_output": direct["normalized"],
                "adapter_output": _normalise_adapter_answer(kind, adapter_answer),
                "normalized_output_match": _sokudan_outputs_match(
                    kind, direct["normalized"], _normalise_adapter_answer(kind, adapter_answer)),
                "screen_match": (sokudan_case["official_api_screen"]
                                 == sokudan_case["adapter"]["screen"]),
                "question_contract_match": _canonical(direct["question"])
                == _canonical(adapter_trace["questions"].get(question_id)),
                "calibration_application_match": _canonical(
                    sokudan_case["official_api_calibration"]) == _canonical(adapter_calibration),
                "official_api_screen": sokudan_case["official_api_screen"],
                "adapter_screen": sokudan_case["adapter"]["screen"],
            }
    laya_rows: dict[str, dict[str, list[float]]] = {"CPU": {}, "NPU": {}}
    decision_counts: dict[str, dict[str, int]] = {"CPU": {"matches": 0, "total": 0},
                                                   "NPU": {"matches": 0, "total": 0}}
    for fixture in DIAGNOSTIC_CASES:
        case_id = fixture["id"]
        for kind in ("choice", "score", "noul"):
            cpu_output = comparison["laya"][case_id][kind]["CPU"]["output"]
            npu_output = comparison["laya"][case_id][kind]["NPU"]["cpu_to_npu_output"]
            decision_counts["CPU"]["total"] += 1
            decision_counts["CPU"]["matches"] += int(cpu_output["decision_match"])
            laya_rows["CPU"].setdefault(kind, []).extend(cpu_output["probability_abs_differences"])
            if kind == "noul" and cpu_output["api_returned_value_abs_difference"] is not None:
                laya_rows["CPU"][kind].append(cpu_output["api_returned_value_abs_difference"])
            decision_counts["NPU"]["total"] += 1
            decision_counts["NPU"]["matches"] += int(npu_output["decision_match"])
            laya_rows["NPU"].setdefault(kind, []).extend(npu_output["probability_abs_differences"])
    gates_result = {}
    for device in ("CPU", "NPU"):
        values = [value for rows in laya_rows[device].values() for value in rows]
        max_difference = max(values, default=0.0)
        mean_difference = sum(values) / len(values) if values else 0.0
        match_ratio = decision_counts[device]["matches"] / max(1, decision_counts[device]["total"])
        if device == "CPU":
            rules = gates["G2"]
            checks = {"max_probability_abs_difference": max_difference <= rules["fp32_max_probability_abs_difference"],
                      "decision_match_ratio": match_ratio >= rules["transformed_decision_match_ratio"]}
        else:
            rules = gates["G3"]
            checks = {"max_probability_abs_difference": max_difference <= rules["fp16_max_probability_abs_difference"],
                      "mean_probability_abs_difference": mean_difference <= rules["fp16_mean_probability_abs_difference"],
                      "decision_match_ratio": match_ratio >= rules["transformed_decision_match_ratio"]}
        gates_result[device] = {"max_probability_abs_difference": max_difference,
                                "mean_probability_abs_difference": mean_difference,
                                "probability_value_denominator": len(values),
                                "decision_match_ratio": match_ratio,
                                "decision_denominator": decision_counts[device]["total"],
                                "thresholds": checks, "checks_pass": all(checks.values()),
                                "scope": "four-input diagnostic subset; not a full G2/G3 acceptance run"}
    comparison["laya_subset_gates"] = gates_result
    comparison["laya_comparison_meaning"] = {
        "CPU": "pinned official CPU/FP32 API output versus OpenVINO CPU/FP32 path; G2 criteria",
        "NPU": "OpenVINO CPU/FP32 path versus OpenVINO NPU/PLUGIN path; existing G3 criteria",
        "official_api_vs_npu": "saved per case for diagnosis; not used as a G3 reference",
    }
    comparison["sokudan_exact_input_and_raw_output_matches"] = all(
        all(item["input"][key] for item in comparison["sokudan"][case_id].values()
            for key in ("input_ids_match", "attention_mask_match", "marker_positions_match",
                        "candidate_ids_order_match", "state_token_ids_match"))
        and all(item["input"]["adapter_truncated"] is False
                and item["input"]["direct_truncated"] is False
                for item in comparison["sokudan"][case_id].values())
        and all(item["question_contract_match"] and item["calibration_application_match"]
                and item["official_answer_equals_adapter_raw_api"]
                for item in comparison["sokudan"][case_id].values())
        for case_id in comparison["sokudan"]
    )
    comparison["laya_input_checks_pass"] = all(
        all(item["input"][key] for key in (
            "active_token_ids_match", "state_token_ids_match", "candidate_ids_order_match",
            "candidate_text_order_match", "question_text_match", "marker_positions_match", "qtype_match",
            "question_id_match", "direct_markers_are_masks", "custom_markers_are_masks",
            "question_contract_match", "padding_only_shape_difference",
            "language_parameter_match", "temperature_bucket_match", "effective_temperature_match",
        )) and not item["input"]["truncation"]["direct"] and not item["input"]["truncation"]["custom"]
        for case_id in comparison["laya"]
        for kind in comparison["laya"][case_id]
        for item in comparison["laya"][case_id][kind].values()
    )
    comparison["sokudan_output_and_screen_checks_pass"] = all(
        all(item["normalized_output_match"] and item["screen_match"]
            for item in comparison["sokudan"][case_id].values())
        for case_id in comparison["sokudan"]
    )
    comparison["implementation_paths_match"] = (
        comparison["laya_input_checks_pass"]
        and comparison["sokudan_exact_input_and_raw_output_matches"]
        and comparison["sokudan_output_and_screen_checks_pass"]
        and all(item["checks_pass"] for item in gates_result.values())
    )
    return comparison


def _run_child(root: Path, run_id: str, phase: str, python: Path) -> None:
    if not python.is_file():
        raise DiagnosticError(f"PYTHON_ENVIRONMENT_MISSING_{phase.upper()}")
    command = [str(python), "-m", "som_probe.diagnostics", "--stage", phase, "--run-id", run_id]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        failure_code = next((line.split(":", 1)[1] for line in result.stdout.splitlines()
                             if line.startswith("DIAGNOSTIC_FAILURE:")), "CHILD_PROCESS_FAILED")
        raise DiagnosticError(f"{phase.upper()}_{failure_code.strip()}")


def run_fixed_diagnostic(root: Path) -> dict[str, Any]:
    run = Run.create(root, "fixed interactive diagnosis of API, input adapter, NPU and display",
                     dataset_version="dev-v3", profile_applicable=False)
    env = root / ".local"
    try:
        _run_child(root, run.run_id, "laya-direct", env / "venv-eval" / "Scripts" / "python.exe")
        _run_child(root, run.run_id, "laya-paths", env / "venv-run" / "Scripts" / "python.exe")
        _run_child(root, run.run_id, "sokudan", env / "venv-sokudan-cpu" / "Scripts" / "python.exe")
        direct_laya = json.loads((run.path / "laya-direct.json").read_text(encoding="utf-8"))
        laya_paths = json.loads((run.path / "laya-paths.json").read_text(encoding="utf-8"))
        sokudan = json.loads((run.path / "sokudan.json").read_text(encoding="utf-8"))
        comparison = _build_comparison(root, direct_laya, laya_paths, sokudan)
        results = {"laya_direct": direct_laya, "laya_paths": laya_paths, "sokudan": sokudan}
        paths_match = comparison["implementation_paths_match"]
        trigger_mismatches = (_target_mismatches({"laya": direct_laya, "sokudan": sokudan})
                              if paths_match else [])
        contrast_status = ("BLOCKED_IMPLEMENTATION_PATH_MISMATCH" if not paths_match
                           else "NOT_RUN_NO_PROVISIONAL_CHOICE_OR_SCORE_MISMATCH")
        if paths_match and trigger_mismatches:
            _run_child(root, run.run_id, "contrasts-laya", env / "venv-eval" / "Scripts" / "python.exe")
            _run_child(root, run.run_id, "contrasts-sokudan", env / "venv-sokudan-cpu" / "Scripts" / "python.exe")
            results["contrasts_laya"] = json.loads((run.path / "contrasts-laya.json").read_text(encoding="utf-8"))
            results["contrasts_sokudan"] = json.loads((run.path / "contrasts-sokudan.json").read_text(encoding="utf-8"))
            contrast_status = "RUN_AFTER_AI_PROVISIONAL_MISMATCH"
        bundle = {
            "run_id": run.run_id, "source_tree_sha256": run.read_manifest()["source_tree_sha256"],
            "git_revision": run.read_manifest()["git_revision"],
            "worktree_status": run.read_manifest()["worktree_status"],
            "source_run_id": SOURCE_RUN_ID,
            "fixture_sha256": _sha256({"cases": DIAGNOSTIC_CASES, "contrasts": CONTRAST_FIXTURES}),
            "execution_commands": [
                f'"{env / "venv-eval" / "Scripts" / "python.exe"}" -m som_probe.diagnostics --stage laya-direct --run-id {run.run_id}',
                f'"{env / "venv-run" / "Scripts" / "python.exe"}" -m som_probe.diagnostics --stage laya-paths --run-id {run.run_id}',
                f'"{env / "venv-sokudan-cpu" / "Scripts" / "python.exe"}" -m som_probe.diagnostics --stage sokudan --run-id {run.run_id}',
            ],
            "implementation_paths_match": paths_match,
            "provisional_assessment_status": ("ASSESSED_AI_DRAFT_ONLY" if paths_match
                                               else "BLOCKED_TECHNICAL_PATH_MISMATCH"),
            "fixed_cases": list(DIAGNOSTIC_CASES),
            "provisional_target_status": "AI_PROVISIONAL_NOT_HUMAN_REVIEWED",
            "provisional_target_mismatches": trigger_mismatches,
            "contrasts_status": contrast_status,
            "path_comparison": comparison,
            "results": results,
            "conclusion_scope": "technical path diagnosis and fixed synthetic contrasts only; not a formal Japanese quality decision",
        }
        run.write_json("fixed-input-diagnostics.json", bundle)
        if not paths_match:
            run.failure("NPU_DIAGNOSTIC", "IMPLEMENTATION_PATH_MISMATCH",
                        RuntimeError("IMPLEMENTATION_PATH_MISMATCH"),
                        implementation_paths_match=False)
            return {"status": "FAILED", "run_id": run.run_id,
                    "code": "IMPLEMENTATION_PATH_MISMATCH"}
        run.stage("NPU_DIAGNOSTIC", "PASS", fixed_cases=4, contrast_status=contrast_status,
                  source_run_id=SOURCE_RUN_ID, fixture_sha256=bundle["fixture_sha256"])
        return {"status": "COMPLETE", "run_id": run.run_id, "contrast_status": contrast_status,
                "provisional_mismatches": trigger_mismatches}
    except Exception as exc:
        code = getattr(exc, "code", "FIXED_DIAGNOSTIC_FAILED")
        run.failure("NPU_DIAGNOSTIC", code, RuntimeError(code))
        return {"status": "FAILED", "run_id": run.run_id, "code": code}


def _run_stage(root: Path, run: Run, stage: str) -> None:
    if stage == "laya-direct":
        output = _run_laya_direct(root, run)
        run.write_json("laya-direct.json", output)
        run.stage("TRYOUT_DIRECT_LAYA", "PASS", cases=4, question_types=3)
    elif stage == "laya-paths":
        output = _run_laya_paths(root, run)
        run.write_json("laya-paths.json", output)
        run.stage("TRYOUT_LAYA_PATHS", "PASS", cases=4, question_types=3,
                  requested_devices=["CPU", "NPU"], compiler_type="PLUGIN")
    elif stage == "sokudan":
        output = _run_sokudan(root, run)
        run.write_json("sokudan.json", output)
        run.stage("TRYOUT_SOKUDAN_PATHS", "PASS", cases=4, question_types=3,
                  backend="torch", device="cpu", dtype="float32")
    elif stage in {"contrasts-laya", "contrasts-sokudan"}:
        model = stage.removeprefix("contrasts-")
        output = _run_contrasts(root, run, model)
        run.write_json(f"{stage}.json", output)
        run.stage(f"TRYOUT_CONTRAST_{model.upper()}", "PASS", examples=len(output["results"]))
    else:
        raise DiagnosticError("DIAGNOSTIC_STAGE_INVALID")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("laya-direct", "laya-paths", "sokudan",
                                             "contrasts-laya", "contrasts-sokudan"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        _run_stage(root, open_run(root, args.run_id), args.stage)
        print("DIAGNOSTIC_STAGE_COMPLETE")
        return 0
    except Exception as exc:
        code = getattr(exc, "code", "DIAGNOSTIC_STAGE_FAILED")
        print(f"DIAGNOSTIC_FAILURE:{code}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

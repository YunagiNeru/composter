"""Candidate-A fixed CPU trials and comparisons against saved dev-v3 outputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .storage import Run, open_run
from .tryout_comparison import FIXED_CASES
from .zero_shot import (
    LANGUAGE_CONDITIONS,
    MODEL_DIRECTORY,
    ZeroShotError,
    classify,
    json_sha256,
    load_config,
    reduce_score,
    sha256_file,
    verify_model_files,
)


LAYA_REFERENCE_RUN_ID = "20261025-041751-58713e9e"
SOKUDAN_REFERENCE_RUN_ID = "20261025-053953-96d1b720"
ZERO_SHOT_TERMINAL_STAGE = "ZERO_SHOT_CPU"
REPEATED_SAMPLE_COUNT = 20
IDLE_WAIT_SECONDS = 60
EXPECTED_DATASET_SHA256 = "8bc720289b99cb8041acf6fc012169cdcb5461a7882e919620d1d9dbc31ec4fb"
EXPECTED_CONTRACT_SHA256 = "8dfaa7b1aa53d261d9d5bdc30441dc407c8add4dc74ae96a90cd69cc07725c5c"


def _strict_json(text: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj

    return json.loads(text, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [_strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_bytes(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_dev_cases(root: Path) -> list[dict[str, Any]]:
    data_path = root / "datasets/dev-v3.jsonl"
    manifest_path = root / "datasets/manifest-v3.json"
    contract_path = root / "configs/evaluation-contract-v2.json"
    if sha256_file(data_path) != EXPECTED_DATASET_SHA256 or sha256_file(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise ZeroShotError("DEV_V3_OR_CONTRACT_FINGERPRINT_MISMATCH")
    manifest = _strict_json(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("dataset_version") != "dev-v3"
            or manifest.get("splits", {}).get("dev", {}).get("sha256") != EXPECTED_DATASET_SHA256
            or manifest.get("scoring_contract", {}).get("sha256") != EXPECTED_CONTRACT_SHA256
            or manifest.get("splits", {}).get("dev", {}).get("cases") != 60):
        raise ZeroShotError("DEV_V3_MANIFEST_MISMATCH")
    rows = _read_jsonl(data_path)
    if len(rows) != 60 or len({row.get("case_id") for row in rows}) != 60:
        raise ZeroShotError("DEV_V3_CASE_COUNT_OR_ID_INVALID")
    cases = []
    for row in rows:
        if not isinstance(row.get("state"), str) or not isinstance(row.get("questions"), dict):
            raise ZeroShotError("DEV_V3_CASE_INVALID")
        if len(row["questions"]) != 1:
            raise ZeroShotError("DEV_V3_QUESTION_COUNT_INVALID")
        question_id, question = next(iter(row["questions"].items()))
        expected = row.get("expected", {}).get(question_id)
        if not isinstance(expected, dict) or not isinstance(expected.get("allowed_answers"), list):
            raise ZeroShotError("DEV_V3_EXPECTATION_INVALID")
        question_type = question.get("type")
        if question_type not in {"choice", "score", "bool"}:
            raise ZeroShotError("DEV_V3_QUESTION_TYPE_INVALID")
        dimension = "noul_deadline" if question_type == "bool" else question_type
        cases.append({
            "case_id": row["case_id"],
            "dataset": "dev-v3",
            "question_id": question_id,
            "dimension": dimension,
            "state": row["state"],
            "allowed_answers": expected["allowed_answers"],
            "rationale": expected.get("rationale"),
            "verification_status": expected.get("verification_status", "ai_draft_unverified"),
            "target_errors": expected.get("target_errors", []),
        })
    return cases


def _fixed_cases(root: Path) -> list[dict[str, Any]]:
    result = []
    score_components = {
        "score-none-minutes": {"date_info": False, "clock_time": False},
        "score-none-material": {"date_info": False, "clock_time": False},
        "score-date-tomorrow": {"date_info": True, "clock_time": False},
        "score-date-next-week": {"date_info": True, "clock_time": False},
        "score-time-15": {"date_info": False, "clock_time": True},
        "score-time-afternoon": {"date_info": False, "clock_time": True},
        "score-both-tomorrow": {"date_info": True, "clock_time": True},
        "score-both-weekday": {"date_info": True, "clock_time": True},
    }
    for fixture in FIXED_CASES:
        dimension = fixture["dimension"]
        # Existing four notification-preference controls remain separate from dev-v3 Noul.
        if dimension == "noul":
            dimension = "notification_preference_diagnostic"
        case = {
            "case_id": fixture["case_id"],
            "dataset": "previous-fixed-tryout-controls",
            "dimension": dimension,
            "state": fixture["state"],
            "allowed_answers": [fixture["expected"]],
            "rationale": fixture["rationale"],
            "verification_status": "ai_fixed_technical_expectation_unverified",
            "target_errors": [],
        }
        if fixture["case_id"] in score_components:
            case["expected_components"] = score_components[fixture["case_id"]]
        result.append(case)
    custom = _strict_json((root / "datasets/zero-shot-controls-v1.json").read_text(encoding="utf-8"))
    if (custom.get("created_before_candidate_a_inference") is not True
            or custom.get("verification_status") != "ai_draft_unverified"):
        raise ZeroShotError("FIXED_CONTROL_FREEZE_INVALID")
    result.extend({**case, "dataset": custom["fixture_version"],
                   "verification_status": custom["verification_status"]}
                  for case in custom.get("cases", []))
    ids = [case["case_id"] for case in result]
    if len(ids) != len(set(ids)):
        raise ZeroShotError("FIXED_CONTROL_ID_DUPLICATE")
    return result


def _case_result(session: Any, condition: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    dimension = case["dimension"]
    if dimension == "score":
        date_info = classify(session, case["state"], "date_info", condition)
        clock_time = classify(session, case["state"], "clock_time", condition)
        if date_info["status"] == "LENGTH_EXCEEDED" or clock_time["status"] == "LENGTH_EXCEEDED":
            overall = {"status": "LENGTH_EXCEEDED"}
            reduced = {"stage_prediction": None, "score_expected_value_from_official_scores": None}
        elif date_info["status"] != "PASS" or clock_time["status"] != "PASS":
            overall = {"status": "CONTRACT_ERROR"}
            reduced = {"stage_prediction": None, "score_expected_value_from_official_scores": None}
        else:
            overall = {"status": "PASS"}
            reduced = reduce_score(date_info, clock_time)
        if overall["status"] == "PASS" and reduced.get("stage_status") == "HOLD":
            overall = {"status": "HOLD"}
        output = {"date_info": date_info, "clock_time": clock_time, **reduced}
        decision = reduced.get("stage_prediction")
        expected_value = reduced.get("score_expected_value_from_official_scores")
    elif dimension in {"choice", "noul_deadline", "notification_preference_diagnostic"}:
        classification = classify(session, case["state"], dimension, condition)
        overall = {"status": classification["status"]}
        output = {"classification": classification}
        predicted = classification.get("common_prediction", {}).get("decision")
        if overall["status"] == "PASS" and classification.get("common_prediction", {}).get("status") == "HOLD":
            overall = {"status": "HOLD"}
        if dimension in {"noul_deadline", "notification_preference_diagnostic"}:
            decision = True if predicted == "yes" else False if predicted == "no" else None
            score_ids = classification.get("candidate_ids", [])
            score_values = classification.get("official_scores_not_calibrated", [])
            expected_value = (score_values[score_ids.index("yes")]
                              if classification.get("status") == "PASS" and "yes" in score_ids else None)
        else:
            decision = predicted
            expected_value = None
    else:
        raise ZeroShotError("FIXTURE_DIMENSION_INVALID")
    end_wall = time.perf_counter()
    end_cpu = time.process_time()
    case_input_sha = json_sha256({"state": case["state"], "dimension": dimension,
                                  "condition": condition, "expected_contract": "evaluation-v2"})
    return {
        **case,
        "state_sha256": _sha256_bytes(case["state"]),
        "case_input_sha256": case_input_sha,
        "status": overall["status"],
        "common_prediction": decision,
        "score_expected_value_not_api_returned_score": expected_value,
        "model_specific_output": output,
        "e2e_time_ms": (end_wall - started_wall) * 1000.0,
        "process_cpu_time_ms": (end_cpu - started_cpu) * 1000.0,
    }


def _allowed(case: dict[str, Any]) -> list[Any]:
    return case.get("allowed_answers", [])


def _metric_summary(cases: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    typed = [case for case in cases if case.get("dimension") == dimension]
    total = len(typed)
    by_status = {status: sum(case.get("status") == status for case in typed)
                 for status in ("PASS", "HOLD", "LENGTH_EXCEEDED", "CONTRACT_ERROR", "FAIL")}
    if dimension == "choice":
        labels = ["task", "none", "need_info"]
        matrix = {truth: {pred: 0 for pred in labels + ["HOLD", "ERROR"]} for truth in labels}
        correct = 0
        scorable = 0
        for case in typed:
            allowed = _allowed(case)
            if len(allowed) != 1 or allowed[0] not in labels:
                continue
            scorable += 1
            pred = case.get("common_prediction")
            column = pred if pred in labels else "HOLD" if case.get("status") == "HOLD" else "ERROR"
            matrix[allowed[0]][column] += 1
            correct += pred in allowed
        per_class = {}
        f1s = []
        for label in labels:
            tp = matrix[label][label]
            fp = sum(matrix[truth][label] for truth in labels if truth != label)
            fn = sum(value for column, value in matrix[label].items() if column != label)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                                "support": sum(matrix[label].values())}
            f1s.append(f1)
        return {
            "total_cases_denominator": total,
            "single_label_scorable_cases": scorable,
            "status_counts": by_status,
            "prediction_counts": {label: sum(case.get("common_prediction") == label for case in typed)
                                  for label in labels} | {"HOLD": sum(case.get("status") == "HOLD" for case in typed)},
            "confusion_matrix": matrix,
            "class_metrics": per_class,
            "macro_f1": sum(f1s) / len(f1s),
            "accuracy_all_scorable_with_hold_as_wrong": correct / scorable if scorable else None,
        }
    if dimension == "score":
        expected_cases = [case for case in typed if len(_allowed(case)) == 1 and type(_allowed(case)[0]) in (int, float)]
        decided = [case for case in expected_cases if type(case.get("common_prediction")) is int]
        predicted_values = [case.get("score_expected_value_not_api_returned_score") for case in expected_cases
                            if type(case.get("score_expected_value_not_api_returned_score")) in (int, float)]
        stage_correct = sum(case["common_prediction"] in _allowed(case) for case in decided)
        # Build a serializable stage confusion matrix, counting abstentions explicitly.
        stage_confusions = {str(value): {str(pred): 0 for pred in (0, 1, 2, "HOLD", "ERROR")} for value in range(3)}
        for case in expected_cases:
            target = str(_allowed(case)[0])
            pred = case.get("common_prediction")
            key = str(pred) if type(pred) is int else "HOLD" if case.get("status") == "HOLD" else "ERROR"
            stage_confusions[target][key] += 1
        stage_only = [abs(case["common_prediction"] - _allowed(case)[0]) for case in decided]
        mae = (sum(abs(float(case["score_expected_value_not_api_returned_score"]) - float(_allowed(case)[0]))
                   for case in expected_cases
                   if type(case.get("score_expected_value_not_api_returned_score")) in (int, float)) /
               len(predicted_values)) if predicted_values else None
        subgroup = {}
        for category, predicate in (
            ("date_only", lambda value: value.get("date_info") is True and value.get("clock_time") is False),
            ("clock_time_only", lambda value: value.get("date_info") is False and value.get("clock_time") is True),
            ("date_and_clock_time", lambda value: value.get("date_info") is True and value.get("clock_time") is True),
            ("neither", lambda value: value.get("date_info") is False and value.get("clock_time") is False),
        ):
            selected = [case for case in typed if predicate(case.get("expected_components", {}))]
            subgroup[category] = {
                "cases": len(selected),
                "stage_correct": sum(case.get("common_prediction") in _allowed(case) for case in selected),
                "component_false_positives": sum(
                    (case.get("model_specific_output", {}).get("date_info_prediction") is True and not case["expected_components"]["date_info"])
                    or (case.get("model_specific_output", {}).get("clock_time_prediction") is True and not case["expected_components"]["clock_time"])
                    for case in selected),
                "component_false_negatives": sum(
                    (case.get("model_specific_output", {}).get("date_info_prediction") is False and case["expected_components"]["date_info"])
                    or (case.get("model_specific_output", {}).get("clock_time_prediction") is False and case["expected_components"]["clock_time"])
                    for case in selected),
            }
        return {
            "total_cases_denominator": total,
            "expected_label_cases": len(expected_cases),
            "decided_stage_cases": len(decided),
            "status_counts": by_status,
            "stage_accuracy_decided_only": stage_correct / len(decided) if decided else None,
            "stage_accuracy_all_expected_with_hold_as_wrong": stage_correct / len(expected_cases) if expected_cases else None,
            "stage_confusion_matrix": stage_confusions,
            "stage_mae_decided_only": sum(stage_only) / len(stage_only) if stage_only else None,
            "score_expectation_mae_over_available_values": mae,
            "score_expectation_value_count": len(predicted_values),
            "score_expectation_is_model_api_score": False,
            "subgroups_with_explicit_component_targets": subgroup,
        }
    if dimension in {"noul_deadline", "notification_preference_diagnostic"}:
        expected_cases = [case for case in typed if len(_allowed(case)) == 1 and type(_allowed(case)[0]) is bool]
        tp = tn = fp = fn = 0
        for case in expected_cases:
            pred = case.get("common_prediction")
            truth = _allowed(case)[0]
            if pred is True and truth is True:
                tp += 1
            elif pred is False and truth is False:
                tn += 1
            elif pred is True and truth is False:
                fp += 1
            elif pred is False and truth is True:
                fn += 1
        return {
            "total_cases_denominator": total,
            "expected_label_cases": len(expected_cases),
            "status_counts": by_status,
            "prediction_counts": {
                "true": sum(case.get("common_prediction") is True for case in typed),
                "false": sum(case.get("common_prediction") is False for case in typed),
                "HOLD": sum(case.get("status") == "HOLD" for case in typed),
            },
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "notification_candidates_false_positive": fp,
            "notification_candidates_missed": fn,
            "valid_prediction_accuracy": (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else None,
            "accuracy_all_expected_with_hold_as_wrong": (tp + tn) / len(expected_cases) if expected_cases else None,
            "probabilities_are_calibrated": False,
        }
    raise ZeroShotError("METRIC_DIMENSION_INVALID")


def _all_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {dimension: _metric_summary(cases, dimension)
            for dimension in ("choice", "score", "noul_deadline", "notification_preference_diagnostic")}


def _baseline_rows(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    laya_run = open_run(root, LAYA_REFERENCE_RUN_ID)
    sokudan_run = open_run(root, SOKUDAN_REFERENCE_RUN_ID)
    for run, stage in ((laya_run, "G1"), (sokudan_run, "SOKUDAN_CPU")):
        manifest = run.read_manifest()
        if (manifest.get("status") != "COMPLETE"
                or manifest.get("dataset", {}).get("dataset_version") != "dev-v3"
                or manifest.get("dataset", {}).get("dataset_sha256") != EXPECTED_DATASET_SHA256
                or manifest.get("dataset", {}).get("scoring_contract_sha256") != EXPECTED_CONTRACT_SHA256
                or manifest.get("stages", {}).get(stage, {}).get("status") not in {"PASS", "UNVERIFIED_LABEL_REFERENCE_ONLY"}):
            raise ZeroShotError("SAVED_BASELINE_IDENTITY_MISMATCH")
    laya_path = laya_run.path / "reference.jsonl"
    sokudan_path = sokudan_run.path / "sokudan-cases.jsonl"
    laya_data = _read_jsonl(laya_path)
    sokudan_data = _read_jsonl(sokudan_path)
    dev = _read_dev_cases(root)
    expected_ids = {case["case_id"] for case in dev}
    if ({row.get("case_id") for row in laya_data} != expected_ids
            or {row.get("case_id") for row in sokudan_data} != expected_ids
            or len(laya_data) != 60 or len(sokudan_data) != 60):
        raise ZeroShotError("SAVED_BASELINE_CASE_SET_MISMATCH")
    laya_rows = []
    for row in laya_data:
        qtype = row["question_type"]
        prediction = row.get("decision")
        if qtype == "noul":
            prediction = row.get("boolean_result")
        laya_rows.append({
            "case_id": row["case_id"], "question_type": qtype,
            "common_prediction": prediction,
            "score_value": row.get("returned_score"),
            "status": "PASS",
        })
    sokudan_rows = []
    for row in sokudan_data:
        question = row["questions"][0]
        adapter = question["adapter"]
        qtype = question["question_type"]
        prediction = adapter.get("decision")
        if qtype == "bool":
            prediction = adapter.get("evaluator_boolean")
        sokudan_rows.append({
            "case_id": row["case_id"], "question_type": qtype,
            "common_prediction": prediction,
            "score_value": adapter.get("api_returned_value") if qtype == "score" else None,
            "status": "PASS" if row.get("status") == "PASS" and question.get("status") == "PASS" else "FAIL",
        })
    baseline_meta = {
        "laya_run_id": LAYA_REFERENCE_RUN_ID,
        "laya_output_sha256": sha256_file(laya_path),
        "laya_commit": laya_run.read_manifest().get("git_revision"),
        "sokudan_run_id": SOKUDAN_REFERENCE_RUN_ID,
        "sokudan_output_sha256": sha256_file(sokudan_path),
        "sokudan_commit": sokudan_run.read_manifest().get("git_revision"),
        "dataset_version": "dev-v3",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "scoring_contract_sha256": EXPECTED_CONTRACT_SHA256,
    }
    return {"laya": laya_rows, "sokudan": sokudan_rows}, baseline_meta


def _baseline_metrics(dev_cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["case_id"]: row for row in rows}
    adapted = []
    for case in dev_cases:
        row = by_id[case["case_id"]]
        qtype = case["dimension"]
        dimension = "noul_deadline" if qtype == "noul_deadline" else qtype
        result = {**case, "dimension": dimension, "status": row["status"],
                  "common_prediction": row["common_prediction"],
                  "score_expected_value_not_api_returned_score": row.get("score_value"),
                  "model_specific_output": {}}
        adapted.append(result)
    return _all_metrics(adapted)


def _majority_baselines(dev_cases: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for dimension, labels in (("choice", ["task", "none", "need_info"]),
                              ("score", [0, 1, 2]), ("noul_deadline", [False, True])):
        selected = [case for case in dev_cases if case["dimension"] == dimension and len(_allowed(case)) == 1]
        counts = {str(label).lower() if isinstance(label, bool) else str(label):
                  sum(_allowed(case)[0] == label for case in selected) for label in labels}
        if not selected:
            continue
        maximum = max(counts.values())
        winner = next(label for label in labels if counts[str(label).lower() if isinstance(label, bool) else str(label)] == maximum)
        predicted = 1 if dimension == "score" else winner
        correct = sum(predicted in _allowed(case) for case in selected)
        entry = {"label_counts": counts, "majority_prediction": predicted,
                 "n": len(selected), "accuracy_same_dev_v3_draft_set": correct / len(selected),
                 "independent_baseline": False}
        if dimension == "score":
            entry["mae_constant_one"] = sum(abs(1 - float(_allowed(case)[0])) for case in selected) / len(selected)
        out[dimension] = entry
    return out


def _metric_comparison(cases: list[dict[str, Any]], baselines: dict[str, list[dict[str, Any]]],
                       dev_cases: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_dev = [case for case in cases if case["dataset"] == "dev-v3"]
    out = {"candidate_a": _all_metrics(candidate_dev)}
    for model, rows in baselines.items():
        out[model] = _baseline_metrics(dev_cases, rows)
    out["same_dev_v3_draft_majority_baselines"] = _majority_baselines(dev_cases)
    out["label_state"] = "ai_draft_unverified"
    out["official_japanese_quality_status"] = "UNDETERMINED"
    return out


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _timing_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for dimension in ("choice", "score", "noul_deadline", "notification_preference_diagnostic"):
        selected = [case for case in cases if case["dimension"] == dimension and case["dataset"] == "dev-v3"
                    and case["status"] in {"PASS", "HOLD"}]
        wall = [case["e2e_time_ms"] for case in selected]
        cpu = [case["process_cpu_time_ms"] for case in selected]
        result[dimension] = {
            "measured_cases": len(wall),
            "e2e_wall_p50_ms": _percentile(wall, 0.50),
            "e2e_wall_p95_ms": _percentile(wall, 0.95),
            "process_cpu_time_p50_ms": _percentile(cpu, 0.50),
            "process_cpu_time_p95_ms": _percentile(cpu, 0.95),
            "errors": sum(case["status"] in {"CONTRACT_ERROR", "FAIL"} for case in cases if case["dimension"] == dimension),
            "length_exceeded": sum(case["status"] == "LENGTH_EXCEEDED" for case in cases if case["dimension"] == dimension),
        }
    return result


def _memory_snapshot() -> dict[str, Any]:
    from .performance import _memory

    return _memory()


def _load_candidate(model_dir: Path, threads: int) -> tuple[Any, float, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if threads not in (1, 2):
        raise ZeroShotError("CPU_THREAD_COUNT_INVALID")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="sdpa",
    )
    model.to("cpu")
    model.eval()
    elapsed = time.perf_counter() - started
    actual_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()})
    config = model.config
    if (actual_devices != ["cpu"] or parameter_dtypes != ["torch.float32"]
            or config.architectures != ["ModernBertForSequenceClassification"]
            or config.id2label != {0: "not_entailment", 1: "entailment"}):
        raise ZeroShotError("ACTUAL_MODEL_DEVICE_PRECISION_OR_LABELS_MISMATCH")
    if torch.cuda.is_available() and any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ZeroShotError("GPU_DEVICE_USAGE_DETECTED")

    class Session:
        entailment_id = int(config.label2id["entailment"])

        def forward(self, tokenized: Any) -> list[list[float]]:
            input_cpu = {key: tensor.to("cpu") for key, tensor in tokenized.items()}
            with torch.inference_mode():
                logits = model(**input_cpu).logits
            return [[float(value) for value in row] for row in logits.detach().cpu().tolist()]

        def __init__(self) -> None:
            self.tokenizer = tokenizer

    runtime = {
        "repository": "Horizon-Labs/multilingual-zeroshot-small",
        "revision": "3fe1f0c48b589cf6220fd21a628a1dd56d3d41f8",
        "architecture": config.architectures[0],
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameter_dtypes": parameter_dtypes,
        "actual_parameter_devices": actual_devices,
        "requested_device": "cpu",
        "backend": "torch",
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_used": False,
        "threads": threads,
        "inter_op_threads": torch.get_num_interop_threads(),
        "intra_op_threads": torch.get_num_threads(),
        "tokenizer_class": type(tokenizer).__name__,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
    }
    return Session(), elapsed, runtime


def _execute_trial(root: Path, run: Run, condition_name: str, trial: int) -> dict[str, Any]:
    if condition_name not in LANGUAGE_CONDITIONS or trial not in (1, 2, 3):
        raise ZeroShotError("TRIAL_CONFIGURATION_INVALID")
    if Path(sys.executable).resolve() != (root / ".local/venv-zero-shot-cpu/Scripts/python.exe").resolve():
        raise ZeroShotError("DEDICATED_CPU_ENVIRONMENT_NOT_ACTIVE")
    model_lock, evaluation = load_config(root)
    model_dir, model_file_hashes = verify_model_files(root, model_lock)
    baselines, baseline_meta = _baseline_rows(root)
    dev_cases = _read_dev_cases(root)
    fixed_cases = _fixed_cases(root)
    all_cases = dev_cases + fixed_cases
    condition = evaluation["conditions"][condition_name]
    condition_sha = json_sha256(condition)
    model_lock_sha = sha256_file(root / "configs/zero-shot-models.lock.json")
    eval_config_sha = sha256_file(root / "configs/zero-shot-evaluation-v1.json")
    control_sha = sha256_file(root / "datasets/zero-shot-controls-v1.json")
    source_sha = sha256_file(Path(__file__))
    adapter_sha = sha256_file(root / "som_probe/zero_shot.py")
    model_loaded, load_seconds, runtime = _load_candidate(model_dir, 2)

    memory_stages = [{"stage": "model_loaded", "captured_at_seconds": 0.0,
                      "memory": _memory_snapshot()}]
    # The first fixed example is repeated below; this first call is the prescribed first-inference memory point.
    sample = next(case for case in fixed_cases if case["case_id"] == "work-none-eaten")
    sample_out = _case_result(model_loaded, condition, sample)
    memory_stages.append({"stage": "first_fixed_inference_complete",
                          "captured_at_seconds": load_seconds + sample_out["e2e_time_ms"] / 1000.0,
                          "memory": _memory_snapshot()})
    cpu_before_idle = time.process_time()
    wall_before_idle = time.perf_counter()
    time.sleep(IDLE_WAIT_SECONDS)
    idle_cpu = time.process_time() - cpu_before_idle
    idle_wall = time.perf_counter() - wall_before_idle
    memory_stages.append({"stage": "idle_wait_complete", "idle_wait_seconds": idle_wall,
                          "idle_process_cpu_time_seconds": idle_cpu,
                          "memory": _memory_snapshot()})

    case_results = [_case_result(model_loaded, condition, case) for case in all_cases]
    memory_stages.append({"stage": "all_fixed_inputs_complete", "memory": _memory_snapshot()})
    repeated_times = []
    repeated_cpu = []
    for _ in range(REPEATED_SAMPLE_COUNT):
        repeated = _case_result(model_loaded, condition, sample)
        if repeated["status"] != "PASS":
            raise ZeroShotError("REPEATED_FIXED_INFERENCE_FAILED")
        repeated_times.append(repeated["e2e_time_ms"])
        repeated_cpu.append(repeated["process_cpu_time_ms"])
    memory_stages.append({"stage": "repeated_fixed_inference_complete",
                          "repeat_count": REPEATED_SAMPLE_COUNT,
                          "memory": _memory_snapshot()})
    memory_stages.append({"stage": "before_process_exit", "memory": _memory_snapshot()})

    summaries = _all_metrics(case_results)
    comparison = _metric_comparison(case_results, baselines, dev_cases)
    result = {
        "run_id": run.run_id,
        "condition": condition_name,
        "condition_definition_sha256": condition_sha,
        "trial": trial,
        "model": {
            "id": "candidate-a",
            "revision": model_lock["candidate_a"]["revision"],
            "files_sha256": model_file_hashes,
            "model_lock_sha256": model_lock_sha,
            "evaluation_config_sha256": eval_config_sha,
            "fixed_controls_sha256": control_sha,
            "adapter_source_sha256": adapter_sha,
            "runner_source_sha256": source_sha,
        },
        "dataset": {
            "version": "dev-v3",
            "sha256": EXPECTED_DATASET_SHA256,
            "scoring_contract_sha256": EXPECTED_CONTRACT_SHA256,
            "case_count": len(dev_cases),
            "label_verification_status": "ai_draft_unverified",
        },
        "additional_fixed_controls": {
            "previous_tryout_controls": len(FIXED_CASES),
            "separate_domain_controls": 8,
            "total_controls": len(fixed_cases),
            "label_status": "ai_fixed_technical_expectation_unverified",
            "notification_preference_cases_are_not_dev_v3_deadline_noul": True,
        },
        "execution": runtime,
        "timing": {
            "model_load_seconds": load_seconds,
            "e2e_case_summary": _timing_summary(case_results),
            "fixed_input_repeat_count": REPEATED_SAMPLE_COUNT,
            "fixed_input_repeat_e2e_p50_ms": _percentile(repeated_times, 0.50),
            "fixed_input_repeat_e2e_p95_ms": _percentile(repeated_times, 0.95),
            "fixed_input_repeat_process_cpu_p50_ms": _percentile(repeated_cpu, 0.50),
            "fixed_input_repeat_process_cpu_p95_ms": _percentile(repeated_cpu, 0.95),
            "idle_wait_wall_seconds": idle_wall,
            "idle_wait_process_cpu_time_seconds": idle_cpu,
        },
        "memory": {
            "api": "Windows GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX",
            "working_set_and_commit_charge_separate": True,
            "peak_fields_are_process_lifetime_cumulative": True,
            "stages": memory_stages,
        },
        "status_counts": {status: sum(case["status"] == status for case in case_results)
                          for status in ("PASS", "HOLD", "LENGTH_EXCEEDED", "CONTRACT_ERROR", "FAIL")},
        "metrics_by_dimension_all_cases": summaries,
        "dev_v3_comparison": comparison,
        "baseline_identity": baseline_meta,
        "cases": case_results,
    }
    run.write_jsonl("candidate-cases.jsonl", case_results)
    run.write_json("candidate-result.json", result)
    run.stage(ZERO_SHOT_TERMINAL_STAGE, "PASS",
              candidate="Horizon-Labs/multilingual-zeroshot-small",
              revision=model_lock["candidate_a"]["revision"], condition=condition_name,
              trial=trial, case_count=len(case_results), model_files_sha256=model_file_hashes,
              dev_v3_sha256=EXPECTED_DATASET_SHA256,
              scoring_contract_sha256=EXPECTED_CONTRACT_SHA256,
              candidate_result_sha256=sha256_file(run.path / "candidate-result.json"),
              candidate_cases_sha256=sha256_file(run.path / "candidate-cases.jsonl"),
              actual_device=runtime["actual_parameter_devices"], actual_dtype=runtime["parameter_dtypes"],
              official_japanese_quality_status="UNDETERMINED")
    del model_loaded
    return {"status": "COMPLETE", "run_id": run.run_id,
            "condition": condition_name, "trial": trial,
            "metrics": comparison["candidate_a"]}


def run_child(root: Path, run_id: str, condition: str, trial: int) -> dict[str, Any]:
    run = open_run(root, run_id)
    try:
        result = _execute_trial(root, run, condition, trial)
        return result
    except Exception as exc:
        code = getattr(exc, "code", "CANDIDATE_A_TRIAL_FAILED")
        run.failure(ZERO_SHOT_TERMINAL_STAGE, code, RuntimeError(code),
                    condition=condition, trial=trial)
        return {"status": "FAILED", "run_id": run_id, "condition": condition,
                "trial": trial, "code": code}


def run_all_trials(root: Path, conditions: list[str], trials: int = 3) -> dict[str, Any]:
    if not conditions or any(condition not in LANGUAGE_CONDITIONS for condition in conditions) or trials != 3:
        raise ZeroShotError("TRIAL_PLAN_INVALID")
    expected_python = (root / ".local/venv-zero-shot-cpu/Scripts/python.exe").resolve()
    if not expected_python.is_file() or Path(sys.executable).resolve() != expected_python:
        raise ZeroShotError("DEDICATED_CPU_ENVIRONMENT_NOT_ACTIVE")

    blocked = Run.create(root, "candidate-B blocked before acquisition", dataset_version="dev-v3",
                         profile_applicable=False)
    blocked.blocked(
        ZERO_SHOT_TERMINAL_STAGE,
        "COMMERCIAL_USAGE_REVIEW_PENDING",
        "商用利用条件を確認できていないため採用審査を保留。法的な商用利用禁止を確定した判断ではない。",
        candidate="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
        artifact_acquired=False,
        inference_performed=False,
    )
    trial_results = []
    for condition in conditions:
        for trial in range(1, trials + 1):
            run = Run.create(root, f"candidate-A CPU {condition} trial {trial}",
                             dataset_version="dev-v3", profile_applicable=False)
            command = [str(expected_python), "-m", "som_probe", "zero-shot-cpu",
                       "--condition", condition, "--trial", str(trial),
                       "--run-id", run.run_id, "--child"]
            process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=7200,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            try:
                child_result = json.loads(process.stdout.splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                child_result = {"status": "FAILED", "run_id": run.run_id,
                                "code": f"CHILD_EXIT_{process.returncode}"}
                if process.returncode == 0:
                    run.failure(ZERO_SHOT_TERMINAL_STAGE, "CHILD_OUTPUT_INVALID",
                                RuntimeError("CHILD_OUTPUT_INVALID"), condition=condition, trial=trial)
            if process.returncode != 0 or child_result.get("status") != "COMPLETE":
                trial_results.append({**child_result, "exit_code": process.returncode})
            else:
                trial_results.append(child_result)
    successful = sum(result.get("status") == "COMPLETE" for result in trial_results)
    return {
        "status": "COMPLETE" if successful == len(trial_results) else "FAILED",
        "candidate_a": "Horizon-Labs/multilingual-zeroshot-small",
        "candidate_b": {"status": "BLOCKED", "run_id": blocked.run_id,
                         "reason": "商用利用条件を確認できていないため採用審査を保留"},
        "trials_per_condition": trials,
        "trial_count": len(trial_results),
        "successful_trials": successful,
        "runs": trial_results,
    }

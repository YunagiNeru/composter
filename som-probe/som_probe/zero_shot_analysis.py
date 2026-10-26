"""Re-score immutable candidate-A trial outputs without loading a model."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .storage import Run, open_run
from .zero_shot import ZeroShotError, json_sha256, sha256_file
from .zero_shot_eval import (
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_DATASET_SHA256,
    LANGUAGE_CONDITIONS,
    LAYA_REFERENCE_RUN_ID,
    SOKUDAN_REFERENCE_RUN_ID,
    _all_metrics,
    _baseline_metrics,
    _baseline_rows,
    _majority_baselines,
    _read_dev_cases,
    _read_jsonl,
)


ANALYSIS_STAGE = "ZERO_SHOT_ANALYSIS"


def official_pipeline_case(row: dict[str, Any]) -> dict[str, Any]:
    """Expose the official single-label top class separately from the HOLD adapter."""
    result = copy.deepcopy(row)
    result["policy_prediction"] = row.get("common_prediction")
    result["policy_status"] = row.get("status")
    detail = row.get("model_specific_output", {})
    dimension = row.get("dimension")
    if dimension == "choice":
        classification = detail.get("classification", {})
        if classification.get("status") != "PASS":
            result["status"] = classification.get("status", "CONTRACT_ERROR")
            result["common_prediction"] = None
        else:
            result["status"] = "PASS"
            result["common_prediction"] = classification.get("official_prediction_id")
    elif dimension == "score":
        date_info = detail.get("date_info", {})
        clock_time = detail.get("clock_time", {})
        if date_info.get("status") != "PASS" or clock_time.get("status") != "PASS":
            result["status"] = "LENGTH_EXCEEDED" if "LENGTH_EXCEEDED" in {
                date_info.get("status"), clock_time.get("status")} else "CONTRACT_ERROR"
            result["common_prediction"] = None
            result["score_expected_value_not_api_returned_score"] = None
        else:
            result["status"] = "PASS"
            result["common_prediction"] = int(date_info["official_prediction_id"] == "yes") + int(
                clock_time["official_prediction_id"] == "yes")
            result["score_expected_value_not_api_returned_score"] = detail.get(
                "score_expected_value_from_official_scores")
    elif dimension in {"noul_deadline", "notification_preference_diagnostic"}:
        classification = detail.get("classification", {})
        if classification.get("status") != "PASS":
            result["status"] = classification.get("status", "CONTRACT_ERROR")
            result["common_prediction"] = None
        else:
            result["status"] = "PASS"
            result["common_prediction"] = classification.get("official_prediction_id") == "yes"
    else:
        raise ZeroShotError("ANALYSIS_CASE_DIMENSION_INVALID")
    return result


def _read_source_run(root: Path, run_id: str) -> tuple[Run, dict[str, Any], list[dict[str, Any]]]:
    run = open_run(root, run_id)
    manifest = run.read_manifest()
    if manifest.get("status") != "COMPLETE" or manifest.get("stages", {}).get("ZERO_SHOT_CPU", {}).get("status") != "PASS":
        raise ZeroShotError("CANDIDATE_SOURCE_RUN_NOT_COMPLETE")
    result_path = run.path / "candidate-result.json"
    cases_path = run.path / "candidate-cases.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cases = _read_jsonl(cases_path)
    if (result.get("run_id") != run_id
            or result.get("dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256
            or result.get("dataset", {}).get("scoring_contract_sha256") != EXPECTED_CONTRACT_SHA256
            or len(cases) != 88
            or result.get("execution", {}).get("actual_parameter_devices") != ["cpu"]
            or result.get("execution", {}).get("parameter_dtypes") != ["torch.float32"]
            or result.get("execution", {}).get("cuda_device_used") is not False):
        raise ZeroShotError("CANDIDATE_SOURCE_IDENTITY_MISMATCH")
    result["source_files_sha256"] = {
        "candidate-result.json": sha256_file(result_path),
        "candidate-cases.jsonl": sha256_file(cases_path),
        "manifest.json": sha256_file(run.path / "manifest.json"),
    }
    return run, result, cases


def _prediction_signature(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    view = official_pipeline_case(row)
    detail = view.get("model_specific_output", {})
    if view["dimension"] == "choice":
        scores = detail.get("classification", {}).get("official_scores_not_calibrated")
    elif view["dimension"] == "score":
        scores = [detail.get("date_info", {}).get("official_scores_not_calibrated"),
                  detail.get("clock_time", {}).get("official_scores_not_calibrated")]
    else:
        scores = detail.get("classification", {}).get("official_scores_not_calibrated")
    return row.get("case_id"), view.get("common_prediction"), scores


def _same_case_set(reference: list[dict[str, Any]], other: list[dict[str, Any]]) -> bool:
    ref = {row.get("case_id"): row for row in reference}
    candidate = {row.get("case_id"): row for row in other}
    if len(ref) != 88 or len(candidate) != 88 or set(ref) != set(candidate):
        return False
    for case_id in ref:
        if (ref[case_id].get("case_input_sha256") != candidate[case_id].get("case_input_sha256")
                or ref[case_id].get("state_sha256") != candidate[case_id].get("state_sha256")):
            return False
    return True


def _case_details(cases: list[dict[str, Any]], baseline_rows: dict[str, list[dict[str, Any]]],
                  condition: str) -> list[dict[str, Any]]:
    by_model = {model: {row["case_id"]: row for row in rows}
                for model, rows in baseline_rows.items()}
    result = []
    for raw in cases:
        row = official_pipeline_case(raw)
        case = {
            "case_id": row["case_id"], "dataset": row["dataset"], "dimension": row["dimension"],
            "state": row["state"], "state_sha256": row["state_sha256"],
            "case_input_sha256": row["case_input_sha256"], "allowed_answers": row["allowed_answers"],
            "rationale": row.get("rationale"), "verification_status": row.get("verification_status"),
            "target_errors": row.get("target_errors", []), "expected_components": row.get("expected_components"),
            "condition": condition, "official_pipeline_prediction": row["common_prediction"],
            "policy_prediction": row["policy_prediction"], "policy_status": row["policy_status"],
            "official_output_status": row["status"],
            "score_expected_value_not_api_returned_score": row.get("score_expected_value_not_api_returned_score"),
            "model_specific_output": row["model_specific_output"],
            "e2e_time_ms": row.get("e2e_time_ms"),
            "process_cpu_time_ms": row.get("process_cpu_time_ms"),
        }
        for model in ("laya", "sokudan"):
            prior = by_model[model].get(row["case_id"]) if row["dataset"] == "dev-v3" else None
            case[f"{model}_saved_prediction"] = prior.get("common_prediction") if prior else None
            case[f"{model}_saved_score_value"] = prior.get("score_value") if prior else None
        case["label_accuracy_comparison_is_formal"] = False
        result.append(case)
    return result


def _resource_summary(source: dict[str, Any]) -> dict[str, Any]:
    result = source["candidate-result.json"]
    memory = result["memory"]
    return {
        "run_id": source["run_id"], "condition": source["condition"], "trial": source["trial"],
        "device": result["execution"]["actual_parameter_devices"],
        "dtype": result["execution"]["parameter_dtypes"],
        "threads": result["execution"]["intra_op_threads"],
        "model_load_seconds": result["timing"]["model_load_seconds"],
        "timing": result["timing"],
        "memory_api": memory["api"],
        "memory_stages": memory["stages"],
        "status_counts": result["status_counts"],
        "model_files_sha256": result["model"]["files_sha256"],
        "condition_definition_sha256": result["condition_definition_sha256"],
    }


def run_analysis(root: Path, source_run_ids: list[str], blocked_run_id: str) -> dict[str, Any]:
    if len(source_run_ids) != 6 or len(set(source_run_ids)) != 6:
        raise ZeroShotError("SIX_DISTINCT_SOURCE_RUNS_REQUIRED")
    analysis_run = Run.create(root, "candidate-A frozen CPU output analysis without inference",
                              dataset_version="dev-v3", profile_applicable=False)
    try:
        source_records = [_read_source_run(root, run_id) for run_id in source_run_ids]
        blocked_run = open_run(root, blocked_run_id)
        blocked_manifest = blocked_run.read_manifest()
        if (blocked_manifest.get("status") != "BLOCKED"
                or blocked_manifest.get("stages", {}).get("ZERO_SHOT_CPU", {}).get("status") != "BLOCKED"
                or blocked_manifest.get("stages", {}).get("ZERO_SHOT_CPU", {}).get("artifacts_acquired") is not False
                or blocked_manifest.get("stages", {}).get("ZERO_SHOT_CPU", {}).get("inference_performed") is not False):
            raise ZeroShotError("CANDIDATE_B_BLOCK_RECORD_INVALID")

        by_condition: dict[str, list[tuple[Run, dict[str, Any], list[dict[str, Any]]]]] = {
            condition: [] for condition in LANGUAGE_CONDITIONS
        }
        for record in source_records:
            result = record[1]
            condition = result.get("condition")
            if condition not in by_condition or result.get("trial") not in (1, 2, 3):
                raise ZeroShotError("SOURCE_CONDITION_OR_TRIAL_INVALID")
            by_condition[condition].append(record)
        if any(len(rows) != 3 or {row[1]["trial"] for row in rows} != {1, 2, 3}
               for rows in by_condition.values()):
            raise ZeroShotError("CONDITION_TRIAL_SET_INCOMPLETE")

        dev_cases = _read_dev_cases(root)
        baselines, baseline_identity = _baseline_rows(root)
        baseline_metrics = {model: _baseline_metrics(dev_cases, rows)
                            for model, rows in baselines.items()}
        majority = _majority_baselines(dev_cases)
        condition_results = {}
        case_details = []
        source_file_hashes = {}
        resource_rows = []
        for condition, records in by_condition.items():
            records.sort(key=lambda item: item[1]["trial"])
            reference_cases = records[0][2]
            raw_hashes = []
            official_trial_metrics = []
            policy_trial_metrics = []
            for run, result, cases in records:
                if result.get("model", {}).get("revision") != "3fe1f0c48b589cf6220fd21a628a1dd56d3d41f8":
                    raise ZeroShotError("CANDIDATE_MODEL_REVISION_MISMATCH")
                if result.get("condition_definition_sha256") != records[0][1].get("condition_definition_sha256"):
                    raise ZeroShotError("CONDITION_DEFINITION_MISMATCH")
                if not _same_case_set(reference_cases, cases):
                    raise ZeroShotError("TRIAL_INPUT_CASES_MISMATCH")
                identity_payload = [
                    {"case_id": row["case_id"], "model_specific_output": row["model_specific_output"]}
                    for row in cases
                ]
                raw_hashes.append(json_sha256(identity_payload))
                official_cases = [official_pipeline_case(row) for row in cases]
                official_trial_metrics.append(_all_metrics(
                    [row for row in official_cases if row["dataset"] == "dev-v3"]))
                policy_trial_metrics.append(_all_metrics(
                    [row for row in cases if row["dataset"] == "dev-v3"]))
                resource_rows.append(_resource_summary({
                    "run_id": run.run_id, "condition": condition, "trial": result["trial"],
                    "candidate-result.json": result,
                }))
                source_file_hashes[run.run_id] = result["source_files_sha256"]
            consistency = {
                "inference_output_sha256_by_trial": raw_hashes,
                "bitwise_identical_outputs_across_three_runs": len(set(raw_hashes)) == 1,
                "official_prediction_agreement_by_case": {},
            }
            trial_views = [[official_pipeline_case(row) for row in record[2]] for record in records]
            reference_by_id = {row["case_id"]: row for row in trial_views[0]}
            for other in trial_views[1:]:
                other_by_id = {row["case_id"]: row for row in other}
                for case_id, reference in reference_by_id.items():
                    status = consistency["official_prediction_agreement_by_case"].setdefault(case_id, 0)
                    if reference.get("common_prediction") == other_by_id[case_id].get("common_prediction"):
                        consistency["official_prediction_agreement_by_case"][case_id] = status + 1
            consistency["all_case_predictions_same_across_trials"] = all(
                count == 2 for count in consistency["official_prediction_agreement_by_case"].values())

            dev_metrics = official_trial_metrics[0]
            condition_results[condition] = {
                "official_pipeline_metrics_by_trial": official_trial_metrics,
                "abstaining_adapter_metrics_by_trial": policy_trial_metrics,
                "saved_baseline_metrics": baseline_metrics,
                "same_dev_v3_draft_majority_baselines": majority,
                "trial_consistency": consistency,
                "candidate_a_vs_saved_baseline": {
                    dimension: {
                        "candidate_a_official_pipeline": dev_metrics[dimension],
                        "laya_saved": baseline_metrics["laya"][dimension],
                        "sokudan_saved": baseline_metrics["sokudan"][dimension],
                        "same_dev_v3_majority_reference": majority.get(dimension),
                    }
                    for dimension in ("choice", "score", "noul_deadline")
                },
                "fixed_control_metrics_by_group": {
                    dataset: _all_metrics([official_pipeline_case(row) for row in reference_cases
                                           if row["dataset"] == dataset])
                    for dataset in sorted({row["dataset"] for row in reference_cases if row["dataset"] != "dev-v3"})
                },
            }
            case_details.extend(_case_details(reference_cases, baselines, condition))

        lock_sha = sha256_file(root / "configs/zero-shot-models.lock.json")
        evaluation_sha = sha256_file(root / "configs/zero-shot-evaluation-v1.json")
        controls_sha = sha256_file(root / "datasets/zero-shot-controls-v1.json")
        result = {
            "analysis_run_id": analysis_run.run_id,
            "analysis_kind": "saved-output reanalysis only; no model loaded and no inference performed",
            "candidate_a_revision": "3fe1f0c48b589cf6220fd21a628a1dd56d3d41f8",
            "candidate_b": {
                "status": "BLOCKED",
                "run_id": blocked_run_id,
                "reason": "商用利用条件を確認できていないため採用審査を保留。商用利用禁止が法的に確定したという判断ではない。",
                "artifacts_acquired": False,
                "inference_performed": False,
            },
            "dataset": {"version": "dev-v3", "sha256": EXPECTED_DATASET_SHA256,
                        "scoring_contract_sha256": EXPECTED_CONTRACT_SHA256,
                        "label_verification_status": "ai_draft_unverified",
                        "official_japanese_quality_status": "UNDETERMINED"},
            "comparison_conditions": {
                "english": "official multilingual zero-shot recommendation: English labels and template",
                "japanese": "fixed separate diagnostic condition with Japanese labels and template",
            },
            "input_policy": {"truncation_forbidden": True, "max_pair_tokens": 8192,
                             "dev_v3_denominator": 60,
                             "diagnostic_controls_separate": True},
            "metrics_semantics": {
                "official_pipeline": "top candidate from softmax of entailment logits; native model output, not calibrated confidence",
                "abstaining_adapter": "reports HOLD when official winner's own pair is not entailment or when the top score ties",
                "candidate_A_comparison_uses": "official pipeline output",
                "quality_status": "reference metrics against unverified AI draft labels only",
            },
            "source_run_ids": source_run_ids,
            "source_file_hashes": source_file_hashes,
            "blocked_candidate_run_id": blocked_run_id,
            "candidate_lock_sha256": lock_sha,
            "evaluation_config_sha256": evaluation_sha,
            "fixed_controls_sha256": controls_sha,
            "baseline_identity": baseline_identity,
            "conditions": condition_results,
            "resource_trials": resource_rows,
            "cases": case_details,
        }
        analysis_run.write_json("analysis-result.json", result)
        analysis_run.write_jsonl("analysis-cases.jsonl", case_details)
        analysis_run.stage(ANALYSIS_STAGE, "PASS", source_run_ids=source_run_ids,
                           blocked_candidate_run_id=blocked_run_id,
                           dataset_sha256=EXPECTED_DATASET_SHA256,
                           scoring_contract_sha256=EXPECTED_CONTRACT_SHA256,
                           analysis_result_sha256=sha256_file(analysis_run.path / "analysis-result.json"),
                           analysis_cases_sha256=sha256_file(analysis_run.path / "analysis-cases.jsonl"),
                           inference_performed=False,
                           official_japanese_quality_status="UNDETERMINED")
        return {"status": "COMPLETE", "run_id": analysis_run.run_id,
                "source_run_ids": source_run_ids,
                "official_japanese_quality_status": "UNDETERMINED"}
    except Exception as exc:
        code = getattr(exc, "code", "ZERO_SHOT_ANALYSIS_FAILED")
        analysis_run.failure(ANALYSIS_STAGE, code, RuntimeError(code))
        return {"status": "FAILED", "run_id": analysis_run.run_id, "code": code}

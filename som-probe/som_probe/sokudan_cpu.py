"""Fixed-artifact Sokudan CPU comparison against dev-v3 and saved Laya output."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import load_dataset
from .storage import Run, open_run, safe_message


class SokudanComparisonError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise SokudanComparisonError("OUTPUT_VALUE_INVALID", f"{field}が有限な数値ではありません。")
    return float(value)


def _display_probabilities(answer: dict[str, Any], keys: list[str]) -> list[float]:
    values = answer.get("probabilities")
    if not isinstance(values, dict) or list(values) != keys:
        raise SokudanComparisonError("PROBABILITY_KEYS_INVALID", "公式APIの確率キーまたは候補順が不正です。")
    result = [_number(values[key], f"probabilities[{key}]") for key in keys]
    if any(not 0.0 <= value <= 1.0 for value in result):
        raise SokudanComparisonError("PROBABILITY_RANGE_INVALID", "公式APIの表示確率が範囲外です。")
    # The official API rounds each displayed probability to four decimal places.
    if abs(sum(result) - 1.0) > len(result) * 0.0000501:
        raise SokudanComparisonError("PROBABILITY_SUM_INVALID", "表示確率の合計が丸め誤差の範囲を外れています。")
    return result


def adapt_official_answer(kind: str, answer: dict[str, Any], candidate_ids: list[Any]) -> dict[str, Any]:
    """Map one official API answer without changing its values or decision."""
    if not isinstance(answer, dict):
        raise SokudanComparisonError("ANSWER_SHAPE_INVALID", "公式APIの回答がオブジェクトではありません。")
    api_type = "bool" if kind == "noul" else kind
    if answer.get("type") != api_type:
        raise SokudanComparisonError("ANSWER_TYPE_INVALID", "公式APIの回答型が質問型と一致しません。")

    common = {
        "question_type": kind,
        "candidate_ids": list(candidate_ids),
        "api_returned_value": None,
        "api_display_probabilities": None,
        "probability_origin": None,
        "stage_prediction": None,
        "stage_prediction_source": None,
        "evaluator_boolean": None,
        "evaluator_boolean_source": None,
    }
    if kind == "choice":
        if not candidate_ids or any(not isinstance(value, str) for value in candidate_ids):
            raise SokudanComparisonError("CANDIDATE_IDS_INVALID", "Choiceの候補IDが不正です。")
        decision = answer.get("choice")
        if decision not in candidate_ids:
            raise SokudanComparisonError("UNKNOWN_CANDIDATE", "公式APIのChoiceが候補IDにありません。")
        probabilities = _display_probabilities(answer, candidate_ids)
        common.update({
            "decision": decision,
            "decision_slot": candidate_ids.index(decision),
            "api_returned_value": decision,
            "api_display_probabilities": probabilities,
            "probability_origin": "official_api_display_rounded_4",
        })
        return common

    if kind == "score":
        expected_ids = list(range(len(candidate_ids)))
        if candidate_ids != expected_ids or len(candidate_ids) < 2:
            raise SokudanComparisonError("SCORE_CANDIDATE_IDS_INVALID", "Scoreの候補IDが0始まりの段階と一致しません。")
        probabilities = _display_probabilities(answer, [str(value) for value in candidate_ids])
        returned_score = _number(answer.get("score"), "score")
        if not 0.0 <= returned_score <= float(len(candidate_ids) - 1):
            raise SokudanComparisonError("SCORE_RANGE_INVALID", "公式APIの連続Scoreが尺度の範囲外です。")
        stage = max(range(len(probabilities)), key=probabilities.__getitem__)
        common.update({
            "decision": stage,
            "decision_slot": stage,
            "api_returned_value": returned_score,
            "api_display_probabilities": probabilities,
            "probability_origin": "official_api_display_rounded_4",
            "stage_prediction": stage,
            "stage_prediction_source": "argmax_of_official_api_display_rounded_4_probabilities",
        })
        return common

    if kind == "noul":
        if candidate_ids != [False, True]:
            raise SokudanComparisonError("NOUL_CANDIDATE_IDS_INVALID", "Noulの候補順はfalse, trueでなければなりません。")
        probability = _number(answer.get("noul"), "noul")
        if not 0.0 <= probability <= 1.0:
            raise SokudanComparisonError("NOUL_PROBABILITY_INVALID", "公式APIのP(true)が範囲外です。")
        common.update({
            "api_returned_value": probability,
            "api_returned_value_rounding": "official_api_python_round_4",
            "evaluator_boolean": probability >= 0.5,
            "evaluator_boolean_source": "returned_P_true_greater_than_or_equal_to_0.5",
        })
        return common

    raise SokudanComparisonError("QUESTION_TYPE_INVALID", "未対応の質問型です。")


def _normalise_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _locked_environment(root: Path) -> dict[str, Any]:
    expected_prefix = (root / ".local" / "venv-sokudan-cpu").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise SokudanComparisonError("DEDICATED_VENV_REQUIRED", "比較専用venvのpython.exeで実行してください。")
    if sys.version_info[:3] != (3, 11, 14) or struct.calcsize("P") * 8 != 64:
        raise SokudanComparisonError("PYTHON_RUNTIME_MISMATCH", "比較環境はPython 3.11.14 x64で固定されています。")
    lock_path = root / "requirements" / "sokudan-cpu-win-x64.lock"
    lock_text = lock_path.read_text(encoding="utf-8")
    expected: dict[str, str] = {}
    for line in lock_text.splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\\\s]+)", line)
        if match:
            expected[_normalise_name(match.group(1))] = match.group(2)
    mismatches = []
    installed: dict[str, str] = {}
    for name, version in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append({"package": name, "expected": version, "actual": "MISSING"})
            continue
        installed[name] = actual
        if actual != version:
            mismatches.append({"package": name, "expected": version, "actual": actual})
    if mismatches:
        raise SokudanComparisonError("DEPENDENCY_LOCK_MISMATCH", "専用venvの依存版が固定ロックと一致しません。")
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "architecture_bits": struct.calcsize("P") * 8,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "dependency_lock_sha256": _sha256_file(lock_path),
        "installed_packages": installed,
    }


def _verify_local_artifacts(root: Path) -> dict[str, Any]:
    command = [sys.executable, str(root / "scripts" / "prepare_sokudan_cpu.py"), "--verify-only"]
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SokudanComparisonError("ARTIFACT_VERIFY_FAILED", "固定取得物のローカル検査を起動できません。") from exc
    if result.returncode != 0:
        raise SokudanComparisonError("ARTIFACT_VERIFY_FAILED", "固定取得物のローカル検査に失敗しました。")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SokudanComparisonError("ARTIFACT_VERIFY_OUTPUT_INVALID", "取得物検査の結果JSONが不正です。") from exc
    if data.get("status") != "PASS":
        raise SokudanComparisonError("ARTIFACT_VERIFY_FAILED", "固定取得物のローカル検査がPASSではありません。")
    return data


def _verify_installed_source(package_root: Path, wheel_path: Path) -> dict[str, Any]:
    checked = 0
    with zipfile.ZipFile(wheel_path) as archive:
        for name in archive.namelist():
            if not name.startswith("sokudan/") or not name.endswith(".py"):
                continue
            installed = package_root.parent / Path(*name.split("/"))
            if not installed.is_file() or installed.read_bytes() != archive.read(name):
                raise SokudanComparisonError("INSTALLED_SOURCE_MISMATCH", "venv内の公式APIソースが固定wheelと一致しません。")
            checked += 1
    if checked != 51:
        raise SokudanComparisonError("INSTALLED_SOURCE_INCOMPLETE", "固定wheelのPythonソース件数が想定と一致しません。")
    return {"wheel_python_files_matched": checked}


def _question_kind(question: dict[str, Any]) -> str:
    return "noul" if question["type"] == "bool" else question["type"]


def _candidate_ids(question: dict[str, Any]) -> list[Any]:
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return list(range(len(question["criteria"])))
    return [False, True]


def _prepare_question_inputs(agent: Any, state: str,
                             questions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from sokudan.encoding.question import QuestionTooLongError, encode_joint
    from sokudan.schema.question import parse_questions

    parsed = parse_questions(questions)
    if len(parsed) != 1:
        raise SokudanComparisonError("QUESTION_COUNT_INVALID", "dev-v3の各ケースは質問1件でなければなりません。")
    results: dict[str, dict[str, Any]] = {}
    for question_id, question in parsed.items():
        try:
            encoded = encode_joint(question, state, agent.tokenizer,
                                   input_order=agent.backend.input_order)
        except QuestionTooLongError:
            # Re-encode only for exact length accounting. The oversized sequence is
            # never passed to the model; prediction is rejected below.
            encoded = encode_joint(question, state, agent.tokenizer,
                                   max_tokens=2_147_483_647,
                                   input_order=agent.backend.input_order)
        if len(encoded.marker_positions) != len(question.labels):
            raise SokudanComparisonError("MARKER_COUNT_MISMATCH", "質問候補数と特殊トークン位置が一致しません。")
        if any(encoded.input_ids[index] != agent.tokenizer.mask_token_id
               for index in encoded.marker_positions):
            raise SokudanComparisonError("MARKER_POSITION_INVALID", "候補マーカーがトークナイザーのmask位置と一致しません。")
        state_token_count = _encode_state_length(agent, state)
        required_total = len(encoded.input_ids) + max(0, state_token_count - encoded.n_state_tokens)
        input_tensor_sha256 = _sha256_bytes(_canonical({
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "marker_positions": encoded.marker_positions,
            "ordered": encoded.ordered,
            "input_order": agent.backend.input_order,
        }).encode("utf-8"))
        suffix_tokens = len(encoded.input_ids) - encoded.n_question_tokens - encoded.n_state_tokens
        results[question_id] = {
            "question_tokens": encoded.n_question_tokens,
            "suffix_tokens": suffix_tokens,
            "state_tokens": state_token_count,
            "required_total_tokens": required_total,
            "truncated": encoded.truncated,
            "candidate_count": len(question.labels),
            "input_tensor_sha256": input_tensor_sha256,
        }
    return results


def _encode_state_length(agent: Any, state: str) -> int:
    return len(agent.tokenizer(state, add_special_tokens=False)["input_ids"])


def _input_fingerprint(case: dict[str, Any]) -> str:
    payload = {"state": case["state"], "questions": case["questions"]}
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(serialized.encode("utf-8"))


def _validate_calibration(agent: Any, official_output: dict[str, Any], qids: list[str],
                          question_types: dict[str, str], question_counts: dict[str, int],
                          expected_temperature: float) -> None:
    selected = {key: float(value) for key, value in agent.temperatures.items()}
    if selected != {("bool", 2): expected_temperature}:
        raise SokudanComparisonError("CALIBRATION_SELECTION_MISMATCH", "読み込まれた校正対象または温度が固定配布物と一致しません。")
    expected_ids = [qid for qid in qids
                    if question_types[qid] == "noul" and question_counts[qid] == 2]
    if official_output.get("calibrated_answers") != expected_ids:
        raise SokudanComparisonError("CALIBRATION_APPLICATION_MISMATCH", "公式APIが校正を適用した回答IDが期待と一致しません。")
    if official_output.get("calibrated") is not bool(expected_ids):
        raise SokudanComparisonError("CALIBRATION_FLAG_MISMATCH", "公式APIの校正適用フラグが期待と一致しません。")


def _laya_prediction(row: dict[str, Any]) -> dict[str, Any]:
    kind = row.get("question_type")
    candidate_ids = row.get("candidate_ids")
    probabilities = row.get("probabilities")
    if kind not in {"choice", "score", "noul"} or not isinstance(candidate_ids, list):
        raise SokudanComparisonError("LAYA_REFERENCE_INVALID", "保存済みLaya出力の質問型または候補IDが不正です。")
    if not isinstance(probabilities, list) or len(probabilities) != len(candidate_ids):
        raise SokudanComparisonError("LAYA_REFERENCE_INVALID", "保存済みLaya出力の確率幅が候補数と一致しません。")
    values = [_number(value, "Laya probability") for value in probabilities]
    if kind == "choice":
        decision = row.get("decision")
        if decision not in candidate_ids:
            raise SokudanComparisonError("LAYA_REFERENCE_INVALID", "保存済みLaya Choiceが候補外です。")
        return {"question_type": kind, "decision": decision, "stage_prediction": None,
                "continuous_score": None, "true_probability": None,
                "display_probabilities": values, "candidate_ids": candidate_ids}
    if kind == "score":
        stage = row.get("decision_slot")
        returned_score = _number(row.get("returned_score"), "Laya returned_score")
        if type(stage) is not int or not 0 <= stage < len(candidate_ids):
            raise SokudanComparisonError("LAYA_REFERENCE_INVALID", "保存済みLaya Score段階が不正です。")
        return {"question_type": kind, "decision": stage, "stage_prediction": stage,
                "continuous_score": returned_score, "true_probability": None,
                "display_probabilities": values, "candidate_ids": candidate_ids}
    probability = _number(row.get("returned_noul"), "Laya returned_noul")
    boolean = row.get("boolean_result")
    if type(boolean) is not bool or not 0.0 <= probability <= 1.0:
        raise SokudanComparisonError("LAYA_REFERENCE_INVALID", "保存済みLaya Noul出力が不正です。")
    return {"question_type": kind, "decision": boolean, "stage_prediction": None,
            "continuous_score": None, "true_probability": probability,
            "display_probabilities": values, "candidate_ids": candidate_ids}


def _normalised_sokudan(row: dict[str, Any]) -> dict[str, Any]:
    adapted = row["adapter"]
    kind = adapted["question_type"]
    return {
        "question_type": kind,
        "decision": adapted["decision"] if kind != "noul" else adapted["evaluator_boolean"],
        "stage_prediction": adapted["stage_prediction"],
        "continuous_score": adapted["api_returned_value"] if kind == "score" else None,
        "true_probability": adapted["api_returned_value"] if kind == "noul" else None,
        "display_probabilities": adapted["api_display_probabilities"],
        "candidate_ids": adapted["candidate_ids"],
    }


def _classes(kind: str, question: dict[str, Any]) -> list[Any]:
    if kind == "choice":
        return list(question["criteria"])
    if kind == "score":
        return list(range(len(question["criteria"])))
    return [False, True]


def _key(value: Any) -> str:
    return _canonical(value)


def _majority(cases: list[dict[str, Any]], kind: str, candidates: list[Any]) -> dict[str, Any]:
    votes: dict[str, float] = {_key(candidate): 0.0 for candidate in candidates}
    by_key = {_key(candidate): candidate for candidate in candidates}
    for case in cases:
        question_id, _question = next(iter(case["questions"].items()))
        label = case["expected"][question_id]["allowed_answers"]
        for answer in label:
            votes[_key(answer)] += 1.0 / len(label)
    selected_key = min(votes, key=lambda item: (-votes[item], candidates.index(by_key[item])))
    selected = by_key[selected_key]
    return {"answer": selected, "votes": votes[selected_key],
            "source": "このdev-v3の未確認ラベル全件から算出",
            "independent_predeclared_baseline": False}


def _per_class_metrics(gold_pred: list[tuple[Any, Any]], candidates: list[Any]) -> dict[str, Any]:
    singleton = [(gold[0], pred) for allowed, pred in gold_pred if len(allowed) == 1 for gold in [allowed]]
    confusion = {str(gold): {str(pred): 0 for pred in candidates} for gold in candidates}
    class_metrics: dict[str, Any] = {}
    f1s = []
    for actual, predicted in singleton:
        confusion[str(actual)][str(predicted)] += 1
    for candidate in candidates:
        tp = sum(actual == candidate and predicted == candidate for actual, predicted in singleton)
        fp = sum(actual != candidate and predicted == candidate for actual, predicted in singleton)
        fn = sum(actual == candidate and predicted != candidate for actual, predicted in singleton)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        class_metrics[str(candidate)] = {"precision": precision, "recall": recall,
                                         "f1": f1, "support": tp + fn,
                                         "predicted_count": tp + fp}
    return {"confusion_matrix_singleton_labels": confusion,
            "confusion_matrix_denominator": len(singleton),
            "class_precision_recall_f1": class_metrics,
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "macro_f1_classes": len(candidates)}


def _model_metrics(cases: list[dict[str, Any]], normalized: dict[tuple[str, str], dict[str, Any]],
                   statuses: Counter[str], kind: str) -> dict[str, Any]:
    selected_cases = []
    candidate_ids: list[Any] | None = None
    for case in cases:
        qid, question = next(iter(case["questions"].items()))
        case_kind = _question_kind(question)
        if case_kind != kind:
            continue
        row = normalized.get((case["case_id"], qid))
        if row is not None:
            selected_cases.append((case, qid, question, row))
            if candidate_ids is None:
                candidate_ids = _classes(kind, question)
    requested = sum(_question_kind(next(iter(case["questions"].values()))) == kind for case in cases)
    candidate_ids = candidate_ids or _classes(kind, next(
        next(iter(case["questions"].values())) for case in cases
        if _question_kind(next(iter(case["questions"].values()))) == kind
    ))
    denominator = len(selected_cases)
    result: dict[str, Any] = {
        "requested_cases": requested,
        "valid_output_denominator": denominator,
        "status_counts": dict(statuses),
        "candidate_ids": candidate_ids,
        "predicted_counts": {str(value): 0 for value in candidate_ids},
        "draft_label_matches": 0,
        "draft_label_match_rate": 0.0,
        "draft_label_status": "ai_draft_unverified",
    }
    gold_pred: list[tuple[list[Any], Any]] = []
    majority = _majority([case for case in cases
                          if _question_kind(next(iter(case["questions"].values()))) == kind],
                         kind, candidate_ids)
    baseline_correct = 0
    for case, qid, _question, row in selected_cases:
        allowed = case["expected"][qid]["allowed_answers"]
        prediction = row["decision"]
        if prediction not in result["predicted_counts"] and str(prediction) not in result["predicted_counts"]:
            raise SokudanComparisonError("PREDICTION_CLASS_INVALID", "予測候補が評価クラスにありません。")
        result["predicted_counts"][str(prediction)] += 1
        result["draft_label_matches"] += int(any(_key(prediction) == _key(answer) for answer in allowed))
        baseline_correct += int(any(_key(majority["answer"]) == _key(answer) for answer in allowed))
        gold_pred.append((allowed, prediction))
    result["draft_label_match_rate"] = result["draft_label_matches"] / denominator if denominator else 0.0
    result["same_dev_majority_baseline"] = majority | {
        "draft_label_matches": baseline_correct,
        "match_rate": baseline_correct / denominator if denominator else 0.0,
        "denominator": denominator,
    }

    if kind == "choice":
        result.update(_per_class_metrics(gold_pred, candidate_ids))
        result["accuracy"] = result["draft_label_match_rate"]
        result["accuracy_denominator"] = denominator
    elif kind == "score":
        stage_matches = 0
        abs_errors: list[float] = []
        normalised_errors: list[float] = []
        baseline_errors: list[float] = []
        stage_pred: list[tuple[list[Any], Any]] = []
        for case, qid, question, row in selected_cases:
            allowed = case["expected"][qid]["allowed_answers"]
            stage = row["stage_prediction"]
            stage_matches += int(any(_key(stage) == _key(answer) for answer in allowed))
            scale_max = len(question["criteria"]) - 1
            value = row["continuous_score"]
            abs_errors.append(min(abs(float(value) - float(answer)) for answer in allowed))
            normalised_errors.append(abs_errors[-1] / scale_max)
            baseline_errors.append(min(abs(1.0 - float(answer)) for answer in allowed))
            stage_pred.append((allowed, stage))
        result.update(_per_class_metrics(stage_pred, candidate_ids))
        result["stage_accuracy"] = stage_matches / denominator if denominator else 0.0
        result["stage_accuracy_denominator"] = denominator
        result["continuous_score_mae"] = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
        result["continuous_score_mae_denominator"] = len(abs_errors)
        result["continuous_score_normalized_mae"] = (
            sum(normalised_errors) / len(normalised_errors) if normalised_errors else 0.0
        )
        result["continuous_score_normalized_mae_denominator"] = len(normalised_errors)
        result["continuous_score_normalization"] = "各ケースの絶対誤差を段階上限(K-1)で割り、その平均"
        result["always_return_1_mae"] = sum(baseline_errors) / len(baseline_errors) if baseline_errors else 0.0
        result["always_return_1_mae_denominator"] = len(baseline_errors)
    else:
        true_count = result["predicted_counts"].get("true", 0)
        false_count = result["predicted_counts"].get("false", 0)
        # JSON stringification of Python booleans is title-cased.
        true_count = result["predicted_counts"].get("True", true_count)
        false_count = result["predicted_counts"].get("False", false_count)
        confusion = {"false": {"false": 0, "true": 0}, "true": {"false": 0, "true": 0}}
        false_positive = false_negative = 0
        for case, qid, _question, row in selected_cases:
            gold = case["expected"][qid]["allowed_answers"]
            predicted = row["decision"]
            if len(gold) == 1:
                confusion[str(gold[0]).lower()][str(predicted).lower()] += 1
            if predicted is True and False in gold:
                false_positive += 1
            if predicted is False and True in gold:
                false_negative += 1
        result.update({
            "predicted_true_count": true_count,
            "predicted_false_count": false_count,
            "confusion_matrix_singleton_labels": confusion,
            "confusion_matrix_denominator": sum(map(sum, (row.values() for row in confusion.values()))),
            "notification_out_of_scope_false_positives": false_positive,
            "notification_candidate_misses": false_negative,
            "real_notification_operations_tested": False,
            "api_value_and_evaluator_separate": True,
        })
        returned = [row["true_probability"] for _, _, _, row in selected_cases]
        result["returned_P_true_values"] = returned
        result["returned_P_true_value_denominator"] = len(returned)
    return result


def _pairwise_metrics(cases: list[dict[str, Any]], sokudan: dict[tuple[str, str], dict[str, Any]],
                      laya: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("choice", "score", "noul"):
        pairs = []
        case_differences = []
        probability_differences: list[float] = []
        value_differences: list[float] = []
        for case in cases:
            qid, question = next(iter(case["questions"].items()))
            if _question_kind(question) != kind:
                continue
            left = sokudan.get((case["case_id"], qid))
            right = laya.get((case["case_id"], qid))
            if left is None or right is None:
                continue
            if left["candidate_ids"] != right["candidate_ids"]:
                raise SokudanComparisonError("CANDIDATE_ORDER_MISMATCH", "LayaとSokudanの候補IDまたは順序が一致しません。")
            pairs.append((case["case_id"], qid, left, right))
            difference: dict[str, Any] = {
                "case_id": case["case_id"], "question_id": qid,
                "sokudan_prediction": left["decision"],
                "saved_laya_prediction": right["decision"],
                "prediction_match": left["decision"] == right["decision"],
            }
            if kind in {"choice", "score"}:
                for a, b in zip(left["display_probabilities"], right["display_probabilities"], strict=True):
                    probability_differences.append(abs(float(a) - float(b)))
                difference["candidate_probabilities"] = [
                    {"candidate_id": candidate_id,
                     "sokudan_official_display_probability": a,
                     "laya_saved_probability": b,
                     "absolute_difference": abs(float(a) - float(b))}
                    for candidate_id, a, b in zip(left["candidate_ids"], left["display_probabilities"],
                                                 right["display_probabilities"], strict=True)
                ]
                if kind == "score":
                    difference["sokudan_returned_score"] = left["continuous_score"]
                    difference["laya_saved_returned_score"] = right["continuous_score"]
                    difference["returned_score_absolute_difference"] = abs(
                        float(left["continuous_score"]) - float(right["continuous_score"])
                    )
            else:
                value_differences.append(abs(float(left["true_probability"]) - float(right["true_probability"])))
                difference.update({
                    "sokudan_returned_P_true": left["true_probability"],
                    "laya_saved_returned_P_true": right["true_probability"],
                    "P_true_absolute_difference": abs(float(left["true_probability"])
                                                       - float(right["true_probability"])),
                })
            case_differences.append(difference)
        matches = sum(left["decision"] == right["decision"] for _, _, left, right in pairs)
        item: dict[str, Any] = {
            "paired_cases": len(pairs),
            "prediction_matches": matches,
            "prediction_agreement_rate": matches / len(pairs) if pairs else 0.0,
            "prediction_agreement_denominator": len(pairs),
            "case_differences": case_differences,
        }
        if kind in {"choice", "score"}:
            item["candidate_probability_abs_difference_mean"] = (
                sum(probability_differences) / len(probability_differences) if probability_differences else 0.0
            )
            item["candidate_probability_value_denominator"] = len(probability_differences)
            item["probability_comparison_note"] = (
                "Sokudan側は公式APIの小数4桁表示値、Laya側は保存済みCPU確率。"
            )
        else:
            item["returned_P_true_abs_difference_mean"] = (
                sum(value_differences) / len(value_differences) if value_differences else 0.0
            )
            item["returned_P_true_value_denominator"] = len(value_differences)
            item["probability_comparison_note"] = (
                "Sokudan側は公式APIの小数4桁返却値、Laya側は保存済みreturned_noul。"
            )
        result[kind] = item
    return result


def _case_diagnostics(cases: list[dict[str, Any]], case_records: list[dict[str, Any]],
                      laya_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sok_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for case_record in case_records:
        for row in case_record["questions"]:
            sok_by_key[(case_record["case_id"], row.get("question_id"))] = row
    laya_by_key = {(row["case_id"], row["question_id"]): row for row in laya_rows}
    results = []
    for case in cases:
        qid, question = next(iter(case["questions"].items()))
        key = (case["case_id"], qid)
        kind = _question_kind(question)
        allowed = case["expected"][qid]["allowed_answers"]
        sok_row = sok_by_key.get(key, {})
        sok_norm = _normalised_sokudan(sok_row) if sok_row.get("status") == "PASS" else None
        laya_raw = laya_by_key.get(key)
        laya_norm = _laya_prediction(laya_raw) if laya_raw is not None else None
        item: dict[str, Any] = {
            "case_id": case["case_id"],
            "question_id": qid,
            "question_type": kind,
            "label_state": "ai_draft_unverified",
            "draft_allowed_answers": allowed,
            "sokudan_status": sok_row.get("status", "HELD"),
            "sokudan_error_code": sok_row.get("error_code"),
            "saved_laya_available": laya_norm is not None,
        }
        if sok_norm is not None:
            predicted = sok_norm["decision"]
            item["sokudan"] = {
                "prediction": predicted,
                "stage_prediction": sok_norm["stage_prediction"],
                "continuous_score": sok_norm["continuous_score"],
                "returned_P_true": sok_norm["true_probability"],
                "display_probabilities": sok_norm["display_probabilities"],
                "candidate_ids": sok_norm["candidate_ids"],
            }
            item["sokudan_match_to_unverified_draft"] = any(
                _key(predicted) == _key(answer) for answer in allowed
            )
        else:
            item["sokudan"] = None
            item["sokudan_match_to_unverified_draft"] = None
        if laya_norm is not None:
            item["saved_laya"] = {
                "prediction": laya_norm["decision"],
                "stage_prediction": laya_norm["stage_prediction"],
                "continuous_score": laya_norm["continuous_score"],
                "returned_P_true": laya_norm["true_probability"],
                "display_probabilities": laya_norm["display_probabilities"],
                "candidate_ids": laya_norm["candidate_ids"],
            }
            item["sokudan_matches_saved_laya"] = (
                sok_norm["decision"] == laya_norm["decision"] if sok_norm is not None else None
            )
        else:
            item["saved_laya"] = None
            item["sokudan_matches_saved_laya"] = None
        if kind == "score":
            item["score_stage_match_to_unverified_draft"] = (
                any(_key(sok_norm["stage_prediction"]) == _key(answer) for answer in allowed)
                if sok_norm is not None else None
            )
            item["score_continuous_abs_error_to_unverified_draft"] = (
                min(abs(float(sok_norm["continuous_score"]) - float(answer)) for answer in allowed)
                if sok_norm is not None else None
            )
        if kind == "noul":
            predicted = sok_norm["decision"] if sok_norm is not None else None
            item["notification_out_of_scope_false_positive_against_draft"] = (
                predicted is True and False in allowed if predicted is not None else None
            )
            item["notification_candidate_miss_against_draft"] = (
                predicted is False and True in allowed if predicted is not None else None
            )
        results.append(item)
    return results


def _validate_reference_run(root: Path, dataset: dict[str, Any], run_id: str) -> tuple[Run, list[dict[str, Any]]]:
    reference_run = open_run(root, run_id)
    manifest = reference_run.read_manifest()
    identity = dataset["identity"]
    reference_identity = manifest.get("dataset", {})
    keys = ("dataset_version", "dataset_sha256", "scoring_contract_version", "scoring_contract_sha256")
    if any(reference_identity.get(key) != identity.get(key) for key in keys):
        raise SokudanComparisonError("LAYA_DATASET_IDENTITY_MISMATCH", "保存済みLaya Runとdev-v3の版・入力・採点契約が一致しません。")
    if manifest.get("stages", {}).get("G1", {}).get("status") != "UNVERIFIED_LABEL_REFERENCE_ONLY":
        raise SokudanComparisonError("LAYA_REFERENCE_STAGE_INVALID", "保存済みLaya CPU参照Runに有効なG1結果がありません。")
    path = reference_run.path / "reference.jsonl"
    if not path.is_file():
        raise SokudanComparisonError("LAYA_REFERENCE_MISSING", "指定されたRunに保存済みLaya参照がありません。")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    inputs_path = reference_run.path / "inputs.jsonl"
    if not inputs_path.is_file():
        raise SokudanComparisonError("LAYA_INPUTS_MISSING", "保存済みLaya Runの入力記録がありません。")
    input_rows = [json.loads(line) for line in inputs_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    expected_keys = {(case["case_id"], qid) for case in dataset["cases"]
                     for qid in case["questions"]}
    actual_keys = [(row.get("case_id"), row.get("question_id")) for row in rows]
    input_keys = [(row.get("case_id"), row.get("question_id")) for row in input_rows]
    if (len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys
            or len(input_keys) != len(set(input_keys)) or set(input_keys) != expected_keys):
        raise SokudanComparisonError("LAYA_REFERENCE_CASE_SET_MISMATCH", "保存済みLaya参照の件数またはcase_id集合がdev-v3と一致しません。")
    by_output_key = {(row["case_id"], row["question_id"]): row for row in rows}
    by_input_key = {(row["case_id"], row["question_id"]): row for row in input_rows}
    cases_by_id = {case["case_id"]: case for case in dataset["cases"]}
    for key in expected_keys:
        output = by_output_key[key]
        source = by_input_key[key]
        question = cases_by_id[key[0]]["questions"][key[1]]
        expected_candidates = _candidate_ids(question)
        if (output.get("input_sha256") != source.get("input_sha256")
                or output.get("evaluation_key") != source.get("evaluation_key")
                or output.get("candidate_ids") != expected_candidates
                or source.get("candidate_ids") != expected_candidates
                or output.get("question_type") != _question_kind(question)
                or source.get("question_type") != _question_kind(question)):
            raise SokudanComparisonError("LAYA_INPUT_IDENTITY_MISMATCH", "保存済みLayaの入力・候補・質問型がdev-v3または入力台帳と一致しません。")
    return reference_run, rows


def _load_lock(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "configs" / "sokudan-cpu.lock.json"
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SokudanComparisonError("SOKUDAN_LOCK_INVALID", "Sokudan固定ロックJSONを読み取れません。") from exc
    if data.get("schema_version") != 1 or data.get("execution", {}).get("backend") != "torch":
        raise SokudanComparisonError("SOKUDAN_LOCK_INVALID", "Sokudan固定ロックの版または実行先が不正です。")
    return data, hashlib.sha256(raw).hexdigest()


def _run_environment(root: Path) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    environment = _locked_environment(root)
    lock, lock_sha256 = _load_lock(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = str((root / ".local" / "cache" / "sokudan-hf").resolve())
    verification = _verify_local_artifacts(root)
    wheel_path = (root / lock["package"]["wheel"]["path"]).resolve()
    import torch
    import sokudan

    if sokudan.__version__ != lock["package"]["version"]:
        raise SokudanComparisonError("SOKUDAN_VERSION_MISMATCH", "読み込まれた公式API版が固定ロックと一致しません。")
    source_check = _verify_installed_source(Path(sokudan.__file__).resolve().parent, wheel_path)
    environment["dependency_lock_sha256"] = _sha256_file(root / "requirements" / "sokudan-cpu-win-x64.lock")
    environment["backend_request"] = {"backend": "torch", "device": "cpu", "dtype": "float32"}
    environment["torch_version"] = torch.__version__
    environment["torch_cuda_build"] = torch.version.cuda
    environment["source_wheel_sha256"] = lock["package"]["wheel"]["sha256"]
    environment["source_package_python_files_matched"] = source_check["wheel_python_files_matched"]
    environment["artifact_lock_sha256"] = lock_sha256
    return environment, sokudan, lock, verification


def _predict_cases(agent: Any, cases: list[dict[str, Any]], dataset_identity: dict[str, Any],
                   expected_temperature: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_records: list[dict[str, Any]] = []
    valid: dict[tuple[str, str], dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    adapter_cases: dict[str, str] = {}
    for case in cases:
        record: dict[str, Any] = {"case_id": case["case_id"], "status": "PASS",
                                  "source_payload_sha256": _input_fingerprint(case), "questions": []}
        if not isinstance(case.get("state"), str):
            row = {"question_id": None, "question_type": None, "status": "CONTRACT_ERROR",
                   "error_code": "STATE_MUST_BE_STRING"}
            record["status"] = "CONTRACT_ERROR"
            record["questions"].append(row)
            case_records.append(record)
            status_counts[row["status"]] += 1
            continue

        try:
            prepared = _prepare_question_inputs(agent, case["state"], case["questions"])
            too_long = [qid for qid, info in prepared.items()
                        if info["truncated"] or info["required_total_tokens"] > 1024]
            if too_long:
                record["status"] = "OVER_LIMIT"
                record["questions"] = [
                    {"question_id": qid, "question_type": _question_kind(case["questions"][qid]),
                     "status": "OVER_LIMIT",
                     "required_total_tokens": prepared[qid]["required_total_tokens"],
                     "limit_tokens": 1024,
                     "truncation_detected_before_prediction": prepared[qid]["truncated"],
                     "prediction_performed": False}
                    for qid in case["questions"]
                ]
                case_records.append(record)
                status_counts["OVER_LIMIT"] += len(record["questions"])
                continue

            before = _canonical(case["questions"])
            official = agent.predict(case["state"], case["questions"], order_marginalize=False)
            if _canonical(case["questions"]) != before:
                raise SokudanComparisonError("INPUT_MUTATED", "公式API呼出しで質問または候補定義が変更されました。")
            if official.get("usage", {}).get("state_truncated") is not False:
                raise SokudanComparisonError("STATE_TRUNCATION_DETECTED", "公式APIがstateを切り捨てたため評価対象外です。")
            if official.get("encoding") != agent.backend.encoding:
                raise SokudanComparisonError("ENCODING_MISMATCH", "公式APIの実行encodingがモデル設定と一致しません。")
            if set(official.get("answers", {})) != set(case["questions"]):
                raise SokudanComparisonError("ANSWER_QUESTION_SET_MISMATCH", "公式APIの回答IDが入力質問と一致しません。")
            if len(prepared) == 1:
                qid, input_record = next(iter(prepared.items()))
                expected_api_question_tokens = input_record["question_tokens"] + input_record["suffix_tokens"]
                usage = official.get("usage", {})
                if (usage.get("state_tokens") != input_record["state_tokens"]
                        or usage.get("question_tokens") != expected_api_question_tokens):
                    raise SokudanComparisonError("TOKENIZATION_USAGE_MISMATCH", "公式APIのusageと固定入力のトークン数が一致しません。")

            kinds = {qid: _question_kind(question) for qid, question in case["questions"].items()}
            counts = {qid: len(_candidate_ids(question)) for qid, question in case["questions"].items()}
            _validate_calibration(agent, official, list(case["questions"]), kinds, counts,
                                  expected_temperature)
            for qid, question in case["questions"].items():
                kind = kinds[qid]
                candidate_ids = _candidate_ids(question)
                official_answer = official["answers"][qid]
                adapted = adapt_official_answer(kind, official_answer, candidate_ids)
                tokenization = {
                    "required_total_tokens": prepared[qid]["required_total_tokens"],
                    "question_tokens": prepared[qid]["question_tokens"],
                    "suffix_tokens": prepared[qid]["suffix_tokens"],
                    "state_tokens": prepared[qid]["state_tokens"],
                    "candidate_count": prepared[qid]["candidate_count"],
                    "marker_count": prepared[qid]["candidate_count"],
                    "truncation_performed": False,
                    "profile_or_npu_shape_applied": False,
                }
                row = {
                    "question_id": qid,
                    "question_type": kind,
                    "status": "PASS",
                    "input_tensor_sha256": prepared[qid]["input_tensor_sha256"],
                    "tokenization": tokenization,
                    "official_api": official,
                    "adapter": adapted,
                }
                if kind not in adapter_cases:
                    adapter_cases[kind] = case["case_id"]
                record["questions"].append(row)
                valid[(case["case_id"], qid)] = {
                    "case_id": case["case_id"], "question_id": qid,
                    "candidate_ids": candidate_ids,
                    "decision": adapted["decision"] if kind != "noul" else adapted["evaluator_boolean"],
                    "stage_prediction": adapted["stage_prediction"],
                    "continuous_score": adapted["api_returned_value"] if kind == "score" else None,
                    "true_probability": adapted["api_returned_value"] if kind == "noul" else None,
                    "display_probabilities": adapted["api_display_probabilities"],
                }
                status_counts["PASS"] += 1
        except Exception as exc:
            code = getattr(exc, "code", "INFERENCE_EXCEPTION")
            status = ("OVER_LIMIT" if code == "QUESTION_TOO_LONG" else
                      "CONTRACT_ERROR" if isinstance(exc, SokudanComparisonError) else
                      "EXECUTION_FAILED")
            if record["questions"]:
                for row in record["questions"]:
                    if row.get("status") == "PASS":
                        row["status"] = status
                        row["error_code"] = code
                        row["error"] = safe_message(str(exc))
                # A multi-question case is invalid as a whole if any question failed.
                record["questions"] = [row if row.get("status") != "PASS" else
                                       {"question_id": row["question_id"],
                                        "question_type": row["question_type"],
                                        "status": status, "error_code": code,
                                        "error": safe_message(str(exc))}
                                       for row in record["questions"]]
            else:
                record["questions"] = [
                    {"question_id": qid, "question_type": _question_kind(question),
                     "status": status, "error_code": code, "error": safe_message(str(exc))}
                    for qid, question in case["questions"].items()
                ]
            record["status"] = status
            for key in list(valid):
                if key[0] == case["case_id"]:
                    valid.pop(key, None)
            status_counts[status] += len(record["questions"])
        case_records.append(record)
    return case_records, {"outputs": valid, "status_counts": dict(status_counts),
                          "adapter_api_validation_case_ids": adapter_cases}


def _metrics(cases: list[dict[str, Any]], case_records: list[dict[str, Any]],
             laya_rows: list[dict[str, Any]], laya_run_id: str,
             identity: dict[str, Any]) -> dict[str, Any]:
    laya_raw = {(row["case_id"], row["question_id"]): row for row in laya_rows}
    laya_norm: dict[tuple[str, str], dict[str, Any]] = {}
    for key, row in laya_raw.items():
        laya_norm[key] = _laya_prediction(row)
    sok_norm = {}
    statuses: dict[str, Counter[str]] = {kind: Counter() for kind in ("choice", "score", "noul")}
    for case_record in case_records:
        for row in case_record["questions"]:
            kind = row.get("question_type")
            if kind in statuses:
                statuses[kind][row["status"]] += 1
            if row["status"] == "PASS":
                sok_norm[(case_record["case_id"], row["question_id"])] = _normalised_sokudan(row)
    dataset_case_ids = {case["case_id"] for case in cases}
    if any(case_id not in dataset_case_ids for case_id, _qid in laya_norm):
        raise SokudanComparisonError("LAYA_REFERENCE_UNKNOWN_CASE", "保存済みLaya出力にdev-v3外のcase_idがあります。")
    models: dict[str, Any] = {}
    for name, norm in (("sokudan", sok_norm), ("laya_saved_cpu_reference", laya_norm)):
        models[name] = {}
        for kind in ("choice", "score", "noul"):
            model_status = (statuses[kind] if name == "sokudan" else Counter({
                "PASS": sum(1 for row in laya_rows if row["question_type"] == kind)
            }))
            models[name][kind] = _model_metrics(cases, norm, model_status, kind)
    pairwise = _pairwise_metrics(cases, sok_norm, laya_norm)
    requested = len(cases)
    case_type = {
        case["case_id"]: _question_kind(next(iter(case["questions"].values())))
        for case in cases
    }
    valid_denominators = Counter(case_type[case_id] for case_id, _qid in sok_norm)
    coverage = {
        "requested_cases": requested,
        "requested_by_type": {kind: sum(_question_kind(next(iter(case["questions"].values()))) == kind
                                          for case in cases)
                              for kind in ("choice", "score", "noul")},
        "sokudan_output_status_by_type": {kind: dict(statuses[kind]) for kind in statuses},
        "sokudan_valid_output_denominators": {kind: valid_denominators[kind] for kind in statuses},
        "length_overflow": sum(statuses[kind]["OVER_LIMIT"] for kind in statuses),
        "contract_errors": sum(statuses[kind]["CONTRACT_ERROR"] for kind in statuses),
        "execution_failures": sum(statuses[kind]["EXECUTION_FAILED"] for kind in statuses),
        "held": sum(statuses[kind]["HELD"] for kind in statuses),
        "silent_exclusion": False,
        "silent_truncation": False,
    }
    return {
        "dataset": {key: identity[key] for key in (
            "dataset_version", "dataset_sha256", "scoring_contract_version", "scoring_contract_sha256")},
        "human_label_state": "ai_draft_unverified",
        "official_japanese_quality_status": "UNDETERMINED",
        "coverage": coverage,
        "models": models,
        "sokudan_vs_saved_laya": pairwise,
        "case_diagnostics": _case_diagnostics(cases, case_records, laya_rows),
        "laya_reference_run_id": laya_run_id,
        "metrics_are_preliminary_not_model_acceptance": True,
    }


def run_sokudan_cpu(root: Path, laya_reference_run_id: str) -> dict[str, Any]:
    run = Run.create(root, "SOKUDAN_CPU", dataset_version="dev-v3", profile_applicable=False)
    case_records: list[dict[str, Any]] = []
    wrote_cases = False
    try:
        lock, lock_sha256 = _load_lock(root)
        dataset = load_dataset(root, "dev-v3")
        dependency_lock_path = root / "requirements" / "sokudan-cpu-win-x64.lock"
        run.stage("SOKUDAN_IDENTITY", "PASS",
                  dataset_version=dataset["identity"]["dataset_version"],
                  dataset_sha256=dataset["identity"]["dataset_sha256"],
                  scoring_contract_sha256=dataset["identity"]["scoring_contract_sha256"],
                  package_version=lock["package"]["version"],
                  package_source_commit=lock["package"]["source_commit"],
                  package_wheel_sha256=lock["package"]["wheel"]["sha256"],
                  package_source_archive_sha256=lock["package"]["source_archive"]["sha256"],
                  package_sdist_sha256=lock["package"]["source_distribution"]["sha256"],
                  model_repository=lock["model"]["repository"],
                  model_revision=lock["model"]["revision"],
                  model_file_sha256={name: entry["sha256"] for name, entry in lock["model"]["files"].items()},
                  backbone_repository=lock["backbone"]["repository"],
                  backbone_revision=lock["backbone"]["revision"],
                  backbone_file_sha256={name: entry["sha256"] for name, entry in lock["backbone"]["files"].items()},
                  calibration_sha256=lock["execution"]["calibration_sha256"],
                  artifact_lock_sha256=lock_sha256,
                  dependency_lock_sha256=_sha256_file(dependency_lock_path),
                  laya_reference_run_id=laya_reference_run_id)
        reference_run, reference_rows = _validate_reference_run(root, dataset, laya_reference_run_id)
        environment, sokudan, lock, artifact_verification = _run_environment(root)
        runtime_info = artifact_verification["runtime_model"]
        runtime_path = (root / runtime_info["directory"]).resolve()
        config_hash = runtime_info["config_sha256"]
        calibration_hash = runtime_info["calibration_sha256"]

        agent = sokudan.load(runtime_path, backend="torch", device="cpu", dtype="float32")
        import torch

        backend = getattr(agent.backend, "name", None)
        device = str(agent.device)
        dtype = str(agent.dtype)
        parameter_dtypes = sorted({str(parameter.dtype) for parameter in agent.model.parameters()
                                   if parameter.is_floating_point()})
        parameter_count = sum(parameter.numel() for parameter in agent.model.parameters())
        if backend != "torch" or device != "cpu" or dtype != "float32" or parameter_dtypes != ["torch.float32"]:
            raise SokudanComparisonError("ACTUAL_EXECUTION_MISMATCH", "実際のバックエンド・デバイス・精度がtorch/CPU/FP32と一致しません。")

        calibration = json.loads((runtime_path / "calibration.json").read_text(encoding="utf-8"))
        expected_temperature = float(calibration["temperatures"]["bool/2"])
        case_records, prediction_data = _predict_cases(
            agent, dataset["cases"], dataset["identity"], expected_temperature
        )
        run.write_jsonl("sokudan-cases.jsonl", case_records)
        wrote_cases = True
        metrics = _metrics(dataset["cases"], case_records, reference_rows,
                           laya_reference_run_id, dataset["identity"])
        case_diagnostics = metrics.pop("case_diagnostics")
        run.write_jsonl("sokudan-case-diagnostics.jsonl", case_diagnostics)
        run.write_json("sokudan-comparison.json", metrics)

        successful = all(row["status"] == "PASS" for row in case_records)
        stage_status = "PASS" if successful else "FAIL"
        result = {
            "status": "PASS" if successful else "FAIL",
            "run_id": run.run_id,
            "run_terminal_status": "COMPLETE" if successful else "FAILED",
            "code": "SOKUDAN_CPU_COMPLETED" if successful else "SOKUDAN_CPU_CASE_FAILURES",
            "requested_cases": len(dataset["cases"]),
            "valid_outputs": sum(row.get("status") == "PASS" for row in case_records),
            "label_state": "ai_draft_unverified",
            "official_japanese_quality_status": "UNDETERMINED",
        }
        provenance = {
            "package_version": lock["package"]["version"],
            "package_source_commit": lock["package"]["source_commit"],
            "package_wheel_sha256": lock["package"]["wheel"]["sha256"],
            "package_source_archive_sha256": lock["package"]["source_archive"]["sha256"],
            "package_sdist_sha256": lock["package"]["source_distribution"]["sha256"],
            "model_repository": lock["model"]["repository"],
            "model_revision": lock["model"]["revision"],
            "model_file_hashes": {name: record["sha256"] for name, record in lock["model"]["files"].items()},
            "backbone_repository": lock["backbone"]["repository"],
            "backbone_revision": lock["backbone"]["revision"],
            "backbone_file_hashes": {name: record["sha256"] for name, record in lock["backbone"]["files"].items()},
            "runtime_model_config_sha256": config_hash,
            "calibration_sha256": calibration_hash,
            "dependency_lock_sha256": environment["dependency_lock_sha256"],
            "artifact_lock_sha256": environment["artifact_lock_sha256"],
            "dataset_sha256": dataset["identity"]["dataset_sha256"],
            "scoring_contract_sha256": dataset["identity"]["scoring_contract_sha256"],
            "laya_reference_run_id": reference_run.run_id,
            "laya_reference_output_sha256": _sha256_file(reference_run.path / "reference.jsonl"),
        }
        actual = {
            "backend": backend, "device": device, "dtype": dtype,
            "floating_parameter_dtypes": parameter_dtypes,
            "parameter_count": parameter_count,
            "encoding": agent.backend.encoding,
            "input_order": agent.backend.input_order,
            "order_marginalize": False,
            "calibration_default": "calibration.json",
            "calibration_temperatures_loaded": {f"{key[0]}/{key[1]}": value
                                                 for key, value in agent.temperatures.items()},
            "tokenizer_class": type(agent.tokenizer).__name__,
            "tokenizer_is_fast": bool(getattr(agent.tokenizer, "is_fast", False)),
            "tokenizer_source": "固定ModernBERT-Jaローカルスナップショット",
            "model_repo_tokenizer_loaded": False,
            "model_repo_tokenizer_downloaded": True,
            "official_loader_self_check_inference": True,
            "backbone_files_loaded_by_official_loader": ["config.json", "model.safetensors",
                                                           "tokenizer.json", "tokenizer_config.json"],
            "API_probability_precision": {
                "choice": "official returned probabilities rounded to 4 decimals",
                "score": "official returned probabilities rounded to 4 decimals; score is the API's separate continuous value",
                "noul": "official returned P(true) rounded to 4 decimals",
                "unrounded_probability_reconstructed": False,
            },
            "adapter_api_validation_case_ids": prediction_data["adapter_api_validation_case_ids"],
        }
        summary = {**result, "provenance": provenance, "environment": environment,
                   "actual_execution": actual, "artifact_verification": artifact_verification,
                   "metrics": metrics}
        run.write_json("sokudan-run-summary.json", summary)
        run.stage("SOKUDAN_CPU", stage_status, **{
            "summary_file": "sokudan-run-summary.json",
            "cases_file": "sokudan-cases.jsonl",
            "case_diagnostics_file": "sokudan-case-diagnostics.jsonl",
            "comparison_file": "sokudan-comparison.json",
            "requested_cases": len(dataset["cases"]),
            "valid_outputs": result["valid_outputs"],
            "label_state": "ai_draft_unverified",
            "official_japanese_quality_status": "UNDETERMINED",
            "provenance": provenance,
            "actual_execution": actual,
        })
        return result
    except Exception as exc:
        code = getattr(exc, "code", "SOKUDAN_CPU_FAILED")
        if not wrote_cases:
            try:
                run.write_jsonl("sokudan-cases.jsonl", case_records)
                wrote_cases = True
            except (OSError, FileExistsError):
                pass
        try:
            run.failure("SOKUDAN_CPU", code, exc,
                        requested_cases=len(case_records),
                        label_state="ai_draft_unverified",
                        official_japanese_quality_status="UNDETERMINED")
        except Exception:
            pass
        return {"status": "FAIL", "run_id": run.run_id, "run_terminal_status": "FAILED",
                "code": code, "error_type": type(exc).__name__,
                "message": safe_message(str(exc)), "recorded_cases": len(case_records)}

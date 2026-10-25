"""外部入力と架空評価ケースの境界検証。"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


class ContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Profile:
    sequence_length: int = 256
    candidate_slots: int = 8
    max_questions: int = 4

    @property
    def name(self) -> str:
        return f"s{self.sequence_length}-k{self.candidate_slots}"


def profile_from_mapping(value: Any) -> Profile:
    required = {"name", "batch", "sequence_length", "candidate_slots"}
    if not isinstance(value, dict) or set(value) != required:
        raise ContractError("PROFILE_INVALID", "固定プロファイルの項目構成が不正です。")
    sequence_length = value["sequence_length"]
    candidate_slots = value["candidate_slots"]
    if (type(value["batch"]) is not int or value["batch"] != 1 or type(sequence_length) is not int
            or type(candidate_slots) is not int or candidate_slots != 8
            or sequence_length not in (128, 256)
            or value["name"] != f"s{sequence_length}-k8"):
        raise ContractError("PROFILE_INVALID", "s256-k8またはs128-k8以外の形状は使用できません。")
    return Profile(sequence_length=sequence_length, candidate_slots=candidate_slots)


def profile_mapping(profile: Profile) -> dict[str, Any]:
    if profile.name not in {"s256-k8", "s128-k8"} or profile.max_questions != 4:
        raise ContractError("PROFILE_INVALID", "承認済みの固定プロファイルではありません。")
    return {"name": profile.name, "batch": 1,
            "sequence_length": profile.sequence_length, "candidate_slots": profile.candidate_slots}


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_TOP_LEVEL_KEYS = {"schema_version", "request_id", "state", "state_version", "questions", "now", "timezone"}


def _json_text(value: Any, field: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_JSON_VALUE", f"{field} は有限なJSON値である必要があります。") from exc


def render_options(question: dict[str, Any]) -> list[str]:
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        return [str(label) if description in (None, "") else f"{label}: {_json_text(description, 'criteria')}"
                for label, description in criteria.items()]
    if kind == "score":
        return [f"level {index}: {_json_text(value, 'criteria')}" for index, value in enumerate(criteria)]
    return ["false: no, the statement does not hold", "true: yes, the statement holds"]


def _check_id(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ContractError("INVALID_IDENTIFIER", f"{field} は英数字から始まる1〜64文字のIDが必要です。")


def _validate_question(question_id: Any, question: Any, profile: Profile) -> None:
    _check_id(question_id, "question_id")
    if not isinstance(question, dict):
        raise ContractError("INVALID_QUESTION", f"質問 {question_id} はオブジェクトである必要があります。")
    if set(question) - {"type", "instructions", "criteria"}:
        raise ContractError("UNKNOWN_QUESTION_FIELD", f"質問 {question_id} に未定義の項目があります。")
    kind = question.get("type")
    if kind not in {"choice", "score", "bool"}:
        raise ContractError("INVALID_QUESTION_TYPE", f"質問 {question_id} のtypeが無効です。")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ContractError("EMPTY_INSTRUCTIONS", f"質問 {question_id} のinstructionsが空です。")
    if kind == "bool":
        if "criteria" in question:
            raise ContractError("UNEXPECTED_CRITERIA", f"bool質問 {question_id} はcriteriaを持てません。")
        return
    choices = question.get("criteria")
    if kind == "choice":
        if not isinstance(choices, dict) or not 2 <= len(choices) <= profile.candidate_slots:
            raise ContractError("INVALID_CANDIDATES", f"choice質問 {question_id} は2〜{profile.candidate_slots}候補が必要です。")
        for candidate_id, description in choices.items():
            _check_id(candidate_id, f"質問 {question_id} の候補ID")
            _json_text(description, "criteria")
    else:
        if not isinstance(choices, list) or not 2 <= len(choices) <= profile.candidate_slots:
            raise ContractError("INVALID_CANDIDATES", f"score質問 {question_id} は順序付きの2〜{profile.candidate_slots}段階が必要です。")
        for value in choices:
            _json_text(value, "criteria")
    rendered = [unicodedata.normalize("NFC", item).strip().casefold() for item in render_options(question)]
    if any(not value for value in rendered):
        raise ContractError("DUPLICATE_CANDIDATE", f"質問 {question_id} の候補が空または重複しています。")
    if kind == "choice":
        labels = [unicodedata.normalize("NFC", str(value)).strip().casefold() for value in choices.values()]
    else:
        labels = [unicodedata.normalize("NFC", _json_text(value, "criteria")).strip().casefold() for value in choices]
    if len(labels) != len(set(labels)):
        raise ContractError("DUPLICATE_CANDIDATE", f"質問 {question_id} の候補が空または重複しています。")


def validate_request(value: Any, profile: Profile = Profile()) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("INVALID_REQUEST", "入力はJSONオブジェクトである必要があります。")
    if set(value) - _TOP_LEVEL_KEYS:
        raise ContractError("UNKNOWN_REQUEST_FIELD", "入力に未定義の項目があります。")
    if value.get("schema_version") != 1:
        raise ContractError("UNSUPPORTED_SCHEMA", "schema_versionは1である必要があります。")
    _check_id(value.get("request_id"), "request_id")
    if isinstance(value.get("state_version"), bool) or not isinstance(value.get("state_version"), int) or value["state_version"] < 0:
        raise ContractError("INVALID_STATE_VERSION", "state_versionは0以上の整数が必要です。")
    state = value.get("state")
    if not isinstance(state, (str, dict, list)) or (isinstance(state, str) and not state.strip()):
        raise ContractError("EMPTY_STATE", "stateは空でない文字列、配列、またはオブジェクトが必要です。")
    _json_text(state, "state")
    questions = value.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= profile.max_questions:
        raise ContractError("INVALID_QUESTIONS", f"questionsは1〜{profile.max_questions}件のオブジェクトが必要です。")
    for question_id, question in questions.items():
        _validate_question(question_id, question, profile)
    if "now" in value and not isinstance(value["now"], str):
        raise ContractError("INVALID_NOW", "nowは文字列である必要があります。")
    if "timezone" in value and not isinstance(value["timezone"], str):
        raise ContractError("INVALID_TIMEZONE", "timezoneは文字列である必要があります。")
    return value


def validate_dataset_case(case: Any, profile: Profile = Profile(), dataset_version: str = "dev-v1") -> None:
    if not isinstance(case, dict) or set(case) != {"case_id", "state", "questions", "expected"}:
        raise ContractError("INVALID_CASE", "評価ケースの項目構成が不正です。")
    _check_id(case["case_id"], "case_id")
    request = {
        "schema_version": 1,
        "request_id": case["case_id"],
        "state": case["state"],
        "state_version": 0,
        "questions": case["questions"],
    }
    validate_request(request, profile)
    expected = case["expected"]
    if not isinstance(expected, dict) or set(expected) != set(case["questions"]):
        raise ContractError("INVALID_EXPECTED", f"ケース {case['case_id']} の期待判断が質問と一致しません。")
    for question_id, label in expected.items():
        required = {"allowed_answers", "rationale", "verification_status"}
        if dataset_version == "dev-v1":
            required.add("target_errors")
        elif dataset_version not in {"dev-v2", "dev-v3"}:
            raise ContractError("DATASET_VERSION_UNKNOWN", "未対応の評価データ版です。")
        if not isinstance(label, dict) or set(label) != required:
            raise ContractError("INVALID_LABEL", f"ケース {case['case_id']} / {question_id} のラベル項目が不正です。")
        if not isinstance(label["allowed_answers"], list) or not label["allowed_answers"]:
            raise ContractError("EMPTY_LABEL", f"ケース {case['case_id']} / {question_id} の許容解が空です。")
        question = case["questions"][question_id]
        kind = question["type"]
        if kind == "choice":
            valid_answers = set(question["criteria"])
        elif kind == "score":
            valid_answers = set(range(len(question["criteria"])))
        else:
            valid_answers = {False, True}
        def answer_is_valid(answer: Any) -> bool:
            if kind == "choice":
                return isinstance(answer, str) and answer in valid_answers
            if kind == "score":
                return isinstance(answer, int) and not isinstance(answer, bool) and answer in valid_answers
            return isinstance(answer, bool)

        if (len(label["allowed_answers"]) != len(set(map(repr, label["allowed_answers"])))
                or any(not answer_is_valid(answer) for answer in label["allowed_answers"])):
            raise ContractError("LABEL_ANSWER_OUT_OF_RANGE", f"ケース {case['case_id']} / {question_id} の許容解が選択肢にありません。")
        if not isinstance(label["rationale"], str) or not label["rationale"].strip():
            raise ContractError("EMPTY_RATIONALE", f"ケース {case['case_id']} / {question_id} の理由が空です。")
        if dataset_version == "dev-v1" and (
            not isinstance(label["target_errors"], list) or not label["target_errors"]
        ):
            raise ContractError("EMPTY_TARGET_ERRORS", f"ケース {case['case_id']} / {question_id} の対象誤りが空です。")
        if label["verification_status"] != "ai_draft_unverified":
            raise ContractError("INVALID_VERIFICATION_STATE", "初期データはai_draft_unverifiedでなければなりません。")


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    if not values or not math.isfinite(temperature) or temperature <= 0:
        raise ContractError("INVALID_LOGITS", "logitsとtemperatureが不正です。")
    scaled = [float(value) / temperature for value in values]
    if any(not math.isfinite(value) for value in scaled):
        raise ContractError("INVALID_LOGITS", "logitsにNaNまたはInfがあります。")
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ContractError("INVALID_LOGITS", "有効な確率分布を作れません。")
    return [value / total for value in weights]

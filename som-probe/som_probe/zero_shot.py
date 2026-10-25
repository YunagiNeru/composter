"""A small CPU adapter for the pinned multilingual NLI classifier."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


MODEL_REVISION = "3fe1f0c48b589cf6220fd21a628a1dd56d3d41f8"
MODEL_REPOSITORY = "Horizon-Labs/multilingual-zeroshot-small"
MODEL_DIRECTORY = f"multilingual-zeroshot-small-{MODEL_REVISION}"
MAX_PAIR_TOKENS = 8192
TRAINED_LENGTH_TOKENS = 1024
LANGUAGE_CONDITIONS = ("english", "japanese")


class ZeroShotError(ValueError):
    """Stable, user-safe error for the candidate-A evaluation."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        model_lock = json.loads((root / "configs/zero-shot-models.lock.json").read_text(encoding="utf-8"))
        evaluation = json.loads((root / "configs/zero-shot-evaluation-v1.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZeroShotError("CANDIDATE_CONFIGURATION_INVALID") from exc
    if evaluation.get("evaluation_version") != "zero-shot-evaluation-v1":
        raise ZeroShotError("EVALUATION_VERSION_MISMATCH")
    if evaluation.get("official_primary_condition") != "english":
        raise ZeroShotError("OFFICIAL_CONDITION_NOT_ENGLISH")
    if evaluation.get("input_policy", {}).get("truncation") != "forbidden":
        raise ZeroShotError("TRUNCATION_POLICY_INVALID")
    return model_lock, evaluation


def verify_model_files(root: Path, model_lock: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    spec = model_lock.get("candidate_a", {})
    if (spec.get("status") != "PINNED_FOR_CPU_EVALUATION"
            or spec.get("repository") != MODEL_REPOSITORY
            or spec.get("revision") != MODEL_REVISION):
        raise ZeroShotError("MODEL_REVISION_LOCK_MISMATCH")
    model_dir = root / ".local" / "models" / MODEL_DIRECTORY
    locked_files = spec.get("files")
    if not isinstance(locked_files, dict) or not locked_files:
        raise ZeroShotError("MODEL_FILE_LOCK_EMPTY")
    actual = {}
    for name, metadata in locked_files.items():
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != metadata.get("size"):
            raise ZeroShotError("MODEL_FILE_MISSING_OR_SIZE_MISMATCH")
        actual[name] = sha256_file(path)
        if actual[name] != metadata.get("sha256"):
            raise ZeroShotError("MODEL_FILE_HASH_MISMATCH")
    return model_dir, actual


def build_hypotheses(state: str, labels: list[dict[str, str]], template: str) -> list[str]:
    if not isinstance(state, str) or not state.strip() or "{}" not in template:
        raise ZeroShotError("INPUT_OR_TEMPLATE_INVALID")
    if len(labels) < 2 or len({item.get("id") for item in labels}) != len(labels):
        raise ZeroShotError("CANDIDATE_LABELS_INVALID")
    hypotheses = [template.format(item["label"]) for item in labels]
    if any(not text.strip() for text in hypotheses):
        raise ZeroShotError("HYPOTHESIS_EMPTY")
    return hypotheses


def official_scores_from_logits(raw_logits: list[list[float]], entailment_id: int = 1) -> dict[str, Any]:
    """Match the official single-label pipeline's entailment-score normalization."""
    if not raw_logits or any(len(row) != 2 for row in raw_logits):
        raise ZeroShotError("LOGIT_SHAPE_INVALID")
    if entailment_id not in (0, 1):
        raise ZeroShotError("ENTAILMENT_LABEL_ID_INVALID")
    for row in raw_logits:
        if any(not math.isfinite(float(value)) for value in row):
            raise ZeroShotError("NONFINITE_LOGIT")
    entailment_logits = [float(row[entailment_id]) for row in raw_logits]
    maximum = max(entailment_logits)
    exponents = [math.exp(value - maximum) for value in entailment_logits]
    denominator = sum(exponents)
    scores = [value / denominator for value in exponents]
    probabilities = []
    for row in raw_logits:
        pair_max = max(float(row[0]), float(row[1]))
        pair_exp = [math.exp(float(value) - pair_max) for value in row]
        pair_total = sum(pair_exp)
        probabilities.append([value / pair_total for value in pair_exp])
    top = max(scores)
    top_indices = [index for index, score in enumerate(scores) if math.isclose(score, top, rel_tol=0.0, abs_tol=1e-12)]
    return {
        "entailment_logits": entailment_logits,
        "official_scores": scores,
        "pair_probabilities": probabilities,
        "official_top_index": top_indices[0],
        "official_top_tie": len(top_indices) > 1,
    }


def common_prediction(labels: list[dict[str, str]], result: dict[str, Any],
                      raw_logits: list[list[float]], entailment_id: int = 1) -> dict[str, Any]:
    index = result["official_top_index"]
    if result["official_top_tie"]:
        return {"decision": None, "status": "HOLD", "reason": "official_score_tie"}
    non_entailment_id = 1 - entailment_id
    if float(raw_logits[index][entailment_id]) <= float(raw_logits[index][non_entailment_id]):
        return {"decision": None, "status": "HOLD", "reason": "top_candidate_not_entailed"}
    return {"decision": labels[index]["id"], "status": "DECIDED", "reason": None}


def classify(session: Any, state: str, dimension: str, condition: dict[str, Any],
             max_pair_tokens: int = MAX_PAIR_TOKENS) -> dict[str, Any]:
    labels = condition.get("dimensions", {}).get(dimension)
    if not isinstance(labels, list):
        raise ZeroShotError("FIXED_DIMENSION_MISSING")
    hypotheses = build_hypotheses(state, labels, condition.get("template", ""))
    tokenized = session.tokenizer(
        [state] * len(hypotheses), hypotheses,
        padding=True, truncation=False, return_tensors="pt",
    )
    pair_lengths = [int(mask.sum().item()) for mask in tokenized["attention_mask"]]
    longest = max(pair_lengths)
    record = {
        "dimension": dimension,
        "candidate_ids": [item["id"] for item in labels],
        "candidate_labels": [item["label"] for item in labels],
        "hypotheses": hypotheses,
        "template": condition["template"],
        "pair_token_lengths": pair_lengths,
        "maximum_pair_tokens": longest,
        "truncation_performed": False,
        "outside_reported_finetuning_length": longest > TRAINED_LENGTH_TOKENS,
        "input_sha256": json_sha256({"premise": state, "hypotheses": hypotheses}),
    }
    if longest > max_pair_tokens:
        return {**record, "status": "LENGTH_EXCEEDED", "raw_logits": None}
    raw_logits = session.forward(tokenized)
    result = official_scores_from_logits(raw_logits, session.entailment_id)
    selected = common_prediction(labels, result, raw_logits, session.entailment_id)
    official_index = result["official_top_index"]
    return {
        **record,
        "status": "PASS",
        "raw_logits": raw_logits,
        "entailment_logits": result["entailment_logits"],
        "pair_probabilities_not_calibrated": result["pair_probabilities"],
        "official_scores_not_calibrated": result["official_scores"],
        "official_prediction_id": labels[official_index]["id"],
        "official_prediction_label": labels[official_index]["label"],
        "official_top_tie": result["official_top_tie"],
        "common_prediction": selected,
    }


def reduce_score(date_result: dict[str, Any], clock_result: dict[str, Any]) -> dict[str, Any]:
    date_yes = _binary_value(date_result)
    clock_yes = _binary_value(clock_result)
    date_scores = _yes_no_scores(date_result)
    clock_scores = _yes_no_scores(clock_result)
    expectation = None
    if date_scores is not None and clock_scores is not None:
        expectation = date_scores["yes"] + clock_scores["yes"]
    stage = None
    if date_yes is not None and clock_yes is not None:
        stage = int(date_yes) + int(clock_yes)
    reason = None
    if stage is None:
        reason = "date_or_clock_component_held"
    return {
        "stage_prediction": stage,
        "stage_status": "DECIDED" if stage is not None else "HOLD",
        "stage_hold_reason": reason,
        "score_expected_value_from_official_scores": expectation,
        "date_info_prediction": date_yes,
        "clock_time_prediction": clock_yes,
        "date_info_scores_not_calibrated": date_scores,
        "clock_time_scores_not_calibrated": clock_scores,
    }


def _binary_value(output: dict[str, Any]) -> bool | None:
    if output.get("status") != "PASS":
        return None
    value = output.get("common_prediction", {}).get("decision")
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def _yes_no_scores(output: dict[str, Any]) -> dict[str, float] | None:
    if output.get("status") != "PASS":
        return None
    ids = output.get("candidate_ids", [])
    scores = output.get("official_scores_not_calibrated", [])
    if len(ids) != 2 or len(scores) != 2 or set(ids) != {"yes", "no"}:
        return None
    return {candidate_id: float(scores[ids.index(candidate_id)]) for candidate_id in ("yes", "no")}

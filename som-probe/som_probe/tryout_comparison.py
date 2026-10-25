"""Fixed comparison of current dev-v3 questions and decomposed questions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from .diagnostics import (
    DiagnosticError,
    _laya_question,
    _official_laya_trace,
    _official_sokudan_candidate_ids,
    _sokudan_official_trace,
)
from .storage import Run, open_run


FIXED_CASES = (
    {"case_id": "work-none-eaten", "dimension": "choice", "state": "スコーン食べた",
     "expected": "none", "rationale": "単なる完了済みの出来事で、本人に残る行動はない。"},
    {"case_id": "work-none-delivery", "dimension": "choice", "state": "明日ヤマトから荷物届く",
     "expected": "none", "rationale": "荷物を届けるのは第三者の行動である。"},
    {"case_id": "work-none-cancelled", "dimension": "choice", "state": "勤怠の提出は取り消した。",
     "expected": "none", "rationale": "対象行動が取消済みである。"},
    {"case_id": "work-none-completed", "dimension": "choice", "state": "資料を提出し終えた。",
     "expected": "none", "rationale": "本人の行動は完了している。"},
    {"case_id": "work-task-v3", "dimension": "choice", "state": "今週中に勤怠まとめて提出",
     "expected": "task", "rationale": "本人が行う未完了行動と期限が明記されている。"},
    {"case_id": "work-task-report", "dimension": "choice", "state": "明日までに報告書を提出する。",
     "expected": "task", "rationale": "本人の未完了行動と実行意思が明記されている。"},
    {"case_id": "work-need-request", "dimension": "choice", "state": "資料の提出をお願いされた。",
     "expected": "need_info", "rationale": "依頼はあるが、本人が引き受けたかは書かれていない。"},
    {"case_id": "work-need-conditional", "dimension": "choice", "state": "必要なら報告書の提出に対応する。",
     "expected": "need_info", "rationale": "対応は条件付きで、実行意思・必要性が確定していない。"},
    {"case_id": "score-none-minutes", "dimension": "score", "state": "会議の議事録を確認する。",
     "expected": 0, "rationale": "日程情報も時計時刻も記載されていない。"},
    {"case_id": "score-none-material", "dimension": "score", "state": "資料を読み返す。",
     "expected": 0, "rationale": "日程情報も時計時刻も記載されていない。"},
    {"case_id": "score-date-tomorrow", "dimension": "score", "state": "明日、荷物が届く。",
     "expected": 1, "rationale": "相対日程はあるが時計時刻はない。"},
    {"case_id": "score-date-next-week", "dimension": "score", "state": "来週、商品を受け取る。",
     "expected": 1, "rationale": "相対日程はあるが時計時刻はない。"},
    {"case_id": "score-time-15", "dimension": "score", "state": "15時に会議を始める。",
     "expected": 1, "rationale": "時計時刻はあるが日程情報はない。"},
    {"case_id": "score-time-afternoon", "dimension": "score", "state": "午後3時に荷物を受け取る。",
     "expected": 1, "rationale": "時計時刻はあるが日程情報はない。"},
    {"case_id": "score-both-tomorrow", "dimension": "score", "state": "明日15時に荷物が届く。",
     "expected": 2, "rationale": "相対日程と時計時刻の両方がある。"},
    {"case_id": "score-both-weekday", "dimension": "score", "state": "来週火曜日の午後3時に会議がある。",
     "expected": 2, "rationale": "曜日を含む相対日程と時計時刻の両方がある。"},
    {"case_id": "notify-yes-direct", "dimension": "noul", "state": "明日作業する。通知してほしい。",
     "expected": True, "rationale": "通知希望が明記されている。"},
    {"case_id": "notify-no-direct", "dimension": "noul", "state": "明日作業する。通知はいらない。",
     "expected": False, "rationale": "通知不要が明記されている。"},
    {"case_id": "notify-yes-paraphrase", "dimension": "noul", "state": "明日作業を行う。終わったら知らせてください。",
     "expected": True, "rationale": "完了後に知らせる希望が明記されている。"},
    {"case_id": "notify-no-paraphrase", "dimension": "noul", "state": "明日作業を行う。完了通知は不要です。",
     "expected": False, "rationale": "完了通知が不要と明記されている。"},
)

SIMPLE_QUESTION_VARIANTS = {
    "work_pending": {
        "base": "本文には、本人が今後行う必要のある未完了の行動候補がありますか。",
        "paraphrase": "本人に残っている、これから対応する行動が本文に含まれていますか。",
    },
    "work_commitment": {
        "base": "本文から、本人の実行意思・約束・必要性を明確に判断できますか。",
        "paraphrase": "本人がその行動を実行する意思または必要性があると本文から分かりますか。",
    },
    "date_info": {
        "base": "本文に、日付・曜日・相対日程のいずれかが書かれていますか。",
        "paraphrase": "本文に、日や期間を示す日程情報がありますか。",
    },
    "clock_time": {
        "base": "本文に具体的な時計の時刻が書かれていますか。",
        "paraphrase": "本文に何時かを示す時刻が明記されていますか。",
    },
    "notification": {
        "base": "本文の書き手は通知を希望していますか。",
        "paraphrase": "本文の書き手は通知を求めていますか。",
    },
}

_QUESTION_IDS = {
    "work_pending": "split_work_pending",
    "work_commitment": "split_work_commitment",
    "date_info": "split_date_info",
    "clock_time": "split_clock_time",
    "notification": "split_notification",
}


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def choice_from_components(pending: bool, commitment_clear: bool) -> str:
    if type(pending) is not bool or type(commitment_clear) is not bool:
        raise ValueError("Choice分解質問は真偽値でなければなりません。")
    if not pending:
        return "none"
    return "task" if commitment_clear else "need_info"


def score_from_components(date_info: bool, clock_time: bool) -> int:
    if type(date_info) is not bool or type(clock_time) is not bool:
        raise ValueError("Score分解質問は真偽値でなければなりません。")
    return 2 if date_info and clock_time else int(date_info or clock_time)


def _simple_questions(model: str, variant: str) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for key, variants in SIMPLE_QUESTION_VARIANTS.items():
        text = variants[variant]
        question_id = _QUESTION_IDS[key]
        if model == "laya":
            question = {"type": "noul", "instructions": text,
                        "criteria": {"false": "いいえ", "true": "はい"},
                        "labels": {"false": "いいえ", "true": "はい"}}
        else:
            question = {"type": "bool", "instructions": text,
                        "false_label": "いいえ", "true_label": "はい"}
        questions[question_id] = question
    return questions


def _trace_summary(model: str, trace: dict[str, Any], question: dict[str, Any], state: str) -> dict[str, Any]:
    candidate_ids = trace.get("candidate_ids")
    state_ids = trace.get("state_token_ids") or []
    if model == "laya":
        tensors = trace.get("input_tensors")
        input_hash = _sha256(tensors) if isinstance(tensors, dict) else None
        token_count = len(state_ids)
        input_token_count = trace.get("required_total_tokens")
        truncated = trace.get("state_truncated")
        question_hash = trace.get("question_sha256") or _sha256(question)
        extras = {key: trace.get(key) for key in (
            "qtype", "temperature_bucket", "effective_temperature", "language_parameter",
        )}
    else:
        input_hash = trace.get("input_tensor_sha256")
        token_count = len(state_ids)
        input_token_count = trace.get("required_total_tokens")
        truncated = trace.get("truncated")
        question_hash = _sha256(question)
        extras = {key: trace.get(key) for key in ("ordered", "input_order")}
    return {
        "state_sha256": hashlib.sha256(state.encode("utf-8")).hexdigest(),
        "question_sha256": question_hash,
        "candidate_ids": candidate_ids,
        "state_token_count": token_count,
        "input_token_count": input_token_count,
        "state_token_ids_sha256": _sha256(state_ids),
        "input_tensors_sha256": input_hash,
        "truncated": truncated,
        **extras,
    }


def _normalise(model: str, kind: str, answer: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    from .diagnostics import _candidate_ids, _normalise_laya_answer, _normalise_sokudan_official_answer

    candidate_ids = _candidate_ids(kind, question) if model == "laya" else _official_sokudan_candidate_ids(question)
    if model == "laya":
        normalized = _normalise_laya_answer(kind, answer, candidate_ids)
    else:
        normalized = _normalise_sokudan_official_answer(kind, answer, candidate_ids)
    if kind in {"choice", "score"}:
        values = normalized.get("probabilities")
        if (not isinstance(values, list) or len(values) != len(candidate_ids)
                or any(type(value) not in (int, float) or not math.isfinite(float(value))
                       or not 0.0 <= float(value) <= 1.0 for value in values)
                or not math.isclose(sum(values), 1.0, rel_tol=0.0,
                                    abs_tol=len(candidate_ids) * 0.0000501)):
            raise DiagnosticError("COMPARISON_CANDIDATE_PROBABILITIES_INVALID")
        if kind == "choice" and normalized.get("decision") not in candidate_ids:
            raise DiagnosticError("COMPARISON_CHOICE_OUTSIDE_CANDIDATES")
        if kind == "score":
            stage = normalized.get("stage_prediction")
            returned_score = normalized.get("returned_score")
            if (type(stage) is not int or not 0 <= stage < len(candidate_ids)
                    or type(returned_score) not in (int, float)
                    or not math.isfinite(float(returned_score))
                    or not 0.0 <= float(returned_score) <= len(candidate_ids) - 1):
                raise DiagnosticError("COMPARISON_SCORE_OUTPUT_INVALID")
    elif (type(normalized.get("boolean_result")) is not bool
          or type(normalized.get("returned_noul", normalized.get("api_returned_value")))
          not in (int, float)
          or not math.isfinite(float(normalized.get("returned_noul", normalized.get("api_returned_value"))))
          or not 0.0 <= float(normalized.get("returned_noul", normalized.get("api_returned_value"))) <= 1.0):
        raise DiagnosticError("COMPARISON_NOUL_OUTPUT_INVALID")
    return normalized


def _run_laya_api(agent: Any, questions: dict[str, dict[str, Any]], state: str,
                  kinds: dict[str, str]) -> dict[str, Any]:
    import torch

    api_questions = {question_id: _laya_question(question)
                     for question_id, question in questions.items()}
    before = _sha256(api_questions)
    with torch.inference_mode():
        output = agent.system_one(state, api_questions, lang=None, max_len=256, head_max_len=192)
    if _sha256(api_questions) != before:
        raise DiagnosticError("LAYA_QUESTION_MUTATED")
    if output.get("usage", {}).get("truncated") is not False:
        raise DiagnosticError("LAYA_COMPARISON_INPUT_TRUNCATED")
    answers = output.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise DiagnosticError("LAYA_COMPARISON_ANSWER_KEYS_INVALID")
    rows = {}
    for question_id, question in questions.items():
        kind = kinds[question_id]
        trace = _official_laya_trace(agent, question_id, question, state)
        raw = answers[question_id]
        rows[question_id] = {
            "question": question,
            "answer": raw,
            "normalized": _normalise("laya", kind, raw, question),
            "input": _trace_summary("laya", trace, question, state),
        }
    return rows


def _run_sokudan_api(session: Any, questions: dict[str, dict[str, Any]], state: str,
                     kinds: dict[str, str]) -> dict[str, Any]:
    from .diagnostics import _sokudan_official_trace
    from .sokudan_cpu import _validate_calibration

    before = _sha256(questions)
    output = session._agent.predict(state, copy.deepcopy(questions), order_marginalize=False)
    if _sha256(questions) != before or output.get("usage", {}).get("state_truncated") is not False:
        raise DiagnosticError("SOKUDAN_COMPARISON_INPUT_CONTRACT_FAILED")
    if output.get("encoding") != session._agent.backend.encoding:
        raise DiagnosticError("SOKUDAN_COMPARISON_ENCODING_MISMATCH")
    answers = output.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise DiagnosticError("SOKUDAN_COMPARISON_ANSWER_KEYS_INVALID")
    counts = {question_id: len(_official_sokudan_candidate_ids(question))
              for question_id, question in questions.items()}
    _validate_calibration(session._agent, output, list(questions), kinds, counts,
                          session._expected_temperature)
    traces = _sokudan_official_trace(session._agent, state, questions)
    return {
        question_id: {
            "question": question,
            "answer": answers[question_id],
            "normalized": _normalise("sokudan", kinds[question_id], answers[question_id], question),
            "input": _trace_summary("sokudan", traces[question_id], question, state),
            "calibration": {
                "calibrated": output.get("calibrated"),
                "calibrated_answers": output.get("calibrated_answers"),
                "temperature": session._expected_temperature if kinds[question_id] == "noul" else None,
            },
        }
        for question_id, question in questions.items()
    }


def _result_decision(kind: str, normalized: dict[str, Any]) -> Any:
    if kind == "choice":
        return normalized.get("decision")
    if kind == "score":
        return normalized.get("stage_prediction")
    return normalized.get("boolean_result")


def _case_result(fixture: dict[str, Any], current: dict[str, Any],
                 split_by_variant: dict[str, dict[str, Any]]) -> dict[str, Any]:
    kind = fixture["dimension"]
    expected = fixture["expected"]
    current_output = current["current"][kind]
    current_prediction = _result_decision(kind, current_output["normalized"])
    split: dict[str, Any] = {}
    for variant, values in split_by_variant.items():
        simple = values["simple"]
        if kind == "choice":
            pending = simple["work_pending"]["normalized"]["boolean_result"]
            commitment = simple["work_commitment"]["normalized"]["boolean_result"]
            prediction = choice_from_components(pending, commitment)
            components = {"pending_action": pending, "commitment_clear": commitment}
        elif kind == "score":
            date_info = simple["date_info"]["normalized"]["boolean_result"]
            clock_time = simple["clock_time"]["normalized"]["boolean_result"]
            prediction = score_from_components(date_info, clock_time)
            components = {"date_info": date_info, "clock_time": clock_time}
        else:
            prediction = simple["notification"]["normalized"]["boolean_result"]
            components = {"notification_requested": prediction}
        split[variant] = {
            "prediction": prediction,
            "expected": expected,
            "mismatch": prediction != expected,
            "components": components,
            "question_results": simple,
        }
    return {
        "case_id": fixture["case_id"], "dimension": kind, "state": fixture["state"],
        "expected": expected, "rationale": fixture["rationale"],
        "label_status": "AI_FIXED_TECHNICAL_EXPECTATION_NOT_HUMAN_REVIEWED",
        "current": {
            "prediction": current_prediction, "mismatch": current_prediction != expected,
            "normalized": current_output["normalized"], "question_result": current_output,
        },
        "decomposed": split,
    }


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for kind in ("choice", "score", "noul"):
        rows = [row for row in cases if row["dimension"] == kind]
        variants = ("base", "paraphrase")
        current_errors = sum(row["current"]["mismatch"] for row in rows)
        variant_summary = {}
        improvements = {}
        regressions = {}
        agreements = {}
        for variant in variants:
            errors = sum(row["decomposed"][variant]["mismatch"] for row in rows)
            improved = sum(row["current"]["mismatch"] and not row["decomposed"][variant]["mismatch"]
                           for row in rows)
            regressed = sum(not row["current"]["mismatch"] and row["decomposed"][variant]["mismatch"]
                            for row in rows)
            agrees = sum(row["current"]["prediction"] == row["decomposed"][variant]["prediction"]
                         for row in rows)
            variant_summary[variant] = {"denominator": len(rows), "mismatches": errors,
                                        "matches": len(rows) - errors}
            improvements[variant] = improved
            regressions[variant] = regressed
            agreements[variant] = agrees
        output[kind] = {
            "case_count": len(rows), "current": {"denominator": len(rows),
                                                   "mismatches": current_errors,
                                                   "matches": len(rows) - current_errors},
            "decomposed": variant_summary,
            "paired_improvements": improvements,
            "paired_regressions": regressions,
            "current_decomposed_decision_agreements": agreements,
            "labels": "AI固定技術期待への参考比較。人手確認前の品質判定ではない。",
        }
    return output


def _source_metadata(root: Path, model: str, loaded: dict[str, Any]) -> dict[str, Any]:
    from .assets import load_lock, verify_assets
    from .assets import sha256_file

    lock = load_lock(root)
    common = {"verified_artifact_sha256": verify_assets(root),
              "models_lock_sha256": sha256_file(root / "configs" / "models.lock.json"),
              "dev_v3_sha256": sha256_file(root / "datasets" / "dev-v3.jsonl"),
              "execution_backend": "CPU", "floating_precision": "FP32"}
    if model == "laya":
        return {
            **common,
            "repository": lock["sources"]["reference_checkpoint"]["repository"],
            "model_revision": lock["sources"]["reference_checkpoint"]["revision"],
            "model_source_revision": lock["sources"]["laya_code"]["commit"],
            "actual_device": loaded["actual_device"],
            "parameter_dtypes": loaded["parameter_dtypes"],
            "openvino_version": loaded["openvino_version"],
        }
    session = loaded["session"]
    model_lock = session._lock
    return {
        **common,
        "repository": model_lock["model"]["repository"],
        "model_revision": model_lock["model"]["revision"],
        "package_version": model_lock["package"]["version"],
        "package_source_revision": model_lock["package"]["source_commit"],
        "actual_backend": session._agent.backend.name,
        "actual_device": str(session._agent.device),
        "actual_dtype": str(session._agent.dtype),
        "calibration_sha256": model_lock["execution"]["calibration_sha256"],
    }


def _run_model(root: Path, run: Run, model: str) -> dict[str, Any]:
    from .tryout import _ExclusiveModelLock, _SokudanSession, _question_templates
    from .diagnostics import _read_dev_v3_questions

    templates, canaries = _question_templates(root)
    question_manifest = _read_dev_v3_questions(root)
    if any(canaries[kind]["question_id"] != question_manifest[kind]["question_id"]
           for kind in ("choice", "score", "noul")):
        raise DiagnosticError("CURRENT_TEMPLATE_ID_MISMATCH")
    current_questions = {
        canaries[kind]["question_id"]: copy.deepcopy(templates[kind])
        for kind in ("choice", "score", "noul")
    }
    current_kinds = {
        canaries[kind]["question_id"]: kind for kind in ("choice", "score", "noul")
    }
    case_results: list[dict[str, Any]] = []
    model_meta: dict[str, Any] = {}
    session = None
    agent = None
    with _ExclusiveModelLock():
        if model == "laya":
            from .diagnostics import _load_official_laya

            agent, _model_lock, parameter_dtypes = _load_official_laya(root, run, contrast=True)
            if str(agent.device).lower() != "cpu" or parameter_dtypes != ["torch.float32"]:
                raise DiagnosticError("LAYA_COMPARISON_DEVICE_OR_PRECISION_MISMATCH")
            try:
                loaded = {"actual_device": str(agent.device).lower(),
                          "parameter_dtypes": parameter_dtypes,
                          "openvino_version": None}
                import openvino as ov
                loaded["openvino_version"] = ov.__version__
                model_meta = _source_metadata(root, model, loaded)
                for fixture in FIXED_CASES:
                    state = fixture["state"]
                    current_rows = _run_laya_api(agent, current_questions, state, current_kinds)
                    current = {"current": {
                        kind: current_rows[canaries[kind]["question_id"]]
                        for kind in ("choice", "score", "noul")}}
                    split_variants = {}
                    for variant in ("base", "paraphrase"):
                        simple_questions = _simple_questions("laya", variant)
                        kinds = {question_id: "noul" for question_id in simple_questions}
                        rows = _run_laya_api(agent, simple_questions, state, kinds)
                        split_variants[variant] = {
                            "simple": {
                                key: rows[_QUESTION_IDS[key]]
                                for key in SIMPLE_QUESTION_VARIANTS
                            }}
                    result = _case_result(fixture, current, split_variants)
                    result["current"]["input_contract"] = {
                        key: current["current"][fixture["dimension"]]["input"].get(key)
                        for key in ("state_sha256", "truncated", "input_tensors_sha256")}
                    case_results.append(result)
            finally:
                agent.__exit__(None, None, None)
                agent = None
        else:
            session = _SokudanSession(root, templates, canaries)
            try:
                actual = (session._agent.backend.name, str(session._agent.device), str(session._agent.dtype))
                if actual != ("torch", "cpu", "float32"):
                    raise DiagnosticError("SOKUDAN_COMPARISON_DEVICE_OR_PRECISION_MISMATCH")
                model_meta = _source_metadata(root, model, {"session": session})
                for fixture in FIXED_CASES:
                    state = fixture["state"]
                    current_rows = _run_sokudan_api(session, current_questions, state, current_kinds)
                    current = {"current": {
                        kind: current_rows[canaries[kind]["question_id"]]
                        for kind in ("choice", "score", "noul")}}
                    split_variants = {}
                    for variant in ("base", "paraphrase"):
                        simple_questions = _simple_questions("sokudan", variant)
                        kinds = {question_id: "noul" for question_id in simple_questions}
                        rows = _run_sokudan_api(session, simple_questions, state, kinds)
                        split_variants[variant] = {
                            "simple": {
                                key: rows[_QUESTION_IDS[key]]
                                for key in SIMPLE_QUESTION_VARIANTS
                            }}
                    result = _case_result(fixture, current, split_variants)
                    result["current"]["input_contract"] = {
                        key: current["current"][fixture["dimension"]]["input"].get(key)
                        for key in ("state_sha256", "truncated", "input_tensors_sha256")}
                    case_results.append(result)
            finally:
                session.close()
                session = None

    return {
        "model": model, "execution": model_meta,
        "question_variants": SIMPLE_QUESTION_VARIANTS,
        "question_variant_sha256": _sha256(SIMPLE_QUESTION_VARIANTS),
        "cases": case_results, "summary": _summary(case_results),
    }


def _child(root: Path, run: Run, model: str, python: Path) -> None:
    if not python.is_file():
        raise DiagnosticError(f"COMPARISON_ENVIRONMENT_MISSING_{model.upper()}")
    command = [str(python), "-B", "-m", "som_probe.tryout_comparison",
               "--stage", model, "--run-id", run.run_id]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=3600)
    run.write_json(f"{model}-process.json", {"model": model,
                                             "command": [Path(command[0]).name, *command[1:]],
                                             "exit_code": result.returncode,
                                             "status": "PASS" if result.returncode == 0 else "FAIL"})
    if result.returncode != 0:
        code = next((line.partition(":")[2].strip() for line in result.stdout.splitlines()
                     if line.startswith("COMPARISON_FAILURE:")), f"CHILD_EXIT_{result.returncode}")
        raise DiagnosticError(f"{model.upper()}_{code}")


def run_comparison(root: Path) -> dict[str, Any]:
    run = Run.create(root, "fixed CPU comparison of current and decomposed tryout questions",
                     dataset_version="dev-v3", profile_applicable=False)
    env = root / ".local"
    try:
        _child(root, run, "laya", env / "venv-eval" / "Scripts" / "python.exe")
        _child(root, run, "sokudan", env / "venv-sokudan-cpu" / "Scripts" / "python.exe")
        outputs = {
            model: json.loads((run.path / f"{model}-comparison.json").read_text(encoding="utf-8"))
            for model in ("laya", "sokudan")
        }
        result = {
            "run_id": run.run_id,
            "git_revision": run.read_manifest()["git_revision"],
            "source_tree_sha256": run.read_manifest()["source_tree_sha256"],
            "worktree_status": run.read_manifest()["worktree_status"],
            "dataset_version": "dev-v3",
            "dataset_sha256": run.read_manifest()["input_sha256"].get("datasets/dev-v3.jsonl"),
            "fixture_sha256": _sha256(FIXED_CASES),
            "fixture_count": len(FIXED_CASES),
            "fixture_label_status": "AI_FIXED_TECHNICAL_EXPECTATIONS_NOT_HUMAN_REVIEWED",
            "current_question_source": "unchanged dev-v3 templates used by current tryout",
            "decomposed_question_sha256": _sha256(SIMPLE_QUESTION_VARIANTS),
            "calibration_and_thresholds_changed": False,
            "results": outputs,
        }
        run.write_json("comparison-result.json", result)
        passed = all(len(outputs[model]["cases"]) == len(FIXED_CASES)
                     for model in outputs)
        run.stage("TRYOUT_COMPARISON", "PASS" if passed else "FAIL",
                  result_file="comparison-result.json", cases_per_model=len(FIXED_CASES),
                  models=["laya-cpu-fp32", "sokudan-cpu-fp32"])
        return {"status": "COMPLETE" if passed else "FAILED", "run_id": run.run_id}
    except Exception as exc:
        code = getattr(exc, "code", "TRYOUT_COMPARISON_FAILED")
        run.failure("TRYOUT_COMPARISON", code, RuntimeError(code))
        return {"status": "FAILED", "run_id": run.run_id, "code": code}


def run_model_stage(root: Path, run: Run, model: str) -> None:
    expected = "venv-eval" if model == "laya" else "venv-sokudan-cpu"
    if expected not in Path(sys.executable).parts:
        raise DiagnosticError("COMPARISON_PYTHON_ENVIRONMENT_MISMATCH")
    output = _run_model(root, run, model)
    run.write_json(f"{model}-comparison.json", output)
    run.stage(f"TRYOUT_COMPARE_{model.upper()}", "PASS", cases=len(output["cases"]),
              backend="CPU", precision="FP32")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("laya", "sokudan"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        run_model_stage(root, open_run(root, args.run_id), args.stage)
        print("COMPARISON_STAGE_COMPLETE")
        return 0
    except Exception as exc:
        code = getattr(exc, "code", "COMPARISON_STAGE_FAILED")
        try:
            open_run(root, args.run_id).write_json(f"{args.stage}-failure.json", {
                "code": code, "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
        print(f"COMPARISON_FAILURE:{code}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

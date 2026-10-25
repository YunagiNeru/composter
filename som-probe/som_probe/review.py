"""架空の日本語ケース60件を確認する自己完結HTMLを生成する。"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from .dataset import load_dataset


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _options(question: dict[str, Any]) -> list[tuple[str, str]]:
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, dict):
            return [(str(key), "" if value in (None, "") else _text(value)) for key, value in criteria.items()]
        return [(str(value), "") for value in criteria]
    if kind == "score":
        return [(str(index), _text(value)) for index, value in enumerate(criteria)]
    return [("false", "いいえ。文が成立しない"), ("true", "はい。文が成立する")]


def _draft_answer(question: dict[str, Any], labels: list[Any]) -> str:
    options = dict(_options(question))
    rendered: list[str] = []
    for label in labels:
        key = str(label).lower() if isinstance(label, bool) else str(label)
        description = options.get(key, "")
        rendered.append(key if not description else f"{key} — {description}")
    return " / ".join(rendered)


def _case_html(case: dict[str, Any], version: str,
               revision: dict[str, Any] | None = None) -> str:
    question_id, question = next(iter(case["questions"].items()))
    expected = case["expected"][question_id]
    kind = "noul" if question["type"] == "bool" else question["type"]
    options = "".join(
        f"<tr><th scope=\"row\">{html.escape(key)}</th><td>{html.escape(description) or '説明なし'}</td></tr>"
        for key, description in _options(question)
    )
    candidate_ids = [key for key, _ in _options(question)]
    if question["type"] == "bool":
        candidate_ids = ["false", "true"]
    allowed = {str(value).lower() if isinstance(value, bool) else str(value)
               for value in expected["allowed_answers"]}
    if version == "dev-v1":
        wrong_ids = expected["target_errors"]
        wrong_title = "AIドラフトの対象誤り"
    else:
        wrong_ids = [value for value in candidate_ids if value not in allowed]
        wrong_title = "許容解から評価器が導出する誤出力"
    errors = "".join(f"<li>{html.escape(_text(value))}</li>" for value in wrong_ids)
    revision_text = ""
    if revision:
        revision_text = f"<div><dt>旧版からの改訂理由</dt><dd>{html.escape(revision.get('reason', ''))}</dd></div>"
    return f"""<article class="review-case" id="{html.escape(case['case_id'])}">
<header class="record__header"><span class="record__id">{html.escape(case['case_id'])}</span><h3 class="record__title">{html.escape(kind.title())} / {html.escape(question_id)}</h3><span class="status status--open">人未確認</span></header>
<dl class="record__meta"><div><dt>データ型</dt><dd>{html.escape(question['type'])}</dd></div><div><dt>モデル型</dt><dd>{html.escape(kind)}</dd></div><div><dt>確認状態</dt><dd>{html.escape(expected['verification_status'])}</dd></div></dl>
<dl class="review-fields">
<div><dt>原文</dt><dd>{html.escape(_text(case['state']))}</dd></div>
<div><dt>質問</dt><dd>{html.escape(question['instructions'])}</dd></div>
<div><dt>全選択肢</dt><dd><table class="review-options"><caption>候補IDと候補文</caption><thead><tr><th scope="col">ID</th><th scope="col">候補</th></tr></thead><tbody>{options}</tbody></table></dd></div>
<div><dt>AI作成の正解候補・許容解</dt><dd>{html.escape(_draft_answer(question, expected['allowed_answers']))}</dd></div>
<div><dt>理由</dt><dd>{html.escape(expected['rationale'])}</dd></div>
{revision_text}
<div><dt>{wrong_title}</dt><dd><ul>{errors}</ul></dd></div>
<div><dt>人手確認</dt><dd>未実施　□ 候補を確認　□ 修正が必要<br>修正メモ：＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿</dd></div>
</dl></article>"""


def build_review_html(root: Path, destination: Path, dataset_version: str = "dev-v1") -> Path:
    snapshot = load_dataset(root, dataset_version)
    cases = snapshot["cases"]
    if len(cases) != 60:
        raise ValueError("確認資料は60件のデータに限ります。")
    dataset_sha256 = snapshot["identity"]["dataset_sha256"]
    revisions = {}
    revision_path = snapshot["identity"]["revision_ledger_path"]
    if revision_path:
        revisions = json.loads((root / revision_path).read_text(encoding="utf-8"))["cases"]
    styles = (root.parent / "docs" / "PLAN.html").read_text(encoding="utf-8")
    match = re.search(r"<style>(.*?)</style>", styles, flags=re.DOTALL)
    if not match:
        raise ValueError("PLAN.htmlから埋め込みスタイルを読み取れません。")
    counts = {kind: 0 for kind in ("choice", "score", "noul")}
    for case in cases:
        question = next(iter(case["questions"].values()))
        counts["noul" if question["type"] == "bool" else question["type"]] += 1
    index = "".join(
        f"<li><a href=\"#{html.escape(case['case_id'])}\">{html.escape(case['case_id'])}</a></li>"
        for case in cases
    )
    cards = "\n".join(_case_html(case, dataset_version, revisions.get(case["case_id"])) for case in cases)
    html_text = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><meta name="description" content="架空の日本語60件について、人が正解候補を確認するための資料。"><title>日本語ケース確認資料</title><style>{match.group(1)}
.review-fields{{display:grid;gap:1rem;margin:1.5rem 0 0}}.review-fields>div{{border-top:1px solid var(--color-rule);padding-top:.75rem}}.review-fields dt{{font-weight:700;margin-bottom:.35rem}}.review-fields dd{{margin:0;overflow-wrap:anywhere}}.review-options{{width:100%;border-collapse:collapse}}.review-options th,.review-options td{{padding:.45rem .6rem;border-bottom:1px solid var(--color-rule);text-align:left}}.review-options th:first-child{{width:8rem}}.review-case{{padding:var(--space-lg);margin-block:var(--space-lg);background:var(--color-paper-raised);border:var(--border-hairline) solid var(--color-rule);border-radius:var(--radius-sm);scroll-margin-top:1rem}}.review-index{{columns:2;gap:2rem}}.review-index li{{break-inside:avoid;margin:.25rem 0}}@media(min-width:900px){{.review-index{{columns:3}}}}@media print{{.review-index{{columns:4}}.review-case{{break-inside:avoid}}}}
</style></head><body><a class="skip-link" href="#main">本文へ移動</a><div class="document"><aside class="rail" aria-label="資料索引"><p class="rail__title">日本語<br>ケース確認</p><nav aria-label="ケース一覧"><ol>{index}</ol></nav></aside><details class="mobile-index"><summary>ケース一覧</summary><nav aria-label="モバイルケース一覧"><ol>{index}</ol></nav></details><main class="page" id="main"><header class="hero"><p class="eyebrow">人による正解確認</p><h1>日本語60件<br>正解候補の確認</h1><p class="lede">開発用の架空ケースに付けたAIドラフト正解を、人が確認する資料です。</p><dl class="document-contract"><div><dt>総数</dt><dd>60件</dd></div><div><dt>Choice</dt><dd>{counts['choice']}件</dd></div><div><dt>Score</dt><dd>{counts['score']}件</dd></div><div><dt>Noul</dt><dd>{counts['noul']}件</dd></div><div><dt>人手状態</dt><dd>全件未確認</dd></div></dl></header>
<section class="section section--wide"><header class="section__header"><h2>確認前提</h2></header><p>原文、質問、全候補、AI作成の正解候補、理由、許容解をケースごとに示します。全ラベルは <code>ai_draft_unverified</code> で、人の確認済みとして扱っていません。対象誤りは許容解の補集合から表示しています。</p><p>この資料にはモデル予測を含めません。予測出力と参照差分は、別Runのローカル記録で確認してください。確認後にこの開発用集合を調整へ使った場合、同じ集合を最終品質の証明に再利用せず、人確認済みの独立評価集合を用意してください。</p><p>データ版: <code>{dataset_version}</code>　データSHA-256: <code>{dataset_sha256}</code></p><nav aria-label="ケース索引"><ol class="review-index">{index}</ol></nav></section>
<section class="section section--wide"><header class="section__header"><h2>確認ケース</h2></header>{cards}</section></main><footer class="foot"><p>ラベル状態: 人手未確認 · 60件は開発用 · 正式な日本語品質は未判定</p></footer></div></body></html>"""
    destination.write_text(html_text, encoding="utf-8")
    return destination

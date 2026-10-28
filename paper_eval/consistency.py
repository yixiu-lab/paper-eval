from __future__ import annotations

from dataclasses import replace
from typing import Any

from .llm import AgentRuntimeConfig, call_plain_json
from .models import AgentJudgement, Evidence, ScoreItem
from .rules import RuleProfile


def _trim(text: str, limit: int = 260) -> str:
    return " ".join(str(text).split())[:limit]


def _review_models(config: AgentRuntimeConfig) -> list[str]:
    configured = list(config.consistency_models)
    if not configured:
        configured = [model for model in config.fallback_models if model != config.model]
    seen: set[str] = set()
    return [
        model
        for model in configured
        if model and model != config.model and not (model in seen or seen.add(model))
    ][: max(0, config.max_consistency_models)]


def _compact_rule_reviews(
    profile: RuleProfile,
    judgements: list[AgentJudgement],
    score_items: list[ScoreItem],
    evidences: list[Evidence],
) -> list[dict[str, Any]]:
    rules = profile.rules_by_id
    evidence_map = {item.evidence_id: item for item in evidences}
    score_map = {item.indicator_id: item for item in score_items}
    rows: list[dict[str, Any]] = []
    for judgement in judgements:
        rule = rules.get(judgement.indicator_id)
        score = score_map.get(judgement.indicator_id)
        if not rule or not score:
            continue
        snippets = []
        for evidence_id in judgement.evidence_ids[:2]:
            evidence = evidence_map.get(evidence_id)
            if evidence:
                snippets.append(
                    {
                        "page": evidence.page,
                        "section": evidence.section,
                        "text": _trim(evidence.text, 220),
                    }
                )
        rows.append(
            {
                "rule_id": rule.rule_id,
                "dimension_id": rule.dimension_id,
                "indicator_name": rule.indicator_name,
                "judgement": judgement.judgement.value,
                "weighted_score": score.weighted_score,
                "confidence": judgement.confidence,
                "reason": _trim(judgement.reason, 220),
                "evidence": snippets,
            }
        )
    return rows


def _review_prompt() -> str:
    return (
        "你是独立复判模型，负责做多模型一致性审核。"
        "请只根据论文画像、维度分、规则判断理由和证据片段判断主评分是否公正。"
        "不要根据期刊、作者、学校、影响因子、引用量加减分。"
        "不要重新生成每条规则评分，只做整体分数复判。"
        "评分标尺：每条规则都取中间档（部分完成）时总分是 60 分；全满足 100 分，全不满足 20 分。"
        "你的 recommended_score 必须按这一标尺给出，否则无法与主评分比较。"
        "如果主评分偏高或偏低，请说明具体来自哪些维度或证据链问题。"
        "输出 JSON："
        '{"recommended_score":0,"score_band":"60-69|70-79|80-89|90-100",'
        '"agreement_level":"high|medium|low","major_disagreements":["..."],'
        '"suspected_biases":["..."],"should_adjust_score":false,'
        '"adjusted_score":0,"reason":"中文复判理由","confidence":0.0}'
    )


def _normalize_review(raw: dict[str, Any], model: str, primary_score: float | None) -> dict[str, Any]:
    def number(name: str, default: float) -> float:
        try:
            return float(raw.get(name, default))
        except (TypeError, ValueError):
            return default

    fallback = float(primary_score or 0)
    recommended = max(0.0, min(100.0, number("recommended_score", fallback)))
    adjusted = max(0.0, min(100.0, number("adjusted_score", recommended)))
    should_adjust = raw.get("should_adjust_score", False)
    if isinstance(should_adjust, str):
        should_adjust = should_adjust.strip().lower() in {"true", "yes", "1", "是", "需要"}
    agreement = str(raw.get("agreement_level") or "medium").lower()
    if agreement not in {"high", "medium", "low"}:
        agreement = "medium"
    return {
        "model": model,
        "recommended_score": round(recommended, 2),
        "score_band": str(raw.get("score_band") or ""),
        "agreement_level": agreement,
        "major_disagreements": [str(item) for item in raw.get("major_disagreements", [])[:6]]
        if isinstance(raw.get("major_disagreements"), list)
        else [],
        "suspected_biases": [str(item) for item in raw.get("suspected_biases", [])[:6]]
        if isinstance(raw.get("suspected_biases"), list)
        else [],
        "should_adjust_score": bool(should_adjust),
        "adjusted_score": round(adjusted, 2),
        "reason": _trim(str(raw.get("reason") or "复判模型未给出充分理由。"), 500),
        "confidence": max(0.0, min(1.0, number("confidence", 0.5))),
    }


def run_multi_model_consistency_review(
    run_id: str,
    profile: RuleProfile,
    judgements: list[AgentJudgement],
    score_items: list[ScoreItem],
    evidences: list[Evidence],
    summary: dict[str, Any],
    paper_context: dict[str, Any] | None,
    config: AgentRuntimeConfig,
) -> dict[str, Any]:
    primary_score = summary.get("total_score")
    if not config.multi_model_review_enabled:
        return {"enabled": False, "reason": "multi_model_review_disabled", "reviews": []}
    models = _review_models(config)
    if not models:
        return {"enabled": False, "reason": "no_consistency_models_configured", "reviews": []}

    payload = {
        "run_id": run_id,
        "paper_context": {
            "paper_profile": (paper_context or {}).get("paper_profile", {}),
            "review_profile": (paper_context or {}).get("review_profile", {}),
            "reading_notes": (paper_context or {}).get("reading_notes", ""),
        },
        "primary_summary": {
            "total_score": primary_score,
            "quality_grade": summary.get("quality_grade"),
            "top_journal_potential": summary.get("top_journal_potential"),
            "scoring_completeness": summary.get("scoring_completeness"),
            "dimension_scores": summary.get("dimension_scores", {}),
            "top_journal_gate_reasons": summary.get("top_journal_gate_reasons", []),
        },
        "rule_reviews": _compact_rule_reviews(profile, judgements, score_items, evidences),
    }
    reviews: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for model in models:
        review_config = replace(config, model=model, fallback_models=(), consistency_models=())
        try:
            raw = call_plain_json(_review_prompt(), payload, review_config)
            reviews.append(_normalize_review(raw, model, float(primary_score) if primary_score is not None else None))
        except Exception as exc:
            failures.append({"model": model, "error": str(exc)})

    usable_scores = [float(item["adjusted_score"] if item["should_adjust_score"] else item["recommended_score"]) for item in reviews]
    primary = float(primary_score) if primary_score is not None else None
    if primary is None or not usable_scores:
        consensus_score = primary
    else:
        reviewer_avg = sum(usable_scores) / len(usable_scores)
        weight = 0.3 if len(usable_scores) >= 2 else 0.2
        consensus_score = round(primary * (1 - weight) + reviewer_avg * weight, 2)
    deltas = [abs(float(item["recommended_score"]) - float(primary or 0)) for item in reviews]
    agreement = "unavailable"
    if reviews:
        max_delta = max(deltas) if deltas else 0
        low_count = sum(1 for item in reviews if item["agreement_level"] == "low")
        if max_delta <= 5 and low_count == 0:
            agreement = "high"
        elif max_delta <= 10 and low_count <= 1:
            agreement = "medium"
        else:
            agreement = "low"
    return {
        "enabled": True,
        "primary_model": config.model,
        "review_models": models,
        "reviews": reviews,
        "failures": failures,
        "primary_score": primary,
        "consensus_score": consensus_score,
        "agreement_level": agreement,
        "score_delta_max": round(max(deltas), 2) if deltas else None,
        "needs_human_review": agreement in {"low", "unavailable"} or bool(failures),
        "calculation": "consensus_score = primary_score * 0.70 + reviewer_average * 0.30 when two reviewers are available; one reviewer uses 0.20.",
    }

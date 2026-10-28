from __future__ import annotations

from dataclasses import replace
from typing import Any

from .llm import AgentRuntimeConfig, call_plain_json
from .models import AgentJudgement, Evidence, ReflectionIssue, ScoreItem
from .rules import RuleProfile


def _clip(text: str, limit: int = 160) -> str:
    return " ".join(text.split())[:limit]


def build_audit_payload(
    run_id: str,
    profile: RuleProfile,
    judgements: list[AgentJudgement],
    score_items: list[ScoreItem],
    evidences: list[Evidence],
    summary: dict[str, Any],
    paper_context: dict[str, Any] | None = None,
    local_issues: list[ReflectionIssue] | None = None,
) -> dict[str, Any]:
    evidence_map = {item.evidence_id: item for item in evidences}
    score_map = {item.indicator_id: item for item in score_items}
    rule_map = profile.rules_by_id
    in_scope_judgements = [
        judgement
        for judgement in judgements
        if not score_map[judgement.indicator_id].out_of_scope
    ]
    high_risk = [
        judgement
        for judgement in in_scope_judgements
        if judgement.needs_review
        or judgement.confidence < 0.45
        or not judgement.evidence_ids
        or score_map[judgement.indicator_id].weighted_score is None
    ]
    high_score = [
        judgement
        for judgement in in_scope_judgements
        if score_map[judgement.indicator_id].weighted_score
        and score_map[judgement.indicator_id].weighted_score >= score_map[judgement.indicator_id].weight * 0.75
    ]
    audit_targets = high_risk[:10] + high_score[:4]
    seen_targets: set[str] = set()
    detail_rows = []
    for judgement in audit_targets:
        if judgement.indicator_id in seen_targets:
            continue
        seen_targets.add(judgement.indicator_id)
        score = score_map[judgement.indicator_id]
        rule = rule_map[judgement.indicator_id]
        detail_rows.append(
            {
                "rule_id": judgement.indicator_id,
                "dimension_id": score.dimension_id,
                "indicator_name": rule.indicator_name,
                "judgement": judgement.judgement.value,
                "weighted_score": score.weighted_score,
                "weight": score.weight,
                "confidence": judgement.confidence,
                "needs_review": judgement.needs_review,
                "reason": judgement.reason,
                "accepted_evidence_pages": [
                    evidence_map[eid].page for eid in judgement.evidence_ids if eid in evidence_map
                ],
                "candidate_evidence_pages": [
                    evidence_map[eid].page for eid in judgement.candidate_evidence_ids if eid in evidence_map
                ],
                "accepted_evidence_preview": [
                    {
                        "evidence_id": eid,
                        "page": evidence_map[eid].page,
                        "text": _clip(evidence_map[eid].text),
                    }
                    for eid in judgement.evidence_ids[:2]
                    if eid in evidence_map
                ],
            }
        )
    return {
        "run_id": run_id,
        "paper_profile": (paper_context or {}).get("paper_profile", {}),
        "read_coverage": {
            key: value
            for key, value in (paper_context or {}).get("dimension_evidence_routes", {}).items()
        },
        "summary": {
            "total_score": summary.get("total_score"),
            "scoring_completeness": summary.get("scoring_completeness"),
            "judgement_counts": summary.get("judgement_counts"),
            "dimension_scores": {
                key: {
                    "dimension_name": value.get("dimension_name"),
                    "normalized_score": value.get("normalized_score"),
                    "uncertain_weight": value.get("uncertain_weight"),
                }
                for key, value in summary.get("dimension_scores", {}).items()
                if isinstance(value, dict)
            },
        },
        "audit_scope": {
            "total_rules": len(in_scope_judgements),
            "excluded_out_of_scope_rules": len(judgements) - len(in_scope_judgements),
            "reviewed_rules": len(detail_rows),
            "selection_policy": "优先审核低置信度、无证据、无法判断、已标记复核和高分项。",
        },
        "local_reflection_issues": [
            {
                "issue_type": item.issue_type,
                "severity": item.severity,
                "message": item.message,
                "related_ids": item.related_ids[:12],
            }
            for item in (local_issues or [])
        ],
        "scoring_details": detail_rows,
    }


def normalize_audit_output(raw: dict[str, Any]) -> dict[str, Any]:
    issues = raw.get("audit_issues", [])
    if not isinstance(issues, list):
        issues = []
    normalized_issues = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        related = issue.get("related_ids", [])
        if isinstance(related, str):
            related = [related]
        if not isinstance(related, list):
            related = []
        normalized_issues.append(
            {
                "issue_type": str(issue.get("issue_type") or "audit_issue"),
                "severity": str(issue.get("severity") or "medium"),
                "message": str(issue.get("message") or "审核智能体发现需要复核的问题。"),
                "related_ids": [str(item) for item in related][:20],
                "suggested_action": str(issue.get("suggested_action") or "人工复核相关评分、证据和理由。"),
            }
        )
    return {
        "audit_status": str(raw.get("audit_status") or "needs_review"),
        "overall_assessment": str(raw.get("overall_assessment") or "审核智能体未给出总体意见。"),
        "score_reliability": str(raw.get("score_reliability") or "unknown"),
        "score_adjustment_needed": bool(raw.get("score_adjustment_needed", False)),
        "audit_issues": normalized_issues,
    }


def run_reflection_audit_agent(
    run_id: str,
    profile: RuleProfile,
    judgements: list[AgentJudgement],
    score_items: list[ScoreItem],
    evidences: list[Evidence],
    summary: dict[str, Any],
    paper_context: dict[str, Any] | None,
    local_issues: list[ReflectionIssue],
    config: AgentRuntimeConfig,
) -> dict[str, Any]:
    prompt = (
        "你是反思审核智能体，负责审核评分过程和评分结果，而不是重新评分。"
        "请检查：论文类型与规则是否匹配、全文证据分配是否覆盖关键页、专家判断是否过度、"
        "高分是否有足够证据、无法判断是否合理、总分是否可能失真。"
        "你不能直接修改分数，只能给出审核结论和需要人工复核的项目。"
        "输出 JSON："
        '{"audit_status":"pass|needs_review|fail","overall_assessment":"中文总体审核意见",'
        '"score_reliability":"high|medium|low","score_adjustment_needed":false,'
        '"audit_issues":[{"issue_type":"evidence_overclaim","severity":"low|medium|high",'
        '"message":"中文说明","related_ids":["规则ID"],"suggested_action":"中文建议"}]}'
    )
    payload = build_audit_payload(
        run_id,
        profile,
        judgements,
        score_items,
        evidences,
        summary,
        paper_context,
        local_issues,
    )
    try:
        audit_config = replace(config, timeout_seconds=min(config.timeout_seconds, 90))
        raw = call_plain_json(prompt, payload, audit_config)
    except Exception as exc:
        return {
            "audit_status": "needs_review",
            "overall_assessment": f"反思审核智能体未能在快速审核时间内完成：{exc}",
            "score_reliability": "unknown",
            "score_adjustment_needed": True,
            "audit_issues": [
                {
                    "issue_type": "audit_timeout",
                    "severity": "medium",
                    "message": "评分主流程已完成，但反思审核 Agent 超时或连接失败。",
                    "related_ids": [],
                    "suggested_action": "人工优先复核低置信度、NA、高分和质量警告项；必要时单独重跑审核。",
                }
            ],
        }
    return normalize_audit_output(raw)


def audit_to_reflection_issues(audit: dict[str, Any]) -> list[ReflectionIssue]:
    issues = []
    for item in audit.get("audit_issues", []):
        issues.append(
            ReflectionIssue(
                issue_type=f"audit:{item['issue_type']}",
                severity=item["severity"],
                message=item["message"],
                related_ids=item["related_ids"],
                suggested_action=item["suggested_action"],
            )
        )
    return issues

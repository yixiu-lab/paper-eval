from __future__ import annotations

from collections import Counter

from .models import AgentJudgement, Evidence, ReflectionIssue, ScoreItem


def reflect_run(judgements: list[AgentJudgement], score_items: list[ScoreItem], evidences: list[Evidence]) -> list[ReflectionIssue]:
    issues: list[ReflectionIssue] = []
    evidence_ids = {item.evidence_id for item in evidences}
    in_scope_ids = {item.indicator_id for item in score_items if not item.out_of_scope}
    in_scope_judgements = [item for item in judgements if item.indicator_id in in_scope_ids]
    in_scope_scores = [item for item in score_items if not item.out_of_scope]
    missing = [j.indicator_id for j in in_scope_judgements if not j.evidence_ids]
    if missing:
        issues.append(
            ReflectionIssue(
                issue_type="evidence_missing",
                severity="medium",
                message=f"{len(missing)} 个指标缺少可用证据，报告应进入人工复核。",
                related_ids=missing[:20],
                suggested_action="补充 PDF 解析质量、检索策略或让 Verifier 重新定位证据。",
            )
        )

    invalid = [j.indicator_id for j in in_scope_judgements for eid in j.evidence_ids if eid not in evidence_ids]
    if invalid:
        issues.append(
            ReflectionIssue(
                issue_type="invalid_evidence_id",
                severity="high",
                message="发现 Agent 引用了不存在的证据 ID。",
                related_ids=invalid,
                suggested_action="退回对应 Agent 输出并重新生成判断。",
            )
        )

    counts = Counter(item.judgement.value for item in in_scope_scores)
    if counts.get("satisfied", 0) / max(len(in_scope_scores), 1) > 0.85:
        issues.append(
            ReflectionIssue(
                issue_type="score_distribution",
                severity="medium",
                message="满足项比例过高，可能存在证据匹配过宽或规则区分度不足。",
                related_ids=[],
                suggested_action="抽样检查高分指标证据，必要时提高 evidence gate 阈值。",
            )
        )

    zero_available = [item.dimension_id for item in in_scope_scores if item.weighted_score is None]
    if len(zero_available) > len(in_scope_scores) * 0.4:
        issues.append(
            ReflectionIssue(
                issue_type="low_completeness",
                severity="medium",
                message="无法判断项过多，评分完整度偏低。",
                related_ids=zero_available[:20],
                suggested_action="优先改善章节识别、证据检索和规则关键词。",
            )
        )
    return issues

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
                message=f"{len(missing)} indicators lack available evidence and the report should be subject to manual review.",
                related_ids=missing[:20],
                suggested_action='Supplement PDF parsing quality, search strategy or let Verifier relocate evidence.',
            )
        )

    invalid = [j.indicator_id for j in in_scope_judgements for eid in j.evidence_ids if eid not in evidence_ids]
    if invalid:
        issues.append(
            ReflectionIssue(
                issue_type="invalid_evidence_id",
                severity="high",
                message='It was discovered that the Agent referenced an evidence ID that does not exist.',
                related_ids=invalid,
                suggested_action='returns the corresponding Agent output and regenerates the judgment.',
            )
        )

    counts = Counter(item.judgement.value for item in in_scope_scores)
    if counts.get("satisfied", 0) / max(len(in_scope_scores), 1) > 0.85:
        issues.append(
            ReflectionIssue(
                issue_type="score_distribution",
                severity="medium",
                message='The proportion of satisfied items is too high, and there may be evidence matching that is too wide or rule discrimination that is insufficient.',
                related_ids=[],
                suggested_action='Randomly check high-scoring indicator evidence and increase the evidence gate threshold if necessary.',
            )
        )

    zero_available = [item.dimension_id for item in in_scope_scores if item.weighted_score is None]
    if len(zero_available) > len(in_scope_scores) * 0.4:
        issues.append(
            ReflectionIssue(
                issue_type="low_completeness",
                severity="medium",
                message='There are too many items that cannot be judged, and the score completeness is low.',
                related_ids=zero_available[:20],
                suggested_action='Prioritize improving chapter identification, evidence retrieval and rule keywords.',
            )
        )
    return issues

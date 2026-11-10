from __future__ import annotations

from .models import AgentJudgement, Judgement
from .rules import RuleProfile


def validate_agent_judgements(profile: RuleProfile, judgements: list[AgentJudgement]) -> list[str]:
    errors: list[str] = []
    expected_rule_ids = {rule.rule_id for rule in profile.rules}
    seen_rule_ids: set[str] = set()
    for item in judgements:
        if item.indicator_id not in expected_rule_ids:
            errors.append(f"Unknown indicator_id: {item.indicator_id}")
        if item.indicator_id in seen_rule_ids:
            errors.append(f"Duplicate judgement for indicator_id: {item.indicator_id}")
        seen_rule_ids.add(item.indicator_id)
        if item.judgement not in set(Judgement):
            errors.append(f"Invalid judgement for {item.indicator_id}: {item.judgement}")
        if not 0 <= item.confidence <= 1:
            errors.append(f"Confidence out of range for {item.indicator_id}: {item.confidence}")
        if item.judgement in {Judgement.SATISFIED, Judgement.PARTIAL} and not item.evidence_ids:
            errors.append(f"{item.indicator_id} has positive judgement but no evidence_ids")
        if not item.reason.strip():
            errors.append(f"{item.indicator_id} missing reason")
        if not item.agent_id.strip():
            errors.append(f"{item.indicator_id} missing agent_id")
    missing = expected_rule_ids - seen_rule_ids
    if missing:
        errors.append(f"Missing judgements for rules: {', '.join(sorted(missing))}")
    return errors

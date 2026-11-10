'Evidence gating self-test: The gating only marks and does not change the model judgment.\n\nRun:\n    python ".\\scripts\\self_test_evidence_gate.py"'

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.evidence_gate import apply_evidence_gate
from paper_eval.models import AgentJudgement, Evidence, Judgement


def evidence(evidence_id: str, page: int, section: str, text: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        paper_id="PAPER-TEST",
        section=section,
        page=page,
        span=(0, len(text)),
        text=text,
        text_hash=f"hash-{evidence_id}",
    )


LONG_METHOD_TEXT = (
    "We collected survey data from 412 employees across 38 teams. "
    "The measurement model was assessed with confirmatory factor analysis, "
    "and the regression analysis controls for tenure, age and firm size. "
) * 1

LONG_THEORY_TEXT = (
    "Our theoretical framework integrates social exchange theory with the literature on "
    "algorithmic management, and specifies the mechanism through which perceived "
    "AI feedback shapes employee contribution. This is our main contribution to the field. "
) * 1


def judgement(
    rule_id: str,
    value: Judgement,
    evidence_ids: list[str],
    reason: str,
) -> AgentJudgement:
    return AgentJudgement(
        task_id="RUN-TEST",
        agent_id="expert_agent",
        indicator_id=rule_id,
        judgement=value,
        evidence_ids=evidence_ids,
        reason=reason,
        confidence=0.9,
        needs_review=False,
        agent_version="self-check",
        model_version="test",
    )


def main() -> int:
    evidences = [
        evidence("EV-1", 4, "Method", LONG_METHOD_TEXT),
        evidence("EV-2", 5, "Method", LONG_METHOD_TEXT),
        evidence("EV-3", 3, "Theory", LONG_THEORY_TEXT),
        evidence("EV-4", 4, "Theory", LONG_THEORY_TEXT),
        evidence("EV-5", 1, "Abstract", "This paper discusses the limitations of the proposed approach in the abstract only."),
    ]

    weak_reason = 'The methods section describes the samples and measurements, but only partial testing is reported.'
    clean_reason = 'The methods chapter gives samples, measurements and regression settings across two pages, with complete evidence.'

    cases = [
        judgement("D-DESIGN-FIT", Judgement.SATISFIED, ["EV-1"], weak_reason),
        judgement("D-MEASUREMENT-VALIDITY", Judgement.SATISFIED, ["EV-1", "EV-2"], clean_reason),
        judgement("B-CONCEPT-MECHANISM-NOVELTY-V1", Judgement.SATISFIED, ["EV-3", "EV-4"], clean_reason),
        judgement("E-RESULTS-COMPLETENESS", Judgement.SATISFIED, [], 'results table complete.'),
        judgement("F-LIMITATIONS", Judgement.PARTIAL, ["EV-5"], 'I only saw limitations mentioned in the abstract.'),
        judgement("A-QUESTION-CLARITY-V1", Judgement.SATISFIED, ["EV-NOT-EXIST", "EV-1"], clean_reason),
    ]
    gated = {item.indicator_id: item for item in apply_evidence_gate(cases, evidences)}

    # 1. Satisfied with weak reasons: the judgment remains unchanged, the reasons remain unchanged, and only the mark is added.
    first = gated["D-DESIGN-FIT"]
    assert first.judgement == Judgement.SATISFIED, first.judgement
    assert first.reason == weak_reason, 'Gate control must not rewrite the model reason'
    assert first.review_flags, 'Weak evidence is satisfied and a review reminder must be left'
    assert first.needs_review is True

    # 2. The evidence is sufficient and the reason is clean: should not be disturbed.
    second = gated["D-MEASUREMENT-VALIDITY"]
    assert second.judgement == Judgement.SATISFIED
    assert second.review_flags == [], second.review_flags
    assert second.needs_review is False

    # 3. Under the same set of evidence, the B-dimension cross-page theoretical evidence will not be downgraded.
    third = gated["B-CONCEPT-MECHANISM-NOVELTY-V1"]
    assert third.judgement == Judgement.SATISFIED, third.judgement

    # 4. The hard threshold is still there: Positive judgment but zero evidence → Unable to judge, the reasons remain the same.
    fourth = gated["E-RESULTS-COMPLETENESS"]
    assert fourth.judgement == Judgement.UNCERTAIN, fourth.judgement
    assert fourth.reason == 'results table complete.', 'Reasons cannot be rewritten even if the threshold is hard'
    assert fourth.review_flags

    # 5. partial + only summary evidence: the judgment remains partial, the reason remains unchanged, and a hint is left.
    fifth = gated["F-LIMITATIONS"]
    assert fifth.judgement == Judgement.PARTIAL, fifth.judgement
    assert fifth.reason == 'I only saw limitations mentioned in the abstract.'
    assert fifth.review_flags

    # 6. Non-existent evidence IDs are still eliminated (traceability).
    sixth = gated["A-QUESTION-CLARITY-V1"]
    assert sixth.evidence_ids == ["EV-1"], sixth.evidence_ids

    print('Evidence gating self-test passed: all 6 assertions are true, judgment/reason has not been rewritten by local rules.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

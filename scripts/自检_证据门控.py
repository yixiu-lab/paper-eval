"""证据门控自检：门控只打标记，不改模型判断。

运行：
    python ".\scripts\自检_证据门控.py"
"""

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

    weak_reason = "方法部分说明了样本与测量，但仅报告了部分检验。"
    clean_reason = "方法章节跨页给出了样本、测量与回归设定，证据完整。"

    cases = [
        judgement("D-DESIGN-FIT", Judgement.SATISFIED, ["EV-1"], weak_reason),
        judgement("D-MEASUREMENT-VALIDITY", Judgement.SATISFIED, ["EV-1", "EV-2"], clean_reason),
        judgement("B-CONCEPT-MECHANISM-NOVELTY-V1", Judgement.SATISFIED, ["EV-3", "EV-4"], clean_reason),
        judgement("E-RESULTS-COMPLETENESS", Judgement.SATISFIED, [], "结果表完整。"),
        judgement("F-LIMITATIONS", Judgement.PARTIAL, ["EV-5"], "只看到摘要里提到局限。"),
        judgement("A-QUESTION-CLARITY-V1", Judgement.SATISFIED, ["EV-NOT-EXIST", "EV-1"], clean_reason),
    ]
    gated = {item.indicator_id: item for item in apply_evidence_gate(cases, evidences)}

    # 1. 弱理由的 satisfied：判断不变，理由不变，只加标记。
    first = gated["D-DESIGN-FIT"]
    assert first.judgement == Judgement.SATISFIED, first.judgement
    assert first.reason == weak_reason, "门控不得改写模型理由"
    assert first.review_flags, "弱证据 satisfied 必须留下复核提示"
    assert first.needs_review is True

    # 2. 证据充分且理由干净：不该被打扰。
    second = gated["D-MEASUREMENT-VALIDITY"]
    assert second.judgement == Judgement.SATISFIED
    assert second.review_flags == [], second.review_flags
    assert second.needs_review is False

    # 3. 同一套证据下，B 维度跨页理论证据不降档。
    third = gated["B-CONCEPT-MECHANISM-NOVELTY-V1"]
    assert third.judgement == Judgement.SATISFIED, third.judgement

    # 4. 硬门槛仍在：正向判断却零证据 → 无法判断，理由保持原样。
    fourth = gated["E-RESULTS-COMPLETENESS"]
    assert fourth.judgement == Judgement.UNCERTAIN, fourth.judgement
    assert fourth.reason == "结果表完整。", "硬门槛也不得改写理由"
    assert fourth.review_flags

    # 5. partial + 只有摘要证据：判断保持 partial，理由不变，留下提示。
    fifth = gated["F-LIMITATIONS"]
    assert fifth.judgement == Judgement.PARTIAL, fifth.judgement
    assert fifth.reason == "只看到摘要里提到局限。"
    assert fifth.review_flags

    # 6. 不存在的证据 ID 仍然被剔除（可追溯性）。
    sixth = gated["A-QUESTION-CLARITY-V1"]
    assert sixth.evidence_ids == ["EV-1"], sixth.evidence_ids

    print("证据门控自检通过：6 项断言全部成立，judgement/reason 未被本地规则改写。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

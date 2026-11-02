'Evidence Gating.\n\nContract (Revised 2026-09): Gating is a **checker**, not a second grader.\n\n- Only `review_flags` and `needs_review` are allowed to be written, and the values given by the model are not allowed to be overwritten.\n  `judgement`/`reason`/`confidence`.\n- The only exception: when forward judgment (satisfied/partial) cannot find any valid evidence ID,\n  The judgment becomes uncertain – this is a hard threshold for traceability, not a quality judgment.\n- The strength of the evidence is just a reminder. Whether to downgrade is determined by manual review and is no longer determined by the number of local keyword hits.'

from __future__ import annotations

from dataclasses import replace

from .models import AgentJudgement, Evidence, Judgement


SUBSTANTIVE_MARKERS = {
    "method": [
        "method",
        "methods",
        "sample",
        "participants",
        "measure",
        "measurement",
        "survey",
        "experiment",
        "regression",
        "model",
        "analysis",
        "procedure",
        "data",
        'method',
        'sample',
        'Measurement',
        'Experiment',
        'Return',
        'model',
        'data',
    ],
    "result": [
        "result",
        "results",
        "finding",
        "findings",
        "coefficient",
        "significant",
        "table",
        "figure",
        "robust",
        "hypothesis",
        'results',
        'found',
        'significant',
        'table',
        'diagram',
        'Robust',
        'Assumption',
    ],
    "theory": [
        "theory",
        "theoretical",
        "framework",
        "mechanism",
        "construct",
        "hypothesis",
        "literature",
        "contribution",
        'Theory',
        'frame',
        'mechanism',
        'Concept',
        'Assumption',
        'Literature',
        'Contribution',
    ],
}


def _is_abstract_only(items: list[Evidence]) -> bool:
    if not items:
        return False
    return all("abstract" in item.section.lower() or item.page == 1 for item in items)


def _substantive_pages(items: list[Evidence]) -> set[int]:
    return {
        item.page
        for item in items
        if "abstract" not in item.section.lower() and len(item.text.strip()) >= 80
    }


def _has_reason_score_conflict(reason: str) -> bool:
    lowered = reason.lower()
    conflict_markers = [
        'Not clear',
        'Unclear',
        'Insufficient',
        'Insufficient',
        'Insufficient',
        'missing',
        'lack',
        'not seen',
        'None',
        'only',
        'just',
        'hint',
        'part',
        'Unable to determine',
        "cannot determine",
        "insufficient",
        "not explicit",
        "not clearly",
        "only suggests",
        "partially",
    ]
    return any(marker in lowered for marker in conflict_markers)


def _marker_hits(items: list[Evidence], marker_group: str) -> int:
    markers = SUBSTANTIVE_MARKERS[marker_group]
    return sum(
        1
        for item in items
        if any(marker in f"{item.section} {item.text}".lower() for marker in markers)
    )


def evidence_strength(items: list[Evidence], indicator_id: str) -> tuple[str, str]:
    if not items:
        return "none", 'No valid evidence'
    if _is_abstract_only(items):
        return "mention_only", 'The evidence mainly comes from the abstract or homepage statement'

    pages = _substantive_pages(items)
    sections = {item.section.strip().lower() for item in items if item.section.strip()}
    long_blocks = [item for item in items if len(item.text.strip()) >= 120 and "abstract" not in item.section.lower()]
    method_hits = _marker_hits(items, "method")
    result_hits = _marker_hits(items, "result")
    theory_hits = _marker_hits(items, "theory")

    if indicator_id.startswith(("D-", "E-")):
        if len(pages) >= 2 and len(long_blocks) >= 2 and (method_hits >= 2 or result_hits >= 2 or method_hits + result_hits >= 2):
            return "strong_support", 'Method/Results Evidence spans two pages and contains substantive method or results information'
        if len(pages) >= 1 and (method_hits >= 1 or result_hits >= 1):
            return "partial_support", 'There is evidence of methods or results, but there is insufficient cross-page or lack of detail.'
        return "mention_only", 'Only saw mentions of methods or results, lacking substantive details'

    if indicator_id.startswith(("B-", "C-")):
        if len(pages) >= 2 and len(long_blocks) >= 2 and theory_hits >= 2:
            return "strong_support", 'Theoretical evidence spans two pages and contains construct, mechanism, framework or literature contribution information'
        if len(pages) >= 1 and theory_hits >= 1:
            return "partial_support", 'There is theoretically relevant evidence, but the mechanism or contribution is insufficiently developed.'
        return "mention_only", 'Only saw mentions related to theory, lack of mechanism or contribution expansion'

    if len(pages) >= 2 and len(long_blocks) >= 2 and len(sections) >= 1:
        return "strong_support", 'The evidence comes from multiple texts and the text is relatively complete.'
    if len(pages) >= 1 and long_blocks:
        return "partial_support", 'Textual evidence is available, but quantity or coverage is limited'
    return "mention_only", 'The evidence is short or stops at general mentions'


def apply_evidence_gate(judgements: list[AgentJudgement], evidences: list[Evidence]) -> list[AgentJudgement]:
    evidence_map = {item.evidence_id: item for item in evidences}
    positive = {Judgement.SATISFIED, Judgement.MOSTLY, Judgement.PARTIAL}
    gated: list[AgentJudgement] = []
    for judgement in judgements:
        valid_ids = [
            evidence_id
            for evidence_id in judgement.evidence_ids
            if evidence_id in evidence_map and len(evidence_map[evidence_id].text.strip()) >= 40
        ]

        # Hard threshold: positive judgment but without any valid evidence, cannot be traced, and becomes impossible to judge.
        if judgement.judgement in positive and not valid_ids:
            gated.append(
                replace(
                    judgement,
                    judgement=Judgement.UNCERTAIN,
                    evidence_ids=[],
                    confidence=min(judgement.confidence, 0.25),
                    needs_review=True,
                    review_flags=[
                        *judgement.review_flags,
                        'Evidence gating: No valid evidence ID or evidence text was found, and it became impossible to judge because it could not be traced back.',
                    ],
                )
            )
            continue

        valid_evidences = [evidence_map[evidence_id] for evidence_id in valid_ids]
        strength, strength_reason = evidence_strength(valid_evidences, judgement.indicator_id)
        flags: list[str] = []
        if judgement.judgement == Judgement.SATISFIED:
            if _has_reason_score_conflict(judgement.reason):
                flags.append(
                    'Evidence gating prompt: There is insufficient evidence or only partially supported statements in the reasons, which are inconsistent with satisfied and require manual review.'
                )
            if strength != "strong_support":
                flags.append(
                    f"Evidence strength reminder: The evidence strength of satisfied is {strength} ({strength_reason}), which requires manual review."
                )
            if len(valid_ids) < 2:
                flags.append(
                    f"High score evidence tip: satisfied only adopts {len(valid_ids)} valid evidence, which requires manual review."
                )
        elif judgement.judgement == Judgement.PARTIAL and strength == "mention_only":
            flags.append(
                f"Evidence strength reminder: The evidence strength of partial is {strength} ({strength_reason}), which requires manual review."
            )

        gated.append(
            replace(
                judgement,
                evidence_ids=valid_ids,
                needs_review=judgement.needs_review or bool(flags),
                review_flags=[*judgement.review_flags, *flags],
            )
        )
    return gated

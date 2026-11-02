from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AgentJudgement, Evidence, EvaluationRun, ExtendedAnalysis, Paper, ReflectionIssue, ScoreItem, to_dict
from .pdf_diagnostics import parse_warning_message
from .rules import RuleProfile


def public_score_summary(summary: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(summary, ensure_ascii=False))
    cleaned.pop("strict_top_journal_reference_score", None)
    cleaned.pop("strict_raw_earned_score", None)
    for item in cleaned.get("dimension_scores", {}).values():
        if isinstance(item, dict):
            item.pop("strict_earned_score", None)
            item.pop("strict_normalized_score", None)
    return cleaned


def public_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = json.loads(json.dumps(trace, ensure_ascii=False))
    for item in cleaned:
        if isinstance(item, dict) and isinstance(item.get("summary"), dict):
            item["summary"] = public_score_summary(item["summary"])
    return cleaned


def build_quality_warnings(
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    reflection_issues: list[ReflectionIssue],
    paper_context: dict[str, Any] | None,
    audit_report: dict[str, Any] | None,
    consistency_report: dict[str, Any] | None = None,
    parse_diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if isinstance(parse_diagnostics, dict) and not parse_diagnostics.get("parse_ok", True):
        warning_items = [str(item) for item in parse_diagnostics.get("warnings", [])]
        warnings.append(
            {
                "level": "high",
                "type": "pdf_parse_diagnostics",
                "message": (
                    'PDF:'
                    f"{parse_warning_message(warning_items)}. It is recommended to first check whether the PDF is a scanned version, a garbled version, or the chapter structure is abnormal.\nThe score integrity of"
                ),
            }
        )
    completeness = float(summary.get("scoring_completeness") or 0)
    if completeness < 0.8:
        warnings.append(
            {
                "level": "high",
                "type": "low_completeness",
                "message": f"is {completeness:.0%}, with many NAs. The total score needs to be combined with manual review.",
            }
        )
    na_by_dimension: dict[str, int] = {}
    for item in details:
        if item.get("out_of_scope"):
            continue
        if item.get("judgement") == "uncertain":
            na_by_dimension[item["dimension_id"]] = na_by_dimension.get(item["dimension_id"], 0) + 1
    for dimension_id, count in sorted(na_by_dimension.items()):
        if count >= 3:
            dimension_name = summary["dimension_scores"].get(dimension_id, {}).get("dimension_name", "")
            warnings.append(
                {
                    "level": "medium",
                    "type": "dimension_na_cluster",
                    "message": f"{dimension_id}. {dimension_name} There are {count} rules that cannot be judged. It is recommended to review the evidence of this dimension first.",
                }
            )
    def _has_flag(item: dict[str, Any], *markers: str) -> bool:
        flags = item.get("review_flags") or []
        return any(marker in str(flag) for flag in flags for marker in markers)

    gate_adjusted = [
        item
        for item in details
        if not item.get("out_of_scope") and _has_flag(item, 'Evidence strength reminder', 'High score evidence prompt', 'Evidence gating prompt')
    ]
    if gate_adjusted:
        warnings.append(
            {
                "level": "medium",
                "type": "high_score_gate",
                "message": (
                    f"{len(gate_adjusted)} There are doubts about the strength of the evidence or the consistency of the reasons in the judgment."
                    'The local check only marks and determines whether the model has not been changed. Manual review is required to see whether the original gear is maintained.'
                ),
            }
        )
    weak_partial = [
        item
        for item in details
        if not item.get("out_of_scope") and item.get("judgement") == "partial" and _has_flag(item, 'Strength of evidence\nThe partially completed judgment of')
    ]
    if weak_partial:
        warnings.append(
            {
                "level": "medium",
                "type": "weak_partial_evidence",
                "message": f"{len(weak_partial)} only generally mentions the evidence, and it is recommended that experts review it first.",
            }
        )
    weak_reasons = [
        item
        for item in details
        if not item.get("out_of_scope") and _has_flag(item, 'Reason is too short or too general')
    ]
    if weak_reasons:
        warnings.append(
            {
                "level": "medium",
                "type": "weak_agent_reason",
                "message": f"{len(weak_reasons)} The scoring reasons are too short or too broad. It is recommended to review the explanation of the corresponding indicators.",
            }
        )
    review_profile = (paper_context or {}).get("review_profile", {}) if isinstance(paper_context, dict) else {}
    if isinstance(paper_context, dict) and paper_context.get("context_error"):
        warnings.append(
            {
                "level": "high",
                "type": "paper_context_fallback",
                "message": (
                    'The full-text understanding Agent failed. This scoring uses local heuristic portraits and lacks full-text sharing context.'
                    'Rule-level evidence and judgment are still valid, but the paper portrait, chapter understanding and consistency judgment require manual review.'
                ),
            }
        )
    if isinstance(paper_context, dict) and (paper_context.get("paper_profile") or {}).get("paper_type_source") == "fallback":
        warnings.append(
            {
                "level": "medium",
                "type": "paper_type_fallback",
                "message": (
                    'The paper type cannot be determined by the full-text understanding agent and has been processed as "mixed" - all rules are applied and no rules are excluded by type.'
                    'If you need to divide the paper by type (for example, theoretical papers do not evaluate data sample rules), please manually confirm the paper type.'
                ),
            }
        )
    if isinstance(paper_context, dict) and paper_context.get("context_incomplete"):
        missing = [str(item) for item in paper_context["context_incomplete"]]
        warnings.append(
            {
                "level": "high",
                "type": "paper_context_incomplete",
                "message": (
                    'The paper portrait returned by the full-text understanding agent is missing key fields ('
                    + "、".join(missing)
                    + '), the follow-up expert Agent lacks this part of the shared context, and the portrait of the paper in the report cannot be taken seriously, and manual review is recommended.'
                ),
            }
        )
    if isinstance(paper_context, dict) and paper_context.get("routing_error"):
        warnings.append(
            {
                "level": "medium",
                "type": "routing_fallback",
                "message": 'The full-text evidence routing Agent is unavailable this time, and the system instead uses the local chapter index to allocate candidate evidence; the final scoring is still completed by the expert Agent.',
            }
        )
    if isinstance(review_profile, dict) and review_profile.get("nonstandard_paradigm"):
        warnings.append(
            {
                "level": "info",
                "type": "nonstandard_paradigm",
                "message": 'The system recognizes unconventional research paradigms and atypical but valuable papers should be protected when scoring.',
            }
        )
    if reflection_issues:
        warnings.append(
            {
                "level": "medium",
                "type": "reflection_issues",
                "message": f"The reflection module found {len(reflection_issues)} issues that need review.",
            }
        )
    if isinstance(audit_report, dict) and audit_report.get("audit_status") in {"needs_review", "failed"}:
        warnings.append(
            {
                "level": "high",
                "type": "audit_agent",
                "message": f"The status of the reflective review agent is {audit_report.get('audit_status')}. It is recommended to check the review comments.",
            }
        )
    if isinstance(consistency_report, dict) and consistency_report.get("agreement_level") == "low":
        warnings.append(
            {
                "level": "high",
                "type": "multi_model_disagreement",
                "message": 'The consistency of multi-model re-judgment is low. It is recommended to manually review the total score and key dimensions.',
            }
        )
    if isinstance(consistency_report, dict) and consistency_report.get("failures"):
        warnings.append(
            {
                "level": "medium",
                "type": "multi_model_review_failure",
                "message": f"{len(consistency_report.get('failures', []))} rejudgment model calls failed. The current consistency conclusion should be used with caution.",
            }
        )
    return warnings


def build_report(
    run: EvaluationRun,
    paper: Paper,
    evidences: list[Evidence],
    profile: RuleProfile,
    judgements: list[AgentJudgement],
    score_items: list[ScoreItem],
    summary: dict[str, Any],
    extended_analyses: list[ExtendedAnalysis] | None = None,
    reflection_issues: list[ReflectionIssue] | None = None,
    paper_context: dict[str, Any] | None = None,
    audit_report: dict[str, Any] | None = None,
    consistency_report: dict[str, Any] | None = None,
    parse_diagnostics: dict[str, Any] | None = None,
    external_retrieval: dict[str, Any] | None = None,
    novelty_verification: dict[str, Any] | None = None,
    external_validation: dict[str, Any] | None = None,
    evidence_repository: dict[str, Any] | None = None,
) -> dict[str, Any]:
    public_summary = public_score_summary(summary)
    evidence_map = {item.evidence_id: item for item in evidences}
    judgement_map = {item.indicator_id: item for item in judgements}
    rules = profile.rules_by_id
    details = []
    for score in score_items:
        judgement = judgement_map[score.indicator_id]
        rule = rules[score.indicator_id]
        details.append(
            {
                "rule_id": rule.rule_id,
                "indicator_name": rule.indicator_name,
                "dimension_id": rule.dimension_id,
                "dimension_name": rule.dimension_name,
                "raw_score": score.raw_score,
                "weighted_score": score.weighted_score,
                "weight": score.weight,
                "role": rule.role,
                "judgement": judgement.judgement.value,
                "confidence": judgement.confidence,
                "reason": judgement.reason,
                "needs_review": judgement.needs_review,
                "review_flags": judgement.review_flags,
                "model_version": judgement.model_version,
                "out_of_scope": score.out_of_scope,
                "rationale": rule.rationale,
                "threshold": rule.threshold,
                "mostly_threshold": rule.mostly_threshold,
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "section": evidence_map[evidence_id].section,
                        "page": evidence_map[evidence_id].page,
                        "source_type": evidence_map[evidence_id].source_type,
                        "text": evidence_map[evidence_id].text[:600],
                    }
                    for evidence_id in judgement.evidence_ids
                    if evidence_id in evidence_map
                ],
                "candidate_evidence": [
                    {
                        "evidence_id": evidence_id,
                        "section": evidence_map[evidence_id].section,
                        "page": evidence_map[evidence_id].page,
                        "source_type": evidence_map[evidence_id].source_type,
                        "text": evidence_map[evidence_id].text[:300],
                    }
                    for evidence_id in judgement.candidate_evidence_ids
                    if evidence_id in evidence_map
                ],
                "rule_source": rule.source_reference,
            }
        )
    read_pages_by_dimension: dict[str, list[int]] = {}
    for item in details:
        # page=0 for external literature blocks and innovative verification blocks: they are not "read text pages" and leaving them in the reading coverage would be misleading.
        pages = sorted({ev["page"] for ev in item.get("candidate_evidence", []) if ev["page"] > 0})
        read_pages_by_dimension.setdefault(item["dimension_id"], [])
        read_pages_by_dimension[item["dimension_id"]].extend(pages)
    read_pages_by_dimension = {
        dimension_id: sorted(set(pages))
        for dimension_id, pages in read_pages_by_dimension.items()
    }
    quality_warnings = build_quality_warnings(
        summary,
        details,
        reflection_issues or [],
        paper_context,
        audit_report,
        consistency_report,
        parse_diagnostics,
    )
    return {
        "run": {**to_dict(run), "trace": public_trace(run.trace)},
        "paper": to_dict(paper),
        "rule_profile": {
            "rule_profile_id": profile.rule_profile_id,
            "version": profile.version,
            "status": profile.status,
        },
        "summary": public_summary,
        "paper_context": paper_context or {},
        "parse_diagnostics": parse_diagnostics or {},
        "audit_report": audit_report or {},
        "multi_model_consistency": consistency_report or {},
        "quality_warnings": quality_warnings,
        "read_coverage": {
            "candidate_pages_by_dimension": read_pages_by_dimension,
            "note": 'candidate_pages_by_dimension indicates the candidate evidence page numbers actually received and read by the expert Agent of each dimension; evidence indicates the evidence finally adopted as the basis for scoring.',
        },
        "details": details,
        "extended_analyses": to_dict(extended_analyses or []),
        "external_retrieval": external_retrieval or {},
        "novelty_verification": novelty_verification or {},
        "external_validation": external_validation or {},
        "evidence_repository": evidence_repository or {},
        "reflection_issues": to_dict(reflection_issues or []),
    }


def save_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = report["run"]["run_id"]
    json_path = out / f"{run_id}_report.json"
    md_path = out / f"{run_id}_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    paper = report["paper"]
    paper_context = report.get("paper_context", {})
    paper_profile = paper_context.get("paper_profile", {}) if isinstance(paper_context, dict) else {}
    total = summary["total_score"]
    total_text = "NA" if total is None else f"{total:.2f}"
    conservative = summary.get("conservative_total_score")
    optimistic = summary.get("optimistic_total_score")
    consistency = report.get("multi_model_consistency", {})
    consensus_score = consistency.get("consensus_score") if isinstance(consistency, dict) else None
    dimension_lines = []
    for dimension_id, item in (summary.get("dimension_scores") or {}).items():
        value = item.get("normalized_score")
        text = "NA" if value is None else f"{value:.2f}/{item['max_score']:.1f}"
        baseline_rate = item.get("baseline_rate")
        bonus_rate = item.get("bonus_rate")
        detail = ""
        if baseline_rate is not None or bonus_rate is not None:
            baseline_text = "NA" if baseline_rate is None else f"{baseline_rate:.0%}"
            bonus_text = "NA" if bonus_rate is None else f"{bonus_rate:.0%}"
            detail = f"(Benchmark achieved {baseline_text}｜Extra points {bonus_text})"
        dimension_lines.append(f"- {dimension_id}. {item['dimension_name']}：{text}{detail}")
    baseline_text = "NA" if summary.get("baseline_rate") is None else f"{summary['baseline_rate']:.0%}"
    bonus_text = "NA" if summary.get("bonus_rate") is None else f"{summary['bonus_rate']:.0%}"
    lines = [
        f"# Literature evaluation report",
        "",
        f"- Paper: {paper['metadata'].get('title')}",
        f"- Run ID：{report['run']['run_id']}",
        f"- Rule version: {report['rule_profile']['rule_profile_id']} / {report['rule_profile']['version']}",
        f"- Actual scoring model: {report['run'].get('model_versions', {}).get('core_agents', 'NA')}",
        f"- Scoring mode: {summary.get('scoring_mode', 'Standard quality baseline')}",
        f"- Scoring scale: {summary.get('scale_version', 'NA')} ({summary.get('scale_anchor', 'NA')})",
        f"- Quality score: {total_text}",
        (
            f"- Baseline score: {summary.get('baseline_score', 0):.2f}/{summary.get('baseline_pool', 0):.0f}"
            f"(Benchmark achievement rate {baseline_text}; passing the benchmark item means full score)"
        ),
        (
            f"- Bonus points: {summary.get('bonus_score', 0):.2f}/{summary.get('bonus_pool', 0):.0f}"
            f"(bonus points {bonus_text}; only satisfied gets full bonus points)"
        ),
        f"- Multi-model consensus calibration score: {'NA' if consensus_score is None else f'{float(consensus_score):.2f}'}",
        f"- Multi-model consistency: {consistency.get('agreement_level', 'NA') if isinstance(consistency, dict) else 'NA'}",
        f"- Quality grade: {summary.get('quality_grade', 'NA')}",
        f"- Top publication potential: {summary.get('top_journal_potential', 'NA')}",
        f"- NA by 0 (strict lower bound): {'NA' if conservative is None else f'{conservative:.2f}'}",
        f"- NA by part (optimistic upper bound): {'NA' if optimistic is None else f'{optimistic:.2f}'}",
        f"- Rating completeness: {summary['scoring_completeness']:.2%}",
        (
            f"- Applicable construct weight: {summary.get('scope_weight', 'NA')}"
            f"; Weight outside the applicable scope: {summary.get('out_of_scope_weight', 0)}"
            f"; Construct weight not evaluated: {summary.get('missing_construct_weight', 0)} (loose caliber is not included in the denominator)"
        ),
        "",
        '## Dimension score',
        *dimension_lines,
    ]
    if paper_profile:
        theoretical_move = paper_profile.get("theoretical_move", {})
        method_design = paper_profile.get("method_design", {})
        evidence_structure = paper_profile.get("evidence_structure", {})
        lines.extend(
            [
                "",
                '## Full-text paper portrait',
                f"- Paper Type: Controlled Type {paper_profile.get('paper_type', 'NA')}"
                f"(Source: {paper_profile.get('paper_type_source', 'NA')}; Description: {paper_profile.get('paper_type_label') or 'NA'})",
                f"- Core issue: {paper_profile.get('central_problem') or paper_profile.get('research_question') or 'NA'}",
                f"- Research Gap: {paper_profile.get('research_gap', 'NA')}",
                f"- Core proposition: {paper_profile.get('core_claim', 'NA')}",
                f"- Main contribution: {paper_profile.get('main_contribution', 'NA')}",
            ]
        )
        if isinstance(theoretical_move, dict):
            lines.append(
                '- Theoretical action:'
                f"{theoretical_move.get('type', 'NA')} / {theoretical_move.get('strength', 'NA')}；"
                f"{theoretical_move.get('description', 'NA')}"
            )
        if isinstance(method_design, dict):
            lines.append(
                '- Method design:'
                f"{method_design.get('design_type', 'NA')}；"
                f"{method_design.get('sample_data', 'NA')}；"
                f"{method_design.get('identification_or_validity_strategy', 'NA')}"
            )
        if isinstance(evidence_structure, dict):
            for label, key in [
                ('Main evidence', "main_evidence"),
                ('Robustness/Verification', "robustness_or_validation"),
                ('Cross-chapter closed loop', "cross_section_support"),
            ]:
                values = evidence_structure.get(key, [])
                if isinstance(values, list) and values:
                    lines.append(f"- {label}：{'；'.join(str(item) for item in values[:4])}")
        pages = paper_profile.get("most_important_pages", [])
        if isinstance(pages, list) and pages:
            lines.append(f"- Key page number: {', '.join(str(page) for page in pages[:8])}")
        signals = paper_profile.get("possible_top_journal_signals", [])
        if isinstance(signals, list) and signals:
            lines.append(f"- Top issue internal signal: {'；'.join(str(item) for item in signals[:5])}")
        weaknesses = paper_profile.get("blocking_weaknesses", [])
        if isinstance(weaknesses, list) and weaknesses:
            lines.append(f"- Blocking Weakness: {'；'.join(str(item) for item in weaknesses[:5])}")
    gate_reasons = summary.get("top_journal_gate_reasons") or []
    if gate_reasons:
        lines.extend(["", '## The reason why the top journal transition failed'])
        for reason in gate_reasons:
            lines.append(f"- {reason}")
    if report.get("quality_warnings"):
        lines.extend(["", '## Quality Warning'])
        for warning in report["quality_warnings"]:
            lines.append(f"- [{warning['level']}] {warning['message']}")
    if isinstance(consistency, dict) and consistency.get("enabled"):
        lines.extend(["", '## Multi-model consistency review'])
        lines.append(f"- Main model score: {consistency.get('primary_score', 'NA')}")
        lines.append(f"- Consistency calibration score: {consistency.get('consensus_score', 'NA')}")
        lines.append(f"- Conformance level: {consistency.get('agreement_level', 'NA')}")
        for review in consistency.get("reviews", []):
            lines.append(
                f"- {review.get('model')}: Recommended {review.get('recommended_score')};"
                f"Consistency {review.get('agreement_level')}; {review.get('reason')}"
            )
    lines.extend(["", '## Scoring Criteria Description'])
    lines.extend(
        [
            f"- Mechanism: {summary.get('scoring_mode', 'NA')}.",
            '- The benchmark item (role=baseline) determines whether the publication level is sufficient. **If you pass the test, you will get full benchmark points**:'
            'mostly (passed with no obvious gaps) and partial (passed with pointed gaps) have the same score.'
            'only has points for weak (not reaching publication standards) and not_satisfied (serious defects); there is no satisfied for the benchmark items.',
            '- Bonus points (role=bonus) are used to determine whether "it exceeds the qualification": satisfied gets full bonus points, mostly gets half,'
            'partial gets a quarter, weak / not_satisfied has no bonus points (no extra points are deducted).',
        ]
    )
    dimension_scores = summary.get("dimension_scores") or {}
    for dimension_id, dimension in dimension_scores.items():
        dimension_rules = [item for item in report["details"] if item["dimension_id"] == dimension_id]
        if not dimension_rules:
            continue
        max_score = dimension.get("max_score")
        title = dimension.get("dimension_name", dimension_id)
        suffix = f"({max_score:.0f} points)" if isinstance(max_score, (int, float)) else ""
        lines.append(f"### {dimension_id}. {title}{suffix}")
        for item in dimension_rules:
            threshold = item.get("threshold") or ""
            mostly = item.get("mostly_threshold") or ""
            flag = '[No scoring observation item]' if not item.get("weight") else ""
            if item.get("role") == "bonus":
                detail = f"Bonus points｜satisfied: {threshold}"
            else:
                detail = f"Benchmark item｜Qualification threshold: {threshold}"
            if mostly:
                detail += f"；mostly：{mostly}"
            lines.append(f"- {item['indicator_name']}：{detail}{flag}")
    lines.extend(["", '## Evidence chain details'])
    for item in report["details"]:
        score = "NA" if item["weighted_score"] is None else f"{item['weighted_score']:.2f}"
        if item.get("out_of_scope"):
            lines.extend(
                [
                    "",
                    f"### {item['rule_id']} {item['indicator_name']}",
                    '- Out of scope: not rated, not included in the total score and completeness of the score.',
                    f"- Description: {item['reason']}",
                ]
            )
            continue
        lines.extend(
            [
                "",
                f"### {item['rule_id']} {item['indicator_name']}",
                f"- Quality score: {score}; Judgment: {item['judgement']}; Confidence: {item['confidence']:.2f}",
                f"- Reason: {item['reason']}",
            ]
        )
        for flag in item.get("review_flags") or []:
            lines.append(f"- Review tip: {flag}")
        lines.append(f"- Source rule: {'；'.join(item['rule_source'])}")
        for ev in item["evidence"]:
            if ev.get("source_type") == "external":
                lines.append(f"- External evidence {ev['evidence_id']}, {ev['section']}: {ev['text']}")
            else:
                lines.append(f"- Evidence {ev['evidence_id']}, page {ev['page']}, {ev['section']}: {ev['text']}")
    if report.get("extended_analyses"):
        lines.extend(["", '## Innovation and relevance'])
        for analysis in report["extended_analyses"]:
            lines.append(f"- {analysis['analysis_type']}: {analysis['summary']}; Confidence: {analysis['confidence']:.2f}")
            for match in analysis["matches"][:5]:
                lines.append(f"- {match['title']}, {match['facet']}, similarity {match['score']:.3f}")
    external = report.get("external_retrieval") or {}
    if external.get("related_works") or external.get("failures") or external.get("error"):
        lines.extend(["", '## External related work'])
        if external.get("error"):
            lines.append(f"- External retrieval failed, skipped: {external['error']}")
        else:
            status = external.get("status") or {}
            lines.append(
                f"- Search source: {'/'.join(str(name) for name in status.get('providers') or []) or 'NA'};"
                f"claim search formula {len(external.get('queries') or [])};"
                f"candidate {external.get('candidates', 0)}; reason for discontinuation: {external.get('stop_reason')}"
            )
            for query in external.get("queries") or []:
                lines.append(f"- Search formula ({query.get('rule_id')}): {query.get('query')}")
        for work in external.get("related_works") or []:
            meta = "，".join(str(piece) for piece in [work.get("year"), work.get("venue")] if piece)
            lines.append(
                f"- **{work.get('title')}** ({meta or 'year/journal unavailable'}; other references {work.get('citation_count')};"
                f"rearrangement points {work.get('rerank_score')})"
            )
            summary = work.get("summary")
            if isinstance(summary, dict):
                for key, label in (("method", 'method'), ("findings", 'found'), ("experiments", 'Experiment'), ("similar", 'similar'), ("different", 'different')):
                    lines.append(f"  - {label}：{summary.get(key, '')}")
            elif work.get("summary_error"):
                lines.append(f"- Digest generation failed: {work['summary_error']}")
        for failure in (external.get("failures") or [])[:5]:
            lines.append(f"- Retrieval failed ({failure.get('claim_id')}): {failure.get('error')}")
    lines.extend(render_external_evidence_sections(report))
    # The reliability segment is rendered by reliability_diagnosis (avoids the reporting layer relying on the internal structure of reliability)
    reliability = report.get("reliability_report")
    if reliability:
        from .reliability_diagnosis import render_reliability_markdown

        lines.extend(render_reliability_markdown(reliability))
    if report.get("reflection_issues"):
        lines.extend(["", '## Reflection and review suggestions'])
        for issue in report["reflection_issues"]:
            lines.append(f"- [{issue['severity']}] {issue['issue_type']}: {issue['message']} Recommendation: {issue['suggested_action']}")
    return "\n".join(lines) + "\n"


def render_external_evidence_sections(report: dict[str, Any]) -> list[str]:
    'Four stages of external evidence: claim extraction → multi-channel recall → innovative verification → external evidence verification + reverse traceability.\n\n    These paragraphs are the implementation of "external conclusions can be traced, inspected, and reproducible":\n    You can go back from Score along details → judgment → claim → XEV-* → external papers (including DOI and metadata verification results).'
    lines: list[str] = []
    external = report.get("external_retrieval") or {}
    verification = report.get("novelty_verification") or {}
    validation = report.get("external_validation") or {}
    repository = report.get("evidence_repository") or {}

    claims = external.get("claims") or []
    if claims:
        lines.extend(["", '## Claim extraction (external verification object)'])
        for claim in claims:
            lines.append(
                f"- {claim.get('claim_id')}｜{claim.get('claim_type_label') or claim.get('claim_type')}"
                f"| Internal evidence {claim.get('source_evidence_id') or 'NA'}: {str(claim.get('claim_text') or '')[:160]}"
            )

    routes = external.get("routes_used") or []
    if routes or external.get("candidate_literature_pool"):
        lines.extend(["", '## Multi-way recall and candidate pool'])
        lines.append(
            f"- Enable recall path: {'、'.join(str(route) for route in routes) or 'NA'};"
            f"candidate pool {external.get('pool_size', 0)} articles; time leakage filtering eliminates {external.get('time_filtered', 0)} articles;"
            f"Delete the article itself {external.get('self_excluded', 0)}; target paper year {external.get('publication_year') or 'NA'}"
        )
        for query in (external.get("queries") or [])[:12]:
            lines.append(
                f"- Search formula [{query.get('route')}/{query.get('facet')}] ({query.get('claim_id')}, source {query.get('provider')}): {query.get('query')}"
            )
        graph = external.get("citation_graph") or {}
        if graph:
            lines.append(f"- Citation map: Target paper \"{graph.get('target') or 'NA'}\", retrieve the {graph.get('groups', 0)} group citation relationship")
        for work in (external.get("external_evidence_set") or [])[:20]:
            features = work.get("rank_features") or {}
            why = "，".join(
                f"{RANK_FEATURE_LABEL_MAP.get(key, key)}={float(value or 0):.2f}"
                for key, value in sorted(features.items(), key=lambda row: -float(row[1] or 0))[:4]
            )
            lines.append(
                f"- {work.get('evidence_id')}｜**{work.get('title')}** ({work.get('year') or 'year unavailable'};"
                f"{work.get('venue') or 'venue unavailable'};other citations {work.get('citation_count')};DOI {work.get('doi') or 'NA'};"
                f"metadata {work.get('metadata_check')}); recall={','.join(str(item) for item in (work.get('routes') or []))};"
                f"claim={','.join(str(item) for item in (work.get('claim_ids') or []))}; correlation {work.get('relevance_score')} ({why})"
            )
            lines.append(f"- Why it was recalled: Search formula \"{str(work.get('source_query') or '')[:100]}\"")
            if work.get("metadata_notes"):
                lines.append(f"- Metadata note: {'；'.join(str(item) for item in work['metadata_notes'])}")

    verdicts = verification.get("verdicts") or []
    if verdicts:
        lines.extend(["", '## Innovative Verification Agent Conclusion'])
        lines.append(
            f"- Availability: {'Unavailable（' + str(verification.get('reason')) + '）' if verification.get('unavailable') else 'Available'};"
            f"Total {len(verdicts)} claim judgments"
        )
        for index, row in enumerate(verdicts, start=1):
            # NOV-<i> is consistent with the number of novelty_agent.verdict_blocks(): this paragraph can be checked back when NOV-* is quoted in the judgment.
            lines.append(
                f"- NOV-{index} | {row.get('claim_id')} | {row.get('judgment')} (Confidence {row.get('confidence')},"
                f"needs to be reviewed {row.get('needs_review')}): {row.get('reason')}"
            )
            lines.append(
                f"- Author's assertion (internal evidence {row.get('internal_evidence_id')}): {str(row.get('author_claim') or '')[:160]}"
            )
            lines.append(
                f"- citing external evidence: {', '.join(str(item) for item in (row.get('external_evidence_ids') or [])) or 'None'}"
            )
            if row.get("supporting_papers"):
                lines.append(f"  - supporting：{'；'.join(str(item) for item in row['supporting_papers'])}")
            if row.get("contradicting_papers"):
                lines.append(f"  - contradicting：{'；'.join(str(item) for item in row['contradicting_papers'])}")

    findings = validation.get("findings") or []
    if validation.get("checked"):
        lines.extend(["", '## External evidence verification (Harness)'])
        lines.append(
            f"- external papers {validation.get('external_paper_count', 0)}; check items {validation.get('finding_count', 0)}"
            f"(high severity {validation.get('high_severity_count', 0)} entry); target year {validation.get('target_year') or 'NA'}"
        )
        for finding in findings[:30]:
            lines.append(f"- [{finding.get('severity')}] {finding.get('code')}：{finding.get('message')}")

    trace_rows = (repository.get("traceability") or {}).get("claims") or []
    if trace_rows:
        lines.extend(["", '## Reverse traceability (Score → Judgment → Claim → External Evidence → External Paper)'])
        lines.append(
            f"- Internal evidence warehouse {repository.get('counts', {}).get('internal', 0)} items;"
            f"external literature warehouse {repository.get('counts', {}).get('external', 0)} items (the two warehouses are stored separately)"
        )
        for row in trace_rows:
            lines.append(
                f"- {row.get('claim_id')} → Internal Evidence {row.get('internal_evidence_id')}"
                f"({'Present' if row.get('internal_evidence_found') else 'Missing'}) → External evidence"
                f"{', '.join(str(item) for item in (row.get('external_evidence_ids') or [])) or 'None'} → DOI"
                f"{', '.join(str(item) for item in (row.get('external_dois') or [])) or 'None'}"
            )
    return lines


# The report layer copies a feature name mapping to avoid importing the search module in order to change a line of copy (the report may be loaded separately).
RANK_FEATURE_LABEL_MAP = {
    "topic_relevance": 'Topic Relevance',
    "construct_overlap": 'Core construct overlap',
    "theory_relevance": 'Theoretical relevance',
    "mechanism_relevance": 'Mechanism dependency',
    "context_relevance": 'Study situational relevance',
    "semantic_similarity": 'Semantic similarity',
    "citation_relationship": "citation relationship",
    "publication_year": "publication year",
}

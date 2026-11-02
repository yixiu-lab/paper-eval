'Reliability layer: calibration sensitivity + uncertainty source diagnosis + final report assembly.\n\nThree pieces of content:\n\n1. **Calibration Uncertainty**: Do not re-run the Agent, change the same batch of judgments to several sets of gear mappings and recalculate the points.\n   Look at the total score, dimensions, paper ordering, and whether Top-K can be read. Purely reuse `scorer.aggregate_scores`,\n   Therefore, it is consistent with the online scoring caliber bit by bit (not writing another set of algorithms).\n2. **Reliability Diagnosis**: Locate the sources of uncertainty and give suggestions based on the four judgment rules given by the requirements.\n3. **Reliability Report**: Assemble into `{run_variance, evidence_variance, judgment_variance,\n   calibration_uncertainty, overall_warning, needs_review}`, **no total score**.'

from __future__ import annotations

from typing import Any, Iterable

from .models import AgentJudgement, Judgement
from .reliability import (
    DIMENSION_IDS,
    RELIABILITY_VERSION,
    ReliabilityConfig,
    RunTrace,
    batch_ranking_stability,
    evidence_variance,
    judgment_variance,
    run_variance,
    spearman,
    stability_scores,
)
from .rules import RuleProfile
from .scorer import aggregate_scores, profile_score_mapping, score_judgements_with_scheme

# --------------------------------------------------------------------------------------
# calibration scheme: three sets of alternative mapping
# --------------------------------------------------------------------------------------
#
# The three sets of solutions required are "satisfied / partial / not_satisfied" level three, but the scale of this project is level five.
# (the benchmark items are also divided into mostly/weak). Here, it is expanded into five levels according to the caliber of the project itself, maintaining the relative relationship between the plans:
#   Scheme A (literal) = the "steepest" scheme in demand, partial is significantly lower than mostly (0.7 vs 0.85)
#   Scheme B (low) = overall reduction, amplifying gear differences
#   Scheme C (high) = overall increase, narrowing the gear difference
# The three sets all swing around the current mapping of the rule base, and are used to answer "if you change to a set of equally reasonable rulers, will the conclusion be reversed?"
CALIBRATION_SCHEMES: dict[str, dict[str, Any]] = {
    "scheme_a_literal": {
        "label": 'Scheme A: satisfied 1.0 / partial 0.7 / not_satisfied 0.35 (expanded to fifth gear according to demand)',
        "baseline": {"mostly": 0.85, "partial": 0.7, "weak": 0.45, "not_satisfied": 0.35},
        "bonus": {"satisfied": 1.0, "mostly": 0.7, "partial": 0.35, "weak": 0.0, "not_satisfied": 0.35},
    },
    "scheme_b_low": {
        "label": 'Scheme B: satisfied 1.0 / partial 0.6 / not_satisfied 0.25 (overall depressed)',
        "baseline": {"mostly": 0.8, "partial": 0.6, "weak": 0.4, "not_satisfied": 0.25},
        "bonus": {"satisfied": 1.0, "mostly": 0.6, "partial": 0.3, "weak": 0.0, "not_satisfied": 0.25},
    },
    "scheme_c_high": {
        "label": 'Scheme C: satisfied 1.0 / partial 0.8 / not_satisfied 0.5 (overall elevation)',
        "baseline": {"mostly": 0.9, "partial": 0.8, "weak": 0.6, "not_satisfied": 0.5},
        "bonus": {"satisfied": 1.0, "mostly": 0.8, "partial": 0.5, "weak": 0.0, "not_satisfied": 0.5},
    },
}


def _judgements_from_trace(trace: RunTrace, profile: RuleProfile | None = None) -> list[AgentJudgement]:
    'RunTrace → AgentJudgement (only moves fields, no judgment is made).\n\n    The report may come from another rule base version (offline playback is used for this), and a rule_id that does not exist in the current rule base appears.\n    Skip directly: Calibration recalculation must select weights and roles according to the rule base. Rules that cannot be calculated cannot be randomly assigned weights.'
    known = set(profile.rules_by_id) if profile is not None else None
    known_judgements = {item.value for item in Judgement}
    records = []
    for record in trace.rule_records:
        if record.judgement not in known_judgements:
            continue
        if known is not None and record.rule_id not in known:
            continue
        records.append(
            AgentJudgement(
                task_id=trace.run_id,
                agent_id="reliability-replay",
                indicator_id=record.rule_id,
                judgement=Judgement(record.judgement),
                evidence_ids=list(record.evidence_ids),
                reason='Reliability Layer For recalculation: the judgment and reasons are taken as they are from the record of the run.',
                confidence=record.confidence,
                needs_review=record.needs_review,
                agent_version="reliability-replay",
                model_version=trace.model_version,
                review_flags=list(record.review_flags),
            )
        )
    return records


def score_under_scheme(
    profile: RuleProfile,
    trace: RunTrace,
    scheme: dict[str, Any] | None,
    out_of_scope_ids: set[str] | None = None,
) -> dict[str, Any]:
    'The score of this paper under a certain set of mapping. `scheme=None` means using the rule base’s own mapping (baseline).'
    judgements = _judgements_from_trace(trace, profile)
    if scheme is None:
        items = score_judgements_with_scheme(
            profile, judgements, {"baseline": profile_score_mapping(profile)}, out_of_scope_ids
        )
    else:
        items = score_judgements_with_scheme(profile, judgements, scheme, out_of_scope_ids)
    return aggregate_scores(profile, items)


def _dimension_delta(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, str]:
    worst_value, worst_dimension = 0.0, ""
    for dimension_id in DIMENSION_IDS:
        left_score = ((left.get("dimension_scores") or {}).get(dimension_id) or {}).get("normalized_score")
        right_score = ((right.get("dimension_scores") or {}).get(dimension_id) or {}).get("normalized_score")
        if left_score is None or right_score is None:
            continue
        delta = abs(float(left_score) - float(right_score))
        if delta > worst_value:
            worst_value, worst_dimension = delta, dimension_id
    return worst_value, worst_dimension


def calibration_uncertainty(
    profile: RuleProfile,
    paper_traces: dict[str, list[RunTrace]],
    config: ReliabilityConfig,
    focus_paper_id: str | None = None,
    out_of_scope_by_paper: dict[str, set[str]] | None = None,
    schemes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    'Change mapping to recalculate scores: total score changes, dimension changes, ranking correlation, Top-K overlap.\n\n    Each paper takes the judgment record of the "most recent run" to recalculate (the same batch of judgments is considered clean);\n    Ranking comparison uses the score vector of common papers within the batch.'
    active_schemes = schemes or CALIBRATION_SCHEMES
    latest: dict[str, RunTrace] = {}
    for paper_id, traces in paper_traces.items():
        usable = [trace for trace in traces if trace.rule_records]
        if usable:
            latest[paper_id] = sorted(usable, key=lambda item: (item.started_at or "", item.run_id))[-1]
    if not latest:
        return {
            "checked": False,
            "robustness": "insufficient_data",
            "note": 'There is no judgment record available for recalculation.',
            "schemes": [],
        }

    per_paper: dict[str, dict[str, Any]] = {}
    for paper_id, trace in latest.items():
        baseline = score_under_scheme(
            profile, trace, None, (out_of_scope_by_paper or {}).get(paper_id)
        )
        per_paper[paper_id] = {"baseline": baseline, "schemes": {}}
        for name, scheme in active_schemes.items():
            per_paper[paper_id]["schemes"][name] = score_under_scheme(
                profile, trace, scheme, (out_of_scope_by_paper or {}).get(paper_id)
            )

    focus_paper = focus_paper_id or (sorted(latest)[0] if latest else "")
    focus = per_paper.get(focus_paper) or next(iter(per_paper.values()))
    focus_baseline = focus["baseline"]

    scheme_reports: list[dict[str, Any]] = []
    max_score_delta = 0.0
    max_score_delta_ratio = 0.0
    max_dimension_delta = 0.0
    max_dimension_id = ""
    ranking_correlations: list[float] = []
    topk_overlaps: list[float] = []
    sensitive_schemes: list[str] = []
    scale = float(profile.scoring_scale or 100.0)

    paper_ids = sorted(per_paper)
    for name, scheme in active_schemes.items():
        deltas: dict[str, float] = {}
        for paper_id in paper_ids:
            base = per_paper[paper_id]["baseline"].get("total_score")
            other = per_paper[paper_id]["schemes"][name].get("total_score")
            if base is None or other is None:
                continue
            deltas[paper_id] = float(other) - float(base)
        if not deltas:
            continue
        worst_paper = max(deltas, key=lambda key: abs(deltas[key]))
        abs_delta = abs(deltas[worst_paper])
        ratio = abs_delta / scale if scale else 0.0

        dimension_delta, dimension_id = _dimension_delta(
            focus_baseline, focus["schemes"][name]
        )

        correlation: float | None = None
        if len(paper_ids) >= 2:
            base_vector = [
                per_paper[paper_id]["baseline"]["total_score"] for paper_id in paper_ids
            ]
            other_vector = [
                per_paper[paper_id]["schemes"][name]["total_score"] for paper_id in paper_ids
            ]
            if all(value is not None for value in base_vector + other_vector):
                correlation = spearman([float(v) for v in base_vector], [float(v) for v in other_vector])
        if correlation is not None:
            ranking_correlations.append(correlation)

        overlap: float | None = None
        top_k = max(1, min(config.top_k, len(paper_ids)))
        if len(paper_ids) >= 2:
            base_top = {
                paper_id
                for paper_id in sorted(
                    paper_ids, key=lambda key: -(per_paper[key]["baseline"]["total_score"] or 0.0)
                )[:top_k]
            }
            other_top = {
                paper_id
                for paper_id in sorted(
                    paper_ids, key=lambda key: -(per_paper[key]["schemes"][name]["total_score"] or 0.0)
                )[:top_k]
            }
            overlap = len(base_top & other_top) / len(base_top | other_top) if base_top | other_top else None
            if overlap is not None:
                topk_overlaps.append(overlap)

        scheme_sensitive = (
            ratio > config.calibration_score_delta_threshold
            or dimension_delta > config.dimension_variance_threshold
            or (correlation is not None and correlation < config.calibration_ranking_correlation_threshold)
            or (overlap is not None and overlap < config.calibration_topk_overlap_threshold)
        )
        if scheme_sensitive:
            sensitive_schemes.append(name)
        max_score_delta = max(max_score_delta, abs_delta)
        max_score_delta_ratio = max(max_score_delta_ratio, ratio)
        if dimension_delta > max_dimension_delta:
            max_dimension_delta, max_dimension_id = dimension_delta, dimension_id
        scheme_reports.append(
            {
                "scheme": name,
                "label": scheme.get("label") or name,
                "score_delta": round(max(deltas.values(), key=abs), 4),
                "max_abs_score_delta": round(abs_delta, 4),
                "score_delta_ratio": round(ratio, 4),
                "worst_paper_id": worst_paper,
                "dimension_delta": round(dimension_delta, 4),
                "dimension_id": dimension_id,
                "ranking_correlation": round(correlation, 4) if correlation is not None else None,
                "top_k_overlap": round(overlap, 4) if overlap is not None else None,
                "sensitive": scheme_sensitive,
            }
        )

    mean_correlation = (
        sum(ranking_correlations) / len(ranking_correlations) if ranking_correlations else None
    )
    mean_overlap = sum(topk_overlaps) / len(topk_overlaps) if topk_overlaps else None
    ranking_checked = mean_correlation is not None
    sensitive = bool(sensitive_schemes)
    robustness = (
        "low" if sensitive
        else "high" if max_score_delta_ratio <= config.calibration_score_delta_threshold * 0.5
        else "medium"
    )
    flags = ["calibration_sensitive"] if sensitive else []
    note = ""
    if not ranking_checked:
        note = 'This time only the total score and dimension score were compared: Ranking relevance requires at least 2 papers from the same batch.'
    return {
        "checked": True,
        "scheme_count": len(scheme_reports),
        "focus_paper_id": focus_paper,
        "baseline_scale": scale,
        "baseline_mapping": {
            key: (None if value is None else round(float(value), 4))
            for key, value in (profile_score_mapping(profile) or {}).items()
        },
        "max_abs_score_delta": round(max_score_delta, 4),
        "max_score_delta_ratio": round(max_score_delta_ratio, 4),
        "max_dimension_delta": round(max_dimension_delta, 4),
        "max_dimension_id": max_dimension_id,
        "ranking_correlation_checked": ranking_checked,
        "mean_ranking_correlation": round(mean_correlation, 4) if mean_correlation is not None else None,
        "min_ranking_correlation": round(min(ranking_correlations), 4) if ranking_correlations else None,
        "mean_top_k_overlap": round(mean_overlap, 4) if mean_overlap is not None else None,
        # alias, convenient for directly matching the field names in the requirements
        "score_difference": round(max_score_delta, 4),
        "ranking_correlation": round(mean_correlation, 4) if mean_correlation is not None else None,
        "robustness": robustness,
        "thresholds": {
            "score_delta_ratio": config.calibration_score_delta_threshold,
            "ranking_correlation": config.calibration_ranking_correlation_threshold,
            "top_k_overlap": config.calibration_topk_overlap_threshold,
            "dimension_delta": config.dimension_variance_threshold,
        },
        "sensitive_schemes": sensitive_schemes,
        "schemes": scheme_reports,
        "per_paper_totals": {
            paper_id: {
                "baseline_total": per_paper[paper_id]["baseline"].get("total_score"),
                "scheme_totals": {
                    name: score.get("total_score")
                    for name, score in per_paper[paper_id]["schemes"].items()
                },
            }
            for paper_id in paper_ids
        },
        "flags": flags,
        "note": note,
    }


# --------------------------------------------------------------------------------------
# Diagnosis: Sources of Uncertainty + Recommendations
# --------------------------------------------------------------------------------------

_RECOMMENDATIONS = {
    "retrieval_instability": 'Increase search candidates (increase evidence recall top-k / increase max_evidence_per_rule), and fix randomness in the search layer; do not directly accept a single result when the evidence is unstable.',
    "agent_reasoning": 'There are repeated judgments on the same rule. It is recommended to improve the sampling stability (temperature=0 / multiple votes to obtain the majority), and to add clearer gear anchor points to the high-frequency flipping rules.',
    "calibration_sensitive": 'Ranking is sensitive to bin mapping, it is recommended to recalibrate score_mapping / weighting and report score intervals across schemes instead of single point scores.',
    "evaluation_reliable": 'All four indicators are within the threshold: the evaluation results are stable under repeated runs and mapping changes, and can be used as a single result.',
    "insufficient_data": 'Insufficient number of comparable runs: the same paper must be rerun at least 2 times (same rule base version) to judge run stability.',
}


def diagnose_reliability(
    run_result: dict[str, Any],
    evidence_result: dict[str, Any],
    judgment_result: dict[str, Any],
    calibration_result: dict[str, Any],
    config: ReliabilityConfig,
) -> dict[str, Any]:
    'locates the sources of uncertainty according to the four types of variances, and gives suggestions and a review list.'
    run_flags = set(run_result.get("flags") or [])
    evidence_flags = set(evidence_result.get("flags") or [])
    judgment_flags = set(judgment_result.get("flags") or [])
    calibration_flags = set(calibration_result.get("flags") or [])
    run_high = bool(run_flags & {"high_variance", "grade_flip", "dimension_variance", "ranking_change"})
    evidence_high = bool(evidence_flags)
    judgment_high = bool(judgment_flags)
    calibration_high = bool(calibration_flags)
    insufficient = str(run_result.get("stability")) == "insufficient_data" and not calibration_high

    if run_high and evidence_high:
        source = "retrieval_instability"
        reason = 'The total score fluctuates greatly between runs, and the evidence recall in different runs is inconsistent: the uncertainty mainly comes from the retrieval layer.'
    elif run_high and not evidence_high and judgment_high:
        source = "agent_reasoning"
        reason = 'Evidence recall is stable but judgment gears are repetitive: uncertainty comes from Agent reasoning, not retrieval.'
    elif evidence_high and judgment_high and not run_high:
        source = "retrieval_and_reasoning"
        reason = 'Both evidence and judgment are unstable (the total score has not fluctuated significantly): uncertainty comes from both retrieval and reasoning, which is a hidden danger.'
    elif not run_high and calibration_high:
        source = "calibration_sensitive"
        reason = 'The results of repeated runs are stable, but the ranking/score changes significantly after changing the gear mapping: the uncertainty comes from the scoring rules themselves.'
    elif insufficient:
        source = "insufficient_data"
        reason = 'There are not enough comparable runs to judge run stability; calibration sensitivity conclusions are still available.'
    else:
        source = "evaluation_reliable"
        reason = 'All four indicators are within the threshold.'

    warnings: list[str] = []
    if run_flags & {"high_variance"}:
        warnings.append(
            f"Total score fluctuates between runs {run_result.get('range')} score (threshold {run_result.get('threshold')}):"
            'Scores from a single run should not be taken as conclusive.'
        )
    if "grade_flip" in run_flags:
        warnings.append(
            'The quality level/top publication potential judgment given by different runs is inconsistent:'
            + "、".join(str(item) for item in (run_result.get("grade_flip_detail") or []))
        )
    for dimension_id in run_result.get("dimension_flagged") or []:
        detail = (run_result.get("dimension_stability") or {}).get(dimension_id) or {}
        warnings.append(
            f"dimension {dimension_id} fluctuates across runs by {detail.get('range')} points (threshold {config.dimension_variance_threshold})."
        )
    if "ranking_change" in run_flags:
        warnings.append(
            f"Paper sorting within a batch is unstable between runs (Spearman {run_result.get('ranking_change_detail') or ''})."
        )
    if evidence_flags:
        warnings.append(
            f"Evidence recall is unstable: average intersection {evidence_result.get('evidence_overlap')}"
            f"(threshold {config.evidence_overlap_threshold}),"
            f"The most unstable rule {', '.join((evidence_result.get('unstable_rule_ids') or [])[:6]) or 'NA'}."
        )
    if judgment_flags:
        warnings.append(
            f"Judgment gear inconsistency: average consistency rate {judgment_result.get('mean_agreement')}"
            f"(threshold {config.judgment_agreement_threshold}),"
            f"The most unstable rule {', '.join((judgment_result.get('unstable_rule_ids') or [])[:6]) or 'NA'}."
        )
    if judgment_result.get("reasoning_only_rule_ids"):
        warnings.append(
            'The evidence is stable but the judgment is reversed (source_of_uncertainty=agent_reasoning):'
            f"{', '.join(judgment_result['reasoning_only_rule_ids'][:6])}。"
        )
    if judgment_result.get("retrieval_and_reasoning_rule_ids"):
        warnings.append(
            'The evidence and judgment are unstable at the same time (source_of_uncertainty=retrieval_and_reasoning):'
            f"{', '.join(judgment_result['retrieval_and_reasoning_rule_ids'][:6])}。"
        )
    if calibration_flags:
        warnings.append(
            f"Sensitive to gear mapping: maximum total score change {calibration_result.get('max_abs_score_delta')} points"
            f"(relative to {calibration_result.get('max_score_delta_ratio')}),"
            f"Scheme {', '.join(calibration_result.get('sensitive_schemes') or [])} The order/score change exceeds the threshold."
        )
    if calibration_result.get("checked") and not calibration_result.get("ranking_correlation_checked"):
        warnings.append(str(calibration_result.get("note") or 'This time only the total score and dimension score were compared: Ranking relevance requires at least 2 papers from the same batch.'))
    elif not calibration_result.get("checked"):
        warnings.append('There is no judgment record that can be recalculated this time, and the calibration sensitivity has not been tested (this conclusion is missing, which does not mean that the rules are robust).')

    needs_review: list[str] = []
    for rule_id in judgment_result.get("unstable_rule_ids") or []:
        if rule_id not in needs_review:
            needs_review.append(rule_id)
    for rule_id in evidence_result.get("unstable_rule_ids") or []:
        if rule_id not in needs_review:
            needs_review.append(rule_id)

    return {
        "main_uncertainty_source": source,
        "reason": reason,
        "recommendation": _RECOMMENDATIONS.get(source, _RECOMMENDATIONS["evaluation_reliable"]),
        "warning_count": len(warnings),
        "warnings": warnings,
        "needs_review": needs_review,
        "components": {
            "run_variance_high": run_high,
            "evidence_variance_high": evidence_high,
            "judgment_variance_high": judgment_high,
            "calibration_uncertainty_high": calibration_high,
        },
    }


# --------------------------------------------------------------------------------------
# Final Report
# --------------------------------------------------------------------------------------

def build_reliability_report(
    profile: RuleProfile,
    traces: Iterable[RunTrace],
    config: ReliabilityConfig,
    *,
    focus_paper_id: str | None = None,
    out_of_scope_by_paper: dict[str, set[str]] | None = None,
    schemes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    'Assembly Evaluation Reliability Report.\n\n    Report structure (alignment requirements):\n    `{run_variance, evidence_variance, judgment_variance, calibration_uncertainty,\n      overall_warning, needs_review}`, plus four 0-1 sub-items `reliability_report` and `diagnosis`.\n    **No total score field**: the four sub-items are independent.'
    trace_list = list(traces)
    grouped = {}
    for trace in trace_list:
        grouped.setdefault(trace.paper_id, []).append(trace)

    focus_id = focus_paper_id or (sorted(grouped)[0] if grouped else "")
    focus_traces = grouped.get(focus_id, [])
    run_result = run_variance(focus_traces, config)
    ranking = batch_ranking_stability(grouped, config)
    if ranking.get("checked"):
        run_result = {
            **run_result,
            "ranking_change": ranking.get("ranking_change"),
            "ranking_change_detail": ranking,
        }
        if ranking.get("ranking_change") and "ranking_change" not in (run_result.get("flags") or []):
            run_result["flags"] = [*(run_result.get("flags") or []), "ranking_change"]
    else:
        run_result = {**run_result, "ranking_change": None, "ranking_change_detail": ranking}

    evidence_result = evidence_variance(focus_traces, config)
    judgment_result = judgment_variance(focus_traces, config, evidence_result)
    calibration_result = calibration_uncertainty(
        profile, grouped, config, focus_paper_id=focus_id, out_of_scope_by_paper=out_of_scope_by_paper, schemes=schemes
    )
    diagnosis = diagnose_reliability(run_result, evidence_result, judgment_result, calibration_result, config)
    scores = stability_scores(run_result, evidence_result, judgment_result, calibration_result)

    overall_warning = list(diagnosis["warnings"])
    if not trace_list:
        overall_warning.append('No analyzable running records were found: Reliability Layer did not generate a conclusion this time.')

    return {
        "reliability_version": RELIABILITY_VERSION,
        "focus_paper_id": focus_id,
        "analyzed_run_count": len(trace_list),
        "analyzed_paper_count": len(grouped),
        "run_variance": run_result,
        "evidence_variance": evidence_result,
        "judgment_variance": judgment_result,
        "calibration_uncertainty": calibration_result,
        "overall_warning": overall_warning,
        "needs_review": diagnosis["needs_review"],
        "reliability_report": {
            "run_stability": scores["run_stability"],
            "evidence_stability": scores["evidence_stability"],
            "judgment_stability": scores["judgment_stability"],
            "calibration_robustness": scores["calibration_robustness"],
            "note": scores["note"],
        },
        "diagnosis": diagnosis,
        "thresholds": {
            "run_variance_threshold": config.run_variance_threshold,
            "evidence_overlap_threshold": config.evidence_overlap_threshold,
            "judgment_agreement_threshold": config.judgment_agreement_threshold,
            "min_runs": config.min_runs,
        },
        "runs": [
            {
                "run_id": trace.run_id,
                "paper_id": trace.paper_id,
                "total_score": trace.total_score,
                "started_at": trace.started_at,
                "rule_profile": trace.rule_profile,
                "model_version": trace.model_version,
                "source_path": trace.source_path,
            }
            for trace in trace_list
        ],
    }


def attach_reliability_to_report(report: dict[str, Any], reliability: dict[str, Any] | None) -> dict[str, Any]:
    'Link the Reliability Report to the existing evaluation report (add a new key and leave no existing keys unchanged).\nThe "Evaluation Reliability" section in the'
    if reliability is None:
        return report
    return {**report, "reliability_report": reliability}


def render_reliability_markdown(reliability: dict[str, Any]) -> list[str]:
    'report: four sub-items + warning + review required + audit chain.'
    if not reliability:
        return []
    lines: list[str] = ["", '## Evaluation Reliability']
    scores = reliability.get("reliability_report") or {}
    lines.append(
        '- Operation stability {run}｜Evidence stability {ev}｜Judgment stability {jud}｜Parameter robustness {cal} (the four sub-items are independent and not combined into the total score)'.format(
            run=_score_text(scores.get("run_stability")),
            ev=_score_text(scores.get("evidence_stability")),
            jud=_score_text(scores.get("judgment_stability")),
            cal=_score_text(scores.get("calibration_robustness")),
        )
    )
    lines.append(
        f"- Analysis scope: {reliability.get('analyzed_paper_count')} papers / {reliability.get('analyzed_run_count')} runs;"
        f"Focus Paper {reliability.get('focus_paper_id')}"
    )
    run = reliability.get("run_variance") or {}
    if run.get("mean_score") is not None and run.get("scored_run_count", 0) >= 2:
        lines.append(
            f"- Total score: mean {run.get('mean_score')}, standard deviation {run.get('std')},"
            f"Extremely poor {run.get('range')} (threshold {run.get('threshold')}), interval {run.get('min_score')}–{run.get('max_score')}"
        )
        for dimension_id, detail in (run.get("dimension_stability") or {}).items():
            lines.append(
                f"- Dimension {dimension_id}: mean {detail.get('mean')}, range {detail.get('range')},"
                f"Stability {detail.get('stability')}"
            )
    else:
        lines.append(f"- Operational stability: {run.get('note') or 'Insufficient data'}")
    ranking = run.get("ranking_change_detail") or {}
    if ranking.get("checked"):
        lines.append(
            f"- Batch sorting: {ranking.get('batch_count')} batches × {ranking.get('paper_count')} papers,"
            f"average Spearman {ranking.get('mean_spearman')} (minimum {ranking.get('min_spearman')}),"
            f"Top-{ranking.get('top_k')} overlap {ranking.get('top_k_overlap')}, stability {ranking.get('stability')}"
        )
    evidence = reliability.get("evidence_variance") or {}
    if evidence.get("evidence_overlap") is not None:
        lines.append(
            f"- Evidence recall: average intersection {evidence.get('evidence_overlap')} (threshold {evidence.get('threshold')}),"
            f"covers variation {evidence.get('coverage_variance')}, stability {evidence.get('stability')}"
        )
    judgment = reliability.get("judgment_variance") or {}
    if judgment.get("mean_agreement") is not None:
        lines.append(
            f"- Judgment consistency rate: average {judgment.get('mean_agreement')} (threshold {judgment.get('threshold')}),"
            f"Stability {judgment.get('stability')}; source of uncertainty {judgment.get('source_of_uncertainty') or 'NA'}"
        )
    for item in (judgment.get("per_rule") or [])[:8]:
        lines.append(
            f"- {item.get('rule_id')}: Consistency rate {item.get('agreement')},"
            f"distribution {item.get('distribution')}, entropy {item.get('entropy')}"
            + (f", source {item['source_of_uncertainty']}" if item.get("source_of_uncertainty") else "")
        )
    calibration = reliability.get("calibration_uncertainty") or {}
    if calibration.get("checked"):
        lines.append(
            f"- Calibration sensitivity: {calibration.get('scheme_count')} set of alternative mappings,"
            f"Maximum total score change {calibration.get('max_abs_score_delta')} points"
            f"(relative to {calibration.get('max_score_delta_ratio')}),"
            f"average sorting related {calibration.get('mean_ranking_correlation')},"
            f"Top-K coincidence {calibration.get('mean_top_k_overlap')}, robustness {calibration.get('robustness')}"
        )
        for scheme in calibration.get("schemes") or []:
            lines.append(
                f"- {scheme.get('scheme')}: ΔTotal Score {scheme.get('score_delta')}"
                f"(relative to {scheme.get('score_delta_ratio')}), Δ dimension {scheme.get('dimension_delta')}"
                f"({scheme.get('dimension_id')}), sort related {scheme.get('ranking_correlation')}"
                f", Top-K overlap {scheme.get('top_k_overlap')}"
                + ('; **Exceeding threshold**' if scheme.get("sensitive") else "")
            )
    diagnosis = reliability.get("diagnosis") or {}
    lines.append("")
    lines.append(
        f"**Uncertainty source: {diagnosis.get('main_uncertainty_source')}** —— {diagnosis.get('reason')}"
    )
    lines.append(f"**Suggestion:**{diagnosis.get('recommendation')}")
    if reliability.get("overall_warning"):
        lines.append("")
        lines.append('**WARNING**')
        for warning in reliability["overall_warning"]:
            lines.append(f"- {warning}")
    if reliability.get("needs_review"):
        lines.append("")
        lines.append('**Requires manual review:**' + "、".join(str(item) for item in reliability["needs_review"]))
    lines.extend(["", '### Reliability audit chain (Score → Judgment → Evidence → Reliability Diagnosis)'])
    for run in (reliability.get("runs") or [])[:12]:
        lines.append(
            f"- {run.get('run_id')} (papers {run.get('paper_id')}, {run.get('rule_profile')},"
            f"Total score {run.get('total_score')}) → See the judgment and evidence of the operation report;"
            f"Source {run.get('source_path')}"
        )
    return lines


def _score_text(value: Any) -> str:
    if value is None:
        return 'Insufficient data\nThere are risks in full text analysis of'
    return f"{float(value):.2f}"

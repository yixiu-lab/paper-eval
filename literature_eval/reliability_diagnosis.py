"""Reliability 层的判定与报告：校准敏感性 + 不确定性来源诊断 + 最终报告组装。

三块内容：

1. **Calibration Uncertainty**：不重跑 Agent，把同一批 judgment 换几套档位映射重新算分，
   看总分、维度、论文排序、Top-K 会不会翻。纯粹复用 `scorer.aggregate_scores`，
   所以与线上算分口径逐位一致（不是另写一套算法）。
2. **Reliability Diagnosis**：按需求给的四条判定规则定位不确定性来源并给建议。
3. **Reliability Report**：组装成 `{run_variance, evidence_variance, judgment_variance,
   calibration_uncertainty, overall_warning, needs_review}`，**不出总分**。
"""

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
# 校准方案：三套替代 mapping
# --------------------------------------------------------------------------------------
#
# 需求给的三套方案是「satisfied / partial / not_satisfied」三级，但本项目的标尺是五档
# （基准项还分 mostly / weak）。这里按项目自己的口径展开成五档，保持方案之间的相对关系：
#   Scheme A（literal）= 需求里最"陡"的那套，partial 明显低于 mostly（0.7 vs 0.85）
#   Scheme B（low）    = 整体压低，放大档位差异
#   Scheme C（high）   = 整体抬高，缩小档位差异
# 三套都围绕规则库当前 mapping 摆动，用来回答"换一套同样合理的标尺，结论会不会翻"。
CALIBRATION_SCHEMES: dict[str, dict[str, Any]] = {
    "scheme_a_literal": {
        "label": "Scheme A：satisfied 1.0 / partial 0.7 / not_satisfied 0.35（按需求口径展开到五档）",
        "baseline": {"mostly": 0.85, "partial": 0.7, "weak": 0.45, "not_satisfied": 0.35},
        "bonus": {"satisfied": 1.0, "mostly": 0.7, "partial": 0.35, "weak": 0.0, "not_satisfied": 0.35},
    },
    "scheme_b_low": {
        "label": "Scheme B：satisfied 1.0 / partial 0.6 / not_satisfied 0.25（整体压低）",
        "baseline": {"mostly": 0.8, "partial": 0.6, "weak": 0.4, "not_satisfied": 0.25},
        "bonus": {"satisfied": 1.0, "mostly": 0.6, "partial": 0.3, "weak": 0.0, "not_satisfied": 0.25},
    },
    "scheme_c_high": {
        "label": "Scheme C：satisfied 1.0 / partial 0.8 / not_satisfied 0.5（整体抬高）",
        "baseline": {"mostly": 0.9, "partial": 0.8, "weak": 0.6, "not_satisfied": 0.5},
        "bonus": {"satisfied": 1.0, "mostly": 0.8, "partial": 0.5, "weak": 0.0, "not_satisfied": 0.5},
    },
}


def _judgements_from_trace(trace: RunTrace, profile: RuleProfile | None = None) -> list[AgentJudgement]:
    """RunTrace → AgentJudgement（只搬字段，不做任何判断）。

    报告可能来自别的规则库版本（离线回放就是干这个的），出现当前规则库没有的 rule_id 时
    直接跳过：校准重算必须按规则库取权重与角色，算不了的规则不能瞎赋权重。
    """
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
                reason="Reliability Layer 重算用：判断与理由原样取自该次运行的记录。",
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
    """某一套 mapping 下这篇论文的分数。`scheme=None` 表示用规则库自己的 mapping（基准）。"""
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
    """换 mapping 重算分：总分变化、维度变化、排序相关、Top-K 重合。

    每个论文取"最近一次运行"的 judgment 记录来重算（同一批判断才算得干净）；
    排序比较用批次内共有论文的分数向量。
    """
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
            "note": "没有可用于重算的判断记录。",
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
        note = "本次只比较了总分与维度分：排序相关性需要同一批次至少 2 篇论文。"
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
        # 别名，方便直接对上需求里的字段名
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
# 诊断：不确定性来源 + 建议
# --------------------------------------------------------------------------------------

_RECOMMENDATIONS = {
    "retrieval_instability": "增加检索候选（提高 evidence recall top-k / 加大 max_evidence_per_rule），并在检索层固定随机性；证据不稳时不要直接采信单次结果。",
    "agent_reasoning": "同一条规则上判断反复，建议提高采样稳定性（temperature=0 / 多次投票取多数），并对高频翻转的规则补充更明确的档位锚点。",
    "calibration_sensitive": "排序对档位映射敏感，建议重新校准 score_mapping / 权重，并报告跨方案的分数区间而不是单点分数。",
    "evaluation_reliable": "四类指标均在阈值内：本次评价结果在重复运行与换映射下都稳定，可按单次结果使用。",
    "insufficient_data": "可比较的运行次数不足：至少重跑同一篇论文 2 次（同一规则库版本）才能判断运行稳定性。",
}


def diagnose_reliability(
    run_result: dict[str, Any],
    evidence_result: dict[str, Any],
    judgment_result: dict[str, Any],
    calibration_result: dict[str, Any],
    config: ReliabilityConfig,
) -> dict[str, Any]:
    """按四类方差定位不确定性来源，给出建议与需复核清单。"""
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
        reason = "运行间总分波动大，且不同运行的证据召回不一致：不确定性主要来自检索层。"
    elif run_high and not evidence_high and judgment_high:
        source = "agent_reasoning"
        reason = "证据召回稳定而判断档位反复：不确定性来自 Agent 推理，不是检索。"
    elif evidence_high and judgment_high and not run_high:
        source = "retrieval_and_reasoning"
        reason = "证据与判断都不稳（总分尚未明显波动）：不确定性同时来自检索与推理，属于隐患。"
    elif not run_high and calibration_high:
        source = "calibration_sensitive"
        reason = "重复运行结果稳定，但换一套档位映射后排序/分数明显变化：不确定性来自评分规则本身。"
    elif insufficient:
        source = "insufficient_data"
        reason = "可比较的运行次数不足，无法判断运行稳定性；校准敏感性结论仍然可用。"
    else:
        source = "evaluation_reliable"
        reason = "四类指标均在阈值内。"

    warnings: list[str] = []
    if run_flags & {"high_variance"}:
        warnings.append(
            f"总分在不同运行间波动 {run_result.get('range')} 分（阈值 {run_result.get('threshold')}）："
            "单次运行的分数不应作为定论。"
        )
    if "grade_flip" in run_flags:
        warnings.append(
            "不同运行给出的质量等级/顶刊潜力判定不一致："
            + "、".join(str(item) for item in (run_result.get("grade_flip_detail") or []))
        )
    for dimension_id in run_result.get("dimension_flagged") or []:
        detail = (run_result.get("dimension_stability") or {}).get(dimension_id) or {}
        warnings.append(
            f"维度 {dimension_id} 跨运行波动 {detail.get('range')} 分（阈值 {config.dimension_variance_threshold}）。"
        )
    if "ranking_change" in run_flags:
        warnings.append(
            f"批次内论文排序在不同运行间不稳定（Spearman {run_result.get('ranking_change_detail') or ''}）。"
        )
    if evidence_flags:
        warnings.append(
            f"证据召回不稳定：平均交集 {evidence_result.get('evidence_overlap')}"
            f"（阈值 {config.evidence_overlap_threshold}），"
            f"最不稳的规则 {', '.join((evidence_result.get('unstable_rule_ids') or [])[:6]) or 'NA'}。"
        )
    if judgment_flags:
        warnings.append(
            f"判断档位不一致：平均一致率 {judgment_result.get('mean_agreement')}"
            f"（阈值 {config.judgment_agreement_threshold}），"
            f"最不稳的规则 {', '.join((judgment_result.get('unstable_rule_ids') or [])[:6]) or 'NA'}。"
        )
    if judgment_result.get("reasoning_only_rule_ids"):
        warnings.append(
            "证据稳定但判断翻转（source_of_uncertainty=agent_reasoning）："
            f"{', '.join(judgment_result['reasoning_only_rule_ids'][:6])}。"
        )
    if judgment_result.get("retrieval_and_reasoning_rule_ids"):
        warnings.append(
            "证据与判断同时不稳（source_of_uncertainty=retrieval_and_reasoning）："
            f"{', '.join(judgment_result['retrieval_and_reasoning_rule_ids'][:6])}。"
        )
    if calibration_flags:
        warnings.append(
            f"对档位映射敏感：最大总分变化 {calibration_result.get('max_abs_score_delta')} 分"
            f"（相对 {calibration_result.get('max_score_delta_ratio')}），"
            f"方案 {', '.join(calibration_result.get('sensitive_schemes') or [])} 下排序/分数变化超阈值。"
        )
    if calibration_result.get("checked") and not calibration_result.get("ranking_correlation_checked"):
        warnings.append(str(calibration_result.get("note") or "本次只比较了总分与维度分：排序相关性需要同一批次至少 2 篇论文。"))
    elif not calibration_result.get("checked"):
        warnings.append("本次没有可重算的判断记录，校准敏感性未检测（该结论缺失，不代表规则稳健）。")

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
# 最终报告
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
    """组装 Evaluation Reliability Report。

    报告结构（对齐需求）：
    `{run_variance, evidence_variance, judgment_variance, calibration_uncertainty,
      overall_warning, needs_review}`，另附 `reliability_report` 四个 0-1 分项与 `diagnosis`。
    **没有总分字段**：四个分项各自独立。
    """
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
        overall_warning.append("没有找到可分析的运行记录：Reliability Layer 本次未产生结论。")

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
    """把 Reliability Report 挂到既有评价报告上（新增一个键，不动任何已有键）。"""
    if reliability is None:
        return report
    return {**report, "reliability_report": reliability}


def render_reliability_markdown(reliability: dict[str, Any]) -> list[str]:
    """报告里的「评价可靠性」段：四个分项 + 警告 + 需复核 + 审计链。"""
    if not reliability:
        return []
    lines: list[str] = ["", "## 评价可靠性（Evaluation Reliability）"]
    scores = reliability.get("reliability_report") or {}
    lines.append(
        "- 运行稳定性 {run}｜证据稳定性 {ev}｜判断稳定性 {jud}｜参数稳健性 {cal}（四个分项独立，不合成总分）".format(
            run=_score_text(scores.get("run_stability")),
            ev=_score_text(scores.get("evidence_stability")),
            jud=_score_text(scores.get("judgment_stability")),
            cal=_score_text(scores.get("calibration_robustness")),
        )
    )
    lines.append(
        f"- 分析范围：{reliability.get('analyzed_paper_count')} 篇论文 / {reliability.get('analyzed_run_count')} 次运行；"
        f"焦点论文 {reliability.get('focus_paper_id')}"
    )
    run = reliability.get("run_variance") or {}
    if run.get("mean_score") is not None and run.get("scored_run_count", 0) >= 2:
        lines.append(
            f"- 总分：均值 {run.get('mean_score')}，标准差 {run.get('std')}，"
            f"极差 {run.get('range')}（阈值 {run.get('threshold')}），区间 {run.get('min_score')}–{run.get('max_score')}"
        )
        for dimension_id, detail in (run.get("dimension_stability") or {}).items():
            lines.append(
                f"  - 维度 {dimension_id}：均值 {detail.get('mean')}，极差 {detail.get('range')}，"
                f"稳定性 {detail.get('stability')}"
            )
    else:
        lines.append(f"- 运行稳定性：{run.get('note') or '数据不足'}")
    ranking = run.get("ranking_change_detail") or {}
    if ranking.get("checked"):
        lines.append(
            f"- 批次排序：{ranking.get('batch_count')} 个批次 × {ranking.get('paper_count')} 篇论文，"
            f"平均 Spearman {ranking.get('mean_spearman')}（最低 {ranking.get('min_spearman')}），"
            f"Top-{ranking.get('top_k')} 重合 {ranking.get('top_k_overlap')}，稳定性 {ranking.get('stability')}"
        )
    evidence = reliability.get("evidence_variance") or {}
    if evidence.get("evidence_overlap") is not None:
        lines.append(
            f"- 证据召回：平均交集 {evidence.get('evidence_overlap')}（阈值 {evidence.get('threshold')}），"
            f"覆盖变异 {evidence.get('coverage_variance')}，稳定性 {evidence.get('stability')}"
        )
    judgment = reliability.get("judgment_variance") or {}
    if judgment.get("mean_agreement") is not None:
        lines.append(
            f"- 判断一致率：平均 {judgment.get('mean_agreement')}（阈值 {judgment.get('threshold')}），"
            f"稳定性 {judgment.get('stability')}；不确定性来源 {judgment.get('source_of_uncertainty') or 'NA'}"
        )
    for item in (judgment.get("per_rule") or [])[:8]:
        lines.append(
            f"  - {item.get('rule_id')}：一致率 {item.get('agreement')}，"
            f"分布 {item.get('distribution')}，熵 {item.get('entropy')}"
            + (f"，来源 {item['source_of_uncertainty']}" if item.get("source_of_uncertainty") else "")
        )
    calibration = reliability.get("calibration_uncertainty") or {}
    if calibration.get("checked"):
        lines.append(
            f"- 校准敏感性：{calibration.get('scheme_count')} 套备选 mapping，"
            f"最大总分变化 {calibration.get('max_abs_score_delta')} 分"
            f"（相对 {calibration.get('max_score_delta_ratio')}），"
            f"平均排序相关 {calibration.get('mean_ranking_correlation')}，"
            f"Top-K 重合 {calibration.get('mean_top_k_overlap')}，稳健性 {calibration.get('robustness')}"
        )
        for scheme in calibration.get("schemes") or []:
            lines.append(
                f"  - {scheme.get('scheme')}：Δ总分 {scheme.get('score_delta')}"
                f"（相对 {scheme.get('score_delta_ratio')}），Δ维度 {scheme.get('dimension_delta')}"
                f"（{scheme.get('dimension_id')}），排序相关 {scheme.get('ranking_correlation')}"
                f"，Top-K 重合 {scheme.get('top_k_overlap')}"
                + ("；**超阈值**" if scheme.get("sensitive") else "")
            )
    diagnosis = reliability.get("diagnosis") or {}
    lines.append("")
    lines.append(
        f"**不确定性来源：{diagnosis.get('main_uncertainty_source')}** —— {diagnosis.get('reason')}"
    )
    lines.append(f"**建议：**{diagnosis.get('recommendation')}")
    if reliability.get("overall_warning"):
        lines.append("")
        lines.append("**警告**")
        for warning in reliability["overall_warning"]:
            lines.append(f"- {warning}")
    if reliability.get("needs_review"):
        lines.append("")
        lines.append("**需人工复核：**" + "、".join(str(item) for item in reliability["needs_review"]))
    lines.extend(["", "### 可靠性审计链（Score → Judgment → Evidence → Reliability Diagnosis）"])
    for run in (reliability.get("runs") or [])[:12]:
        lines.append(
            f"- {run.get('run_id')}（论文 {run.get('paper_id')}，{run.get('rule_profile')}，"
            f"总分 {run.get('total_score')}）→ 见该次运行报告的判断与证据；"
            f"来源 {run.get('source_path')}"
        )
    return lines


def _score_text(value: Any) -> str:
    if value is None:
        return "数据不足"
    return f"{float(value):.2f}"

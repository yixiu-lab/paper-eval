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
                    "PDF 全文解析存在风险："
                    f"{parse_warning_message(warning_items)}。建议先检查 PDF 是否为扫描版、乱码版或章节结构异常。"
                ),
            }
        )
    completeness = float(summary.get("scoring_completeness") or 0)
    if completeness < 0.8:
        warnings.append(
            {
                "level": "high",
                "type": "low_completeness",
                "message": f"评分完整度为 {completeness:.0%}，NA 较多，总分需要结合人工复核。",
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
                    "message": f"{dimension_id}. {dimension_name} 有 {count} 条规则无法判断，建议优先复核该维度证据。",
                }
            )
    def _has_flag(item: dict[str, Any], *markers: str) -> bool:
        flags = item.get("review_flags") or []
        return any(marker in str(flag) for flag in flags for marker in markers)

    gate_adjusted = [
        item
        for item in details
        if not item.get("out_of_scope") and _has_flag(item, "证据强度提示", "高分证据提示", "证据门控提示")
    ]
    if gate_adjusted:
        warnings.append(
            {
                "level": "medium",
                "type": "high_score_gate",
                "message": (
                    f"{len(gate_adjusted)} 条判断在证据强度或理由一致性上存在疑点。"
                    "本地检查只做标记、未改动模型判断，需人工复核是否维持原档位。"
                ),
            }
        )
    weak_partial = [
        item
        for item in details
        if not item.get("out_of_scope") and item.get("judgement") == "partial" and _has_flag(item, "证据强度")
    ]
    if weak_partial:
        warnings.append(
            {
                "level": "medium",
                "type": "weak_partial_evidence",
                "message": f"{len(weak_partial)} 条部分完成判断只有一般性提及证据，建议专家优先复核。",
            }
        )
    weak_reasons = [
        item
        for item in details
        if not item.get("out_of_scope") and _has_flag(item, "理由过短或过泛")
    ]
    if weak_reasons:
        warnings.append(
            {
                "level": "medium",
                "type": "weak_agent_reason",
                "message": f"{len(weak_reasons)} 条评分理由过短或过泛，建议复核对应指标解释。",
            }
        )
    review_profile = (paper_context or {}).get("review_profile", {}) if isinstance(paper_context, dict) else {}
    if isinstance(paper_context, dict) and paper_context.get("context_error"):
        warnings.append(
            {
                "level": "high",
                "type": "paper_context_fallback",
                "message": (
                    "全文理解 Agent 失败，本次评分使用本地启发式画像，缺少全文共享上下文。"
                    "规则级证据与判断仍然有效，但论文画像、章节理解和一致性判断需人工复核。"
                ),
            }
        )
    if isinstance(paper_context, dict) and (paper_context.get("paper_profile") or {}).get("paper_type_source") == "fallback":
        warnings.append(
            {
                "level": "medium",
                "type": "paper_type_fallback",
                "message": (
                    "论文类型未能由全文理解 Agent 确定，已按「混合」处理——适用全部规则，未按类型排除任何规则。"
                    "如需按类型分流（例如理论类论文不评数据样本类规则），请人工确认论文类型。"
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
                    "全文理解 Agent 返回的论文画像缺少关键字段（"
                    + "、".join(missing)
                    + "），后续专家 Agent 缺少这部分共享上下文，报告中的论文画像不可当真，建议人工复核。"
                ),
            }
        )
    if isinstance(paper_context, dict) and paper_context.get("routing_error"):
        warnings.append(
            {
                "level": "medium",
                "type": "routing_fallback",
                "message": "全文证据路由 Agent 本次不可用，系统改用本地章节索引分配候选证据；最终评分仍由专家 Agent 完成。",
            }
        )
    if isinstance(review_profile, dict) and review_profile.get("nonstandard_paradigm"):
        warnings.append(
            {
                "level": "info",
                "type": "nonstandard_paradigm",
                "message": "系统识别到非常规研究范式，评分时应保护非典型但有价值的论文。",
            }
        )
    if reflection_issues:
        warnings.append(
            {
                "level": "medium",
                "type": "reflection_issues",
                "message": f"反思模块发现 {len(reflection_issues)} 个需要复核的问题。",
            }
        )
    if isinstance(audit_report, dict) and audit_report.get("audit_status") in {"needs_review", "failed"}:
        warnings.append(
            {
                "level": "high",
                "type": "audit_agent",
                "message": f"反思审核智能体状态为 {audit_report.get('audit_status')}，建议查看审核意见。",
            }
        )
    if isinstance(consistency_report, dict) and consistency_report.get("agreement_level") == "low":
        warnings.append(
            {
                "level": "high",
                "type": "multi_model_disagreement",
                "message": "多模型复判一致性较低，建议人工复核总分和关键维度。",
            }
        )
    if isinstance(consistency_report, dict) and consistency_report.get("failures"):
        warnings.append(
            {
                "level": "medium",
                "type": "multi_model_review_failure",
                "message": f"{len(consistency_report.get('failures', []))} 个复判模型调用失败，当前一致性结论需谨慎使用。",
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
        # 外部文献块与创新性验证块的 page=0：它们不是"读过的正文页"，留在阅读覆盖里会误导。
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
            "note": "candidate_pages_by_dimension 表示各维度专家 Agent 实际收到并阅读过的候选证据页码；evidence 表示最终采纳为评分依据的证据。",
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
            detail = f"（基准达成 {baseline_text}｜加分 {bonus_text}）"
        dimension_lines.append(f"- {dimension_id}. {item['dimension_name']}：{text}{detail}")
    baseline_text = "NA" if summary.get("baseline_rate") is None else f"{summary['baseline_rate']:.0%}"
    bonus_text = "NA" if summary.get("bonus_rate") is None else f"{summary['bonus_rate']:.0%}"
    lines = [
        f"# 文献评价报告",
        "",
        f"- 论文：{paper['metadata'].get('title')}",
        f"- Run ID：{report['run']['run_id']}",
        f"- 规则版本：{report['rule_profile']['rule_profile_id']} / {report['rule_profile']['version']}",
        f"- 实际评分模型：{report['run'].get('model_versions', {}).get('core_agents', 'NA')}",
        f"- 评分模式：{summary.get('scoring_mode', '普通质量基准')}",
        f"- 评分标尺：{summary.get('scale_version', 'NA')}（{summary.get('scale_anchor', 'NA')}）",
        f"- 质量分：{total_text}",
        (
            f"- 基准分：{summary.get('baseline_score', 0):.2f}/{summary.get('baseline_pool', 0):.0f}"
            f"（基准达成率 {baseline_text}；基准项合格即满分）"
        ),
        (
            f"- 加分：{summary.get('bonus_score', 0):.2f}/{summary.get('bonus_pool', 0):.0f}"
            f"（加成分 {bonus_text}；只有 satisfied 拿满加分）"
        ),
        f"- 多模型一致性校准分：{'NA' if consensus_score is None else f'{float(consensus_score):.2f}'}",
        f"- 多模型一致性：{consistency.get('agreement_level', 'NA') if isinstance(consistency, dict) else 'NA'}",
        f"- 质量等级：{summary.get('quality_grade', 'NA')}",
        f"- 顶刊潜力：{summary.get('top_journal_potential', 'NA')}",
        f"- NA按0（严格下界）：{'NA' if conservative is None else f'{conservative:.2f}'}",
        f"- NA按部分（乐观上界）：{'NA' if optimistic is None else f'{optimistic:.2f}'}",
        f"- 评分完整度：{summary['scoring_completeness']:.2%}",
        (
            f"- 适用构念权重：{summary.get('scope_weight', 'NA')}"
            f"；适用范围外权重：{summary.get('out_of_scope_weight', 0)}"
            f"；未评到构念权重：{summary.get('missing_construct_weight', 0)}（宽松口径不计入分母）"
        ),
        "",
        "## 维度得分",
        *dimension_lines,
    ]
    if paper_profile:
        theoretical_move = paper_profile.get("theoretical_move", {})
        method_design = paper_profile.get("method_design", {})
        evidence_structure = paper_profile.get("evidence_structure", {})
        lines.extend(
            [
                "",
                "## 全文论文画像",
                f"- 论文类型：受控类型 {paper_profile.get('paper_type', 'NA')}"
                f"（来源：{paper_profile.get('paper_type_source', 'NA')}；说明：{paper_profile.get('paper_type_label') or 'NA'}）",
                f"- 核心问题：{paper_profile.get('central_problem') or paper_profile.get('research_question') or 'NA'}",
                f"- 研究缺口：{paper_profile.get('research_gap', 'NA')}",
                f"- 核心主张：{paper_profile.get('core_claim', 'NA')}",
                f"- 主要贡献：{paper_profile.get('main_contribution', 'NA')}",
            ]
        )
        if isinstance(theoretical_move, dict):
            lines.append(
                "- 理论动作："
                f"{theoretical_move.get('type', 'NA')} / {theoretical_move.get('strength', 'NA')}；"
                f"{theoretical_move.get('description', 'NA')}"
            )
        if isinstance(method_design, dict):
            lines.append(
                "- 方法设计："
                f"{method_design.get('design_type', 'NA')}；"
                f"{method_design.get('sample_data', 'NA')}；"
                f"{method_design.get('identification_or_validity_strategy', 'NA')}"
            )
        if isinstance(evidence_structure, dict):
            for label, key in [
                ("主要证据", "main_evidence"),
                ("稳健性/验证", "robustness_or_validation"),
                ("跨章节闭环", "cross_section_support"),
            ]:
                values = evidence_structure.get(key, [])
                if isinstance(values, list) and values:
                    lines.append(f"- {label}：{'；'.join(str(item) for item in values[:4])}")
        pages = paper_profile.get("most_important_pages", [])
        if isinstance(pages, list) and pages:
            lines.append(f"- 关键页码：{', '.join(str(page) for page in pages[:8])}")
        signals = paper_profile.get("possible_top_journal_signals", [])
        if isinstance(signals, list) and signals:
            lines.append(f"- 顶刊内部信号：{'；'.join(str(item) for item in signals[:5])}")
        weaknesses = paper_profile.get("blocking_weaknesses", [])
        if isinstance(weaknesses, list) and weaknesses:
            lines.append(f"- 阻断弱点：{'；'.join(str(item) for item in weaknesses[:5])}")
    gate_reasons = summary.get("top_journal_gate_reasons") or []
    if gate_reasons:
        lines.extend(["", "## 顶刊跃迁未通过原因"])
        for reason in gate_reasons:
            lines.append(f"- {reason}")
    if report.get("quality_warnings"):
        lines.extend(["", "## 质量警告"])
        for warning in report["quality_warnings"]:
            lines.append(f"- [{warning['level']}] {warning['message']}")
    if isinstance(consistency, dict) and consistency.get("enabled"):
        lines.extend(["", "## 多模型一致性复判"])
        lines.append(f"- 主模型分数：{consistency.get('primary_score', 'NA')}")
        lines.append(f"- 一致性校准分：{consistency.get('consensus_score', 'NA')}")
        lines.append(f"- 一致性等级：{consistency.get('agreement_level', 'NA')}")
        for review in consistency.get("reviews", []):
            lines.append(
                f"- {review.get('model')}：建议 {review.get('recommended_score')}；"
                f"一致性 {review.get('agreement_level')}；{review.get('reason')}"
            )
    lines.extend(["", "## 评分标准说明"])
    lines.extend(
        [
            f"- 机制：{summary.get('scoring_mode', 'NA')}。",
            "- 基准项（role=baseline）判「够不够发表水准」，**合格即拿满基准分**："
            "mostly（合格且无明显缺口）与 partial（合格但有可指出的缺口）同分，"
            "只有 weak（未达发表水准）和 not_satisfied（严重缺陷）扣分；基准项不设 satisfied。",
            "- 加分项（role=bonus）判「有没有超出合格」：satisfied 拿满加分，mostly 拿一半，"
            "partial 拿四分之一，weak / not_satisfied 没有加分（也不额外扣分）。",
        ]
    )
    dimension_scores = summary.get("dimension_scores") or {}
    for dimension_id, dimension in dimension_scores.items():
        dimension_rules = [item for item in report["details"] if item["dimension_id"] == dimension_id]
        if not dimension_rules:
            continue
        max_score = dimension.get("max_score")
        title = dimension.get("dimension_name", dimension_id)
        suffix = f"（{max_score:.0f} 分）" if isinstance(max_score, (int, float)) else ""
        lines.append(f"### {dimension_id}. {title}{suffix}")
        for item in dimension_rules:
            threshold = item.get("threshold") or ""
            mostly = item.get("mostly_threshold") or ""
            flag = "〔不计分观察项〕" if not item.get("weight") else ""
            if item.get("role") == "bonus":
                detail = f"加分项｜satisfied：{threshold}"
            else:
                detail = f"基准项｜合格门槛：{threshold}"
            if mostly:
                detail += f"；mostly：{mostly}"
            lines.append(f"- {item['indicator_name']}：{detail}{flag}")
    lines.extend(["", "## 证据链明细"])
    for item in report["details"]:
        score = "NA" if item["weighted_score"] is None else f"{item['weighted_score']:.2f}"
        if item.get("out_of_scope"):
            lines.extend(
                [
                    "",
                    f"### {item['rule_id']} {item['indicator_name']}",
                    "- 适用范围外：未评分，不计入总分与评分完整度。",
                    f"- 说明：{item['reason']}",
                ]
            )
            continue
        lines.extend(
            [
                "",
                f"### {item['rule_id']} {item['indicator_name']}",
                f"- 质量得分：{score}；判断：{item['judgement']}；置信度：{item['confidence']:.2f}",
                f"- 理由：{item['reason']}",
            ]
        )
        for flag in item.get("review_flags") or []:
            lines.append(f"- 复核提示：{flag}")
        lines.append(f"- 来源规则：{'；'.join(item['rule_source'])}")
        for ev in item["evidence"]:
            if ev.get("source_type") == "external":
                lines.append(f"- 外部证据 {ev['evidence_id']}，{ev['section']}：{ev['text']}")
            else:
                lines.append(f"- 证据 {ev['evidence_id']}，第 {ev['page']} 页，{ev['section']}：{ev['text']}")
    if report.get("extended_analyses"):
        lines.extend(["", "## 创新性与相关性"])
        for analysis in report["extended_analyses"]:
            lines.append(f"- {analysis['analysis_type']}：{analysis['summary']}；置信度：{analysis['confidence']:.2f}")
            for match in analysis["matches"][:5]:
                lines.append(f"  - {match['title']}，{match['facet']}，相似度 {match['score']:.3f}")
    external = report.get("external_retrieval") or {}
    if external.get("related_works") or external.get("failures") or external.get("error"):
        lines.extend(["", "## 外部相关工作"])
        if external.get("error"):
            lines.append(f"- 外部检索失败，已跳过：{external['error']}")
        else:
            status = external.get("status") or {}
            lines.append(
                f"- 检索源：{'/'.join(str(name) for name in status.get('providers') or []) or 'NA'}；"
                f"claim 检索式 {len(external.get('queries') or [])} 条；"
                f"候选 {external.get('candidates', 0)} 篇；停止原因：{external.get('stop_reason')}"
            )
            for query in external.get("queries") or []:
                lines.append(f"- 检索式（{query.get('rule_id')}）：{query.get('query')}")
        for work in external.get("related_works") or []:
            meta = "，".join(str(piece) for piece in [work.get("year"), work.get("venue")] if piece)
            lines.append(
                f"- **{work.get('title')}**（{meta or '年份/期刊缺失'}；他引 {work.get('citation_count')}；"
                f"重排分 {work.get('rerank_score')}）"
            )
            summary = work.get("summary")
            if isinstance(summary, dict):
                for key, label in (("method", "方法"), ("findings", "发现"), ("experiments", "实验"), ("similar", "相似"), ("different", "不同")):
                    lines.append(f"  - {label}：{summary.get(key, '')}")
            elif work.get("summary_error"):
                lines.append(f"  - 摘要生成失败：{work['summary_error']}")
        for failure in (external.get("failures") or [])[:5]:
            lines.append(f"- 检索失败（{failure.get('claim_id')}）：{failure.get('error')}")
    lines.extend(render_external_evidence_sections(report))
    # 可靠性段由 reliability_diagnosis 渲染（避免报告层依赖 reliability 的内部结构）
    reliability = report.get("reliability_report")
    if reliability:
        from .reliability_diagnosis import render_reliability_markdown

        lines.extend(render_reliability_markdown(reliability))
    if report.get("reflection_issues"):
        lines.extend(["", "## 反思与复核建议"])
        for issue in report["reflection_issues"]:
            lines.append(f"- [{issue['severity']}] {issue['issue_type']}：{issue['message']} 建议：{issue['suggested_action']}")
    return "\n".join(lines) + "\n"


def render_external_evidence_sections(report: dict[str, Any]) -> list[str]:
    """外部证据四段：claim 抽取 → 多路召回 → 创新性验证 → 外部证据校验 + 反向追溯。

    这几段是「外部的结论可追溯、可检查、可复现」的落地：
    从 Score 沿 details → 判断 → claim → XEV-* → 外部论文（含 DOI 与元数据校验结果）一路能走回去。
    """
    lines: list[str] = []
    external = report.get("external_retrieval") or {}
    verification = report.get("novelty_verification") or {}
    validation = report.get("external_validation") or {}
    repository = report.get("evidence_repository") or {}

    claims = external.get("claims") or []
    if claims:
        lines.extend(["", "## Claim 抽取（外部验证对象）"])
        for claim in claims:
            lines.append(
                f"- {claim.get('claim_id')}｜{claim.get('claim_type_label') or claim.get('claim_type')}"
                f"｜内部证据 {claim.get('source_evidence_id') or 'NA'}：{str(claim.get('claim_text') or '')[:160]}"
            )

    routes = external.get("routes_used") or []
    if routes or external.get("candidate_literature_pool"):
        lines.extend(["", "## 多路召回与候选池"])
        lines.append(
            f"- 启用召回路径：{'、'.join(str(route) for route in routes) or 'NA'}；"
            f"候选池 {external.get('pool_size', 0)} 篇；时间泄漏过滤剔除 {external.get('time_filtered', 0)} 篇；"
            f"剔除本文自身 {external.get('self_excluded', 0)} 条；目标论文年份 {external.get('publication_year') or 'NA'}"
        )
        for query in (external.get("queries") or [])[:12]:
            lines.append(
                f"- 检索式[{query.get('route')}/{query.get('facet')}]（{query.get('claim_id')}，源 {query.get('provider')}）：{query.get('query')}"
            )
        graph = external.get("citation_graph") or {}
        if graph:
            lines.append(f"- 引文图：目标论文「{graph.get('target') or 'NA'}」，取回 {graph.get('groups', 0)} 组引文关系")
        for work in (external.get("external_evidence_set") or [])[:20]:
            features = work.get("rank_features") or {}
            why = "，".join(
                f"{RANK_FEATURE_LABEL_MAP.get(key, key)}={float(value or 0):.2f}"
                for key, value in sorted(features.items(), key=lambda row: -float(row[1] or 0))[:4]
            )
            lines.append(
                f"- {work.get('evidence_id')}｜**{work.get('title')}**（{work.get('year') or '年份未知'}；"
                f"{work.get('venue') or '期刊未知'}；他引 {work.get('citation_count')}；DOI {work.get('doi') or 'NA'}；"
                f"元数据 {work.get('metadata_check')}）；召回={','.join(str(item) for item in (work.get('routes') or []))}；"
                f"claim={','.join(str(item) for item in (work.get('claim_ids') or []))}；相关度 {work.get('relevance_score')}（{why}）"
            )
            lines.append(f"  - 为什么召回它：检索式「{str(work.get('source_query') or '')[:100]}」")
            if work.get("metadata_notes"):
                lines.append(f"  - 元数据备注：{'；'.join(str(item) for item in work['metadata_notes'])}")

    verdicts = verification.get("verdicts") or []
    if verdicts:
        lines.extend(["", "## 创新性验证 Agent 结论"])
        lines.append(
            f"- 可用性：{'不可用（' + str(verification.get('reason')) + '）' if verification.get('unavailable') else '正常'}；"
            f"共 {len(verdicts)} 条 claim 判断"
        )
        for index, row in enumerate(verdicts, start=1):
            # NOV-<i> 与 novelty_agent.verdict_blocks() 的编号一致：判断里引用 NOV-* 时能回查到这一段。
            lines.append(
                f"- NOV-{index}｜{row.get('claim_id')}｜{row.get('judgment')}（置信度 {row.get('confidence')}，"
                f"需复核 {row.get('needs_review')}）：{row.get('reason')}"
            )
            lines.append(
                f"  - 作者主张（内部证据 {row.get('internal_evidence_id')}）：{str(row.get('author_claim') or '')[:160]}"
            )
            lines.append(
                f"  - 引用外部证据：{', '.join(str(item) for item in (row.get('external_evidence_ids') or [])) or '无'}"
            )
            if row.get("supporting_papers"):
                lines.append(f"  - supporting：{'；'.join(str(item) for item in row['supporting_papers'])}")
            if row.get("contradicting_papers"):
                lines.append(f"  - contradicting：{'；'.join(str(item) for item in row['contradicting_papers'])}")

    findings = validation.get("findings") or []
    if validation.get("checked"):
        lines.extend(["", "## 外部证据校验（Harness）"])
        lines.append(
            f"- 外部论文 {validation.get('external_paper_count', 0)} 篇；校验项 {validation.get('finding_count', 0)} 条"
            f"（高严重度 {validation.get('high_severity_count', 0)} 条）；目标年份 {validation.get('target_year') or 'NA'}"
        )
        for finding in findings[:30]:
            lines.append(f"- [{finding.get('severity')}] {finding.get('code')}：{finding.get('message')}")

    trace_rows = (repository.get("traceability") or {}).get("claims") or []
    if trace_rows:
        lines.extend(["", "## 反向追溯（Score → Judgment → Claim → External Evidence → External Paper）"])
        lines.append(
            f"- 内部证据仓库 {repository.get('counts', {}).get('internal', 0)} 条；"
            f"外部文献仓库 {repository.get('counts', {}).get('external', 0)} 条（两个仓库分开存储）"
        )
        for row in trace_rows:
            lines.append(
                f"- {row.get('claim_id')} → 内部证据 {row.get('internal_evidence_id')}"
                f"（{'存在' if row.get('internal_evidence_found') else '缺失'}） → 外部证据 "
                f"{', '.join(str(item) for item in (row.get('external_evidence_ids') or [])) or '无'} → DOI "
                f"{', '.join(str(item) for item in (row.get('external_dois') or [])) or '无'}"
            )
    return lines


# 报告层复制一份特征名映射，避免为了改一行文案去 import 检索模块（报告可能被单独加载）。
RANK_FEATURE_LABEL_MAP = {
    "topic_relevance": "主题相关性",
    "construct_overlap": "核心构念重合度",
    "theory_relevance": "理论相关性",
    "mechanism_relevance": "机制相关性",
    "context_relevance": "研究情境相关性",
    "semantic_similarity": "语义相似度",
    "citation_relationship": "citation relationship",
    "publication_year": "publication year",
}

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _list_text(values: Any, limit: int = 5) -> str:
    if not isinstance(values, list):
        return ""
    parts = []
    for item in values[:limit]:
        if isinstance(item, dict):
            parts.append("；".join(f"{key}: {value}" for key, value in item.items() if value))
        else:
            parts.append(str(item))
    return "；".join(parts)


def render_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    details = report["details"]
    paper_context = report.get("paper_context", {})
    paper_profile = paper_context.get("paper_profile", {}) if isinstance(paper_context, dict) else {}
    review_profile = paper_context.get("review_profile", {}) if isinstance(paper_context, dict) else {}
    section_map = paper_context.get("section_map", []) if isinstance(paper_context, dict) else []
    read_coverage = report.get("read_coverage", {})
    pages_by_dimension = read_coverage.get("candidate_pages_by_dimension", {}) if isinstance(read_coverage, dict) else {}
    dimension_routes = paper_context.get("dimension_evidence_routes", {}) if isinstance(paper_context, dict) else {}
    routing_notes = paper_context.get("routing_notes", {}) if isinstance(paper_context, dict) else {}
    audit_report = report.get("audit_report", {})
    audit_issues = audit_report.get("audit_issues", []) if isinstance(audit_report, dict) else []
    consistency = report.get("multi_model_consistency", {})
    consistency_reviews = consistency.get("reviews", []) if isinstance(consistency, dict) else []
    quality_warnings = report.get("quality_warnings", [])
    total = _fmt(summary["total_score"])
    nonstandard_notice = ""
    if isinstance(review_profile, dict) and review_profile.get("nonstandard_paradigm"):
        nonstandard_notice = (
            "<section class=\"card\"><b>非常规研究范式提示</b>"
            f"<p>{html.escape(str(review_profile.get('anti_homogenization_policy', '')))}</p>"
            f"<p>{html.escape(str(review_profile.get('rule_focus_notes', '')))}</p></section>"
        )
    profile_cards = "".join(
        f"<div class=\"metric\">{html.escape(label)}<b>{html.escape(str(value or 'NA'))}</b></div>"
        for label, value in [
            ("论文类型", f"受控类型 {paper_profile.get('paper_type')}（来源：{paper_profile.get('paper_type_source', 'NA')}；说明：{paper_profile.get('paper_type_label') or 'NA'}）"),
            ("核心问题", paper_profile.get("central_problem") or paper_profile.get("research_question")),
            ("核心主张", paper_profile.get("core_claim")),
            ("方法路线", paper_profile.get("method_type")),
        ]
    )
    theoretical_move = paper_profile.get("theoretical_move", {}) if isinstance(paper_profile, dict) else {}
    method_design = paper_profile.get("method_design", {}) if isinstance(paper_profile, dict) else {}
    evidence_structure = paper_profile.get("evidence_structure", {}) if isinstance(paper_profile, dict) else {}
    paper_profile_rows = ""
    for label, value in [
        ("研究缺口", paper_profile.get("research_gap")),
        ("主要贡献", paper_profile.get("main_contribution")),
        (
            "理论动作",
            (
                f"{theoretical_move.get('type', 'NA')} / {theoretical_move.get('strength', 'NA')}；"
                f"{theoretical_move.get('description', 'NA')}"
            )
            if isinstance(theoretical_move, dict)
            else "",
        ),
        (
            "方法设计",
            (
                f"{method_design.get('design_type', 'NA')}；{method_design.get('sample_data', 'NA')}；"
                f"{method_design.get('identification_or_validity_strategy', 'NA')}"
            )
            if isinstance(method_design, dict)
            else "",
        ),
        ("主要发现", paper_profile.get("main_findings")),
        ("边界条件", paper_profile.get("boundary_conditions")),
        (
            "证据结构",
            (
                f"主要证据：{_list_text(evidence_structure.get('main_evidence', []), 4)} "
                f"稳健性/验证：{_list_text(evidence_structure.get('robustness_or_validation', []), 4)} "
                f"跨章节闭环：{_list_text(evidence_structure.get('cross_section_support', []), 4)}"
            )
            if isinstance(evidence_structure, dict)
            else "",
        ),
        ("关键页码", ", ".join(str(page) for page in paper_profile.get("most_important_pages", [])[:8]) if isinstance(paper_profile.get("most_important_pages"), list) else ""),
        ("顶刊内部信号", _list_text(paper_profile.get("possible_top_journal_signals", []), 5)),
        ("阻断弱点", _list_text(paper_profile.get("blocking_weaknesses", []), 5)),
    ]:
        if value:
            paper_profile_rows += f"<tr><td>{html.escape(label)}</td><td>{html.escape(str(value))}</td></tr>"
    section_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(str(item.get("section", "")))}</td>
          <td>{html.escape(", ".join(str(page) for page in item.get("pages", [])))}</td>
          <td>{html.escape(str(item.get("summary", ""))[:260])}</td>
        </tr>
        """
        for item in section_map[:12]
        if isinstance(item, dict)
    )
    coverage_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(dim_id)}</td>
          <td>{html.escape(summary["dimension_scores"].get(dim_id, {}).get("dimension_name", ""))}</td>
          <td>{html.escape(", ".join(str(page) for page in pages))}</td>
        </tr>
        """
        for dim_id, pages in pages_by_dimension.items()
    )
    route_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(dim_id)}</td>
          <td>{html.escape(summary["dimension_scores"].get(dim_id, {}).get("dimension_name", ""))}</td>
          <td>{len(ids)}</td>
          <td>{html.escape(str(routing_notes.get(dim_id, "")) if isinstance(routing_notes, dict) else "")}</td>
        </tr>
        """
        for dim_id, ids in dimension_routes.items()
        if isinstance(ids, list)
    )
    dimensions = "".join(
        f"""
        <tr>
          <td>{html.escape(dim_id)}</td>
          <td>{html.escape(item["dimension_name"])}</td>
          <td>{_fmt(item["normalized_score"])}</td>
          <td>{_fmt(item["max_score"])}</td>
          <td>{_fmt(item["available_weight"])}</td>
        </tr>
        """
        for dim_id, item in summary["dimension_scores"].items()
    )
    detail_cards = "".join(
        f"""
        <section class="card">
          <h3>{html.escape(item["rule_id"])} · {html.escape(item["indicator_name"])}</h3>
          <p><b>质量得分</b> {_fmt(item["weighted_score"])} <b>判断</b> {html.escape(item["judgement"])} <b>置信度</b> {_fmt(item["confidence"])}</p>
          <p><b>依据</b> {html.escape(item["reason"])}</p>
          {''.join(f'<p class="flag"><b>复核提示</b> {html.escape(str(flag))}</p>' for flag in item.get("review_flags") or [])}
          <details>
            <summary>证据链</summary>
            {''.join(f'<p class="evidence">第 {ev["page"]} 页 · {html.escape(ev["section"])}<br>{html.escape(ev["text"])}</p>' for ev in item["evidence"])}
          </details>
          <details>
            <summary>Agent 实际阅读片段</summary>
            {''.join(f'<p class="evidence">第 {ev["page"]} 页 · {html.escape(ev["section"])}<br>{html.escape(ev["text"])}</p>' for ev in item.get("candidate_evidence", [])[:6])}
          </details>
        </section>
        """
        for item in details
    )
    analyses = "".join(
        f"""
        <section class="card">
          <h3>{html.escape(item["analysis_type"])}</h3>
          <p>{html.escape(item["summary"])}</p>
          <ul>{''.join(f'<li>{html.escape(match["title"])} · {html.escape(match["facet"])} · {_fmt(match["score"], 3)}</li>' for match in item["matches"][:5])}</ul>
        </section>
        """
        for item in report.get("extended_analyses", [])
    )
    issues = "".join(
        f"<li><b>{html.escape(item['severity'])}</b> {html.escape(item['issue_type'])}: {html.escape(item['message'])} {html.escape(item['suggested_action'])}</li>"
        for item in report.get("reflection_issues", [])
    )
    audit_issue_rows = "".join(
        f"<li><b>{html.escape(str(item.get('severity', '')))}</b> {html.escape(str(item.get('issue_type', '')))}: {html.escape(str(item.get('message', '')))} {html.escape(str(item.get('suggested_action', '')))}</li>"
        for item in audit_issues
        if isinstance(item, dict)
    )
    consistency_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(str(item.get("model", "")))}</td>
          <td>{html.escape(str(item.get("recommended_score", "NA")))}</td>
          <td>{html.escape(str(item.get("agreement_level", "NA")))}</td>
          <td>{html.escape(str(item.get("reason", "")))}</td>
        </tr>
        """
        for item in consistency_reviews
        if isinstance(item, dict)
    )
    warning_rows = "".join(
        f"<li><b>{html.escape(str(item.get('level', '')))}</b> {html.escape(str(item.get('message', '')))}</li>"
        for item in quality_warnings
        if isinstance(item, dict)
    )
    external = report.get("external_retrieval") or {}
    external_block = ""
    if external.get("related_works") or external.get("failures") or external.get("error"):
        pieces: list[str] = ["<h2>外部相关工作</h2><div class=\"grid\">"]
        status = external.get("status") or {}
        if external.get("error"):
            pieces.append(f"<section class=\"card\"><p>外部检索失败，已跳过：{html.escape(str(external['error']))}</p></section>")
        else:
            overview = (
                f"检索源：{'/'.join(str(name) for name in status.get('providers') or []) or 'NA'}；"
                f"claim 检索式 {len(external.get('queries') or [])} 条；候选 {external.get('candidates', 0)} 篇；"
                f"停止原因：{external.get('stop_reason')}"
            )
            queries = "".join(
                f"<li>{html.escape(str(query.get('rule_id')))}：{html.escape(str(query.get('query')))}</li>"
                for query in (external.get("queries") or [])
            )
            pieces.append(f"<section class=\"card\"><h3>检索概览</h3><p>{html.escape(overview)}</p><ul>{queries}</ul></section>")
        for work in external.get("related_works") or []:
            # 变量名不能叫 summary：那会覆盖外层报告 summary，后面渲染就崩。
            work_summary = work.get("summary") if isinstance(work.get("summary"), dict) else {}
            detail = "".join(
                f"<li><b>{label}</b> {html.escape(str(work_summary.get(key, '')))}</li>"
                for key, label in (("method", "方法"), ("findings", "发现"), ("experiments", "实验"), ("similar", "相似"), ("different", "不同"))
            )
            meta = "，".join(str(piece) for piece in [work.get("year"), work.get("venue")] if piece)
            footer = f"{meta or '年份/期刊缺失'}；他引 {work.get('citation_count')}；重排分 {work.get('rerank_score')}"
            pieces.append(
                f"<section class=\"card\"><h3>{html.escape(str(work.get('title')))}</h3>"
                f"<p>{html.escape(footer)}</p><ul>{detail}</ul></section>"
            )
        pieces.append("</div>")
        external_block = "".join(pieces)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>文献评价报告</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #1f2933; background: #f6f7f9; }}
    header {{ padding: 28px 36px; background: #ffffff; border-bottom: 1px solid #dde3ea; }}
    main {{ padding: 24px 36px 40px; max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
    .metric, .card {{ background: #fff; border: 1px solid #dde3ea; border-radius: 8px; padding: 14px; }}
    .metric b {{ display: block; font-size: 22px; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dde3ea; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e8edf2; text-align: left; vertical-align: top; }}
    th {{ background: #edf2f7; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .evidence {{ padding: 10px; background: #f8fafc; border-left: 3px solid #417690; }}
    .warning {{ background: #fff9ed; border: 1px solid #e6c987; border-radius: 8px; padding: 14px; }}
    @media (max-width: 760px) {{ .metrics, .grid {{ grid-template-columns: 1fr; }} main, header {{ padding-left: 18px; padding-right: 18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(report["paper"]["metadata"].get("title", "文献评价报告"))}</h1>
    <div>Run ID: {html.escape(report["run"]["run_id"])} · 规则: {html.escape(report["rule_profile"]["version"])}</div>
    <div class="metrics">
      <div class="metric">质量分<b>{total}</b></div>
      <div class="metric">基准分<b>{_fmt(summary.get("baseline_score"))}/{_fmt(summary.get("baseline_pool"), 0)}</b></div>
      <div class="metric">加分<b>{_fmt(summary.get("bonus_score"))}/{_fmt(summary.get("bonus_pool"), 0)}</b></div>
      <div class="metric">评分完整度<b>{summary.get("scoring_completeness", 0):.0%}</b></div>
      <div class="metric">评分标尺<b>{html.escape(str(summary.get("scale_version", "NA")))}</b></div>
      <div class="metric">一致性校准分<b>{html.escape(str(consistency.get("consensus_score", "NA")) if isinstance(consistency, dict) else "NA")}</b></div>
      <div class="metric">质量等级<b>{html.escape(str(summary.get("quality_grade", "NA")))}</b></div>
      <div class="metric">顶刊潜力<b>{html.escape(str(summary.get("top_journal_potential", "NA")))}</b></div>
      <div class="metric">多模型一致性<b>{html.escape(str(consistency.get("agreement_level", "NA")) if isinstance(consistency, dict) else "NA")}</b></div>
    </div>
  </header>
  <main>
    <h2>评分模式</h2>
    <section class="card">
      <p><b>{html.escape(str(summary.get("scoring_mode", "普通质量基准")))}</b> · 评分完整度 {summary["scoring_completeness"]:.0%}</p>
      <p>主分用于判断普通学术论文质量；顶刊潜力由总分和核心维度门槛共同判断。</p>
      <ul>{''.join(f'<li>{html.escape(str(reason))}</li>' for reason in summary.get("top_journal_gate_reasons", [])) or "<li>暂无顶刊跃迁扣分门槛提示</li>"}</ul>
    </section>
    <h2>全文理解</h2>
    <div class="metrics">{profile_cards}</div>
    {nonstandard_notice}
    <p>{html.escape(str(paper_context.get("reading_notes", ""))) if isinstance(paper_context, dict) else ""}</p>
    <table><thead><tr><th>画像字段</th><th>全文理解 Agent 判断</th></tr></thead><tbody>{paper_profile_rows}</tbody></table>
    <table><thead><tr><th>章节</th><th>页码</th><th>系统理解摘要</th></tr></thead><tbody>{section_rows}</tbody></table>
    <h2>全文证据分配</h2>
    <table><thead><tr><th>ID</th><th>专家维度</th><th>分配证据数</th><th>分配说明</th></tr></thead><tbody>{route_rows}</tbody></table>
    <h2>专家阅读页码覆盖</h2>
    <table><thead><tr><th>ID</th><th>维度</th><th>实际阅读候选页码</th></tr></thead><tbody>{coverage_rows}</tbody></table>
    <h2>反思审核智能体</h2>
    <section class="card">
      <p><b>审核状态</b> {html.escape(str(audit_report.get("audit_status", "NA")) if isinstance(audit_report, dict) else "NA")} <b>分数可信度</b> {html.escape(str(audit_report.get("score_reliability", "NA")) if isinstance(audit_report, dict) else "NA")}</p>
      <p>{html.escape(str(audit_report.get("overall_assessment", "")) if isinstance(audit_report, dict) else "")}</p>
      <ul>{audit_issue_rows or "<li>暂无审核问题</li>"}</ul>
    </section>
    <h2>多模型一致性复判</h2>
    <section class="card">
      <p><b>主模型分数</b> {html.escape(str(consistency.get("primary_score", "NA")) if isinstance(consistency, dict) else "NA")} <b>一致性校准分</b> {html.escape(str(consistency.get("consensus_score", "NA")) if isinstance(consistency, dict) else "NA")} <b>一致性等级</b> {html.escape(str(consistency.get("agreement_level", "NA")) if isinstance(consistency, dict) else "NA")}</p>
      <p>{html.escape(str(consistency.get("calculation", "")) if isinstance(consistency, dict) else "")}</p>
      <table><thead><tr><th>复判模型</th><th>建议分</th><th>一致性</th><th>复判理由</th></tr></thead><tbody>{consistency_rows or "<tr><td colspan='4'>暂无复判结果</td></tr>"}</tbody></table>
    </section>
    <h2>质量警告</h2>
    <section class="warning"><ul>{warning_rows or "<li>暂无质量警告</li>"}</ul></section>
    <h2>维度得分</h2>
    <table><thead><tr><th>ID</th><th>维度</th><th>质量分</th><th>满分</th><th>可用权重</th></tr></thead><tbody>{dimensions}</tbody></table>
    <h2>创新性与相关性</h2>
    <div class="grid">{analyses}</div>
    {external_block}
    <h2>反思与复核建议</h2>
    <ul>{issues or "<li>暂无</li>"}</ul>
    <h2>证据链明细</h2>
    {detail_cards}
  </main>
</body>
</html>
"""


def save_html_report(json_report_path: str | Path, out_path: str | Path | None = None) -> Path:
    source = Path(json_report_path)
    report = json.loads(source.read_text(encoding="utf-8"))
    target = Path(out_path) if out_path else source.with_suffix(".html")
    target.write_text(render_html(report), encoding="utf-8")
    return target

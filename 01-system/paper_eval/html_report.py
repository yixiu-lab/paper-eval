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
            "<section class=\"card\"><b>Nonstandard paradigm notice</b>"
            f"<p>{html.escape(str(review_profile.get('anti_homogenization_policy', '')))}</p>"
            f"<p>{html.escape(str(review_profile.get('rule_focus_notes', '')))}</p></section>"
        )
    profile_cards = "".join(
        f"<div class=\"metric\">{html.escape(label)}<b>{html.escape(str(value or 'NA'))}</b></div>"
        for label, value in [
            ("Paper type", f"Controlled type {paper_profile.get('paper_type')}（Source：{paper_profile.get('paper_type_source', 'NA')}；Description：{paper_profile.get('paper_type_label') or 'NA'}）"),
            ("Central problem", paper_profile.get("central_problem") or paper_profile.get("research_question")),
            ("Core claims", paper_profile.get("core_claim")),
            ("Methodological approach", paper_profile.get("method_type")),
        ]
    )
    theoretical_move = paper_profile.get("theoretical_move", {}) if isinstance(paper_profile, dict) else {}
    method_design = paper_profile.get("method_design", {}) if isinstance(paper_profile, dict) else {}
    evidence_structure = paper_profile.get("evidence_structure", {}) if isinstance(paper_profile, dict) else {}
    paper_profile_rows = ""
    for label, value in [
        ("Research gap", paper_profile.get("research_gap")),
        ("Main contribution", paper_profile.get("main_contribution")),
        (
            "Theoretical contribution",
            (
                f"{theoretical_move.get('type', 'NA')} / {theoretical_move.get('strength', 'NA')}；"
                f"{theoretical_move.get('description', 'NA')}"
            )
            if isinstance(theoretical_move, dict)
            else "",
        ),
        (
            "Research design",
            (
                f"{method_design.get('design_type', 'NA')}；{method_design.get('sample_data', 'NA')}；"
                f"{method_design.get('identification_or_validity_strategy', 'NA')}"
            )
            if isinstance(method_design, dict)
            else "",
        ),
        ("Main findings", paper_profile.get("main_findings")),
        ("Boundary conditions", paper_profile.get("boundary_conditions")),
        (
            "Evidence structure",
            (
                f"Key evidence：{_list_text(evidence_structure.get('main_evidence', []), 4)} "
                f"Robustness/Validation：{_list_text(evidence_structure.get('robustness_or_validation', []), 4)} "
                f"Cross-section evidence chain：{_list_text(evidence_structure.get('cross_section_support', []), 4)}"
            )
            if isinstance(evidence_structure, dict)
            else "",
        ),
        ("Key pages", ", ".join(str(page) for page in paper_profile.get("most_important_pages", [])[:8]) if isinstance(paper_profile.get("most_important_pages"), list) else ""),
        ("Top-journal signals", _list_text(paper_profile.get("possible_top_journal_signals", []), 5)),
        ("Critical weaknesses", _list_text(paper_profile.get("blocking_weaknesses", []), 5)),
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
          <p><b>Quality score</b> {_fmt(item["weighted_score"])} <b>Judgment</b> {html.escape(item["judgement"])} <b>Confidence</b> {_fmt(item["confidence"])}</p>
          <p><b>Rationale</b> {html.escape(item["reason"])}</p>
          {''.join(f'<p class="flag"><b>Review flag</b> {html.escape(str(flag))}</p>' for flag in item.get("review_flags") or [])}
          <details>
            <summary>Evidence chain</summary>
            {''.join(f'<p class="evidence">Page {ev["page"]}  · {html.escape(ev["section"])}<br>{html.escape(ev["text"])}</p>' for ev in item["evidence"])}
          </details>
          <details>
            <summary>Agent Passages reviewed</summary>
            {''.join(f'<p class="evidence">Page {ev["page"]}  · {html.escape(ev["section"])}<br>{html.escape(ev["text"])}</p>' for ev in item.get("candidate_evidence", [])[:6])}
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
        pieces: list[str] = ["<h2>Related external work</h2><div class=\"grid\">"]
        status = external.get("status") or {}
        if external.get("error"):
            pieces.append(f"<section class=\"card\"><p>External retrieval failed，Skipped：{html.escape(str(external['error']))}</p></section>")
        else:
            overview = (
                f"Retrieval source：{'/'.join(str(name) for name in status.get('providers') or []) or 'NA'}；"
                f"claim Queries {len(external.get('queries') or [])} queries；candidates {external.get('candidates', 0)} papers；"
                f"Stopping reason：{external.get('stop_reason')}"
            )
            queries = "".join(
                f"<li>{html.escape(str(query.get('rule_id')))}：{html.escape(str(query.get('query')))}</li>"
                for query in (external.get("queries") or [])
            )
            pieces.append(f"<section class=\"card\"><h3>Retrieval overview</h3><p>{html.escape(overview)}</p><ul>{queries}</ul></section>")
        for work in external.get("related_works") or []:
            # Do not name this variable summary：That would overwrite the outer report summary，and break downstream rendering。
            work_summary = work.get("summary") if isinstance(work.get("summary"), dict) else {}
            detail = "".join(
                f"<li><b>{label}</b> {html.escape(str(work_summary.get(key, '')))}</li>"
                for key, label in (("method", "Method"), ("findings", "Findings"), ("experiments", "Experiments"), ("similar", "Similarities"), ("different", "Differences"))
            )
            meta = "，".join(str(piece) for piece in [work.get("year"), work.get("venue")] if piece)
            footer = f"{meta or 'Year or journal unavailable'}；Citations {work.get('citation_count')}；Rerank score {work.get('rerank_score')}"
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
  <title>PaperEval Report</title>
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
    <h1>{html.escape(report["paper"]["metadata"].get("title", "PaperEval Report"))}</h1>
    <div>Run ID: {html.escape(report["run"]["run_id"])} · Rule profile: {html.escape(report["rule_profile"]["version"])}</div>
    <div class="metrics">
      <div class="metric">Quality score<b>{total}</b></div>
      <div class="metric">Baseline score<b>{_fmt(summary.get("baseline_score"))}/{_fmt(summary.get("baseline_pool"), 0)}</b></div>
      <div class="metric">Bonus score<b>{_fmt(summary.get("bonus_score"))}/{_fmt(summary.get("bonus_pool"), 0)}</b></div>
      <div class="metric">Scoring completeness<b>{summary.get("scoring_completeness", 0):.0%}</b></div>
      <div class="metric">Scoring scale<b>{html.escape(str(summary.get("scale_version", "NA")))}</b></div>
      <div class="metric">Consistency-adjusted score<b>{html.escape(str(consistency.get("consensus_score", "NA")) if isinstance(consistency, dict) else "NA")}</b></div>
      <div class="metric">Quality grade<b>{html.escape(str(summary.get("quality_grade", "NA")))}</b></div>
      <div class="metric">Top-journal potential<b>{html.escape(str(summary.get("top_journal_potential", "NA")))}</b></div>
      <div class="metric">Cross-model agreement<b>{html.escape(str(consistency.get("agreement_level", "NA")) if isinstance(consistency, dict) else "NA")}</b></div>
    </div>
  </header>
  <main>
    <h2>Scoring mode</h2>
    <section class="card">
      <p><b>{html.escape(str(summary.get("scoring_mode", "Standard quality baseline")))}</b> · Scoring completeness {summary["scoring_completeness"]:.0%}</p>
      <p>The primary score measures overall scholarly quality；Top-journal potential is assessed using the total score and core-dimension thresholds。</p>
      <ul>{''.join(f'<li>{html.escape(str(reason))}</li>' for reason in summary.get("top_journal_gate_reasons", [])) or "<li>No top-journal threshold deductions were triggered</li>"}</ul>
    </section>
    <h2>Full-text understanding</h2>
    <div class="metrics">{profile_cards}</div>
    {nonstandard_notice}
    <p>{html.escape(str(paper_context.get("reading_notes", ""))) if isinstance(paper_context, dict) else ""}</p>
    <table><thead><tr><th>Profile field</th><th>Full-text understanding Agent Judgment</th></tr></thead><tbody>{paper_profile_rows}</tbody></table>
    <table><thead><tr><th>Section</th><th>Page</th><th>System summary</th></tr></thead><tbody>{section_rows}</tbody></table>
    <h2>Full-text evidence routing</h2>
    <table><thead><tr><th>ID</th><th>Reviewer dimension</th><th>Evidence items assigned</th><th>Routing rationale</th></tr></thead><tbody>{route_rows}</tbody></table>
    <h2>Reviewer page coverage</h2>
    <table><thead><tr><th>ID</th><th>Dimension</th><th>Candidate pages reviewed</th></tr></thead><tbody>{coverage_rows}</tbody></table>
    <h2>Reflection audit</h2>
    <section class="card">
      <p><b>Audit status</b> {html.escape(str(audit_report.get("audit_status", "NA")) if isinstance(audit_report, dict) else "NA")} <b>Score reliability</b> {html.escape(str(audit_report.get("score_reliability", "NA")) if isinstance(audit_report, dict) else "NA")}</p>
      <p>{html.escape(str(audit_report.get("overall_assessment", "")) if isinstance(audit_report, dict) else "")}</p>
      <ul>{audit_issue_rows or "<li>No audit issues</li>"}</ul>
    </section>
    <h2>Cross-model consistency review</h2>
    <section class="card">
      <p><b>Primary-model score</b> {html.escape(str(consistency.get("primary_score", "NA")) if isinstance(consistency, dict) else "NA")} <b>Consistency-adjusted score</b> {html.escape(str(consistency.get("consensus_score", "NA")) if isinstance(consistency, dict) else "NA")} <b>Agreement level</b> {html.escape(str(consistency.get("agreement_level", "NA")) if isinstance(consistency, dict) else "NA")}</p>
      <p>{html.escape(str(consistency.get("calculation", "")) if isinstance(consistency, dict) else "")}</p>
      <table><thead><tr><th>Review model</th><th>Recommended score</th><th>Agreement</th><th>Review rationale</th></tr></thead><tbody>{consistency_rows or "<tr><td colspan='4'>No review result</td></tr>"}</tbody></table>
    </section>
    <h2>Quality warnings</h2>
    <section class="warning"><ul>{warning_rows or "<li>No quality warnings</li>"}</ul></section>
    <h2>Dimension scores</h2>
    <table><thead><tr><th>ID</th><th>Dimension</th><th>Quality score</th><th>Maximum score</th><th>Applicable weight</th></tr></thead><tbody>{dimensions}</tbody></table>
    <h2>Novelty and relevance</h2>
    <div class="grid">{analyses}</div>
    {external_block}
    <h2>Reflection and review notes</h2>
    <ul>{issues or "<li>None</li>"}</ul>
    <h2>Evidence-chain details</h2>
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

from __future__ import annotations

from collections.abc import Callable
import html
import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from .agent_graph import run_evaluation_graph
from .consistency import run_multi_model_consistency_review
from .llm import load_agent_runtime_config
from .models import EvaluationRun, Paper, RunStatus, now_iso, to_dict
from .pdf_diagnostics import diagnose_parsed_pdf, parse_warning_message, should_block_agent_scoring
from .pdf_parser import parse_pdf
from .html_report import save_html_report
from .reliability import (
    discover_report_paths,
    load_reliability_config,
    load_traces_from_reports,
    run_trace_from_report,
)
from .reliability_diagnosis import build_reliability_report
from .report import build_report, save_report
from .rules import load_rule_profile, validate_rule_profile
from .runtime_errors import classify_runtime_error
from .storage import save_run_bundle


HARNESS_VERSION = "harness-0.2"


def _notify(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback:
        progress_callback(message)


def _existing_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def _optional_existing_file(path: str | Path | None, label: str) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def _start_stage(trace: list[dict[str, Any]], step: str, message: str, progress_callback: Callable[[str], None] | None) -> float:
    _notify(progress_callback, message)
    trace.append({"step": step, "status": "started", "at": now_iso()})
    return time.perf_counter()


def _finish_stage(trace: list[dict[str, Any]], step: str, started: float, **extra: Any) -> None:
    item: dict[str, Any] = {
        "step": step,
        "status": "completed",
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    item.update(extra)
    trace.append(item)


def _write_manifest(
    out_dir: Path,
    run_id: str,
    status: RunStatus,
    started_at: str,
    completed_at: str,
    paths: dict[str, str | None],
    trace: list[dict[str, Any]],
    error_info: dict[str, str] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{run_id}_manifest.json"
    manifest = {
        "run_id": run_id,
        "status": status.value,
        "harness_version": HARNESS_VERSION,
        "started_at": started_at,
        "completed_at": completed_at,
        "paths": paths,
        "error_info": error_info,
        "trace": trace,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _save_failure_report(
    out_dir: Path,
    run: EvaluationRun,
    paper: Paper | None,
    trace: list[dict[str, Any]],
    error_info: dict[str, str],
    raw_error: str,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run.run_id}_report.json"
    md_path = out_dir / f"{run.run_id}_report.md"
    html_path = out_dir / f"{run.run_id}_report.html"
    report = {
        "run": to_dict(run),
        "paper": to_dict(paper) if paper else None,
        "summary": {
            "total_score": None,
            "scoring_completeness": 0,
            "quality_grade": "运行失败",
            "top_journal_potential": "NA",
        },
        "error_info": error_info,
        "raw_error": raw_error,
        "quality_warnings": [
            {
                "level": "high",
                "type": error_info["type"],
                "message": error_info["user_message"],
            }
        ],
        "details": [],
        "reflection_issues": [],
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# 文献评价报告",
                "",
                f"- Run ID：{run.run_id}",
                "- 状态：failed",
                f"- 失败类型：{error_info['title']}",
                f"- 说明：{error_info['user_message']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-CN">',
                "<head>",
                '<meta charset="utf-8">',
                "<title>文献评价失败</title>",
                '<style>body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:40px;line-height:1.6;color:#172033}main{max-width:880px}.badge{display:inline-block;padding:4px 10px;border-radius:999px;background:#fee2e2;color:#991b1b;font-weight:700}pre{white-space:pre-wrap;background:#f8fafc;padding:16px;border-radius:8px}</style>',
                "</head>",
                "<body><main>",
                '<span class="badge">运行失败</span>',
                f"<h1>{html.escape(error_info['title'])}</h1>",
                f"<p>{html.escape(error_info['user_message'])}</p>",
                f"<p><strong>Run ID：</strong>{html.escape(run.run_id)}</p>",
                f"<pre>{html.escape(raw_error)}</pre>",
                "</main></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    run.trace = trace
    return json_path, md_path, html_path


def _run_reliability_layer(
    report: dict[str, Any],
    profile: Any,
    paper: Paper,
    out_path: Path,
    config_path: str | Path | None,
    progress_callback: Callable[[str], None] | None,
) -> dict[str, Any] | None:
    """评分结束后的**只读后处理**：算评价可靠性，挂到报告上。

    刻意放在 `build_report` 之后、`save_report` 之前：报告结构不变，只是多一个
    `reliability_report` 键。它不改判断、不改分数、不改任何已有 Agent 的行为；
    关掉开关（`reliability.enabled: false`，默认）时整个函数直接返回 None，流程与原来逐位一致。

    输入是"同一篇论文的历史运行报告 + 本次运行"：本次运行还没落盘，所以用内存里的报告对象
    直接建 trace，历史运行从报告目录里读。因此第一次跑某篇论文时会得到
    `insufficient_data`（运行稳定性需要至少 2 次运行），这是设计如此，不是故障。
    """
    reliability_config = load_reliability_config(config_path)
    if not reliability_config.enabled:
        return None
    _notify(progress_callback, "正在计算评价可靠性（只读后处理，不重跑 Agent）")
    traces = []
    try:
        for path in discover_report_paths(out_path):
            if path.name == f"{report['run']['run_id']}_report.json":
                continue
            traces.extend(load_traces_from_reports([path]))
        traces.append(run_trace_from_report(report, out_path / f"{report['run']['run_id']}_report.json"))
        return build_reliability_report(
            profile,
            traces,
            reliability_config,
            focus_paper_id=str((report.get("paper") or {}).get("paper_id") or paper.paper_id),
        )
    except Exception as exc:  # 可靠性层任何失败都不允许影响已经算好的评价结果
        return {
            "reliability_version": "error",
            "error": f"Reliability Layer 计算失败：{str(exc)[:300]}",
            "overall_warning": ["可靠性层异常，本次未产出可靠性结论；论文评分结果不受影响。"],
            "needs_review": [],
        }


def evaluate_pdf(
    pdf_path: str | Path,
    rules_path: str | Path,
    out_dir: str | Path,
    index_path: str | Path | None = None,
    user_topic: str | None = None,
    db_path: str | Path | None = None,
    agent_config_path: str | Path | None = None,
    agent_runtime_overrides: dict[str, object] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    run_id = f"RUN-{uuid.uuid4().hex[:12].upper()}"
    started_at = now_iso()
    out_path = Path(out_dir)
    trace_db = Path(db_path) if db_path else out_path.parent / "evaluation_trace.sqlite"
    trace: list[dict[str, Any]] = []
    run: EvaluationRun | None = None
    paper: Paper | None = None
    evidences = []
    judgements: list[Any] = []
    score_items: list[Any] = []
    parse_diagnostics: dict[str, Any] | None = None
    profile = None
    json_path: Path | None = None
    md_path: Path | None = None
    html_path: Path | None = None
    manifest_path: Path | None = None

    paths = {
        "pdf_path": str(pdf_path),
        "rules_path": str(rules_path),
        "index_path": str(index_path) if index_path else None,
        "agent_config_path": str(agent_config_path) if agent_config_path else None,
        "out_dir": str(out_path),
        "trace_db": str(trace_db),
    }

    try:
        setup_started = _start_stage(trace, "harness.prepare_inputs", "正在准备评价输入", progress_callback)
        pdf_file = _existing_file(pdf_path, "PDF")
        rules_file = _existing_file(rules_path, "Rule profile")
        resolved_index = _optional_existing_file(index_path, "Retrieval index")
        resolved_agent_config = _optional_existing_file(agent_config_path, "Agent runtime config")
        out_path.mkdir(parents=True, exist_ok=True)
        _finish_stage(
            trace,
            "harness.prepare_inputs",
            setup_started,
            pdf_path=str(pdf_file),
            rules_path=str(rules_file),
            index_path=str(resolved_index) if resolved_index else None,
            agent_config_path=str(resolved_agent_config) if resolved_agent_config else None,
        )

        rules_started = _start_stage(trace, "harness.load_rules", "正在读取评分规则", progress_callback)
        profile = load_rule_profile(rules_file)
        errors = validate_rule_profile(profile)
        if errors:
            raise ValueError("Rule profile validation failed: " + "; ".join(errors))
        _finish_stage(
            trace,
            "harness.load_rules",
            rules_started,
            rule_profile=f"{profile.rule_profile_id}:{profile.version}",
            rule_count=len(profile.rules),
            dimension_count=len(profile.dimensions),
        )

        parse_started = _start_stage(trace, "harness.parse_pdf", "正在解析 PDF 全文", progress_callback)
        paper, evidences = parse_pdf(pdf_file)
        parse_diagnostics = diagnose_parsed_pdf(paper, evidences)
        _finish_stage(
            trace,
            "harness.parse_pdf",
            parse_started,
            paper_id=paper.paper_id,
            evidence_count=len(evidences),
            section_count=len(paper.sections),
            parse_ok=parse_diagnostics["parse_ok"],
            parse_warnings=parse_diagnostics["warnings"],
            pages_detected=parse_diagnostics["pages_detected"],
            facet_counts=parse_diagnostics["facet_counts"],
        )
        if not parse_diagnostics["parse_ok"]:
            _notify(progress_callback, "PDF 解析存在风险，系统会继续评分并在报告中标记")
        if should_block_agent_scoring(parse_diagnostics):
            raise RuntimeError(
                "PDF_PARSE_BLOCKED: "
                + parse_warning_message([str(item) for item in parse_diagnostics.get("warnings", [])])
                + "。系统已在调用云端模型前停止，避免浪费 API 额度。"
            )
        run = EvaluationRun(
            run_id=run_id,
            paper_id=paper.paper_id,
            harness_version=HARNESS_VERSION,
            model_versions={"core_agents": "agent-only-openai-compatible"},
            rule_profile=f"{profile.rule_profile_id}:{profile.version}",
            status=RunStatus.PARTIAL,
            started_at=started_at,
            trace=trace,
        )

        graph_started = _start_stage(trace, "harness.run_agent_graph", "正在启动 LangGraph 智能体评分流程", progress_callback)
        graph_state = run_evaluation_graph(
            run_id,
            profile,
            paper,
            evidences,
            agent_config_path=resolved_agent_config,
            agent_runtime_overrides=agent_runtime_overrides,
            progress_callback=progress_callback,
        )
        judgements = graph_state["judgements"]
        score_items = graph_state["score_items"]
        summary = graph_state["summary"]
        reflection_issues = graph_state["reflection_issues"]
        # 记录每条判断实际使用的模型：回退链可能换过模型，run 级不能只写主模型。
        used_models = sorted(
            {item.model_version for item in judgements if item.model_version and item.model_version != "n/a"}
        )
        if used_models:
            run.model_versions = {"core_agents": ", ".join(used_models)}
        trace.extend(graph_state.get("trace", []))
        _finish_stage(
            trace,
            "harness.run_agent_graph",
            graph_started,
            judgement_count=len(judgements),
            reflection_issue_count=len(reflection_issues),
        )

        consistency_started = _start_stage(trace, "harness.multi_model_consistency", "正在运行多模型一致性复判", progress_callback)
        consistency_config = load_agent_runtime_config(resolved_agent_config, overrides=agent_runtime_overrides)
        # 证据预算治理：分数绝对值依赖每条规则的证据条数（实测 10→20 段整体上移约 7 分），
        # 所以规则库声明的预算与实际生效值必须一致；不一致就记进 trace 并提示，避免静默重标定。
        declared_budget = profile.max_evidence_per_rule
        effective_budget = consistency_config.max_evidence_per_rule
        budget_mismatch = declared_budget is not None and declared_budget != effective_budget
        if budget_mismatch:
            _notify(
                progress_callback,
                f"证据预算与规则库声明不一致：声明 {declared_budget} 段、实际 {effective_budget} 段；"
                "分数会在不同预算间不可比，请同步升规则库版本",
            )
        consistency_report = run_multi_model_consistency_review(
            run_id,
            profile,
            judgements,
            score_items,
            evidences,
            summary,
            graph_state.get("paper_context"),
            consistency_config,
        )
        _finish_stage(
            trace,
            "harness.multi_model_consistency",
            consistency_started,
            enabled=consistency_report.get("enabled"),
            agreement_level=consistency_report.get("agreement_level"),
            consensus_score=consistency_report.get("consensus_score"),
            review_count=len(consistency_report.get("reviews", [])),
            failure_count=len(consistency_report.get("failures", [])),
            evidence_budget_declared=declared_budget,
            evidence_budget_effective=effective_budget,
            evidence_budget_mismatch=budget_mismatch,
        )

        # 外部文献和社会问题证据已经在评分图中直接进入六个 Reviewer。
        # 不再重复运行旧扩展 Agent，避免把它们误解为额外评分主体。
        extended_analyses = []

        # 外部文献检索已挪到评分图内、评分之前（langgraph.retrieve_external_evidence），
        # 因为判断「有没有人做过」的规则必须在评分时就看到外部对照。这里只取结果用于报告。
        external_retrieval: dict[str, Any] = graph_state.get("external_retrieval") or {}
        novelty_verification: dict[str, Any] = graph_state.get("novelty_verification") or {}
        external_validation: dict[str, Any] = graph_state.get("external_validation") or {}
        evidence_repository: dict[str, Any] = graph_state.get("evidence_repository") or {}
        external_evidence_count = len(graph_state.get("external_evidences") or [])
        if external_evidence_count:
            evidences = [*evidences, *(graph_state.get("external_evidences") or [])]
        novelty_evidence_count = len(graph_state.get("novelty_evidences") or [])
        if novelty_evidence_count:
            evidences = [*evidences, *(graph_state.get("novelty_evidences") or [])]

        report_started = _start_stage(trace, "harness.save_reports", "正在生成并保存评价报告", progress_callback)
        run.status = RunStatus.NEEDS_REVIEW if any(item.needs_review for item in judgements) or reflection_issues else RunStatus.SUCCESS
        run.completed_at = now_iso()
        run.trace = trace
        report = build_report(
            run,
            paper,
            evidences,
            profile,
            judgements,
            score_items,
            summary,
            extended_analyses,
            reflection_issues,
            paper_context=graph_state.get("paper_context"),
            audit_report=graph_state.get("audit_report"),
            consistency_report=consistency_report,
            parse_diagnostics=parse_diagnostics,
            external_retrieval=external_retrieval,
            novelty_verification=novelty_verification,
            external_validation=external_validation,
            evidence_repository=evidence_repository,
        )
        # Reliability Layer：纯后处理，挂一个 reliability_report 键；关掉时返回值逐位不变。
        reliability_started = _start_stage(trace, "harness.reliability_layer", "正在计算评价可靠性", progress_callback)
        reliability_report = _run_reliability_layer(
            report, profile, paper, out_path, resolved_agent_config, progress_callback
        )
        if reliability_report is not None:
            report["reliability_report"] = reliability_report
        _finish_stage(
            trace,
            "harness.reliability_layer",
            reliability_started,
            enabled=reliability_report is not None,
            analyzed_run_count=(reliability_report or {}).get("analyzed_run_count"),
            main_uncertainty_source=((reliability_report or {}).get("diagnosis") or {}).get("main_uncertainty_source"),
            warning_count=len((reliability_report or {}).get("overall_warning") or []),
            needs_review_count=len((reliability_report or {}).get("needs_review") or []),
        )
        json_path, md_path = save_report(report, out_path)
        html_path = save_html_report(json_path)
        save_run_bundle(trace_db, run, paper, evidences, judgements, score_items, reflection_issues, json_path, md_path, html_path)
        _finish_stage(
            trace,
            "harness.save_reports",
            report_started,
            json_report=str(json_path),
            markdown_report=str(md_path),
            html_report=str(html_path),
            trace_db=str(trace_db),
        )
        run.trace = trace
        manifest_path = _write_manifest(out_path, run_id, run.status, started_at, run.completed_at, paths, trace)
        return {
            "run_id": run_id,
            "status": run.status.value,
            "paper_id": paper.paper_id,
            "evidence_count": len(evidences),
            "judgement_count": len(judgements),
            "total_score": summary["total_score"],
            "consensus_score": consistency_report.get("consensus_score"),
            "consistency_agreement": consistency_report.get("agreement_level"),
            "scoring_completeness": summary["scoring_completeness"],
            "json_report": str(json_path),
            "markdown_report": str(md_path),
            "html_report": str(html_path),
            "manifest": str(manifest_path),
            "reflection_issue_count": len(reflection_issues),
            "extended_analysis_count": len(extended_analyses),
            "external_related_work_count": len(external_retrieval.get("related_works") or []),
            "external_claim_count": len(external_retrieval.get("claims") or []),
            "novelty_verdict_count": len(novelty_verification.get("verdicts") or []),
            "external_validation_finding_count": external_validation.get("finding_count", 0),
            "external_evidence_unavailable": bool(external_retrieval.get("external_evidence_unavailable")),
            # Reliability Layer 摘要（未启用时为 None）：只报告，不参与任何评分。
            "reliability": None if reliability_report is None else {
                "main_uncertainty_source": ((reliability_report.get("diagnosis") or {}).get("main_uncertainty_source")),
                "analyzed_run_count": reliability_report.get("analyzed_run_count"),
                "reliability_report": reliability_report.get("reliability_report"),
                "warning_count": len(reliability_report.get("overall_warning") or []),
                "needs_review": reliability_report.get("needs_review") or [],
            },
            "trace_db": str(trace_db),
        }
    except Exception as exc:
        completed_at = now_iso()
        raw_error = str(exc)
        error_info = classify_runtime_error(raw_error)
        trace.append(
            {
                "step": "harness.failure",
                "status": "failed",
                "at": completed_at,
                "error_type": error_info["type"],
                "error_title": error_info["title"],
                "error": raw_error,
                "traceback": traceback.format_exc(limit=8),
            }
        )
        _notify(progress_callback, error_info["title"])
        failed_run = run or EvaluationRun(
            run_id=run_id,
            paper_id=paper.paper_id if paper else "UNKNOWN",
            harness_version=HARNESS_VERSION,
            model_versions={"core_agents": "agent-only-openai-compatible"},
            rule_profile=f"{profile.rule_profile_id}:{profile.version}" if profile else "UNKNOWN",
            status=RunStatus.FAILED,
            started_at=started_at,
        )
        failed_run.status = RunStatus.FAILED
        failed_run.completed_at = completed_at
        failed_run.trace = trace
        if paper is not None:
            json_path, md_path, html_path = _save_failure_report(out_path, failed_run, paper, trace, error_info, raw_error)
            # 打分已经跑完、只是后面某一步失败时，判断和分数仍要落库，否则一次渲染 bug 就让整轮 API 花费作废。
            save_run_bundle(
                trace_db,
                failed_run,
                paper,
                evidences,
                judgements,
                score_items,
                [],
                json_path,
                md_path,
                html_path,
            )
        manifest_path = _write_manifest(out_path, run_id, RunStatus.FAILED, started_at, completed_at, paths, trace, error_info)
        return {
            "run_id": run_id,
            "status": RunStatus.FAILED.value,
            "paper_id": paper.paper_id if paper else None,
            "evidence_count": len(evidences),
            "judgement_count": 0,
            "total_score": None,
            "scoring_completeness": 0,
            "json_report": str(json_path) if json_path else None,
            "markdown_report": str(md_path) if md_path else None,
            "html_report": str(html_path) if html_path else None,
            "manifest": str(manifest_path),
            "reflection_issue_count": 0,
            "extended_analysis_count": 0,
            "trace_db": str(trace_db),
            "error": raw_error,
            "error_info": error_info,
        }

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .harness import evaluate_pdf
from .runtime_errors import classify_runtime_error


def risk_score_from_result(result: dict[str, Any]) -> int:
    if result.get("status") == "failed":
        return 999
    risk = 0
    completeness = float(result.get("scoring_completeness") or 0)
    if completeness < 0.8:
        risk += 40
    if result.get("status") == "needs_review":
        risk += 20
    risk += int(result.get("reflection_issue_count") or 0) * 10
    risk += int(result.get("quality_warning_count") or 0) * 3
    if result.get("parse_ok") is False:
        risk += 30
    return risk


def enrich_result_from_report(result: dict[str, Any]) -> dict[str, Any]:
    json_report = result.get("json_report")
    if not json_report:
        return result
    path = Path(str(json_report))
    if not path.exists():
        return result
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return result
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    consistency = report.get("multi_model_consistency", {}) if isinstance(report, dict) else {}
    paper = report.get("paper", {}) if isinstance(report, dict) else {}
    metadata = paper.get("metadata", {}) if isinstance(paper, dict) else {}
    result = {**result}
    result["paper_title"] = metadata.get("title")
    result["quality_grade"] = summary.get("quality_grade")
    result["top_journal_potential"] = summary.get("top_journal_potential")
    result["consensus_score"] = consistency.get("consensus_score") if isinstance(consistency, dict) else None
    result["consistency_agreement"] = consistency.get("agreement_level") if isinstance(consistency, dict) else None
    result["consistency_review_count"] = len(consistency.get("reviews", [])) if isinstance(consistency, dict) else 0
    result["quality_warning_count"] = len(report.get("quality_warnings", [])) if isinstance(report, dict) else 0
    result["parse_ok"] = (report.get("parse_diagnostics") or {}).get("parse_ok") if isinstance(report, dict) else None
    return result


def overview_item(pdf: Path, result: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    result = enrich_result_from_report(result)
    return {
        "pdf": str(pdf),
        "name": pdf.name,
        "paper_title": result.get("paper_title"),
        "ok": result.get("status") != "failed",
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "paper_id": result.get("paper_id"),
        "total_score": result.get("total_score"),
        "consensus_score": result.get("consensus_score"),
        "consistency_agreement": result.get("consistency_agreement"),
        "consistency_review_count": result.get("consistency_review_count"),
        "scoring_completeness": result.get("scoring_completeness"),
        "quality_grade": result.get("quality_grade"),
        "top_journal_potential": result.get("top_journal_potential"),
        "quality_warning_count": result.get("quality_warning_count"),
        "parse_ok": result.get("parse_ok"),
        "risk_score": risk_score_from_result(result),
        "reflection_issue_count": result.get("reflection_issue_count"),
        "html_report": result.get("html_report"),
        "json_report": result.get("json_report"),
        "error_info": result.get("error_info"),
        "error": result.get("error"),
        "elapsed_seconds": round(elapsed_seconds, 2),
    }


def overview_item_from_report(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run = report.get("run", {})
    paper = report.get("paper", {})
    metadata = paper.get("metadata", {}) if isinstance(paper, dict) else {}
    summary = report.get("summary", {})
    pdf = Path(str(metadata.get("source_path") or report_path.name))
    result = {
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "paper_id": paper.get("paper_id") if isinstance(paper, dict) else None,
        "total_score": summary.get("total_score"),
        "scoring_completeness": summary.get("scoring_completeness"),
        "json_report": str(report_path),
        "html_report": str(report_path.with_suffix(".html")),
        "reflection_issue_count": len(report.get("reflection_issues", [])),
    }
    return overview_item(pdf, result, 0)


def sort_overview(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    top_rank = {"高": 0, "中": 1, "低": 2, "NA": 3, None: 3}
    return sorted(
        items,
        key=lambda item: (
            not bool(item.get("ok")),
            int(item.get("risk_score") or 0),
            top_rank.get(item.get("top_journal_potential"), 3),
            -(float(item.get("consensus_score") or item.get("total_score") or -1)),
            str(item.get("name") or ""),
        ),
    )


def build_batch_summary(batch_id: str, pdf_dir: Path, limit: int, items: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sort_overview(items)
    scored = [item for item in items if item.get("ok") and item.get("total_score") is not None]
    return {
        "batch_id": batch_id,
        "pdf_dir": str(pdf_dir),
        "limit": limit,
        "checked": len(items),
        "completed": sum(1 for item in items if item.get("ok")),
        "failed": sum(1 for item in items if not item.get("ok")),
        "average_score": None
        if not scored
        else round(sum(float(item["total_score"]) for item in scored) / len(scored), 4),
        "average_consensus_score": None
        if not [item for item in items if item.get("ok") and item.get("consensus_score") is not None]
        else round(
            sum(float(item["consensus_score"]) for item in items if item.get("ok") and item.get("consensus_score") is not None)
            / len([item for item in items if item.get("ok") and item.get("consensus_score") is not None]),
            4,
        ),
        "overview": ranked,
    }


def write_batch_summary(batch_dir: Path, summary: dict[str, Any]) -> Path:
    summary_path = batch_dir / f"{summary['batch_id']}_overview.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def build_overview_from_reports(batch_dir: Path, pdf_dir: Path | None = None, batch_id: str | None = None) -> dict[str, Any]:
    resolved_batch_id = batch_id or batch_dir.name
    items = [overview_item_from_report(path) for path in sorted(batch_dir.glob("*_report.json"))]
    summary = build_batch_summary(resolved_batch_id, pdf_dir or batch_dir, len(items), items)
    summary["is_partial"] = False
    summary["summary_path"] = str(write_batch_summary(batch_dir, summary))
    return summary


def run_batch_evaluation(
    pdf_dir: Path,
    limit: int,
    out_dir: Path,
    rules_path: Path,
    index_path: Path | None,
    db_path: Path,
    agent_config_path: Path,
    user_topic: str | None = None,
    agent_runtime_overrides: dict[str, object] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    start_index: int = 0,
    batch_id: str | None = None,
) -> dict[str, Any]:
    all_pdfs = sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())
    pdfs = all_pdfs[start_index : start_index + limit]
    batch_id = batch_id or f"BATCH-{int(time.time())}"
    batch_dir = out_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for index, pdf in enumerate(pdfs, start=1):
        if progress_callback:
            progress_callback(f"批量评价 {start_index + index}/{start_index + len(pdfs)}：{pdf.name}")
        started = time.monotonic()
        try:
            result = evaluate_pdf(
                pdf,
                rules_path,
                batch_dir,
                index_path=index_path if index_path and index_path.exists() else None,
                user_topic=user_topic,
                db_path=db_path,
                agent_config_path=agent_config_path,
                agent_runtime_overrides=agent_runtime_overrides,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            result = {"status": "failed", "error": str(exc), "error_info": classify_runtime_error(str(exc))}
        items.append(overview_item(pdf, result, time.monotonic() - started))
        error_type = (result.get("error_info") or {}).get("type")
        if error_type in {"insufficient_balance", "insufficient_quota"}:
            stop_reason = error_type
        partial = build_batch_summary(batch_id, pdf_dir, start_index + limit, items)
        partial["start_index"] = start_index
        partial["is_partial"] = len(items) < len(pdfs)
        if stop_reason:
            partial["stop_reason"] = stop_reason
        write_batch_summary(batch_dir, partial)
        if stop_reason:
            break

    summary = build_batch_summary(batch_id, pdf_dir, start_index + limit, items)
    summary["start_index"] = start_index
    summary["is_partial"] = len(items) < len(pdfs)
    if stop_reason:
        summary["stop_reason"] = stop_reason
    summary_path = write_batch_summary(batch_dir, summary)
    summary["summary_path"] = str(summary_path)
    return summary

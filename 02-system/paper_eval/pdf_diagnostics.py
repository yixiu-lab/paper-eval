from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Evidence, Paper
from .paper_context import build_evidence_index
from .pdf_parser import parse_pdf


KEY_FACETS = ["method", "results", "discussion", "tables_figures", "appendix", "middle_body"]
CORE_SECTIONS = ["Introduction", "Methods", "Results", "Discussion", "Conclusion"]
SEVERE_WARNINGS = {"only_first_pages", "low_evidence_count", "missing_middle_body"}


def diagnose_parsed_pdf(paper: Paper, evidences: list[Evidence], elapsed_seconds: float | None = None) -> dict[str, Any]:
    sections = Counter(item.section for item in evidences)
    pages = sorted({item.page for item in evidences})
    index = build_evidence_index(evidences)
    facet_counts = {facet: len(index.get(facet, [])) for facet in KEY_FACETS}
    missing_facets = [facet for facet, count in facet_counts.items() if count == 0 and facet != "appendix"]
    core_section_count = sum(1 for section in CORE_SECTIONS if sections.get(section, 0) > 0)
    warnings: list[str] = []
    if len(pages) <= 2:
        warnings.append("only_first_pages")
    if len(evidences) < 80:
        warnings.append("low_evidence_count")
    if core_section_count < 3:
        warnings.append("few_core_sections")
    if facet_counts["method"] == 0:
        warnings.append("missing_method_facet")
    if facet_counts["results"] == 0:
        warnings.append("missing_results_facet")
    if facet_counts["discussion"] == 0:
        warnings.append("missing_discussion_facet")
    if facet_counts["middle_body"] == 0:
        warnings.append("missing_middle_body")

    return {
        "paper_id": paper.paper_id,
        "pages_detected": len(pages),
        "first_page": min(pages) if pages else None,
        "last_page": max(pages) if pages else None,
        "evidence_count": len(evidences),
        "section_count": len(sections),
        "core_section_count": core_section_count,
        "sections": dict(sections.most_common()),
        "facet_counts": facet_counts,
        "missing_facets": missing_facets,
        "warnings": warnings,
        "parse_ok": len(warnings) == 0,
        "elapsed_seconds": elapsed_seconds,
    }


def diagnose_pdf_path(pdf_path: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    paper, evidences = parse_pdf(pdf_path)
    result = diagnose_parsed_pdf(paper, evidences, elapsed_seconds=round(time.monotonic() - started, 3))
    result.update({"pdf": str(pdf_path), "name": Path(pdf_path).name})
    return result


def parse_warning_message(warnings: list[str]) -> str:
    labels = {
        "only_first_pages": "只识别到前两页或更少页码",
        "low_evidence_count": "可用文本证据数量偏少",
        "few_core_sections": "核心章节识别不足",
        "missing_method_facet": "方法证据池为空",
        "missing_results_facet": "结果证据池为空",
        "missing_discussion_facet": "讨论证据池为空",
        "missing_middle_body": "中后段正文证据池为空",
        "parse_failed": "PDF 解析失败",
    }
    return "；".join(labels.get(item, item) for item in warnings)


def should_block_agent_scoring(diagnostics: dict[str, Any]) -> bool:
    warnings = {str(item) for item in diagnostics.get("warnings", [])}
    return bool(warnings & SEVERE_WARNINGS)

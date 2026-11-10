from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from .models import Evidence, Paper


SECTION_PATTERNS = [
    (re.compile('^(?:\\d+\\.?\\s*)?abstract\\b|^Abstract$', re.I), "Abstract"),
    (re.compile('^(?:\\d+\\.?\\s*)?introduction\\b|^Introduction$', re.I), "Introduction"),
    (re.compile('^(?:\\d+\\.?\\s*)?(literature review|theory|theoretical background|background|related work|hypotheses?)\\b|^(theory|literature review|research hypotheses)$', re.I), "Literature Review"),
    (re.compile('^(?:\\d+(?:\\.\\d+)*\\.?\\s*)?(methods?|methodology|research methodology|research design|data collection|data and sample|sample and procedure|sample characteristics|measures?|model architecture|training)\\b|^(methods|research design|data and sample|variable measurement|data collection|sample characteristics)$', re.I), "Methods"),
    (re.compile('^(?:\\d+(?:\\.\\d+)*\\.?\\s*)?(results?|findings?|analysis|experiments?|evaluation|measurement model evaluation|structural model evaluation|mediation effect|moderation test|control variables|hypothesis testing)\\b|^(results|research results|empirical results|analysis results)$', re.I), "Results"),
    (re.compile('^(?:\\d+(?:\\.\\d+)*\\.?\\s*)?(discussion|(?:theoretical|practical|managerial) implications)\\b|^Discussion$', re.I), "Discussion"),
    (re.compile('^(?:\\d+(?:\\.\\d+)*\\.?\\s*)?(conclusion|conclusions|discussion and conclusion|general discussion|limitations?|future research|implications?)\\b|^(conclusion|limitations|future research|implications)$', re.I), "Conclusion"),
    (re.compile('^(?:\\d+(?:\\.\\d+)*\\.?\\s*)?(appendix|appendices|supplementary material)\\b|^(Appendix|Supplementary material)$', re.I), "Appendix"),
    (re.compile('^(?:\\d+(?:\\.\\d+)*\\.?\\s*)?(references|bibliography)\\b|^References$', re.I), "References"),
]
TABLE_PATTERN = re.compile('^(table|figure|fig\\.?)\\s+\\d+|^(table|figure)\\s*\\d+', re.I)
HEADING_NUMBER_PATTERN = re.compile(r"^\s*(?:\d+|[IVX]+)(?:\.\d+)*\.?\s+[A-Z][A-Za-z ,/&-]{2,80}$")


def stable_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12].upper()
    return f"{prefix}-{digest}"


def normalize_line(line: str) -> str:
    line = line.replace("\u2003", " ").replace("\u2002", " ").replace("\xa0", " ")
    line = re.sub(r"\s*[|｜]\s*", " ", line)
    return re.sub(r"\s+", " ", line).strip()


def detect_heading_section(line: str) -> str | None:
    candidate = normalize_line(line).strip(" .:：")
    if not candidate or len(candidate) > 100:
        return None
    if candidate.endswith("-"):
        return None
    # Full prose sentences such as "Findings indicated ..." are not headings.
    heading_text = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+", "", candidate, flags=re.I)
    if len(candidate.split()) > 10 or re.search(r"[.!?。！？]\s+\w", heading_text):
        return None
    lowered_candidate = candidate.lower()
    for heading_word in ["findings", "results", "discussion", "conclusion", "limitations", "method", "methods"]:
        if lowered_candidate.startswith(heading_word + " ") and len(lowered_candidate.split()) > 2:
            return None
    for pattern, section in SECTION_PATTERNS:
        if pattern.search(candidate):
            return section
    if HEADING_NUMBER_PATTERN.match(candidate):
        lowered = candidate.lower()
        for pattern, section in SECTION_PATTERNS:
            if pattern.search(re.sub(r"^\s*(?:\d+|[ivx]+)(?:\.\d+)*\.?\s+", "", lowered, flags=re.I)):
                return section
    return None


def guess_page_section(text: str, current: str) -> str:
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    for line in lines[:12]:
        section = detect_heading_section(line)
        if section:
            return section
    head = normalize_line(text)[:220]
    for pattern, section in SECTION_PATTERNS:
        if pattern.search(head):
            return section
    return current


def guess_paragraph_section(paragraph: str, current: str) -> str:
    first_line = normalize_line(paragraph.splitlines()[0] if paragraph.splitlines() else paragraph)
    if TABLE_PATTERN.search(first_line):
        return "Tables and Figures"
    section = detect_heading_section(first_line)
    return section or current


def iter_paragraphs_with_sections(text: str, current: str) -> list[tuple[str, str]]:
    cleaned = text.replace("\x00", " ")
    parts = re.split(r"\n\s*\n|(?<=[.!?。！？])\s+(?=[A-Z\u4e00-\u9fff])", cleaned)
    rows: list[tuple[str, str]] = []
    section = current
    for part in parts:
        raw = part.strip()
        if not raw:
            continue
        detected = guess_paragraph_section(raw, section)
        section = detected
        paragraph = re.sub(r"\s+", " ", raw).strip()
        if len(paragraph) >= 40:
            rows.append((section, paragraph))
    return rows


def parse_pdf(path: str | Path) -> tuple[Paper, list[Evidence]]:
    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    paper_id = stable_id(pdf_path.name, pdf_path.stat().st_size, prefix="PAPER")
    metadata = {
        "title": reader.metadata.title if reader.metadata and reader.metadata.title else pdf_path.stem,
        "source_path": str(pdf_path),
        "page_count": len(reader.pages),
    }
    sections: list[dict[str, object]] = []
    evidences: list[Evidence] = []
    references: list[str] = []
    current_section = "Unknown"

    for page_index, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        current_section = guess_page_section(text, current_section)
        sections.append({"section": current_section, "page": page_index, "chars": len(text)})
        cursor = 0
        for paragraph_index, (section, paragraph) in enumerate(iter_paragraphs_with_sections(text, current_section), 1):
            start = text.find(paragraph, cursor)
            if start < 0:
                start = cursor
            end = start + len(paragraph)
            cursor = end
            current_section = section if section not in {"Tables and Figures"} else current_section
            if section == "References":
                references.append(paragraph)
            evidence_id = stable_id(paper_id, page_index, paragraph_index, paragraph[:80], prefix="EV")
            evidences.append(
                Evidence(
                    evidence_id=evidence_id,
                    paper_id=paper_id,
                    section=section,
                    page=page_index,
                    span=(start, end),
                    text=paragraph,
                    text_hash=hashlib.sha1(paragraph.encode("utf-8")).hexdigest(),
                )
            )

    return Paper(
        paper_id=paper_id,
        metadata=metadata,
        sections=sections,
        references=references,
    ), evidences

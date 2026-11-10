from __future__ import annotations

from pathlib import Path

from .models import ExtendedAnalysis, Paper, RetrievalMatch
from .retrieval import FACETS, load_index, search_top_k


def _query_from_paper(paper: Paper) -> str:
    bits = [str(paper.metadata.get("title", ""))]
    for section in paper.sections[:12]:
        bits.append(str(section.get("section", "")))
    return " ".join(bits)


def _as_match(row: dict[str, object]) -> RetrievalMatch:
    return RetrievalMatch(
        doc_id=str(row["doc_id"]),
        title=str(row["title"]),
        facet=str(row["facet"]),
        score=float(row["score"]),
        source_path=str(row["source_path"]),
        publish_year=row.get("publish_year") if isinstance(row.get("publish_year"), int) else None,
        preview=str(row.get("preview", "")),
    )


def run_novelty_agent(paper: Paper, index_path: str | Path | None, top_k: int = 5) -> ExtendedAnalysis:
    if not index_path or not Path(index_path).exists():
        return ExtendedAnalysis(
            agent_id="novelty_agent",
            analysis_type="novelty",
            summary='does not provide a historical document index, and innovative Agents cannot currently perform time-constrained Top-K comparisons.',
            matches=[],
            confidence=0.0,
            needs_review=True,
        )
    docs = load_index(index_path)
    query = _query_from_paper(paper)
    before_year = None
    if paper.publish_date and paper.publish_date[:4].isdigit():
        before_year = int(paper.publish_date[:4])
    rows = []
    for facet in ["topic", "problem", "method", "contribution"]:
        rows.extend(search_top_k(docs, query, facet=facet, k=top_k, before_year=before_year))
    rows.sort(key=lambda item: item["score"], reverse=True)
    matches = [_as_match(item) for item in rows[:top_k]]
    if matches:
        summary = 'Similar documents have been returned based on the local historical document index for subsequent manual or LLM judgment of innovative differences.'
        confidence = min(0.8, 0.35 + matches[0].score)
        needs_review = True
    else:
        summary = 'No valid similar documents were retrieved. To judge the innovativeness, supplementary literature database or search strategy is required.'
        confidence = 0.2
        needs_review = True
    return ExtendedAnalysis("novelty_agent", "novelty", summary, matches, confidence, needs_review)


def run_relevance_agent(paper: Paper, user_topic: str | None, index_path: str | Path | None, top_k: int = 5) -> ExtendedAnalysis:
    if not user_topic:
        return ExtendedAnalysis(
            agent_id="relevance_agent",
            analysis_type="relevance",
            summary='The user research topic is not provided, and the correlation agent cannot currently determine the topic/method/data matching.',
            matches=[],
            confidence=0.0,
            needs_review=True,
        )
    matches: list[RetrievalMatch] = []
    if index_path and Path(index_path).exists():
        docs = load_index(index_path)
        rows = []
        for facet in FACETS:
            rows.extend(search_top_k(docs, user_topic, facet=facet, k=top_k))
        rows.sort(key=lambda item: item["score"], reverse=True)
        matches = [_as_match(item) for item in rows[:top_k]]
    summary = 'Relevance search references have been generated based on user research topics; the final relevance should be manually reviewed based on paper evidence and user topics.'
    confidence = 0.45 if matches else 0.25
    return ExtendedAnalysis("relevance_agent", "relevance", summary, matches, confidence, True)

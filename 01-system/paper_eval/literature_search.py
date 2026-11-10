'External Literature Retrieval: Multi-channel recall → Deduplication → Time leak filtering → Reranker → External Evidence Set.\n\nThree-way recall (**not allowed** only searches the title of the paper):\n\n1. **Query Retrieval**: Each claim generates 3-5 search terms (one for each core construct/theory/mechanism/research situation),\n   Go to Semantic Scholar search (search_external, source can be changed).\n2. **Citation Retrieval**: First locate the target paper and read its references; then select references based on highly relevant references.\n   Form a local citation graph. The citation relationship can find documents that "do not have a common word in the title, but are indeed in the same theoretical context."\n3. **Semantic Retrieval**: Use claim + abstract + research question and candidate document abstract to calculate the lexical semantic relevance (TF-IDF cosine),\n   There is no additional API call, so it is always available, ensuring that there is still a way to recall when the external API is completely hung.\n\nTime leakage control: To verify the innovativeness of a certain paper, only research published before the publication year of the target paper can be used, so the candidate pool is\nFilter by year (`time_leak_guard`) before reranking. Leaking out of the pool is safer than contamination judgment.\n\nAll search logic lives in this module. The Agent only consumes `external_evidence_set` and does not issue searches by itself.'

from __future__ import annotations

import math
import re
import time
from typing import Any, Callable

from .external_retrieval import (
    DEFAULT_CLAIM_QUERY_EN,
    ExternalRetrievalConfig,
    RelatedWork,
    _cached_or_fetch,
    is_self_reference,
    rerank as token_rerank,
    search_external,
)
from .models import Evidence, now_iso
from .semantic_scholar import S2Config
from .semantic_scholar import find_by_title as s2_find_by_title
from .semantic_scholar import references as s2_references

ROUTE_QUERY = "query"
ROUTE_CITATION = "citation"
ROUTE_SEMANTIC = "semantic"
ROUTE_ORDER = (ROUTE_CITATION, ROUTE_SEMANTIC, ROUTE_QUERY)
ROUTE_WEIGHT = {ROUTE_CITATION: 0.10, ROUTE_SEMANTIC: 0.08, ROUTE_QUERY: 0.05}

# The hit bonus of thesis topic words in the candidates. The subject word is "what this paper studies", which is more distinguishable than the general word claim:
# When this item is missing in the actual test, general reviews such as "fairness review" will rely on common words to hit the test (the real priors on the same topic are suppressed).
TOPIC_BONUS_PER_HIT = 0.3
TOPIC_BONUS_CAP = 0.6

SEMANTIC_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "is", "are", "be",
    "as", "by", "that", "this", "we", "our", "it", "its", "from", "at", "how", "when", "what",
    "are", "was", "were", "has", "have", "had", "not", "but", "can", "may", "also", "than",
    "which", "who", "whom", "their", "they", "them", "these", "those", "such", "into", "over",
    "more", "most", "other", "some", "any", "all", "new", "first", "study", "review", "paper",
    "research", "findings", "results", "analysis", "based", "using", "used", "present", "show",
    'Research', 'Paper', 'This article', '', 'and', 'and', 'and', 'or',
}

# rerank The weight of each signal comes from the configuration. If it is changed, the scale is changed and must be reported.
RANK_FEATURE_LABELS = {
    "topic_relevance": 'Topic Relevance',
    "construct_overlap": 'Core construct overlap',
    "theory_relevance": 'Theoretical relevance',
    "mechanism_relevance": 'Mechanism dependency',
    "context_relevance": 'Study situational relevance',
    "semantic_similarity": 'Semantic similarity',
    "citation_relationship": "citation relationship",
    "publication_year": "publication year",
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}|[\u4e00-\u9fff]{2,}", str(text or "").lower())
    return [word for word in words if word not in SEMANTIC_STOPWORDS]


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _as_terms(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {token for item in value for token in _tokens(str(item))}
    return set(_tokens(str(value or "")))


def expand_query_routes(claim: Any, paper_title: str, limit: int = 5) -> list[dict[str, str]]:
    'Expand a claim into 3-5 search terms: core construct/theory/mechanism/research situation/claim English translation.\n\n    It is not enough to just send a "paper title" search query: the same innovative claim may be under other construct names or other theoretical names.\n    Someone has already done it. The purpose of multi-path search is to hit those documents whose names are changed without changing the content.'
    from .external_retrieval import _topic_tokens

    title_terms = " ".join(_topic_tokens(paper_title, 6))
    concepts = " ".join(str(item) for item in (getattr(claim, "concepts", None) or []))
    claim_terms = str(getattr(claim, "keywords_en", "") or "")
    claim_type = str(getattr(claim, "claim_type", "") or getattr(claim, "claim_id", "") or "")
    fallback = DEFAULT_CLAIM_QUERY_EN.get(claim_type, "")

    facets: list[tuple[str, str]] = []
    if concepts:
        facets.append(("construct", f"{title_terms} {concepts}"))
    if claim_terms:
        facets.append(("claim", f"{title_terms} {claim_terms}"))
    theory = str(getattr(claim, "theory", "") or "").replace("_", " ").strip()
    if theory:
        facets.append(("theory", f"{title_terms} {theory}"))
    mechanism = str(getattr(claim, "mechanism", "") or "").strip()
    if mechanism:
        facets.append(("mechanism", f"{title_terms} {mechanism}"))
    context = str(getattr(claim, "context", "") or "").strip()
    if context:
        facets.append(("context", f"{title_terms} {context}"))
    if fallback:
        facets.append(("claim_type", f"{title_terms} {fallback}"))

    routes: list[dict[str, str]] = []
    seen: set[str] = set()
    for facet, terms in facets:
        query = " ".join(piece for piece in (f"{title_terms} {terms}").split())
        if not title_terms:
            query = " ".join(piece for piece in terms.split())
        query = query[:220]
        if not query or query in seen:
            continue
        seen.add(query)
        routes.append({"route": facet, "query": query})
        if len(routes) >= max(1, int(limit)):
            break
    return routes


def _citation_cache_key(paper_id: str, kind: str, limit: int) -> str:
    return f"s2cite|{kind}|{paper_id}|{limit}"


def citation_graph(
    paper_title: str,
    config: ExternalRetrievalConfig,
    *,
    allow_later_works: bool = False,
    expand_depth: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    'Local citation map: target paper → references → references of highly relevant references.\n\n    Return {"papers": {query: [item]}, "target": {...}, "failures": [...], "requests": n}.\n    The keys of papers are marked as `references:target` / `refs:<cited paper title>`, so that provenance can tell the origin clearly.\n\n    `expand_depth=False` only reads the target paper\'s own references and no longer drills down article by article: when the query path has recalled enough candidates\n    There is no need to type Semantic Scholar again for each reference (that is the most time-consuming part).\n    The default is not to collect later documents "citing this article" (`allow_later_works=False`): they are after the publication of the target paper.\n    The research used to determine the innovativeness of this article is time leakage.'
    result: dict[str, Any] = {"papers": {}, "target": None, "failures": [], "requests": 0}
    s2_config = S2Config.from_external(config)
    title = str(paper_title or "").strip()
    if not title:
        return result
    try:
        # _cached_or_fetch returns (entry list, error) - here is the list, only the 0th entry is taken when it hits.
        hits = _cached_or_fetch(
            "semantic_scholar",
            _citation_cache_key(title, "target", 1),
            config,
            lambda: ([found] if (found := s2_find_by_title(title, s2_config)) else []),
        )[0]
        target = hits[0] if hits else None
    except Exception as exc:  # The failure of the citation path is not fatal. After marking, the query path will take the lead.
        result["failures"].append({"route": ROUTE_CITATION, "query": f"target:{title[:60]}", "error": str(exc)[:200]})
        return result
    if not target:
        result["failures"].append({"route": ROUTE_CITATION, "query": f"target:{title[:60]}", "error": 'Semantic Scholar did not locate the target paper'})
        return result
    result["target"] = target
    target_id = str((target.get("externalIds") or {}).get("DOI") or "") or str(target.get("paperId") or "")
    if not target_id:
        return result

    kinds = ["references"]
    if allow_later_works:
        kinds.append("citations")
    for kind in kinds:
        result["requests"] += 1
        if progress_callback:
            progress_callback(f"Citation path: read target paper {kind}")
        try:
            rows = _cached_or_fetch(
                "semantic_scholar",
                _citation_cache_key(target_id, kind, config.citation_related_limit),
                config,
                lambda k=kind, tid=target_id: (
                    s2_references(tid, config.citation_related_limit, s2_config)
                    if k == "references"
                    else _s2_citations(tid, config.citation_related_limit, s2_config)
                ),
            )[0]
        except Exception as exc:
            result["failures"].append({"route": ROUTE_CITATION, "query": f"{kind}:{target_id}", "error": str(exc)[:200]})
            continue
        result["papers"][f"{kind}:target"] = rows
        if kind != "references" or config.citation_graph_depth <= 0 or not expand_depth:
            continue
        # Depth 1: Take another layer of references around the previous references to form a partial citation map.
        for row in rows[: config.citation_related_limit]:
            row_id = str((row.get("externalIds") or {}).get("DOI") or "") or str(row.get("paperId") or "")
            if not row_id:
                continue
            result["requests"] += 1
            try:
                deeper = _cached_or_fetch(
                    "semantic_scholar",
                    _citation_cache_key(row_id, "references", config.citation_related_limit),
                    config,
                    lambda rid=row_id: s2_references(rid, config.citation_related_limit, s2_config),
                )[0]
            except Exception as exc:
                result["failures"].append({"route": ROUTE_CITATION, "query": f"refs:{row_id}", "error": str(exc)[:200]})
                continue
            result["papers"][f"refs:{str(row.get('title') or row_id)[:80]}"] = deeper
    return result


def _s2_citations(paper_id: str, limit: int, config: S2Config) -> list[dict[str, Any]]:
    from .semantic_scholar import citations as s2_citations

    return s2_citations(paper_id, limit, config)


def semantic_scores(
    claim_terms: set[str],
    query_terms: set[str],
    candidates: dict[str, RelatedWork],
) -> dict[str, float]:
    'Semantic relevance: TF-IDF cosine of claim/search term/research question and candidate title+abstract.\n\n    Only lexical TF-IDF (standard library implementation) is used, because it is available offline, without key, and without vector service.\n    Moreover, the source of each point can be printed and audited - even when the external API is completely down, this path can still be sorted.'
    docs: dict[str, list[str]] = {
        key: _tokens(f"{work.title} {work.abstract}") for key, work in candidates.items()
    }
    if not docs:
        return {}
    document_frequency: dict[str, int] = {}
    for tokens in docs.values():
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    total_docs = len(docs)

    def weight(token: str) -> float:
        return math.log((1 + total_docs) / (1 + document_frequency.get(token, 0))) + 1.0

    claim_vector = {token: weight(token) for token in (claim_terms | query_terms)}
    claim_norm = math.sqrt(sum(value * value for value in claim_vector.values())) or 1.0
    scores: dict[str, float] = {}
    for key, tokens in docs.items():
        if not tokens:
            scores[key] = 0.0
            continue
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        doc_vector = {token: (1 + math.log(count)) * weight(token) for token, count in counts.items()}
        doc_norm = math.sqrt(sum(value * value for value in doc_vector.values())) or 1.0
        shared = set(claim_vector) & set(doc_vector)
        scores[key] = sum(claim_vector[token] * doc_vector[token] for token in shared) / (claim_norm * doc_norm)
    return scores


def passes_time_guard(
    work: RelatedWork,
    target_year: int | None,
    config: ExternalRetrievalConfig,
) -> bool:
    'Time leakage control: prior literature must be before the publication year of the target paper (including tolerance grace).\n\n    Candidates without any year information are **released** and marked `unknown_year` - "receive one less article" when there is insufficient evidence.\n    It is safe to "use the literature of 2026 as a priori for 2019", but throwing it away directly will make all new papers (preprints of that year) useless.'
    if not config.time_leak_guard or not target_year:
        return True
    if work.year is None:
        return True
    return int(work.year) <= int(target_year) + max(0, int(config.year_tolerance))


def rank_features(
    work: RelatedWork,
    claim: Any,
    semantic: float,
    config: ExternalRetrievalConfig,
) -> dict[str, float]:
    'The eight items of information listed according to the requirements are scored one by one, and each item can be printed out separately for verification.'
    title_tokens = set(_tokens(work.title))
    body_tokens = set(_tokens(f"{work.title} {work.abstract}"))
    topic_terms = _as_terms(getattr(claim, "topic_terms", None)) or _as_terms(getattr(claim, "concepts", None))
    construct_terms = _as_terms(getattr(claim, "concepts", None))
    theory_terms = _as_terms(getattr(claim, "theory", ""))
    mechanism_terms = _as_terms(getattr(claim, "mechanism", ""))
    context_terms = _as_terms(getattr(claim, "context", ""))
    claim_type = str(getattr(claim, "claim_type", "") or "")

    features = {
        "topic_relevance": max(_overlap(topic_terms, body_tokens), _overlap(topic_terms, title_tokens)),
        "construct_overlap": max(_overlap(construct_terms, body_tokens), _overlap(construct_terms, title_tokens)),
        "theory_relevance": _overlap(theory_terms, body_tokens),
        "mechanism_relevance": _overlap(mechanism_terms, body_tokens),
        "context_relevance": _overlap(context_terms, body_tokens),
        "semantic_similarity": round(float(semantic or 0.0), 4),
        "citation_relationship": 1.0 if ROUTE_CITATION in work.routes else 0.0,
        # publication year: The most relevant if it falls within the window before the target year, and will be slightly attenuated if it is too old (the success rate of old documents in the same field is low)
        "publication_year": _year_feature(work, config),
    }
    topic_hits = len(topic_terms & (body_tokens | title_tokens))
    features["topic_term_hits"] = float(topic_hits)
    if claim_type == "research_gap" and work.citation_count and work.citation_count >= 50:
        features["topic_relevance"] = min(1.0, features["topic_relevance"] + 0.05)
    return features


def _year_feature(work: RelatedWork, config: ExternalRetrievalConfig) -> float:
    year = work.year
    if not year:
        return 0.35
    return 0.9 if int(year) >= 2015 else 0.7


RANK_WEIGHTS = {
    "topic_relevance": 0.26,
    "construct_overlap": 0.20,
    "theory_relevance": 0.10,
    "mechanism_relevance": 0.10,
    "context_relevance": 0.08,
    "semantic_similarity": 0.18,
    "citation_relationship": 0.05,
    "publication_year": 0.03,
}


def literature_rerank(
    work: RelatedWork,
    claim: Any,
    semantic: float,
    claim_tokens: set[str],
    claim_phrases: list[str],
    topic_tokens: set[str],
    config: ExternalRetrievalConfig,
) -> tuple[float, dict[str, float], dict[str, float]]:
    'Reranker: Eight signal weighted scores + subject word hit bonus + existing word hit score.\n\n    Returns (total score, signal details, word hit details). The subject words are added in a separate column because they are direct evidence of "whether they are talking about the same thing."\n    It is not the same thing as claim word hit ("why find it"), and it cannot explain the ordering if mixed together.'
    features = rank_features(work, claim, semantic, config)
    signal_total = sum(RANK_WEIGHTS[key] * features.get(key, 0.0) for key in RANK_WEIGHTS)
    token_score, token_detail = token_rerank(work, claim_tokens, claim_phrases, topic_tokens)
    # After normalization, the word hit score is of the same magnitude as the signal score (the original score is 0-40, divided by 20 to cap 1)
    normalized_token = min(1.0, token_score / 20.0)
    topic_bonus = min(TOPIC_BONUS_CAP, TOPIC_BONUS_PER_HIT * features.get("topic_term_hits", 0.0))
    total = round(10.0 * (0.55 * signal_total + 0.45 * normalized_token) + topic_bonus, 4)
    return total, features, token_detail


def select_external_evidence_set(
    works: list[RelatedWork],
    claim: Any,
    config: ExternalRetrievalConfig,
    top_k: int,
) -> list[RelatedWork]:
    'Top-K Highly relevant literature. Sorting is determined: total score → citation relationship → other citations → year → title.'
    ordered = sorted(
        works,
        key=lambda work: (
            -work.relevance_score,
            -work.rank_features.get("citation_relationship", 0.0),
            -(work.citation_count or 0),
            -(work.year or 0),
            work.title,
        ),
    )
    return ordered[: max(1, int(top_k))]


def _claim_tokens(claim: Any, paper_title: str) -> tuple[set[str], list[str], set[str]]:
    from .external_retrieval import claim_rerank_vocabulary

    if getattr(claim, "claim_type", "") and getattr(claim, "keywords_en", ""):
        return claim_rerank_vocabulary(str(claim.keywords_en), "claim_extractor", paper_title)
    from .external_retrieval import claim_retrieval_terms

    rule = getattr(claim, "rule", None)
    if rule is not None:
        terms, _, origin = claim_retrieval_terms(rule, ExternalRetrievalConfig())
        return claim_rerank_vocabulary(terms, origin, paper_title, rule)
    return claim_rerank_vocabulary(str(getattr(claim, "keywords_en", "")), "override", paper_title)


def run_literature_search(
    claims: list[Any],
    paper_title: str,
    config: ExternalRetrievalConfig,
    *,
    paper_abstract: str = "",
    research_question: str = "",
    target_year: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    fetcher: Callable[[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    'Three-way recall → merge and deduplication → time filtering → rearrangement → External Evidence Set.\n\n    `candidate_literature_pool` in the returned structure is the candidate pool (with provenance) after merging and deduplication.\n    `external_evidence_set` is the Top-K of each claim, and `queries` is the complete retrieval process log.'
    cfg = config
    started = time.time()
    result: dict[str, Any] = {
        "enabled": cfg.enabled,
        "queries": [],
        "routes_used": [],
        "candidate_literature_pool": [],
        "external_evidence_set": [],
        "per_claim": [],
        "time_filtered": 0,
        "self_excluded": 0,
        "failures": [],
        "citation_graph": {"requests": 0, "target": None},
        "pool_size": 0,
        "duration_seconds": 0.0,
        "stop_reason": None,
    }
    if not cfg.enabled or not claims:
        result["stop_reason"] = "disabled" if not cfg.enabled else "no_claims"
        return result

    pool: dict[str, RelatedWork] = {}
    claim_meta: dict[str, Any] = {}

    def register(item: dict[str, Any], claim: Any, query: str, route: str, relationship: str = "") -> None:
        title = str(item.get("title") or "").strip()
        if not title or is_self_reference(title, paper_title):
            if title:
                result["self_excluded"] += 1
            return
        doi = str((item.get("externalIds") or {}).get("DOI") or "").strip()
        key = doi.lower() if doi else title.lower()
        work = pool.get(key)
        if work is None:
            if len(pool) >= cfg.candidate_pool_limit:
                return
            work = RelatedWork(
                title=title,
                year=item.get("year"),
                venue=str(item.get("venue") or ""),
                authors=[str(a.get("name")) for a in (item.get("authors") or []) if isinstance(a, dict)][:6],
                abstract=str(item.get("abstract") or ""),
                citation_count=item.get("citationCount"),
                external_ids=item.get("externalIds") or {},
                source_query=query,
                paper_id=str(item.get("paperId") or doi or key),
                doi=doi,
                retrieval_source=str(item.get("retrieval_source") or ""),
                retrieval_time=now_iso(),
                citation_relationship=relationship,
            )
            pool[key] = work
        else:
            # Multi-path hits do not add new records, but only accumulate provenance (whichever paths recall it).
            if item.get("citationCount") and not work.citation_count:
                work.citation_count = item.get("citationCount")
            if item.get("abstract") and not work.abstract:
                work.abstract = str(item.get("abstract"))
        if route not in work.routes:
            work.routes.append(route)
        if query and query not in work.source_query:
            work.source_query = f"{work.source_query}; {query}"[:400]
        if relationship and not work.citation_relationship:
            work.citation_relationship = relationship
        if claim.claim_id not in work.claim_ids:
            work.claim_ids.append(claim.claim_id)

    for claim in claims:
        claim_meta[claim.claim_id] = claim
        query_routes = expand_query_routes(claim, paper_title, cfg.max_queries) if cfg.query_route_enabled else []
        if not query_routes and cfg.query_route_enabled:
            fallback = str(getattr(claim, "keywords_en", "") or "")
            query_routes = [{"route": "claim", "query": fallback[:220]}] if fallback else []
        claim_log: dict[str, Any] = {
            "claim_id": claim.claim_id,
            "claim_type": str(getattr(claim, "claim_type", "") or ""),
            "source_evidence_id": str(getattr(claim, "source_evidence_id", "") or ""),
            "claim_text": str(getattr(claim, "text", ""))[:200],
            "query_routes": [],
            "merged_candidates": 0,
            "top_k": [],
        }
        for entry in query_routes:
            query = entry["query"]
            claim_log["query_routes"].append({"facet": entry["route"], "query": query, "provider": None, "returned": 0})
            if progress_callback:
                progress_callback(f"Query Retrieval（{claim.claim_id}/{entry['route']}）：{query[:40]}")
            items, error, provider = search_external(query, cfg, fetcher=fetcher)
            claim_log["query_routes"][-1]["provider"] = provider
            claim_log["query_routes"][-1]["returned"] = len(items)
            result["queries"].append(
                {
                    "claim_id": claim.claim_id,
                    "route": ROUTE_QUERY,
                    "facet": entry["route"],
                    "query": query,
                    "provider": provider,
                    "returned": len(items),
                }
            )
            if error:
                result["failures"].append({"claim_id": claim.claim_id, "route": ROUTE_QUERY, "query": query, "error": error[:200]})
                continue
            for item in items:
                item = {**item, "retrieval_source": provider or ""}
                register(item, claim, query, ROUTE_QUERY)
        result["per_claim"].append(claim_log)

    if ROUTE_QUERY in {route for work in pool.values() for route in work.routes}:
        result["routes_used"].append(ROUTE_QUERY)

    # Road 2: Citation graph (hanging on the first claim, the citation relationship does not belong to the semantic retrieval of a certain claim).
# When
    # query has recalled enough candidates, it will only read the target paper's own references and no longer drill down one by one:
    # Article-by-article references are the most expensive part of this link (one Semantic Scholar request per article).
    if cfg.citation_route_enabled:
        expand_depth = len(pool) < max(4, cfg.max_evidence_blocks)
        if progress_callback:
            progress_callback('Citation Retrieval: Read target paper references')
        graph = citation_graph(
            paper_title,
            cfg,
            allow_later_works=cfg.allow_citing_later_works,
            expand_depth=expand_depth,
            progress_callback=progress_callback,
        )
        result["citation_graph"] = {
            "requests": graph["requests"],
            "target": (graph.get("target") or {}).get("title") if isinstance(graph.get("target"), dict) else None,
            "groups": len(graph["papers"]),
            "expanded_depth": expand_depth,
        }
        result["failures"].extend(graph["failures"])
        anchor = claims[0]
        for label, rows in graph["papers"].items():
            # The citation direction must be clearly remembered: refs: = references of references (earlier a priori), references: = references of this article.
            relationship = (
                "reference_of_reference" if label.startswith("refs:")
                else "target_reference" if label.startswith("references:")
                else "cites_target"
            )
            for item in rows:
                item = {**item, "retrieval_source": "semantic_scholar"}
                register(item, anchor, label, ROUTE_CITATION, relationship=relationship)
        if graph["papers"]:
            result["routes_used"].append(ROUTE_CITATION)

    result["pool_size"] = len(pool)

    # Time leak filtering: filter first and then rearrange to avoid "post-hoc explanation why a certain article did not enter the pool"
    kept: dict[str, RelatedWork] = {}
    for key, work in pool.items():
        if passes_time_guard(work, target_year, cfg):
            kept[key] = work
        else:
            result["time_filtered"] += 1
    pool = kept

    # claim-by-claim rearrangement + semantic path
    top_k = cfg.top_k if cfg.top_k > 0 else 10
    evidence_set: list[RelatedWork] = []
    # A document may be recalled by multiple claims at the same time: only the "one with the highest score" is retained, otherwise the score of the later claim will overwrite the previous one.
    # The relevance in the External Evidence Set is disconnected from which claim it is actually supported by (especially fatal when judging novel).
    best: dict[int, tuple[float, dict[str, float], dict[str, float], float]] = {}
    for claim in claims:
        claim_tokens, claim_phrases, topic_tokens = _claim_tokens(claim, paper_title)
        setattr(claim, "topic_terms", sorted(topic_tokens))
        query_terms: set[str] = set()
        for entry in result["per_claim"][claims.index(claim)]["query_routes"]:
            query_terms |= set(_tokens(entry["query"]))
        query_terms |= set(_tokens(research_question)) | set(_tokens(paper_abstract)) | topic_tokens
        semantic_map = semantic_scores(claim_tokens, query_terms, pool) if cfg.semantic_route_enabled else {}
        if semantic_map and ROUTE_SEMANTIC not in result["routes_used"]:
            result["routes_used"].append(ROUTE_SEMANTIC)
        scored: list[RelatedWork] = []
        for key, work in pool.items():
            if claim.claim_id not in work.claim_ids:
                continue
            semantic = float(semantic_map.get(key, 0.0))
            total, features, token_detail = literature_rerank(
                work, claim, semantic, claim_tokens, claim_phrases, topic_tokens, cfg
            )
            selection_score = round(
                total
                + ROUTE_WEIGHT.get(ROUTE_CITATION, 0.0) * features.get("citation_relationship", 0.0)
                + (ROUTE_WEIGHT.get(ROUTE_SEMANTIC, 0.0) if semantic >= 0.15 else 0.0),
                4,
            )
            previous = best.get(id(work))
            if previous is None or selection_score > previous[0]:
                best[id(work)] = (
                    selection_score,
                    features,
                    {**token_detail, "matched_claim": claim.claim_id, "claim_type": str(getattr(claim, "claim_type", "") or "")},
                    total,
                )
                work.relevance_score = selection_score
            scored.append(work)
        chosen = select_external_evidence_set(scored, claim, cfg, cfg.max_evidence_blocks or top_k)
        for work in chosen:
            if work not in evidence_set:
                evidence_set.append(work)

    # After selecting, write the characteristics of "the highest score" back to the object, so that the scores in the External Evidence Set are consistent with the explanation.
    for work in evidence_set:
        entry = best.get(id(work))
        if entry is None:
            continue
        work.relevance_score, work.rank_features, work.rerank_detail, work.rerank_score = entry
        result["per_claim"][claims.index(claim)]["merged_candidates"] = len(scored)
        result["per_claim"][claims.index(claim)]["top_k"] = [
            {
                "title": work.title,
                "year": work.year,
                "doi": work.doi,
                "routes": list(work.routes),
                "relevance_score": work.relevance_score,
                "why": {k: round(v, 3) for k, v in work.rank_features.items()},
            }
            for work in chosen
        ]

    external_set = select_external_evidence_set(evidence_set, None, cfg, max(top_k, len(evidence_set)))
    result["external_evidence_set"] = [work.to_dict() for work in external_set]
    result["candidate_literature_pool"] = [work.to_dict() for work in pool.values()]
    result["routes_used"] = [route for route in ROUTE_ORDER if route in result["routes_used"]]
    result["stop_reason"] = "ok" if external_set else ("time_filtered_all" if result["time_filtered"] else "no_candidates")
    result["duration_seconds"] = round(time.time() - started, 3)
    if not external_set:
        result["external_evidence_unavailable"] = True
    return result


def attach_external_evidence_ids(result: dict[str, Any]) -> list[dict[str, Any]]:
    'Assign a citable external evidence ID (XEV-1..N) to each paper in the External Evidence Set and write back the results.'
    works = list(result.get("external_evidence_set") or [])
    for index, work in enumerate(works, start=1):
        work["evidence_id"] = f"XEV-{index}"
    result["external_evidence_set"] = works
    return works


def external_evidence_blocks_from_set(
    result: dict[str, Any],
    paper_id: str,
) -> list[Evidence]:
    'External Evidence Set → Quotable evidence block. For each item, write clearly "why it is recalled" and "which claim it corresponds to".'
    import hashlib

    blocks: list[Evidence] = []
    for index, work in enumerate(attach_external_evidence_ids(result), start=1):
        routes = "/".join(str(route) for route in (work.get("routes") or [])) or "unknown"
        claim_ids = ",".join(str(item) for item in (work.get("claim_ids") or [])) or "NA"
        features = work.get("rank_features") or {}
        why = "；".join(
            f"{RANK_FEATURE_LABELS.get(key, key)}={value:.2f}"
            for key, value in sorted(features.items(), key=lambda row: -float(row[1] or 0))[:4]
        )
        lines = [
            (
                f"【External literature {index}｜{work.get('evidence_id')}｜{work.get('year') or 'year unavailable'}｜"
                f"{work.get('venue') or 'venue unavailable'}｜He cited {work.get('citation_count')}]"
            ),
            f"Title: {work.get('title')}",
            f"DOI: {work.get('doi') or 'None'} (metadata check: {work.get('metadata_check') or 'unchecked'})",
            f"Recall path: {routes}; corresponding claim: {claim_ids}; search formula: {str(work.get('source_query') or '')[:120]}",
            f"Correlation: {work.get('relevance_score')} ({'，'.join(piece for piece in [why] if piece)})",
        ]
        abstract = str(work.get("abstract") or "").strip()
        lines.append(f"Summary: {abstract[:600] if abstract else '（No abstract available）'}")
        summary = work.get("summary") if isinstance(work.get("summary"), dict) else {}
        for key, label in (("method", 'method'), ("findings", 'found'), ("similar", 'similar'), ("different", 'different')):
            value = str(summary.get(key) or "").strip()
            if value:
                lines.append(f"{label}：{value}")
        text = "\n".join(lines)
        blocks.append(
            Evidence(
                evidence_id=str(work.get("evidence_id") or f"XEV-{index}"),
                paper_id=paper_id,
                section=EXTERNAL_EVIDENCE_SECTION,
                page=0,
                span=(0, len(text)),
                text=text,
                text_hash=hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
                source_type="external",
            )
        )
    return blocks


# The section name of the external document evidence block. In agents.py, it is used to determine "whether this article is an external document" + source_type.
EXTERNAL_EVIDENCE_SECTION = 'External documentation (multiple recall)'


def coverage_of(result: dict[str, Any]) -> dict[str, Any]:
    'prior literature coverage: Harness relies on Novelty Agent to decide whether to lower the confidence when judging novel.'
    works = list(result.get("external_evidence_set") or [])
    prior = [work for work in works if work.get("year") is not None]
    routes = {route for work in works for route in (work.get("routes") or [])}
    top = max((float(work.get("relevance_score") or 0.0) for work in works), default=0.0)
    return {
        "paper_count": len(works),
        "prior_paper_count": len(prior),
        "routes": sorted(routes),
        "top_relevance": round(top, 4),
        "sufficient": len(works) >= 1 and bool(works),
    }

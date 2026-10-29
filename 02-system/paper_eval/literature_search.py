"""External Literature Retrieval：多路召回 → 去重 → 时间泄漏过滤 → Reranker → External Evidence Set。

三路召回（**不允许**只搜论文标题）：

1. **Query Retrieval**：每个 claim 生成 3-5 条检索式（核心构念 / 理论 / 机制 / 研究情境各一路），
   走 Semantic Scholar 检索（search_external，源可换）。
2. **Citation Retrieval**：先定位目标论文，读它的 references；再围绕高相关参考文献取 references，
   形成局部 citation graph。引文关系能捞到"标题里没有一个共同词、但确实是同一条理论脉络"的文献。
3. **Semantic Retrieval**：用 claim + 摘要 + 研究问题 与候选文献摘要算词法语义相关度（TF-IDF 余弦），
   不额外打 API，所以它总是可用，保证外部 API 全挂时仍有一路召回。

时间泄漏控制：验证某篇论文创新性只能用**目标论文发表年份之前**已发表的研究，所以候选池在
rerank 之前就按年份过滤（`time_leak_guard`），漏在池外比污染判断更安全。

检索逻辑全部住在这个模块里，Agent 只消费 `external_evidence_set`，不自己发检索。
"""

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

# 论文主题词在候选里的命中加成。主题词是"这篇论文研究什么"，比 claim 通用词更有区分度：
# 实测缺了这项时，"fairness review" 这类泛泛综述会靠通用词命中考前（真正同主题的先验被压下去）。
TOPIC_BONUS_PER_HIT = 0.3
TOPIC_BONUS_CAP = 0.6

SEMANTIC_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "is", "are", "be",
    "as", "by", "that", "this", "we", "our", "it", "its", "from", "at", "how", "when", "what",
    "are", "was", "were", "has", "have", "had", "not", "but", "can", "may", "also", "than",
    "which", "who", "whom", "their", "they", "them", "these", "those", "such", "into", "over",
    "more", "most", "other", "some", "any", "all", "new", "first", "study", "review", "paper",
    "research", "findings", "results", "analysis", "based", "using", "used", "present", "show",
    "研究", "论文", "本文", "的", "与", "和", "及", "或",
}

# rerank 各信号的权重，全部来自配置，改了就是改了标尺，必须进报告。
RANK_FEATURE_LABELS = {
    "topic_relevance": "主题相关性",
    "construct_overlap": "核心构念重合度",
    "theory_relevance": "理论相关性",
    "mechanism_relevance": "机制相关性",
    "context_relevance": "研究情境相关性",
    "semantic_similarity": "语义相似度",
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
    """把一个 claim 展开成 3-5 条检索式：核心构念 / 理论 / 机制 / 研究情境 / claim 英译。

    只发一条「论文标题」式的检索式是不够的：同一个创新主张可能在别的构念名、别的理论名下
    已经有人做过，多路检索式就是为了撞上那些换名不换事的文献。
    """
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
    """局部引文图：目标论文 → references → 高相关参考文献的 references。

    返回 {"papers": {query: [item]}, "target": {...}, "failures": [...], "requests": n}。
    papers 的键标成 `references:target` / `refs:<被引文献标题>`，让 provenance 说得清来路。

    `expand_depth=False` 时只读目标论文自己的参考文献，不再逐篇下钻：query 路已经召回够多候选时
    没必要为每篇参考文献再打一次 Semantic Scholar（那是最费时的一段）。
    默认**不**收集"引用本文"的后发文献（`allow_later_works=False`）：它们是目标论文发表之后的
    研究，用来判定本文创新性就是时间泄漏。
    """
    result: dict[str, Any] = {"papers": {}, "target": None, "failures": [], "requests": 0}
    s2_config = S2Config.from_external(config)
    title = str(paper_title or "").strip()
    if not title:
        return result
    try:
        # _cached_or_fetch 返回 (条目列表, 错误)——这里是列表，命中时才取第 0 条。
        hits = _cached_or_fetch(
            "semantic_scholar",
            _citation_cache_key(title, "target", 1),
            config,
            lambda: ([found] if (found := s2_find_by_title(title, s2_config)) else []),
        )[0]
        target = hits[0] if hits else None
    except Exception as exc:  # 引文路失败不致命，标记后由 query 路兜底
        result["failures"].append({"route": ROUTE_CITATION, "query": f"target:{title[:60]}", "error": str(exc)[:200]})
        return result
    if not target:
        result["failures"].append({"route": ROUTE_CITATION, "query": f"target:{title[:60]}", "error": "Semantic Scholar 未定位到目标论文"})
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
            progress_callback(f"引文路：读取目标论文 {kind}")
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
        # 深度 1：围绕前几篇参考文献再取一层 references，形成局部引文图。
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
    """语义相关度：claim/检索式/研究问题 与 候选标题+摘要 的 TF-IDF 余弦。

    只做词法 TF-IDF（标准库实现），因为它在离线、无 key、无向量服务时都可用，
    而且每一分的来源都能打印出来审计——外部 API 全挂时这一路仍然给得出排序。
    """
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
    """时间泄漏控制：prior literature 必须在目标论文发表年份之前（含 tolerance 宽限）。

    没有任何年份信息的候选**放行**并标 `unknown_year`——证据不足时"少收一篇"比
    "把 2026 年的文献当 2019 年的先验"安全，但直接丢掉会让新论文（当年预印本）全部落空。
    """
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
    """按需求列出的八项信息逐项打分，每一项都能单独打印出来核对。"""
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
        # publication year：落在目标年份前一档窗口内最相关，太老略微衰减（同一领域老文献成功率低）
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
    """Reranker：八项信号加权分 + 主题词命中加成 + 既有词命中分。

    返回 (总分, 信号明细, 词命中明细)。主题词加成单列，因为它是"是不是讲同一件事"的直接证据，
    与 claim 词命中（"为什么找它"）不是一回事，混在一起就没法解释排序了。
    """
    features = rank_features(work, claim, semantic, config)
    signal_total = sum(RANK_WEIGHTS[key] * features.get(key, 0.0) for key in RANK_WEIGHTS)
    token_score, token_detail = token_rerank(work, claim_tokens, claim_phrases, topic_tokens)
    # 词命中分归一化后与信号分同量级（原分数量级 0-40，除以 20 封顶 1）
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
    """Top-K 高相关文献。排序确定：总分 → 引文关系 → 他引数 → 年份 → 标题。"""
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
    """三路召回 → 合并去重 → 时间过滤 → 重排 → External Evidence Set。

    返回结构里 `candidate_literature_pool` 是合并去重后的候选池（带 provenance），
    `external_evidence_set` 是每个 claim 的 Top-K，`queries` 是完整的检索过程日志。
    """
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
            # 多路命中不新增记录，只累积 provenance（哪几条路都召回了它）。
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

    # 路 2：引文图（挂在第一个 claim 上，引文关系不属于某一条 claim 的语义检索）。
    # query 路已经召回足够多候选时只读目标论文自己的参考文献，不再逐篇下钻：
    # 逐篇 references 是这条链路最贵的部分（每篇一次 Semantic Scholar 请求）。
    if cfg.citation_route_enabled:
        expand_depth = len(pool) < max(4, cfg.max_evidence_blocks)
        if progress_callback:
            progress_callback("Citation Retrieval：读取目标论文 references")
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
            # 引文方向必须记清楚：refs: = 参考文献的参考文献（更早的先验），references: = 本文参考文献。
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

    # 时间泄漏过滤：先过滤再重排，避免"post-hoc 解释为什么某篇没进池"
    kept: dict[str, RelatedWork] = {}
    for key, work in pool.items():
        if passes_time_guard(work, target_year, cfg):
            kept[key] = work
        else:
            result["time_filtered"] += 1
    pool = kept

    # 逐 claim 重排 + 语义路
    top_k = cfg.top_k if cfg.top_k > 0 else 10
    evidence_set: list[RelatedWork] = []
    # 一篇文献可能同时被多条 claim 召回：只保留"最高分的那次"，否则后一条 claim 的分数会覆盖前一条，
    # External Evidence Set 里的相关度就与它实际被哪条 claim 支持脱节（判 novel 时尤其致命）。
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

    # 选完再把"最高分那次"的特征写回对象，让 External Evidence Set 里的分数与解释一致
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
    """给 External Evidence Set 里每篇论文分配可引用的外部证据 ID（XEV-1..N），写回结果。"""
    works = list(result.get("external_evidence_set") or [])
    for index, work in enumerate(works, start=1):
        work["evidence_id"] = f"XEV-{index}"
    result["external_evidence_set"] = works
    return works


def external_evidence_blocks_from_set(
    result: dict[str, Any],
    paper_id: str,
) -> list[Evidence]:
    """External Evidence Set → 可引用的证据块。每条写清"为什么召回它"和"对应哪条 claim"。"""
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
                f"【外部文献 {index}｜{work.get('evidence_id')}｜{work.get('year') or '年份未知'}｜"
                f"{work.get('venue') or '期刊未知'}｜他引 {work.get('citation_count')}】"
            ),
            f"标题：{work.get('title')}",
            f"DOI：{work.get('doi') or '无'}（元数据校验：{work.get('metadata_check') or 'unchecked'}）",
            f"召回路径：{routes}；对应 claim：{claim_ids}；检索式：{str(work.get('source_query') or '')[:120]}",
            f"相关度：{work.get('relevance_score')}（{'，'.join(piece for piece in [why] if piece)}）",
        ]
        abstract = str(work.get("abstract") or "").strip()
        lines.append(f"摘要：{abstract[:600] if abstract else '（该文献无摘要）'}")
        summary = work.get("summary") if isinstance(work.get("summary"), dict) else {}
        for key, label in (("method", "方法"), ("findings", "发现"), ("similar", "相似"), ("different", "不同")):
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


# 外部文献证据块的 section 名。agents.py 里判断"这条是不是外部文献"用的就是它 + source_type。
EXTERNAL_EVIDENCE_SECTION = "外部文献（多路召回）"


def coverage_of(result: dict[str, Any]) -> dict[str, Any]:
    """prior literature coverage：Novelty Agent 判 novel 时 Harness 要靠它决定是否降置信度。"""
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

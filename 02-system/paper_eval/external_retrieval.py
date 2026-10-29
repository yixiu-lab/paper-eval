"""外部文献检索：OpenAlex / Semantic Scholar + claim-conditioned retrieval + 检索预算 + 结构化摘要。

设计要点（对齐 CLAIMCHECK 那批要求）：

- **Claim-conditioned Retrieval**：检索式按 claim（= 评分规则）生成，不是整篇论文做一次 RAG。
  `build_claim_query()` 把规则的指标名、定义、观察点与论文标题关键词合成检索式。
- **检索源**：默认 Crossref + OpenAlex + Semantic Scholar bulk（都免 key，配置里 `providers` 可多源）。
  Semantic Scholar 的 /paper/search 需要 key（无 key 匿名共享池实测常年 429），所以免 key 走
  /paper/search/bulk（另一个池，匿名实测稳定）；存在 `SEMANTIC_SCHOLAR_API_KEY` 时自动追加
  /paper/search 作为相关性排序源。
  所有源都归一化成同一字段形状，下游重排/去重/摘要逻辑不区分来源。
- **Rerank**：先用外部源的 relevance 排序作先验，再用本地确定性打分重排
  （claim 关键词命中 + 观察点命中 + 摘要长度），每一分的来源都可解释，不依赖黑箱模型。
- **检索预算与停止条件**：`min_queries`（至少发出多少条不同的 claim 检索式）、
  `max_candidates`（候选总数上限）、`max_queries`、`error_budget`，
  以及"重排后够用即停"。避免 Agent 随手搜两篇就下结论。
- **缓存**：按「检索源 + 检索式」做磁盘缓存，重复运行不重复打外部 API。
- **失败不致命**：外部 API 不可用时返回带 `failures` 的结构化结果，不抛异常、不中断评分。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import yaml
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from .llm import AgentRuntimeConfig, call_plain_json
from .models import Evidence
from .rules import IndicatorRule, RuleProfile, rule_in_scope
from .semantic_scholar import SEARCH_FIELDS, S2Config
from .semantic_scholar import search_bulk as s2_bulk
from .semantic_scholar import search_relevance as s2_relevance

# S2 的端点、key、限流、重试、缓存都在 paper_eval/semantic_scholar.py；
# 这里保留 API_FIELDS 这个名字，免得动到已有自检脚本。
API_FIELDS = SEARCH_FIELDS
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
CROSSREF_SELECT = "DOI,title,abstract,issued,container-title,author,is-referenced-by-count"
# 默认多源冗余：任一源成功即可。Crossref 免 key 且限流宽松（实测稳定），
# OpenAlex 匿名共享池会按 IP 限流（实测 429/503、Retry-After 35-60 秒）；
# semantic_scholar_bulk 免 key 可用，排在最后兜底。有 S2 key 时 resolved_providers 再补 /paper/search 相关性源。
DEFAULT_PROVIDERS = ("crossref", "openalex", "semantic_scholar_bulk")
# 默认 claim 集：判断确实依赖「外部文献里有没有人做过」的那几条规则。
DEFAULT_CLAIM_RULE_IDS = (
    "A-LITERATURE-POSITIONING-V1",
    "A-GAP-PUZZLE-V1",
    "B-DIALOGUE-ADVANCE-V1",
    "B-CONCEPT-MECHANISM-NOVELTY-V1",
)
STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "is", "are", "be",
    "研究", "论文", "本文", "是否", "判断", "评价", "的", "与", "和", "及", "或",
    # 疑问词必须剔除：论文标题常写成「Challenge or hindrance? How and when ...」，
    # 这些词占掉主题名额，会把真正有区分度的 "artificial intelligence" 挤出检索式。
    "how", "when", "what", "why", "which", "who", "does", "do", "did", "can", "could", "should",
}
# 规则库是中文写的，但 OpenAlex / S2 的语料以英文为主：拿中文当检索式会检索到中文文献、
# 重排分全为 0。所以 claim 另配一份英文检索词，同时用作检索式和重排词表。
# 没配英文的规则退回规则原文（中文），并在 queries 里标 origin=rule_text 便于排查。
DEFAULT_CLAIM_QUERY_EN = {
    "A-LITERATURE-POSITIONING-V1": "literature positioning theoretical conversation prior research dialogue",
    "A-GAP-PUZZLE-V1": "research gap puzzle contradictory findings unresolved question",
    "B-DIALOGUE-ADVANCE-V1": "challenging assumptions integrating conflicting findings theoretical advancement",
    "B-CONCEPT-MECHANISM-NOVELTY-V1": "novel construct new mechanism theoretical proposition conceptual contribution",
}
# 默认把外部文献当证据插进阅读包的规则：判断「是否已有人做过 / 相对既有文献的增量」的那几条。
DEFAULT_EVIDENCE_RULE_IDS = (
    "A-LITERATURE-POSITIONING-V1",
    "A-GAP-PUZZLE-V1",
    "B-CONCEPT-MECHANISM-NOVELTY-V1",
    "B-DIALOGUE-ADVANCE-V1",
    "B-THEORY-INTEGRATION-V1",
    "B-INTERESTINGNESS-V1",
)
# 从 paper_context 抽 claim 的字段。claim 用论文自己的研究问题/缺口/贡献，比用规则定义更贴。
CLAIM_FIELDS = (
    ("RQ", "研究问题", ("research_question", "central_problem")),
    ("GAP", "研究缺口", ("research_gap",)),
    ("CONTRIB", "主要贡献", ("main_contribution",)),
    ("CLAIM", "核心主张", ("core_claim",)),
)
CONTEXT_PLACEHOLDER = "待 Agent 结合全文判断"
# 翻译失败时的确定性兜底：claim 类型 → 通用英文检索词（主题约束仍由论文标题提供）。
DEFAULT_CONTEXT_CLAIM_EN = {
    "RQ": "research question study investigates examines",
    "GAP": "research gap puzzle contradictory findings unresolved question",
    "CONTRIB": "theoretical contribution extends advances prior literature",
    "CLAIM": "core claim main argument proposition findings",
}
# 这些词几乎每篇论文摘要都有，留在重排词表里只会把排序推向"泛泛相关"。
# 只从重排词表里剔除，检索式仍保留（OpenAlex 的检索需要这些词）。
GENERIC_ACADEMIC_TOKENS = {
    "research", "study", "studies", "prior", "previous", "existing", "findings", "theoretical",
    "theory", "question", "questions", "paper", "article", "contribution", "contributions",
    "evidence", "results", "analysis", "approach", "based", "using", "review", "future",
}


@dataclass(frozen=True)
class ExternalRetrievalConfig:
    enabled: bool = False
    providers: list[str] = field(default_factory=lambda: list(DEFAULT_PROVIDERS))
    api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY"
    mailto: str = ""              # OpenAlex 礼貌池邮箱；填了会显著降低被限流概率
    claim_rule_ids: list[str] = field(default_factory=lambda: list(DEFAULT_CLAIM_RULE_IDS))
    claim_query_en: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CLAIM_QUERY_EN))
    # 哪些规则要把外部文献当作证据块插进阅读包（判断「是否已有人做过」的那几条）。
    evidence_rule_ids: list[str] = field(default_factory=lambda: list(DEFAULT_EVIDENCE_RULE_IDS))
    # 检索不到任何外部文献时是否拒绝继续。
    # **默认 false（故障安全）**：外部 API 不可用时评分任务不失败，而是在报告里标
    # `external_evidence_unavailable`，并由 Harness 拦住依赖领域比较的规则不给 satisfied。
    # 想改成"检索不到就别跑"（宁可停也不出降级结果）把它设成 true；此时检索跑在评分之前，
    # 失败还没花掉评分调用，硬门槛也说得过去。
    require_external_evidence: bool = False
    # 插进阅读包的外部文献条数上限（每条会占几百字符输入）。
    max_evidence_blocks: int = 8
    # 限流防护：OpenAlex 未进礼貌池时连续请求会返回 429（实测 4 条查询里失败 2-3 条）。
    request_retries: int = 3
    retry_backoff_seconds: float = 1.5
    min_interval_seconds: float = 0.6
    cache_dir: str | None = None
    timeout_seconds: int = 20
    # 检索预算与停止条件
    min_queries: int = 2          # 至少要发这么多条不同检索式，才允许结束
    max_queries: int = 6
    max_candidates: int = 60      # 候选文献总数上限
    error_budget: int = 3         # 连续失败这么多次就停
    per_query_limit: int = 20     # 单次检索返回上限
    top_k: int = 10               # 重排后保留多少篇
    summary_top_k: int = 5        # 其中几篇生成结构化摘要（0 = 不生成，省调用）
    min_top_score: float = 0.12   # 重排分数达到此值即认为"够用"，可提前停止
    rerank_weight_phrase: float = 3.0
    rerank_weight_token: float = 1.0
    year_from: int | None = None
    year_to: int | None = None
    # 限流防护：OpenAlex 未进礼貌池时突发请求会返回 429（实测 4 条查询失败 2-3 条）。
    request_retries: int = 4
    retry_backoff_seconds: float = 2.0
    min_interval_seconds: float = 1.2
    # 首轮全军覆没（限流风暴）时，冷却后再整轮重试一次，然后才判定失败。
    # 失败的查询不写缓存，所以这一轮一定重新走真实网络。
    empty_retry_seconds: float = 20.0
    # 尊重服务端 Retry-After：超过这个秒数就别等了，直接报出可操作的原因（实测 OpenAlex
    # 匿名共享池会要求等 35-60 秒；用比它短的退避必然失败）。
    max_retry_after_seconds: float = 30.0

    # --- 外部文献检索 + 外部证据验证（第二阶段增量）---
    # external_evidence_verification_enabled：创新性验证 Agent + Harness 外部证据校验的开关。
    # 检索（enabled）与验证分两级：只要外部文献就开 enabled，还要做"相对既有研究新增了什么"的
    # 比较判断（每条 claim 一次模型调用 + Crossref 校验）再开这个。关掉它时依赖领域比较的规则
    # 会被 Harness 拦住不给 satisfied，而不是让模型凭自己的知识断言"没人做过"。
    external_evidence_verification_enabled: bool = True
    # Claim Extractor：先用 5 类 claim 抽提，再逐条检索（关掉则退回旧的画像字段抽提）。
    claim_extraction_enabled: bool = True
    # 多路召回开关：query = 每个 claim 3-5 条检索式；citation = 目标论文 references 局部引文图；
    # semantic = claim/摘要/研究问题与候选摘要的语义相关度重排。
    query_route_enabled: bool = True
    citation_route_enabled: bool = True
    semantic_route_enabled: bool = True
    # 时间泄漏控制：prior literature 只取目标论文发表年份之前（含宽限期）的研究。
    time_leak_guard: bool = True
    year_tolerance: int = 0
    # citation 路：围绕目标论文 references 继续取引文关系，以及是否把引用本文的后发文献纳入候选。
    citation_graph_depth: int = 1
    citation_related_limit: int = 8
    allow_citing_later_works: bool = False
    # 候选池 → External Evidence Set
    candidate_pool_limit: int = 40
    # Crossref 元数据校验（DOI/标题/年份一致性）
    crossref_verify: bool = True
    crossref_max_checks: int = 8
    # 判断「依赖领域比较」的规则至少要有多少篇 prior literature，才允许给 satisfied
    min_external_papers: int = 2
    # Novelty Agent 判 novel 但 prior literature 覆盖不足时，confidence 降到这个上限
    low_coverage_confidence_cap: float = 0.4


@dataclass
class RelatedWork:
    """一篇候外部文献。加 provenance 字段后即可当 External Evidence Repository 的论文记录用。"""

    title: str
    year: int | None
    venue: str
    authors: list[str]
    abstract: str
    citation_count: int | None
    external_ids: dict[str, Any]
    source_query: str
    rerank_score: float = 0.0
    rerank_detail: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, str] | None = None
    # provenance：这篇文献从哪条路、哪条检索式、什么时刻被召回，用来做可复现审计
    paper_id: str = ""
    doi: str = ""
    retrieval_source: str = ""
    retrieval_time: str = ""
    routes: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    rank_features: dict[str, float] = field(default_factory=dict)
    citation_relationship: str = ""
    evidence_id: str = ""
    # 元数据校验结果（Crossref）：ok / mismatch / unchecked
    metadata_check: str = "unchecked"
    metadata_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "authors": self.authors,
            "doi": self.doi,
            "abstract": self.abstract[:800],
            "citation_count": self.citation_count,
            "external_ids": self.external_ids,
            "source_query": self.source_query,
            # retrieval_query 与 source_query 同义：provenance 对外统一叫 retrieval_query，
            # source_query 是旧字段名，报告与旧自检还在读它，两个都留着。
            "retrieval_query": self.source_query,
            "retrieval_source": self.retrieval_source,
            "retrieval_time": self.retrieval_time,
            "routes": list(self.routes),
            "claim_ids": list(self.claim_ids),
            "relevance_score": round(self.relevance_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "rank_features": {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in self.rank_features.items()},
            "rerank_detail": {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in self.rerank_detail.items()},
            "citation_relationship": self.citation_relationship,
            "evidence_id": self.evidence_id,
            "metadata_check": self.metadata_check,
            "metadata_notes": list(self.metadata_notes),
            "summary": self.summary,
        }


@dataclass
class RetrievalClaim:
    """一条检索诉请：claim 来自论文自己的研究问题/缺口/贡献，检索词用英文（语料是英文）。

    claim_extraction 模块会补齐 claim_type / source_evidence_id 等字段；
    这里全部给默认值，旧的规则兜底路径（claims_from_rules）不受影响。
    """

    claim_id: str
    label: str          # 中文标签，进报告
    text: str           # claim 原文（中文），进报告与排查
    keywords_en: str    # 英文检索词，用于检索式与重排
    source: str         # paper_context | rule | fallback | claim_extractor
    claim_type: str = ""
    source_evidence_id: str = ""
    concepts: list[str] = field(default_factory=list)
    theory: str = ""
    mechanism: str = ""
    context: str = ""
    publication_year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "label": self.label,
            "text": self.text[:200],
            "keywords_en": self.keywords_en,
            "origin": self.source,
            "claim_type": self.claim_type,
            "source_evidence_id": self.source_evidence_id,
            "concepts": list(self.concepts),
            "theory": self.theory,
            "mechanism": self.mechanism,
            "context": self.context,
            "publication_year": self.publication_year,
        }


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}|[\u4e00-\u9fff]{2,}", str(text).lower())
    return [word for word in words if word not in STOPWORDS]


def context_claim_texts(paper_context: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    """从 paper_context 的论文画像里抽出可检索的 claim：(claim_id, 中文标签, 中文原文)。"""
    profile = (paper_context or {}).get("paper_profile") or {}
    if not isinstance(profile, dict):
        return []
    claims: list[tuple[str, str, str]] = []
    for claim_id, label, keys in CLAIM_FIELDS:
        text = ""
        for key in keys:
            value = str(profile.get(key) or "").strip()
            if value and value != CONTEXT_PLACEHOLDER:
                text = value
                break
        if text:
            claims.append((claim_id, label, text))
    return claims


def translate_claims_to_english(
    claims: list[tuple[str, str, str]],
    paper_title: str,
    runtime_config: AgentRuntimeConfig | None,
) -> dict[str, str]:
    """把中文 claim 改写成英文检索词串。

    中文 claim 直接去检索英文语料会检索到中文文献、重排分趋零（V9 实测过），
    所以必须先落到英文。翻译失败退回确定性兜底词，并在 claim 来源里标出来。
    """
    if not claims or runtime_config is None:
        return {}
    prompt = (
        "你在为学术检索准备关键词。把每条中文研究主张改写成一条英文检索关键词串："
        "8-14 个英文词，只放名词与术语，不要整句、不要引号、不要 and/or 这类布尔符号。"
        '输出 JSON：{"claims":[{"claim_id":"原样返回","search_terms":"英文关键词串"}]}'
    )
    payload = {
        "paper_title": clean_paper_title(paper_title),
        "claims": [{"claim_id": cid, "label": label, "claim": text} for cid, label, text in claims],
    }
    try:
        raw = call_plain_json(prompt, payload, runtime_config)
    except Exception:
        return {}
    items = raw.get("claims") if isinstance(raw, dict) else None
    result: dict[str, str] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("claim_id") or "").strip()
        terms = re.sub(r"[\"'“”]", "", str(item.get("search_terms") or "")).strip()
        if claim_id and terms:
            result[claim_id] = terms
    return result


def claims_from_paper_context(
    paper_context: dict[str, Any] | None,
    paper_title: str,
    runtime_config: AgentRuntimeConfig | None = None,
) -> list[RetrievalClaim]:
    """用论文自己的研究问题/缺口/贡献构造 claim（首选）。"""
    raw_claims = context_claim_texts(paper_context)
    translated = translate_claims_to_english(raw_claims, paper_title, runtime_config)
    claims: list[RetrievalClaim] = []
    for claim_id, label, text in raw_claims:
        terms = translated.get(claim_id)
        source = "paper_context" if terms else "fallback"
        claims.append(
            RetrievalClaim(
                claim_id=claim_id,
                label=label,
                text=text,
                keywords_en=terms or DEFAULT_CONTEXT_CLAIM_EN.get(claim_id, ""),
                source=source,
                claim_type=claim_id,
            )
        )
    return claims


def claims_from_rules(
    rules: list[tuple[str, IndicatorRule]],
    config: ExternalRetrievalConfig,
) -> list[RetrievalClaim]:
    """用规则定义构造 claim（paper_context 抽不出 claim 时的兜底）。"""
    claims: list[RetrievalClaim] = []
    for claim_id, rule in rules:
        terms, _, origin = claim_retrieval_terms(rule, config)
        claims.append(
            RetrievalClaim(
                claim_id=claim_id,
                label=rule.indicator_name,
                text=rule.definition,
                keywords_en=terms,
                source=origin,
            )
        )
    return claims


# 论文标题是从 PDF 文件名来的，形如「Choudhury et al. - 2020 - 真实标题」。
# 作者名和年份留在检索式里有两个害处：拿作者名去检索会拉回该作者自己的论文，
# 而且白占主题词额度（前 5 个主题词里 2 个被浪费）。
FILENAME_TITLE_PREFIX = re.compile(r"^\s*.{0,90}?\b(?:19|20)\d{2}\b\s*[-–—]\s*")


def clean_paper_title(title: str) -> str:
    """剥掉文件名带来的「作者 - 年份 - 」前缀，得到真实标题。"""
    text = str(title or "").strip()
    cleaned = FILENAME_TITLE_PREFIX.sub("", text).strip()
    return cleaned if len(cleaned) >= 12 else text


def _topic_tokens(paper_title: str, limit: int = 8) -> list[str]:
    """论文主题词。默认取 8 个：文件名常把标题截断，取少了会漏掉 "artificial intelligence" 这类关键词。"""
    return _tokens(clean_paper_title(paper_title))[:limit]


def build_claim_query(rule: IndicatorRule, paper_title: str = "", extra: str = "") -> str:
    """按 claim（规则）生成检索式：指标名 + 定义要点 + 观察点关键词，末尾带论文主题词。"""
    parts: list[str] = [rule.indicator_name]
    definition_tokens = _tokens(rule.definition)[:6]
    parts.extend(definition_tokens)
    for point in rule.observation_points[:2]:
        parts.extend(_tokens(point)[:2])
    topic = " ".join(_topic_tokens(paper_title, 4))
    if topic:
        parts.append(topic)
    if extra:
        parts.extend(_tokens(extra)[:3])
    seen: set[str] = set()
    ordered = [piece for piece in parts if piece and not (piece in seen or seen.add(piece))]
    return " ".join(ordered)[:220]


def _normalized_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def is_self_reference(candidate_title: str, paper_title: str) -> bool:
    """判断候选文献是不是正在评的这篇论文自己。

    检索式是用本文标题 + 研究问题拼出来的，所以本文几乎必然是最高分命中（实测 rerank 34.5，
    远超真实相关工作）。不剔除的话，「有没有人做过」的对照位会被本文自己占掉，
    重排分整体虚高，判断反而失真——本次只是模型碰巧看出「即本文自身」才没出错。
    """
    candidate = str(candidate_title or "").strip()
    paper = clean_paper_title(paper_title)
    if not candidate or not paper:
        return False
    candidate_tokens = set(_tokens(candidate))
    paper_tokens = set(_tokens(paper))
    if candidate_tokens and paper_tokens:
        union = candidate_tokens | paper_tokens
        if union and len(candidate_tokens & paper_tokens) / len(union) >= 0.6:
            return True
    # 标题互相包含（OpenAlex 有时只给主标题，本文标题来自文件名可能带副标题）
    a, b = _normalized_title(candidate), _normalized_title(paper)
    if a and b and min(len(a), len(b)) >= 20 and (a in b or b in a):
        return True
    return False


def _compose_query(terms: str, paper_title: str) -> str:
    """论文主题词 + claim 检索词，去重后拼成检索式。

    主题词必须放在最前面：检索式有长度上限，被截断时应该是通用的方法论 claim 词先掉，
    而不是把「这篇论文到底在研究什么」截掉。早前的写法把主题放末尾，实际发出去的检索式
    结尾是「...prior research dialogue choudhury machine learn」，主题只剩半个词。
    """
    topic = " ".join(_topic_tokens(paper_title, 8))
    seen: set[str] = set()
    ordered = [piece for piece in f"{topic} {terms}".split() if piece and not (piece in seen or seen.add(piece))]
    return " ".join(ordered)[:220]


def claim_retrieval_terms(rule: IndicatorRule, config: ExternalRetrievalConfig) -> tuple[str, list[str], str]:
    """返回 (claim 检索词, 重排词表, 来源)。

    英文 override 优先；缺失则退回规则中文原文，来源标 `rule_text`。
    """
    override = str((config.claim_query_en or {}).get(rule.rule_id) or "").strip()
    if override:
        return override, _tokens(override), "override"
    source_text = f"{rule.indicator_name} {rule.definition} {' '.join(rule.observation_points)}"
    return build_claim_query(rule), _tokens(source_text), "rule_text"


def claim_rerank_vocabulary(
    terms: str,
    origin: str,
    paper_title: str,
    rule: IndicatorRule | None = None,
) -> tuple[set[str], list[str], set[str]]:
    """重排词表：(claim 词, 短语, 论文主题词)。

    主题词单独返回：它代表"这篇论文在研究什么"，和 claim 词不是一回事，打分会分开记账。
    通用学术词从 claim 词里剔除，否则排序会被"研究/发现/理论"这类词拉平。
    """
    claim_tokens = {token for token in _tokens(terms) if token not in GENERIC_ACADEMIC_TOKENS}
    topic_tokens = {token for token in _topic_tokens(paper_title, 8) if token not in GENERIC_ACADEMIC_TOKENS}
    if origin == "rule_text" and rule is not None:
        phrases = [rule.indicator_name] + [piece for piece in _tokens(rule.definition)[:4] if len(piece) >= 4]
    else:
        ordered = _tokens(terms)
        phrases = [" ".join(ordered[index:index + 2]) for index in range(0, max(0, min(len(ordered), 8) - 1), 2)]
    return claim_tokens, phrases, topic_tokens


def rerank(
    candidate: RelatedWork,
    claim_tokens: set[str],
    claim_phrases: list[str],
    topic_tokens: set[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """本地确定性重排：短语命中权重高，词命中累积，摘要过短略降。每一分都可解释。

    主题词命中单独加权（2.0/词）：claim 词说明"为什么找它"，主题词说明"是不是讲同一件事"，
    只有两层都对上的文献才应该排在最前。
    """
    haystack = f"{candidate.title} {candidate.abstract}".lower()
    title_lower = candidate.title.lower()
    phrase_hits = sum(1 for phrase in claim_phrases if phrase and phrase in haystack)
    title_phrase_hits = sum(1 for phrase in claim_phrases if phrase and phrase in title_lower)
    token_hits = sum(1 for token in claim_tokens if token in haystack)
    topic_hits = sum(1 for token in (topic_tokens or set()) if token in haystack)
    coverage = token_hits / max(1, len(claim_tokens))
    abstract_bonus = 0.05 if len(candidate.abstract) >= 200 else 0.0
    detail = {
        "phrase_hits": float(phrase_hits),
        "title_phrase_hits": float(title_phrase_hits),
        "token_hits": float(token_hits),
        "topic_hits": float(topic_hits),
        "token_coverage": coverage,
        "abstract_bonus": abstract_bonus,
    }
    score = (
        3.0 * phrase_hits
        + 1.5 * title_phrase_hits
        + 1.0 * token_hits
        + 2.0 * topic_hits
        + 0.5 * coverage
        + abstract_bonus
    )
    return score, detail


def resolved_providers(config: ExternalRetrievalConfig) -> list[str]:
    """实际使用的检索源：配置顺序；只要存在 S2 key 就自动补上 Semantic Scholar。"""
    providers = [str(name) for name in (config.providers or DEFAULT_PROVIDERS)]
    if os.environ.get(config.api_key_env) and "semantic_scholar" not in providers:
        providers.append("semantic_scholar")
    return providers or ["openalex"]


def _cache_path(cache_dir: str | None, key: str) -> Path | None:
    if not cache_dir:
        return None
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"ext_{digest}.json"


def _openalex_abstract(inverted: Any) -> str:
    """OpenAlex 的摘要存成倒排索引（词 → 位置列表），按位置拼回原文。"""
    if not isinstance(inverted, dict):
        return ""
    positions: dict[int, str] = {}
    for word, indexes in inverted.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                positions[index] = str(word)
    return " ".join(positions[key] for key in sorted(positions))


def _normalize_openalex(payload: Any) -> list[dict[str, Any]]:
    """把 OpenAlex 条目转成 Semantic Scholar 的字段形状，下游重排/去重逻辑完全不用改。"""
    results = payload.get("results") if isinstance(payload, dict) else None
    items: list[dict[str, Any]] = []
    for row in results if isinstance(results, list) else []:
        if not isinstance(row, dict):
            continue
        doi = str(row.get("doi") or "").removeprefix("https://doi.org/")
        source = ((row.get("primary_location") or {}) if isinstance(row.get("primary_location"), dict) else {}).get("source")
        items.append(
            {
                "title": str(row.get("display_name") or row.get("title") or "").strip(),
                "abstract": _openalex_abstract(row.get("abstract_inverted_index")),
                "year": row.get("publication_year"),
                "venue": str((source or {}).get("display_name") or "") if isinstance(source, dict) else "",
                "authors": [
                    {"name": str((entry.get("author") or {}).get("display_name") or "")}
                    for entry in (row.get("authorships") or [])
                    if isinstance(entry, dict)
                ][:6],
                "citationCount": row.get("cited_by_count"),
                "externalIds": {"DOI": doi} if doi else {"OpenAlex": str(row.get("id") or "")},
            }
        )
    return items


def _http_search_semantic_scholar(query: str, config: ExternalRetrievalConfig) -> dict[str, Any]:
    """相关性检索源。匿名常年 429（见 paper_eval/semantic_scholar.py 文档），
    失败会抛出并让 search_external 落到下一个源；有 key 时这里才是首选源。"""
    rows = s2_relevance(query, config.per_query_limit, S2Config.from_external(config), config.year_from, config.year_to)
    return {"data": rows}


def _http_search_semantic_scholar_bulk(query: str, config: ExternalRetrievalConfig) -> dict[str, Any]:
    """S2 批量检索源（免 key 可用），截断成与 /paper/search 相同的信封形状。

    ponytail: bulk 无相关性排序，客户端按 citationCount:desc 取前 N 条
    （"命中最多的那批里被引最高的前 N 篇"）。拿到 key 后换成 semantic_scholar 源即恢复相关性排序。
    """
    rows = s2_bulk(query, config.per_query_limit, S2Config.from_external(config), config.year_from, config.year_to)
    return {"data": rows}


def _http_search_openalex(query: str, config: ExternalRetrievalConfig) -> dict[str, Any]:
    params: dict[str, str] = {"search": query, "per-page": str(config.per_query_limit)}
    if config.mailto:
        params["mailto"] = config.mailto
    if config.year_from:
        window = f"from_publication_date:{config.year_from}-01-01"
        if config.year_to:
            window += f",to_publication_date:{config.year_to}-12-31"
        params["filter"] = window
    url = OPENALEX_ENDPOINT + "?" + urllib.parse.urlencode(params)
    agent = f"paper-eval/1.0 ({config.mailto})" if config.mailto else "paper-eval/1.0"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": agent})
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _strip_markup(text: str) -> str:
    """去掉 Crossref 摘要里的 JATS/HTML 标签与实体。"""
    plain = re.sub(r"<[^>]+>", " ", str(text or ""))
    for entity, char in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"'), ("&#x2010;", "-"), ("&nbsp;", " ")):
        plain = plain.replace(entity, char)
    return " ".join(plain.split())


def _normalize_crossref(payload: Any) -> list[dict[str, Any]]:
    """把 Crossref 条目转成 Semantic Scholar 的字段形状，下游重排/去重逻辑不用改。"""
    message = payload.get("message") if isinstance(payload, dict) else None
    items_raw = message.get("items") if isinstance(message, dict) else None
    items: list[dict[str, Any]] = []
    for row in items_raw if isinstance(items_raw, list) else []:
        if not isinstance(row, dict):
            continue
        titles = row.get("title") or []
        title = str(titles[0]).strip() if isinstance(titles, list) and titles else str(titles or "").strip()
        containers = row.get("container-title") or []
        venue = str(containers[0]) if isinstance(containers, list) and containers else ""
        parts = ((row.get("issued") or {}).get("date-parts") or [[None]]) if isinstance(row.get("issued"), dict) else [[None]]
        year = parts[0][0] if parts and isinstance(parts[0], list) and parts[0] else None
        authors = []
        for entry in row.get("author") or []:
            if not isinstance(entry, dict):
                continue
            name = " ".join(piece for piece in (str(entry.get("given") or "").strip(), str(entry.get("family") or "").strip()) if piece)
            if name:
                authors.append({"name": name})
        doi = str(row.get("DOI") or "").strip()
        items.append(
            {
                "title": title,
                "abstract": _strip_markup(row.get("abstract") or ""),
                "year": year,
                "venue": venue,
                "authors": authors[:6],
                "citationCount": row.get("is-referenced-by-count"),
                "externalIds": {"DOI": doi} if doi else {},
            }
        )
    return items


def _http_search_crossref(query: str, config: ExternalRetrievalConfig) -> dict[str, Any]:
    params: dict[str, str] = {
        "query.bibliographic": query,
        "rows": str(config.per_query_limit),
        "select": CROSSREF_SELECT,
    }
    if config.mailto:
        params["mailto"] = config.mailto
    url = CROSSREF_ENDPOINT + "?" + urllib.parse.urlencode(params)
    agent = f"paper-eval/1.0 (mailto:{config.mailto})" if config.mailto else "paper-eval/1.0"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": agent})
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _raw_fetch(query: str, provider: str, config: ExternalRetrievalConfig) -> list[dict[str, Any]]:
    if provider == "openalex":
        return _normalize_openalex(_http_search_openalex(query, config))
    if provider == "crossref":
        return _normalize_crossref(_http_search_crossref(query, config))
    if provider == "semantic_scholar":
        payload = _http_search_semantic_scholar(query, config)
        data = payload.get("data") if isinstance(payload, dict) else None
        return [item for item in (data if isinstance(data, list) else []) if isinstance(item, dict)]
    if provider == "semantic_scholar_bulk":
        payload = _http_search_semantic_scholar_bulk(query, config)
        data = payload.get("data") if isinstance(payload, dict) else None
        return [item for item in (data if isinstance(data, list) else []) if isinstance(item, dict)]
    raise ValueError(f"未知外部检索源: {provider}")


def _cached_or_fetch(
    provider: str,
    query: str,
    config: ExternalRetrievalConfig,
    loader: Callable[[], Any],
) -> tuple[list[dict[str, Any]], str | None]:
    """返回 (条目, 错误信息)。缓存按「检索源 + 检索式」隔离，换源不会读到旧结果。"""
    cache_file = _cache_path(config.cache_dir, f"{provider}|{query}|{config.per_query_limit}")
    if cache_file and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8")).get("data", []), None
        except (OSError, json.JSONDecodeError):
            pass
    try:
        items = [item for item in (loader() or []) if isinstance(item, dict)]
    except Exception as exc:  # 外部检索失败不致命
        return [], f"{provider}: {exc}"
    if cache_file:
        cache_file.write_text(
            json.dumps({"provider": provider, "query": query, "data": items}, ensure_ascii=False),
            encoding="utf-8",
        )
    return items, None


def _fetcher_items(payload: Any) -> list[dict[str, Any]]:
    """注入 fetcher 的返回兼容两种形状：S2 信封 {"data": [...]} 或直接的条目列表。"""
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _retry_after_seconds(exc: Exception) -> float | None:
    """读服务端的 Retry-After（秒）。它比本地退避权威得多。"""
    headers = getattr(exc, "headers", None)
    raw = headers.get("Retry-After") if headers else None
    if not raw:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _fetch_with_retry(provider: str, query: str, config: ExternalRetrievalConfig) -> list[dict[str, Any]]:
    """真实网络请求带重试退避，并尊重服务端的 Retry-After。

    外部学术 API 在突发请求下会限流（OpenAlex 未进礼貌池时实测返回 429，并要求等 35-60 秒）。
    只对可重试错误重试：限流、超时、5xx；其余（如 400 参数错误）直接抛出，避免空转。
    """
    attempts = max(1, int(config.request_retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _raw_fetch(query, provider, config)
        except Exception as exc:  # noqa: BLE001 - 需要按错误文本与响应头判断是否可重试
            last_error = exc
            message = str(exc).lower()
            retriable = any(
                marker in message
                for marker in ("429", "too many requests", "503", "timed out", "timeout", "502", "504", "remote end closed")
            )
            if not retriable:
                raise
            wait = _retry_after_seconds(exc)
            if wait is not None and wait > config.max_retry_after_seconds:
                raise RuntimeError(
                    f"{exc}（检索源要求等待 {wait:.0f} 秒，超过上限 {config.max_retry_after_seconds:.0f} 秒："
                    f"{provider} 匿名共享池限流，建议在 external_retrieval.mailto 填入联系邮箱进入 OpenAlex 礼貌池）"
                ) from exc
            if attempt + 1 >= attempts:
                if wait is not None:
                    raise RuntimeError(
                        f"{exc}（检索源要求等待 {wait:.0f} 秒：{provider} 匿名共享池限流，"
                        "建议在 external_retrieval.mailto 填入联系邮箱进入礼貌池）"
                    ) from exc
                raise
            # 服务端给了 Retry-After 就听它的；没给才用本地指数退避
            time.sleep(wait if wait is not None else config.retry_backoff_seconds * (2 ** attempt))
    if last_error is not None:
        raise last_error
    return []


def search_external(
    query: str,
    config: ExternalRetrievalConfig,
    fetcher: Callable[[str, int], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """按配置顺序尝试检索源，返回 (条目, 错误, 命中的源)。

    每个源单独计错：一个源整体不可用时会自动落到下一个源，全部失败才算失败。
    `fetcher` 用于离线自检，注入时只用第一个源，保证离线结果确定（也跳过重试与限速）。
    """
    if fetcher is not None:
        provider = resolved_providers(config)[0]
        items, error = _cached_or_fetch(provider, query, config, lambda: _fetcher_items(fetcher(query, config.per_query_limit)))
        return items, error, (None if error else provider)
    last_error: str | None = None
    for index, provider in enumerate(resolved_providers(config)):
        if index > 0 and config.min_interval_seconds > 0:
            time.sleep(config.min_interval_seconds)
        items, error = _cached_or_fetch(
            provider, query, config, lambda p=provider: _fetch_with_retry(p, query, config)
        )
        if error is None:
            return items, None, provider
        last_error = error
    return [], last_error, None


def retrieve_related_works(
    claims: list[RetrievalClaim],
    paper_title: str = "",
    config: ExternalRetrievalConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
    fetcher: Callable[[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """按 claim 检索并重排相关文献。

    预算控制：至少 `min_queries` 条不同检索式；候选总数不超过 `max_candidates`；
    连续失败达到 `error_budget` 或超过 `max_queries` 即停；
    重排后最高分达到 `min_top_score` 且候选数够 `top_k` 时提前停止。
    """
    cfg = config or ExternalRetrievalConfig()
    result: dict[str, Any] = {
        "enabled": cfg.enabled,
        "queries": [],
        "candidates": [],
        "related_works": [],
        "failures": [],
        "stop_reason": None,
    }
    if not cfg.enabled:
        result["stop_reason"] = "disabled"
        return result

    pool: dict[str, RelatedWork] = {}
    claim_terms: dict[str, tuple[str, str]] = {}   # claim_id -> (检索词, 来源)
    consecutive_errors = 0
    issued = 0

    for claim in claims:
        if issued >= cfg.max_queries:
            result["stop_reason"] = "max_queries"
            break
        query = _compose_query(claim.keywords_en, paper_title)
        if not query:
            continue
        issued += 1
        claim_terms[claim.claim_id] = (claim.keywords_en, claim.source)
        result["queries"].append(
            {
                "claim_id": claim.claim_id,
                "claim_label": claim.label,
                "claim_text": claim.text[:160],
                "query": query,
                "origin": claim.source,
            }
        )
        if progress_callback:
            progress_callback(f"外部检索（{claim.label}）：{query[:40]}")
        items, error, provider = search_external(query, cfg, fetcher=fetcher)
        result["queries"][-1]["provider"] = provider
        if error:
            consecutive_errors += 1
            result["failures"].append({"claim_id": claim.claim_id, "query": query, "error": error[:200]})
            if consecutive_errors >= cfg.error_budget:
                result["stop_reason"] = "error_budget"
                break
            continue
        consecutive_errors = 0
        for item in items:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            if is_self_reference(title, paper_title):
                result["self_excluded"] = int(result.get("self_excluded") or 0) + 1
                continue
            key = str((item.get("externalIds") or {}).get("DOI") or title.lower())
            if key in pool:
                continue
            pool[key] = RelatedWork(
                title=title,
                year=item.get("year"),
                venue=str(item.get("venue") or ""),
                authors=[str(a.get("name")) for a in (item.get("authors") or []) if isinstance(a, dict)][:6],
                abstract=str(item.get("abstract") or ""),
                citation_count=item.get("citationCount"),
                external_ids=item.get("externalIds") or {},
                source_query=query,
            )
            if len(pool) >= cfg.max_candidates:
                break
        if len(pool) >= cfg.max_candidates:
            result["stop_reason"] = "max_candidates"
            break

    if issued < cfg.min_queries and result["stop_reason"] is None:
        result["stop_reason"] = "below_min_queries"

    # 重排：claim 关键词/短语作打分依据，对每个 claim 各排一次，合并去重取最高分
    for claim in claims:
        terms, origin = claim_terms.get(claim.claim_id, ("", claim.source))
        if not terms:
            continue
        tokens, phrases, topic_tokens = claim_rerank_vocabulary(terms, origin, paper_title)
        for work in pool.values():
            score, detail = rerank(work, tokens, phrases, topic_tokens)
            if score > work.rerank_score:
                work.rerank_score = score
                work.rerank_detail = {**detail, "matched_claim": claim.claim_id, "claim_origin": origin}

    ranked = sorted(pool.values(), key=lambda w: (-w.rerank_score, -(w.citation_count or 0)))
    result["candidates"] = len(ranked)
    result["related_works"] = [work.to_dict() for work in ranked[: cfg.top_k]]
    if result["stop_reason"] is None:
        top = ranked[0].rerank_score if ranked else 0.0
        result["stop_reason"] = "enough" if (top >= cfg.min_top_score and len(ranked) >= cfg.top_k) else "queries_exhausted"
    return result


def summarize_related_works(
    works: list[dict[str, Any]],
    paper_title: str,
    paper_abstract: str = "",
    config: AgentRuntimeConfig | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """给每篇相关工作生成结构化摘要（方法/发现/实验/相似/不同）。失败记录进 summary_error。"""
    if config is None or top_k <= 0:
        return works
    prompt = (
        "你在为一篇待评论文整理相关工作。针对给定的这篇相关工作，输出 JSON："
        '{"method":"它的方法是什么","findings":"主要发现是什么","experiments":"实验结果是什么",'
        '"similar":"与本论文什么地方相似","different":"什么地方不同"}。'
        "每项一到两句中文，只依据给出的标题与摘要，摘要缺失就写“摘要缺失，无法判断”，不要编造。"
    )
    enriched: list[dict[str, Any]] = []
    for index, work in enumerate(works):
        if index < top_k:
            payload = {
                "current_paper_title": paper_title,
                "current_paper_abstract": paper_abstract[:600],
                "related_work": {
                    "title": work.get("title"),
                    "year": work.get("year"),
                    "venue": work.get("venue"),
                    "abstract": work.get("abstract"),
                },
            }
            try:
                raw = call_plain_json(prompt, payload, config)
                work = {**work, "summary": {k: str(raw.get(k, "")) for k in ("method", "findings", "experiments", "similar", "different")}}
            except Exception as exc:
                work = {**work, "summary": None, "summary_error": str(exc)[:200]}
        enriched.append(work)
    return enriched


def _claim_rules(
    profile: RuleProfile,
    config: ExternalRetrievalConfig,
    paper_type: str | None,
) -> list[tuple[str, IndicatorRule]]:
    """外部检索的 claim 集：配置里点名的规则 ∩ 本篇论文类型在范围内。"""
    claims: list[tuple[str, IndicatorRule]] = []
    for rule_id in config.claim_rule_ids:
        rule = profile.rules_by_id.get(rule_id)
        if rule is None or not rule_in_scope(rule.applicable_type, paper_type):
            continue
        claims.append((rule_id, rule))
    return claims


def collect_related_works(
    profile: RuleProfile,
    paper_title: str,
    paper_abstract: str = "",
    paper_type: str | None = None,
    retrieval_config: ExternalRetrievalConfig | None = None,
    runtime_config: AgentRuntimeConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
    fetcher: Callable[[str, int], dict[str, Any]] | None = None,
    paper_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """外部检索一整段：抽 claim → 真实检索 → 重排 → 五问摘要。

    claim 首选论文自己的研究问题/缺口/贡献（paper_context），抽不出来才退回规则定义。
    `require_external_evidence` 打开时，检索不到任何外部文献就抛错——检索现在跑在评分之前，
    此时还没有花掉评分调用，所以「宁可停」远好过让模型凭空断言「没人做过」。

    评分管线现在走 `collect_external_literature`（多路召回 + XEV-* 证据块）；这个函数只保留
    旧画像/规则 claim 单路检索路径，供既有自检与手工排查使用。
    """
    cfg = retrieval_config or ExternalRetrievalConfig()
    if not cfg.enabled:
        return {"enabled": False, "stop_reason": "disabled", "related_works": [], "queries": [], "failures": [], "status": external_retrieval_status(cfg)}
    # 标题来自文件名，先剥掉「作者 - 年份 - 」前缀；检索和摘要提示词都用干净标题。
    paper_title = clean_paper_title(paper_title)

    claims = claims_from_paper_context(paper_context, paper_title, runtime_config)
    claim_source = "paper_context"
    if not claims:
        claims = claims_from_rules(_claim_rules(profile, cfg, paper_type), cfg)
        claim_source = "rule_fallback"

    result = retrieve_related_works(
        claims,
        paper_title,
        cfg,
        progress_callback=progress_callback,
        fetcher=fetcher,
    )
    result["claim_source"] = claim_source
    result["claims"] = [
        {"claim_id": claim.claim_id, "label": claim.label, "text": claim.text[:200], "keywords_en": claim.keywords_en, "origin": claim.source}
        for claim in claims
    ]
    result["related_works"] = summarize_related_works(
        result["related_works"],
        paper_title,
        paper_abstract,
        runtime_config,
        cfg.summary_top_k,
    )
    result["status"] = external_retrieval_status(cfg)

    if cfg.require_external_evidence and not result["related_works"]:
        # 限流风暴下首轮可能整轮为空。先冷却再整轮重试一次，仍为空才判定失败。
        if cfg.empty_retry_seconds > 0:
            if progress_callback:
                progress_callback(f"外部检索首轮为空，等待 {cfg.empty_retry_seconds:.0f} 秒后整轮重试")
            time.sleep(cfg.empty_retry_seconds)
            retry = retrieve_related_works(
                claims, paper_title, cfg, progress_callback=progress_callback, fetcher=fetcher
            )
            if retry.get("related_works"):
                retry["claim_source"] = claim_source
                retry["claims"] = result["claims"]
                retry["first_pass_failures"] = result.get("failures")
                retry["related_works"] = summarize_related_works(
                    retry["related_works"], paper_title, paper_abstract, runtime_config, cfg.summary_top_k
                )
                retry["status"] = external_retrieval_status(cfg)
                result = retry
    if cfg.require_external_evidence and not result["related_works"]:
        # 走真实网络也拿不到文献：直接停，不要让模型在没有外部对照的情况下判「没人做过」。
        detail = "; ".join(item.get("error", "") for item in result["failures"][:3]) or result.get("stop_reason")
        raise RuntimeError(
            "EXTERNAL_EVIDENCE_REQUIRED: 外部文献检索未取得任何结果，已在调用评分模型之前停止。"
            f"检索来源={result['status'].get('providers')}；原因={detail}。"
            "若检索源持续限流，请在 external_retrieval.mailto 填入联系邮箱进入 OpenAlex 礼貌池；"
            "如需在离线环境继续，请把 require_external_evidence 设为 false。"
        )
    return result


def collect_external_literature(
    profile: RuleProfile,
    paper_title: str,
    paper_abstract: str = "",
    *,
    paper_id: str = "",
    evidences: list[Evidence] | None = None,
    paper_type: str | None = None,
    publication_year: int | None = None,
    research_question: str = "",
    retrieval_config: ExternalRetrievalConfig | None = None,
    runtime_config: AgentRuntimeConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
    fetcher: Callable[[str, int], dict[str, Any]] | None = None,
    paper_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """外部文献检索主入口（第二阶段）：Claim Extractor → 三路召回 → 时间过滤 → Reranker
    → Crossref 校验 → External Evidence Set → XEV-* 证据块。

    返回结构同时保留旧字段（related_works / queries / claims），报告层与旧自检不用改。
    `external_evidence_unavailable` 为真表示这次没拿到外部证据——评分可以继续，
    但依赖领域比较的规则会被 Harness 拦住不给 satisfied。
    """
    from .claim_extraction import extract_claims, publication_year_of
    from .literature_search import (
        attach_external_evidence_ids,
        coverage_of,
        run_literature_search,
    )

    cfg = retrieval_config or ExternalRetrievalConfig()
    if not cfg.enabled:
        return {
            "enabled": False,
            "stop_reason": "disabled",
            "related_works": [],
            "queries": [],
            "failures": [],
            "claims": [],
            "external_evidence_set": [],
            "external_evidence_unavailable": True,
            "status": external_retrieval_status(cfg),
        }

    paper_title = clean_paper_title(paper_title)
    evidences = evidences or []
    if publication_year is None:
        publication_year = publication_year_of((paper_context or {}).get("paper_profile"))

    # B1：内部贡献抽取（claim 一律带 source_evidence_id，指回本文 Evidence Repository）
    claims: list[RetrievalClaim] = []
    claim_source = "paper_context"
    if cfg.claim_extraction_enabled and (paper_context or evidences):
        claims = [
            claim.to_retrieval_claim(paper_title)
            for claim in extract_claims(paper_context, evidences, publication_year)
        ]
        claim_source = "claim_extractor"
    if not claims:
        claims = claims_from_paper_context(paper_context, paper_title, runtime_config)
        claim_source = "paper_context"
    if not claims:
        claims = claims_from_rules(_claim_rules(profile, cfg, paper_type), cfg)
        claim_source = "rule_fallback"

    for claim in claims:
        setattr(claim, "paper_title", paper_title)

    # B2：外部检索（三路召回 + 重排）
    result = run_literature_search(
        claims,
        paper_title,
        cfg,
        paper_abstract=paper_abstract,
        research_question=research_question,
        target_year=publication_year,
        progress_callback=progress_callback,
        fetcher=fetcher,
    )
    result["claim_source"] = claim_source
    result["claims"] = [claim.to_dict() for claim in claims]
    result["publication_year"] = publication_year

    # Crossref 元数据校验（DOI / 标题 / 年份一致性）
    from .external_evidence import verify_crossref

    external_set = list(result.get("external_evidence_set") or [])
    if cfg.crossref_verify and external_set:
        verify_crossref(
            external_set,
            cfg,
            cache_dir=cfg.cache_dir,
            fetcher=None,
            max_checks=cfg.crossref_max_checks,
        )

    attach_external_evidence_ids(result)
    external_set = list(result.get("external_evidence_set") or [])
    result["coverage"] = coverage_of(result)
    result["external_evidence_unavailable"] = not bool(external_set)
    result["related_works"] = external_set
    result["candidates"] = result.get("pool_size") or 0
    result["status"] = external_retrieval_status(cfg)
    return result


def external_evidence_blocks(
    result: dict[str, Any],
    paper_id: str,
    max_blocks: int = 8,
) -> list[Evidence]:
    """把检索到的相关工作变成可引用的证据块（EXT-*），每条带确定性的相关度档位。

    相关度档位按同批重排分的相对位置算（高 ≥ 75% 最高分、中 ≥ 50%、其余低），
    这样模型和读者看到的都是可核对的相对强弱，而不是一个孤立的分数。

    多路召回（claim_extractor 路径）会走 literature_search 的 XEV-* 证据块；这条 EXT-* 路径
    保留给旧的画像/规则 claim 流程与既有自检。
    """
    works = result.get("related_works") or []
    if not works:
        return []
    top = max(float(work.get("rerank_score") or 0.0) for work in works[:max_blocks])
    blocks: list[Evidence] = []
    for index, work in enumerate(works[:max_blocks], start=1):
        score = float(work.get("rerank_score") or 0.0)
        tier = "高" if top > 0 and score >= top * 0.75 else ("中" if top > 0 and score >= top * 0.5 else "低")
        summary = work.get("summary") if isinstance(work.get("summary"), dict) else {}
        lines = [
            (
                f"【外部文献 {index}｜相关度 {tier}｜{work.get('year') or '年份未知'}｜"
                f"{work.get('venue') or '期刊未知'}｜他引 {work.get('citation_count')}】"
            ),
            f"标题：{work.get('title')}",
        ]
        if work.get("doi"):
            lines.append(f"DOI：{work.get('doi')}")
        abstract = str(work.get("abstract") or "").strip()
        lines.append(f"摘要：{abstract[:600] if abstract else '（该文献无摘要）'}")
        for key, label in (("method", "方法"), ("findings", "发现"), ("similar", "相似"), ("different", "不同")):
            value = str(summary.get(key) or "").strip()
            if value:
                lines.append(f"{label}：{value}")
        text = "\n".join(lines)
        blocks.append(
            Evidence(
                evidence_id=f"EXT-{index}",
                paper_id=paper_id,
                section="外部文献（OpenAlex 检索）",
                page=0,
                span=(0, len(text)),
                text=text,
                text_hash=hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
                source_type="external",
            )
        )
    return blocks


def is_external_evidence(item: Evidence) -> bool:
    return str(getattr(item, "source_type", "") or "") == "external"


def load_external_retrieval_config(path: str | Path | None) -> ExternalRetrievalConfig:
    """从配置文件顶层读 `external_retrieval:` 段落。

    刻意放在 `agent_runtime` 之外：`AgentRuntimeConfig` 会把该段落的键直接展开成构造参数，
    多一个未知键就会报错，所以外部检索的配置必须独立成段。
    """
    if not path:
        return ExternalRetrievalConfig()
    config_path = Path(path)
    if not config_path.exists():
        return ExternalRetrievalConfig()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    values = data.get("external_retrieval") or {}
    allowed = {item.name for item in fields(ExternalRetrievalConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"external_retrieval 配置含未知键: {unknown}")
    return ExternalRetrievalConfig(**values)


def external_retrieval_status(config: ExternalRetrievalConfig) -> dict[str, Any]:
    """给工作台/报告用的状态摘要。"""
    providers = resolved_providers(config)
    return {
        "enabled": config.enabled,
        "provider": providers[0] if providers else None,
        "providers": providers,
        "api_key_present": bool(os.environ.get(config.api_key_env)),
        "api_key_env": config.api_key_env,
        "mailto_present": bool(config.mailto),
        "claim_rule_ids": list(config.claim_rule_ids),
        "evidence_rule_ids": list(config.evidence_rule_ids),
        "require_external_evidence": config.require_external_evidence,
        "max_evidence_blocks": config.max_evidence_blocks,
        "cache_dir": config.cache_dir,
        "budget": {
            "min_queries": config.min_queries,
            "max_queries": config.max_queries,
            "max_candidates": config.max_candidates,
            "error_budget": config.error_budget,
            "top_k": config.top_k,
            "summary_top_k": config.summary_top_k,
        },
    }

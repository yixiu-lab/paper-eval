'External literature search: OpenAlex / Semantic Scholar + claim-conditioned retrieval + search budget + structured abstract.\n\nDesign points (aligned with CLAIMCHECK requirements):\n\n- **Claim-conditioned Retrieval**: The retrieval query is generated based on the claim (= scoring rule), instead of doing a RAG for the entire paper.\n  `build_claim_query()` synthesizes the index name, definition, observation point of the rule and the keywords of the paper title into a search formula.\n- **Search source**: Default Crossref + OpenAlex + Semantic Scholar bulk (key-free for both, `providers` in the configuration can have multiple sources).\n  Semantic Scholar\'s /paper/search requires a key (the actual test of the anonymous shared pool without a key is 429 all year round), so no need to use the key\n  /paper/search/bulk (another pool, anonymous actual measurement is stable); automatically added when `SEMANTIC_SCHOLAR_API_KEY` exists\n  /paper/search as the relevance ranking source.\n  All sources are normalized to the same field shape and downstream reordering/deduplication/summarization logic does not differentiate between sources.\n- **Rerank**: First use the relevance ranking of external sources as a priori, and then use the local certainty score to rerank\n  (claim keyword hit + observation point hit + summary length), the source of each point can be explained, and does not rely on the black box model.\n- **Search budget and stopping conditions**: `min_queries` (at least how many different claim search terms are issued),\n  `max_candidates` (the upper limit of the total number of candidates), `max_queries`, `error_budget`,\n  And "Stop as soon as it is enough after rearrangement". Avoid Agents from jumping to conclusions after just searching two articles.\n- **Cache**: Press "Retrieval Source + Retrieval Type" to do disk caching. Repeated operations will not repeatedly hit the external API.\n- **Failure is not fatal**: When the external API is unavailable, a structured result with `failures` is returned, no exception is thrown, and scoring is not interrupted.'

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

# S2 are all in paper_eval/semantic_scholar.py;
# The name API_FIELDS is reserved here to avoid touching the existing self-test script.
API_FIELDS = SEARCH_FIELDS
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
CROSSREF_SELECT = "DOI,title,abstract,issued,container-title,author,is-referenced-by-count"
# Default multi-source redundancy: any source can succeed. Crossref is key-free and has loose current limits (stable in actual testing).
# OpenAlex anonymous shared pool will limit the flow by IP (actual measurement 429/503, Retry-After 35-60 seconds);
# semantic_scholar_bulk is available without key and is ranked last. When there is an S2 key, resolved_providers will add the /paper/search correlation source.
DEFAULT_PROVIDERS = ("crossref", "openalex", "semantic_scholar_bulk")
# Default claim set: The judgment really depends on the rules "has anyone done it in external literature?"
DEFAULT_CLAIM_RULE_IDS = (
    "A-LITERATURE-POSITIONING-V1",
    "A-GAP-PUZZLE-V1",
    "B-DIALOGUE-ADVANCE-V1",
    "B-CONCEPT-MECHANISM-NOVELTY-V1",
)
STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "is", "are", "be",
    'Research', 'Paper', 'This article', 'Whether', 'Judgment', 'Reviews', '', 'and', 'and', 'and', 'or',
    # Question words must be eliminated: the title of the paper is often written as "Challenge or hindrance? How and when...",
    # These words take up the subject quota and will squeeze the truly distinctive "artificial intelligence" out of the search terms.
# Drop generic question words that contribute little to search specificity.
    "how", "when", "what", "why", "which", "who", "does", "do", "did", "can", "could", "should",
}
# The rule profile and the indexed prior-work corpus are in English. Claim-conditioned searches use English terms to align queries with that corpus.
# When a rule has no dedicated retrieval terms, the query records that it fell back to the rule text.
DEFAULT_CLAIM_QUERY_EN = {
    "A-LITERATURE-POSITIONING-V1": "literature positioning theoretical conversation prior research dialogue",
    "A-GAP-PUZZLE-V1": "research gap puzzle contradictory findings unresolved question",
    "B-DIALOGUE-ADVANCE-V1": "challenging assumptions integrating conflicting findings theoretical advancement",
    "B-CONCEPT-MECHANISM-NOVELTY-V1": "novel construct new mechanism theoretical proposition conceptual contribution",
}
# The default rules for inserting external documents as evidence into the reading package: those that determine "whether someone has already done it / the increment relative to the existing literature".
DEFAULT_EVIDENCE_RULE_IDS = (
    "A-LITERATURE-POSITIONING-V1",
    "A-GAP-PUZZLE-V1",
    "B-CONCEPT-MECHANISM-NOVELTY-V1",
    "B-DIALOGUE-ADVANCE-V1",
    "B-THEORY-INTEGRATION-V1",
    "B-INTERESTINGNESS-V1",
)
# Extract the claim field from paper_context. Claim uses the paper's own research questions/gaps/contributions, which is more relevant than using rule definitions.
CLAIM_FIELDS = (
    ("RQ", 'Research Question', ("research_question", "central_problem")),
    ("GAP", 'Research Gap', ("research_gap",)),
    ("CONTRIB", 'Main contribution', ("main_contribution",)),
    ("CLAIM", 'Core proposition', ("core_claim",)),
)
CONTEXT_PLACEHOLDER = 'To be judged by Agent based on the full text'
# Certainty when translation fails: claim type → general English search term (topic constraints are still provided by the paper title).
DEFAULT_CONTEXT_CLAIM_EN = {
    "RQ": "research question study investigates examines",
    "GAP": "research gap puzzle contradictory findings unresolved question",
    "CONTRIB": "theoretical contribution extends advances prior literature",
    "CLAIM": "core claim main argument proposition findings",
}
# These words are found in almost every paper abstract. Leaving them in the rearranged word list will only push the ranking to "generally related".
# is only removed from the rearranged word list, and the search terms are still retained (OpenAlex searches require these words).
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
    mailto: str = ""              # OpenAlex polite pool mailbox; filling it out will significantly reduce the probability of being restricted.
    claim_rule_ids: list[str] = field(default_factory=lambda: list(DEFAULT_CLAIM_RULE_IDS))
    claim_query_en: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CLAIM_QUERY_EN))
    # Which rules should be used to insert external documents as evidence blocks into the reading package (the ones that determine "whether someone has already done it").
    evidence_rule_ids: list[str] = field(default_factory=lambda: list(DEFAULT_EVIDENCE_RULE_IDS))
    # Whether to refuse to continue when no external documents can be retrieved.
    # **Default false (fail-safe)**: The scoring task does not fail when the external API is unavailable, but is marked in the report.
    # `external_evidence_unavailable`, and the rules that depend on field comparisons are blocked by Harness from being satisfied.
    # If you want to change it to "Don't run if you can't retrieve it" (I would rather stop than get a downgrade result), set it to true; at this time, the retrieval will run before the scoring.
    # The failure has not spent the score call, so the hard threshold is reasonable.
    require_external_evidence: bool = False
    # The upper limit of the number of external documents inserted into the reading package (each article will occupy several hundred characters of input).
    max_evidence_blocks: int = 8
    # Current limiting protection: When OpenAlex does not enter the courtesy pool, continuous requests will return 429 (2-3 of the 4 queries failed in the actual test).
    request_retries: int = 3
    retry_backoff_seconds: float = 1.5
    min_interval_seconds: float = 0.6
    cache_dir: str | None = None
    timeout_seconds: int = 20
    # Retrieve budget and stop conditions
    min_queries: int = 2          # At least this many different search terms must be sent before the end is allowed.
    max_queries: int = 6
    max_candidates: int = 60      # Maximum total number of candidate documents
    error_budget: int = 3         # Stop after failing this many times in a row
    per_query_limit: int = 20     # Single search return upper limit
    top_k: int = 10               # How many articles will be retained after rearrangement?
    summary_top_k: int = 5        # Several of these articles generate structured abstracts (0 = do not generate, save calls)
    min_top_score: float = 0.12   # When the rearrangement score reaches this value, it is considered "enough" and can be stopped early.
    rerank_weight_phrase: float = 3.0
    rerank_weight_token: float = 1.0
    year_from: int | None = None
    year_to: int | None = None
    # Current limiting protection: When OpenAlex does not enter the polite pool, burst requests will return 429 (actually 4 queries failed and 2-3 failed).
    request_retries: int = 4
    retry_backoff_seconds: float = 2.0
    min_interval_seconds: float = 1.2
    # When the entire army is wiped out in the first round (limited storm), retry the whole round after cooling down, and then determine failure.
    # The failed query does not write to the cache, so this round must go to the real network again.
    empty_retry_seconds: float = 20.0
    # Respect the server Retry-After: Don’t wait beyond this number of seconds, directly report the actionable reason (actual test OpenAlex
    # The anonymous shared pool will require a wait of 35-60 seconds; using a shorter backoff will inevitably fail).
    max_retry_after_seconds: float = 30.0

    # --- External literature search + external evidence verification (second phase increment) ---
    # external_evidence_verification_enabled: Innovative verification Agent + Harness switch for external evidence verification.
    # Search (enabled) and verification are divided into two levels: as long as external literature is enabled, it is also necessary to do "what is new compared to existing research"
    # Comparison judgment (one model call per claim + Crossref verification) and then open this. Turning it off depends on the rules for field comparisons
    # will be blocked by Harness and not satisfied, instead of letting the model assert "no one has done it" based on its own knowledge.
    external_evidence_verification_enabled: bool = True
    # Claim Extractor: First use 5 types of claim extraction, and then search one by one (turn it off to return to the old portrait field extraction).
    claim_extraction_enabled: bool = True
    # Multiple recall switch: query = 3-5 search terms for each claim; citation = local citation map of target paper references;
    # semantic = Semantic relevance rearrangement of claim/abstract/research question and candidate abstracts.
    query_route_enabled: bool = True
    citation_route_enabled: bool = True
    semantic_route_enabled: bool = True
    # Time leakage control: Prior literature only takes research before the publication year of the target paper (including the grace period).
    time_leak_guard: bool = True
    year_tolerance: int = 0
    # citation path: Continue to obtain citation relationships around the references of the target paper, and whether to include later documents that cite this article as candidates.
    citation_graph_depth: int = 1
    citation_related_limit: int = 8
    allow_citing_later_works: bool = False
    # Candidate Pool → External Evidence Set
    candidate_pool_limit: int = 40
    # Crossref metadata verification (DOI/title/year consistency)
    crossref_verify: bool = True
    crossref_max_checks: int = 8
    # The rule for judging "dependence on field comparison" requires at least how many pieces of prior literature to be satisfied.
    min_external_papers: int = 2
    # When the Novelty Agent is judged as novel but the prior literature coverage is insufficient, the confidence drops to this upper limit.
    low_coverage_confidence_cap: float = 0.4


@dataclass
class RelatedWork:
    'A candidate external document. After adding the provenance field, it can be used as a paper record in the External Evidence Repository.'

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
    # provenance: From which path, which search query, and when was this document recalled for reproducible auditing?
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
    # Metadata verification result (Crossref): ok / mismatch / unchecked
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
            # retrieval_query is synonymous with source_query: provenance is called retrieval_query externally.
            # source_query is the old field name, the report and the old self-test are still reading it, both are retained.
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
    'A search request: claim comes from the paper’s own research questions/gaps/contributions, and the search terms are in English (the corpus is in English).\n\n    The claim_extraction module will complete fields such as claim_type / source_evidence_id;\n    All default values are given here, and the old rule path (claims_from_rules) will not be affected.'

    claim_id: str
    label: str          # Claim label shown in reports
    text: str           # Original claim text retained for reports and diagnostics
    keywords_en: str    # English search term, used for search formula and rearrangement
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
    'Extracts searchable claims from paper_context as (claim_id, label, original claim text).'
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
    'Rewrite each research claim as an English search query. If translation fails, return a deterministic fallback term and mark its source.'
    if not claims or runtime_config is None:
        return {}
    prompt = (
        'You are preparing keywords for academic search. Rewrite each research claim as an English search keyword string:'
        '8-14 English words, only nouns and terms, no whole sentences, no quotation marks, no Boolean symbols such as and/or.'
        'output JSON: {"claims":[{"claim_id":"echo unchanged","search_terms":"English keyword string"}]}'
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
    'Construct a claim using the paper’s own research questions/gaps/contributions (preferred).'
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
    'Use rules to define and construct claim (a fallback when paper_context cannot extract claim).'
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


# The title of the paper is derived from the PDF file name, in the form of "Choudhury et al. - 2020 - real title".
# Leaving the author's name and year in the search query has two disadvantages: searching with the author's name will pull back the author's own papers.
# And the quota of keywords is wasted (2 of the first 5 keywords are wasted).
FILENAME_TITLE_PREFIX = re.compile(r"^\s*.{0,90}?\b(?:19|20)\d{2}\b\s*[-–—]\s*")


def clean_paper_title(title: str) -> str:
    'Strip off the "Author-Year-" prefix brought by the file name to get the real title.'
    text = str(title or "").strip()
    cleaned = FILENAME_TITLE_PREFIX.sub("", text).strip()
    return cleaned if len(cleaned) >= 12 else text


def _topic_tokens(paper_title: str, limit: int = 8) -> list[str]:
    'Thesis topic. The default value is 8: file names often truncate the title, and if the number is less, keywords such as "artificial intelligence" will be missed.'
    return _tokens(clean_paper_title(paper_title))[:limit]


def build_claim_query(rule: IndicatorRule, paper_title: str = "", extra: str = "") -> str:
    'Generate search terms according to claim (rule): indicator name + definition points + observation point keywords, with the thesis subject heading at the end.'
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
    'Determine whether the candidate document is the paper being reviewed.\n\n    The search formula is made up of the title of this article + the research question, so this article is almost certainly the highest-scoring hit (actual measured rerank 34.5,\n    Much more than real related work). If not eliminated, the comparison position of "Has anyone done this" will be occupied by this article itself.\n    The overall rearrangement score is artificially high, and the judgment is distorted - this time it was just that the model happened to see that "it is the article itself" and it did not make an error.'
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
    # titles include each other (OpenAlex sometimes only gives the main title, the title of this article comes from the file name and may have subtitles)
    a, b = _normalized_title(candidate), _normalized_title(paper)
    if a and b and min(len(a), len(b)) >= 20 and (a in b or b in a):
        return True
    return False


def _compose_query(terms: str, paper_title: str) -> str:
    'Paper subject word + claim search word, remove duplicates and spell it into search formula.\n\n    The subject words must be placed first: the search terms have an upper limit on length. When truncated, the general methodology claim words should be dropped first.\n    Instead of cutting off "What is this paper studying?" The previous way of writing put the subject at the end, and the actual search query was issued\n    It ends with "...prior research dialogue choudhury machine learn", with only half a word left on the topic.'
    topic = " ".join(_topic_tokens(paper_title, 8))
    seen: set[str] = set()
    ordered = [piece for piece in f"{topic} {terms}".split() if piece and not (piece in seen or seen.add(piece))]
    return " ".join(ordered)[:220]


def claim_retrieval_terms(rule: IndicatorRule, config: ExternalRetrievalConfig) -> tuple[str, list[str], str]:
    'Returns the claim search term, reranking vocabulary, and source. A configured English override takes priority; otherwise the rule text is used and marked as `rule_text`.'
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
    'Rearrange the word list: (claim word, phrase, thesis topic).\n\n    The subject word is returned separately: it represents "what this paper is studying" and is not the same thing as the claim word, and the scores will be recorded separately.\n    Common academic words are eliminated from the claim words, otherwise the ranking will be flattened by words such as "research/discovery/theory".'
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
    'Local deterministic rearrangement: phrase hits have high weight, word hits accumulate, and abstracts that are too short are slightly downgraded. Every point is open to interpretation.\n\n    Subject word hits are individually weighted (2.0/word): the claim word explains "why you are looking for it", the subject word explains "whether it talks about the same thing",\n    Only documents that match both levels should be ranked first.'
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
    'Actual search source: configuration sequence; Semantic Scholar will be automatically added as long as the S2 key exists.'
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
    'The abstract of OpenAlex is saved as an inverted index (word → position list), and the original text is spelled back by position.'
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
    'Convert OpenAlex entries into the field shape of Semantic Scholar, and the downstream rearrangement/deduplication logic does not need to be changed at all.'
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
    'Relevance search source. Anonymous perennial 429 (see paper_eval/semantic_scholar.py documentation),\n    Failure will throw and let search_external fall to the next source; this is the preferred source when there is a key.'
    rows = s2_relevance(query, config.per_query_limit, S2Config.from_external(config), config.year_from, config.year_to)
    return {"data": rows}


def _http_search_semantic_scholar_bulk(query: str, config: ExternalRetrievalConfig) -> dict[str, Any]:
    'S2 batch search source (available without key), truncated into the same envelope shape as /paper/search.\n\n    ponytail: bulk has no relevance sorting, the client takes the first N items according to citationCount:desc\n    ("The top N most cited articles in the batch with the most hits"). After getting the key, replace it with the semantic_scholar source to restore the relevance sorting.'
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
    'Remove JATS/HTML tags and entities in Crossref summaries.'
    plain = re.sub(r"<[^>]+>", " ", str(text or ""))
    for entity, char in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"'), ("&#x2010;", "-"), ("&nbsp;", " ")):
        plain = plain.replace(entity, char)
    return " ".join(plain.split())


def _normalize_crossref(payload: Any) -> list[dict[str, Any]]:
    'Convert Crossref entries into the field shape of Semantic Scholar, and the downstream rearrangement/deduplication logic does not need to be changed.'
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
    raise ValueError(f"Unknown external search source: {provider}")


def _cached_or_fetch(
    provider: str,
    query: str,
    config: ExternalRetrievalConfig,
    loader: Callable[[], Any],
) -> tuple[list[dict[str, Any]], str | None]:
    'returns (entry, error message). The cache is isolated according to "retrieval source + search formula", and old results will not be read when changing the source.'
    cache_file = _cache_path(config.cache_dir, f"{provider}|{query}|{config.per_query_limit}")
    if cache_file and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8")).get("data", []), None
        except (OSError, json.JSONDecodeError):
            pass
    try:
        items = [item for item in (loader() or []) if isinstance(item, dict)]
    except Exception as exc:  # External retrieval failure is not fatal
        return [], f"{provider}: {exc}"
    if cache_file:
        cache_file.write_text(
            json.dumps({"provider": provider, "query": query, "data": items}, ensure_ascii=False),
            encoding="utf-8",
        )
    return items, None


def _fetcher_items(payload: Any) -> list[dict[str, Any]]:
    'The return of the injected fetcher is compatible with two shapes: S2 envelope {"data": [...]} or a direct list of entries.'
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _retry_after_seconds(exc: Exception) -> float | None:
    'Read the server’s Retry-After (seconds). It is much more authoritative than local backoff.'
    headers = getattr(exc, "headers", None)
    raw = headers.get("Retry-After") if headers else None
    if not raw:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _fetch_with_retry(provider: str, query: str, config: ExternalRetrievalConfig) -> list[dict[str, Any]]:
    "The real network request has retry backoff and respects the server's Retry-After.\n\n    The external academic API will limit the flow under sudden requests (OpenAlex returns 429 when it is not in the polite pool and requires to wait 35-60 seconds).\n    Only retry retryable errors: current limit, timeout, 5xx; the rest (such as 400 parameter error) are directly thrown to avoid idling."
    attempts = max(1, int(config.request_retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _raw_fetch(query, provider, config)
        except Exception as exc:  # noqa: BLE001 - You need to judge whether it can be retried based on the error text and response header.
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
                    f"{exc} (retrieval source requires waiting {wait:.0f} seconds, exceeding the upper limit of {config.max_retry_after_seconds:.0f} seconds:"
                    f"{provider} Anonymous shared pool current limit, it is recommended to fill in the contact email in external_retrieval.mailto to enter the OpenAlex courtesy pool)"
                ) from exc
            if attempt + 1 >= attempts:
                if wait is not None:
                    raise RuntimeError(
                        f"{exc} (retrieval source requires waiting {wait:.0f} seconds: {provider} anonymous shared pool current limit,"
                        'It is recommended to fill in the contact email in external_retrieval.mailto to enter the polite pool)'
                    ) from exc
                raise
            # If the server is given Retry-After, it will listen to it; if it is not given, it will use local exponential backoff.
            time.sleep(wait if wait is not None else config.retry_backoff_seconds * (2 ** attempt))
    if last_error is not None:
        raise last_error
    return []


def search_external(
    query: str,
    config: ExternalRetrievalConfig,
    fetcher: Callable[[str, int], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    'attempts to retrieve sources in the configured order, returning (entry, error, source hit).\n\n    Each source is calculated separately: when a source is unavailable as a whole, it will automatically fall to the next source. Failure will only be considered as failure if all sources fail.\n    `fetcher` is used for offline self-test. Only the first source is used when injecting to ensure that the offline result is determined (retry and rate limiting are also skipped).'
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
    'Retrieve and rearrange related documents by claim.\n\n    Budget control: at least `min_queries` different search terms; the total number of candidates does not exceed `max_candidates`;\n    Stop when continuous failure reaches `error_budget` or exceeds `max_queries`;\n    After rearrangement, the process will stop early when the highest score reaches `min_top_score` and the number of candidates reaches `top_k`.'
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
    claim_terms: dict[str, tuple[str, str]] = {}   # claim_id -> (search term, source)
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
            progress_callback(f"External search ({claim.label}): {query[:40]}")
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

    # Rearrange: claim keywords/phrases are used as the basis for scoring. Each claim is ranked once, and the highest score is obtained by merging and eliminating duplicates.
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
    'Generate structured abstracts (methods/findings/experiments/similar/different) for each relevant work. Failures are logged into summary_error.'
    if config is None or top_k <= 0:
        return works
    prompt = (
        'You are organizing related work for an article to be reviewed. For the given piece of related work, output JSON:'
        '{"method":"What is its method","findings":"What are the main findings","experiments":"What are the experimental results",'
        '"similar":"What is similar to this paper", "different":"What is different"}.'
        'Write one or two concise English sentences for each item, based only on the supplied title and abstract. If the abstract is missing, write "The abstract is missing and cannot be judged." Do not invent details.'
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
    'Claim set for external retrieval: the rule named in the configuration ∩ The type of this paper is within the scope.'
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
    'External search for a whole paragraph: claim → real search → rearrange → five-question summary.\n\n    The claim is preferred to the paper\'s own research questions/gaps/contributions (paper_context). If it cannot be extracted, the rule definition will be returned.\n    When `require_external_evidence` is opened, an error will be thrown if no external documents can be retrieved - retrieval now runs before scoring.\n    At this point, no scoring calls have been spent, so it is far better to "stop" than to let the model assert "no one has done it" out of thin air.\n\n    The scoring pipeline now goes `collect_external_literature` (Multiple Recall + XEV-* Evidence Block); this function only retains\n    Old image/rule claim single search path, used for existing self-checks and manual troubleshooting.'
    cfg = retrieval_config or ExternalRetrievalConfig()
    if not cfg.enabled:
        return {"enabled": False, "stop_reason": "disabled", "related_works": [], "queries": [], "failures": [], "status": external_retrieval_status(cfg)}
    # The title comes from the file name, first strip off the "Author - Year -" prefix; use clean titles for both search and abstract prompt words.
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
        # The first round may be empty during the current restriction storm. Cool down first and then try again in the whole round. If it is still empty, it will be judged as a failure.
        if cfg.empty_retry_seconds > 0:
            if progress_callback:
                progress_callback(f"The first round of external retrieval is empty, wait {cfg.empty_retry_seconds:.0f} seconds and try again in the whole round.")
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
        # You can't get the literature even if you go to the real network: just stop, don't let the model judge "no one has done it" without external comparison.
        detail = "; ".join(item.get("error", "") for item in result["failures"][:3]) or result.get("stop_reason")
        raise RuntimeError(
            'EXTERNAL_EVIDENCE_REQUIRED: The external literature search yielded no results and was stopped before invoking the scoring model.'
            f"Search source={result['status'].get('providers')}; Reason={detail}."
            'If the retrieval source continues to be limited, please fill in the contact email in external_retrieval.mailto to enter the OpenAlex courtesy pool;'
            'To continue in an offline environment, please set require_external_evidence to false.'
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
    'Main entrance to external literature retrieval (second stage): Claim Extractor → Three-way recall → Time filter → Reranker\n    → Crossref Check → External Evidence Set → XEV-* Evidence Block.\n\n    The return structure retains the old fields (related_works / queries / claims), and the reporting layer and old self-test do not need to be changed.\n    If `external_evidence_unavailable` is true, it means that no external evidence was obtained this time - the scoring can continue,\n    But rules that rely on field comparisons will be blocked by Harness from being satisfied.'
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

    # B1: Internal contribution extraction (claims always contain source_evidence_id, pointing back to this article’s Evidence Repository)
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

    # B2: External search (three-way recall + rearrangement)
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

    # Crossref Metadata Check (DOI/Title/Year Consistency)
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
    'Turn the retrieved relevant work into citationable evidence blocks (EXT-*), each with a deterministic relevance level.\n\n    The relevance level is calculated based on the relative position of the rearranged scores in the same batch (high ≥ 75% with the highest score, medium ≥ 50%, and the rest are low),\n    This way the model and readers see verifiable relative strengths rather than an isolated score.\n\n    Multi-way recall (claim_extractor path) will go through the XEV-* evidence block of literature_search; this EXT-* path\n    Reserved for old portrait/rule claim processes and existing self-checks.'
    works = result.get("related_works") or []
    if not works:
        return []
    top = max(float(work.get("rerank_score") or 0.0) for work in works[:max_blocks])
    blocks: list[Evidence] = []
    for index, work in enumerate(works[:max_blocks], start=1):
        score = float(work.get("rerank_score") or 0.0)
        tier = 'high' if top > 0 and score >= top * 0.75 else ('Medium' if top > 0 and score >= top * 0.5 else 'low')
        summary = work.get("summary") if isinstance(work.get("summary"), dict) else {}
        lines = [
            (
                f"[External literature {index}｜Relevance {tier}｜{work.get('year') or 'year unavailable'}｜"
                f"{work.get('venue') or 'venue unavailable'}｜He cited {work.get('citation_count')}]"
            ),
            f"Title: {work.get('title')}",
        ]
        if work.get("doi"):
            lines.append(f"DOI：{work.get('doi')}")
        abstract = str(work.get("abstract") or "").strip()
        lines.append(f"Summary: {abstract[:600] if abstract else '（No abstract available）'}")
        for key, label in (("method", 'method'), ("findings", 'found'), ("similar", 'similar'), ("different", 'different')):
            value = str(summary.get(key) or "").strip()
            if value:
                lines.append(f"{label}：{value}")
        text = "\n".join(lines)
        blocks.append(
            Evidence(
                evidence_id=f"EXT-{index}",
                paper_id=paper_id,
                section='External literature (OpenAlex search)',
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
    'Reads the `external_retrieval:` section from the top level of the configuration file.\n\n    Deliberately placed outside `agent_runtime`: `AgentRuntimeConfig` will directly expand the keys of this paragraph into construction parameters.\n    An additional unknown key will result in an error, so the configuration of external retrieval must be separated into separate segments.'
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
        raise ValueError(f"external_retrieval configuration contains unknown keys: {unknown}")
    return ExternalRetrievalConfig(**values)


def external_retrieval_status(config: ExternalRetrievalConfig) -> dict[str, Any]:
    'Status summary for workbench/report use.'
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

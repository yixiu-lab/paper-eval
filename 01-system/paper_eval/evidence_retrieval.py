'Evidence retrieval layer: bag-of-words recall + LLM rearrangement.\n\nWhy is there this layer: `agents.build_reading_pack` is a purely deterministic heuristic that relies on hard-coded English keywords to select candidates\nEvidence, incomplete coverage. The measured D dimension (research design and method quality) with a score of 25 points only received an average of 8.9 candidates, and the model repeatedly judged\nPartial, the reason is that "candidate evidence does not cover reliability sources and validity test instructions" and "does not cover endogeneity or selection bias" - these\nThe content was included in the paper, but it didn’t hit the keyword list and didn’t make it into the candidate package at all.\n\nThe approach here is two-stage: first use the English search terms declared by the rules to do bag-of-word recall (which can cover expressions not written in the keyword list),\nThen let the model select the ones that really should be included in the reading package from the recalled candidates (which can suppress the false positives of bag-of-word recall).\nPure standard library, reusing tokenize/vectorize/cosine of retrieval.py without introducing any third-party dependencies.\n\nPerformance: hundreds of paragraphs × dozens of query terms, milliseconds in pure Python, without optimizations such as inverted indexing.'

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .external_retrieval import is_external_evidence
from .llm import AgentRuntimeConfig, call_plain_json
from .models import Evidence
from .retrieval import cosine, vectorize
from .rules import IndicatorRule

# Rule id and evidence_location are not free text, and most of the words extracted have no retrieval value.
# (dimension prefix, version number, pure chapter name). If these words are mixed into the query vector, the entire vector will be normalized.
# pulls the bias, making the weight of the real search terms lower. So list it explicitly and throw it away.
_GENERIC_TERMS = {
    "the", "and", "for", "with", "that", "this", "from", "v1", "v2", "v3",
    "abstract", "introduction", "conclusion", "discussion", "methods", "method",
    "results", "result", "findings", "finding", "theory", "appendix", "literature",
    "review", "hypotheses", "tables", "figures", "paper", "study", "research",
    "section", "practical", "implications", "limitations", "data", "analysis",
    "supplementary", "materials", "availability", "design", "sample", "measures",
    "coding", "motivation", "robustness", "checks",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]{2,}")


def _is_reference_evidence(item: Evidence) -> bool:
    'is equivalent to `agents.is_reference_evidence`.\n\n    Why rewrite it here instead of import: agents.py and then import this module (replace select_evidence\n    Connect to the call point of build_reading_pack), this module then reversely import agents to form a loop. There are only two lines of logic,\n    Copying is more cost-effective than introducing the complexity of lazy import; this needs to be synchronized when changing agents.is_reference_evidence.'
    return item.section.lower() == "references" or item.text.strip().lower().startswith(("references", "bibliography"))


def _excluded(item: Evidence) -> bool:
    'Evidence that it cannot enter the search pool: the reference block has no substantial content; EXT-* external documents are only explicitly inserted into the specified rules.'
    return _is_reference_evidence(item) or is_external_evidence(item)


def build_evidence_vectors(evidences: list[Evidence]) -> dict[str, dict[str, int]]:
    'evidence_id → word frequency vector. With section on the key: the same paragraph has different search values \u200b\u200bin Methods and Discussion.'
    vectors: dict[str, dict[str, int]] = {}
    for item in evidences:
        if _excluded(item):
            continue
        vectors[item.evidence_id] = vectorize(f"{item.section}\n{item.text}")
    return vectors


def build_query_string(query_terms: list[str]) -> str:
    return " ".join(term for term in query_terms if term)


def recall_evidence(
    query_terms: list[str],
    vectors: dict[str, dict[str, int]],
    evidences_by_id: dict[str, Evidence],
    top_k: int,
) -> list[tuple[float, Evidence]]:
    'Bag-of-word recall: Return the top_k in descending order of cosine similarity, and discard those with 0 points.\n\n    0 points are lost directly: when none of the query words are hit, the cosine similarity is 0 instead of "somewhat related".\n    Leaving them in only pushes the evidence of real hits out of top_k.'
    query_vector = vectorize(build_query_string(query_terms))
    if not query_vector:
        return []
    scored: list[tuple[float, Evidence]] = []
    for evidence_id, vector in vectors.items():
        item = evidences_by_id.get(evidence_id)
        if item is None:
            continue
        score = cosine(query_vector, vector)
        if score > 0:
            scored.append((score, item))
    # is sorted stably by evidence_id at the same time to ensure that the recall sequence is reproducible (LLM only rearranges and does not generate from scratch, certainty is very important).
    scored.sort(key=lambda pair: (-pair[0], pair[1].evidence_id))
    return scored[: max(0, int(top_k))]


def build_query_terms(rule: IndicatorRule, paper_context: dict[str, Any] | None = None) -> list[str]:
    'rule.\n\n    Prefer `rule.retrieval_terms`, the English search vocabulary declared in the rule profile. If it is empty, fall back to English tokens from the rule ID and evidence location. Rule prose is not parsed into search terms.'
    terms = [str(term).strip().lower() for term in (getattr(rule, "retrieval_terms", None) or []) if str(term).strip()]
    if not terms:
        fallback: list[str] = []
        fallback.extend(match.lower() for match in _TOKEN_RE.findall(str(rule.rule_id)))
        for location in getattr(rule, "evidence_location", None) or []:
            fallback.extend(match.lower() for match in _TOKEN_RE.findall(str(location)))
        fallback.extend(match.lower() for match in _TOKEN_RE.findall(str(getattr(rule, "indicator_name", ""))))
        terms = [term for term in fallback if term not in _GENERIC_TERMS]

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term and term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped


def _config_int(config: Any, name: str, default: int) -> int:
    'reads the positive integer in the configuration. Writing empty values \u200b\u200bor strings in the configuration file will not interrupt the process.'
    try:
        value = int(getattr(config, name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def build_rerank_payload(
    rule: IndicatorRule,
    candidates: list[Evidence],
    limit: int,
    max_chars: int,
) -> dict[str, Any]:
    'rearranges the user payload of the call. The evidence text is truncated here - this is the copy for the model, not the final reading package.'
    judgement_values = getattr(rule, "judgement_values", None) or {}
    return {
        "indicator_name": str(getattr(rule, "indicator_name", "") or ""),
        "definition": str(getattr(rule, "definition", "") or ""),
        "judgement_values": {str(key): str(value) for key, value in judgement_values.items()},
        "budget": int(limit),
        "candidates": [
            {
                "evidence_id": item.evidence_id,
                "section": item.section,
                "page": item.page,
                "text": (item.text or "")[: max(0, int(max_chars))],
            }
            for item in candidates
        ],
    }


# Rearrange prompt. The model is required to only do "filtering + sorting + one sentence reason" and does not produce any scoring content:
# The value of this step is to suppress the false positives in bag-of-word recall. Unauthorized judgment will pollute the subsequent scoring links.
_RERANK_SYSTEM_PROMPT = (
    'You are an academic paper evidence screening assistant. Give you a scoring rule (indicator name, definition, gear anchor point) and a batch of candidate evidence fragments,'
    'Your only task is to pick out the fragments from the candidates that the judges really need to read when judging this rule, and rank them in order of importance from high to low.'
    'Judgment basis: for each element mentioned in the gear anchor point of this rule, whether there is a corresponding expression in the candidate; whether the fragment describes the specific'
    'practices, data, tests or results, rather than background or general conclusions.'
    'Constraints:'
    '1. You can only choose from candidates, and you are not allowed to make up evidence_id;'
    f"2. Select at most {{budget}} items (this is the hard upper limit). Select as many candidates as you can. You would rather not select those that are obviously irrelevant; \n"
    '3. Only filter and sort, do not give grade judgment or score;'
    '4. For each item, give a concise English reason (within 30 words) explaining which part of the rule it covers.'
    'only returns a JSON object, no markdown, no explanation, format:'
    '{"selected":[{"evidence_id":"EV-01","reason":"Relevant evidence for the assigned rule"}]}'
)


def _extract_selected_ids(parsed: Any) -> tuple[list[str], dict[str, str]]:
    'Extract the evidence_id list and reasons from the model return.\n\n    Fault-tolerant to the extent that "as long as the id appears anywhere, it can be used": the model occasionally returns plain text ({"ids": [...]}),\n    Occasionally return {"selected": [...]}, occasionally with a reason object. If it cannot be obtained, it will return empty, and the caller will downgrade it.'
    if not isinstance(parsed, dict):
        return [], {}
    raw = parsed.get("selected")
    if raw is None:
        raw = parsed.get("evidence_ids")
    if raw is None:
        raw = parsed.get("ids")
    if raw is None:
        return [], {}

    ids: list[str] = []
    reasons: dict[str, str] = {}
    if isinstance(raw, dict):
        raw = list(raw.items())
    if not isinstance(raw, list):
        return [], {}
    for entry in raw:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, dict):
            evidence_id = entry.get("evidence_id") or entry.get("id")
            if evidence_id:
                ids.append(str(evidence_id))
                reason = entry.get("reason")
                if reason:
                    reasons[str(evidence_id)] = str(reason)
        elif isinstance(entry, (list, tuple)) and entry:
            ids.append(str(entry[0]))
            if len(entry) > 1 and entry[1]:
                reasons[str(entry[0])] = str(entry[1])
    # Take a step back before you get the reasons: align the two parallel lists of top-level evidence_ids and reasons by position.
    if not reasons:
        flat_ids = parsed.get("evidence_ids")
        flat_reasons = parsed.get("reasons")
        if isinstance(flat_ids, list) and isinstance(flat_reasons, list):
            for index, evidence_id in enumerate(flat_ids):
                if index < len(flat_reasons):
                    reasons[str(evidence_id)] = str(flat_reasons[index])
    return [str(item) for item in ids], reasons


def _call_rerank_llm(payload: dict[str, Any], config: AgentRuntimeConfig) -> Any:
    'Extract a separate layer: the monkeypatch function in the self-test can replace the model response without actually adjusting the API.'
    budget = payload.get("budget", 0)
    system_prompt = _RERANK_SYSTEM_PROMPT.replace("{budget}", str(budget))
    return call_plain_json(system_prompt, payload, config)


def _fill_to_limit(ordered: list[Evidence], padding: list[Evidence], limit: int) -> list[Evidence]:
    'Remove duplicates according to evidence_id, and fill ordered to the limit bar (if insufficient, return to the existing one).\n\n    Complete the results of reading the package using heuristics instead of "select only a few of the models". There are two reasons:\n    1. The number of reading packages directly affects the absolute value of the score (the actual measurement of 10 paragraphs → 20 paragraphs moves up by about 7 points), and the number of items shrinks.\n       It is not comparable to the full number of rules;\n    2. What’s even more terrible is the reverse effect: the original intention of the retrieval layer is to expand evidence coverage. If there are few recall hits, the package will drop from 10 to 2.\n       The rule of constant partial judgment will be even more fatal - worse than not opening the search layer.'
    seen = {item.evidence_id for item in ordered}
    result = list(ordered)
    for item in padding:
        if len(result) >= limit:
            break
        if item.evidence_id in seen:
            continue
        result.append(item)
        seen.add(item.evidence_id)
    return result


def rerank_evidence(
    rule: IndicatorRule,
    candidates: list[Evidence],
    config: AgentRuntimeConfig,
    limit: int,
    fallback: list[Evidence] | None = None,
) -> list[Evidence]:
    'One LLM call reorders candidates; any exception/bad return will be returned to the recall order, and exceptions will never be thrown to interrupt the entire scoring.\n\n    The scoring process is a chain: failure to rearrange a certain rule cannot cause the overall scoring of this paper to fail - failure will return to the recall order,\n    The worst thing is to just return the sub-optimal candidate package of "bag of words recall + heuristic completion", which is more controllable than the failure of the entire article.\n    `fallback` is the result of build_reading_pack, used to make up for the limit; if not passed, only candidates are used to make up.'
    if not candidates:
        return (fallback or [])[:limit]
    # Completion pool = recall order first + heuristic package last, _fill_to_limit will remove duplicates.
    # fills up the limit unconditionally and does not do "discount based on the number of selected models": the number of evidence items itself will move the score
    # (actually measured 10 segments → 20 segments moved up by about 7 points), the number of old and new links is inconsistent, and the A/B score difference cannot be distinguished
    # Is “the evidence more relevant” or “the evidence is less”. The added paragraphs also passed the relevance threshold of the old heuristic.
# The reason given by the
    padding = [*candidates, *(fallback or [])]
    max_chars = _config_int(config, "evidence_rerank_max_chars", 240)
    payload = build_rerank_payload(rule, candidates, limit, max_chars)
    try:
        parsed = _call_rerank_llm(payload, config)
        # model is temporarily discarded (ponytail: currently we only need to audit "whether the evidence_id in the final package is correct",
        # has no consumers. When A+B actually runs the batch and finds that the rearrangement is biased, add the reasons_out parameter of rerank_evidence).
        selected_ids, _reasons = _extract_selected_ids(parsed)
    except Exception:
        # LLM error/timeout/illegal JSON (call_plain_json will throw RuntimeError internally): silent downgrade.
        return _fill_to_limit(candidates[:limit], padding, limit)

    known = {item.evidence_id: item for item in candidates}
    ordered: list[Evidence] = []
    seen: set[str] = set()
    for evidence_id in selected_ids:
        item = known.get(evidence_id)
        if item is None or evidence_id in seen:
            continue
        ordered.append(item)
        seen.add(evidence_id)
    if not ordered:
        # id is not known at all/the model returns empty: the recall sequence is also returned.
        return _fill_to_limit(candidates[:limit], padding, limit)

    # When the model selection is not satisfied with the limit, it is filled in the recall order and then used heuristics (see the description of _fill_to_limit).
    return _fill_to_limit(ordered, padding, limit)


PACK_CACHE_FORMAT = 2


def evidence_cache_scope(evidences: list[Evidence], limit: int, config: Any) -> str:
    'The scope of the evidence package cache: paper evidence set + number of items + recall/truncation parameters (all deterministic input).\n\n    Why not put the "candidate sequence" into the hit condition: the candidate comes from the search term, and the search term comes from `paper_context`, which is\n    LLM products, **each batch is different** (actually tested the same paper twice, and the 8 subfields of paper_profile are all different).\n    Requiring candidates to be consistent is equivalent to requiring LLM to be self-consistent, and the cache must be completely invalid - this is the reason why 17/34 rules are still floating in the actual measurement on 2026-09-29.\n    Here it is changed to only recognize "the paper has not changed, the rules have not changed, and the budget has not changed", so that the evidence package is truly frozen; the cost is that when paper_context\n    When making a major change (changing the image caliber), the packages will still be the ones selected in the previous batch, and you need to manually clear the cache directory to force re-selection.'
    payload = json.dumps(
        {
            "evidence": sorted(item.evidence_id for item in evidences),
            "limit": limit,
            "top_k": _config_int(config, "evidence_recall_top_k", 30),
            "rerank_max_chars": _config_int(config, "evidence_rerank_max_chars", 400),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _rule_fingerprint(rule: IndicatorRule) -> str:
    'Rule fingerprint: Changing the search terms/dimensions/applicable scope is equivalent to changing a rule, and the cache must be invalidated.'
    payload = json.dumps(
        {
            "rule_id": rule.rule_id,
            "dimension": rule.dimension_id,
            "terms": sorted(str(term) for term in (getattr(rule, "retrieval_terms", None) or [])),
            "locations": sorted(str(item) for item in (getattr(rule, "evidence_location", None) or [])),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _pack_cache_path(cache_dir: str, scope: str, rule: IndicatorRule, limit: int) -> Path:
    digest = hashlib.sha1(
        f"v{PACK_CACHE_FORMAT}|{scope}|{_rule_fingerprint(rule)}|{rule.rule_id}|{limit}".encode("utf-8")
    ).hexdigest()[:16]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"pack_{digest}.json"


def _cached_pack(path: Path, evidences_by_id: dict[str, Evidence], limit: int) -> list[Evidence] | None:
    'Hit condition: The format/number of entries are consistent, and each ID in the package can be found in the current evidence set (if it cannot be found, it means that the paper analysis has changed).'
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("format") != PACK_CACHE_FORMAT or data.get("limit") != limit:
        return None
    pack_ids = data.get("pack") or []
    if any(evidence_id not in evidences_by_id for evidence_id in pack_ids):
        return None
    return [evidences_by_id[evidence_id] for evidence_id in pack_ids]


def _store_pack(path: Path, candidates: list[Evidence], pack: list[Evidence], limit: int) -> None:
    'only stores the id sequence and does not store the text (there is no paper content in the cache file and can be safely placed under runs/).'
    path.write_text(
        json.dumps(
            {
                "format": PACK_CACHE_FORMAT,
                "limit": limit,
                "candidates": [item.evidence_id for item in candidates],
                "pack": [item.evidence_id for item in pack],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def select_evidence(
    rule: IndicatorRule,
    evidences: list[Evidence],
    agent: Any,
    limit: int,
    paper_context: dict[str, Any] | None = None,
    config: AgentRuntimeConfig | None = None,
    cache_scope: str | None = None,
) -> list[Evidence]:
    'General entrance for evidence selection. The parent agent can change the call point of agents.build_reading_pack here.\n\n    Contract: The number of items returned is not less than the number of items in build_reading_pack under the same parameters - both are limit when the text is enough.\n    When the switch is turned off, the behavior is exactly the same as the old build_reading_pack (transparent transmission one by one, without any packaging),\n    In this way, "turning off the new link = returning to the old behavior" can be proven by self-examination, not by reading comprehension.\n\n    When `cache_scope` is given (in conjunction with `evidence_pack_cache_dir` in the configuration), the LLM rearrangement hop will be reused by the cache:\n    Only this jump is non-deterministic, recall and completion are purely deterministic calculations.'
    # agents.py will import this module later, so the import must be delayed within the function, and module-level import will form a loop.
    from .agents import build_reading_pack

    if config is None or not bool(getattr(config, "evidence_retrieval_enabled", False)):
        return build_reading_pack(rule, evidences, agent, limit, paper_context)

    # The heuristic package is good here: it not only does the whole package fallback when the recall is empty, but also makes up for the insufficient recall hits.
    legacy = build_reading_pack(rule, evidences, agent, limit, paper_context)

    vectors = build_evidence_vectors(evidences)
    evidences_by_id = {item.evidence_id: item for item in evidences}
    query_terms = build_query_terms(rule, paper_context)
    top_k = _config_int(config, "evidence_recall_top_k", 30)
    recalled = recall_evidence(query_terms, vectors, evidences_by_id, top_k)
    if not recalled:
        # The recall is empty (the search term has no intersection with the text): fallback heuristic, do not give empty packets to the model.
        return legacy

    candidates = [item for _score, item in recalled]
    if not bool(getattr(config, "evidence_rerank_enabled", True)):
        return _fill_to_limit(candidates[:limit], legacy, limit)

    cache_dir = str(getattr(config, "evidence_pack_cache_dir", "") or "")
    path = _pack_cache_path(cache_dir, cache_scope, rule, limit) if (cache_dir and cache_scope) else None
    if path is not None:
        cached = _cached_pack(path, evidences_by_id, limit)
        if cached is not None:
            return cached
    pack = rerank_evidence(rule, candidates, config, limit, legacy)
    if path is not None:
        _store_pack(path, candidates, pack, limit)
    return pack


__all__ = [
    "build_evidence_vectors",
    "build_query_string",
    "build_query_terms",
    "build_rerank_payload",
    "evidence_cache_scope",
    "recall_evidence",
    "rerank_evidence",
    "select_evidence",
]

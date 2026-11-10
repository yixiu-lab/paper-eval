'External retrieval self-test: budget and stop conditions, claim sources and retrieval formulas, reordering interpretability, caching, failure downgrade, multiple sources, external evidence blocks, hard thresholds.\n\nFor the offline part, the injected fake fetcher is used, and no external API is used; finally, a real probe can be optional (default is OpenAlex, no key required).\n\nRun:\n    python ".\\scripts\\self_test_external_retrieval.py" # Offline only\n    python ".\\scripts\\self_test_external_retrieval.py" --online # Extra exploration'

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.external_retrieval import (
    ExternalRetrievalConfig,
    _normalize_crossref,
    _normalize_openalex,
    build_claim_query,
    claims_from_paper_context,
    claims_from_rules,
    clean_paper_title,
    collect_related_works,
    external_evidence_blocks,
    load_external_retrieval_config,
    rerank,
    resolved_providers,
    retrieve_related_works,
    search_external,
)
from paper_eval.rules import load_rule_profile

ROOT = Path(__file__).resolve().parents[1]
FAKE_ITEMS = [
    {
        "title": "Algorithmic management and employee perceptions of justice",
        "abstract": "We examine how algorithmic management shapes perceived justice and employee reactions in selection contexts. " * 4,
        "year": 2021,
        "venue": "Journal of Management",
        "authors": [{"name": "A. Author"}],
        "citationCount": 120,
        "externalIds": {"DOI": "10.1/fake-1"},
    },
    {
        "title": "Unrelated work on protein folding",
        "abstract": "A study of molecular dynamics.",
        "year": 2019,
        "venue": "Nature",
        "authors": [{"name": "B. Author"}],
        "citationCount": 900,
        "externalIds": {"DOI": "10.1/fake-2"},
    },
]
OPENALEX_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.5465/amj.2020.0001",
            "display_name": "Algorithmic management and employee justice perceptions",
            "publication_year": 2021,
            "cited_by_count": 210,
            "primary_location": {"source": {"display_name": "Academy of Management Journal"}},
            "authorships": [{"author": {"display_name": "C. Scholar"}}],
            "abstract_inverted_index": {
                "Algorithmic": [0],
                "management": [1],
                "shapes": [2],
                "justice": [3],
                "perceptions.": [4],
            },
        }
    ]
}


CROSSREF_PAYLOAD = {
    "message": {
        "items": [
            {
                "DOI": "10.1/crossref-demo",
                "title": ["Algorithmic Management: Scoping Review on Employee Behaviour"],
                "abstract": "<jats:p>We review <jats:italic>algorithmic</jats:italic> management and its effects.</jats:p>",
                "issued": {"date-parts": [[2025, 3]]},
                "container-title": ["Human Resource Management Review"],
                "author": [{"given": "A.", "family": "Author"}],
                "is-referenced-by-count": 42,
            }
        ]
    }
}


def main() -> int:
    profile = load_rule_profile(ROOT / 'rules' / "rules_v1.yaml")

    # 1) Generate search formulas according to rules: different rules give different search formulas
    rule_a = profile.rules_by_id["A-QUESTION-CLARITY-V1"]
    rule_d = profile.rules_by_id["D-MODEL-IDENTIFICATION-V1"]
    query_a = build_claim_query(rule_a, "Machine learning and human capital")
    query_d = build_claim_query(rule_d, "Machine learning and human capital")
    assert query_a and query_d and query_a != query_d, (query_a, query_d)
    assert rule_a.indicator_name in query_a
    print(f"Rule search formula example: \n {rule_a.rule_id} -> {query_a[:80]}\n {rule_d.rule_id} -> {query_d[:80]}")

    # 2) Offline running: budget, stop conditions, deduplication, rearrangement
    calls = {"count": 0}

    def fake_fetcher(query: str, limit: int):
        calls["count"] += 1
        return {"data": FAKE_ITEMS}

    config = ExternalRetrievalConfig(
        enabled=True, min_queries=2, max_queries=4, max_candidates=10, top_k=5, summary_top_k=0
    )
    rule_claims = claims_from_rules([(rule.rule_id, rule) for rule in profile.rules[:4]], config)
    result = retrieve_related_works(rule_claims, "Machine learning and human capital", config, fetcher=fake_fetcher)
    assert result["enabled"] is True
    assert calls["count"] >= config.min_queries, calls
    assert calls["count"] <= config.max_queries, calls
    assert result["candidates"] == 2, result["candidates"]        # Two DOIs, only 2 articles after deduplication
    assert result["stop_reason"] in {"enough", "queries_exhausted", "max_queries", "below_min_queries"}
    top = result["related_works"][0]
    assert "justice" in top["title"].lower() or "algorithmic" in top["title"].lower(), top["title"]
    assert top["rerank_score"] > result["related_works"][1]["rerank_score"]
    assert "matched_claim" in top["rerank_detail"]
    print(f"Offline search: Issue {calls['count']} claim search terms, candidate {result['candidates']} articles,"
          f"Stop reason {result['stop_reason']}, Top1 \"{top['title'][:36]}\" score {top['rerank_score']:.2f}")

    # 3) Rearrangement can be explained: the hit item must appear in the details
    score, detail = rerank(
        type("W", (), {"title": "algorithmic management justice", "abstract": "algorithmic management " * 20})(),
        {"algorithmic", "management"},
        ["algorithmic management"],
    )
    assert detail["phrase_hits"] >= 1 and detail["token_hits"] >= 2, detail
    assert score > 0
    print(f"rearrangement can be explained: score {score:.2f}, detail {detail}")

    # 4) Failure is not fatal: fetcher keeps throwing errors → go to error_budget to stop and record failures
    def broken(query: str, limit: int):
        raise RuntimeError("HTTP 429 rate limited")

    failed = retrieve_related_works(rule_claims, "t", ExternalRetrievalConfig(enabled=True, error_budget=2), fetcher=broken)
    assert failed["failures"] and failed["stop_reason"] == "error_budget", failed["stop_reason"]
    assert failed["related_works"] == []
    print(f"failed downgrade: record {len(failed['failures'])} failures, stop reason {failed['stop_reason']}, no exception was thrown")

    # 4b) Hard threshold (when explicitly turned on): If no external literature can be retrieved, an error must be thrown before scoring;
    # Default is failsafe mode (require_external_evidence=False): downgrade returns + mark without interrupting scoring.
    assert ExternalRetrievalConfig().require_external_evidence is False, (
        'The default should be fail-safe: failure of the scoring task cannot be criticized when the external API is unavailable (the lack of external evidence is covered by Harness)'
    )
    gate_cfg = ExternalRetrievalConfig(enabled=True, error_budget=1, summary_top_k=0, require_external_evidence=True)
    try:
        collect_related_works(profile, "t", "", paper_type='Mix', retrieval_config=gate_cfg, runtime_config=None, fetcher=broken)
        raise AssertionError('When require_external_evidence=True, an error should be thrown when getting an empty result')
    except RuntimeError as exc:
        assert "EXTERNAL_EVIDENCE_REQUIRED" in str(exc), exc
    lenient = collect_related_works(
        profile, "t", "", paper_type='Mix',
        retrieval_config=ExternalRetrievalConfig(enabled=True, error_budget=1, summary_top_k=0, require_external_evidence=False),
        runtime_config=None, fetcher=broken,
    )
    assert lenient["related_works"] == [] and lenient["failures"]
    print('Hard threshold: throw EXTERNAL_EVIDENCE_REQUIRED before scoring when external literature cannot be obtained; allow downgrade return after explicitly turning it off')

    # 4c) claim comes from the paper portrait: Chinese claim will be translated into English search terms; when the translation is unavailable, the deterministic word will be returned
    paper_context = {
        "paper_profile": {
            "research_question": 'How algorithm management affects employee fairness perceptions',
            "research_gap": 'Existing research has not examined the fairness mechanism in choice situations.',
            "main_contribution": 'proposes an integrated framework of algorithm management and human-machine complementarity',
            "core_claim": 'Algorithmic management affects employee reactions through procedural justice',
            "paper_type": 'Evidence',
        }
    }
    context_claims = claims_from_paper_context(paper_context, "Choudhury et al. - 2020 - Machine learning and human capital", None)
    ids = [c.claim_id for c in context_claims]
    assert ids == ["RQ", "GAP", "CONTRIB", "CLAIM"], ids
    assert all(c.source == "fallback" for c in context_claims), [c.source for c in context_claims]
    assert all(c.keywords_en for c in context_claims), context_claims
    # The search terms must be in English: Chinese search terms will retrieve Chinese documents, and the rearrangement points will be zero.
    assert not any(re.search(r"[\u4e00-\u9fff]", c.keywords_en) for c in context_claims)
    assert claims_from_paper_context({"paper_profile": {"research_question": 'To be judged by Agent based on the full text'}}, "t", None) == []
    print(f"claim source: Extract {len(context_claims)} claims ({','.join(ids)}) from the paper portrait, the search terms are all in English")

    # 4d) External evidence block: EXT-* number + relevance level + source_type=external
    block_result = collect_related_works(
        profile, "Machine learning and human capital", "", paper_type='Mix',
        retrieval_config=ExternalRetrievalConfig(
            enabled=True, min_queries=2, max_queries=4, top_k=5, summary_top_k=0, require_external_evidence=False
        ),
        runtime_config=None, fetcher=fake_fetcher,
    )
    blocks = external_evidence_blocks(block_result, "PAPER-X", max_blocks=8)
    assert blocks and blocks[0].evidence_id == "EXT-1", [b.evidence_id for b in blocks]
    assert all(b.source_type == "external" for b in blocks)
    assert all(b.section.startswith('External documentation') for b in blocks)
    assert 'correlation' in blocks[0].text, blocks[0].text[:80]
    assert 'High correlation' in blocks[0].text, blocks[0].text[:120]
    print(f"External evidence block: generate {len(blocks)} pieces, the first piece is numbered {blocks[0].evidence_id}, with correlation level and EXT-* number")

    # 4e) Current limiting protection: 429 must be retried, non-retryable errors must not be idle
    import paper_eval.external_retrieval as ext_mod

    attempts = {"n": 0}
    original_raw_fetch = ext_mod._raw_fetch

    def flaky_raw_fetch(query, provider, cfg):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("HTTP Error 429: Too Many Requests")
        return [{"title": "Recovered work", "abstract": "a" * 250, "year": 2020,
                 "venue": "V", "authors": [], "citationCount": 1, "externalIds": {"DOI": "10.1/rec"}}]

    retry_cfg = ExternalRetrievalConfig(enabled=True, request_retries=3, retry_backoff_seconds=0.01)
    ext_mod._raw_fetch = flaky_raw_fetch
    try:
        recovered, error, provider = ext_mod.search_external("retry probe", retry_cfg)
    finally:
        ext_mod._raw_fetch = original_raw_fetch
    assert error is None and recovered and attempts["n"] == 3, (error, recovered, attempts)
    assert provider == resolved_providers(retry_cfg)[0], provider

    attempts["n"] = 0

    def fatal_raw_fetch(query, provider, cfg):
        attempts["n"] += 1
        raise RuntimeError("HTTP Error 400: Bad Request")

    ext_mod._raw_fetch = fatal_raw_fetch
    try:
        _, fatal_error, _ = ext_mod.search_external("fatal probe", retry_cfg)
    finally:
        ext_mod._raw_fetch = original_raw_fetch
    # 400 should not be retried: each source is hit only once, so the total number of times = number of sources (not number of sources × number of retries)
    assert fatal_error and attempts["n"] == len(resolved_providers(retry_cfg)), (fatal_error, attempts)
    print('Current limiting protection: 429 retry until the third success; 400 non-retry errors will only be made once and will not be idle')

    # 4e2) Server Retry-After: If the upper limit is not exceeded, wait as it is; if the upper limit is exceeded, the operable reason will be directly reported (no idling)
    import urllib.error
    import email.message

    def make_http_error(code: int, retry_after: str) -> urllib.error.HTTPError:
        headers = email.message.Message()
        headers["Retry-After"] = retry_after
        return urllib.error.HTTPError("https://api.openalex.org/works", code, "Too Many Requests", headers, None)

    waits: list[float] = []
    original_sleep = ext_mod.time.sleep
    ext_mod.time.sleep = lambda seconds: waits.append(float(seconds))
    attempts["n"] = 0

    def throttled_then_ok(query, provider, cfg):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise make_http_error(429, "1")
        return [{"title": "After retry-after", "abstract": "b" * 250, "year": 2021, "venue": "V",
                 "authors": [], "citationCount": 2, "externalIds": {"DOI": "10.1/ra"}}]

    ext_mod._raw_fetch = throttled_then_ok
    try:
        ra_items, ra_error, _ = ext_mod.search_external("retry-after probe", retry_cfg)
    finally:
        ext_mod._raw_fetch = original_raw_fetch
        ext_mod.time.sleep = original_sleep
    assert ra_error is None and ra_items and waits and abs(waits[0] - 1.0) < 1e-6, (ra_error, waits)
    print(f"respects Retry-After: when the server asks to wait for 1 second, it waits (actually waits for {waits[0]:.1f}s) instead of local backoff")

    attempts["n"] = 0

    def long_throttle(query, provider, cfg):
        attempts["n"] += 1
        raise make_http_error(429, "3600")

    ext_mod._raw_fetch = long_throttle
    try:
        _, long_error, _ = ext_mod.search_external("long throttle probe", retry_cfg)
    finally:
        ext_mod._raw_fetch = original_raw_fetch
    assert long_error and 'Polite Pool' in long_error and attempts["n"] == len(resolved_providers(retry_cfg)), (long_error, attempts)
    print('Ultra-long Retry-After: When it is required to wait for 3600 seconds, the actionable reason for "filling in the mailto into the polite pool" is directly reported instead of waiting for 3600 seconds.')

    # 4f) Eliminate the article itself: the search term is composed of the title of this article, and this article is almost certainly the highest-scoring hit
    from paper_eval.external_retrieval import is_self_reference

    paper_title = "Choudhury et al. - 2020 - Machine learning and human capital complementarities: Experimental evidence on bias mitigation"
    assert is_self_reference("Machine learning and human capital complementarities: Experimental evidence on bias mitigation", paper_title)
    assert is_self_reference("Machine learning and human capital complementarities", paper_title)
    assert not is_self_reference("Human-AI Ensembles: When Can They Work?", paper_title)
    assert not is_self_reference("Artificial intelligence and management: The automation–augmentation paradox", paper_title)

    self_items = [
        {"title": "Machine learning and human capital complementarities: Experimental evidence on bias mitigation",
         "abstract": "the paper itself " * 20, "year": 2020, "venue": "SMJ", "authors": [],
         "citationCount": 281, "externalIds": {"DOI": "10.1/self"}},
        {"title": "Human-AI Ensembles: When Can They Work?", "abstract": "related " * 40, "year": 2023,
         "venue": "Journal of Management", "authors": [], "citationCount": 195, "externalIds": {"DOI": "10.1/other"}},
    ]
    self_result = retrieve_related_works(
        rule_claims, paper_title,
        ExternalRetrievalConfig(enabled=True, min_queries=1, max_queries=1, top_k=5, summary_top_k=0),
        fetcher=lambda q, limit: {"data": self_items},
    )
    titles = [w["title"] for w in self_result["related_works"]]
    assert len(titles) == 1 and titles[0].startswith("Human-AI"), titles
    assert self_result["self_excluded"] == 1, self_result["self_excluded"]
    print(f"Remove this article itself: Remove {self_result['self_excluded']} (this article) from the candidates, and retain the real related work \"{titles[0][:34]}\"")

    # 5) config reads: an independent paragraph outside agent_runtime, which does not affect AgentRuntimeConfig parsing.
    # Here we only verify that "the configuration can be correctly parsed into controlled fields + unknown keys will report an error", and do not assert the specific value——
    # enabled / top_k These are the deployment knobs (the cloud machine turns external retrieval to false in order to save 20 hours),
    # Nailing the value will cause the self-test to fail as the deployment configuration drifts.
    loaded = load_external_retrieval_config(ROOT / "configs" / "agent_runtime.cloud.yaml")
    assert isinstance(loaded.enabled, bool), loaded.enabled
    assert loaded.providers, loaded.providers
    assert loaded.claim_rule_ids and loaded.evidence_rule_ids, (loaded.claim_rule_ids, loaded.evidence_rule_ids)
    assert loaded.top_k >= 1 and loaded.max_evidence_blocks >= 1 and loaded.summary_top_k >= 0
    # The default value is not coupled to the deployment configuration: the explicitly written external evidence verification switch value must be a definite Boolean,
    # When not written, the default value is True (turn off external retrieval ≠ turn off external evidence verification).
    assert isinstance(loaded.external_evidence_verification_enabled, bool)
    assert ExternalRetrievalConfig().external_evidence_verification_enabled is True, (
        'The default value of external_evidence_verification_enabled should be True: otherwise offline deployment will silently lose innovative verification capabilities'
    )
    # Independent paragraph: agent_runtime parsing is not affected by external paragraphs (an additional unknown key must report an error to avoid silent configuration failure)
    try:
        load_external_retrieval_config(ROOT / "configs" / "agent_runtime.example.yaml")
    except ValueError:
        pass
    print(f"configuration read: enabled={loaded.enabled} providers={loaded.providers} claim_rule_ids={len(loaded.claim_rule_ids)}"
          f"evidence_rule_ids={len(loaded.evidence_rule_ids)} require_external_evidence={loaded.require_external_evidence}"
          f"External evidence verification={loaded.external_evidence_verification_enabled}"
          f"min_queries={loaded.min_queries} max_queries={loaded.max_queries} top_k={loaded.top_k} "
          f"max_evidence_blocks={loaded.max_evidence_blocks} summary_top_k={loaded.summary_top_k} cache={loaded.cache_dir}")

    # 6) Source normalization: Both original formats of inverted index/JATS tags must be restored to the same field shape
    items = _normalize_openalex(OPENALEX_PAYLOAD)
    assert len(items) == 1, items
    item = items[0]
    assert item["abstract"] == "Algorithmic management shapes justice perceptions.", item["abstract"]
    assert item["title"].startswith("Algorithmic management"), item["title"]
    assert item["year"] == 2021 and item["citationCount"] == 210
    assert item["venue"] == "Academy of Management Journal", item["venue"]
    assert item["authors"] == [{"name": "C. Scholar"}], item["authors"]
    assert item["externalIds"]["DOI"] == "10.5465/amj.2020.0001", item["externalIds"]
    crossref_items = _normalize_crossref(CROSSREF_PAYLOAD)
    assert len(crossref_items) == 1, crossref_items
    cr = crossref_items[0]
    assert cr["title"] == "Algorithmic Management: Scoping Review on Employee Behaviour", cr["title"]
    assert cr["abstract"] == "We review algorithmic management and its effects.", cr["abstract"]
    assert cr["year"] == 2025 and cr["citationCount"] == 42, cr
    assert cr["venue"] == "Human Resource Management Review", cr["venue"]
    assert cr["authors"] == [{"name": "A. Author"}], cr["authors"]
    assert cr["externalIds"]["DOI"] == "10.1/crossref-demo", cr["externalIds"]
    print(
        f"source normalization: OpenAlex inverted index restoration \"{item['abstract'][:30]}\";"
        f"Crossref JATS label stripping and restoration \"{cr['abstract'][:30]}\" - the two source fields have the same shape"
    )

    # 7) Multi-source: Once configured with S2 key, it will be automatically added as a backup source without changing the configuration.
    assert resolved_providers(loaded) == ["crossref", "openalex", "semantic_scholar_bulk"], resolved_providers(loaded)
    previous = os.environ.get(loaded.api_key_env)
    os.environ[loaded.api_key_env] = "dummy-key-for-selfcheck"
    try:
        upgraded = resolved_providers(loaded)
        assert upgraded == ["crossref", "openalex", "semantic_scholar_bulk", "semantic_scholar"], upgraded
    finally:
        if previous is None:
            os.environ.pop(loaded.api_key_env, None)
        else:
            os.environ[loaded.api_key_env] = previous
    print('Multi-source redundancy: when key-free, try crossref → openalex → semantic_scholar_bulk in sequence;'
          'When S2 key exists, semantic_scholar (/paper/search relevance sorting) is automatically added as the first choice.')

    # 8) Cache: The same search query will not hit the fetcher for the second time
    cache_dir = ROOT / "runs" / "external_cache_selfcheck"
    cached_config = ExternalRetrievalConfig(enabled=True, cache_dir=str(cache_dir), min_queries=1, max_queries=1)
    counter = {"n": 0}
    # Each search query is rotated: the test only verifies the "second cache hit" and does not rely on the cache files left in the previous round.
    cache_query = f"cache test query {time.time_ns()}"

    def counting_fetcher(query: str, limit: int):
        counter["n"] += 1
        return {"data": FAKE_ITEMS}

    first = search_external(cache_query, cached_config, fetcher=counting_fetcher)
    second = search_external(cache_query, cached_config, fetcher=counting_fetcher)
    assert counter["n"] == 1, counter
    assert first[0] and second[0] and first[2] == second[2] == resolved_providers(cached_config)[0], (first[2], second[2])
    print('Cache takes effect: The same retrieval hits the disk cache for the second time, and there is no repeated request.')

    # 9) Titles contaminated by file names must be stripped clean: the title comes from the PDF file name, in the form of "Choudhury et al. - 2020 - real title".
    dirty = "Choudhury et al. - 2020 - Machine learning and human capital complementarities"
    assert clean_paper_title(dirty) == "Machine learning and human capital complementarities", clean_paper_title(dirty)
    assert clean_paper_title("AI in 2020: a review") == "AI in 2020: a review"
    assert clean_paper_title("Robots and risk of COVID-19 workplace contagion") == "Robots and risk of COVID-19 workplace contagion"

    # 10) The search query must start with the topic of the paper: the search query has an upper limit on length, and the truncated words should be general claim words, not the topic.
    topical = retrieve_related_works(
        rule_claims,
        dirty,
        ExternalRetrievalConfig(enabled=True, min_queries=1, max_queries=1, summary_top_k=0),
        fetcher=fake_fetcher,
    )
    query = topical["queries"][0]["query"]
    assert query.startswith("machine learning human capital"), query
    assert "choudhury" not in query.lower(), query
    print(f"Title cleaning and search query sequence: file name title → \"{clean_paper_title(dirty)[:44]}...\"; search query starts with subject \"{query[:52]}...\"")

    # 11) End-to-end wiring (rule-based path): collect_related_works claim → retrieval → summary
    collected = collect_related_works(
        profile,
        "Algorithmic management and employee justice",
        "We study how algorithmic management affects justice perceptions.",
        paper_type='Mix',
        retrieval_config=ExternalRetrievalConfig(
            enabled=True, min_queries=2, max_queries=4, top_k=5, summary_top_k=0, require_external_evidence=False
        ),
        runtime_config=None,
        fetcher=fake_fetcher,
    )
    assert collected["enabled"] is True
    assert collected["claim_source"] == "rule_fallback", collected["claim_source"]
    assert collected["related_works"], collected
    assert collected["status"]["provider"] == resolved_providers(ExternalRetrievalConfig())[0]
    print(f"end-to-end wiring (return rule when there is no image claim): claim {len(collected['claims'])},"
          f"search formula {len(collected['queries'])} articles, related work {len(collected['related_works'])} articles,"
          f"Stop reason {collected['stop_reason']}")

    # 12) Citation route: target paper → references. This road leads directly to Semantic Scholar,
    #     does not inject the fetcher, so a fake S2 client is used for piling. When this check is missing, the target paper is successfully located.
    #     will instead throw an AttributeError (target is treated as a single element package and unpacked into a list), and it can never be measured offline.
    import paper_eval.literature_search as ls

    real_find, real_refs = ls.s2_find_by_title, ls.s2_references
    target_paper = {"paperId": "S2-TARGET", "externalIds": {"DOI": "10.1/target"},
                    "title": "Target paper on algorithmic management"}
    ref_rows = [{"paperId": "S2-REF-1", "externalIds": {"DOI": "10.2/ref1"}, "title": "Prior work one", "year": 2015},
                {"paperId": "S2-REF-2", "externalIds": {}, "title": "Prior work two", "year": 2018}]
    calls = {"find": 0, "refs": 0}
    try:
        ls.s2_find_by_title = lambda title, cfg=None, year_from=None: (calls.__setitem__("find", calls["find"] + 1)
                                                                      or target_paper)
        ls.s2_references = lambda pid, limit=20, cfg=None: (calls.__setitem__("refs", calls["refs"] + 1)
                                                            or ref_rows)
        graph = ls.citation_graph("Target paper on algorithmic management",
                                 ExternalRetrievalConfig(enabled=True, cache_dir=None),
                                 expand_depth=False)
    finally:
        ls.s2_find_by_title, ls.s2_references = real_find, real_refs
    assert calls["find"] == 1, f"Positioning target paper should be requested only once, actual {calls['find']}"
    assert graph["target"] == target_paper, graph["target"]
    assert graph["papers"].get("references:target") == ref_rows, graph["papers"]
    assert not graph["failures"], graph["failures"]
    print(f"Citation path: Target paper 1 time (DOI {graph['target']['externalIds']['DOI']})"
          f"→ Retrieve references {len(ref_rows)}")

    # 13) Optional exploration (OpenAlex key-free)
    if "--online" in sys.argv:
        live_config = ExternalRetrievalConfig(per_query_limit=3, enabled=True, mailto=os.environ.get("OPENALEX_MAILTO", ""))
        live_items, error, provider = search_external(
            "algorithmic management employee justice", live_config
        )
        if error:
            print(f"Online detection failed (does not affect offline verification): {error[:160]}")
        else:
            print(f"Online detection successful: {provider} returns {len(live_items)} items,"
                  f"Example \"{str(live_items[0].get('title'))[:50]}\"")

    print('External retrieval self-test passed.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

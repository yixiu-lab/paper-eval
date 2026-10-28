"""外部检索自检：预算与停止条件、claim 来源与检索式、重排可解释性、缓存、失败降级、多源、外部证据块、硬门槛。

离线部分用注入的假 fetcher，不打外部 API；最后可选打一次真实探活（默认 OpenAlex，免 key）。

运行：
    python ".\scripts\自检_外部检索.py"          # 仅离线
    python ".\scripts\自检_外部检索.py" --online  # 额外探活
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from literature_eval.external_retrieval import (
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
from literature_eval.rules import load_rule_profile

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
    profile = load_rule_profile(ROOT / "评价规则库" / "rules_v1.yaml")

    # 1) 按规则生成检索式：不同规则给出不同检索式
    rule_a = profile.rules_by_id["A-QUESTION-CLARITY-V1"]
    rule_d = profile.rules_by_id["D-MODEL-IDENTIFICATION-V1"]
    query_a = build_claim_query(rule_a, "Machine learning and human capital")
    query_d = build_claim_query(rule_d, "Machine learning and human capital")
    assert query_a and query_d and query_a != query_d, (query_a, query_d)
    assert rule_a.indicator_name in query_a
    print(f"规则检索式示例：\n   {rule_a.rule_id} -> {query_a[:80]}\n   {rule_d.rule_id} -> {query_d[:80]}")

    # 2) 离线跑：预算、停止条件、去重、重排
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
    assert result["candidates"] == 2, result["candidates"]        # 两个 DOI 去重后仅 2 篇
    assert result["stop_reason"] in {"enough", "queries_exhausted", "max_queries", "below_min_queries"}
    top = result["related_works"][0]
    assert "justice" in top["title"].lower() or "algorithmic" in top["title"].lower(), top["title"]
    assert top["rerank_score"] > result["related_works"][1]["rerank_score"]
    assert "matched_claim" in top["rerank_detail"]
    print(f"离线检索：发出 {calls['count']} 条 claim 检索式，候选 {result['candidates']} 篇，"
          f"停止原因 {result['stop_reason']}，Top1「{top['title'][:36]}」得分 {top['rerank_score']:.2f}")

    # 3) 重排可解释：命中项必须出现在明细里
    score, detail = rerank(
        type("W", (), {"title": "algorithmic management justice", "abstract": "algorithmic management " * 20})(),
        {"algorithmic", "management"},
        ["algorithmic management"],
    )
    assert detail["phrase_hits"] >= 1 and detail["token_hits"] >= 2, detail
    assert score > 0
    print(f"重排可解释：得分 {score:.2f}，明细 {detail}")

    # 4) 失败不致命：fetcher 一直抛错 → 走 error_budget 停止并记录 failures
    def broken(query: str, limit: int):
        raise RuntimeError("HTTP 429 rate limited")

    failed = retrieve_related_works(rule_claims, "t", ExternalRetrievalConfig(enabled=True, error_budget=2), fetcher=broken)
    assert failed["failures"] and failed["stop_reason"] == "error_budget", failed["stop_reason"]
    assert failed["related_works"] == []
    print(f"失败降级：记录 {len(failed['failures'])} 条失败，停止原因 {failed['stop_reason']}，未抛异常")

    # 4b) 硬门槛（显式打开时）：检索不到任何外部文献必须在评分前抛错；
    # 默认是故障安全模式（require_external_evidence=False）：降级返回 + 标记，不中断评分。
    assert ExternalRetrievalConfig().require_external_evidence is False, (
        "默认应为故障安全：外部 API 不可用时不能整批评分任务失败（外部证据缺失由 Harness 兜）"
    )
    gate_cfg = ExternalRetrievalConfig(enabled=True, error_budget=1, summary_top_k=0, require_external_evidence=True)
    try:
        collect_related_works(profile, "t", "", paper_type="混合", retrieval_config=gate_cfg, runtime_config=None, fetcher=broken)
        raise AssertionError("require_external_evidence=True 时拿到空结果应当抛错")
    except RuntimeError as exc:
        assert "EXTERNAL_EVIDENCE_REQUIRED" in str(exc), exc
    lenient = collect_related_works(
        profile, "t", "", paper_type="混合",
        retrieval_config=ExternalRetrievalConfig(enabled=True, error_budget=1, summary_top_k=0, require_external_evidence=False),
        runtime_config=None, fetcher=broken,
    )
    assert lenient["related_works"] == [] and lenient["failures"]
    print("硬门槛：拿不到外部文献时在评分前抛 EXTERNAL_EVIDENCE_REQUIRED；显式关掉后允许降级返回")

    # 4c) claim 来自论文画像：中文 claim 会被翻成英文检索词；翻译不可用时退回确定性兜底词
    paper_context = {
        "paper_profile": {
            "research_question": "算法管理如何影响员工公平感知",
            "research_gap": "既有研究未检验选择情境下的公平机制",
            "main_contribution": "提出算法管理与人机互补的整合框架",
            "core_claim": "算法管理通过程序正义影响员工反应",
            "paper_type": "实证",
        }
    }
    context_claims = claims_from_paper_context(paper_context, "Choudhury et al. - 2020 - Machine learning and human capital", None)
    ids = [c.claim_id for c in context_claims]
    assert ids == ["RQ", "GAP", "CONTRIB", "CLAIM"], ids
    assert all(c.source == "fallback" for c in context_claims), [c.source for c in context_claims]
    assert all(c.keywords_en for c in context_claims), context_claims
    # 检索词必须是英文：中文进检索式会检索到中文文献、重排分趋零
    assert not any(re.search(r"[\u4e00-\u9fff]", c.keywords_en) for c in context_claims)
    assert claims_from_paper_context({"paper_profile": {"research_question": "待 Agent 结合全文判断"}}, "t", None) == []
    print(f"claim 来源：从论文画像抽出 {len(context_claims)} 条 claim（{','.join(ids)}），检索词全为英文")

    # 4d) 外部证据块：EXT-* 编号 + 相关度档位 + source_type=external
    block_result = collect_related_works(
        profile, "Machine learning and human capital", "", paper_type="混合",
        retrieval_config=ExternalRetrievalConfig(
            enabled=True, min_queries=2, max_queries=4, top_k=5, summary_top_k=0, require_external_evidence=False
        ),
        runtime_config=None, fetcher=fake_fetcher,
    )
    blocks = external_evidence_blocks(block_result, "PAPER-X", max_blocks=8)
    assert blocks and blocks[0].evidence_id == "EXT-1", [b.evidence_id for b in blocks]
    assert all(b.source_type == "external" for b in blocks)
    assert all(b.section.startswith("外部文献") for b in blocks)
    assert "相关度" in blocks[0].text, blocks[0].text[:80]
    assert "相关度 高" in blocks[0].text, blocks[0].text[:120]
    print(f"外部证据块：生成 {len(blocks)} 条，首条编号 {blocks[0].evidence_id}，带相关度档位与 EXT-* 编号")

    # 4e) 限流防护：429 必须重试，非可重试错误不得空转
    import literature_eval.external_retrieval as ext_mod

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
    # 400 不该被重试：每个源只打一次，所以总次数 = 源数量（不是 源数量 × 重试次数）
    assert fatal_error and attempts["n"] == len(resolved_providers(retry_cfg)), (fatal_error, attempts)
    print("限流防护：429 重试到第 3 次成功；400 这类不可重试错误只打一次，不空转")

    # 4e2) 服务端 Retry-After：不超过上限就照它等；超过上限直接报出可操作原因（不空转）
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
    print(f"尊重 Retry-After：服务端要求等 1 秒时按其等待（实际等待 {waits[0]:.1f}s）而非本地退避")

    attempts["n"] = 0

    def long_throttle(query, provider, cfg):
        attempts["n"] += 1
        raise make_http_error(429, "3600")

    ext_mod._raw_fetch = long_throttle
    try:
        _, long_error, _ = ext_mod.search_external("long throttle probe", retry_cfg)
    finally:
        ext_mod._raw_fetch = original_raw_fetch
    assert long_error and "礼貌池" in long_error and attempts["n"] == len(resolved_providers(retry_cfg)), (long_error, attempts)
    print("超长 Retry-After：要求等 3600 秒时不硬等，直接报出「填 mailto 进礼貌池」的可操作原因")

    # 4f) 剔除本文自身：检索式由本文标题拼成，本文几乎必然是最高分命中
    from literature_eval.external_retrieval import is_self_reference

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
    print(f"剔除本文自身：候选里剔除 {self_result['self_excluded']} 条（本文），保留真实相关工作「{titles[0][:34]}」")

    # 5) config 读取：agent_runtime 之外的独立段落，不影响 AgentRuntimeConfig 解析。
    # 这里只验"配置能被正确解析成受控字段 + 未知键会报错"，不断言具体取值——
    # enabled / top_k 这些是部署旋钮（cloud 那台机器为了省 20 小时把外部检索关成 false），
    # 钉死取值会让自检随部署配置漂移而失败。
    loaded = load_external_retrieval_config(ROOT / "configs" / "agent_runtime.cloud.yaml")
    assert isinstance(loaded.enabled, bool), loaded.enabled
    assert loaded.providers, loaded.providers
    assert loaded.claim_rule_ids and loaded.evidence_rule_ids, (loaded.claim_rule_ids, loaded.evidence_rule_ids)
    assert loaded.top_k >= 1 and loaded.max_evidence_blocks >= 1 and loaded.summary_top_k >= 0
    # 默认值不与部署配置耦合：显式写出的外部证据验证开关取值必须是确定的布尔，
    # 不写时按默认值 True（关掉外部检索 ≠ 关掉外部证据验证）。
    assert isinstance(loaded.external_evidence_verification_enabled, bool)
    assert ExternalRetrievalConfig().external_evidence_verification_enabled is True, (
        "external_evidence_verification_enabled 的默认值应为 True：否则离线部署会静默失去创新性验证能力"
    )
    # 独立段落：agent_runtime 解析不受外部段落影响（多一个未知键必须报错，避免配置静默失效）
    try:
        load_external_retrieval_config(ROOT / "configs" / "agent_runtime.example.yaml")
    except ValueError:
        pass
    print(f"配置读取：enabled={loaded.enabled} providers={loaded.providers} claim_rule_ids={len(loaded.claim_rule_ids)} 条 "
          f"evidence_rule_ids={len(loaded.evidence_rule_ids)} 条 require_external_evidence={loaded.require_external_evidence} "
          f"外部证据验证={loaded.external_evidence_verification_enabled} "
          f"min_queries={loaded.min_queries} max_queries={loaded.max_queries} top_k={loaded.top_k} "
          f"max_evidence_blocks={loaded.max_evidence_blocks} summary_top_k={loaded.summary_top_k} cache={loaded.cache_dir}")

    # 6) 源归一化：倒排索引 / JATS 标签两种原始格式都要还原成同一字段形状
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
        f"源归一化：OpenAlex 倒排索引还原「{item['abstract'][:30]}」；"
        f"Crossref JATS 标签剥离还原「{cr['abstract'][:30]}」——两源字段形状一致"
    )

    # 7) 多源：配了 S2 key 就自动追加为备用源，不改配置
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
    print("多源冗余：免 key 时 crossref → openalex → semantic_scholar_bulk 依次尝试；"
          "存在 S2 key 时自动追加 semantic_scholar（/paper/search 相关性排序）为首选兜底")

    # 8) 缓存：同一检索式第二次不再打 fetcher
    cache_dir = ROOT / "runs" / "external_cache_selfcheck"
    cached_config = ExternalRetrievalConfig(enabled=True, cache_dir=str(cache_dir), min_queries=1, max_queries=1)
    counter = {"n": 0}
    # 每轮换一个检索式：测试只验证「第二次命中缓存」，不依赖上一轮留下的缓存文件。
    cache_query = f"cache test query {time.time_ns()}"

    def counting_fetcher(query: str, limit: int):
        counter["n"] += 1
        return {"data": FAKE_ITEMS}

    first = search_external(cache_query, cached_config, fetcher=counting_fetcher)
    second = search_external(cache_query, cached_config, fetcher=counting_fetcher)
    assert counter["n"] == 1, counter
    assert first[0] and second[0] and first[2] == second[2] == resolved_providers(cached_config)[0], (first[2], second[2])
    print("缓存生效：同一检索式第二次命中磁盘缓存，未重复请求")

    # 9) 文件名污染的标题必须被剥干净：标题来自 PDF 文件名，形如「Choudhury et al. - 2020 - 真实标题」。
    dirty = "Choudhury et al. - 2020 - Machine learning and human capital complementarities"
    assert clean_paper_title(dirty) == "Machine learning and human capital complementarities", clean_paper_title(dirty)
    assert clean_paper_title("AI in 2020: a review") == "AI in 2020: a review"
    assert clean_paper_title("Robots and risk of COVID-19 workplace contagion") == "Robots and risk of COVID-19 workplace contagion"

    # 10) 检索式必须以论文主题开头：检索式有长度上限，被截断的应该是通用 claim 词，不是主题。
    topical = retrieve_related_works(
        rule_claims,
        dirty,
        ExternalRetrievalConfig(enabled=True, min_queries=1, max_queries=1, summary_top_k=0),
        fetcher=fake_fetcher,
    )
    query = topical["queries"][0]["query"]
    assert query.startswith("machine learning human capital"), query
    assert "choudhury" not in query.lower(), query
    print(f"标题清洗与检索式顺序：文件名标题 →「{clean_paper_title(dirty)[:44]}…」；检索式以主题开头「{query[:52]}…」")

    # 11) 端到端接线（规则兜底路径）：collect_related_works 抽 claim → 检索 → 摘要
    collected = collect_related_works(
        profile,
        "Algorithmic management and employee justice",
        "We study how algorithmic management affects justice perceptions.",
        paper_type="混合",
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
    print(f"端到端接线（无画像时退回规则 claim）：claim {len(collected['claims'])} 条，"
          f"检索式 {len(collected['queries'])} 条，相关工作 {len(collected['related_works'])} 篇，"
          f"停止原因 {collected['stop_reason']}")

    # 12) 引文路（citation route）：目标论文 → references。这条路直连 Semantic Scholar，
    #     不走注入 fetcher，所以用假的 S2 客户端打桩。缺了这条检查时，目标论文被成功定位
    #     反而会抛 AttributeError（target 被当成单元素包装解包成 list），离线永远测不出来。
    import literature_eval.literature_search as ls

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
    assert calls["find"] == 1, f"定位目标论文应只请求一次，实际 {calls['find']}"
    assert graph["target"] == target_paper, graph["target"]
    assert graph["papers"].get("references:target") == ref_rows, graph["papers"]
    assert not graph["failures"], graph["failures"]
    print(f"引文路：定位目标论文 1 次（DOI {graph['target']['externalIds']['DOI']}）"
          f"→ 取回 references {len(ref_rows)} 篇")

    # 13) 可选探活（OpenAlex 免 key）
    if "--online" in sys.argv:
        live_config = ExternalRetrievalConfig(per_query_limit=3, enabled=True, mailto=os.environ.get("OPENALEX_MAILTO", ""))
        live_items, error, provider = search_external(
            "algorithmic management employee justice", live_config
        )
        if error:
            print(f"在线探活失败（不影响离线校验）：{error[:160]}")
        else:
            print(f"在线探活成功：{provider} 返回 {len(live_items)} 条，"
                  f"示例「{str(live_items[0].get('title'))[:50]}」")

    print("外部检索自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

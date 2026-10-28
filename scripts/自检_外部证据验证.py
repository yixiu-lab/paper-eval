"""外部文献检索 + 外部证据验证 自检（全程离线，不打外部 API）。

覆盖第一阶段交付：
1. Claim Extractor：5 类 claim + source_evidence_id 映射（claim 必须回指真实证据块）
2. 多路召回：每个 claim 3-5 条检索式、引文路、语义路；合并去重后形成候选池
3. 时间泄漏控制：晚于目标论文的候选被剔除
4. Reranker：八项信号打分 + Top-K External Evidence Set + 保留召回原因与对应 claim
5. Crossref 元数据校验：DOI/标题/年份不一致必须标 mismatch
6. Novelty Agent：只输出有限枚举，引用的证据 ID 必须真实存在
7. Harness 外部证据校验：时间泄漏、引用不存在、prior coverage 不足 → 降置信度/needs_review
8. 仓库分离 + 反向追溯：Score → Judgment → Claim → External Evidence → External Paper

运行：
    python ".\scripts\自检_外部证据验证.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_eval import novelty_agent  # noqa: E402
from paper_eval.agent_graph import validate_external_evidence_node, verify_novelty  # noqa: E402
from paper_eval.claim_extraction import extract_claims  # noqa: E402
from paper_eval.external_evidence import (  # noqa: E402
    apply_coverage_confidence_cap,
    apply_external_validation_to_judgements,
    build_repositories,
    compare_metadata,
    repository_to_dict,
    validate_external_evidence,
    verify_crossref,
)
from paper_eval.external_retrieval import (  # noqa: E402
    ExternalRetrievalConfig,
    collect_external_literature,
)
from paper_eval.literature_search import (  # noqa: E402
    attach_external_evidence_ids,
    expand_query_routes,
    run_literature_search,
)
from paper_eval.models import AgentJudgement, Evidence, Judgement, Paper  # noqa: E402
from paper_eval.rules import load_rule_profile  # noqa: E402

PAPER_TITLE = "Choudhury et al. - 2020 - Machine learning and human capital complementarities"
PAPER_YEAR = 2020

EVIDENCE_TEXTS = [
    (
        "EV-0001",
        "Abstract",
        1,
        "We study how algorithmic management shapes employee justice perceptions in selection contexts. "
        "Little is known about whether fairness perceptions transmit the effect of algorithmic decisions on employee reactions.",
    ),
    (
        "EV-0002",
        "Introduction",
        2,
        "Existing research has not examined the mediating mechanism of procedural justice in algorithmic management settings. "
        "We are the first to test this mechanism with experimental evidence.",
    ),
    (
        "EV-0003",
        "Theory",
        3,
        "Our theoretical contribution is to extend organizational justice theory to algorithmic management contexts. "
        "We develop a mechanism in which perceived procedural justice mediates the effect of algorithmic management on employee reactions.",
    ),
    (
        "EV-0004",
        "Discussion",
        4,
        "We contribute to theory by integrating organizational justice theory with algorithmic management research. "
        "The boundary condition is that the mechanism holds only when employees can observe the decision process.",
    ),
]

PAPER_PROFILE = {
    "title": "Machine learning and human capital complementarities",
    "paper_type": "实证",
    "central_problem": "算法管理如何影响员工公平感知与反应",
    "research_gap": "既有研究未检验选择情境下程序正义的中介机制",
    "research_question": "算法管理通过什么机制影响员工反应，边界条件是什么",
    "core_claim": "算法管理通过程序正义影响员工反应",
    "main_contribution": "把组织公平理论推进到算法管理情境，提出新的中介机制",
    "boundary_conditions": "only when employees can observe the decision process",
    "theoretical_move": {"type": "new_mechanism", "description": "procedural justice mediation", "strength": "moderate"},
}

# 三路召回用的假外部文献：一篇先验高度相关、一篇先验外围相关、一篇晚于目标论文（用来验时间过滤）
PRIOR_STRONG = {
    "title": "Algorithmic management and procedural justice perceptions",
    "abstract": "We examine procedural justice perceptions of algorithmic management and employee reactions in selection contexts. " * 3,
    "year": 2018,
    "venue": "Academy of Management Journal",
    "authors": [{"name": "P. Prior"}],
    "citationCount": 180,
    "externalIds": {"DOI": "10.5465/amj.2018.0001"},
}
PRIOR_PERIPHERAL = {
    "title": "Big data and human resource analytics: fairness concerns in review",
    "abstract": "A review of fairness concerns in human resource analytics and big data management.",
    "year": 2016,
    "venue": "Human Resource Management Review",
    "authors": [{"name": "R. Reviewer"}],
    "citationCount": 60,
    "externalIds": {"DOI": "10.1016/j.hrmr.2016.01.001"},
}
LEAKED_LATER = {
    "title": "Algorithmic management and justice: a replication after 2020",
    "abstract": "We replicate and extend algorithmic management justice findings in selection contexts. " * 3,
    "year": 2023,
    "venue": "Journal of Management",
    "authors": [{"name": "L. Later"}],
    "citationCount": 12,
    "externalIds": {"DOI": "10.1177/01492063231111111"},
}
CITATION_RELATED = {
    "title": "Justice perceptions in automated decision making",
    "abstract": "Automated decision making and justice perceptions in organizations. " * 2,
    "year": 2017,
    "venue": "Organization Science",
    "authors": [{"name": "C. Cited"}],
    "citationCount": 90,
    "externalIds": {"DOI": "10.1287/orsc.2017.0002"},
}


class StubRuntimeConfig:
    """离线自检用的运行时配置占位：只声明 Novelty Agent 需要的开关。"""

    enable_llm_agents = True
    model = "stub"


def _evidences() -> list[Evidence]:
    return [
        Evidence(
            evidence_id=eid,
            paper_id="PAPER-SELFTEST",
            section=section,
            page=page,
            span=(0, len(text)),
            text=text,
            text_hash=eid,
        )
        for eid, section, page, text in EVIDENCE_TEXTS
    ]


def _paper() -> Paper:
    return Paper(
        paper_id="PAPER-SELFTEST",
        metadata={
            "title": PAPER_TITLE,
            "abstract": "Algorithmic management and employee justice perceptions.",
            "year": "2020",
        },
        sections=[],
        references=[],
        publish_date="2020-06-01",
    )


def _fake_fetcher(query: str, limit: int):
    """Query 路假检索：任何检索式都返回两篇先验 + 一篇晚于目标论文的文献。"""
    return {"data": [PRIOR_STRONG, PRIOR_PERIPHERAL, LEAKED_LATER]}


def step_1_claims() -> list:
    claims = extract_claims({"paper_profile": PAPER_PROFILE}, _evidences(), PAPER_YEAR)
    types = {claim.claim_type for claim in claims}
    assert {
        "research_gap",
        "novelty_claim",
        "theoretical_contribution",
        "mechanism_claim",
        "boundary_claim",
    } <= types, types
    known = {item.evidence_id for item in _evidences()}
    for claim in claims:
        assert claim.source_evidence_id in known, (claim.claim_id, claim.source_evidence_id)
        assert claim.claim_text and claim.publication_year == PAPER_YEAR
    print(
        f"1) Claim Extractor：抽出 {len(claims)} 条 claim，类型 {sorted(types)}；"
        f"每条都带真实 source_evidence_id（例：{claims[1].claim_id}→{claims[1].source_evidence_id}）"
    )
    return claims


def step_2_to_4(claims) -> tuple[dict, list]:
    retrieval_claims = [claim.to_retrieval_claim(PAPER_TITLE) for claim in claims]
    for claim in retrieval_claims:
        routes = expand_query_routes(claim, PAPER_TITLE, 5)
        assert 3 <= len(routes) <= 5, (claim.claim_id, len(routes))
    facets = {row["route"] for row in expand_query_routes(retrieval_claims[0], PAPER_TITLE, 5)}
    assert len(facets) >= 3, facets

    # 引文路：离线注入引文图（真实实现走 Semantic Scholar，这里只验合并、provenance 与去重）
    import paper_eval.literature_search as ls

    real_graph = ls.citation_graph

    def fake_graph(paper_title, config, *, allow_later_works=False, expand_depth=True, progress_callback=None):
        return {
            "papers": {"references:target": [CITATION_RELATED]},
            "target": {"title": paper_title, "externalIds": {"DOI": "10.5465/amj.2020.0009"}},
            "failures": [],
            "requests": 1,
        }

    ls.citation_graph = fake_graph
    try:
        result = run_literature_search(
            retrieval_claims,
            PAPER_TITLE,
            ExternalRetrievalConfig(
                enabled=True,
                max_queries=5,
                top_k=5,
                max_evidence_blocks=5,
                summary_top_k=0,
                candidate_pool_limit=40,
                time_leak_guard=True,
            ),
            paper_abstract="Algorithmic management and employee justice perceptions.",
            research_question="算法管理通过什么机制影响员工反应",
            target_year=PAPER_YEAR,
            fetcher=_fake_fetcher,
        )
    finally:
        ls.citation_graph = real_graph

    for route in ("query", "citation", "semantic"):
        assert route in result["routes_used"], result["routes_used"]
    assert result["time_filtered"] >= 1, result["time_filtered"]
    attach_external_evidence_ids(result)

    titles = [work["title"] for work in result["external_evidence_set"]]
    assert LEAKED_LATER["title"] not in titles, titles
    assert PRIOR_STRONG["title"] in titles, titles
    for work in result["external_evidence_set"]:
        assert work["routes"] and work["claim_ids"], work
        assert work["retrieval_time"] and work["retrieval_query"], work
        assert work["evidence_id"].startswith("XEV-"), work["evidence_id"]
    top = result["external_evidence_set"][0]
    assert top["title"] == PRIOR_STRONG["title"], titles
    assert top["relevance_score"] >= result["external_evidence_set"][-1]["relevance_score"]
    required_signals = {
        "topic_relevance",
        "construct_overlap",
        "theory_relevance",
        "mechanism_relevance",
        "context_relevance",
        "semantic_similarity",
        "citation_relationship",
        "publication_year",
    }
    assert required_signals <= set(top["rank_features"]), top["rank_features"]
    print(
        f"2-4) 多路召回：{len(retrieval_claims)} 条 claim × 3-5 条检索式，路径 {result['routes_used']}；"
        f"候选池 {result['pool_size']} 篇，时间过滤剔除 {result['time_filtered']} 篇，"
        f"合并去重后 External Evidence Set {len(titles)} 篇；"
        f"Top1「{top['title'][:40]}」分 {top['relevance_score']}"
    )
    return result, retrieval_claims


def step_5_crossref() -> None:
    status, notes = compare_metadata(
        {"title": PRIOR_STRONG["title"], "year": 2018},
        {"title": PRIOR_STRONG["title"], "year": 2019, "doi": "10.5465/amj.2018.0001"},
    )
    assert status == "mismatch" and any("年份不一致" in note for note in notes), (status, notes)
    ok_status, _ = compare_metadata(
        {"title": PRIOR_STRONG["title"], "year": 2018},
        {"title": PRIOR_STRONG["title"], "year": 2018, "doi": "10.5465/amj.2018.0001"},
    )
    assert ok_status == "ok", ok_status

    works = [
        {"title": PRIOR_STRONG["title"], "year": 2018, "doi": "10.5465/amj.2018.0001", "external_ids": {}},
        {"title": PRIOR_PERIPHERAL["title"], "year": 2016, "doi": "10.1016/j.hrmr.2016.01.001", "external_ids": {}},
    ]
    verify_crossref(
        works,
        ExternalRetrievalConfig(),
        cache_dir=None,
        fetcher=lambda doi: {
            "message": {
                "DOI": doi,
                "title": [PRIOR_STRONG["title"]] if doi.endswith("0001") else [PRIOR_PERIPHERAL["title"]],
                "issued": {"date-parts": [[2019 if doi.endswith("0001") else 2016]]},
            }
        },
        max_checks=2,
    )
    assert works[0]["metadata_check"] == "mismatch", works[0]
    assert works[1]["metadata_check"] == "ok", works[1]
    assert works[0]["metadata_notes"], works[0]
    print("5) Crossref 校验：标题/年份不一致标 mismatch，一致标 ok；结论与备注写回文献记录")


def step_6_novelty(result: dict, retrieval_claims: list) -> list[dict]:
    external_set = result["external_evidence_set"]
    captured: dict = {}

    def fake_call(system_prompt, payload, config):
        captured["payload"] = payload
        assert "只允许" in system_prompt
        return {
            "judgements": [
                {
                    "claim_id": retrieval_claims[0].claim_id,
                    "judgment": "already_established",
                    "reason": "外部文献已在同一机制上做过同问题，本文为增量。",
                    "supporting_papers": [PRIOR_STRONG["title"]],
                    "contradicting_papers": [],
                    "external_evidence_ids": [external_set[0]["evidence_id"], "XEV-999"],
                    "confidence": 0.7,
                    "needs_review": False,
                },
                {
                    "claim_id": retrieval_claims[1].claim_id,
                    "judgment": "novel",
                    "reason": "首次提出该机制，未见已有研究。",
                    "supporting_papers": ["不存在的文献标题"],
                    "contradicting_papers": [],
                    "external_evidence_ids": [],
                    "confidence": 0.9,
                    "needs_review": False,
                },
            ]
        }

    real_call = novelty_agent.call_plain_json
    novelty_agent.call_plain_json = fake_call
    try:
        verification = novelty_agent.run_novelty_verification(
            retrieval_claims,
            _evidences(),
            external_set,
            _paper(),
            PAPER_YEAR,
            StubRuntimeConfig(),
            ExternalRetrievalConfig(enabled=True),
        )
    finally:
        novelty_agent.call_plain_json = real_call

    assert captured["payload"]["claims"], captured
    verdicts = verification["verdicts"]
    assert len(verdicts) == len(retrieval_claims), len(verdicts)
    assert all(row["judgment"] in set(novelty_agent.NOVELTY_JUDGEMENTS) for row in verdicts), verdicts
    first = verdicts[0]
    assert first["judgment"] == "already_established", first
    assert "XEV-999" not in first["external_evidence_ids"], first
    assert first["needs_review"] is True, first
    assert first["comparison_level"] in novelty_agent.NOVELTY_COMPARISON_LEVELS, first
    assert first["comparison_features"]["top_k"] == 10, first
    assert first["top_prior_works"], first
    assert "downgrade_constraints" in first, first
    second = verdicts[1]
    assert second["judgment"] == "novel", second
    assert second["needs_review"] is True and "核对提示" in second["reason"], second
    assert second["comparison_features"]["available_prior_works"] >= 1, second
    print(
        f"6) Novelty Agent：{len(verdicts)} 条 claim 全部落在受限枚举内（{sorted({row['judgment'] for row in verdicts})}）；"
        f"不存在的外部证据 ID 被剔除；点名证据集外文献的 novel 判断被标待复核；"
        f"每条 verdict 均包含五维比较特征与 Top-{first['comparison_features']['top_k']} prior works"
    )
    return verdicts


def step_7_harness(result: dict, retrieval_claims: list, verdicts: list[dict]) -> dict:
    claim_rows = [claim.to_dict() for claim in retrieval_claims]
    validation = validate_external_evidence(
        claims=claim_rows,
        external_set=result["external_evidence_set"],
        novelty_judgements=verdicts,
        target_year=PAPER_YEAR,
        indicator_ids=["B-CONCEPT-MECHANISM-NOVELTY-V1"],
        min_external_papers=2,
        low_coverage_confidence_cap=0.4,
    )
    codes = {item["code"] for item in validation["findings"]}
    assert "external_stance_paper_not_in_evidence_set" in codes, codes
    assert "external_stance_inconsistent" in codes, codes

    # 引用不存在的外部证据 ID：Novelty Agent 侧已剔除，但 Harness 必须能独立抓住（防绕过）
    forged = [
        {
            "claim_id": "CLM-01",
            "judgment": "novel",
            "reason": "首次提出",
            "confidence": 0.9,
            "needs_review": False,
            "external_evidence_ids": ["XEV-999"],
            "supporting_papers": [],
            "contradicting_papers": [],
        }
    ]
    forged_validation = validate_external_evidence(
        claims=claim_rows,
        external_set=result["external_evidence_set"],
        novelty_judgements=forged,
        target_year=PAPER_YEAR,
    )
    forged_codes = {item["code"] for item in forged_validation["findings"]}
    assert "external_evidence_id_not_found" in forged_codes, forged_codes

    # prior literature 覆盖不足 → 下调 confidence 并标 needs_review（判断本身不被删除）
    weak = [
        {
            "claim_id": "CLM-99",
            "judgment": "novel",
            "reason": "首次提出",
            "confidence": 0.9,
            "needs_review": False,
            "external_evidence_ids": [],
            "supporting_papers": [],
            "contradicting_papers": [],
        }
    ]
    low_cov = validate_external_evidence(
        claims=[{"claim_id": "CLM-99", "source_evidence_id": "EV-0002"}],
        external_set=result["external_evidence_set"],
        novelty_judgements=weak,
        target_year=PAPER_YEAR,
        min_external_papers=2,
        low_coverage_confidence_cap=0.4,
    )
    low_codes = {item["code"] for item in low_cov["findings"]}
    assert "low_prior_literature_coverage" in low_codes, low_codes
    capped = apply_coverage_confidence_cap(weak, low_cov, 0.4)
    assert capped[0]["confidence"] == 0.4 and capped[0]["needs_review"] is True, capped[0]

    # 时间泄漏必须被抓到：人工塞一篇晚于目标论文的文献
    leaked = dict(LEAKED_LATER)
    leaked.update({"evidence_id": "XEV-LEAK", "paper_id": "leak", "relevance_score": 1.0, "claim_ids": ["CLM-01"]})
    leak_validation = validate_external_evidence(
        claims=claim_rows,
        external_set=[leaked],
        novelty_judgements=[],
        target_year=PAPER_YEAR,
    )
    assert any(item["code"] == "external_time_leak" for item in leak_validation["findings"]), leak_validation

    # 元数据不一致同样要被抓到
    mismatch = dict(result["external_evidence_set"][0])
    mismatch["metadata_check"] = "mismatch"
    mismatch["metadata_notes"] = ["年份不一致：检索结果 2018 vs Crossref 2019"]
    mismatch_validation = validate_external_evidence(
        claims=claim_rows,
        external_set=[mismatch],
        novelty_judgements=[],
        target_year=PAPER_YEAR,
    )
    assert any(item["code"] == "external_metadata_mismatch" for item in mismatch_validation["findings"])

    # 没有任何外部证据时：依赖领域比较的规则不得给 satisfied（转 uncertain，模型理由不被改写）
    empty_validation = validate_external_evidence(
        claims=claim_rows,
        external_set=[],
        novelty_judgements=[],
        target_year=PAPER_YEAR,
        indicator_ids=["B-CONCEPT-MECHANISM-NOVELTY-V1"],
    )
    judgement = AgentJudgement(
        task_id="RUN-T",
        agent_id="theory",
        indicator_id="B-CONCEPT-MECHANISM-NOVELTY-V1",
        judgement=Judgement.SATISFIED,
        evidence_ids=["EV-0003"],
        reason="新机制证据充分",
        confidence=0.8,
        needs_review=False,
        agent_version="t",
        model_version="stub",
    )
    gated = apply_external_validation_to_judgements([judgement], empty_validation)
    assert gated[0].judgement == Judgement.UNCERTAIN, gated[0]
    assert gated[0].needs_review and gated[0].review_flags, gated[0]
    assert gated[0].reason == "新机制证据充分", "门控不得改写模型理由"
    print(
        "7) Harness 外部校验：抓到时间泄漏 / 元数据不一致 / 引用不存在证据 / prior 覆盖不足（置信度封顶 0.4）；"
        "没有外部证据时依赖领域比较的 satisfied 被转为 uncertain，模型 reason 未被改写"
    )
    return validation


def step_8_repository(result: dict, retrieval_claims: list) -> None:
    repository = build_repositories(
        "PAPER-SELFTEST",
        _evidences(),
        [claim.to_dict() for claim in retrieval_claims],
        result["external_evidence_set"],
    )
    payload = repository_to_dict(repository)
    assert payload["counts"]["internal"] == len(EVIDENCE_TEXTS), payload["counts"]
    assert payload["counts"]["external"] == len(result["external_evidence_set"]), payload["counts"]
    internal_ids = {row["evidence_id"] for row in payload["internal_repository"]}
    external_ids = {row["evidence_id"] for row in payload["external_repository"]}
    assert not (internal_ids & external_ids), internal_ids & external_ids
    rows = payload["traceability"]["claims"]
    assert rows and all(row["internal_evidence_found"] for row in rows), rows
    traced = [row for row in rows if row["external_evidence_ids"]]
    assert traced, rows
    for row in traced:
        assert row["external_paper_ids"] and all(row["external_paper_ids"]), row
    print(
        f"8) 仓库分离与反向追溯：内部 {payload['counts']['internal']} 条 / 外部 {payload['counts']['external']} 条互不混装；"
        f"{len(traced)} 条 claim 走通 Claim → External Evidence → External Paper（含 DOI）"
    )


def step_9_graph_nodes(result: dict, retrieval_claims: list) -> None:
    """直接跑图节点：确认 verify_novelty → validate_external_evidence 接线可用（离线 stub 模型）。"""
    profile = load_rule_profile(ROOT / "评价规则库" / "rules_v1.yaml")
    config_path = ROOT / "configs" / "agent_runtime.deepseek_retrieval.yaml"
    state = {
        "run_id": "RUN-SELFTEST",
        "profile": profile,
        "paper": _paper(),
        "evidences": _evidences(),
        "agent_config_path": config_path,
        "progress_callback": None,
        "trace": [],
        # 图状态里的 external_retrieval 是 collect_external_literature 的产物（带 claims）
        "external_retrieval": {**result, "claims": [claim.to_dict() for claim in retrieval_claims]},
    }
    real_call = novelty_agent.call_plain_json
    novelty_agent.call_plain_json = lambda system_prompt, payload, config: {"judgements": []}
    try:
        state.update(verify_novelty(state))
        state.update(validate_external_evidence_node({**state, "judgements": []}))
    finally:
        novelty_agent.call_plain_json = real_call
    assert state["novelty_verification"]["reason"] == "direct_external_evidence_to_six_reviewers", state["novelty_verification"]
    assert not state["novelty_verification"]["verdicts"], state["novelty_verification"]
    assert state["evidence_repository"]["counts"]["internal"] == len(EVIDENCE_TEXTS)
    assert state["external_validation"]["checked"] is True
    steps = [item["step"] for item in state["trace"]]
    assert "langgraph.verify_novelty" in steps and "langgraph.validate_external_evidence" in steps, steps
    print(
        f"9) 图节点接线：外部先行文献直接交给六个 Reviewer，"
        f"B Reviewer 负责创新性比较；未生成独立 NOV-* 结论块；"
        f"validate_external_evidence 写出 {state['external_validation']['finding_count']} 条校验项"
    )


def step_10_collect_end_to_end() -> None:
    """collect_external_literature 端到端（离线 fetcher），确认旧字段仍然齐全。"""
    import paper_eval.external_evidence as ee
    import paper_eval.literature_search as ls

    profile = load_rule_profile(ROOT / "评价规则库" / "rules_v1.yaml")
    real_graph = ls.citation_graph
    real_crossref = ee._crossref_payload
    ls.citation_graph = lambda *args, **kwargs: {"papers": {}, "target": None, "failures": [], "requests": 0}
    ee._crossref_payload = lambda doi, timeout, mailto="": {
        "message": {"DOI": doi, "title": [PRIOR_STRONG["title"]], "issued": {"date-parts": [[2018]]}}
    }
    try:
        collected = collect_external_literature(
            profile,
            PAPER_TITLE,
            "Algorithmic management and employee justice perceptions.",
            paper_id="PAPER-SELFTEST",
            evidences=_evidences(),
            paper_type="混合",
            publication_year=PAPER_YEAR,
            retrieval_config=ExternalRetrievalConfig(
                enabled=True, max_queries=4, top_k=5, max_evidence_blocks=5, summary_top_k=0
            ),
            runtime_config=None,
            paper_context={"paper_profile": PAPER_PROFILE},
            fetcher=_fake_fetcher,
        )
    finally:
        ls.citation_graph = real_graph
        ee._crossref_payload = real_crossref

    assert collected["claim_source"] == "claim_extractor", collected["claim_source"]
    assert collected["claims"] and collected["related_works"], sorted(collected.keys())
    assert collected["coverage"]["paper_count"] >= 1, collected["coverage"]
    assert collected["external_evidence_unavailable"] is False
    assert all(work["evidence_id"] for work in collected["external_evidence_set"])
    assert collected["status"]["providers"], collected["status"]
    print(
        f"10) 端到端：claim 来源 {collected['claim_source']}，claim {len(collected['claims'])} 条，"
        f"外部证据 {collected['coverage']['paper_count']} 篇，停止原因 {collected['stop_reason']}"
    )


def main() -> int:
    claims = step_1_claims()
    result, retrieval_claims = step_2_to_4(claims)
    step_5_crossref()
    verdicts = step_6_novelty(result, retrieval_claims)
    step_7_harness(result, retrieval_claims, verdicts)
    step_8_repository(result, retrieval_claims)
    step_9_graph_nodes(result, retrieval_claims)
    step_10_collect_end_to_end()
    print("外部证据验证自检通过（全程离线）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

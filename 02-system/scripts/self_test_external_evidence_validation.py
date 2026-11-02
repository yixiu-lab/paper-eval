'External literature retrieval + external evidence verification self-inspection (the whole process is offline, no external API is used).\n\nCovering the first phase of delivery:\n1. Claim Extractor: 5 types of claim + source_evidence_id mapping (claim must refer back to the real evidence block)\n2. Multi-channel recall: 3-5 search terms, citation paths, and semantic paths for each claim; merge and remove duplicates to form a candidate pool\n3. Time leakage control: Candidates that are later than the target paper are eliminated\n4. Reranker: eight signal scoring + Top-K External Evidence Set + retain recall reasons and corresponding claims\n5. Crossref metadata verification: DOI/title/year inconsistencies must be marked as mismatch\n6. Novelty Agent: Only a limited enumeration is output, and the referenced evidence ID must actually exist\n7. Harness external evidence verification: time leakage, non-existent references, insufficient prior coverage → lower confidence/needs_review\n8. Warehouse separation + reverse traceability: Score → Judgment → Claim → External Evidence → External Paper\n\nRun:\n    python ".\\scripts\\self_test_external_evidence_validation.py"'

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
    "paper_type": 'Evidence',
    "central_problem": 'How algorithm management affects employee fairness perceptions and reactions',
    "research_gap": 'Existing research has not examined the mediating mechanism of procedural justice in choice situations.',
    "research_question": 'What mechanism does algorithmic management use to influence employee response, and what are the boundary conditions?',
    "core_claim": 'Algorithmic management affects employee reactions through procedural justice',
    "main_contribution": 'advances organizational justice theory to algorithmic management situations and proposes a new intermediary mechanism',
    "boundary_conditions": "only when employees can observe the decision process",
    "theoretical_move": {"type": "new_mechanism", "description": "procedural justice mediation", "strength": "moderate"},
}

# Fake external documents used for three-way recall: one a priori highly relevant, one a priori peripherally relevant, and one later than the target paper (used to test time filtering)
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
    'Runtime configuration placeholder for offline self-test: only declare the switches required by Novelty Agent.'

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
    'Query false search: Any search query returns two prior papers + one document that is later than the target paper.'
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
        f"1) Claim Extractor: Extract {len(claims)} claims, type {sorted(types)};"
        f"Each item has a real source_evidence_id (example: {claims[1].claim_id}→{claims[1].source_evidence_id})"
    )
    return claims


def step_2_to_4(claims) -> tuple[dict, list]:
    retrieval_claims = [claim.to_retrieval_claim(PAPER_TITLE) for claim in claims]
    for claim in retrieval_claims:
        routes = expand_query_routes(claim, PAPER_TITLE, 5)
        assert 3 <= len(routes) <= 5, (claim.claim_id, len(routes))
    facets = {row["route"] for row in expand_query_routes(retrieval_claims[0], PAPER_TITLE, 5)}
    assert len(facets) >= 3, facets

    # Citation path: offline injection of citation graph (the real implementation is through Semantic Scholar, here only merging, provenance and deduplication are tested)
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
            research_question='Through what mechanism does algorithmic management affect employee reactions?',
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
        f"2-4) Multi-channel recall: {len(retrieval_claims)} claims × 3-5 search formulas, path {result['routes_used']};"
        f"candidate pool {result['pool_size']} articles, time filtering eliminates {result['time_filtered']} articles,"
        f"External Evidence Set {len(titles)} after merging and deduplication;"
        f"Top1 \"{top['title'][:40]}\" points {top['relevance_score']}"
    )
    return result, retrieval_claims


def step_5_crossref() -> None:
    status, notes = compare_metadata(
        {"title": PRIOR_STRONG["title"], "year": 2018},
        {"title": PRIOR_STRONG["title"], "year": 2019, "doi": "10.5465/amj.2018.0001"},
    )
    assert status == "mismatch" and any('The year is inconsistent' in note for note in notes), (status, notes)
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
    print('5) Crossref verification: mismatch for title/year inconsistencies, ok for consistency; conclusions and remarks are written back to the literature record')


def step_6_novelty(result: dict, retrieval_claims: list) -> list[dict]:
    external_set = result["external_evidence_set"]
    captured: dict = {}

    def fake_call(system_prompt, payload, config):
        captured["payload"] = payload
        assert 'only allowed' in system_prompt
        return {
            "judgements": [
                {
                    "claim_id": retrieval_claims[0].claim_id,
                    "judgment": "already_established",
                    "reason": 'External literature has already done the same problem on the same mechanism, and this article is an increment.',
                    "supporting_papers": [PRIOR_STRONG["title"]],
                    "contradicting_papers": [],
                    "external_evidence_ids": [external_set[0]["evidence_id"], "XEV-999"],
                    "confidence": 0.7,
                    "needs_review": False,
                },
                {
                    "claim_id": retrieval_claims[1].claim_id,
                    "judgment": "novel",
                    "reason": 'proposed this mechanism for the first time, but there is no previous research on it.',
                    "supporting_papers": ['Document title does not exist'],
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
    assert second["needs_review"] is True and 'Verification Tips' in second["reason"], second
    assert second["comparison_features"]["available_prior_works"] >= 1, second
    print(
        f"6) Novelty Agent: {len(verdicts)} claims all fall within the restricted enumeration ({sorted({row['judgment'] for row in verdicts})});"
        f"Non-existent external evidence IDs are removed; novel judgments naming documents outside the evidence set are marked for review;"
        f"Each verdict contains five-dimensional comparative features and Top-{first['comparison_features']['top_k']} prior works"
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

    # refers to non-existent external evidence ID: The Novelty Agent side has been eliminated, but the Harness must be able to grasp it independently (anti-bypass)
    forged = [
        {
            "claim_id": "CLM-01",
            "judgment": "novel",
            "reason": 'first proposed',
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

    # prior literature coverage is insufficient → lower confidence and mark needs_review (the judgment itself will not be deleted)
    weak = [
        {
            "claim_id": "CLM-99",
            "judgment": "novel",
            "reason": 'first proposed',
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

    # Time leak must be caught: artificially stuffing a document later than the target paper
    leaked = dict(LEAKED_LATER)
    leaked.update({"evidence_id": "XEV-LEAK", "paper_id": "leak", "relevance_score": 1.0, "claim_ids": ["CLM-01"]})
    leak_validation = validate_external_evidence(
        claims=claim_rows,
        external_set=[leaked],
        novelty_judgements=[],
        target_year=PAPER_YEAR,
    )
    assert any(item["code"] == "external_time_leak" for item in leak_validation["findings"]), leak_validation

    # Inconsistent metadata will also be caught
    mismatch = dict(result["external_evidence_set"][0])
    mismatch["metadata_check"] = "mismatch"
    mismatch["metadata_notes"] = ['Year inconsistency: Search results 2018 vs Crossref 2019']
    mismatch_validation = validate_external_evidence(
        claims=claim_rows,
        external_set=[mismatch],
        novelty_judgements=[],
        target_year=PAPER_YEAR,
    )
    assert any(item["code"] == "external_metadata_mismatch" for item in mismatch_validation["findings"])

    # When there is no external evidence: rules that rely on domain comparison must not be satisfied (turn to uncertain, the model reason will not be rewritten)
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
        reason='There is sufficient evidence for the new mechanism',
        confidence=0.8,
        needs_review=False,
        agent_version="t",
        model_version="stub",
    )
    gated = apply_external_validation_to_judgements([judgement], empty_validation)
    assert gated[0].judgement == Judgement.UNCERTAIN, gated[0]
    assert gated[0].needs_review and gated[0].review_flags, gated[0]
    assert gated[0].reason == 'There is sufficient evidence for the new mechanism', 'Gate control must not rewrite the model reason'
    print(
        '7) Harness external verification: catches time leaks/metadata inconsistencies/references to non-existent evidence/insufficient prior coverage (confidence capped at 0.4);'
        'When there is no external evidence, satisfied that relies on domain comparison is converted to uncertain, and model reason is not rewritten.'
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
        f"8) Warehouse separation and reverse traceability: internal {payload['counts']['internal']} strips / external {payload['counts']['external']} strips do not mix with each other;"
        f"{len(traced)} claim claim → External Evidence → External Paper (including DOI)"
    )


def step_9_graph_nodes(result: dict, retrieval_claims: list) -> None:
    'Directly run the graph node: Confirm that the verify_novelty → validate_external_evidence wiring is available (offline stub model).'
    profile = load_rule_profile(ROOT / 'rules' / "rules_v1.yaml")
    config_path = ROOT / "configs" / "agent_runtime.deepseek_retrieval.yaml"
    state = {
        "run_id": "RUN-SELFTEST",
        "profile": profile,
        "paper": _paper(),
        "evidences": _evidences(),
        "agent_config_path": config_path,
        "progress_callback": None,
        "trace": [],
        # external_retrieval in the graph state is the product of collect_external_literature (with claims)
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
        f"9) Graph node wiring: External prior documents are directly handed over to six Reviewers."
        f"B Reviewer is responsible for innovative comparison; no independent NOV-* conclusion block is generated;"
        f"validate_external_evidence Write out {state['external_validation']['finding_count']} verification items"
    )


def step_10_collect_end_to_end() -> None:
    'collect_external_literature end-to-end (offline fetcher), confirm that the old fields are still complete.'
    import paper_eval.external_evidence as ee
    import paper_eval.literature_search as ls

    profile = load_rule_profile(ROOT / 'rules' / "rules_v1.yaml")
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
            paper_type='Mix',
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
        f"10) End-to-end: claim source {collected['claim_source']}, claim {len(collected['claims'])} strip,"
        f"External evidence {collected['coverage']['paper_count']}, reason for stopping {collected['stop_reason']}"
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
    print('External evidence verification self-test passed (the entire process was offline).')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

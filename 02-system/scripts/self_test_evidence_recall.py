'Evidence retrieval layer self-test: bag-of-words recall sorting, LLM rearrangement order, three degradation paths, switch off does not change behavior.\n\nWhy downgrades must be overridden: Reflow is a model call that will fail (timeout, bad JSON, model bullshit id).\nScoring is a chain. Failure to rearrange any rule will not cause the score of the entire paper to collapse, so "failure returns to the recall order"\nIt is a contract rather than a best effort, and there must be operational evidence.\n\nThe real API is not adjusted in the whole process: the rearranged call point `_call_rerank_llm` is monkeypatched into a false response.\n\nRun:\n    python ".\\scripts\\self_test_evidence_recall.py"'

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval import evidence_retrieval
from paper_eval.agents import AGENTS, build_reading_pack
from paper_eval.evidence_retrieval import (
    build_evidence_vectors,
    build_query_terms,
    recall_evidence,
    rerank_evidence,
    select_evidence,
)
from paper_eval.llm import parse_json_content  # noqa: The bad response of F401 rearrangement is rescued/thrown out at this layer
from paper_eval.models import Evidence
from paper_eval.rules import IndicatorRule, load_rule_profile

RULES = Path(__file__).resolve().parents[1] / 'rules' / "rules_v1.yaml"
AGENT_FOR_DIMENSION = {dimension: agent for agent in AGENTS for dimension in agent.dimension_ids}


@dataclass
class FakeConfig:
    'Minimal configuration stand-in: only fields read by the retrieval layer to avoid self-checking relying on real YAML/environment variables.'

    evidence_retrieval_enabled: bool = True
    evidence_recall_top_k: int = 30
    evidence_rerank_enabled: bool = True
    evidence_rerank_max_chars: int = 240


def evidence(evidence_id: str, page: int, section: str, text: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        paper_id="PAPER-TEST",
        section=section,
        page=page,
        span=(0, len(text)),
        text=text,
        text_hash=f"h-{evidence_id}",
    )


def build_corpus() -> list[Evidence]:
    '10 pieces of mixed Chinese and English evidence. EV-05 talks about samples and data collection, EV-06 talks about identification strategies and endogeneity, and the rest is noise.'
    return [
        evidence(
            "EV-01", 1, "Introduction",
            'This article studies the impact of digital transformation on corporate innovation performance. This study examines how digital transformation'
            "affects innovation performance in Chinese manufacturing firms, a topic widely discussed.",
        ),
        evidence(
            "EV-02", 1, "Theory",
            'Dynamic capabilities theory believes that enterprises need to reconstruct their resource base. The dynamic capability view argues that firms must'
            "reconfigure their resource base to cope with environmental change.",
        ),
        evidence(
            "EV-03", 2, "Theory",
            'This article puts forward three theoretical propositions. We develop three propositions linking organizational slack to'
            "exploratory learning, building on the attention-based view of the firm.",
        ),
        evidence(
            "EV-04", 3, "Literature Review",
            'The existing literature mainly focuses on large enterprises. Prior work has focused on large incumbents, leaving small and'
            "medium-sized enterprises largely unexplored in the digital context.",
        ),
        evidence(
            "EV-05", 6, "Methods",
            'Data source and sample: We collected panel data of Shanghai and Shenzhen A-share manufacturing listed companies from 2015 to 2020.'
            'Total 1,846 firm-year observations. Sample and data collection proceeded in three steps: we drew the'
            "sample from the CSMAR database, dropped firms with missing key variables, and matched the "
            "survey responses with archival financial data. The final sample covers 1,846 firm-year "
            "observations. Response rate was 62 percent and non-response bias tests showed no systematic "
            "difference between respondents and non-respondents.",
        ),
        evidence(
            "EV-06", 7, "Methods",
            'Model setting and identification strategy: In order to alleviate the endogeneity problem, we use the instrumental variable method to use the digital data of other companies in the same industry.'
            'degree means are used as instrumental variables and first-stage F statistics are reported. Identification strategy: to address'
            "endogeneity and selection bias we instrument digital transformation with the industry-year "
            "mean excluding the focal firm. The first-stage F statistic is 34.7, above the weak-instrument "
            "threshold, and the exclusion restriction is argued to hold because peer adoption shocks are "
            "plausibly unrelated to a focal firm's idiosyncratic innovation shock.",
        ),
        evidence(
            "EV-07", 8, "Results",
            'The main regression result shows that the coefficient is significantly positive. The baseline regression shows a positive and significant'
            "coefficient of 0.18 on digital transformation, robust to firm and year fixed effects.",
        ),
        evidence(
            "EV-08", 9, "Discussion",
            'This study has implications for practice. Our findings suggest managers should invest in digital infrastructure,'
            "and policymakers may consider supporting SME digital adoption programmes.",
        ),
        evidence(
            "EV-09", 10, "Conclusion",
            'This article has limitations. This paper has limitations: the sample covers only listed manufacturing firms,'
            "so external validity to service industries remains untested.",
        ),
        evidence(
            "EV-10", 11, "References",
            "References\nZhang, Y. (2019). Digital transformation and firm performance. Journal of "
            "Management Studies, 56(3), 445-478. sample data collection identification strategy endogeneity "
            "instrumental variable validity reliability.",
        ),
    ]


def recall_rank(terms: list[str], evidences: list[Evidence], top_k: int) -> list[str]:
    vectors = build_evidence_vectors(evidences)
    evidences_by_id = {item.evidence_id: item for item in evidences}
    return [item.evidence_id for _score, item in recall_evidence(terms, vectors, evidences_by_id, top_k)]


def legacy_pack(rule: IndicatorRule, evidences: list[Evidence], limit: int) -> list[Evidence]:
    'The result of the old implementation (heuristic reading package) is used as the basis for completing the source and number of items.'
    return build_reading_pack(rule, evidences, AGENT_FOR_DIMENSION[rule.dimension_id], limit, None)


def test_size_parity() -> None:
    '0. The number of items does not shrink: When there are very few recall hits, the package opened by the switch cannot be smaller than the old implementation.\n\n    Reproduce the defects reported by the parent agent: 6 texts + 24 Appendix noise, rules D-MEASUREMENT-VALIDITY-V1,\n    evidence_recall_top_k=30, reordering is turned off - only 2 paragraphs hit validity/reliability,\n    So the old implementation gave 10 items, and the retrieval layer only gave 2 items. This is a reverse change and must be nailed down.'
    profile = load_rule_profile(RULES)
    rule = profile.rules_by_id["D-MEASUREMENT-VALIDITY-V1"]
    evidences = [
        evidence(
            f"EV-BODY-{index}", 5 + index, "Methods",
            f"method section {index}: We describe the measurement validity and reliability test of the construct,"
            f"reports Cronbach's alpha and confirmatory factor analysis results." * 3,
        )
        for index in range(6)
    ] + [
        evidence(f"EV-APP-{index}", 20 + index, "Appendix", f"Appendix Noise Section {index}: Variable definition table and supplementary instructions." * 3)
        for index in range(24)
    ]

    legacy = legacy_pack(rule, evidences, 10)
    assert len(legacy) == 10, f"The old implementation would have filled limit: {len(legacy)}"

    for rerank_on in (False, True):
        on = FakeConfig(evidence_retrieval_enabled=True, evidence_rerank_enabled=rerank_on)
        if rerank_on:
            # Rearrangement is on but the model only selects 1: This is the defective site (less model selection + less recall hits).
            original = evidence_retrieval._call_rerank_llm
            evidence_retrieval._call_rerank_llm = lambda _payload, _config: {
                "selected": [{"evidence_id": "EV-BODY-0", "reason": 'reported reliability coefficients'}]
            }
            try:
                pack = select_evidence(rule, evidences, AGENT_FOR_DIMENSION["D"], 10, None, on)
            finally:
                evidence_retrieval._call_rerank_llm = original
        else:
            pack = select_evidence(rule, evidences, AGENT_FOR_DIMENSION["D"], 10, None, on)
        got = [item.evidence_id for item in pack]
        assert len(got) == len(legacy), f"The number of items in the search layer (rearrangement={rerank_on}) has shrunk: {len(got)} != {len(legacy)}"
        assert len(set(got)) == len(got), f"Duplication appears in completion: {got}"
    assert "EV-BODY-0" in got, got

    # Only if the text is indeed insufficient, less than limit is allowed: 3 pieces of evidence. When limit=10, both sides can only provide 3 pieces of evidence.
    tiny = evidences[:3]
    on = FakeConfig(evidence_retrieval_enabled=True, evidence_rerank_enabled=False)
    tiny_legacy = legacy_pack(rule, tiny, 10)
    tiny_on = select_evidence(rule, tiny, AGENT_FOR_DIMENSION["D"], 10, None, on)
    assert len(tiny_on) == len(tiny_legacy) == 3, (len(tiny_on), len(tiny_legacy))


def test_recall_ordering() -> None:
    '1. Recall: The two items of sample/identification strategy must be ranked first, and the reference block must not enter the pool.'
    evidences = build_corpus()
    assert "EV-10" not in build_evidence_vectors(evidences), 'Reference evidence should not be entered into the search pool'

    # Data and sample search terms
    sample_terms = ["sample", "data collection", "respondents", "non-response bias"]
    ranked = recall_rank(sample_terms, evidences, 30)
    assert ranked, 'Sample class search word recall is empty'
    assert ranked[0] == "EV-05", f"The first sample category recall is not EV-05: {ranked}"
    # There is only one edge in the noise (EV-09 mentions sample, EV-06 mentions bias), and it must be ranked after EV-05.
    assert ranked.index("EV-05") < ranked.index("EV-09"), ranked
    assert ranked.index("EV-05") < ranked.index("EV-06"), ranked

    # identification strategy/endogenous search terms
    identification_terms = ["identification strategy", "endogeneity", "instrumental variable", "selection bias"]
    ranked = recall_rank(identification_terms, evidences, 30)
    assert ranked, 'Recognition strategy search word recall is empty'
    assert ranked[0] == "EV-06", f"The first identification strategy recall is not EV-06: {ranked}"
    assert ranked.index("EV-06") < ranked.index("EV-05"), ranked

    # top_k truncation and 0 points discarded
    ranked_tiny = recall_rank(sample_terms, evidences, 2)
    assert len(ranked_tiny) == 2, ranked_tiny
    assert not recall_rank(["zzyzx"], evidences, 30), 'A completely missed search term should not recall any evidence.'

    # Search pool excludes external documents (EXT-*)
    ext = evidence("EXT-1", 0, 'External literature (OpenAlex search)', "sample data collection endogeneity")
    ext.source_type = "external"
    assert "EXT-1" not in build_evidence_vectors(evidences + [ext]), 'External documents should not be entered into the general search pool'


def test_query_terms() -> None:
    '2. Search terms: priority rule.retrieval_terms; when empty, fall back to the English token of rule_id + evidence_location.'
    profile = load_rule_profile(RULES)
    rule = profile.rules_by_id["D-DATA-SAMPLE-V1"]
    terms = build_query_terms(rule)
    assert terms, 'The fallback search term is empty'
    assert all(term.strip() for term in terms), terms
    assert "d" not in terms and "v1" not in terms, f"Dimension prefix/version number of rule id should be filtered: {terms}"

    # retrieval_terms must be given priority after filling, no longer look at rule_id
    fake = IndicatorRule(
        rule_id="D-DATA-SAMPLE-V1",
        version="1.0",
        dimension_id="D",
        dimension_name='Study design and methodological quality',
        indicator_name='The data source and sample are reasonable',
        definition='Chinese definition should not be included in the search terms',
        required_evidence="",
        evidence_location=["Methods"],
        observation_points=[],
        judgement_values={"partial": 'Sample description is incomplete'},
        score_mapping={"partial": 0.6},
        weight=1.0,
        applicable_type='General',
        source_reference=[],
        status="active",
        retrieval_terms=["sample", "data collection", "response rate"],
    )
    assert build_query_terms(fake) == ["sample", "data collection", "response rate"], build_query_terms(fake)


def test_rerank_order_and_subset() -> None:
    '3. Rearrange: Return the order and subset according to the evidence_id given by the model, and only feed the model with the truncated text.'
    evidences = build_corpus()
    config = FakeConfig()
    rule = load_rule_profile(RULES).rules_by_id["D-MODEL-IDENTIFICATION-V1"]

    def fake_llm(payload, _config):
        # False response: Deliberately pick the "two items at the back of the recall order" and give them in reverse order to verify that the return order indeed follows the model and does not follow the recall.
        assert payload["budget"] == 3, payload["budget"]
        assert payload["judgement_values"], 'gear anchor point must enter payload'
        for candidate in payload["candidates"]:
            assert len(candidate["text"]) <= 240, len(candidate["text"])
        return {
            "selected": [
                {"evidence_id": "EV-06", "reason": 'gives the instrumental variables and the first-stage F value, directly covering the endogeneity treatment.'},
                {"evidence_id": "EV-09", "reason": 'explains sample boundaries and external validity limitations'},
            ]
        }

    original = evidence_retrieval._call_rerank_llm
    evidence_retrieval._call_rerank_llm = fake_llm
    try:
        vectors = build_evidence_vectors(evidences)
        evidences_by_id = {item.evidence_id: item for item in evidences}
        recalled = [item for _score, item in recall_evidence(
            ["identification", "endogeneity", "sample"], vectors, evidences_by_id, 30
        )]
        recalled_ids = [item.evidence_id for item in recalled]
        assert recalled_ids[:2] == ["EV-05", "EV-06"], recalled_ids
        reranked = rerank_evidence(rule, recalled, config, 3)
    finally:
        evidence_retrieval._call_rerank_llm = original

    got = [item.evidence_id for item in reranked]
    assert got[:2] == ["EV-06", "EV-09"], f"The return order must follow the order given by the model, not the recall order: {got}\nWhen"
    assert len(got) == 3 and got[2] == recalled_ids[0], f"is not satisfied with the limit, the recall sequence should be used to complete it: {got}"
    assert len(set(got)) == 3, 'Completion cannot produce duplicates'
    assert reranked[1].text.endswith("untested."), 'What is included in the reading package must be the complete original text, not a truncated version.'


def test_rerank_fallbacks() -> None:
    '4. Three degradation paths: throwing exception / illegal JSON / unknown id, all must return to the recall sequence without throwing exceptions.'
    evidences = build_corpus()
    config = FakeConfig()
    rule = load_rule_profile(RULES).rules_by_id["D-MODEL-IDENTIFICATION-V1"]
    vectors = build_evidence_vectors(evidences)
    evidences_by_id = {item.evidence_id: item for item in evidences}
    # The search terms are deliberately widened to recall most of the corpus, so that "cut to limit=4" really measures truncation.
    recalled = [item for _score, item in recall_evidence(
        ["identification", "endogeneity", "sample", "data", "firm", "performance"], vectors, evidences_by_id, 30
    )]
    expected = [item.evidence_id for item in recalled[:4]]
    assert len(recalled) >= 4, [item.evidence_id for item in recalled]

    def raises(_payload, _config):
        raise RuntimeError("Cloud model timed out after 300 seconds")

    def bad_json(_payload, _config):
        # call_plain_json will throw a RuntimeError when parsing fails. Here, the same layer of errors is simulated.
        raise RuntimeError("model json parse failed: expecting value")

    def unknown_ids(_payload, _config):
        return {"selected": [{"evidence_id": "NOPE-1"}, {"evidence_id": "NOPE-2"}]}

    def empty_selection(_payload, _config):
        return {"selected": []}

    def illegal_shape(_payload, _config):
        return {"note": 'I refuse to output JSON'}

    cases = {
        'throw exception': raises,
        'Illegal JSON': bad_json,
        'unknown id': unknown_ids,
        'returns empty': empty_selection,
        'illegal structure': illegal_shape,
    }
    # The downgrade path also needs to be nailed to "the number of entries will not shrink": only giving 2 when downgrading is worse than directly returning to the old implementation.
    legacy_ids = [
        item.evidence_id
        for item in build_reading_pack(rule, evidences, AGENT_FOR_DIMENSION[rule.dimension_id], 4, None)
    ]
    assert len(legacy_ids) == 4, legacy_ids
    original = evidence_retrieval._call_rerank_llm
    try:
        for name, fake in cases.items():
            evidence_retrieval._call_rerank_llm = fake
            reranked = rerank_evidence(rule, recalled, config, 4, legacy_pack(rule, evidences, 4))
            got = [item.evidence_id for item in reranked]
            assert got[: len(expected)] == expected, f"The downgrade path \"{name}\" does not return the recall sequence: {got} != {expected}"
            assert len(got) == len(legacy_ids), f"The number of downgrade paths \"{name}\" has shrunk: {len(got)} < {len(legacy_ids)}"
            assert len(set(got)) == len(got), f"The downgrade path \"{name}\" is duplicated: {got}\nThe more common form of bad response for"
    finally:
        evidence_retrieval._call_rerank_llm = original

    # real calls is "the parser throws an error" rather than "the parsed structure is incorrect":
    # call_plain_json → parse_json_content will throw RuntimeError when encountering non-JSON text. The bad_json above covers this.
    # Here is another layer: JSON that has been rescued by the parser and wrapped in markdown code blocks should be used normally without triggering downgrade.
    wrapped = '```json\n{"selected": [{"evidence_id": "EV-06", "reason": "endogeneity handling"}], "note": "ok"}\n```'
    parsed = parse_json_content(wrapped)
    ids, reasons = evidence_retrieval._extract_selected_ids(parsed)
    assert ids == ["EV-06"], ids
    assert reasons.get("EV-06") == 'Endogenous treatment', reasons


def test_config_gate() -> None:
    '4b. When configuring missing fields (old configuration/stand-in object), press "Close". You cannot change the link just because the switch cannot be read.'
    import dataclasses

    from paper_eval.llm import AgentRuntimeConfig

    fields = {field.name for field in dataclasses.fields(AgentRuntimeConfig)}
    assert {
        "evidence_retrieval_enabled",
        "evidence_recall_top_k",
        "evidence_rerank_enabled",
        "evidence_rerank_max_chars",
    } <= fields, sorted(fields)

    defaults = AgentRuntimeConfig()
    assert defaults.evidence_retrieval_enabled is False, 'The default must be closed'
    assert defaults.evidence_recall_top_k == 30
    assert defaults.evidence_rerank_enabled is True
    # Don’t copy constants, keep properties: the median length of 1630 paragraphs of evidence in three articles measured is 137 words, P90 is 272 words,
    # Truncation below 400 will cut off about 14% of the tail of the segment, and rearrangement will misjudge that the segment is not relevant.
    assert defaults.evidence_rerank_max_chars >= 400, (
        f"Rearranged and truncated {defaults.evidence_rerank_max_chars} is too short and will cut off the end of the evidence fragment (P90 = 272 words)"
    )

    class LegacyConfig:
        'simulates the old configuration object without these fields.'

    profile = load_rule_profile(RULES)
    evidences = build_corpus()
    rule = profile.rules_by_id["D-DATA-SAMPLE-V1"]
    agent = AGENT_FOR_DIMENSION["D"]
    legacy = build_reading_pack(rule, evidences, agent, 4, None)
    assert [
        item.evidence_id for item in select_evidence(rule, evidences, agent, 4, None, LegacyConfig())
    ] == [item.evidence_id for item in legacy]


def test_switch_off_matches_legacy() -> None:
    '5. Switch off: select_evidence must be consistent with build_reading_pack one by one (the proof does not change the existing behavior).'
    profile = load_rule_profile(RULES)
    evidences = build_corpus()
    evidences += [
        evidence(f"EV-M{i}", 6 + i, "Methods", f"Method section {i}: measurement validity and reliability of the scales." * 4)
        for i in range(6)
    ]
    evidences += [
        evidence(f"EV-D{i}", 12 + i, "Discussion", f"Discussion section {i}: limitations and future research directions." * 4)
        for i in range(6)
    ]
    off = FakeConfig(evidence_retrieval_enabled=False)

    checked = 0
    for rule in profile.rules:
        agent = AGENT_FOR_DIMENSION[rule.dimension_id]
        legacy = build_reading_pack(rule, evidences, agent, 6, None)
        via_layer = select_evidence(rule, evidences, agent, 6, None, off)
        assert [item.evidence_id for item in via_layer] == [item.evidence_id for item in legacy], rule.rule_id
        checked += 1
    assert checked == len(profile.rules), checked

    # config=None must also adopt the old behavior (the parent agent cannot change the link silently when the config is missed)
    rule = profile.rules_by_id["D-DATA-SAMPLE-V1"]
    agent = AGENT_FOR_DIMENSION["D"]
    assert [
        item.evidence_id for item in select_evidence(rule, evidences, agent, 6, None, None)
    ] == [item.evidence_id for item in build_reading_pack(rule, evidences, agent, 6, None)]


def test_switch_on_uses_recall() -> None:
    '6. The switch is on: the relevant evidence (sample/endogeneity) must actually enter the reading package - this is the problem to be solved at this level.'
    profile = load_rule_profile(RULES)
    evidences = build_corpus()
    on = FakeConfig(evidence_retrieval_enabled=True, evidence_rerank_enabled=False)

    # Use rule_id fallback word when there is no retrieval_terms: D-DATA-SAMPLE-V1 → sample / data
    rule = profile.rules_by_id["D-DATA-SAMPLE-V1"]
    pack = [item.evidence_id for item in select_evidence(rule, evidences, AGENT_FOR_DIMENSION["D"], 3, None, on)]
    assert "EV-05" in pack, f"Sample evidence not included: {pack}"

    rule = profile.rules_by_id["D-MODEL-IDENTIFICATION-V1"]
    pack = [item.evidence_id for item in select_evidence(rule, evidences, AGENT_FOR_DIMENSION["D"], 3, None, on)]
    assert "EV-06" in pack, f"The identification strategy evidence was not included in the package: {pack}"

    # When the recall is empty, the heuristic must be rolled back and empty packets cannot be delivered.
    empty_rule = IndicatorRule(
        rule_id="X-NOTHING-V1",
        version="1.0",
        dimension_id="D",
        dimension_name="d",
        indicator_name="n",
        definition="d",
        required_evidence="",
        evidence_location=["Methods"],
        observation_points=[],
        judgement_values={},
        score_mapping={},
        weight=1.0,
        applicable_type='General',
        source_reference=[],
        status="active",
        retrieval_terms=["zzyzx", "qqqqq"],
    )
    fallback = select_evidence(empty_rule, evidences, AGENT_FOR_DIMENSION["D"], 3, None, on)
    assert [item.evidence_id for item in fallback] == [
        item.evidence_id for item in build_reading_pack(empty_rule, evidences, AGENT_FOR_DIMENSION["D"], 3, None)
    ], 'No fallback heuristic when recall is empty'


def test_no_new_dependency() -> None:
    '7. There are no new third-party dependencies: the search layer only relies on the standard library + the existing modules of the project.'
    for banned in ("numpy", "faiss", "sklearn", "scipy", "torch", "rank_bm25", "jieba"):
        assert banned not in sys.modules, f"introduces an undeclared third-party dependency: {banned}"
    source = (Path(__file__).resolve().parents[1] / "paper_eval" / "evidence_retrieval.py").read_text(
        encoding="utf-8"
    )
    assert "import numpy" not in source and "import faiss" not in source
    # Reuse existing search primitives, don’t rewrite a set of participles/cosines yourself
    from paper_eval.retrieval import cosine, tokenize, vectorize  # noqa: F401

    assert evidence_retrieval.vectorize is vectorize
    assert evidence_retrieval.cosine is cosine


def main() -> int:
    test_size_parity()
    test_recall_ordering()
    test_query_terms()
    test_rerank_order_and_subset()
    test_rerank_fallbacks()
    test_config_gate()
    test_switch_off_matches_legacy()
    test_switch_on_uses_recall()
    test_no_new_dependency()
    print(
        'Evidence recall self-test passed: the number of retrieval layer items is not less than the old implementation (heuristics are used to complete the recall hits when there are few); the word bag recall order is correct;'
        'LLM rearranges and returns in model order and subsets; throw exception/illegal JSON/unknown id/null return/illegal structure five downgrades will return to the recall order without shrinking;\nWhen the'
        'switch is turned off, select_evidence and build_reading_pack are consistent one by one; no third-party dependencies are added.'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

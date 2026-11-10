'Evidence retrieval wiring self-test: Verify that after select_evidence is connected to the reading package, the behavior of both states of the switch is correct.\n\nThis self-test is about "wiring" rather than "retrieval quality":\n- When the switch is turned off, it must be the same as the old build_reading_pack **everything** (otherwise "turning off the new link = return to the old behavior" is just a lip service)\n- When the switch is turned on, recall must select the rule-related paragraphs into the package, and build_dimension_reading_packs also gets config\n- External documents (EXT-*) are still only inserted into the specified rules\n\nRunning method:\n    python ".\\scripts\\self_test_evidence_retrieval_wiring.py"'

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_eval.agents import (  # noqa: E402
    METHOD_QUALITY_AGENT,
    build_dimension_reading_packs,
    build_reading_pack,
)
from paper_eval.evidence_retrieval import (  # noqa: E402
    build_evidence_vectors,
    build_query_terms,
    recall_evidence,
    select_evidence,
)
from paper_eval.external_retrieval import external_evidence_blocks  # noqa: E402
from paper_eval.llm import AgentRuntimeConfig  # noqa: E402
from paper_eval.models import Evidence  # noqa: E402
from paper_eval.rules import load_rule_profile  # noqa: E402

RULES = ROOT / 'rules' / "rules_v1.yaml"
TARGET_RULE = "D-MEASUREMENT-VALIDITY-V1"
# Which rule should each piece of construction evidence be recalled by. The search term must allow the "right rule" to find the "right paragraph".
# So here we verify separately according to the rules instead of using a piece of evidence to test all the rules.
RECALL_EXPECTATIONS = {
    "D-MEASUREMENT-VALIDITY-V1": "EV-VALID",
    "D-MODEL-IDENTIFICATION-V1": "EV-IDENT",
    "D-DATA-SAMPLE-V1": "EV-SAMPLE",
}


def make_evidence(evidence_id: str, section: str, page: int, text: str, offset: int = 0) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        paper_id="PAPER-SELFTEST",
        section=section,
        page=page,
        span=(offset, offset + len(text)),
        text=text,
        text_hash=f"HASH-{evidence_id}",
    )


def build_evidences() -> list[Evidence]:
    'Construct a small set of text evidence: two are directly related to D-MEASUREMENT, and the rest is noise.'
    items = [
        make_evidence(
            "EV-VALID",
            "Methods",
            5,
            "We assessed construct validity by reporting Cronbach's alpha, composite reliability, "
            "and average variance extracted for each construct, and confirmed discriminant validity "
            "with the Fornell-Larcker criterion and the heterotrait-monotrait ratio. Measurement items "
            "were adapted from prior scales and the adaptation rationale is described below.",
        ),
        make_evidence(
            "EV-IDENT",
            "Methods",
            6,
            "Identification strategy: because treatment assignment may be endogenous, we exploit the "
            "exogenous variation in the rollout timing and report instrumental variable estimates, "
            "along with tests for weak instruments, selection bias, and common method variance.",
        ),
        make_evidence(
            "EV-SAMPLE",
            "Methods",
            4,
            "Sample and procedure: our sample comprised 949 experts drawn from a multi-wave panel, "
            "and the response rate, inclusion criteria, and sampling frame are described in Appendix A.",
        ),
        make_evidence(
            "EV-THEORY",
            "Theory",
            3,
            "The theoretical framework integrates the transactional theory of stress with the "
            "literature on algorithmic management to explain appraisal formation.",
        ),
        make_evidence(
            "EV-RESULTS",
            "Results",
            9,
            "Results indicate a positive and significant effect of the focal predictor on the outcome, "
            "and the coefficients are reported in Table 2.",
        ),
        make_evidence(
            "EV-LIMIT",
            "Discussion",
            14,
            "This study has limitations: the design cannot fully rule out reverse causality, and the "
            "findings may not generalize beyond the sampled industry.",
        ),
    ]
    # Noise: has nothing to do with the D dimension and does not fall in the Methods chapter
    for index in range(24):
        items.append(
            make_evidence(
                f"EV-NOISE-{index:02d}",
                "Appendix",
                18,
                f"Supplementary checklist item {index}: administrative notes on formatting, "
                "cover sheet, and submission metadata that carry no substantive content.",
                offset=index,
            )
        )
    return items


def make_config(**overrides) -> AgentRuntimeConfig:
    'does not read the independent configuration of .env: AgentRuntimeConfig is all fields with default values \u200b\u200band does not trigger any network calls.'
    return AgentRuntimeConfig(**overrides)


def main() -> int:
    profile = load_rule_profile(RULES)
    rule = profile.rules_by_id[TARGET_RULE]
    evidences = build_evidences()

    off = make_config(evidence_retrieval_enabled=False)
    on_no_rerank = make_config(
        evidence_retrieval_enabled=True,
        evidence_recall_top_k=30,
        evidence_rerank_enabled=False,
    )

    # 1. Switch off: same as old build_reading_pack item by item
    legacy = build_reading_pack(rule, evidences, METHOD_QUALITY_AGENT, 10, None)
    switched_off = select_evidence(rule, evidences, METHOD_QUALITY_AGENT, 10, None, off)
    assert [item.evidence_id for item in legacy] == [item.evidence_id for item in switched_off], (
        'When the switch is turned off, select_evidence must be consistent with build_reading_pack one by one.',
        [item.evidence_id for item in legacy],
        [item.evidence_id for item in switched_off],
    )
    assert switched_off, 'Empty packets should not be returned even when the switch is turned off.'

    # 2. When config is None, the old behavior will also be returned (the behavior cannot be changed if the caller misses config)
# Build_reading_pack should be returned when
    assert [
        item.evidence_id
        for item in select_evidence(rule, evidences, METHOD_QUALITY_AGENT, 10, None, None)
    ] == [item.evidence_id for item in legacy], 'config=None'

    # 2b. When using paper_context, it must also be consistent one by one.
    # paper_context contains evidence routing table and facet index, which is the real source of evidence for build_reading_pack;
    # If the position parameter is missed or wrongly passed during wiring, it will be exposed here (and will not be exposed in the use case of paper_context=None).
    paper_context = {
        "evidence_index": {
            facet: [item.evidence_id for item in evidences if item.section == section]
            for facet, section in (("method", "Methods"), ("results", "Results"), ("discussion", "Discussion"))
        },
        "dimension_evidence_routes": {"D": ["EV-SAMPLE", "EV-IDENT", "EV-VALID"]},
    }
    legacy_ctx = build_reading_pack(rule, evidences, METHOD_QUALITY_AGENT, 10, paper_context)
    off_ctx = select_evidence(rule, evidences, METHOD_QUALITY_AGENT, 10, paper_context, off)
    assert [item.evidence_id for item in legacy_ctx] == [
        item.evidence_id for item in off_ctx
    ], 'When the switch is turned off with paper_context, it must be equal to build_reading_pack one by one.\nWhen'
    on_ctx = select_evidence(rule, evidences, METHOD_QUALITY_AGENT, 10, paper_context, on_no_rerank)
    assert len(on_ctx) == len(legacy_ctx), (
        f"has paper_context, the number of entries shrinks: new {len(on_ctx)} vs old {len(legacy_ctx)}"
    )

    # 3. Recall itself can hand over the "right paragraph" to the "right rule"
# The search terms for
    vectors = build_evidence_vectors(evidences)
    evidences_by_id = {item.evidence_id: item for item in evidences}
    recalled_by_rule: dict[str, list[str]] = {}
    for rule_id, expected in RECALL_EXPECTATIONS.items():
        target = profile.rules_by_id[rule_id]
        terms = build_query_terms(target, None)
        assert terms, f"{rule_id} are empty and cannot be recalled (are retrieval_terms missing?)"
        ranked = recall_evidence(terms, vectors, evidences_by_id, 30)
        ids = [item.evidence_id for _score, item in ranked]
        recalled_by_rule[rule_id] = ids
        assert expected in ids, f"{rule_id} was not recalled to {expected}, actual hit: {ids}"
        assert not ids or ids[0] != "EV-NOISE-00", f"{rule_id} Putting noise first: {ids[:5]}"

    # 4. Switch on (rearrangement is off to avoid true adjustment of the model): related paragraphs are included in the package
    on_pack = select_evidence(rule, evidences, METHOD_QUALITY_AGENT, 10, None, on_no_rerank)
    on_ids = [item.evidence_id for item in on_pack]
    assert "EV-VALID" in on_ids, f"The search layer opening validity evidence was not included in the package: {on_ids}"
    # The search layer cannot "give less evidence": when there is enough text, the number of items must be equal to the old heuristic.
    # The recall only hits 2 items and then shrinks the package to 2 items. This is to reduce the evidence coverage, which is contrary to the goal of this transformation.
    assert len(on_pack) == len(legacy), (
        f"When the search layer is opened, only {len(on_pack)} items are given, and the old heuristic gives {len(legacy)} items: backfilling is required when there are insufficient recall hits.",
        on_ids,
    )

    # 5. build_dimension_reading_packs must penetrate the config (this is evidence whether the wiring is really effective)
    dim_off = build_dimension_reading_packs([rule], evidences, METHOD_QUALITY_AGENT, 10, None, config=None)
    dim_on = build_dimension_reading_packs(
        [rule], evidences, METHOD_QUALITY_AGENT, 10, None, config=on_no_rerank
    )
    assert [
        item.evidence_id for item in dim_off[rule.rule_id]
    ] == [item.evidence_id for item in legacy], 'build_dimension_reading_packs should equal old behavior when config=None'
    assert "EV-VALID" in [item.evidence_id for item in dim_on[rule.rule_id]], (
        'build_dimension_reading_packs does not pass config to select_evidence',
        [item.evidence_id for item in dim_on[rule.rule_id]],
    )

    # 6. External documents are still only inserted into specified rules
    external = external_evidence_blocks(
        {
            "related_works": [
                {
                    "title": "Related work on algorithmic management and appraisal",
                    "year": 2022,
                    "rerank_score": 1.0,
                    "doi": "10.1/selftest",
                    "source": "crossref",
                    "summary": {"method": "survey", "finding": "algorithmic management affects appraisal"},
                }
            ]
        },
        "PAPER-SELFTEST",
    )
    assert external, 'Failed to construct external document block, self-test prerequisite is not established'
    with_external = [*evidences, *external]
    designated = {rule.rule_id}
    packs = build_dimension_reading_packs(
        [rule], with_external, METHOD_QUALITY_AGENT, 10, None, external_rule_ids=designated, config=on_no_rerank
    )
    pack_ids = [item.evidence_id for item in packs[rule.rule_id]]
    assert any(item.startswith("EXT-") for item in pack_ids), 'The specified rule did not obtain the external document block'
    other = profile.rules_by_id["A-TOPIC-SIGNIFICANCE-V1"]
    packs_other = build_dimension_reading_packs(
        [other], with_external, METHOD_QUALITY_AGENT, 10, None, external_rule_ids=designated, config=on_no_rerank
    )
    assert not any(
        item.startswith("EXT-") for item in [ev.evidence_id for ev in packs_other[other.rule_id]]
    ), 'Unspecified rules should not get external document blocks'

    print(
        'Evidence retrieval wiring self-test passed:'
        f"switch off is consistent with the old behavior item by item ({len(legacy)} item);"
        + "；".join(
            f"{rule_id} Recall {expected} (No. {recalled_by_rule[rule_id].index(expected) + 1})"
            for rule_id, expected in RECALL_EXPECTATIONS.items()
        )
        + f"; When the switch is turned on, {len(on_pack)} bars will not shrink;"
        'build_dimension_reading_packs has transparently transmitted config; external documents are only inserted into the specified rules.'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

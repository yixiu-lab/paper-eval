"""Offline checks for deterministic evidence routing in dimensions E and F.

The checks verify that each E/F rule receives a non-empty, bounded evidence pack containing both high-support and low-support material, that repeated calls return the same evidence IDs, and that other dimensions retain their expected routing behavior.

Run with: python scripts/self_test_ef_evidence_routing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_eval.agents import (  # noqa: E402
    AGENTS,
    build_dimension_reading_packs,
    build_reading_pack,
    evidence_form_channels,
)
from paper_eval.llm import AgentRuntimeConfig  # noqa: E402
from paper_eval.models import Evidence  # noqa: E402
from paper_eval.rules import load_rule_profile  # noqa: E402

RULES = ROOT / 'rules' / "rules_v1.yaml"
LIMIT = 10
AGENT_FOR_DIMENSION = {dimension: agent for agent in AGENTS for dimension in agent.dimension_ids}


def evidence(eid: str, page: int, section: str, text: str) -> Evidence:
    return Evidence(
        evidence_id=eid,
        paper_id="PAPER-EF-SELFTEST",
        section=section,
        page=page,
        span=(0, len(text)),
        text=text,
        text_hash=f"h-{eid}",
    )


# supports **high-end** paragraphs: sentences that clearly explain the results/values/availability/boundaries.
HIGH_SUPPORT = [
    evidence(
        "EV-HI-STATS",
        6,
        "Results",
        "Table 2 reports the OLS estimates. The coefficient for the focal predictor is b = 0.34 "
        "(standard error = 0.08, p < .001), the 95% confidence interval [0.18, 0.50] excludes zero, "
        "and the model explains R2 = 0.27 of the variance in the outcome.",
    ),
    evidence(
        "EV-HI-MEDIATION",
        7,
        "Results",
        "Table 3 reports the mediation and moderation estimates. The indirect effect is 0.09 with a "
        "95% bias-corrected confidence interval of [0.03, 0.17]; the index of moderated mediation is "
        "significant, supporting Hypothesis 2 and partially supporting Hypothesis 3.",
    ),
    evidence(
        "EV-HI-AVAILABILITY",
        5,
        "Methods",
        "Data availability: the data and the analysis code are available in the project repository at "
        "osf.io/example. Supplementary materials contain the survey instrument and the full coding scheme.",
    ),
    evidence(
        "EV-HI-BOUNDARY",
        12,
        "Discussion",
        "Several limitations should be noted. The design cannot rule out reverse causality, and the "
        "findings cannot be generalized beyond the sampled industry; future research should examine "
        "other settings and longer time horizons.",
    ),
]

# Paragraphs that support **low-grade**: the author writes out what he "failed to do" (not reported/not significant/not available/the conclusion is stronger than the evidence).
LOW_SUPPORT = [
    evidence(
        "EV-LO-NOTSIG",
        8,
        "Results",
        "We do not find a significant effect of the moderator on the outcome. The coefficient was not "
        "significant and the pattern of results is only suggestive of a weaker relationship.",
    ),
    evidence(
        "EV-LO-NOTROBUST",
        12,
        "Discussion",
        "The effect did not hold in the subsample of smaller firms, and the estimate was not robust to "
        "alternative specifications of the dependent variable.",
    ),
    evidence(
        "EV-LO-UNREPORTED",
        9,
        "Results",
        "Descriptive statistics for two control variables were not reported because the data were not "
        "available for the second wave of the panel.",
    ),
    evidence(
        "EV-LO-OVERCLAIM",
        13,
        "Conclusion",
        "The design proves that the intervention causes higher performance in all cases, and the result "
        "generalizes to every organization that adopts the same technology.",
    ),
    evidence(
        "EV-LO-PROPRIETARY",
        5,
        "Methods",
        "The firm-level data are proprietary and cannot be disclosed by the research team; the analysis "
        "code is not publicly available and no replication package is provided.",
    ),
    evidence(
        "EV-LO-NOTDESCRIBED",
        13,
        "Appendix",
        "The interview protocol was not described in the manuscript and the coding decisions were not "
        "reported, so the procedure cannot be followed by other researchers.",
    ),
]
LOW_SUPPORT_IDS = {item.evidence_id for item in LOW_SUPPORT}
HIGH_SUPPORT_IDS = {item.evidence_id for item in HIGH_SUPPORT}

# Method section for D-dimensional regression anchor point: changes should not affect the evidence collection of D.
METHOD_ITEMS = [
    evidence(
        f"EV-METHOD-{index}",
        4 + index,
        "Methods",
        'Research procedures and sample description section. We collected survey data from respondents in a multi-wave panel; the'
        "response rate and the sampling frame are described, every construct was measured with validated "
        "scales, and the identification strategy addresses endogeneity with instrumental variables. " * 2,
    )
    for index in range(6)
]

FILLER = [
    evidence(
        f"EV-FILL-{index}",
        2 + index,
        "Introduction",
        'background paving section. The remaining sections of this manuscript proceed as follows and introduce each part'
        "of the argument in turn for the reader, without reporting any estimate or procedure. " * 2,
    )
    for index in range(4)
]


def build_pool() -> list[Evidence]:
    return [*FILLER, *METHOD_ITEMS, *HIGH_SUPPORT, *LOW_SUPPORT]


# D-dimensional expected sequence before change (limit=10, fixed pool). See the module docstring for the generation method.
EXPECTED_D_PACKS: dict[str, list[str]] = {
    "D-DESIGN-FIT-V1": ['EV-METHOD-0', 'EV-METHOD-1', 'EV-METHOD-2', 'EV-METHOD-3', 'EV-METHOD-4', 'EV-METHOD-5', 'EV-HI-BOUNDARY', 'EV-LO-NOTDESCRIBED', 'EV-FILL-2', 'EV-HI-AVAILABILITY'],
    "D-DATA-SAMPLE-V1": ['EV-METHOD-0', 'EV-METHOD-1', 'EV-METHOD-2', 'EV-METHOD-3', 'EV-METHOD-4', 'EV-METHOD-5', 'EV-HI-BOUNDARY', 'EV-LO-NOTDESCRIBED', 'EV-FILL-2', 'EV-HI-AVAILABILITY'],
    "D-MEASUREMENT-VALIDITY-V1": ['EV-FILL-0', 'EV-METHOD-0', 'EV-METHOD-1', 'EV-METHOD-2', 'EV-METHOD-3', 'EV-METHOD-4', 'EV-METHOD-5', 'EV-LO-NOTDESCRIBED', 'EV-FILL-2', 'EV-HI-AVAILABILITY'],
    "D-MODEL-IDENTIFICATION-V1": ['EV-METHOD-0', 'EV-METHOD-1', 'EV-HI-STATS', 'EV-HI-MEDIATION', 'EV-METHOD-4', 'EV-METHOD-5', 'EV-LO-NOTDESCRIBED', 'EV-HI-AVAILABILITY', 'EV-METHOD-2', 'EV-METHOD-3'],
    "D-METHOD-EXECUTION-V1": ['EV-FILL-0', 'EV-FILL-1', 'EV-METHOD-0', 'EV-METHOD-1', 'EV-METHOD-2', 'EV-METHOD-3', 'EV-METHOD-4', 'EV-METHOD-5', 'EV-LO-NOTDESCRIBED', 'EV-FILL-2'],
    "D-QUAL-GENRE-FIT-V1": ['EV-FILL-0', 'EV-METHOD-0', 'EV-METHOD-1', 'EV-METHOD-2', 'EV-METHOD-3', 'EV-METHOD-4', 'EV-METHOD-5', 'EV-LO-NOTDESCRIBED', 'EV-FILL-2', 'EV-HI-AVAILABILITY'],
}


def dump_d(profile) -> int:
    pool = build_pool()
    for rule in profile.rules:
        if rule.dimension_id != "D":
            continue
        pack = build_reading_pack(rule, pool, AGENT_FOR_DIMENSION["D"], LIMIT, None)
        print(f'    "{rule.rule_id}": {[item.evidence_id for item in pack]!r},')
    return 0


def main() -> int:
    profile = load_rule_profile(RULES)
    if "--dump-d" in sys.argv:
        print('Current D-dimensional reading package (should be the baseline before modification and must be consistent after modification):')
        return dump_d(profile)

    pool = build_pool()
    ef_rules = [rule for rule in profile.rules if rule.dimension_id in {"E", "F"}]
    assert ef_rules, 'There is no E/F rule in the rule base, and the self-test prerequisite is not established.'

    # 1) E/F: Not empty, not exceeding limit, both types of evidence appear
# The reading package of
    for rule in ef_rules:
        agent = AGENT_FOR_DIMENSION[rule.dimension_id]
        pack = build_reading_pack(rule, pool, agent, LIMIT, None)
        ids = [item.evidence_id for item in pack]
        assert pack, f"{rule.rule_id} is empty"
        assert len(pack) <= LIMIT, f"{rule.rule_id} Super hair: {len(pack)}"
        assert len(set(ids)) == len(ids), f"{rule.rule_id} Duplicate evidence appears: {ids}"
        high_present = HIGH_SUPPORT_IDS & set(ids)
        low_present = LOW_SUPPORT_IDS & set(ids)
        assert high_present, f"{rule.rule_id} package does not support high-end paragraphs: {ids}"
        assert low_present, f"{rule.rule_id} package does not support low-end paragraphs: {ids}"

    # 2) Deterministic: Same as inputting two calls one by one
    for rule in ef_rules:
        agent = AGENT_FOR_DIMENSION[rule.dimension_id]
        first = [item.evidence_id for item in build_reading_pack(rule, pool, agent, LIMIT, None)]
        second = [item.evidence_id for item in build_reading_pack(rule, pool, agent, LIMIT, None)]
        assert first == second, f"{rule.rule_id} The results of the two calls are different: {first} vs {second}"

    # 3) The final package that the model actually sees (retrieval layer on, reordering off) must also have both types, and the number of items should not be reduced.
    config = AgentRuntimeConfig(
        evidence_retrieval_enabled=True,
        evidence_rerank_enabled=False,
        evidence_recall_top_k=30,
    )
    for rule in ef_rules:
        agent = AGENT_FOR_DIMENSION[rule.dimension_id]
        legacy = [item.evidence_id for item in build_reading_pack(rule, pool, agent, LIMIT, None)]
        final_pack = build_dimension_reading_packs(
            [rule], pool, agent, LIMIT, None, config=config
        )[rule.rule_id]
        final_ids = [item.evidence_id for item in final_pack]
        assert len(final_ids) == len(legacy), (
            f"{rule.rule_id} The final number of packets is inconsistent with the old heuristic: {len(final_ids)} vs {len(legacy)}"
        )
        assert HIGH_SUPPORT_IDS & set(final_ids), f"{rule.rule_id} The final package lost the paragraph supporting high-end: {final_ids}"
        assert LOW_SUPPORT_IDS & set(final_ids), f"{rule.rule_id} The final package lost the paragraph supporting low-end: {final_ids}"
        again = build_dimension_reading_packs([rule], pool, agent, LIMIT, None, config=config)[rule.rule_id]
        assert final_ids == [item.evidence_id for item in again], f"{rule.rule_id} The final package cannot be reproduced"

    # 4) The non-E/F dimension (D) is equal to the expected sequence before the change one by one - the change does not overflow
    d_rules = [rule for rule in profile.rules if rule.dimension_id == "D"]
    assert EXPECTED_D_PACKS, 'EXPECTED_D_PACKS is empty, please run --dump-d first to generate'
    assert {rule.rule_id for rule in d_rules} == set(EXPECTED_D_PACKS), (
        'The D-dimensional rule set has changed, and the expected sequence needs to be regenerated.\nThe reading package of',
        sorted(rule.rule_id for rule in d_rules),
    )
    for rule in d_rules:
        agent = AGENT_FOR_DIMENSION["D"]
        got = [item.evidence_id for item in build_reading_pack(rule, pool, agent, LIMIT, None)]
        assert got == EXPECTED_D_PACKS[rule.rule_id], (
            f"{rule.rule_id} is affected by E/F changes: {got} != {EXPECTED_D_PACKS[rule.rule_id]}"
        )

    # 5) The morphological channel itself: only returns E/F candidates, and their internal order is determined
    sample_rule = ef_rules[0]
    channels = evidence_form_channels(sample_rule, pool)
    assert set(channels) == {"supports_high", "supports_low"}, sorted(channels)
    assert channels["supports_high"] and channels["supports_low"], 'shape channel is empty'
    repeat = evidence_form_channels(sample_rule, pool)
    assert [item.evidence_id for item in channels["supports_high"]] == [
        item.evidence_id for item in repeat["supports_high"]
    ], 'support_high channel sequence is unstable'
    assert [item.evidence_id for item in channels["supports_low"]] == [
        item.evidence_id for item in repeat["supports_low"]
    ], 'support_low channel sequence is unstable'

    print(
        f"E/F evidence routing self-test passed: {len(ef_rules)} The reading package of E/F rules has two types of evidence,"
        f"≤ {LIMIT}, repeated calls are consistent one by one; the {len(d_rules)} D rule reading package is exactly the same as before the change."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

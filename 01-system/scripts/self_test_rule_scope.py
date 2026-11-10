'Rule application scope self-test: controlled enumeration, scope filtering, denominator and completeness.\n\nRun:\n    python ".\\scripts\\self_test_rule_scope.py"'

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.models import AgentJudgement, Evidence, Judgement
from paper_eval.agents import AGENTS, build_second_pass_reading_pack, out_of_scope_rule_ids_of
from paper_eval.agent_graph import aggregate_agent_scores
from paper_eval.audit_agent import build_audit_payload
from paper_eval.pdf_parser import detect_heading_section
from paper_eval.reflection import reflect_run
from paper_eval.rules import (
    load_rule_profile,
    normalize_paper_type,
    out_of_scope_rule_ids,
    rule_in_scope,
)
from paper_eval.scorer import aggregate_scores, rule_score_mapping, score_judgements

RULES = Path(__file__).resolve().parents[1] / 'rules' / "rules_v1.yaml"


def judgement(rule_id: str, value: Judgement = Judgement.SATISFIED) -> AgentJudgement:
    return AgentJudgement(
        task_id="RUN-TEST",
        agent_id="self-check",
        indicator_id=rule_id,
        judgement=value,
        evidence_ids=["EV-1"],
        reason='Self-test space occupation reason.',
        confidence=0.8,
        needs_review=False,
        agent_version="self-check",
        model_version="test",
    )


def main() -> int:
    # 1. Controlled enumeration mapping
    assert normalize_paper_type('Review Paper') == 'Summary'
    assert normalize_paper_type('Empirical Research on Management') == 'Evidence'
    assert normalize_paper_type('Qualitative or case study') == 'Qualitative'
    assert normalize_paper_type('Theoretical or conceptual paper') == 'Theory'
    assert normalize_paper_type('Review or opinion paper') == 'Theory'
    assert normalize_paper_type('Empirical research (questionnaire survey + PLS-SEM structural equation)') == 'Evidence'
    assert normalize_paper_type('Mixed methods study') == 'Mix'
    assert normalize_paper_type('This paper adopts a mixed design combining interviews and questionnaires.') == 'Mix'
    assert normalize_paper_type('Completely incomprehensible type', None) is None
    assert normalize_paper_type("", 'Mix') == 'Mix'

    # 2. Scope of application of a single rule
    assert rule_in_scope('General', 'Summary') is True
    assert rule_in_scope('Evidence', 'Evidence') is True
    assert rule_in_scope('Quantitative', 'Evidence') is True          # Vocabulary alias normalization
    assert rule_in_scope('Qualitative', 'Evidence') is False
    assert rule_in_scope('Qualitative', 'Mix') is True          # Mixed method contains qualitative ingredients
    assert rule_in_scope('Evidence', 'Theory') is False
    assert rule_in_scope('is not included in the vocabulary list', 'Theory') is True     # For unknown values, use universal to avoid accidental damage.

    profile = load_rule_profile(RULES)

    # 3. Review papers should not be graded according to empirical standards
    review_scope = out_of_scope_rule_ids(profile, 'Summary')
    assert "D-DATA-SAMPLE-V1" in review_scope
    assert "E-ROBUSTNESS-V1" in review_scope
    assert "A-TOPIC-SIGNIFICANCE-V1" not in review_scope
    # Qualitative special rules should not fall on empirical papers
    empirical_scope = out_of_scope_rule_ids(profile, 'Evidence')
    assert "D-QUAL-GENRE-FIT-V1" in empirical_scope
    assert "D-DATA-SAMPLE-V1" not in empirical_scope
    assert out_of_scope_rule_ids_of(profile.rules, 'Evidence') == empirical_scope
    assert out_of_scope_rule_ids_of(profile.rules, 'Mix') == set()
    runtime_judgements = [judgement(rule.rule_id) for rule in profile.rules]
    graph_result = aggregate_agent_scores(
        {
            "profile": profile,
            "judgements": runtime_judgements,
            "paper_context": {"paper_profile": {"paper_type": 'Evidence'}},
        }
    )
    assert graph_result["trace"][-1]["out_of_scope_rule_count"] == len(empirical_scope)
    assert all(item.out_of_scope for item in graph_result["score_items"] if item.rule_id in empirical_scope)
    audit_payload = build_audit_payload(
        "RUN-TEST",
        profile,
        runtime_judgements,
        graph_result["score_items"],
        [],
        graph_result["summary"],
        {"paper_profile": {"paper_type": 'Evidence'}},
    )
    assert audit_payload["audit_scope"]["excluded_out_of_scope_rules"] == len(empirical_scope)
    assert all(item["rule_id"] != "D-QUAL-GENRE-FIT-V1" for item in audit_payload["scoring_details"])
    reflection_issues = reflect_run(
        runtime_judgements,
        graph_result["score_items"],
        [Evidence("EV-1", "P-TEST", "Methods", 1, (0, 20), "A valid evidence span for the scope self-check.", "h1")],
    )
    assert not any("D-QUAL-GENRE-FIT-V1" in issue.related_ids for issue in reflection_issues)
    # General rules are always in scope
    assert not any(rule.applicable_type == 'General' for rule in profile.rules if rule.rule_id in empirical_scope)

    # 4. End-to-end: Rules outside the scope do not enter the numerator or denominator.
    judgements = [judgement(rule.rule_id) for rule in profile.rules]
    items = score_judgements(profile, judgements, empirical_scope)
    summary = aggregate_scores(profile, items)
    skipped_weight = sum(rule.weight for rule in profile.rules if rule.rule_id in empirical_scope)
    assert summary["out_of_scope_weight"] == skipped_weight, summary["out_of_scope_weight"]
    # The two accumulation sequences are different, and there will be 1e-14 level floating point noise (the construct weight contains amortized decimals such as 2.666667), so == cannot be used
    assert abs(summary["scope_weight"] - (profile.scoring_scale - skipped_weight)) < 1e-9, (
        summary["scope_weight"],
        profile.scoring_scale - skipped_weight,
    )
    assert summary["available_weight"] == summary["scope_weight"], 'Out-of-scope rules should not enter available weights'
    assert summary["scoring_completeness"] == 1.0, summary["scoring_completeness"]
    assert abs(summary["total_score"] - profile.scoring_scale) < 1e-9, summary["total_score"]
    # Out-of-scope rules must not become needs_review noise
    assert all(not item.needs_review for item in judgements if item.indicator_id in empirical_scope)

    # 5. Construct aggregation: Rules for the same construct are first weighted and averaged, and votes are no longer counted separately.
    cluster_keys = {rule.rule_id: rule.cluster_key for rule in profile.rules}
    assert cluster_keys["B-DIALOGUE-ADVANCE-V1"] == "B-DIALOGUE"
    dialogue_members = [rule for rule in profile.rules if rule.cluster == "B-DIALOGUE"]
    assert len(dialogue_members) == 1, dialogue_members   # Three dialogue rules have been merged into one
    # This rule is currently marked as generic and the runtime should respect the rule base rather than the hard-coded expectations in the legacy self-test.
    assert profile.rules_by_id["B-THEORY-INTEGRATION-V1"].applicable_type == 'General'
    assert "B-THEORY-INTEGRATION-V1" not in empirical_scope
    # Quantitative/mixed rules should not be used to grade theoretical papers
    assert "D-MODEL-IDENTIFICATION-V1" in out_of_scope_rule_ids(profile, 'Theory')
    assert "D-MODEL-IDENTIFICATION-V1" not in out_of_scope_rule_ids(profile, 'Evidence')

    # Common level numbers and complete title phrases must hit the chapter routing to avoid all main chapters falling into the summary.
    assert detect_heading_section("3. Research methodology") == "Methods"
    assert detect_heading_section("3.3. Data collection") == "Methods"
    assert detect_heading_section("3.5. Measurement model evaluation") == "Results"
    assert detect_heading_section("3.6.1. Mediation effect") == "Results"
    assert detect_heading_section("4. Discussion and implications") == "Discussion"
    assert detect_heading_section("4.1. Theoretical implications") == "Discussion"
    assert detect_heading_section("5. Conclusion, limitations, and future research directions") == "Conclusion"

    # The second review should give priority to finding new rule-related paragraphs and not be crowded out by earlier titles/generalized paragraphs on the same page.
    f_agent = next(agent for agent in AGENTS if "F" in agent.dimension_ids)
    limit_rule = profile.rules_by_id["F-LIMITATIONS-V1"]
    second_pass_evidence = [
        Evidence("EV-OLD", "P-TEST", "Introduction", 1, (0, 180), "Prior evidence says the study has a large sample and uses an online survey.", "h1"),
        Evidence("EV-LIMIT-HEADING", "P-TEST", "Conclusion", 9, (0, 180), "Conclusion, limitations, and future research directions: this study examines resilience.", "h2"),
        Evidence("EV-LIMIT-SPECIFIC", "P-TEST", "Conclusion", 9, (181, 380), "The study uses data from a single country and industry, creating sampling and geographical limitations; future studies should test other sectors.", "h3"),
        Evidence("EV-GENERIC", "P-TEST", "Discussion", 8, (0, 180), "The paper discusses its theoretical contribution to the literature.", "h4"),
    ]
    second_pass_pack = build_second_pass_reading_pack(
        limit_rule, second_pass_evidence, f_agent, 6, prior_candidate_ids=["EV-OLD"]
    )
    assert "EV-LIMIT-SPECIFIC" in {item.evidence_id for item in second_pass_pack}

    cluster_judged = {"B-DIALOGUE-ADVANCE-V1": Judgement.PARTIAL}   # Three conversations have been merged into one
    mixed = [
        judgement(rule.rule_id, cluster_judged.get(rule.rule_id, Judgement.UNCERTAIN))
        for rule in profile.rules
    ]
    mixed_summary = aggregate_scores(profile, score_judgements(profile, mixed))
    dimension_b = mixed_summary["dimension_scores"]["B"]
    dialogue_members = [r for r in profile.rules if r.cluster == "B-DIALOGUE"]
    b_construct_count = len({c.construct_id for c in profile.constructs if c.dimension_id == "B"})
    assert dimension_b["cluster_count"] == b_construct_count, dimension_b["cluster_count"]
    assert dimension_b["available_cluster_count"] == 1, dimension_b["available_cluster_count"]

    # Each dimension must have Benchmark: Bonus = 6:4. This is the only condition for "dimension full score == declared dimension weight" to be true:
    # scorer is apportioned by the pool (pool × rule weight of the dimension / total weight of the pool). Only when each dimension is 6:4, the apportioned result is equal to the declared weight.
    for dimension in profile.dimensions:
        base_weight = sum(
            r.weight for r in profile.rules
            if r.dimension_id == dimension.dimension_id and r.role == "baseline"
        )
        assert abs(base_weight - dimension.weight * 0.6) < 1e-9, (
            dimension.dimension_id, base_weight, dimension.weight,
        )

    b_rules = [r for r in profile.rules if r.dimension_id == "B"]
    dimension_b_weight = next(d.weight for d in profile.dimensions if d.dimension_id == "B")
    baseline_weight_total = sum(r.weight for r in profile.rules if r.role == "baseline")
    bonus_weight_total = sum(r.weight for r in profile.rules if r.role == "bonus")
    expected_b_max = (
        profile.baseline_pool
        * (sum(r.weight for r in b_rules if r.role == "baseline") / baseline_weight_total)
        + profile.bonus_pool
        * (sum(r.weight for r in b_rules if r.role == "bonus") / bonus_weight_total)
    )
    assert abs(dimension_b["max_score"] - expected_b_max) < 1e-9, dimension_b["max_score"]
    assert abs(dimension_b["max_score"] - dimension_b_weight) < 1e-9, (
        dimension_b["max_score"], dimension_b_weight,
    )
    # Only this construct was rated: the loose caliber is normalized according to the rated constructs, and the dimension score = the full score of the pool where the rule is located × the achievement rate
    dialogue_rule = profile.rules_by_id["B-DIALOGUE-ADVANCE-V1"]
    pool_max = dimension_b["baseline_max" if dialogue_rule.role == "baseline" else "bonus_max"]
    expected_normalized = pool_max * rule_score_mapping(dialogue_rule)["partial"]
    assert abs(dimension_b["normalized_score"] - expected_normalized) < 1e-9, dimension_b["normalized_score"]
    # The sum of the full scores in each dimension must equal the total score (still true after the two pools are divided)
    total_dim_max = sum(item["max_score"] for item in mixed_summary["dimension_scores"].values())
    assert abs(total_dim_max - profile.scoring_scale) < 1e-9, total_dim_max
    assert mixed_summary["scoring_completeness"] < 1.0, 'Unrated constructs only affect the completeness, not the looseness score.'
    assert mixed_summary["na_as_zero_total_score"] < mixed_summary["total_score"]
    # The three dialogue rules originally totaled 11 points (4+3+4). After the merger, the entire construct only occupies the weight of one rule and is no longer dragged down by averaging.
    dialogue = profile.constructs_by_id["B-DIALOGUE"]
    assert abs(dialogue.weight - sum(r.weight for r in dialogue_members)) < 1e-9, dialogue.weight
    # The construct weight must be equalized to the declared dimension weight (the most easily missed link when changing the weight)
    b_construct_total = sum(c.weight for c in profile.constructs if c.dimension_id == "B")
    assert abs(b_construct_total - dimension_b_weight) < 1e-9, (b_construct_total, dimension_b_weight)

    # 6. Non-scoring observation items (weight 0) shall not affect the total score
    all_satisfied = [judgement(rule.rule_id) for rule in profile.rules]
    writing_uncertain = [
        judgement(item.indicator_id, Judgement.UNCERTAIN if item.indicator_id == "C-WRITING-QUALITY-V1" else Judgement.SATISFIED)
        for item in all_satisfied
    ]
    score_a = aggregate_scores(profile, score_judgements(profile, all_satisfied))["total_score"]
    score_b = aggregate_scores(profile, score_judgements(profile, writing_uncertain))["total_score"]
    assert abs(score_a - score_b) < 1e-9, (score_a, score_b)

    print(
        'Applicable scope self-test passed:'
        f"Applicable weight for empirical papers {summary['scope_weight']:.0f}/{profile.scoring_scale:.0f},"
        f"out-of-range {len(empirical_scope)} rules (weight {skipped_weight:.0f}) are not scored;"
        f"The number of constructs and unscored items have been verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

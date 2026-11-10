'Scoring scale self-inspection: two-pool mechanism (base score + bonus points).\n\nContract:\n1. **Full marks will be obtained if the benchmark items are passed**: Most and partial of the benchmark items will receive the same benchmark points; only weak / not_satisfied will deduct points.\n2. **Extra points are only for those who are satisfied**: Mostly gets half, partial gets a quarter, and weak/not_satisfied is 0.\n3. Add the two pools = total score: base pool 70 + bonus pool 30.\n4. Fully partial (qualified in all aspects) must be significantly higher than fully weak (with shortcomings), and lower than fully satisfied.\n5. Loose caliber: Constructs that cannot be evaluated will not be included in the denominator and the total score will not be changed.\n\nRun:\n    python ".\\scripts\\self_test_scoring_scale.py"'

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.models import AgentJudgement, Judgement
from paper_eval.rules import load_rule_profile, validate_rule_profile
from paper_eval.scorer import (
    BASELINE_MAPPING,
    BONUS_MAPPING,
    aggregate_scores,
    rule_score_mapping,
    score_judgements,
)

RULES = Path(__file__).resolve().parents[1] / 'rules' / "rules_v1.yaml"


def judgement(rule_id: str, value: Judgement) -> AgentJudgement:
    return AgentJudgement(
        task_id="RUN-TEST",
        agent_id="self-check",
        indicator_id=rule_id,
        judgement=value,
        evidence_ids=["EV-1"],
        reason='The reason why the ruler is occupied by self-test.',
        confidence=0.8,
        needs_review=False,
        agent_version="self-check",
        model_version="test",
    )


def summary_for(profile, value: Judgement) -> dict:
    judgements = [judgement(rule.rule_id, value) for rule in profile.rules if rule.weight > 0]
    return aggregate_scores(profile, score_judgements(profile, judgements))


def main() -> int:
    profile = load_rule_profile(RULES)
    baseline_rule = next(r for r in profile.rules if r.role == "baseline" and r.weight > 0)
    bonus_rule = next(r for r in profile.rules if r.role == "bonus" and r.weight > 0)

    # 1) The two pools are tied, and the true source of the score is the rule base
    assert profile.baseline_pool == 60 and profile.bonus_pool == 40, (profile.baseline_pool, profile.bonus_pool)
    base_map = rule_score_mapping(baseline_rule)
    bonus_map = rule_score_mapping(bonus_rule)
    assert base_map["mostly"] == 1.0 and base_map["partial"] == 1.0, base_map
    assert base_map["weak"] == 0.5 and base_map["not_satisfied"] == 0.0, base_map
    assert "satisfied" not in base_map, 'The benchmark items should not be satisfied: passing means full marks, not excellence.'
    assert bonus_map["satisfied"] == 1.0 and bonus_map["mostly"] == 0.5, bonus_map
    assert bonus_map["partial"] == 0.25 and bonus_map["weak"] == 0.0, bonus_map
    assert BASELINE_MAPPING[Judgement.PARTIAL] == 1.0 and BONUS_MAPPING[Judgement.SATISFIED] == 1.0
    print(
        f"Two pool scores: benchmark items mostly/partial are {base_map['mostly']:.2f} (passing means full score), weak {base_map['weak']:.2f};"
        f"Bonus points satisfied {bonus_map['satisfied']:.2f} / mostly {bonus_map['mostly']:.2f} / partial {bonus_map['partial']:.2f}"
    )

    # 2) Total score for all gears
    all_partial = summary_for(profile, Judgement.PARTIAL)
    all_mostly = summary_for(profile, Judgement.MOSTLY)
    all_satisfied = summary_for(profile, Judgement.SATISFIED)
    all_weak = summary_for(profile, Judgement.WEAK)
    all_unsatisfied = summary_for(profile, Judgement.NOT_SATISFIED)

    assert abs(all_partial["baseline_score"] - 60.0) < 1e-9, all_partial["baseline_score"]
    assert abs(all_mostly["baseline_score"] - 60.0) < 1e-9, all_mostly["baseline_score"]
    assert abs(all_satisfied["total_score"] - 100.0) < 1e-9, all_satisfied["total_score"]
    assert abs(all_partial["total_score"] - 70.0) < 1e-9, all_partial["total_score"]
    assert abs(all_mostly["total_score"] - 80.0) < 1e-9, all_mostly["total_score"]
    assert abs(all_weak["total_score"] - 30.0) < 1e-9, all_weak["total_score"]
    assert abs(all_unsatisfied["total_score"] - 0.0) < 1e-9, all_unsatisfied["total_score"]
    print(
        f"Total score for all gears: fully satisfied {all_satisfied['total_score']:.1f} / mostly {all_mostly['total_score']:.1f} /"
        f"Full partial (qualified in all aspects) {all_partial['total_score']:.1f} / Full weak {all_weak['total_score']:.1f} /"
        f"full not_satisfied {all_unsatisfied['total_score']:.1f}"
    )

    # 3) Key contract: passing the test means full score - the basic scores for all partial and all most must be the same, and the difference only comes from the bonus points
    assert all_partial["baseline_score"] == all_mostly["baseline_score"] == 60.0
    assert all_partial["bonus_score"] < all_mostly["bonus_score"] < all_satisfied["bonus_score"]
    ladder = [all_unsatisfied["total_score"], all_weak["total_score"], all_partial["total_score"],
              all_mostly["total_score"], all_satisfied["total_score"]]
    assert ladder == sorted(ladder), ladder
    assert all_weak["total_score"] < all_partial["total_score"], 'has shortcomings and must be lower than qualified in all aspects'
    print('Pass is a full score contract: the benchmark score for all partial and most is both 60.0, the difference is only in the bonus points')

    # 4) Loose caliber: constructs that cannot be evaluated will not be included in the denominator and the total score will not be changed.
    for target_id in ("E-RESULTS-ALIGNMENT-V1", "B-DIALOGUE-ADVANCE-V1"):
        judgements = [
            judgement(rule.rule_id, Judgement.UNCERTAIN if rule.rule_id == target_id else Judgement.PARTIAL)
            for rule in profile.rules
            if rule.weight > 0
        ]
        summary = aggregate_scores(profile, score_judgements(profile, judgements))
        assert abs(summary["total_score"] - all_partial["total_score"]) < 1e-9, (target_id, summary["total_score"])
        assert summary["scoring_completeness"] < 1.0
        assert summary["na_as_zero_total_score"] < summary["total_score"], 'The strict lower bound must be lower than the loose caliber'
        assert summary["na_as_partial_total_score"] > summary["na_as_zero_total_score"]
    print('Loose approach: remove any rule (basic item or bonus item) and the total score remains unchanged, only affecting the completeness and strict lower bound')

    # 5) Top publication threshold: no shortcomings + outstanding bonus points
    assert all_partial["top_journal_potential"] == 'low', all_partial["top_journal_potential"]
    assert all_satisfied["top_journal_gate_pass"] is True
    shortfall = summary_for(profile, Judgement.WEAK)
    assert shortfall["top_journal_gate_pass"] is False
    print(
        f"Top publication threshold: fully satisfied and passed={all_satisfied['top_journal_gate_pass']};"
        f"full partial potential={all_partial['top_journal_potential']}; full weak pass={shortfall['top_journal_gate_pass']}"
    )

    # 6) Strict verification: weights are not equal/roles are illegal/two pools are not equal/baseline terms are satisfied/constructs cross roles must be rejected
    def errors_for(mutated) -> list[str]:
        return validate_rule_profile(mutated)

    assert errors_for(profile) == [], errors_for(profile)

    bumped_construct = replace(
        profile,
        constructs=[replace(c, weight=c.weight + 1) if c.construct_id == "A-TOPIC" else c for c in profile.constructs],
    )
    assert any("construct weight mismatch" in e for e in errors_for(bumped_construct)), errors_for(bumped_construct)

    bumped_rule = replace(
        profile,
        rules=[replace(r, weight=r.weight + 1) if r.rule_id == "B-DIALOGUE-ADVANCE-V1" else r for r in profile.rules],
    )
    assert any("rule weight sum" in e for e in errors_for(bumped_rule)), errors_for(bumped_rule)

    orphan_rule = replace(
        profile,
        rules=[replace(r, cluster=None) if r.rule_id == "D-DESIGN-FIT-V1" else r for r in profile.rules],
    )
    assert any("has no construct" in e for e in errors_for(orphan_rule)), errors_for(orphan_rule)

    unknown_construct_rule = replace(
        profile,
        rules=[replace(r, cluster="NOPE") if r.rule_id == "E-RESULTS-ALIGNMENT-V1" else r for r in profile.rules],
    )
    assert any("unknown construct" in e for e in errors_for(unknown_construct_rule))

    empty_construct = replace(profile, constructs=[*profile.constructs, replace(profile.constructs[0], construct_id="EMPTY", weight=0)])
    assert any("has no rules" in e for e in errors_for(empty_construct))

    bad_pool = replace(profile, baseline_pool=80.0)
    assert any('The two pools are not even.' in e for e in errors_for(bad_pool)), errors_for(bad_pool)

    bad_role = replace(profile, rules=[replace(profile.rules[0], role="nope"), *profile.rules[1:]])
    assert any("unknown role" in e for e in errors_for(bad_role)), errors_for(bad_role)

    baseline_with_satisfied = replace(
        profile,
        rules=[
            replace(r, score_mapping={**r.score_mapping, "satisfied": 1.0}) if r.role == "baseline" else r
            for r in profile.rules
        ],
    )
    assert any('should not declare satisfied' in e for e in errors_for(baseline_with_satisfied))

    # Constructs cannot cross roles: F-REPRO-TRANSPARENCY The next two items are both baseline items. Changing one of them to a bonus item should be rejected.
    mixed_construct = replace(
        profile,
        rules=[replace(r, role="bonus") if r.rule_id == "F-DATA-CODE-TRANSPARENCY-V1" else r for r in profile.rules],
    )
    assert any('cannot be returned to the pool' in e for e in errors_for(mixed_construct)), errors_for(mixed_construct)

    print('Passed strict verification of the rule base: unbalanced addition / no attribution / reference to a non-existent construct / empty construct / unfair addition of two pools / illegal role / baseline item band satisfied / construct with cross-role, all will be rejected.')

    # 7) Dimension full score is a rule base attribute and should not change with the paper type (scope of application)
# The full score of
    from paper_eval.rules import out_of_scope_rule_ids

    def dim_maxes(paper_type: str) -> dict[str, float]:
        skipped = out_of_scope_rule_ids(profile, paper_type)
        items = score_judgements(profile, [judgement(r.rule_id, Judgement.PARTIAL) for r in profile.rules], skipped)
        return {k: v["max_score"] for k, v in aggregate_scores(profile, items)["dimension_scores"].items()}

    empirical = dim_maxes('Evidence')
    mixed = dim_maxes('Mix')
    assert empirical == mixed, (empirical, mixed)
    assert abs(sum(mixed.values()) - profile.scoring_scale) < 1e-6, sum(mixed.values())
    print(f"dimension has nothing to do with the paper type, and the total = total score: {' '.join(f'{k}={v:.1f}' for k, v in mixed.items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

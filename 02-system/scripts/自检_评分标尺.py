"""评分标尺自检：两池机制（基准分 + 加分项）。

契约：
1. **基准项合格即满分**：基准项的 mostly 与 partial 拿一样的基准分；只有 weak / not_satisfied 扣分。
2. **加分项只有 satisfied 拿满**：mostly 拿一半，partial 拿四分之一，weak / not_satisfied 为 0。
3. 两池相加 = 总分：基准池 70 + 加分池 30。
4. 全 partial（各方面都合格）必须明显高于全 weak（有短板），且低于全 satisfied。
5. 宽松口径：评不到的构念不进分母，不改变总分。

运行：
    python ".\\scripts\\自检_评分标尺.py"
"""

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

RULES = Path(__file__).resolve().parents[1] / "评价规则库" / "rules_v1.yaml"


def judgement(rule_id: str, value: Judgement) -> AgentJudgement:
    return AgentJudgement(
        task_id="RUN-TEST",
        agent_id="self-check",
        indicator_id=rule_id,
        judgement=value,
        evidence_ids=["EV-1"],
        reason="标尺自检占位理由。",
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

    # 1) 两池加平，且分值真源是规则库
    assert profile.baseline_pool == 60 and profile.bonus_pool == 40, (profile.baseline_pool, profile.bonus_pool)
    base_map = rule_score_mapping(baseline_rule)
    bonus_map = rule_score_mapping(bonus_rule)
    assert base_map["mostly"] == 1.0 and base_map["partial"] == 1.0, base_map
    assert base_map["weak"] == 0.5 and base_map["not_satisfied"] == 0.0, base_map
    assert "satisfied" not in base_map, "基准项不应有 satisfied：合格即满分，不判优秀"
    assert bonus_map["satisfied"] == 1.0 and bonus_map["mostly"] == 0.5, bonus_map
    assert bonus_map["partial"] == 0.25 and bonus_map["weak"] == 0.0, bonus_map
    assert BASELINE_MAPPING[Judgement.PARTIAL] == 1.0 and BONUS_MAPPING[Judgement.SATISFIED] == 1.0
    print(
        f"两池分值：基准项 mostly/partial 均 {base_map['mostly']:.2f}（合格即满分）、weak {base_map['weak']:.2f}；"
        f"加分项 satisfied {bonus_map['satisfied']:.2f} / mostly {bonus_map['mostly']:.2f} / partial {bonus_map['partial']:.2f}"
    )

    # 2) 全档位总分
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
        f"全档位总分：全 satisfied {all_satisfied['total_score']:.1f} / 全 mostly {all_mostly['total_score']:.1f} / "
        f"全 partial（各方面合格）{all_partial['total_score']:.1f} / 全 weak {all_weak['total_score']:.1f} / "
        f"全 not_satisfied {all_unsatisfied['total_score']:.1f}"
    )

    # 3) 关键契约：合格即满分——全 partial 与全 mostly 的基准分必须相同，差异只来自加分项
    assert all_partial["baseline_score"] == all_mostly["baseline_score"] == 60.0
    assert all_partial["bonus_score"] < all_mostly["bonus_score"] < all_satisfied["bonus_score"]
    ladder = [all_unsatisfied["total_score"], all_weak["total_score"], all_partial["total_score"],
              all_mostly["total_score"], all_satisfied["total_score"]]
    assert ladder == sorted(ladder), ladder
    assert all_weak["total_score"] < all_partial["total_score"], "有短板必须低于各方面合格"
    print("合格即满分契约：全 partial 与全 mostly 基准分都是 60.0，差异只出在加分项")

    # 4) 宽松口径：评不到的构念不进分母，不改变总分
    for target_id in ("E-RESULTS-ALIGNMENT-V1", "B-DIALOGUE-ADVANCE-V1"):
        judgements = [
            judgement(rule.rule_id, Judgement.UNCERTAIN if rule.rule_id == target_id else Judgement.PARTIAL)
            for rule in profile.rules
            if rule.weight > 0
        ]
        summary = aggregate_scores(profile, score_judgements(profile, judgements))
        assert abs(summary["total_score"] - all_partial["total_score"]) < 1e-9, (target_id, summary["total_score"])
        assert summary["scoring_completeness"] < 1.0
        assert summary["na_as_zero_total_score"] < summary["total_score"], "严格下界必须低于宽松口径"
        assert summary["na_as_partial_total_score"] > summary["na_as_zero_total_score"]
    print("宽松口径：抽掉任意一条规则（基准项或加分项）总分不变，只影响完整度与严格下界")

    # 5) 顶刊门槛：基准无短板 + 加分项突出
    assert all_partial["top_journal_potential"] == "低", all_partial["top_journal_potential"]
    assert all_satisfied["top_journal_gate_pass"] is True
    shortfall = summary_for(profile, Judgement.WEAK)
    assert shortfall["top_journal_gate_pass"] is False
    print(
        f"顶刊门槛：全 satisfied 通过={all_satisfied['top_journal_gate_pass']}；"
        f"全 partial 潜力={all_partial['top_journal_potential']}；全 weak 通过={shortfall['top_journal_gate_pass']}"
    )

    # 6) 严格校验：权重加不平 / 角色非法 / 两池加不平 / 基准项带 satisfied / 构念跨角色 都必须被拒绝
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
    assert any("两池加不平" in e for e in errors_for(bad_pool)), errors_for(bad_pool)

    bad_role = replace(profile, rules=[replace(profile.rules[0], role="nope"), *profile.rules[1:]])
    assert any("unknown role" in e for e in errors_for(bad_role)), errors_for(bad_role)

    baseline_with_satisfied = replace(
        profile,
        rules=[
            replace(r, score_mapping={**r.score_mapping, "satisfied": 1.0}) if r.role == "baseline" else r
            for r in profile.rules
        ],
    )
    assert any("不应声明 satisfied" in e for e in errors_for(baseline_with_satisfied))

    # 构念不能跨角色：F-REPRO-TRANSPARENCY 下两条都是基准项，把其中一条改成加分项就该被拒
    mixed_construct = replace(
        profile,
        rules=[replace(r, role="bonus") if r.rule_id == "F-DATA-CODE-TRANSPARENCY-V1" else r for r in profile.rules],
    )
    assert any("无法归池" in e for e in errors_for(mixed_construct)), errors_for(mixed_construct)

    print("规则库严格校验通过：加不平 / 无归属 / 引用不存在的构念 / 空构念 / 两池加不平 / 非法角色 / 基准项带 satisfied / 构念跨角色 全部会被拒绝。")

    # 7) 维度满分是规则库属性，不应随论文类型（适用范围）变化
    from paper_eval.rules import out_of_scope_rule_ids

    def dim_maxes(paper_type: str) -> dict[str, float]:
        skipped = out_of_scope_rule_ids(profile, paper_type)
        items = score_judgements(profile, [judgement(r.rule_id, Judgement.PARTIAL) for r in profile.rules], skipped)
        return {k: v["max_score"] for k, v in aggregate_scores(profile, items)["dimension_scores"].items()}

    empirical = dim_maxes("实证")
    mixed = dim_maxes("混合")
    assert empirical == mixed, (empirical, mixed)
    assert abs(sum(mixed.values()) - profile.scoring_scale) < 1e-6, sum(mixed.values())
    print(f"维度满分与论文类型无关，且合计 = 总分：{'  '.join(f'{k}={v:.1f}' for k, v in mixed.items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""规则适用范围自检：受控枚举、作用域过滤、分母与完整度。

运行：
    python ".\scripts\自检_规则适用范围.py"
"""

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

RULES = Path(__file__).resolve().parents[1] / "评价规则库" / "rules_v1.yaml"


def judgement(rule_id: str, value: Judgement = Judgement.SATISFIED) -> AgentJudgement:
    return AgentJudgement(
        task_id="RUN-TEST",
        agent_id="self-check",
        indicator_id=rule_id,
        judgement=value,
        evidence_ids=["EV-1"],
        reason="自检占位理由。",
        confidence=0.8,
        needs_review=False,
        agent_version="self-check",
        model_version="test",
    )


def main() -> int:
    # 1. 受控枚举映射
    assert normalize_paper_type("综述论文") == "综述"
    assert normalize_paper_type("管理学实证研究") == "实证"
    assert normalize_paper_type("质性或案例研究") == "质性"
    assert normalize_paper_type("理论或概念论文") == "理论"
    assert normalize_paper_type("评论或观点论文") == "理论"
    assert normalize_paper_type("实证类研究（问卷调研 + PLS-SEM 结构方程）") == "实证"
    assert normalize_paper_type("混合方法研究") == "混合"
    assert normalize_paper_type("本论文采用访谈与问卷结合的混合设计") == "混合"
    assert normalize_paper_type("完全看不懂的类型", None) is None
    assert normalize_paper_type("", "混合") == "混合"

    # 2. 单条规则适用范围
    assert rule_in_scope("通用", "综述") is True
    assert rule_in_scope("实证", "实证") is True
    assert rule_in_scope("定量", "实证") is True          # 词表别名归一
    assert rule_in_scope("质性", "实证") is False
    assert rule_in_scope("质性", "混合") is True          # 混合方法含质性成分
    assert rule_in_scope("实证", "理论") is False
    assert rule_in_scope("未收录词表", "理论") is True     # 未知取值按通用，避免误伤

    profile = load_rule_profile(RULES)

    # 3. 综述论文不该被按实证标准评分
    review_scope = out_of_scope_rule_ids(profile, "综述")
    assert "D-DATA-SAMPLE-V1" in review_scope
    assert "E-ROBUSTNESS-V1" in review_scope
    assert "A-TOPIC-SIGNIFICANCE-V1" not in review_scope
    # 质性专用规则不该落在实证论文上
    empirical_scope = out_of_scope_rule_ids(profile, "实证")
    assert "D-QUAL-GENRE-FIT-V1" in empirical_scope
    assert "D-DATA-SAMPLE-V1" not in empirical_scope
    assert out_of_scope_rule_ids_of(profile.rules, "实证") == empirical_scope
    assert out_of_scope_rule_ids_of(profile.rules, "混合") == set()
    runtime_judgements = [judgement(rule.rule_id) for rule in profile.rules]
    graph_result = aggregate_agent_scores(
        {
            "profile": profile,
            "judgements": runtime_judgements,
            "paper_context": {"paper_profile": {"paper_type": "实证"}},
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
        {"paper_profile": {"paper_type": "实证"}},
    )
    assert audit_payload["audit_scope"]["excluded_out_of_scope_rules"] == len(empirical_scope)
    assert all(item["rule_id"] != "D-QUAL-GENRE-FIT-V1" for item in audit_payload["scoring_details"])
    reflection_issues = reflect_run(
        runtime_judgements,
        graph_result["score_items"],
        [Evidence("EV-1", "P-TEST", "Methods", 1, (0, 20), "A valid evidence span for the scope self-check.", "h1")],
    )
    assert not any("D-QUAL-GENRE-FIT-V1" in issue.related_ids for issue in reflection_issues)
    # 通用规则永远在范围内
    assert not any(rule.applicable_type == "通用" for rule in profile.rules if rule.rule_id in empirical_scope)

    # 4. 端到端：范围外规则不进分子也不进分母
    judgements = [judgement(rule.rule_id) for rule in profile.rules]
    items = score_judgements(profile, judgements, empirical_scope)
    summary = aggregate_scores(profile, items)
    skipped_weight = sum(rule.weight for rule in profile.rules if rule.rule_id in empirical_scope)
    assert summary["out_of_scope_weight"] == skipped_weight, summary["out_of_scope_weight"]
    # 两次累加顺序不同，会有 1e-14 级浮点噪声（构念权重含 2.666667 这类摊分小数），不能用 ==
    assert abs(summary["scope_weight"] - (profile.scoring_scale - skipped_weight)) < 1e-9, (
        summary["scope_weight"],
        profile.scoring_scale - skipped_weight,
    )
    assert summary["available_weight"] == summary["scope_weight"], "范围外规则不应进入可用权重"
    assert summary["scoring_completeness"] == 1.0, summary["scoring_completeness"]
    assert abs(summary["total_score"] - profile.scoring_scale) < 1e-9, summary["total_score"]
    # 范围外规则不得变成 needs_review 噪音
    assert all(not item.needs_review for item in judgements if item.indicator_id in empirical_scope)

    # 5. 构念聚合：同一构念的规则先取加权平均，不再各自计票
    cluster_keys = {rule.rule_id: rule.cluster_key for rule in profile.rules}
    assert cluster_keys["B-DIALOGUE-ADVANCE-V1"] == "B-DIALOGUE"
    dialogue_members = [rule for rule in profile.rules if rule.cluster == "B-DIALOGUE"]
    assert len(dialogue_members) == 1, dialogue_members   # 三条对话类规则已合并为一条
    # 该规则当前标为通用，运行时应尊重规则库而非旧版自检中的硬编码预期。
    assert profile.rules_by_id["B-THEORY-INTEGRATION-V1"].applicable_type == "通用"
    assert "B-THEORY-INTEGRATION-V1" not in empirical_scope
    # 定量/混合类规则不该给理论论文打分
    assert "D-MODEL-IDENTIFICATION-V1" in out_of_scope_rule_ids(profile, "理论")
    assert "D-MODEL-IDENTIFICATION-V1" not in out_of_scope_rule_ids(profile, "实证")

    # 常见层级编号和完整标题短语必须命中章节路由，避免主体章节全部落入综述。
    assert detect_heading_section("3. Research methodology") == "Methods"
    assert detect_heading_section("3.3. Data collection") == "Methods"
    assert detect_heading_section("3.5. Measurement model evaluation") == "Results"
    assert detect_heading_section("3.6.1. Mediation effect") == "Results"
    assert detect_heading_section("4. Discussion and implications") == "Discussion"
    assert detect_heading_section("4.1. Theoretical implications") == "Discussion"
    assert detect_heading_section("5. Conclusion, limitations, and future research directions") == "Conclusion"

    # 二次复判应优先找到新的规则相关段落，不能被同页较早的标题/泛化段落挤掉。
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

    cluster_judged = {"B-DIALOGUE-ADVANCE-V1": Judgement.PARTIAL}   # 对话类三条已合并为一条
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

    # 每个维度都必须 基准:加分 = 6:4。这是「维度满分 == 声明的维度权重」成立的唯一条件：
    # scorer 按池摊分（pool × 该维规则权重 / 池总权重），只有每维都 6:4 时摊分结果才等于声明权重。
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
    # 只有这一个构念被评到：宽松口径按已评构念归一，维度分 = 该规则所在池的满分 × 达成率
    dialogue_rule = profile.rules_by_id["B-DIALOGUE-ADVANCE-V1"]
    pool_max = dimension_b["baseline_max" if dialogue_rule.role == "baseline" else "bonus_max"]
    expected_normalized = pool_max * rule_score_mapping(dialogue_rule)["partial"]
    assert abs(dimension_b["normalized_score"] - expected_normalized) < 1e-9, dimension_b["normalized_score"]
    # 各维度满分相加必须等于总分（两池摊分后仍然成立）
    total_dim_max = sum(item["max_score"] for item in mixed_summary["dimension_scores"].values())
    assert abs(total_dim_max - profile.scoring_scale) < 1e-9, total_dim_max
    assert mixed_summary["scoring_completeness"] < 1.0, "未评到的构念只影响完整度，不影响宽松分"
    assert mixed_summary["na_as_zero_total_score"] < mixed_summary["total_score"]
    # 对话类三条原合计 11 分（4+3+4），合并后整个构念只占一条规则的权重，不再被取平均拖低
    dialogue = profile.constructs_by_id["B-DIALOGUE"]
    assert abs(dialogue.weight - sum(r.weight for r in dialogue_members)) < 1e-9, dialogue.weight
    # 构念权重必须加平到声明的维度权重（改权重时最容易漏的一环）
    b_construct_total = sum(c.weight for c in profile.constructs if c.dimension_id == "B")
    assert abs(b_construct_total - dimension_b_weight) < 1e-9, (b_construct_total, dimension_b_weight)

    # 6. 不计分观察项（weight 0）不得影响总分
    all_satisfied = [judgement(rule.rule_id) for rule in profile.rules]
    writing_uncertain = [
        judgement(item.indicator_id, Judgement.UNCERTAIN if item.indicator_id == "C-WRITING-QUALITY-V1" else Judgement.SATISFIED)
        for item in all_satisfied
    ]
    score_a = aggregate_scores(profile, score_judgements(profile, all_satisfied))["total_score"]
    score_b = aggregate_scores(profile, score_judgements(profile, writing_uncertain))["total_score"]
    assert abs(score_a - score_b) < 1e-9, (score_a, score_b)

    print(
        "适用范围自检通过："
        f"实证论文适用权重 {summary['scope_weight']:.0f}/{profile.scoring_scale:.0f}，"
        f"范围外 {len(empirical_scope)} 条规则（权重 {skipped_weight:.0f}）不计分；"
        f"构念数与不计分项均已验证。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E/F 证据路由自检：两条"证据形态"通道必须都有确定性的入选路径。

背景：E/F 的阅读包以前只由 `score_results_boundary_hits` 排序决定，而那套词全是
"论文把结果写好了"的信号（results indicate / coefficient / robustness check /
hypothesis supported）。按它排序，进包的一定是最像"好结果"的段落；而"没报数值""只给方向"
"检验不显著""数据不可得""结论强过证据"这些能支撑低档的段落只能靠排序尾部的偶然进包。
改法见 `literature_eval/agents.py` 的 `evidence_form_channels` / `guarantee_evidence_forms`。

本自检钉四件事：
  1) E/F 每条规则的包**非空**、**不超 limit**，并且"支持高档"与"支持低档"两类段落都出现；
  2) 同等输入重复调用，包逐条相同（证据选择确定性，不靠 dict/set 顺序）；
  3) 非 E/F 维度（D）的包逐条等于改动前的期望序列 —— 改动不外溢；
  4) 模型真正看到的最终包（`build_dimension_reading_packs`）同样两类证据齐备。

第 3 条的期望值是改动前跑 `python "scripts/自检_EF证据路由.py" --dump-d` 记下来的。
如果 `评价规则库/rules_v1.yaml` 里 D 维的 retrieval_terms / evidence_location 改了，
这个常量需要重新生成（这是刻意的：它就该在规则库变动时失败提醒）。

运行：
    python ".\scripts\自检_EF证据路由.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from literature_eval.agents import (  # noqa: E402
    AGENTS,
    build_dimension_reading_packs,
    build_reading_pack,
    evidence_form_channels,
)
from literature_eval.llm import AgentRuntimeConfig  # noqa: E402
from literature_eval.models import Evidence  # noqa: E402
from literature_eval.rules import load_rule_profile  # noqa: E402

RULES = ROOT / "评价规则库" / "rules_v1.yaml"
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


# 支持**高档**的段落：把结果/数值/可得性/边界交代清楚的句子。
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

# 支持**低档**的段落：作者自己写出"没做到"的地方（未报告 / 不显著 / 不可得 / 结论强过证据）。
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

# D 维回归锚点用的方法段：改动不该影响 D 的取证据。
METHOD_ITEMS = [
    evidence(
        f"EV-METHOD-{index}",
        4 + index,
        "Methods",
        "研究程序与样本说明段。We collected survey data from respondents in a multi-wave panel; the "
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
        "背景铺垫段。The remaining sections of this manuscript proceed as follows and introduce each part "
        "of the argument in turn for the reader, without reporting any estimate or procedure. " * 2,
    )
    for index in range(4)
]


def build_pool() -> list[Evidence]:
    return [*FILLER, *METHOD_ITEMS, *HIGH_SUPPORT, *LOW_SUPPORT]


# 改动前的 D 维期望序列（limit=10，池子固定）。生成方式见模块 docstring。
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
        print("当前 D 维阅读包（改动前应为基准，改动后必须一致）：")
        return dump_d(profile)

    pool = build_pool()
    ef_rules = [rule for rule in profile.rules if rule.dimension_id in {"E", "F"}]
    assert ef_rules, "规则库里没有 E/F 规则，自检前提不成立"

    # 1) E/F：非空、不超 limit、两类证据都出现
    for rule in ef_rules:
        agent = AGENT_FOR_DIMENSION[rule.dimension_id]
        pack = build_reading_pack(rule, pool, agent, LIMIT, None)
        ids = [item.evidence_id for item in pack]
        assert pack, f"{rule.rule_id} 的阅读包为空"
        assert len(pack) <= LIMIT, f"{rule.rule_id} 超发：{len(pack)}"
        assert len(set(ids)) == len(ids), f"{rule.rule_id} 出现重复证据：{ids}"
        high_present = HIGH_SUPPORT_IDS & set(ids)
        low_present = LOW_SUPPORT_IDS & set(ids)
        assert high_present, f"{rule.rule_id} 的包里没有支持高档的段落：{ids}"
        assert low_present, f"{rule.rule_id} 的包里没有支持低档的段落：{ids}"

    # 2) 确定性：同输入两次调用逐条相同
    for rule in ef_rules:
        agent = AGENT_FOR_DIMENSION[rule.dimension_id]
        first = [item.evidence_id for item in build_reading_pack(rule, pool, agent, LIMIT, None)]
        second = [item.evidence_id for item in build_reading_pack(rule, pool, agent, LIMIT, None)]
        assert first == second, f"{rule.rule_id} 两次调用结果不同：{first} vs {second}"

    # 3) 模型真正看到的最终包（检索层开、重排关）也要两类齐备，且条数不缩水
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
            f"{rule.rule_id} 最终包条数与旧启发式不一致：{len(final_ids)} vs {len(legacy)}"
        )
        assert HIGH_SUPPORT_IDS & set(final_ids), f"{rule.rule_id} 最终包丢了支持高档的段落：{final_ids}"
        assert LOW_SUPPORT_IDS & set(final_ids), f"{rule.rule_id} 最终包丢了支持低档的段落：{final_ids}"
        again = build_dimension_reading_packs([rule], pool, agent, LIMIT, None, config=config)[rule.rule_id]
        assert final_ids == [item.evidence_id for item in again], f"{rule.rule_id} 最终包不可复现"

    # 4) 非 E/F 维度（D）逐条等于改动前的期望序列 —— 改动不外溢
    d_rules = [rule for rule in profile.rules if rule.dimension_id == "D"]
    assert EXPECTED_D_PACKS, "EXPECTED_D_PACKS 为空，请先跑 --dump-d 生成"
    assert {rule.rule_id for rule in d_rules} == set(EXPECTED_D_PACKS), (
        "D 维规则集合变了，期望序列需要重新生成",
        sorted(rule.rule_id for rule in d_rules),
    )
    for rule in d_rules:
        agent = AGENT_FOR_DIMENSION["D"]
        got = [item.evidence_id for item in build_reading_pack(rule, pool, agent, LIMIT, None)]
        assert got == EXPECTED_D_PACKS[rule.rule_id], (
            f"{rule.rule_id} 的阅读包被 E/F 改动影响：{got} != {EXPECTED_D_PACKS[rule.rule_id]}"
        )

    # 5) 形态通道自身：只返回 E/F 的候选，且各自内部顺序确定
    sample_rule = ef_rules[0]
    channels = evidence_form_channels(sample_rule, pool)
    assert set(channels) == {"supports_high", "supports_low"}, sorted(channels)
    assert channels["supports_high"] and channels["supports_low"], "形态通道为空"
    repeat = evidence_form_channels(sample_rule, pool)
    assert [item.evidence_id for item in channels["supports_high"]] == [
        item.evidence_id for item in repeat["supports_high"]
    ], "support_high 通道顺序不稳定"
    assert [item.evidence_id for item in channels["supports_low"]] == [
        item.evidence_id for item in repeat["supports_low"]
    ], "support_low 通道顺序不稳定"

    print(
        f"E/F 证据路由自检通过：{len(ef_rules)} 条 E/F 规则的阅读包两类证据齐备、"
        f"条数 ≤ {LIMIT}、重复调用逐条一致；{len(d_rules)} 条 D 规则阅读包与改动前完全相同。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""证据检索接线自检：验证 select_evidence 接进阅读包后，开关两态的行为都正确。

这个自检管的是「接线」而不是「检索质量」：
- 开关关闭时，必须与旧 build_reading_pack **逐条相同**（否则「关掉新链路 = 回到旧行为」只是口头承诺）
- 开关打开时，召回必须把规则相关的段落选进包，并且 build_dimension_reading_packs 也吃到了 config
- 外部文献（EXT-*）仍然只插给指定的那几条规则

跑法：
    python ".\scripts\自检_证据检索接线.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from literature_eval.agents import (  # noqa: E402
    METHOD_QUALITY_AGENT,
    build_dimension_reading_packs,
    build_reading_pack,
)
from literature_eval.evidence_retrieval import (  # noqa: E402
    build_evidence_vectors,
    build_query_terms,
    recall_evidence,
    select_evidence,
)
from literature_eval.external_retrieval import external_evidence_blocks  # noqa: E402
from literature_eval.llm import AgentRuntimeConfig  # noqa: E402
from literature_eval.models import Evidence  # noqa: E402
from literature_eval.rules import load_rule_profile  # noqa: E402

RULES = ROOT / "评价规则库" / "rules_v1.yaml"
TARGET_RULE = "D-MEASUREMENT-VALIDITY-V1"
# 每段构造证据应该被哪条规则召回。检索词必须让「对的规则」捞到「对的段落」，
# 所以这里按规则分别验证，而不是拿一段证据去测所有规则。
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
    """构造一小组正文证据：两条与 D-MEASUREMENT 直接相关，其余是噪音。"""
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
    # 噪音：与 D 维无关、且不落在 Methods 章节
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
    """不读 .env 的独立配置：AgentRuntimeConfig 全是带默认值的字段，不触发任何网络调用。"""
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

    # 1. 开关关闭：与旧 build_reading_pack 逐条相同
    legacy = build_reading_pack(rule, evidences, METHOD_QUALITY_AGENT, 10, None)
    switched_off = select_evidence(rule, evidences, METHOD_QUALITY_AGENT, 10, None, off)
    assert [item.evidence_id for item in legacy] == [item.evidence_id for item in switched_off], (
        "开关关闭时 select_evidence 必须与 build_reading_pack 逐条一致",
        [item.evidence_id for item in legacy],
        [item.evidence_id for item in switched_off],
    )
    assert switched_off, "开关关闭时也不该返回空包"

    # 2. config 为 None 时同样退回旧行为（调用方漏传 config 不能改变行为）
    assert [
        item.evidence_id
        for item in select_evidence(rule, evidences, METHOD_QUALITY_AGENT, 10, None, None)
    ] == [item.evidence_id for item in legacy], "config=None 时应退回 build_reading_pack"

    # 2b. 带 paper_context 时同样必须逐条一致。
    # paper_context 里带证据路由表与 facet 索引，是 build_reading_pack 真正的取证据来源；
    # 接线时漏传或传错位置参数，这里就会露出来（且不会在 paper_context=None 的用例里暴露）。
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
    ], "带 paper_context 时开关关闭也必须逐条等于 build_reading_pack"
    on_ctx = select_evidence(rule, evidences, METHOD_QUALITY_AGENT, 10, paper_context, on_no_rerank)
    assert len(on_ctx) == len(legacy_ctx), (
        f"带 paper_context 时条数缩水：新 {len(on_ctx)} vs 旧 {len(legacy_ctx)}"
    )

    # 3. 召回本身能把「对的段落」交给「对的规则」
    vectors = build_evidence_vectors(evidences)
    evidences_by_id = {item.evidence_id: item for item in evidences}
    recalled_by_rule: dict[str, list[str]] = {}
    for rule_id, expected in RECALL_EXPECTATIONS.items():
        target = profile.rules_by_id[rule_id]
        terms = build_query_terms(target, None)
        assert terms, f"{rule_id} 的检索词为空，无法召回（retrieval_terms 是否漏填？）"
        ranked = recall_evidence(terms, vectors, evidences_by_id, 30)
        ids = [item.evidence_id for _score, item in ranked]
        recalled_by_rule[rule_id] = ids
        assert expected in ids, f"{rule_id} 没召回到 {expected}，实际命中：{ids}"
        assert not ids or ids[0] != "EV-NOISE-00", f"{rule_id} 把噪音排到了首位：{ids[:5]}"

    # 4. 开关打开（重排关闭，避免真调模型）：相关段落进包
    on_pack = select_evidence(rule, evidences, METHOD_QUALITY_AGENT, 10, None, on_no_rerank)
    on_ids = [item.evidence_id for item in on_pack]
    assert "EV-VALID" in on_ids, f"检索层开时效度证据没进包：{on_ids}"
    # 检索层不能反而「少给证据」：正文够多时，条数必须与旧启发式持平。
    # 召回只命中 2 条就把包缩到 2 条，是把证据覆盖做小，与本次改造的目标相反。
    assert len(on_pack) == len(legacy), (
        f"检索层开时只给了 {len(on_pack)} 条，旧启发式给 {len(legacy)} 条：召回命中不足时必须回填",
        on_ids,
    )

    # 5. build_dimension_reading_packs 必须把 config 透下去（这是接线是否真的生效的证据）
    dim_off = build_dimension_reading_packs([rule], evidences, METHOD_QUALITY_AGENT, 10, None, config=None)
    dim_on = build_dimension_reading_packs(
        [rule], evidences, METHOD_QUALITY_AGENT, 10, None, config=on_no_rerank
    )
    assert [
        item.evidence_id for item in dim_off[rule.rule_id]
    ] == [item.evidence_id for item in legacy], "build_dimension_reading_packs 在 config=None 时应等于旧行为"
    assert "EV-VALID" in [item.evidence_id for item in dim_on[rule.rule_id]], (
        "build_dimension_reading_packs 没有把 config 透给 select_evidence",
        [item.evidence_id for item in dim_on[rule.rule_id]],
    )

    # 6. 外部文献仍只插给指定规则
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
    assert external, "构造外部文献块失败，自检前提不成立"
    with_external = [*evidences, *external]
    designated = {rule.rule_id}
    packs = build_dimension_reading_packs(
        [rule], with_external, METHOD_QUALITY_AGENT, 10, None, external_rule_ids=designated, config=on_no_rerank
    )
    pack_ids = [item.evidence_id for item in packs[rule.rule_id]]
    assert any(item.startswith("EXT-") for item in pack_ids), "指定规则没拿到外部文献块"
    other = profile.rules_by_id["A-TOPIC-SIGNIFICANCE-V1"]
    packs_other = build_dimension_reading_packs(
        [other], with_external, METHOD_QUALITY_AGENT, 10, None, external_rule_ids=designated, config=on_no_rerank
    )
    assert not any(
        item.startswith("EXT-") for item in [ev.evidence_id for ev in packs_other[other.rule_id]]
    ), "未指定的规则不该拿到外部文献块"

    print(
        "证据检索接线自检通过："
        f"开关关闭与旧行为逐条一致（{len(legacy)} 条）；"
        + "；".join(
            f"{rule_id} 召回 {expected}（第 {recalled_by_rule[rule_id].index(expected) + 1} 位）"
            for rule_id, expected in RECALL_EXPECTATIONS.items()
        )
        + f"；开关打开仍给 {len(on_pack)} 条不缩水；"
        "build_dimension_reading_packs 已透传 config；外部文献仅插给指定规则。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

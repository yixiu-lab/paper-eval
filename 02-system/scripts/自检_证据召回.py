"""证据检索层自检：词袋召回排序、LLM 重排顺序、三条降级路径、开关关闭不改行为。

为什么必须覆盖降级：重排是一次会失败的模型调用（超时、坏 JSON、模型胡说 id）。
评分是一条链，任何一条规则的重排失败都不能让整篇论文的评分崩掉，所以「失败退回召回顺序」
是契约而不是尽力而为，必须有可运行的证据。

全程不调真实 API：重排调用点 `_call_rerank_llm` 被 monkeypatch 成假响应。

运行：
    python ".\\scripts\\自检_证据召回.py"
"""

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
from paper_eval.llm import parse_json_content  # noqa: F401  重排的坏响应就是在这一层被救回来 / 抛出去
from paper_eval.models import Evidence
from paper_eval.rules import IndicatorRule, load_rule_profile

RULES = Path(__file__).resolve().parents[1] / "评价规则库" / "rules_v1.yaml"
AGENT_FOR_DIMENSION = {dimension: agent for agent in AGENTS for dimension in agent.dimension_ids}


@dataclass
class FakeConfig:
    """最小配置替身：只带检索层读的字段，避免自检依赖真实 YAML/环境变量。"""

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
    """10 条中英混合证据。EV-05 讲样本与数据收集，EV-06 讲识别策略与内生性，其余是噪音。"""
    return [
        evidence(
            "EV-01", 1, "Introduction",
            "本文研究数字化转型对企业创新绩效的影响。This study examines how digital transformation "
            "affects innovation performance in Chinese manufacturing firms, a topic widely discussed.",
        ),
        evidence(
            "EV-02", 1, "Theory",
            "动态能力理论认为企业需要重构资源基础。The dynamic capability view argues that firms must "
            "reconfigure their resource base to cope with environmental change.",
        ),
        evidence(
            "EV-03", 2, "Theory",
            "本文提出三个理论命题。We develop three propositions linking organizational slack to "
            "exploratory learning, building on the attention-based view of the firm.",
        ),
        evidence(
            "EV-04", 3, "Literature Review",
            "既有文献主要关注大型企业。Prior work has focused on large incumbents, leaving small and "
            "medium-sized enterprises largely unexplored in the digital context.",
        ),
        evidence(
            "EV-05", 6, "Methods",
            "数据来源与样本：我们收集了 2015-2020 年沪深 A 股制造业上市公司的面板数据，"
            "共 1,846 个企业-年度观测值。Sample and data collection proceeded in three steps: we drew the "
            "sample from the CSMAR database, dropped firms with missing key variables, and matched the "
            "survey responses with archival financial data. The final sample covers 1,846 firm-year "
            "observations. Response rate was 62 percent and non-response bias tests showed no systematic "
            "difference between respondents and non-respondents.",
        ),
        evidence(
            "EV-06", 7, "Methods",
            "模型设定与识别策略：为缓解内生性问题，我们采用工具变量法，以同行业其他企业的数字化"
            "程度均值作为工具变量，并报告了第一阶段 F 统计量。Identification strategy: to address "
            "endogeneity and selection bias we instrument digital transformation with the industry-year "
            "mean excluding the focal firm. The first-stage F statistic is 34.7, above the weak-instrument "
            "threshold, and the exclusion restriction is argued to hold because peer adoption shocks are "
            "plausibly unrelated to a focal firm's idiosyncratic innovation shock.",
        ),
        evidence(
            "EV-07", 8, "Results",
            "主回归结果显示系数显著为正。The baseline regression shows a positive and significant "
            "coefficient of 0.18 on digital transformation, robust to firm and year fixed effects.",
        ),
        evidence(
            "EV-08", 9, "Discussion",
            "本研究对实践有启示。Our findings suggest managers should invest in digital infrastructure, "
            "and policymakers may consider supporting SME digital adoption programmes.",
        ),
        evidence(
            "EV-09", 10, "Conclusion",
            "本文存在局限。This paper has limitations: the sample covers only listed manufacturing firms, "
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
    """旧实现（启发式阅读包）的结果，用来当补齐来源与条数基准。"""
    return build_reading_pack(rule, evidences, AGENT_FOR_DIMENSION[rule.dimension_id], limit, None)


def test_size_parity() -> None:
    """0. 条数不缩水：召回命中很少时，开关打开的包也不能比旧实现小。

    复现父 agent 报的缺陷：6 条正文 + 24 条 Appendix 噪音、规则 D-MEASUREMENT-VALIDITY-V1、
    evidence_recall_top_k=30、重排关闭——只有 2 条段落命中 validity/reliability，
    于是旧实现给 10 条、检索层只给 2 条。这是反向改动，必须钉死。
    """
    profile = load_rule_profile(RULES)
    rule = profile.rules_by_id["D-MEASUREMENT-VALIDITY-V1"]
    evidences = [
        evidence(
            f"EV-BODY-{index}", 5 + index, "Methods",
            f"方法段 {index}：我们说明构念的 measurement validity 与 reliability 检验，"
            f"报告 Cronbach's alpha 与验证性因子分析结果。 " * 3,
        )
        for index in range(6)
    ] + [
        evidence(f"EV-APP-{index}", 20 + index, "Appendix", f"附录噪音段落 {index}：变量定义表与补充说明。 " * 3)
        for index in range(24)
    ]

    legacy = legacy_pack(rule, evidences, 10)
    assert len(legacy) == 10, f"旧实现本该填满 limit：{len(legacy)}"

    for rerank_on in (False, True):
        on = FakeConfig(evidence_retrieval_enabled=True, evidence_rerank_enabled=rerank_on)
        if rerank_on:
            # 重排开着但模型只挑 1 条：这正是缺陷现场（模型少选 + 召回命中少）。
            original = evidence_retrieval._call_rerank_llm
            evidence_retrieval._call_rerank_llm = lambda _payload, _config: {
                "selected": [{"evidence_id": "EV-BODY-0", "reason": "报告了信度系数"}]
            }
            try:
                pack = select_evidence(rule, evidences, AGENT_FOR_DIMENSION["D"], 10, None, on)
            finally:
                evidence_retrieval._call_rerank_llm = original
        else:
            pack = select_evidence(rule, evidences, AGENT_FOR_DIMENSION["D"], 10, None, on)
        got = [item.evidence_id for item in pack]
        assert len(got) == len(legacy), f"检索层（重排={rerank_on}）条数缩水：{len(got)} != {len(legacy)}"
        assert len(set(got)) == len(got), f"补齐出现重复：{got}"
    assert "EV-BODY-0" in got, got

    # 正文确实不足 limit 时才允许少于 limit：3 条证据、limit=10 时两边都只能给 3 条。
    tiny = evidences[:3]
    on = FakeConfig(evidence_retrieval_enabled=True, evidence_rerank_enabled=False)
    tiny_legacy = legacy_pack(rule, tiny, 10)
    tiny_on = select_evidence(rule, tiny, AGENT_FOR_DIMENSION["D"], 10, None, on)
    assert len(tiny_on) == len(tiny_legacy) == 3, (len(tiny_on), len(tiny_legacy))


def test_recall_ordering() -> None:
    """1. 召回：讲样本 / 讲识别策略的两条必须排在最前，参考文献块必须进不了池子。"""
    evidences = build_corpus()
    assert "EV-10" not in build_evidence_vectors(evidences), "参考文献证据不该进检索池"

    # 数据与样本类检索词
    sample_terms = ["sample", "data collection", "respondents", "non-response bias"]
    ranked = recall_rank(sample_terms, evidences, 30)
    assert ranked, "样本类检索词召回为空"
    assert ranked[0] == "EV-05", f"样本类召回首位不是 EV-05：{ranked}"
    # 噪音里只有一处擦边（EV-09 提了 sample、EV-06 提了 bias），必须排在 EV-05 之后。
    assert ranked.index("EV-05") < ranked.index("EV-09"), ranked
    assert ranked.index("EV-05") < ranked.index("EV-06"), ranked

    # 识别策略 / 内生性类检索词
    identification_terms = ["identification strategy", "endogeneity", "instrumental variable", "selection bias"]
    ranked = recall_rank(identification_terms, evidences, 30)
    assert ranked, "识别策略类检索词召回为空"
    assert ranked[0] == "EV-06", f"识别策略类召回首位不是 EV-06：{ranked}"
    assert ranked.index("EV-06") < ranked.index("EV-05"), ranked

    # top_k 截断与 0 分丢弃
    ranked_tiny = recall_rank(sample_terms, evidences, 2)
    assert len(ranked_tiny) == 2, ranked_tiny
    assert not recall_rank(["zzyzx"], evidences, 30), "完全不命中的检索词不该召回任何证据"

    # 检索池排除外部文献（EXT-*）
    ext = evidence("EXT-1", 0, "外部文献（OpenAlex 检索）", "sample data collection endogeneity")
    ext.source_type = "external"
    assert "EXT-1" not in build_evidence_vectors(evidences + [ext]), "外部文献不该进通用检索池"


def test_query_terms() -> None:
    """2. 检索词：优先 rule.retrieval_terms；为空时回退 rule_id + evidence_location 的英文 token。"""
    profile = load_rule_profile(RULES)
    rule = profile.rules_by_id["D-DATA-SAMPLE-V1"]
    terms = build_query_terms(rule)
    assert terms, "回退检索词为空"
    assert all(term.strip() for term in terms), terms
    assert "d" not in terms and "v1" not in terms, f"规则 id 的维度前缀/版本号应被过滤：{terms}"

    # retrieval_terms 填充后必须优先，不再看 rule_id
    fake = IndicatorRule(
        rule_id="D-DATA-SAMPLE-V1",
        version="1.0",
        dimension_id="D",
        dimension_name="研究设计与方法质量",
        indicator_name="数据来源与样本合理",
        definition="中文定义不该进检索词",
        required_evidence="",
        evidence_location=["Methods"],
        observation_points=[],
        judgement_values={"partial": "样本说明不完整"},
        score_mapping={"partial": 0.6},
        weight=1.0,
        applicable_type="通用",
        source_reference=[],
        status="active",
        retrieval_terms=["sample", "data collection", "response rate"],
    )
    assert build_query_terms(fake) == ["sample", "data collection", "response rate"], build_query_terms(fake)


def test_rerank_order_and_subset() -> None:
    """3. 重排：按模型给的 evidence_id 顺序与子集返回，并且只用截断后的文本喂模型。"""
    evidences = build_corpus()
    config = FakeConfig()
    rule = load_rule_profile(RULES).rules_by_id["D-MODEL-IDENTIFICATION-V1"]

    def fake_llm(payload, _config):
        # 假响应：故意挑「召回顺序里靠后的两条」并倒序给出，验证返回顺序确实跟模型走，不跟召回走。
        assert payload["budget"] == 3, payload["budget"]
        assert payload["judgement_values"], "档位锚点必须进 payload"
        for candidate in payload["candidates"]:
            assert len(candidate["text"]) <= 240, len(candidate["text"])
        return {
            "selected": [
                {"evidence_id": "EV-06", "reason": "给出工具变量与第一阶段 F 值，直接覆盖内生性处理"},
                {"evidence_id": "EV-09", "reason": "交代样本边界与外部效度限制"},
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
    assert got[:2] == ["EV-06", "EV-09"], f"返回顺序必须跟模型给的顺序，而不是召回顺序：{got}"
    assert len(got) == 3 and got[2] == recalled_ids[0], f"选不满 limit 时应用召回顺序补齐：{got}"
    assert len(set(got)) == 3, "补齐不能产生重复"
    assert reranked[1].text.endswith("untested."), "进阅读包的必须是完整原文，不是截断版"


def test_rerank_fallbacks() -> None:
    """4. 三条降级路径：抛异常 / 非法 JSON / 未知 id，都必须退回召回顺序且不抛异常。"""
    evidences = build_corpus()
    config = FakeConfig()
    rule = load_rule_profile(RULES).rules_by_id["D-MODEL-IDENTIFICATION-V1"]
    vectors = build_evidence_vectors(evidences)
    evidences_by_id = {item.evidence_id: item for item in evidences}
    # 检索词故意放宽到能召回大半语料，这样「截到 limit=4」才真的在测截断。
    recalled = [item for _score, item in recall_evidence(
        ["identification", "endogeneity", "sample", "data", "firm", "performance"], vectors, evidences_by_id, 30
    )]
    expected = [item.evidence_id for item in recalled[:4]]
    assert len(recalled) >= 4, [item.evidence_id for item in recalled]

    def raises(_payload, _config):
        raise RuntimeError("Cloud model timed out after 300 seconds")

    def bad_json(_payload, _config):
        # call_plain_json 解析失败时会抛 RuntimeError，这里模拟同一层的抛错。
        raise RuntimeError("model json parse failed: expecting value")

    def unknown_ids(_payload, _config):
        return {"selected": [{"evidence_id": "NOPE-1"}, {"evidence_id": "NOPE-2"}]}

    def empty_selection(_payload, _config):
        return {"selected": []}

    def illegal_shape(_payload, _config):
        return {"note": "我拒绝输出 JSON"}

    cases = {
        "抛异常": raises,
        "非法 JSON": bad_json,
        "未知 id": unknown_ids,
        "返回空": empty_selection,
        "结构非法": illegal_shape,
    }
    # 降级路径还要一起钉死「条数不缩水」：降级时只给 2 条比直接退回旧实现更糟。
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
            assert got[: len(expected)] == expected, f"降级路径「{name}」没有退回召回顺序：{got} != {expected}"
            assert len(got) == len(legacy_ids), f"降级路径「{name}」条数缩水：{len(got)} < {len(legacy_ids)}"
            assert len(set(got)) == len(got), f"降级路径「{name}」出现重复：{got}"
    finally:
        evidence_retrieval._call_rerank_llm = original

    # 真调用的坏响应更常见的形态是「解析器抛错」而不是「解析出的结构不对」：
    # call_plain_json → parse_json_content 遇到非 JSON 文本会抛 RuntimeError。上面的 bad_json 覆盖这条。
    # 这里再补一层：解析器已经救回来的、外面裹着 markdown 代码块的 JSON 应该正常用上，不触发降级。
    wrapped = '```json\n{"selected": [{"evidence_id": "EV-06", "reason": "内生性处理"}], "note": "ok"}\n```'
    parsed = parse_json_content(wrapped)
    ids, reasons = evidence_retrieval._extract_selected_ids(parsed)
    assert ids == ["EV-06"], ids
    assert reasons.get("EV-06") == "内生性处理", reasons


def test_config_gate() -> None:
    """4b. 配置缺字段（老配置/替身对象）时按「关闭」处理，不能因为读不到开关就换链路。"""
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
    assert defaults.evidence_retrieval_enabled is False, "默认必须是关闭"
    assert defaults.evidence_recall_top_k == 30
    assert defaults.evidence_rerank_enabled is True
    # 不抄常量，守性质：实测三篇 1630 段证据的中位长度 137 字、P90 272 字，
    # 截断低于 400 会把约 14% 的片段尾部切掉，重排会误判该段不相关。
    assert defaults.evidence_rerank_max_chars >= 400, (
        f"重排截断 {defaults.evidence_rerank_max_chars} 太短，会切掉证据片段尾部（P90 = 272 字）"
    )

    class LegacyConfig:
        """模拟没有这几个字段的旧配置对象。"""

    profile = load_rule_profile(RULES)
    evidences = build_corpus()
    rule = profile.rules_by_id["D-DATA-SAMPLE-V1"]
    agent = AGENT_FOR_DIMENSION["D"]
    legacy = build_reading_pack(rule, evidences, agent, 4, None)
    assert [
        item.evidence_id for item in select_evidence(rule, evidences, agent, 4, None, LegacyConfig())
    ] == [item.evidence_id for item in legacy]


def test_switch_off_matches_legacy() -> None:
    """5. 开关关闭：select_evidence 必须与 build_reading_pack 逐条一致（证明不改现有行为）。"""
    profile = load_rule_profile(RULES)
    evidences = build_corpus()
    evidences += [
        evidence(f"EV-M{i}", 6 + i, "Methods", f"方法段 {i}：measurement validity and reliability of the scales. " * 4)
        for i in range(6)
    ]
    evidences += [
        evidence(f"EV-D{i}", 12 + i, "Discussion", f"讨论段 {i}：limitations and future research directions. " * 4)
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

    # config=None 也必须走旧行为（父 agent 接线漏传 config 时不能静默换链路）
    rule = profile.rules_by_id["D-DATA-SAMPLE-V1"]
    agent = AGENT_FOR_DIMENSION["D"]
    assert [
        item.evidence_id for item in select_evidence(rule, evidences, agent, 6, None, None)
    ] == [item.evidence_id for item in build_reading_pack(rule, evidences, agent, 6, None)]


def test_switch_on_uses_recall() -> None:
    """6. 开关打开：相关证据（样本 / 内生性）必须真的进阅读包——这是这层要解决的问题。"""
    profile = load_rule_profile(RULES)
    evidences = build_corpus()
    on = FakeConfig(evidence_retrieval_enabled=True, evidence_rerank_enabled=False)

    # 没有 retrieval_terms 时用 rule_id 回退词：D-DATA-SAMPLE-V1 → sample / data
    rule = profile.rules_by_id["D-DATA-SAMPLE-V1"]
    pack = [item.evidence_id for item in select_evidence(rule, evidences, AGENT_FOR_DIMENSION["D"], 3, None, on)]
    assert "EV-05" in pack, f"样本证据没进包：{pack}"

    rule = profile.rules_by_id["D-MODEL-IDENTIFICATION-V1"]
    pack = [item.evidence_id for item in select_evidence(rule, evidences, AGENT_FOR_DIMENSION["D"], 3, None, on)]
    assert "EV-06" in pack, f"识别策略证据没进包：{pack}"

    # 召回为空时必须回退启发式，不能交空包
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
        applicable_type="通用",
        source_reference=[],
        status="active",
        retrieval_terms=["zzyzx", "qqqqq"],
    )
    fallback = select_evidence(empty_rule, evidences, AGENT_FOR_DIMENSION["D"], 3, None, on)
    assert [item.evidence_id for item in fallback] == [
        item.evidence_id for item in build_reading_pack(empty_rule, evidences, AGENT_FOR_DIMENSION["D"], 3, None)
    ], "召回为空时没有回退启发式"


def test_no_new_dependency() -> None:
    """7. 没有新增第三方依赖：检索层只靠标准库 + 项目已有模块。"""
    for banned in ("numpy", "faiss", "sklearn", "scipy", "torch", "rank_bm25", "jieba"):
        assert banned not in sys.modules, f"引入了未声明的第三方依赖：{banned}"
    source = (Path(__file__).resolve().parents[1] / "paper_eval" / "evidence_retrieval.py").read_text(
        encoding="utf-8"
    )
    assert "import numpy" not in source and "import faiss" not in source
    # 复用既有检索原语，别自己重写一套分词/余弦
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
        "证据召回自检通过：检索层条数不少于旧实现（召回命中少时用启发式补齐）；词袋召回排序正确；"
        "LLM 重排按模型顺序与子集返回；抛异常/非法 JSON/未知 id/空返回/结构非法五条降级都退回召回顺序且不缩水；"
        "开关关闭时 select_evidence 与 build_reading_pack 逐条一致；无新增第三方依赖。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

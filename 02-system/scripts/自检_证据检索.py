"""证据检索自检：规则英文检索词、中文规则文本不再被当检索词、C/D 维度专属召回、阅读包限额。

背景（三个病灶）：
  1) DIMENSION_KEYWORD_HINTS / RULE_KEYWORD_HINTS 里混进了自检样例论文（Zhai 2023，教育测量 /
     CDM / Q-matrix 主题）的专属术语，对真正要评的 AI-HRM 管理学论文几乎零命中；
  2) rule_keyword_hints 把中文规则文本用正则抽成"关键词"，再拿去搜英文正文 item.text → 恒不命中；
  3) C / D 维度在 build_reading_pack 里没有任何专属检索环节，D（25 分，权重最大）实测平均只有
     8.9 条候选证据（A/B 有 14-16.7 条）。

运行：
    python ".\scripts\自检_证据检索.py"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.agents import (
    AGENTS,
    DIMENSION_KEYWORD_HINTS,
    RULE_KEYWORD_HINTS,
    batch_response_instruction,
    build_reading_pack,
    rule_keyword_hints,
    score_method_rigor_hits,
    score_theory_rigor_hits,
)
from paper_eval.models import Evidence
from paper_eval.rules import load_rule_profile

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "评价规则库" / "rules_v1.yaml"
AGENT_FOR_DIMENSION = {dimension: agent for agent in AGENTS for dimension in agent.dimension_ids}

# 自检样例论文（Zhai 2023：教育测量 / CDM / Q-matrix / 机器-人工一致性）专属术语。
# 这些词落在检索词表里就是污染：换成 AI-HRM 管理学论文后它们只会把候选证据挤向无关段落，
# 或者命中错内容。任何一路检索词都不允许再出现。
POLLUTION_BLACKLIST = [
    "cognitive diagnostic", "cdm", "cdm results", "q-matrix", "cognitive skills", "cognitive patterns",
    "automatic scoring", "automated scoring", "automated scoring systems", "automated systems",
    "fine-grained", "coarse-grained", "personalized", "personalized feedback",
    "machine-human agreement", "cohen", "kappa", "srmsr", "heatmap", "probabilities",
    "aligned with prior", "diagnostic", "diagnostic information", "removed item", "test without items",
    "rubric", "participants", "real-time feedback", "tutorials", "follow-up tutorials",
    "instructional support", "private school", "public middle schools", "middle school",
    "same total ability score", "applied cdm", "timely feedback", "automatic guidance",
    "human experts encountered", "not unexpected", "not all students require", "assessment",
]

# 一段连续 6 个以上汉字：旧实现从中文规则文本里正则抽出来的"关键词"就是这种形状。
LONG_CHINESE_RUN = re.compile(r"[\u4e00-\u9fff]{6,}")

# 评分 prompt（batch_response_instruction）里不允许再出现的样例论文术语。
# 与检索词黑名单分开列：prompt 是直接引导模型的，判据允许更严（含"教育技术/教师诊断"这类领域词）。
PROMPT_DOMAIN_BLACKLIST = [
    "Q-matrix", "SRMSR", "heatmap", "CDM", "Cohen's kappa",
    "学习进阶", "细粒度诊断", "学生模式", "教师诊断", "学习干预", "评分量规", "教育技术", "异常题项",
]
# 从规则 definition / indicator_name 里直接摘的原文，出现即说明又回去抽中文字面词了。
CHINESE_RULE_TEXT_SAMPLES = [
    "研究问题重要性",
    "判断论文是否说明",
    "判断论文是否",
    "需要能定位到",
    "研究对象与研究问题明确",
    "观测点",
]


def blacklist_hit(text: str) -> str | None:
    """整词匹配黑名单；返回命中的词，未命中返回 None。"""
    lowered = str(text).lower()
    for term in POLLUTION_BLACKLIST:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered):
            return term
    return None


def evidence(eid: str, page: int, section: str, text: str) -> Evidence:
    return Evidence(
        evidence_id=eid,
        paper_id="PAPER-TEST",
        section=section,
        page=page,
        span=(0, len(text)),
        text=text,
        text_hash=f"h-{eid}",
    )


# 造出来的英文证据片段。section 用 "Main Body"：任何规则的 evidence_location 与 agent.section_hints
# 都匹配不上，所以能不能进阅读包只能由专属召回那一路决定，测试才有区分度。
FIXTURES = {
    "SAMPLE": evidence(
        "EV-SAMPLE", 1, "Main Body",
        "We collected survey data from 412 respondents in 87 firms. The data collection followed a "
        "two-stage sampling procedure, the response rate was 61 percent, and the final sample size "
        "is adequate for the estimation we report below.",
    ),
    "IDENT": evidence(
        "EV-IDENT", 2, "Main Body",
        "Identification rests on an instrumental variable strategy. We address endogeneity with "
        "two-stage least squares and a full set of control variables; the estimates survive checks "
        "for omitted variable bias and selection concerns.",
    ),
    "DESIGN": evidence(
        "EV-DESIGN", 3, "Main Body",
        "Our research design is a multi-wave longitudinal field study. The design matches the causal "
        "question because the time lag between waves lets us observe change; random assignment was "
        "not feasible, so the panel structure carries the inference.",
    ),
    "MEASURE": evidence(
        "EV-MEASURE", 4, "Main Body",
        "Every construct was measured with validated scales adapted from prior work. Each measurement "
        "scale used five items, factor analysis supports construct validity, and the reliability of "
        "every scale exceeds the conventional threshold.",
    ),
    "EXEC": evidence(
        "EV-EXEC", 5, "Main Body",
        "The procedure we followed is reported step by step. We collected the data in 2019, then ran "
        "the main regression and a structural equation model. The coding process and every data "
        "analysis decision are documented in the appendix.",
    ),
    "QUAL": evidence(
        "EV-QUAL", 6, "Main Body",
        "We adopt a qualitative case study design. We conducted 34 interviews and analyzed the "
        "transcripts with grounded theory coding; the thematic analysis followed an inductive logic "
        "and we report triangulation across informants and field notes.",
    ),
    "THEORY": evidence(
        "EV-THEORY", 7, "Main Body",
        "We define algorithmic discretion as the latitude employees have in interpreting system "
        "output. This conceptual definition is distinct from related constructs such as autonomy. "
        "We argue that the underlying assumption of prior work is taken for granted; therefore we "
        "derive hypotheses about its boundary conditions. The theoretical logic builds on agency "
        "theory, and we assume managers respond to perceived control. We also consider an "
        "alternative explanation: reverse causality and omitted variable bias could produce the "
        "same pattern, so we cannot rule out confounding.",
    ),
}

# 与上面五条方法片段完全不沾边的填充段：用来验证召回是按内容排序，不是靠兜底凑数。
FILLER_TEXTS = [
    "The author thanks the editor and anonymous referees for their helpful comments on earlier versions of this manuscript and for their guidance.",
    "An earlier version of this paper was presented at the annual meeting of the academy and benefited from the discussion that followed.",
    "The remaining sections of this manuscript proceed as follows and introduce each part of the argument in turn for the reader.",
    "Support from the research fund of the university is gratefully acknowledged by the authors of this study and their colleagues.",
    "Any remaining errors and all interpretations offered in this article are those of the authors alone and not of their institutions.",
    "We are indebted to the two reviewers whose suggestions substantially improved the clarity of the exposition in this article.",
    "The order of authorship is alphabetical and both authors contributed equally to every part of the work reported here.",
    "A previous draft circulated widely among colleagues, and the questions we received shaped the framing adopted in this version.",
    "The views expressed in this article do not necessarily reflect those of any organization with which the authors are affiliated.",
    "We thank the participants of the doctoral consortium for their generous engagement with an earlier draft of this research note.",
]

# 每条 D 规则应当被召回的那条方法片段。
EXPECTED_D_FIXTURE = {
    "D-DESIGN-FIT-V1": "DESIGN",
    "D-DATA-SAMPLE-V1": "SAMPLE",
    "D-MEASUREMENT-VALIDITY-V1": "MEASURE",
    "D-MODEL-IDENTIFICATION-V1": "IDENT",
    "D-METHOD-EXECUTION-V1": "EXEC",
    "D-QUAL-GENRE-FIT-V1": "QUAL",
}


def method_fixture_pool() -> list[Evidence]:
    pool = list(FIXTURES.values())
    for index, text in enumerate(FILLER_TEXTS):
        pool.append(evidence(f"EV-FILL-{index}", 20 + index, "Main Body", text))
    return pool


def main() -> int:
    profile = load_rule_profile(RULES)
    assert len(profile.rules) == 34, len(profile.rules)

    # 1) 34 条规则都必须有非空 retrieval_terms，且不落在污染黑名单里
    for rule in profile.rules:
        terms = list(rule.retrieval_terms)
        assert terms, f"{rule.rule_id} 没有 retrieval_terms（英文检索词）"
        assert 6 <= len(terms) <= 12, f"{rule.rule_id} retrieval_terms 共 {len(terms)} 个，应在 6-12 之间"
        for term in terms:
            assert term.isascii(), f"{rule.rule_id} 的检索词不是英文：{term}"
            assert ": " not in term, f"{rule.rule_id} 的检索词含 ASCII 冒号加空格，会破坏 YAML：{term}"
            hit = blacklist_hit(term)
            assert hit is None, f"{rule.rule_id} 的检索词 {term!r} 命中污染黑名单 {hit!r}"
    print(f"检索词表：34 条规则全部声明 retrieval_terms（{min(len(r.retrieval_terms) for r in profile.rules)}"
          f"-{max(len(r.retrieval_terms) for r in profile.rules)} 个），无一命中样例论文污染黑名单")

    # 2) 维度词表与规则类型词表也不许有污染术语
    for dimension_id, hints in DIMENSION_KEYWORD_HINTS.items():
        # 不锁死"15-45 个"这种魔数：词表长度本身不是性质，E 维从 45 涨到 48 既没坏处也没好处。
        # 真正要守的是下面两件（它们坏了才会有功能性后果）。
        assert hints, f"DIMENSION_KEYWORD_HINTS[{dimension_id}] 为空"
        assert any(hint.isascii() for hint in hints), f"{dimension_id} 维没有任何英文关键词"
        assert len(hints) <= 64, f"{dimension_id} 维词表 {len(hints)} 个，疑似失控膨胀"
        for hint in hints:
            hit = blacklist_hit(hint)
            assert hit is None, f"DIMENSION_KEYWORD_HINTS[{dimension_id}] 的 {hint!r} 命中污染黑名单 {hit!r}"
    for marker, hints in RULE_KEYWORD_HINTS.items():
        for hint in hints:
            hit = blacklist_hit(hint)
            assert hit is None, f"RULE_KEYWORD_HINTS[{marker}] 的 {hint!r} 命中污染黑名单 {hit!r}"
    print("词表清理：DIMENSION_KEYWORD_HINTS / RULE_KEYWORD_HINTS 已无样例论文专属术语")

    # 2b) 词袋截断不得吃掉规则自带的检索词。
    # rule_keyword_hints 末尾有 [:48] 截断，词源顺序是 retrieval_terms → 维度通用词 → 规则类型词，
    # 所以维度词表膨胀只会截掉通用词的尾巴，不会截掉 rule.retrieval_terms。
    # 这条性质比"词表几个词"重要得多：一旦词源顺序被调换，规则专属检索词会被静默丢弃。
    for rule in profile.rules:
        hints = rule_keyword_hints(rule)
        missing = [term for term in rule.retrieval_terms if term.lower() not in hints]
        assert not missing, f"{rule.rule_id} 的检索词被词袋截断吃掉了：{missing}"
    print("词袋截断：34 条规则的 retrieval_terms 全部存活在 rule_keyword_hints 返回值里")

    # 3) rule_keyword_hints 不再从中文规则文本抽字面词
    for rule in profile.rules:
        hints = rule_keyword_hints(rule)
        assert hints, f"{rule.rule_id} 的检索词袋为空"
        joined = " ".join(hints)
        long_run = LONG_CHINESE_RUN.search(joined)
        assert long_run is None, f"{rule.rule_id} 的检索词袋出现长中文短语：{long_run.group(0)!r}"
        for sample in CHINESE_RULE_TEXT_SAMPLES:
            assert sample not in joined, f"{rule.rule_id} 的检索词袋混入了中文规则原文：{sample!r}"
        for hint in hints:
            hit = blacklist_hit(hint)
            assert hit is None, f"{rule.rule_id} 的检索词 {hint!r} 命中污染黑名单 {hit!r}"
        # 规则自带的英文检索词必须出现在词袋里（最具体的一路不能被截断掉）
        for term in rule.retrieval_terms:
            assert term.lower() in hints, f"{rule.rule_id} 的检索词 {term!r} 没进 rule_keyword_hints"
    # 中文通用词仍要保留（对中文正文有用），不能一刀切成纯英文
    assert any(re.search(r"[\u4e00-\u9fff]", hint) for hint in rule_keyword_hints(profile.rules[0]))
    print("词袋重建：只由英文检索词 + 通用词表构成，不再从中文规则文本抽字面词，中文通用词保留")

    # 4) C / D 专属召回函数：只对自己的维度生效，且能召回构造的相关片段
    pool = method_fixture_pool()
    for rule in profile.rules:
        if rule.dimension_id == "C":
            assert score_method_rigor_hits(rule, pool) == [], f"{rule.rule_id} 不该走 D 召回"
        elif rule.dimension_id == "D":
            assert score_theory_rigor_hits(rule, pool) == [], f"{rule.rule_id} 不该走 C 召回"

    for rule in profile.rules:
        if rule.dimension_id != "D":
            continue
        expected = EXPECTED_D_FIXTURE[rule.rule_id]
        hits = score_method_rigor_hits(rule, pool)
        hit_ids = [item.evidence_id for item in hits]
        assert FIXTURES[expected].evidence_id in hit_ids, f"{rule.rule_id} 的 D 专属召回没命中 {expected}"
        # 相关片段必须排在填充段前面：召回是按内容打分，不是靠兜底顺序
        assert hit_ids.index(FIXTURES[expected].evidence_id) < len(FILLER_TEXTS), (
            f"{rule.rule_id} 的相关片段排在填充段之后：{hit_ids[:6]}"
        )
        pack = build_reading_pack(rule, pool, AGENT_FOR_DIMENSION["D"], 5, None)
        pack_ids = [item.evidence_id for item in pack]
        assert FIXTURES[expected].evidence_id in pack_ids, (
            f"{rule.rule_id} 的阅读包没选进 {expected}：{pack_ids}"
        )
        # D 路排在兜底之前：整包 5 条必须全是方法片段，一条填充段都不该混进来
        filler_ids = {f"EV-FILL-{index}" for index in range(len(FILLER_TEXTS))}
        assert not (set(pack_ids) & filler_ids), (
            f"{rule.rule_id} 的阅读包靠兜底填充段凑数：{pack_ids}"
        )
    print(f"D 维度专属召回：6 条规则的 sample/data collection 与 identification/endogeneity 片段全部进包并排在最前")

    # 5) C 维度专属召回：理论严谨性片段进包
    for rule in profile.rules:
        if rule.dimension_id != "C":
            continue
        hits = score_theory_rigor_hits(rule, pool)
        assert hits and hits[0].evidence_id == FIXTURES["THEORY"].evidence_id, (
            f"{rule.rule_id} 的 C 专属召回首位不是理论严谨性片段：{[i.evidence_id for i in hits[:3]]}"
        )
        pack = build_reading_pack(rule, pool, AGENT_FOR_DIMENSION["C"], 5, None)
        pack_ids = [item.evidence_id for item in pack]
        assert FIXTURES["THEORY"].evidence_id in pack_ids, f"{rule.rule_id} 的阅读包没选进理论片段：{pack_ids}"
    print("C 维度专属召回：7 条规则的理论严谨性片段全部进包")

    # 6) 过短片段（<90 字符）不参与 C / D 打分
    short = evidence("EV-SHORT", 1, "Main Body", "We define the construct and report the sample.")
    assert len(short.text) < 90
    for rule in profile.rules:
        if rule.dimension_id in {"C", "D"}:
            hits = score_theory_rigor_hits(rule, [short]) + score_method_rigor_hits(rule, [short])
            assert hits == [], f"{rule.rule_id} 把过短片段也当成候选了"
    print("最短长度门槛：<90 字符的片段不参与 C / D 专属召回")

    # 7) 限额：任何维度、任何规则，build_reading_pack 都不得超发
    big_pool = []
    for index in range(40):
        section = ["Introduction", "Literature Review", "Theory", "Methods", "Data", "Sample",
                   "Measures", "Results", "Discussion", "Conclusion", "Appendix"][index % 11]
        text = (
            f"{section} paragraph {index}. We collected survey data from respondents, measured every "
            "construct with validated scales, estimated a regression model with control variables, and "
            "report robustness checks. The theoretical contribution is developed in the discussion, "
            "together with limitations, practical implications and directions for future research."
        )
        big_pool.append(evidence(f"EV-BIG-{index}", 1 + index // 3, section, text))
    big_pool.append(evidence("EV-REF-1", 30, "References", "References " + "Smith, J. 2020. A study of organizations. " * 6))
    big_pool.append(evidence("EV-REF-2", 31, "References", "References " + "Lee, K. 2021. Another study of organizations. " * 6))

    for rule in profile.rules:
        pack = build_reading_pack(rule, big_pool, AGENT_FOR_DIMENSION[rule.dimension_id], 10, None)
        assert len(pack) <= 10, f"{rule.rule_id} 超发：{len(pack)} 条"
        assert len({item.evidence_id for item in pack}) == len(pack), f"{rule.rule_id} 出现重复证据"
        assert all(item.section != "References" for item in pack), f"{rule.rule_id} 把参考文献段当成了证据"
    print("阅读包限额：34 条规则在限额 10 下均未超发，且无重复、无参考文献段")

    # 8) 评分 prompt 里的领域校准文本也不许带样例论文术语。
    # 这比关键词表更严重：它不是检索不到证据，而是直接告诉模型什么算贡献、什么算稳健性证据。
    prompt_text = "\n".join(batch_response_instruction(agent) for agent in AGENTS)
    for banned in PROMPT_DOMAIN_BLACKLIST:
        assert banned not in prompt_text, f"评分 prompt 里仍有样例论文术语：{banned!r}"
    # 设计意图必须保留：不能因为论文没写固定字样、或不是传统回归稳健性就判 not_satisfied
    for intent in [
        "不要求出现固定关键词",
        "E-ROBUSTNESS 至少应考虑 partial",
        "F-REPRODUCIBILITY 可给 partial",
        "F-LIMITATIONS 和 F-EXTERNAL-VALIDITY",
        "不应只因未出现固定的管理学理论措辞而判 not_satisfied",
        "应落到管理实践上",
    ]:
        assert intent in prompt_text, f"评分 prompt 丢了原有校准意图：{intent!r}"
    # A-F 六个维度每个都要有自己的一份校准文本（B/E/F 三块必须真的注进了 prompt）
    for marker in ["B 维度评分校准", "E 维度评分校准", "F 维度评分校准"]:
        assert marker in prompt_text, marker
    print("评分 prompt 去污染：B/E/F 三块校准文本已换成管理学口径，原有校准意图保留")

    print("证据检索自检通过：英文检索词就位，中文规则文本不再当检索词，C/D 维度各有专属召回。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

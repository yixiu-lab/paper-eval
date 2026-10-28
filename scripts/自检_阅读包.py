"""阅读包自检：规则声明的章节必须优先进入候选证据。

背景：规则库用 Method / Tables 这种单数写法，解析结果是 Methods / Tables and Figures。
旧的 `in` 匹配 + “排在最后且每章节只取 1 条”导致 F 维度实践启示类规则拿不到 Discussion
证据，被判 6/6 不满足。

运行：
    python ".\scripts\自检_阅读包.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.agents import AGENTS, build_reading_pack, section_matches
from paper_eval.models import Evidence
from paper_eval.rules import load_rule_profile

RULES = Path(__file__).resolve().parents[1] / "评价规则库" / "rules_v1.yaml"
AGENT_FOR_DIMENSION = {dimension: agent for agent in AGENTS for dimension in agent.dimension_ids}


def evidence(evidence_id: str, page: int, section: str, length: int = 200) -> Evidence:
    text = (f"{section} 段落内容，用于自检。 " * ((length // 12) + 1))[:length]
    return Evidence(
        evidence_id=evidence_id,
        paper_id="PAPER-TEST",
        section=section,
        page=page,
        span=(0, len(text)),
        text=text,
        text_hash=f"h-{evidence_id}",
    )


def main() -> int:
    # 1. 章节宽容匹配
    assert section_matches("Method", "Methods")
    assert section_matches("Tables", "Tables and Figures")
    assert section_matches("Figure", "Tables and Figures")
    assert section_matches("Discussion", "Discussion")
    assert section_matches("结论", "结论")
    assert not section_matches("Discussion", "Introduction")
    assert not section_matches("", "Discussion")

    profile = load_rule_profile(RULES)

    # 2. 证据池：Discussion / Conclusion 排在最后，靠关键词完全拿不到
    evidences = [evidence(f"EV-INTRO-{i}", 1 + i, "Introduction") for i in range(12)]
    evidences += [evidence(f"EV-METHOD-{i}", 6 + i, "Methods") for i in range(12)]
    evidences += [evidence(f"EV-DISC-{i}", 12 + i, "Discussion") for i in range(4)]
    evidences += [evidence(f"EV-CONC-{i}", 16 + i, "Conclusion") for i in range(4)]

    rule = profile.rules_by_id["F-PRACTICAL-IMPLICATIONS-V1"]
    agent = AGENT_FOR_DIMENSION["F"]
    pack = build_reading_pack(rule, evidences, agent, 5, None)
    sections = {item.section for item in pack}
    assert len(pack) == 5, len(pack)
    assert {"Discussion", "Conclusion"} & sections, f"实践启示规则没拿到 Discussion/Conclusion：{sections}"

    # 3. 声明了 Methods 的规则必须拿到 Methods
    rule = profile.rules_by_id["F-REPRODUCIBILITY-V1"]
    pack = build_reading_pack(rule, evidences, agent, 5, None)
    assert any(item.section == "Methods" for item in pack), {item.section for item in pack}

    # 4. 不声明章节也要能出满包（不能因为优先章节而空转）
    rule = profile.rules_by_id["B-INTERESTINGNESS-V1"]
    pack = build_reading_pack(rule, evidences, AGENT_FOR_DIMENSION["B"], 5, None)
    assert len(pack) == 5, len(pack)

    print("阅读包自检通过：章节宽容匹配生效，规则声明的章节优先进入候选证据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

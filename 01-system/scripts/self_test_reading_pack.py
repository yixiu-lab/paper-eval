'Reading package self-test: Chapters declared by the rules must be entered into candidate evidence first.\n\nBackground: The rule base uses the singular word Method / Tables, and the parsing result is Methods / Tables and Figures.\nDiscussion\nEvidence, sentenced 6/6 not satisfied.\n\nRun:\n    python ".\\scripts\\self_test_reading_pack.py"'

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.agents import AGENTS, build_reading_pack, section_matches
from paper_eval.models import Evidence
from paper_eval.rules import load_rule_profile

RULES = Path(__file__).resolve().parents[1] / 'rules' / "rules_v1.yaml"
AGENT_FOR_DIMENSION = {dimension: agent for agent in AGENTS for dimension in agent.dimension_ids}


def evidence(evidence_id: str, page: int, section: str, length: int = 200) -> Evidence:
    text = (f"{section} paragraph content, used for self-test." * ((length // 12) + 1))[:length]
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
    # 1. Chapter tolerance matching
    assert section_matches("Method", "Methods")
    assert section_matches("Tables", "Tables and Figures")
    assert section_matches("Figure", "Tables and Figures")
    assert section_matches("Discussion", "Discussion")
    assert section_matches('Conclusion', 'Conclusion')
    assert not section_matches("Discussion", "Introduction")
    assert not section_matches("", "Discussion")

    profile = load_rule_profile(RULES)

    # 2. Evidence pool: Discussion/Conclusion is ranked last and cannot be obtained by relying on keywords.
    evidences = [evidence(f"EV-INTRO-{i}", 1 + i, "Introduction") for i in range(12)]
    evidences += [evidence(f"EV-METHOD-{i}", 6 + i, "Methods") for i in range(12)]
    evidences += [evidence(f"EV-DISC-{i}", 12 + i, "Discussion") for i in range(4)]
    evidences += [evidence(f"EV-CONC-{i}", 16 + i, "Conclusion") for i in range(4)]

    rule = profile.rules_by_id["F-PRACTICAL-IMPLICATIONS-V1"]
    agent = AGENT_FOR_DIMENSION["F"]
    pack = build_reading_pack(rule, evidences, agent, 5, None)
    sections = {item.section for item in pack}
    assert len(pack) == 5, len(pack)
    assert {"Discussion", "Conclusion"} & sections, f"Practice enlightenment rules not obtained Discussion/Conclusion: {sections}"

    # 3. Rules that declare Methods must obtain Methods
    rule = profile.rules_by_id["F-REPRODUCIBILITY-V1"]
    pack = build_reading_pack(rule, evidences, agent, 5, None)
    assert any(item.section == "Methods" for item in pack), {item.section for item in pack}

    # 4. You must be able to produce a full package without declaring chapters (cannot idle because of priority chapters)
    rule = profile.rules_by_id["B-INTERESTINGNESS-V1"]
    pack = build_reading_pack(rule, evidences, AGENT_FOR_DIMENSION["B"], 5, None)
    assert len(pack) == 5, len(pack)

    print('Reading package self-test passed: Chapter tolerance matching takes effect, and chapters declared by the rules are given priority to enter candidate evidence.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

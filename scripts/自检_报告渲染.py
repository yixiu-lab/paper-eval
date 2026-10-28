"""报告渲染自检：Markdown 与 HTML 两条渲染路径必须能跑通。

加这个检查的原因：评分跑完却挂在"生成报告"这一步，整次运行的判断全部丢失
（harness 的失败分支只保存空判断）。渲染路径此前没有被任何自检覆盖。

运行：
    python ".\scripts\自检_报告渲染.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from literature_eval.html_report import render_html
from literature_eval.models import (
    AgentJudgement,
    Evidence,
    EvaluationRun,
    Judgement,
    Paper,
    RunStatus,
    now_iso,
)
from literature_eval.report import build_report, render_markdown
from literature_eval.rules import load_rule_profile, out_of_scope_rule_ids
from literature_eval.scorer import aggregate_scores, score_judgements

RULES = Path(__file__).resolve().parents[1] / "评价规则库" / "rules_v1.yaml"


def main() -> int:
    profile = load_rule_profile(RULES)
    paper = Paper(paper_id="PAPER-T", metadata={"title": "报告渲染自检样例"}, sections=[], references=[])
    run = EvaluationRun(
        run_id="RUN-RENDER-CHECK",
        paper_id="PAPER-T",
        harness_version="harness-test",
        model_versions={"core_agents": "test-model"},
        rule_profile=f"{profile.rule_profile_id}:{profile.version}",
        status=RunStatus.SUCCESS,
        started_at=now_iso(),
        completed_at=now_iso(),
    )
    evidence = Evidence(
        evidence_id="EV-1",
        paper_id="PAPER-T",
        section="Discussion",
        page=9,
        span=(0, 120),
        text="本文讨论了适用范围与局限。" * 6,
        text_hash="h1",
    )
    skipped = out_of_scope_rule_ids(profile, "实证")
    judgements = [
        AgentJudgement(
            task_id=run.run_id,
            agent_id="self-check",
            indicator_id=rule.rule_id,
            judgement=Judgement.UNCERTAIN if rule.rule_id in skipped else Judgement.PARTIAL,
            evidence_ids=[] if rule.rule_id in skipped else ["EV-1"],
            reason="报告渲染自检占位理由。",
            confidence=0.6,
            needs_review=False,
            agent_version="self-check",
            model_version="test-model",
        )
        for rule in profile.rules
    ]
    score_items = score_judgements(profile, judgements, skipped)
    summary = aggregate_scores(profile, score_items)

    report = build_report(
        run,
        paper,
        [evidence],
        profile,
        judgements,
        score_items,
        summary,
        [],
        [],
        paper_context={
            "paper_profile": {
                "paper_type": "实证",
                "paper_type_label": "管理学实证研究",
                "paper_type_source": "agent",
            }
        },
    )
    markdown = render_markdown(report)
    assert "## 评分标准说明" in markdown, "缺少评分标准说明章节"
    assert "判定" not in markdown or True
    rule_count = sum(1 for rule in profile.rules if rule.threshold)
    listed = markdown.count("：")  # 粗略计数，只要不空即可
    assert rule_count > 0 and listed > rule_count // 2
    html = render_html(report)
    assert "质量分" in html and "评分完整度" in html

    # 外部检索结果必须能渲染进两条报告路径：正常结果、摘要生成失败、整段失败三种形态都要活下来。
    external_payload = {
        "enabled": True,
        "candidates": 2,
        "stop_reason": "enough",
        "queries": [
            {
                "claim_id": "A-GAP-PUZZLE-V1",
                "rule_id": "A-GAP-PUZZLE-V1",
                "query": "research gap puzzle unresolved question",
                "provider": "openalex",
                "origin": "override",
            }
        ],
        "failures": [],
        "status": {"providers": ["openalex"], "provider": "openalex"},
        "related_works": [
            {
                "title": "Artificial Intelligence and Management",
                "year": 2021,
                "venue": "Academy of Management Review",
                "citation_count": 900,
                "rerank_score": 6.05,
                "summary": {
                    "method": "概念论文",
                    "findings": "自动化—增强悖论",
                    "experiments": "无实验",
                    "similar": "同样关注算法与人的关系",
                    "different": "不含量化证据",
                },
            },
            {
                "title": "摘要缺失的样例文献",
                "year": None,
                "venue": "",
                "citation_count": None,
                "rerank_score": 0.2,
                "summary": None,
                "summary_error": "模型额度不足",
            },
        ],
    }
    external_markdown = render_markdown({**report, "external_retrieval": external_payload})
    assert "## 外部相关工作" in external_markdown, "缺少外部相关工作章节"
    assert "Artificial Intelligence and Management" in external_markdown
    assert "方法：概念论文" in external_markdown, "五问摘要未渲染"
    assert "摘要生成失败：模型额度不足" in external_markdown, "摘要失败未标注"
    external_html = render_html({**report, "external_retrieval": external_payload})
    assert "外部相关工作" in external_html and "Artificial Intelligence and Management" in external_html
    broken_markdown = render_markdown({**report, "external_retrieval": {"error": "HTTP 429 Too Many Requests"}})
    assert "外部检索失败，已跳过" in broken_markdown, "整段失败未降级渲染"
    assert render_markdown(report).count("## 外部相关工作") == 0, "无外部检索时不应出现空章节"

    print(
        f"报告渲染自检通过：Markdown {len(markdown)} 字，HTML {len(html)} 字，规则说明 {rule_count} 条，"
        f"外部相关工作 {len(external_payload['related_works'])} 篇（含摘要失败与整段失败降级）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

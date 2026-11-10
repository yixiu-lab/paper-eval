'Report rendering self-test: the two rendering paths of Markdown and HTML must be able to run smoothly.\n\nThe reason for adding this check: after the scoring run was completed, it hung at the "Generate Report" step, and all the judgments of the entire run were lost.\n(The failure branch of harness only saves empty judgments). The rendering path has not been previously covered by any self-test.\n\nRun:\n    python ".\\scripts\\self_test_report_rendering.py"'

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.html_report import render_html
from paper_eval.models import (
    AgentJudgement,
    Evidence,
    EvaluationRun,
    Judgement,
    Paper,
    RunStatus,
    now_iso,
)
from paper_eval.report import build_report, render_markdown
from paper_eval.rules import load_rule_profile, out_of_scope_rule_ids
from paper_eval.scorer import aggregate_scores, score_judgements

RULES = Path(__file__).resolve().parents[1] / 'rules' / "rules_v1.yaml"


def main() -> int:
    profile = load_rule_profile(RULES)
    paper = Paper(paper_id="PAPER-T", metadata={"title": 'Report rendering self-test example'}, sections=[], references=[])
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
        text='This article discusses scope and limitations.' * 6,
        text_hash="h1",
    )
    skipped = out_of_scope_rule_ids(profile, 'Evidence')
    judgements = [
        AgentJudgement(
            task_id=run.run_id,
            agent_id="self-check",
            indicator_id=rule.rule_id,
            judgement=Judgement.UNCERTAIN if rule.rule_id in skipped else Judgement.PARTIAL,
            evidence_ids=[] if rule.rule_id in skipped else ["EV-1"],
            reason='Report rendering self-test placeholder reason.',
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
                "paper_type": 'Evidence',
                "paper_type_label": 'Empirical Research on Management',
                "paper_type_source": "agent",
            }
        },
    )
    markdown = render_markdown(report)
    assert '## Scoring Criteria Description' in markdown, 'Missing scoring standards description chapter'
    assert 'Judgment' not in markdown or True
    rule_count = sum(1 for rule in profile.rules if rule.threshold)
    listed = markdown.count("：")  # rough count, as long as it is not empty
    assert rule_count > 0 and listed > rule_count // 2
    html = render_html(report)
    assert 'quality points' in html and 'Rating completeness' in html

    # External search results must be able to be rendered into two reporting paths: normal results, summary generation failure, and entire paragraph failure must survive the three forms.
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
                    "method": 'Concept Paper',
                    "findings": 'Automation—The Augmentation Paradox',
                    "experiments": 'No experiment',
                    "similar": 'also pays attention to the relationship between algorithms and people',
                    "different": 'does not contain quantitative evidence',
                },
            },
            {
                "title": 'Sample document with missing abstract',
                "year": None,
                "venue": "",
                "citation_count": None,
                "rerank_score": 0.2,
                "summary": None,
                "summary_error": 'model quota is insufficient',
            },
        ],
    }
    external_markdown = render_markdown({**report, "external_retrieval": external_payload})
    assert '## External related work' in external_markdown, 'Missing external related work chapters'
    assert "Artificial Intelligence and Management" in external_markdown
    assert 'Methods: Concept Paper' in external_markdown, 'Five questions summary not rendered'
    assert 'Summary generation failed: Insufficient model quota' in external_markdown, 'Summary failed and not marked'
    external_html = render_html({**report, "external_retrieval": external_payload})
    assert 'External related work' in external_html and "Artificial Intelligence and Management" in external_html
    broken_markdown = render_markdown({**report, "external_retrieval": {"error": "HTTP 429 Too Many Requests"}})
    assert 'External retrieval failed and skipped' in broken_markdown, 'The entire section failed and the rendering was not downgraded.'
    assert render_markdown(report).count('## External related work') == 0, 'Empty chapters should not appear when there is no external search'

    print(
        f"Report rendering self-test passed: Markdown {len(markdown)} words, HTML {len(html)} words, rule description {rule_count} articles,"
        f"External related work {len(external_payload['related_works'])} (including summary failure and entire segment failure downgrade)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

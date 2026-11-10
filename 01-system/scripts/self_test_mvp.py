#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_eval.harness import evaluate_pdf
from paper_eval.pdf_parser import parse_pdf
from paper_eval.rules import load_rule_profile, validate_rule_profile


RULES = ROOT / 'rules' / "rules_v1.yaml"
# This is the only true-tuned model kit. The configuration must be specified explicitly: the default configuration points to Alibaba Cloud glm-5.2,
# will crash with 403 after the free quota is exhausted (shown as a false failure with judgment_count == 0).
AGENT_CONFIG = ROOT / "configs" / "agent_runtime.deepseek.yaml"
SAMPLE_PDF = ROOT / 'Method Reference 1-Zhai et al. - 2023 - Assessing Argumentation Using Machine Learning and.pdf'
OUT_DIR = ROOT / "runs" / "self_check"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    profile = load_rule_profile(RULES)
    errors = validate_rule_profile(profile)
    check(not errors, f"Rule base verification failed: {errors}")
    check(len(profile.dimensions) == 6, 'The number of first-level dimensions should be 6')
    check(len(profile.rules) == 34, 'The number of scoring rules should be 34 (24 baseline + 10 bonus points)')

    paper, evidences = parse_pdf(SAMPLE_PDF)
    check(paper.paper_id.startswith("PAPER-"), 'Paper ID format error')
    check(len(evidences) > 0, 'PDF Unresolved evidence fragments')
    check(all(item.evidence_id.startswith("EV-") for item in evidences[:10]), 'Evidence ID format error')

    result = evaluate_pdf(SAMPLE_PDF, RULES, OUT_DIR, agent_config_path=AGENT_CONFIG)
    check(result["judgement_count"] == len(profile.rules), 'The number of Agent judgments should be equal to the number of rules')
    # External document search will add EXT-* blocks to the evidence set, so it can only be "no less than the analysis result"
    check(result["evidence_count"] >= len(evidences), 'Harness The number of evidence should not be less than the analysis result')
    check(Path(result["json_report"]).exists(), 'JSON report not generated')
    check(Path(result["markdown_report"]).exists(), 'Markdown report not generated')
    check(0 <= result["scoring_completeness"] <= 1, 'Rating completeness must be between 0-1')

    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

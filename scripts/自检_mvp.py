#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from literature_eval.harness import evaluate_pdf
from literature_eval.pdf_parser import parse_pdf
from literature_eval.rules import load_rule_profile, validate_rule_profile


RULES = ROOT / "评价规则库" / "rules_v1.yaml"
# 这是唯一真调模型的套件。必须显式指定配置：默认配置指向阿里云 glm-5.2，
# 免费额度用尽后会 403 挂掉（表现为 judgement_count == 0 的假失败）。
AGENT_CONFIG = ROOT / "configs" / "agent_runtime.deepseek.yaml"
SAMPLE_PDF = ROOT / "方法参考文献1-Zhai 等 - 2023 - Assessing Argumentation Using Machine Learning and.pdf"
OUT_DIR = ROOT / "runs" / "self_check"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    profile = load_rule_profile(RULES)
    errors = validate_rule_profile(profile)
    check(not errors, f"规则库校验失败: {errors}")
    check(len(profile.dimensions) == 6, "一级维度数量应为 6")
    check(len(profile.rules) == 34, "评分规则数量应为 34（24 基准 + 10 加分）")

    paper, evidences = parse_pdf(SAMPLE_PDF)
    check(paper.paper_id.startswith("PAPER-"), "Paper ID 格式错误")
    check(len(evidences) > 0, "PDF 未解析出证据片段")
    check(all(item.evidence_id.startswith("EV-") for item in evidences[:10]), "Evidence ID 格式错误")

    result = evaluate_pdf(SAMPLE_PDF, RULES, OUT_DIR, agent_config_path=AGENT_CONFIG)
    check(result["judgement_count"] == len(profile.rules), "Agent 判断数量应等于规则数量")
    # 外部文献检索会往证据集里追加 EXT-* 块，所以只能是「不少于解析结果」
    check(result["evidence_count"] >= len(evidences), "Harness 证据数量不应少于解析结果")
    check(Path(result["json_report"]).exists(), "JSON 报告未生成")
    check(Path(result["markdown_report"]).exists(), "Markdown 报告未生成")
    check(0 <= result["scoring_completeness"] <= 1, "评分完整度必须在 0-1 之间")

    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

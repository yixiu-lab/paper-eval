"""Evaluation Reliability Layer 自检（全程离线，不重跑 Agent、不打外部 API）。

覆盖：
1. 配置化阈值读取（`reliability:` 段 + 未知键报错）
2. Run Variance：总分均值/标准差/极差、维度波动、等级翻转 → high_variance
3. 单次运行必须老实报 insufficient_data（不许假装算得出来）
4. 批次排序稳定性：换序批次 → ranking_change；同序批次 → high
5. Evidence Variance：证据完全一致 → overlap 1.0；完全换掉 → evidence_instability
6. Judgment Variance：一致率、分布、熵；证据稳而判断不稳 → agent_reasoning
7. Calibration Uncertainty：换映射重算分 / 排序相关 / Top-K 重合 / calibration_sensitive
8. Reliability Diagnosis 四种情况的归因与建议
9. 最终 Reliability Report 结构 + **不含总可靠性分**
10. 审计链：Score → Judgment → Evidence → Reliability Diagnosis
11. 离线重算 CLI 端到端（真报告 JSON 走一遍）

运行：
    python ".\scripts\\自检_可靠性层.py"
    python ".\scripts\\自检_可靠性层.py" --backtests   # 额外跑真实 v23 四批数据（若存在）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from literature_eval.reliability import (  # noqa: E402
    ReliabilityConfig,
    RuleRecord,
    RunTrace,
    batch_ranking_stability,
    evidence_variance,
    group_by_paper,
    judgment_variance,
    load_reliability_config,
    run_variance,
    spearman,
)
from literature_eval.reliability_diagnosis import (  # noqa: E402
    CALIBRATION_SCHEMES,
    build_reliability_report,
    calibration_uncertainty,
    diagnose_reliability,
    render_reliability_markdown,
)
from literature_eval.rules import load_rule_profile  # noqa: E402

RULES_PATH = ROOT / "评价规则库" / "rules_v1.yaml"
CONFIG_PATH = ROOT / "configs" / "agent_runtime.deepseek.yaml"
BACKTEST_RUNS = Path(r"D:\创业\agent文献打分\初始文件\article reviewer\runs")


def make_trace(
    run_id: str,
    paper_id: str,
    total: float,
    judgements: dict[str, str],
    evidence: dict[str, list[str]],
    dimension_scores: dict[str, float] | None = None,
    grade: str = "好论文",
    potential: str = "中",
    started_at: str = "2026-01-01T00:00:00Z",
    source_path: str = "",
    confidence: float = 0.7,
) -> RunTrace:
    records = []
    for rule_id, judgement in judgements.items():
        evidence_ids = list(evidence.get(rule_id, []))
        records.append(
            RuleRecord(
                rule_id=rule_id,
                dimension_id=rule_id.split("-")[0],
                judgement=judgement,
                confidence=confidence,
                evidence_ids=evidence_ids,
                raw_score=1.0,
                weight=1.0,
                evidence_sections=["Introduction"] if evidence_ids else [],
                evidence_pages=[1] if evidence_ids else [],
            )
        )
    return RunTrace(
        run_id=run_id,
        paper_id=paper_id,
        rule_records=records,
        total_score=total,
        dimension_scores=dimension_scores or {dim: 10.0 for dim in ("A", "B", "C", "D", "E", "F")},
        quality_grade=grade,
        top_journal_potential=potential,
        started_at=started_at,
        source_path=source_path,
    )


def step_1_config() -> ReliabilityConfig:
    config = load_reliability_config(CONFIG_PATH)
    assert config.run_variance_threshold == 5.0, config
    assert config.evidence_overlap_threshold == 0.6, config
    assert config.judgment_agreement_threshold == 0.8, config
    assert ReliabilityConfig().enabled is False, "默认必须关闭：不启用时评价流程逐位不变"
    try:
        load_reliability_config(ROOT / "configs" / "agent_runtime.cloud.yaml")
    except ValueError:
        raise AssertionError("cloud 配置没有 reliability 段，不应报错")
    # 未知键必须报错，而不是静默忽略（配置写错会让人以为阈值生效了）
    temp = Path(tempfile.gettempdir()) / "_reliability_unknown_key.yaml"
    temp.write_text("reliability:\n  run_variance_threshold: 5.0\n  typo_key: 1\n", encoding="utf-8")
    try:
        load_reliability_config(temp)
        raise AssertionError("未知键必须报错")
    except ValueError:
        pass
    finally:
        temp.unlink(missing_ok=True)
    print(
        f"1) 配置化阈值：run_variance_threshold={config.run_variance_threshold}、"
        f"evidence_overlap_threshold={config.evidence_overlap_threshold}、"
        f"judgment_agreement_threshold={config.judgment_agreement_threshold}；未知键报错；默认关闭"
    )
    return config


def step_2_run_variance(config: ReliabilityConfig) -> dict:
    judgements = {f"A-R{i}-V1": "mostly" for i in range(4)}
    evidence = {rule_id: [f"EV-{index}"] for index, rule_id in enumerate(judgements)}
    # 稳定批次：总分只差 1 分
    stable = [
        make_trace("RUN-S1", "PAPER-STABLE", 90.0, judgements, evidence, grade="好论文"),
        make_trace("RUN-S2", "PAPER-STABLE", 91.0, judgements, evidence, grade="好论文"),
    ]
    stable_result = run_variance(stable, config)
    assert stable_result["stability"] == "high", stable_result
    assert stable_result["range"] == 1.0 and stable_result["grade_flip"] is False, stable_result

    # 波动批次：总分差 8 分且等级翻转
    volatile = [
        make_trace("RUN-V1", "PAPER-VOLATILE", 88.0, judgements, evidence, grade="好论文"),
        make_trace("RUN-V2", "PAPER-VOLATILE", 96.0, judgements, evidence, grade="顶刊级"),
    ]
    volatile_result = run_variance(volatile, config)
    assert "high_variance" in volatile_result["flags"], volatile_result
    assert volatile_result["stability"] == "low", volatile_result
    assert volatile_result["grade_flip"] is True, volatile_result
    assert volatile_result["mean_score"] == 92.0 and volatile_result["std"] == 4.0, volatile_result

    # 维度波动：C 维度在两次运行间差 5 分
    dimension = [
        make_trace("RUN-D1", "PAPER-DIM", 80.0, judgements, evidence, dimension_scores={"A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0, "E": 10.0, "F": 10.0}),
        make_trace("RUN-D2", "PAPER-DIM", 81.0, judgements, evidence, dimension_scores={"A": 10.0, "B": 10.0, "C": 15.0, "D": 10.0, "E": 10.0, "F": 10.0}),
    ]
    dimension_result = run_variance(dimension, config)
    assert "C" in dimension_result["dimension_flagged"], dimension_result
    assert "dimension_variance" in dimension_result["flags"], dimension_result

    single = run_variance([stable[0]], config)
    assert single["stability"] == "insufficient_data" and single["range"] is None, single
    print(
        f"2-3) Run Variance：稳定批稳定性 {stable_result['stability']}（极差 {stable_result['range']}）；"
        f"波动批 {volatile_result['stability']}（极差 {volatile_result['range']}，等级翻转 "
        f"{volatile_result['grade_flip_detail']}）；维度 C 波动被抓到；"
        f"单次运行老实报 {single['stability']}"
    )
    return {"volatile": volatile_result, "stable": stable_result, "dimension": dimension_result}


def step_4_ranking(config: ReliabilityConfig) -> None:
    # 同一批次目录下 3 篇论文，两个批次给出相同排序 → high
    same = {
        "PAPER-A": [make_trace("R1", "PAPER-A", 90.0, {"A-1": "mostly"}, {"A-1": ["EV-1"]}, source_path="batch1/R1.json")],
        "PAPER-B": [make_trace("R2", "PAPER-B", 80.0, {"A-1": "mostly"}, {"A-1": ["EV-1"]}, source_path="batch1/R2.json")],
        "PAPER-C": [make_trace("R3", "PAPER-C", 70.0, {"A-1": "mostly"}, {"A-1": ["EV-1"]}, source_path="batch1/R3.json")],
    }
    for paper_id, scores in (("PAPER-A", 90.0), ("PAPER-B", 80.0), ("PAPER-C", 70.0)):
        same[paper_id].append(
            make_trace(f"R-{paper_id}", paper_id, scores, {"A-1": "mostly"}, {"A-1": ["EV-1"]}, source_path="batch2/R.json")
        )
    stable_ranking = batch_ranking_stability(same, config)
    assert stable_ranking["checked"] and stable_ranking["mean_spearman"] == 1.0, stable_ranking
    assert stable_ranking["ranking_change"] is False, stable_ranking

    # 第二个批次把顺序颠倒 → ranking_change
    flipped = {
        "PAPER-A": [make_trace("R1", "PAPER-A", 70.0, {"A-1": "mostly"}, {"A-1": ["EV-1"]}, source_path="batch1/R1.json")],
        "PAPER-B": [make_trace("R2", "PAPER-B", 80.0, {"A-1": "mostly"}, {"A-1": ["EV-1"]}, source_path="batch1/R2.json")],
        "PAPER-C": [make_trace("R3", "PAPER-C", 90.0, {"A-1": "mostly"}, {"A-1": ["EV-1"]}, source_path="batch1/R3.json")],
    }
    for paper_id, scores in (("PAPER-A", 95.0), ("PAPER-B", 80.0), ("PAPER-C", 60.0)):
        flipped[paper_id].append(
            make_trace(f"R-{paper_id}", paper_id, scores, {"A-1": "mostly"}, {"A-1": ["EV-1"]}, source_path="batch2/R.json")
        )
    flip_ranking = batch_ranking_stability(flipped, config)
    assert flip_ranking["ranking_change"] is True, flip_ranking
    assert flip_ranking["mean_spearman"] < config.ranking_correlation_threshold, flip_ranking
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert spearman([1, 2, 3], [2, 4, 6]) == 1.0
    print(
        f"4) 批次排序：同序批次 Spearman {stable_ranking['mean_spearman']}；"
        f"颠倒批次 Spearman {flip_ranking['mean_spearman']} → ranking_change；"
        f"Top-{stable_ranking['top_k']} 重合 {stable_ranking['top_k_overlap']}"
    )


def step_5_evidence(config: ReliabilityConfig) -> None:
    same_evidence = {"A-1": ["EV-1", "EV-2"], "B-1": ["EV-3"]}
    identical = [
        make_trace("R1", "P1", 80.0, {"A-1": "mostly", "B-1": "mostly"}, same_evidence),
        make_trace("R2", "P1", 80.0, {"A-1": "mostly", "B-1": "mostly"}, same_evidence),
    ]
    identical_result = evidence_variance(identical, config)
    assert identical_result["evidence_overlap"] == 1.0, identical_result
    assert identical_result["stability"] == "high", identical_result

    moved = [
        make_trace("R1", "P1", 80.0, {"A-1": "mostly", "B-1": "mostly"}, {"A-1": ["EV-1", "EV-2"], "B-1": ["EV-3"]}),
        make_trace("R2", "P1", 80.0, {"A-1": "mostly", "B-1": "mostly"}, {"A-1": ["EV-9"], "B-1": ["EV-8"]}),
    ]
    moved_result = evidence_variance(moved, config)
    assert moved_result["evidence_overlap"] == 0.0, moved_result
    assert "evidence_instability" in moved_result["flags"], moved_result
    assert moved_result["stability"] == "low", moved_result
    assert "A-1" in moved_result["unstable_rule_ids"], moved_result
    print(
        f"5) Evidence Variance：证据完全一致 → 交集 {identical_result['evidence_overlap']}；"
        f"整批换证据 → 交集 {moved_result['evidence_overlap']}，标记 {moved_result['flags']}"
    )
    return moved_result


def step_6_judgment(config: ReliabilityConfig) -> dict:
    stable_evidence = {"B1": ["EV-1"], "B2": ["EV-2"]}
    # 证据稳、判断在 satisfied/partial 之间摇 → agent_reasoning
    reasoning = [
        make_trace("R1", "P1", 80.0, {"B1": "satisfied", "B2": "satisfied"}, stable_evidence),
        make_trace("R2", "P1", 80.0, {"B1": "partial", "B2": "satisfied"}, stable_evidence),
        make_trace("R3", "P1", 80.0, {"B1": "satisfied", "B2": "satisfied"}, stable_evidence),
    ]
    evidence_result = evidence_variance(reasoning, config)
    judgment_result = judgment_variance(reasoning, config, evidence_result)
    first = next(item for item in judgment_result["per_rule"] if item["rule_id"] == "B1")
    assert first["distribution"] == {"partial": 0.3333, "satisfied": 0.6667}, first
    assert abs(first["agreement"] - 2 / 3) < 1e-3, first
    assert first["source_of_uncertainty"] == "agent_reasoning", first
    assert "judgment_instability" in judgment_result["flags"], judgment_result
    assert judgment_result["source_of_uncertainty"] == "agent_reasoning", judgment_result
    assert "B1" in judgment_result["unstable_rule_ids"], judgment_result

    # 证据与判断同时不稳 → retrieval_and_reasoning
    both = [
        make_trace("R1", "P1", 80.0, {"B1": "satisfied"}, {"B1": ["EV-1"]}),
        make_trace("R2", "P1", 80.0, {"B1": "partial"}, {"B1": ["EV-9"]}),
        make_trace("R3", "P1", 80.0, {"B1": "not_satisfied"}, {"B1": ["EV-5"]}),
    ]
    both_evidence = evidence_variance(both, config)
    both_judgment = judgment_variance(both, config, both_evidence)
    assert both_judgment["source_of_uncertainty"] == "retrieval_and_reasoning", both_judgment
    third = next(item for item in both_judgment["per_rule"] if item["rule_id"] == "B1")
    assert third["entropy"] > 1.5, third
    print(
        f"6) Judgment Variance：B1 分布 {first['distribution']}，一致率 {first['agreement']}，熵 {first['entropy']}"
        f" → 来源 {first['source_of_uncertainty']}；"
        f"证据+判断同不稳 → {both_judgment['source_of_uncertainty']}"
    )
    return {"agent_reasoning": judgment_result, "both": both_judgment}


def step_7_calibration(profile, config: ReliabilityConfig) -> dict:
    rule_ids = [rule.rule_id for rule in profile.rules]
    # 用规则库真实规则构一批"baseline 大多是 partial、bonus 大多是 satisfied"的判断记录：
    # partial 在基准池里是 1.0，换到 Scheme B（partial=0.6）后分数必须明显下降 → calibration_sensitive
    judgements: dict[str, str] = {}
    for rule in profile.rules:
        judgements[rule.rule_id] = "satisfied" if rule.role == "bonus" else "partial"
    evidence = {rule_id: ["EV-1"] for rule_id in rule_ids}
    # 两篇论文必须分数不同，否则 Spearman 没有方差可算（返回 None）。
    # 只差一条**加分**规则：判 uncertain 的那篇不进那条规则的分母，总分略低。
    swap_rule = next(rule.rule_id for rule in profile.rules if rule.role == "bonus")
    # paper2 只把一条加分规则降成 partial：基准项（partial=1.0）不受影响，但总分低于 paper1，
    # 排序才有方差可算（两篇分数完全相同 Spearman 会返回 None，那是"算不出"而不是"稳定"）。
    weaker = {rule_id: ("partial" if rule_id == swap_rule else value) for rule_id, value in judgements.items()}
    traces = {
        "PAPER-1": [make_trace("R1", "PAPER-1", 90.0, judgements, evidence, source_path="b1/R1.json")],
        "PAPER-2": [make_trace("R2", "PAPER-2", 70.0, weaker, evidence, source_path="b1/R2.json")],
    }
    result = calibration_uncertainty(profile, traces, config)
    assert result["checked"] is True, result
    assert result["scheme_count"] == len(CALIBRATION_SCHEMES), result
    assert result["max_abs_score_delta"] > 0, result
    assert result["ranking_correlation"] == 1.0, result
    assert result["mean_top_k_overlap"] == 1.0, result
    assert "calibration_sensitive" in result["flags"], result
    assert result["robustness"] == "low", result
    low_scheme = next(item for item in result["schemes"] if item["scheme"] == "scheme_b_low")
    assert low_scheme["score_delta"] < 0, low_scheme
    assert low_scheme["sensitive"] is True, low_scheme

    # 反过来：加分项普遍判 satisfied 时，换映射对分数的影响必须显著更小
    #（证明这个检测不是恒真——它对"判断分布"有反应）。
    bonus_heavy = {rule.rule_id: "satisfied" for rule in profile.rules}
    bonus_heavy_weaker = {
        rule_id: ("not_satisfied" if rule_id == swap_rule else value) for rule_id, value in bonus_heavy.items()
    }
    flat_traces = {
        "PAPER-1": [make_trace("R1", "PAPER-1", 90.0, bonus_heavy, evidence, source_path="b1/R1.json")],
        "PAPER-2": [make_trace("R2", "PAPER-2", 70.0, bonus_heavy_weaker, evidence, source_path="b1/R2.json")],
    }
    flat = calibration_uncertainty(profile, flat_traces, config)
    assert flat["schemes"], flat
    assert flat["score_difference"] < result["score_difference"], (flat["score_difference"], result["score_difference"])
    print(
        f"7) Calibration Uncertainty：{result['scheme_count']} 套 mapping，最大 Δ总分 "
        f"{result['max_abs_score_delta']}（相对 {result['max_score_delta_ratio']}），"
        f"排序相关 {result['ranking_correlation']}，Top-K 重合 {result['mean_top_k_overlap']}，"
        f"稳健性 {result['robustness']}（方案 {result['sensitive_schemes']}）；"
        f"全 satisfied 批次 Δ总分仅 {flat['score_difference']}，说明检测不是恒真"
    )
    return result


def step_8_diagnosis(run_result: dict, evidence_instability: dict, judgment_result: dict, calibration_result: dict, config: ReliabilityConfig) -> None:
    # 情况 1：运行高波动 + 证据不稳 → retrieval_instability
    case1 = diagnose_reliability(
        {"flags": ["high_variance"], "stability": "low"},
        {"flags": ["evidence_instability"], "unstable_rule_ids": ["A-1"]},
        {"flags": [], "stability": "high"},
        {"checked": False, "flags": []},
        config,
    )
    assert case1["main_uncertainty_source"] == "retrieval_instability", case1
    assert "检索候选" in case1["recommendation"], case1

    # 情况 2：运行波动 + 证据稳 + 判断不稳 → agent_reasoning
    case2 = diagnose_reliability(
        {"flags": ["high_variance"], "stability": "low"},
        {"flags": [], "stability": "high"},
        {"flags": ["judgment_instability"], "unstable_rule_ids": ["B1"]},
        {"checked": False, "flags": []},
        config,
    )
    assert case2["main_uncertainty_source"] == "agent_reasoning", case2
    assert "投票" in case2["recommendation"], case2
    assert case2["needs_review"] == ["B1"], case2

    # 情况 3：运行稳 + 校准高 → calibration_sensitive
    case3 = diagnose_reliability(
        {"flags": [], "stability": "high"},
        {"flags": [], "stability": "high"},
        {"flags": [], "stability": "high"},
        {"checked": True, "flags": ["calibration_sensitive"]},
        config,
    )
    assert case3["main_uncertainty_source"] == "calibration_sensitive", case3
    assert "校准" in case3["recommendation"], case3

    # 情况 4：全稳 → evaluation_reliable（校准检测完成且排序也检过，才允许"零警告"）
    case4 = diagnose_reliability(
        {"flags": [], "stability": "high"},
        {"flags": [], "stability": "high"},
        {"flags": [], "stability": "high"},
        {"checked": True, "flags": [], "robustness": "high", "ranking_correlation_checked": True},
        config,
    )
    assert case4["main_uncertainty_source"] == "evaluation_reliable", case4
    assert case4["warnings"] == [], case4

    # 证据与判断都不稳但总分没波动 → 隐患也要报出来
    case5 = diagnose_reliability(
        {"flags": [], "stability": "high"},
        {"flags": ["evidence_instability"]},
        {"flags": ["judgment_instability"]},
        {"checked": True, "flags": []},
        config,
    )
    assert case5["main_uncertainty_source"] == "retrieval_and_reasoning", case5
    print(
        f"8) Reliability Diagnosis：检索不稳→{case1['main_uncertainty_source']}；"
        f"推理不稳→{case2['main_uncertainty_source']}；规则敏感→{case3['main_uncertainty_source']}；"
        f"全稳→{case4['main_uncertainty_source']}；隐藏隐患→{case5['main_uncertainty_source']}"
    )


def step_9_final_report(profile, run_result: dict, evidence_result: dict, judgment_result: dict, calibration_result: dict, config: ReliabilityConfig) -> dict:
    # 用真实规则 ID：校准重算要按规则库取权重/角色，假 rule_id 会取不到。
    rule_ids = [rule.rule_id for rule in profile.rules[:4]]
    judgements = {rule_id: "mostly" for rule_id in rule_ids}
    traces = [
        make_trace("R1", "PAPER-X", 90.0, judgements, {rule_ids[0]: ["EV-1"]}, grade="好论文"),
        make_trace("R2", "PAPER-X", 92.0, judgements, {rule_ids[0]: ["EV-9"]}, grade="顶刊级"),
    ]
    report = build_reliability_report(profile, traces, config, focus_paper_id="PAPER-X")
    for key in (
        "run_variance",
        "evidence_variance",
        "judgment_variance",
        "calibration_uncertainty",
        "overall_warning",
        "needs_review",
        "reliability_report",
        "diagnosis",
    ):
        assert key in report, key
    assert "total" not in report and "overall_score" not in report, "需求明确：不许合成一个总可靠性分"
    assert set(report["reliability_report"]) >= {
        "run_stability",
        "evidence_stability",
        "judgment_stability",
        "calibration_robustness",
    }, report["reliability_report"]
    assert report["overall_warning"], report
    assert report["needs_review"], report
    markdown = "\n".join(render_reliability_markdown(report))
    for heading in ("## 评价可靠性", "可靠性审计链", "需人工复核", "**建议：**"):
        assert heading in markdown, heading
    assert "总可靠性" not in markdown
    print(
        f"9) 最终报告：四个分项 {report['reliability_report']['run_stability']}/"
        f"{report['reliability_report']['evidence_stability']}/"
        f"{report['reliability_report']['judgment_stability']}/"
        f"{report['reliability_report']['calibration_robustness']}，"
        f"警告 {len(report['overall_warning'])} 条，需复核 {len(report['needs_review'])} 条，"
        f"无总可靠性分"
    )
    return report


def step_10_audit_chain(profile, config: ReliabilityConfig) -> None:
    """审计链：Score → Judgment → Evidence → Reliability Diagnosis 每一环都能落地。"""
    judgements = {"A-1": "mostly", "B-1": "satisfied"}
    evidence = {"A-1": ["EV-1", "EV-2"], "B-1": ["EV-3"]}
    traces = [
        make_trace("R1", "PAPER-AUDIT", 88.0, judgements, evidence, source_path="/runs/x/R1_report.json"),
        make_trace("R2", "PAPER-AUDIT", 89.0, judgements, evidence, source_path="/runs/x/R2_report.json"),
    ]
    report = build_reliability_report(profile, traces, config, focus_paper_id="PAPER-AUDIT")
    # Score → Judgment：报告里能指到每次运行的总分
    assert {item["run_id"] for item in report["runs"]} == {"R1", "R2"}, report["runs"]
    # Judgment → Evidence：judgment_variance 的 per_rule 能指到规则
    assert any(item["rule_id"] == "A-1" for item in report["judgment_variance"]["per_rule"]), report["judgment_variance"]
    # Evidence：evidence_variance 的 per_rule 能指到每条规则的证据交集
    assert any(item["rule_id"] == "A-1" for item in report["evidence_variance"]["per_rule"]), report["evidence_variance"]
    # Reliability Diagnosis：诊断能指到来源与建议
    assert report["diagnosis"]["main_uncertainty_source"], report["diagnosis"]
    assert report["diagnosis"]["recommendation"], report["diagnosis"]
    print(
        f"10) 审计链：{len(report['runs'])} 次运行的 Score 可回查 → "
        f"{len(report['judgment_variance']['per_rule'])} 条 Judgment → "
        f"{len(report['evidence_variance']['per_rule'])} 条 Evidence → 诊断 "
        f"{report['diagnosis']['main_uncertainty_source']}"
    )


def step_11_cli() -> None:
    """离线重算 CLI：用合成报告 JSON 走一遍真实入口。"""
    out_dir = ROOT / "runs" / "_reliability_selftest"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, score in enumerate((90.0, 94.0), start=1):
        report = {
            "run": {
                "run_id": f"RUN-CLI-{index}",
                "paper_id": "PAPER-CLI",
                "harness_version": "harness-0.2",
                "rule_profile": "MGMT-LIT-EVAL-V1:V16",
                "started_at": f"2026-01-0{index}T00:00:00Z",
                "model_versions": {"core_agents": "stub"},
            },
            "paper": {"paper_id": "PAPER-CLI", "metadata": {"title": "CLI fixture"}},
            "summary": {"total_score": score, "dimension_scores": {"A": {"normalized_score": 12.0}}},
            "details": [
                {
                    "rule_id": "A-TOPIC-SIGNIFICANCE-V1",
                    "dimension_id": "A",
                    "judgement": "mostly",
                    "confidence": 0.7,
                    "evidence": [{"evidence_id": f"EV-{index}", "section": "Introduction", "page": 1}],
                    "raw_score": 1.0,
                    "weight": 3.0,
                    "role": "baseline",
                    "needs_review": False,
                    "review_flags": [],
                }
            ],
        }
        (out_dir / f"RUN-CLI-{index}_report.json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                str(ROOT / "scripts" / "可靠性_离线重算.py"),
                "--runs-dir",
                str(out_dir),
                "--out",
                str(out_dir / "out"),
                "--print",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        # --print 会先输出完整 JSON，再输出一行摘要，所以只解码第一个 JSON 对象
        payload, _ = json.JSONDecoder().raw_decode(result.stdout[result.stdout.index("{") :])
        assert payload["analyzed_run_count"] == 2, payload["analyzed_run_count"]
        assert payload["run_variance"]["range"] == 4.0, payload["run_variance"]
        assert (out_dir / "out").glob("Reliability_*.json"), "未生成可靠性报告文件"
        print(
            f"11) 离线重算 CLI：读到 {payload['analyzed_run_count']} 次运行，"
            f"极差 {payload['run_variance']['range']}，稳定性 {payload['run_variance']['stability']}，"
            f"并写出独立报告文件"
        )
    finally:
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()


def backtests(profile, config: ReliabilityConfig) -> None:
    if not BACKTEST_RUNS.exists():
        print(f"    （跳过真实数据回测：找不到 {BACKTEST_RUNS}）")
        return
    from literature_eval.reliability import discover_report_paths, load_traces_from_reports

    paths: list[Path] = []
    for name in ("v23_ef_3papers", "v23b_ef_repeat_3papers", "v23c_ef_3papers", "v23d_ef_3papers"):
        paths.extend(discover_report_paths(BACKTEST_RUNS / name))
    traces = load_traces_from_reports(paths)
    assert len(traces) == 12, len(traces)
    report = build_reliability_report(profile, traces, config)
    run = report["run_variance"]
    assert run["scored_run_count"] == 4, run
    assert abs(run["range"] - 5.25) < 0.01, run
    assert run["grade_flip"] is True, run
    assert report["evidence_variance"]["evidence_overlap"] is not None
    assert report["judgment_variance"]["mean_agreement"] is not None
    assert report["calibration_uncertainty"]["checked"] is True
    assert report["diagnosis"]["main_uncertainty_source"] in {
        "agent_reasoning",
        "retrieval_instability",
        "retrieval_and_reasoning",
        "calibration_sensitive",
    }, report["diagnosis"]
    print(
        f"12) 真实数据回测（v23 四批 × 3 篇）：总分区间 {run['min_score']}–{run['max_score']}、"
        f"极差 {run['range']}、等级翻转 {run['grade_flip_detail']}；"
        f"证据交集 {report['evidence_variance']['evidence_overlap']}；"
        f"判断一致率 {report['judgment_variance']['mean_agreement']}；"
        f"不确定性来源 {report['diagnosis']['main_uncertainty_source']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtests", action="store_true", help="额外跑真实 v23 四批数据")
    args = parser.parse_args()

    profile = load_rule_profile(RULES_PATH)
    config = step_1_config()
    run_result = step_2_run_variance(config)
    step_4_ranking(config)
    evidence_instability = step_5_evidence(config)
    judgment_result = step_6_judgment(config)
    calibration_result = step_7_calibration(profile, config)
    step_8_diagnosis(run_result, evidence_instability, judgment_result, calibration_result, config)
    step_9_final_report(profile, run_result, evidence_instability, judgment_result, calibration_result, config)
    step_10_audit_chain(profile, config)
    step_11_cli()
    if args.backtests:
        backtests(profile, config)
    print("Evaluation Reliability Layer 自检通过（全程离线，未重跑任何 Agent）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

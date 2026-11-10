'Evaluation Reliability Layer self-test (entirely offline, no re-running Agent, no external API).\n\nCoverage:\n1. Configurable threshold reading (`reliability:` segment + unknown key error)\n2. Run Variance: Total score mean/standard deviation/range, dimension fluctuation, level flip → high_variance\n3. A single run must report insufficient_data honestly (don’t pretend to calculate it)\n4. Batch ranking stability: Change order of batches → ranking_change; Same order batches → high\n5. Evidence Variance: The evidence is completely consistent → overlap 1.0; completely replaced → evidence_instability\n6. Judgment Variance: consistency rate, distribution, entropy; evidence is stable but judgment is unstable → agent_reasoning\n7. Calibration Uncertainty: Mapping recalculation / sorting related / Top-K coincidence / calibration_sensitive\n8. Reliability Diagnosis Attribution and suggestions for four situations\n9. Final Reliability Report Structure + **Excluding Total Reliability Score**\n10. Audit chain: Score → Judgment → Evidence → Reliability Diagnosis\n11. Offline recalculation CLI end-to-end (real report JSON go through)\n\nRun:\n    python ".\\scripts\\self_test_reliability.py"\n    python ".\\scripts\\self_test_reliability.py" --backtests # Run four additional batches of real v23 data (if existing)'

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_eval.reliability import (  # noqa: E402
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
from paper_eval.reliability_diagnosis import (  # noqa: E402
    CALIBRATION_SCHEMES,
    build_reliability_report,
    calibration_uncertainty,
    diagnose_reliability,
    render_reliability_markdown,
)
from paper_eval.rules import load_rule_profile  # noqa: E402

RULES_PATH = ROOT / 'rules' / "rules_v1.yaml"
CONFIG_PATH = ROOT / "configs" / "agent_runtime.deepseek.yaml"
BACKTEST_RUNS = Path('D:\\Entrepreneurship\\agent document scoring\\initial document\\article reviewer\\runs')


def make_trace(
    run_id: str,
    paper_id: str,
    total: float,
    judgements: dict[str, str],
    evidence: dict[str, list[str]],
    dimension_scores: dict[str, float] | None = None,
    grade: str = 'Good paper',
    potential: str = 'Medium',
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
    assert ReliabilityConfig().enabled is False, 'Must be turned off by default: when not enabled, the evaluation process remains unchanged bit by bit.'
    try:
        load_reliability_config(ROOT / "configs" / "agent_runtime.cloud.yaml")
    except ValueError:
        raise AssertionError('cloud configuration does not have a reliability section and no error should be reported')
    # Unknown keys must report an error instead of silently ignoring it (wrongly written configuration will make people think that the threshold has taken effect)
    temp = Path(tempfile.gettempdir()) / "_reliability_unknown_key.yaml"
    temp.write_text("reliability:\n  run_variance_threshold: 5.0\n  typo_key: 1\n", encoding="utf-8")
    try:
        load_reliability_config(temp)
        raise AssertionError('Unknown keys must report an error')
    except ValueError:
        pass
    finally:
        temp.unlink(missing_ok=True)
    print(
        f"1) Configurable threshold: run_variance_threshold={config.run_variance_threshold},"
        f"evidence_overlap_threshold={config.evidence_overlap_threshold}、"
        f"judgment_agreement_threshold={config.judgment_agreement_threshold}; unknown key error; closed by default"
    )
    return config


def step_2_run_variance(config: ReliabilityConfig) -> dict:
    judgements = {f"A-R{i}-V1": "mostly" for i in range(4)}
    evidence = {rule_id: [f"EV-{index}"] for index, rule_id in enumerate(judgements)}
    # stable batch: only 1 point difference in total score
    stable = [
        make_trace("RUN-S1", "PAPER-STABLE", 90.0, judgements, evidence, grade='Good paper'),
        make_trace("RUN-S2", "PAPER-STABLE", 91.0, judgements, evidence, grade='Good paper'),
    ]
    stable_result = run_variance(stable, config)
    assert stable_result["stability"] == "high", stable_result
    assert stable_result["range"] == 1.0 and stable_result["grade_flip"] is False, stable_result

    # Fluctuating batch: total score difference is 8 points and grade flipped
    volatile = [
        make_trace("RUN-V1", "PAPER-VOLATILE", 88.0, judgements, evidence, grade='Good paper'),
        make_trace("RUN-V2", "PAPER-VOLATILE", 96.0, judgements, evidence, grade='Top publication level'),
    ]
    volatile_result = run_variance(volatile, config)
    assert "high_variance" in volatile_result["flags"], volatile_result
    assert volatile_result["stability"] == "low", volatile_result
    assert volatile_result["grade_flip"] is True, volatile_result
    assert volatile_result["mean_score"] == 92.0 and volatile_result["std"] == 4.0, volatile_result

    # Dimension fluctuation: C dimension differs by 5 points between two runs
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
        f"2-3) Run Variance: Stable batch stability {stable_result['stability']} (extremely poor {stable_result['range']});"
        f"Fluctuating batch {volatile_result['stability']} (extremely poor {volatile_result['range']}, level flipped"
        f"{volatile_result['grade_flip_detail']}); Dimension C fluctuation was caught;"
        f"Honest report for a single operation {single['stability']}"
    )
    return {"volatile": volatile_result, "stable": stable_result, "dimension": dimension_result}


def step_4_ranking(config: ReliabilityConfig) -> None:
    # 3 papers in the same batch directory, the two batches give the same ranking → high
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

    # The second batch reverses the order → ranking_change
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
        f"4) Batch sorting: batches in the same order Spearman {stable_ranking['mean_spearman']};"
        f"reverse batch Spearman {flip_ranking['mean_spearman']} → ranking_change;"
        f"Top-{stable_ranking['top_k']} overlap {stable_ranking['top_k_overlap']}"
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
        f"5) Evidence Variance: The evidence is completely consistent → Intersection {identical_result['evidence_overlap']};"
        f"Whole batch of evidence → intersection {moved_result['evidence_overlap']}, marked {moved_result['flags']}"
    )
    return moved_result


def step_6_judgment(config: ReliabilityConfig) -> dict:
    stable_evidence = {"B1": ["EV-1"], "B2": ["EV-2"]}
    # The evidence is stable and the judgment wavers between satisfied/partial → agent_reasoning
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

    # Evidence and judgment are unstable at the same time → retrieval_and_reasoning
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
        f"6) Judgment Variance: B1 distribution {first['distribution']}, consistency rate {first['agreement']}, entropy {first['entropy']}"
        f"→ source {first['source_of_uncertainty']};"
        f"Evidence + judgment are both unstable → {both_judgment['source_of_uncertainty']}"
    )
    return {"agent_reasoning": judgment_result, "both": both_judgment}


def step_7_calibration(profile, config: ReliabilityConfig) -> dict:
    rule_ids = [rule.rule_id for rule in profile.rules]
    # Use the real rules of the rule base to construct a batch of judgment records of "baseline is mostly partial, bonus is mostly satisfied":
    # partial is 1.0 in the benchmark pool. After switching to Scheme B (partial=0.6), the score must drop significantly → calibration_sensitive
    judgements: dict[str, str] = {}
    for rule in profile.rules:
        judgements[rule.rule_id] = "satisfied" if rule.role == "bonus" else "partial"
    evidence = {rule_id: ["EV-1"] for rule_id in rule_ids}
    # The two papers must have different scores, otherwise Spearman has no variance to calculate (returns None).
    # There is only one **extra point** rule missing: the article that is judged as uncertain does not enter the denominator of that rule, and the total score is slightly lower.
    swap_rule = next(rule.rule_id for rule in profile.rules if rule.role == "bonus")
    # paper2 only reduces one bonus point rule to partial: the benchmark item (partial=1.0) is not affected, but the total score is lower than paper1,
# Only
    # sorting can calculate the variance (the scores of the two articles are exactly the same and Spearman will return None, which means "cannot be calculated" rather than "stable").
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

    # On the other hand: when the extra points are generally satisfied, the impact of changing the mapping on the score must be significantly smaller
    #(proves that this test is not always true - it responds to the "judgment distribution").
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
        f"7) Calibration Uncertainty: {result['scheme_count']} set of mapping, maximum Δ total score"
        f"{result['max_abs_score_delta']} (relative to {result['max_score_delta_ratio']}),"
        f"sorting related {result['ranking_correlation']}, Top-K overlap {result['mean_top_k_overlap']},"
        f"Robustness {result['robustness']} (scheme {result['sensitive_schemes']});"
        f"is fully satisfied. The total score of the batch is only {flat['score_difference']}, indicating that the test is not constant."
    )
    return result


def step_8_diagnosis(run_result: dict, evidence_instability: dict, judgment_result: dict, calibration_result: dict, config: ReliabilityConfig) -> None:
    # Case 1: High operational volatility + unstable evidence → retrieval_instability
    case1 = diagnose_reliability(
        {"flags": ["high_variance"], "stability": "low"},
        {"flags": ["evidence_instability"], "unstable_rule_ids": ["A-1"]},
        {"flags": [], "stability": "high"},
        {"checked": False, "flags": []},
        config,
    )
    assert case1["main_uncertainty_source"] == "retrieval_instability", case1
    assert 'Search Candidates' in case1["recommendation"], case1

    # Situation 2: Operation fluctuation + stable evidence + unstable judgment → agent_reasoning
    case2 = diagnose_reliability(
        {"flags": ["high_variance"], "stability": "low"},
        {"flags": [], "stability": "high"},
        {"flags": ["judgment_instability"], "unstable_rule_ids": ["B1"]},
        {"checked": False, "flags": []},
        config,
    )
    assert case2["main_uncertainty_source"] == "agent_reasoning", case2
    assert 'vote' in case2["recommendation"], case2
    assert case2["needs_review"] == ["B1"], case2

    # Case 3: Stable operation + high calibration → calibration_sensitive
    case3 = diagnose_reliability(
        {"flags": [], "stability": "high"},
        {"flags": [], "stability": "high"},
        {"flags": [], "stability": "high"},
        {"checked": True, "flags": ["calibration_sensitive"]},
        config,
    )
    assert case3["main_uncertainty_source"] == "calibration_sensitive", case3
    assert 'Calibration' in case3["recommendation"], case3

    # Case 4: Fully stable → evaluation_reliable ("zero warning" is allowed only after the calibration test is completed and the sorting is also checked)
    case4 = diagnose_reliability(
        {"flags": [], "stability": "high"},
        {"flags": [], "stability": "high"},
        {"flags": [], "stability": "high"},
        {"checked": True, "flags": [], "robustness": "high", "ranking_correlation_checked": True},
        config,
    )
    assert case4["main_uncertainty_source"] == "evaluation_reliable", case4
    assert case4["warnings"] == [], case4

    # The evidence and judgment are unstable but the total score does not fluctuate → Hidden risks must also be reported
    case5 = diagnose_reliability(
        {"flags": [], "stability": "high"},
        {"flags": ["evidence_instability"]},
        {"flags": ["judgment_instability"]},
        {"checked": True, "flags": []},
        config,
    )
    assert case5["main_uncertainty_source"] == "retrieval_and_reasoning", case5
    print(
        f"8) Reliability Diagnosis: Retrieval is unstable → {case1['main_uncertainty_source']};"
        f"Unstable reasoning→{case2['main_uncertainty_source']}; Rule sensitive→{case3['main_uncertainty_source']};"
        f"Fully stable→{case4['main_uncertainty_source']}; hidden danger→{case5['main_uncertainty_source']}"
    )


def step_9_final_report(profile, run_result: dict, evidence_result: dict, judgment_result: dict, calibration_result: dict, config: ReliabilityConfig) -> dict:
    # Use real rule ID: Calibration recalculation requires weight/role according to the rule base, false rule_id will not be obtained.
    rule_ids = [rule.rule_id for rule in profile.rules[:4]]
    judgements = {rule_id: "mostly" for rule_id in rule_ids}
    traces = [
        make_trace("R1", "PAPER-X", 90.0, judgements, {rule_ids[0]: ["EV-1"]}, grade='Good paper'),
        make_trace("R2", "PAPER-X", 92.0, judgements, {rule_ids[0]: ["EV-9"]}, grade='Top publication level'),
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
    assert "total" not in report and "overall_score" not in report, 'The requirement is clear: it is not allowed to combine a total reliability score'
    assert set(report["reliability_report"]) >= {
        "run_stability",
        "evidence_stability",
        "judgment_stability",
        "calibration_robustness",
    }, report["reliability_report"]
    assert report["overall_warning"], report
    assert report["needs_review"], report
    markdown = "\n".join(render_reliability_markdown(report))
    for heading in ('## Evaluation reliability', 'Reliability Audit Chain', 'requires manual review', '**Suggestion:**'):
        assert heading in markdown, heading
    assert 'Total Reliability' not in markdown
    print(
        f"9) Final report: four sub-items {report['reliability_report']['run_stability']}/"
        f"{report['reliability_report']['evidence_stability']}/"
        f"{report['reliability_report']['judgment_stability']}/"
        f"{report['reliability_report']['calibration_robustness']}，"
        f"warning {len(report['overall_warning'])} items, need to review {len(report['needs_review'])} items,"
        f"No overall reliability score"
    )
    return report


def step_10_audit_chain(profile, config: ReliabilityConfig) -> None:
    'Audit chain: Score → Judgment → Evidence → Reliability Diagnosis Each link can be implemented.'
    judgements = {"A-1": "mostly", "B-1": "satisfied"}
    evidence = {"A-1": ["EV-1", "EV-2"], "B-1": ["EV-3"]}
    traces = [
        make_trace("R1", "PAPER-AUDIT", 88.0, judgements, evidence, source_path="/runs/x/R1_report.json"),
        make_trace("R2", "PAPER-AUDIT", 89.0, judgements, evidence, source_path="/runs/x/R2_report.json"),
    ]
    report = build_reliability_report(profile, traces, config, focus_paper_id="PAPER-AUDIT")
    # Score → Judgment: The report can refer to the total score of each run
    assert {item["run_id"] for item in report["runs"]} == {"R1", "R2"}, report["runs"]
    # Judgment → Evidence: per_rule of judgment_variance can point to the rule
    assert any(item["rule_id"] == "A-1" for item in report["judgment_variance"]["per_rule"]), report["judgment_variance"]
    # Evidence: per_rule of evidence_variance can point to the intersection of evidence for each rule
    assert any(item["rule_id"] == "A-1" for item in report["evidence_variance"]["per_rule"]), report["evidence_variance"]
    # Reliability Diagnosis: Diagnosis points to sources and recommendations
    assert report["diagnosis"]["main_uncertainty_source"], report["diagnosis"]
    assert report["diagnosis"]["recommendation"], report["diagnosis"]
    print(
        f"10) Audit chain: Score of {len(report['runs'])} runs can be reviewed →"
        f"{len(report['judgment_variance']['per_rule'])} Judgment →"
        f"{len(report['evidence_variance']['per_rule'])} Evidence → Diagnosis"
        f"{report['diagnosis']['main_uncertainty_source']}"
    )


def step_11_cli() -> None:
    'Offline recalculation CLI: Use synthetic report JSON to walk through the real entrance.'
    out_dir = ROOT / "runs" / "_reliability_selftest"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, score in enumerate((90.0, 94.0), start=1):
        report = {
            "run": {
                "run_id": f"RUN-CLI-{index}",
                "paper_id": "PAPER-CLI",
                "harness_version": "harness-0.2",
                "rule_profile": "expert_calibrated_scoring_rules",
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
                str(ROOT / "scripts" / 'reliability_offline recalculation.py'),
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
        # --print will output the complete JSON first, and then output a one-line summary, so only the first JSON object will be decoded
        payload, _ = json.JSONDecoder().raw_decode(result.stdout[result.stdout.index("{") :])
        assert payload["analyzed_run_count"] == 2, payload["analyzed_run_count"]
        assert payload["run_variance"]["range"] == 4.0, payload["run_variance"]
        assert (out_dir / "out").glob("Reliability_*.json"), 'Reliability report file not generated'
        print(
            f"11) Offline recalculation CLI: Read {payload['analyzed_run_count']} times,"
            f"Extremely poor {payload['run_variance']['range']}, stability {payload['run_variance']['stability']},"
            f"and write an independent report document"
        )
    finally:
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()


def backtests(profile, config: ReliabilityConfig) -> None:
    if not BACKTEST_RUNS.exists():
        print(f"(Skip real data backtest: {BACKTEST_RUNS} not found)")
        return
    from paper_eval.reliability import discover_report_paths, load_traces_from_reports

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
        f"12) Real data backtest (v23 four batches × 3 articles): Total partition interval {run['min_score']}–{run['max_score']},"
        f"extremely poor {run['range']}, level flip {run['grade_flip_detail']};"
        f"Evidence intersection {report['evidence_variance']['evidence_overlap']};"
        f"Judgment consistency rate {report['judgment_variance']['mean_agreement']};"
        f"Source of uncertainty {report['diagnosis']['main_uncertainty_source']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtests", action="store_true", help='runs four additional batches of real v23 data')
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
    print('Evaluation Reliability Layer self-test passed (the whole process was offline, no Agent was re-run).')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

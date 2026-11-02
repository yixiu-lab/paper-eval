"""Reproduce the deterministic Harness fault-injection study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import groupby
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "02-system"))

from paper_eval.evidence_gate import apply_evidence_gate  # noqa: E402
from paper_eval.models import AgentJudgement, Evidence, Judgement  # noqa: E402
from paper_eval.rules import DimensionRule, IndicatorRule, RuleProfile  # noqa: E402
from paper_eval.schema_validation import validate_agent_judgements  # noqa: E402


REPEATS_PER_FAULT = 20
CLEAN_CONTROLS = 40
FAULTS = (
    "unknown_evidence_id",
    "positive_without_evidence",
    "unknown_rule_id",
    "duplicate_rule_judgement",
    "missing_rule_judgement",
    "invalid_judgement_label",
    "confidence_out_of_range",
    "empty_reason",
    "reason_judgement_conflict",
)


def make_profile() -> RuleProfile:
    dimensions = [
        DimensionRule(letter, f"Dimension {letter}", 1 / 6, "Fixed fault-test dimension")
        for letter in "ABCDEF"
    ]
    rules = [
        IndicatorRule(
            rule_id=f"{letter}-01",
            version="fault-test",
            dimension_id=letter,
            dimension_name=f"Dimension {letter}",
            indicator_name=f"Test criterion {letter}",
            definition="A synthetic criterion used to exercise Harness validation.",
            required_evidence="At least one traceable evidence item.",
            evidence_location=["paper body"],
            observation_points=["judgment and evidence reference"],
            judgement_values={item.value: item.value for item in Judgement},
            score_mapping={item.value: 1.0 for item in Judgement},
            weight=1 / 6,
            applicable_type="General",
            source_reference=[],
            status="active",
        )
        for letter in "ABCDEF"
    ]
    return RuleProfile(
        rule_profile_id="fault-injection-profile",
        rule_profile_name="Harness fault-injection profile",
        version="1",
        status="active",
        scoring_scale=100,
        dimensions=dimensions,
        rules=rules,
    )


def make_evidence(index: int, page: int, letter: str) -> Evidence:
    topics = {
        "A": "research questions and gaps",
        "B": "theoretical mechanisms and contributions",
        "C": "constructs and hypothesis logic",
        "D": "methods, samples, and analysis",
        "E": "findings, robustness, and hypothesis tests",
        "F": "discussion, boundaries, and limitations",
    }
    text = (f"The paper describes {topics[letter]} with criterion-specific detail. " * 8)[:220]
    return Evidence(
        evidence_id=f"EV-{index:03d}",
        paper_id="PAPER-FAULT-TEST",
        section="Methods" if letter == "D" else ("Results" if letter == "E" else "Theory"),
        page=page,
        span=(0, len(text)),
        text=text,
        text_hash=f"hash-{index:03d}",
    )


def base_case(repeat: int) -> tuple[list[AgentJudgement], list[Evidence]]:
    evidence = [make_evidence(1, 3, "B"), make_evidence(2, 5, "B")]
    judgments = [
        AgentJudgement(
            task_id=f"TASK-{repeat:03d}-{letter}",
            agent_id=f"Reviewer-{letter}",
            indicator_id=f"{letter}-01",
            judgement=Judgement.MOSTLY,
            evidence_ids=["EV-001", "EV-002"],
            reason="The cited evidence supports this judgment.",
            confidence=0.8,
            needs_review=False,
            agent_version="fault-test",
            model_version="fixed-fixture",
        )
        for letter in "ABCDEF"
    ]
    return judgments, evidence


def inject_fault(fault: str, judgments: list[AgentJudgement], repeat: int) -> list[AgentJudgement]:
    target = judgments[1]
    if fault == "unknown_evidence_id":
        target.evidence_ids = [f"EV-MISSING-{repeat:03d}"]
    elif fault == "positive_without_evidence":
        target.evidence_ids = []
    elif fault == "unknown_rule_id":
        target.indicator_id = f"B-UNKNOWN-{repeat:03d}"
    elif fault == "duplicate_rule_judgement":
        judgments.append(target)
    elif fault == "missing_rule_judgement":
        judgments = [item for item in judgments if item.indicator_id != "B-01"]
    elif fault == "invalid_judgement_label":
        target.judgement = "excellent"  # type: ignore[assignment]
    elif fault == "confidence_out_of_range":
        target.confidence = 1.2
    elif fault == "empty_reason":
        target.reason = "   "
    elif fault == "reason_judgement_conflict":
        target.judgement = Judgement.SATISFIED
        target.reason = "The available evidence is insufficient to support this conclusion."
    else:
        raise ValueError(f"Unknown fault type: {fault}")
    return judgments


def run_case(profile: RuleProfile, case_id: str, fault: str | None, repeat: int) -> dict[str, object]:
    judgments, evidence = base_case(repeat)
    if fault:
        judgments = inject_fault(fault, judgments, repeat)

    errors = validate_agent_judgements(profile, judgments)
    blocked = bool(errors)
    gated = [] if blocked else apply_evidence_gate(judgments, evidence)
    review_needed = any(item.needs_review for item in gated)
    changed = any(
        item.judgement != original.judgement or item.evidence_ids != original.evidence_ids
        for item, original in zip(gated, judgments)
    )
    detected = blocked or review_needed or changed
    contained = blocked or changed
    return {
        "case_id": case_id,
        "fault_type": fault or "clean_control",
        "injected_error": fault is not None,
        "harness_off_detected": False,
        "harness_detected": detected,
        "harness_blocked": blocked,
        "needs_review": review_needed,
        "automatically_contained": contained,
        "flagged_for_review_only": bool(fault) and not contained and review_needed,
        "gate_changed_output": changed,
        "schema_error_count": len(errors),
        "schema_errors": " | ".join(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reproduced" / "harness-fault-injection",
    )
    args = parser.parse_args()
    profile = make_profile()
    rows = []
    for fault in FAULTS:
        for repeat in range(1, REPEATS_PER_FAULT + 1):
            rows.append(run_case(profile, f"{fault}-{repeat:02d}", fault, len(rows) + 1))
    for repeat in range(1, CLEAN_CONTROLS + 1):
        rows.append(run_case(profile, f"clean-{repeat:02d}", None, len(rows) + 1))

    faulty = [row for row in rows if row["injected_error"]]
    clean = [row for row in rows if not row["injected_error"]]
    count_detected = sum(bool(row["harness_detected"]) for row in faulty)
    count_contained = sum(bool(row["automatically_contained"]) for row in faulty)
    count_review_only = sum(bool(row["flagged_for_review_only"]) for row in faulty)
    false_positives = sum(bool(row["harness_detected"]) for row in clean)
    summary = {
        "fault_types": list(FAULTS),
        "faulty_cases": len(faulty),
        "clean_controls": len(clean),
        "total_cases": len(rows),
        "harness_off_detection_rate": 0.0,
        "fault_detection_rate": count_detected / len(faulty),
        "automatic_containment_rate": count_contained / len(faulty),
        "review_only_rate": count_review_only / len(faulty),
        "clean_control_false_positive_rate": false_positives / len(clean),
        "by_fault": {},
    }
    for fault, group in groupby(
        sorted(faulty, key=lambda row: str(row["fault_type"])), key=lambda row: row["fault_type"]
    ):
        items = list(group)
        summary["by_fault"][fault] = {
            "n": len(items),
            "detected": sum(bool(row["harness_detected"]) for row in items),
            "automatically_contained": sum(bool(row["automatically_contained"]) for row in items),
            "review_only": sum(bool(row["flagged_for_review_only"]) for row in items),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "case-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "by_fault"}, indent=2))


if __name__ == "__main__":
    main()

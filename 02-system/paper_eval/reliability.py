'Evaluation Reliability Layer: Determine "whether the evaluation result itself is stable and credible."\n\nPositioning (must be stated clearly, otherwise it can easily be misused):\n\n- **Pure post-processing**. It does not participate in the judgment of paper quality, does not change the scoring caliber of `scorer.py`, and does not modify the output of any existing Agent.\n  It does not affect the return value structure of `harness.evaluate_pdf` - it only reads the product after the scoring is completed to calculate stability.\n- **Does not call any models**. The entire layer is deterministic: the same batch of runs is recalculated and the conclusion is the same byte by byte.\n  This is not "asking another Agent to evaluate again", but "using the existing judgment records for variance analysis".\n- **No total score**. Operational stability/evidence stability/judgment stability/parameter robustness are issues in four different dimensions.\n  Forcing a number into one simply introduces a new set of subjective weights. The report only gives four sub-items + warning list + items that need to be reviewed.\n\nFour types of detection (inputs are all from existing products, see `RunTrace`):\n\n1. `run_variance`: Is the total score/dimension/ranking of the same paper run multiple times stable (`run_variance`)\n2. `evidence_variance`: Is the recall of rule → evidence_ids stable in multiple runs (`evidence_variance`)\n3. `judgment_variance`: Distribution and consistency rate of Agent judgment gears on the same rule (`judgment_variance`)\n4. `calibration_uncertainty`: Change the gear mapping and recalculate the points, and check whether the sorting is reversed (see reliability_diagnosis.py)\n\nThe evidence is stable but the judgment is unstable → `agent_reasoning`; both are unstable → `retrieval_and_reasoning`.'

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

import yaml

RELIABILITY_VERSION = "reliability-0.1"

# stability level → 0-1 performance points. It is only used to give readers a large number in reports and does not participate in any weighted synthesis.
STABILITY_SCORE = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
    "insufficient_data": None,
    "not_checked": None,
}

DIMENSION_IDS = ("A", "B", "C", "D", "E", "F")


@dataclass(frozen=True)
class ReliabilityConfig:
    'All thresholds are configurable. The default value is taken from measured fluctuation magnitudes that have been recorded in the project.'

    enabled: bool = False
    # When the same paper is run less than this number of times, the variances in the three blocks of running/evidence/judgment will be marked as insufficient_data (don’t pretend to calculate it)
    min_runs: int = 2
    # Total score range threshold (points). The rule base has a perfect score of 100.
    # Actual measurement basis: Choudhury repeated four batches of runs 96.63 / 91.38 / 93.44 / 92.31, a very poor score of 5.25 points - that level is enough to flip
    # is used to determine the level of "top publication level/strong publication edge", so the default threshold is set at 5.0 points, which is of the same magnitude as the actual measured noise.
    run_variance_threshold: float = 5.0
    # Dimension score extreme difference threshold (points, absolute score before conversion from dimension full score)
    dimension_variance_threshold: float = 4.0
    # Cross-run paper sorting correlation threshold (Spearman): below it, the relative order of multiple papers is considered unstable
    ranking_correlation_threshold: float = 0.8
    # Evidence intersection threshold: lower than it considers Evidence Router/retrieval unstable
    evidence_overlap_threshold: float = 0.6
    # Tolerance for variation in the number of evidence pieces (coefficient of variation)
    evidence_coverage_cv_threshold: float = 0.25
    # Judgment consistency rate threshold: lower than it considers the Agent’s judgment on the same rule to be unstable
    judgment_agreement_threshold: float = 0.8
    # Judgment distribution entropy threshold (bit, 5th gear maximum 2.32): if it exceeds it, the gears are considered dispersed
    judgment_entropy_threshold: float = 1.0
    # Calibration sensitivity: Relative change in total score (|Δ| / full score) exceeds what it considers sensitive to mapping
    calibration_score_delta_threshold: float = 0.05
    # Calibration sensitivity: cross-scheme ranking correlation is lower than it considers ranking unstable
    calibration_ranking_correlation_threshold: float = 0.9
    # Calibration sensitivity: The Top-K set coincidence rate is lower than it considers the candidate set unstable
    calibration_topk_overlap_threshold: float = 0.8
    # Top-K Select a few papers (sorting stability in batches of multiple papers)
    top_k: int = 5


def load_reliability_config(path: str | Path | None) -> ReliabilityConfig:
    'Reads the `reliability:` section from the top level of the configuration file.\n\n    Same as `external_retrieval`, deliberately placed outside `agent_runtime`: `AgentRuntimeConfig`\n    The keys of the paragraph will be directly expanded into construction parameters. If there is an unknown key, an error will be reported.'
    if not path:
        return ReliabilityConfig()
    config_path = Path(path)
    if not config_path.exists():
        return ReliabilityConfig()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    values = data.get("reliability") or {}
    allowed = {item.name for item in fields(ReliabilityConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"reliability configuration contains unknown key: {unknown}")
    return ReliabilityConfig(**values)


# --------------------------------------------------------------------------------------
# Run trace: a complete record of each judgment in an evaluation (the only input unit of the reliability layer)
# --------------------------------------------------------------------------------------

@dataclass
class RuleRecord:
    rule_id: str
    dimension_id: str
    judgement: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    raw_score: float | None = None
    weight: float = 0.0
    role: str = "baseline"
    needs_review: bool = False
    review_flags: list[str] = field(default_factory=list)
    evidence_sections: list[str] = field(default_factory=list)
    evidence_pages: list[int] = field(default_factory=list)


@dataclass
class RunTrace:
    'Judgment record + score record for one run. Trace convention in field alignment requirements.'

    run_id: str
    paper_id: str
    rule_records: list[RuleRecord] = field(default_factory=list)
    total_score: float | None = None
    dimension_scores: dict[str, float | None] = field(default_factory=dict)
    quality_grade: str = ""
    top_journal_potential: str = ""
    scoring_completeness: float | None = None
    rule_profile: str = ""
    model_version: str = ""
    started_at: str = ""
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "paper_id": self.paper_id,
            "total_score": self.total_score,
            "dimension_scores": self.dimension_scores,
            "quality_grade": self.quality_grade,
            "top_journal_potential": self.top_journal_potential,
            "scoring_completeness": self.scoring_completeness,
            "rule_profile": self.rule_profile,
            "model_version": self.model_version,
            "started_at": self.started_at,
            "source_path": self.source_path,
            "rule_records": [
                {
                    "rule_id": item.rule_id,
                    "dimension_id": item.dimension_id,
                    "evidence_ids": list(item.evidence_ids),
                    "judgment": item.judgement,
                    "confidence": item.confidence,
                    "score": item.raw_score,
                    "weight": item.weight,
                    "role": item.role,
                    "needs_review": item.needs_review,
                }
                for item in self.rule_records
            ],
        }


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def run_trace_from_report(report: dict[str, Any], source_path: str | Path = "") -> RunTrace:
    'Build RunTrace from the report JSON output by `build_report()`.\n\n    The data source is the existing product: `details[]` contains rule_id / judgment / confidence / evidence[],\n    There are total scores and dimensional scores in `summary`. **Do not change report format**, read only.'
    run = report.get("run") or {}
    paper = report.get("paper") or {}
    summary = report.get("summary") or {}
    details = report.get("details") or []
    records: list[RuleRecord] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        evidence = [ev for ev in (item.get("evidence") or []) if isinstance(ev, dict)]
        records.append(
            RuleRecord(
                rule_id=str(item.get("rule_id") or ""),
                dimension_id=str(item.get("dimension_id") or ""),
                judgement=str(item.get("judgement") or ""),
                confidence=float(item.get("confidence") or 0.0),
                evidence_ids=[str(ev.get("evidence_id") or "") for ev in evidence],
                raw_score=_float_or_none(item.get("raw_score")),
                weight=float(item.get("weight") or 0.0),
                role=str(item.get("role") or "baseline"),
                needs_review=bool(item.get("needs_review")),
                review_flags=[str(flag) for flag in (item.get("review_flags") or [])],
                evidence_sections=[str(ev.get("section") or "") for ev in evidence],
                evidence_pages=[int(ev.get("page") or 0) for ev in evidence],
            )
        )
    dimension_scores = {
        str(key): _float_or_none((value or {}).get("normalized_score"))
        for key, value in (summary.get("dimension_scores") or {}).items()
        if isinstance(value, dict)
    }
    model_versions = run.get("model_versions") or {}
    return RunTrace(
        run_id=str(run.get("run_id") or ""),
        paper_id=str(paper.get("paper_id") or ""),
        rule_records=records,
        total_score=_float_or_none(summary.get("total_score")),
        dimension_scores=dimension_scores,
        quality_grade=str(summary.get("quality_grade") or ""),
        top_journal_potential=str(summary.get("top_journal_potential") or ""),
        scoring_completeness=_float_or_none(summary.get("scoring_completeness")),
        rule_profile=str(run.get("rule_profile") or ""),
        model_version=str(model_versions.get("core_agents") or ""),
        started_at=str(run.get("started_at") or ""),
        source_path=str(source_path),
    )


def load_traces_from_reports(paths: Iterable[str | Path]) -> list[RunTrace]:
    traces: list[RunTrace] = []
    for path in paths:
        resolved = Path(path)
        try:
            report = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict) or "details" not in report:
            continue
        trace = run_trace_from_report(report, resolved)
        if trace.run_id:
            traces.append(trace)
    return traces


def discover_report_paths(root: str | Path, pattern: str = "*_report.json") -> list[Path]:
    'Recursive collection report JSON. Failure reports (those without details) are skipped by default.'
    directory = Path(root)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob(pattern) if path.is_file())


def load_traces_from_db(db_path: str | Path, paper_id: str | None = None) -> list[RunTrace]:
    'reads the run trace from the SQLite runtime library of `storage.py` (another entry for offline recalculation).\n\n    Read only, do not write to the library; `agent_judgements` / `score_items` already has all the required fields.'
    path = Path(db_path)
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        runs = conn.execute(
            "SELECT * FROM evaluation_runs" + (" WHERE paper_id = ?" if paper_id else ""),
            ((paper_id,) if paper_id else ()),
        ).fetchall()
        traces: list[RunTrace] = []
        for run in runs:
            run_id = run["run_id"]
            score_rows = conn.execute("SELECT * FROM score_items WHERE run_id = ?", (run_id,)).fetchall()
            if not score_rows:
                continue
            judgement_rows = {
                row["indicator_id"]: row
                for row in conn.execute("SELECT * FROM agent_judgements WHERE run_id = ?", (run_id,)).fetchall()
            }
            records: list[RuleRecord] = []
            dimensions: dict[str, float] = {}
            for row in score_rows:
                judgement = judgement_rows.get(row["indicator_id"])
                records.append(
                    RuleRecord(
                        rule_id=str(row["rule_id"]),
                        dimension_id=str(row["dimension_id"]),
                        judgement=str(row["judgement"]),
                        confidence=float(judgement["confidence"]) if judgement else 0.0,
                        evidence_ids=json.loads(judgement["evidence_ids_json"]) if judgement else [],
                        raw_score=_float_or_none(row["raw_score"]),
                        weight=float(row["weight"] or 0.0),
                        needs_review=bool(judgement["needs_review"]) if judgement else False,
                    )
                )
                weight = float(row["weight"] or 0.0)
                if row["raw_score"] is not None and weight > 0:
                    dimensions[str(row["dimension_id"])] = dimensions.get(str(row["dimension_id"]), 0.0) + float(
                        row["raw_score"]
                    ) * weight
            model_versions = json.loads(run["model_versions_json"] or "{}")
            traces.append(
                RunTrace(
                    run_id=run_id,
                    paper_id=str(run["paper_id"]),
                    rule_records=records,
                    total_score=None,   # There is no total score column in the library: the total score is calculated by aggregate, see reliability_diagnosis
                    dimension_scores=dimensions,
                    rule_profile=str(run["rule_profile"]),
                    model_version=str(model_versions.get("core_agents") or ""),
                    started_at=str(run["started_at"]),
                    source_path=str(path),
                )
            )
        return traces


def group_by_paper(traces: Iterable[RunTrace]) -> dict[str, list[RunTrace]]:
    grouped: dict[str, list[RunTrace]] = {}
    for trace in traces:
        grouped.setdefault(trace.paper_id, []).append(trace)
    return grouped


# --------------------------------------------------------------------------------------
# 1. Run Variance: The same paper is run multiple times, and the total score/dimension/ranking is unstable.
# --------------------------------------------------------------------------------------

def _stability_from_range(value: float | None, threshold: float) -> str:
    if value is None:
        return "insufficient_data"
    if value <= threshold * 0.5:
        return "high"
    if value <= threshold:
        return "medium"
    return "low"


def _pairwise_jaccard(sets: list[set[str]]) -> float | None:
    pairs = [
        (left, right)
        for index, left in enumerate(sets)
        for right in sets[index + 1 :]
    ]
    if not pairs:
        return None
    scores = []
    for left, right in pairs:
        union = left | right
        scores.append(1.0 if not union else len(left & right) / len(union))
    return sum(scores) / len(scores)


def spearman(left: list[float], right: list[float]) -> float | None:
    'Spearman rank correlation. The average rank is used for parallel values. Returns None if there are fewer than 2 points or the variance is 0.'
    if len(left) != len(right) or len(left) < 2:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                result[order[index]] = average
            position = end + 1
        return result

    rank_left, rank_right = ranks(left), ranks(right)
    n = len(left)
    mean_left, mean_right = sum(rank_left) / n, sum(rank_right) / n
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(rank_left, rank_right))
    var_left = sum((a - mean_left) ** 2 for a in rank_left)
    var_right = sum((b - mean_right) ** 2 for b in rank_right)
    if var_left <= 0 or var_right <= 0:
        return None
    return cov / math.sqrt(var_left * var_right)


def run_variance(traces: list[RunTrace], config: ReliabilityConfig) -> dict[str, Any]:
    'Total score/dimension/level stability across multiple runs of the same paper.\n\n    Ranking_change requires "multiple papers" to be calculated, so here we also receive traces of other papers in a batch:\n    For a single paper, return None and explain it in notes instead of compiling a ranking.'
    scored = [trace for trace in traces if trace.total_score is not None]
    if len(scored) < max(2, config.min_runs):
        return {
            "run_count": len(traces),
            "scored_run_count": len(scored),
            "stability": "insufficient_data",
            "note": f"Insufficient number of comparable runs ({len(scored)} < {max(2, config.min_runs)}): Operation stability cannot be judged in a single run.",
            "mean_score": scored[0].total_score if len(scored) == 1 else None,
            "std": None,
            "range": None,
            "min_score": min((t.total_score for t in scored), default=None),
            "max_score": max((t.total_score for t in scored), default=None),
            "dimension_stability": {},
            "grade_flip": False,
            "grade_flip_detail": [],
            "flags": [],
        }

    scores = [float(trace.total_score) for trace in scored]
    mean_score = sum(scores) / len(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    score_range = max(scores) - min(scores)

    dimension_stability: dict[str, Any] = {}
    dimension_flagged: list[str] = []
    for dimension_id in DIMENSION_IDS:
        values = [
            trace.dimension_scores[dimension_id]
            for trace in scored
            if trace.dimension_scores.get(dimension_id) is not None
        ]
        if len(values) < 2:
            continue
        spread = max(values) - min(values)
        dimension_stability[dimension_id] = {
            "mean": round(sum(values) / len(values), 4),
            "range": round(spread, 4),
            "std": round(statistics.pstdev(values), 4),
            "stability": _stability_from_range(spread, config.dimension_variance_threshold),
        }
        if spread > config.dimension_variance_threshold:
            dimension_flagged.append(dimension_id)

    grades = [f"{trace.quality_grade}|{trace.top_journal_potential}" for trace in scored]
    grade_flip = len(set(grades)) > 1

    flags: list[str] = []
    if score_range > config.run_variance_threshold:
        flags.append("high_variance")
    if grade_flip:
        flags.append("grade_flip")
    if dimension_flagged:
        flags.append("dimension_variance")

    return {
        "run_count": len(traces),
        "scored_run_count": len(scored),
        "mean_score": round(mean_score, 4),
        "std": round(std, 4),
        "range": round(score_range, 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "coefficient_of_variation": round(std / mean_score, 4) if mean_score else None,
        "stability": _stability_from_range(score_range, config.run_variance_threshold),
        "threshold": config.run_variance_threshold,
        "dimension_stability": dimension_stability,
        "dimension_flagged": dimension_flagged,
        "grade_flip": grade_flip,
        "grade_flip_detail": sorted(set(grades)) if grade_flip else [],
        "runs": [
            {
                "run_id": trace.run_id,
                "total_score": trace.total_score,
                "quality_grade": trace.quality_grade,
                "top_journal_potential": trace.top_journal_potential,
                "started_at": trace.started_at,
            }
            for trace in scored
        ],
        "flags": flags,
        "ranking_change": None,
        "ranking_note": 'Ranking_change for a single paper requires multiple papers within a batch: see `batch_ranking_stability()`.',
    }


def batch_ranking_stability(paper_traces: dict[str, list[RunTrace]], config: ReliabilityConfig) -> dict[str, Any]:
    'Batch-level sorting stability: Treat each run as a complete sorting, and calculate the overlap between Spearman and Top-K in pairs.'
    # Each run position → {paper_id: score}, only the papers common to the same batch are retained
    run_scores: dict[str, dict[str, float]] = {}
    for paper_id, traces in paper_traces.items():
        for index, trace in enumerate(sorted(traces, key=lambda item: item.started_at or item.run_id)):
            if trace.total_score is None:
                continue
            run_scores.setdefault(f"{paper_id}#{index}", {})[paper_id] = float(trace.total_score)

    # Batch concept: Runs in the same directory are sorted once. Group by directory of source_path.
    by_dir: dict[str, dict[str, float]] = {}
    for paper_id, traces in paper_traces.items():
        for trace in traces:
            if trace.total_score is None:
                continue
            directory = str(Path(trace.source_path).parent) if trace.source_path else "unknown"
            by_dir.setdefault(directory, {})[paper_id] = max(
                by_dir.get(directory, {}).get(paper_id, float("-inf")), float(trace.total_score)
            )
    rankings = [scores for scores in by_dir.values() if len(scores) >= 2]
    if len(rankings) < 2:
        return {
            "checked": False,
            "stability": "insufficient_data",
            "note": (
                'sorting stability requires at least 2 papers in the same batch directory × at least 2 batches;'
                'The current batch is not sufficient to calculate paper ordering changes.'
            ),
            "pair_count": 0,
        }

    paper_ids = sorted({paper_id for scores in rankings for paper_id in scores})
    common = [paper_id for paper_id in paper_ids if all(paper_id in scores for scores in rankings)]
    if len(common) < 2:
        return {
            "checked": False,
            "stability": "insufficient_data",
            "note": 'There are not enough common papers (at least 2) between batches for comparative ranking.',
            "pair_count": 0,
        }

    vectors = [[scores[paper_id] for paper_id in common] for scores in rankings]
    correlations: list[float] = []
    for index, left in enumerate(vectors):
        for right in vectors[index + 1 :]:
            value = spearman(left, right)
            if value is not None:
                correlations.append(value)
    top_k = max(1, min(config.top_k, len(common)))
    top_sets = [
        {common[i] for i in sorted(range(len(common)), key=lambda i, v=vector: -v[i])[:top_k]}
        for vector in vectors
    ]
    overlap = _pairwise_jaccard(top_sets)
    mean_correlation = sum(correlations) / len(correlations) if correlations else None
    flagged = mean_correlation is not None and mean_correlation < config.ranking_correlation_threshold
    return {
        "checked": True,
        "batch_count": len(rankings),
        "paper_count": len(common),
        "pair_count": len(correlations),
        "mean_spearman": round(mean_correlation, 4) if mean_correlation is not None else None,
        "min_spearman": round(min(correlations), 4) if correlations else None,
        "top_k": top_k,
        "top_k_overlap": round(overlap, 4) if overlap is not None else None,
        "ranking_change": bool(flagged),
        "stability": (
            "insufficient_data" if mean_correlation is None
            else "low" if flagged
            else "high" if mean_correlation >= 0.95
            else "medium"
        ),
        "threshold": config.ranking_correlation_threshold,
        "flags": ["ranking_change"] if flagged else [],
    }


# --------------------------------------------------------------------------------------
# 2. Evidence Variance: Is the recall of rule → evidence_ids stable during multiple runs?
# --------------------------------------------------------------------------------------

def evidence_variance(traces: list[RunTrace], config: ReliabilityConfig) -> dict[str, Any]:
    usable = [trace for trace in traces if trace.rule_records]
    if len(usable) < max(2, config.min_runs):
        return {
            "run_count": len(traces),
            "stability": "insufficient_data",
            "note": f"Insufficient number of comparable runs ({len(usable)} < {max(2, config.min_runs)}): A single run cannot judge the stability of evidence retrieval.",
            "evidence_overlap": None,
            "coverage_variance": None,
            "per_rule": [],
            "flags": [],
        }

    # rule_id → Evidence collection for each run
    per_rule_sets: dict[str, list[set[str]]] = {}
    per_rule_counts: dict[str, list[int]] = {}
    per_rule_sections: dict[str, list[set[str]]] = {}
    for trace in usable:
        for record in trace.rule_records:
            per_rule_sets.setdefault(record.rule_id, []).append(set(record.evidence_ids))
            per_rule_counts.setdefault(record.rule_id, []).append(len(record.evidence_ids))
            per_rule_sections.setdefault(record.rule_id, []).append(
                {section for section in record.evidence_sections if section}
            )

    per_rule: list[dict[str, Any]] = []
    overlaps: list[float] = []
    coverage_cvs: list[float] = []
    for rule_id in sorted(per_rule_sets):
        sets = per_rule_sets[rule_id]
        if len(sets) < 2:
            continue
        overlap = _pairwise_jaccard(sets)
        counts = per_rule_counts[rule_id]
        mean_count = sum(counts) / len(counts)
        cv = (statistics.pstdev(counts) / mean_count) if mean_count else 0.0
        section_overlap = _pairwise_jaccard(per_rule_sections[rule_id])
        unstable = (overlap is not None and overlap < config.evidence_overlap_threshold) or cv > config.evidence_coverage_cv_threshold
        if overlap is not None:
            overlaps.append(overlap)
        coverage_cvs.append(cv)
        per_rule.append(
            {
                "rule_id": rule_id,
                "run_count": len(sets),
                "evidence_overlap": round(overlap, 4) if overlap is not None else None,
                "evidence_count_mean": round(mean_count, 3),
                "evidence_count_cv": round(cv, 4),
                "section_consistency": round(section_overlap, 4) if section_overlap is not None else None,
                "unstable": unstable,
            }
        )

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else None
    mean_cv = sum(coverage_cvs) / len(coverage_cvs) if coverage_cvs else None
    unstable_rules = [item["rule_id"] for item in per_rule if item["unstable"]]
    flags: list[str] = []
    if mean_overlap is not None and mean_overlap < config.evidence_overlap_threshold:
        flags.append("evidence_instability")
    if mean_cv is not None and mean_cv > config.evidence_coverage_cv_threshold:
        flags.append("evidence_coverage_variance")
    stability = (
        "insufficient_data" if mean_overlap is None
        else "low" if "evidence_instability" in flags
        else "high" if mean_overlap >= 0.85 and not unstable_rules
        else "medium"
    )
    return {
        "run_count": len(usable),
        "evidence_overlap": round(mean_overlap, 4) if mean_overlap is not None else None,
        "coverage_variance": round(mean_cv, 4) if mean_cv is not None else None,
        "stability": stability,
        "threshold": config.evidence_overlap_threshold,
        "unstable_rule_ids": unstable_rules,
        "per_rule": sorted(per_rule, key=lambda item: (item["evidence_overlap"] if item["evidence_overlap"] is not None else 1.0)),
        "flags": flags,
    }


# --------------------------------------------------------------------------------------
# 3. Judgment Variance: Whether the Agent’s judgment on the same rule is stable or not
# --------------------------------------------------------------------------------------

def _entropy_bits(counts: dict[str, int], total: int) -> float:
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        value -= probability * math.log2(probability)
    return value


def judgment_variance(
    traces: list[RunTrace],
    config: ReliabilityConfig,
    evidence_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    'The gear distribution, consistency rate, and entropy of the same rule, and attribution of uncertainty sources are given.\n\n    Attribution rules (judgment criteria in requirements):\n    - The evidence is stable but the judgment is unstable → `agent_reasoning`\n    - Unstable evidence and unstable judgment → `retrieval_and_reasoning`'
    usable = [trace for trace in traces if trace.rule_records]
    if len(usable) < max(2, config.min_runs):
        return {
            "run_count": len(traces),
            "stability": "insufficient_data",
            "note": f"Insufficient number of comparable runs ({len(usable)} < {max(2, config.min_runs)}): The stability cannot be judged in a single run.",
            "mean_agreement": None,
            "per_rule": [],
            "source_of_uncertainty": "insufficient_data",
            "flags": [],
        }

    by_rule: dict[str, list[RuleRecord]] = {}
    for trace in usable:
        for record in trace.rule_records:
            by_rule.setdefault(record.rule_id, []).append(record)

    evidence_by_rule = {
        item["rule_id"]: item for item in (evidence_result or {}).get("per_rule", [])
    }
    agree_rates: list[float] = []
    per_rule: list[dict[str, Any]] = []
    unstable_rules: list[str] = []
    reasoning_only: list[str] = []
    retrieval_and_reasoning: list[str] = []
    for rule_id in sorted(by_rule):
        records = by_rule[rule_id]
        if len(records) < 2:
            continue
        counts: dict[str, int] = {}
        for record in records:
            counts[record.judgement] = counts.get(record.judgement, 0) + 1
        total = sum(counts.values())
        agreement = max(counts.values()) / total if total else 0.0
        entropy = _entropy_bits(counts, total)
        confidence_mean = sum(record.confidence for record in records) / len(records)
        unstable = agreement < config.judgment_agreement_threshold or entropy > config.judgment_entropy_threshold
        evidence_item = evidence_by_rule.get(rule_id) or {}
        evidence_unstable = bool(evidence_item.get("unstable"))
        source = ""
        if unstable:
            source = "retrieval_and_reasoning" if evidence_unstable else "agent_reasoning"
            unstable_rules.append(rule_id)
            if source == "agent_reasoning":
                reasoning_only.append(rule_id)
            else:
                retrieval_and_reasoning.append(rule_id)
        agree_rates.append(agreement)
        per_rule.append(
            {
                "rule_id": rule_id,
                "run_count": len(records),
                "distribution": {key: round(value / total, 4) for key, value in sorted(counts.items())},
                "agreement": round(agreement, 4),
                "entropy": round(entropy, 4),
                "confidence_mean": round(confidence_mean, 4),
                "confidence_range": round(max(r.confidence for r in records) - min(r.confidence for r in records), 4),
                "needs_review_any": any(record.needs_review for record in records),
                "evidence_unstable": evidence_unstable,
                "source_of_uncertainty": source,
                "judgements": [record.judgement for record in records],
            }
        )

    mean_agreement = sum(agree_rates) / len(agree_rates) if agree_rates else None
    flags: list[str] = []
    # As long as one rule is unstable, it will be reported: the overall consistency rate will be averaged out by a large number of stable rules.
    #(actual measurement of four batches of v23: overall 0.90, but the rule consistency rate on C/D/E is only 0.5).
    if mean_agreement is not None and mean_agreement < config.judgment_agreement_threshold:
        flags.append("judgment_instability")
    if unstable_rules:
        flags.append("judgment_instability")
        flags.append("per_rule_judgment_instability")
    flags = list(dict.fromkeys(flags))
    if retrieval_and_reasoning:
        source_of_uncertainty = "retrieval_and_reasoning"
    elif reasoning_only:
        source_of_uncertainty = "agent_reasoning"
    else:
        source_of_uncertainty = "" if mean_agreement is not None else "insufficient_data"
    stability = (
        "insufficient_data" if mean_agreement is None
        else "low" if "judgment_instability" in flags
        else "high" if mean_agreement >= 0.9 and not unstable_rules
        else "medium"
    )
    return {
        "run_count": len(usable),
        "mean_agreement": round(mean_agreement, 4) if mean_agreement is not None else None,
        "stability": stability,
        "threshold": config.judgment_agreement_threshold,
        "unstable_rule_ids": unstable_rules,
        "reasoning_only_rule_ids": reasoning_only,
        "retrieval_and_reasoning_rule_ids": retrieval_and_reasoning,
        "source_of_uncertainty": source_of_uncertainty,
        # In ascending order according to the consistency rate: the most unstable ones are ranked first, and the report directly says "which ones should be looked at"
        "per_rule": sorted(per_rule, key=lambda item: item["agreement"]),
        "flags": flags,
    }


def stability_scores(
    run_result: dict[str, Any],
    evidence_result: dict[str, Any],
    judgment_result: dict[str, Any],
    calibration_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    'converts the four conclusions into 0-1 performance points for quick comparison by report readers.\n\n    **Deliberately not combined into one total score**: The four sub-items are presented independently, and combining them will introduce new weights (clearly not required in the requirements).\n    A breakdown with insufficient data returns None instead of pretending to be stable with 1.0.'
    return {
        "run_stability": STABILITY_SCORE.get(str(run_result.get("stability")), None),
        "evidence_stability": STABILITY_SCORE.get(str(evidence_result.get("stability")), None),
        "judgment_stability": STABILITY_SCORE.get(str(judgment_result.get("stability")), None),
        "calibration_robustness": (
            None if not calibration_result or not calibration_result.get("checked")
            else STABILITY_SCORE.get(str(calibration_result.get("robustness")), None)
        ),
        "note": 'The four sub-items are independent and will not be weighted and combined; None means that there is insufficient data in this dimension and does not mean that it is stable.\nDetermination and reporting of',
    }

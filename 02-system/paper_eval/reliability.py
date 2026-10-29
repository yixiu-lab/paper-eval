"""Evaluation Reliability Layer：判断「这次评价结果本身稳不稳、可信不可信」。

定位（必须说清楚，否则很容易被误用）：

- **纯后处理**。它不参与论文质量判断，不改变 `scorer.py` 的算分口径，不修改任何已有 Agent 的输出，
  也不影响 `harness.evaluate_pdf` 的返回值结构——只是在评分结束之后读产物、算稳定性。
- **不调用任何模型**。整层是确定性的：同一批 run 重新算一次，结论逐字节相同。
  这不是"再请一个 Agent 评一遍"，而是"把已有判断记录拿来做方差分析"。
- **不出总分**。运行稳定性 / 证据稳定性 / 判断稳定性 / 参数稳健性 是四个不同维度的问题，
  强行合成一个数字只会引入一套新的主观权重。报告只给四个分项 + 警告清单 + 需复核项。

四类检测（输入全部来自已有产物，见 `RunTrace`）：

1. `run_variance`：同一篇论文多次运行的总分 / 维度 / 排名是否稳定（`run_variance`）
2. `evidence_variance`：多次运行里 rule → evidence_ids 的召回是否稳定（`evidence_variance`）
3. `judgment_variance`：同一 rule 上 Agent 判断档位的分布与一致率（`judgment_variance`）
4. `calibration_uncertainty`：换一套档位映射重新算分，排序是否翻（见 reliability_diagnosis.py）

证据稳定但判断不稳 → `agent_reasoning`；两者都不稳 → `retrieval_and_reasoning`。
"""

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

# 稳定性档位 → 0-1 表现分。只用于报告里给读者一个大数，不参与任何加权合成。
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
    """所有阈值配置化。默认值取自项目里已知的实测波动量级（见 technical doc 的 V16 重标定记录）。"""

    enabled: bool = False
    # 同一篇论文少于这个运行次数时，运行/证据/判断三块方差一律标 insufficient_data（不假装算得出来）
    min_runs: int = 2
    # 总分极差阈值（分）。规则库满分 100。
    # 实测依据：Choudhury 四批重复跑 96.63 / 91.38 / 93.44 / 92.31，极差 5.25 分——那个量级已经足够翻转
    # 「顶刊级 / 强刊边缘」的等级判定，所以默认阈值定在 5.0 分，与实测噪声同量级。
    run_variance_threshold: float = 5.0
    # 维度得分极差阈值（分，按维度满分折算前的绝对分）
    dimension_variance_threshold: float = 4.0
    # 跨 run 的论文排序相关阈值（Spearman）：低于它就认为多篇论文的相对次序不稳
    ranking_correlation_threshold: float = 0.8
    # 证据交集阈值：低于它认为 Evidence Router / 检索不稳
    evidence_overlap_threshold: float = 0.6
    # 证据条数变异的容忍度（变异系数）
    evidence_coverage_cv_threshold: float = 0.25
    # 判断一致率阈值：低于它认为 Agent 在同一条规则上判断不稳
    judgment_agreement_threshold: float = 0.8
    # 判断分布熵阈值（bit，5 档最大 2.32）：超过它就认为档位分散
    judgment_entropy_threshold: float = 1.0
    # 校准敏感性：总分相对变化（|Δ| / 满分）超过它认为对映射敏感
    calibration_score_delta_threshold: float = 0.05
    # 校准敏感性：跨方案排序相关低于它认为排序不稳
    calibration_ranking_correlation_threshold: float = 0.9
    # 校准敏感性：Top-K 集合重合率低于它认为候选集不稳
    calibration_topk_overlap_threshold: float = 0.8
    # Top-K 取几篇（多篇论文批次里的排序稳定性）
    top_k: int = 5


def load_reliability_config(path: str | Path | None) -> ReliabilityConfig:
    """从配置文件顶层读 `reliability:` 段。

    与 `external_retrieval` 同理，刻意放在 `agent_runtime` 之外：`AgentRuntimeConfig`
    会把该段落的键直接展开成构造参数，多一个未知键就报错。
    """
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
        raise ValueError(f"reliability 配置含未知键: {unknown}")
    return ReliabilityConfig(**values)


# --------------------------------------------------------------------------------------
# 运行 trace：一次评价里每一条判断的完整记录（reliability 层的唯一输入单元）
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
    """一次运行的判断记录 + 分数记录。字段对齐需求里的 trace 约定。"""

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
    """从 `build_report()` 产出的报告 JSON 建 RunTrace。

    数据来源就是既有产物：`details[]` 里有 rule_id / judgement / confidence / evidence[]，
    `summary` 里有总分与维度分。**不改报告格式**，只读。
    """
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
    """递归收集报告 JSON。默认跳过失败报告（没有 details 的那些）。"""
    directory = Path(root)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob(pattern) if path.is_file())


def load_traces_from_db(db_path: str | Path, paper_id: str | None = None) -> list[RunTrace]:
    """从 `storage.py` 的 SQLite 运行库读 run trace（离线重算的另一条入口）。

    只读，不写库；`agent_judgements` / `score_items` 里已经有全部需要的字段。
    """
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
                    total_score=None,   # 库里没有总分列：总分由 aggregate 现算，见 reliability_diagnosis
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
# 一、Run Variance：同一篇论文多次运行，总分 / 维度 / 排名稳不稳
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
    """Spearman 秩相关。并列值用平均秩。少于 2 个点或方差为 0 时返回 None。"""
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
    """同一篇论文多次运行的总分 / 维度 / 等级稳定性。

    ranking_change 需要"多篇论文"才能算，所以这里同时接收一个批次里的其他论文 trace：
    单篇论文时返回 None，并在 notes 里说明，而不是编一个排名出来。
    """
    scored = [trace for trace in traces if trace.total_score is not None]
    if len(scored) < max(2, config.min_runs):
        return {
            "run_count": len(traces),
            "scored_run_count": len(scored),
            "stability": "insufficient_data",
            "note": f"可比较的运行次数不足（{len(scored)} < {max(2, config.min_runs)}）：单次运行无法判断运行稳定性。",
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
        "ranking_note": "单篇论文的 ranking_change 需要批次内多篇论文：见 `batch_ranking_stability()`。",
    }


def batch_ranking_stability(paper_traces: dict[str, list[RunTrace]], config: ReliabilityConfig) -> dict[str, Any]:
    """批次级排序稳定性：把每个 run 看成一次完整排序，算两两 Spearman 与 Top-K 重合。"""
    # 每个 run 位置 → {paper_id: score}，只保留同一次批次里共有的论文
    run_scores: dict[str, dict[str, float]] = {}
    for paper_id, traces in paper_traces.items():
        for index, trace in enumerate(sorted(traces, key=lambda item: item.started_at or item.run_id)):
            if trace.total_score is None:
                continue
            run_scores.setdefault(f"{paper_id}#{index}", {})[paper_id] = float(trace.total_score)

    # 批次概念：同一目录下的 run 才是一次排序。按 source_path 的目录分组。
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
                "排序稳定性需要同一批次目录下至少 2 篇论文 × 至少 2 个批次；"
                "当前批次不足以计算论文排序变化。"
            ),
            "pair_count": 0,
        }

    paper_ids = sorted({paper_id for scores in rankings for paper_id in scores})
    common = [paper_id for paper_id in paper_ids if all(paper_id in scores for scores in rankings)]
    if len(common) < 2:
        return {
            "checked": False,
            "stability": "insufficient_data",
            "note": "批次之间没有足够的共同论文（至少 2 篇）用于比较排序。",
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
# 二、Evidence Variance：多次运行里 rule → evidence_ids 召回稳不稳
# --------------------------------------------------------------------------------------

def evidence_variance(traces: list[RunTrace], config: ReliabilityConfig) -> dict[str, Any]:
    usable = [trace for trace in traces if trace.rule_records]
    if len(usable) < max(2, config.min_runs):
        return {
            "run_count": len(traces),
            "stability": "insufficient_data",
            "note": f"可比运行次数不足（{len(usable)} < {max(2, config.min_runs)}）：单次运行无法判断证据检索稳定性。",
            "evidence_overlap": None,
            "coverage_variance": None,
            "per_rule": [],
            "flags": [],
        }

    # rule_id → 每个 run 的证据集合
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
# 三、Judgment Variance：同一 rule 上 Agent 判断稳不稳
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
    """同一条 rule 的档位分布、一致率、熵，并给出不确定性来源归因。

    归因规则（需求里的判定口径）：
    - 证据稳、判断不稳 → `agent_reasoning`
    - 证据不稳、判断不稳 → `retrieval_and_reasoning`
    """
    usable = [trace for trace in traces if trace.rule_records]
    if len(usable) < max(2, config.min_runs):
        return {
            "run_count": len(traces),
            "stability": "insufficient_data",
            "note": f"可比运行次数不足（{len(usable)} < {max(2, config.min_runs)}）：单次运行无法判断判断稳定性。",
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
    # 只要有一条规则不稳就报出来：整体一致率会被大量稳定规则平均掉
    #（实测 v23 四批：整体 0.90，但 C/D/E 上有规则一致率只有 0.5）。
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
        # 按一致率升序：最不稳的排最前，报告里直接就是"该看哪几条"
        "per_rule": sorted(per_rule, key=lambda item: item["agreement"]),
        "flags": flags,
    }


def stability_scores(
    run_result: dict[str, Any],
    evidence_result: dict[str, Any],
    judgment_result: dict[str, Any],
    calibration_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把四块结论折算成 0-1 表现分，供报告读者快速比较。

    **刻意不合成一个总分**：四个分项各自独立呈现，合成会引入新权重（需求里明确不要）。
    数据不足的分项返回 None，而不是用 1.0 假装稳定。
    """
    return {
        "run_stability": STABILITY_SCORE.get(str(run_result.get("stability")), None),
        "evidence_stability": STABILITY_SCORE.get(str(evidence_result.get("stability")), None),
        "judgment_stability": STABILITY_SCORE.get(str(judgment_result.get("stability")), None),
        "calibration_robustness": (
            None if not calibration_result or not calibration_result.get("checked")
            else STABILITY_SCORE.get(str(calibration_result.get("robustness")), None)
        ),
        "note": "四个分项独立，不做加权合成；None 表示该维度数据不足，不代表稳定。",
    }

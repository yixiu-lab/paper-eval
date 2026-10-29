from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Judgement(str, Enum):
    """五档判断 + 无法判断。

    标尺锚点：partial = 60 分（及格线）。原来只有 satisfied/partial/not_satisfied 三档时，
    60% 以上的判断挤在 partial 一档，总分被压成常数；拆出 mostly 与 weak 让中间散开。
    """

    SATISFIED = "satisfied"
    MOSTLY = "mostly"
    PARTIAL = "partial"
    WEAK = "weak"
    NOT_SATISFIED = "not_satisfied"
    UNCERTAIN = "uncertain"


class RunStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


@dataclass
class Paper:
    paper_id: str
    metadata: dict[str, Any]
    sections: list[dict[str, Any]]
    references: list[str]
    publish_date: str | None = None
    parse_version: str = "pdf-parser-0.1"


@dataclass
class Evidence:
    evidence_id: str
    paper_id: str
    section: str
    page: int
    span: tuple[int, int]
    text: str
    text_hash: str
    source_type: str = "paragraph"


@dataclass
class EvaluationRun:
    run_id: str
    paper_id: str
    harness_version: str
    model_versions: dict[str, str]
    rule_profile: str
    status: RunStatus
    started_at: str
    completed_at: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentJudgement:
    task_id: str
    agent_id: str
    indicator_id: str
    judgement: Judgement
    evidence_ids: list[str]
    reason: str
    confidence: float
    needs_review: bool
    agent_version: str
    model_version: str
    candidate_evidence_ids: list[str] = field(default_factory=list)
    # 本地启发式的观察结果。约定：本地规则只写这里 + needs_review，
    # 不改 judgement / reason / confidence，模型判断与理由保持原样可审计。
    review_flags: list[str] = field(default_factory=list)


@dataclass
class ScoreItem:
    indicator_id: str
    judgement: Judgement
    rule_id: str
    raw_score: float | None
    weighted_score: float | None
    weight: float
    dimension_id: str
    dimension_name: str
    strict_raw_score: float | None = None
    strict_weighted_score: float | None = None
    # 李克特呈现层（1–5）：只是判断档位的编号，供报告与论文方法学描述使用；
    # 参与总分计算的仍是 raw_score（两池映射，基准项 mostly/partial 同分）。
    likert_score: int | None = None
    # 适用范围外：不评分、不进分子分母，只在报告里列出。
    out_of_scope: bool = False


@dataclass
class RetrievalMatch:
    doc_id: str
    title: str
    facet: str
    score: float
    source_path: str
    publish_year: int | None = None
    preview: str = ""


@dataclass
class ExtendedAnalysis:
    agent_id: str
    analysis_type: str
    summary: str
    matches: list[RetrievalMatch]
    confidence: float
    needs_review: bool


@dataclass
class ReflectionIssue:
    issue_type: str
    severity: str
    message: str
    related_ids: list[str]
    suggested_action: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_dict(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_dict(v) for v in value]
    if isinstance(value, tuple):
        return [to_dict(v) for v in value]
    if isinstance(value, dict):
        return {k: to_dict(v) for k, v in value.items()}
    return value

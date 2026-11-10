from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Judgement(str, Enum):
    'Fifth gear judgment + Unable to judge.\n\n    Scale anchor point: partial = 60 points (passing line). It turns out that there is only satisfied/partial/not_satisfied in the third gear.\n    More than 60% of the judgments are squeezed into the partial level, and the total score is reduced to a constant; most and weak are separated to spread out in the middle.'

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
    # Observations of local heuristics. Convention: local rules only write here + needs_review,
    # does not change judgment / reason / confidence, the model judgment and reasons remain unchanged and can be audited.
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
    # Likert presentation layer (1–5): just the number of the judgment gear, used for report and paper methodology description;
    # still uses raw_score (two-pool mapping, benchmark items mostly/partially have the same score) to participate in the calculation of the total score.
    likert_score: int | None = None
    # Outside the applicable scope: no scoring, no inclusion in the numerator and denominator, only listed in the report.
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

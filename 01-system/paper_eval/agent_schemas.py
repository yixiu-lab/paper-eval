from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


JudgementLiteral = Literal["satisfied", "mostly", "partial", "weak", "not_satisfied", "uncertain"]

ALLOWED_JUDGEMENTS = ["satisfied", "mostly", "partial", "weak", "not_satisfied", "uncertain"]


class AgentEvidenceInput(BaseModel):
    evidence_id: str
    section: str
    page: int
    text: str


class AgentRuleInput(BaseModel):
    rule_id: str
    indicator_name: str
    definition: str
    required_evidence: str
    evidence_location: list[str]
    observation_points: list[str]
    # Rule role: baseline (eligibility threshold, passing means full points, only gaps will be judged) or bonus (additional points, only satisfied ones get full points).
    role: str = "baseline"
    allowed_judgements: list[str] = Field(default_factory=lambda: list(ALLOWED_JUDGEMENTS))
    # The gear level definition of each rule. In the past, it only existed in the rule base and did not enter the model input. The model could only guess the gear position by relying on common vocabulary.
    judgement_anchors: dict[str, str] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    indicator_id: str
    judgement: JudgementLiteral
    evidence_ids: list[str]
    reason: str
    confidence: float = Field(ge=0, le=1)
    needs_review: bool

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be empty")
        return value


class AgentTaskPayload(BaseModel):
    run_id: str
    agent_id: str
    paper_context: dict[str, Any] = Field(default_factory=dict)
    rule: AgentRuleInput
    evidences: list[AgentEvidenceInput]

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
    # 规则角色：baseline（合格性门槛，合格即满分，只判缺口）或 bonus（加分项，只有 satisfied 拿满）。
    role: str = "baseline"
    allowed_judgements: list[str] = Field(default_factory=lambda: list(ALLOWED_JUDGEMENTS))
    # 每条规则的档位定义。以前只存在规则库里、没有进入模型输入，模型只能靠通用话术猜档位。
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

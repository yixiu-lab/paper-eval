from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import out_of_scope_rule_ids_of, paper_class, retry_uncertain_judgements, run_core_agents
from .audit_agent import audit_to_reflection_issues, run_reflection_audit_agent
from .claim_extraction import publication_year_of
from .evidence_gate import apply_evidence_gate
from .external_evidence import (
    apply_coverage_confidence_cap,
    apply_external_validation_to_judgements,
    build_repositories,
    repository_to_dict,
    validate_external_evidence,
)
from .external_retrieval import (
    collect_external_literature,
    load_external_retrieval_config,
)
from .literature_search import external_evidence_blocks_from_set
from .llm import load_agent_runtime_config
from .models import AgentJudgement, Evidence, Paper, ReflectionIssue, ScoreItem
from .paper_context import build_global_context
from .reflection import reflect_run
from .rules import RuleProfile
from .schema_validation import validate_agent_judgements
from .scorer import aggregate_scores, score_judgements
from .social_relevance import social_issue_evidence_blocks


class EvaluationGraphState(TypedDict, total=False):
    run_id: str
    profile: RuleProfile
    paper: Paper
    evidences: list[Evidence]
    agent_config_path: str | Path | None
    agent_runtime_overrides: dict[str, object] | None
    paper_context_override: dict[str, Any]
    external_retrieval_override: dict[str, Any]
    social_relevance_override: list[Evidence]
    paper_context: dict[str, Any]
    external_retrieval: dict[str, Any]
    external_evidences: list[Evidence]
    novelty_evidences: list[Evidence]
    social_relevance_evidences: list[Evidence]
    novelty_verification: dict[str, Any]
    external_validation: dict[str, Any]
    evidence_repository: dict[str, Any]
    judgements: list[AgentJudgement]
    ungated_judgements: list[AgentJudgement]
    schema_errors: list[str]
    score_items: list[ScoreItem]
    summary: dict[str, Any]
    reflection_issues: list[ReflectionIssue]
    audit_report: dict[str, Any]
    trace: list[dict[str, Any]]
    progress_callback: Callable[[str], None] | None


def append_trace(state: EvaluationGraphState, item: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("trace", []), item]


def notify_progress(state: EvaluationGraphState, stage: str) -> None:
    callback = state.get("progress_callback")
    if callback:
        callback(stage)


def build_paper_context(state: EvaluationGraphState) -> EvaluationGraphState:
    notify_progress(state, 'Full text understanding Agent is creating a paper portrait')
    if state.get("paper_context_override") is not None:
        context = state["paper_context_override"]
        return {"paper_context": context, "trace": append_trace(state, {"step": "langgraph.build_paper_context", "paper_type": (context.get("paper_profile") or {}).get("paper_type"), "context_evidence_count": len(context.get("context_evidence_ids", [])), "paired_input_reused": True})}
    config = load_agent_runtime_config(
        state.get("agent_config_path"),
        overrides=state.get("agent_runtime_overrides"),
    )
    context = build_global_context(
        state["run_id"],
        state["paper"],
        state["evidences"],
        state["profile"],
        config,
    )
    profile = context.get("paper_profile", {})
    return {
        "paper_context": context,
        "trace": append_trace(
            state,
            {
                "step": "langgraph.build_paper_context",
                "paper_type": profile.get("paper_type"),
                "context_evidence_count": len(context.get("context_evidence_ids", [])),
            },
        ),
    }


def ground_social_relevance(state: EvaluationGraphState) -> EvaluationGraphState:
    'Social issue grounding input layer: Convert annual social/international issues into citable evidence blocks.\n\n    This node only provides SIV-* grounding for F-SOCIAL bonus rules. Ratings will remain the same as before\n    Six Reviewers judge that Harness is responsible for checking whether the evidence ID actually exists.'
    config = load_agent_runtime_config(state.get("agent_config_path"), overrides=state.get("agent_runtime_overrides"))
    if not config.social_relevance_grounding_enabled:
        return {
            "social_relevance_evidences": [],
            "trace": append_trace(state, {"step": "langgraph.ground_social_relevance", "evidence_block_count": 0, "disabled": True}),
        }
    if state.get("social_relevance_override") is not None:
        blocks = list(state["social_relevance_override"])
        return {
            "social_relevance_evidences": blocks,
            "evidences": [*state["evidences"], *blocks],
            "trace": append_trace(state, {"step": "langgraph.ground_social_relevance", "evidence_block_count": len(blocks), "evidence_ids": [item.evidence_id for item in blocks], "paired_input_reused": True}),
        }
    notify_progress(state, 'is matching annual social issue grounding evidence')
    blocks = social_issue_evidence_blocks(
        state["paper"],
        state.get("paper_context"),
        limit=3,
    )
    return {
        "social_relevance_evidences": blocks,
        "evidences": [*state["evidences"], *blocks],
        "trace": append_trace(
            state,
            {
                "step": "langgraph.ground_social_relevance",
                "evidence_block_count": len(blocks),
                "evidence_ids": [item.evidence_id for item in blocks],
            },
        ),
    }


def retrieve_external_evidence(state: EvaluationGraphState) -> EvaluationGraphState:
    'Do true external retrieval before scoring: Claim Extraction → Three-Way Recall → Rerank → External Evidence Set.\n\n    Put it here instead of after the score: the rules that determine "whether someone has already done it" must be externally compared.\n    Otherwise, the model can only trust the self-statement of the paper. Failed searches will not interrupt scoring - missing external evidence will be credited\n    `external_evidence_unavailable`, and the rules that block dependent field comparisons on the Harness side are not satisfied.'
    notify_progress(state, 'is retrieving external related work (Claim → Multi-way Recall → Rearrange)')
    config = load_agent_runtime_config(
        state.get("agent_config_path"),
        overrides=state.get("agent_runtime_overrides"),
    )
    retrieval_config = load_external_retrieval_config(state.get("agent_config_path"))
    runtime = load_agent_runtime_config(state.get("agent_config_path"), overrides=state.get("agent_runtime_overrides"))
    if state.get("external_retrieval_override") is not None and runtime.external_grounding_enabled:
        retrieval = state["external_retrieval_override"]
        blocks = external_evidence_blocks_from_set(retrieval, state["paper"].paper_id)
        return {
            "external_retrieval": retrieval,
            "external_evidences": blocks,
            "evidences": [*state["evidences"], *blocks],
            "trace": append_trace(state, {"step": "langgraph.retrieve_external_evidence", "external_evidence_count": len(retrieval.get("external_evidence_set") or []), "evidence_block_count": len(blocks), "paired_input_reused": True, "stop_reason": "paired_input"}),
        }
    if not runtime.external_grounding_enabled or not retrieval_config.enabled:
        retrieval = {"status": {"provider": None}, "claims": [], "external_evidence_set": [], "external_evidence_unavailable": True, "stop_reason": "disabled"}
        return {
            "external_retrieval": retrieval,
            "external_evidences": [],
            "trace": append_trace(state, {"step": "langgraph.retrieve_external_evidence", "external_evidence_count": 0, "disabled": True, "stop_reason": "disabled"}),
        }
    paper = state["paper"]
    profile_data = (state.get("paper_context") or {}).get("paper_profile") or {}
    publication_year = publication_year_of(paper.metadata, paper.publish_date)
    retrieval = collect_external_literature(
        state["profile"],
        str(paper.metadata.get("title") or ""),
        str(paper.metadata.get("abstract") or ""),
        paper_id=paper.paper_id,
        evidences=state["evidences"],
        paper_type=profile_data.get("paper_type") if isinstance(profile_data, dict) else None,
        publication_year=publication_year,
        research_question=str(profile_data.get("research_question") or "") if isinstance(profile_data, dict) else "",
        retrieval_config=retrieval_config,
        runtime_config=config,
        progress_callback=state.get("progress_callback"),
        paper_context=state.get("paper_context"),
    )
    blocks = external_evidence_blocks_from_set(retrieval, paper.paper_id)
    return {
        "external_retrieval": retrieval,
        "external_evidences": blocks,
        # External evidence must be entered into the total evidence table, otherwise the evidence gate cannot parse XEV-* references.
        "evidences": [*state["evidences"], *blocks],
        "trace": append_trace(
            state,
            {
                "step": "langgraph.retrieve_external_evidence",
                "provider": (retrieval.get("status") or {}).get("provider"),
                "claim_source": retrieval.get("claim_source"),
                "claim_count": len(retrieval.get("claims") or []),
                "routes_used": retrieval.get("routes_used"),
                "query_count": len(retrieval.get("queries") or []),
                "pool_size": retrieval.get("pool_size"),
                "time_filtered": retrieval.get("time_filtered"),
                "external_evidence_count": len(retrieval.get("external_evidence_set") or []),
                "evidence_block_count": len(blocks),
                "failure_count": len(retrieval.get("failures") or []),
                "external_evidence_unavailable": retrieval.get("external_evidence_unavailable"),
                "stop_reason": retrieval.get("stop_reason"),
            },
        ),
    }


def verify_novelty(state: EvaluationGraphState) -> EvaluationGraphState:
    'collates prior literature comparison evidence, and B Reviewer completes the final innovativeness judgment.'
    notify_progress(state, 'is compiling the comparative evidence of prior literature and submitting it to B Reviewer')
    retrieval_config = load_external_retrieval_config(state.get("agent_config_path"))
    retrieval = state.get("external_retrieval") or {}
    paper = state["paper"]
    claims = retrieval.get("claims") or []
    external_set = retrieval.get("external_evidence_set") or []
    claim_objects = [_ClaimView(row) for row in claims] if retrieval_config.claim_extraction_enabled else []
    verification = {
        "verdicts": [],
        "unavailable": False,
        "reason": "direct_external_evidence_to_six_reviewers",
        "handled_by": "B Reviewer",
    }
    blocks: list[Evidence] = []
    repository = build_repositories(
        paper.paper_id,
        state["evidences"],
        claim_objects,
        external_set,
    )
    return {
        "novelty_verification": verification,
        "novelty_evidences": blocks,
        "evidence_repository": repository_to_dict(repository),
        "evidences": [*state["evidences"], *blocks],
        "trace": append_trace(
            state,
            {
                "step": "langgraph.verify_novelty",
                "claim_count": len(claim_objects),
                "verdict_count": 0,
                "judgement_distribution": {},
                "unavailable": verification.get("unavailable"),
                "reason": verification.get("reason"),
                "internal_evidence_count": repository.counts()["internal"],
                "external_paper_count": repository.counts()["external"],
            },
        ),
    }


def validate_external_evidence_node(state: EvaluationGraphState) -> EvaluationGraphState:
    'Harness External Evidence Validation: Whether the external evidence is true, consistent, sufficient, and cited.\n\n    Only write review_flags / needs_review / confidence, do not rewrite judgment and reason;\n    Positive judgments that rely on field comparisons without external evidence directly turn to uncertain (hard threshold for traceability).'
    notify_progress(state, 'Verifying external evidence and references')
    retrieval_config = load_external_retrieval_config(state.get("agent_config_path"))
    runtime = load_agent_runtime_config(state.get("agent_config_path"), overrides=state.get("agent_runtime_overrides"))
    if not runtime.external_validation_enabled:
        return {
            "external_validation": {"finding_count": 0, "findings": [], "disabled": True},
            "trace": append_trace(state, {"step": "langgraph.validate_external_evidence", "finding_count": 0, "disabled": True}),
        }
    retrieval = state.get("external_retrieval") or {}
    paper = state["paper"]
    validation = validate_external_evidence(
        claims=retrieval.get("claims") or [],
        external_set=retrieval.get("external_evidence_set") or [],
        novelty_judgements=list((state.get("novelty_verification") or {}).get("verdicts") or []),
        target_year=publication_year_of(paper.metadata, paper.publish_date),
        indicator_ids=[rule.rule_id for rule in state["profile"].rules],
        min_external_papers=retrieval_config.min_external_papers,
        low_coverage_confidence_cap=retrieval_config.low_coverage_confidence_cap,
    )
    judgements = apply_external_validation_to_judgements(state["judgements"], validation)
    verification = state.get("novelty_verification") or {}
    capped_verdicts = apply_coverage_confidence_cap(
        list(verification.get("verdicts") or []),
        validation,
        retrieval_config.low_coverage_confidence_cap,
    )
    if capped_verdicts:
        verification = {**verification, "verdicts": capped_verdicts}
    return {
        "external_validation": validation,
        "novelty_verification": verification,
        "judgements": judgements,
        "trace": append_trace(
            state,
            {
                "step": "langgraph.validate_external_evidence",
                "external_paper_count": validation.get("external_paper_count"),
                "finding_count": validation.get("finding_count"),
                "high_severity_count": validation.get("high_severity_count"),
                "codes": sorted({str(item.get("code")) for item in validation.get("findings") or []}),
            },
        ),
    }


def _distribution(verdicts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in verdicts:
        key = str(row.get("judgment") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


class _ClaimView:
    'Restore the claim dict circulating in the workflow into an attribute object recognized by Novelty Agent.\n\n    Only read-only mapping is performed, and business logic is not copied: the field definition of claim is still in claim_extraction.'

    def __init__(self, row: dict[str, Any]) -> None:
        self.claim_id = str(row.get("claim_id") or "")
        self.claim_type = str(row.get("claim_type") or "")
        self.text = str(row.get("text") or "")
        self.source_evidence_id = str(row.get("source_evidence_id") or "")
        self.concepts = list(row.get("concepts") or [])
        self.theory = str(row.get("theory") or "")
        self.mechanism = str(row.get("mechanism") or "")
        self.context = str(row.get("context") or "")
        self.keywords_en = str(row.get("keywords_en") or "")
        self.publication_year = row.get("publication_year")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "claim_text": self.text,
            "source_evidence_id": self.source_evidence_id,
            "concepts": self.concepts,
            "theory": self.theory,
            "mechanism": self.mechanism,
            "context": self.context,
            "publication_year": self.publication_year,
        }


def call_scoring_agents(state: EvaluationGraphState) -> EvaluationGraphState:
    runtime = load_agent_runtime_config(state.get("agent_config_path"), overrides=state.get("agent_runtime_overrides"))
    notify_progress(state, 'Unified Reviewer is rating' if runtime.unified_reviewer_enabled else '6 expert agents are scoring in parallel')
    judgements = run_core_agents(
        state["run_id"],
        state["profile"].rules,
        state["evidences"],
        agent_config_path=state.get("agent_config_path"),
        agent_runtime_overrides=state.get("agent_runtime_overrides"),
        paper_context=state.get("paper_context"),
        external_rule_ids=set(
            load_external_retrieval_config(state.get("agent_config_path")).evidence_rule_ids
        ),
        novelty_evidences=state.get("novelty_evidences") or [],
    )
    return {
        "judgements": judgements,
        "trace": append_trace(state, {"step": "langgraph.call_scoring_agents", "judgement_count": len(judgements)}),
    }


def retry_uncertain_agent_outputs(state: EvaluationGraphState) -> EvaluationGraphState:
    notify_progress(state, 'The second evidence recall and re-judgment is in progress for items that cannot be judged.')
    before = sum(1 for item in state["judgements"] if item.judgement.value == "uncertain")
    judgements = retry_uncertain_judgements(
        state["run_id"],
        state["profile"].rules,
        state["evidences"],
        state["judgements"],
        agent_config_path=state.get("agent_config_path"),
        agent_runtime_overrides=state.get("agent_runtime_overrides"),
        paper_context=state.get("paper_context"),
        novelty_evidences=state.get("novelty_evidences") or [],
    )
    after = sum(1 for item in judgements if item.judgement.value == "uncertain")
    return {
        "judgements": judgements,
        "trace": append_trace(
            state,
            {
                "step": "langgraph.retry_uncertain_agent_outputs",
                "uncertain_before": before,
                "uncertain_after": after,
                "changed_count": sum(
                    1
                    for old, new in zip(state["judgements"], judgements)
                    if old.judgement != new.judgement or old.evidence_ids != new.evidence_ids
                ),
            },
        ),
    }


def validate_agent_outputs(state: EvaluationGraphState) -> EvaluationGraphState:
    notify_progress(state, 'Verifying Agent output structure')
    config = load_agent_runtime_config(state.get("agent_config_path"), overrides=state.get("agent_runtime_overrides"))
    if not config.schema_validation_enabled:
        return {"schema_errors": [], "trace": append_trace(state, {"step": "langgraph.validate_agent_outputs", "error_count": 0, "disabled": True})}
    errors = validate_agent_judgements(state["profile"], state["judgements"])
    if errors:
        raise ValueError("Agent judgement schema validation failed: " + "; ".join(errors))
    return {
        "schema_errors": errors,
        "trace": append_trace(state, {"step": "langgraph.validate_agent_outputs", "error_count": len(errors)}),
    }


def gate_evidence(state: EvaluationGraphState) -> EvaluationGraphState:
    notify_progress(state, 'is checking the evidence chain and high score evidence threshold')
    ungated_judgements = list(state["judgements"])
    config = load_agent_runtime_config(state.get("agent_config_path"), overrides=state.get("agent_runtime_overrides"))
    if not config.evidence_gate_enabled:
        return {
            "ungated_judgements": ungated_judgements,
            "judgements": ungated_judgements,
            "trace": append_trace(state, {"step": "langgraph.gate_evidence", "needs_review_count": sum(1 for item in ungated_judgements if item.needs_review), "disabled": True}),
        }
    judgements = apply_evidence_gate(ungated_judgements, state["evidences"])
    return {
        "ungated_judgements": ungated_judgements,
        "judgements": judgements,
        "trace": append_trace(
            state,
            {
                "step": "langgraph.gate_evidence",
                "needs_review_count": sum(1 for item in judgements if item.needs_review),
            },
        ),
    }


def aggregate_agent_scores(state: EvaluationGraphState) -> EvaluationGraphState:
    notify_progress(state, 'Aggregating dimension scores and total score')
    # Only performs range filtering on special rules whose portraits are clearly identified as mutually exclusive types; "mixed" remains compatible.
    skipped = out_of_scope_rule_ids_of(state["profile"].rules, paper_class(state.get("paper_context")))
    score_items = score_judgements(state["profile"], state["judgements"], skipped)
    summary = aggregate_scores(state["profile"], score_items)
    return {
        "score_items": score_items,
        "summary": summary,
        "trace": append_trace(
            state,
            {
                "step": "langgraph.aggregate_agent_scores",
                "out_of_scope_rule_count": len(skipped),
                "scope_weight": summary.get("scope_weight"),
                "summary": summary,
            },
        ),
    }


def external_validation_issues(validation: dict[str, Any]) -> list[ReflectionIssue]:
    'External evidence verification conclusion → Review suggestion (severity is mapped to the existing gear).'
    severity_map = {"high": "high", "medium": "medium", "low": "info"}
    issues: list[ReflectionIssue] = []
    for finding in validation.get("findings") or []:
        issues.append(
            ReflectionIssue(
                issue_type=f"external_evidence:{finding.get('code')}",
                severity=severity_map.get(str(finding.get("severity")), "info"),
                message=str(finding.get("message") or ""),
                related_ids=[
                    str(item)
                    for item in [finding.get("claim_id"), finding.get("evidence_id"), finding.get("indicator_id")]
                    if item
                ],
                suggested_action=(
                    'Manually check the metadata and year of publication of this external evidence, or re-check the relevant documents and then run again;'
                    'This item should not be accepted as satisfied when there is insufficient external evidence.'
                ),
            )
        )
    return issues


def review_agent_outputs(state: EvaluationGraphState) -> EvaluationGraphState:
    notify_progress(state, 'is generating reflection and review suggestions')
    issues = reflect_run(state["judgements"], state["score_items"], state["evidences"])
    issues = [*issues, *external_validation_issues(state.get("external_validation") or {})]
    return {
        "reflection_issues": issues,
        "trace": append_trace(state, {"step": "langgraph.review_agent_outputs", "issue_count": len(issues)}),
    }


def audit_scoring_process(state: EvaluationGraphState) -> EvaluationGraphState:
    notify_progress(state, 'The reflective review agent is reviewing the scoring process')
    config = load_agent_runtime_config(
        state.get("agent_config_path"),
        overrides=state.get("agent_runtime_overrides"),
    )
    audit = run_reflection_audit_agent(
        state["run_id"],
        state["profile"],
        state["judgements"],
        state["score_items"],
        state["evidences"],
        state["summary"],
        state.get("paper_context"),
        state.get("reflection_issues", []),
        config,
    )
    issues = [*state.get("reflection_issues", []), *audit_to_reflection_issues(audit)]
    return {
        "audit_report": audit,
        "reflection_issues": issues,
        "trace": append_trace(
            state,
            {
                "step": "langgraph.audit_scoring_process",
                "audit_status": audit.get("audit_status"),
                "audit_issue_count": len(audit.get("audit_issues", [])),
            },
        ),
    }


def build_evaluation_graph():
    graph = StateGraph(EvaluationGraphState)
    graph.add_node("build_paper_context", build_paper_context)
    graph.add_node("ground_social_relevance", ground_social_relevance)
    graph.add_node("retrieve_external_evidence", retrieve_external_evidence)
    graph.add_node("verify_novelty", verify_novelty)
    graph.add_node("call_scoring_agents", call_scoring_agents)
    graph.add_node("validate_external_evidence", validate_external_evidence_node)
    graph.add_node("retry_uncertain_agent_outputs", retry_uncertain_agent_outputs)
    graph.add_node("validate_agent_outputs", validate_agent_outputs)
    graph.add_node("gate_evidence", gate_evidence)
    graph.add_node("aggregate_agent_scores", aggregate_agent_scores)
    graph.add_node("review_agent_outputs", review_agent_outputs)
    graph.add_node("audit_scoring_process", audit_scoring_process)

    # Sequence: Portrait → External search → Organize external comparative evidence → Grounding of social issues → Six Reviewer ratings →
    # Harness external verification → secondary recall → structure and evidence gating → scoring → reflection → review.
    # External verification is deliberately arranged before the secondary recall: after the item judged to "rely on field comparison but no external evidence" becomes uncertain,
    # There is still a chance to supplement evidence during the second recall; conversely, there is no window for remediation.
    graph.add_edge(START, "build_paper_context")
    graph.add_edge("build_paper_context", "retrieve_external_evidence")
    graph.add_edge("retrieve_external_evidence", "verify_novelty")
    graph.add_edge("verify_novelty", "ground_social_relevance")
    graph.add_edge("ground_social_relevance", "call_scoring_agents")
    graph.add_edge("call_scoring_agents", "validate_external_evidence")
    graph.add_edge("validate_external_evidence", "retry_uncertain_agent_outputs")
    graph.add_edge("retry_uncertain_agent_outputs", "validate_agent_outputs")
    graph.add_edge("validate_agent_outputs", "gate_evidence")
    graph.add_edge("gate_evidence", "aggregate_agent_scores")
    graph.add_edge("aggregate_agent_scores", "review_agent_outputs")
    graph.add_edge("review_agent_outputs", "audit_scoring_process")
    graph.add_edge("audit_scoring_process", END)
    return graph.compile()


def run_evaluation_graph(
    run_id: str,
    profile: RuleProfile,
    paper: Paper,
    evidences: list[Evidence],
    agent_config_path: str | Path | None,
    agent_runtime_overrides: dict[str, object] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    paper_context_override: dict[str, Any] | None = None,
    external_retrieval_override: dict[str, Any] | None = None,
    social_relevance_override: list[Evidence] | None = None,
) -> EvaluationGraphState:
    app = build_evaluation_graph()
    return app.invoke(
        {
            "run_id": run_id,
            "profile": profile,
            "paper": paper,
            "evidences": evidences,
            "agent_config_path": agent_config_path,
            "agent_runtime_overrides": agent_runtime_overrides,
            "paper_context_override": paper_context_override,
            "external_retrieval_override": external_retrieval_override,
            "social_relevance_override": social_relevance_override,
            "progress_callback": progress_callback,
            "trace": [],
        }
    )

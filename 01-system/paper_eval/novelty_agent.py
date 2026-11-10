'Novelty Verification Agent: Use external documentary evidence to make **limited enumeration** judgment on a single claim.\n\nIt does not generate free scores and only allows you to choose one of five levels:\n\n- There is no comparable existing study before publication of `novel`\n- `partially_novel` has new ingredients, but some of them have been covered by existing research\n- `incremental` The main body has been done, this article only does the increment\n- `already_established` has been established by existing research\n- `uncertain` Insufficient external evidence, unable to make a verdict\n\nThe core task is not to "summarize relevant literature", but to answer: **Compared to the existing research before publication, what exactly does this contribution add? **\n\nTwo things must be distinguished:\n1. The author himself claims innovation (from internal evidence `internal_evidence` in this article);\n2. External documentary evidence supports its innovation (`supporting_papers` / `contradicting_papers`, from External Evidence Set).\nWhen there is a conflict between the two, external evidence shall prevail, and the conflict shall be written into reason.\n\nWhen LLM is unavailable (no key/offline/local small model call failure), no judgment will be made, and uncertain + needs_review will be output.\nLet Harness proceed with "insufficient external evidence."'

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

from .claim_extraction import CLAIM_TYPE_LABELS
from .external_evidence import judgement_support_consistency
from .external_retrieval import ExternalRetrievalConfig
from .llm import AgentRuntimeConfig, call_plain_json
from .models import Evidence

# The section name used for the innovative verification conclusion block. The prompt word for agents.py identifies NOV-* entries by this string.
NOVELTY_EVIDENCE_SECTION = 'Innovation Verification Agent Conclusion'
NOVELTY_JUDGEMENTS = ("novel", "partially_novel", "incremental", "already_established", "uncertain")
# Levels that are meaningful only if external evidence is insufficient: falling into these two levels is not allowed when external evidence is insufficient.
NOVELTY_POSITIVE = {"novel", "partially_novel"}

NOVELTY_SYSTEM_PROMPT = (
    'You are an external innovation verification agent. Your task is not to summarize the relevant literature, but to answer a specific question:'
    '**Compared to the related research that existed before the publication of the target paper, what exactly does this claim add? **'
    'You will receive: target_claim (claim to be verified, including claim_type / concepts / theory / mechanism / context),'
    'internal_evidence (internal evidence in this article, representing **the author’s own claims**), external_evidence_set (actually retrieved,'
    'Candidate documents published before the target paper, each with relevance_score and retrieval_query),'
    'target_paper (target paper metadata and publication year).'
    'Judgment rules:'
    '1) The status of external documents can only be determined based on the given external_evidence_set. **You may not** rely on your own knowledge to claim a certain research'
    '"For the first time", "Established" or "There have been a lot of studies"; if it is not in the evidence, it means it is not there.'
    '2) It is necessary to distinguish between "the author claims innovation" (words in internal_evidence) and "external documentary evidence supports his innovation"'
    '(content that can be compared in external_evidence_set). If there is any inconsistency between the two, external evidence shall prevail.'
    '3) When external_evidence_set is empty or obviously irrelevant to claim, judgment can only be uncertain.'
    '4) When citing documents, you must use the external_evidence_id (XEV-*) given in the evidence, supporting_papers /'
    'Only titles or DOIs that actually exist in external_evidence_set can be named in contradicting_papers.'
    '5) judgment only allows: novel | partially_novel | incremental | already_established | uncertain.'
    '6) No free scores are allowed to be output; scores are calculated by Harness.'
    'Output JSON: {"judgements":[{"claim_id":"Return as is","judgment":"Enumeration value",'
    '"reason":"Concise English review explaining what is new compared with prior research","supporting_papers":["Title or DOI"],'
    '"contradicting_papers":["Title or DOI"],"external_evidence_ids":["XEV-1"],'
    '"confidence":0.0,"needs_review":true}]}. A judgment must be returned for each input claim.'
)


@dataclass
class NoveltyVerdict:
    claim_id: str
    judgment: str
    reason: str
    supporting_papers: list[str] = field(default_factory=list)
    contradicting_papers: list[str] = field(default_factory=list)
    external_evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = True
    claim_type: str = ""
    coverage: dict[str, Any] = field(default_factory=dict)
    author_claim: str = ""
    internal_evidence_id: str = ""
    llm_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "claim_type_label": CLAIM_TYPE_LABELS.get(self.claim_type, self.claim_type),
            "judgment": self.judgment,
            "reason": self.reason,
            "supporting_papers": list(self.supporting_papers),
            "contradicting_papers": list(self.contradicting_papers),
            "external_evidence_ids": list(self.external_evidence_ids),
            "confidence": round(float(self.confidence), 3),
            "needs_review": bool(self.needs_review),
            "coverage": dict(self.coverage),
            "author_claim": self.author_claim[:240],
            "internal_evidence_id": self.internal_evidence_id,
            "llm_error": self.llm_error,
        }


def coverage_of_claim(claim_id: str, external_set: list[dict[str, Any]]) -> dict[str, Any]:
    'External evidence coverage of a certain claim. Harness relies on it to decide whether to downgrade the confidence level.'
    works = [
        work
        for work in external_set
        if claim_id in [str(item) for item in (work.get("claim_ids") or [])]
        or str(work.get("claim_id") or "") == claim_id
    ]
    prior = [work for work in works if work.get("year") is not None]
    return {
        "paper_count": len(works),
        "prior_paper_count": len(prior),
        "external_evidence_ids": [str(work.get("evidence_id") or "") for work in works],
        "top_relevance": round(max((float(work.get("relevance_score") or 0.0) for work in works), default=0.0), 4),
    }


def build_novelty_payload(
    claim: Any,
    internal_evidences: list[Evidence],
    external_set: list[dict[str, Any]],
    paper: Any,
    publication_year: int | None,
) -> dict[str, Any]:
    source_id = str(getattr(claim, "source_evidence_id", "") or "")
    ordered = sorted(
        internal_evidences,
        key=lambda item: (0 if item.evidence_id == source_id else 1, item.page, item.span[0]),
    )
    return {
        "target_paper": {
            "title": str((getattr(paper, "metadata", {}) or {}).get("title") or ""),
            "publication_year": publication_year,
            "doi": str((getattr(paper, "metadata", {}) or {}).get("doi") or ""),
        },
        "target_claim": {
            "claim_id": str(getattr(claim, "claim_id", "")),
            "claim_type": str(getattr(claim, "claim_type", "")),
            "claim_type_label": CLAIM_TYPE_LABELS.get(str(getattr(claim, "claim_type", "")), ""),
            "claim_text": str(getattr(claim, "text", ""))[:500],
            "concepts": list(getattr(claim, "concepts", None) or []),
            "theory": str(getattr(claim, "theory", "") or ""),
            "mechanism": str(getattr(claim, "mechanism", "") or ""),
            "context": str(getattr(claim, "context", "") or ""),
            "author_claims_this_is_novel": True,
        },
        "internal_evidence": [
            {"evidence_id": item.evidence_id, "section": item.section, "page": item.page, "text": item.text[:500]}
            for item in ordered[:3]
        ],
        "external_evidence_set": [
            {
                "external_evidence_id": str(work.get("evidence_id") or ""),
                "title": work.get("title"),
                "year": work.get("year"),
                "venue": work.get("venue"),
                "doi": work.get("doi"),
                "citation_count": work.get("citation_count"),
                "relevance_score": work.get("relevance_score"),
                "retrieval_query": str(work.get("source_query") or "")[:120],
                "routes": work.get("routes"),
                "abstract": str(work.get("abstract") or "")[:400],
            }
            for work in external_set
        ],
    }


def normalize_novelty_output(
    raw: Any,
    claims: list[Any],
    external_set: list[dict[str, Any]],
    publication_year: int | None,
) -> list[NoveltyVerdict]:
    'Converge model output into a controlled structure: enumeration values, evidence IDs, supporting/contradicting are all checked against the warehouse.'
    known_ids = {str(work.get("evidence_id") or "") for work in external_set}
    items = raw.get("judgements") if isinstance(raw, dict) else None
    by_claim: dict[str, dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            by_claim[str(item.get("claim_id") or "")] = item

    verdicts: list[NoveltyVerdict] = []
    for claim in claims:
        claim_id = str(getattr(claim, "claim_id", ""))
        row = by_claim.get(claim_id)
        coverage = coverage_of_claim(claim_id, external_set)
        if row is None:
            verdicts.append(
                NoveltyVerdict(
                    claim_id=claim_id,
                    judgment="uncertain",
                    reason='The model does not return the innovativeness judgment of the claim and is treated as insufficient external evidence.',
                    confidence=0.0,
                    needs_review=True,
                    claim_type=str(getattr(claim, "claim_type", "")),
                    coverage=coverage,
                    author_claim=str(getattr(claim, "text", "")),
                    internal_evidence_id=str(getattr(claim, "source_evidence_id", "")),
                    llm_error="missing_in_model_output",
                )
            )
            continue
        judgment = str(row.get("judgment") or row.get("judgement") or "").strip().lower().replace(" ", "_")
        if judgment not in NOVELTY_JUDGEMENTS:
            judgment = "uncertain"
        referenced = [str(item) for item in (row.get("external_evidence_ids") or []) if isinstance(item, (str, int))]
        unknown = [item for item in referenced if item not in known_ids]
        referenced = [item for item in referenced if item in known_ids]
        supporting = [str(item) for item in (row.get("supporting_papers") or [])]
        contradicting = [str(item) for item in (row.get("contradicting_papers") or [])]
        try:
            confidence = max(0.0, min(1.0, float(row.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        needs_review = bool(row.get("needs_review", True))
        notes: list[str] = []
        if unknown:
            needs_review = True
            notes.append(f"refers to non-existent external evidence ID: {', '.join(unknown)}")

        def backed(name: str) -> bool:
            lowered = str(name or "").strip().lower()
            if not lowered:
                return False
            return any(
                lowered in str(work.get("title") or "").lower()
                or lowered == str(work.get("doi") or "").lower()
                for work in external_set
            )

        unbacked_support = [name for name in supporting if not backed(name)]
        unbacked_contra = [name for name in contradicting if not backed(name)]
        if unbacked_support or unbacked_contra:
            needs_review = True
            notes.append(
                'named documents outside the External Evidence Set:'
                + "；".join([*unbacked_support, *unbacked_contra])[:120]
            )
        cited = [
            work for work in external_set
            if str(work.get("evidence_id") or "") in referenced
        ]
        if judgment in NOVELTY_POSITIVE and not cited and not supporting:
            notes.append('is judged to be novel but no supporting documents are provided, and the confidence level is lowered by Harness.')
            needs_review = True
        if judgment in NOVELTY_POSITIVE and not coverage["paper_count"]:
            notes.append('is judged novel but the claim does not have any external reference literature, and the confidence level is lowered by Harness.')
            needs_review = True
        if cited:
            ok, note = judgement_support_consistency(judgment, cited, str(row.get("reason") or ""))
            if not ok:
                notes.append(note)
                needs_review = True
        reason = str(row.get("reason") or "").strip()
        if notes:
            reason = f"{reason} (check tip: {'；'.join(notes)})"
        verdicts.append(
            NoveltyVerdict(
                claim_id=claim_id,
                judgment=judgment,
                reason=reason[:600],
                supporting_papers=supporting,
                contradicting_papers=contradicting,
                external_evidence_ids=referenced,
                confidence=confidence,
                needs_review=needs_review,
                claim_type=str(getattr(claim, "claim_type", "")),
                coverage=coverage,
                author_claim=str(getattr(claim, "text", "")),
                internal_evidence_id=str(getattr(claim, "source_evidence_id", "")),
            )
        )
    return verdicts


def fallback_verdicts(claims: list[Any], external_set: list[dict[str, Any]], error: str) -> list[NoveltyVerdict]:
    'Determination degradation when external validation is not available (no model/offline): all uncertain + needs_review.'
    return [
        NoveltyVerdict(
            claim_id=str(getattr(claim, "claim_id", "")),
            judgment="uncertain",
            reason='The innovation verification agent is unavailable (or no available external evidence has been obtained), and no innovation judgment will be made on the claim this time.',
            external_evidence_ids=[],
            confidence=0.0,
            needs_review=True,
            claim_type=str(getattr(claim, "claim_type", "")),
            coverage=coverage_of_claim(str(getattr(claim, "claim_id", "")), external_set),
            author_claim=str(getattr(claim, "text", "")),
            internal_evidence_id=str(getattr(claim, "source_evidence_id", "")),
            llm_error=error[:200],
        )
        for claim in claims
    ]


def verdict_blocks(verdicts: list[dict[str, Any]], paper_id: str) -> list[Evidence]:
    'Innovative verification conclusion → Quotable evidence block (NOV-*), inserted into rules that rely on field comparison to make B4 contribution judgments.\n\n    These blocks are not "evidence", but "checked judgments": write down the location, reasons, which external evidence IDs were cited,\n    This not only prevents the scoring agent from having to redo the comparison itself, but also allows Harness to trace back NOV-* → XEV-* → external papers.'
    blocks: list[Evidence] = []
    for index, row in enumerate(verdicts, start=1):
        lines = [
            f"【Innovative verification conclusion NOV-{index}｜claim {row.get('claim_id')}｜{row.get('claim_type_label') or row.get('claim_type')}】",
            f"Judgment: {row.get('judgment')} (confidence {row.get('confidence')}; need to review {row.get('needs_review')})",
            f"Author's assertion (internal evidence of this article {row.get('internal_evidence_id')}): {row.get('author_claim')}",
            f"External literature comparison conclusion: {row.get('reason')}",
            f"External evidence cited: {', '.join(str(item) for item in (row.get('external_evidence_ids') or [])) or 'None'}",
        ]
        if row.get("supporting_papers"):
            lines.append(f"Documents supporting this judgment: {'；'.join(str(item) for item in row['supporting_papers'])}")
        if row.get("contradicting_papers"):
            lines.append(f"Documents constituting comparison/counterexample: {'；'.join(str(item) for item in row['contradicting_papers'])}")
        coverage = row.get("coverage") or {}
        lines.append(
            f"External evidence coverage: {coverage.get('paper_count', 0)} articles (including {coverage.get('prior_paper_count', 0)} articles with year)"
        )
        text = "\n".join(lines)
        blocks.append(
            Evidence(
                evidence_id=f"NOV-{index}",
                paper_id=paper_id,
                section=NOVELTY_EVIDENCE_SECTION,
                page=0,
                span=(0, len(text)),
                text=text,
                text_hash=hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
                source_type="novelty_verdict",
            )
        )
    return blocks


def run_novelty_verification(
    claims: list[Any],
    internal_evidences: list[Evidence],
    external_set: list[dict[str, Any]],
    paper: Any,
    publication_year: int | None,
    config: AgentRuntimeConfig | None,
    retrieval_config: ExternalRetrievalConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    'runs innovative verification. Return {"verdicts": [...], "unavailable": bool, "reason": str}.\n\n    Only make one batch call (all claims together). If it fails, the whole call will be downgraded to uncertain - no retrying one by one.\n    Because "insufficient external evidence" is itself a permissible conclusion, retrying will just burn the call.'
    if not claims:
        return {"verdicts": [], "unavailable": True, "reason": "no_claims"}
    if not external_set:
        return {
            "verdicts": [item.to_dict() for item in fallback_verdicts(claims, external_set, "external_evidence_unavailable")],
            "unavailable": True,
            "reason": "external_evidence_unavailable",
        }
    if config is None or not getattr(config, "enable_llm_agents", True):
        return {
            "verdicts": [item.to_dict() for item in fallback_verdicts(claims, external_set, "llm_disabled")],
            "unavailable": True,
            "reason": "llm_disabled",
        }
    payload = {
        "target_paper": build_novelty_payload(claims[0], internal_evidences, external_set, paper, publication_year)["target_paper"],
        "claims": [
            build_novelty_payload(claim, internal_evidences, external_set, paper, publication_year)
            for claim in claims
        ],
    }
    if progress_callback:
        progress_callback(f"Innovation Verification Agent is comparing {len(claims)} claims with {len(external_set)} external documents")
    try:
        raw = call_plain_json(NOVELTY_SYSTEM_PROMPT, payload, config)
    except Exception as exc:
        return {
            "verdicts": [item.to_dict() for item in fallback_verdicts(claims, external_set, str(exc))],
            "unavailable": True,
            "reason": f"llm_call_failed: {str(exc)[:160]}",
        }
    verdicts = normalize_novelty_output(raw, claims, external_set, publication_year)
    return {"verdicts": [item.to_dict() for item in verdicts], "unavailable": False, "reason": "ok"}

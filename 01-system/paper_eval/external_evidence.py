'External Evidence Repository: External literature warehouse + external evidence verification + reverse traceability.\n\nStorage partitions (deliberately not mixed):\n\n- **Internal Evidence Repository**: PDF parsed evidence block (EV-*) of this article, source_type=paragraph.\n- **External Literature Repository**: Published literature retrieved externally (XEV-*), source_type=external.\n\nBoth bring provenance, and ultimately need to be able to\n`Score → Judgment → Claim → External Evidence → External Paper` Walk all the way back,\nSo each record here also records claim_id, evidence_id, paper_id and the search source.\n\nHarness’ External Evidence Validation is also here:\nWhether the external paper actually exists, whether the DOI/title/year is consistent, whether it is earlier than the target paper, whether it is related to the claim,\nDetermine whether the external_evidence_id referenced by Agent exists and whether the supporting/contradicting paper really supports it.\n**No model readme is accepted for any field**: Only what is in the warehouse is recognized.'

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .external_retrieval import CROSSREF_ENDPOINT
from .models import Evidence, Judgement, now_iso

# The section name of the external evidence block. literature_search also uses this string when generating evidence blocks.
EXTERNAL_EVIDENCE_SECTION = 'External documentation (multiple recall)'
INTERNAL_EVIDENCE_SECTION = 'Evidence for this article'


@dataclass
class EvidenceRecord:
    'A record in the Internal Evidence Repository.'

    evidence_id: str
    paper_id: str
    section: str
    page: int
    text: str
    text_hash: str
    source_type: str = "paragraph"


@dataclass
class ExternalPaperRecord:
    'A record in External Literature Repository: one external literature + all provenance.'

    paper_id: str
    evidence_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str
    doi: str
    abstract: str
    citation_count: int | None
    references: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    retrieval_source: str = ""
    retrieval_query: str = ""
    retrieval_time: str = ""
    relevance_score: float = 0.0
    claim_id: str = ""
    claim_ids: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    citation_relationship: str = ""
    rank_features: dict[str, float] = field(default_factory=dict)
    # Metadata verification: ok / mismatch / unchecked
    metadata_check: str = "unchecked"
    metadata_notes: list[str] = field(default_factory=list)
    crossref_record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "evidence_id": self.evidence_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "abstract": self.abstract[:600],
            "citation_count": self.citation_count,
            "references": list(self.references)[:20],
            "citations": list(self.citations)[:20],
            "retrieval_source": self.retrieval_source,
            "retrieval_query": self.retrieval_query,
            "retrieval_time": self.retrieval_time,
            "relevance_score": self.relevance_score,
            "claim_id": self.claim_id,
            "claim_ids": list(self.claim_ids),
            "routes": list(self.routes),
            "citation_relationship": self.citation_relationship,
            "rank_features": dict(self.rank_features),
            "metadata_check": self.metadata_check,
            "metadata_notes": list(self.metadata_notes),
        }


@dataclass
class ExternalEvidenceRepository:
    'Two warehouses + reverse traceability table. Internal / external partition storage, do not write to each other.'

    paper_id: str
    internal: list[EvidenceRecord] = field(default_factory=list)
    external: list[ExternalPaperRecord] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    # --- Write ---
    def add_internal(self, evidences: list[Evidence]) -> None:
        for item in evidences:
            if getattr(item, "source_type", "paragraph") == "external":
                continue
            self.internal.append(
                EvidenceRecord(
                    evidence_id=item.evidence_id,
                    paper_id=item.paper_id,
                    section=item.section,
                    page=item.page,
                    text=item.text,
                    text_hash=item.text_hash,
                    source_type="paragraph",
                )
            )

    def add_external_works(self, works: list[dict[str, Any]]) -> None:
        for work in works:
            evidence_id = str(work.get("evidence_id") or "")
            claim_ids = [str(item) for item in (work.get("claim_ids") or [])]
            self.external.append(
                ExternalPaperRecord(
                    paper_id=str(work.get("paper_id") or work.get("doi") or work.get("title") or ""),
                    evidence_id=evidence_id,
                    title=str(work.get("title") or ""),
                    authors=[str(item) for item in (work.get("authors") or [])],
                    year=work.get("year") if isinstance(work.get("year"), int) else None,
                    venue=str(work.get("venue") or ""),
                    doi=str(work.get("doi") or ""),
                    abstract=str(work.get("abstract") or ""),
                    citation_count=work.get("citation_count") if isinstance(work.get("citation_count"), int) else None,
                    references=[str(item) for item in (work.get("references") or [])],
                    citations=[str(item) for item in (work.get("citations") or [])],
                    retrieval_source=str(work.get("retrieval_source") or ""),
                    retrieval_query=str(work.get("source_query") or ""),
                    retrieval_time=str(work.get("retrieval_time") or now_iso()),
                    relevance_score=float(work.get("relevance_score") or 0.0),
                    claim_id=claim_ids[0] if claim_ids else "",
                    claim_ids=claim_ids,
                    routes=[str(item) for item in (work.get("routes") or [])],
                    citation_relationship=str(work.get("citation_relationship") or ""),
                    rank_features={str(k): float(v) for k, v in (work.get("rank_features") or {}).items()},
                    metadata_check=str(work.get("metadata_check") or "unchecked"),
                    metadata_notes=[str(item) for item in (work.get("metadata_notes") or [])],
                )
            )

    def add_claims(self, claims: list[Any]) -> None:
        for claim in claims:
            if hasattr(claim, "to_dict"):
                self.claims.append(claim.to_dict())
            elif isinstance(claim, dict):
                self.claims.append(claim)

    # --- Query (for Harness/Report for reverse tracing) ---
    def external_by_evidence_id(self, evidence_id: str) -> ExternalPaperRecord | None:
        return next((item for item in self.external if item.evidence_id == evidence_id), None)

    def internal_by_id(self, evidence_id: str) -> EvidenceRecord | None:
        return next((item for item in self.internal if item.evidence_id == evidence_id), None)

    def trace(self, judgement_indicator_id: str | None = None) -> dict[str, Any]:
        'Reverse traceability table: claim → external evidence → external paper, and whether the internal evidence ID exists.'
        rows: list[dict[str, Any]] = []
        for claim in self.claims:
            source_id = str(claim.get("source_evidence_id") or "")
            external_rows = [item for item in self.external if claim.get("claim_id") in (item.claim_ids or [item.claim_id])]
            rows.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "claim_type": claim.get("claim_type"),
                    "internal_evidence_id": source_id,
                    "internal_evidence_found": self.internal_by_id(source_id) is not None,
                    "external_evidence_ids": [item.evidence_id for item in external_rows],
                    "external_paper_ids": [item.paper_id for item in external_rows],
                    "external_dois": [item.doi for item in external_rows if item.doi],
                }
            )
        return {
            "paper_id": self.paper_id,
            "created_at": self.created_at,
            "internal_count": len(self.internal),
            "external_count": len(self.external),
            "claims": rows,
        }

    def counts(self) -> dict[str, int]:
        return {"internal": len(self.internal), "external": len(self.external), "claims": len(self.claims)}


def build_repositories(
    paper_id: str,
    internal_evidences: list[Evidence],
    claims: list[Any],
    external_works: list[dict[str, Any]],
) -> ExternalEvidenceRepository:
    repo = ExternalEvidenceRepository(paper_id=paper_id)
    repo.add_internal(internal_evidences)
    repo.add_claims(claims)
    repo.add_external_works(external_works)
    return repo


# --------------------------------------------------------------------------------------
# Crossref metadata verification
# --------------------------------------------------------------------------------------

def _normalized_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _title_similarity(left: str, right: str) -> float:
    a, b = _normalized_title(left), _normalized_title(right)
    if not a or not b:
        return 0.0
    shorter, longer = sorted((a, b), key=len)
    if shorter in longer:
        return 0.6 + 0.4 * len(shorter) / len(longer)
    return 0.0


def _crossref_payload(doi: str, timeout: int, mailto: str = "") -> dict[str, Any]:
    url = f"{CROSSREF_ENDPOINT}/{urllib.parse.quote(doi)}"
    if mailto:
        url += "?" + urllib.parse.urlencode({"mailto": mailto})
    agent = f"paper-eval/1.0 (mailto:{mailto})" if mailto else "paper-eval/1.0"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _crossref_record(payload: Any) -> dict[str, Any]:
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, dict):
        return {}
    titles = message.get("title") or []
    title = str(titles[0]).strip() if isinstance(titles, list) and titles else str(titles or "").strip()
    parts = ((message.get("issued") or {}).get("date-parts") or [[None]]) if isinstance(message.get("issued"), dict) else [[None]]
    year = parts[0][0] if parts and isinstance(parts[0], list) and parts[0] else None
    authors = []
    for entry in message.get("author") or []:
        if not isinstance(entry, dict):
            continue
        name = " ".join(
            piece for piece in (str(entry.get("given") or "").strip(), str(entry.get("family") or "").strip()) if piece
        )
        if name:
            authors.append(name)
    containers = message.get("container-title") or []
    return {
        "doi": str(message.get("DOI") or ""),
        "title": title,
        "year": year,
        "authors": authors[:10],
        "venue": str(containers[0]) if isinstance(containers, list) and containers else "",
    }


def compare_metadata(work: dict[str, Any], record: dict[str, Any]) -> tuple[str, list[str]]:
    'compares DOI/Title/Year. Return (ok|mismatch, notes).'
    notes: list[str] = []
    status = "ok"
    record_title = str(record.get("title") or "")
    if record_title:
        similarity = _title_similarity(str(work.get("title") or ""), record_title)
        if similarity < 0.5:
            status = "mismatch"
            notes.append(f"Title inconsistency: Search result \"{str(work.get('title'))[:60]}\" vs Crossref \"{record_title[:60]}\"")
    record_year = record.get("year")
    work_year = work.get("year")
    if record_year and work_year and int(record_year) != int(work_year):
        status = "mismatch"
        notes.append(f"Year inconsistent: Search results {work_year} vs Crossref {record_year}")
    if not record_title and not record_year:
        status = "mismatch"
        notes.append('Crossref did not return an available title and year, and the existence of this document could not be confirmed.')
    return status, notes


def verify_crossref(
    works: list[dict[str, Any]],
    config: Any,
    *,
    cache_dir: str | Path | None = None,
    fetcher: Callable[[str], dict[str, Any]] | None = None,
    max_checks: int = 8,
) -> list[dict[str, Any]]:
    'Perform Crossref verification on external documents with DOI one by one, and write the results back to works (metadata_check / metadata_notes).\n\n    Verification failure is not fatal: network failure time stamp `unchecked` and write the reason in notes, Harness will respond accordingly\n    "External metadata not verified" is included in the review prompt instead of failing the entire score.'
    checked = 0
    for work in works:
        doi = str(work.get("doi") or (work.get("external_ids") or {}).get("DOI") or "").strip()
        if not doi or checked >= max_checks:
            if not doi:
                work["metadata_check"] = "unchecked"
                work.setdefault("metadata_notes", []).append('No DOI, no Crossref check')
            continue
        checked += 1
        cache_file = None
        if cache_dir:
            digest = hashlib.sha1(f"crossref|{doi}".encode("utf-8")).hexdigest()[:16]
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)
            cache_file = directory / f"cr_{digest}.json"
        try:
            if cache_file and cache_file.exists():
                record = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                payload = fetcher(doi) if fetcher is not None else _crossref_payload(
                    doi, int(getattr(config, "timeout_seconds", 20)), str(getattr(config, "mailto", ""))
                )
                record = _crossref_record(payload)
                if cache_file and record:
                    cache_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:  # External unavailable not fatal
            work["metadata_check"] = "unchecked"
            work.setdefault("metadata_notes", []).append(f"Crossref verification failed: {str(exc)[:120]}")
            continue
        status, notes = compare_metadata(work, record)
        work["metadata_check"] = status
        work["metadata_notes"] = [*(work.get("metadata_notes") or []), *notes]
        work["crossref_record"] = record
        if record.get("doi") and not work.get("doi"):
            work["doi"] = record["doi"]
    return works


# --------------------------------------------------------------------------------------
# External Evidence Validation (Harness side, all certainty)
# --------------------------------------------------------------------------------------

EXTERNAL_EVIDENCE_RULES = (
    "A-LITERATURE-POSITIONING-V1",
    "A-GAP-PUZZLE-V1",
    "B-CONCEPT-MECHANISM-NOVELTY-V1",
    "B-DIALOGUE-ADVANCE-V1",
    "B-THEORY-INTEGRATION-V1",
    "B-INTERESTINGNESS-V1",
)
# These rules must be judged by "field comparison" (research gaps/innovation/theoretical contributions), and they are not allowed to be satisfied without external evidence.
DOMAIN_COMPARISON_MARKERS = (
    "gap", "puzzle", "novelty", "novel", "literature", "dialogue", "theory-integration",
    "interestingness", "contribution", 'Innovation', 'Blank', 'notch', 'Literature', 'Theory',
)

CONTRADICTION_MARKERS = (
    "no overlap", "not overlap", "different", "does not", "not the same", "unrelated",
    'overlaps only in', 'different mechanisms', 'Different situations', 'does not constitute', 'has no overlap', 'Peripheral',
)
LEADING_MARKERS = ("first", 'First time', "no prior", "no existing", 'None yet', 'not seen', 'Lack of research', "nobody")
SUPPORT_MARKERS = (
    "overlap", "same mechanism", "same question", "prior work", "extends", "builds on",
    'consistent', 'overlap', 'Same mechanism', 'Same problem', 'has been studied', 'follows the', 'Undertake',
)


def rule_needs_domain_comparison(indicator_id: str) -> bool:
    lowered = str(indicator_id or "").lower()
    return any(marker in lowered for marker in DOMAIN_COMPARISON_MARKERS)


def judgement_support_consistency(
    stance: str,
    cited_papers: list[dict[str, Any]],
    reason: str,
) -> tuple[bool, str]:
    "supporting / contradicting paper whether it really supports corresponding judgment.\n\n    The criterion is a two-level check of certainty, and the second model is not invited to evaluate:\n    1. The citation position must be consistent with the text of the reason (supporting must be cited if there is overlap, contradicting must be quoted if it is different/no overlap);\n    2. The cited document itself must be able to carry this direction (citation relationship, relevance, existence).\n    Inconsistencies only record flag: Harness does not rewrite the model's reason."
    if not cited_papers:
        return False, 'does not cite any external literature and cannot check and determine the direction.'
    text = str(reason or "").lower()
    if stance == "novel" and any(marker in text for marker in SUPPORT_MARKERS) and not any(
        marker in text for marker in CONTRADICTION_MARKERS
    ):
        return False, 'is judged as novel, but the statement "overlap of existing research" appears in the reason, and the supporting/contradicting direction may be reversed.'
    if stance in {"already_established", "incremental"} and any(marker in text for marker in LEADING_MARKERS):
        return False, f"Judgment {stance} However, there are \"first time/not yet\" expressions in the reason, and the direction may be reversed.\nThe relevance of the external literature cited by"
    weak = [work for work in cited_papers if float(work.get("relevance_score") or 0.0) <= 0.0]
    if len(weak) == len(cited_papers):
        return False, 'is all 0, which is not enough to support this judgment.'
    return True, 'The reference bit is consistent with the judgment direction\nExternal evidence verification for'


def validate_external_evidence(
    *,
    claims: list[dict[str, Any]],
    external_set: list[dict[str, Any]],
    novelty_judgements: list[dict[str, Any]],
    target_year: int | None,
    indicator_ids: list[str] | None = None,
    min_external_papers: int = 2,
    low_coverage_confidence_cap: float = 0.4,
) -> dict[str, Any]:
    'Harness. Return findings (each with code / severity / message) without throwing exceptions.\n\n    Findings will be recorded in judgment.review_flags and reflection_issues, and confidence will be lowered if necessary.\n    But judgment and reason will never be rewritten - the judgment lies with the model, and the effectiveness control lies with Harness.'
    findings: list[dict[str, Any]] = []
    known_evidence_ids = {str(work.get("evidence_id") or "") for work in external_set}
    known_by_id = {str(work.get("evidence_id") or ""): work for work in external_set}
    claim_by_id = {str(claim.get("claim_id") or ""): claim for claim in claims}
    target_year_int = int(target_year) if target_year else None

    def add(code: str, severity: str, message: str, **extra: Any) -> None:
        findings.append({"code": code, "severity": severity, "message": message, **extra})

    if not external_set:
        add(
            "external_evidence_unavailable",
            "high",
            'No external documentary evidence was obtained this time: all judgments relying on field comparisons shall not be considered satisfied.',
        )

    # 1-4) External documents one by one: existence, metadata, time, relevance
    for work in external_set:
        evidence_id = str(work.get("evidence_id") or "")
        title = str(work.get("title") or "")
        if not title or not str(work.get("paper_id") or ""):
            add("external_paper_missing_identity", "high", f"{evidence_id} The title or paper_id is missing, and the existence of the document cannot be confirmed.\nThe DOI/title/year of", evidence_id=evidence_id)
        if work.get("metadata_check") == "mismatch":
            add(
                "external_metadata_mismatch",
                "high",
                f"{evidence_id} \"{title[:50]}\" does not match the Crossref: {'；'.join(work.get('metadata_notes') or [])[:160]}",
                evidence_id=evidence_id,
            )
        elif work.get("metadata_check") != "ok":
            add(
                "external_metadata_unverified",
                "low",
                f"{evidence_id} \"{title[:50]}\" did not complete Crossref metadata verification ({'; '.join(work.get('metadata_notes') or [])[:80] or 'unverified'}).",
                evidence_id=evidence_id,
            )
        if target_year_int and work.get("year") and int(work["year"]) > target_year_int:
            add(
                "external_time_leak",
                "high",
                f"{evidence_id} \"{title[:50]}\" was published in {work['year']}, which is later than the target paper {target_year_int}, and cannot be used as an innovative priori.",
                evidence_id=evidence_id,
            )
        if float(work.get("relevance_score") or 0.0) <= 0.0 and not (work.get("claim_ids") or []):
            add(
                "external_evidence_irrelevant",
                "medium",
                f"{evidence_id} \"{title[:50]}\" has no corresponding claim and the correlation is 0, which is irrelevant to this judgment.",
                evidence_id=evidence_id,
            )

    # 5-6) Novelty judgment one by one: whether the cited evidence ID exists and whether the supporting/contradicting is tenable
    for row in novelty_judgements:
        claim_id = str(row.get("claim_id") or "")
        referenced = [str(item) for item in (row.get("external_evidence_ids") or [])]
        unknown = [item for item in referenced if item not in known_evidence_ids]
        if unknown:
            add(
                "external_evidence_id_not_found",
                "high",
                f"{claim_id} references non-existent external evidence ID: {', '.join(unknown)}. Models must not supplement the literature on their own.",
                claim_id=claim_id,
            )
        cited = [known_by_id[item] for item in referenced if item in known_by_id]
        for stance_field in ("supporting_papers", "contradicting_papers"):
            named = [str(item) for item in (row.get(stance_field) or [])]
            unbacked = [
                name for name in named
                if not any(
                    name.lower() in str(work.get("title") or "").lower() or name == str(work.get("doi") or "")
                    for work in cited
                )
            ]
            if unbacked:
                add(
                    "external_stance_paper_not_in_evidence_set",
                    "high",
                    f"{stance_field} of {claim_id} named a document that is not in the External Evidence Set: {', '.join(unbacked)[:120]}.",
                    claim_id=claim_id,
                )
        stance = str(row.get("judgment") or "")
        if stance in {"novel", "partially_novel"}:
            check_papers = cited or []
            ok, note = judgement_support_consistency(stance, check_papers, str(row.get("reason") or ""))
            if not ok:
                add("external_stance_inconsistent", "medium", f"{claim_id} is judged {stance} but {note}.", claim_id=claim_id)
            coverage = row.get("coverage") or {}
            prior_count = int(coverage.get("prior_paper_count") or 0)
            if prior_count < int(min_external_papers):
                add(
                    "low_prior_literature_coverage",
                    "high",
                    f"{claim_id} is judged as {stance}, but the prior literature coverage is insufficient ({prior_count} < {int(min_external_papers)}),"
                    'The confidence level of this judgment has been lowered.\nThe source_evidence_id must have the form EV-<id>.',
                    claim_id=claim_id,
                    confidence_cap=float(low_coverage_confidence_cap),
                )
        claim = claim_by_id.get(claim_id)
        if claim and claim.get("source_evidence_id"):
            source_id = str(claim["source_evidence_id"])
            if not source_id.startswith(("EV-", "EXT-", "XEV-")) and source_id:
                add("claim_source_evidence_invalid", "medium", f"{claim_id} is suspicious.", claim_id=claim_id)

    # 7) Dependence field comparison rules: no satisfaction is allowed without external evidence
    coverage_ok = bool(external_set)
    for indicator_id in indicator_ids or []:
        if not coverage_ok and rule_needs_domain_comparison(indicator_id):
            add(
                "external_comparison_required_but_missing",
                "high",
                f"{indicator_id} must be judged by field comparison, but there is no external documentary evidence this time: it must not be satisfied.",
                indicator_id=indicator_id,
            )

    high = [item for item in findings if item["severity"] == "high"]
    return {
        "checked": True,
        "external_paper_count": len(external_set),
        "claim_count": len(claims),
        "novelty_judgement_count": len(novelty_judgements),
        "finding_count": len(findings),
        "high_severity_count": len(high),
        "findings": findings,
        "needs_review": any(item["severity"] in {"high", "medium"} for item in findings),
        "target_year": target_year_int,
    }


def apply_external_validation_to_judgements(
    judgements: list[Any],
    validation: dict[str, Any],
) -> list[Any]:
    'Put the external verification conclusion into the scoring judgment (only review_flags / needs_review / confidence).'
    from dataclasses import replace

    by_indicator: dict[str, list[dict[str, Any]]] = {}
    for finding in validation.get("findings") or []:
        indicator_id = str(finding.get("indicator_id") or "")
        if indicator_id:
            by_indicator.setdefault(indicator_id, []).append(finding)

    updated: list[Any] = []
    for judgement in judgements:
        flags: list[str] = []
        confidence = judgement.confidence
        if not validation.get("external_paper_count") and rule_needs_domain_comparison(judgement.indicator_id):
            if judgement.judgement.value in {"satisfied", "mostly"}:
                # relies on field comparison but has no external evidence at all: the positive judgment is not traceable → it becomes impossible to judge (the same hard threshold caliber as evidence gating).
                updated.append(
                    replace(
                        judgement,
                        judgement=Judgement.UNCERTAIN,
                        confidence=min(confidence, 0.25),
                        needs_review=True,
                        review_flags=[
                            *judgement.review_flags,
                            'External evidence verification: This judgment relies on field comparison, but without any external documentary evidence, it has become impossible to judge.',
                        ],
                    )
                )
                continue
        for finding in by_indicator.get(judgement.indicator_id, []):
            flags.append(f"External evidence verification ({finding['code']}): {finding['message']}")
            if finding.get("confidence_cap") is not None:
                confidence = min(confidence, float(finding["confidence_cap"]))
        if flags:
            updated.append(
                replace(
                    judgement,
                    confidence=confidence,
                    needs_review=True,
                    review_flags=[*judgement.review_flags, *flags],
                )
            )
        else:
            updated.append(judgement)
    return updated


def apply_coverage_confidence_cap(
    novelty_judgements: list[dict[str, Any]],
    validation: dict[str, Any],
    cap: float,
) -> list[dict[str, Any]]:
    'Novelty Agent is judged novel but the prior literature coverage is insufficient: confidence is automatically lowered and marked as needs_review.'
    capped: list[dict[str, Any]] = []
    low_coverage_claims = {
        str(item.get("claim_id") or "")
        for item in validation.get("findings") or []
        if item.get("code") == "low_prior_literature_coverage"
    }
    for row in novelty_judgements:
        if str(row.get("claim_id") or "") in low_coverage_claims:
            row = {
                **row,
                "confidence": min(float(row.get("confidence") or 0.0), float(cap)),
                "needs_review": True,
                "confidence_capped_by": "low_prior_literature_coverage",
            }
        capped.append(row)
    return capped


def repository_to_dict(repo: ExternalEvidenceRepository) -> dict[str, Any]:
    return {
        "paper_id": repo.paper_id,
        "created_at": repo.created_at,
        "counts": repo.counts(),
        "internal_repository": [
            {
                "evidence_id": item.evidence_id,
                "paper_id": item.paper_id,
                "section": item.section,
                "page": item.page,
                "text_hash": item.text_hash,
                "text_preview": item.text[:160],
            }
            for item in repo.internal
        ],
        "external_repository": [item.to_dict() for item in repo.external],
        "claims": repo.claims,
        "traceability": repo.trace(),
    }

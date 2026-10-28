"""External Evidence Repository：外部文献仓库 + 外部证据验证 + 反向追溯。

存储分区（刻意不混）：

- **Internal Evidence Repository**：PDF 解析出来的本文证据块（EV-*），source_type=paragraph。
- **External Literature Repository**：外部检索到的已发表文献（XEV-*），source_type=external。

两者都带 provenance，最终要能从
`Score → Judgment → Claim → External Evidence → External Paper` 一路走回去，
所以这里的每条记录都同时记 claim_id、evidence_id、paper_id 和检索来路。

Harness 的 External Evidence Validation 也在这里：
外部论文是否真实存在、DOI/标题/年份是否一致、是否早于目标论文、是否与 claim 相关、
Agent 引用的 external_evidence_id 是否存在、supporting/contradicting paper 是否真的支持判断。
**任何一个字段都不接受模型自述**：只认仓库里有什么。
"""

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

# 外部证据块的 section 名。literature_search 生成证据块时用的也是这个字符串。
EXTERNAL_EVIDENCE_SECTION = "外部文献（多路召回）"
INTERNAL_EVIDENCE_SECTION = "本文证据"


@dataclass
class EvidenceRecord:
    """Internal Evidence Repository 的一条记录。"""

    evidence_id: str
    paper_id: str
    section: str
    page: int
    text: str
    text_hash: str
    source_type: str = "paragraph"


@dataclass
class ExternalPaperRecord:
    """External Literature Repository 的一条记录：一篇外部文献 + 全部 provenance。"""

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
    # 元数据校验：ok / mismatch / unchecked
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
    """两个仓库 + 反向追溯表。internal / external 分区存储，互不写入。"""

    paper_id: str
    internal: list[EvidenceRecord] = field(default_factory=list)
    external: list[ExternalPaperRecord] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    # --- 写入 ---
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

    # --- 查询（供 Harness / 报告做反向追溯） ---
    def external_by_evidence_id(self, evidence_id: str) -> ExternalPaperRecord | None:
        return next((item for item in self.external if item.evidence_id == evidence_id), None)

    def internal_by_id(self, evidence_id: str) -> EvidenceRecord | None:
        return next((item for item in self.internal if item.evidence_id == evidence_id), None)

    def trace(self, judgement_indicator_id: str | None = None) -> dict[str, Any]:
        """反向追溯表：claim → 外部证据 → 外部论文，以及内部证据 ID 是否存在。"""
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
    agent = f"literature-eval/1.0 (mailto:{mailto})" if mailto else "literature-eval/1.0"
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
    """比对 DOI / 标题 / 年份。返回 (ok|mismatch, notes)。"""
    notes: list[str] = []
    status = "ok"
    record_title = str(record.get("title") or "")
    if record_title:
        similarity = _title_similarity(str(work.get("title") or ""), record_title)
        if similarity < 0.5:
            status = "mismatch"
            notes.append(f"标题不一致：检索结果「{str(work.get('title'))[:60]}」 vs Crossref「{record_title[:60]}」")
    record_year = record.get("year")
    work_year = work.get("year")
    if record_year and work_year and int(record_year) != int(work_year):
        status = "mismatch"
        notes.append(f"年份不一致：检索结果 {work_year} vs Crossref {record_year}")
    if not record_title and not record_year:
        status = "mismatch"
        notes.append("Crossref 未返回可用标题与年份，无法确认该文献存在")
    return status, notes


def verify_crossref(
    works: list[dict[str, Any]],
    config: Any,
    *,
    cache_dir: str | Path | None = None,
    fetcher: Callable[[str], dict[str, Any]] | None = None,
    max_checks: int = 8,
) -> list[dict[str, Any]]:
    """对带 DOI 的外部文献逐条做 Crossref 校验，结果写回 works（metadata_check / metadata_notes）。

    校验失败不致命：网络不通时标 `unchecked` 并在 notes 里写原因，Harness 会据此把
    "外部元数据未校验"记进复核提示，而不是让整次评分失败。
    """
    checked = 0
    for work in works:
        doi = str(work.get("doi") or (work.get("external_ids") or {}).get("DOI") or "").strip()
        if not doi or checked >= max_checks:
            if not doi:
                work["metadata_check"] = "unchecked"
                work.setdefault("metadata_notes", []).append("无 DOI，未做 Crossref 校验")
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
        except Exception as exc:  # 外部不可用不致命
            work["metadata_check"] = "unchecked"
            work.setdefault("metadata_notes", []).append(f"Crossref 校验失败：{str(exc)[:120]}")
            continue
        status, notes = compare_metadata(work, record)
        work["metadata_check"] = status
        work["metadata_notes"] = [*(work.get("metadata_notes") or []), *notes]
        work["crossref_record"] = record
        if record.get("doi") and not work.get("doi"):
            work["doi"] = record["doi"]
    return works


# --------------------------------------------------------------------------------------
# External Evidence Validation（Harness 侧，全部确定性）
# --------------------------------------------------------------------------------------

EXTERNAL_EVIDENCE_RULES = (
    "A-LITERATURE-POSITIONING-V1",
    "A-GAP-PUZZLE-V1",
    "B-CONCEPT-MECHANISM-NOVELTY-V1",
    "B-DIALOGUE-ADVANCE-V1",
    "B-THEORY-INTEGRATION-V1",
    "B-INTERESTINGNESS-V1",
)
# 这些规则必须靠"领域比较"才能判断（研究空白 / 创新性 / 理论贡献），没外部证据不许给 satisfied。
DOMAIN_COMPARISON_MARKERS = (
    "gap", "puzzle", "novelty", "novel", "literature", "dialogue", "theory-integration",
    "interestingness", "contribution", "创新", "空白", "缺口", "文献", "理论",
)

CONTRADICTION_MARKERS = (
    "no overlap", "not overlap", "different", "does not", "not the same", "unrelated",
    "重叠只在", "不同机制", "不同情境", "不构成", "并无重叠", "外围",
)
LEADING_MARKERS = ("first", "首次", "no prior", "no existing", "尚无", "未见", "缺乏研究", "nobody")
SUPPORT_MARKERS = (
    "overlap", "same mechanism", "same question", "prior work", "extends", "builds on",
    "一致", "重叠", "同机制", "同问题", "已有研究", "沿用了", "承接",
)


def rule_needs_domain_comparison(indicator_id: str) -> bool:
    lowered = str(indicator_id or "").lower()
    return any(marker in lowered for marker in DOMAIN_COMPARISON_MARKERS)


def judgement_support_consistency(
    stance: str,
    cited_papers: list[dict[str, Any]],
    reason: str,
) -> tuple[bool, str]:
    """supporting / contradicting paper 是否真的支持对应判断。

    判据是确定性的两层检查，不请第二个模型来评：
    1. 引用位与理由文本一致（说"有重叠"就得引 supporting，说"不同/无重叠"就得引 contradicting）；
    2. 引用的那篇文献本身要能承载这个方向（引文关系、相关度、是否存在）。
    不一致只记 flag：Harness 不改写模型的 reason。
    """
    if not cited_papers:
        return False, "未引用任何外部文献，无法核对判断方向"
    text = str(reason or "").lower()
    if stance == "novel" and any(marker in text for marker in SUPPORT_MARKERS) and not any(
        marker in text for marker in CONTRADICTION_MARKERS
    ):
        return False, "判 novel 但理由里出现了「已有研究重叠」的表述，supporting/contradicting 方向可能颠倒"
    if stance in {"already_established", "incremental"} and any(marker in text for marker in LEADING_MARKERS):
        return False, f"判 {stance} 但理由里出现「首次/尚无」类表述，方向可能颠倒"
    weak = [work for work in cited_papers if float(work.get("relevance_score") or 0.0) <= 0.0]
    if len(weak) == len(cited_papers):
        return False, "引用的外部文献相关度全为 0，不足以支撑该判断"
    return True, "引用位与判断方向一致"


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
    """Harness 的外部证据校验。返回 findings（每条带 code / severity / message），不抛异常。

    findings 会被记进 judgement.review_flags 与 reflection_issues，必要时下调 confidence，
    但绝不改写 judgement 与 reason——判断权在模型，有效性控制权在 Harness。
    """
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
            "本次没有取得任何外部文献证据：所有依赖领域比较的判断不得视为 satisfied。",
        )

    # 1-4) 逐篇外部文献：存在性、元数据、时间、相关性
    for work in external_set:
        evidence_id = str(work.get("evidence_id") or "")
        title = str(work.get("title") or "")
        if not title or not str(work.get("paper_id") or ""):
            add("external_paper_missing_identity", "high", f"{evidence_id} 缺少标题或 paper_id，无法确认文献存在。", evidence_id=evidence_id)
        if work.get("metadata_check") == "mismatch":
            add(
                "external_metadata_mismatch",
                "high",
                f"{evidence_id}「{title[:50]}」的 DOI/标题/年份与 Crossref 不一致：{'；'.join(work.get('metadata_notes') or [])[:160]}",
                evidence_id=evidence_id,
            )
        elif work.get("metadata_check") != "ok":
            add(
                "external_metadata_unverified",
                "low",
                f"{evidence_id}「{title[:50]}」未完成 Crossref 元数据校验（{'; '.join(work.get('metadata_notes') or [])[:80] or '未校验'}）。",
                evidence_id=evidence_id,
            )
        if target_year_int and work.get("year") and int(work["year"]) > target_year_int:
            add(
                "external_time_leak",
                "high",
                f"{evidence_id}「{title[:50]}」发表于 {work['year']}，晚于目标论文 {target_year_int} 年，不能作为创新性先验。",
                evidence_id=evidence_id,
            )
        if float(work.get("relevance_score") or 0.0) <= 0.0 and not (work.get("claim_ids") or []):
            add(
                "external_evidence_irrelevant",
                "medium",
                f"{evidence_id}「{title[:50]}」没有对应 claim 且相关度为 0，与本次判断无关。",
                evidence_id=evidence_id,
            )

    # 5-6) 逐条 Novelty 判断：引用的证据 ID 是否存在、supporting/contradicting 是否站得住
    for row in novelty_judgements:
        claim_id = str(row.get("claim_id") or "")
        referenced = [str(item) for item in (row.get("external_evidence_ids") or [])]
        unknown = [item for item in referenced if item not in known_evidence_ids]
        if unknown:
            add(
                "external_evidence_id_not_found",
                "high",
                f"{claim_id} 引用了不存在的外部证据 ID：{', '.join(unknown)}。模型不得自行补充文献。",
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
                    f"{claim_id} 的 {stance_field} 点名了不在 External Evidence Set 里的文献：{', '.join(unbacked)[:120]}。",
                    claim_id=claim_id,
                )
        stance = str(row.get("judgment") or "")
        if stance in {"novel", "partially_novel"}:
            check_papers = cited or []
            ok, note = judgement_support_consistency(stance, check_papers, str(row.get("reason") or ""))
            if not ok:
                add("external_stance_inconsistent", "medium", f"{claim_id} 判 {stance} 但{note}。", claim_id=claim_id)
            coverage = row.get("coverage") or {}
            prior_count = int(coverage.get("prior_paper_count") or 0)
            if prior_count < int(min_external_papers):
                add(
                    "low_prior_literature_coverage",
                    "high",
                    f"{claim_id} 判 {stance}，但 prior literature 覆盖不足（{prior_count} < {int(min_external_papers)} 篇），"
                    "该项判断的置信度已被下调。",
                    claim_id=claim_id,
                    confidence_cap=float(low_coverage_confidence_cap),
                )
        claim = claim_by_id.get(claim_id)
        if claim and claim.get("source_evidence_id"):
            source_id = str(claim["source_evidence_id"])
            if not source_id.startswith(("EV-", "EXT-", "XEV-")) and source_id:
                add("claim_source_evidence_invalid", "medium", f"{claim_id} 的 source_evidence_id={source_id} 形态可疑。", claim_id=claim_id)

    # 7) 依赖领域比较的规则：没有外部证据不许 satisfied
    coverage_ok = bool(external_set)
    for indicator_id in indicator_ids or []:
        if not coverage_ok and rule_needs_domain_comparison(indicator_id):
            add(
                "external_comparison_required_but_missing",
                "high",
                f"{indicator_id} 必须靠领域比较判断，但本次没有外部文献证据：不得给 satisfied。",
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
    """把外部校验结论落到评分判断上（只碰 review_flags / needs_review / confidence）。"""
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
                # 依赖领域比较却完全没有外部证据：正向判断不可追溯 → 转无法判断（与证据门控同一硬门槛口径）。
                updated.append(
                    replace(
                        judgement,
                        judgement=Judgement.UNCERTAIN,
                        confidence=min(confidence, 0.25),
                        needs_review=True,
                        review_flags=[
                            *judgement.review_flags,
                            "外部证据校验：本条判断依赖领域比较，但没有任何外部文献证据，已转为无法判断。",
                        ],
                    )
                )
                continue
        for finding in by_indicator.get(judgement.indicator_id, []):
            flags.append(f"外部证据校验（{finding['code']}）：{finding['message']}")
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
    """Novelty Agent 判 novel 但 prior literature 覆盖不足：自动下调 confidence 并标 needs_review。"""
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

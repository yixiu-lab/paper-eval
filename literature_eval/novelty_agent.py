"""Novelty Verification Agent：拿外部文献证据给单条 claim 做**有限枚举**判断。

它不产出自由分数，只允许在五个档里选一个：

- `novel`                发表之前没有可比的已有研究
- `partially_novel`      有新成分，但已有研究已覆盖其中一部分
- `incremental`          主体已被做过，本文只做了增量
- `already_established`  已被既有研究确立
- `uncertain`            外部证据不足，判不了

核心任务不是"总结相关文献"，而是回答：**相对发表之前的已有研究，这条贡献究竟新增了什么。**

必须区分两件事：
1. 作者自己声称创新（来自本文内部证据 `internal_evidence`）；
2. 外部文献证据支持其创新（`supporting_papers` / `contradicting_papers`，来自 External Evidence Set）。
两者冲突时以外部证据为准，并把冲突写进 reason。

LLM 不可用（无 key / 离线 / 本地小模型调用失败）时不编造判断，产出 uncertain + needs_review，
让 Harness 按"外部证据不足"处理。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

from .claim_extraction import CLAIM_TYPE_LABELS
from .external_evidence import judgement_support_consistency
from .external_retrieval import ExternalRetrievalConfig
from .llm import AgentRuntimeConfig, call_plain_json
from .models import Evidence

# 创新性验证结论块用的 section 名。agents.py 的提示词按这个字符串识别 NOV-* 条目。
NOVELTY_EVIDENCE_SECTION = "创新性验证 Agent 结论"
NOVELTY_JUDGEMENTS = ("novel", "partially_novel", "incremental", "already_established", "uncertain")
# 依赖外部证据才有意义的档位：外部证据不足时不允许落到这两档。
NOVELTY_POSITIVE = {"novel", "partially_novel"}

NOVELTY_SYSTEM_PROMPT = (
    "你是外部创新性验证 Agent。你的任务不是总结相关文献，而是回答一个具体问题："
    "**相对于目标论文发表之前已经存在的相关研究，这条 claim 究竟新增了什么。**\n"
    "你会收到：target_claim（待验证主张，含 claim_type / concepts / theory / mechanism / context）、"
    "internal_evidence（本文内部证据，代表**作者自己的声称**）、external_evidence_set（真实检索到的、"
    "发表于目标论文之前的候选文献，各带 relevance_score 与 retrieval_query）、"
    "target_paper（目标论文元数据与发表年份）。\n"
    "判断规则：\n"
    "1) 只能依据给出的 external_evidence_set 判断外部文献情况。**不得**凭自己的知识声称某研究"
    "『首次』『已确立』或『已有大量研究』；证据里没有就是没有。\n"
    "2) 必须区分「作者声称创新」（internal_evidence 里的话）与「外部文献证据支持其创新」"
    "（external_evidence_set 里能对照上的内容）。两者不一致时以外部证据为准。\n"
    "3) external_evidence_set 为空或与 claim 明显无关时，judgment 只能是 uncertain。\n"
    "4) 引用文献时必须用证据里给出的 external_evidence_id（XEV-*），supporting_papers / "
    "contradicting_papers 里只能点名 external_evidence_set 中真实存在的标题或 DOI。\n"
    "5) judgment 只允许：novel | partially_novel | incremental | already_established | uncertain。\n"
    "6) 不允许输出任何自由分数；分数由 Harness 计算。\n"
    "输出 JSON：{\"judgements\":[{\"claim_id\":\"原样返回\",\"judgment\":\"枚举值\","
    "\"reason\":\"中文短评，说明相对既有研究新增了什么\",\"supporting_papers\":[\"标题或DOI\"],"
    "\"contradicting_papers\":[\"标题或DOI\"],\"external_evidence_ids\":[\"XEV-1\"],"
    "\"confidence\":0.0,\"needs_review\":true}]}。必须为每条输入 claim 返回一条判断。"
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
    """某条 claim 的外部证据覆盖情况。Harness 靠它决定要不要下调置信度。"""
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
    """把模型输出收敛成受控结构：枚举值、证据 ID、supporting/contradicting 都按仓库核对。"""
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
                    reason="模型未返回该 claim 的创新性判断，按外部证据不足处理。",
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
            notes.append(f"引用了不存在的外部证据 ID：{', '.join(unknown)}")

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
                "点名了 External Evidence Set 之外的文献："
                + "；".join([*unbacked_support, *unbacked_contra])[:120]
            )
        cited = [
            work for work in external_set
            if str(work.get("evidence_id") or "") in referenced
        ]
        if judgment in NOVELTY_POSITIVE and not cited and not supporting:
            notes.append("判novel但未给出任何支撑文献，置信度由 Harness 下调")
            needs_review = True
        if judgment in NOVELTY_POSITIVE and not coverage["paper_count"]:
            notes.append("判novel但该 claim 没有任何外部对照文献，置信度由 Harness 下调")
            needs_review = True
        if cited:
            ok, note = judgement_support_consistency(judgment, cited, str(row.get("reason") or ""))
            if not ok:
                notes.append(note)
                needs_review = True
        reason = str(row.get("reason") or "").strip()
        if notes:
            reason = f"{reason}（核对提示：{'；'.join(notes)}）"
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
    """外部验证不可用（无模型/离线）时的确定性降级：全部 uncertain + needs_review。"""
    return [
        NoveltyVerdict(
            claim_id=str(getattr(claim, "claim_id", "")),
            judgment="uncertain",
            reason="创新性验证 Agent 不可用（或未取得可用外部证据），本次不对该 claim 做创新性判断。",
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
    """创新性验证结论 → 可引用证据块（NOV-*），插给依赖领域比较的规则做 B4 贡献判断。

    这些块不是"证据"，是"已核对的判断"：写清档位、理由、引用了哪些外部证据 ID，
    既让评分 Agent 不必自己重做比对，也让 Harness 能顺着 NOV-* → XEV-* → 外部论文一路追溯。
    """
    blocks: list[Evidence] = []
    for index, row in enumerate(verdicts, start=1):
        lines = [
            f"【创新性验证结论 NOV-{index}｜claim {row.get('claim_id')}｜{row.get('claim_type_label') or row.get('claim_type')}】",
            f"判断：{row.get('judgment')}（置信度 {row.get('confidence')}；需复核 {row.get('needs_review')}）",
            f"作者的主张（本文内部证据 {row.get('internal_evidence_id')}）：{row.get('author_claim')}",
            f"外部文献对照结论：{row.get('reason')}",
            f"引用的外部证据：{', '.join(str(item) for item in (row.get('external_evidence_ids') or [])) or '无'}",
        ]
        if row.get("supporting_papers"):
            lines.append(f"支持该判断的文献：{'；'.join(str(item) for item in row['supporting_papers'])}")
        if row.get("contradicting_papers"):
            lines.append(f"构成对照/反例的文献：{'；'.join(str(item) for item in row['contradicting_papers'])}")
        coverage = row.get("coverage") or {}
        lines.append(
            f"外部证据覆盖：{coverage.get('paper_count', 0)} 篇（其中带年份 {coverage.get('prior_paper_count', 0)} 篇）"
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
    """跑创新性验证。返回 {"verdicts": [...], "unavailable": bool, "reason": str}。

    只做一次批量调用（所有 claim 一起），失败即整体降级为 uncertain——不逐条重试，
    因为"外部证据不足"本身就是允许的结论，重试只会烧调用。
    """
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
        progress_callback(f"创新性验证 Agent 正在比对 {len(claims)} 条 claim 与 {len(external_set)} 篇外部文献")
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

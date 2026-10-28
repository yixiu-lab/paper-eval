"""Claim Extractor：从目标论文里抽出**需要外部文献对照**的关键 claim。

为什么不能直接拿整篇论文去检索：整篇论文的检索式会被主题词淹没，检索回的是"领域相关"
而不是"能证伪这条创新主张的相关"。所以先把论文拆成一条条可对照的 claim，再逐条检索。

五类 claim（只抽这几类，方法/结果类判断仍主要靠内部证据）：

- research_gap              作者声称已有研究缺什么
- novelty_claim             作者声称首次提出 / 首次发现 / 首次验证什么
- theoretical_contribution  作者声称推进了什么理论
- mechanism_claim           作者提出了什么新的作用机制
- boundary_claim            作者扩展了什么理论边界或研究情境

每条 claim 必须带 `source_evidence_id`——指向本文 Evidence Repository 里的 EV-*，
不能是模型凭空生成的句子。抽不到正文依据的 claim 一律不产生（宁可少抽，不可编造）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .external_retrieval import (
    CLAIM_FIELDS,
    CONTEXT_PLACEHOLDER,
    RetrievalClaim,
    clean_paper_title,
)
from .models import Evidence

RESEARCH_GAP = "research_gap"
NOVELTY_CLAIM = "novelty_claim"
THEORETICAL_CONTRIBUTION = "theoretical_contribution"
MECHANISM_CLAIM = "mechanism_claim"
BOUNDARY_CLAIM = "boundary_claim"
CLAIM_TYPES = (RESEARCH_GAP, NOVELTY_CLAIM, THEORETICAL_CONTRIBUTION, MECHANISM_CLAIM, BOUNDARY_CLAIM)

CLAIM_TYPE_LABELS = {
    RESEARCH_GAP: "研究空白",
    NOVELTY_CLAIM: "创新性主张",
    THEORETICAL_CONTRIBUTION: "理论贡献",
    MECHANISM_CLAIM: "机制主张",
    BOUNDARY_CLAIM: "边界/情境主张",
}

# paper_profile 字段 → claim 类型。一个字段对应一类 claim，抽取顺序固定，保证可复现。
PROFILE_CLAIM_FIELDS = (
    ("research_gap", RESEARCH_GAP),
    ("core_claim", NOVELTY_CLAIM),
    ("main_contribution", THEORETICAL_CONTRIBUTION),
    ("central_problem", RESEARCH_GAP),
    ("research_question", BOUNDARY_CLAIM),
)

# 从正文里找 claim 依据时的定位词。中文是规则库/论文常用的表述，英文是语料里的常见写法。
CLAIM_MARKERS = {
    RESEARCH_GAP: (
        "研究缺口", "研究空白", "尚未", "少有研究", "缺乏研究", "未被检验", "文献不足", "少有文献",
        "research gap", "has not been", "have not been", "little is known", "remains unclear",
        "underexplored", "unaddressed", "few studies", "remains unknown", "scant",
    ),
    NOVELTY_CLAIM: (
        "首次", "首次提出", "首次检验", "首次验证", "创新性", "新概念", "新机制", "新框架", "新命题",
        "首次实证", "尚未有人", "尚无研究",
        "first to", "we are the first", "first study", "novel ", "novelty", "newly proposed",
        "introduce a new", "we develop a new", "heretofore", "originality", "new contribution",
    ),
    THEORETICAL_CONTRIBUTION: (
        "理论贡献", "贡献在于", "推进了", "拓展了理论", "丰富了理论", "理论意义", "整合理论",
        "theoretical contribution", "we contribute", "contributes to theory", "advance theory",
        "extend theory", "theoretical implications", "enrich",
    ),
    MECHANISM_CLAIM: (
        "作用机制", "中介机制", "机制在于", "通过", "传导", "路径", "调节作用", "边界机制",
        "mechanism", "mediat", "underlying process", "pathway", "transmit", "explain why",
        "process through which", "moderat", "conditional effect", "channel",
    ),
    BOUNDARY_CLAIM: (
        "边界条件", "情境", "外部效度", "适用条件", "推广", "不同情境", "跨文化", "样本情境",
        "boundary condition", "context", "contextual", "generalizab", "external validity",
        "under what conditions", "moderating role of", "across settings",
    ),
}

# claim 文本 → 检索词时，把中文里的英文术语捞出来（理论名、机制名、情境名都是英文最有用）。
# 只收 3 个字母以上的词：claim 依据句常带 "are the first test this mechanism" 这类虚词，
# 收进来会把检索式和重排词表灌满噪声（2 字母的 "we/is/are" 尤其致命）。
ASCII_TERM_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")

# 检索词表里必须剔除的通用学术词与虚词。
# 实测：claim 依据句是英文时（"We develop a new mechanism in which perceived procedural justice
# mediates ..."），整句进检索词会把 "underlying/process/develop/new/which" 这类词带上，
# 而每个候选摘要都有这些词——重排被拉平到"谁词多谁靠前"，真正同主题的先验反而排第二。
GENERIC_KEYWORD_TOKENS = {
    "underlying", "process", "processes", "explanatory", "pathway", "develop", "developed",
    "develops", "new", "which", "perceived", "mediates", "mediating", "mediation", "review",
    "study", "studies", "research", "paper", "article", "findings", "results", "analysis",
    "effect", "effects", "evidence", "theory", "theoretical", "contribution", "contributions",
    "framework", "novel", "novelty", "first", "approach", "based", "using", "used", "present",
    "presented", "examine", "examines", "examined", "investigate", "investigates", "show",
    "shows", "shown", "provide", "provides", "propose", "proposes", "proposed", "test",
    "tests", "tested", "unresolved", "prior", "previous", "existing", "question", "questions",
    "gap", "gaps", "limitation", "limitations", "open", "settings", "context", "contextual",
    "moderators", "generalizability", "mechanism", "mechanisms", "conditions", "boundary",
    "contexts", "settings", "whether", "this", "that", "these", "those", "with", "from",
    "about", "into", "over", "more", "most", "such", "also", "than", "they", "them", "their",
    "have", "has", "are", "was", "were", "not", "but", "can", "may", "all", "any", "some",
}
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;.!?])\s*|\n+")

# 抽 claim 时不要用参考文献区，那里全是别人的标题。
SKIP_SECTIONS = ("references", "bibliography", "参考文献")


@dataclass
class ExtractedClaim:
    """一条待外部验证的 claim。字段对齐需求里的 claim 结构。"""

    claim_id: str
    claim_type: str
    claim_text: str
    source_evidence_id: str
    concepts: list[str] = field(default_factory=list)
    theory: str = ""
    mechanism: str = ""
    context: str = ""
    publication_year: int | None = None
    # 抽取来源，仅用于排查：paper_context | evidence_markers
    extraction_source: str = "paper_context"
    keywords_en: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "claim_type_label": CLAIM_TYPE_LABELS.get(self.claim_type, self.claim_type),
            "claim_text": self.claim_text,
            "source_evidence_id": self.source_evidence_id,
            "concepts": list(self.concepts),
            "theory": self.theory,
            "mechanism": self.mechanism,
            "context": self.context,
            "publication_year": self.publication_year,
            "extraction_source": self.extraction_source,
            "keywords_en": self.keywords_en,
            "confidence": round(self.confidence, 3),
        }

    def to_retrieval_claim(self, paper_title: str) -> RetrievalClaim:
        """转成检索层认的 claim（检索层只关心 claim_id/文本/英文检索词）。"""
        return RetrievalClaim(
            claim_id=self.claim_id,
            label=CLAIM_TYPE_LABELS.get(self.claim_type, self.claim_type),
            text=self.claim_text,
            keywords_en=self.keywords_en or default_keywords(self.claim_type, self.theory, self.mechanism),
            source=f"claim_extractor:{self.extraction_source}",
            claim_type=self.claim_type,
            source_evidence_id=self.source_evidence_id,
            concepts=list(self.concepts),
            theory=self.theory,
            mechanism=self.mechanism,
            context=self.context,
            publication_year=self.publication_year,
        )


def _is_reference(item: Evidence) -> bool:
    section = str(item.section or "").lower()
    return any(marker in section for marker in SKIP_SECTIONS)


def _usable(evidences: list[Evidence]) -> list[Evidence]:
    return [
        item
        for item in evidences
        if not _is_reference(item) and len(str(item.text or "").strip()) >= 40
    ]


def _sentences(text: str) -> list[str]:
    return [piece.strip() for piece in SENTENCE_SPLIT_RE.split(str(text or "")) if len(piece.strip()) >= 24]


def _ascii_terms(text: str, limit: int = 8, *, drop_generic: bool = True) -> list[str]:
    """捞英文术语。`drop_generic=True` 时剔除通用学术词（做检索词/构念时都要剔）。"""
    seen: set[str] = set()
    terms: list[str] = []
    for match in ASCII_TERM_RE.findall(str(text or "")):
        lowered = match.lower()
        if lowered in seen:
            continue
        if drop_generic and lowered in GENERIC_KEYWORD_TOKENS:
            continue
        seen.add(lowered)
        terms.append(match)
        if len(terms) >= limit:
            break
    return terms


def claim_marker_hits(claim_type: str, text: str) -> int:
    lowered = str(text or "").lower()
    return sum(1 for marker in CLAIM_MARKERS.get(claim_type, ()) if marker in lowered)


def evidence_support(evidences: list[Evidence], claim_type: str, limit: int = 3) -> list[tuple[Evidence, str]]:
    """在正文证据里找最能支撑某类 claim 的 (证据, 依据句)，按「定位词命中 → 句子长度」排序。

    依据句只从证据原文切出来，不做任何改写：claim 必须是论文自己写的句子，
    否则「外部文献对照」就变成对照一个模型编出来的主张。
    """
    scored: list[tuple[int, int, Evidence, str]] = []
    for item in _usable(evidences):
        best_sentence = ""
        best_hits = 0
        for sentence in _sentences(item.text):
            hits = claim_marker_hits(claim_type, sentence)
            if hits > best_hits or (hits == best_hits and hits > 0 and len(sentence) > len(best_sentence)):
                best_hits, best_sentence = hits, sentence
        if best_hits <= 0:
            continue
        scored.append((best_hits, len(best_sentence), item, best_sentence))
    scored.sort(key=lambda row: (-row[0], -row[1]))
    return [(item, sentence) for _, _, item, sentence in scored[:limit]]


def _profile_text(profile: dict[str, Any], key: str) -> str:
    value = str(profile.get(key) or "").strip()
    if not value or value == CONTEXT_PLACEHOLDER or value.startswith("待 Agent"):
        return ""
    return value


def _theory_from_profile(profile: dict[str, Any]) -> str:
    """理论标签：优先画像里的理论动作类型，其次其英文描述（检索词要用得上）。"""
    move = profile.get("theoretical_move")
    if not isinstance(move, dict):
        return ""
    parts: list[str] = []
    move_type = str(move.get("type") or "").strip()
    if move_type and move_type != "unclear":
        parts.append(move_type.replace("_", " "))
    description = str(move.get("description") or "").strip()
    if description and not description.startswith("待 Agent"):
        parts.extend(_ascii_terms(description, limit=4))
    return " ".join(parts).strip()


def _concepts_for(claim_type: str, profile: dict[str, Any], evidence: Evidence | None, sentence: str) -> list[str]:
    """核心构念：优先取证据/依据句里的英文术语，其次取 profile 里的理论动作关键词。"""
    terms = _ascii_terms(f"{sentence} {evidence.text if evidence else ''}", limit=6)
    if not terms:
        terms = _ascii_terms(str(profile.get("title") or ""), limit=4)
    return terms


def _keywords_for(claim_type: str, profile: dict[str, Any], evidence: Evidence | None, sentence: str) -> str:
    """英文检索词：claim 类型词 + 依据句里的英文术语 + 论文标题术语。

    中文内容这里不做翻译——检索走的是英文语料，翻不动就退回该类型的中性英文检索词，
    并在 confidence 上留痕；宁可检索式泛一点，也不要编一个不存在的英文术语。
    """
    parts = [default_keywords(claim_type)]
    parts.extend(_ascii_terms(sentence, limit=6))
    if evidence is not None:
        parts.extend(_ascii_terms(evidence.text, limit=4))
    parts.extend(_ascii_terms(str(profile.get("title") or ""), limit=4))
    seen: set[str] = set()
    ordered = [piece for piece in " ".join(parts).split() if piece and not (piece in seen or seen.add(piece))]
    return " ".join(ordered)[:220]


def default_keywords(claim_type: str, theory: str = "", mechanism: str = "") -> str:
    """claim 类型 → 中性英文检索词（兜底，保证每条 claim 都有可检索的词）。"""
    base = DEFAULT_CLAIM_KEYWORDS.get(claim_type, "prior literature theoretical advance")
    extra = " ".join(piece for piece in (theory.replace("_", " "), mechanism) if piece)
    return f"{base} {extra}".strip()


DEFAULT_CLAIM_KEYWORDS = {
    RESEARCH_GAP: "unresolved research gap prior studies limitations open question",
    NOVELTY_CLAIM: "novel contribution new construct first study originality",
    THEORETICAL_CONTRIBUTION: "theoretical contribution advance theory extend framework",
    MECHANISM_CLAIM: "underlying mechanism mediating process explanatory pathway",
    BOUNDARY_CLAIM: "boundary conditions contextual moderators generalizability settings",
}


def extract_claims(
    paper_context: dict[str, Any] | None,
    evidences: list[Evidence],
    publication_year: int | None = None,
) -> list[ExtractedClaim]:
    """抽 claim。优先用论文画像的显式字段，字段缺失时用正文定位词兜底。

    `source_evidence_id` 一律指向真实存在的 EV-*：画像字段本身没有证据 ID，
    所以必须回正文里找到写这句话的那一段，找不到就不产出这条 claim。
    """
    profile = (paper_context or {}).get("paper_profile") or {}
    if not isinstance(profile, dict):
        profile = {}
    theory = _theory_from_profile(profile)
    contexts = _ascii_terms(str(profile.get("boundary_conditions") or ""), limit=4)

    claims: list[ExtractedClaim] = []
    seen_types: dict[str, str] = {}   # claim_type -> 已用过的依据句，避免同一句重复成多条 claim

    for field_name, claim_type in PROFILE_CLAIM_FIELDS:
        text = _profile_text(profile, field_name)
        if not text:
            continue
        supports = evidence_support(evidences, claim_type, limit=1)
        if not supports:
            continue
        evidence, sentence = supports[0]
        if seen_types.get(claim_type) == sentence:
            continue
        seen_types[claim_type] = sentence
        claims.append(
            ExtractedClaim(
                claim_id=f"CLM-{len(claims) + 1:02d}",
                claim_type=claim_type,
                claim_text=text[:400],
                source_evidence_id=evidence.evidence_id,
                concepts=_concepts_for(claim_type, profile, evidence, sentence),
                theory=theory,
                mechanism=(_ascii_terms(sentence, limit=4) or [""])[0] if claim_type == MECHANISM_CLAIM else "",
                context="; ".join(contexts),
                publication_year=publication_year,
                extraction_source="paper_context",
                keywords_en=_keywords_for(claim_type, profile, evidence, sentence),
                confidence=0.8,
            )
        )

    # 画像抽完后再补一次正文兜底：画像字段里没有对应表述的 claim 类型（例如没写
    # mechanism 字段的论文），只要正文有明确机制/边界定位词就补一条。
    # 每类最多补一条，避免把检索预算耗在同一类 claim 的重复表述上。
    if claims:
        used_types = {claim.claim_type for claim in claims}
        for claim_type in CLAIM_TYPES:
            if claim_type in used_types:
                continue
            supports = evidence_support(evidences, claim_type, limit=1)
            if not supports:
                continue
            evidence, sentence = supports[0]
            claims.append(
                ExtractedClaim(
                    claim_id=f"CLM-{len(claims) + 1:02d}",
                    claim_type=claim_type,
                    claim_text=sentence[:400],
                    source_evidence_id=evidence.evidence_id,
                    concepts=_ascii_terms(sentence, limit=6),
                    theory=theory,
                    context="; ".join(contexts),
                    publication_year=publication_year,
                    extraction_source="evidence_markers",
                    keywords_en=_keywords_for(claim_type, profile, evidence, sentence),
                    confidence=0.5,
                )
            )

    # 画像没抽出任何 claim（画像生成失败或字段全是占位符）时的兜底：只按正文定位词抽。
    if not claims:
        for claim_type in CLAIM_TYPES:
            supports = evidence_support(evidences, claim_type, limit=1)
            if not supports:
                continue
            evidence, sentence = supports[0]
            claims.append(
                ExtractedClaim(
                    claim_id=f"CLM-{len(claims) + 1:02d}",
                    claim_type=claim_type,
                    claim_text=sentence[:400],
                    source_evidence_id=evidence.evidence_id,
                    concepts=_ascii_terms(sentence, limit=6),
                    theory=theory,
                    context="; ".join(contexts),
                    publication_year=publication_year,
                    extraction_source="evidence_markers",
                    keywords_en=_keywords_for(claim_type, profile, evidence, sentence),
                    confidence=0.5,
                )
            )
    return claims


def publication_year_of(metadata: dict[str, Any] | None, publish_date: str | None = None) -> int | None:
    """目标论文发表年份。用于时间泄漏控制：prior literature 必须早于它。"""
    text = str((metadata or {}).get("year") or "") or str(publish_date or "")
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group(0)) if match else None

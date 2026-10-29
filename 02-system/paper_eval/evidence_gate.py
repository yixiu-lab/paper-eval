"""证据门控。

契约（2026-09 修订）：门控是**检查器**，不是第二个评分员。

- 只允许写 `review_flags` 和 `needs_review`，不得改写模型给出的
  `judgement` / `reason` / `confidence`。
- 唯一例外：正向判断（satisfied / partial）却找不到任何有效证据 ID 时，
  判断转为 uncertain —— 这是可追溯性硬门槛，不是质量判断。
- 证据强弱只是提示，是否降档由人工复核决定，不再由本地关键词命中数决定。
"""

from __future__ import annotations

from dataclasses import replace

from .models import AgentJudgement, Evidence, Judgement


SUBSTANTIVE_MARKERS = {
    "method": [
        "method",
        "methods",
        "sample",
        "participants",
        "measure",
        "measurement",
        "survey",
        "experiment",
        "regression",
        "model",
        "analysis",
        "procedure",
        "data",
        "方法",
        "样本",
        "测量",
        "实验",
        "回归",
        "模型",
        "数据",
    ],
    "result": [
        "result",
        "results",
        "finding",
        "findings",
        "coefficient",
        "significant",
        "table",
        "figure",
        "robust",
        "hypothesis",
        "结果",
        "发现",
        "显著",
        "表",
        "图",
        "稳健",
        "假设",
    ],
    "theory": [
        "theory",
        "theoretical",
        "framework",
        "mechanism",
        "construct",
        "hypothesis",
        "literature",
        "contribution",
        "理论",
        "框架",
        "机制",
        "构念",
        "假设",
        "文献",
        "贡献",
    ],
}


def _is_abstract_only(items: list[Evidence]) -> bool:
    if not items:
        return False
    return all("abstract" in item.section.lower() or item.page == 1 for item in items)


def _substantive_pages(items: list[Evidence]) -> set[int]:
    return {
        item.page
        for item in items
        if "abstract" not in item.section.lower() and len(item.text.strip()) >= 80
    }


def _has_reason_score_conflict(reason: str) -> bool:
    lowered = reason.lower()
    conflict_markers = [
        "未明确",
        "不明确",
        "未充分",
        "不充分",
        "不足",
        "缺少",
        "缺乏",
        "未见",
        "没有",
        "仅",
        "只是",
        "暗示",
        "部分",
        "无法判断",
        "cannot determine",
        "insufficient",
        "not explicit",
        "not clearly",
        "only suggests",
        "partially",
    ]
    return any(marker in lowered for marker in conflict_markers)


def _marker_hits(items: list[Evidence], marker_group: str) -> int:
    markers = SUBSTANTIVE_MARKERS[marker_group]
    return sum(
        1
        for item in items
        if any(marker in f"{item.section} {item.text}".lower() for marker in markers)
    )


def evidence_strength(items: list[Evidence], indicator_id: str) -> tuple[str, str]:
    if not items:
        return "none", "没有有效证据"
    if _is_abstract_only(items):
        return "mention_only", "证据主要来自摘要或首页声明"

    pages = _substantive_pages(items)
    sections = {item.section.strip().lower() for item in items if item.section.strip()}
    long_blocks = [item for item in items if len(item.text.strip()) >= 120 and "abstract" not in item.section.lower()]
    method_hits = _marker_hits(items, "method")
    result_hits = _marker_hits(items, "result")
    theory_hits = _marker_hits(items, "theory")

    if indicator_id.startswith(("D-", "E-")):
        if len(pages) >= 2 and len(long_blocks) >= 2 and (method_hits >= 2 or result_hits >= 2 or method_hits + result_hits >= 2):
            return "strong_support", "方法/结果证据跨页且包含实质方法或结果信息"
        if len(pages) >= 1 and (method_hits >= 1 or result_hits >= 1):
            return "partial_support", "有方法或结果证据，但跨页或细节强度不足"
        return "mention_only", "只看到方法或结果相关提及，缺少实质细节"

    if indicator_id.startswith(("B-", "C-")):
        if len(pages) >= 2 and len(long_blocks) >= 2 and theory_hits >= 2:
            return "strong_support", "理论证据跨页且包含构念、机制、框架或文献贡献信息"
        if len(pages) >= 1 and theory_hits >= 1:
            return "partial_support", "有理论相关证据，但机制或贡献展开不足"
        return "mention_only", "只看到理论相关提及，缺少机制或贡献展开"

    if len(pages) >= 2 and len(long_blocks) >= 2 and len(sections) >= 1:
        return "strong_support", "证据来自多处正文且文本较完整"
    if len(pages) >= 1 and long_blocks:
        return "partial_support", "有正文证据，但数量或覆盖范围有限"
    return "mention_only", "证据较短或停留在一般性提及"


def apply_evidence_gate(judgements: list[AgentJudgement], evidences: list[Evidence]) -> list[AgentJudgement]:
    evidence_map = {item.evidence_id: item for item in evidences}
    positive = {Judgement.SATISFIED, Judgement.MOSTLY, Judgement.PARTIAL}
    gated: list[AgentJudgement] = []
    for judgement in judgements:
        valid_ids = [
            evidence_id
            for evidence_id in judgement.evidence_ids
            if evidence_id in evidence_map and len(evidence_map[evidence_id].text.strip()) >= 40
        ]

        # 硬门槛：正向判断但没有任何有效证据，无法追溯，转为无法判断。
        if judgement.judgement in positive and not valid_ids:
            gated.append(
                replace(
                    judgement,
                    judgement=Judgement.UNCERTAIN,
                    evidence_ids=[],
                    confidence=min(judgement.confidence, 0.25),
                    needs_review=True,
                    review_flags=[
                        *judgement.review_flags,
                        "证据门控：未找到有效证据 ID 或证据文本，因无法追溯转为无法判断。",
                    ],
                )
            )
            continue

        valid_evidences = [evidence_map[evidence_id] for evidence_id in valid_ids]
        strength, strength_reason = evidence_strength(valid_evidences, judgement.indicator_id)
        flags: list[str] = []
        if judgement.judgement == Judgement.SATISFIED:
            if _has_reason_score_conflict(judgement.reason):
                flags.append(
                    "证据门控提示：理由中出现证据不足或仅部分支持的表述，与 satisfied 不一致，需人工复核。"
                )
            if strength != "strong_support":
                flags.append(
                    f"证据强度提示：satisfied 的证据强度为 {strength}（{strength_reason}），需人工复核。"
                )
            if len(valid_ids) < 2:
                flags.append(
                    f"高分证据提示：satisfied 只采纳 {len(valid_ids)} 条有效证据，需人工复核。"
                )
        elif judgement.judgement == Judgement.PARTIAL and strength == "mention_only":
            flags.append(
                f"证据强度提示：partial 的证据强度为 {strength}（{strength_reason}），需人工复核。"
            )

        gated.append(
            replace(
                judgement,
                evidence_ids=valid_ids,
                needs_review=judgement.needs_review or bool(flags),
                review_flags=[*judgement.review_flags, *flags],
            )
        )
    return gated

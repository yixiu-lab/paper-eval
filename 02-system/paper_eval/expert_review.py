from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _review_path(root: str | Path, run_id: str) -> Path:
    safe_run_id = "".join(ch for ch in run_id if ch.isalnum() or ch in {"-", "_"})
    return Path(root) / "expert_reviews" / f"{safe_run_id}_expert_reviews.jsonl"


def _find_ai_detail(report: dict[str, Any], indicator_id: str) -> dict[str, Any] | None:
    for item in report.get("details", []):
        if isinstance(item, dict) and item.get("rule_id") == indicator_id:
            return item
    return None


def _difference_type(ai_detail: dict[str, Any] | None, expert_judgement: str, expert_score: float | None) -> str:
    if not ai_detail:
        return "missing_ai_rule"
    ai_judgement = str(ai_detail.get("judgement") or "")
    if ai_judgement != expert_judgement:
        return "judgement_changed"
    ai_score = ai_detail.get("weighted_score")
    if expert_score is not None and ai_score is not None and abs(float(ai_score) - expert_score) >= 1:
        return "score_changed"
    return "aligned"


def save_expert_review(
    root: str | Path,
    report: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(review["run_id"])
    indicator_id = str(review["indicator_id"])
    ai_detail = _find_ai_detail(report, indicator_id)
    expert_score = review.get("expert_score")
    if expert_score in {"", None}:
        expert_score = None
    else:
        expert_score = float(expert_score)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "indicator_id": indicator_id,
        "expert_name": review.get("expert_name") or "anonymous",
        "expert_judgement": review["expert_judgement"],
        "expert_score": expert_score,
        "comment": review.get("comment") or "",
        "ai_judgement": ai_detail.get("judgement") if ai_detail else None,
        "ai_weighted_score": ai_detail.get("weighted_score") if ai_detail else None,
        "ai_reason": ai_detail.get("reason") if ai_detail else None,
        "difference_type": _difference_type(ai_detail, str(review["expert_judgement"]), expert_score),
    }
    path = _review_path(root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def list_expert_reviews(root: str | Path, run_id: str) -> list[dict[str, Any]]:
    path = _review_path(root, run_id)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def summarize_expert_reviews(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    differences = [item for item in records if item.get("difference_type") != "aligned"]
    by_type: dict[str, int] = {}
    for item in records:
        key = str(item.get("difference_type") or "unknown")
        by_type[key] = by_type.get(key, 0) + 1
    return {
        "total": total,
        "difference_count": len(differences),
        "alignment_rate": None if total == 0 else round((total - len(differences)) / total, 4),
        "by_type": by_type,
    }

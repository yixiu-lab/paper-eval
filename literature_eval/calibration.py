from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DIMENSION_IDS = ["A", "B", "C", "D", "E", "F"]


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_expert_benchmark(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def compare_report_to_expert(report: dict[str, Any], expert_row: dict[str, str]) -> dict[str, Any]:
    summary = report["summary"]
    expert_total = _float_or_none(expert_row.get("expert_total_score"))
    system_total = summary.get("total_score")
    dimension_errors = {}
    for dim in DIMENSION_IDS:
        expert_value = _float_or_none(expert_row.get(f"expert_{dim}"))
        system_value = summary["dimension_scores"].get(dim, {}).get("normalized_score")
        if expert_value is not None and system_value is not None:
            dimension_errors[dim] = {
                "expert": expert_value,
                "system": system_value,
                "absolute_error": abs(system_value - expert_value),
            }
    total_error = None if expert_total is None or system_total is None else abs(system_total - expert_total)
    return {
        "sample_id": expert_row["sample_id"],
        "system_total_score": system_total,
        "expert_total_score": expert_total,
        "total_absolute_error": total_error,
        "dimension_errors": dimension_errors,
    }


def summarize_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_errors = [row["total_absolute_error"] for row in rows if row["total_absolute_error"] is not None]
    dimension_mae = {}
    for dim in DIMENSION_IDS:
        errors = [
            row["dimension_errors"][dim]["absolute_error"]
            for row in rows
            if dim in row["dimension_errors"]
        ]
        dimension_mae[dim] = None if not errors else mean(errors)
    return {
        "sample_count": len(rows),
        "total_mae": None if not total_errors else mean(total_errors),
        "dimension_mae": dimension_mae,
        "rows": rows,
    }


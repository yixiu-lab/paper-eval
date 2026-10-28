from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateThresholds:
    max_total_mae: float = 12.0
    max_dimension_mae: float = 4.0


def evaluate_version_gate(calibration_summary: dict[str, Any], thresholds: GateThresholds | None = None) -> dict[str, Any]:
    thresholds = thresholds or GateThresholds()
    failures = []
    total_mae = calibration_summary.get("total_mae")
    if total_mae is None:
        failures.append("缺少 total_mae，不能升级。")
    elif total_mae > thresholds.max_total_mae:
        failures.append(f"total_mae={total_mae:.2f} 超过阈值 {thresholds.max_total_mae:.2f}")

    for dim, value in calibration_summary.get("dimension_mae", {}).items():
        if value is not None and value > thresholds.max_dimension_mae:
            failures.append(f"dimension {dim} mae={value:.2f} 超过阈值 {thresholds.max_dimension_mae:.2f}")

    return {
        "approved": not failures,
        "failures": failures,
        "thresholds": {
            "max_total_mae": thresholds.max_total_mae,
            "max_dimension_mae": thresholds.max_dimension_mae,
        },
    }

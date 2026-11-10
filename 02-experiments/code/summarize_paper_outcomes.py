"""Summarize the released paper-level scoring table by evaluation group."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    scored = [row for row in rows if row.get("record_status") == "scored"]

    def score_summary(values: list[float]) -> dict[str, float | int]:
        return {
            "n": len(values),
            "mean": round(statistics.mean(values), 4),
            "sample_sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "median": round(statistics.median(values), 4),
            "minimum": round(min(values), 4),
            "maximum": round(max(values), 4),
        }

    overall = [float(row["score_total"]) for row in scored]
    return {
        "papers_evaluated": len(rows),
        "completed_scores": len(scored),
        "blocked_before_scoring": sum(row.get("record_status") != "scored" for row in rows),
        "overall": score_summary(overall),
        "dimension_scores": {
            dimension: score_summary(
                [float(row[f"dimension_{dimension}"]) for row in scored if row.get(f"dimension_{dimension}")]
            )
            for dimension in "ABCDEF"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "03-data" / "paper-outcomes.csv",
    )
    args = parser.parse_args()
    print(json.dumps(summarize(read_csv(args.input)), indent=2))


if __name__ == "__main__":
    main()

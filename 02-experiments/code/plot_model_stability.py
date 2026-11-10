"""Create a figure of observed repeat-run variation and the capability hypothesis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


RUN_DIFFERENCES = {
    "Qwen3.8-Flash": [1.88, 3.53, 6.09],
    "Qwen3.5-35B-A3B": [1.30, 3.22, 2.53],
    "Qwen3.7-Flash": [0.80, 0.56, 1.97],
    "DeepSeek-Flash": [1.33, 0.50, 0.00],
    "Kimi-K2.5": [1.50, 5.81, 4.17],
}


def create_figure(output: Path) -> None:
    summaries = sorted(
        ((name, values, sum(values) / len(values)) for name, values in RUN_DIFFERENCES.items()),
        key=lambda item: item[2],
        reverse=True,
    )
    fig, (observed, hypothesis) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.2),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    colors = ["#52718A", "#64839B", "#7898AE", "#8EABBD", "#A7BECC"]

    for row, (name, values, mean) in enumerate(summaries):
        color = colors[row]
        observed.barh(row, mean, height=0.48, color=color, alpha=0.72, zorder=1)
        offsets = [-0.12, 0.0, 0.12]
        observed.scatter(values, [row + offset for offset in offsets], s=36, color="#D17A45", edgecolor="white", linewidth=0.6, zorder=3)
        observed.text(mean + 0.09, row, f"{mean:.2f}", va="center", ha="left", fontsize=9)

    observed.set_yticks(range(len(summaries)), [item[0] for item in summaries])
    observed.invert_yaxis()
    observed.set_xlim(0, 6.85)
    observed.set_xlabel("Absolute difference between two runs (points)")
    observed.set_title("Observed repeat-run variation", loc="left", fontweight="bold", pad=12)
    observed.grid(axis="x", color="#DDE3E8", linewidth=0.8)
    observed.set_axisbelow(True)
    observed.spines[["top", "right", "left"]].set_visible(False)
    observed.tick_params(axis="y", length=0, pad=8)
    observed.text(
        0.0,
        -0.18,
        "Bars: model mean absolute difference; dots: the three paper-level differences. Lower values indicate greater repeatability.",
        transform=observed.transAxes,
        fontsize=8,
        color="#4A5560",
        va="top",
        wrap=True,
    )

    hypothesis.set_xlim(0, 1)
    hypothesis.set_ylim(0, 1)
    hypothesis.axis("off")
    hypothesis.set_title("Capability–stability hypothesis", loc="left", fontweight="bold", pad=12)
    hypothesis.text(0.02, 0.86, "Proposed relationship for the next study", fontsize=9, color="#4A5560")

    boxes = [
        (0.07, 0.63, 0.86, 0.16, "Higher model capability", "measured independently"),
        (0.07, 0.38, 0.86, 0.16, "More consistent assessment", "rule interpretation + evidence use"),
        (0.07, 0.13, 0.86, 0.16, "Lower run-to-run variation", "greater score stability"),
    ]
    for x, y, width, height, heading, detail in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#7898AE",
            facecolor="#F2F6F8",
        )
        hypothesis.add_patch(patch)
        hypothesis.text(x + width / 2, y + height * 0.62, heading, ha="center", va="center", fontsize=10, fontweight="bold", color="#233746")
        hypothesis.text(x + width / 2, y + height * 0.27, detail, ha="center", va="center", fontsize=8.5, color="#4A5560")

    for y_start, y_end in [(0.61, 0.56), (0.36, 0.31)]:
        hypothesis.add_patch(
            FancyArrowPatch(
                (0.5, y_start),
                (0.5, y_end),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.3,
                color="#D17A45",
            )
        )

    fig.suptitle("Model capability and assessment stability", x=0.05, y=1.02, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.19, right=0.98, top=0.86, bottom=0.24, wspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    default_output = Path(__file__).resolve().parents[2] / "04-ethics-access" / ".assets" / "experiments" / "model-capability-stability-hypothesis.png"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    create_figure(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

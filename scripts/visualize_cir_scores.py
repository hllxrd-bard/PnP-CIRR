#!/usr/bin/env python3
"""Visualize CIR score distributions from a JSON output file.

Example:
    python scripts/visualize_cir_scores.py \
        --input output.smoke.json \
        --output-dir outputs/score_analysis \
        --top-k 100 \
        --show

The script creates:
    01_score_distributions.png
    02_scores_by_rank.png
    03_score_correlation.png
    04_target_vs_reference_keep.png
    05_matched_query_distribution.png
    06_score_boxplots_by_query.png
    score_summary.csv
    flattened_results.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_SCORE_COLUMNS = [
    "score",
    "composed",
    "target",
    "reference_keep",
    "direction",
    "metadata",
    "edit_score",
    "edit_gate_penalty",
    "negative_penalty",
]

RAW_SCORE_COLUMNS = [
    "raw_composed",
    "raw_target",
    "raw_reference_keep",
    "raw_direction",
    "raw_metadata",
    "raw_negative_penalty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize score distributions from a CIR output JSON file."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to CIR output JSON, e.g. output.smoke.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/score_analysis"),
        help="Directory for PNG and CSV outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only analyze the first K ranked results. Default: all results.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution. Default: 180.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    parser.add_argument(
        "--style",
        default="whitegrid",
        choices=["whitegrid", "darkgrid", "white", "dark", "ticks"],
        help="Seaborn style.",
    )
    return parser.parse_args()


def load_output(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Input JSON does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    status = payload.get("status")
    if status != "success":
        raise ValueError(
            "CIR output is not successful. "
            f"status={status!r}, error={payload.get('error')!r}"
        )

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("CIR output contains no results to visualize.")

    return payload


def flatten_results(payload: dict[str, Any], top_k: int | None) -> pd.DataFrame:
    results = payload["results"]
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("--top-k must be greater than zero.")
        results = results[:top_k]

    rows: list[dict[str, Any]] = []

    for position, result in enumerate(results, start=1):
        scores = result.get("scores") or {}
        raw_scores = result.get("raw_scores") or {}
        metadata = result.get("metadata") or {}

        row = {
            "rank": result.get("rank", position),
            "id": result.get("id"),
            "video_name": result.get("video_name"),
            "frame_name": result.get("frame_name"),
            "timestamp": result.get("timestamp"),
            "frame_id": result.get("frame_id"),
            "cluster_id": result.get("cluster_id"),
            "matched_query": result.get("matched_query") or "unknown",
            "matched_query_strength": result.get("matched_query_strength"),
            "score": result.get("score"),
            "composed": scores.get("composed"),
            "target": scores.get("target"),
            "reference_keep": scores.get("reference_keep"),
            "direction": scores.get("direction"),
            "metadata": scores.get("metadata"),
            "edit_score": scores.get("edit_score"),
            "edit_gate_penalty": scores.get("edit_gate_penalty", 0.0),
            "negative_penalty": scores.get("negative_penalty", 0.0),
            "raw_composed": raw_scores.get("composed"),
            "raw_target": raw_scores.get("target"),
            "raw_reference_keep": raw_scores.get("reference_keep"),
            "raw_direction": raw_scores.get("direction"),
            "raw_metadata": raw_scores.get("metadata"),
            "raw_negative_penalty": raw_scores.get("negative_penalty", 0.0),
            "shot": metadata.get("shot"),
            "image_path": result.get("image_path"),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)

    numeric_columns = [
        "rank",
        "timestamp",
        "frame_id",
        "matched_query_strength",
        *DEFAULT_SCORE_COLUMNS,
        *RAW_SCORE_COLUMNS,
        "shot",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values("rank").reset_index(drop=True)
    return frame


def available_score_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in DEFAULT_SCORE_COLUMNS
        if column in frame.columns
        and frame[column].notna().any()
        and frame[column].nunique(dropna=True) > 1
    ]


def available_raw_score_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in RAW_SCORE_COLUMNS
        if column in frame.columns
        and frame[column].notna().any()
        and frame[column].nunique(dropna=True) > 1
    ]


def save_figure(path: Path, dpi: int, show: bool) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_score_distributions(
    frame: pd.DataFrame,
    score_columns: list[str],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    n_columns = 2
    n_rows = math.ceil(len(score_columns) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(13, max(4, n_rows * 3.5)),
        squeeze=False,
    )

    for axis, column in zip(axes.flat, score_columns):
        values = frame[column].dropna()
        if values.empty:
            axis.set_visible(False)
            continue

        # KDE is unstable for a constant or single-value column.
        use_kde = values.nunique() > 1 and len(values) >= 3
        sns.histplot(
            data=frame,
            x=column,
            bins="auto",
            kde=use_kde,
            ax=axis,
        )
        axis.axvline(values.mean(), linestyle="--", linewidth=1.4, label="Mean")
        axis.axvline(values.median(), linestyle=":", linewidth=1.4, label="Median")
        axis.set_title(f"Distribution: {column}")
        axis.set_xlabel("Similarity / score")
        axis.set_ylabel("Number of candidates")
        axis.legend()

    for axis in axes.flat[len(score_columns) :]:
        axis.set_visible(False)

    figure.suptitle("CIR score distributions", y=1.01)
    save_figure(output_dir / "01_score_distributions.png", dpi, show)


def plot_scores_by_rank(
    frame: pd.DataFrame,
    score_columns: list[str],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    long_frame = frame.melt(
        id_vars=["rank"],
        value_vars=score_columns,
        var_name="score_type",
        value_name="value",
    ).dropna(subset=["value"])

    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=long_frame,
        x="rank",
        y="value",
        hue="score_type",
        marker="o",
        linewidth=1.6,
    )
    plt.title("Score profile by result rank")
    plt.xlabel("Rank")
    plt.ylabel("Similarity / score")
    plt.legend(title="Score", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(output_dir / "02_scores_by_rank.png", dpi, show)


def plot_score_correlation(
    frame: pd.DataFrame,
    score_columns: list[str],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    correlation = frame[score_columns].corr(numeric_only=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title("Correlation between CIR score components")
    save_figure(output_dir / "03_score_correlation.png", dpi, show)


def plot_target_vs_keep(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    required = {"target", "reference_keep", "score", "matched_query", "rank"}
    if not required.issubset(frame.columns):
        return

    plot_frame = frame.dropna(subset=["target", "reference_keep", "score"]).copy()
    if plot_frame.empty:
        return

    # Avoid nearly invisible points when final scores are close together.
    minimum = plot_frame["score"].min()
    maximum = plot_frame["score"].max()
    if maximum > minimum:
        plot_frame["point_size"] = 70 + 280 * (
            (plot_frame["score"] - minimum) / (maximum - minimum)
        )
    else:
        plot_frame["point_size"] = 140

    plt.figure(figsize=(11, 8))
    axis = sns.scatterplot(
        data=plot_frame,
        x="reference_keep",
        y="target",
        hue="matched_query",
        size="point_size",
        sizes=(60, 350),
        alpha=0.8,
    )

    # Label only the highest-ranked points to keep the plot readable.
    for _, row in plot_frame.nsmallest(min(12, len(plot_frame)), "rank").iterrows():
        axis.annotate(
            str(int(row["rank"])),
            (row["reference_keep"], row["target"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    plt.title("Target satisfaction vs reference preservation")
    plt.xlabel("Reference keep similarity")
    plt.ylabel("Target-text similarity")
    plt.legend(title="Matched query / final score", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(output_dir / "04_target_vs_reference_keep.png", dpi, show)


def plot_matched_query_distribution(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    counts = (
        frame["matched_query"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("matched_query")
        .reset_index(name="count")
    )

    plt.figure(figsize=(11, max(4, 0.55 * len(counts) + 2)))
    axis = sns.barplot(
        data=counts,
        y="matched_query",
        x="count",
        orient="h",
    )
    axis.bar_label(axis.containers[0], padding=3)
    plt.title("Which query vector produced the best match?")
    plt.xlabel("Number of returned candidates")
    plt.ylabel("Matched query")
    save_figure(output_dir / "05_matched_query_distribution.png", dpi, show)


def plot_boxplots_by_query(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    value_columns = [
        column
        for column in ["score", "target", "reference_keep", "direction"]
        if column in frame.columns and frame[column].notna().any()
    ]
    if not value_columns:
        return

    long_frame = frame.melt(
        id_vars=["matched_query"],
        value_vars=value_columns,
        var_name="score_type",
        value_name="value",
    ).dropna(subset=["value"])

    plt.figure(figsize=(14, 7))
    sns.boxplot(
        data=long_frame,
        x="matched_query",
        y="value",
        hue="score_type",
    )
    plt.title("Score components grouped by matched query")
    plt.xlabel("Matched query")
    plt.ylabel("Similarity / score")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Score", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(output_dir / "06_score_boxplots_by_query.png", dpi, show)


def write_summary(
    payload: dict[str, Any],
    frame: pd.DataFrame,
    score_columns: list[str],
    output_dir: Path,
) -> None:
    summary = frame[score_columns].describe().T
    summary["median"] = frame[score_columns].median()
    summary["missing"] = frame[score_columns].isna().sum()
    summary.to_csv(output_dir / "score_summary.csv", encoding="utf-8")
    frame.to_csv(output_dir / "flattened_results.csv", index=False, encoding="utf-8")

    timings = payload.get("timings_ms") or {}
    query = payload.get("query") or {}

    report = {
        "number_of_results": int(len(frame)),
        "candidate_pool_size": query.get("candidate_pool_size"),
        "used_vlm": query.get("used_vlm"),
        "total_latency_ms": timings.get("total"),
        "matched_query_counts": {
            str(key): int(value)
            for key, value in frame["matched_query"].value_counts().items()
        },
        "score_means": {
            column: (
                None if pd.isna(frame[column].mean()) else float(frame[column].mean())
            )
            for column in score_columns
        },
    }

    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    sns.set_theme(style=args.style, context="notebook")

    try:
        payload = load_output(args.input)
        frame = flatten_results(payload, args.top_k)
        score_columns = available_score_columns(frame)

        if not score_columns:
            raise ValueError("No numeric score columns were found in the CIR output.")

        args.output_dir.mkdir(parents=True, exist_ok=True)

        plot_score_distributions(
            frame, score_columns, args.output_dir, args.dpi, args.show
        )
        plot_scores_by_rank(frame, score_columns, args.output_dir, args.dpi, args.show)
        plot_score_correlation(
            frame, score_columns, args.output_dir, args.dpi, args.show
        )
        plot_target_vs_keep(frame, args.output_dir, args.dpi, args.show)
        plot_matched_query_distribution(frame, args.output_dir, args.dpi, args.show)
        plot_boxplots_by_query(frame, args.output_dir, args.dpi, args.show)
        write_summary(payload, frame, score_columns, args.output_dir)

        print(f"Analyzed {len(frame)} results from: {args.input}")
        print(f"Outputs written to: {args.output_dir.resolve()}")
        for path in sorted(args.output_dir.iterdir()):
            print(f"  - {path.name}")
        return 0

    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

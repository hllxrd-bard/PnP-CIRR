#!/usr/bin/env python3
"""Objective diagnostics for CIR reranking outputs.

This script deliberately separates four concepts:

1. Raw similarities:
   Cosine similarities produced by the embedding model.
2. Normalized component scores:
   Candidate-pool percentile/min-max scores used by the reranker.
3. Weighted contributions:
   The actual contribution of each normalized component to the final score.
4. Edit-gate penalties:
   The amount subtracted when target/direction satisfaction is insufficient.

The legacy output field ``matched_query`` is interpreted as
``best_composed_query`` when the output declares:
    composed_query_policy = directional_and_geodesic_only

This matches CIR reranker v2, where ``matched_query`` is assigned after
considering only directional/geodesic query vectors. It is not treated as
ANN provenance.

Example:
    python scripts/visualize_cir_scores_objective.py \
        --input outputs/output.analysis.json \
        --output-dir outputs/score_analysis_objective
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


POSITIVE_COMPONENTS = [
    "composed",
    "target",
    "reference_keep",
    "direction",
    "metadata",
]

NORMALIZED_COLUMNS = [
    "score",
    *POSITIVE_COMPONENTS,
    "edit_score",
    "edit_gate_penalty",
    "negative_penalty",
]

RAW_COLUMNS = [
    "raw_composed",
    "raw_target",
    "raw_reference_keep",
    "raw_direction",
    "raw_metadata",
    "raw_negative_penalty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate objective diagnostics for a CIR reranking output."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/score_analysis_objective"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Analyze only the first K returned results. Default: all returned results.",
    )
    parser.add_argument(
        "--rank-plot-k",
        type=int,
        default=40,
        help="Number of top ranks shown in rank-profile plots.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--negative-penalty-weight",
        type=float,
        default=0.20,
        help="Used to reconstruct final scores when not present in output metadata.",
    )
    parser.add_argument(
        "--style",
        default="whitegrid",
        choices=["whitegrid", "darkgrid", "white", "dark", "ticks"],
    )
    return parser.parse_args()


def load_output(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Input JSON does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("status") != "success":
        raise ValueError(
            f"CIR request did not succeed: status={payload.get('status')!r}, "
            f"error={payload.get('error')!r}"
        )

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("The CIR output contains no results.")

    return payload


def normalized_weights(payload: dict[str, Any]) -> dict[str, float]:
    reranking = (payload.get("query") or {}).get("reranking") or {}
    configured = reranking.get("weights") or {}

    defaults = {
        "composed": 0.30,
        "target": 0.30,
        "reference_keep": 0.15,
        "direction": 0.20,
        "metadata": 0.05,
    }
    parsed = {
        name: max(0.0, float(configured.get(name, defaults[name])))
        for name in POSITIVE_COMPONENTS
    }
    total = sum(parsed.values())
    if total <= 0:
        raise ValueError("Reranking weights must contain at least one positive value.")
    return {name: value / total for name, value in parsed.items()}


def edit_gate_config(payload: dict[str, Any]) -> dict[str, float | bool]:
    reranking = (payload.get("query") or {}).get("reranking") or {}
    cfg = reranking.get("edit_gate") or {}

    target_weight = max(0.0, float(cfg.get("target_weight", 0.55)))
    direction_weight = max(0.0, float(cfg.get("direction_weight", 0.45)))
    total = target_weight + direction_weight
    if total <= 0:
        target_weight = direction_weight = 0.5
    else:
        target_weight /= total
        direction_weight /= total

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "target_weight": target_weight,
        "direction_weight": direction_weight,
        "minimum_score": float(cfg.get("minimum_score", 0.35)),
        "penalty_weight": max(0.0, float(cfg.get("penalty_weight", 0.20))),
    }


def infer_query_semantics(payload: dict[str, Any]) -> tuple[str, list[str]]:
    query = payload.get("query") or {}
    reranking = query.get("reranking") or {}
    policy = str(reranking.get("composed_query_policy", "")).strip()

    warnings: list[str] = []
    if policy == "directional_and_geodesic_only":
        label = "best_composed_query"
    else:
        label = "matched_query"
        warnings.append(
            "The output does not explicitly declare the directional/geodesic-only "
            "composed-query policy. The legacy matched_query field is kept as-is."
        )
    return label, warnings


def flatten_results(
    payload: dict[str, Any],
    top_k: int | None,
    negative_penalty_weight: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    results = payload["results"]
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("--top-k must be positive.")
        results = results[:top_k]

    query_label, warnings = infer_query_semantics(payload)
    weights = normalized_weights(payload)
    gate = edit_gate_config(payload)

    rows: list[dict[str, Any]] = []
    for position, result in enumerate(results, start=1):
        scores = result.get("scores") or {}
        raw_scores = result.get("raw_scores") or {}
        metadata = result.get("metadata") or {}

        best_composed_query = (
            result.get("best_composed_query")
            or result.get("matched_query")
            or "unknown"
        )

        row: dict[str, Any] = {
            "rank": result.get("rank", position),
            "id": result.get("id"),
            "video_name": result.get("video_name"),
            "frame_name": result.get("frame_name"),
            "timestamp": result.get("timestamp"),
            "frame_id": result.get("frame_id"),
            "cluster_id": result.get("cluster_id"),
            "best_composed_query": best_composed_query,
            "best_composed_strength": (
                result.get("best_composed_query_strength")
                if result.get("best_composed_query_strength") is not None
                else result.get("matched_query_strength")
            ),
            "best_ann_query": result.get("best_ann_query"),
            "retrieved_by": result.get("retrieved_by"),
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

        for name in POSITIVE_COMPONENTS:
            value = row.get(name)
            row[f"contribution_{name}"] = (
                None if value is None else weights[name] * float(value)
            )

        gate_penalty_value = float(row.get("edit_gate_penalty") or 0.0)
        negative_penalty_value = float(row.get("negative_penalty") or 0.0)
        row["contribution_edit_gate"] = -gate["penalty_weight"] * gate_penalty_value
        row["contribution_negative"] = (
            -max(0.0, negative_penalty_weight) * negative_penalty_value
        )

        contribution_names = [
            f"contribution_{name}" for name in POSITIVE_COMPONENTS
        ] + ["contribution_edit_gate", "contribution_negative"]
        values = [row.get(name) for name in contribution_names]
        if all(value is not None for value in values):
            row["reconstructed_score"] = float(sum(values))
            row["score_reconstruction_error"] = (
                float(row["score"]) - row["reconstructed_score"]
                if row.get("score") is not None
                else None
            )
        else:
            row["reconstructed_score"] = None
            row["score_reconstruction_error"] = None

        row["gate_active"] = gate_penalty_value > 1e-12
        rows.append(row)

    frame = pd.DataFrame(rows)

    numeric_columns = [
        "rank",
        "timestamp",
        "frame_id",
        "best_composed_strength",
        *NORMALIZED_COLUMNS,
        *RAW_COLUMNS,
        *[f"contribution_{name}" for name in POSITIVE_COMPONENTS],
        "contribution_edit_gate",
        "contribution_negative",
        "reconstructed_score",
        "score_reconstruction_error",
        "shot",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values("rank").reset_index(drop=True)

    metadata = {
        "query_field_interpretation": query_label,
        "weights": weights,
        "edit_gate": gate,
        "warnings": warnings,
    }
    return frame, metadata


def save_figure(path: Path, dpi: int, show: bool) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def variable_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column in frame
        and frame[column].notna().any()
        and frame[column].nunique(dropna=True) > 1
    ]


def plot_normalized_ecdf(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    columns = variable_numeric_columns(
        frame,
        ["composed", "target", "reference_keep", "direction", "metadata", "edit_score"],
    )
    if not columns:
        return

    long_frame = frame.melt(
        value_vars=columns,
        var_name="component",
        value_name="normalized_score",
    ).dropna()

    plt.figure(figsize=(11, 7))
    sns.ecdfplot(
        data=long_frame,
        x="normalized_score",
        hue="component",
        linewidth=2,
    )
    plt.xlim(-0.02, 1.02)
    plt.title("Normalized component scores across returned candidates")
    plt.xlabel("Candidate-pool normalized score")
    plt.ylabel("Cumulative proportion")
    plt.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(output_dir / "01_normalized_score_ecdf.png", dpi, show)


def plot_raw_distributions(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    columns = variable_numeric_columns(frame, RAW_COLUMNS)
    if not columns:
        return

    ncols = 2
    nrows = math.ceil(len(columns) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, max(4, 3.5 * nrows)),
        squeeze=False,
    )

    for axis, column in zip(axes.flat, columns):
        values = frame[column].dropna()
        sns.histplot(
            data=frame,
            x=column,
            bins="auto",
            kde=len(values) >= 30 and values.nunique() > 2,
            ax=axis,
        )
        axis.axvline(values.mean(), linestyle="--", linewidth=1.2, label="Mean")
        axis.axvline(values.median(), linestyle=":", linewidth=1.2, label="Median")
        axis.set_title(column.replace("raw_", "Raw similarity: "))
        axis.set_xlabel("Raw cosine similarity")
        axis.set_ylabel("Candidates")
        axis.legend()

    for axis in axes.flat[len(columns):]:
        axis.set_visible(False)

    figure.suptitle(
        "Raw model similarities — do not compare their absolute scales directly",
        y=1.01,
    )
    save_figure(output_dir / "02_raw_similarity_distributions.png", dpi, show)


def plot_weighted_contributions(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
    rank_plot_k: int,
) -> None:
    subset = frame.nsmallest(min(rank_plot_k, len(frame)), "rank").copy()
    contribution_columns = [
        *[f"contribution_{name}" for name in POSITIVE_COMPONENTS],
        "contribution_edit_gate",
        "contribution_negative",
    ]
    available = [
        column for column in contribution_columns
        if column in subset and subset[column].notna().any()
    ]
    if not available:
        return

    plot_frame = subset.set_index("rank")[available]
    positive = [column for column in available if not column.startswith("contribution_edit") and not column.startswith("contribution_negative")]
    negative = [column for column in available if column not in positive]

    palette = sns.color_palette("deep", n_colors=max(len(positive), 1))
    fig, ax = plt.subplots(figsize=(15, 8))

    bottom = np.zeros(len(plot_frame), dtype=float)
    x = np.arange(len(plot_frame))
    for color, column in zip(palette, positive):
        values = plot_frame[column].fillna(0.0).to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=column.replace("contribution_", ""),
            color=color,
        )
        bottom += values

    neg_bottom = np.zeros(len(plot_frame), dtype=float)
    for column in negative:
        values = plot_frame[column].fillna(0.0).to_numpy()
        if np.allclose(values, 0.0):
            continue
        ax.bar(
            x,
            values,
            bottom=neg_bottom,
            label=column.replace("contribution_", ""),
            alpha=0.7,
        )
        neg_bottom += values

    ax.plot(
        x,
        subset["score"].to_numpy(),
        color="black",
        marker="o",
        linewidth=1.5,
        label="stored final score",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subset["rank"].astype(int))
    ax.set_title("Actual weighted contributions to the final ranking")
    ax.set_xlabel("Returned rank")
    ax.set_ylabel("Contribution to final score")
    ax.legend(title="Term", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(output_dir / "03_weighted_contributions_by_rank.png", dpi, show)


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = (
            (x >= x[i])
            & (y >= y[i])
            & ((x > x[i]) | (y > y[i]))
        )
        if np.any(dominated):
            mask[i] = False
    return mask


def plot_target_keep_pareto(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    needed = ["target", "reference_keep", "edit_score", "score", "rank"]
    if any(column not in frame for column in needed):
        return

    plot_frame = frame.dropna(subset=needed).copy()
    if plot_frame.empty:
        return

    plot_frame["pareto"] = pareto_mask(
        plot_frame["reference_keep"].to_numpy(),
        plot_frame["target"].to_numpy(),
    )
    plot_frame["gate"] = np.where(plot_frame["gate_active"], "Gate penalty", "No gate penalty")

    plt.figure(figsize=(12, 8))
    axis = sns.scatterplot(
        data=plot_frame,
        x="reference_keep",
        y="target",
        hue="best_composed_query",
        style="gate",
        size="edit_score",
        sizes=(45, 260),
        alpha=0.78,
    )

    frontier = plot_frame[plot_frame["pareto"]].sort_values("reference_keep")
    if len(frontier) >= 2:
        axis.plot(
            frontier["reference_keep"],
            frontier["target"],
            linestyle="--",
            linewidth=1.3,
            color="black",
            label="Pareto frontier",
        )

    for _, row in plot_frame.nsmallest(min(20, len(plot_frame)), "rank").iterrows():
        axis.annotate(
            str(int(row["rank"])),
            (row["reference_keep"], row["target"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlim(-0.03, 1.03)
    plt.ylim(-0.03, 1.03)
    plt.title("Target satisfaction vs reference preservation")
    plt.xlabel("Normalized reference preservation")
    plt.ylabel("Normalized target satisfaction")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(output_dir / "04_target_keep_pareto.png", dpi, show)


def plot_edit_gate(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    if "edit_score" not in frame or "score" not in frame:
        return

    plot_frame = frame.dropna(subset=["edit_score", "score"]).copy()
    if plot_frame.empty:
        return

    threshold = float(metadata["edit_gate"]["minimum_score"])
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=plot_frame,
        x="edit_score",
        y="score",
        hue="gate_active",
        style="best_composed_query",
        alpha=0.8,
    )
    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"Gate threshold = {threshold:.2f}",
    )
    plt.xlim(-0.03, 1.03)
    plt.title("Edit-gate behavior")
    plt.xlabel("Normalized edit score")
    plt.ylabel("Final reranking score")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(output_dir / "05_edit_gate_diagnostics.png", dpi, show)


def query_sort_key(value: str) -> tuple[int, float, str]:
    value = str(value)
    if value == "reference":
        return (0, 0.0, value)
    if value == "target_text":
        return (1, 0.0, value)
    if value.startswith("directional_"):
        try:
            return (2, float(value.split("_", 1)[1]), value)
        except ValueError:
            return (2, 999.0, value)
    if value.startswith("geodesic_"):
        try:
            return (3, float(value.split("_", 1)[1]), value)
        except ValueError:
            return (3, 999.0, value)
    return (4, 0.0, value)


def plot_query_groups(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    counts = (
        frame["best_composed_query"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("best_composed_query")
        .reset_index(name="count")
    )
    counts["sort_key"] = counts["best_composed_query"].map(query_sort_key)
    counts = counts.sort_values("sort_key").drop(columns="sort_key")

    plt.figure(figsize=(11, max(4, 0.55 * len(counts) + 2)))
    axis = sns.barplot(
        data=counts,
        x="count",
        y="best_composed_query",
        orient="h",
    )
    axis.bar_label(axis.containers[0], padding=3)
    plt.title("Best composed query among directional/geodesic probes")
    plt.xlabel("Returned candidates")
    plt.ylabel("Best composed query")
    save_figure(output_dir / "06_best_composed_query_counts.png", dpi, show)

    group_count = frame["best_composed_query"].nunique(dropna=True)
    if group_count < 2:
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.axis("off")
        only_group = str(frame["best_composed_query"].dropna().iloc[0])
        axis.text(
            0.5,
            0.5,
            "Only one best-composed-query group is present:\n"
            f"{only_group}\n\nA between-query boxplot would not be informative.",
            ha="center",
            va="center",
            fontsize=13,
        )
        save_figure(
            output_dir / "07_scores_by_best_composed_query.png",
            dpi,
            show,
        )
        return

    value_columns = [
        column
        for column in ["score", "target", "reference_keep", "direction", "edit_score"]
        if column in frame and frame[column].notna().any()
    ]
    long_frame = frame.melt(
        id_vars=["best_composed_query"],
        value_vars=value_columns,
        var_name="component",
        value_name="value",
    ).dropna()

    ordered_queries = sorted(
        frame["best_composed_query"].dropna().astype(str).unique(),
        key=query_sort_key,
    )
    plt.figure(figsize=(15, 8))
    sns.boxplot(
        data=long_frame,
        x="best_composed_query",
        y="value",
        hue="component",
        order=ordered_queries,
    )
    plt.ylim(-0.03, 1.03)
    plt.title("Normalized scores grouped by best composed query")
    plt.xlabel("Best composed query")
    plt.ylabel("Normalized score")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(output_dir / "07_scores_by_best_composed_query.png", dpi, show)


def plot_rank_correlations(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> dict[str, float]:
    columns = [
        column
        for column in [
            "composed",
            "target",
            "reference_keep",
            "direction",
            "metadata",
            "edit_score",
            "edit_gate_penalty",
        ]
        if column in frame and frame[column].notna().any()
    ]
    correlations: dict[str, float] = {}
    for column in columns:
        value = frame[["score", column]].corr(method="spearman").iloc[0, 1]
        correlations[column] = None if pd.isna(value) else float(value)

    correlation_frame = pd.DataFrame(
        [
            {"component": key, "spearman_with_final": value}
            for key, value in correlations.items()
            if value is not None
        ]
    ).sort_values("spearman_with_final")

    if correlation_frame.empty:
        return correlations

    plt.figure(figsize=(10, 6))
    axis = sns.barplot(
        data=correlation_frame,
        x="spearman_with_final",
        y="component",
        orient="h",
    )
    axis.axvline(0.0, linewidth=1.0)
    plt.xlim(-1.0, 1.0)
    plt.title("Spearman association with the final reranking score")
    plt.xlabel("Spearman correlation")
    plt.ylabel("Component")
    save_figure(output_dir / "08_component_rank_correlations.png", dpi, show)
    return correlations


def plot_raw_vs_normalized(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    pairs = [
        ("raw_composed", "composed"),
        ("raw_target", "target"),
        ("raw_reference_keep", "reference_keep"),
        ("raw_direction", "direction"),
        ("raw_metadata", "metadata"),
    ]
    pairs = [
        pair for pair in pairs
        if pair[0] in frame and pair[1] in frame
        and frame[pair[0]].notna().any()
        and frame[pair[1]].notna().any()
    ]
    if not pairs:
        return

    ncols = 2
    nrows = math.ceil(len(pairs) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, max(4, 3.7 * nrows)),
        squeeze=False,
    )
    for axis, (raw_column, normalized_column) in zip(axes.flat, pairs):
        sns.scatterplot(
            data=frame,
            x=raw_column,
            y=normalized_column,
            hue="best_composed_query",
            alpha=0.7,
            legend=False,
            ax=axis,
        )
        axis.set_title(f"{raw_column} → {normalized_column}")
        axis.set_xlabel("Raw cosine similarity")
        axis.set_ylabel("Candidate-pool normalized score")
        axis.set_ylim(-0.03, 1.03)

    for axis in axes.flat[len(pairs):]:
        axis.set_visible(False)

    figure.suptitle("Effect of candidate-pool normalization", y=1.01)
    save_figure(output_dir / "09_raw_vs_normalized.png", dpi, show)


def plot_rank_profile(
    frame: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    show: bool,
    rank_plot_k: int,
) -> None:
    subset = frame.nsmallest(min(rank_plot_k, len(frame)), "rank")
    columns = [
        column
        for column in ["score", "target", "reference_keep", "direction", "edit_score"]
        if column in subset and subset[column].notna().any()
    ]
    if not columns:
        return

    long_frame = subset.melt(
        id_vars=["rank"],
        value_vars=columns,
        var_name="component",
        value_name="value",
    ).dropna()

    plt.figure(figsize=(15, 7))
    sns.lineplot(
        data=long_frame,
        x="rank",
        y="value",
        hue="component",
        marker="o",
        linewidth=1.5,
    )
    plt.ylim(-0.03, 1.03)
    plt.title(f"Normalized score profile for the top {len(subset)} results")
    plt.xlabel("Returned rank")
    plt.ylabel("Normalized score")
    plt.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(output_dir / "10_top_rank_score_profile.png", dpi, show)


def write_outputs(
    payload: dict[str, Any],
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    correlations: dict[str, float],
    output_dir: Path,
) -> None:
    frame.to_csv(output_dir / "flattened_results.csv", index=False, encoding="utf-8")

    score_columns = [
        column
        for column in [*NORMALIZED_COLUMNS, *RAW_COLUMNS]
        if column in frame and frame[column].notna().any()
    ]
    if score_columns:
        summary = frame[score_columns].describe().T
        summary["median"] = frame[score_columns].median()
        summary["missing"] = frame[score_columns].isna().sum()
        summary.to_csv(output_dir / "score_summary.csv", encoding="utf-8")

    query = payload.get("query") or {}
    request = payload.get("request") or {}
    candidate_pool_size = query.get("candidate_pool_size")
    returned_count = len(frame)

    warnings = list(metadata["warnings"])
    if returned_count < 50:
        warnings.append(
            "Fewer than 50 returned results were analyzed; distribution plots may be unstable."
        )
    if bool((request.get("deduplication") or {}).get("enabled", False)):
        warnings.append(
            "Deduplication was enabled. The analyzed distribution is selection-biased "
            "toward diverse results rather than the original reranking pool."
        )
    if isinstance(candidate_pool_size, int) and candidate_pool_size > returned_count:
        warnings.append(
            f"Only {returned_count}/{candidate_pool_size} reranked candidates are present "
            "in the output. The analysis describes returned top-K results, not the full pool."
        )

    policy = (
        ((query.get("reranking") or {}).get("composed_query_policy"))
        if isinstance(query, dict)
        else None
    )
    invalid_reference_count = int(
        (frame["best_composed_query"].astype(str) == "reference").sum()
    )
    if policy == "directional_and_geodesic_only" and invalid_reference_count > 0:
        warnings.append(
            f"{invalid_reference_count} results report best_composed_query='reference' "
            "despite the directional/geodesic-only policy. The running container may "
            "still be using an older reranker.py or an older output JSON."
        )

    reconstruction_error = frame["score_reconstruction_error"].abs().dropna()
    max_reconstruction_error = (
        None if reconstruction_error.empty else float(reconstruction_error.max())
    )
    if max_reconstruction_error is not None and max_reconstruction_error > 1e-4:
        warnings.append(
            "Final-score reconstruction differs from the stored score. Check whether "
            "the negative penalty weight or reranking formula differs from this script."
        )

    pareto_count = 0
    if {"target", "reference_keep"}.issubset(frame.columns):
        valid = frame.dropna(subset=["target", "reference_keep"])
        if not valid.empty:
            pareto_count = int(
                pareto_mask(
                    valid["reference_keep"].to_numpy(),
                    valid["target"].to_numpy(),
                ).sum()
            )

    top_n = min(20, returned_count)
    report = {
        "number_of_results_analyzed": returned_count,
        "candidate_pool_size_reported": candidate_pool_size,
        "returned_pool_coverage": (
            None
            if not isinstance(candidate_pool_size, int) or candidate_pool_size <= 0
            else returned_count / candidate_pool_size
        ),
        "query_field_interpretation": metadata["query_field_interpretation"],
        "weights_used_for_reconstruction": metadata["weights"],
        "edit_gate": metadata["edit_gate"],
        "gate_active_count": int(frame["gate_active"].sum()),
        "gate_active_rate": float(frame["gate_active"].mean()),
        "best_composed_query_counts": {
            str(key): int(value)
            for key, value in frame["best_composed_query"].value_counts().items()
        },
        "spearman_with_final_score": correlations,
        "mean_normalized_scores_all": {
            column: float(frame[column].mean())
            for column in [
                "composed", "target", "reference_keep", "direction", "metadata", "edit_score"
            ]
            if column in frame and frame[column].notna().any()
        },
        f"mean_normalized_scores_top_{top_n}": {
            column: float(frame.nsmallest(top_n, "rank")[column].mean())
            for column in [
                "composed", "target", "reference_keep", "direction", "metadata", "edit_score"
            ]
            if column in frame and frame[column].notna().any()
        },
        "mean_weighted_contributions": {
            column.replace("contribution_", ""): float(frame[column].mean())
            for column in [
                *[f"contribution_{name}" for name in POSITIVE_COMPONENTS],
                "contribution_edit_gate",
                "contribution_negative",
            ]
            if column in frame and frame[column].notna().any()
        },
        "pareto_frontier_size_target_vs_keep": pareto_count,
        "max_score_reconstruction_error": max_reconstruction_error,
        "warnings": warnings,
    }

    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    markdown_lines = [
        "# CIR objective score analysis",
        "",
        f"- Results analyzed: **{returned_count}**",
        f"- Reported candidate pool: **{candidate_pool_size}**",
        f"- Gate active: **{report['gate_active_count']} "
        f"({report['gate_active_rate']:.1%})**",
        f"- Target/keep Pareto frontier size: **{pareto_count}**",
        "",
        "## Best composed query counts",
        "",
    ]
    for key, value in report["best_composed_query_counts"].items():
        markdown_lines.append(f"- `{key}`: {value}")

    markdown_lines.extend(["", "## Warnings", ""])
    if warnings:
        markdown_lines.extend(f"- {warning}" for warning in warnings)
    else:
        markdown_lines.append("- None.")

    (output_dir / "REPORT.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    sns.set_theme(style=args.style, context="notebook")

    try:
        payload = load_output(args.input)
        frame, metadata = flatten_results(
            payload,
            args.top_k,
            args.negative_penalty_weight,
        )
        if frame.empty:
            raise ValueError("No rows remain after applying --top-k.")

        args.output_dir.mkdir(parents=True, exist_ok=True)

        plot_normalized_ecdf(frame, args.output_dir, args.dpi, args.show)
        plot_raw_distributions(frame, args.output_dir, args.dpi, args.show)
        plot_weighted_contributions(
            frame,
            args.output_dir,
            args.dpi,
            args.show,
            args.rank_plot_k,
        )
        plot_target_keep_pareto(frame, args.output_dir, args.dpi, args.show)
        plot_edit_gate(frame, metadata, args.output_dir, args.dpi, args.show)
        plot_query_groups(frame, args.output_dir, args.dpi, args.show)
        correlations = plot_rank_correlations(
            frame, args.output_dir, args.dpi, args.show
        )
        plot_raw_vs_normalized(frame, args.output_dir, args.dpi, args.show)
        plot_rank_profile(
            frame,
            args.output_dir,
            args.dpi,
            args.show,
            args.rank_plot_k,
        )
        write_outputs(
            payload,
            frame,
            metadata,
            correlations,
            args.output_dir,
        )

        print(f"Analyzed {len(frame)} returned results from: {args.input}")
        print(f"Outputs written to: {args.output_dir.resolve()}")
        for path in sorted(args.output_dir.iterdir()):
            print(f"  - {path.name}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

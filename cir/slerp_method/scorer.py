from __future__ import annotations

from typing import Any

import numpy as np

from ..reranker import RankedCandidate
from ..utils import as_numpy, cosine_similarity


def rank_candidates_by_slerp_cosine(
    candidates: list[dict[str, Any]],
    image_vector_field: str,
    composed_vector: np.ndarray,
    query_name: str,
    alpha: float,
) -> list[RankedCandidate]:
    """Exact local cosine ranking for the pure training-free SLERP method."""

    valid_entities: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for entity in candidates:
        value = entity.get(image_vector_field)
        if value is None:
            continue
        vector = as_numpy(value)
        if vector.size != composed_vector.size:
            continue
        valid_entities.append(entity)
        vectors.append(vector)

    if not vectors:
        return []

    raw_scores = cosine_similarity(
        composed_vector,
        np.stack(vectors).astype(np.float32, copy=False),
    )

    ranked: list[RankedCandidate] = []
    for entity, score_value in zip(valid_entities, raw_scores, strict=True):
        score = float(score_value)
        ranked.append(
            RankedCandidate(
                entity=entity,
                score=score,
                component_scores={
                    "composed": score,
                    "target": 0.0,
                    "reference_keep": 0.0,
                    "direction": 0.0,
                    "metadata": 0.0,
                    "edit_score": score,
                    "edit_gate_penalty": 0.0,
                    "negative_penalty": 0.0,
                    "removal_penalty": 0.0,
                },
                raw_component_scores={
                    "composed": score,
                    "target": 0.0,
                    "reference_keep": 0.0,
                    "direction": 0.0,
                    "metadata": 0.0,
                    "negative_penalty": 0.0,
                    "removal_penalty": 0.0,
                },
                matched_query=query_name,
                matched_query_strength=float(alpha),
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked

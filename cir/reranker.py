from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import AppConfig
from .query_composer import ComposedQuery
from .utils import l2_normalize, normalize_rows


@dataclass
class RankedCandidate:
    entity: dict[str, Any]
    score: float
    component_scores: dict[str, float]
    raw_component_scores: dict[str, float]
    matched_query: str | None
    matched_query_strength: float | None


class VectorReranker:
    """Fast vector reranker for explicit Edit/Add and Remove fields.

    ``target`` is retained in score JSON for backward compatibility, but in this
    mode it means similarity to the Edit/Add text.  Remove-only requests disable
    target and metadata weights dynamically, avoiding a fake neutral target score.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.cfg = config.section("reranking")
        self.removal_cfg = config.section("object_removal")
        self.fields = config.get("milvus.fields", {})
        self.epsilon = float(config.get("composition.epsilon", 1e-8))

    @staticmethod
    def _query_is_composed(name: str) -> bool:
        return name.startswith("explicit_")

    def _normalize_component(
        self,
        values: np.ndarray,
        method: str,
        *,
        constant_value: float = 0.5,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return values

        finite = np.isfinite(values)
        if not finite.all():
            replacement = float(np.nanmedian(values[finite])) if finite.any() else 0.0
            values = np.where(finite, values, replacement).astype(np.float32)

        method = method.strip().lower()
        spread = float(values.max() - values.min())
        if spread <= self.epsilon:
            return np.full_like(values, float(constant_value), dtype=np.float32)
        if method == "none":
            return values.astype(np.float32)
        if method == "minmax":
            return ((values - values.min()) / max(spread, self.epsilon)).astype(
                np.float32
            )
        if method != "percentile":
            raise ValueError(
                "Unsupported reranking.score_normalization value: "
                f"{method!r}. Expected one of: none, minmax, percentile."
            )

        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(values.size, dtype=np.float32)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and np.isclose(
                sorted_values[end],
                sorted_values[start],
                rtol=1e-6,
                atol=self.epsilon,
            ):
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return (ranks / max(values.size - 1, 1)).astype(np.float32)

    @staticmethod
    def _normalized_weights(
        weights: dict[str, Any],
        *,
        has_edit_text: bool,
    ) -> dict[str, float]:
        names = ("composed", "target", "reference_keep", "direction", "metadata")
        parsed = {name: max(0.0, float(weights.get(name, 0.0))) for name in names}

        # Remove-only mode has no positive Edit/Add vector. Do not let neutral
        # target/metadata values consume score weight.
        if not has_edit_text:
            parsed["target"] = 0.0
            parsed["metadata"] = 0.0

        total = sum(parsed.values())
        if total <= 0.0:
            raise ValueError("At least one active positive reranking weight is required.")
        return {name: value / total for name, value in parsed.items()}

    def _penalty_scores(
        self,
        image_matrix: np.ndarray,
        vectors: np.ndarray | None,
        normalization: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = np.zeros(image_matrix.shape[0], dtype=np.float32)
        if vectors is not None and len(vectors) > 0:
            matrix = normalize_rows(np.asarray(vectors, dtype=np.float32), self.epsilon)
            raw = np.maximum(image_matrix @ matrix.T, 0.0).max(axis=1)
        if np.all(np.abs(raw) <= self.epsilon):
            return raw, np.zeros_like(raw)
        return raw, self._normalize_component(
            raw,
            normalization,
            constant_value=0.0,
        )

    def rank(
        self,
        candidates: list[dict[str, Any]],
        reference_vector: np.ndarray,
        composed: ComposedQuery,
        negative_vectors: np.ndarray | None = None,
        removal_vectors: np.ndarray | None = None,
    ) -> list[RankedCandidate]:
        if not candidates:
            return []

        image_field = self.fields["image_vector"]
        text_field = self.fields["text_vector"]
        valid_candidates: list[dict[str, Any]] = []
        image_vectors: list[np.ndarray] = []
        text_vectors: list[np.ndarray | None] = []

        for entity in candidates:
            vector = entity.get(image_field)
            if vector is None:
                continue
            array = np.asarray(vector, dtype=np.float32).reshape(-1)
            if array.size == 0:
                continue
            valid_candidates.append(entity)
            image_vectors.append(array)
            text_value = entity.get(text_field)
            text_vectors.append(
                np.asarray(text_value, dtype=np.float32).reshape(-1)
                if text_value is not None
                else None
            )

        if not valid_candidates:
            return []

        image_matrix = normalize_rows(np.stack(image_vectors, axis=0), self.epsilon)
        reference = l2_normalize(reference_vector, self.epsilon)
        direction = composed.direction_vector
        edit_vector = composed.edit_vector
        has_edit_text = edit_vector is not None

        composed_query_items = [
            item for item in composed.named_queries if self._query_is_composed(item.name)
        ]
        if not composed_query_items:
            composed_query_items = [
                item for item in composed.named_queries if item.name == "edit_text"
            ]
        if not composed_query_items:
            composed_query_items = [
                item for item in composed.named_queries if item.name != "reference"
            ]
        if not composed_query_items:
            composed_query_items = list(composed.named_queries)

        composed_query_matrix = normalize_rows(
            np.stack([item.vector for item in composed_query_items], axis=0),
            self.epsilon,
        )
        raw_composed_all = image_matrix @ composed_query_matrix.T
        best_composed_indices = np.argmax(raw_composed_all, axis=1)
        raw_composed = raw_composed_all[
            np.arange(len(valid_candidates)), best_composed_indices
        ]

        if has_edit_text:
            assert edit_vector is not None
            raw_target = image_matrix @ edit_vector
        else:
            raw_target = np.zeros(len(valid_candidates), dtype=np.float32)

        raw_keep = image_matrix @ reference
        candidate_directions = normalize_rows(
            image_matrix - reference[None, :],
            self.epsilon,
        )
        raw_direction = candidate_directions @ direction

        raw_metadata = np.zeros(len(valid_candidates), dtype=np.float32)
        if has_edit_text:
            assert edit_vector is not None
            valid_text_indices: list[int] = []
            valid_text_vectors: list[np.ndarray] = []
            for index, vector in enumerate(text_vectors):
                if vector is not None and vector.size == edit_vector.size:
                    valid_text_indices.append(index)
                    valid_text_vectors.append(vector)
            if valid_text_vectors:
                text_matrix = normalize_rows(
                    np.stack(valid_text_vectors, axis=0),
                    self.epsilon,
                )
                raw_metadata[np.asarray(valid_text_indices)] = text_matrix @ edit_vector

        normalization = str(self.cfg.get("score_normalization", "percentile"))
        composed_scores = self._normalize_component(raw_composed, normalization)
        target_scores = (
            self._normalize_component(raw_target, normalization)
            if has_edit_text
            else np.zeros_like(raw_target)
        )
        keep_scores = self._normalize_component(raw_keep, normalization)
        direction_scores = self._normalize_component(raw_direction, normalization)
        metadata_scores = (
            self._normalize_component(raw_metadata, normalization)
            if has_edit_text and np.any(np.abs(raw_metadata) > self.epsilon)
            else np.zeros_like(raw_metadata)
        )

        raw_negative_penalty, negative_penalties = self._penalty_scores(
            image_matrix,
            negative_vectors,
            normalization,
        )
        raw_removal_penalty, removal_penalties = self._penalty_scores(
            image_matrix,
            removal_vectors if composed.remove_texts else None,
            normalization,
        )

        weights = self._normalized_weights(
            self.cfg.get("weights", {}),
            has_edit_text=has_edit_text,
        )
        base_scores = (
            weights["composed"] * composed_scores
            + weights["target"] * target_scores
            + weights["reference_keep"] * keep_scores
            + weights["direction"] * direction_scores
            + weights["metadata"] * metadata_scores
        )

        edit_gate_cfg = self.cfg.get("edit_gate", {})
        gate_enabled = bool(edit_gate_cfg.get("enabled", True))
        target_gate_weight = max(
            0.0,
            float(edit_gate_cfg.get("target_weight", 0.55)),
        )
        direction_gate_weight = max(
            0.0,
            float(edit_gate_cfg.get("direction_weight", 0.45)),
        )

        if has_edit_text:
            gate_weight_sum = target_gate_weight + direction_gate_weight
            if gate_weight_sum <= 0.0:
                target_gate_weight, direction_gate_weight = 0.5, 0.5
            else:
                target_gate_weight /= gate_weight_sum
                direction_gate_weight /= gate_weight_sum
            edit_scores = (
                target_gate_weight * target_scores
                + direction_gate_weight * direction_scores
            ).astype(np.float32)
        else:
            # For remove-only queries, direction and removal penalty carry the edit.
            edit_scores = direction_scores.astype(np.float32)

        if gate_enabled:
            minimum_edit_score = max(
                0.0,
                min(1.0, float(edit_gate_cfg.get("minimum_score", 0.40))),
            )
            if minimum_edit_score > self.epsilon:
                edit_gate_penalties = np.maximum(
                    minimum_edit_score - edit_scores,
                    0.0,
                ) / minimum_edit_score
            else:
                edit_gate_penalties = np.zeros_like(edit_scores)
            edit_gate_penalty_weight = max(
                0.0,
                float(edit_gate_cfg.get("penalty_weight", 0.25)),
            )
        else:
            edit_gate_penalties = np.zeros_like(edit_scores)
            edit_gate_penalty_weight = 0.0

        negative_penalty_weight = max(
            0.0,
            float(self.cfg.get("negative_penalty_weight", 0.20)),
        )
        removal_penalty_weight = (
            max(
                0.0,
                float(self.removal_cfg.get("removal_penalty_weight", 0.35)),
            )
            if composed.remove_texts
            else 0.0
        )

        final_scores = (
            base_scores
            - edit_gate_penalty_weight * edit_gate_penalties
            - negative_penalty_weight * negative_penalties
            - removal_penalty_weight * removal_penalties
        )

        ranked: list[RankedCandidate] = []
        for index, entity in enumerate(valid_candidates):
            query_item = composed_query_items[int(best_composed_indices[index])]
            ranked.append(
                RankedCandidate(
                    entity=entity,
                    score=float(final_scores[index]),
                    component_scores={
                        "composed": float(composed_scores[index]),
                        "target": float(target_scores[index]),
                        "reference_keep": float(keep_scores[index]),
                        "direction": float(direction_scores[index]),
                        "metadata": float(metadata_scores[index]),
                        "edit_score": float(edit_scores[index]),
                        "edit_gate_penalty": float(edit_gate_penalties[index]),
                        "negative_penalty": float(negative_penalties[index]),
                        "removal_penalty": float(removal_penalties[index]),
                    },
                    raw_component_scores={
                        "composed": float(raw_composed[index]),
                        "target": float(raw_target[index]),
                        "reference_keep": float(raw_keep[index]),
                        "direction": float(raw_direction[index]),
                        "metadata": float(raw_metadata[index]),
                        "negative_penalty": float(raw_negative_penalty[index]),
                        "removal_penalty": float(raw_removal_penalty[index]),
                    },
                    matched_query=query_item.name,
                    matched_query_strength=query_item.strength,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

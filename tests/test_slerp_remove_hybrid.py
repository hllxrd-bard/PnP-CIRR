from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from cir.reranker import RankedCandidate
from cir.schemas import CIRRequest
from cir.slerp_method.remove_pipeline import search_slerp_remove


class _Config:
    def __init__(self) -> None:
        self.values = {
            "milvus.search.metric_type": "COSINE",
            "milvus.search.params.ef": 256,
            "retrieval.candidate_k_per_query": 150,
            "retrieval.max_candidate_pool": 700,
            "retrieval.default_top_k": 10,
            "retrieval.max_top_k": 300,
            "retrieval.exclude_reference": True,
            "composition.use_reference_query": True,
            "composition.use_edit_text_query": True,
            "slerp_remove.default_add_alpha": 0.4,
            "slerp_remove.default_gamma": 0.2,
            "slerp_remove.epsilon": 1e-8,
            "milvus.raw_text_paths": [],
            "reranking.score_normalization": "percentile",
            "reranking.weights": {},
            "reranking.edit_gate": {},
            "object_removal": {"removal_penalty_weight": 0.35},
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class _Composer:
    @staticmethod
    def split_remove_field(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @staticmethod
    def expand_remove_texts(values: list[str]) -> list[str]:
        return values + ["water lily"]


class _Encoder:
    @staticmethod
    def encode_texts(texts: list[str]) -> np.ndarray:
        mapping = {
            "pond": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "lotus": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "water lily": np.array([0.9, 0.1, 0.0], dtype=np.float32),
        }
        return np.stack([mapping[text] for text in texts], axis=0)


class _Store:
    def __init__(self) -> None:
        self.query_count = 0

    def search(self, query_vectors, limit, expression):
        self.query_count = len(query_vectors)
        return [
            [{"id": 2, "distance": 0.8 + index * 0.01}]
            for index in range(self.query_count)
        ]

    @staticmethod
    def fetch_entities(ids, include_vectors=True):
        assert ids == [2]
        return [
            {
                "id": 2,
                "image_embedding": [0.0, 1.0, 0.0],
                "text_embedding": [0.0, 1.0, 0.0],
                "video_name": "L30_V071",
                "frame_name": "frame_035",
                "timestamp": 1.0,
                "frame_id": 35,
                "cluster_id": "c1",
            }
        ]


class _Reranker:
    def __init__(self) -> None:
        self.called = False
        self.composed = None
        self.removal_vectors = None

    def rank(
        self,
        candidates,
        reference_vector,
        composed,
        negative_vectors=None,
        removal_vectors=None,
    ):
        self.called = True
        self.composed = composed
        self.removal_vectors = removal_vectors
        return [
            RankedCandidate(
                entity=candidates[0],
                score=0.75,
                component_scores={
                    "composed": 0.9,
                    "target": 0.8,
                    "reference_keep": 0.7,
                    "direction": 0.6,
                    "metadata": 0.5,
                    "edit_score": 0.7,
                    "edit_gate_penalty": 0.0,
                    "negative_penalty": 0.0,
                    "removal_penalty": 0.2,
                },
                raw_component_scores={
                    "composed": 0.8,
                    "target": 0.7,
                    "reference_keep": 0.6,
                    "direction": 0.5,
                    "metadata": 0.4,
                    "negative_penalty": 0.0,
                    "removal_penalty": 0.3,
                },
                matched_query="explicit_spherical_replace_a0.400_g0.200",
                matched_query_strength=0.2,
            )
        ]


class _Deduplicator:
    @staticmethod
    def apply(ranked, overrides, top_k):
        return ranked[:top_k]


class _Engine:
    def __init__(self) -> None:
        self.config = _Config()
        self.composer = _Composer()
        self.encoder = _Encoder()
        self.store = _Store()
        self.reranker = _Reranker()
        self.deduplicator = _Deduplicator()
        self.fields = {
            "id": "id",
            "image_vector": "image_embedding",
            "text_vector": "text_embedding",
            "video_name": "video_name",
            "frame_name": "frame_name",
            "timestamp": "timestamp",
            "frame_id": "frame_id",
            "cluster_id": "cluster_id",
            "metadata": "metadata",
        }

    @staticmethod
    def _resolve_reference(request):
        return (
            {
                "id": 1,
                "video_name": "L30_V071",
                "frame_name": "frame_034",
                "timestamp": 0.0,
                "image_embedding": [0.7, 0.7, 0.0],
            },
            np.array([0.7, 0.7, 0.0], dtype=np.float32),
            None,
        )

    @staticmethod
    def _build_expression(request, reference_entity):
        return "id != 1"

    @staticmethod
    def _passes_client_filters(entity, request):
        return True

    @staticmethod
    def _entity_image_path(entity):
        return None


def test_spherical_remove_reuses_directional_candidate_and_rerank_pipeline() -> None:
    request = CIRRequest(
        reference={"id": 1},
        composition_mode="slerp_remove",
        edit_text="pond",
        remove_text="lotus",
        slerp_alpha=0.4,
        slerp_remove_gamma=0.2,
        top_k=5,
        use_vlm=False,
    )
    engine = _Engine()

    output = search_slerp_remove(engine, request)

    assert engine.store.query_count == 3
    assert engine.reranker.called is True
    assert engine.reranker.removal_vectors.shape == (2, 3)
    assert [item.name for item in engine.reranker.composed.named_queries] == [
        "reference",
        "edit_text",
        "explicit_spherical_replace_a0.400_g0.200",
    ]
    assert output.query["expanded_remove_objects"] == ["lotus", "water lily"]
    assert output.query["reranking"]["method"] == "directional_vector_reranker"
    assert output.query["composition"]["single_milvus_query"] is False
    assert output.results[0].retrieved_by == [
        "reference",
        "edit_text",
        "explicit_spherical_replace_a0.400_g0.200",
    ]

from pathlib import Path
import copy

import numpy as np
import pytest

from cir.config import AppConfig, DEFAULT_CONFIG
from cir.schemas import CIRRequest
from cir.slerp_method.composer import spherical_linear_interpolation
from cir.slerp_method.scorer import rank_candidates_by_slerp_cosine


def test_slerp_endpoints_and_unit_norm():
    image = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    text = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    np.testing.assert_allclose(
        spherical_linear_interpolation(image, text, 0.0), image, atol=1e-6
    )
    np.testing.assert_allclose(
        spherical_linear_interpolation(image, text, 1.0), text, atol=1e-6
    )
    middle = spherical_linear_interpolation(image, text, 0.8)
    assert np.isfinite(middle).all()
    assert np.linalg.norm(middle) == pytest.approx(1.0, abs=1e-6)
    assert middle[1] > middle[0]


def test_slerp_parallel_fallback_is_finite():
    image = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    text = np.array([1.0, 1e-9, 0.0], dtype=np.float32)
    output = spherical_linear_interpolation(image, text, 0.8)
    assert np.isfinite(output).all()
    assert np.linalg.norm(output) == pytest.approx(1.0, abs=1e-6)


def test_slerp_alpha_validation():
    vector = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        spherical_linear_interpolation(vector, vector, -0.01)
    with pytest.raises(ValueError):
        spherical_linear_interpolation(vector, vector, 1.01)



def test_schema_defaults_to_directional_and_accepts_slerp():
    legacy = CIRRequest.model_validate(
        {"reference": {"id": 1}, "edit_text": "pond"}
    )
    assert legacy.composition_mode == "directional"

    request = CIRRequest.model_validate(
        {
            "reference": {"id": 1},
            "composition_mode": "slerp",
            "edit_text": "pond",
            "remove_text": "",
            "slerp_alpha": 0.8,
        }
    )
    assert request.composition_mode == "slerp"
    assert request.slerp_alpha == pytest.approx(0.8)


def test_exact_slerp_scorer_uses_only_composed_cosine():
    config = AppConfig(data=copy.deepcopy(DEFAULT_CONFIG), source_path=Path("config.yaml"))
    image_field = config.get("milvus.fields.image_vector")
    candidates = [
        {"id": 1, image_field: [1.0, 0.0]},
        {"id": 2, image_field: [0.0, 1.0]},
    ]
    ranked = rank_candidates_by_slerp_cosine(
        candidates=candidates,
        image_vector_field=image_field,
        composed_vector=np.array([1.0, 0.0], dtype=np.float32),
        query_name="slerp_0.800",
        alpha=0.8,
    )
    assert ranked[0].entity["id"] == 1
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[0].component_scores["target"] == 0.0
    assert ranked[0].component_scores["removal_penalty"] == 0.0


def test_pure_slerp_pipeline_uses_one_milvus_query_and_no_directional_reranker():
    from cir.slerp_method.pipeline import search_slerp

    config_data = copy.deepcopy(DEFAULT_CONFIG)
    config_data["slerp"] = {
        "default_alpha": 0.8,
        "candidate_k": 10,
        "max_candidate_pool": 10,
        "epsilon": 1e-8,
    }
    config_data["deduplication"]["enabled"] = False
    config = AppConfig(data=config_data, source_path=Path("config.yaml"))
    fields = config_data["milvus"]["fields"]

    reference = {
        fields["id"]: 1,
        fields["image_vector"]: [1.0, 0.0],
        fields["video_name"]: "L00_V001",
        fields["frame_name"]: "frame_001",
        fields["timestamp"]: 1.0,
        fields["metadata"]: {},
    }
    candidates = {
        2: {
            fields["id"]: 2,
            fields["image_vector"]: [1.0, 0.0],
            fields["video_name"]: "L00_V002",
            fields["frame_name"]: "frame_002",
            fields["timestamp"]: 2.0,
            fields["metadata"]: {},
        },
        3: {
            fields["id"]: 3,
            fields["image_vector"]: [0.0, 1.0],
            fields["video_name"]: "L00_V003",
            fields["frame_name"]: "frame_003",
            fields["timestamp"]: 3.0,
            fields["metadata"]: {},
        },
    }

    class FakeStore:
        def __init__(self):
            self.query_vectors = None

        def search(self, query_vectors, limit, expression=None):
            self.query_vectors = query_vectors
            return [[
                {"id": 2, "distance": 0.2, "entity": {}},
                {"id": 3, "distance": 0.9, "entity": {}},
            ]]

        def fetch_entities(self, ids, include_vectors=True):
            return [candidates[int(entity_id)] for entity_id in ids]

    class FakeEncoder:
        def encode_texts(self, texts):
            assert texts == ["pond"]
            return np.asarray([[0.0, 1.0]], dtype=np.float32)

    class FakeDeduplicator:
        def apply(self, ranked, overrides, top_k):
            return ranked[:top_k]

    class FakeEngine:
        def __init__(self):
            self.config = config
            self.fields = fields
            self.store = FakeStore()
            self.encoder = FakeEncoder()
            self.deduplicator = FakeDeduplicator()

        def _resolve_reference(self, request):
            return reference, np.asarray([1.0, 0.0], dtype=np.float32), None

        def _build_expression(self, request, reference_entity):
            return None

        def _passes_client_filters(self, entity, request):
            return True

        def _entity_image_path(self, entity):
            return None

    engine = FakeEngine()
    request = CIRRequest.model_validate(
        {
            "reference": {"id": 1},
            "composition_mode": "slerp",
            "edit_text": "pond",
            "remove_text": "",
            "slerp_alpha": 0.8,
            "top_k": 2,
            "use_vlm": True,
            "deduplication": {"enabled": False},
        }
    )
    output = search_slerp(engine, request)

    assert len(engine.store.query_vectors) == 1
    assert output.query["composition_mode"] == "slerp"
    assert output.query["intent_text"] == "pond"
    assert output.query["used_vlm"] is False
    assert output.results[0].id == 3
    assert output.results[0].matched_query == "slerp_0.800"
    assert output.warnings

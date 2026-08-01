from pathlib import Path
import copy

import numpy as np
import pytest

from cir.config import AppConfig, DEFAULT_CONFIG
from cir.query_composer import QueryComposer
from cir.schemas import CIRRequest
from cir.utils import l2_normalize


def make_config() -> AppConfig:
    return AppConfig(data=copy.deepcopy(DEFAULT_CONFIG), source_path=Path("config.yaml"))


def test_schema_accepts_edit_only():
    request = CIRRequest.model_validate(
        {
            "reference": {"id": 1},
            "edit_text": "holding a microphone",
            "use_vlm": False,
        }
    )
    assert request.edit_text == "holding a microphone"
    assert request.remove_text == ""


def test_schema_accepts_remove_only():
    request = CIRRequest.model_validate(
        {
            "reference": {"id": 1},
            "edit_text": "",
            "remove_text": "hat",
            "use_vlm": False,
        }
    )
    assert request.remove_text == "hat"


def test_schema_rejects_empty_edit_and_remove():
    with pytest.raises(ValueError):
        CIRRequest.model_validate(
            {
                "reference": {"id": 1},
                "edit_text": "",
                "remove_text": "",
            }
        )


def test_selected_strength_creates_one_explicit_query():
    composer = QueryComposer(make_config())
    reference = l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    edit_vector = l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))

    output = composer.compose(
        reference_image_vector=reference,
        edit_text="helmet",
        edit_vector=edit_vector,
        remove_texts=[],
        expanded_remove_texts=[],
        removal_vectors=None,
        edit_strength=0.95,
    )

    explicit = [item for item in output.named_queries if item.name.startswith("explicit_")]
    assert [item.name for item in explicit] == ["explicit_edit_0.950"]
    assert explicit[0].strength == 0.95


def test_remove_field_and_alias_expansion():
    composer = QueryComposer(make_config())
    raw = composer.split_remove_field("the hat, glasses\ncar")
    assert raw == ["hat", "glasses", "car"]
    expanded = composer.expand_remove_texts(["hat"])
    assert expanded == ["hat", "cap", "headwear"]


def test_legacy_one_box_removal_parser_extracts_only_object():
    composer = QueryComposer(make_config())
    assert composer.parse_removal_texts("remove the hat from the man") == ["hat"]
    assert composer.parse_removal_texts("xóa chiếc mũ khỏi người đàn ông") == ["mũ"]


def test_explicit_replace_direction_adds_and_subtracts():
    composer = QueryComposer(make_config())
    reference = l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    helmet = l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    hat = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)

    output = composer.compose(
        reference_image_vector=reference,
        edit_text="helmet",
        edit_vector=helmet,
        remove_texts=["hat"],
        expanded_remove_texts=["hat"],
        removal_vectors=hat,
        edit_strength=0.95,
    )

    assert output.operation == "replace"
    assert output.direction_vector[1] > 0
    assert output.direction_vector[2] < 0
    assert any(item.name == "explicit_replace_0.950" for item in output.named_queries)


def test_removal_penalty_demotes_candidate_with_removed_object():
    from cir.reranker import VectorReranker

    config_data = copy.deepcopy(DEFAULT_CONFIG)
    config_data["reranking"]["weights"] = {
        "composed": 0.25,
        "target": 0.35,
        "reference_keep": 0.10,
        "direction": 0.25,
        "metadata": 0.05,
    }
    config_data["object_removal"]["removal_penalty_weight"] = 0.80
    config_data["reranking"]["score_normalization"] = "minmax"
    config = AppConfig(data=config_data, source_path=Path("config.yaml"))

    composer = QueryComposer(config)
    reference = l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    removal = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    composed = composer.compose(
        reference_image_vector=reference,
        edit_text="",
        edit_vector=None,
        remove_texts=["hat"],
        expanded_remove_texts=["hat"],
        removal_vectors=removal,
        edit_strength=0.95,
    )

    fields = config_data["milvus"]["fields"]
    no_hat = {
        fields["id"]: 1,
        fields["image_vector"]: [0.55, 0.83, 0.02],
        fields["text_vector"]: [0.55, 0.83, 0.02],
    }
    with_hat = {
        fields["id"]: 2,
        fields["image_vector"]: [0.55, 0.65, 0.52],
        fields["text_vector"]: [0.55, 0.65, 0.52],
    }

    ranked = VectorReranker(config).rank(
        [with_hat, no_hat],
        reference_vector=reference,
        composed=composed,
        removal_vectors=removal,
    )
    assert ranked[0].entity[fields["id"]] == 1
    assert (
        ranked[0].component_scores["removal_penalty"]
        < ranked[1].component_scores["removal_penalty"]
    )
    assert ranked[0].component_scores["target"] == 0.0


def test_engine_remove_only_does_not_call_vlm():
    from cir.deduplicator import CandidateDeduplicator
    from cir.engine import CIREngine
    from cir.reranker import VectorReranker

    config_data = copy.deepcopy(DEFAULT_CONFIG)
    config_data["retrieval"]["default_top_k"] = 2
    config_data["deduplication"]["enabled"] = False
    config_data["reranking"]["score_normalization"] = "minmax"
    config_data["object_removal"]["aliases"] = {"hat": ["cap", "headwear"]}
    config = AppConfig(data=config_data, source_path=Path("config.yaml"))
    fields = config_data["milvus"]["fields"]

    reference = {
        fields["id"]: 1,
        fields["image_vector"]: [1.0, 0.0, 0.0],
        fields["text_vector"]: [0.8, 0.2, 0.0],
        fields["video_name"]: "L29_V001",
        fields["frame_name"]: "frame_001",
        fields["timestamp"]: 1.0,
        fields["cluster_id"]: 1,
        fields["metadata"]: {},
    }
    candidates = {
        2: {
            fields["id"]: 2,
            fields["image_vector"]: [0.55, 0.83, 0.02],
            fields["text_vector"]: [0.55, 0.83, 0.02],
            fields["video_name"]: "L29_V002",
            fields["frame_name"]: "frame_002",
            fields["timestamp"]: 2.0,
            fields["cluster_id"]: 2,
            fields["metadata"]: {},
        },
        3: {
            fields["id"]: 3,
            fields["image_vector"]: [0.55, 0.65, 0.52],
            fields["text_vector"]: [0.55, 0.65, 0.52],
            fields["video_name"]: "L29_V003",
            fields["frame_name"]: "frame_003",
            fields["timestamp"]: 3.0,
            fields["cluster_id"]: 3,
            fields["metadata"]: {},
        },
    }

    class FakeStore:
        def get_by_id(self, entity_id, include_vectors=True):
            return reference if int(entity_id) == 1 else candidates.get(int(entity_id))

        def get_by_video_frame(self, video_name, frame_name, include_vectors=True):
            return None

        def search(self, query_vectors, limit, expression=None):
            return [
                [
                    {"id": 2, "distance": 0.8, "entity": {}},
                    {"id": 3, "distance": 0.7, "entity": {}},
                ]
                for _ in query_vectors
            ]

        def fetch_entities(self, ids, include_vectors=True):
            return [candidates[int(entity_id)] for entity_id in ids]

    class FakeEncoder:
        def encode_texts(self, texts):
            vectors = []
            for text in texts:
                if text.casefold() in {"hat", "cap", "headwear"}:
                    vectors.append([0.0, 0.0, 1.0])
                else:
                    vectors.append([0.0, 1.0, 0.0])
            return np.asarray(vectors, dtype=np.float32)

        def encode_images(self, paths):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    class FailIfCalledVLM:
        def rewrite(self, *args, **kwargs):
            raise AssertionError("VLM must not be called when use_vlm=false")

    engine = CIREngine.__new__(CIREngine)
    engine.config = config
    engine.store = FakeStore()
    engine.encoder = FakeEncoder()
    engine.composer = QueryComposer(config)
    engine.reranker = VectorReranker(config)
    engine.deduplicator = CandidateDeduplicator(config)
    engine.vlm = FailIfCalledVLM()
    engine.fields = fields

    request = CIRRequest.model_validate(
        {
            "reference": {"id": 1},
            "edit_text": "",
            "remove_text": "hat",
            "top_k": 2,
            "use_vlm": False,
            "edit_strength": 0.95,
            "deduplication": {"enabled": False},
        }
    )
    output = engine.search(request)
    assert output.status == "success"
    assert output.query["used_vlm"] is False
    assert output.query["target_text"] is None
    assert output.query["operation"] == "remove"
    assert output.query["remove_objects"] == ["hat"]
    assert output.query["expanded_remove_objects"] == ["hat", "cap", "headwear"]
    assert output.query["query_vectors"][-1]["name"] == "explicit_remove_0.950"

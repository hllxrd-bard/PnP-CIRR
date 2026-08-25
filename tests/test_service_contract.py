import pytest
from pydantic import ValidationError

from cir.service_schemas import PublicCIRRequest, PublicReferenceInput


def test_reference_requires_exactly_one_locator():
    with pytest.raises(ValidationError):
        PublicReferenceInput(id=1, path="/tmp/a.jpg")


def test_embedding_requires_metadata():
    with pytest.raises(ValidationError):
        PublicReferenceInput(id=1, image_embedding=[0.1, 0.2])


def test_mode_aliases_are_normalized():
    pure = PublicCIRRequest(
        reference={"id": 1}, composition_mode="slerp", edit_text="target"
    )
    hybrid = PublicCIRRequest(
        reference={"id": 1},
        composition_mode="slerp_remove",
        remove_text="hat",
    )
    assert pure.canonical_mode == "pure_slerp"
    assert hybrid.canonical_mode == "slerp_hybrid"


def test_public_filters_reject_raw_milvus_expression():
    with pytest.raises(ValidationError):
        PublicCIRRequest(
            reference={"id": 1},
            edit_text="helmet",
            filters={"milvus_expression": "id > 0"},
        )

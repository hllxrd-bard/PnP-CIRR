from __future__ import annotations

import numpy as np
import pytest

from cir.schemas import CIRRequest
from cir.slerp_method.remove_composer import (
    normalized_mean,
    spherical_move_away,
)


def test_gamma_zero_returns_anchor() -> None:
    anchor = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    negative = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    output = spherical_move_away(anchor, negative, gamma=0.0)
    np.testing.assert_allclose(output, anchor, atol=1e-6)


def test_move_away_reduces_negative_similarity() -> None:
    anchor = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    negative = np.array([0.5, 0.8660254, 0.0], dtype=np.float32)
    output = spherical_move_away(anchor, negative, gamma=0.25)
    assert float(np.dot(output, negative)) < float(np.dot(anchor, negative))
    assert np.isclose(np.linalg.norm(output), 1.0, atol=1e-6)


def test_parallel_vectors_remain_finite() -> None:
    anchor = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    output = spherical_move_away(anchor, anchor, gamma=0.2)
    assert np.all(np.isfinite(output))
    assert np.isclose(np.linalg.norm(output), 1.0, atol=1e-6)
    assert float(np.dot(output, anchor)) < 1.0


def test_opposite_negative_keeps_anchor() -> None:
    anchor = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    negative = -anchor
    output = spherical_move_away(anchor, negative, gamma=0.5)
    np.testing.assert_allclose(output, anchor, atol=1e-6)


def test_normalized_mean() -> None:
    values = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    output = normalized_mean(values)
    np.testing.assert_allclose(output, np.array([2**-0.5, 2**-0.5]), atol=1e-6)


def test_schema_accepts_slerp_remove() -> None:
    request = CIRRequest(
        reference={"path": "/tmp/reference.webp"},
        composition_mode="slerp_remove",
        remove_text="lotus",
        slerp_remove_gamma=0.2,
    )
    assert request.composition_mode == "slerp_remove"


def test_schema_requires_remove_for_slerp_remove() -> None:
    with pytest.raises(ValueError, match="requires remove_text"):
        CIRRequest(
            reference={"path": "/tmp/reference.webp"},
            composition_mode="slerp_remove",
            edit_text="pond",
        )


def test_pure_slerp_rejects_remove() -> None:
    with pytest.raises(ValueError, match="does not accept remove_text"):
        CIRRequest(
            reference={"path": "/tmp/reference.webp"},
            composition_mode="slerp",
            edit_text="pond",
            remove_text="lotus",
        )

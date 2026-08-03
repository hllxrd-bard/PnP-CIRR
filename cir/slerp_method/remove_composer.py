from __future__ import annotations

import math

import numpy as np


def l2_normalize(vector: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= epsilon:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return value / norm


def normalized_mean(vectors: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array.")
    normalized = np.stack([l2_normalize(item, epsilon) for item in values], axis=0)
    return l2_normalize(normalized.mean(axis=0), epsilon)


def spherical_move_away(
    anchor_vector: np.ndarray,
    negative_vector: np.ndarray,
    gamma: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Move from ``anchor_vector`` away from ``negative_vector`` on the unit sphere.

    ``gamma`` is the geodesic angle in radians.  A value of zero returns the
    normalized anchor.  This is an experimental spherical remove extension,
    not the interpolation formula from the SLERP-TAT paper.
    """

    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0 or gamma > 1.5:
        raise ValueError("gamma must be a finite value in [0, 1.5] radians.")

    anchor = l2_normalize(anchor_vector, epsilon)
    negative = l2_normalize(negative_vector, epsilon)
    if gamma <= epsilon:
        return anchor

    dot = float(np.clip(np.dot(anchor, negative), -1.0, 1.0))
    tangent = negative - dot * anchor
    tangent_norm = float(np.linalg.norm(tangent))

    if tangent_norm <= epsilon:
        # If negative is already opposite to the anchor, the anchor is already
        # maximally far from it.  Moving would only make the score worse.
        if dot < 0.0:
            return anchor

        # If the vectors are parallel, every tangent direction is initially
        # equally valid.  Pick a deterministic orthogonal direction.
        basis = np.zeros_like(anchor)
        basis[int(np.argmin(np.abs(anchor)))] = 1.0
        tangent = basis - float(np.dot(basis, anchor)) * anchor
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= epsilon:
            raise ValueError("Could not construct a stable tangent direction.")

    tangent_unit = tangent / tangent_norm
    moved = math.cos(gamma) * anchor - math.sin(gamma) * tangent_unit
    return l2_normalize(moved, epsilon)

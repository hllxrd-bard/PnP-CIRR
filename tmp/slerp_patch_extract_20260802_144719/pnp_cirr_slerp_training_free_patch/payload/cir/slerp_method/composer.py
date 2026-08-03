from __future__ import annotations

import math

import numpy as np

from ..utils import l2_normalize


def spherical_linear_interpolation(
    image_vector: np.ndarray,
    text_vector: np.ndarray,
    alpha: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Return the training-free SLERP composed query.

    ``alpha=0`` is image-only and ``alpha=1`` is text-only. Both inputs and the
    returned vector are L2-normalized.
    """

    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("slerp_alpha must be a finite number in [0, 1].")

    image = l2_normalize(np.asarray(image_vector, dtype=np.float32), epsilon)
    text = l2_normalize(np.asarray(text_vector, dtype=np.float32), epsilon)
    if not np.any(image):
        raise ValueError("Cannot SLERP from an empty image embedding.")
    if not np.any(text):
        raise ValueError("Cannot SLERP to an empty text embedding.")

    if alpha == 0.0:
        return image
    if alpha == 1.0:
        return text

    dot = float(np.clip(np.dot(image, text), -1.0, 1.0))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)

    # Stable fallback for almost parallel vectors. Exact antipodal vectors are
    # extraordinarily unlikely for VLP embeddings; normalized LERP is still the
    # safest deterministic fallback for that numerical edge case.
    if abs(sin_theta) <= epsilon or abs(dot) > 0.9995:
        blended = (1.0 - alpha) * image + alpha * text
        normalized = l2_normalize(blended, epsilon)
        if np.any(normalized):
            return normalized
        return image if alpha < 0.5 else text

    image_weight = math.sin((1.0 - alpha) * theta) / sin_theta
    text_weight = math.sin(alpha * theta) / sin_theta
    return l2_normalize(image_weight * image + text_weight * text, epsilon)

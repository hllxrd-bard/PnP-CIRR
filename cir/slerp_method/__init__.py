"""Training-free SLERP retrieval method.

This package is intentionally separate from the existing directional
Add/Remove composer and reranker.
"""

from .composer import spherical_linear_interpolation
from .intent_builder import SlerpIntent, build_slerp_intent
from .pipeline import search_slerp

__all__ = [
    "SlerpIntent",
    "build_slerp_intent",
    "search_slerp",
    "spherical_linear_interpolation",
]

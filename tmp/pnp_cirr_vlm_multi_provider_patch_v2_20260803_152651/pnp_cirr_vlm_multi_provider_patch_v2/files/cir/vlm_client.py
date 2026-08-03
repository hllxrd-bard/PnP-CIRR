"""Backward-compatible import path for the multi-provider VLM router."""

from .vlm import VLMRouter

VLMClient = VLMRouter

__all__ = ["VLMClient"]

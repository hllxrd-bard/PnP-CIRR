from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .providers import GeminiProvider, QwenProvider


LOGGER = logging.getLogger(__name__)


_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "qwen": {
        "base_url": "http://192.168.20.150:8018/v1",
        "chat_completions_path": "/chat/completions",
        "api_key": None,
        "model": "Qwen3.5-9B-Q8_0.gguf",
        "timeout_seconds": 30.0,
        "temperature": 0.0,
        "max_tokens": 512,
        "enable_thinking": False,
        "force_no_think_prompt": True,
        "response_format_json": True,
        "send_reference_image": True,
        "image_detail": None,
        "verify_tls": True,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "chat_completions_path": "/chat/completions",
        "api_key": None,
        "model": "gemini-3.6-flash",
        "timeout_seconds": 90.0,
        "temperature": 0.0,
        "max_tokens": 256,
        "reasoning_effort": "minimal",
        "force_no_think_prompt": False,
        "response_format_json": True,
        "send_reference_image": True,
        "image_detail": None,
        "verify_tls": True,
        # Thêm hai dòng này
        "image_max_side": 384,
        "image_jpeg_quality": 82,
    },
}

_QWEN_LEGACY_KEYS = {
    "base_url",
    "chat_completions_path",
    "api_key",
    "model",
    "timeout_seconds",
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "enable_thinking",
    "force_no_think_prompt",
    "response_format_json",
    "send_reference_image",
    "image_detail",
    "verify_tls",
    "system_prompt",
}


def _expand_env(value: Any) -> Any:
    import os

    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


class VLMRouter:
    """Route one normalized VLM request to Qwen local or Gemini API."""

    def __init__(self, config: AppConfig):
        self.cfg = config.section("vlm")

    @property
    def default_provider(self) -> str:
        return str(self.cfg.get("default_provider", "qwen")).strip().lower()

    def available_providers(self) -> tuple[str, ...]:
        return ("qwen", "gemini")

    def _profile(self, provider: str) -> dict[str, Any]:
        provider = provider.strip().lower()
        if provider not in _BUILTIN_PROFILES:
            raise ValueError(
                f"Unknown VLM provider {provider!r}; expected qwen or gemini."
            )

        profile = dict(_BUILTIN_PROFILES[provider])

        # Backward compatibility: the project's existing flat `vlm:` block is
        # the Qwen profile. This preserves the exact current local Qwen config.
        if provider == "qwen":
            for key in _QWEN_LEGACY_KEYS:
                if key in self.cfg:
                    profile[key] = self.cfg[key]

        # An explicit nested provider profile wins over legacy flat values.
        configured_profiles = self.cfg.get("providers")
        if isinstance(configured_profiles, dict):
            override = configured_profiles.get(provider)
            if isinstance(override, dict):
                profile.update(override)

        profile = _expand_env(profile)
        if provider == "gemini" and not profile.get("api_key"):
            import os

            profile["api_key"] = os.environ.get("GEMINI_API_KEY")
        return profile

    def rewrite(
        self,
        edit_text: str,
        source_text: str | None = None,
        image_path: str | Path | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        edit_text = str(edit_text).strip()
        if not edit_text:
            raise ValueError("edit_text must not be empty.")

        selected = str(provider or self.default_provider).strip().lower()
        profile = self._profile(selected)
        if selected == "qwen":
            client = QwenProvider(profile)
        elif selected == "gemini":
            client = GeminiProvider(profile)
        else:  # guarded by _profile, retained for type checkers
            raise ValueError(f"Unsupported VLM provider: {selected}")

        result = client.rewrite(
            edit_text=edit_text,
            source_text=source_text,
            image_path=str(image_path) if image_path is not None else None,
        )
        meta = result.get("_meta") or {}
        LOGGER.info(
            "VLM rewrite succeeded | provider=%s | model=%s | latency_ms=%s | "
            "operation=%s | target=%r",
            meta.get("provider"),
            meta.get("model"),
            meta.get("latency_ms"),
            result.get("operation"),
            result.get("target_description"),
        )
        return result

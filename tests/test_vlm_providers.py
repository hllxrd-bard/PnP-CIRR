from __future__ import annotations

from pathlib import Path

from cir.config import AppConfig
from cir.vlm.providers import GeminiProvider, QwenProvider
from cir.vlm.router import VLMRouter


def _config() -> AppConfig:
    return AppConfig(
        data={
            "vlm": {
                "enabled_by_default": False,
                "default_provider": "qwen",
                "base_url": "http://192.168.20.150:8018/v1",
                "chat_completions_path": "/chat/completions",
                "api_key": None,
                "model": "Qwen3.5-9B-Q8_0.gguf",
            }
        },
        source_path=Path("test.yaml"),
    )


def test_router_preserves_flat_qwen_config() -> None:
    router = VLMRouter(_config())
    profile = router._profile("qwen")
    assert profile["base_url"] == "http://192.168.20.150:8018/v1"
    assert profile["model"] == "Qwen3.5-9B-Q8_0.gguf"


def test_qwen_payload_has_qwen_only_thinking_switch() -> None:
    payload = QwenProvider(
        {
            "model": "Qwen3.5-9B-Q8_0.gguf",
            "enable_thinking": False,
            "response_format_json": True,
        }
    )._build_payload([])
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in payload


def test_gemini_payload_has_reasoning_effort_only() -> None:
    payload = GeminiProvider(
        {
            "model": "gemini-3.6-flash",
            "reasoning_effort": "low",
            "response_format_json": True,
        }
    )._build_payload([])
    assert payload["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in payload

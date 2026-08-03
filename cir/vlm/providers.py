from __future__ import annotations

import time
from typing import Any

import httpx

from .common import build_messages, content_to_text, normalize_vlm_output


class OpenAICompatibleProvider:
    provider_name = "openai_compatible"

    def __init__(self, profile: dict[str, Any]):
        self.profile = dict(profile)

    def _url(self) -> str:
        base = str(self.profile.get("base_url", "")).strip().rstrip("/")
        path = str(
            self.profile.get("chat_completions_path", "/chat/completions")
        ).strip()
        if not base:
            raise ValueError(f"VLM provider {self.provider_name!r} has empty base_url.")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.profile.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    def rewrite(
        self,
        edit_text: str,
        source_text: str | None = None,
        image_path: str | None = None,
    ) -> dict[str, Any]:
        model = str(self.profile.get("model", "")).strip()
        if not model:
            raise ValueError(f"VLM provider {self.provider_name!r} has empty model.")

        messages = build_messages(
            self.profile,
            edit_text=edit_text,
            source_text=source_text,
            image_path=image_path,
        )
        payload = self._build_payload(messages)

        started = time.perf_counter()
        try:
            with httpx.Client(
                timeout=float(self.profile.get("timeout_seconds", 30.0)),
                verify=bool(self.profile.get("verify_tls", True)),
            ) as client:
                response = client.post(
                    self._url(),
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"{self.provider_name} VLM request failed with HTTP "
                f"{exc.response.status_code}: {exc.response.text[:1000]}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"{self.provider_name} VLM request failed: {exc}"
            ) from exc
        except ValueError as exc:
            raise RuntimeError(
                f"{self.provider_name} VLM returned a non-JSON HTTP response."
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self.provider_name} VLM returned no choices.")

        choice = choices[0]
        message = choice.get("message") or {}
        content_text = content_to_text(message.get("content")).strip()
        finish_reason = choice.get("finish_reason")

        if finish_reason == "length":
            raise RuntimeError(
                f"{self.provider_name} VLM output was truncated because max_tokens "
                f"was reached. Partial content: {content_text[:1000]!r}"
            )
        if not content_text:
            reasoning_text = content_to_text(
                message.get("reasoning_content")
            ).strip()
            if reasoning_text:
                raise RuntimeError(
                    f"{self.provider_name} returned only reasoning_content and no "
                    "final content. Check provider thinking settings."
                )
            raise RuntimeError(
                f"{self.provider_name} returned empty content "
                f"(finish_reason={finish_reason!r})."
            )

        result = normalize_vlm_output(content_text)
        result["_meta"] = {
            "provider": self.provider_name,
            "model": model,
            "latency_ms": round(latency_ms, 3),
            "finish_reason": finish_reason,
            "usage": data.get("usage") or {},
        }
        return result


class QwenProvider(OpenAICompatibleProvider):
    provider_name = "qwen"

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": str(self.profile["model"]),
            "messages": messages,
            "temperature": float(self.profile.get("temperature", 0.0)),
            "max_tokens": int(self.profile.get("max_tokens", 512)),
        }
        if "top_p" in self.profile:
            payload["top_p"] = float(self.profile["top_p"])
        if "seed" in self.profile:
            payload["seed"] = int(self.profile["seed"])
        if not bool(self.profile.get("enable_thinking", False)):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if bool(self.profile.get("response_format_json", True)):
            payload["response_format"] = {"type": "json_object"}
        return payload


class GeminiProvider(OpenAICompatibleProvider):
    provider_name = "gemini"

    def _headers(self) -> dict[str, str]:
        api_key = str(self.profile.get("api_key") or "").strip()
        if not api_key or "${" in api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is empty. Export GEMINI_API_KEY before starting "
                "the CIR service."
            )
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": str(self.profile["model"]),
            "messages": messages,
            "temperature": float(self.profile.get("temperature", 0.0)),
            "max_tokens": int(self.profile.get("max_tokens", 1024)),
            "reasoning_effort": str(
                self.profile.get("reasoning_effort", "low")
            ),
        }
        if bool(self.profile.get("response_format_json", True)):
            payload["response_format"] = {"type": "json_object"}
        return payload

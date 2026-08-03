from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from .config import AppConfig
from .utils import image_to_data_url, safe_json_extract


logger = logging.getLogger(__name__)


class VLMClient:
    def __init__(self, config: AppConfig):
        self.cfg = config.section("vlm")

    def _url(self) -> str:
        base = str(self.cfg.get("base_url", "")).strip().rstrip("/")
        path = str(
            self.cfg.get("chat_completions_path", "/chat/completions")
        ).strip()
        if not base:
            raise ValueError("vlm.base_url is empty.")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _build_messages(
        self,
        edit_text: str,
        source_text: str | None,
        image_path: str | Path | None,
    ) -> list[dict[str, Any]]:
        default_system_prompt = (
            "Inspect the reference image when one is supplied. Return only one valid "
            "JSON object using exactly this schema: "
            '{"source_description":"string","operation":"modify|remove",'
            '"target_description":"string","preserve":["string"],'
            '"change":["string"],"remove_objects":["string"],'
            '"negative":["string"]}. '
            "Describe visible content rather than copying television OCR, ticker text, "
            "timestamps, filenames, or unrelated ASR. For object removal, set operation "
            "to remove, put only the removed visual object names in remove_objects, and "
            "describe a positive replacement state in target_description, such as an "
            "uncovered head instead of merely saying without a hat. Preserve all visual "
            "details that are not explicitly changed. All list fields must be arrays. "
            "Do not return explanations or Markdown."
        )
        system_prompt = str(
            self.cfg.get("system_prompt", default_system_prompt)
        ).strip()

        force_no_think = bool(self.cfg.get("force_no_think_prompt", True))
        if force_no_think and "/no_think" not in system_prompt.lower():
            system_prompt += " /no_think"

        has_image = bool(image_path and Path(image_path).is_file())
        user_text = (
            "Create a source description and a modified target-image description for "
            "image retrieval.\n"
            f"Edit instruction: {edit_text}\n"
        )
        if source_text:
            user_text += (
                "Optional noisy metadata; use only visually relevant information: "
                f"{source_text}\n"
            )
        if has_image:
            user_text += "Use the attached reference image as the primary source of truth.\n"
        else:
            user_text += "No reference image is attached; infer only from the supplied text.\n"
        user_text += "Return JSON only."
        if force_no_think:
            user_text += "\n/no_think"

        send_image = bool(self.cfg.get("send_reference_image", False))
        user_content: str | list[dict[str, Any]]
        if send_image and has_image:
            image_url: dict[str, Any] = {
                "url": image_to_data_url(image_path),
            }
            # Some OpenAI-compatible vision servers do not accept image_url.detail.
            # Only include it when the user explicitly configures a non-empty value.
            image_detail = self.cfg.get("image_detail")
            if image_detail:
                image_url["detail"] = str(image_detail)
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": image_url},
            ]
        else:
            user_content = user_text

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        model = str(self.cfg.get("model", "")).strip()
        if not model:
            raise ValueError("vlm.model is empty.")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(self.cfg.get("temperature", 0.0)),
            "max_tokens": int(self.cfg.get("max_tokens", 256)),
        }
        if "top_p" in self.cfg:
            payload["top_p"] = float(self.cfg["top_p"])
        if "seed" in self.cfg:
            payload["seed"] = int(self.cfg["seed"])
        if not bool(self.cfg.get("enable_thinking", False)):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if bool(self.cfg.get("response_format_json", True)):
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text is not None:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            output: list[str] = []
            seen: set[str] = set()
            for item in value:
                text = str(item).strip()
                key = text.casefold()
                if text and key not in seen:
                    output.append(text)
                    seen.add(key)
            return output
        text = str(value).strip()
        return [text] if text else []

    def rewrite(
        self,
        edit_text: str,
        source_text: str | None = None,
        image_path: str | Path | None = None,
    ) -> dict[str, Any]:
        edit_text = str(edit_text).strip()
        if not edit_text:
            raise ValueError("edit_text must not be empty.")

        payload = self._build_payload(
            self._build_messages(edit_text, source_text, image_path)
        )
        headers = {"Content-Type": "application/json"}
        api_key = self.cfg.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            with httpx.Client(
                timeout=float(self.cfg.get("timeout_seconds", 30.0)),
                verify=bool(self.cfg.get("verify_tls", True)),
            ) as client:
                response = client.post(self._url(), headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"VLM request failed with HTTP {exc.response.status_code}: "
                f"{exc.response.text[:1000]}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"VLM request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(
                "VLM endpoint returned a non-JSON HTTP response."
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("VLM endpoint returned no choices.")

        choice = choices[0]
        message = choice.get("message") or {}
        content_text = self._content_to_text(message.get("content")).strip()
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "VLM output was truncated because max_tokens was reached. "
                f"Partial content: {content_text[:1000]!r}"
            )
        if not content_text:
            reasoning_text = self._content_to_text(
                message.get("reasoning_content")
            ).strip()
            if reasoning_text:
                raise RuntimeError(
                    "VLM returned empty content and only reasoning_content. "
                    "Check enable_thinking=false and /no_think."
                )
            raise RuntimeError(
                "VLM returned empty content "
                f"(finish_reason={choice.get('finish_reason')!r})."
            )

        parsed = safe_json_extract(content_text)
        if not isinstance(parsed, dict) or not parsed:
            logger.warning(
                "Could not parse VLM content as JSON: %r", content_text[:1000]
            )
            raise RuntimeError(
                "VLM response did not contain a valid JSON object."
            )

        target = str(parsed.get("target_description", "")).strip()
        if not target:
            raise RuntimeError("VLM JSON is missing target_description.")

        result = dict(parsed)
        result["source_description"] = str(
            parsed.get("source_description", "")
        ).strip()
        result["target_description"] = target
        for key in ("preserve", "change", "remove_objects", "negative"):
            result[key] = self._normalize_string_list(parsed.get(key))

        operation = str(parsed.get("operation", "modify")).strip().lower()
        if operation not in {"modify", "remove"}:
            operation = "remove" if result["remove_objects"] else "modify"
        if result["remove_objects"]:
            operation = "remove"
        result["operation"] = operation

        logger.info(
            "VLM rewrite succeeded | operation=%s | target=%r | "
            "remove_objects=%s",
            operation,
            result["target_description"],
            result["remove_objects"],
        )
        return result

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import image_to_data_url, safe_json_extract


DEFAULT_SYSTEM_PROMPT = (
    "Inspect the reference image when one is supplied. Return only one valid JSON "
    "object using exactly this schema: "
    '{"source_description":"string","operation":"modify|remove",'
    '"target_description":"string","preserve":["string"],'
    '"change":["string"],"remove_objects":["string"],'
    '"negative":["string"]}. '
    "Keep source_description and target_description concise: one sentence each, "
    "normally no more than 30 words. Use at most 5 preserve items, 4 change items, "
    "4 remove_objects, and 6 negative items. Describe visible scene content. Ignore "
    "logos, OCR, banners, subtitles, ticker text, timestamps, filenames, and unrelated "
    "ASR unless the edit explicitly concerns them. For object removal, set operation "
    "to remove, put only removed visual object names in remove_objects, and describe "
    "the desired target scene while preserving unrelated content. All list fields must "
    "be arrays. Do not return explanations or Markdown."
)


def build_messages(
    profile: dict[str, Any],
    edit_text: str,
    source_text: str | None,
    image_path: str | Path | None,
) -> list[dict[str, Any]]:
    system_prompt = str(profile.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
    force_no_think = bool(profile.get("force_no_think_prompt", False))
    if force_no_think and "/no_think" not in system_prompt.lower():
        system_prompt += " /no_think"

    has_image = bool(image_path and Path(image_path).is_file())
    user_text = (
        "Create a concise source description and target-image description for image "
        "retrieval.\n"
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
        user_text += "No reference image is attached; infer only from supplied text.\n"
    user_text += "Return JSON only."
    if force_no_think:
        user_text += "\n/no_think"

    send_image = bool(profile.get("send_reference_image", True))
    user_content: str | list[dict[str, Any]]
    if send_image and has_image:
        image_url: dict[str, Any] = {"url": image_to_data_url(image_path)}
        image_detail = profile.get("image_detail")
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


def content_to_text(content: Any) -> str:
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


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list):
        values = [values]

    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = " ".join(str(item).split()).strip()
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def normalize_vlm_output(content_text: str) -> dict[str, Any]:
    parsed = safe_json_extract(content_text)
    if not isinstance(parsed, dict) or not parsed:
        raise RuntimeError("VLM response did not contain a valid JSON object.")

    target = str(parsed.get("target_description", "")).strip()
    if not target:
        raise RuntimeError("VLM JSON is missing target_description.")

    result = dict(parsed)
    result["source_description"] = str(
        parsed.get("source_description", "")
    ).strip()
    result["target_description"] = target
    for key in ("preserve", "change", "remove_objects", "negative"):
        result[key] = normalize_string_list(parsed.get(key))

    operation = str(parsed.get("operation", "modify")).strip().lower()
    if operation not in {"modify", "remove"}:
        operation = "remove" if result["remove_objects"] else "modify"
    if result["remove_objects"]:
        operation = "remove"
    result["operation"] = operation
    return result

from __future__ import annotations

import re

_SPLIT_PATTERN = re.compile(r"[,;|\n\r]+")


def split_text_concepts(value: str | None) -> list[str]:
    """Split a user field into unique, non-empty concepts while preserving order."""

    output: list[str] = []
    seen: set[str] = set()
    for item in _SPLIT_PATTERN.split(str(value or "")):
        text = " ".join(item.split()).strip(" \t\r\n:,-.;!?")
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output

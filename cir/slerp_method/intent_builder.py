from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SlerpIntent:
    """Deterministically built textual intent for training-free SLERP."""

    text: str
    edit_text: str
    remove_objects: list[str]


def _clean_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip(" \t\r\n:,-.;!?")


def split_remove_objects(value: str | None) -> list[str]:
    pieces = re.split(r"[,;|\n\r]+", str(value or ""))
    output: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        text = _clean_text(piece)
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def build_slerp_intent(edit_text: str | None, remove_text: str | None) -> SlerpIntent:
    """Build one full textual intent without using an LLM.

    The paper interpolates the reference image embedding with one textual-intent
    embedding. The existing two-field UI is retained, so this helper converts the
    fields into one deterministic phrase:

    - Add + Remove: ``<add> without <remove>``
    - Add only: ``<add>``
    - Remove only: ``the same scene without <remove>``
    """

    edit = _clean_text(edit_text)
    remove_objects = split_remove_objects(remove_text)
    remove_phrase = " and ".join(remove_objects)

    if edit and remove_phrase:
        intent = f"{edit} without {remove_phrase}"
    elif edit:
        intent = edit
    elif remove_phrase:
        intent = f"the same scene without {remove_phrase}"
    else:
        raise ValueError("SLERP requires Edit/Add or Remove text.")

    return SlerpIntent(
        text=intent,
        edit_text=edit,
        remove_objects=remove_objects,
    )

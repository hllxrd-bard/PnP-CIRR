from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from .config import AppConfig
from .utils import l2_normalize, normalize_rows


@dataclass(frozen=True)
class NamedQuery:
    name: str
    vector: np.ndarray
    strength: float | None = None


@dataclass(frozen=True)
class ComposedQuery:
    edit_text: str
    edit_vector: np.ndarray | None
    direction_vector: np.ndarray
    named_queries: list[NamedQuery]
    operation: str
    remove_texts: list[str]
    expanded_remove_texts: list[str]
    selected_strength: float


class QueryComposer:
    """Build explicit add/remove composed queries without target-text rewriting.

    The default query is:

        q = normalize(reference + strength * normalize(add - remove))

    where ``add`` is the optional Edit/Add text embedding and ``remove`` is the
    mean embedding of the explicit Remove concepts.  The reference image itself
    is the implicit keep signal, so users do not need to provide a Keep field.
    """

    _REMOVE_PATTERNS = (
        re.compile(
            r"\b(?:remove|delete|erase|take\s+off)\s+"
            r"(?:the\s+|a\s+|an\s+)?"
            r"(?P<object>.+?)"
            r"(?:\s+from\b|\s+off\b|\s+on\b|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:without|no\s+longer\s+(?:wearing|holding|carrying|having)?)\s+"
            r"(?:the\s+|a\s+|an\s+)?(?P<object>.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:bỏ|xóa|xoá|loại\s+bỏ)\s+"
            r"(?:cái\s+|chiếc\s+|một\s+)?"
            r"(?P<object>.+?)"
            r"(?:\s+khỏi\b|\s+ra\s+khỏi\b|\s+trên\b|\s+ở\b|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:không\s+có|không\s+còn)\s+"
            r"(?:cái\s+|chiếc\s+|một\s+)?(?P<object>.+)$",
            re.IGNORECASE,
        ),
    )

    def __init__(self, config: AppConfig):
        self.config = config
        self.cfg = config.section("composition")
        self.removal_cfg = config.section("object_removal")
        self.epsilon = float(self.cfg.get("epsilon", 1e-8))

    @staticmethod
    def _deduplicate(items: Iterable[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = re.sub(r"\s+", " ", str(item)).strip(" \t\r\n:,-.;!?")
            key = value.casefold()
            if not value or key in seen:
                continue
            seen.add(key)
            output.append(value)
        return output

    @staticmethod
    def _clean_remove_object(value: str) -> str:
        text = str(value).strip(" \t\r\n:,-.;!?")
        text = re.sub(
            r"^(?:remove|delete|erase|take\s+off|without|bỏ|xóa|xoá|loại\s+bỏ)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"^(?:the|a|an|cái|chiếc|một)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.split(
            r"\s+(?:from|off|on|of|khỏi|ra\s+khỏi|trên|ở)\s+",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return re.sub(r"\s+", " ", text).strip(" \t\r\n:,-.;!?")

    def split_remove_field(self, remove_text: str | None) -> list[str]:
        """Split the explicit Remove field by comma, semicolon, pipe, or line."""
        if not remove_text or not self.removal_cfg.get("enabled", True):
            return []
        pieces = re.split(r"[,;|\n\r]+", str(remove_text))
        cleaned = [self._clean_remove_object(piece) for piece in pieces]
        max_items = max(1, int(self.removal_cfg.get("max_remove_objects", 8)))
        return self._deduplicate(cleaned)[:max_items]

    def parse_removal_texts(self, edit_text: str) -> list[str]:
        """Legacy compatibility for old one-box requests such as 'remove the hat'."""
        if not edit_text or not self.removal_cfg.get("enabled", True):
            return []
        candidates: list[str] = []
        for pattern in self._REMOVE_PATTERNS:
            match = pattern.search(edit_text)
            if not match:
                continue
            cleaned = self._clean_remove_object(match.group("object"))
            if cleaned:
                candidates.append(cleaned)
        max_items = max(1, int(self.removal_cfg.get("max_remove_objects", 8)))
        return self._deduplicate(candidates)[:max_items]

    def expand_remove_texts(self, remove_texts: list[str]) -> list[str]:
        """Apply a small configurable alias dictionary for low-effort removal."""
        base = self._deduplicate(remove_texts)
        if not bool(self.removal_cfg.get("expand_aliases", True)):
            return base

        aliases = self.removal_cfg.get("aliases", {}) or {}
        if not isinstance(aliases, dict):
            return base

        expanded: list[str] = list(base)
        for item in base:
            item_key = item.casefold()
            for canonical, values in aliases.items():
                candidates = [str(canonical)]
                if isinstance(values, list):
                    candidates.extend(str(value) for value in values)
                candidate_keys = {candidate.casefold() for candidate in candidates}
                if item_key in candidate_keys:
                    expanded.extend(candidates)
                    break

        max_expanded = max(
            len(base),
            int(self.removal_cfg.get("max_expanded_remove_texts", 12)),
        )
        return self._deduplicate(expanded)[:max_expanded]

    @staticmethod
    def detect_operation(edit_text: str, remove_texts: list[str]) -> str:
        has_edit = bool(str(edit_text).strip())
        has_remove = bool(remove_texts)
        if has_edit and has_remove:
            return "replace"
        if has_remove:
            return "remove"
        return "edit"

    def selected_strength(self, edit_strength: float | None) -> float:
        value = (
            float(edit_strength)
            if edit_strength is not None
            else float(self.cfg.get("default_edit_strength", 0.95))
        )
        if not np.isfinite(value):
            raise ValueError("edit_strength must be a finite number.")
        return max(-3.0, min(5.0, value))

    def compose(
        self,
        reference_image_vector: np.ndarray,
        edit_text: str,
        edit_vector: np.ndarray | None,
        remove_texts: list[str],
        expanded_remove_texts: list[str],
        removal_vectors: np.ndarray | None,
        edit_strength: float | None = None,
    ) -> ComposedQuery:
        reference = l2_normalize(reference_image_vector, self.epsilon)

        normalized_edit_vector: np.ndarray | None = None
        if edit_vector is not None and np.linalg.norm(edit_vector) > self.epsilon:
            normalized_edit_vector = l2_normalize(edit_vector, self.epsilon)

        removal_center: np.ndarray | None = None
        if removal_vectors is not None and len(removal_vectors) > 0:
            matrix = normalize_rows(
                np.asarray(removal_vectors, dtype=np.float32),
                self.epsilon,
            )
            removal_center = l2_normalize(matrix.mean(axis=0), self.epsilon)

        add_weight = max(0.0, float(self.cfg.get("explicit_add_weight", 1.0)))
        remove_weight = max(0.0, float(self.cfg.get("explicit_remove_weight", 1.0)))

        delta = np.zeros_like(reference, dtype=np.float32)
        if normalized_edit_vector is not None:
            delta = delta + add_weight * normalized_edit_vector
        if removal_center is not None:
            delta = delta - remove_weight * removal_center

        if np.linalg.norm(delta) <= self.epsilon:
            raise ValueError(
                "The explicit edit produced an empty direction. Provide Edit/Add or Remove text."
            )

        direction = l2_normalize(delta, self.epsilon)
        strength = self.selected_strength(edit_strength)
        operation = self.detect_operation(edit_text, remove_texts)

        named: list[NamedQuery] = []
        if bool(self.cfg.get("use_reference_query", True)):
            named.append(NamedQuery("reference", reference, None))

        if (
            normalized_edit_vector is not None
            and bool(
                self.cfg.get(
                    "use_edit_text_query",
                    self.cfg.get("use_target_text_query", True),
                )
            )
        ):
            named.append(NamedQuery("edit_text", normalized_edit_vector, None))

        if bool(self.cfg.get("use_explicit_query", True)):
            explicit_query = l2_normalize(reference + strength * direction, self.epsilon)
            named.append(
                NamedQuery(
                    f"explicit_{operation}_{strength:.3f}",
                    explicit_query,
                    strength,
                )
            )

        if not named:
            raise ValueError(
                "No query vectors were generated. Enable reference, edit-text, or explicit query."
            )

        return ComposedQuery(
            edit_text=str(edit_text).strip(),
            edit_vector=normalized_edit_vector,
            direction_vector=direction,
            named_queries=named,
            operation=operation,
            remove_texts=list(remove_texts),
            expanded_remove_texts=list(expanded_remove_texts),
            selected_strength=strength,
        )

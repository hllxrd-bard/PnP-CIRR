from __future__ import annotations

import base64
import json
import math
import mimetypes
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np


def as_numpy(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array


def l2_normalize(vector: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= epsilon:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)


def normalize_rows(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, epsilon)
    return matrix / norms


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vector = l2_normalize(vector)
    matrix = normalize_rows(matrix)
    return matrix @ vector


def slerp(start: np.ndarray, end: np.ndarray, alpha: float, epsilon: float = 1e-8) -> np.ndarray:
    start_n = l2_normalize(start, epsilon)
    end_n = l2_normalize(end, epsilon)
    dot = float(np.clip(np.dot(start_n, end_n), -1.0, 1.0))
    if abs(dot) > 0.9995:
        return l2_normalize((1.0 - alpha) * start_n + alpha * end_n, epsilon)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if abs(sin_theta) <= epsilon:
        return start_n
    result = (
        math.sin((1.0 - alpha) * theta) / sin_theta * start_n
        + math.sin(alpha * theta) / sin_theta * end_n
    )
    return l2_normalize(result, epsilon)


def nested_get(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_nonempty_text(entity: dict[str, Any], paths: Iterable[str]) -> str | None:
    parts: list[str] = []
    seen: set[str] = set()
    for path in paths:
        value = nested_get(entity, path)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            text = " ".join(str(item) for item in value if item is not None)
        elif isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return " | ".join(parts) if parts else None


def safe_json_extract(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def image_to_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_frame_path(
    frames_root: str,
    path_template: str,
    video_name: Any,
    frame_name: Any,
) -> Path | None:
    if video_name is None or frame_name is None:
        return None
    rendered = path_template.format(
        frames_root=frames_root,
        video_name=str(video_name),
        frame_name=str(frame_name),
    )
    return Path(rendered).expanduser().resolve()


def is_path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_image_url(video_name: Any, frame_name: Any) -> str | None:
    if video_name is None or frame_name is None:
        return None
    return f"/api/frame?video_name={quote(str(video_name))}&frame_name={quote(str(frame_name))}"


def quote_milvus_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def coerce_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

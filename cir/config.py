from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the YAML configuration is invalid."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "device": "cuda",
        "dtype": "float16",
        "hf_home": None,
        "offline": False,
        "seed": 42,
        "log_level": "INFO",
        "warmup_text": "a person standing in a room",
    },
    "model": {
        "name_or_path": "google/siglip2-large-patch16-512",
        "trust_remote_code": False,
        "local_files_only": False,
        "max_text_length": 64,
        "normalize_embeddings": True,
    },
    "milvus": {
        "uri": "http://192.168.20.150:6050",
        "token": None,
        "database": "default",
        "collection": "multimodal_index_siglip_large_full",
        "load_collection": True,
        "consistency_level": "Bounded",
        "search": {
            "metric_type": "COSINE",
            "params": {"ef": 128},
        },
        "fields": {
            "id": "id",
            "image_vector": "image_embedding",
            "text_vector": "text_embedding",
            "video_name": "video_name",
            "frame_name": "frame_name",
            "timestamp": "timestamp",
            "frame_id": "frame_id",
            "frame_specify": "frame_specify",
            "cluster_id": "cluster_id",
            "metadata": "metadata",
        },
        "raw_text_paths": [
            "metadata.caption",
            "metadata.text",
            "metadata.asr",
            "metadata.ocr",
            "frame_specify",
        ],
    },
    "frames": {
        "root": "/workingspace_aiclub/WorkingSpace/Personal/chinhnm/AIC2026/frames",
        "path_template": "{frames_root}/{video_name}/{frame_name}",
        "allowed_extensions": [".jpg", ".jpeg", ".png", ".webp"],
    },
    "composition": {
        "prompt_template": "{edit_text}",
        "default_edit_strength": 0.95,
        "explicit_add_weight": 1.0,
        "explicit_remove_weight": 1.0,
        "use_reference_query": True,
        "use_edit_text_query": True,
        "use_explicit_query": True,
        # Legacy keys are retained so older YAML files still merge cleanly.
        "use_target_text_query": True,
        "use_directional_queries": False,
        "use_geodesic_queries": False,
        "epsilon": 1e-8,
    },
    "slerp": {
        "default_mode": "directional",
        "default_alpha": 0.8,
        "candidate_k": 300,
        "max_candidate_pool": 500,
        "epsilon": 1e-8,
    },
    "slerp_remove": {
        "default_add_alpha": 0.4,
        "default_gamma": 0.2,
        "candidate_k": 250,
        "max_candidate_pool": 250,
        "epsilon": 1e-8,
    },
    "retrieval": {
        "candidate_k_per_query": 150,
        "max_candidate_pool": 700,
        "default_top_k": 60,
        "max_top_k": 300,
        "exclude_reference": True,
        "output_vector_fields": False,
        "search_output_fields": [
            "id",
            "video_name",
            "frame_name",
            "timestamp",
            "frame_id",
            "cluster_id",
        ],
        "entity_output_fields": [],
    },
    "reranking": {
        "weights": {
            "composed": 0.25,
            "target": 0.35,
            "reference_keep": 0.10,
            "direction": 0.25,
            "metadata": 0.05,
        },
        "score_normalization": "percentile",
        "edit_gate": {
            "enabled": True,
            "target_weight": 0.55,
            "direction_weight": 0.45,
            "minimum_score": 0.40,
            "penalty_weight": 0.25,
        },
        "negative_penalty_weight": 0.20,
        "negative_markers": [
            "không có",
            "không còn",
            "bỏ",
            "xóa",
            "xoá",
            "without",
            "remove",
            "no longer",
        ],
        "enable_negative_parser": True,
    },
    "object_removal": {
        "enabled": True,
        "removal_penalty_weight": 0.35,
        "max_remove_objects": 8,
        "expand_aliases": True,
        "max_expanded_remove_texts": 12,
        "aliases": {
            "hat": ["cap", "headwear"],
            "mũ": ["nón", "headwear"],
            "helmet": ["protective helmet", "headwear"],
            "car": ["automobile", "vehicle"],
            "xe ô tô": ["ô tô", "car", "vehicle"],
            "glasses": ["eyeglasses", "spectacles"],
            "kính": ["kính mắt", "eyeglasses"],
            "microphone": ["mic"],
        },
    },
    "deduplication": {
        "enabled": True,
        "timestamp_window_seconds": 1.5,
        "max_frames_per_video": 5,
        "max_frames_per_cluster": None,
        "prefer_higher_score": True,
    },
    "vlm": {
        "enabled_by_default": False,
        "base_url": "http://127.0.0.1:8000/v1",
        "chat_completions_path": "/chat/completions",
        "api_key": "EMPTY",
        "model": "Qwen3.5-9B-Q8_0.gguf",
        "timeout_seconds": 20.0,
        "temperature": 0.0,
        "max_tokens": 256,
        "enable_thinking": False,
        "force_no_think_prompt": True,
        "response_format_json": True,
        "send_reference_image": True,
        "image_detail": None,
        "fallback_to_no_vlm": True,
        "verify_tls": True,
        "system_prompt": (
            "Inspect the supplied reference image and return strict JSON with keys "
            "source_description, operation, target_description, preserve, change, "
            "remove_objects, negative. For object removal, put only removed visual "
            "object names in remove_objects and describe a positive replacement state. "
            "Ignore television OCR, ticker text, timestamps, filenames, and unrelated ASR. "
            "Do not return explanations or Markdown. /no_think"
        ),
    },
    "web": {
        "host": "0.0.0.0",
        "port": 8088,
        "workers": 1,
        "page_size": 30,
        "title": "Interactive CIR Viewer",
        "allow_local_reference_path": True,
    },
}


@dataclass(frozen=True)
class AppConfig:
    data: dict[str, Any]
    source_path: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Missing or invalid config section: {name}")
        return value

    def get(self, dotted_path: str, default: Any = None) -> Any:
        current: Any = self.data
        for key in dotted_path.split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def load_config(path: str | Path) -> AppConfig:
    source_path = Path(path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {source_path}")

    with source_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    if not isinstance(user_config, dict):
        raise ConfigError("The root YAML value must be an object/mapping.")

    merged = _expand(_deep_merge(DEFAULT_CONFIG, user_config))

    hf_home = merged["runtime"].get("hf_home")
    if hf_home:
        os.environ["HF_HOME"] = str(hf_home)
    if merged["runtime"].get("offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        merged["model"]["local_files_only"] = True

    return AppConfig(data=merged, source_path=source_path)

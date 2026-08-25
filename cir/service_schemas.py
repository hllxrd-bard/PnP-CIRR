from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CanonicalMode = Literal["directional", "pure_slerp", "slerp_hybrid"]
AcceptedMode = Literal[
    "directional",
    "pure_slerp",
    "slerp_hybrid",
    "slerp",
    "slerp_remove",
]

MODE_ALIASES: dict[str, CanonicalMode] = {
    "directional": "directional",
    "pure_slerp": "pure_slerp",
    "slerp_hybrid": "slerp_hybrid",
    "slerp": "pure_slerp",
    "slerp_remove": "slerp_hybrid",
}

INTERNAL_MODES: dict[CanonicalMode, str] = {
    "directional": "directional",
    "pure_slerp": "slerp",
    "slerp_hybrid": "slerp_remove",
}


class PublicReferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | str | None = None
    video_name: str | None = None
    frame_name: str | None = None
    path: str | None = None

    image_embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_reference(self) -> "PublicReferenceInput":
        by_id = self.id is not None
        by_fields = bool(self.video_name and self.frame_name)
        partial_fields = bool(self.video_name) != bool(self.frame_name)
        by_path = self.path is not None

        if partial_fields:
            raise ValueError("video_name and frame_name must be provided together")
        if sum([by_id, by_fields, by_path]) != 1:
            raise ValueError(
                "reference must contain exactly one locator: id; "
                "video_name + frame_name; or path"
            )

        if self.image_embedding is not None:
            if self.embedding_model is None or not self.embedding_model.strip():
                raise ValueError(
                    "embedding_model is required when image_embedding is provided"
                )
            if self.embedding_dimension is None:
                raise ValueError(
                    "embedding_dimension is required when image_embedding is provided"
                )
        elif self.embedding_model is not None or self.embedding_dimension is not None:
            raise ValueError(
                "embedding_model and embedding_dimension are only valid with image_embedding"
            )
        return self


class PublicSearchOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_k_per_query: int | None = Field(default=None, ge=1, le=5000)
    max_candidate_pool: int | None = Field(default=None, ge=1, le=20000)


class PublicFilterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclude_reference: bool | None = None
    include_video_prefixes: list[str] = Field(default_factory=list)
    exclude_video_prefixes: list[str] = Field(default_factory=list)
    exclude_video_names: list[str] = Field(default_factory=list)


class PublicDeduplicationOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    timestamp_window_seconds: float | None = Field(default=None, ge=0)
    max_frames_per_video: int | None = Field(default=None, ge=1)
    max_frames_per_cluster: int | None = Field(default=None, ge=1)


class PublicCIRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: PublicReferenceInput
    composition_mode: AcceptedMode = "directional"
    edit_text: str = Field(default="", max_length=2000)
    remove_text: str = Field(default="", max_length=2000)

    top_k: int | None = Field(default=None, ge=1, le=1000)
    edit_strength: float | None = Field(default=None, ge=-3.0, le=5.0)
    slerp_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    slerp_hybrid_gamma: float | None = Field(default=None, ge=0.0, le=1.5)

    use_vlm: bool | None = None
    vlm_provider: Literal["qwen", "gemini"] | None = None

    search: PublicSearchOverrides = Field(default_factory=PublicSearchOverrides)
    filters: PublicFilterInput = Field(default_factory=PublicFilterInput)
    deduplication: PublicDeduplicationOverrides = Field(
        default_factory=PublicDeduplicationOverrides
    )

    @model_validator(mode="after")
    def normalize_text(self) -> "PublicCIRRequest":
        self.edit_text = str(self.edit_text or "").strip()
        self.remove_text = str(self.remove_text or "").strip()
        return self

    @property
    def canonical_mode(self) -> CanonicalMode:
        return MODE_ALIASES[self.composition_mode]

    @property
    def uses_deprecated_mode_alias(self) -> bool:
        return self.composition_mode in {"slerp", "slerp_remove"}


class ResolveReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference: PublicReferenceInput


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="allow")
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    request_id: str
    error: ErrorDetail

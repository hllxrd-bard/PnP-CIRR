from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | str | None = None
    video_name: str | None = None
    frame_name: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> "ReferenceInput":
        by_id = self.id is not None
        by_fields = bool(self.video_name and self.frame_name)
        by_path = self.path is not None
        if sum([by_id, by_fields, by_path]) != 1:
            raise ValueError(
                "reference must contain exactly one of: id; video_name + frame_name; or path"
            )
        return self


class SearchOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_k_per_query: int | None = Field(default=None, ge=1, le=5000)
    max_candidate_pool: int | None = Field(default=None, ge=1, le=20000)
    # Retained for backward-compatible JSON parsing. Explicit mode uses one
    # user-selected strength and does not expand this list.
    strengths: list[float] | None = None
    geodesic_alphas: list[float] | None = None


class FilterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    milvus_expression: str | None = None
    exclude_reference: bool | None = None
    include_video_prefixes: list[str] = Field(default_factory=list)
    exclude_video_prefixes: list[str] = Field(default_factory=list)
    exclude_video_names: list[str] = Field(default_factory=list)


class DeduplicationOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    timestamp_window_seconds: float | None = Field(default=None, ge=0)
    max_frames_per_video: int | None = Field(default=None, ge=1)
    max_frames_per_cluster: int | None = Field(default=None, ge=1)


class CIRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: ReferenceInput
    # New explicit UI/API semantics:
    # - edit_text: what should be added or changed to
    # - remove_text: what should be removed
    # At least one must be non-empty.
    edit_text: str = Field(default="", max_length=2000)
    remove_text: str = Field(default="", max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    use_vlm: bool | None = None
    edit_strength: float | None = Field(default=None, ge=-3.0, le=5.0)
    search: SearchOverrides = Field(default_factory=SearchOverrides)
    filters: FilterInput = Field(default_factory=FilterInput)
    deduplication: DeduplicationOverrides = Field(default_factory=DeduplicationOverrides)

    @model_validator(mode="after")
    def validate_explicit_edit(self) -> "CIRRequest":
        self.edit_text = str(self.edit_text or "").strip()
        self.remove_text = str(self.remove_text or "").strip()
        if not self.edit_text and not self.remove_text:
            raise ValueError("At least one of edit_text or remove_text must be non-empty.")
        return self


class ScoreBreakdown(BaseModel):
    composed: float
    # Backward-compatible name. In explicit mode, target means Edit/Add similarity.
    target: float
    reference_keep: float
    direction: float
    metadata: float
    edit_score: float = 0.0
    edit_gate_penalty: float = 0.0
    negative_penalty: float = 0.0
    removal_penalty: float = 0.0


class RawScoreBreakdown(BaseModel):
    composed: float
    # Backward-compatible name. In explicit mode, target means Edit/Add similarity.
    target: float
    reference_keep: float
    direction: float
    metadata: float
    negative_penalty: float = 0.0
    removal_penalty: float = 0.0


class CIRResultItem(BaseModel):
    rank: int
    id: int | str
    video_name: str | None = None
    frame_name: str | None = None
    timestamp: float | int | str | None = None
    frame_id: int | str | None = None
    cluster_id: int | str | None = None
    image_path: str | None = None
    image_url: str | None = None
    score: float
    scores: ScoreBreakdown
    raw_scores: RawScoreBreakdown
    matched_query: str | None = None
    matched_query_strength: float | None = None
    best_composed_query: str | None = None
    best_composed_query_strength: float | None = None
    best_ann_query: str | None = None
    retrieved_by: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class TimingInfo(BaseModel):
    reference_lookup: float = 0.0
    vlm: float = 0.0
    text_encoding: float = 0.0
    milvus_search: float = 0.0
    candidate_fetch: float = 0.0
    reranking: float = 0.0
    deduplication: float = 0.0
    total: float = 0.0


class CIROutput(BaseModel):
    status: Literal["success", "error"]
    request: dict[str, Any]
    reference: dict[str, Any] | None = None
    query: dict[str, Any] | None = None
    timings_ms: TimingInfo
    warnings: list[str] = Field(default_factory=list)
    results: list[CIRResultItem] = Field(default_factory=list)
    error: str | None = None

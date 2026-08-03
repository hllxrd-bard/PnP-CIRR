from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig
from .deduplicator import CandidateDeduplicator
from .encoder import Siglip2Encoder
from .milvus_store import MilvusStore
from .query_composer import QueryComposer
from .reranker import VectorReranker
from .schemas import (
    CIRRequest,
    CIROutput,
    CIRResultItem,
    RawScoreBreakdown,
    ScoreBreakdown,
    TimingInfo,
)
from .utils import (
    as_numpy,
    build_image_url,
    first_nonempty_text,
    quote_milvus_string,
    resolve_frame_path,
)
from .vlm_client import VLMClient

LOGGER = logging.getLogger(__name__)


class CIREngine:
    def __init__(self, config: AppConfig, warmup: bool = False):
        self.config = config
        self.store = MilvusStore(config)
        self.encoder = Siglip2Encoder(config)
        self.composer = QueryComposer(config)
        self.reranker = VectorReranker(config)
        self.deduplicator = CandidateDeduplicator(config)
        self.vlm = VLMClient(config)
        self.fields = config.get("milvus.fields", {})
        if warmup:
            self.encoder.warmup(
                str(config.get("runtime.warmup_text", "a person standing in a room"))
            )

    def _entity_image_path(self, entity: dict[str, Any]) -> Path | None:
        return resolve_frame_path(
            frames_root=str(self.config.get("frames.root")),
            path_template=str(self.config.get("frames.path_template")),
            video_name=entity.get(self.fields["video_name"]),
            frame_name=entity.get(self.fields["frame_name"]),
        )

    def _resolve_reference(
        self,
        request: CIRRequest,
    ) -> tuple[dict[str, Any], np.ndarray, Path | None]:
        reference = request.reference
        image_field = self.fields["image_vector"]

        if reference.id is not None:
            entity = self.store.get_by_id(reference.id, include_vectors=True)
            if entity is None:
                raise LookupError(f"Reference id was not found in Milvus: {reference.id}")
            vector = entity.get(image_field)
            if vector is None:
                raise ValueError(f"Reference entity has no '{image_field}' vector.")
            return entity, as_numpy(vector), self._entity_image_path(entity)

        if reference.video_name and reference.frame_name:
            entity = self.store.get_by_video_frame(
                reference.video_name,
                reference.frame_name,
                include_vectors=True,
            )
            if entity is None:
                raise LookupError(
                    "Reference frame was not found in Milvus: "
                    f"{reference.video_name}/{reference.frame_name}"
                )
            vector = entity.get(image_field)
            if vector is None:
                raise ValueError(f"Reference entity has no '{image_field}' vector.")
            return entity, as_numpy(vector), self._entity_image_path(entity)

        assert reference.path is not None
        path = Path(reference.path).expanduser().resolve()
        vector = self.encoder.encode_images([path])[0]
        entity = {
            self.fields["id"]: f"local:{path}",
            self.fields["video_name"]: None,
            self.fields["frame_name"]: path.name,
            self.fields["timestamp"]: None,
            "local_path": str(path),
        }
        return entity, vector, path

    @staticmethod
    def _literal(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return quote_milvus_string(str(value))

    @staticmethod
    def _deduplicate_texts(items: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = " ".join(str(item).split()).strip(" \t\r\n:,-.;!?")
            key = text.casefold()
            if text and key not in seen:
                output.append(text)
                seen.add(key)
        return output

    def _build_expression(
        self,
        request: CIRRequest,
        reference_entity: dict[str, Any],
    ) -> str | None:
        clauses: list[str] = []
        if request.filters.milvus_expression:
            clauses.append(f"({request.filters.milvus_expression})")

        video_field = self.fields["video_name"]
        if request.filters.include_video_prefixes:
            clauses.append(
                "("
                + " or ".join(
                    f"{video_field} like {quote_milvus_string(prefix + '%')}"
                    for prefix in request.filters.include_video_prefixes
                )
                + ")"
            )
        for prefix in request.filters.exclude_video_prefixes:
            clauses.append(
                f"not ({video_field} like {quote_milvus_string(prefix + '%')})"
            )
        for video_name in request.filters.exclude_video_names:
            clauses.append(f"{video_field} != {quote_milvus_string(video_name)}")

        exclude_reference = self.config.get("retrieval.exclude_reference", True)
        if request.filters.exclude_reference is not None:
            exclude_reference = request.filters.exclude_reference
        reference_id = reference_entity.get(self.fields["id"])
        if (
            exclude_reference
            and reference_id is not None
            and not str(reference_id).startswith("local:")
        ):
            clauses.append(f"{self.fields['id']} != {self._literal(reference_id)}")

        return " and ".join(clauses) if clauses else None

    def _passes_client_filters(
        self,
        entity: dict[str, Any],
        request: CIRRequest,
    ) -> bool:
        video_name = str(entity.get(self.fields["video_name"], ""))
        if request.filters.include_video_prefixes and not any(
            video_name.startswith(prefix)
            for prefix in request.filters.include_video_prefixes
        ):
            return False
        if any(
            video_name.startswith(prefix)
            for prefix in request.filters.exclude_video_prefixes
        ):
            return False
        return video_name not in set(request.filters.exclude_video_names)

    def search(self, request: CIRRequest) -> CIROutput:
        started = time.perf_counter()
        timing = TimingInfo()
        warnings: list[str] = []

        lookup_started = time.perf_counter()
        reference_entity, reference_vector, reference_path = self._resolve_reference(request)
        timing.reference_lookup = (time.perf_counter() - lookup_started) * 1000.0

        raw_source_text = first_nonempty_text(
            reference_entity,
            self.config.get("milvus.raw_text_paths", []),
        )

        # Explicit mode: Edit/Add and Remove are independent user inputs.  The
        # stored reference text_embedding is not used to compose the direction.
        effective_edit_text = str(request.edit_text or "").strip()
        explicit_remove_texts = self.composer.split_remove_field(request.remove_text)

        # Backward compatibility for old one-box requests such as
        # edit_text="remove the hat from the man".
        legacy_remove_texts: list[str] = []
        if not explicit_remove_texts and effective_edit_text:
            legacy_remove_texts = self.composer.parse_removal_texts(effective_edit_text)
            if legacy_remove_texts:
                effective_edit_text = ""

        remove_texts = self._deduplicate_texts(
            explicit_remove_texts + legacy_remove_texts
        )

        use_vlm = (
            bool(self.config.get("vlm.enabled_by_default", False))
            if request.use_vlm is None
            else bool(request.use_vlm)
        )
        vlm_payload: dict[str, Any] | None = None
        vlm_negative: list[str] = []

        # VLM remains an explicitly requested slow assist mode.  It is never
        # called automatically in the default explicit path.
        if use_vlm:
            vlm_started = time.perf_counter()
            try:
                vlm_instruction = request.edit_text.strip()
                if request.remove_text.strip():
                    suffix = f"Remove: {request.remove_text.strip()}"
                    vlm_instruction = (
                        f"{vlm_instruction}\n{suffix}" if vlm_instruction else suffix
                    )
                vlm_payload = self.vlm.rewrite(
                    edit_text=vlm_instruction,
                    source_text=raw_source_text,
                    image_path=reference_path,
                )
                vlm_target = str(vlm_payload.get("target_description", "")).strip()
                vlm_remove = [
                    str(item).strip()
                    for item in vlm_payload.get("remove_objects", [])
                    if str(item).strip()
                ]
                vlm_negative = [
                    str(item).strip()
                    for item in vlm_payload.get("negative", [])
                    if str(item).strip()
                ]
                if vlm_target:
                    effective_edit_text = vlm_target
                remove_texts = self._deduplicate_texts(remove_texts + vlm_remove)
            except Exception as exc:
                if not self.config.get("vlm.fallback_to_no_vlm", True):
                    raise
                warnings.append(f"VLM failed; explicit vector fallback was used: {exc}")
                LOGGER.warning(warnings[-1])
            finally:
                timing.vlm = (time.perf_counter() - vlm_started) * 1000.0

        if not effective_edit_text and not remove_texts:
            raise ValueError("Provide at least one Edit/Add or Remove concept.")

        expanded_remove_texts = self.composer.expand_remove_texts(remove_texts)
        negative_texts = self._deduplicate_texts(vlm_negative)
        removal_keys = {text.casefold() for text in expanded_remove_texts}
        negative_texts = [
            text for text in negative_texts if text.casefold() not in removal_keys
        ]

        text_started = time.perf_counter()
        texts_to_encode: list[str] = []

        edit_index: int | None = None
        if effective_edit_text:
            edit_index = len(texts_to_encode)
            texts_to_encode.append(effective_edit_text)

        negative_start = len(texts_to_encode)
        texts_to_encode.extend(negative_texts)
        negative_end = len(texts_to_encode)

        removal_start = len(texts_to_encode)
        texts_to_encode.extend(expanded_remove_texts)
        removal_end = len(texts_to_encode)

        if not texts_to_encode:
            raise ValueError("No text concepts were available for encoding.")

        encoded = self.encoder.encode_texts(texts_to_encode)
        edit_vector = encoded[edit_index] if edit_index is not None else None
        negative_vectors = (
            encoded[negative_start:negative_end]
            if negative_end > negative_start
            else None
        )
        removal_vectors = (
            encoded[removal_start:removal_end]
            if removal_end > removal_start
            else None
        )
        timing.text_encoding = (time.perf_counter() - text_started) * 1000.0

        if edit_vector is not None and edit_vector.size != reference_vector.size:
            raise ValueError(
                "Embedding dimension mismatch: SigLIP2 text encoder returned "
                f"{edit_vector.size}, but Milvus reference image_embedding has "
                f"{reference_vector.size}."
            )
        if removal_vectors is not None and removal_vectors.shape[1] != reference_vector.size:
            raise ValueError(
                "Removal text embedding dimension differs from the reference image embedding."
            )

        composed = self.composer.compose(
            reference_image_vector=reference_vector,
            edit_text=effective_edit_text,
            edit_vector=edit_vector,
            remove_texts=remove_texts,
            expanded_remove_texts=expanded_remove_texts,
            removal_vectors=removal_vectors,
            edit_strength=request.edit_strength,
        )

        candidate_k = int(
            request.search.candidate_k_per_query
            or self.config.get("retrieval.candidate_k_per_query", 150)
        )
        max_pool = int(
            request.search.max_candidate_pool
            or self.config.get("retrieval.max_candidate_pool", 700)
        )
        top_k = min(
            int(request.top_k or self.config.get("retrieval.default_top_k", 60)),
            int(self.config.get("retrieval.max_top_k", 300)),
        )

        expression = self._build_expression(request, reference_entity)
        search_started = time.perf_counter()
        raw_hits = self.store.search(
            query_vectors=[item.vector.tolist() for item in composed.named_queries],
            limit=candidate_k,
            expression=expression,
        )
        timing.milvus_search = (time.perf_counter() - search_started) * 1000.0

        metric_type = str(
            self.config.get("milvus.search.metric_type", "COSINE")
        ).upper()
        higher_is_better = metric_type not in {"L2", "EUCLIDEAN"}
        best_ann_score: dict[Any, float] = {}
        best_ann_query: dict[Any, str] = {}
        retrieved_by: dict[Any, list[str]] = {}

        for query_index, query_hits in enumerate(raw_hits):
            query_name = composed.named_queries[query_index].name
            for hit in query_hits:
                entity_id = hit["id"]
                score = float(hit["distance"])
                retrieved_by.setdefault(entity_id, []).append(query_name)
                previous = best_ann_score.get(entity_id)
                if previous is None or (
                    score > previous if higher_is_better else score < previous
                ):
                    best_ann_score[entity_id] = score
                    best_ann_query[entity_id] = query_name

        sorted_ids = sorted(
            best_ann_score,
            key=best_ann_score.get,
            reverse=higher_is_better,
        )[:max_pool]

        fetch_started = time.perf_counter()
        candidates = self.store.fetch_entities(sorted_ids, include_vectors=True)
        timing.candidate_fetch = (time.perf_counter() - fetch_started) * 1000.0
        candidates = [
            entity
            for entity in candidates
            if self._passes_client_filters(entity, request)
        ]

        reference_id = reference_entity.get(self.fields["id"])
        exclude_reference = self.config.get("retrieval.exclude_reference", True)
        if request.filters.exclude_reference is not None:
            exclude_reference = request.filters.exclude_reference
        if exclude_reference:
            candidates = [
                entity
                for entity in candidates
                if entity.get(self.fields["id"]) != reference_id
            ]

        rerank_started = time.perf_counter()
        ranked = self.reranker.rank(
            candidates=candidates,
            reference_vector=reference_vector,
            composed=composed,
            negative_vectors=negative_vectors,
            removal_vectors=removal_vectors,
        )
        timing.reranking = (time.perf_counter() - rerank_started) * 1000.0

        dedup_started = time.perf_counter()
        selected = self.deduplicator.apply(
            ranked=ranked,
            overrides=request.deduplication,
            top_k=top_k,
        )
        timing.deduplication = (time.perf_counter() - dedup_started) * 1000.0

        results: list[CIRResultItem] = []
        metadata_field = self.fields.get("metadata")
        for rank, candidate in enumerate(selected, start=1):
            entity = candidate.entity
            entity_id = entity.get(self.fields["id"])
            image_path = self._entity_image_path(entity)
            video_name = entity.get(self.fields["video_name"])
            frame_name = entity.get(self.fields["frame_name"])
            results.append(
                CIRResultItem(
                    rank=rank,
                    id=entity_id,
                    video_name=video_name,
                    frame_name=frame_name,
                    timestamp=entity.get(self.fields["timestamp"]),
                    frame_id=entity.get(self.fields.get("frame_id")),
                    cluster_id=entity.get(self.fields.get("cluster_id")),
                    image_path=str(image_path) if image_path is not None else None,
                    image_url=build_image_url(video_name, frame_name),
                    score=candidate.score,
                    scores=ScoreBreakdown(**candidate.component_scores),
                    raw_scores=RawScoreBreakdown(**candidate.raw_component_scores),
                    matched_query=candidate.matched_query,
                    matched_query_strength=candidate.matched_query_strength,
                    best_composed_query=candidate.matched_query,
                    best_composed_query_strength=candidate.matched_query_strength,
                    best_ann_query=best_ann_query.get(entity_id),
                    retrieved_by=list(
                        dict.fromkeys(retrieved_by.get(entity_id, []))
                    ),
                    metadata=(
                        entity.get(metadata_field)
                        if metadata_field
                        and isinstance(entity.get(metadata_field), dict)
                        else None
                    ),
                )
            )

        timing.total = (time.perf_counter() - started) * 1000.0
        return CIROutput(
            status="success",
            request=request.model_dump(mode="json"),
            reference={
                "id": reference_id,
                "video_name": reference_entity.get(self.fields["video_name"]),
                "frame_name": reference_entity.get(self.fields["frame_name"]),
                "timestamp": reference_entity.get(self.fields["timestamp"]),
                "image_path": str(reference_path) if reference_path is not None else None,
                "image_url": build_image_url(
                    reference_entity.get(self.fields["video_name"]),
                    reference_entity.get(self.fields["frame_name"]),
                ),
                "source_text": raw_source_text,
                "source_text_used_for_composition": False,
            },
            query={
                "original_edit_text": request.edit_text,
                "original_remove_text": request.remove_text,
                "edit_text": effective_edit_text or None,
                # Retained only for clients that still expect the old key.
                "target_text": None,
                "operation": composed.operation,
                "selected_strength": composed.selected_strength,
                "used_vlm": bool(vlm_payload),
                "vlm_output": vlm_payload,
                "remove_objects": remove_texts,
                "expanded_remove_objects": expanded_remove_texts,
                "negative_texts": negative_texts,
                "query_vectors": [
                    {"name": item.name, "strength": item.strength}
                    for item in composed.named_queries
                ],
                "candidate_pool_size": len(candidates),
                "milvus_expression": expression,
                "composition": {
                    "formula": "normalize(reference + strength * normalize(edit_add - remove))",
                    "uses_target_text_rewrite": False,
                    "uses_reference_text_embedding": False,
                },
                "reranking": {
                    "score_normalization": self.config.get(
                        "reranking.score_normalization", "percentile"
                    ),
                    "weights": self.config.get("reranking.weights", {}),
                    "edit_gate": self.config.get("reranking.edit_gate", {}),
                    "object_removal": self.config.get("object_removal", {}),
                    "score_semantics": {
                        "target": "similarity to Edit/Add text; zero in remove-only mode",
                        "removal_penalty": "similarity to explicit Remove concepts",
                    },
                    "composed_query_policy": "single_user_selected_strength",
                },
            },
            timings_ms=timing,
            warnings=warnings,
            results=results,
        )

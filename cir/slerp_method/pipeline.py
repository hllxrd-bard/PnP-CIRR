from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..schemas import (
    CIRRequest,
    CIROutput,
    CIRResultItem,
    RawScoreBreakdown,
    ScoreBreakdown,
    TimingInfo,
)
from ..utils import build_image_url, first_nonempty_text
from .composer import spherical_linear_interpolation
from .scorer import rank_candidates_by_slerp_cosine

if TYPE_CHECKING:
    from ..engine import CIREngine


def search_slerp(engine: "CIREngine", request: CIRRequest) -> CIROutput:
    """Run the pure training-free SLERP retrieval path.

    This path is deliberately isolated from the existing directional composer,
    vector reranker, removal penalty, edit gate, and VLM.
    """

    started = time.perf_counter()
    timing = TimingInfo()
    warnings: list[str] = []

    lookup_started = time.perf_counter()
    reference_entity, reference_vector, reference_path = engine._resolve_reference(request)
    timing.reference_lookup = (time.perf_counter() - lookup_started) * 1000.0

    intent_text = str(request.edit_text or "").strip()
    if not intent_text:
        raise ValueError("Pure SLERP requires a non-empty textual intent in Edit/Add.")
    if str(request.remove_text or "").strip():
        raise ValueError("Pure SLERP does not accept Remove; use slerp_remove mode.")
    alpha = (
        float(request.slerp_alpha)
        if request.slerp_alpha is not None
        else float(engine.config.get("slerp.default_alpha", 0.8))
    )
    if request.use_vlm:
        warnings.append("SLERP mode ignores use_vlm=true and remains training-free.")

    text_started = time.perf_counter()
    intent_vector = engine.encoder.encode_texts([intent_text])[0]
    timing.text_encoding = (time.perf_counter() - text_started) * 1000.0

    if intent_vector.size != reference_vector.size:
        raise ValueError(
            "Embedding dimension mismatch: SLERP text embedding has "
            f"{intent_vector.size} values, but the reference image embedding has "
            f"{reference_vector.size}."
        )

    epsilon = float(engine.config.get("slerp.epsilon", 1e-8))
    composed_vector = spherical_linear_interpolation(
        reference_vector,
        intent_vector,
        alpha=alpha,
        epsilon=epsilon,
    )
    query_name = f"slerp_{alpha:.3f}"

    requested_candidate_k = int(
        request.search.candidate_k_per_query
        or engine.config.get("slerp.candidate_k", 300)
    )
    hnsw_ef = engine.config.get("milvus.search.params.ef")
    candidate_k = requested_candidate_k
    if hnsw_ef is not None:
        try:
            safe_k = max(1, int(hnsw_ef) - 1)
        except (TypeError, ValueError):
            safe_k = requested_candidate_k
        if candidate_k > safe_k:
            warnings.append(
                f"Pure SLERP candidate_k was clamped from {candidate_k} to {safe_k} "
                f"because Milvus HNSW ef={hnsw_ef}."
            )
            candidate_k = safe_k
    max_pool = int(
        request.search.max_candidate_pool
        or engine.config.get("slerp.max_candidate_pool", candidate_k)
    )
    top_k = min(
        int(request.top_k or engine.config.get("retrieval.default_top_k", 60)),
        int(engine.config.get("retrieval.max_top_k", 300)),
    )

    expression = engine._build_expression(request, reference_entity)
    search_started = time.perf_counter()
    raw_hits = engine.store.search(
        query_vectors=[composed_vector.tolist()],
        limit=candidate_k,
        expression=expression,
    )
    timing.milvus_search = (time.perf_counter() - search_started) * 1000.0

    metric_type = str(engine.config.get("milvus.search.metric_type", "COSINE")).upper()
    higher_is_better = metric_type not in {"L2", "EUCLIDEAN"}
    ann_scores: dict[Any, float] = {}
    for hit in raw_hits[0] if raw_hits else []:
        ann_scores[hit["id"]] = float(hit["distance"])

    sorted_ids = sorted(
        ann_scores,
        key=ann_scores.get,
        reverse=higher_is_better,
    )[:max_pool]

    fetch_started = time.perf_counter()
    candidates = engine.store.fetch_entities(sorted_ids, include_vectors=True)
    timing.candidate_fetch = (time.perf_counter() - fetch_started) * 1000.0
    candidates = [
        entity
        for entity in candidates
        if engine._passes_client_filters(entity, request)
    ]

    reference_id = reference_entity.get(engine.fields["id"])
    exclude_reference = engine.config.get("retrieval.exclude_reference", True)
    if request.filters.exclude_reference is not None:
        exclude_reference = request.filters.exclude_reference
    if exclude_reference:
        candidates = [
            entity
            for entity in candidates
            if entity.get(engine.fields["id"]) != reference_id
        ]

    rerank_started = time.perf_counter()
    ranked = rank_candidates_by_slerp_cosine(
        candidates=candidates,
        image_vector_field=engine.fields["image_vector"],
        composed_vector=composed_vector,
        query_name=query_name,
        alpha=alpha,
    )
    timing.reranking = (time.perf_counter() - rerank_started) * 1000.0

    dedup_started = time.perf_counter()
    selected = engine.deduplicator.apply(
        ranked=ranked,
        overrides=request.deduplication,
        top_k=top_k,
    )
    timing.deduplication = (time.perf_counter() - dedup_started) * 1000.0

    results: list[CIRResultItem] = []
    metadata_field = engine.fields.get("metadata")
    for rank, candidate in enumerate(selected, start=1):
        entity = candidate.entity
        entity_id = entity.get(engine.fields["id"])
        image_path = engine._entity_image_path(entity)
        video_name = entity.get(engine.fields["video_name"])
        frame_name = entity.get(engine.fields["frame_name"])
        results.append(
            CIRResultItem(
                rank=rank,
                id=entity_id,
                video_name=video_name,
                frame_name=frame_name,
                timestamp=entity.get(engine.fields["timestamp"]),
                frame_id=entity.get(engine.fields.get("frame_id")),
                cluster_id=entity.get(engine.fields.get("cluster_id")),
                image_path=str(image_path) if image_path is not None else None,
                image_url=build_image_url(video_name, frame_name),
                score=candidate.score,
                scores=ScoreBreakdown(**candidate.component_scores),
                raw_scores=RawScoreBreakdown(**candidate.raw_component_scores),
                matched_query=query_name,
                matched_query_strength=alpha,
                best_composed_query=query_name,
                best_composed_query_strength=alpha,
                best_ann_query=query_name,
                retrieved_by=[query_name],
                metadata=(
                    entity.get(metadata_field)
                    if metadata_field
                    and isinstance(entity.get(metadata_field), dict)
                    else None
                ),
            )
        )

    raw_source_text = first_nonempty_text(
        reference_entity,
        engine.config.get("milvus.raw_text_paths", []),
    )
    timing.total = (time.perf_counter() - started) * 1000.0
    return CIROutput(
        status="success",
        request=request.model_dump(mode="json"),
        reference={
            "id": reference_id,
            "video_name": reference_entity.get(engine.fields["video_name"]),
            "frame_name": reference_entity.get(engine.fields["frame_name"]),
            "timestamp": reference_entity.get(engine.fields["timestamp"]),
            "image_path": str(reference_path) if reference_path is not None else None,
            "image_url": build_image_url(
                reference_entity.get(engine.fields["video_name"]),
                reference_entity.get(engine.fields["frame_name"]),
            ),
            "source_text": raw_source_text,
            "source_text_used_for_composition": False,
        },
        query={
            "composition_mode": "slerp",
            "original_edit_text": request.edit_text,
            "original_remove_text": request.remove_text,
            "intent_text": intent_text,
            "edit_text": intent_text,
            "target_text": intent_text,
            "operation": "slerp",
            "selected_strength": None,
            "slerp_alpha": alpha,
            "used_vlm": False,
            "vlm_output": None,
            "remove_objects": [],
            "expanded_remove_objects": [],
            "negative_texts": [],
            "query_vectors": [{"name": query_name, "strength": alpha}],
            "candidate_pool_size": len(candidates),
            "milvus_expression": expression,
            "composition": {
                "formula": "slerp(reference_image, textual_intent, alpha)",
                "uses_target_text_rewrite": False,
                "uses_reference_text_embedding": False,
                "training_free": True,
                "single_milvus_query": True,
            },
            "reranking": {
                "method": "exact_local_cosine",
                "score_semantics": {
                    "composed": "exact cosine(candidate_image, slerp_query)",
                    "other_components": "zero; directional heuristics are not used",
                },
            },
        },
        timings_ms=timing,
        warnings=warnings,
        results=results,
    )

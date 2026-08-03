from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..query_composer import ComposedQuery, NamedQuery
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
from .remove_composer import l2_normalize, normalized_mean, spherical_move_away

if TYPE_CHECKING:
    from ..engine import CIREngine


def _candidate_k_with_hnsw_guard(
    engine: "CIREngine",
    request: CIRRequest,
    default_k: int,
    warnings: list[str],
) -> int:
    """Use the directional retrieval candidate count without violating HNSW ef."""

    requested = int(request.search.candidate_k_per_query or default_k)
    ef = engine.config.get("milvus.search.params.ef")
    if ef is None:
        return requested

    try:
        ef_value = int(ef)
    except (TypeError, ValueError):
        return requested

    # The deployed Milvus build requires ef > k, not merely ef >= k.
    safe_k = max(1, ef_value - 1)
    if requested > safe_k:
        warnings.append(
            f"SLERP Remove candidate_k_per_query was clamped from {requested} "
            f"to {safe_k} because Milvus HNSW ef={ef_value}."
        )
        return safe_k
    return requested


def _spherical_direction(
    reference_vector: np.ndarray,
    composed_vector: np.ndarray,
    negative_vector: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Return the candidate-motion direction expected by VectorReranker.

    Normally this is the normalized chord from the original reference to the
    spherical composed query.  At the degenerate gamma=0/alpha=0 point, use the
    negative concept's opposite direction so the reranker remains well-defined.
    """

    reference = l2_normalize(reference_vector, epsilon)
    composed = l2_normalize(composed_vector, epsilon)
    chord = composed - reference
    if float(np.linalg.norm(chord)) > epsilon:
        return l2_normalize(chord, epsilon)
    return l2_normalize(-negative_vector, epsilon)


def search_slerp_remove(engine: "CIREngine", request: CIRRequest) -> CIROutput:
    """Run spherical composition with the full directional retrieval/reranker.

    Geometry is the only intentionally changed component:

    * remove-only: spherical_move_away(reference, remove, gamma)
    * add+remove: spherical_move_away(slerp(reference, add, alpha), remove, gamma)

    Candidate generation, reference safety-net retrieval, Edit/Add retrieval,
    vector reranking, edit gate, explicit removal penalty and deduplication are
    shared with the existing directional pipeline.
    """

    started = time.perf_counter()
    timing = TimingInfo()
    warnings: list[str] = []

    lookup_started = time.perf_counter()
    reference_entity, reference_vector, reference_path = engine._resolve_reference(
        request
    )
    timing.reference_lookup = (time.perf_counter() - lookup_started) * 1000.0

    edit_text = str(request.edit_text or "").strip()

    # Reuse exactly the directional pipeline's parsing and alias expansion.
    remove_objects = engine.composer.split_remove_field(request.remove_text)
    if not remove_objects:
        raise ValueError("SLERP Remove requires at least one Remove concept.")
    expanded_remove_objects = engine.composer.expand_remove_texts(remove_objects)

    if request.use_vlm:
        warnings.append(
            "SLERP Remove ignores use_vlm=true; spherical composition remains "
            "training-free while reusing the directional vector reranker."
        )

    alpha = (
        float(request.slerp_alpha)
        if request.slerp_alpha is not None
        else float(engine.config.get("slerp_remove.default_add_alpha", 0.4))
    )
    gamma = (
        float(request.slerp_remove_gamma)
        if request.slerp_remove_gamma is not None
        else float(engine.config.get("slerp_remove.default_gamma", 0.2))
    )
    epsilon = float(engine.config.get("slerp_remove.epsilon", 1e-8))

    texts_to_encode = ([edit_text] if edit_text else []) + expanded_remove_objects
    text_started = time.perf_counter()
    encoded = np.asarray(
        engine.encoder.encode_texts(texts_to_encode),
        dtype=np.float32,
    )
    timing.text_encoding = (time.perf_counter() - text_started) * 1000.0

    if encoded.ndim != 2 or encoded.shape[0] != len(texts_to_encode):
        raise ValueError("Text encoder returned an unexpected embedding matrix shape.")

    reference = l2_normalize(reference_vector, epsilon)
    offset = 0
    normalized_edit_vector: np.ndarray | None = None

    if edit_text:
        add_vector = encoded[0]
        offset = 1
        if add_vector.size != reference.size:
            raise ValueError(
                "Embedding dimension mismatch between Edit/Add text and "
                "the reference image."
            )
        normalized_edit_vector = l2_normalize(add_vector, epsilon)
        anchor_vector = spherical_linear_interpolation(
            reference,
            normalized_edit_vector,
            alpha=alpha,
            epsilon=epsilon,
        )
    else:
        anchor_vector = reference

    removal_vectors = encoded[offset:]
    if (
        removal_vectors.ndim != 2
        or removal_vectors.shape[0] == 0
        or removal_vectors.shape[1] != reference.size
    ):
        raise ValueError(
            "Embedding dimension mismatch between Remove text and the "
            "reference image."
        )

    negative_center = normalized_mean(removal_vectors, epsilon)
    composed_vector = spherical_move_away(
        anchor_vector,
        negative_center,
        gamma=gamma,
        epsilon=epsilon,
    )
    direction_vector = _spherical_direction(
        reference,
        composed_vector,
        negative_center,
        epsilon,
    )

    operation = "replace" if edit_text else "remove"
    explicit_query_name = (
        f"explicit_spherical_replace_a{alpha:.3f}_g{gamma:.3f}"
        if edit_text
        else f"explicit_spherical_remove_g{gamma:.3f}"
    )

    # Reuse the same multiprobe policy as the directional composer.  The
    # explicit spherical query is always present because it defines this mode.
    named_queries: list[NamedQuery] = []
    if bool(engine.config.get("composition.use_reference_query", True)):
        named_queries.append(NamedQuery("reference", reference, None))
    if edit_text and bool(
        engine.config.get(
            "composition.use_edit_text_query",
            engine.config.get("composition.use_target_text_query", True),
        )
    ):
        assert normalized_edit_vector is not None
        named_queries.append(NamedQuery("edit_text", normalized_edit_vector, None))
    named_queries.append(
        NamedQuery(explicit_query_name, composed_vector, gamma)
    )

    composed = ComposedQuery(
        edit_text=edit_text,
        edit_vector=normalized_edit_vector,
        direction_vector=direction_vector,
        named_queries=named_queries,
        operation=operation,
        remove_texts=list(remove_objects),
        expanded_remove_texts=list(expanded_remove_objects),
        selected_strength=gamma,
    )

    default_candidate_k = int(
        engine.config.get("retrieval.candidate_k_per_query", 150)
    )
    candidate_k = _candidate_k_with_hnsw_guard(
        engine,
        request,
        default_candidate_k,
        warnings,
    )
    max_pool = int(
        request.search.max_candidate_pool
        or engine.config.get("retrieval.max_candidate_pool", 700)
    )
    top_k = min(
        int(request.top_k or engine.config.get("retrieval.default_top_k", 60)),
        int(engine.config.get("retrieval.max_top_k", 300)),
    )

    expression = engine._build_expression(request, reference_entity)
    search_started = time.perf_counter()
    raw_hits = engine.store.search(
        query_vectors=[item.vector.tolist() for item in composed.named_queries],
        limit=candidate_k,
        expression=expression,
    )
    timing.milvus_search = (time.perf_counter() - search_started) * 1000.0

    metric_type = str(
        engine.config.get("milvus.search.metric_type", "COSINE")
    ).upper()
    higher_is_better = metric_type not in {"L2", "EUCLIDEAN"}
    best_ann_score: dict[Any, float] = {}
    best_ann_query: dict[Any, str] = {}
    retrieved_by: dict[Any, list[str]] = {}

    for query_index, query_hits in enumerate(raw_hits):
        if query_index >= len(composed.named_queries):
            break
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

    # This is the material change from the raw spherical baseline: preserve all
    # directional reranking signals, including explicit removal penalty.
    rerank_started = time.perf_counter()
    ranked = engine.reranker.rank(
        candidates=candidates,
        reference_vector=reference,
        composed=composed,
        negative_vectors=None,
        removal_vectors=removal_vectors,
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

    raw_source_text = first_nonempty_text(
        reference_entity,
        engine.config.get("milvus.raw_text_paths", []),
    )
    timing.total = (time.perf_counter() - started) * 1000.0
    formula = (
        "spherical_move_away(slerp(reference, add, alpha), remove, gamma)"
        if edit_text
        else "spherical_move_away(reference, remove, gamma)"
    )

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
            "composition_mode": "slerp_remove",
            "original_edit_text": request.edit_text,
            "original_remove_text": request.remove_text,
            "intent_text": edit_text or None,
            "edit_text": edit_text or None,
            "target_text": None,
            "operation": operation,
            "selected_strength": gamma,
            "slerp_alpha": alpha if edit_text else None,
            "slerp_remove_gamma": gamma,
            "used_vlm": False,
            "vlm_output": None,
            "remove_objects": remove_objects,
            "expanded_remove_objects": expanded_remove_objects,
            "negative_texts": [],
            "query_vectors": [
                {"name": item.name, "strength": item.strength}
                for item in composed.named_queries
            ],
            "candidate_pool_size": len(candidates),
            "milvus_expression": expression,
            "composition": {
                "formula": formula,
                "geometry": "spherical",
                "uses_target_text_rewrite": False,
                "uses_reference_text_embedding": False,
                "training_free": True,
                "single_milvus_query": False,
                "experimental_extension": True,
            },
            "reranking": {
                "method": "directional_vector_reranker",
                "score_normalization": engine.config.get(
                    "reranking.score_normalization", "percentile"
                ),
                "weights": engine.config.get("reranking.weights", {}),
                "edit_gate": engine.config.get("reranking.edit_gate", {}),
                "object_removal": engine.config.get("object_removal", {}),
                "score_semantics": {
                    "composed": "similarity to explicit spherical query",
                    "target": "similarity to Edit/Add text; zero in remove-only mode",
                    "reference_keep": "same reference preservation as directional",
                    "direction": "candidate motion toward spherical composed query",
                    "removal_penalty": "same explicit Remove penalty as directional",
                },
                "candidate_generation": "reference + optional edit_text + explicit spherical query",
                "composed_query_policy": "single_user_selected_gamma",
            },
        },
        timings_ms=timing,
        warnings=warnings,
        results=results,
    )

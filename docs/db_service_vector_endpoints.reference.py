"""Reference implementation of the two endpoints CIR needs from the DB service.

NOT APPLIED, NOT IMPORTED. This file lives in the CIR repo purely as something
to hand to the owner of src/apps/database.py. Nothing in CIR imports it.

To apply: paste both endpoints into src/apps/database.py, and add the two
request models beside the existing ones (near BaseSearchRequest). No other
change is needed -- everything else already exists in that file.

Why these are needed
--------------------
CIR composes its query vector locally -- normalize(reference + strength *
direction) for directional mode, or a spherical interpolation for SLERP -- and
then reranks candidates with exact cosine over their image embeddings.

The existing endpoints cannot serve that:
  * /v1/search/text  takes a string and encodes server-side
  * /v1/search/image takes an image file and encodes server-side
A composed vector is neither a string nor an image, and no current endpoint
returns embeddings, so the reranking step has nothing to work with.

Both endpoints below are thin wrappers over machinery already present.
SearcherMixin._execute_search already accepts precomputed_vector= and skips
encoding when it is supplied (src/core/database/milvus/searcher.py:81,111-112);
it simply is not reachable over HTTP. These go straight to the collection for
clarity, matching how searcher.py itself calls it.

Verified: CIR was run against a mock implementing exactly this contract, and
all seven composition-mode cases returned results byte-identical to CIR's
direct-to-Milvus backend.
"""

from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request models -- add these next to BaseSearchRequest in database.py
# ---------------------------------------------------------------------------


class VectorSearchRequest(BaseModel):
    """ANN search using a caller-supplied query vector."""

    vectors: List[List[float]] = Field(
        description="One or more query vectors. Batched to match Milvus semantics."
    )
    model_name: Optional[str] = None
    anns_field: str = "image_embedding"
    metric_type: str = "COSINE"
    top_k: int = 150
    expr: Optional[str] = Field(
        default=None, description="Optional Milvus boolean expression."
    )
    output_fields: Optional[List[str]] = None


class EntityFetchRequest(BaseModel):
    """Fetch entities by primary key, or by filter expression.

    CIR needs both forms: `ids` for candidate fetch during reranking, `filter`
    for resolving a reference by video_name + frame_name, which has no id yet.
    """

    model_name: Optional[str] = None
    ids: Optional[List[int]] = None
    filter: Optional[str] = None
    limit: Optional[int] = None
    include_vectors: bool = True
    output_fields: Optional[List[str]] = None


DEFAULT_SCALAR_FIELDS = [
    "id",
    "video_name",
    "frame_name",
    "timestamp",
    "frame_id",
    "cluster_id",
]


def _resolve_collection(mgr, model_name: Optional[str]):
    """Same resolution rule SearcherMixin.search uses."""
    if not model_name and len(mgr.model_names) == 1:
        model_name = mgr.model_names[0]
    if model_name not in mgr.collections:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not initialized. Available: {mgr.model_names}",
        )
    return mgr.collections[model_name]


# ---------------------------------------------------------------------------
# Endpoint 1 -- ANN search by supplied vector
# ---------------------------------------------------------------------------


@app.post("/v1/search/vector")  # noqa: F821  (app is defined in database.py)
async def search_vector(req: VectorSearchRequest) -> Dict[str, Any]:
    """Search using a precomputed query vector rather than text or an image."""
    t0 = time.time()  # noqa: F821
    mgr = get_milvus()  # noqa: F821
    collection = _resolve_collection(mgr, req.model_name)
    output_fields = req.output_fields or DEFAULT_SCALAR_FIELDS

    def execute():
        return collection.search(
            data=req.vectors,
            anns_field=req.anns_field,
            param={
                "metric_type": req.metric_type,
                # ef must be at least the requested limit or HNSW rejects the search.
                "params": {"ef": max(int(req.top_k), 256)},
            },
            limit=int(req.top_k),
            output_fields=output_fields,
            expr=req.expr,
        )

    try:
        raw = await run_blocking_search(execute)  # noqa: F821

        results = []
        for query_hits in raw:
            hits = []
            for hit in query_hits:
                entity = {field: hit.entity.get(field) for field in output_fields}
                entity.setdefault("id", hit.id)
                hits.append(
                    {
                        "id": hit.id,
                        "distance": float(hit.distance),
                        "entity": entity,
                    }
                )
            results.append(hits)

        return {
            "status": "success",
            "results": results,
            "latency_ms": round((time.time() - t0) * 1000, 2),  # noqa: F821
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search error: {str(e)}")


# ---------------------------------------------------------------------------
# Endpoint 2 -- fetch entities, optionally including their embeddings
# ---------------------------------------------------------------------------


@app.post("/v1/entities/fetch")  # noqa: F821
async def fetch_entities(req: EntityFetchRequest) -> Dict[str, Any]:
    """Fetch entities by id or by filter, optionally with their vectors.

    include_vectors is what makes this endpoint distinct from every existing
    one: CIR reranks candidates locally and needs the raw image embeddings.
    """
    t0 = time.time()  # noqa: F821
    mgr = get_milvus()  # noqa: F821
    collection = _resolve_collection(mgr, req.model_name)

    if not req.ids and not req.filter:
        raise HTTPException(
            status_code=400, detail="Provide either 'ids' or 'filter'."
        )

    fields = list(req.output_fields or DEFAULT_SCALAR_FIELDS)
    if req.include_vectors:
        fields += ["image_embedding", "text_embedding"]

    if req.ids:
        expr = f"id in {list(req.ids)}"
        limit = len(req.ids)
    else:
        expr = req.filter
        limit = req.limit or 1

    def execute():
        return collection.query(expr=expr, limit=limit, output_fields=fields)

    try:
        rows = await run_blocking_search(execute)  # noqa: F821
        return {
            "status": "success",
            "entities": [dict(row) for row in rows],
            "latency_ms": round((time.time() - t0) * 1000, 2),  # noqa: F821
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entity fetch error: {str(e)}")

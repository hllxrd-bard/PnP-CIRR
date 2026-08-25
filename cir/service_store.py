from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx

from .config import AppConfig
from .store_base import StoreBackendError
from .utils import quote_milvus_string

SEARCH_ENDPOINT = "/v1/search/vector"
FETCH_ENDPOINT = "/v1/entities/fetch"

_MISSING_ENDPOINTS_HINT = """\
The database microservice at {base_url} does not expose the vector endpoints CIR
needs: {missing}.

CIR composes its query vector locally (reference + edit - remove, or SLERP) and
reranks with exact cosine over candidate image embeddings, so it needs to send a
vector and to read vectors back. The service's text and image search endpoints
encode server-side and never return embeddings, so they cannot stand in.

Both endpoints are thin wrappers over machinery that already exists --
SearcherMixin._execute_search already accepts precomputed_vector= and skips
encoding when given one. See docs/CIR_SERVICE_API.md for the request and
response shapes.

Until they ship, run CIR with milvus.backend: direct."""


class ServiceStore:
    """FrameStore backed by the database microservice over HTTP.

    Mirrors MilvusStore's contract so the two are interchangeable behind
    build_store(). Requires the vector endpoints listed above; construction
    fails fast with a precise message when they are absent, rather than
    surfacing a confusing 404 partway through a search.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        milvus_cfg = config.section("milvus")
        service_cfg = dict(milvus_cfg.get("service", {}) or {})

        base_url = str(service_cfg.get("base_url", "")).strip()
        if not base_url:
            raise StoreBackendError(
                "milvus.service.base_url is required when milvus.backend is 'service'."
            )
        self.base_url = base_url.rstrip("/")
        self.model_name = str(service_cfg.get("model_name", config.get("model.name_or_path")))
        self.collection = str(milvus_cfg["collection"])
        self.fields = dict(milvus_cfg["fields"])
        self.embedding_dim = int(service_cfg.get("embedding_dim", 1024))
        self.metric_type = str(milvus_cfg.get("search", {}).get("metric_type", "COSINE"))

        timeout = float(service_cfg.get("timeout_seconds", 30.0))
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
        self._verify_capabilities()

    # -- setup ------------------------------------------------------------

    def _verify_capabilities(self) -> None:
        try:
            health = self.client.get("/health")
        except httpx.HTTPError as error:
            raise StoreBackendError(
                f"Database service at {self.base_url} is unreachable: {error}"
            ) from error
        if health.status_code != 200:
            raise StoreBackendError(
                f"Database service at {self.base_url} is unhealthy "
                f"[{health.status_code}]: {health.text}"
            )

        try:
            spec = self.client.get("/openapi.json")
            paths = set(spec.json().get("paths", {})) if spec.status_code == 200 else set()
        except (httpx.HTTPError, ValueError):
            # No machine-readable spec. Let the first real call decide.
            return

        if not paths:
            return
        missing = [name for name in (SEARCH_ENDPOINT, FETCH_ENDPOINT) if name not in paths]
        if missing:
            raise StoreBackendError(
                _MISSING_ENDPOINTS_HINT.format(
                    base_url=self.base_url, missing=", ".join(missing)
                )
            )

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.post(endpoint, json=payload)
        except httpx.HTTPError as error:
            raise StoreBackendError(f"{endpoint} request failed: {error}") from error
        if response.status_code != 200:
            raise StoreBackendError(
                f"{endpoint} returned [{response.status_code}]: {response.text}"
            )
        return response.json()

    # -- field helpers ----------------------------------------------------

    @property
    def id_field(self) -> str:
        return str(self.fields["id"])

    @property
    def image_vector_field(self) -> str:
        return str(self.fields["image_vector"])

    @property
    def text_vector_field(self) -> str:
        return str(self.fields["text_vector"])

    def describe(self) -> dict[str, Any]:
        """Synthesize a Milvus-shaped description.

        The service exposes no schema endpoint, so the image vector's dimension
        comes from milvus.service.embedding_dim rather than from the collection
        itself. Keep that value in step with the model when either changes.
        """
        return {
            "collection_name": self.collection,
            "fields": [
                {
                    "name": self.image_vector_field,
                    "params": {"dim": self.embedding_dim},
                },
                {
                    "name": self.text_vector_field,
                    "params": {"dim": self.embedding_dim},
                },
            ],
        }

    # -- FrameStore -------------------------------------------------------

    def get_by_id(
        self, entity_id: int | str, include_vectors: bool = True
    ) -> dict[str, Any] | None:
        entities = self.fetch_entities([entity_id], include_vectors=include_vectors)
        return entities[0] if entities else None

    def get_by_video_frame(
        self, video_name: str, frame_name: str, include_vectors: bool = True
    ) -> dict[str, Any] | None:
        expression = (
            f"{self.fields['video_name']} == {quote_milvus_string(video_name)} and "
            f"{self.fields['frame_name']} == {quote_milvus_string(frame_name)}"
        )
        payload = self._post(
            FETCH_ENDPOINT,
            {
                "model_name": self.model_name,
                "filter": expression,
                "limit": 1,
                "include_vectors": bool(include_vectors),
            },
        )
        entities = payload.get("entities") or []
        return dict(entities[0]) if entities else None

    def fetch_entities(
        self, ids: Iterable[int | str], include_vectors: bool = True
    ) -> list[dict[str, Any]]:
        id_list = list(dict.fromkeys(ids))
        if not id_list:
            return []
        payload = self._post(
            FETCH_ENDPOINT,
            {
                "model_name": self.model_name,
                "ids": id_list,
                "include_vectors": bool(include_vectors),
            },
        )
        return [dict(entity) for entity in (payload.get("entities") or [])]

    def search(
        self,
        query_vectors: list[list[float]],
        limit: int,
        expression: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        payload = self._post(
            SEARCH_ENDPOINT,
            {
                "model_name": self.model_name,
                "vectors": [list(vector) for vector in query_vectors],
                "anns_field": self.image_vector_field,
                "metric_type": self.metric_type,
                "top_k": int(limit),
                "expr": expression or None,
            },
        )
        results = payload.get("results") or []
        normalized: list[list[dict[str, Any]]] = []
        for query_hits in results:
            hits: list[dict[str, Any]] = []
            for hit in query_hits:
                entity = dict(hit.get("entity") or {})
                hit_id = hit.get("id", entity.get(self.id_field))
                entity.setdefault(self.id_field, hit_id)
                hits.append(
                    {
                        "id": hit_id,
                        "distance": float(hit.get("distance", hit.get("score", 0.0))),
                        "entity": entity,
                    }
                )
            normalized.append(hits)
        return normalized

    def close(self) -> None:
        self.client.close()

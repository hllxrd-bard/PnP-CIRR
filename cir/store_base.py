from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from .config import AppConfig


class StoreBackendError(RuntimeError):
    """Raised when a store backend is unusable or misconfigured."""


@runtime_checkable
class FrameStore(Protocol):
    """Storage operations the CIR pipeline depends on.

    Both the direct-to-Milvus store and the HTTP store implement this. These are
    the only store methods reached from outside the implementation; everything
    else on MilvusStore is internal.
    """

    def describe(self) -> dict[str, Any]:
        """Collection description shaped like Milvus's own.

        Only the ``fields`` list is consumed, to read the image vector's
        dimension. See ServiceCIREngine.expected_embedding_dimension.
        """
        ...

    def get_by_id(
        self, entity_id: int | str, include_vectors: bool = True
    ) -> dict[str, Any] | None:
        """One entity by primary key, or None. Keys are physical field names."""
        ...

    def get_by_video_frame(
        self, video_name: str, frame_name: str, include_vectors: bool = True
    ) -> dict[str, Any] | None:
        """One entity by video/frame pair, or None."""
        ...

    def fetch_entities(
        self, ids: Iterable[int | str], include_vectors: bool = True
    ) -> list[dict[str, Any]]:
        """Entities by primary key, deduplicated. Missing ids are omitted."""
        ...

    def search(
        self,
        query_vectors: list[list[float]],
        limit: int,
        expression: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """ANN search, one result list per query vector.

        Each hit is ``{"id": ..., "distance": float, "entity": {...}}``.
        """
        ...


def build_store(config: AppConfig) -> FrameStore:
    """Construct the store backend named by ``milvus.backend``.

    ``direct`` (the default) talks to Milvus over pymilvus. ``service`` talks to
    the database microservice over HTTP, and requires vector endpoints that
    service does not expose yet -- see cir/service_store.py.
    """
    backend = str(config.get("milvus.backend", "direct") or "direct").strip().lower()

    if backend == "direct":
        from .milvus_store import MilvusStore

        return MilvusStore(config)

    if backend == "service":
        from .service_store import ServiceStore

        return ServiceStore(config)

    raise StoreBackendError(
        f"Unknown milvus.backend {backend!r}. Expected 'direct' or 'service'."
    )

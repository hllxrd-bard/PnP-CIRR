from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pymilvus import DataType, MilvusClient

from .config import AppConfig
from .utils import quote_milvus_string


class MilvusStore:
    def __init__(self, config: AppConfig):
        self.config = config
        cfg = config.section("milvus")
        kwargs: dict[str, Any] = {
            "uri": cfg["uri"],
            "db_name": cfg.get("database", "default"),
        }
        if cfg.get("token"):
            kwargs["token"] = cfg["token"]
        self.client = MilvusClient(**kwargs)
        self.collection = str(cfg["collection"])
        self.fields = dict(cfg["fields"])
        self.search_cfg = dict(cfg.get("search", {}))
        self.consistency_level = cfg.get("consistency_level", "Bounded")
        self._description = self.client.describe_collection(collection_name=self.collection)
        self._field_names = {
            str(field.get("name")) for field in self._description.get("fields", [])
        }
        self._primary_field = next(
            (field for field in self._description.get("fields", []) if field.get("is_primary")),
            None,
        )
        if cfg.get("load_collection", True):
            self.client.load_collection(collection_name=self.collection)

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
        return dict(self._description)

    def has_field(self, name: str) -> bool:
        return str(name) in self._field_names

    def coerce_id(self, value: int | str) -> int | str:
        if self._primary_field is None:
            return value
        raw_type = self._primary_field.get("type")
        type_name = getattr(raw_type, "name", str(raw_type).split(".")[-1]).upper()
        if type_name in {"INT8", "INT16", "INT32", "INT64"}:
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        return str(value)

    def scalar_fields(self) -> list[str]:
        described = self.describe()
        vector_types = {
            DataType.FLOAT_VECTOR,
            DataType.BINARY_VECTOR,
            DataType.FLOAT16_VECTOR,
            DataType.BFLOAT16_VECTOR,
            DataType.SPARSE_FLOAT_VECTOR,
        }
        vector_names = {item.name for item in vector_types}
        fields = []
        for field in described.get("fields", []):
            raw_dtype = field.get("type")
            dtype_name = getattr(raw_dtype, "name", str(raw_dtype).split(".")[-1]).upper()
            if raw_dtype not in vector_types and dtype_name not in vector_names:
                fields.append(str(field["name"]))
        return fields

    def _resolve_configured_fields(self, values: list[Any]) -> list[str]:
        existing = set(self.scalar_fields())
        resolved: list[str] = []
        for item in values:
            name = str(item)
            physical = str(self.fields.get(name, name))
            if physical in existing:
                resolved.append(physical)
        return list(dict.fromkeys(resolved))

    def entity_output_fields(self) -> list[str]:
        configured = self.config.get("retrieval.entity_output_fields", []) or []
        if configured:
            fields = self._resolve_configured_fields(list(configured))
            if fields:
                return fields
        return self.scalar_fields()

    def search_output_fields(self) -> list[str]:
        configured = self.config.get("retrieval.search_output_fields", []) or []
        fields = self._resolve_configured_fields(list(configured)) if configured else []
        if not fields:
            fields = self.entity_output_fields()
        return fields

    def default_output_fields(self) -> list[str]:
        # Backwards-compatible alias for entity retrieval.
        return self.entity_output_fields()

    def get_by_id(self, entity_id: int | str, include_vectors: bool = True) -> dict[str, Any] | None:
        output_fields = self.default_output_fields()
        if include_vectors:
            if self.has_field(self.image_vector_field):
                output_fields.append(self.image_vector_field)
            if self.has_field(self.text_vector_field):
                output_fields.append(self.text_vector_field)
        rows = self.client.get(
            collection_name=self.collection,
            ids=[self.coerce_id(entity_id)],
            output_fields=list(dict.fromkeys(output_fields)),
            consistency_level=self.consistency_level,
        )
        return dict(rows[0]) if rows else None

    def get_by_video_frame(
        self,
        video_name: str,
        frame_name: str,
        include_vectors: bool = True,
    ) -> dict[str, Any] | None:
        video_field = str(self.fields["video_name"])
        frame_field = str(self.fields["frame_name"])
        expression = (
            f"{video_field} == {quote_milvus_string(video_name)} and "
            f"{frame_field} == {quote_milvus_string(frame_name)}"
        )
        output_fields = self.default_output_fields()
        if include_vectors:
            if self.has_field(self.image_vector_field):
                output_fields.append(self.image_vector_field)
            if self.has_field(self.text_vector_field):
                output_fields.append(self.text_vector_field)
        rows = self.client.query(
            collection_name=self.collection,
            filter=expression,
            output_fields=list(dict.fromkeys(output_fields)),
            limit=1,
            consistency_level=self.consistency_level,
        )
        return dict(rows[0]) if rows else None

    def fetch_entities(self, ids: Iterable[int | str], include_vectors: bool = True) -> list[dict[str, Any]]:
        id_list = list(dict.fromkeys(ids))
        if not id_list:
            return []
        output_fields = self.default_output_fields()
        if include_vectors:
            if self.has_field(self.image_vector_field):
                output_fields.append(self.image_vector_field)
            if self.has_field(self.text_vector_field):
                output_fields.append(self.text_vector_field)
        rows = self.client.get(
            collection_name=self.collection,
            ids=id_list,
            output_fields=list(dict.fromkeys(output_fields)),
            consistency_level=self.consistency_level,
        )
        return [dict(row) for row in rows]

    def search(
        self,
        query_vectors: list[list[float]],
        limit: int,
        expression: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        params = {
            "metric_type": self.search_cfg.get("metric_type", "COSINE"),
            "params": self.search_cfg.get("params", {}),
        }
        kwargs: dict[str, Any] = {
            "collection_name": self.collection,
            "data": query_vectors,
            "anns_field": self.image_vector_field,
            "limit": int(limit),
            "search_params": params,
            "output_fields": self.search_output_fields(),
            "consistency_level": self.consistency_level,
        }
        if expression:
            kwargs["filter"] = expression
        raw_results = self.client.search(**kwargs)
        normalized: list[list[dict[str, Any]]] = []
        for query_hits in raw_results:
            hits: list[dict[str, Any]] = []
            for hit in query_hits:
                if isinstance(hit, dict):
                    entity = dict(hit.get("entity") or {})
                    entity.setdefault(self.id_field, hit.get("id"))
                    hits.append(
                        {
                            "id": hit.get("id", entity.get(self.id_field)),
                            "distance": float(hit.get("distance", hit.get("score", 0.0))),
                            "entity": entity,
                        }
                    )
                else:
                    entity = dict(getattr(hit, "entity", {}) or {})
                    hit_id = getattr(hit, "id", entity.get(self.id_field))
                    entity.setdefault(self.id_field, hit_id)
                    hits.append(
                        {
                            "id": hit_id,
                            "distance": float(getattr(hit, "distance", 0.0)),
                            "entity": entity,
                        }
                    )
            normalized.append(hits)
        return normalized

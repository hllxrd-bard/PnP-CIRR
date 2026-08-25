from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np

from .engine import CIREngine
from .schemas import CIRRequest, ReferenceInput
from .service_schemas import INTERNAL_MODES, PublicCIRRequest, PublicReferenceInput
from .utils import as_numpy

_DIRECT_REFERENCE: ContextVar[PublicReferenceInput | None] = ContextVar(
    "cir_direct_reference", default=None
)


class ServiceCIREngine(CIREngine):
    """CIREngine adapter for the stable /v1 service contract.

    Existing directional/Slerp implementations remain untouched. This adapter
    adds canonical public mode names and an optional validated reference-vector
    fast path.
    """

    def expected_embedding_dimension(self) -> int:
        image_field = str(self.fields["image_vector"])
        description = self.store.describe()
        for field in description.get("fields", []):
            if str(field.get("name")) != image_field:
                continue
            params = field.get("params") or {}
            for key in ("dim", "dimension"):
                value = params.get(key, field.get(key))
                if value is not None:
                    return int(value)
        # Current SigLIP2-large gallery contract. Kept as a defensive fallback.
        return 1024

    def _validate_direct_embedding(
        self, reference: PublicReferenceInput
    ) -> np.ndarray:
        assert reference.image_embedding is not None
        expected_dim = self.expected_embedding_dimension()
        expected_model = str(self.encoder.model_name)

        if reference.embedding_dimension != expected_dim:
            raise ValueError(
                "Reference embedding dimension mismatch: "
                f"expected {expected_dim}, received {reference.embedding_dimension}."
            )
        if len(reference.image_embedding) != expected_dim:
            raise ValueError(
                "Reference image_embedding length mismatch: "
                f"expected {expected_dim}, received {len(reference.image_embedding)}."
            )
        if str(reference.embedding_model) != expected_model:
            raise ValueError(
                "Reference embedding model mismatch: "
                f"expected '{expected_model}', received '{reference.embedding_model}'."
            )

        vector = np.asarray(reference.image_embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size != expected_dim:
            raise ValueError("Reference image_embedding must be a one-dimensional vector.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Reference image_embedding contains NaN or infinite values.")

        epsilon = float(self.config.get("composition.epsilon", 1e-8))
        norm = float(np.linalg.norm(vector))
        if norm <= epsilon:
            raise ValueError("Reference image_embedding norm is too small.")
        return vector / norm

    def _resolve_reference(
        self, request: CIRRequest
    ) -> tuple[dict[str, Any], np.ndarray, Path | None]:
        direct = _DIRECT_REFERENCE.get()
        if direct is None or direct.image_embedding is None:
            return super()._resolve_reference(request)

        vector = self._validate_direct_embedding(direct)
        if direct.id is not None:
            entity = self.store.get_by_id(direct.id, include_vectors=False)
            if entity is None:
                raise LookupError(f"Reference id was not found in Milvus: {direct.id}")
            return entity, vector, self._entity_image_path(entity)

        if direct.video_name and direct.frame_name:
            entity = self.store.get_by_video_frame(
                direct.video_name,
                direct.frame_name,
                include_vectors=False,
            )
            if entity is None:
                raise LookupError(
                    "Reference frame was not found in Milvus: "
                    f"{direct.video_name}/{direct.frame_name}"
                )
            return entity, vector, self._entity_image_path(entity)

        assert direct.path is not None
        path = Path(direct.path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {path}")
        entity = {
            self.fields["id"]: f"local:{path}",
            self.fields["video_name"]: None,
            self.fields["frame_name"]: path.name,
            self.fields["timestamp"]: None,
            "local_path": str(path),
        }
        return entity, vector, path

    def to_internal_request(self, request: PublicCIRRequest) -> CIRRequest:
        reference = request.reference
        internal_reference = ReferenceInput(
            id=reference.id,
            video_name=reference.video_name,
            frame_name=reference.frame_name,
            path=reference.path,
        )
        return CIRRequest(
            reference=internal_reference,
            composition_mode=INTERNAL_MODES[request.canonical_mode],
            edit_text=request.edit_text,
            remove_text=request.remove_text,
            top_k=request.top_k,
            use_vlm=request.use_vlm,
            vlm_provider=request.vlm_provider,
            edit_strength=request.edit_strength,
            slerp_alpha=request.slerp_alpha,
            slerp_remove_gamma=request.slerp_hybrid_gamma,
            search=request.search.model_dump(mode="python"),
            filters={
                **request.filters.model_dump(mode="python"),
                "milvus_expression": None,
            },
            deduplication=request.deduplication.model_dump(mode="python"),
        )

    def search_public(self, request: PublicCIRRequest):
        internal = self.to_internal_request(request)
        token = _DIRECT_REFERENCE.set(request.reference)
        try:
            return super().search(internal)
        finally:
            _DIRECT_REFERENCE.reset(token)

    def resolve_public_reference(
        self, reference: PublicReferenceInput
    ) -> tuple[dict[str, Any], np.ndarray, Path | None, str]:
        internal = CIRRequest(
            reference=ReferenceInput(
                id=reference.id,
                video_name=reference.video_name,
                frame_name=reference.frame_name,
                path=reference.path,
            ),
            edit_text="reference preview",
        )
        token = _DIRECT_REFERENCE.set(reference)
        try:
            entity, vector, path = self._resolve_reference(internal)
        finally:
            _DIRECT_REFERENCE.reset(token)

        if reference.image_embedding is not None:
            source = "request"
        elif reference.id is not None or (
            reference.video_name and reference.frame_name
        ):
            source = "milvus"
        else:
            source = "encoded_path"
        return entity, as_numpy(vector), path, source

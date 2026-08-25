from __future__ import annotations

import argparse
import logging
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from cir.config import AppConfig, load_config
from cir.service_engine import ServiceCIREngine
from cir.service_schemas import (
    ErrorResponse,
    PublicCIRRequest,
    ResolveReferenceRequest,
)
from cir.utils import is_path_within, resolve_frame_path

SERVICE_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


class ServiceAPIError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", f"cir_{uuid.uuid4().hex}")


def _error_body(
    request_id: str,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ErrorResponse(
        request_id=request_id,
        error={"code": code, "message": message, "details": details or {}},
    ).model_dump(mode="json")


def _validate_semantics(payload: PublicCIRRequest, config: AppConfig) -> None:
    mode = payload.canonical_mode
    if mode == "directional":
        if not payload.edit_text and not payload.remove_text:
            raise ServiceAPIError(
                422,
                "MISSING_EDIT_OPERATION",
                "directional requires edit_text, remove_text, or both.",
            )
    elif mode == "pure_slerp":
        if not payload.edit_text:
            raise ServiceAPIError(
                422,
                "MISSING_TEXTUAL_INTENT",
                "pure_slerp requires non-empty edit_text as the full textual intent.",
            )
        if payload.remove_text:
            raise ServiceAPIError(
                409,
                "UNSUPPORTED_MODE_COMBINATION",
                "pure_slerp does not support remove_text.",
                {
                    "composition_mode": "pure_slerp",
                    "unsupported_field": "remove_text",
                    "suggested_mode": "slerp_hybrid",
                },
            )
    elif mode == "slerp_hybrid" and not payload.remove_text:
        raise ServiceAPIError(
            422,
            "MISSING_REMOVE_TEXT",
            "slerp_hybrid requires non-empty remove_text; edit_text is optional.",
        )

    max_top_k = int(config.get("retrieval.max_top_k", 300))
    if payload.top_k is not None and payload.top_k > max_top_k:
        raise ServiceAPIError(
            422,
            "TOP_K_EXCEEDS_LIMIT",
            f"top_k must be less than or equal to {max_top_k}.",
            {"max_top_k": max_top_k, "received_top_k": payload.top_k},
        )

    if payload.search.candidate_k_per_query and payload.search.max_candidate_pool:
        if payload.search.max_candidate_pool < payload.top_k if payload.top_k else False:
            raise ServiceAPIError(
                422,
                "INVALID_CANDIDATE_POOL",
                "max_candidate_pool cannot be smaller than top_k.",
            )


def _map_runtime_error(exc: Exception) -> ServiceAPIError:
    message = str(exc)
    module = type(exc).__module__.lower()
    name = type(exc).__name__

    if isinstance(exc, LookupError) or isinstance(exc, FileNotFoundError):
        return ServiceAPIError(404, "REFERENCE_NOT_FOUND", message)
    if isinstance(exc, TimeoutError) or "timeout" in message.lower():
        return ServiceAPIError(504, "MILVUS_TIMEOUT", message)
    if "pymilvus" in module or "milvus" in name.lower() or "milvus" in message.lower():
        return ServiceAPIError(502, "MILVUS_ERROR", message)
    if isinstance(exc, ValueError):
        if "embedding model mismatch" in message.lower():
            code = "EMBEDDING_MODEL_MISMATCH"
        elif "embedding dimension" in message.lower() or "embedding length" in message.lower():
            code = "EMBEDDING_DIMENSION_MISMATCH"
        elif "nan" in message.lower() or "infinite" in message.lower():
            code = "INVALID_EMBEDDING_VALUES"
        else:
            code = "INVALID_REQUEST"
        return ServiceAPIError(422, code, message)
    return ServiceAPIError(500, "INTERNAL_ERROR", "Internal CIR service error.")


def create_service_app(config: AppConfig, warmup: bool = True) -> FastAPI:
    app = FastAPI(
        title="PnP-CIRR Service API",
        version=SERVICE_VERSION,
        description=(
            "Versioned composed-image-retrieval API backed directly by pymilvus. "
            "The legacy viewer endpoints remain in visualize.py."
        ),
    )
    engine = ServiceCIREngine(config, warmup=warmup)
    app.state.engine = engine
    app.state.config = config

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or (
            f"cir_{uuid.uuid4().hex}"
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ServiceAPIError)
    async def service_error_handler(request: Request, exc: ServiceAPIError):
        if exc.status_code >= 500:
            LOGGER.exception("CIR service request failed: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                _request_id(request), exc.code, exc.message, exc.details
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_error_body(
                _request_id(request),
                "VALIDATION_ERROR",
                "Request body validation failed.",
                {"errors": exc.errors()},
            ),
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "pnp-cirr",
            "service_version": SERVICE_VERSION,
        }

    @app.get("/ready")
    def ready(request: Request) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        try:
            description = engine.store.client.describe_collection(
                collection_name=engine.store.collection
            )
            checks["milvus"] = {
                "status": "ok",
                "collection": engine.store.collection,
                "field_count": len(description.get("fields", [])),
            }
            frames_root = Path(str(config.get("frames.root"))).expanduser().resolve()
            checks["frames_root"] = {
                "status": "ok" if frames_root.exists() else "error",
                "path": str(frames_root),
            }
            checks["encoder"] = {
                "status": "ok",
                "model": engine.encoder.model_name,
                "loaded": bool(engine.encoder.is_loaded),
                "dimension": engine.expected_embedding_dimension(),
            }
            if not frames_root.exists():
                raise ServiceAPIError(
                    503,
                    "SERVICE_NOT_READY",
                    "Configured frames root does not exist.",
                    checks,
                )
            return {
                "status": "ready",
                "service_version": SERVICE_VERSION,
                "checks": checks,
            }
        except ServiceAPIError:
            raise
        except Exception as exc:
            raise ServiceAPIError(
                503,
                "SERVICE_NOT_READY",
                str(exc),
                checks,
            ) from exc

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "service_version": SERVICE_VERSION,
            "embedding_model": engine.encoder.model_name,
            "embedding_dimension": engine.expected_embedding_dimension(),
            "reference_locators": ["id", "video_name+frame_name", "path"],
            "direct_reference_embedding": {
                "supported": True,
                "recommended": False,
                "requires_model_and_dimension": True,
            },
            "composition_modes": {
                "directional": {
                    "supports_edit": True,
                    "supports_remove": True,
                    "experimental": False,
                },
                "pure_slerp": {
                    "supports_edit": True,
                    "supports_remove": False,
                    "experimental": False,
                },
                "slerp_hybrid": {
                    "supports_edit": True,
                    "supports_remove": True,
                    "edit_optional": True,
                    "experimental": True,
                },
            },
            "deprecated_mode_aliases": {
                "slerp": "pure_slerp",
                "slerp_remove": "slerp_hybrid",
            },
            "defaults": {
                "top_k": int(config.get("retrieval.default_top_k", 60)),
                "edit_strength": float(
                    config.get("composition.default_edit_strength", 0.95)
                ),
                "slerp_alpha": float(config.get("slerp.default_alpha", 0.8)),
                "slerp_hybrid_add_alpha": float(
                    config.get("slerp_remove.default_add_alpha", 0.4)
                ),
                "slerp_hybrid_gamma": float(
                    config.get("slerp_remove.default_gamma", 0.2)
                ),
                "candidate_k_per_query": int(
                    config.get("retrieval.candidate_k_per_query", 150)
                ),
                "max_candidate_pool": int(
                    config.get("retrieval.max_candidate_pool", 700)
                ),
            },
            "limits": {
                "max_top_k": int(config.get("retrieval.max_top_k", 300)),
            },
            "filters": [
                "exclude_reference",
                "include_video_prefixes",
                "exclude_video_prefixes",
                "exclude_video_names",
            ],
        }

    @app.post("/v1/references/resolve")
    def resolve_reference(
        payload: ResolveReferenceRequest, request: Request
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            entity, _, path, embedding_source = engine.resolve_public_reference(
                payload.reference
            )
        except Exception as exc:
            raise _map_runtime_error(exc) from exc

        fields = engine.fields
        video_name = entity.get(fields["video_name"])
        frame_name = entity.get(fields["frame_name"])
        if payload.reference.path:
            image_url = "/v1/local-frames?path=" + quote(
                payload.reference.path, safe=""
            )
        elif video_name is not None and frame_name is not None:
            image_url = (
                "/v1/frames?video_name="
                + quote(str(video_name))
                + "&frame_name="
                + quote(str(frame_name))
            )
        else:
            image_url = None
        return {
            "status": "success",
            "request_id": _request_id(request),
            "service_version": SERVICE_VERSION,
            "reference": {
                "id": entity.get(fields["id"]),
                "video_name": video_name,
                "frame_name": frame_name,
                "timestamp": entity.get(fields["timestamp"]),
                "frame_id": entity.get(fields.get("frame_id")),
                "cluster_id": entity.get(fields.get("cluster_id")),
                "image_path": str(path) if path else None,
                "image_url": image_url,
                "embedding_source": embedding_source,
            },
            "timings_ms": {
                "total": (time.perf_counter() - started) * 1000.0
            },
        }

    @app.post("/v1/search")
    def search(payload: PublicCIRRequest, request: Request) -> dict[str, Any]:
        _validate_semantics(payload, config)
        try:
            output = engine.search_public(payload)
        except Exception as exc:
            raise _map_runtime_error(exc) from exc

        data = output.model_dump(mode="json")
        canonical_mode = payload.canonical_mode
        public_request = payload.model_dump(mode="json")
        public_request["composition_mode"] = canonical_mode

        warnings: list[Any] = list(data.get("warnings") or [])
        if payload.uses_deprecated_mode_alias:
            warnings.append(
                {
                    "code": "DEPRECATED_COMPOSITION_MODE",
                    "message": (
                        f"'{payload.composition_mode}' is deprecated; "
                        f"use '{canonical_mode}'."
                    ),
                }
            )

        reference = data.get("reference") or {}
        if payload.reference.image_embedding is not None:
            reference["embedding_source"] = "request"
        elif payload.reference.id is not None or (
            payload.reference.video_name and payload.reference.frame_name
        ):
            reference["embedding_source"] = "milvus"
        else:
            reference["embedding_source"] = "encoded_path"

        if payload.reference.path:
            reference["image_url"] = "/v1/local-frames?path=" + quote(
                payload.reference.path, safe=""
            )

        for item in data.get("results") or []:
            video_name = item.get("video_name")
            frame_name = item.get("frame_name")
            if video_name is not None and frame_name is not None:
                item["image_url"] = (
                    "/v1/frames?video_name="
                    + quote(str(video_name))
                    + "&frame_name="
                    + quote(str(frame_name))
                )

        return {
            "status": "success",
            "request_id": _request_id(request),
            "service_version": SERVICE_VERSION,
            "composition_mode": canonical_mode,
            "request": public_request,
            "reference": reference,
            "query": data.get("query"),
            "timings_ms": data.get("timings_ms"),
            "warnings": warnings,
            "results": data.get("results") or [],
        }

    @app.get("/v1/frames")
    def frame(
        video_name: str = Query(...), frame_name: str = Query(...)
    ) -> FileResponse:
        root = Path(str(config.get("frames.root"))).expanduser().resolve()
        path = resolve_frame_path(
            frames_root=str(root),
            path_template=str(config.get("frames.path_template")),
            video_name=video_name,
            frame_name=frame_name,
        )
        if path is None or not is_path_within(path, root):
            raise HTTPException(status_code=403, detail="Invalid frame path")
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Frame not found: {path}")
        if path.suffix.lower() not in set(config.get("frames.allowed_extensions", [])):
            raise HTTPException(status_code=415, detail="Unsupported image extension")
        return FileResponse(path)

    @app.get("/v1/local-frames")
    def local_frame(path: str = Query(...)) -> FileResponse:
        if not config.get("web.allow_local_reference_path", True):
            raise HTTPException(status_code=403, detail="Local reference paths are disabled")
        image_path = Path(path).expanduser().resolve()
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail="Local image does not exist")
        if image_path.suffix.lower() not in set(config.get("frames.allowed_extensions", [])):
            raise HTTPException(status_code=415, detail="Unsupported image extension")
        return FileResponse(image_path)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the versioned CIR service API.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, str(config.get("runtime.log_level", "INFO")).upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app = create_service_app(config, warmup=not args.no_warmup)
    uvicorn.run(
        app,
        host=args.host or str(config.get("web.host", "0.0.0.0")),
        port=args.port or int(config.get("web.port", 8088)),
        workers=1,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cir.config import AppConfig, load_config
from cir.engine import CIREngine
from cir.schemas import CIRRequest, ReferenceInput
from cir.utils import is_path_within, resolve_frame_path

ROOT = Path(__file__).resolve().parent


def create_app(config: AppConfig, warmup: bool = True) -> FastAPI:
    app = FastAPI(title=str(config.get("web.title", "Interactive CIR Viewer")))
    templates = Jinja2Templates(directory=str(ROOT / "web" / "templates"))
    app.mount("/static", StaticFiles(directory=str(ROOT / "web" / "static")), name="static")

    @app.middleware("http")
    async def disable_browser_cache(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


    engine = CIREngine(config, warmup=warmup)
    app.state.engine = engine
    app.state.config = config

    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "title": config.get("web.title", "Interactive CIR Viewer"),
                "default_top_k": config.get("retrieval.default_top_k", 60),
                "default_edit_strength": config.get("composition.default_edit_strength", 0.95),
                "default_composition_mode": config.get("slerp.default_mode", "directional"),
                "default_slerp_alpha": config.get("slerp.default_alpha", 0.8),
                "default_slerp_remove_gamma": config.get("slerp_remove.default_gamma", 0.2),
                "page_size": config.get("web.page_size", 30),
                "vlm_default": config.get("vlm.enabled_by_default", False),
                "vlm_model": config.get("vlm.model"),
            },
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "collection": engine.store.collection,
            "model": engine.encoder.model_name,
            "model_loaded": engine.encoder.is_loaded,
            "device": str(engine.encoder.device),
        }

    @app.post("/api/reference")
    def lookup_reference(reference: ReferenceInput = Body(...)) -> dict[str, Any]:
        try:
            entity, _, path = engine._resolve_reference(  # intentional reuse of validated engine logic
                CIRRequest(reference=reference, edit_text="preview")
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        fields = engine.fields
        image_url = None
        if reference.path is not None:
            image_url = f"/api/local-frame?path={quote(reference.path, safe='')}"
        elif entity.get(fields["video_name"]) is not None and entity.get(fields["frame_name"]) is not None:
            image_url = (
                "/api/frame?video_name="
                + quote(str(entity.get(fields["video_name"])))
                + "&frame_name="
                + quote(str(entity.get(fields["frame_name"])))
            )
        return {
            "id": entity.get(fields["id"]),
            "video_name": entity.get(fields["video_name"]),
            "frame_name": entity.get(fields["frame_name"]),
            "timestamp": entity.get(fields["timestamp"]),
            "image_path": str(path) if path else None,
            "image_url": image_url,
        }

    @app.post("/api/search")
    def search(request_payload: CIRRequest) -> dict[str, Any]:
        try:
            output = engine.search(request_payload)
        except Exception as exc:
            logging.exception("CIR web request failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        data = output.model_dump(mode="json")
        if request_payload.reference.path and data.get("reference"):
            data["reference"]["image_url"] = (
                f"/api/local-frame?path={quote(request_payload.reference.path, safe='')}"
            )
        return data

    @app.get("/api/frame")
    def frame(
        video_name: str = Query(...),
        frame_name: str = Query(...),
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

    @app.get("/api/local-frame")
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
    parser = argparse.ArgumentParser(description="Run the FastAPI CIR visualizer.")
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
    app = create_app(config, warmup=not args.no_warmup)
    uvicorn.run(
        app,
        host=args.host or str(config.get("web.host", "0.0.0.0")),
        port=args.port or int(config.get("web.port", 8088)),
        workers=1,
    )


if __name__ == "__main__":
    main()

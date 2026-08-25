from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .config import AppConfig
from .utils import normalize_rows


class Siglip2Encoder:
    """Lazy, thread-safe SigLIP2 image/text encoder."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._lock = threading.RLock()
        runtime = config.section("runtime")
        model_cfg = config.section("model")

        requested_device = str(runtime.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        dtype_name = str(runtime.get("dtype", "float16")).lower()
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype_name, torch.float16)
        if self.device.type == "cpu" and self.dtype == torch.float16:
            self.dtype = torch.float32

        self.model_name = str(model_cfg["name_or_path"])
        self.max_text_length = int(model_cfg.get("max_text_length", 64))
        # SigLIP2 pads to max_length; CLIP-family models pad to the longest item in
        # the batch. This must match how the gallery embeddings were produced.
        self.text_padding = model_cfg.get("text_padding", "max_length")
        self.normalize = bool(model_cfg.get("normalize_embeddings", True))
        self.trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
        self.local_files_only = bool(model_cfg.get("local_files_only", False))


    @staticmethod
    def _extract_feature_tensor(output: object) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        pooler_output = getattr(output, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output
        if isinstance(output, (tuple, list)):
            two_dimensional = [
                item for item in output if isinstance(item, torch.Tensor) and item.ndim == 2
            ]
            if two_dimensional:
                return two_dimensional[-1]
        raise TypeError(
            "Could not extract a pooled embedding tensor from the Transformers model output."
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                low_cpu_mem_usage=True,
            )
            self._model.to(self.device)
            self._model.eval()

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        text_list = [str(text).strip() for text in texts]
        if not text_list or any(not text for text in text_list):
            raise ValueError("All text inputs must be non-empty strings.")
        self.load()
        assert self._processor is not None
        assert self._model is not None

        with self._lock, torch.inference_mode():
            inputs = self._processor(
                text=text_list,
                padding=self.text_padding,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self._model.get_text_features(**inputs)
            features = self._extract_feature_tensor(output)
            array = features.detach().float().cpu().numpy().astype(np.float32, copy=False)
        return normalize_rows(array) if self.normalize else array

    def encode_images(self, image_paths: Iterable[str | Path]) -> np.ndarray:
        paths = [Path(path).expanduser().resolve() for path in image_paths]
        if not paths:
            raise ValueError("At least one image path is required.")
        images: list[Image.Image] = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Image does not exist: {path}")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        self.load()
        assert self._processor is not None
        assert self._model is not None
        with self._lock, torch.inference_mode():
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self._model.get_image_features(**inputs)
            features = self._extract_feature_tensor(output)
            array = features.detach().float().cpu().numpy().astype(np.float32, copy=False)
        return normalize_rows(array) if self.normalize else array

    def warmup(self, text: str) -> None:
        self.encode_texts([text])

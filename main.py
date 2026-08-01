from __future__ import annotations

import json
import logging

from cir.config import load_config
from cir.engine import CIREngine
from cir.schemas import CIRRequest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # 1. Đọc config
    config = load_config("config.yaml")

    # 2. Khởi tạo engine
    # Engine sẽ load SigLIP2, kết nối Milvus và chuẩn bị reranker.
    engine = CIREngine(
        config=config,
        warmup=True,
    )

    # 3. Tạo request
    request = CIRRequest.model_validate(
        {
            "reference": {
                "path": (
                    "/workingspace_aiclub/WorkingSpace/Personal/"
                    "chinhnm/AIC2026/frames/L30_V071/frame_034.webp"
                )
            },
            "edit_text": "pond",
            "remove_text": "lotus",
            "top_k": 20,
            "use_vlm": False,
            "edit_strength": 0.95,
            "deduplication": {
                "enabled": True,
            },
        }
    )

    # 4. Chạy CIR
    output = engine.search(request)

    # 5. Chuyển Pydantic output thành dict/JSON
    output_dict = output.model_dump(mode="json")

    print(
        json.dumps(
            output_dict,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
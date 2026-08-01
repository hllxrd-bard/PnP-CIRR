from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cir.config import load_config  # noqa: E402
from cir.encoder import Siglip2Encoder  # noqa: E402
from cir.milvus_store import MilvusStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate config, Milvus, frame root, and SigLIP2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    checks = []
    try:
        store = MilvusStore(config)
        checks.append({"name": "milvus", "ok": True, "collection": store.collection})
        description = store.describe()
        checks.append({"name": "schema", "ok": True, "fields": [f.get("name") for f in description.get("fields", [])]})
    except Exception as exc:
        checks.append({"name": "milvus", "ok": False, "error": str(exc)})

    frame_root = Path(str(config.get("frames.root"))).expanduser()
    checks.append({"name": "frames_root", "ok": frame_root.exists(), "path": str(frame_root)})

    if not args.skip_model:
        try:
            encoder = Siglip2Encoder(config)
            vector = encoder.encode_texts([str(config.get("runtime.warmup_text"))])[0]
            checks.append({"name": "siglip2", "ok": True, "dimension": int(vector.size), "device": str(encoder.device)})
        except Exception as exc:
            checks.append({"name": "siglip2", "ok": False, "error": str(exc)})

    print(json.dumps({"checks": checks}, ensure_ascii=False, indent=2))
    return 0 if all(item["ok"] for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

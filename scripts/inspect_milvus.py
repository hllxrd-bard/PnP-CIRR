from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cir.config import load_config  # noqa: E402
from cir.milvus_store import MilvusStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Milvus schema, indexes, samples, and runtime.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="milvus_inspection.json")
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--vector-head", type=int, default=5)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def summarize_value(value: Any, vector_head: int) -> Any:
    if isinstance(value, (list, tuple)) and len(value) > 32:
        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            return {
                "kind": "vector_summary",
                "dimension": int(array.size),
                "norm": float(np.linalg.norm(array)),
                "head": array[:vector_head].tolist(),
            }
        except (TypeError, ValueError):
            return {"kind": "long_list", "length": len(value), "head": list(value[:vector_head])}
    if isinstance(value, dict):
        return {str(key): summarize_value(item, vector_head) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize_value(item, vector_head) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def field_type_name(field: dict[str, Any]) -> str:
    raw = field.get("type")
    return getattr(raw, "name", str(raw).split(".")[-1])


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    store = MilvusStore(config)
    client = store.client
    collection = store.collection

    report: dict[str, Any] = {
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "transformers": package_version("transformers"),
            "pymilvus": package_version("pymilvus"),
            "fastapi": package_version("fastapi"),
        },
        "configured": {
            "uri": config.get("milvus.uri"),
            "database": config.get("milvus.database"),
            "collection": collection,
            "fields": config.get("milvus.fields"),
            "model": config.get("model.name_or_path"),
            "frames_root": config.get("frames.root"),
        },
        "warnings": [],
    }
    if torch.cuda.is_available():
        report["runtime"]["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(index).total_memory / (1024**3), 3
                ),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]

    description = client.describe_collection(collection_name=collection)
    fields = description.get("fields", [])
    report["collection"] = summarize_value(description, args.vector_head)
    report["collection"]["field_summary"] = [
        {
            "name": field.get("name"),
            "type": field_type_name(field),
            "is_primary": bool(field.get("is_primary", False)),
            "dimension": (field.get("params") or {}).get("dim", field.get("dim")),
            "nullable": field.get("nullable"),
        }
        for field in fields
    ]

    try:
        index_names = client.list_indexes(collection_name=collection)
        report["indexes"] = [
            summarize_value(
                client.describe_index(collection_name=collection, index_name=index_name),
                args.vector_head,
            )
            for index_name in index_names
        ]
    except Exception as exc:
        report["indexes"] = []
        report["warnings"].append(f"Could not inspect indexes: {exc}")

    for key, function in (
        ("load_state", client.get_load_state),
        ("stats", client.get_collection_stats),
    ):
        try:
            report[key] = summarize_value(function(collection_name=collection), args.vector_head)
        except Exception as exc:
            report[key] = None
            report["warnings"].append(f"Could not read {key}: {exc}")

    configured_fields = config.get("milvus.fields", {})
    existing_names = {str(field.get("name")) for field in fields}
    missing = {
        logical: physical
        for logical, physical in configured_fields.items()
        if physical and str(physical) not in existing_names
    }
    if missing:
        report["warnings"].append(f"Configured fields missing from collection: {missing}")

    scalar_fields = store.scalar_fields()
    samples: list[dict[str, Any]] = []
    try:
        rows = client.query(
            collection_name=collection,
            filter="",
            output_fields=scalar_fields,
            limit=max(1, args.sample_size),
            consistency_level=config.get("milvus.consistency_level", "Bounded"),
        )
        for row in rows:
            samples.append(summarize_value(dict(row), args.vector_head))
    except Exception as exc:
        report["warnings"].append(f"Scalar sample query failed: {exc}")

    id_field = configured_fields.get("id")
    vector_samples: list[dict[str, Any]] = []
    if id_field:
        for sample in samples:
            entity_id = sample.get(id_field)
            if entity_id is None:
                continue
            try:
                entity = store.get_by_id(entity_id, include_vectors=True)
                if entity:
                    vector_samples.append(summarize_value(entity, args.vector_head))
            except Exception as exc:
                report["warnings"].append(f"Vector sample fetch failed for {entity_id}: {exc}")
            if len(vector_samples) >= args.sample_size:
                break
    report["samples_scalar"] = samples
    report["samples_with_vector_summaries"] = vector_samples

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nSaved inspection report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "cir/slerp_method/remove_pipeline.py"

if not TARGET.is_file():
    raise SystemExit(f"Missing: {TARGET}")

text = TARGET.read_text(encoding="utf-8")
required = {
    "directional reranker call": "engine.reranker.rank(",
    "reference multiprobe": 'NamedQuery("reference"',
    "edit multiprobe": 'NamedQuery("edit_text"',
    "explicit spherical query": "explicit_spherical_",
    "directional removal aliases": "engine.composer.expand_remove_texts",
    "removal vectors to reranker": "removal_vectors=removal_vectors",
}
missing = [name for name, marker in required.items() if marker not in text]
if missing:
    raise SystemExit("Hybrid verification failed; missing: " + ", ".join(missing))

ast.parse(text, filename=str(TARGET))
print("SLERP Remove hybrid source verification: OK")

# PnP-CIRR Training-free SLERP Patch

This patch adds a second retrieval method without replacing the existing directional Add/Remove pipeline.

## Modes

- `directional`: existing pipeline and default behavior.
- `slerp`: pure training-free SLERP baseline.

The SLERP code lives under `cir/slerp_method/` and performs:

```text
reference image embedding
+ deterministic full textual intent embedding
→ SLERP(alpha)
→ one Milvus image_embedding search
→ exact local cosine ranking
→ deduplication
→ Top-K
```

SLERP mode does not use VLM, TAT, LoRA, directional subtraction, removal penalty, or edit gate.

## API example

```json
{
  "reference": {"path": "/absolute/path/to/reference.webp"},
  "composition_mode": "slerp",
  "edit_text": "pond",
  "remove_text": "lotus flowers",
  "slerp_alpha": 0.8,
  "top_k": 60,
  "use_vlm": false
}
```

The deterministic textual intent becomes:

```text
pond without lotus flowers
```

The old API remains compatible because `composition_mode` defaults to `directional`.

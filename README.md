# Low-Latency Composed Image Retrieval Module

This package implements an interactive **reference frame + edit text → ranked keyframes** workflow.
It has no image generation component.

The same engine is exposed through:

1. `run_cir.py`: one input JSON to one output JSON;
2. `visualize.py`: a lightweight FastAPI web viewer;
3. `scripts/inspect_milvus.py`: schema, index, sample entity, vector, and runtime inspection;
4. `scripts/benchmark_latency.py`: repeated end-to-end latency measurement.

## Retrieval design

For a selected reference image embedding `z_ref`, its stored `text_embedding` `z_source_text`,
and the SigLIP2 embedding of the requested edit `z_target_text`, the default direction is:

```text
direction = normalize(z_target_text - z_source_text)
q_lambda = normalize(z_ref + lambda * direction)
```

The engine searches Milvus with one batch containing:

- the reference image vector;
- the target text vector;
- several directional query vectors.

It unions the ANN candidates and reranks them using:

```text
final = 0.30 * composed
      + 0.22 * target
      + 0.20 * reference_keep
      + 0.18 * direction_consistency
      + 0.10 * candidate_text_metadata
      - negative_penalty
```

All weights, strengths, candidate counts, filters, and deduplication rules are in one YAML file.
This is a practical vector-only baseline, not a claim that the fixed weights are universally optimal.
Tune them on user selections or labeled CIR triplets.

## Package layout

```text
cir_module/
├── cir/                         Core retrieval library
├── examples/                    Input/output JSON examples
├── scripts/
│   ├── inspect_milvus.py        Schema + sample entity inspection
│   ├── validate_setup.py        Connectivity/model validation
│   └── benchmark_latency.py     Repeated latency benchmark
├── web/                         FastAPI template and vanilla JavaScript
├── config.example.yaml          Complete configuration
├── Dockerfile
├── docker-compose.example.yml
├── run_cir.py
└── visualize.py
```

## 1. Prerequisites

Recommended host environment:

- Linux;
- NVIDIA driver compatible with CUDA 12.4 runtime;
- approximately 10 GB GPU memory available;
- Milvus reachable from the process/container;
- frame files mounted at the path defined by `frames.root`;
- internet access for the first Hugging Face download, or a pre-populated cache.

The Dockerfile uses:

```text
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
```

The image contains Python and CUDA-enabled PyTorch. The remaining Python dependencies are installed
from `requirements.txt`.

## 2. Hugging Face model download and HF_HOME

The configured checkpoint is:

```text
google/siglip2-large-patch16-512
```

`AutoProcessor.from_pretrained()` and `AutoModel.from_pretrained()` automatically download it on
first use when it is absent from the cache and the machine can access Hugging Face.

Set the cache before starting the process:

```bash
export HF_HOME=/workingspace_aiclub/cache/huggingface
```

Or set `runtime.hf_home` in `config.yaml`. For offline execution after the model is cached:

```yaml
runtime:
  hf_home: /workingspace_aiclub/cache/huggingface
  offline: true
```

The model must be exactly the model used to build the Milvus gallery embeddings. Matching only the
1024-dimensional output is insufficient; a different checkpoint creates a different vector space.

## 3. Native installation

Create an environment with Python 3.10 or 3.11:

```bash
cd cir_module
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build appropriate for the machine, then install the package dependencies:

```bash
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Create the working config:

```bash
cp config.example.yaml config.yaml
```

Edit at least:

```yaml
milvus:
  uri: http://192.168.20.150:6050
  collection: multimodal_index_siglip_large_full

frames:
  root: /workingspace_aiclub/WorkingSpace/Personal/chinhnm/AIC2026/frames
```

## 4. Docker installation

```bash
cp config.example.yaml config.yaml
mkdir -p hf_cache
docker build -t cir-module:latest .
```

Run the inspector:

```bash
docker run --rm --gpus all --network host \
  -e HF_HOME=/cache/huggingface \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  -v "$PWD/hf_cache:/cache/huggingface" \
  -v /workingspace_aiclub/WorkingSpace/Personal/chinhnm/AIC2026/frames:/workingspace_aiclub/WorkingSpace/Personal/chinhnm/AIC2026/frames:ro \
  cir-module:latest \
  python scripts/inspect_milvus.py \
    --config /app/config.yaml \
    --output /app/milvus_inspection.json
```

For a persistent viewer, use `docker-compose.example.yml` as a starting point:

```bash
cp docker-compose.example.yml docker-compose.yml
docker compose up --build
```

## 5. Inspect Milvus before the first retrieval

Run:

```bash
python scripts/inspect_milvus.py \
  --config config.yaml \
  --sample-size 3 \
  --output milvus_inspection.json
```

The report includes:

- runtime versions, CUDA availability, GPU memory;
- collection schema and primary key;
- vector field dimensions;
- index type and metric;
- load state and collection statistics;
- scalar entity samples;
- sample `image_embedding` and `text_embedding` dimension, norm, and first values;
- missing configured field warnings.

Use it to correct `milvus.fields`. In particular, verify:

```yaml
milvus:
  fields:
    id: id
    image_vector: image_embedding
    text_vector: text_embedding
```

Also match the ANN parameters to the index:

```yaml
# HNSW
search:
  metric_type: COSINE
  params:
    ef: 128

# IVF example
search:
  metric_type: COSINE
  params:
    nprobe: 32
```

## 6. Validate the complete setup

Validate Milvus, frame root, model download, CUDA, and output dimension:

```bash
python scripts/validate_setup.py --config config.yaml
```

To inspect Milvus and paths without downloading/loading SigLIP2:

```bash
python scripts/validate_setup.py --config config.yaml --skip-model
```

## 7. JSON-to-JSON execution

Copy the example and replace the reference ID:

```bash
cp examples/input.example.json input.json
python run_cir.py \
  --config config.yaml \
  --input input.json \
  --output output.json \
  --warmup
```

Reference by Milvus primary key:

```json
{
  "reference": {"id": 123456},
  "edit_text": "người đàn ông đang cầm micro",
  "top_k": 60,
  "use_vlm": false
}
```

Reference by video and frame fields:

```json
{
  "reference": {
    "video_name": "L29_V001",
    "frame_name": "000123.jpg"
  },
  "edit_text": "same scene at night",
  "top_k": 60,
  "use_vlm": false
}
```

Reference by a server-local image path:

```json
{
  "reference": {
    "path": "/absolute/path/to/reference.jpg"
  },
  "edit_text": "the person is holding a microphone",
  "top_k": 60,
  "use_vlm": false
}
```

The local-path mode runs SigLIP2 image encoding once because there is no stored reference vector.
Using a Milvus ID is faster.

### Input overrides

```json
{
  "search": {
    "candidate_k_per_query": 200,
    "max_candidate_pool": 700,
    "strengths": [0.4, 0.7, 1.0],
    "geodesic_alphas": [0.4, 0.7]
  },
  "filters": {
    "milvus_expression": "cluster_id >= 0",
    "include_video_prefixes": ["L29_"],
    "exclude_video_prefixes": ["L26_"],
    "exclude_video_names": ["L29_V999"],
    "exclude_reference": true
  },
  "deduplication": {
    "enabled": true,
    "timestamp_window_seconds": 1.5,
    "max_frames_per_video": 5,
    "max_frames_per_cluster": null
  }
}
```

`edit_strength` replaces the configured list with three probes around the selected value:

```text
strength - 0.25, strength, strength + 0.25
```

Leave it absent to use `composition.directional_strengths`.

## 7.1 Embed the engine in another Python module

For repeated low-latency requests, keep one engine instance alive instead of launching the one-shot
CLI for every query:

```python
import json

from cir.config import load_config
from cir.engine import CIREngine
from cir.schemas import CIRRequest

config = load_config("config.yaml")
engine = CIREngine(config, warmup=True)

payload = {
    "reference": {"id": 123456},
    "edit_text": "người đàn ông đang cầm micro",
    "top_k": 60,
    "use_vlm": False,
}
request = CIRRequest.model_validate(payload)
output = engine.search(request)
print(json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2))
```

The FastAPI `/api/search` route uses the same persistent-engine pattern.

## 8. VLM mode

The configured remote model name is exactly:

```text
Qwen3.5-9B-Q8_0.gguf
```

The client expects an OpenAI-compatible endpoint:

```yaml
vlm:
  base_url: http://127.0.0.1:8000/v1
  chat_completions_path: /chat/completions
  model: Qwen3.5-9B-Q8_0.gguf
```

### VLM disabled

Request:

```json
"use_vlm": false
```

Behavior:

- no HTTP request;
- no endpoint health check;
- no local GGUF loading;
- no VLM timeout;
- only SigLIP2 and Milvus are used.

This is the default and lowest-latency mode.

### VLM enabled

Request:

```json
"use_vlm": true
```

The client asks the endpoint for strict JSON:

```json
{
  "target_description": "The same man on the stage holding a microphone",
  "preserve": ["person", "stage", "viewpoint"],
  "change": ["add a microphone"],
  "negative": []
}
```

By default:

```yaml
send_reference_image: false
```

Therefore, Qwen receives the first available source text from `milvus.raw_text_paths` plus the edit
instruction. Set `send_reference_image: true` only when the actual GGUF server and its multimodal
projector accept OpenAI-style `image_url` content.

If the endpoint fails and this is enabled:

```yaml
fallback_to_no_vlm: true
```

the request continues in vector-only mode and records a warning in the output JSON.

## 9. FastAPI viewer

Start:

```bash
python visualize.py --config config.yaml
```

Open:

```text
http://SERVER_IP:8088
```

The viewer supports:

- reference ID;
- video name + frame name;
- optional server-local image path;
- modification text;
- VLM on/off;
- edit strength;
- deduplication on/off;
- paginated rendering;
- lazy thumbnail loading;
- per-component score inspection;
- JSON output download;
- selecting a returned frame as the next reference.

Only the returned `top_k` entries enter the browser. Images are loaded when approaching the viewport,
so the full frame collection is never inserted into the page.

### Security

`web.allow_local_reference_path: true` allows the browser to request a path on the server. This is
convenient on a trusted internal machine but should be disabled when exposing the service publicly.
The regular `/api/frame` endpoint is restricted to `frames.root`.

## 10. Latency benchmark

```bash
python scripts/benchmark_latency.py \
  --config config.yaml \
  --input input.json \
  --warmup-runs 2 \
  --runs 20
```

The report shows mean, median, minimum, maximum, p95, and the latest component breakdown.

Recommended low-latency settings:

```yaml
runtime:
  device: cuda
  dtype: float16

composition:
  use_geodesic_queries: false
  directional_strengths: [0.35, 0.60, 0.85, 1.10]

retrieval:
  candidate_k_per_query: 100   # increase after measuring recall
  max_candidate_pool: 400
```

Operational recommendations:

- keep one FastAPI worker per GPU so the model is not duplicated;
- keep SigLIP2 resident and use warmup;
- send all query vectors in one Milvus `nq` batch;
- return only lightweight scalar fields during ANN search, then fetch full entities and vectors for unique candidates;
- keep Milvus on the same LAN;
- avoid VLM in the default path;
- mount frames read-only;
- cache thumbnails at a reverse proxy if image transfer dominates perceived latency.

## 11. Output interpretation

Each result exposes:

- `composed`: best similarity among the generated query vectors;
- `target`: similarity to target edit text;
- `reference_keep`: similarity to the selected reference frame;
- `direction`: whether candidate-reference follows the requested edit direction;
- `metadata`: similarity between candidate `text_embedding` and target text;
- `negative_penalty`: penalty for simple remove/without expressions;
- `matched_query`: which reference/text/directional probe retrieved the best match.

The complete vector fields are deliberately omitted from output JSON.

## 12. Common failures

### Text dimension differs from the Milvus image dimension

The wrong SigLIP/SigLIP2 checkpoint is being loaded, or the configured vector field is wrong.
Run `inspect_milvus.py` and confirm both dimensions and the original indexing checkpoint.

### `text_embedding` is absent

The engine falls back to:

```text
direction = normalize(target_text - reference_image)
```

This remains operational but is less semantically clean. Update `milvus.fields.text_vector` if the
collection does contain a matching text vector.

### Search rejects `ef`

The collection may use an IVF index. Replace `ef` with `nprobe` after checking the index report.

### Images rank correctly but do not display

Check:

- `frames.root`;
- `frames.path_template`;
- actual `video_name` and `frame_name` values from the inspection samples;
- container volume mounts.

### CUDA out of memory

The default only loads SigLIP2 Large in float16; Qwen is remote. If another process consumes memory:

- stop it or select another GPU with `CUDA_VISIBLE_DEVICES`;
- reduce concurrent requests;
- keep `workers: 1`;
- use CPU as a functional fallback, accepting higher text-encoding latency.

## Direct `docker run` workflow used on HCMAIC

This package includes `docker_cir.sh`, which runs directly from:

```text
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
```

The source directory, persistent virtual environment and config are expected at:

```text
/workingspace_aiclub/WorkingSpace/Personal/chinhnm/AIC2026/src/core/cir
```

Prepare the config and environment once:

```bash
cd /workingspace_aiclub/WorkingSpace/Personal/chinhnm/AIC2026/src/core/cir
cp config.hcmaic.example.yaml config.yaml
./docker_cir.sh setup
./docker_cir.sh inspect
./docker_cir.sh validate
```

Start the viewer:

```bash
./docker_cir.sh viewer
```

The viewer uses host networking and listens on the `web.host` and `web.port`
values in `config.yaml`, which default to `0.0.0.0:8088`.

The broad `/workingspace_aiclub:/workingspace_aiclub` mount already exposes the
source, `.venv`, config, Hugging Face cache and frames under their original
absolute paths. The script additionally overlays `config.yaml` as a read-only
file mount. No source-code path changes are required.

The persistent `.venv` is created using `--system-site-packages`, so it reuses
the CUDA-enabled PyTorch bundled in the base image. Use it with the same Docker
image. To rebuild it:

```bash
RECREATE_VENV=1 ./docker_cir.sh setup
```

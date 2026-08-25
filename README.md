# Low-Latency Composed Image Retrieval (CIR)

Module này thực hiện **Composed Image Retrieval** trên tập keyframe đã được index trong Milvus.

Đầu vào gồm:

- một ảnh/reference frame;
- nội dung cần **thêm hoặc chuyển thành** qua `edit_text`;
- nội dung cần **loại bỏ** qua `remove_text`.

Đầu ra là danh sách top-K keyframe đã được retrieval, rerank và deduplicate.

> Đây là hệ thống retrieval, không phải image generation, segmentation hay inpainting.  
> `Remove` có nghĩa là tìm frame khác ít mang khái niệm cần loại bỏ hơn, không phải chỉnh sửa trực tiếp pixel của ảnh reference.

---

## 1. Thiết kế retrieval hiện tại

Pipeline mặc định không dùng LLM/VLM:

```text
reference image
+ Edit/Add text vector
- Remove text vector
→ Milvus ANN retrieval
→ vector reranking
→ deduplication
```

Query chính được tạo theo công thức:

```text
direction = normalize(add_vector - remove_vector)
query     = normalize(reference_vector + strength * direction)
```

Tùy request:

```text
Add-only:
query = normalize(reference + strength * add)

Remove-only:
query = normalize(reference - strength * remove)

Replace:
query = normalize(reference + strength * normalize(add - remove))
```

Ảnh reference đóng vai trò **implicit keep signal**, nên không cần field `keep_text`.

### Ý nghĩa của `edit_strength`

`edit_strength` không phải phần trăm edit/remove.

Nó là độ dài bước dịch trong embedding space:

```text
0.50  thay đổi nhẹ, giữ reference nhiều hơn
0.70  mức tương đối an toàn
0.95  mặc định
1.20  thay đổi mạnh hơn, dễ trôi khỏi reference
```

Mỗi request chỉ tạo **một explicit query ứng với đúng strength được chọn**. Hệ thống không tự mở rộng thành `strength ± 0.25`.

### VLM

VLM mặc định tắt:

```yaml
vlm:
  enabled_by_default: false
```

Khi `use_vlm=false`:

- không có HTTP request tới VLM endpoint;
- `edit_text` và `remove_text` được encode trực tiếp bằng SigLIP2;
- đây là chế độ mặc định và có latency thấp nhất.

VLM chỉ được gọi khi request chủ động đặt:

```json
"use_vlm": true
```

---

## 2. Cấu trúc project

```text
cir/
├── main.py                         Ví dụ tích hợp engine trực tiếp trong Python
├── run_cir.py                      CLI: một input JSON → một output JSON
├── visualize.py                    FastAPI API + web viewer
├── config.yaml                     Cấu hình runtime hiện tại
├── config.example.yaml             Cấu hình mẫu, nếu còn được duy trì
├── requirements.txt
├── README.md
│
├── cir/
│   ├── __init__.py
│   ├── config.py                   Load và validate YAML config
│   ├── schemas.py                  Pydantic request/response models
│   ├── encoder.py                  SigLIP2 image/text encoding
│   ├── store_base.py               FrameStore protocol và build_store() factory
│   ├── milvus_store.py             Store backend direct: pymilvus tới Milvus
│   ├── service_store.py            Store backend service: HTTP tới database microservice
│   ├── query_composer.py           Tạo explicit add/remove query
│   ├── reranker.py                 Vector reranking và removal penalty
│   ├── deduplicator.py             Loại frame gần trùng
│   ├── engine.py                   Điều phối toàn bộ pipeline
│   ├── utils.py                    Helper dùng chung
│   └── vlm_client.py               Optional VLM client
│
├── examples/
│   ├── input.example.json
│   ├── input.video_frame.example.json
│   ├── output.example.json
│   └── ...
│
├── scripts/
│   ├── inspect_milvus.py
│   ├── validate_setup.py
│   ├── benchmark_latency.py
│   ├── visualize_cir_scores.py
│   └── visualize_cir_scores_objective.py
│
├── tests/
│   ├── conftest.py
│   └── test_core.py
│
└── web/
    ├── templates/
    │   └── index.html
    └── static/
        ├── app.js
        └── style.css
```

### `examples/` có phải API definition không?

Không.

`examples/` chỉ chứa request/response mẫu để developer xem nhanh hoặc chạy smoke test.

Nguồn định nghĩa chính thức theo thứ tự:

1. `cir/schemas.py`;
2. FastAPI OpenAPI tại `/openapi.json`;
3. FastAPI Swagger UI tại `/docs`;
4. `examples/*.json`;
5. README này.

Khi schema và example không đồng nhất, ưu tiên `cir/schemas.py` và OpenAPI.

---

## 3. Yêu cầu môi trường

Khuyến nghị:

- Linux;
- Python 3.10 hoặc 3.11;
- NVIDIA GPU;
- CUDA-enabled PyTorch;
- khoảng 10–11 GB VRAM trống cho SigLIP2;
- Milvus có thể truy cập từ process/container;
- frame root được mount đúng đường dẫn;
- Hugging Face cache đã có model hoặc máy có internet trong lần chạy đầu.

Model hiện tại:

```text
google/siglip2-large-patch16-512
```

Model dùng lúc query phải khớp với model đã dùng để tạo gallery embedding trong Milvus. Chỉ cùng dimension là chưa đủ.

### Cặp model ↔ collection

`model.name_or_path` và `milvus.collection` là một cặp. Mỗi model có collection riêng:

```text
google/siglip2-large-patch16-512  ->  multimodal_index_siglip_large_v3
```

`model.text_padding` cũng phải khớp với cách gallery embedding được tạo:

```text
SigLIP2         text_padding: max_length   (max_text_length: 64)
CLIP-family     text_padding: true         (max_text_length: 77)
```

Database microservice (`src/apps/database.py`, port `6090`) dùng đúng collection `_v3` này, nên entity id dùng chung được giữa hai service: id do service đó trả về có thể truyền thẳng vào CIR qua `reference.id`.

### Store backend

`milvus.backend` chọn nơi CIR đọc dữ liệu:

```text
direct    pymilvus kết nối thẳng Milvus. Mặc định, đang dùng.
service   HTTP tới database microservice.
```

`service` cần hai endpoint mà microservice **chưa có** (`/v1/search/vector` và `/v1/entities/fetch`); xem `docs/CIR_SERVICE_API.md` mục 15. Khi thiếu, CIR báo lỗi ngay lúc khởi tạo kèm tên endpoint còn thiếu chứ không fail giữa request.

Các collection cũ không có hậu tố `_v3` chứa cùng frame nhưng **primary key khác**. Trỏ sai collection sẽ làm `reference.id` fail âm thầm, trong khi `video_name` + `frame_name` và `path` vẫn chạy.

---

## 4. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Khi cần cài PyTorch CUDA 12.4 riêng:

```bash
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Thiết lập Hugging Face cache:

```bash
export HF_HOME=/workingspace_aiclub/cache/huggingface
```

Hoặc cấu hình trong `config.yaml`:

```yaml
runtime:
  hf_home: /workingspace_aiclub/cache/huggingface
  offline: false
```

---

## 5. Cấu hình tối thiểu

Kiểm tra ít nhất các phần sau trong `config.yaml`:

```yaml
milvus:
  backend: direct
  uri: http://192.168.20.150:6050
  collection: multimodal_index_siglip_large_v3

  fields:
    id: id
    image_vector: image_embedding
    text_vector: text_embedding
    video_name: video_name
    frame_name: frame_name
    timestamp: timestamp
    cluster_id: cluster_id

frames:
  root: /workingspace_aiclub/WorkingSpace/Personal/chinhnm/AIC2026/frames
  path_template: "{frames_root}/{video_name}/{frame_name}.webp"

composition:
  default_edit_strength: 0.95
  explicit_add_weight: 1.0
  explicit_remove_weight: 1.0

object_removal:
  enabled: true
  removal_penalty_weight: 0.35
  expand_aliases: true

vlm:
  enabled_by_default: false

web:
  host: 0.0.0.0
  port: 8029
  workers: 1
```

`0.0.0.0` là địa chỉ bind của server. Khi gọi API từ cùng container, dùng:

```text
http://127.0.0.1:8029
```

---

## 6. Kiểm tra setup

Kiểm tra Milvus schema và sample entities:

```bash
python scripts/inspect_milvus.py \
  --config config.yaml \
  --sample-size 3 \
  --output milvus_inspection.json
```

Kiểm tra Milvus, frame root, model, CUDA và embedding dimension:

```bash
python scripts/validate_setup.py --config config.yaml
```

Bỏ qua bước load model:

```bash
python scripts/validate_setup.py \
  --config config.yaml \
  --skip-model
```

---

## 7. Dùng trực tiếp từ `main.py`

`main.py` là ví dụ đơn giản nhất cho developer muốn nhúng engine vào backend Python.

Các import chính:

```python
from cir.config import load_config
from cir.engine import CIREngine
from cir.schemas import CIRRequest
```

Pattern chuẩn:

```python
from __future__ import annotations

import json

from cir.config import load_config
from cir.engine import CIREngine
from cir.schemas import CIRRequest


def main() -> None:
    config = load_config("config.yaml")

    # Chỉ khởi tạo một lần để không reload model và kết nối Milvus
    engine = CIREngine(config, warmup=True)

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
                "enabled": True
            }
        }
    )

    output = engine.search(request)

    print(
        json.dumps(
            output.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
```

Chạy từ project root:

```bash
source .venv/bin/activate
python main.py
```

Không nên khởi tạo `CIREngine` lại cho từng request. Backend nên giữ một engine instance sống trong suốt vòng đời process.

---

## 8. Dùng dưới dạng thư viện trong backend

Ví dụ service wrapper:

```python
from typing import Any

from cir.config import load_config
from cir.engine import CIREngine
from cir.schemas import CIRRequest


_config = load_config("config.yaml")
_engine = CIREngine(_config, warmup=True)


def search_cir(payload: dict[str, Any]) -> dict[str, Any]:
    request = CIRRequest.model_validate(payload)
    output = _engine.search(request)
    return output.model_dump(mode="json")
```

Backend không nên tự triển khai lại:

- operation detection;
- remove alias expansion;
- query composition;
- removal penalty;
- score normalization;
- deduplication.

Các logic này thuộc core package `cir/`.

---

## 9. Request schema cơ bản

Ít nhất một trong hai field sau phải có nội dung:

```text
edit_text
remove_text
```

### Reference theo Milvus ID

```json
{
  "reference": {
    "id": 123456
  },
  "edit_text": "holding a microphone",
  "remove_text": "",
  "top_k": 20,
  "use_vlm": false,
  "edit_strength": 0.95
}
```

### Reference theo video/frame

```json
{
  "reference": {
    "video_name": "L30_V071",
    "frame_name": "frame_034"
  },
  "edit_text": "pond",
  "remove_text": "lotus",
  "top_k": 20,
  "use_vlm": false,
  "edit_strength": 0.95
}
```

### Reference theo local path

```json
{
  "reference": {
    "path": "/absolute/path/to/reference.webp"
  },
  "edit_text": "helmet",
  "remove_text": "hat",
  "top_k": 20,
  "use_vlm": false,
  "edit_strength": 0.95
}
```

Chỉ được dùng đúng một kiểu reference:

- `id`;
- hoặc `video_name` + `frame_name`;
- hoặc `path`.

Reference bằng Milvus ID/video-frame thường nhanh hơn local path vì có thể tái sử dụng embedding đã lưu.

---

## 10. Ba chế độ query

### Add-only

```json
{
  "reference": {"id": 123456},
  "edit_text": "holding a microphone",
  "remove_text": "",
  "top_k": 20,
  "use_vlm": false,
  "edit_strength": 0.95
}
```

### Remove-only

```json
{
  "reference": {"id": 123456},
  "edit_text": "",
  "remove_text": "hat",
  "top_k": 20,
  "use_vlm": false,
  "edit_strength": 0.70
}
```

Remove-only hợp lệ nhưng có thể kém ổn định vì query chỉ biết cần tránh điều gì mà không có positive destination.

### Replace

```json
{
  "reference": {"id": 123456},
  "edit_text": "helmet",
  "remove_text": "hat",
  "top_k": 20,
  "use_vlm": false,
  "edit_strength": 0.95
}
```

Replace thường ổn định hơn remove-only vì có cả hướng cần thêm và hướng cần tránh.

Không nên điền các từ chung chung như `scene`, `image` hoặc `photo` chỉ để tránh để trống `edit_text`.

---

## 11. Chạy JSON-to-JSON CLI

```bash
python run_cir.py \
  --config config.yaml \
  --input examples/input.example.json \
  --output output.json \
  --warmup
```

`run_cir.py`:

1. load config;
2. validate input bằng `CIRRequest`;
3. khởi tạo `CIREngine`;
4. chạy retrieval;
5. ghi output JSON.

CLI phù hợp với smoke test, batch nhỏ hoặc debugging. Với backend phục vụ nhiều request, nên dùng persistent engine.

---

## 12. FastAPI viewer

Khởi động:

```bash
python visualize.py --config config.yaml
```

Truy cập:

```text
http://SERVER_IP:8029
```

Swagger:

```text
http://SERVER_IP:8029/docs
```

OpenAPI:

```text
http://SERVER_IP:8029/openapi.json
```

Health check trong cùng container:

```bash
curl http://127.0.0.1:8029/health
```

Các endpoint chính:

```text
GET  /health
POST /api/reference
POST /api/search
GET  /api/frame
GET  /
```

Frontend hiện có hai field:

```text
Edit / Add
Remove
```

Static assets và trang root được cấu hình không cache trong giai đoạn phát triển để tránh browser dùng `app.js` cũ.

---

## 13. Output chính

Output top-level thường gồm:

```text
status
request
reference
query
timings_ms
warnings
results
error
```

Mỗi result có thể gồm:

```text
rank
id
video_name
frame_name
timestamp
frame_id
cluster_id
image_path
image_url
score
scores
raw_scores
matched_query
matched_query_strength
metadata
```

### Score components

- `composed`: similarity với explicit composed query;
- `target`: similarity với `Edit/Add`; tên được giữ để tương thích;
- `reference_keep`: mức giữ nội dung reference;
- `direction`: candidate có đi đúng hướng add/remove hay không;
- `metadata`: similarity với candidate text embedding khi có positive edit;
- `edit_score`: score edit tổng hợp dùng cho edit gate;
- `edit_gate_penalty`: phạt candidate không đạt edit;
- `negative_penalty`: phạt negative concepts từ VLM/legacy logic;
- `removal_penalty`: phạt candidate vẫn giống các khái niệm trong `remove_text`.

Trong remove-only mode, `target` và `metadata` không được dùng như positive signal.

### Query metadata

Các field quan trọng:

```text
original_edit_text
original_remove_text
edit_text
target_text
operation
selected_strength
used_vlm
remove_objects
expanded_remove_objects
query_vectors
candidate_pool_size
```

`target_text` được giữ để tương thích với client cũ nhưng trong explicit mode có thể là `null` và không tham gia retrieval.

Tên query thường có dạng:

```text
reference
edit_text
explicit_edit_0.950
explicit_remove_0.950
explicit_replace_0.950
```

---

## 14. Latency

Output có breakdown:

```text
reference_lookup
vlm
text_encoding
milvus_search
candidate_fetch
reranking
deduplication
total
```

Benchmark:

```bash
python scripts/benchmark_latency.py \
  --config config.yaml \
  --input examples/input.example.json \
  --warmup-runs 2 \
  --runs 20
```

Khuyến nghị:

- giữ một engine instance;
- giữ SigLIP2 resident trên GPU;
- `workers: 1` cho mỗi GPU;
- warm up trước khi nhận request;
- dùng Milvus ID/video-frame khi có thể;
- giữ VLM tắt trong default path;
- chỉ trả scalar fields cần thiết từ ANN search;
- để Milvus trên cùng LAN.

---

## 15. Testing

Chạy test:

```bash
PYTHONPATH=. pytest -q
```

Các case tối thiểu nên được bảo vệ:

- add-only;
- remove-only;
- replace;
- cả `edit_text` và `remove_text` cùng trống bị reject;
- VLM không tự động được gọi;
- một request chỉ dùng đúng selected strength;
- removal penalty làm giảm final score;
- legacy request vẫn tương thích.

---

## 16. Lưu ý cho frontend/backend

### Frontend

Frontend chỉ nên:

- thu nhận reference;
- thu nhận `edit_text` và `remove_text`;
- gửi request;
- hiển thị output;
- hiển thị score breakdown khi cần debug.

Frontend không tự tính query vector hay score.

### Backend

Backend nên:

- tạo `CIREngine` đúng một lần;
- validate request qua `CIRRequest`;
- quản lý lifecycle và concurrency;
- cung cấp API contract từ OpenAPI;
- log timing và warning;
- không duplicate core retrieval logic.

---

## 17. Common issues

### `ModuleNotFoundError: No module named 'cir'`

Chạy script từ project root:

```bash
PYTHONPATH=. python path/to/script.py
```

Hoặc cài project dưới dạng package trước khi chạy từ thư mục khác.

### Hugging Face trả `404` cho `additional_chat_templates`

Đây thường không phải lỗi load model nếu model weights vẫn được load thành công.

### Installer ghi `node unavailable`

FastAPI không phụ thuộc Node.js để chạy.

Thông báo đó chỉ có nghĩa installer không chạy được bước kiểm tra cú pháp JavaScript bằng Node. `app.js` vẫn được browser thực thi.

### Browser hiển thị logic cũ

Kiểm tra:

- server đã restart;
- static asset URL đang được serve đúng;
- response có no-cache/no-store;
- browser hard refresh một lần sau khi deploy.

### `Remove` không cho kết quả như mong đợi

`Remove` là vector retrieval, không phải object erasing.

Thử:

- thêm một positive anchor vào `edit_text`;
- giảm `edit_strength`;
- mô tả remove concept rõ hơn;
- kiểm tra `removal_penalty`;
- kiểm tra prompt-image alignment của SigLIP2.

Ví dụ tốt hơn:

```text
Edit/Add: pond
Remove: lotus
```

thay vì chỉ:

```text
Remove: lotus
```

### Ảnh rank đúng nhưng không hiển thị

Kiểm tra:

- `frames.root`;
- `frames.path_template`;
- extension thực tế;
- `video_name` và `frame_name`;
- Docker volume mount.

### CUDA out of memory

- chọn GPU khác bằng `CUDA_VISIBLE_DEVICES`;
- giảm concurrent request;
- giữ một worker trên mỗi GPU;
- kiểm tra process khác đang chiếm VRAM.

---

## 18. Security

`web.allow_local_reference_path: true` cho phép client tham chiếu path nằm trên server.

Chỉ bật trong môi trường nội bộ đáng tin cậy. Khi public service:

```yaml
web:
  allow_local_reference_path: false
```

Nên thêm authentication, request size limit và path validation ở tầng backend/reverse proxy.

---

## Training-free SLERP mode

<!-- PNP_CIRR_SLERP_TRAINING_FREE_V1 -->

SLERP được triển khai tách biệt trong `cir/slerp_method/`. Pipeline directional cũ vẫn là mặc định và không đổi hành vi.

Request chọn method bằng:

```json
{
  "composition_mode": "slerp",
  "edit_text": "pond",
  "remove_text": "lotus flowers",
  "slerp_alpha": 0.8
}
```

SLERP mode thực hiện:

```text
reference image embedding
+ one deterministic textual-intent embedding
→ spherical linear interpolation
→ one Milvus cosine search
→ exact local cosine reranking
→ deduplication
→ Top-K
```

Nó không gọi VLM, không dùng directional `add-remove`, không dùng removal penalty, edit gate hoặc TAT/LoRA.

---

## SLERP spherical remove (experimental)

Ba composition mode được tách riêng:

- `directional`: pipeline Add/Remove hiện tại, không thay đổi;
- `slerp`: pure training-free SLERP giữa reference image và positive `edit_text`;
- `slerp_remove`: tạo positive anchor tùy chọn bằng SLERP, sau đó đi khỏi
  `remove_text` theo một geodesic angle `slerp_remove_gamma`.

Remove-only:

```text
q = spherical_move_away(reference, remove, gamma)
```

Add + Remove:

```text
anchor = slerp(reference, add, alpha)
q = spherical_move_away(anchor, remove, gamma)
```

`gamma` là góc tính theo radian; vùng thử ban đầu nên là `0.10–0.30`.
Đây là extension training-free để thử nghiệm, không phải công thức remove được
đề xuất trong paper SLERP-TAT.


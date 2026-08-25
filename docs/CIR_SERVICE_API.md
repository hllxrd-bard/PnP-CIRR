# PnP-CIRR Service API

## 1. Mục đích

Tài liệu này định nghĩa service contract chính thức để backend gọi Composed Image Retrieval (CIR). Service nhận một reference frame cùng yêu cầu thêm/xóa nội dung, tạo query embedding, tìm trong Milvus, rerank, deduplicate và trả danh sách keyframe đã xếp hạng.

Service này là retrieval service. Nó không sửa pixel, không inpainting và không sinh ảnh.

## 2. Kiến trúc triển khai hiện tại

```text
Backend chính
    |
    | HTTP /v1/*
    v
PnP-CIRR Service
    |- SigLIP2 encoder
    |- Directional / pure Slerp / Slerp hybrid composition
    |- Local reranking và deduplication
    |
    | pymilvus
    v
Milvus
```

CIR Service kết nối trực tiếp Milvus bằng `pymilvus`. URI Milvus, collection, field mapping, frame root, model và hyperparameter mặc định nằm trong `config.yaml` của service. Backend không gửi các cấu hình hạ tầng này theo từng request.

## 3. Khởi chạy

```bash
python service.py \
  --config config.yaml \
  --host 0.0.0.0 \
  --port 8088
```

Swagger UI:

```text
http://<cir-host>:8088/docs
```

OpenAPI:

```text
http://<cir-host>:8088/openapi.json
```

Service dùng một worker vì encoder GPU được giữ trong process. Khi cần scale, chạy nhiều replica độc lập và đặt load balancer phía trước.

## 4. Endpoint

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/health` | Process liveness |
| `GET` | `/ready` | Kiểm tra Milvus, encoder và frame root |
| `GET` | `/v1/capabilities` | Mode, defaults, limits và embedding contract |
| `POST` | `/v1/references/resolve` | Resolve reference và kiểm tra metadata |
| `POST` | `/v1/search` | CIR search chính |
| `GET` | `/v1/frames` | Trả frame từ frame root |
| `GET` | `/v1/local-frames` | Trả ảnh path local khi được bật |

Mỗi response có header `X-Request-ID`. Backend có thể gửi sẵn header này; nếu không, service tự sinh.

## 5. Reference contract

Reference phải có đúng một locator:

### 5.1. ID — khuyến nghị

```json
{
  "reference": {
    "id": 123456
  }
}
```

Service lấy metadata và `image_embedding` đã index từ Milvus. Đây là cách backend nên dùng mặc định vì không phải đọc và encode lại ảnh.

### 5.2. Video và frame name

```json
{
  "reference": {
    "video_name": "L30_V071",
    "frame_name": "frame_034.webp"
  }
}
```

Service resolve entity và embedding từ Milvus.

### 5.3. Path

```json
{
  "reference": {
    "path": "/shared/frames/L30_V071/frame_034.webp"
  }
}
```

Service đọc ảnh và encode bằng model đang chạy. Path phải nhìn thấy được từ process/container CIR.

### 5.4. Optional direct embedding fast path

```json
{
  "reference": {
    "id": 123456,
    "image_embedding": [0.0123, -0.0456],
    "embedding_model": "google/siglip2-large-patch16-512",
    "embedding_dimension": 1024
  }
}
```

Quy tắc:

1. Có `image_embedding`: service validate và dùng trực tiếp.
2. Không có embedding nhưng có `id`: lấy embedding từ Milvus.
3. Không có `id` nhưng có `video_name + frame_name`: resolve từ Milvus.
4. Chỉ có `path`: encode ảnh.

Service kiểm tra direct embedding:

- dimension khai báo bằng dimension hiện tại của collection;
- số phần tử đúng dimension;
- mọi giá trị hữu hạn, không NaN/Inf;
- norm lớn hơn epsilon;
- model name khớp model CIR;
- L2-normalize lại trước khi dùng.

Backend nên truyền `reference.id`. Không nên truyền embedding trừ khi embedding đã có sẵn và được sinh bởi đúng model/version của CIR Service.

## 6. Composition modes

### 6.1. `directional`

Hỗ trợ edit-only, remove-only hoặc cả hai.

```text
Add-only:    normalize(reference + strength * add)
Remove-only: normalize(reference - strength * remove)
Replace:     normalize(reference + strength * normalize(add - remove))
```

Validation: ít nhất một trong `edit_text`, `remove_text` phải có nội dung.

### 6.2. `pure_slerp`

Training-free Slerp giữa reference image embedding và một positive textual intent.

- `edit_text`: bắt buộc, phải là full textual intent.
- `remove_text`: không được hỗ trợ.
- `slerp_alpha`: trong `[0, 1]`.

Ví dụ textual intent: `an open pond with clear water`.

### 6.3. `slerp_hybrid`

Experimental mode:

1. Nếu có `edit_text`, Slerp từ reference về positive edit anchor.
2. Di chuyển trên hypersphere ra xa negative concept trong `remove_text`.

- `remove_text`: bắt buộc.
- `edit_text`: optional.
- `slerp_alpha`: add-anchor strength.
- `slerp_hybrid_gamma`: góc move-away.

### 6.4. Deprecated aliases

Service vẫn nhận để tương thích:

```text
slerp        -> pure_slerp
slerp_remove -> slerp_hybrid
```

Response luôn dùng tên canonical và trả warning deprecated.

## 7. Search request

```json
{
  "reference": {
    "id": 123456
  },
  "composition_mode": "directional",
  "edit_text": "helmet",
  "remove_text": "hat",
  "top_k": 60,
  "edit_strength": 0.95,
  "slerp_alpha": null,
  "slerp_hybrid_gamma": null,
  "use_vlm": false,
  "vlm_provider": null,
  "search": {
    "candidate_k_per_query": null,
    "max_candidate_pool": null
  },
  "filters": {
    "exclude_reference": true,
    "include_video_prefixes": [],
    "exclude_video_prefixes": [],
    "exclude_video_names": []
  },
  "deduplication": {
    "enabled": true,
    "timestamp_window_seconds": null,
    "max_frames_per_video": null,
    "max_frames_per_cluster": null
  }
}
```

Field bị bỏ hoặc `null` dùng server default. Effective defaults lấy từ `GET /v1/capabilities`.

Public `/v1/search` không nhận raw `milvus_expression`. Backend chỉ dùng simple filters đã định nghĩa.

## 8. Search response

```json
{
  "status": "success",
  "request_id": "cir_01J...",
  "service_version": "1.0.0",
  "composition_mode": "directional",
  "request": {},
  "reference": {
    "id": 123456,
    "video_name": "L30_V071",
    "frame_name": "frame_034.webp",
    "timestamp": 12.8,
    "image_path": "/shared/frames/L30_V071/frame_034.webp",
    "image_url": "/v1/frames?video_name=L30_V071&frame_name=frame_034.webp",
    "embedding_source": "milvus"
  },
  "query": {
    "operation": "replace",
    "selected_strength": 0.95,
    "candidate_pool_size": 438
  },
  "timings_ms": {
    "reference_lookup": 5.2,
    "vlm": 0.0,
    "text_encoding": 16.4,
    "milvus_search": 71.2,
    "candidate_fetch": 25.7,
    "reranking": 4.8,
    "deduplication": 1.6,
    "total": 126.1
  },
  "warnings": [],
  "results": [
    {
      "rank": 1,
      "id": 987654,
      "video_name": "L29_V123",
      "frame_name": "frame_081.webp",
      "timestamp": 32.4,
      "frame_id": 81,
      "cluster_id": "cluster_091",
      "image_path": "/shared/frames/L29_V123/frame_081.webp",
      "image_url": "/v1/frames?video_name=L29_V123&frame_name=frame_081.webp",
      "score": 0.8231,
      "scores": {},
      "raw_scores": {},
      "matched_query": "explicit_replace_0.950",
      "best_ann_query": "explicit_replace_0.950",
      "retrieved_by": [
        "reference",
        "edit_text",
        "explicit_replace_0.950"
      ],
      "metadata": {}
    }
  ]
}
```

`embedding_source` có thể là:

- `milvus`;
- `encoded_path`;
- `request`.

## 9. Validation matrix

| Mode | `edit_text` | `remove_text` | Kết quả |
|---|---:|---:|---|
| directional | có | không | hợp lệ |
| directional | không | có | hợp lệ |
| directional | có | có | hợp lệ |
| directional | không | không | 422 |
| pure_slerp | có | không | hợp lệ |
| pure_slerp | có | có | 409 |
| pure_slerp | không | bất kỳ | 422 |
| slerp_hybrid | optional | có | hợp lệ |
| slerp_hybrid | bất kỳ | không | 422 |

## 10. Error response và HTTP status

```json
{
  "status": "error",
  "request_id": "cir_01J...",
  "error": {
    "code": "UNSUPPORTED_MODE_COMBINATION",
    "message": "pure_slerp does not support remove_text.",
    "details": {
      "suggested_mode": "slerp_hybrid"
    }
  }
}
```

| HTTP | Code điển hình | Ý nghĩa |
|---:|---|---|
| 400 | `INVALID_REQUEST` | Request không hợp lệ ở mức giao thức |
| 404 | `REFERENCE_NOT_FOUND` | Không tìm thấy reference/path |
| 409 | `UNSUPPORTED_MODE_COMBINATION` | Field không được mode hỗ trợ |
| 415 | `UNSUPPORTED_IMAGE_TYPE` | Extension ảnh không hỗ trợ |
| 422 | `VALIDATION_ERROR`, `TOP_K_EXCEEDS_LIMIT`, embedding errors | Schema/semantic validation |
| 502 | `MILVUS_ERROR` | Milvus trả lỗi |
| 503 | `SERVICE_NOT_READY` | Encoder/storage chưa sẵn sàng |
| 504 | `MILVUS_TIMEOUT` | Timeout storage |
| 500 | `INTERNAL_ERROR` | Lỗi không phân loại |

## 11. Curl examples

### Directional replace

```bash
curl -sS -X POST http://127.0.0.1:8088/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "reference": {"id": 123456},
    "composition_mode": "directional",
    "edit_text": "helmet",
    "remove_text": "hat",
    "top_k": 60
  }'
```

### Pure Slerp

```bash
curl -sS -X POST http://127.0.0.1:8088/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "reference": {"id": 123456},
    "composition_mode": "pure_slerp",
    "edit_text": "an open pond with clear water",
    "slerp_alpha": 0.8,
    "top_k": 60
  }'
```

### Slerp hybrid removal

```bash
curl -sS -X POST http://127.0.0.1:8088/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "reference": {"id": 123456},
    "composition_mode": "slerp_hybrid",
    "edit_text": "pond with green leaves",
    "remove_text": "pink lotus flowers",
    "slerp_alpha": 0.4,
    "slerp_hybrid_gamma": 0.2,
    "top_k": 60
  }'
```

## 12. Python client example

```python
import httpx

payload = {
    "reference": {"id": 123456},
    "composition_mode": "directional",
    "edit_text": "helmet",
    "remove_text": "hat",
    "top_k": 60,
}

with httpx.Client(base_url="http://cir-service:8088", timeout=10.0) as client:
    response = client.post("/v1/search", json=payload)
    response.raise_for_status()
    results = response.json()["results"]
```

## 13. Backend integration recommendation

1. Backend lưu và truyền Milvus frame ID làm `reference.id`.
2. Backend không cần truyền path hoặc embedding trong luồng bình thường.
3. Backend gọi `/ready` khi health-check deployment.
4. Backend đọc defaults từ `/v1/capabilities`, không hardcode giới hạn.
5. Backend đặt timeout cao hơn latency p95 của CIR và log `X-Request-ID`.
6. VLM để tắt theo mặc định vì làm tăng latency đáng kể.
7. Không expose `/v1/search` trực tiếp ra public internet nếu chưa có authentication/rate limiting ở gateway.

## 14. Legacy viewer

`visualize.py` và các endpoint `/api/*` hiện tại không bị thay đổi. `service.py` là entrypoint riêng cho backend integration. Điều này giúp service contract ổn định mà không làm hỏng web viewer đang dùng để debug.

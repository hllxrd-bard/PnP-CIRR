# PnP-CIRR multi-provider VLM patch v2

Bản v2 sửa installer để tương thích với cấu trúc `app.js` hiện tại và thêm lựa chọn VLM provider cho `directional` mode:

- `qwen`: Qwen 3.5 9B local hiện tại
- `gemini`: Gemini OpenAI-compatible API

Không thay đổi logic directional, pure SLERP hoặc spherical remove.

## Qwen mặc định

```yaml
vlm:
  enabled_by_default: false
  default_provider: qwen
  base_url: http://192.168.20.150:8018/v1
  chat_completions_path: /chat/completions
  api_key: null
  model: Qwen3.5-9B-Q8_0.gguf
```

Flat `vlm:` block hiện tại tiếp tục được hiểu là profile Qwen, nên config cũ vẫn tương thích.

## Gemini mặc định

Router có sẵn profile:

```yaml
vlm:
  providers:
    gemini:
      base_url: https://generativelanguage.googleapis.com/v1beta/openai
      chat_completions_path: /chat/completions
      model: gemini-3.6-flash
      timeout_seconds: 90
      max_tokens: 1024
      reasoning_effort: low
```

API key không nằm trong YAML hoặc browser payload. Export trước khi chạy service:

```bash
read -rsp 'Gemini API key: ' GEMINI_API_KEY
export GEMINI_API_KEY
echo
python visualize.py --config config.yaml
```

## API payload

Qwen:

```json
{
  "reference": {"path": "/absolute/path/to/reference.webp"},
  "composition_mode": "directional",
  "edit_text": "",
  "remove_text": "lotus flowers",
  "use_vlm": true,
  "vlm_provider": "qwen",
  "edit_strength": 0.95,
  "top_k": 60
}
```

Gemini:

```json
{
  "reference": {"path": "/absolute/path/to/reference.webp"},
  "composition_mode": "directional",
  "edit_text": "",
  "remove_text": "lotus flowers",
  "use_vlm": true,
  "vlm_provider": "gemini",
  "edit_strength": 0.95,
  "top_k": 60
}
```

Response query metadata có thêm:

```json
{
  "used_vlm": true,
  "vlm_provider": "gemini",
  "vlm_model": "gemini-3.6-flash",
  "vlm_http_latency_ms": 1824.3
}
```

`timings_ms.vlm` vẫn là VLM end-to-end latency ở engine; `vlm_http_latency_ms` là riêng HTTP call.

## File mới

```text
cir/vlm/
├── __init__.py
├── common.py
├── providers.py
└── router.py

tests/test_vlm_providers.py
examples/input.vlm.qwen.json
examples/input.vlm.gemini.json
```

## File tích hợp nhẹ

```text
cir/vlm_client.py
cir/config.py
cir/schemas.py
cir/engine.py
visualize.py
web/templates/index.html
web/static/app.js
```

Không thêm dependency mới. Repo hiện tại đã dùng `httpx`, Python và PyYAML.

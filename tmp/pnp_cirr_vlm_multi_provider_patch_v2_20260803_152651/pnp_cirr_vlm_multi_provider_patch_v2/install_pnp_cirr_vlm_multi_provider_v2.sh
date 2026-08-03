#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(pwd)"
TMP_DIR="$ROOT_DIR/tmp"
mkdir -p "$TMP_DIR"

PYTHON_BIN=""
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Không tìm thấy Python. Cài bằng: apt-get update && apt-get install -y python3" >&2
  exit 1
fi

ZIP_ARG="${1:-}"
if [[ -z "$ZIP_ARG" ]]; then
  if [[ -f "$ROOT_DIR/pnp_cirr_vlm_multi_provider_patch_v2.zip" ]]; then
    ZIP_ARG="$ROOT_DIR/pnp_cirr_vlm_multi_provider_patch_v2.zip"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$SCRIPT_DIR/pnp_cirr_vlm_multi_provider_patch_v2.zip" ]]; then
      ZIP_ARG="$SCRIPT_DIR/pnp_cirr_vlm_multi_provider_patch_v2.zip"
    else
      echo "Usage: $0 /path/to/pnp_cirr_vlm_multi_provider_patch_v2.zip" >&2
      exit 2
    fi
  fi
fi

ZIP_SOURCE="$(realpath "$ZIP_ARG")"
if [[ ! -f "$ZIP_SOURCE" ]]; then
  echo "Không tìm thấy ZIP: $ZIP_SOURCE" >&2
  exit 2
fi

ZIP_TARGET="$TMP_DIR/pnp_cirr_vlm_multi_provider_patch_v2.zip"
if [[ "$ZIP_SOURCE" != "$ZIP_TARGET" ]]; then
  mv -f "$ZIP_SOURCE" "$ZIP_TARGET"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
EXTRACT_DIR="$TMP_DIR/pnp_cirr_vlm_multi_provider_patch_v2_$STAMP"
BACKUP_DIR="$TMP_DIR/vlm_multi_provider_backup_$STAMP"
mkdir -p "$EXTRACT_DIR"

"$PYTHON_BIN" - "$ZIP_TARGET" "$EXTRACT_DIR" <<'PY'
from pathlib import Path
import sys
import zipfile

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(out_dir)
PY

PAYLOAD_DIR="$EXTRACT_DIR/pnp_cirr_vlm_multi_provider_patch_v2"
if [[ ! -f "$PAYLOAD_DIR/patch.py" ]]; then
  echo "ZIP không đúng cấu trúc: thiếu patch.py" >&2
  exit 3
fi

restore_backup() {
  local rel
  for rel in \
    cir/config.py \
    cir/schemas.py \
    cir/engine.py \
    cir/vlm_client.py \
    visualize.py \
    web/templates/index.html \
    web/static/app.js; do
    if [[ -f "$BACKUP_DIR/$rel" ]]; then
      mkdir -p "$(dirname "$ROOT_DIR/$rel")"
      cp -f "$BACKUP_DIR/$rel" "$ROOT_DIR/$rel"
    fi
  done
  rm -rf "$ROOT_DIR/cir/vlm"
  rm -f \
    "$ROOT_DIR/tests/test_vlm_providers.py" \
    "$ROOT_DIR/examples/input.vlm.qwen.json" \
    "$ROOT_DIR/examples/input.vlm.gemini.json"
}

trap 'echo "Cài đặt lỗi; đang khôi phục file cũ..." >&2; restore_backup' ERR

"$PYTHON_BIN" "$PAYLOAD_DIR/patch.py" \
  --repo "$ROOT_DIR" \
  --payload "$PAYLOAD_DIR" \
  --backup "$BACKUP_DIR"

"$PYTHON_BIN" -m compileall -q \
  "$ROOT_DIR/cir" \
  "$ROOT_DIR/visualize.py"

"$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [name for name in ("httpx", "yaml") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Thiếu Python package đã vốn cần bởi project: " + ", ".join(missing)
    )
PY

if "$PYTHON_BIN" -c 'import pytest' >/dev/null 2>&1; then
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -m pytest -q \
    "$ROOT_DIR/tests/test_vlm_providers.py"
else
  echo "pytest không có trong environment; bỏ qua unit test, compile check đã pass."
fi

if command -v node >/dev/null 2>&1; then
  node --check "$ROOT_DIR/web/static/app.js"
else
  echo "Node.js không có; bỏ qua JS syntax check. Node không cần cho runtime FastAPI."
fi

trap - ERR

echo
echo "========== VLM MULTI-PROVIDER PATCH V2 DONE =========="
echo "ZIP:      $ZIP_TARGET"
echo "Extract:  $EXTRACT_DIR"
echo "Backup:   $BACKUP_DIR"
echo
echo "Qwen local mặc định:"
echo "  http://192.168.20.150:8018/v1"
echo "  Qwen3.5-9B-Q8_0.gguf"
echo
echo "Để dùng Gemini, export key trong cùng shell chạy service:"
echo "  read -rsp 'Gemini API key: ' GEMINI_API_KEY"
echo "  export GEMINI_API_KEY"
echo "  echo"
echo
echo "Sau đó restart:"
echo "  python visualize.py --config config.yaml"

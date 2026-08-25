#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-./pnp_cirr_service_patch.zip}"
PROJECT_ROOT="$(pwd)"
TMP_ROOT="$PROJECT_ROOT/tmp"
STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="$TMP_ROOT/pnp_cirr_service_patch_$STAMP"
BACKUP_DIR="$TMP_ROOT/pnp_cirr_service_backup_$STAMP"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Patch ZIP not found: $ZIP_PATH" >&2
  exit 2
fi
if [[ ! -f "$PROJECT_ROOT/cir/engine.py" || ! -f "$PROJECT_ROOT/cir/schemas.py" ]]; then
  echo "Run this installer from the PnP-CIRR project root." >&2
  exit 2
fi
if ! command -v python >/dev/null 2>&1; then
  echo "Python is required but was not found." >&2
  exit 2
fi

mkdir -p "$TMP_ROOT" "$WORK_DIR" "$BACKUP_DIR"
ZIP_BASENAME="$(basename "$ZIP_PATH")"
ZIP_ABS="$(python - <<'PY' "$ZIP_PATH"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
TARGET_ZIP="$TMP_ROOT/$ZIP_BASENAME"
if [[ "$ZIP_ABS" != "$(python - <<'PY' "$TARGET_ZIP"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)" ]]; then
  mv "$ZIP_PATH" "$TARGET_ZIP"
else
  TARGET_ZIP="$ZIP_PATH"
fi

python - <<'PY' "$TARGET_ZIP" "$WORK_DIR"
from pathlib import Path
from zipfile import ZipFile
import sys
zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
PY

PAYLOAD="$WORK_DIR/pnp_cirr_service_patch"
if [[ ! -d "$PAYLOAD" ]]; then
  echo "Invalid patch ZIP: payload directory missing." >&2
  exit 2
fi

FILES=(
  "service.py"
  "cir/service_schemas.py"
  "cir/service_engine.py"
  "docs/CIR_SERVICE_API.md"
  "tests/test_service_contract.py"
  "examples/v1.search.directional.json"
  "examples/v1.search.pure_slerp.json"
  "examples/v1.search.slerp_hybrid.json"
)

restore() {
  for rel in "${FILES[@]}"; do
    if [[ -f "$BACKUP_DIR/$rel" ]]; then
      mkdir -p "$PROJECT_ROOT/$(dirname "$rel")"
      cp -a "$BACKUP_DIR/$rel" "$PROJECT_ROOT/$rel"
    elif [[ -f "$PROJECT_ROOT/$rel" ]]; then
      rm -f "$PROJECT_ROOT/$rel"
    fi
  done
}
trap 'echo "Installation failed; restoring previous files." >&2; restore' ERR

for rel in "${FILES[@]}"; do
  if [[ -f "$PROJECT_ROOT/$rel" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$rel")"
    cp -a "$PROJECT_ROOT/$rel" "$BACKUP_DIR/$rel"
  fi
  mkdir -p "$PROJECT_ROOT/$(dirname "$rel")"
  cp -a "$PAYLOAD/$rel" "$PROJECT_ROOT/$rel"
done

python -m py_compile \
  service.py \
  cir/service_schemas.py \
  cir/service_engine.py

if command -v pytest >/dev/null 2>&1; then
  PYTHONPATH=. pytest -q tests/test_service_contract.py
else
  echo "pytest not found; skipped unit tests."
fi

trap - ERR
echo
echo "PnP-CIRR service patch installed successfully."
echo "Patch ZIP: $TARGET_ZIP"
echo "Backup:    $BACKUP_DIR"
echo
echo "Start service:"
echo "  python service.py --config config.yaml --host 0.0.0.0 --port 8088"
echo
echo "Docs: docs/CIR_SERVICE_API.md"

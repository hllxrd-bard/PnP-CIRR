#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
ZIP_ARG="${1:-./pnp_cirr_slerp_training_free_patch.zip}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TMP_ROOT="$PROJECT_ROOT/tmp"
BACKUP_ROOT="$PROJECT_ROOT/backups/slerp_training_free_${TIMESTAMP}"
EXTRACT_ROOT="$TMP_ROOT/slerp_patch_extract_${TIMESTAMP}"
MOVED_ZIP="$TMP_ROOT/$(basename "$ZIP_ARG")"
INSTALL_LOG="$TMP_ROOT/slerp_patch_install_${TIMESTAMP}.log"
INSTALLED=0

mkdir -p "$TMP_ROOT" "$BACKUP_ROOT"
cd "$PROJECT_ROOT"

log() {
  printf '%s\n' "$*" | tee -a "$INSTALL_LOG"
}

fail() {
  log "ERROR: $*"
  exit 1
}

command -v python >/dev/null 2>&1 || fail "python is required. Run install_pnp_cirr_slerp_prereqs.sh first."

[[ -f "$ZIP_ARG" ]] || fail "Patch ZIP not found: $ZIP_ARG"

ZIP_ABS="$(python - "$ZIP_ARG" <<'PYRESOLVE'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PYRESOLVE
)"
MOVED_ABS="$(python - "$MOVED_ZIP" <<'PYRESOLVE'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PYRESOLVE
)"
if [[ "$ZIP_ABS" != "$MOVED_ABS" ]]; then
  mv "$ZIP_ARG" "$MOVED_ZIP"
fi
log "Patch ZIP moved to: $MOVED_ZIP"

rm -rf "$EXTRACT_ROOT"
mkdir -p "$EXTRACT_ROOT"
python - "$MOVED_ZIP" "$EXTRACT_ROOT" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
with ZipFile(source) as archive:
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination.resolve() not in target.parents and target != destination.resolve():
            raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
    archive.extractall(destination)
PY

BUNDLE="$EXTRACT_ROOT/pnp_cirr_slerp_training_free_patch"
[[ -f "$BUNDLE/apply_patch.py" ]] || fail "Invalid bundle: apply_patch.py missing"
[[ -d "$BUNDLE/payload/cir/slerp_method" ]] || fail "Invalid bundle: SLERP payload missing"

BACKUP_PATHS=(
  "cir/engine.py"
  "cir/schemas.py"
  "cir/config.py"
  "config.yaml"
  "visualize.py"
  "web/templates/index.html"
  "web/static/app.js"
  "web/static/style.css"
  "README.md"
  "cir/slerp_method"
  "tests/test_slerp.py"
  "examples/input.slerp.example.json"
)

mkdir -p "$BACKUP_ROOT/files"
: > "$BACKUP_ROOT/PRESENT_PATHS.txt"
: > "$BACKUP_ROOT/ABSENT_PATHS.txt"
for relative in "${BACKUP_PATHS[@]}"; do
  if [[ -e "$PROJECT_ROOT/$relative" ]]; then
    printf '%s\n' "$relative" >> "$BACKUP_ROOT/PRESENT_PATHS.txt"
    mkdir -p "$BACKUP_ROOT/files/$(dirname "$relative")"
    cp -a "$PROJECT_ROOT/$relative" "$BACKUP_ROOT/files/$relative"
  else
    printf '%s\n' "$relative" >> "$BACKUP_ROOT/ABSENT_PATHS.txt"
  fi
done

cat > "$BACKUP_ROOT/restore.sh" <<'RESTORE'
#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${1:-$(cd "$HERE/../.." && pwd)}"

while IFS= read -r relative; do
  [[ -n "$relative" ]] || continue
  rm -rf "$PROJECT_ROOT/$relative"
  mkdir -p "$PROJECT_ROOT/$(dirname "$relative")"
  cp -a "$HERE/files/$relative" "$PROJECT_ROOT/$relative"
done < "$HERE/PRESENT_PATHS.txt"

while IFS= read -r relative; do
  [[ -n "$relative" ]] || continue
  rm -rf "$PROJECT_ROOT/$relative"
done < "$HERE/ABSENT_PATHS.txt"

echo "Restored project files from: $HERE"
RESTORE
chmod +x "$BACKUP_ROOT/restore.sh"

rollback() {
  status=$?
  if [[ $status -ne 0 && $INSTALLED -eq 1 ]]; then
    log "Installation failed; rolling back from $BACKUP_ROOT"
    "$BACKUP_ROOT/restore.sh" "$PROJECT_ROOT" | tee -a "$INSTALL_LOG" || true
  fi
  exit "$status"
}
trap rollback EXIT

log "Applying SLERP integration..."
INSTALLED=1
python "$BUNDLE/apply_patch.py" \
  --project-root "$PROJECT_ROOT" \
  --payload-root "$BUNDLE/payload" | tee -a "$INSTALL_LOG"

log "Compiling Python files..."
PYTHONPATH="$PROJECT_ROOT" python -m compileall -q \
  "$PROJECT_ROOT/cir" \
  "$PROJECT_ROOT/visualize.py"

log "Validating YAML..."
python - "$PROJECT_ROOT/config.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8"))
assert isinstance(data, dict)
assert isinstance(data.get("slerp"), dict)
assert 0.0 <= float(data["slerp"]["default_alpha"]) <= 1.0
print("YAML OK")
PY

if PYTHONPATH="$PROJECT_ROOT" python -c 'import pytest' >/dev/null 2>&1; then
  log "Running SLERP focused tests..."
  PYTHONPATH="$PROJECT_ROOT" python -m pytest -q \
    "$PROJECT_ROOT/tests/test_slerp.py" | tee -a "$INSTALL_LOG"
  if [[ "${RUN_FULL_TESTS:-0}" == "1" ]]; then
    log "RUN_FULL_TESTS=1: running existing core tests..."
    PYTHONPATH="$PROJECT_ROOT" python -m pytest -q \
      "$PROJECT_ROOT/tests/test_core.py" | tee -a "$INSTALL_LOG"
  fi
else
  log "pytest unavailable; skipped tests. Python compile and YAML validation passed."
fi

if command -v node >/dev/null 2>&1; then
  log "Checking JavaScript syntax with Node..."
  node --check "$PROJECT_ROOT/web/static/app.js" | tee -a "$INSTALL_LOG"
else
  log "Node is unavailable; JavaScript syntax check skipped. Node is not required by FastAPI."
fi

printf 'success\n' > "$BACKUP_ROOT/INSTALL_RESULT.txt"
INSTALLED=0
trap - EXIT

log "Installation completed successfully."
log "Backup: $BACKUP_ROOT"
log "Restore: $BACKUP_ROOT/restore.sh"
log "Restart FastAPI to load the new schema and web UI."

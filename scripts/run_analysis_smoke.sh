#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workingspace_aiclub/WorkingSpace/Personal/chinhnm/AIC2026/src/core/cir}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-$ROOT/config.yaml}"
INPUT="${1:-$ROOT/examples/input.analysis.json}"
OUTPUT="${2:-$ROOT/outputs/output.analysis.json}"
ANALYSIS_DIR="${3:-$ROOT/outputs/score_analysis_objective}"

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")" "$ANALYSIS_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

echo "[1/2] Running CIR analysis request..."
"$PYTHON" run_cir.py \
  --config "$CONFIG" \
  --input "$INPUT" \
  --output "$OUTPUT"

echo "[2/2] Generating objective score diagnostics..."
"$PYTHON" scripts/visualize_cir_scores_objective.py \
  --input "$OUTPUT" \
  --output-dir "$ANALYSIS_DIR"

echo
echo "CIR output:     $OUTPUT"
echo "Analysis files: $ANALYSIS_DIR"

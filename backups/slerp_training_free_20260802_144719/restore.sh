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

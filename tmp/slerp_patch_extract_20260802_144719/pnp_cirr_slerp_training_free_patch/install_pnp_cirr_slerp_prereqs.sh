#!/usr/bin/env bash
set -Eeuo pipefail

# The patch installer intentionally uses only Bash and Python's standard library.
# Node.js and unzip are not required. This script installs Python only when it is absent.

if command -v python >/dev/null 2>&1; then
  echo "python already available: $(python --version 2>&1)"
  exit 0
fi

command -v apt-get >/dev/null 2>&1 || {
  echo "python is missing and apt-get is unavailable." >&2
  exit 1
}

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3 \
  python3-venv \
  ca-certificates

if ! command -v python >/dev/null 2>&1; then
  ln -s "$(command -v python3)" /usr/local/bin/python
fi

python --version

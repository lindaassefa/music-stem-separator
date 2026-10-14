#!/usr/bin/env bash
# Create the project virtualenv and install the separation stack.
# Demucs is installed with --no-deps so it cannot replace torchaudio 2.4.1
# with the torchaudio<2.1 pin published in demucs 4.0.1.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Could not find $PYTHON_BIN. Install Python 3.12 or set PYTHON to another 3.10+ interpreter." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install "demucs==4.0.1" --no-deps
# Re-apply the pin in case openunmix or another package moved PyTorch.
.venv/bin/python -m pip install "torch==2.4.1" "torchaudio==2.4.1"

echo
echo "Installed interpreters and PyTorch device support:"
.venv/bin/python -m stem_ai.device

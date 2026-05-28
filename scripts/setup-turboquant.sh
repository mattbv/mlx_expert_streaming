#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
VENV_DIR="${TQ_VENV:-$HOME/tq-env}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python 3.12 not found at $PYTHON_BIN" >&2
  echo "Install it with: brew install python@3.12" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install --upgrade "turboquant-mlx-full>=0.4.1" requests

echo "TurboQuant environment ready: $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"

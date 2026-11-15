#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PAPER_PYTHON:-/export/abdullah/ode-env/bin/python}"
RESULT_DIR="${PAPER_RESULTS:-/export/abdullah/ode-results/paper-v1}"
mkdir -p "$RESULT_DIR/matplotlib"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$RESULT_DIR/matplotlib"
"$PYTHON_BIN" -m pip freeze > "$RESULT_DIR/environment.txt"
"$PYTHON_BIN" -m unittest discover -s "$REPO_DIR/tests" -v

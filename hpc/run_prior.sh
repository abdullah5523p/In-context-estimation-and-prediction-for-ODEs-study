#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PAPER_PYTHON:-/export/abdullah/ode-env/bin/python}"
LOG_DIR="${PAPER_PRIOR_LOGS:-/export/abdullah/ode-results/prior-screen/logs}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON_BIN" -u -m ode_paper.prior_cli "$@" 2>&1 | tee "$LOG_DIR/$(date +%Y%m%d_%H%M%S)_$$.log"


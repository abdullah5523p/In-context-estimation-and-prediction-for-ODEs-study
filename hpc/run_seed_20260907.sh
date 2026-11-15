#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PAPER_PYTHON:-/export/abdullah/ode-env/bin/python}"
RESULT_DIR="${PAPER_RESULTS:-/export/abdullah/ode-results/paper-seed-20260907}"
mkdir -p "$RESULT_DIR/logs" "$RESULT_DIR/matplotlib"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$RESULT_DIR/matplotlib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ "$#" -eq 0 ]; then
  echo "Usage: bash hpc/run_seed_20260907.sh {list|train|evaluate|plot}"
  exit 2
fi
"$PYTHON_BIN" -u "$REPO_DIR/hpc/seed_20260907.py" --root "$RESULT_DIR" --previous-root /export/abdullah/ode-results/paper-v1 "$@" 2>&1 | tee "$RESULT_DIR/logs/seed_20260907_$(date +%Y%m%d_%H%M%S)_$$.log"

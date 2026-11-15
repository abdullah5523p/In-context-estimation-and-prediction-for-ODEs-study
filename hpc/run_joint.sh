#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PAPER_PYTHON:-/export/abdullah/ode-env/bin/python}"
LOG_DIR="${PAPER_JOINT_LOGS:-/export/abdullah/ode-results/joint-experiment/logs}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$LOG_DIR/matplotlib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ "$#" -eq 0 ]; then
  echo "Usage: bash hpc/run_joint.sh {list|train|evaluate|export}"
  exit 2
fi
"$PYTHON_BIN" -u -m ode_paper.joint_cli "$@" 2>&1 | tee "$LOG_DIR/$(date +%Y%m%d_%H%M%S)_$$.log"


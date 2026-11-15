#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PAPER_PYTHON:-/export/abdullah/ode-env/bin/python}"
SEED=20260906
ROOT_OVERRIDE=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --seed) SEED="${ARGS[i+1]:-}" ;;
    --seed=*) SEED="${ARGS[i]#--seed=}" ;;
    --root) ROOT_OVERRIDE="${ARGS[i+1]:-}" ;;
    --root=*) ROOT_OVERRIDE="${ARGS[i]#--root=}" ;;
  esac
done
if [[ "$SEED" == 20260906 ]]; then
  DEFAULT_RESULTS=/export/abdullah/ode-results/paper-v1
else
  DEFAULT_RESULTS="/export/abdullah/ode-results/paper-seed-$SEED"
fi
RESULT_DIR="${ROOT_OVERRIDE:-${PAPER_RESULTS:-$DEFAULT_RESULTS}}"
mkdir -p "$RESULT_DIR/logs" "$RESULT_DIR/matplotlib"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$RESULT_DIR/matplotlib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ "$#" -eq 0 ]; then
  echo "Usage: bash hpc/run.sh {list|train|evaluate|plot} [--seed 20260906|20260907] [--index 0 | --all]"
  exit 2
fi
"$PYTHON_BIN" -u -m ode_paper.cli --root "$RESULT_DIR" "$@" 2>&1 | tee "$RESULT_DIR/logs/$(date +%Y%m%d_%H%M%S)_$$.log"

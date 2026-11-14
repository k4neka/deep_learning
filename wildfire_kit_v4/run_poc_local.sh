#!/bin/bash
# Runs the proof-of-concept curriculum on this computer (WSL / Linux), without the VSC cluster.
#   bash run_poc_local.sh            # all stages, 16 parallel environments
#   NENV=20 bash run_poc_local.sh    # more environments if the CPU has spare threads
# Progress: runs/logs/*.log   Results for the report: runs/poc_*/summary.md
set -e
cd "$(dirname "$0")"
VENV="$HOME/venv-dl"
if [ ! -x "$VENV/bin/python" ]; then
    echo "== creating $VENV (first run only, a few minutes)"
    python3 -m venv "$VENV" || { echo "python3-venv is missing: run  sudo apt install -y python3-venv  and start again"; exit 1; }
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
    "$VENV/bin/pip" install -q gymnasium stable-baselines3 scipy matplotlib
fi
PY="$VENV/bin/python"
NENV="${NENV:-16}"
mkdir -p runs/logs
run() {   # name, then train_poc.py arguments
    local name=$1; shift
    if [ -f "runs/$name/summary.md" ]; then echo "== $name already done, skipping"; return; fi
    echo "== $name started $(date)"
    "$PY" train_poc.py "$@" --steps 2000000 --n-envs "$NENV" --subproc --out "runs/$name" > "runs/logs/$name.log" 2>&1
    echo "== $name finished $(date)"
}
run poc_fe4    --fire-every 4
run poc_fe2    --fire-every 2 --init-from runs/poc_fe4/model.zip
run poc_fe1    --fire-every 1 --init-from runs/poc_fe2/model.zip
run poc_fe4_s1 --fire-every 4 --seed 1
echo "== all stages done $(date)"

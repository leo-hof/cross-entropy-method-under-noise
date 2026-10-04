#!/usr/bin/env bash
# Train + evaluate every variant on every noisy environment with 5 seeds (80 runs).
# Runs 8 jobs in parallel and skips runs that already have a results/*_eval.json.
# ~13 min on a 10-core laptop CPU.  Usage: bash scripts/run_grid.sh
cd "$(dirname "$0")/.." || exit 1
PYTHON="${PYTHON:-.venv/bin/python}"
JOBS="${JOBS:-8}"

jobs_file="$(mktemp)"
for env in obs reward transition start; do
  for seed in 0 1 2 3 4; do
    for variant in "cem 1" "cem 5" "baseline 1" "advantage 1"; do
      read -r agent rollouts <<< "$variant"
      [ -f "results/${agent}_${env}_r${rollouts}_s${seed}_eval.json" ] || echo "$agent $env $rollouts $seed" >> "$jobs_file"
    done
  done
done
echo "runs to do: $(wc -l < "$jobs_file" | tr -d ' ')"

export OMP_NUM_THREADS=1 PYTHON  # one thread per job, since jobs already run in parallel
xargs -P "$JOBS" -L 1 bash -c '"$PYTHON" run.py --agent "$0" --env "$1" --rollouts "$2" --seed "$3" > /dev/null 2>&1 || echo "FAILED: $0 $1 $2 $3"' < "$jobs_file"
rm -f "$jobs_file"
echo "result files: $(ls results/*_eval.json | wc -l | tr -d ' ')"

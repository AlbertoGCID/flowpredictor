#!/usr/bin/env bash
# Sensitivity experiments: tau=0.90 baselines with the published seed (92) and
# seeds 42 and 123 for M1/M2/M5, at 48h, on the 2017-2023 expanding-window CV
# and the 2025 holdout. Sequential on purpose: every step writes to
# results_consolidated.csv. Run it inside tmux with tee, for example:
#   tmux new-session -d -s sens_seeds \
#     "bash resultados/prueba63/tools/run_sensitivity.sh 2>&1 | tee experiment_sens_seeds.log; exec bash"
set -u
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TOOLS_DIR/../../.." && pwd)"
# dataset.root_raw (experiments_config.toml) is relative to the parent directory of flowpredictor.
cd "$(dirname "$REPO_ROOT")"
export PYTHONPATH="$REPO_ROOT"
MOD=resultados.prueba63.src.run_experiments_phase3
HOLD="$TOOLS_DIR/variant_holdout.py"

step() { echo; echo "=================== $(date '+%F %T') · $* ==================="; }

step "[seed 92] baselines tau=0.50+0.90 · expanding CV (M5 already complete: skipped)"
python3 -m $MOD --run_cv
step "[seed 92] baselines tau=0.50+0.90 · holdout 2025"
python3 "$HOLD" --seed 92 --baselines

for SEED in 42 123; do
  step "[seed $SEED] M1, M2 · expanding CV"
  python3 -m $MOD --run_ablation --configs M1 M2 --seed $SEED
  step "[seed $SEED] M5 · expanding CV"
  python3 -m $MOD --run_cv --seed $SEED
  step "[seed $SEED] M1, M2, M5 · holdout 2025"
  python3 "$HOLD" --seed $SEED --configs M1 M2 M5
done
step "DONE"

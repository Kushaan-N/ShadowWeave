#!/bin/bash
# Submit the whole pipeline as a dependency chain:
#   generate data → train world model → train RL → evaluate
# Each stage starts only if the previous one succeeded.
#
#   ./slurm/pipeline.sh
#   SW_EPISODES=800 SW_RL_STEPS=2000000 ./slurm/pipeline.sh
#   SW_SKIP_DATA=1 ./slurm/pipeline.sh      # reuse existing rollouts

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

echo "Submitting ShadowWeave pipeline"

DEP=""
if [[ "${SW_SKIP_DATA:-0}" != "1" ]]; then
  TRAIN_DATA=$(sbatch --parsable slurm/gen_data.sbatch)
  echo "  gen_data (train) : ${TRAIN_DATA}"

  VAL_DATA=$(SW_SPLIT=val SW_EPISODES="${SW_VAL_EPISODES:-80}" \
    sbatch --parsable --array=0-3 slurm/gen_data.sbatch)
  echo "  gen_data (val)   : ${VAL_DATA}"

  # afterok on an array job waits for every task in it.
  DEP="--dependency=afterok:${TRAIN_DATA}:${VAL_DATA}"
fi

WM=$(sbatch --parsable ${DEP} slurm/train_worldmodel.sbatch)
echo "  world model      : ${WM}"

RL=$(sbatch --parsable --dependency=afterok:"${WM}" slurm/train_rl.sbatch)
echo "  local agent (RL) : ${RL}"

EV=$(sbatch --parsable --dependency=afterok:"${RL}" slurm/eval.sbatch)
echo "  eval             : ${EV}"

echo
echo "Watch with:  squeue -u \$USER"
echo "Cancel all:  scancel ${TRAIN_DATA:-} ${VAL_DATA:-} ${WM} ${RL} ${EV}"

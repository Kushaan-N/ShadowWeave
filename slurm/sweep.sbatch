#!/bin/bash
#SBATCH --job-name=sw-sweep
#SBATCH --output=slurm/logs/sweep_%A_%a.out
#SBATCH --error=slurm/logs/sweep_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#
# Hyperparameter sweep — one array task per configuration, each with its own
# checkpoint directory so runs never overwrite one another.
#
#   sbatch slurm/sweep.sbatch
#
# Edit CONFIGS below and keep --array in range (0 .. N-1).

source slurm/env.sh
mkdir -p slurm/logs

CONFIGS=(
  "world_model.base_channels=32 world_model.lr=1e-4"
  "world_model.base_channels=64 world_model.lr=1e-4"
  "world_model.base_channels=64 world_model.lr=3e-4"
  "world_model.base_channels=64 world_model.lr=1e-4 world_model.pos_weight=12.0"
  "world_model.base_channels=96 world_model.lr=1e-4"
  "world_model.architecture=convlstm world_model.base_channels=64"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#CONFIGS[@]} )); then
  echo "no config at index ${IDX} (have ${#CONFIGS[@]})"
  exit 0
fi

CFG="${CONFIGS[$IDX]}"
RUN_DIR="checkpoints/sweep/run${IDX}"
echo "sweep run ${IDX}: ${CFG}"

srun python -m shadowweave.world_model.train \
  --data "${SW_DATA_ROOT:-data/rollouts}" \
  --overrides ${CFG} \
    world_model.checkpoint_dir="${RUN_DIR}" \
    wandb.project=shadowweave-sweep

echo "sweep run ${IDX} finished → ${RUN_DIR}"

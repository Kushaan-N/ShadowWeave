#!/bin/bash
#SBATCH --job-name=sw-valshadow
#SBATCH --output=slurm/logs/valshadow_%j.out
#SBATCH --error=slurm/logs/valshadow_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#
# Fixed-validation-set shadow-IOU for one world-model checkpoint, on a GPU node (fast,
# full val set — no --max-batches). Confound-free: no policy, identical data per model.
#
#   SW_CKPT=<ckpt> SW_DATA_ROOT=<val root> SW_VAL_OUT=<out.json> sbatch slurm/val_shadow.sbatch
#
# SW_CKPT       world model checkpoint to score
# SW_DATA_ROOT  rollout root (expects <root>/val/)
# SW_VAL_OUT    where to write the metrics json
# SW_OVERRIDES  optional extra config overrides (usually none — arch/channels come from ckpt)

source slurm/env.sh
mkdir -p slurm/logs

CKPT="${SW_CKPT:?set SW_CKPT to the world model checkpoint}"
DATA_ROOT="${SW_DATA_ROOT:?set SW_DATA_ROOT to the val rollout root}"
OUT="${SW_VAL_OUT:-val_shadow.json}"
mkdir -p "$(dirname "$OUT")"

srun python scripts/eval_val_shadow.py \
  --ckpt "${CKPT}" \
  --data "${DATA_ROOT}" \
  --split val \
  --batch-size 64 \
  --out "${OUT}" \
  --overrides ${SW_OVERRIDES:-}

echo "val-shadow done -> ${OUT}"

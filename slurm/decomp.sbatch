#!/bin/bash
#SBATCH --job-name=sw-decomp
#SBATCH --output=slurm/logs/decomp_%j.out
#SBATCH --error=slurm/logs/decomp_%j.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#
# Static/dynamic decomposition + calibration + bootstrap-CI shadow gains for one checkpoint
# on one val set (full val, no --max-batches), on a GPU node.
#
#   SW_CKPT=<ckpt> SW_DATA_ROOT=<val root> SW_VAL_OUT=<out.json> sbatch slurm/decomp.sbatch

source slurm/env.sh
mkdir -p slurm/logs

CKPT="${SW_CKPT:?set SW_CKPT}"
DATA_ROOT="${SW_DATA_ROOT:?set SW_DATA_ROOT (expects <root>/val/)}"
OUT="${SW_VAL_OUT:-val_decomp.json}"
mkdir -p "$(dirname "$OUT")"

srun python scripts/eval_val_decomp.py \
  --ckpt "${CKPT}" --data "${DATA_ROOT}" --split val \
  --batch-size 64 --bootstrap "${SW_BOOTSTRAP:-10000}" --out "${OUT}" \
  --overrides ${SW_OVERRIDES:-}

echo "decomp done -> ${OUT}"

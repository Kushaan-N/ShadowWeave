#!/bin/bash
#SBATCH --job-name=sw-valdecomp
#SBATCH --output=slurm/logs/valdecomp_%j.out
#SBATCH --error=slurm/logs/valdecomp_%j.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#
# Static/dynamic shadow decomposition + calibration-in-shadow for one world-model
# checkpoint, per difficulty tier, in a SINGLE GPU allocation. Reuses the per-tier val
# sets built by val_pertier.sbatch (rollouts_val_<tier>); errors if they are missing.
#
#   SW_CKPT=<ckpt> SW_VAL_DECOMP_DIR=<out dir> sbatch slurm/val_decomp.sbatch

source slurm/env.sh
mkdir -p slurm/logs

WS="${WS:-/scratch4/workspace/knaskar_umass_edu-shadowweave}"
CKPT="${SW_CKPT:-$WS/checkpoints/world_model/best.pt}"
OUTDIR="${SW_VAL_DECOMP_DIR:-$WS/val_decomp}"
mkdir -p "$OUTDIR"

for tier in static moving debris; do
  DATA=$WS/rollouts_val_${tier}
  if [[ "$(ls "$DATA/val/"*.npz 2>/dev/null | wc -l)" -lt 1 ]]; then
    echo "ERROR: no per-tier val set at $DATA/val — run run_val_pertier.sh first"; exit 1
  fi
  echo "=== tier ${tier} ==="
  srun python scripts/eval_val_decomp.py \
    --ckpt "${CKPT}" --data "${DATA}" --split val \
    --batch-size 64 --out "${OUTDIR}/${tier}.json"
done

echo "decomposition done -> ${OUTDIR}/{static,moving,debris}.json"

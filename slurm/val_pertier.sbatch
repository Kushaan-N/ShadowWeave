#!/bin/bash
#SBATCH --job-name=sw-valpertier
#SBATCH --output=slurm/logs/valpertier_%j.out
#SBATCH --error=slurm/logs/valpertier_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#
# Confound-free per-tier shadow gain in a SINGLE GPU allocation: generate a clean
# single-difficulty val set for each tier and score one world model on it, all serially.
# One queue wait instead of six (gen+eval per tier), and one fair-share hit — the whole
# thing is ~10-15 min, well under the wall limit. Everything on scratch.
#
#   SW_CKPT=<ckpt> sbatch slurm/val_pertier.sbatch
#
# SW_CKPT            world model checkpoint to score (default: the full model)
# SW_TIER_EPISODES   val episodes per tier (default 30)

source slurm/env.sh
mkdir -p slurm/logs

WS="${WS:-/scratch4/workspace/knaskar_umass_edu-shadowweave}"
CKPT="${SW_CKPT:-$WS/checkpoints/world_model/best.pt}"
EPS="${SW_TIER_EPISODES:-30}"
STEPS="${SW_STEPS:-300}"
OUTDIR="${SW_VAL_PERTIER_DIR:-$WS/val_pertier}"
mkdir -p "$OUTDIR"

echo "per-tier val-shadow (single job) for checkpoint: $CKPT"

for tier in static moving debris; do
  DATA=$WS/rollouts_val_${tier}
  echo "=== tier ${tier}: generate ${EPS} episodes ==="
  # Regenerate only if the tier's val set isn't already present (idempotent on requeue).
  if [[ "$(ls "$DATA/val/"*.npz 2>/dev/null | wc -l)" -lt "$EPS" ]]; then
    srun python -m shadowweave.sim.synthetic_data \
      --episodes "${EPS}" --split val --steps "${STEPS}" --seed0 900000 \
      --out "${DATA}" --overrides eval.difficulty_tiers=[${tier}]
  else
    echo "  reusing existing $DATA/val"
  fi

  echo "=== tier ${tier}: score model ==="
  srun python scripts/eval_val_shadow.py \
    --ckpt "${CKPT}" --data "${DATA}" --split val \
    --batch-size 64 --out "${OUTDIR}/${tier}.json"
done

echo "per-tier done -> ${OUTDIR}/{static,moving,debris}.json"

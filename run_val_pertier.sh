#!/bin/bash
# Confound-free per-difficulty-tier shadow gain, on the FIXED validation set.
#
# The existing val set cycles static/moving/debris per episode and does not record the tier,
# so it can't be split after the fact. This generates a small CLEAN per-tier val set (all
# episodes one difficulty) for each tier, then scores the full world model on each with the
# no-policy eval_val_shadow — so the per-tier numbers differ only by scene difficulty, not by
# policy trajectory. Answers the reviewer question "is the +0.36 gain just static-layout
# memorisation, or does it hold where objects actually move into the shadow?"
#
# All compute runs as SLURM batch jobs on GPU nodes (nothing on the login node). Everything
# on scratch. Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu

CKPT="${SW_CKPT:-$WS/checkpoints/world_model/best.pt}"   # the full model by default
EPS="${SW_TIER_EPISODES:-30}"                             # val episodes per tier
mkdir -p "$WS/val_pertier"

echo "per-tier val-shadow for checkpoint: $CKPT"
for tier in static moving debris; do
  DATA=$WS/rollouts_val_${tier}

  # 1) generate a clean single-tier val set (2 shards) -> $DATA/val/
  GEN=$(SW_SPLIT=val SW_EPISODES=$EPS SW_DATA_ROOT=$DATA \
        SW_OVERRIDES="eval.difficulty_tiers=[${tier}]" \
        sbatch --parsable --array=0-1 slurm/gen_data.sbatch)

  # 2) score the model on that tier once generation succeeds
  EV=$(SW_CKPT="$CKPT" SW_DATA_ROOT="$DATA" SW_VAL_OUT="$WS/val_pertier/${tier}.json" \
       sbatch --parsable --dependency=afterok:"$GEN" slurm/val_shadow.sbatch)

  echo "  tier=${tier}:  gen=$GEN  ->  val-shadow=$EV  ->  $WS/val_pertier/${tier}.json"
done

echo
echo "when all finish, compare val_shadow_gain_5s across $WS/val_pertier/{static,moving,debris}.json"
echo "expect: static <= moving <= debris  (persistence is strong on static; the model should"
echo "pull ahead most where objects move into the shadow => forecasting, not memorisation)."

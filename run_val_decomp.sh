#!/bin/bash
# Launch the static/dynamic decomposition + calibration-in-shadow analysis (eval-only) for
# the full and no-visibility world models, per difficulty tier, each as a single GPU batch
# job. Reuses the per-tier val sets from run_val_pertier.sh, so both checkpoints are scored
# on identical data. Everything on scratch. Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu

declare -A CKPTS=(
  [full]=$WS/checkpoints/world_model/best.pt
  [novis]=$WS/checkpoints/world_model_noshadow/best.pt
)

for tag in full novis; do
  JOB=$(SW_CKPT="${CKPTS[$tag]}" SW_VAL_DECOMP_DIR="$WS/val_decomp/$tag" \
        sbatch --parsable --gres="${SW_TRAIN_GPU:-gpu:1}" slurm/val_decomp.sbatch)
  echo "decomp [$tag]: $JOB  ->  $WS/val_decomp/$tag/{static,moving,debris}.json"
done

echo
echo "when done:  python scripts/plot_decomp.py --dir $WS/val_decomp --out figures/"
echo "read: dynamic_iou/dynamic_recall rising static<moving<debris = forecasting (not just"
echo "static completion); ece_shadow ~ ece_observed = faithful uncertainty in unseen space."

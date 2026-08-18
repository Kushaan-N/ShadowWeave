#!/bin/bash
# Analysis — break the shadow-forecasting gain out by difficulty tier (static / moving /
# debris), instead of the single pooled +0.320. This defuses the sharpest reviewer attack
# ("is the gain just amodal hole-filling of STATIC occluders, i.e. memorisation?"): if the
# gain concentrates in the moving/debris tiers — where objects actually enter the shadow and
# nothing can be memorised — that is direct evidence of forecasting, not static in-painting.
#
# Eval-ONLY: reuses the already-trained full world model + policy, NO retraining. The eval
# loops tiers but pools all metrics into one summary, so we run it once per tier (each with
# a single-tier override) to get separate numbers. Everything on scratch.
# Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_DATA_ROOT=$WS/rollouts
export SW_CKPT_DIR=$WS/checkpoints/world_model
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu

GPU="${SW_TRAIN_GPU:-gpu:1}"
WM_CKPT=$WS/checkpoints/world_model/best.pt   # the trained full world model (reused)

for tier in static moving debris; do
  EV=$(SW_CKPT="$WM_CKPT" SW_RESULTS_DIR=$WS/results_tier_${tier} SW_EVAL_EPISODES=30 \
       SW_OVERRIDES="eval.difficulty_tiers=[${tier}]" \
       sbatch --parsable --gres="$GPU" slurm/eval.sbatch)
  echo "tier=${tier}  eval=$EV  ->  $WS/results_tier_${tier}/eval_summary.json"
done

echo
echo "when all three finish, compare model_shadow_gain_over_best_baseline_5s across tiers:"
echo "  expect static <= moving <= debris  (persistence is strong on static; the model should"
echo "  pull ahead most where objects move into the shadow). full pooled value = +0.320."

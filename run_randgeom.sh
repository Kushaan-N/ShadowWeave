#!/bin/bash
# Randomized-geometry experiment: regenerate data with per-episode room geometry, retrain the
# U-Net world model on it, and re-measure the shadow gain (rollout + fixed-val decomposition
# with bootstrap CIs). Kills the fixed-6x6-wall memorization confound. Writes to *_randgeom
# names so the fixed-geometry baseline (checkpoints/world_model, results/) is untouched and
# the two can be reported side by side. Everything on scratch. Untracked helper.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu
GPU="${SW_TRAIN_GPU:-gpu:1}"

DATA=$WS/rollouts_randgeom
CKPTS=$WS/checkpoints/world_model_randgeom
RND="sim.randomize_room=true"          # opt into randomized geometry for gen + eval env

# 1) regenerate data with randomized geometry (train + val; val seeds are +900000, disjoint).
GTR=$(SW_SPLIT=train SW_EPISODES=400 SW_DATA_ROOT=$DATA SW_OVERRIDES="$RND" \
      sbatch --parsable slurm/gen_data.sbatch)
GVA=$(SW_SPLIT=val SW_EPISODES=80 SW_DATA_ROOT=$DATA SW_OVERRIDES="$RND" \
      sbatch --parsable --array=0-3 slurm/gen_data.sbatch)

# 2) retrain the U-Net world model on the randomized data (after both gens succeed).
WM=$(SW_DATA_ROOT=$DATA SW_CKPT_DIR=$CKPTS \
     sbatch --parsable --gres="$GPU" --time=08:00:00 \
     --dependency=afterok:"$GTR":"$GVA" slurm/train_worldmodel.sbatch)

# 3a) rollout eval in randomized rooms (shadow gain, headline table form).
EV=$(SW_CKPT=$CKPTS/best.pt SW_CKPT_DIR=$CKPTS SW_RESULTS_DIR=$WS/results_randgeom SW_EVAL_EPISODES=90 \
     SW_OVERRIDES="$RND" \
     sbatch --parsable --gres="$GPU" --dependency=afterok:"$WM" slurm/eval.sbatch)

# 3b) fixed-val decomposition + bootstrap-CI on the randomized val set (the reviewer-proof
#     nostatic micro gain with an error bar, on geometry the model never memorized).
DC=$(SW_CKPT=$CKPTS/best.pt SW_DATA_ROOT=$DATA SW_VAL_OUT=$WS/val_decomp/randgeom.json \
     sbatch --parsable --gres="$GPU" --dependency=afterok:"$WM" slurm/decomp.sbatch)

echo "randgeom chain:"
echo "  gen train=$GTR  val=$GVA"
echo "  worldmodel=$WM  (-> $CKPTS)"
echo "  rollout eval=$EV  (-> results_randgeom/eval_summary.json)"
echo "  decomp+CI  =$DC  (-> $WS/val_decomp/randgeom.json)"
echo "compare shadow gain + nostatic gain (with CI) vs fixed-geom to show the gain survives."

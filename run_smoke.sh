#!/bin/bash
# Fast go/no-go on the CORE claim (world-model shadow gain): a 25-epoch U-Net + a small
# eval. Skips the 8h PPO — collision rate needs PPO, but the shadow-gain / IOU metrics
# come from the world model's predictions and do not. Run after run_check_data.sh says OK.
# Writes to *_smoke dirs so it never clobbers the full run. Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_DATA_ROOT=$WS/rollouts
export SW_CKPT_DIR=$WS/checkpoints/world_model_smoke
export SW_PYTHON_MODULE=python/3.11.7
export SW_CUDA_MODULE=cuda/12.6
export SBATCH_PARTITION=gpu
export SBATCH_ACCOUNT=pi_sniekum_umass_edu   # this is a Sniekum-lab project; don't bill pi_andrewlan

GPU="${SW_TRAIN_GPU:-gpu:1}"     # any GPU; override e.g. SW_TRAIN_GPU=gpu:l40s:1 for speed

# The smoke is ~1.5-2h on an a100, but train_worldmodel.sbatch asks for 12h, which
# blocks backfill into otherwise-free GPUs. Request a short limit so it schedules now.
WM=$(SW_OVERRIDES="world_model.epochs=25" sbatch --parsable --gres="$GPU" --time="${SW_TIME:-03:00:00}" slurm/train_worldmodel.sbatch)
EV=$(SW_CKPT="$SW_CKPT_DIR/best.pt" SW_RESULTS_DIR=results_smoke SW_EVAL_EPISODES=30 \
     sbatch --parsable --gres="$GPU" --dependency=afterok:"$WM" slurm/eval.sbatch)

echo "smoke chain:  worldmodel(25ep)=$WM  ->  eval=$EV"
echo "results land in ./results_smoke/eval_summary.json when $EV finishes"
echo "watch:  squeue -u $USER   |   tail -f slurm/logs/worldmodel_*.out"
